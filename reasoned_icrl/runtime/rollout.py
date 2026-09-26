"""The greedy rollout of a trained policy, and everything that needs it.

One fresh trajectory cache per task, the actions the policy chooses, no learner
update; the cache *interventions* an environment declares clear that cache at a
task boundary while leaving the environment, rewards, budget and the current
transition packet untouched. :func:`evaluate` drives one checkpoint over a
declared roster and hands the per-task results to the pure record builders in
:mod:`reasoned_icrl.experiments.evaluation`; :func:`checkpoint_series` repeats
that at every scheduled label; the scripted probes replay public streams under
the frozen policy for the mechanism diagnostics.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Protocol, cast

import numpy as np
import torch
from amago.envs.amago_env import SequenceWrapper
from numpy.typing import NDArray

from reasoned_icrl.environments.base import BaseEnv
from reasoned_icrl.environments.concentration import ConcentrationEnv
from reasoned_icrl.environments.count_recall import CountRecallEnv
from reasoned_icrl.environments.match_pattern import MatchPatternEnv
from reasoned_icrl.environments.mazerunner import MazeRunnerEnv
from reasoned_icrl.environments.tmaze import TMazeEnv
from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import (
    ContractError,
    architecture_uses_history_packet,
)
from reasoned_icrl.experiments.evaluation import (
    HISTORY_MODES,
    SUMMARY_CLEARED,
    AttemptTaskResult,
    ConcentrationResult,
    CountRecallStreamResult,
    MatchPatternResult,
    MazeRunnerEpisodeResult,
    TMazeEpisodeResult,
    continued_history_modes,
    evaluation_environment,
    events,
    run_record,
    secondary,
    write_evaluation,
)
from reasoned_icrl.experiments.qualification import CheckpointScore, primary_from_events
from reasoned_icrl.experiments.records import (
    CHECKPOINT_RULES,
    BenchmarkEvent,
    BenchmarkRun,
    CheckpointRule,
)
from reasoned_icrl.runtime.environments import amago_environment

Boundary = Callable[[Mapping[str, Any]], NDArray[np.bool_]]


class DecisionProbe(Protocol):
    """An evaluation-only observer of a rollout's policy calls.

    :meth:`begin_chunk` receives the task identity of every row of a chunk;
    :meth:`after_policy` runs right after each policy call with the live rows,
    each row's decision index (the decisions it had already taken) and the
    carrier state the call left; :meth:`after_actions` then receives the same
    rows and indices with the call's public input packets (every key,
    ``[rows, ...]``) and the chosen actions (``[rows, ...]``). A probe must not
    change the rollout; the attention diagnostics of
    :mod:`reasoned_icrl.runtime.attention` and the summary capture of
    :mod:`reasoned_icrl.runtime.representation` use it to align what they saw
    with the decisions it served.
    """

    def begin_chunk(self, task_ids: Sequence[int]) -> None: ...

    def after_policy(
        self, live_rows: Sequence[int], steps: NDArray[np.int64], hidden: Any
    ) -> None: ...

    def after_actions(
        self,
        live_rows: Sequence[int],
        steps: NDArray[np.int64],
        observation: Mapping[str, NDArray[Any]],
        actions: NDArray[Any],
    ) -> None: ...


def _attempt_boundary(info: Mapping[str, Any]) -> NDArray[np.bool_]:
    return np.asarray(info["attempt_done"], dtype=np.bool_).reshape(-1)


def _goal_boundary(info: Mapping[str, Any]) -> NDArray[np.bool_]:
    return np.asarray(info["goal_reached"], dtype=np.bool_).reshape(-1)


def _query_boundary(info: Mapping[str, Any]) -> NDArray[np.bool_]:
    return np.asarray([bool(info["query_ready"])], dtype=np.bool_)


def _every_step(info: Mapping[str, Any]) -> NDArray[np.bool_]:
    del info
    return np.ones(1, dtype=np.bool_)


BOUNDARIES: dict[str, Boundary] = {
    "attempt-cleared": _attempt_boundary,
    "current-token": _every_step,
    "goal-cleared": _goal_boundary,
}


def _extract(
    env: BaseEnv,
    task_id: int,
    rollout_seed: int,
    steps: int,
    *,
    writes: tuple[int, ...] | None = None,
    values: tuple[int, ...] = (),
) -> Any:
    """Read the evaluator-only records of the task that just finished."""
    if writes is not None and len(writes) != steps:
        raise ContractError("Summary write counters must cover every decision.")
    if isinstance(env, MatchPatternEnv):
        return MatchPatternResult(task_id, rollout_seed, env.scored_decision, steps)
    if isinstance(env, CountRecallEnv):
        return CountRecallStreamResult(
            task_id,
            rollout_seed,
            env.scored_queries,
            env.stream_return,
            steps,
            writes,
            values,
            streams=env.streams,
        )
    if isinstance(env, ConcentrationEnv):
        return ConcentrationResult(
            task_id,
            rollout_seed,
            env.flips,
            env.matched_pairs,
            env.pairs,
            env.episode_return,
            steps,
            writes,
        )
    if isinstance(env, TMazeEnv):
        if env.episode is None:
            raise ContractError("A T-Maze result needs a finished episode.")
        return TMazeEpisodeResult(task_id, rollout_seed, env.episode, steps, writes)
    if isinstance(env, MazeRunnerEnv) and env.meta_horizon is not None:
        # The repeated-laps axis scores one attempt record per lap.
        return AttemptTaskResult(
            task_id,
            rollout_seed,
            env.completed_attempts,
            env.partial_attempt,
            env.task_return,
            steps,
            writes,
        )
    if isinstance(env, MazeRunnerEnv):
        return MazeRunnerEpisodeResult(
            task_id,
            rollout_seed,
            env.goals,
            env.completed_goals,
            env.episode_return,
            steps,
            writes,
        )
    attempt_env = cast(Any, env)
    return AttemptTaskResult(
        task_id,
        rollout_seed,
        attempt_env.completed_attempts,
        attempt_env.partial_attempt,
        attempt_env.task_return,
        steps,
        writes,
    )


EVALUATION_BATCH_VARIABLE = "REASONED_ICRL_EVALUATION_BATCH"
DEFAULT_EVALUATION_BATCH = 64


def evaluation_batch(requested: int | None = None) -> int:
    """How many roster tasks one rollout drives at once.

    Every carrier keeps its rows independent (own cache, memory and counters,
    reset by a per-row mask), so the batch changes only how many policy calls
    a roster costs, never what any task sees. ``None`` reads
    ``REASONED_ICRL_EVALUATION_BATCH`` (default 64).
    """
    if requested is None:
        raw = os.environ.get(EVALUATION_BATCH_VARIABLE, "").strip()
        requested = int(raw) if raw else DEFAULT_EVALUATION_BATCH
    if isinstance(requested, bool) or int(requested) < 1:
        raise ContractError("The evaluation batch must be a positive integer.")
    return int(requested)


def _carries_summary(encoder: Any) -> bool:
    """Whether the carrier writes a summary that ``summary-cleared`` can drop:
    the summary carrier in its ``summary`` regime, or the Memo comparator,
    whose boundaries then discard the accumulated summaries."""
    from reasoned_icrl.model.trajectory_encoder import MemoTrajEncoder

    if isinstance(encoder, MemoTrajEncoder):
        return True
    return getattr(getattr(encoder, "spec", None), "regime", None) == "summary"


def rollout(
    experiment: Any,
    environment: Any,
    *,
    task_ids: Sequence[int],
    rollout_seed: int,
    history: str = "retained",
    sample_actions: bool = False,
    concentration_probe: list[dict[str, int]] | None = None,
    count_probe: list[dict[str, object]] | None = None,
    decision_probe: DecisionProbe | None = None,
) -> tuple[tuple[Any, ...], dict[str, float]]:
    """Run one greedy episode per declared task identity, without any update.

    ``decision_probe`` (evaluation only) observes every chunk and policy call
    without changing the rollout; see :class:`DecisionProbe`.

    ``environment`` is one AMAGO-wrapped environment or a sequence of them.
    The roster is driven in chunks of that many tasks, one task per
    environment and one policy call per step for the whole chunk; rows are
    independent in every carrier, so a task's rollout does not depend on which
    tasks share its chunk. The trajectory cache is created fresh for every
    chunk, so no identity's history can leak into the next one. A cache
    intervention additionally drops the cache wherever the task reports its
    declared boundary.
    """
    environments = (
        tuple(environment) if isinstance(environment, (list, tuple)) else (environment,)
    )
    if not environments:
        raise ContractError("A rollout needs at least one environment.")
    bases = [cast(BaseEnv, env.unwrapped) for env in environments]
    base = bases[0]
    if concentration_probe is not None and not isinstance(base, ConcentrationEnv):
        raise ContractError("The scripted partner probe requires Concentration.")
    if count_probe is not None and not isinstance(base, CountRecallEnv):
        raise ContractError("The scripted count probe requires CountRecall.")
    name = str(base.__class__.__name__)
    allowed = HISTORY_MODES.get(_environment_name(base), ())
    if isinstance(base, CountRecallEnv) and base.streams > 1:
        allowed = (*allowed, "attempt-cleared")  # the stream-cleared companion
    if isinstance(base, MazeRunnerEnv) and base.meta_horizon is not None:
        allowed = (*allowed, "attempt-cleared")  # the lap-cleared companion
    if history not in allowed:
        raise ContractError(f"Unknown {name} history intervention: {history!r}.")
    if any(type(other) is not type(base) for other in bases):
        raise ContractError("Every environment of one rollout must share a class.")
    boundary = BOUNDARIES.get(history)
    if isinstance(base, MatchPatternEnv) and history == "current-token":
        boundary = _query_boundary
    # The summary write bookkeeping (boundaries crossed before a decision and
    # since its evidence) is defined for the carrier's own lifecycle: under a
    # cache-clearing intervention the boundary counter restarts at every
    # declared boundary and the evidence it would date has been erased, so
    # the counters are not recorded there (``summary-cleared`` keeps them: it
    # replaces the carried memory but never clears the segment).
    track_writes = boundary is None
    policy = experiment.policy
    policy.eval()
    encoder = policy.traj_encoder
    summary_cleared = history == SUMMARY_CLEARED
    if summary_cleared and not _carries_summary(encoder):
        raise ContractError(
            f"{SUMMARY_CLEARED} applies to a carrier whose memory regime is "
            "'summary' or 'accumulated'; this carrier writes no carried summary."
        )
    device = experiment.DEVICE
    sequences = [
        SequenceWrapper(env, save_trajs_to=None, save_every=None)
        for env in environments
    ]
    # Match the collection-time precision of history conditions exactly, and
    # leave the memoryless reference in full precision.
    uses_history = architecture_uses_history_packet(
        str(experiment.encoder_architecture_id)
    )
    count_recall = isinstance(base, CountRecallEnv)
    mazerunner = isinstance(base, MazeRunnerEnv)
    results: list[Any] = []
    decisions = 0
    inference_total = 0.0
    charged = physical = reset_only = 0  # R4: the evaluation's own clocks
    intervention_count = intervention_episodes = post_intervention_decisions = 0
    blocked_moves = movement_attempts = 0
    action_counts: dict[int, int] = {}
    if sample_actions:
        torch.manual_seed(rollout_seed)
    started = time.perf_counter()
    width = len(environments)
    for first in range(0, len(task_ids), width):
        chunk = [int(task_id) for task_id in task_ids[first : first + width]]
        rows = len(chunk)
        clocks_before = [bases[row].collection_counters() for row in range(rows)]
        for row, task_id in enumerate(chunk):
            bases[row].set_task(task_id)
            sequences[row].reset(seed=rollout_seed)
        hidden = encoder.init_hidden_state(rows, device)
        if summary_cleared:
            hidden.summary_cleared = True
        if decision_probe is not None:
            decision_probe.begin_chunk(chunk)
        live = np.ones(rows, dtype=np.bool_)
        steps = np.zeros(rows, dtype=np.int64)
        episode_intervened = np.zeros(rows, dtype=np.bool_)
        writes: list[list[int]] = [[] for _ in range(rows)]
        values: list[list[int]] = [[] for _ in range(rows)]
        while live.any():
            current = [sequences[row].current_timestep for row in range(rows)]
            observation = {
                key: np.concatenate([obs[key] for obs, _, _ in current], axis=0)
                for key in current[0][0]
            }
            rl2 = np.concatenate([rl2_row for _, rl2_row, _ in current], axis=0)
            time_index = np.concatenate([times for _, _, times in current], axis=0)
            live_rows = [int(row) for row in np.flatnonzero(live)]
            if count_recall:
                for row in live_rows:
                    values[row].append(
                        cast(Any, bases[row]).decode(
                            {"current": observation["current"][row]}
                        )[0]
                    )
            tensors = {
                key: torch.from_numpy(np.array(value, copy=True))
                .to(device)
                .unsqueeze(1)
                for key, value in observation.items()
            }
            inference_started = time.perf_counter()
            precision = experiment.caster() if uses_history else nullcontext()
            with torch.inference_mode(), precision:
                actions, hidden = policy.get_actions(
                    obs=tensors,
                    rl2s=torch.from_numpy(np.array(rl2, copy=True))
                    .to(device)
                    .unsqueeze(1),
                    time_idxs=torch.from_numpy(np.array(time_index, copy=True))
                    .to(device)
                    .unsqueeze(1),
                    sample=sample_actions,
                    hidden_state=hidden,
                )
            inference_total += time.perf_counter() - inference_started
            if decision_probe is not None:
                decision_probe.after_policy(live_rows, steps, hidden)
            segment = getattr(hidden, "segment", None) if track_writes else None
            if segment is not None:
                # Boundaries are crossed before a decision's record is read, so
                # the counter after the call is the writes before the decision.
                counters = segment.detach().cpu().numpy().reshape(-1)
                for row in live_rows:
                    writes[row].append(int(counters[row]))
            chosen = actions.squeeze(1).cpu().numpy()
            if decision_probe is not None:
                decision_probe.after_actions(live_rows, steps, observation, chosen)
            resets = np.zeros(rows, dtype=np.bool_)
            for row in live_rows:
                action = chosen[row]
                if isinstance(bases[row], MatchPatternEnv):
                    action = cast(MatchPatternEnv, bases[row]).canonical_action(action)
                action_index = int(np.asarray(action).reshape(-1)[0])
                if concentration_probe is not None:
                    board_env = cast(ConcentrationEnv, bases[row])
                    targets, reveal = board_env.retrieval_targets()
                    if targets:
                        assert reveal is not None
                        concentration_probe.append(
                            {
                                "task_id": chunk[row],
                                "flip": int(steps[row]) + 1,
                                "prediction": action_index,
                                "eligible": len(targets),
                                "hit": int(action_index in targets),
                                "evidence_age_flips": int(steps[row]) - reveal,
                            }
                        )
                    # Same public history across cells: twice each position 0..25,
                    # then repeat. The policy prediction never changes the script.
                    action_index = (int(steps[row]) // 2) % (board_env.cards // 2)
                    action = np.full_like(action, action_index)
                if count_probe is not None:
                    stream_env = cast(CountRecallEnv, bases[row])
                    value, query, _ = stream_env.decode(
                        {"current": observation["current"][row]}
                    )
                    index = int(steps[row]) + 1
                    boundary_index = ((index - 1) // 32) * 32
                    truth = values[row].count(query)
                    digest = hashlib.sha256()
                    for key in sorted(observation):
                        digest.update(observation[key][row].tobytes())
                    digest.update(rl2[row].tobytes())
                    digest.update(time_index[row].tobytes())
                    count_probe.append(
                        {
                            "task_id": chunk[row],
                            "query_index": index,
                            "value": value,
                            "query": query,
                            "true_count": truth,
                            "prediction": action_index,
                            "correct": int(action_index == truth),
                            "absolute_error": abs(action_index - truth),
                            "count_before_segment": values[row][:boundary_index].count(
                                query
                            ),
                            "count_in_segment": values[row][boundary_index:].count(
                                query
                            ),
                            "input_sha256": digest.hexdigest(),
                        }
                    )
                    action_index = (
                        0  # same RL2/action/reward history for every checkpoint
                    )
                    action = np.full_like(action, 0)
                action_counts[action_index] = action_counts.get(action_index, 0) + 1
                post_intervention_decisions += int(episode_intervened[row])
                before = np.asarray(observation["current"][row]).reshape(-1)[:2].copy()
                _, _, terminated, truncated, info = sequences[row].step(action)
                if mazerunner:
                    # Evaluator-only native mapping distinguishes no-op from
                    # blocked movement under randomized controls. Never passed
                    # to the policy.
                    dirs = np.asarray(
                        cast(Any, bases[row]).native.unwrapped.action_dirs
                    )
                    if np.any(dirs[action_index]):
                        movement_attempts += 1
                        after_obs, _, _ = sequences[row].current_timestep
                        after = np.asarray(after_obs["current"]).reshape(-1)[:2]
                        blocked_moves += int(np.array_equal(before, after))
                done = bool(np.logical_or(terminated, truncated).reshape(-1)[0])
                steps[row] += 1
                decisions += 1
                if boundary is not None and not done:
                    reset = boundary(info)
                    intervention_count += int(np.count_nonzero(reset))
                    if np.any(reset):
                        episode_intervened[row] = True
                        resets[row] = True
                if done:
                    live[row] = False
            if resets.any():
                hidden = encoder.reset_hidden_state(hidden, resets)
        intervention_episodes += int(np.count_nonzero(episode_intervened))
        for row in range(rows):
            clocks_after = bases[row].collection_counters()
            earlier = clocks_before[row]
            charged += clocks_after["charged_calls"] - earlier["charged_calls"]
            physical += clocks_after["physical_actions"] - earlier["physical_actions"]
            reset_only += clocks_after["reset_only_steps"] - earlier["reset_only_steps"]
        for row, task_id in enumerate(chunk):
            results.append(
                _extract(
                    bases[row],
                    task_id,
                    rollout_seed,
                    int(steps[row]),
                    writes=tuple(writes[row]) if writes[row] else None,
                    values=tuple(values[row]),
                )
            )
    elapsed = time.perf_counter() - started
    if charged != decisions:
        raise ContractError(
            "The environments' charged calls disagree with the evaluator's decisions."
        )
    metrics = {
        "runtime_seconds": elapsed,
        "evaluation_batch": float(width),
        "sample_actions": float(sample_actions),
        "charged_calls": float(charged),
        "physical_actions": float(physical),
        "reset_only_steps": float(reset_only),
        "intervention_count": float(intervention_count),
        "intervention_episodes": float(intervention_episodes),
        "post_intervention_decisions": float(post_intervention_decisions),
        "movement_attempts": float(movement_attempts),
        "blocked_moves": float(blocked_moves),
        "blocked_move_fraction": blocked_moves / max(movement_attempts, 1),
        **{
            f"action_{index}_count": float(count)
            for index, count in action_counts.items()
        },
        "inference_ms": 1000.0 * inference_total / max(decisions, 1),
        "rollout_ms": 1000.0 * elapsed / max(decisions, 1),
    }
    return tuple(results), metrics


def scripted_count_recall(
    experiment: Any,
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    split: str,
    checkpoint: str,
    history: str = "retained",
    task_cap: int | None = None,
) -> dict[str, object]:
    """Predictions under a fixed answer-zero script; RL2 and timer stay native."""
    if contract.protocol != "count-recall-medium":
        raise ContractError("The R8 shared-stream probe requires official Medium.")
    roster = contract.roster(split)
    if task_cap is not None:
        roster = roster[:task_cap]
    width = min(evaluation_batch(None), len(roster))
    environments = [
        amago_environment(
            evaluation_environment(contract, config, split=split, seed=0),
            name="CountRecallScripted",
        )
        for _ in range(width)
    ]
    probe: list[dict[str, object]] = []
    try:
        _, metrics = rollout(
            experiment,
            environments,
            task_ids=roster,
            rollout_seed=0,
            history=history,
            count_probe=probe,
        )
    finally:
        for environment in environments:
            environment.close()
    return {
        "schema": "count-recall-medium-scripted-counts.v1",
        "protocol": contract.protocol,
        "condition": config.condition,
        "seed": config.seed,
        "checkpoint": checkpoint,
        "split": split,
        "history": history,
        "task_ids": list(roster),
        "partial_task_cap": task_cap,
        "script": "answer=0; predictions do not affect public histories; C32 bins",
        "charged_calls": int(metrics["charged_calls"]),
        "queries": probe,
    }


def scripted_concentration_retrieval(
    experiment: Any,
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    split: str,
    checkpoint: str,
    task_cap: int | None = None,
) -> dict[str, object]:
    """Frozen partner predictions on identical scripted public histories."""
    if contract.name != "concentration":
        raise ContractError("The scripted retrieval diagnostic requires Concentration.")
    roster = contract.roster(split)
    if task_cap is not None:
        roster = roster[:task_cap]
    width = min(evaluation_batch(None), len(roster))
    environments = [
        amago_environment(
            evaluation_environment(contract, config, split=split, seed=0),
            name="ConcentrationScripted",
        )
        for _ in range(width)
    ]
    probe: list[dict[str, int]] = []
    try:
        _, metrics = rollout(
            experiment,
            environments,
            task_ids=roster,
            rollout_seed=0,
            concentration_probe=probe,
        )
    finally:
        for environment in environments:
            environment.close()
    return {
        "schema": "concentration-scripted-retrieval.v1",
        "protocol": contract.protocol,
        "condition": config.condition,
        "seed": config.seed,
        "checkpoint": checkpoint,
        "split": split,
        "task_ids": list(roster),
        "partial_task_cap": task_cap,
        "script": "position=(zero_based_flip//2)%26; 104 flips; retained history",
        "charged_calls": int(metrics["charged_calls"]),
        "opportunities": probe,
        "opportunity_count": len(probe),
        "hits": sum(p["hit"] for p in probe),
    }


def _environment_name(env: BaseEnv) -> str:
    from reasoned_icrl.environments import (
        CountRecallEnv,
        DarkKeyToDoorEnv,
        DarkRoomEnv,
        MazeRunnerEnv,
        XLandMiniGridEnv,
        XLandOneRuleEnv,
    )

    # The one-rule task subclasses the broad adapter: test it first.
    for name, kind in (
        ("match_pattern", MatchPatternEnv),
        ("darkroom", DarkRoomEnv),
        ("dark_key_to_door", DarkKeyToDoorEnv),
        ("count_recall", CountRecallEnv),
        ("mazerunner", MazeRunnerEnv),
        ("tmaze", TMazeEnv),
        ("concentration", ConcentrationEnv),
        ("xland_one_rule", XLandOneRuleEnv),
        ("xland_minigrid", XLandMiniGridEnv),
    ):
        if isinstance(env, kind):
            return name
    raise ContractError(f"Unknown environment class: {type(env).__name__}.")


def evaluate(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    experiment: Any,
    *,
    checkpoint: str,
    split: str,
    history: str = "retained",
    sample_actions: bool = False,
    task_cap: int | None = None,
    prior: Sequence[int] | None = None,
    checkpoint_rule: CheckpointRule = "selected",
    batch_size: int | None = None,
    layout_period: int | None = None,
    decision_probe: DecisionProbe | None = None,
) -> tuple[BenchmarkRun, tuple[BenchmarkEvent, ...], dict[str, float | None]]:
    """Evaluate one checkpoint over a declared roster and its rollout seed.

    ``decision_probe`` (evaluation only) is handed to :func:`rollout`; it
    observes the policy calls and changes nothing that is recorded here.

    ``checkpoint_rule`` names which rule chose ``checkpoint`` and travels on
    the run and every event, so the development-selected panel and the
    fixed-final-checkpoint supplement are validated as separate panels.
    ``batch_size`` tasks are rolled out at once (:func:`evaluation_batch`);
    the width is recorded in the run's metrics as ``evaluation_batch``.
    ``layout_period``
    (evaluation only, Key-to-Door) replaces the hidden layout at the first
    attempt boundary after every that many calls; the environment must
    support ``set_layout_period``.
    """
    if checkpoint_rule not in CHECKPOINT_RULES:
        raise ContractError(f"Unknown checkpoint rule: {checkpoint_rule!r}.")
    if sample_actions and contract.name != "mazerunner":
        raise ContractError("Sampled actions are only a MazeRunner policy diagnostic.")
    allowed_modes = continued_history_modes(contract.environment)
    if history not in allowed_modes:
        raise ContractError(
            f"{contract.name} history modes: {', '.join(allowed_modes)}"
        )
    plan = contract.evaluation
    if len(plan.rollout_seeds) != 1:
        raise ContractError("Every protocol declares exactly one rollout seed.")
    rollout_seed = plan.rollout_seeds[0]
    roster = contract.roster(split)
    if task_cap is not None:
        roster = roster[:task_cap]
    width = min(evaluation_batch(batch_size), max(len(roster), 1))
    wrapped = []
    for _ in range(width):
        environment = evaluation_environment(
            contract, config, split=split, seed=rollout_seed
        )
        if layout_period is not None:
            setter = getattr(environment, "set_layout_period", None)
            if setter is None:
                raise ContractError(
                    f"{contract.name!r} has no layout-change continuation."
                )
            setter(int(layout_period))
        wrapped.append(
            amago_environment(
                environment,
                name=f"{type(environment).__name__}-{split}",
                seed=rollout_seed,
            )
        )
    try:
        results, metrics = rollout(
            experiment,
            wrapped,
            task_ids=roster,
            rollout_seed=rollout_seed,
            history=history,
            sample_actions=sample_actions,
            decision_probe=decision_probe,
        )
    finally:
        for env in wrapped:
            env.close()
    rows = events(
        contract,
        config,
        results,
        checkpoint=checkpoint,
        split=split,
        history=history,
        checkpoint_rule=checkpoint_rule,
        layout_period=layout_period,
    )
    parameters = sum(
        p.numel() for p in experiment.policy.parameters() if p.requires_grad
    )
    run = run_record(
        contract,
        config,
        checkpoint=checkpoint,
        split=split,
        history=history,
        metrics=metrics,
        parameter_count=parameters,
        checkpoint_rule=checkpoint_rule,
        layout_period=layout_period,
    )
    if contract.name == "match_pattern":
        from reasoned_icrl.experiments.match_pattern import manifest_hash

        path = (
            config.run_directory / checkpoint
            if checkpoint.endswith(".pt")
            else config.run_directory / "ckpts" / "policy_weights" / f"{checkpoint}.pt"
        )
        run = replace(
            run,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            corpus_sha256=manifest_hash(),
        )
    summary: dict[str, float | None] = {
        **metrics,
        **secondary(config.environment, results, prior=prior),
    }
    return run, rows, summary


def checkpoint_series(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    experiment: Any,
    *,
    epochs: Sequence[int],
    task_cap: int | None = None,
) -> tuple[CheckpointScore, ...]:
    """Evaluate the development split at each scheduled training checkpoint."""
    if not epochs:
        raise ContractError("A C2 decision needs at least one checkpoint.")
    scores: list[CheckpointScore] = []
    for epoch in epochs:
        started = time.perf_counter()
        if epoch == -1:
            from reasoned_icrl.runtime.training import load_initial_checkpoint

            load_initial_checkpoint(experiment, config)
        else:
            experiment.load_checkpoint(int(epoch), resume_training_state=False)
        checkpoint = "initial_checkpoint.pt" if epoch == -1 else f"policy_epoch_{epoch}"
        run, events, secondary_values = evaluate(
            contract,
            config,
            experiment,
            checkpoint=checkpoint,
            split="development",
            task_cap=task_cap,
        )
        calls = None
        digest = None
        if contract.name == "match_pattern":
            import json

            path = (
                config.run_directory / "initial_checkpoint.pt"
                if epoch == -1
                else config.run_directory
                / "ckpts"
                / "policy_weights"
                / f"{checkpoint}.pt"
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            calls = 0 if epoch == -1 else None
            if epoch != -1:
                row = json.loads(path.with_suffix(".metadata.json").read_text())
                if row["checkpoint_sha256"] != digest:
                    raise ContractError(
                        "Checkpoint weights disagree with measured-counter metadata."
                    )
                calls = int(row["charged_calls"])
            # Store every full-panel checkpoint separately for A1, without
            # replacing the development-selected or endpoint panels.
            write_evaluation(
                config.run_directory / "development_series" / str(epoch),
                contract,
                run,
                events,
                secondary_values,
                split="development",
                history="retained",
                task_cap=task_cap,
            )
        scores.append(
            CheckpointScore(
                epoch=int(epoch),
                primary=primary_from_events(events, contract.evaluation.primary_metric),
                evaluation_seconds=time.perf_counter() - started,
                charged_calls=calls,
                checkpoint_sha256=digest,
            )
        )
    return tuple(scores)


__all__ = [
    "BOUNDARIES",
    "checkpoint_series",
    "evaluate",
    "evaluation_batch",
    "rollout",
    "scripted_concentration_retrieval",
    "scripted_count_recall",
]
