"""Reference and checkpoint-score records of the qualification gates.

The pure half of the C1-C3 machinery: the evaluation-only references a
benchmark's designated full-history cell must beat, the task-side diagnostic
record, the per-checkpoint development score, and the label arithmetic over a
run's retained policy weights. Nothing here loads a learner: the checkpoint
series that scores a trained policy and the task diagnostics that drive an
AMAGO sequence wrapper live in :mod:`reasoned_icrl.runtime.rollout` and
:mod:`reasoned_icrl.runtime.diagnostics`. The decisions themselves (C2/C3 on
the 0.4M grid, the plateau screen) are the study's, in
:mod:`reasoned_icrl.experiments.summary_memory.revised`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from reasoned_icrl.environments.base import TRAINING_TASKS
from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.environments import build_environment
from reasoned_icrl.experiments.evaluation import (
    CountRecallStreamResult,
    count_prior,
    count_recall_actions,
    cue_blind_tmaze_success,
    evaluation_environment,
    prior_accuracy,
    random_policy_attempt_success,
    random_policy_goal_fraction,
    random_policy_pair_fraction,
    random_reference_accuracy,
)
from reasoned_icrl.experiments.records import BenchmarkEvent, primary_value


@dataclass(frozen=True, slots=True)
class TaskDiagnostic:
    """One benchmark's task-side C3 evidence."""

    benchmark: str
    question: str
    passed: bool
    measurements: Mapping[str, float]
    statement: str

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "question": self.question,
            "passed": self.passed,
            "measurements": dict(self.measurements),
            "statement": self.statement,
        }


def primary_from_events(
    events: Sequence[BenchmarkEvent], metric: str | None = None
) -> float:
    """The declared primary metric, averaged equally over evaluation units.

    With no metric named this is the legacy mean of ``numerator / denominator``
    over every event (Key-to-Door's eight scored attempts, CountRecall's
    queries, MazeRunner's goal fraction). A named metric follows its own
    membership and aggregation (R4): ``doors_completed`` sums every retained
    attempt's successes per task before averaging tasks, and
    ``door_success_first8`` reads only the first eight completed attempts of a
    complete record.
    """
    if not events:
        raise ContractError("The primary metric needs at least one event.")
    if metric is None:
        return float(np.mean([event.numerator / event.denominator for event in events]))
    return primary_value(events, metric)


@dataclass(frozen=True, slots=True)
class Reference:
    """One declared useful reference the baseline must beat."""

    name: str
    primary: float
    kind: str
    note: str
    complete: bool = True


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    """The development primary metric at one scheduled checkpoint."""

    epoch: int
    primary: float
    evaluation_seconds: float
    charged_calls: int | None = None
    checkpoint_sha256: str | None = None


def measure_reference(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    task_cap: int | None = None,
    generator_seed: int = 0,
) -> Reference:
    """Measure the best declared evaluation-only reference for one benchmark.

    Every contract names a random-policy floor; CountRecall additionally names a
    development-fitted count/time prior. The trained current-token reference it
    also names is combined by the CLI after this function. Until that evidence
    is supplied, complete=False prevents a CountRecall C2 pass.
    """
    name = contract.environment.name
    roster = contract.roster("development")
    if task_cap is not None:
        roster = roster[:task_cap]
    if name == "count_recall":
        # Fit the prior on the disjoint training identity band and score it on
        # the development roster. Fitting and scoring on the same streams would
        # inflate the reference and make C2 artificially hard to clear.
        fitting = tuple(TRAINING_TASKS[: len(roster)])
        prior = count_prior(
            _count_recall_streams(config, fitting, split="train"),
            horizon=contract.environment.horizon,
        )
        held_out = prior_accuracy(prior, _count_recall_streams(config, roster))
        actions = count_recall_actions(contract.protocol)
        uniform = random_reference_accuracy(actions)
        if held_out >= uniform:
            return Reference(
                name="held-out count/time prior",
                primary=held_out,
                kind="evaluation-only",
                complete=False,
                note=(
                    f"Position-only modal count fitted on {len(fitting)} training-band "
                    "streams and scored on the development roster, so the two sets are "
                    "disjoint. This positional prior is not information-free; "
                    "the trained reference separately reads current observation "
                    "and RL2."
                ),
            )
        return Reference(
            name="uniform random answer",
            primary=uniform,
            kind="evaluation-only",
            note=(
                f"One of {actions} answers, chosen uniformly; trained "
                "current-token reference missing."
            ),
            complete=False,
        )
    environment = evaluation_environment(contract, config, split="development", seed=0)
    try:
        if name == "mazerunner":
            measured = random_policy_goal_fraction(
                cast(Any, environment), task_ids=roster, generator_seed=generator_seed
            )
            return Reference(
                name="random-policy goal fraction",
                primary=float(measured["random_reference_goal_fraction"]),
                kind="evaluation-only",
                note="Uniform actions on the same development maps.",
            )
        if name == "concentration":
            measured = random_policy_pair_fraction(
                cast(Any, environment), task_ids=roster, generator_seed=generator_seed
            )
            return Reference(
                name="random-policy pair fraction",
                primary=float(measured["random_reference_pair_fraction"]),
                kind="evaluation-only",
                note="Uniform flips on the same development boards.",
            )
        if name == "tmaze":
            measured = cue_blind_tmaze_success(
                cast(Any, environment), task_ids=roster, generator_seed=generator_seed
            )
            blind = float(measured["cue_blind_reference_goal_success"])
            random = float(measured["random_reference_goal_success"])
            return Reference(
                name="cue-blind forward-then-turn success",
                primary=max(blind, random),
                kind="evaluation-only",
                note=(
                    "Forward along the corridor and a fixed upward turn at the "
                    f"junction: {blind:.3f} on the development roster (the fraction "
                    "of tasks whose cue is up); uniform random actions "
                    f"{random:.3f}. The comparison level is the larger, so C2 asks "
                    "the reference to beat the best memoryless policy by delta."
                ),
            )
        if name in ("xland_minigrid", "xland_one_rule"):
            measured = random_policy_attempt_success(
                cast(Any, environment), task_ids=roster, generator_seed=generator_seed
            )
            return Reference(
                name="random-policy success on the scored episodes",
                primary=float(measured["random_reference_scored_attempt_success"]),
                kind="evaluation-only",
                note=(
                    "Uniform actions on the same development rulesets and layout "
                    "schedule, averaged over episodes 4-5; first-episode success "
                    f"{measured['random_reference_first_attempt_success']:.3f}."
                ),
            )
        success = _random_attempts(environment, roster, generator_seed)
    finally:
        environment.close()
    return Reference(
        name="random-policy scored-attempt success",
        primary=success,
        kind="evaluation-only",
        note="Uniform actions on the same development tasks.",
    )


def _count_recall_streams(
    config: ExperimentConfig, roster: Sequence[int], *, split: str = "development"
) -> tuple[CountRecallStreamResult, ...]:
    environment = build_environment(config.as_runtime_mapping(), split=split, seed=0)
    results = []
    try:
        for stream_id in roster:
            environment.set_task(int(stream_id))
            environment.reset()
            done = False
            while not done:
                _, _, terminated, truncated, _ = environment.step(0)
                done = terminated or truncated
            results.append(
                CountRecallStreamResult(
                    task_id=int(stream_id),
                    rollout_seed=0,
                    queries=cast(Any, environment).scored_queries,
                    stream_return=cast(Any, environment).stream_return,
                    decisions=cast(Any, environment).horizon,
                )
            )
    finally:
        environment.close()
    return tuple(results)


def _random_attempts(
    environment: Any, roster: Sequence[int], generator_seed: int
) -> float:
    """Mean scored-attempt success of uniform actions on an attempt task."""
    generator = np.random.default_rng(generator_seed)
    scored = environment.attempts
    first = int(getattr(environment, "scored_from", 1))
    fractions: list[float] = []
    for task_id in roster:
        environment.set_task(int(task_id))
        environment.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = environment.step(
                int(generator.integers(environment.action_count))
            )
            done = terminated or truncated
        attempts = environment.completed_attempts[:scored]
        if len(attempts) < scored:
            raise ContractError("The reference roster lost a scored attempt.")
        attempts = attempts[first - 1 :]
        fractions.append(float(np.mean([record.success for record in attempts])))
    return float(np.mean(fractions))


def scheduled_epochs(run_directory: Path) -> list[int]:
    """The epochs whose policy weights the run retained for selection."""
    weights = run_directory / "ckpts" / "policy_weights"
    return sorted(
        int(path.stem.rsplit("_", 1)[1]) for path in weights.glob("policy_epoch_*.pt")
    )


__all__ = [
    "CheckpointScore",
    "Reference",
    "TaskDiagnostic",
    "measure_reference",
    "primary_from_events",
    "scheduled_epochs",
]
