"""Passive T-Maze: native boundaries, the exact budget, leakage, records and the
C1 lifecycle (bounded-summary paper, third benchmark).

These are execution-integrity checks. Nothing here trains to convergence or
claims that the pinned task is learnable at 8M.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from amago.envs.amago_env import SequenceWrapper

from reasoned_icrl.environments.base import (
    CONFIRMATION_TASKS,
    DEVELOPMENT_TASKS,
    TRAINING_TASKS,
)
from reasoned_icrl.environments.tmaze import (
    BACK,
    DOWN,
    FORWARD,
    TMAZE_CORRIDOR,
    TMAZE_MOVEMENT_PENALTY,
    TMAZE_V3_PROTOCOL,
    TMAZE_V3_TRAINING_CORRIDORS,
    UP,
    TMazeEnv,
    TMazeEpisode,
    corridor_for_task,
    tmaze_horizon,
)
from reasoned_icrl.experiments.benchmarks import experiment_config, load_contract
from reasoned_icrl.experiments.contracts import (
    ContractError,
    ResultValidationError,
    architecture_uses_history_packet,
)
from reasoned_icrl.experiments.evaluation import (
    SUMMARY_CLEARED,
    TMazeEpisodeResult,
    cue_blind_tmaze_success,
    evaluation_environment,
    events,
    history_modes,
    run_record,
    secondary,
)
from reasoned_icrl.experiments.horizon import (
    check_episode_panel,
    extended_horizon,
)
from reasoned_icrl.experiments.qualification import measure_reference
from reasoned_icrl.experiments.records import validate_benchmark_results
from reasoned_icrl.experiments.summary_memory.configs import load_tmaze_v3_study
from reasoned_icrl.runtime.diagnostics import task_diagnostic, tmaze_diagnostic
from reasoned_icrl.runtime.environments import amago_environment
from reasoned_icrl.runtime.replay import create_replay_dataset
from reasoned_icrl.runtime.rollout import evaluate, rollout
from reasoned_icrl.runtime.training import load_experiment, train_experiment

ROOT = Path(__file__).resolve().parents[2]
PAPER_CELLS = (
    "full_context",
    "full_gru",
    "raw_summary",
    "raw_segment",
    "memo",
    "memo_fixed",
)


def contract() -> Any:
    return load_tmaze_v3_study().contract("tmaze")


def resolved(condition: str = "raw_summary", **overrides: Any) -> Any:
    settings: dict[str, Any] = {"seed": 42, "repository": ROOT, "device": "cpu"}
    settings.update(overrides)
    return experiment_config(
        contract(), load_tmaze_v3_study(), condition=condition, **settings
    )


def environment(corridor: int = 8, **overrides: Any) -> TMazeEnv:
    settings: dict[str, Any] = {
        "corridor_length": corridor,
        "split": "development",
        "initial_seed": 0,
    }
    settings.update(overrides)
    return TMazeEnv(**settings)


def walk(
    env: TMazeEnv, task: int, turn: int | None = None, *, waste_at: int | None = None
):
    """Forward along the corridor, then ``turn`` (the cue when None); one
    wasted lateral move at step ``waste_at`` if given. Returns the episode."""
    packet, _ = env.reset(options={"task_index": task})
    cue = env.public_fields(packet)["cue_or_lateral"]
    done = False
    step = 0
    while not done:
        step += 1
        fields = env.public_fields(packet)
        if fields["at_junction"]:
            action = ((cue if turn is None else turn) == 1 and UP) or DOWN
        elif waste_at == step:
            action = UP
        else:
            action = FORWARD
        packet, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    assert env.episode is not None
    return env.episode


# ----------------------------------------------------------------------
# Native boundaries and the exact budget
# ----------------------------------------------------------------------


def test_the_cue_is_shown_once_and_the_junction_is_observable() -> None:
    env = environment(8)
    try:
        packet, info = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        fields = env.public_fields(packet)
        assert fields["at_junction"] == 0 and fields["cue_or_lateral"] in (-1, 1)
        assert env.cue == fields["cue_or_lateral"]
        assert packet["event"].tolist() == [0, 1, 0]
        assert not packet["previous"].any() and not packet["outcome"].any()
        assert info["step_in_task"] == 0 and not info["at_junction"]
        for step in range(1, 8):
            packet, reward, terminated, truncated, info = env.step(FORWARD)
            assert reward == 0.0 and not terminated and not truncated
            assert env.public_fields(packet) == {"at_junction": 0, "cue_or_lateral": 0}
            assert packet["event"].tolist() == [1, 0, 0]
            np.testing.assert_array_equal(packet["outcome"], packet["current"])
            assert info["step_in_task"] == step and not info["at_junction"]
        packet, reward, terminated, truncated, info = env.step(FORWARD)
        assert reward == 0.0 and not terminated and not truncated
        assert env.public_fields(packet) == {"at_junction": 1, "cue_or_lateral": 0}
        assert info["at_junction"] and info["step_in_task"] == 8
        turn = UP if env.cue == 1 else DOWN
        packet, reward, terminated, truncated, info = env.step(turn)
        assert reward == 1.0 and terminated and truncated
        assert env.public_fields(packet) == {
            "at_junction": 1,
            "cue_or_lateral": env.cue,
        }
        assert packet["event"].tolist() == [1, 0, 1]
        assert info["episode_done"] and info["episode_success"]
        episode = env.episode
        assert episode is not None and episode.success
        assert (episode.steps, episode.junction_step, episode.forward_moves) == (
            9,
            8,
            8,
        )
        assert episode.native_return == 1.0 and episode.cue == env.cue
        with pytest.raises(ContractError, match="requires an outer reset"):
            env.step(FORWARD)
    finally:
        env.close()


def test_the_budget_has_no_spare_step() -> None:
    """corridor_length forward moves plus one turn is the whole budget, so a
    single lateral move in the corridor fails the episode; every successful
    corridor action is the forward move and cannot encode the cue."""
    env = environment(8)
    try:
        task = DEVELOPMENT_TASKS[3]
        assert tmaze_horizon(8) == 9 == env.horizon
        success = walk(env, task)
        assert success.success and success.forward_moves == 8
        wrong = walk(env, task, turn=-success.cue)
        assert not wrong.success and wrong.lateral == -success.cue
        assert wrong.native_return == 0.0 and wrong.junction_step == 8
        # One wasted lateral move anywhere in the corridor: the junction is
        # first seen on the last decision and no step is left to turn.
        wasted = walk(env, task, waste_at=1)
        assert not wasted.success and wasted.junction_step == 9
        assert wasted.forward_moves == 8 and wasted.lateral == 0
        # v2: the wasted move paid the movement penalty (-1 / corridor); the
        # final decision never does, so the wrong turn above paid nothing.
        assert wasted.penalised_moves == 1
        assert wasted.native_return == pytest.approx(TMAZE_MOVEMENT_PENALTY)
        assert wrong.penalised_moves == 0 and success.penalised_moves == 0
        late = walk(env, task, waste_at=8)
        assert not late.success and late.junction_step == 9
        assert late.steps == 9 and late.lateral == 0
        assert late.penalised_moves == 1
        assert late.native_return == pytest.approx(TMAZE_MOVEMENT_PENALTY)
    finally:
        env.close()


def test_a_backward_move_at_the_start_is_blocked_and_still_costs_a_step() -> None:
    env = environment(4)
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[1]})
        packet, reward, _, _, _ = env.step(BACK)
        # Blocked, still a non-forward move: the paper's penalty (-1 / 128 on
        # every contract, whatever the corridor) is paid.
        assert reward == pytest.approx(TMAZE_MOVEMENT_PENALTY)
        assert env.public_fields(packet) == {"at_junction": 0, "cue_or_lateral": 0}
        assert (
            env.public_fields({"current": packet["previous"]})["cue_or_lateral"]
            == env.cue
        )
        # The blocked move cost the only step: the junction is first seen on
        # the last decision and the episode fails without a turn.
        for _ in range(3):
            packet, *_ = env.step(FORWARD)
            assert env.public_fields(packet)["at_junction"] == 0
        packet, reward, terminated, truncated, _ = env.step(FORWARD)
        assert env.public_fields(packet)["at_junction"] == 1
        assert reward == 0.0 and terminated and truncated
        assert env.episode is not None and not env.episode.success
        assert env.episode.junction_step == 5 and env.episode.forward_moves == 4
        assert env.episode.penalised_moves == 1
        assert env.episode.native_return == pytest.approx(TMAZE_MOVEMENT_PENALTY)
    finally:
        env.close()


def test_the_task_identity_fixes_the_cue_and_both_sides_are_present() -> None:
    env = environment(4, split="confirmation")
    try:
        cues = {}
        for task in CONFIRMATION_TASKS[:64]:
            env.reset(options={"task_index": task})
            cues[task] = env.cue
        for task in list(cues)[:8]:
            env.reset(options={"task_index": task})
            assert env.cue == cues[task]
        assert set(cues.values()) == {-1, 1}
        assert 8 <= sum(cue == 1 for cue in cues.values()) <= 56
    finally:
        env.close()
    other = TMazeEnv(corridor_length=4, split="confirmation", initial_seed=7)
    try:
        for task in list(cues)[:8]:
            other.reset(options={"task_index": task})
            assert other.cue == cues[task]  # the actor seed never changes a task
    finally:
        other.close()


def test_the_packet_carries_only_the_two_public_native_values() -> None:
    env = environment(8)
    try:
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        assert set(packet) == {"current", "previous", "outcome", "event", "valid"}
        assert packet["current"].shape == (2,) and env.width == 2
        assert [field.name for field in env.fields] == ["at_junction", "cue_or_lateral"]
        assert env.contract["environment_protocol"] == TMAZE_V3_PROTOCOL
        assert env.action_count == 4
        for _ in range(8):
            packet, *_ = env.step(FORWARD)
            assert np.all(np.abs(packet["current"]) <= 1.0)
        with pytest.raises(ContractError, match="width"):
            env.public_fields({"current": np.zeros(3, dtype=np.float32)})
    finally:
        env.close()


def test_state_round_trips_mid_corridor() -> None:
    env = environment(8)
    twin = environment(8)
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[5]})
        for _ in range(5):
            env.step(FORWARD)
        state = env.state_dict()  # the runtime checkpoints it with torch, not JSON
        twin.reset(options={"task_index": DEVELOPMENT_TASKS[5]})
        twin.load_state_dict(state)
        assert twin.cue == env.cue and twin.state_dict() == env.state_dict()
        for _ in range(3):
            left, *_ = env.step(FORWARD)
            right, *_ = twin.step(FORWARD)
            np.testing.assert_array_equal(left["current"], right["current"])
        turn = UP if env.cue == 1 else DOWN
        assert env.step(turn)[1] == twin.step(turn)[1] == 1.0
        assert env.episode == twin.episode
        mismatch = environment(9)
        try:
            mismatch.reset(options={"task_index": DEVELOPMENT_TASKS[5]})
            with pytest.raises(ContractError, match="contract does not match"):
                mismatch.load_state_dict(state)
        finally:
            mismatch.close()
    finally:
        env.close()
        twin.close()


def test_episode_records_refuse_inconsistent_outcomes() -> None:
    with pytest.raises(ContractError, match="disagrees with the public outcome"):
        TMazeEpisode(1, -1, True, 9, 8, 8, 1.0)
    with pytest.raises(ContractError, match="native return"):
        TMazeEpisode(1, 1, True, 9, 8, 8, 0.0)
    with pytest.raises(ContractError, match="requires the junction"):
        TMazeEpisode(1, 1, True, 9, None, 8, 1.0)
    penalised = TMazeEpisode(1, 1, True, 9, 8, 7, 1.0 - 0.125, 1, -0.125)
    assert penalised.native_return == pytest.approx(0.875)
    with pytest.raises(ContractError, match="native return"):
        TMazeEpisode(1, 1, True, 9, 8, 7, 1.0, 1, -0.125)
    with pytest.raises(ContractError, match="positive integer"):
        tmaze_horizon(0)


# ----------------------------------------------------------------------
# Contract, study and records
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["environment"].update(horizon=130), "no slack"),
        (lambda d: d["environment"].update(size=64, horizon=65), "corridor length 128"),
        (lambda d: d["environment"].update(attempts=2, horizon=129), "meta-attempt"),
        (
            lambda d: d["environment"].update(benchmark="tmaze-passive-l64-v1"),
            "requires tmaze-passive-l32-256-v3",
        ),
        (lambda d: d["environment"].update(movement_penalty=0.5), "finite value <= 0"),
        (lambda d: d["environment"].pop("movement_penalty"), "declares its movement"),
        (lambda d: d["training"].update(exploration="random_walk"), "exploration"),
        (lambda d: d["evaluation"].update(retention="complete"), "complete retention"),
    ],
)
def test_the_contract_refuses_a_different_corridor_or_budget(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "configs/environments/8m/tmaze_v3.yaml").read_text())
    mutation(raw)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError, match=message):
        load_contract(path)


def test_every_paper_cell_resolves_with_the_locked_memory_geometry() -> None:
    for condition in PAPER_CELLS:
        config = resolved(condition)
        assert config.environment.name == "tmaze"
        assert config.environment.outer_length == 129
        if config.model.summary is not None:
            assert (
                config.model.summary.segment_length,
                config.model.summary.memory_tokens,
            ) == (32, 4)
        if config.model.memo is not None:
            assert (
                config.model.memo.segment_length,
                config.model.memo.summary_tokens,
            ) == (32, 4)
            assert config.model.memo.training_segment_jitter == (
                0.0 if condition == "memo_fixed" else 0.2
            )
    smoke = resolved("raw_summary", smoke=True)
    assert smoke.environment.size == 7 and smoke.environment.horizon == 8
    assert smoke.environment.outer_length == smoke.training.max_sequence_length == 8


def _scripted_results(
    active: Any, config: Any, tasks: list[int], *, wrong: set[int] = frozenset()
):
    env = evaluation_environment(active, config, split="development", seed=0)
    results = []
    try:
        for task in tasks:
            episode = walk(env, task)
            if task in wrong:
                episode = walk(env, task, turn=-episode.cue)
            results.append(TMazeEpisodeResult(task, 0, episode, episode.steps))
    finally:
        env.close()
    return results


def test_events_score_one_episode_per_task_and_validate_against_the_roster() -> None:
    active = contract()
    config = resolved("full_context")
    tasks = list(active.roster("development"))
    wrong = set(tasks[:16])
    results = _scripted_results(active, config, tasks, wrong=wrong)
    rows = events(
        active,
        config,
        results,
        checkpoint="policy_epoch_999",
        split="development",
        history="retained",
        checkpoint_rule="endpoint",
    )
    assert len(rows) == 64 and {row.kind for row in rows} == {"episode"}
    assert all(
        row.step == 129 and (row.start_step, row.end_step) == (1, 129) for row in rows
    )
    assert sum(row.numerator for row in rows) == 48
    assert all(row.native_return == float(row.numerator) for row in rows)
    summary = secondary(active.environment, results)
    assert summary["goal_success"] == pytest.approx(48 / 64)
    assert (
        summary["junction_reached_rate"] == 1.0
        and summary["forward_moves_mean"] == 128.0
    )
    assert summary["tasks_cue_up"] + summary["tasks_cue_down"] == 64.0
    run = run_record(
        active,
        config,
        checkpoint="policy_epoch_999",
        split="development",
        history="retained",
        metrics={
            "runtime_seconds": 1.0,
            "charged_calls": 64 * 129,
            "physical_actions": 64 * 129,
            "reset_only_steps": 0,
        },
        checkpoint_rule="endpoint",
    )
    assert run.outer_length == 129
    validate_benchmark_results([active], [run], list(rows))
    with pytest.raises(ResultValidationError, match="incomplete/unexpected"):
        validate_benchmark_results([active], [run], list(rows[:-1]))
    short = replace(
        results[0], decisions=128, episode=replace(results[0].episode, steps=128)
    )
    with pytest.raises(ResultValidationError, match="exactly 129"):
        events(
            active,
            config,
            [short],
            checkpoint="x",
            split="development",
            history="retained",
        )
    assert check_episode_panel(rows, rows) == {"compared": 64, "units": 64}
    with pytest.raises(ContractError, match="differs at the trained size"):
        check_episode_panel(
            rows, [replace(rows[0], numerator=1 - rows[0].numerator), *rows[1:]]
        )


def test_the_references_are_the_cue_blind_turn_and_uniform_random() -> None:
    active = contract()
    config = resolved("full_context")
    env = evaluation_environment(active, config, split="development", seed=0)
    try:
        measured = cue_blind_tmaze_success(env, task_ids=active.roster("development"))
        ups = sum(walk(env, task).cue == 1 for task in active.roster("development"))
    finally:
        env.close()
    assert measured["cue_blind_reference_goal_success"] == pytest.approx(ups / 64)
    assert measured["random_reference_goal_success"] <= 1 / 64
    reference = measure_reference(active, config)
    assert reference.primary == pytest.approx(max(measured.values()))
    assert reference.name.startswith("cue-blind") and reference.complete


def test_the_junction_input_is_identical_across_tasks_and_the_cue_resolves_it() -> None:
    active = contract()
    config = resolved("full_context")
    diagnostic = task_diagnostic(active, config)
    assert diagnostic.passed
    assert diagnostic.measurements["distinct_junction_inputs"] == 1.0
    assert diagnostic.measurements["public_recall_success_tasks"] == 64.0
    assert diagnostic.measurements["verified_pairs"] >= 32
    assert diagnostic.measurements["tasks_in_verified_pairs"] >= 16
    with pytest.raises(ContractError, match="needs tasks"):
        tmaze_diagnostic(environment(8), task_ids=())


# ----------------------------------------------------------------------
# Extended-horizon adapter
# ----------------------------------------------------------------------


def test_the_corridor_has_no_extended_horizon_adapter() -> None:
    """The T-Maze is evaluated at its trained corridor only."""
    with pytest.raises(ContractError, match="no extended-horizon adapter"):
        extended_horizon(contract(), resolved("memo"), 2048)


# ----------------------------------------------------------------------
# C1 lifecycle
# ----------------------------------------------------------------------


@pytest.mark.parametrize("condition", ["raw_summary", "memo", "full_gru"])
def test_cpu_train_export_evaluate_lifecycle(condition: str, tmp_path: Path) -> None:
    active = contract()
    config = resolved(condition, output_root=tmp_path, smoke=True)
    train_experiment(config)
    run = config.run_directory
    assert (run / "checkpoint.pt").is_file()
    telemetry = [
        json.loads(line)
        for line in (run / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert any(
        row["panel"] == "train-update" and row["gradient_steps"] >= 1
        for row in telemetry
    )
    experiment = load_experiment(config, work_directory=run)
    smoke_contract = replace(
        active, environment=config.environment, training=config.training
    )
    try:
        record, rows, summary = evaluate(
            smoke_contract,
            config,
            experiment,
            checkpoint="checkpoint.pt",
            split="development",
            task_cap=4,
        )
        assert record.status == "completed" and len(rows) == 4
        assert summary["tasks"] == 4.0 and 0.0 <= summary["goal_success"] <= 1.0
        assert record.outer_length == 8
        with pytest.raises(ResultValidationError, match="incomplete/unexpected"):
            validate_benchmark_results([smoke_contract], [record], list(rows))
        for history in ("current-token", SUMMARY_CLEARED):
            if history == SUMMARY_CLEARED and condition == "full_gru":
                continue
            other, other_rows, _ = evaluate(
                smoke_contract,
                config,
                experiment,
                checkpoint="checkpoint.pt",
                split="development",
                history=history,
                task_cap=2,
            )
            assert other.history == history and len(other_rows) == 2
        env = amago_environment(
            evaluation_environment(smoke_contract, config, split="development", seed=0),
            name="check",
            seed=0,
        )
        try:
            forward, _ = rollout(
                experiment,
                env,
                task_ids=[DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[1]],
                rollout_seed=0,
            )
            reverse, _ = rollout(
                experiment,
                env,
                task_ids=[DEVELOPMENT_TASKS[1], DEVELOPMENT_TASKS[0]],
                rollout_seed=0,
            )
        finally:
            env.close()
        assert forward[0].episode == reverse[1].episode
        assert forward[1].episode == reverse[0].episode
        if architecture_uses_history_packet(config.model.architecture_id):
            _assert_dense_and_cached_agree(experiment, smoke_contract, config)
    finally:
        for name in ("train_envs", "val_envs"):
            closer = getattr(getattr(experiment, name, None), "close", None)
            if callable(closer):
                closer()


def test_rollout_and_replay_agree_on_every_causal_field(tmp_path: Path) -> None:
    """One whole episode through the AMAGO boundary, saved and re-read."""
    env = environment(4)
    wrapped = amago_environment(env, name="TMaze-test", seed=0)
    dataset = create_replay_dataset(
        tmp_path, capacity=8, full_tasks=True, reset_only_terminal=False
    )
    sequence = SequenceWrapper(
        wrapped,
        save_trajs_to=dataset.save_new_trajs_to,
        save_every=None,
        save_trajs_as="npz-compressed",
    )
    env.set_task(DEVELOPMENT_TASKS[10])
    observations: list[dict[str, np.ndarray]] = []
    rewards: list[float] = []
    taken: list[int] = []
    packet, _ = sequence.reset()
    observations.append({k: np.asarray(v)[0].copy() for k, v in packet.items()})
    cue = env.cue
    for action in [FORWARD] * 4 + [UP if cue == 1 else DOWN]:
        packet, reward, terminated, truncated, _ = sequence.step(
            np.array([action], dtype=np.int64)
        )
        observations.append({k: np.asarray(v)[0].copy() for k, v in packet.items()})
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        taken.append(action)
        if bool(np.logical_or(terminated, truncated).reshape(-1)[0]):
            break
    sequence.save_finished_trajs()
    dataset.configure(
        items_per_epoch=1,
        max_seq_len=len(taken),
        padded_sampling="none",
        has_edit_rights=True,
    )
    dataset._refresh_files()
    env.close()
    assert rewards == [0.0, 0.0, 0.0, 0.0, 1.0] and len(taken) == 5
    data = dataset.sample_random_trajectory()
    length = len(data)
    assert length == 5 and data.obs["current"].shape[0] == 6
    assert bool(data.dones[-1].item())
    assert data.time_idxs.reshape(-1).tolist() == list(range(6))
    for step, expected in enumerate(observations):
        for name, value in expected.items():
            np.testing.assert_allclose(
                data.obs[name][step].numpy(), value, err_msg=f"{name}@{step}"
            )
    np.testing.assert_allclose(data.rews.reshape(-1).numpy(), rewards)
    for step, action in enumerate(taken):
        expected = np.zeros(5, dtype=np.float32)
        expected[0] = rewards[step]
        expected[1 + action] = 1.0
        np.testing.assert_allclose(data.rl2s[step + 1].numpy(), expected)
    np.testing.assert_allclose(data.rl2s[0].numpy(), np.zeros(5, dtype=np.float32))


def _assert_dense_and_cached_agree(experiment: Any, active: Any, config: Any) -> None:
    env = evaluation_environment(active, config, split="development", seed=0)
    try:
        packets: list[dict[str, np.ndarray]] = []
        feedback: list[np.ndarray] = []
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        packets.append(packet)
        feedback.append(np.zeros(5, dtype=np.float32))
        done = False
        while not done:
            action = UP if env.public_fields(packet)["at_junction"] else FORWARD
            packet, reward, terminated, truncated, _ = env.step(action)
            row = np.zeros(5, dtype=np.float32)
            row[0] = reward
            row[1 + action] = 1.0
            packets.append(packet)
            feedback.append(row)
            done = terminated or truncated
    finally:
        env.close()
    device = experiment.DEVICE
    observation = {
        name: torch.as_tensor(
            np.stack([step[name] for step in packets])[None], device=device
        )
        for name in packets[0]
    }
    rl2 = torch.as_tensor(np.stack(feedback)[None], device=device)
    times = torch.arange(len(packets), device=device).view(1, -1, 1)
    policy = experiment.policy
    carrier = policy.traj_encoder
    with torch.no_grad():
        tokens = policy.tstep_encoder(observation, rl2)
        dense, _ = carrier(tokens, times, None)
        hidden = carrier.init_hidden_state(1, device)
        cached = []
        for step in range(tokens.shape[1]):
            output, hidden = carrier(
                tokens[:, step : step + 1], times[:, step : step + 1], hidden
            )
            cached.append(output)
    torch.testing.assert_close(dense, torch.cat(cached, dim=1), rtol=2e-4, atol=2e-5)


def test_training_tasks_never_draw_an_evaluation_identity() -> None:
    env = TMazeEnv(corridor_length=4, split="train", initial_seed=3)
    try:
        for _ in range(32):
            env.reset()
            assert env.evaluator_task_index in TRAINING_TASKS
    finally:
        env.close()


# ----------------------------------------------------------------------
# v2 exploration schedule
# ----------------------------------------------------------------------


def test_the_tmaze_exploration_holds_low_noise_inside_the_corridor() -> None:
    """AMAGO's T-Maze schedule: 0.5 / L inside the corridor, the annealed
    epsilon on the first decision and the last ``end_window`` decisions."""
    from reasoned_icrl.runtime.amago import (
        SeededTMazeEpsilonGreedy,
        exploration_wrapper_type,
    )

    assert exploration_wrapper_type({"exploration": "tmaze_epsilon_greedy"}) is (
        SeededTMazeEpsilonGreedy
    )
    env = amago_environment(environment(8), name="tmaze", seed=0)
    wrapper = SeededTMazeEpsilonGreedy(
        env, corridor_length=8, eps_start=1.0, eps_end=0.05, steps_anneal=100
    )
    try:
        wrapper.reset()
        wrapper.global_multiplier = np.ones(1)
        eps = {
            step: float(wrapper.current_eps(np.array([step]))[0])
            for step in (0, 1, 4, 5, 8)
        }
        assert eps[0] == pytest.approx(1.0) and eps[8] == pytest.approx(1.0)
        assert eps[5] == pytest.approx(1.0)  # the last end_window decisions
        assert eps[1] == eps[4] == pytest.approx(0.5 / 8)
        state = wrapper.state_dict()
        assert state["schema"] == "seeded-tmaze-epsilon-greedy.v0.3"
    finally:
        env.close()


# ----------------------------------------------------------------------
# v3: the training corridor drawn per task
# ----------------------------------------------------------------------


V3_CELLS = (*PAPER_CELLS, "raw_summary_residual")


def test_v3_draws_the_corridor_per_task_and_keeps_the_exact_budget() -> None:
    lengths = (8, 16, 24)
    env = environment(
        8, corridor_lengths=lengths, split="train", protocol=TMAZE_V3_PROTOCOL
    )
    twin = environment(
        8, corridor_lengths=lengths, split="train", protocol=TMAZE_V3_PROTOCOL
    )
    seen: dict[int, set[int]] = {length: set() for length in lengths}
    for task in TRAINING_TASKS[:48]:
        assert env.corridor_for_task(task) == corridor_for_task(task, lengths)
        assert twin.corridor_for_task(task) == env.corridor_for_task(task)
        episode = walk(env, task)
        assert env.corridor_length == corridor_for_task(task, lengths)
        assert env.horizon == tmaze_horizon(env.corridor_length)
        assert episode.success and episode.steps == env.corridor_length + 1
        assert episode.forward_moves == env.corridor_length
        seen[env.corridor_length].add(episode.cue)
    # Every declared length is drawn and both cues occur at every length: the
    # length draw is independent of the native cue draw.
    assert all(cues == {-1, 1} for cues in seen.values()), seen
    # A wasted move fails a drawn corridor exactly as it fails the fixed one.
    task = next(t for t in TRAINING_TASKS if corridor_for_task(t, lengths) == 24)
    failed = walk(env, task, waste_at=5)
    assert not failed.success and failed.steps == 25 and failed.penalised_moves == 1
    # The identity is the declared geometry, not the live task's corridor.
    assert env._identity() == (8, 9, TMAZE_MOVEMENT_PENALTY, lengths)
    # Without a draw the corridor is fixed whatever the task.
    fixed = environment(8, split="train")
    assert all(fixed.corridor_for_task(t) == 8 for t in TRAINING_TASKS[:8])
    assert fixed._identity() == (8, 9, TMAZE_MOVEMENT_PENALTY)


def test_v3_refuses_a_bad_draw_or_protocol() -> None:
    with pytest.raises(ContractError, match="distinct positive"):
        environment(8, corridor_lengths=(8, 8), protocol=TMAZE_V3_PROTOCOL)
    with pytest.raises(ContractError, match="distinct positive"):
        environment(8, corridor_lengths=(0, 8), protocol=TMAZE_V3_PROTOCOL)
    with pytest.raises(ContractError, match="Unknown T-Maze protocol"):
        environment(8, protocol="tmaze-passive-l128-v9")


def test_v3_state_round_trips_across_corridor_lengths() -> None:
    lengths = (8, 16)
    long_task = next(t for t in TRAINING_TASKS if corridor_for_task(t, lengths) == 16)
    short_task = next(t for t in TRAINING_TASKS if corridor_for_task(t, lengths) == 8)
    env = environment(
        8, corridor_lengths=lengths, split="train", protocol=TMAZE_V3_PROTOCOL
    )
    packet, _ = env.reset(options={"task_index": long_task})
    cue = env.public_fields(packet)["cue_or_lateral"]
    for _ in range(6):
        packet, *_ = env.step(FORWARD)
    snapshot = env.state_dict()
    assert snapshot["corridor_length"] == 16
    # A fresh environment sitting in a short task restores the long one.
    other = environment(
        8, corridor_lengths=lengths, split="train", protocol=TMAZE_V3_PROTOCOL
    )
    other.reset(options={"task_index": short_task})
    assert other.corridor_length == 8
    other.load_state_dict(snapshot)
    assert other.corridor_length == 16 and other.horizon == 17
    done = False
    steps = 6
    while not done:
        steps += 1
        fields = other.public_fields(packet)
        action = ((cue == 1 and UP) or DOWN) if fields["at_junction"] else FORWARD
        packet, _, terminated, truncated, _ = other.step(action)
        done = terminated or truncated
    assert other.episode is not None and other.episode.success and steps == 17
    # A snapshot of an undeclared corridor is refused.
    bad = dict(snapshot)
    bad["corridor_length"] = 12
    with pytest.raises(ContractError, match="undeclared corridor"):
        environment(
            8, corridor_lengths=lengths, split="train", protocol=TMAZE_V3_PROTOCOL
        ).load_state_dict(bad)


def test_the_v3_contract_pins_the_training_draw_and_the_fixed_evaluation() -> None:
    study = load_tmaze_v3_study()
    active = study.contract("tmaze")
    env = active.environment
    assert (active.protocol, env.benchmark, env.size, env.horizon) == (
        TMAZE_V3_PROTOCOL,
        TMAZE_V3_PROTOCOL,
        TMAZE_CORRIDOR,
        129,
    )
    assert env.attempts == 1
    assert env.training_corridors == TMAZE_V3_TRAINING_CORRIDORS
    assert env.outer_length == 129 and env.longest_training_length == 257
    assert env.movement_penalty == pytest.approx(-1 / 128) == TMAZE_MOVEMENT_PENALTY
    assert active.training.exploration == "tmaze_epsilon_greedy"
    assert active.event_kind == "episode"
    assert active.evaluation.primary_metric == "goal_success"
    assert active.evaluation.retention == "scored-band"
    assert set(active.evaluation.splits) == {"development", "final", "confirmation"}
    assert active.memory is not None and active.memory["summary"] == {
        "segment_length": 32,
        "memory_tokens": 4,
    }
    assert history_modes("tmaze") == ("retained", "current-token", SUMMARY_CLEARED)
    assert active.training.max_sequence_length == 257
    assert active.training.trajectory_length == 258
    assert active.training.exploration_rollout_horizon == 257
    assert active.training.validation_timesteps == 129
    assert (
        active.training.epochs * active.training.timesteps_per_epoch * env.parallel_envs
        == 8_000_000
    )
    assert active.roster("confirmation") == tuple(CONFIRMATION_TASKS)
    plan = study.tier("tmaze")
    assert plan.primary == V3_CELLS
    # T3: the gated write rides beside the paper cells.
    assert plan.supplementary == ("raw_summary_gated",)
    assert plan.practical_effect == 0.25
    assert [c.name for c in plan.contrasts][:7] == [
        "RSM - RSM no carry",
        "full history - RSM",
        "RSM - Memo",
        "RSM - Memo fixed",
        "Memo - Memo fixed",
        "RSM - GRU",
        "RSM - RSM overwrite",
    ]
    assert plan.contrasts[0].left == "raw_summary_residual"
    # Training tasks draw their corridor; every evaluation split keeps 128.
    config = experiment_config(
        active,
        study,
        condition="raw_summary_residual",
        seed=42,
        repository=ROOT,
        device="cpu",
    )
    from reasoned_icrl.experiments.environments import build_environment

    mapping = config.as_runtime_mapping()
    train_env = build_environment(mapping, split="train", seed=0)
    assert isinstance(train_env, TMazeEnv)
    assert train_env.corridor_lengths == TMAZE_V3_TRAINING_CORRIDORS
    assert train_env.protocol == TMAZE_V3_PROTOCOL
    drawn = {train_env.corridor_for_task(t) for t in TRAINING_TASKS[:64]}
    assert len(drawn) >= 6 and drawn <= set(TMAZE_V3_TRAINING_CORRIDORS)
    for split in ("development", "confirmation"):
        held = build_environment(mapping, split=split, seed=0)
        assert isinstance(held, TMazeEnv) and held.corridor_lengths is None
        assert held.corridor_length == 128 and held.horizon == 129


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["environment"].pop("training_corridors"), "declares its training"),
        (
            lambda d: d["environment"].update(training_corridors=[32, 32, 64]),
            "distinct positive",
        ),
        (
            lambda d: d["environment"].update(training_corridors=[160, 192, 224, 256]),
            "inside the training draw",
        ),
        (lambda d: d["environment"].update(size=64, horizon=65), "corridor length 128"),
        (
            lambda d: d["training"].update(
                max_sequence_length=129, trajectory_length=130
            ),
            "complete outer-task replay",
        ),
    ],
)
def test_the_v3_contract_refuses_a_bad_draw(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "configs/environments/8m/tmaze_v3.yaml").read_text())
    mutation(raw)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw))
    if "replay" in message:
        # The sequence rule is the resolved config's, not the contract loader's.
        active = load_contract(path)
        with pytest.raises(ContractError, match=message):
            experiment_config(
                active,
                load_tmaze_v3_study(),
                condition="raw_summary_residual",
                seed=42,
                repository=ROOT,
                device="cpu",
            )
        return
    with pytest.raises(ContractError, match=message):
        load_contract(path)


def test_the_v3_exploration_follows_the_live_corridor() -> None:
    """The low-noise window is the live task's corridor, so a short corridor
    keeps the ordinary schedule at its own junction."""
    from reasoned_icrl.runtime.amago import SeededTMazeEpsilonGreedy

    lengths = (8, 16)
    base = environment(
        8, corridor_lengths=lengths, split="train", protocol=TMAZE_V3_PROTOCOL
    )
    env = amago_environment(base, name="tmaze", seed=0)
    wrapper = SeededTMazeEpsilonGreedy(
        env, corridor_length=16, eps_start=1.0, eps_end=0.05, steps_anneal=100
    )
    try:
        wrapper.reset()
        wrapper.global_multiplier = np.ones(1)
        for length in lengths:
            task = next(
                t for t in TRAINING_TASKS if corridor_for_task(t, lengths) == length
            )
            base.reset(options={"task_index": task})
            assert wrapper.live_corridor() == length
            eps = {
                step: float(wrapper.current_eps(np.array([step]))[0])
                for step in (1, length - 4, length - 3, length)
            }
            assert eps[1] == eps[length - 4] == pytest.approx(0.5 / length)
            assert eps[length - 3] == pytest.approx(1.0) and eps[
                length
            ] == pytest.approx(1.0)
    finally:
        env.close()


@pytest.mark.parametrize("condition", ["raw_summary_residual", "memo"])
def test_v3_cpu_lifecycle_trains_over_drawn_corridors(
    condition: str, tmp_path: Path
) -> None:
    """The v3 smoke profile draws three short corridors per task: training,
    export and evaluation run end to end over variable episode lengths, the
    context and replay sized to the longest, evaluation at the fixed corridor."""
    study = load_tmaze_v3_study()
    active = study.contract("tmaze")
    config = experiment_config(
        active,
        study,
        condition=condition,
        seed=42,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert config.environment.training_corridors == (3, 5, 7)
    assert config.environment.size == 7 and config.environment.outer_length == 8
    assert config.training.max_sequence_length == 8
    train_experiment(config)
    run = config.run_directory
    assert (run / "checkpoint.pt").is_file()
    experiment = load_experiment(config, work_directory=run)
    smoke_contract = replace(
        active, environment=config.environment, training=config.training
    )
    try:
        record, rows, summary = evaluate(
            smoke_contract,
            config,
            experiment,
            checkpoint="checkpoint.pt",
            split="development",
            task_cap=4,
        )
        assert record.status == "completed" and len(rows) == 4
        assert record.outer_length == 8
        assert summary["tasks"] == 4.0 and 0.0 <= summary["goal_success"] <= 1.0
    finally:
        for name in ("train_envs", "val_envs"):
            closer = getattr(getattr(experiment, name, None), "close", None)
            if closer is not None:
                closer()
