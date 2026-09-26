"""M2 exit: native Key-to-Door boundaries, leakage, replay and the C1 lifecycle.

These are execution-integrity checks. Nothing here trains to convergence or
claims that the pinned task is learnable; that is M5 work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from amago.envs.amago_env import SequenceWrapper

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    FINAL_TASKS,
    TRAINING_TASKS,
    benchmark_task_sources,
)
from reasoned_icrl.environments.dark_key_to_door import (
    KEY_TO_DOOR_FIELDS,
    KEY_TO_DOOR_PROTOCOL,
    DarkKeyToDoorEnv,
)
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import (
    ContractError,
    ResultValidationError,
    architecture_uses_history_packet,
)
from reasoned_icrl.experiments.evaluation import (
    AttemptTaskResult,
    attempt_secondary,
    evaluation_environment,
    events,
)
from reasoned_icrl.experiments.records import validate_benchmark_results
from reasoned_icrl.runtime.environments import amago_environment
from reasoned_icrl.runtime.replay import create_replay_dataset
from reasoned_icrl.runtime.rollout import (
    evaluate,
    rollout,
)
from reasoned_icrl.runtime.training import (
    load_experiment,
    train_experiment,
)
from tests.experiments.fixtures import load_fixture_study

ROOT = Path(__file__).resolve().parents[2]
# The native action table: left, up, right, down, stay.
LEFT, UP, RIGHT, DOWN, STAY = 0, 1, 2, 3, 4
NATIVE_DIRS = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]


def contract() -> Any:
    return load_fixture_study("dat_benchmarks").contract("dark_key_to_door")


def resolved(condition: str = "transition", **overrides: Any) -> Any:
    settings: dict[str, Any] = {"seed": 0, "repository": ROOT, "device": "cpu"}
    settings.update(overrides)
    return experiment_config(
        contract(),
        load_fixture_study("dat_benchmarks"),
        condition=condition,
        **settings,
    )


def native_environment(**overrides: Any) -> DarkKeyToDoorEnv:
    settings: dict[str, Any] = {
        "size": 8,
        "physical_horizon": 50,
        "meta_horizon": 500,
        "scored_attempts": 8,
        "split": "development",
        "initial_seed": 0,
    }
    settings.update(overrides)
    return DarkKeyToDoorEnv(**settings)


def pin_layout(
    env: DarkKeyToDoorEnv,
    *,
    start: tuple[int, int],
    key: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    """Replace the sampled hidden layout; evaluator-side test scaffolding only."""
    native = env.native.unwrapped
    native.start = np.array(start)
    native.key = np.array(key)
    native.goal = np.array(goal)
    native.dirs = [list(pair) for pair in NATIVE_DIRS]
    native.pos = np.array(start)
    native.episode_time = 0
    native.has_key = False
    native.reset_next_step = False
    # Keep the adapter's cached endpoint consistent with the replaced layout.
    env._current = env._project(native.obs())
    env._pending_reset = False
    env._has_key = False


def decoded(env: DarkKeyToDoorEnv, packet: dict[str, np.ndarray]) -> list[float]:
    fields = env.public_fields(packet)
    return [fields[name] for name in KEY_TO_DOOR_FIELDS]


# ----------------------------------------------------------------------
# M2.1/M2.2 boundaries
# ----------------------------------------------------------------------


def test_key_door_and_reset_boundaries_follow_the_pinned_native_lifecycle() -> None:
    env = native_environment()
    try:
        packet, _ = env.reset(options={"task_index": 1_000_000})
        pin_layout(env, start=(0, 0), key=(0, 1), goal=(0, 2))
        assert packet["event"].tolist() == [0, 1, 0]
        assert not packet["previous"].any() and not packet["outcome"].any()

        packet, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 1.0 and not terminated and not truncated
        assert packet["event"].tolist() == [1, 0, 0]
        assert decoded(env, packet) == pytest.approx([0.0, 1 / 8, 1.0, 1 / 50])
        # A physical step's endpoint is the observation it returned.
        np.testing.assert_array_equal(packet["outcome"], packet["current"])
        assert decoded(env, {"current": packet["previous"]}) == pytest.approx(
            [0.0, 0.0, 0.0, 0.0]
        )
        assert not info["attempt_done"] and not info["reset_only"]

        packet, reward, _, _, info = env.step(RIGHT)
        assert reward == 1.0 and info["attempt_done"] and info["attempt_success"]
        assert packet["event"].tolist() == [1, 0, 1]
        assert (info["attempt_return"], info["attempt_steps"]) == (2.0, 2)

        # The next step is a soft reset that ignores whichever action it gets.
        packet, reward, _, _, info = env.step(DOWN)
        assert reward == 0.0 and info["reset_only"] and not info["attempt_done"]
        assert packet["event"].tolist() == [0, 1, 0]
        assert not packet["previous"].any() and not packet["outcome"].any()
        assert decoded(env, packet) == pytest.approx([0.0, 0.0, 0.0, 0.0])

        record = env.completed_attempts[0]
        assert (record.index, record.steps, record.success) == (0, 2, True)
        assert (record.key_step, record.door_step) == (1, 2)
        assert record.native_return == 2.0 and record.complete
    finally:
        env.close()


def test_a_timed_out_attempt_ends_at_the_physical_limit_without_a_door() -> None:
    env = native_environment(physical_horizon=6, meta_horizon=100)
    try:
        env.reset(options={"task_index": 1_000_001})
        pin_layout(env, start=(0, 0), key=(7, 7), goal=(7, 6))
        for step in range(1, 6):
            packet, reward, _, _, info = env.step(STAY)
            assert reward == 0.0 and not info["attempt_done"]
            assert packet["event"].tolist() == [1, 0, 0]
            assert decoded(env, packet)[3] == pytest.approx(step / 6)
        packet, _, _, _, info = env.step(STAY)
        assert info["attempt_done"] and not info["attempt_success"]
        assert packet["event"].tolist() == [1, 0, 1]
        record = env.completed_attempts[0]
        assert (record.steps, record.native_return, record.key_step) == (6, 0.0, None)
        assert record.door_step is None and not record.success
    finally:
        env.close()


def test_the_outer_boundary_terminates_and_truncates_once_at_the_meta_budget() -> None:
    env = native_environment(physical_horizon=4, meta_horizon=10, scored_attempts=2)
    try:
        env.reset(options={"task_index": 1_000_002})
        pin_layout(env, start=(0, 0), key=(7, 7), goal=(7, 6))
        for index in range(9):
            _, _, terminated, truncated, _ = env.step(STAY)
            assert not terminated and not truncated, index
        _, _, terminated, truncated, info = env.step(STAY)
        assert terminated and truncated and info["step_in_task"] == 10
        # Step 5 and step 10 are the soft resets; two attempts completed.
        assert len(env.completed_attempts) == 2
        assert env.partial_attempt is None
        with pytest.raises(ContractError, match="requires an outer reset"):
            env.step(STAY)
    finally:
        env.close()


def test_a_partial_attempt_at_the_outer_boundary_is_reported_separately() -> None:
    env = native_environment(physical_horizon=4, meta_horizon=12, scored_attempts=2)
    try:
        env.reset(options={"task_index": 1_000_003})
        pin_layout(env, start=(0, 0), key=(7, 7), goal=(7, 6))
        for _ in range(12):
            _, _, terminated, _, _ = env.step(STAY)
        assert terminated
        assert len(env.completed_attempts) == 2
        partial = env.partial_attempt
        assert partial is not None and not partial.complete
        assert (partial.steps, partial.success, partial.door_step) == (2, False, None)
    finally:
        env.close()


def test_hidden_locations_persist_while_possession_resets_each_attempt() -> None:
    env = native_environment(physical_horizon=3, meta_horizon=40)
    try:
        env.reset(options={"task_index": 1_000_004})
        pin_layout(env, start=(0, 0), key=(0, 1), goal=(7, 7))
        for _ in range(2):
            packet, reward, _, _, _ = env.step(RIGHT)
            assert reward == 1.0 and decoded(env, packet)[2] == 1.0
            env.step(STAY)
            packet, _, _, _, info = env.step(STAY)
            assert info["attempt_done"] and decoded(env, packet)[2] == 1.0
            packet, _, _, _, info = env.step(STAY)
            # Possession and the physical clock reset; the layout does not.
            assert info["reset_only"] and decoded(env, packet)[2] == 0.0
        native = env.native.unwrapped
        assert native.key.tolist() == [0, 1] and native.goal.tolist() == [7, 7]
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_meta_budget_always_completes_the_declared_scored_attempts(
    seed: int,
) -> None:
    """The worst case is a policy that never finishes an attempt early."""
    native = contract().environment
    assert native.meta_horizon >= native.attempts * (native.horizon + 1)
    env = native_environment()
    generator = np.random.default_rng(seed)
    try:
        for policy in ("stay", "random"):
            env.reset(options={"task_index": DEVELOPMENT_TASKS[seed]})
            done = False
            while not done:
                action = STAY if policy == "stay" else int(generator.integers(5))
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            assert len(env.completed_attempts) >= native.attempts
    finally:
        env.close()


# ----------------------------------------------------------------------
# Observation leakage
# ----------------------------------------------------------------------


def test_the_packet_carries_only_the_four_public_native_values() -> None:
    env = native_environment()
    try:
        packet, info = env.reset(options={"task_index": 1_000_005})
        assert set(packet) == {"current", "previous", "outcome", "event", "valid"}
        assert set(env.observation_space.spaces) == set(packet)
        assert packet["current"].shape == (len(KEY_TO_DOOR_FIELDS),) == (4,)
        native = env.native.unwrapped
        for _ in range(20):
            packet, _, _, _, info = env.step(int(np.random.default_rng(3).integers(5)))
            expected = [
                native.pos[0] / native.size,
                native.pos[1] / native.size,
                float(native.has_key),
                native.episode_time / native.H,
            ]
            assert decoded(env, packet) == pytest.approx(expected)
        hidden = {"key", "goal", "start", "dirs", "task_seed"}
        assert not hidden & set(info)
    finally:
        env.close()


def test_two_different_hidden_layouts_produce_identical_untouched_traces() -> None:
    """Nothing about the key or door reaches the policy until it is reached."""
    left, right = native_environment(), native_environment()
    try:
        left.reset(options={"task_index": 1_000_006})
        right.reset(options={"task_index": 1_000_007})
        pin_layout(left, start=(3, 3), key=(7, 7), goal=(7, 6))
        pin_layout(right, start=(3, 3), key=(0, 0), goal=(1, 0))
        for action in (STAY, UP, DOWN, STAY, LEFT, RIGHT):
            first_packet, first_reward, *_ = left.step(action)
            second_packet, second_reward, *_ = right.step(action)
            assert first_reward == second_reward == 0.0
            for name in first_packet:
                np.testing.assert_array_equal(
                    first_packet[name], second_packet[name], err_msg=name
                )
    finally:
        left.close()
        right.close()


def test_task_rosters_are_disjoint_and_match_the_contract() -> None:
    active = contract()
    assert active.roster("development") == tuple(DEVELOPMENT_TASKS)
    assert active.roster("final") == tuple(FINAL_TASKS)
    for left, right in (
        (TRAINING_TASKS, DEVELOPMENT_TASKS),
        (TRAINING_TASKS, FINAL_TASKS),
        (DEVELOPMENT_TASKS, FINAL_TASKS),
    ):
        assert not set(left) & set(right)
    assert benchmark_task_sources("train") is TRAINING_TASKS
    with pytest.raises(ContractError, match="Unknown benchmark split"):
        benchmark_task_sources("iid")
    assert contract().protocol == KEY_TO_DOOR_PROTOCOL


def test_a_task_seed_selects_the_same_hidden_layout_for_every_condition() -> None:
    traces = []
    for actor_seed in (0, 17, 4096):
        env = native_environment(initial_seed=actor_seed, split="final")
        try:
            packet, _ = env.reset(options={"task_index": 2_000_000})
            steps = [packet["current"].copy()]
            for action in (RIGHT, RIGHT, DOWN, DOWN, LEFT):
                packet, reward, *_ = env.step(action)
                steps.append(packet["current"].copy())
                steps.append(np.array([reward], dtype=np.float32))
            traces.append(np.concatenate(steps))
        finally:
            env.close()
    for other in traces[1:]:
        np.testing.assert_array_equal(traces[0], other)


# ----------------------------------------------------------------------
# Replay alignment
# ----------------------------------------------------------------------


def collect_task(
    directory: Path, *, actions: list[int], size: int = 8
) -> tuple[DarkKeyToDoorEnv, list[dict[str, np.ndarray]], list[float], list[int]]:
    """Run one whole task through the AMAGO boundary and save its replay file."""
    env = native_environment(
        size=size, physical_horizon=4, meta_horizon=10, scored_attempts=2
    )
    wrapped = amago_environment(env, name="DarkKeyToDoorNative-test", seed=0)
    dataset = create_replay_dataset(
        directory, capacity=8, full_tasks=True, reset_only_terminal=True
    )
    sequence = SequenceWrapper(
        wrapped,
        save_trajs_to=dataset.save_new_trajs_to,
        save_every=None,
        save_trajs_as="npz-compressed",
    )
    env.set_task(1_000_010)
    observations: list[dict[str, np.ndarray]] = []
    rewards: list[float] = []
    taken: list[int] = []
    packet, _ = sequence.reset()
    observations.append({k: np.asarray(v)[0].copy() for k, v in packet.items()})
    for action in actions:
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
    return dataset, observations, rewards, taken


def test_rollout_and_replay_agree_on_every_causal_field(tmp_path: Path) -> None:
    dataset, observations, rewards, taken = collect_task(tmp_path, actions=[STAY] * 10)
    data = dataset.sample_random_trajectory()
    length = len(data)
    assert length == len(taken) == 10
    assert data.obs["current"].shape[0] == length + 1
    assert bool(data.dones[-1].item())
    assert data.time_idxs.reshape(-1).tolist() == list(range(length + 1))
    for step, expected in enumerate(observations):
        for name, value in expected.items():
            np.testing.assert_allclose(
                data.obs[name][step].numpy(), value, err_msg=f"{name}@{step}"
            )
    np.testing.assert_allclose(data.rews.reshape(-1).numpy(), rewards)
    for step, action in enumerate(taken):
        # RL2 at t+1 is exactly [reward_t, one-hot action_t]; feedback appears
        # once, on the token that follows the decision that earned it.
        expected = np.zeros(6, dtype=np.float32)
        expected[0] = rewards[step]
        expected[1 + action] = 1.0
        np.testing.assert_allclose(data.rl2s[step + 1].numpy(), expected)
    np.testing.assert_allclose(data.rl2s[0].numpy(), np.zeros(6, dtype=np.float32))


def test_a_reset_only_terminal_token_is_accepted_only_where_it_is_native(
    tmp_path: Path,
) -> None:
    dataset, observations, _, _ = collect_task(tmp_path, actions=[STAY] * 10)
    # The tenth decision is the soft reset that follows the second timeout.
    assert observations[-1]["event"].tolist() == [0, 1, 0]
    assert observations[-2]["event"].tolist() == [1, 0, 1]
    assert dataset.sample_random_trajectory() is not None
    dataset.reset_only_terminal = False
    with pytest.raises(ContractError, match="lost its terminal event"):
        dataset.sample_random_trajectory()


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


def test_events_score_the_first_eight_attempts_and_reject_a_short_task() -> None:
    active = contract()
    config = resolved()
    env = native_environment()
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(STAY)
            done = terminated or truncated
        result = AttemptTaskResult(
            task_id=DEVELOPMENT_TASKS[0],
            rollout_seed=0,
            attempts=env.completed_attempts,
            partial=env.partial_attempt,
            task_return=env.task_return,
            decisions=active.environment.outer_length,
        )
    finally:
        env.close()
    rows = events(
        active,
        config,
        [result],
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert len(rows) == active.environment.attempts
    assert [event.event_index for event in rows] == list(range(1, 9))
    assert all(event.numerator == 0 and event.native_return == 0.0 for event in rows)
    assert all(event.step == active.environment.horizon for event in rows)
    short = AttemptTaskResult(
        task_id=result.task_id,
        rollout_seed=0,
        attempts=result.attempts[:3],
        partial=None,
        task_return=0.0,
        decisions=10,
    )
    with pytest.raises(ResultValidationError, match="completed 3 attempts"):
        events(
            active,
            config,
            [short],
            checkpoint="checkpoint.pt",
            split="development",
            history="retained",
        )


def test_secondary_records_use_the_evaluated_budget_and_mark_undefined_means() -> None:
    config = resolved()
    env = native_environment()
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(STAY)
            done = terminated or truncated
        result = AttemptTaskResult(
            task_id=DEVELOPMENT_TASKS[0],
            rollout_seed=0,
            attempts=env.completed_attempts,
            partial=env.partial_attempt,
            task_return=env.task_return,
            decisions=500,
        )
    finally:
        env.close()
    summary = attempt_secondary(config.environment, [result])
    assert summary["completion_steps_success_only_mean"] is None
    # The meta budget outlasts the eight scored attempts, so later ones
    # exist and are reported apart from the primary endpoint.
    assert summary["later_attempt_count"] == 1.0
    assert summary["later_attempt_success"] == 0.0
    assert summary["completed_attempts_mean"] == 9.0
    assert summary["completion_steps_fixed_budget_mean"] == 50.0
    assert summary["first_key_step_fixed_budget_mean"] == 500.0
    assert summary["tasks_without_key"] == summary["tasks_without_door"] == 1.0
    assert summary["full_budget_return_mean"] == 0.0
    assert json.dumps(summary, allow_nan=False)


# ----------------------------------------------------------------------
# C1 lifecycle
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("condition", ("transition", "transition_dat"))
def test_native_cpu_train_export_evaluate_lifecycle(
    condition: str, tmp_path: Path
) -> None:
    active = contract()
    config = resolved(condition, output_root=tmp_path, smoke=True)
    assert config.environment.name == "dark_key_to_door"
    assert config.environment.outer_length == config.environment.meta_horizon
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
    try:
        run_record, rows, summary = evaluate(
            active,
            config,
            experiment,
            checkpoint="checkpoint.pt",
            split="development",
            task_cap=3,
        )
        assert run_record.status == "completed"
        assert run_record.parameter_count and run_record.parameter_count > 0
        assert len(rows) == 3 * active.environment.attempts
        assert summary["tasks"] == 3.0
        with pytest.raises(ResultValidationError, match="incomplete/unexpected"):
            validate_benchmark_results([active], [run_record], list(rows))

        environment = amago_environment(
            _evaluation_environment(config), name="check", seed=0
        )
        try:
            forward, _ = rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[1]],
                rollout_seed=0,
            )
            reverse, _ = rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[1], DEVELOPMENT_TASKS[0]],
                rollout_seed=0,
            )
        finally:
            environment.close()
        # An outer reset clears the cache, so roster order cannot change a task.
        assert forward[0].attempts == reverse[1].attempts
        assert forward[1].attempts == reverse[0].attempts

        if architecture_uses_history_packet(config.model.architecture_id):
            _assert_dense_and_cached_agree(experiment, config)
    finally:
        for name in ("train_envs", "val_envs"):
            closer = getattr(getattr(experiment, name, None), "close", None)
            if callable(closer):
                closer()


def _evaluation_environment(config: Any) -> Any:
    return evaluation_environment(contract(), config, split="development", seed=0)


def _assert_dense_and_cached_agree(experiment: Any, config: Any) -> None:
    """One rollout, encoded once as a whole sequence and once step by step."""
    environment = _evaluation_environment(config)
    try:
        packets: list[dict[str, np.ndarray]] = []
        feedback: list[np.ndarray] = []
        packet, _ = environment.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        packets.append(packet)
        feedback.append(np.zeros(6, dtype=np.float32))
        done = False
        action = STAY
        while not done:
            packet, reward, terminated, truncated, _ = environment.step(action)
            row = np.zeros(6, dtype=np.float32)
            row[0] = reward
            row[1 + action] = 1.0
            packets.append(packet)
            feedback.append(row)
            done = terminated or truncated
    finally:
        environment.close()
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


def test_the_layout_change_continuation_swaps_the_layout_at_an_attempt_boundary() -> (
    None
):
    """A new hidden layout at the first attempt boundary after every period,
    invisible to the packet, deterministic per task, recorded per attempt."""
    import random

    plain = native_environment(physical_horizon=3, meta_horizon=40)
    changing = native_environment(physical_horizon=3, meta_horizon=40)
    try:
        changing.set_layout_period(10)
        assert changing.layout_period == 10 and changing.layout_index == 0
        for env in (plain, changing):
            env.reset(options={"task_index": 1_000_004})
            pin_layout(env, start=(0, 0), key=(0, 1), goal=(7, 7))
        native = changing.native.unwrapped
        # Attempts last three physical calls plus a reset-only call, so the
        # first boundary at or after call 10 is the reset-only call 12.
        seen: list[tuple[int, int, bool]] = []
        for call in range(1, 25):
            packet_plain, *_ = plain.step(STAY)
            packet, _, _, _, info = changing.step(STAY)
            seen.append((call, changing.layout_index, bool(info["reset_only"])))
            if call < 12:
                assert decoded(changing, packet) == decoded(plain, packet_plain), call
        assert next(s for s in seen if s[1] == 1) == (12, 1, True)
        assert next(s for s in seen if s[1] == 2) == (20, 2, True)
        draw = random.Random("relayout:1000004:2")
        assert native.start.tolist() == draw.choices(range(8), k=2)
        assert native.key.tolist() == draw.choices(range(8), k=2)
        assert native.goal.tolist() == draw.choices(range(8), k=2)
        layouts = [record.layout for record in changing.completed_attempts]
        assert layouts == [0, 0, 0, 1, 1, 2]
        assert all(r.layout == 0 for r in plain.completed_attempts)
        state = changing._state()
        assert state["layout_period"] == 10 and state["layout"] == 2
        assert state["source"] == 1_000_004
        with pytest.raises(ContractError, match="positive number of calls"):
            changing.set_layout_period(0)
    finally:
        plain.close()
        changing.close()
