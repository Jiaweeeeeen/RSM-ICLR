"""M4 exit: MazeRunner semantics, hindsight relabeling and the C1 lifecycle.

These are execution-integrity checks. Nothing here trains to convergence or
claims that the pinned maze task is learnable; that is M5 work.
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
from amago.hindsight import FrozenTraj

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    FINAL_TASKS,
    TRAINING_TASKS,
)
from reasoned_icrl.environments.mazerunner import (
    GOAL_SENTINEL,
    MAZERUNNER_ACTIONS,
    MAZERUNNER_OBSERVATION_WIDTH,
    HindsightGoalRelabeler,
    MazeRunnerEnv,
    decode_grid,
    encode_grid,
    mazerunner_protocol,
    mazerunner_variant,
    public_width,
    rebuild_transition_packet,
)
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.config import environment_config
from reasoned_icrl.experiments.contracts import (
    ContractError,
    ResultValidationError,
    architecture_uses_history_packet,
)
from reasoned_icrl.experiments.evaluation import (
    MazeRunnerEpisodeResult,
    evaluation_environment,
    events,
    history_modes,
    mazerunner_secondary,
    random_policy_goal_fraction,
)
from reasoned_icrl.experiments.records import validate_benchmark_results
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
)
from reasoned_icrl.runtime.environments import (
    amago_environment,
    benchmark_relabeler,
)
from reasoned_icrl.runtime.replay import (
    ReconstructingRelabeler,
    create_replay_dataset,
)
from reasoned_icrl.runtime.rollout import (
    evaluate,
    rollout,
)
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
    load_selected_checkpoint,
    train_experiment,
)
from tests.experiments.fixtures import load_fixture_study

ROOT = Path(__file__).resolve().parents[2]
MAZE = 11
GOALS = 3
HORIZON = 250
# Native action table: west, north, east, south, stay.
WEST, NORTH, EAST, SOUTH, STAY = 0, 1, 2, 3, 4


def contract() -> Any:
    return load_fixture_study("dat_benchmarks").contract("mazerunner")


def resolved(condition: str = "transition", **overrides: Any) -> Any:
    settings: dict[str, Any] = {"seed": 0, "repository": ROOT, "device": "cpu"}
    settings.update(overrides)
    return experiment_config(
        contract(),
        load_fixture_study("dat_benchmarks"),
        condition=condition,
        **settings,
    )


def maze_environment(**overrides: Any) -> MazeRunnerEnv:
    settings: dict[str, Any] = {
        "size": MAZE,
        "goals": GOALS,
        "horizon": HORIZON,
        "variant": "fixed-actions",
        "split": "development",
        "initial_seed": 0,
    }
    settings.update(overrides)
    return MazeRunnerEnv(**settings)


# ----------------------------------------------------------------------
# M4.1 native learning task
# ----------------------------------------------------------------------


def test_the_pinned_task_matches_the_planned_starting_values() -> None:
    active = contract().environment
    assert (active.size, active.goals, active.horizon) == (MAZE, GOALS, HORIZON)
    env = maze_environment()
    try:
        packet, info = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        native = env.native.unwrapped
        assert native.maze_dim == MAZE and native.time_limit == HORIZON
        assert len(native.goal_positions) == GOALS
        assert native.start == (MAZE - 2, MAZE // 2)
        assert native.action_dirs.tolist() == [
            [0, -1],
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, 0],
        ]
        assert env.action_space.n == MAZERUNNER_ACTIONS
        assert packet["current"].shape == (public_width(GOALS),) == (13,)
        assert packet["event"].tolist() == [0, 1, 0]
        assert env.decode(packet)["position"] == native.start
        assert "achieved" not in info and "maze" not in info
    finally:
        env.close()


def test_reaching_the_final_goal_terminates_and_a_timeout_also_terminates() -> None:
    env = maze_environment(horizon=12)
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[1]})
        native = env.native.unwrapped
        # Walk the agent onto its goals in order by pinning the goal sequence to
        # squares it is standing on; the native rule checks one goal per step.
        native.goal_positions = [native.start] * GOALS
        native.active_goal_idx = 0
        rewards = []
        for _ in range(GOALS):
            _, reward, terminated, truncated, info = env.step(STAY)
            rewards.append(reward)
        assert rewards == [1.0, 1.0, 1.0]
        assert terminated and not truncated
        assert env.episode_return == 3.0 and len(env.completed_goals) == GOALS
        assert [record.step for record in env.completed_goals] == [1, 2, 3]
        assert info["goals_completed"] == GOALS
        with pytest.raises(ContractError, match="requires an outer reset"):
            env.step(STAY)
    finally:
        env.close()

    timeout = maze_environment(horizon=5)
    try:
        timeout.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        for step in range(1, 6):
            _, _, terminated, truncated, _ = timeout.step(STAY)
            assert terminated == (step == 5) and truncated == (step == 5)
        # The native task promotes its finite-horizon timeout to `terminated`.
        assert timeout.episode_return == 0.0
    finally:
        timeout.close()


def test_the_completed_goal_sentinel_is_published_in_place() -> None:
    env = maze_environment(horizon=12)
    try:
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[3]})
        native = env.native.unwrapped
        assert not (env.decode(packet)["goals"] == GOAL_SENTINEL).any()
        native.goal_positions = [native.start] * GOALS
        native.active_goal_idx = 0
        packet, _, _, _, _ = env.step(STAY)
        goals = env.decode(packet)["goals"]
        assert goals[0].tolist() == [GOAL_SENTINEL, GOAL_SENTINEL]
        assert not (goals[1:] == GOAL_SENTINEL).any()
    finally:
        env.close()


def test_the_public_projection_is_the_pinned_wrapper_and_hides_the_maze() -> None:
    env = maze_environment()
    try:
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[4]})
        native = env.native.unwrapped
        generator = np.random.default_rng(0)
        for _ in range(25):
            packet, _, terminated, truncated, info = env.step(
                int(generator.integers(MAZERUNNER_ACTIONS))
            )
            current = packet["current"]
            # The first seven values are the native goal-free observation.
            np.testing.assert_allclose(current[:7], native._get_obs(), rtol=0, atol=0)
            # `achieved` is exactly the public position, so it is never stored.
            assert env.decode(packet)["position"] == tuple(native.pos)
            privileged = {"maze", "action_dirs", "achieved", "goal_positions"}
            assert not privileged & set(info)
            if terminated or truncated:
                break
        assert set(env.observation_space.spaces) == set(packet)
    finally:
        env.close()


def test_action_permutation_is_deterministic_and_only_for_the_named_protocol() -> None:
    fixed = [
        maze_environment(variant="fixed-actions", initial_seed=seed) for seed in (0, 9)
    ]
    randomized = [
        MazeRunnerEnv(
            size=MAZE,
            goals=GOALS,
            horizon=HORIZON,
            variant="randomized-actions",
            split="development",
            initial_seed=seed,
        )
        for seed in (0, 9)
    ]
    try:
        for env in fixed + randomized:
            env.reset(options={"task_index": DEVELOPMENT_TASKS[5]})
        base = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
        assert all(e.native.unwrapped.action_dirs.tolist() == base for e in fixed)
        left, right = (e.native.unwrapped.action_dirs.tolist() for e in randomized)
        # The permutation is a function of the map identity, not the actor seed.
        assert left == right and sorted(left) == sorted(base)
        assert randomized[0].protocol == "mazerunner-randomized-actions"
        assert fixed[0].protocol == "mazerunner-fixed-actions"
    finally:
        for env in fixed + randomized:
            env.close()


def test_map_rosters_are_disjoint_and_match_the_contract() -> None:
    active = contract()
    assert active.roster("development") == tuple(DEVELOPMENT_TASKS)
    assert active.roster("final") == tuple(FINAL_TASKS)
    for left, right in (
        (TRAINING_TASKS, DEVELOPMENT_TASKS),
        (TRAINING_TASKS, FINAL_TASKS),
        (DEVELOPMENT_TASKS, FINAL_TASKS),
    ):
        assert not set(left) & set(right)
    assert active.protocol == mazerunner_protocol("fixed-actions", MAZE)
    assert mazerunner_variant("mazerunner-randomized-actions") == (
        "randomized-actions",
        MAZE,
    )
    assert mazerunner_variant("mazerunner-15-randomized-actions") == (
        "randomized-actions",
        15,
    )
    with pytest.raises(ContractError, match="Unknown MazeRunner protocol"):
        mazerunner_variant("mazerunner-diagonal")
    with pytest.raises(ContractError, match="Unknown MazeRunner variant"):
        mazerunner_protocol("diagonal", MAZE)
    # The size is part of the identity: no 15x15 fixed-action protocol exists.
    with pytest.raises(ContractError, match="No MazeRunner protocol"):
        mazerunner_protocol("fixed-actions", 15)


def test_the_same_map_seed_reproduces_the_same_maze_for_every_condition() -> None:
    traces = []
    for actor_seed in (0, 31, 4096):
        env = maze_environment(initial_seed=actor_seed, split="final")
        try:
            packet, _ = env.reset(options={"task_index": FINAL_TASKS[0]})
            steps = [packet["current"].copy()]
            for action in (NORTH, NORTH, WEST, EAST, SOUTH):
                packet, reward, *_ = env.step(action)
                steps.append(packet["current"].copy())
                steps.append(np.array([reward], dtype=np.float32))
            traces.append(np.concatenate(steps))
        finally:
            env.close()
    for other in traces[1:]:
        np.testing.assert_array_equal(traces[0], other)


# ----------------------------------------------------------------------
# M6.1 the 15x15 randomized protocol (the summary-memory study's)
# ----------------------------------------------------------------------

MAZE_15 = 15
HORIZON_15 = 500


def contract_15() -> Any:
    return load_retired_summary_memory_study().contract("mazerunner")


def maze_15(**overrides: Any) -> MazeRunnerEnv:
    settings: dict[str, Any] = {
        "size": MAZE_15,
        "goals": GOALS,
        "horizon": HORIZON_15,
        "variant": "randomized-actions",
        "split": "development",
        "initial_seed": 0,
    }
    settings.update(overrides)
    return MazeRunnerEnv(**settings)


def test_the_15x15_protocol_carries_its_size_and_declares_the_shared_recipe() -> None:
    active = contract_15()
    reference = contract()
    assert active.protocol == "mazerunner-15-randomized-actions"
    assert active.status == "C0"
    declared = active.environment
    assert (declared.size, declared.goals, declared.horizon) == (MAZE_15, GOALS, 500)
    assert declared.randomized_actions and not declared.has_fixed_native_horizon
    assert active.training.max_sequence_length == HORIZON_15
    assert active.training.trajectory_length == HORIZON_15 + 1
    assert active.training.validation_timesteps == HORIZON_15
    assert active.training.exploration_rollout_horizon == HORIZON_15
    for name in (
        "epochs",
        "timesteps_per_epoch",
        "batches_per_epoch",
        "batch_size",
        "learning_rate",
        "warmup_steps",
        "epsilon_anneal_steps",
        "reward_multiplier",
        "mixed_precision",
        "validation_interval",
        "checkpoint_interval",
    ):
        assert getattr(active.training, name) == getattr(reference.training, name)
    assert active.evaluation == reference.evaluation
    assert active.roster("development") == tuple(DEVELOPMENT_TASKS)
    env = maze_15()
    try:
        packet, info = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        native = env.native.unwrapped
        assert env.protocol == active.protocol and env.randomized_actions
        assert native.maze_dim == MAZE_15 and native.time_limit == HORIZON_15
        assert len(native.goal_positions) == GOALS
        assert native.start == (MAZE_15 - 2, MAZE_15 // 2) == (13, 7)
        assert env.decode(packet)["position"] == native.start
        # The public width does not depend on the maze size; coordinates are
        # divided by 15 and decode back onto the 15x15 grid.
        assert packet["current"].shape == (public_width(GOALS),) == (13,)
        assert env.action_space.n == MAZERUNNER_ACTIONS
        goals = env.decode(packet)["goals"]
        assert goals.shape == (GOALS, 2) and np.all((goals >= 0) & (goals < MAZE_15))
        assert "achieved" not in info and "action_dirs" not in info
    finally:
        env.close()
    with pytest.raises(ContractError, match="No MazeRunner protocol"):
        MazeRunnerEnv(size=MAZE_15, variant="fixed-actions", split="development")
    raw = {
        "name": "mazerunner",
        "benchmark": active.protocol,
        "size": MAZE_15,
        "attempts": 1,
        "horizon": HORIZON_15,
        "goals": GOALS,
        "randomized_actions": True,
        "parallel_envs": 1,
    }
    assert environment_config(raw).size == MAZE_15
    with pytest.raises(ContractError, match="size disagrees with its protocol"):
        environment_config({**raw, "size": MAZE})
    with pytest.raises(ContractError, match="randomized_actions disagrees"):
        environment_config({**raw, "randomized_actions": False})
    study = load_retired_summary_memory_study()
    smoke = experiment_config(
        active,
        study,
        condition="raw",
        seed=0,
        repository=ROOT,
        device="cpu",
        smoke=True,
    )
    # The smoke profile shrinks the timer, never the maze or the protocol.
    assert smoke.environment.size == MAZE_15 and smoke.environment.horizon == 8
    assert smoke.environment.benchmark == active.protocol


def test_the_permutation_is_drawn_once_per_outer_reset_and_fixed_within_it() -> None:
    base = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
    seen: dict[int, list[list[int]]] = {}
    env = maze_15()
    try:
        for task in DEVELOPMENT_TASKS[:6]:
            env.reset(options={"task_index": task})
            native = env.native.unwrapped
            dirs = native.action_dirs.tolist()
            assert sorted(dirs) == sorted(base)
            generator = np.random.default_rng(task)
            for _ in range(40):
                _, _, terminated, truncated, _ = env.step(
                    int(generator.integers(MAZERUNNER_ACTIONS))
                )
                # One permutation per outer reset: nothing changes it mid-episode.
                assert native.action_dirs.tolist() == dirs
                if terminated or truncated:
                    break
            seen[task] = dirs
            # A second reset of the same map draws the same permutation.
            env.reset(options={"task_index": task})
            assert native.action_dirs.tolist() == dirs
    finally:
        env.close()
    # Different maps draw different permutations ...
    assert len({tuple(map(tuple, dirs)) for dirs in seen.values()}) > 1
    # ... as a function of the map identity, never of the actor seed.
    other = maze_15(initial_seed=7)
    try:
        for task, dirs in seen.items():
            other.reset(options={"task_index": task})
            assert other.native.unwrapped.action_dirs.tolist() == dirs
    finally:
        other.close()


def test_relabeling_on_the_15x15_protocol_is_one_function_for_every_cell() -> None:
    study = load_retired_summary_memory_study()
    active = contract_15()
    built = []
    for condition in ("raw", "raw_summary", "raw_dat_summary", "raw_gru"):
        config = experiment_config(
            active, study, condition=condition, seed=5, repository=ROOT, device="cpu"
        )
        relabel = benchmark_relabeler(config.as_runtime_mapping())
        assert relabel is not None
        built.append(relabel)
    native = [cast_native(relabel) for relabel in built]
    assert {(r.size, r.goals, r.horizon, r.strategy, r.seed) for r in native} == {
        (MAZE_15, GOALS, HORIZON_15, "some", 5)
    }
    # A three-goal trajectory on the 15x15 grid at the native horizon: one
    # original goal reached, then a timeout terminal.
    walk = [(13, 7), (13, 8), (12, 8), (12, 9), (11, 9), (11, 10), (10, 10)]
    path = walk + [walk[-1]] * (HORIZON_15 + 1 - len(walk))
    goals = [(12, 8), (2, 2), (3, 3)]
    rewards = [0.0, 1.0] + [0.0] * (HORIZON_15 - 2)
    frozen = toy_trajectory(path, goals, rewards, horizon=HORIZON_15, maze=MAZE_15)
    outputs = [relabel(frozen) for relabel in built]
    for other in outputs[1:]:
        np.testing.assert_array_equal(outputs[0].obs["current"], other.obs["current"])
        np.testing.assert_array_equal(outputs[0].rews, other.rews)
        np.testing.assert_array_equal(outputs[0].dones, other.dones)
    first = outputs[0]
    tokens = first.obs["current"].shape[0]
    # Positions are untouched and every published goal lies on the 15x15 grid.
    np.testing.assert_array_equal(
        first.obs["current"][:, :2], np.asarray(frozen.obs["current"])[:tokens, :2]
    )
    instruction = decode_grid(
        first.obs["current"][0, MAZERUNNER_OBSERVATION_WIDTH:], MAZE_15
    ).reshape(GOALS, 2)
    assert np.all((instruction >= 0) & (instruction < MAZE_15))
    assert float(first.rews.sum()) >= 1.0 and bool(first.dones[-1])


# ----------------------------------------------------------------------
# M4.2 relabeling: an independently checked toy trajectory
# ----------------------------------------------------------------------


def toy_trajectory(
    positions: list[tuple[int, int]],
    goals: list[tuple[int, int]],
    rewards: list[float],
    *,
    horizon: int,
    maze: int = MAZE,
) -> FrozenTraj:
    """Build a synthetic trajectory with a hand-chosen path and instruction."""
    tokens = len(positions)
    width = public_width(len(goals))
    current = np.zeros((tokens, width), dtype=np.float32)
    active = 0
    for token, spot in enumerate(positions):
        current[token, :2] = encode_grid(np.asarray(spot), maze)
        current[token, MAZERUNNER_OBSERVATION_WIDTH - 1] = np.float32(token / horizon)
        published = np.array(goals, dtype=np.int64)
        published[:active] = GOAL_SENTINEL
        current[token, MAZERUNNER_OBSERVATION_WIDTH:] = encode_grid(
            published.reshape(-1), maze
        )
        if token < tokens - 1 and rewards[token]:
            active += 1
    actions = np.zeros((tokens - 1, MAZERUNNER_ACTIONS), dtype=np.float32)
    actions[np.arange(tokens - 1), np.arange(tokens - 1) % MAZERUNNER_ACTIONS] = 1.0
    rews = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    dones = np.zeros((tokens - 1, 1), dtype=bool)
    dones[-1, 0] = True
    return FrozenTraj(
        obs={"current": current},
        rl2s=np.zeros((tokens, 1 + MAZERUNNER_ACTIONS), dtype=np.float32),
        time_idxs=np.arange(tokens, dtype=np.int64).reshape(-1, 1),
        rews=rews,
        dones=dones,
        actions=actions,
    )


TOY_PATH = [
    (9, 5),  # start
    (9, 6),
    (9, 7),  # original goal 0 reached here
    (8, 7),
    (8, 8),
    (8, 8),  # blocked move: a repeated position
    (7, 8),
    (8, 8),  # revisit
    (7, 8),  # revisit
]
TOY_GOALS = [(9, 7), (2, 2)]
TOY_REWARDS = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def relabeler(**overrides: Any) -> ReconstructingRelabeler:
    settings: dict[str, Any] = {
        "size": MAZE,
        "goals": len(TOY_GOALS),
        "horizon": 8,
        "strategy": "all",
        "seed": 0,
    }
    settings.update(overrides)
    return ReconstructingRelabeler(
        HindsightGoalRelabeler(**settings), rebuild_transition_packet
    )


def test_a_toy_trajectory_agrees_field_by_field_after_relabeling() -> None:
    frozen = toy_trajectory(TOY_PATH, TOY_GOALS, TOY_REWARDS, horizon=8)
    result = relabeler()(frozen)

    # Independently replay the randomly selected instruction on the original
    # public positions, rather than assert the retired last-position sampler.
    instruction = (
        decode_grid(result.obs["current"][0, MAZERUNNER_OBSERVATION_WIDTH:], MAZE)
        .reshape(2, 2)
        .tolist()
    )
    active = 0
    expected_rewards = []
    expected_progress = [0]
    for spot in TOY_PATH[1:]:
        reward = float(list(spot) == instruction[active])
        active += int(reward)
        expected_rewards.append(reward)
        expected_progress.append(active)
        if active == 2:
            break
    tokens = len(expected_rewards) + 1
    assert active == 2
    assert result.obs["current"].shape == (tokens, public_width(2))
    assert result.rews.reshape(-1).tolist() == expected_rewards
    assert result.dones.reshape(-1).tolist() == [False] * (tokens - 2) + [True]
    assert result.time_idxs.reshape(-1).tolist() == list(range(tokens))
    assert result.obs["valid"].reshape(-1).tolist() == [1.0] * tokens
    for token in range(tokens):
        published = np.array(instruction, dtype=np.int64)
        published[: expected_progress[token]] = GOAL_SENTINEL
        goals = decode_grid(
            result.obs["current"][token, MAZERUNNER_OBSERVATION_WIDTH:], MAZE
        )
        assert goals.tolist() == published.reshape(-1).tolist()
        np.testing.assert_array_equal(
            result.obs["current"][token, :MAZERUNNER_OBSERVATION_WIDTH],
            frozen.obs["current"][token, :MAZERUNNER_OBSERVATION_WIDTH],
        )
    # Every goal-bearing endpoint is rebuilt from the relabeled observations.
    np.testing.assert_array_equal(result.obs["outcome"][0], np.zeros(public_width(2)))
    np.testing.assert_array_equal(result.obs["previous"][0], np.zeros(public_width(2)))
    for token in range(1, tokens):
        np.testing.assert_array_equal(
            result.obs["previous"][token], result.obs["current"][token - 1]
        )
        np.testing.assert_array_equal(
            result.obs["outcome"][token], result.obs["current"][token]
        )
    assert result.obs["event"][0].tolist() == [0, 1, 0]
    assert all(row.tolist() == [1, 0, 0] for row in result.obs["event"][1:])

    # RL2 reward and action agree with the relabeled transitions.
    np.testing.assert_array_equal(result.rl2s[1:, :1], result.rews)
    np.testing.assert_array_equal(result.rl2s[1:, 1:], result.actions)
    np.testing.assert_array_equal(result.rl2s[0], np.zeros(1 + MAZERUNNER_ACTIONS))
    np.testing.assert_array_equal(result.actions, frozen.actions[: tokens - 1])


def test_relabeling_ignores_stale_precomputed_transition_fields() -> None:
    """No reward, goal or endpoint survives from before relabeling."""
    frozen = toy_trajectory(TOY_PATH, TOY_GOALS, TOY_REWARDS, horizon=8)
    clean = relabeler()(frozen)
    poisoned = toy_trajectory(TOY_PATH, TOY_GOALS, TOY_REWARDS, horizon=8)
    poisoned.obs["previous"] = np.full_like(poisoned.obs["current"], 0.5)
    poisoned.obs["outcome"] = np.full_like(poisoned.obs["current"], -0.5)
    poisoned.obs["event"] = np.ones((len(TOY_PATH), 3), dtype=np.float32)
    poisoned.obs["valid"] = np.zeros((len(TOY_PATH), 1), dtype=np.float32)
    poisoned.rl2s = np.full_like(poisoned.rl2s, 7.0)
    result = relabeler()(poisoned)
    for name, value in clean.obs.items():
        np.testing.assert_array_equal(result.obs[name], value, err_msg=name)
    np.testing.assert_array_equal(result.rl2s, clean.rl2s)
    np.testing.assert_array_equal(result.rews, clean.rews)


def test_a_hindsight_goal_is_never_placed_on_the_start_square() -> None:
    """Time zero: the native task never checks or samples a goal at the start."""
    path = [(9, 5), (9, 6), (9, 5), (9, 4), (9, 5), (8, 4)]
    frozen = toy_trajectory(path, [(2, 2)], [0.0] * 5, horizon=5)
    result = relabeler(goals=1, horizon=5)(frozen)
    instruction = decode_grid(
        result.obs["current"][0, MAZERUNNER_OBSERVATION_WIDTH:], MAZE
    )
    assert instruction.tolist() != [9, 5]
    assert tuple(instruction.tolist()) in path[1:]
    assert result.rews.sum() == 1.0
    assert bool(result.dones[-1])


def test_repeated_positions_never_become_repeated_goals() -> None:
    path = [(9, 5), (8, 5), (8, 5), (8, 5), (7, 5), (7, 5), (6, 5)]
    frozen = toy_trajectory(path, [(2, 2), (3, 3)], [0.0] * 6, horizon=6)
    result = relabeler(goals=2, horizon=6)(frozen)
    instruction = decode_grid(
        result.obs["current"][0, MAZERUNNER_OBSERVATION_WIDTH:], MAZE
    ).reshape(2, 2)
    assert len({tuple(row) for row in instruction.tolist()}) == 2
    assert all(tuple(row) in path[1:] for row in instruction.tolist())
    assert result.rews.reshape(-1).sum() == 2.0


def test_a_trajectory_that_cannot_be_filled_is_returned_unchanged() -> None:
    """Without enough distinct new squares the terminal cannot be guaranteed."""
    path = [(9, 5), (9, 5), (9, 5)]
    frozen = toy_trajectory(path, [(2, 2), (3, 3)], [0.0, 0.0], horizon=2)
    result = relabeler(goals=2, horizon=2)(frozen)
    assert result.rews.reshape(-1).tolist() == [0.0, 0.0]
    assert result.obs["current"].shape[0] == 3
    instruction = decode_grid(
        result.obs["current"][0, MAZERUNNER_OBSERVATION_WIDTH:], MAZE
    ).reshape(2, 2)
    assert instruction.tolist() == [[2, 2], [3, 3]]
    assert bool(result.dones[-1])


def test_an_already_complete_instruction_is_left_alone() -> None:
    path = [(9, 5), (9, 6), (9, 7)]
    frozen = toy_trajectory(path, [(9, 6), (9, 7)], [1.0, 1.0], horizon=2)
    result = relabeler(goals=2, horizon=2)(frozen)
    assert result.rews.reshape(-1).tolist() == [1.0, 1.0]
    instruction = decode_grid(
        result.obs["current"][0, MAZERUNNER_OBSERVATION_WIDTH:], MAZE
    ).reshape(2, 2)
    assert instruction.tolist() == [[9, 6], [9, 7]]


def test_the_declared_strategies_control_how_often_a_trajectory_is_rewritten() -> None:
    frozen = toy_trajectory(TOY_PATH, TOY_GOALS, TOY_REWARDS, horizon=8)
    untouched = relabeler(strategy="none")(frozen)
    assert untouched.obs["current"].shape[0] == len(TOY_PATH)
    assert untouched.rews.reshape(-1).sum() == 1.0
    rewritten = [
        relabeler(strategy="some", seed=seed)(frozen).rews.sum() for seed in range(30)
    ]
    assert set(rewritten) == {1.0, 2.0}
    with pytest.raises(ContractError, match="Unknown relabeling strategy"):
        HindsightGoalRelabeler(size=MAZE, goals=2, horizon=8, strategy="most")


def test_partial_hindsight_keeps_timeout_and_uncompleted_goals() -> None:
    path = [(9, 5), (9, 6), (9, 7), (9, 8), (8, 8), (7, 8)]
    frozen = toy_trajectory(path, [(2, 2), (3, 3), (4, 4)], [0.0] * 5, horizon=5)
    results = [
        relabeler(goals=3, horizon=5, strategy="some", seed=seed)(frozen)
        for seed in range(50)
    ]
    assert {float(r.rews.sum()) for r in results} == {0.0, 1.0, 2.0, 3.0}
    for result in results:
        if result.rews.sum() < 3:
            assert len(result.rews) == 5 and result.dones[-1]
            goals = decode_grid(
                result.obs["current"][-1, MAZERUNNER_OBSERVATION_WIDTH:], MAZE
            )
            assert np.any(goals >= 0)
        np.testing.assert_array_equal(result.rl2s[1:, :1], result.rews)
        np.testing.assert_array_equal(result.rl2s[1:, 1:], result.actions)


def test_relabeling_is_the_same_function_for_every_attention_condition() -> None:
    built = []
    for condition in ("transition", "transition_dat", "transition_dual_content"):
        config = resolved(condition, seed=7)
        relabel = benchmark_relabeler(config.as_runtime_mapping())
        assert relabel is not None
        built.append(relabel)
    native = [cast_native(relabel) for relabel in built]
    assert {(r.size, r.goals, r.horizon, r.strategy, r.seed) for r in native} == {
        (MAZE, GOALS, HORIZON, "some", 7)
    }
    # Match the real three-goal, finite-horizon recipe, not the two-goal toy.
    path = TOY_PATH + [TOY_PATH[-1]] * (HORIZON + 1 - len(TOY_PATH))
    frozen = toy_trajectory(
        path,
        [*TOY_GOALS, (3, 3)],
        TOY_REWARDS + [0.0] * (HORIZON - len(TOY_REWARDS)),
        horizon=HORIZON,
    )
    outputs = [relabel(frozen) for relabel in built]
    for other in outputs[1:]:
        np.testing.assert_array_equal(outputs[0].obs["current"], other.obs["current"])
        np.testing.assert_array_equal(outputs[0].rews, other.rews)


def cast_native(relabel: ReconstructingRelabeler) -> Any:
    return relabel.native


def test_other_benchmarks_declare_no_relabeler() -> None:
    study = load_fixture_study("dat_benchmarks")
    for name in ("dark_key_to_door", "count_recall"):
        config = experiment_config(
            study.contract(name),
            study,
            condition="transition",
            seed=0,
            repository=ROOT,
            device="cpu",
        )
        assert benchmark_relabeler(config.as_runtime_mapping()) is None


# ----------------------------------------------------------------------
# Replay alignment through the real dataset path
# ----------------------------------------------------------------------


def collect_episode(directory: Path, *, relabel: Any = None) -> tuple[Any, ...]:
    """Run one whole episode through the AMAGO boundary and save its replay."""
    env = maze_environment(horizon=30)
    wrapped = amago_environment(env, name="MazeRunner-test", seed=0)
    dataset = create_replay_dataset(
        directory, capacity=8, full_tasks=True, relabeler=relabel
    )
    sequence = SequenceWrapper(
        wrapped,
        save_trajs_to=dataset.save_new_trajs_to,
        save_every=None,
        save_trajs_as="npz-compressed",
    )
    env.set_task(DEVELOPMENT_TASKS[9])
    packet, _ = sequence.reset()
    observations = [{k: np.asarray(v)[0].copy() for k, v in packet.items()}]
    rewards: list[float] = []
    taken: list[int] = []
    generator = np.random.default_rng(2)
    done = False
    while not done:
        action = int(generator.integers(MAZERUNNER_ACTIONS))
        packet, reward, terminated, truncated, _ = sequence.step(
            np.array([action], dtype=np.int64)
        )
        observations.append({k: np.asarray(v)[0].copy() for k, v in packet.items()})
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        taken.append(action)
        done = bool(np.logical_or(terminated, truncated).reshape(-1)[0])
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
    dataset, observations, rewards, taken = collect_episode(tmp_path)
    data = dataset.sample_random_trajectory()
    length = len(data)
    assert length == len(taken) == 30
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
        expected_rl2 = np.zeros(MAZERUNNER_ACTIONS + 1, dtype=np.float32)
        expected_rl2[0] = rewards[step]
        expected_rl2[1 + action] = 1.0
        np.testing.assert_allclose(data.rl2s[step + 1].numpy(), expected_rl2)
    np.testing.assert_allclose(
        data.rl2s[0].numpy(), np.zeros(MAZERUNNER_ACTIONS + 1, dtype=np.float32)
    )


def test_a_relabeled_trajectory_still_satisfies_the_full_task_replay_contract(
    tmp_path: Path,
) -> None:
    """MazeRunner needs no terminal allowance: every token carries evidence."""
    relabel = ReconstructingRelabeler(
        HindsightGoalRelabeler(
            size=MAZE, goals=GOALS, horizon=30, strategy="all", seed=0
        ),
        rebuild_transition_packet,
    )
    dataset, _, _, taken = collect_episode(tmp_path, relabel=relabel)
    assert not dataset.reset_only_terminal
    data = dataset.sample_random_trajectory()
    # The instruction was actually rewritten and is completed by construction.
    assert float(data.rews.sum().item()) == float(GOALS)
    assert len(data) <= len(taken)
    assert bool(data.dones[-1].item())
    assert data.obs["event"][0, 0] == 0 and data.obs["event"][-1, 0] == 1
    assert int(data.time_idxs[0].item()) == 0
    # Action identities are preserved: RL2 still one-hots the executed action.
    for step in range(len(data)):
        one_hot = data.rl2s[step + 1, 1:].numpy()
        assert one_hot.sum() == 1.0
        np.testing.assert_allclose(one_hot, data.actions[step].numpy())


# ----------------------------------------------------------------------
# M4.3 goal evaluation
# ----------------------------------------------------------------------


def sample_results(maps: tuple[int, ...], *, horizon: int = 30) -> tuple[Any, ...]:
    results = []
    env = maze_environment(horizon=horizon)
    generator = np.random.default_rng(11)
    try:
        for map_id in maps:
            env.set_task(map_id)
            env.reset()
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(
                    int(generator.integers(MAZERUNNER_ACTIONS))
                )
                done = terminated or truncated
            results.append(
                MazeRunnerEpisodeResult(
                    task_id=map_id,
                    rollout_seed=0,
                    goals_total=GOALS,
                    completions=env.completed_goals,
                    episode_return=env.episode_return,
                    decisions=env.decisions,
                )
            )
    finally:
        env.close()
    return tuple(results)


def test_events_score_the_goal_fraction_and_reject_a_wrong_goal_count() -> None:
    active = contract()
    config = resolved()
    results = sample_results(tuple(DEVELOPMENT_TASKS[:3]))
    rows = events(
        active,
        config,
        results,
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert len(rows) == 3
    for event in rows:
        assert event.kind == "episode" and event.event_index == 1
        assert event.denominator == GOALS
        assert event.native_return == float(event.numerator)
        assert event.true_count is None
        assert 1 <= event.step <= HORIZON
    wrong = MazeRunnerEpisodeResult(
        task_id=results[0].task_id,
        rollout_seed=0,
        goals_total=GOALS + 1,
        completions=(),
        episode_return=0.0,
        decisions=5,
    )
    with pytest.raises(ResultValidationError, match="ran 4 goals"):
        events(
            active,
            config,
            [wrong],
            checkpoint="checkpoint.pt",
            split="development",
            history="retained",
        )


def test_secondary_records_include_failures_and_label_success_only_means() -> None:
    config = resolved()
    results = sample_results(tuple(DEVELOPMENT_TASKS[:4]))
    summary = mazerunner_secondary(config.environment, results)
    assert summary["maps"] == 4.0
    assert summary["goal_fraction"] == pytest.approx(
        float(np.mean([r.goal_fraction for r in results]))
    )
    assert summary["full_sequence_rate"] == pytest.approx(
        float(np.mean([r.full_sequence for r in results]))
    )
    assert summary["native_return_mean"] == pytest.approx(
        float(np.mean([r.episode_return for r in results]))
    )
    for index in range(GOALS):
        rate = summary[f"goal_{index}_reached_rate"]
        assert rate is not None and 0.0 <= rate <= 1.0
        fixed = summary[f"goal_{index}_steps_fixed_budget_mean"]
        assert fixed is not None
        if rate == 0.0:
            # A goal nobody reached costs the whole native horizon, and its
            # success-conditioned mean is undefined rather than zero.
            assert fixed == pytest.approx(float(HORIZON))
            assert summary[f"goal_{index}_steps_success_only_mean"] is None
    assert json.dumps(summary, allow_nan=False)
    with pytest.raises(ResultValidationError, match="at least one map"):
        mazerunner_secondary(config.environment, [])


def test_the_declared_random_reference_runs_without_a_policy() -> None:
    env = maze_environment(horizon=30)
    try:
        reference = random_policy_goal_fraction(
            env, task_ids=tuple(DEVELOPMENT_TASKS[:4]), generator_seed=3
        )
    finally:
        env.close()
    assert 0.0 <= reference["random_reference_goal_fraction"] <= 1.0
    assert 0.0 <= reference["random_reference_full_sequence"] <= 1.0
    assert json.dumps(reference, allow_nan=False)


def test_history_modes_are_declared_per_benchmark() -> None:
    assert history_modes("mazerunner") == (
        "retained",
        "goal-cleared",
        "summary-cleared",
    )
    assert history_modes("count_recall") == (
        "retained",
        "current-token",
        "summary-cleared",
    )
    with pytest.raises(ContractError, match="No evaluator is implemented"):
        history_modes("multidomain_popgym")


# ----------------------------------------------------------------------
# C1 lifecycle
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("condition", ("transition", "transition_dat"))
def test_mazerunner_cpu_train_export_evaluate_lifecycle(
    condition: str, tmp_path: Path
) -> None:
    active = contract()
    config = resolved(condition, output_root=tmp_path, smoke=True)
    assert config.environment.name == "mazerunner"
    assert config.environment.goals == GOALS
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
        # The declared relabeler is live on the replay this run trains from.
        assert isinstance(
            experiment.reasoned_dataset.relabeler, ReconstructingRelabeler
        )
        run_record, rows, summary = evaluate(
            active,
            config,
            experiment,
            checkpoint="checkpoint.pt",
            split="development",
            task_cap=3,
        )
        assert run_record.status == "completed"
        assert len(rows) == 3
        assert summary["maps"] == 3.0
        with pytest.raises(ResultValidationError, match="incomplete/unexpected"):
            validate_benchmark_results([active], [run_record], list(rows))
    finally:
        close_experiment(experiment)
    # The repeated-laps axis on the same weights: every map replayed for two
    # native budgets, its first lap the plain episode map for map.
    from reasoned_icrl.experiments.horizon import (
        check_laps_prefix,
        continued_laps,
        window_goal_counts,
    )

    # The smoke recipe shrinks the episode, so the laps task starts from the
    # contract as the smoke run actually played it.
    smoke_contract = replace(active, environment=config.environment)
    budget = 2 * config.environment.horizon
    laps_contract, laps_config = continued_laps(smoke_contract, config, budget)
    laps_experiment = load_experiment(
        laps_config, work_directory=run, persist_configuration=False
    )
    try:
        load_selected_checkpoint(laps_experiment, "checkpoint.pt")
        laps_run, lap_rows, laps_summary = evaluate(
            laps_contract,
            laps_config,
            laps_experiment,
            checkpoint="checkpoint.pt",
            split="development",
            task_cap=3,
        )
        assert laps_run.status == "completed" and laps_run.retention == "complete"
        assert laps_run.outer_length == budget
        assert laps_run.charged_calls == 3 * budget
        assert laps_run.reset_only_steps is not None
        assert laps_run.reset_only_steps >= 3
        assert all(row.kind == "attempt" for row in lap_rows)
        assert all(row.denominator == GOALS for row in lap_rows)
        assert all(row.outer_length == budget for row in lap_rows)
        by_task: dict[int, list[Any]] = {}
        for row in lap_rows:
            by_task.setdefault(row.task_id, []).append(row)
        assert set(by_task) == set(DEVELOPMENT_TASKS[:3])
        for task_rows in by_task.values():
            ordered = sorted(task_rows, key=lambda r: r.event_index)
            assert [r.event_index for r in ordered] == list(range(1, len(ordered) + 1))
            assert ordered[0].start_step == 1
            assert all(r.complete for r in ordered[:-1])
            assert all(
                r.start_step <= min(r.goal_steps, default=r.start_step)
                and max(r.goal_steps, default=r.end_step) <= r.end_step
                for r in ordered
            )
            assert all(len(r.goal_steps) == r.numerator for r in ordered)
            assert ordered[-1].end_step in (budget - 1, budget)
        assert check_laps_prefix(rows, lap_rows) == {"compared": 3, "units": 3}
        windows = window_goal_counts(lap_rows, outer_length=budget, window=budget // 2)
        assert all(len(v) == 2 for v in windows.values())
        assert laps_summary["maps"] == 3.0 and laps_summary["laps_finished_mean"] >= 1
        # The lap-cleared companion runs on the same task and is scored the same way.
        cleared_run, cleared_rows, _ = evaluate(
            laps_contract,
            laps_config,
            laps_experiment,
            checkpoint="checkpoint.pt",
            split="development",
            task_cap=3,
            history="attempt-cleared",
        )
        assert cleared_run.status == "completed" and len(cleared_rows) >= 3
        assert check_laps_prefix(rows, cleared_rows) == {"compared": 3, "units": 3}
        with pytest.raises(ContractError, match="history modes"):
            evaluate(
                active,
                config,
                laps_experiment,
                checkpoint="checkpoint.pt",
                split="development",
                task_cap=1,
                history="attempt-cleared",
            )
    finally:
        close_experiment(laps_experiment)
    experiment = load_experiment(config, work_directory=run)
    try:
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
            cleared, coverage = rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[0]],
                rollout_seed=0,
                history="goal-cleared",
            )
        finally:
            environment.close()
        # An outer reset clears the cache, so roster order cannot change a map.
        assert forward[0].completions == reverse[1].completions
        assert forward[1].completions == reverse[0].completions
        assert cleared[0].goals_total == GOALS
        completed_before_end = sum(
            g.step < cleared[0].decisions for g in cleared[0].completions
        )
        assert coverage["intervention_count"] == completed_before_end
        assert coverage["intervention_episodes"] == int(completed_before_end > 0)
        assert (
            sum(
                v
                for k, v in coverage.items()
                if k.startswith("action_") and k.endswith("_count")
            )
            == cleared[0].decisions
        )
        assert 0 <= coverage["blocked_move_fraction"] <= 1
        with pytest.raises(ContractError, match="history intervention"):
            rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[0]],
                rollout_seed=0,
                history="current-token",
            )

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
    """One episode, encoded once as a whole sequence and once step by step."""
    environment = _evaluation_environment(config)
    try:
        packet, _ = environment.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        packets = [packet]
        feedback = [np.zeros(MAZERUNNER_ACTIONS + 1, dtype=np.float32)]
        done = False
        action = STAY
        while not done:
            packet, reward, terminated, truncated, _ = environment.step(action)
            row = np.zeros(MAZERUNNER_ACTIONS + 1, dtype=np.float32)
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


# ----------------------------------------------------------------------
# The larger-maze axis: the trained protocol on a larger maze
# ----------------------------------------------------------------------


def test_a_larger_maze_keeps_the_trained_protocol_and_only_enlarges_the_grid() -> None:
    env = maze_15(size=25, horizon=1389, protocol_size=MAZE_15)
    try:
        packet, info = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        native = env.native.unwrapped
        assert env.protocol == "mazerunner-15-randomized-actions"
        assert env.protocol_size == MAZE_15 and env.size == 25
        assert native.maze_dim == 25 and native.time_limit == 1389
        assert len(native.goal_positions) == GOALS and env.randomized_actions
        # The packet width is the trained one; coordinates are divided by the
        # maze played and decode back onto the 25x25 grid.
        assert packet["current"].shape == (public_width(GOALS),) == (13,)
        assert env.decode(packet)["position"] == native.start == (23, 12)
        goals = env.decode(packet)["goals"]
        assert goals.shape == (GOALS, 2) and np.all((goals >= 0) & (goals < 25))
        assert env.decode(packet)["timer"] == 0
        assert "achieved" not in info
        # The state identity carries the maze played and its budget, so a
        # trained-size snapshot never restores into a larger-maze task.
        trained = maze_15()
        try:
            trained.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
            with pytest.raises(ContractError, match="contract does not match"):
                env.load_state_dict(trained.state_dict())
        finally:
            trained.close()
    finally:
        env.close()
    with pytest.raises(ContractError, match="smaller than the trained"):
        maze_15(size=11, protocol_size=MAZE_15)
    with pytest.raises(ContractError, match="odd maze"):
        maze_15(size=16, protocol_size=MAZE_15)
    # Without the trained size, 25 names no protocol: the axis is explicit.
    with pytest.raises(ContractError, match="No MazeRunner protocol"):
        maze_15(size=25)
    raw = {
        "name": "mazerunner",
        "benchmark": "mazerunner-15-randomized-actions",
        "size": 25,
        "attempts": 1,
        "horizon": 1389,
        "goals": GOALS,
        "randomized_actions": True,
        "parallel_envs": 1,
        "protocol_size": MAZE_15,
    }
    declared = environment_config(raw)
    assert (declared.size, declared.protocol_size, declared.outer_length) == (
        25,
        MAZE_15,
        1389,
    )
    assert (
        environment_config(
            {k: v for k, v in raw.items() if k != "protocol_size"}
            | {"size": MAZE_15, "horizon": 500}
        ).protocol_size
        is None
    )
    with pytest.raises(ContractError, match="protocol_size disagrees"):
        environment_config({**raw, "protocol_size": MAZE})
    with pytest.raises(ContractError, match="at least the protocol"):
        environment_config({**raw, "size": MAZE})
    with pytest.raises(ContractError, match="size disagrees with its protocol"):
        environment_config({k: v for k, v in raw.items() if k != "protocol_size"})
    with pytest.raises(ContractError, match="does not accept a MazeRunner protocol"):
        environment_config(
            {
                "name": "darkroom",
                "benchmark": "darkroom",
                "size": 5,
                "attempts": 2,
                "horizon": 20,
                "parallel_envs": 1,
                "protocol_size": MAZE_15,
            }
        )


# ----------------------------------------------------------------------
# The repeated-laps axis
# ----------------------------------------------------------------------


def test_repeated_laps_replay_the_same_map_across_reset_only_boundaries() -> None:
    from reasoned_icrl.environments.mazerunner import MazeLap

    task = DEVELOPMENT_TASKS[2]
    plain = maze_environment(horizon=5)
    laps = maze_environment(horizon=5, meta_horizon=17)
    try:
        first_plain, _ = plain.reset(options={"task_index": task})
        first_laps, info = laps.reset(options={"task_index": task})
        assert all(np.array_equal(first_plain[k], first_laps[k]) for k in first_plain)
        assert (info["attempt_index"], info["reset_only"], info["step_in_task"]) == (
            0,
            False,
            0,
        )
        native = laps.native.unwrapped
        maze = np.array(native.maze)
        goals = list(native.goal_positions)
        dirs = np.array(native.action_dirs)
        # Lap 1 is the plain episode, record for record: same packets, rewards.
        script = [STAY, 0, 1, 2, 3]
        for call, action in enumerate(script, 1):
            plain_packet, plain_reward, plain_term, plain_trunc, _ = plain.step(action)
            packet, reward, terminated, truncated, info = laps.step(action)
            assert all(np.array_equal(plain_packet[k], packet[k]) for k in packet)
            assert reward == plain_reward and not terminated and not truncated
            assert (info["step_in_task"], info["step_in_episode"]) == (call, call)
            assert info["reset_only"] is False
        assert plain_term and plain_trunc  # the plain task ends at its timer
        assert info["attempt_done"] and not laps.partial_attempt
        assert laps.completed_attempts == (
            MazeLap(
                index=0,
                first_step=1,
                last_step=5,
                steps=5,
                success=False,
                native_return=0.0,
                complete=True,
                goals=0,
                goal_steps=(),
            ),
        )
        # The reset-only call: nothing executes, nothing is paid, the call is
        # charged, and the packet is the task's own first reset packet again.
        packet, reward, terminated, truncated, info = laps.step(STAY)
        assert reward == 0.0 and not terminated and not truncated
        assert (info["reset_only"], info["attempt_done"], info["attempt_index"]) == (
            True,
            False,
            1,
        )
        assert info["step_in_task"] == 6 and info["step_in_episode"] == 0
        assert all(np.array_equal(first_laps[k], packet[k]) for k in packet)
        assert np.array_equal(np.array(native.maze), maze)
        assert list(native.goal_positions) == goals
        assert np.array_equal(np.array(native.action_dirs), dirs)
        assert native.pos == native.start and native.timer == 0
        assert native.active_goal_idx == 0
        assert laps.collection_counters() == {
            "charged_calls": 6,
            "physical_actions": 5,
            "reset_only_steps": 1,
            "tasks_started": 1,
            "tasks_completed": 0,
        }
        # Lap 2 reaches every goal (pinned onto the start, as the goal test
        # does); its goal steps are calls of the outer task.
        native.goal_positions = [native.start] * GOALS
        rewards = []
        for _ in range(GOALS):
            _, reward, terminated, truncated, info = laps.step(STAY)
            rewards.append(reward)
        assert rewards == [1.0, 1.0, 1.0] and not terminated
        assert info["attempt_done"] and info["goals_completed"] == GOALS
        lap = laps.completed_attempts[1]
        assert (lap.first_step, lap.last_step, lap.steps, lap.goals) == (7, 9, 3, 3)
        assert lap.goal_steps == (7, 8, 9) and lap.success and lap.complete
        assert laps.task_return == 3.0 and laps.partial_attempt is None
        # The reset redraws the map's own goals: the pinned ones are gone.
        _, _, _, _, info = laps.step(STAY)
        assert info["reset_only"] and info["step_in_task"] == 10
        assert list(native.goal_positions) == goals
        # Lap 3 (calls 11-15) ends by the timer, call 16 resets, and the
        # budget cuts lap 4 after one call: the partial lap.
        for call in range(11, 18):
            _, _, terminated, truncated, info = laps.step(STAY)
            assert terminated == (call == 17) and not truncated
        assert [r.index for r in laps.completed_attempts] == [0, 1, 2]
        third = laps.completed_attempts[2]
        assert (third.first_step, third.last_step) == (11, 15)
        partial = laps.partial_attempt
        assert partial is not None and not partial.complete and not partial.success
        assert (partial.index, partial.first_step, partial.last_step) == (3, 17, 17)
        assert partial.steps == 1
        assert laps.collection_counters() == {
            "charged_calls": 17,
            "physical_actions": 14,
            "reset_only_steps": 3,
            "tasks_started": 1,
            "tasks_completed": 1,
        }
        with pytest.raises(ContractError, match="requires an outer reset"):
            laps.step(STAY)
        # The state round-trips with its laps, and never into a one-episode task.
        twin = maze_environment(horizon=5, meta_horizon=17)
        try:
            twin.load_state_dict(laps.state_dict())
            assert twin.completed_attempts == laps.completed_attempts
            assert twin.partial_attempt == partial and twin.task_return == 3.0
        finally:
            twin.close()
        with pytest.raises(ContractError, match="contract does not match"):
            plain.load_state_dict(laps.state_dict())
    finally:
        plain.close()
        laps.close()


def test_the_laps_budget_is_an_evaluation_only_field_of_the_trained_maze() -> None:
    env = maze_15(meta_horizon=4000)
    try:
        assert env.meta_horizon == 4000 and env.size == MAZE_15
        assert env.protocol == "mazerunner-15-randomized-actions"
        # Nothing in the packet names a lap: width and fields are the trained ones.
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        assert packet["current"].shape == (public_width(GOALS),)
        assert [field.name for field in env.fields] == [
            field.name for field in maze_15().fields
        ]
    finally:
        env.close()
    with pytest.raises(ContractError, match="at least one native episode"):
        maze_15(meta_horizon=400)
    with pytest.raises(ContractError, match="separate axes"):
        maze_15(size=17, horizon=642, protocol_size=MAZE_15, meta_horizon=4000)
    raw = {
        "name": "mazerunner",
        "benchmark": "mazerunner-15-randomized-actions",
        "size": MAZE_15,
        "attempts": 1,
        "horizon": HORIZON_15,
        "goals": GOALS,
        "randomized_actions": True,
        "parallel_envs": 1,
        "meta_horizon": 4000,
    }
    declared = environment_config(raw)
    assert (declared.meta_horizon, declared.outer_length, declared.protocol_size) == (
        4000,
        4000,
        None,
    )
    with pytest.raises(ContractError, match="at least one native episode"):
        environment_config({**raw, "meta_horizon": 400})
    with pytest.raises(ContractError, match="separate axes"):
        environment_config(
            {**raw, "size": 17, "horizon": 642, "protocol_size": MAZE_15}
        )
    with pytest.raises(ContractError, match="does not accept a native meta budget"):
        environment_config(
            {
                "name": "tmaze",
                "benchmark": "tmaze-passive-l32-256-v3",
                "size": 128,
                "attempts": 1,
                "horizon": 129,
                "parallel_envs": 1,
                "movement_penalty": -0.0078125,
                "training_corridors": [32, 64, 128, 256],
                "meta_horizon": 1000,
            }
        )


def test_a_map_change_redraws_maze_and_goals_and_keeps_the_action_map() -> None:
    from reasoned_icrl.environments.mazerunner import RELAYOUT_TASKS

    task = DEVELOPMENT_TASKS[3]
    env = maze_15(meta_horizon=2000)
    twin = maze_15(meta_horizon=2000)
    try:
        with pytest.raises(ContractError, match="inside the trained episode"):
            env.set_layout_period(400)
        env.set_layout_period(600)
        twin.set_layout_period(600)
        first, _ = env.reset(options={"task_index": task})
        twin.reset(options={"task_index": task})
        native = env.native.unwrapped
        maze0 = np.array(native.maze)
        goals0 = list(native.goal_positions)
        dirs0 = np.array(native.action_dirs)
        seen: list[tuple[int, int]] = []
        calls = 0
        while not env._done:
            before = env.layout_index
            packet, _, _, _, info = env.step(STAY)
            twin_packet, *_ = twin.step(STAY)
            assert all(np.array_equal(packet[k], twin_packet[k]) for k in packet)
            calls += 1
            if info["reset_only"]:
                seen.append((calls, env.layout_index))
                if env.layout_index != before:
                    # A new 15x15 maze and new goals; the action map is kept
                    # and nothing in the packet names the change.
                    assert env.layout_index == before + 1
                    assert native.maze_dim == MAZE_15
                    assert not (
                        np.array_equal(np.array(native.maze), maze0)
                        and list(native.goal_positions) == goals0
                    )
                    assert np.array_equal(np.array(native.action_dirs), dirs0)
                    assert packet["current"].shape == first["current"].shape
        # Timer-ended laps of 500 steps: resets at calls 501, 1002, 1503; the
        # first boundary at or after 600 is 1002, the next after 1200 is 1503,
        # so laps 1-2 play the roster map, lap 3 map 1 and the cut lap map 2.
        assert seen == [(501, 0), (1002, 1), (1503, 2)]
        laps = env.completed_attempts
        assert [lap.layout for lap in laps] == [0, 0, 1]
        assert env.partial_attempt is not None and env.partial_attempt.layout == 2
        assert all(env.relayout_seed(k) in RELAYOUT_TASKS for k in (1, 2, 3))
        assert len({env.relayout_seed(k) for k in (1, 2, 3)}) == 3
        # The state round-trips mid-task with its map and action map.
        restored = maze_15(meta_horizon=2000)
        try:
            restored.load_state_dict(env.state_dict())
            assert restored.layout_index == env.layout_index
            assert restored.completed_attempts == laps
        finally:
            restored.close()
    finally:
        env.close()
        twin.close()
    plain = maze_15()
    try:
        with pytest.raises(ContractError, match="repeated-laps budget"):
            plain.set_layout_period(1000)
    finally:
        plain.close()
