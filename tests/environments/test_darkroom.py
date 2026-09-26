"""DarkRoom: the K-attempt lifecycle, goal rosters, packets and restorable state."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from reasoned_icrl.environments.base import PACKET_KEYS
from reasoned_icrl.environments.darkroom import (
    DARKROOM_IID_SOURCE_INDICES,
    DARKROOM_OOD_SOURCE_INDICES,
    DARKROOM_TRAIN_SOURCE_INDICES,
    DarkRoomEnv,
    darkroom_source_indices,
)
from reasoned_icrl.experiments.contracts import ContractError, arrays_equal

UP, RIGHT, DOWN, LEFT, STAY = range(5)


def test_darkroom_is_seeded_and_gymnasium_compatible() -> None:
    check_env(DarkRoomEnv(size=5, attempts=5, horizon=32), skip_render_check=True)
    first, second = DarkRoomEnv(initial_seed=7), DarkRoomEnv(initial_seed=7)
    first_packet, first_info = first.reset(seed=19)
    second_packet, second_info = second.reset(seed=19)
    assert arrays_equal(first_packet, second_packet)
    assert first_info["evaluator_task_index"] == second_info["evaluator_task_index"]
    assert set(first_packet) == set(PACKET_KEYS)
    assert first.observation_space.contains(first_packet)
    # The initial token carries no physical evidence and starts an attempt.
    assert first_packet["event"].tolist() == [0.0, 1.0, 0.0]


def test_attempts_reset_internally_and_only_the_task_end_truncates() -> None:
    env = DarkRoomEnv(size=3, attempts=3, horizon=1, goal_partition=None)
    env.set_task(0)
    env.reset(seed=0)
    for attempt in range(3):
        packet, reward, terminated, truncated, info = env.step(STAY)
        assert reward == 0.0 and terminated is False
        assert truncated is (attempt == 2)
        assert info["attempt_done"] is True and info["attempt_index"] == attempt
        fields = env.public_fields(packet)
        # After an internal reset the current token starts the next attempt at
        # the centre with the boundary flag raised; the outcome keeps the
        # physical endpoint; the last attempt's boundary is the outer end.
        if attempt < 2:
            assert fields["attempt_boundary"] == 1 and fields["attempt"] == attempt + 1
            assert packet["event"].tolist() == [1.0, 1.0, 1.0]
        else:
            assert packet["event"].tolist() == [1.0, 0.0, 1.0]
    assert [record.index for record in env.completed_attempts] == [0, 1, 2]
    assert env.partial_attempt is None
    with pytest.raises(ContractError, match="outer reset"):
        env.step(STAY)


def test_reaching_the_goal_pays_once_and_ends_the_attempt() -> None:
    env = DarkRoomEnv(size=5, attempts=2, horizon=8)
    env.set_task(DARKROOM_TRAIN_SOURCE_INDICES[0])  # goal cell 1 = (0, 1)
    env.reset(seed=0)
    assert env.goal == (0, 1)
    rewards = []
    for action in (UP, UP, LEFT):
        packet, reward, _, truncated, info = env.step(action)
        rewards.append(reward)
    assert rewards == [0.0, 0.0, 1.0]
    assert info["attempt_done"] and info["attempt_success"]
    assert env.completed_attempts[0].success and env.completed_attempts[0].steps == 3
    # The outcome token is the goal cell; the current token restarts the task.
    assert env.public_fields({"current": packet["outcome"]})["position"] == (0, 1)
    assert env.public_fields(packet)["position"] == (2, 2)
    assert not truncated


def test_quadrant_partition_rosters_are_frozen_and_disjoint() -> None:
    train, iid, ood = (
        darkroom_source_indices(size=5, split=split, goal_partition="quadrant-v1")
        for split in ("train", "iid", "ood")
    )
    assert (train, iid, ood) == (
        DARKROOM_TRAIN_SOURCE_INDICES,
        DARKROOM_IID_SOURCE_INDICES,
        DARKROOM_OOD_SOURCE_INDICES,
    )
    assert not (set(train) & set(iid)) and not (set(train) & set(ood))
    assert not set(iid) & set(ood)
    assert (
        darkroom_source_indices(
            size=5, split="validation", goal_partition="quadrant-v1"
        )
        == iid
    )
    assert darkroom_source_indices(size=3, split="train", goal_partition=None) == tuple(
        range(9)
    )
    with pytest.raises(ContractError, match="no OOD"):
        darkroom_source_indices(size=3, split="ood", goal_partition=None)
    with pytest.raises(ContractError, match="5x5"):
        darkroom_source_indices(size=3, split="train", goal_partition="quadrant-v1")
    with pytest.raises(ContractError, match="outside the active partition"):
        DarkRoomEnv(split="ood", source_indices=(DARKROOM_TRAIN_SOURCE_INDICES[0],))


def test_sampling_stays_inside_the_split_roster_and_is_reproducible() -> None:
    first = DarkRoomEnv(split="train", initial_seed=17)
    second = DarkRoomEnv(split="train", initial_seed=17)
    drawn = []
    for _ in range(128):
        _, info = first.reset()
        _, other = second.reset()
        assert info["evaluator_task_index"] == other["evaluator_task_index"]
        drawn.append(int(info["evaluator_task_index"]))
    assert set(drawn) == set(DARKROOM_TRAIN_SOURCE_INDICES)
    ood = DarkRoomEnv(split="ood")
    with pytest.raises(ContractError, match="outside the roster"):
        ood.reset(options={"task_index": DARKROOM_TRAIN_SOURCE_INDICES[0]})


def test_checkpoint_restores_the_exact_next_step_and_rejects_other_contracts() -> None:
    original = DarkRoomEnv(attempts=2, horizon=3, split="iid")
    original.reset(seed=13)
    original.step(RIGHT)
    state = original.state_dict()
    expected = original.step(DOWN)
    restored = DarkRoomEnv(attempts=2, horizon=3, split="iid")
    restored.reset(seed=99)
    restored.load_state_dict(state)
    actual = restored.step(DOWN)
    assert arrays_equal(expected[0], actual[0])
    assert expected[1:4] == actual[1:4] and expected[4] == actual[4]
    other = DarkRoomEnv(attempts=2, horizon=3, split="ood")
    with pytest.raises(ContractError, match="contract does not match"):
        other.load_state_dict(state)


def test_render_marks_agent_and_goal() -> None:
    env = DarkRoomEnv(size=5, attempts=1, horizon=3, split="iid")
    env.set_task(DARKROOM_IID_SOURCE_INDICES[0])
    env.reset(seed=0)
    rows = [row.split() for row in env.render().splitlines()]
    assert rows[2][2] == "A"
    goal_row, goal_column = divmod(DARKROOM_IID_SOURCE_INDICES[0], 5)
    assert rows[goal_row][goal_column] == "G"
    assert np.asarray(rows).shape == (5, 5)
