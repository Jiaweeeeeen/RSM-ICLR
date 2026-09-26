"""The shared BaseEnv contract, checked on every environment.

Every environment publishes the same five-key packet, selects tasks from a
declared roster on reset, refuses tasks outside it, and restores its exact
next step from ``state_dict``. These checks are parametrised over all four so
that a new environment cannot drift from the contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from reasoned_icrl.environments import (
    ConcentrationEnv,
    CountRecallEnv,
    DarkKeyToDoorEnv,
    DarkRoomEnv,
    MazeRunnerEnv,
    XLandMiniGridEnv,
)
from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    OUTCOME_PROTOCOL,
    PACKET_KEYS,
    Attempt,
    BaseEnv,
    PublicDecision,
)
from reasoned_icrl.experiments.contracts import ContractError, arrays_equal

FACTORIES: dict[str, Callable[[], BaseEnv]] = {
    "darkroom": lambda: DarkRoomEnv(split="iid", attempts=2, horizon=6),
    "dark_key_to_door": lambda: DarkKeyToDoorEnv(
        physical_horizon=4, meta_horizon=12, scored_attempts=2, split="development"
    ),
    "concentration": lambda: ConcentrationEnv(split="development"),
    "count_recall": lambda: CountRecallEnv(split="development"),
    "count_recall_hard": lambda: CountRecallEnv(variant="hard", split="development"),
    "mazerunner": lambda: MazeRunnerEnv(horizon=20, split="development"),
    "mazerunner_15": lambda: MazeRunnerEnv(
        size=15, variant="randomized-actions", horizon=20, split="development"
    ),
    "xland_minigrid": lambda: XLandMiniGridEnv(
        physical_horizon=6, attempts=2, scored_from=2, split="development"
    ),
}
FIRST_TASK = {
    "darkroom": 0,
    "dark_key_to_door": DEVELOPMENT_TASKS[0],
    "concentration": DEVELOPMENT_TASKS[0],
    "count_recall": DEVELOPMENT_TASKS[0],
    "count_recall_hard": DEVELOPMENT_TASKS[0],
    "mazerunner": DEVELOPMENT_TASKS[0],
    "mazerunner_15": DEVELOPMENT_TASKS[0],
    "xland_minigrid": DEVELOPMENT_TASKS[0],
}


@pytest.fixture(params=sorted(FACTORIES))
def environment(request: pytest.FixtureRequest) -> Any:
    env = FACTORIES[request.param]()
    env.name = request.param
    yield env
    env.close()


def test_packet_contract_is_shared(environment: BaseEnv) -> None:
    assert isinstance(environment, BaseEnv)
    assert environment.contract["schema"] == OUTCOME_PROTOCOL
    assert environment.contract["environment_protocol"] == environment.protocol
    assert set(environment.observation_space.spaces) == set(PACKET_KEYS)
    width = environment.width
    assert environment.observation_space["current"].shape == (width,)
    assert width == sum(len(field.low) for field in environment.fields)
    assert environment.action_space.n == environment.action_count


def test_reset_selects_from_the_roster_and_refuses_outsiders(environment: Any) -> None:
    packet, info = environment.reset(seed=3)
    assert set(packet) == set(PACKET_KEYS)
    assert environment.observation_space.contains(packet)
    assert packet["event"].tolist() == [0.0, 1.0, 0.0]
    assert np.all(packet["previous"] == 0) and np.all(packet["outcome"] == 0)
    assert info["evaluator_task_index"] == environment.evaluator_task_index
    assert environment.evaluator_task_index in environment.source_indices
    outsider = 987_654_321
    with pytest.raises(ContractError, match="outside the roster"):
        environment.set_task(outsider)
    with pytest.raises(ContractError, match="outside the roster"):
        environment.reset(options={"task_index": outsider})
    with pytest.raises(ContractError, match="must be an integer"):
        environment.reset(options={"task_index": True})


def test_set_task_pins_the_next_reset(environment: Any) -> None:
    task = FIRST_TASK[environment.name]
    environment.set_task(task)
    _, info = environment.reset(seed=1)
    assert info["evaluator_task_index"] == task
    _, info = environment.reset()
    assert info["evaluator_task_index"] == task


def test_every_step_carries_physical_evidence_or_a_reset(environment: Any) -> None:
    environment.set_task(FIRST_TASK[environment.name])
    environment.reset(seed=0)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        action = int(rng.integers(environment.action_count))
        packet, reward, terminated, truncated, _ = environment.step(action)
        available, new_attempt, physical_done = packet["event"].tolist()
        assert environment.observation_space.contains(packet)
        if available:
            assert np.array_equal(
                packet["outcome"], packet["outcome"].astype(np.float32)
            )
        else:
            # A reset-only step executed no physical action and paid nothing.
            assert reward == 0.0 and new_attempt == 1.0 and physical_done == 0.0
            assert np.all(packet["previous"] == 0) and np.all(packet["outcome"] == 0)
        done = terminated or truncated
    with pytest.raises(ContractError, match="outer reset"):
        environment.step(0)
    with pytest.raises(ContractError, match="outside the action space"):
        environment.reset()
        environment.step(environment.action_count)


def test_state_dict_restores_the_exact_next_step(environment: Any) -> None:
    environment.set_task(FIRST_TASK[environment.name])
    environment.reset(seed=5)
    rng = np.random.default_rng(1)
    for _ in range(3):
        environment.step(int(rng.integers(environment.action_count)))
    state = environment.state_dict()
    assert state["schema"] == environment.state_schema
    action = int(rng.integers(environment.action_count))
    expected = environment.step(action)
    twin = FACTORIES[environment.name]()
    try:
        twin.set_task(FIRST_TASK[environment.name])
        twin.reset(seed=99)
        twin.load_state_dict(state)
        actual = twin.step(action)
    finally:
        twin.close()
    assert arrays_equal(expected[0], actual[0])
    assert expected[1:4] == actual[1:4]
    assert expected[4] == actual[4]
    # R6: the measured counters travel with the snapshot, so a resumed run
    # keeps counting from the saved state; a snapshot without them (pre-R6)
    # leaves the twin's own counters alone.
    assert state["counters"]["charged_calls"] == 3
    counted = FACTORIES[environment.name]()
    try:
        counted.set_task(FIRST_TASK[environment.name])
        counted.reset(seed=99)
        counted.load_state_dict(state)
        assert counted.collection_counters() == state["counters"]
        counted.step(action)
        assert counted.collection_counters()["charged_calls"] == 4
        legacy = {k: v for k, v in state.items() if k != "counters"}
        counted.load_state_dict(legacy)
        assert counted.collection_counters()["charged_calls"] == 4
    finally:
        counted.close()
    with pytest.raises(ContractError, match="contract does not match"):
        twin.load_state_dict({**state, "contract": ("other",)})


def test_public_decision_refuses_inconsistent_records() -> None:
    current = np.zeros(3, dtype=np.float32)
    packet, feedback = PublicDecision(current=current, new_attempt=True).inputs(4)
    assert (
        packet["event"].tolist() == [0.0, 1.0, 0.0] and feedback.tolist() == [0.0] * 5
    )
    packet, feedback = PublicDecision(
        current, current, current, executed_action=2, reward=0.5, physical_done=True
    ).inputs(4)
    assert packet["event"].tolist() == [1.0, 0.0, 1.0]
    assert feedback.tolist() == [0.5, 0.0, 0.0, 1.0, 0.0]
    with pytest.raises(ContractError, match="requires previous/outcome/action"):
        PublicDecision(current, previous=current).inputs(4)
    with pytest.raises(ContractError, match="cannot hide feedback"):
        PublicDecision(current, reward=1.0).inputs(4)
    with pytest.raises(ContractError, match="requires a real outcome"):
        PublicDecision(current, physical_done=True).inputs(4)
    with pytest.raises(ContractError, match="outside native action space"):
        PublicDecision(current, current, current, executed_action=4).inputs(4)


def test_attempt_records_validate_their_counters() -> None:
    Attempt(
        index=0,
        first_step=1,
        last_step=3,
        steps=3,
        success=True,
        native_return=1.0,
        complete=True,
    )
    with pytest.raises(ContractError, match="span disagrees"):
        Attempt(0, 1, 5, 3, False, 0.0, True)
    with pytest.raises(ContractError, match="partial attempt cannot report success"):
        Attempt(0, 1, 3, 3, True, 1.0, False)


def test_collection_counters_measure_what_the_actor_actually_did() -> None:
    """R1: the recipe's nominal product is not a measured interaction count."""
    from reasoned_icrl.environments.dark_key_to_door import DarkKeyToDoorEnv

    env = DarkKeyToDoorEnv(
        size=8,
        physical_horizon=50,
        meta_horizon=60,
        scored_attempts=8,
        randomized_actions=False,
        split="development",
        source_indices=range(1_000_000, 1_000_004),
        initial_seed=0,
    )
    assert env.collection_counters() == {
        "charged_calls": 0,
        "physical_actions": 0,
        "reset_only_steps": 0,
        "tasks_started": 0,
        "tasks_completed": 0,
    }
    env.reset(seed=0)
    charged = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        charged += 1
        done = terminated or truncated
    counters = env.collection_counters()
    assert counters["charged_calls"] == charged
    assert counters["physical_actions"] + counters["reset_only_steps"] == charged
    # Key-to-Door soft-resets between attempts, so some calls execute nothing.
    assert counters["reset_only_steps"] > 0
    assert counters["tasks_started"] == 1
    assert counters["tasks_completed"] == 1

    env.reset(seed=1)
    assert env.collection_counters()["tasks_started"] == 2
    assert env.collection_counters()["tasks_completed"] == 1
