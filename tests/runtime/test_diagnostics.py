"""Task-side information diagnostics against the real pinned environments.

They are evaluation-only: witness search and optimal labels may use evaluator
state, but input equality and history-based resolution use only public
streams. A passing diagnostic says the task can reward history; it does not say
any trained agent uses it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reasoned_icrl.environments.base import DEVELOPMENT_TASKS
from reasoned_icrl.environments.count_recall import CountRecallEnv
from reasoned_icrl.environments.dark_key_to_door import DarkKeyToDoorEnv
from reasoned_icrl.environments.darkroom import DarkRoomEnv
from reasoned_icrl.environments.mazerunner import MazeRunnerEnv
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.runtime.diagnostics import (
    count_recall_diagnostic,
    darkroom_diagnostic,
    incompatible_actions,
    key_to_door_diagnostic,
    mazerunner_diagnostic,
    public_signature,
    task_diagnostic,
)
from tests.experiments.fixtures import load_fixture_study

ROOT = Path(__file__).resolve().parents[2]
SLICE = 6
NAMES = ("dark_key_to_door", "count_recall", "mazerunner", "darkroom")


def study() -> Any:
    return load_fixture_study("stage1")


def contract(name: str) -> Any:
    return study().contract(name)


def config_for(name: str) -> Any:
    return experiment_config(
        contract(name),
        study(),
        condition="transition",
        seed=101,
        repository=ROOT,
        device="cpu",
    )


def test_count_recall_needs_history_at_every_scored_position() -> None:
    environment = CountRecallEnv(variant="easy", split="development", initial_seed=0)
    try:
        result = count_recall_diagnostic(
            environment, stream_ids=tuple(DEVELOPMENT_TASKS[:SLICE])
        )
    finally:
        environment.close()
    assert result.measurements["scored_decisions"] == float(SLICE * 51)
    assert result.measurements["distinct_current_packets"] > 102
    assert result.passed == (result.measurements["ambiguous_packet_fraction"] >= 0.5)
    assert "RL2 and time" in result.statement


def test_mazerunner_requires_observed_full_input_conflicts() -> None:
    environment = MazeRunnerEnv(
        size=11,
        goals=3,
        horizon=250,
        variant="fixed-actions",
        split="development",
        initial_seed=0,
    )
    try:
        result = mazerunner_diagnostic(
            environment, map_ids=tuple(DEVELOPMENT_TASKS[:SLICE])
        )
    finally:
        environment.close()
    assert result.measurements["reachable_states"] > 0
    assert result.measurements["reachable_states"] <= SLICE * 250
    assert result.passed == (result.measurements["ambiguous_packet_fraction"] >= 0.05)
    assert "Tied optimal routes alone do not count" in result.statement


def test_key_to_door_uses_public_discovery_and_real_recall() -> None:
    environment = DarkKeyToDoorEnv(
        size=8,
        physical_horizon=50,
        meta_horizon=500,
        scored_attempts=8,
        split="development",
        initial_seed=0,
    )
    try:
        result = key_to_door_diagnostic(
            environment, task_ids=tuple(DEVELOPMENT_TASKS[:SLICE])
        )
    finally:
        environment.close()
    assert (
        0
        <= result.measurements["public_recall_success_tasks"]
        <= result.measurements["public_discovery_tasks"]
    )
    assert "memoryless_later_attempt_success" not in result.measurements
    assert "not proof" in result.statement


def test_optimal_ties_do_not_manufacture_a_history_requirement() -> None:
    assert not incompatible_actions([{0, 1}])
    assert not incompatible_actions([{0, 1}, {1, 2}])
    assert not incompatible_actions([set(), {1}])
    assert incompatible_actions([{0, 1}, {2, 3}])


def test_full_input_signature_includes_feedback_previous_and_time() -> None:
    from types import SimpleNamespace

    import numpy as np

    obs = {
        key: np.zeros((1, 3), dtype=np.float32)
        for key in ("current", "previous", "outcome", "event", "valid")
    }
    rl2 = np.zeros((1, 6), dtype=np.float32)
    time = np.zeros((1, 1), dtype=np.int64)
    sequence = SimpleNamespace(current_timestep=(obs, rl2, time))
    original = public_signature(sequence)
    for value in (*obs.values(), rl2, time):
        value.flat[0] = 1
        assert public_signature(sequence) != original
        value.flat[0] = 0


def test_the_diagnostics_refuse_an_empty_roster() -> None:
    door = DarkKeyToDoorEnv(
        size=8,
        physical_horizon=50,
        meta_horizon=500,
        scored_attempts=8,
        split="development",
        initial_seed=0,
    )
    try:
        with pytest.raises(ContractError, match="needs tasks"):
            key_to_door_diagnostic(door, task_ids=())
    finally:
        door.close()


@pytest.mark.parametrize("name", NAMES)
def test_each_contract_resolves_to_its_own_task_diagnostic(name: str) -> None:
    result = task_diagnostic(contract(name), config_for(name), task_cap=3)
    assert result.benchmark == contract(name).environment.name
    assert result.statement and result.question.endswith("?")


def test_darkroom_public_outcome_resolves_the_second_attempt() -> None:
    environment = DarkRoomEnv(split="iid", initial_seed=0)
    try:
        result = darkroom_diagnostic(environment, task_ids=(0, 3, 8, 14, 17))
    finally:
        environment.close()
    assert result.measurements["public_recall_success_tasks"] == 5.0
    assert result.measurements["incompatible_input_groups"] >= 1.0
    assert result.passed
