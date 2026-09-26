"""Foundations shared by every study: contract validation, native sampling,
causal records, replay relabeling and the benchmark record round trip.

The retired DAT roster is used as a fixture because its transition arms and
its three contracts exercise every branch of the record validator.
"""

from __future__ import annotations

import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from amago.hindsight import FrozenTraj, NoOpRelabeler

from reasoned_icrl.environments.base import PublicDecision
from reasoned_icrl.experiments.benchmarks import load_contract
from reasoned_icrl.experiments.contracts import ContractError, ResultValidationError
from reasoned_icrl.experiments.environments import build_environment
from reasoned_icrl.experiments.records import (
    read_benchmark_results,
    validate_benchmark_results,
    write_benchmark_results,
)
from reasoned_icrl.runtime.replay import ReconstructingRelabeler
from tests.experiments.fixtures import fixture_results, load_fixture_study

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda d: d["training"].update(epsilon_anneal_steps=250001), "anneal"),
        (lambda d: d["training"].update(max_sequence_length=49), "prefix"),
        (
            lambda d: d["evaluation"]["splits"].update(
                final={"source": "development", "count": 64}
            ),
            "overlap",
        ),
        (lambda d: d["qualification"].update(maximum_primary=2), "criteria"),
        (lambda d: d["environment"].update(meta_horizon=400), "guarantee"),
        (lambda d: d.update(status="C2"), "unverified"),
        (lambda d: d.update(typo="ignored"), "sections"),
        (lambda d: d["environment"].update(name="count_recall"), "meta budget"),
        # MazeRunner accepts a meta budget (the evaluation-only repeated-laps axis) and
        # is refused on its protocol instead.
        (lambda d: d["environment"].update(name="mazerunner"), "MazeRunner protocol"),
    ],
)
def test_contract_rejects_invalid_decisions(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    raw = yaml.safe_load(
        (ROOT / "configs/environments/dark_key_to_door.yaml").read_text()
    )
    mutation(raw)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError, match=message):
        load_contract(path)


@pytest.mark.parametrize("name", ["dark_key_to_door", "count_recall", "mazerunner"])
def test_native_sampling_is_isolated_and_repeatable(name: str) -> None:
    contract = load_fixture_study("dat_benchmarks").contract(name)
    environment = contract.environment
    if name == "mazerunner":
        environment = replace(
            environment,
            benchmark="mazerunner-randomized-actions",
            randomized_actions=True,
        )
    mapping = {"environment": asdict(environment)}
    caller = random.getstate()
    left = build_environment(mapping, split="development", seed=31)
    right = build_environment(mapping, split="development", seed=31)
    try:
        for _ in range(2):
            a, _ = left.reset()
            # Unrelated global draws must not affect native task sampling.
            random.random()
            b, _ = right.reset()
            np.testing.assert_array_equal(a["current"], b["current"])
            for action in [0, 3, 4, 2, 1]:
                a, ar, at, ax, _ = left.step(action)
                b, br, bt, bx, _ = right.step(action)
                np.testing.assert_array_equal(a["current"], b["current"])
                assert (ar, at, ax) == (br, bt, bx)
    finally:
        left.close()
        right.close()
        random.setstate(caller)


def test_causal_record_distinguishes_zero_outcome_reset_and_outer_terminal() -> None:
    state = np.zeros(4, dtype=np.float32)
    physical = PublicDecision(state, state, state, 0, 1.0, physical_done=True)
    packet, rl2 = physical.inputs(5)
    assert packet["event"].tolist() == [1, 0, 1]
    assert rl2.tolist() == [1, 1, 0, 0, 0, 0]
    assert not physical.learner_terminal
    reset = PublicDecision(state, new_attempt=True)
    packet, rl2 = reset.inputs(5)
    assert packet["event"].tolist() == [0, 1, 0]
    assert not rl2.any()
    assert replace(physical, outer_truncated=True).learner_terminal


def test_relabeling_precedes_reconstruction_and_rejects_stale_feedback() -> None:
    trajectory = FrozenTraj(
        obs={"raw": np.zeros((3, 2))},
        rl2s=np.zeros((3, 3)),
        time_idxs=np.arange(3)[:, None],
        rews=np.zeros((2, 1)),
        dones=np.array([[False], [True]]),
        actions=np.zeros((2, 2)),
    )
    order: list[str] = []

    class ChangeReward(NoOpRelabeler):  # type: ignore[misc]
        def relabel(self, traj: Any) -> Any:
            order.append("relabel")
            traj.rews[1] = 1
            traj.rl2s[2, 0] = 1
            return traj

    def reconstruct(traj: Any) -> Any:
        order.append("reconstruct")
        assert traj.rews[1, 0] == 1
        assert set(traj.obs) == {"raw"}
        state = np.zeros((3, 2), dtype=np.float32)
        traj.obs = dict(
            current=state,
            previous=state.copy(),
            outcome=state.copy(),
            event=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=np.float32),
            valid=np.ones((3, 1), dtype=np.float32),
        )
        return traj

    output = ReconstructingRelabeler(ChangeReward(), reconstruct)(trajectory)
    assert order == ["relabel", "reconstruct"]
    assert not trajectory.rews.any() and set(trajectory.obs) == {"raw"}
    assert output.rews[1, 0] == 1

    def stale(traj: Any) -> Any:
        result = reconstruct(traj)
        result.rl2s[:] = 0
        return result

    with pytest.raises(ContractError, match="alignment"):
        ReconstructingRelabeler(ChangeReward(), stale)(trajectory)


def test_records_roundtrip_pairing_and_missing_runs(tmp_path: Path) -> None:
    study, runs, events = fixture_results()
    path = write_benchmark_results(
        tmp_path / "results.json", study.contracts, runs, events
    )
    saved_runs, saved_events = read_benchmark_results(path, study.contracts)
    assert saved_runs == tuple(runs) and saved_events == tuple(events)
    with pytest.raises(ResultValidationError, match="roster"):
        validate_benchmark_results(study.contracts, runs, events[:-1])
    with pytest.raises(ResultValidationError, match="Duplicate"):
        validate_benchmark_results(study.contracts, runs, [*events, events[0]])
    failed = replace(runs[0], status="failed", note="evaluation interrupted")
    validate_benchmark_results(study.contracts, [failed], [])
    with pytest.raises(ResultValidationError, match="completed"):
        validate_benchmark_results(study.contracts, [failed], [events[0]])
    with pytest.raises(ResultValidationError, match="denominator"):
        validate_benchmark_results(
            study.contracts, runs, [replace(events[0], denominator=0), *events[1:]]
        )
