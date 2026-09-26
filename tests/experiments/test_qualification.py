"""The pure qualification records: primary metric, declared references, labels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.qualification import (
    measure_reference,
    primary_from_events,
    scheduled_epochs,
)
from reasoned_icrl.experiments.records import BenchmarkEvent
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


def event(numerator: int, denominator: int = 1) -> BenchmarkEvent:
    return BenchmarkEvent(
        protocol="native-keydoor-fixed500-first8",
        benchmark="dark_key_to_door",
        condition="transition",
        training_seed=101,
        checkpoint="policy_epoch_10",
        split="development",
        history="retained",
        task_id=1_000_000,
        cluster_id=1_000_000,
        rollout_seed=0,
        kind="attempt",
        event_index=1,
        step=5,
        numerator=numerator,
        denominator=denominator,
        native_return=2.0 if numerator else 0.0,
    )


def test_the_primary_metric_is_the_mean_scored_fraction() -> None:
    assert primary_from_events([event(1), event(0), event(1), event(0)]) == 0.5
    assert primary_from_events(
        [
            replace(event(2, 3), native_return=2.0),
            replace(event(3, 3), native_return=3.0),
        ]
    ) == pytest.approx((2 / 3 + 1.0) / 2)
    with pytest.raises(ContractError, match="at least one event"):
        primary_from_events([])


@pytest.mark.parametrize("name", NAMES)
def test_every_reference_is_measured_and_evaluation_only(name: str) -> None:
    reference = measure_reference(contract(name), config_for(name), task_cap=3)
    assert reference.kind == "evaluation-only"
    assert 0.0 <= reference.primary <= 1.0
    assert reference.note


def test_the_count_recall_reference_prefers_the_fitted_prior_over_uniform() -> None:
    reference = measure_reference(
        contract("count_recall"), config_for("count_recall"), task_cap=SLICE
    )
    assert reference.primary >= 1 / 27
    assert "prior" in reference.name or "random" in reference.name
    if "prior" in reference.name:
        # Fitted on the training band, scored on development: disjoint sets.
        assert "disjoint" in reference.note
        assert "not information-free" in reference.note


def test_scheduled_epochs_reads_the_retained_policy_labels(tmp_path: Path) -> None:
    weights = tmp_path / "ckpts" / "policy_weights"
    weights.mkdir(parents=True)
    for label in (100, 50, 999, 950):
        (weights / f"policy_epoch_{label}.pt").write_bytes(b"")
    (weights / "other.pt").write_bytes(b"")
    assert scheduled_epochs(tmp_path) == [50, 100, 950, 999]
    assert scheduled_epochs(tmp_path / "missing") == []
