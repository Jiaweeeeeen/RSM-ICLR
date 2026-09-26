"""Seed/task bootstrap estimates, paired contrasts, curves and run collection."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from reasoned_icrl.analysis import curves, estimates, paired_contrasts, training_curve
from reasoned_icrl.analysis.runs import collect, complete_events, markdown_table
from reasoned_icrl.experiments.contracts import ResultValidationError
from reasoned_icrl.experiments.records import write_benchmark_results
from tests.experiments.fixtures import fixture_results


def test_estimates_average_cells_then_bootstrap_seeds_and_tasks() -> None:
    study, _, events = fixture_results()
    protocol = study.contracts[0].protocol
    rows = estimates([e for e in events if e.protocol == protocol], samples=200)
    by_condition = {row.condition: row for row in rows}
    assert set(by_condition) == set(study.conditions)
    assert by_condition["transition_dat"].estimate == 1.0
    assert by_condition["transition"].estimate == 0.0
    assert (
        by_condition["transition"].seeds == 3 and by_condition["transition"].tasks == 2
    )
    for row in rows:
        assert row.lower == row.estimate == row.upper  # constant fixture
        assert set(row.per_seed) == set(study.training_seeds)


def test_paired_contrasts_require_identical_rosters() -> None:
    study, _, events = fixture_results()
    protocol = study.contracts[1].protocol
    panel = [e for e in events if e.protocol == protocol]
    rows = paired_contrasts(panel, [("transition_dat", "transition")], samples=100)
    assert len(rows) == 1 and rows[0].estimate == 1.0
    assert paired_contrasts(panel, [("transition_dat", "missing")]) == ()
    # A whole seed missing on one side is an incomplete matrix, not corrupt data:
    # the contrast is estimated on the seeds the two conditions share and says so.
    fewer = [
        e for e in panel if not (e.condition == "transition" and e.training_seed == 2)
    ]
    partial = paired_contrasts(fewer, [("transition_dat", "transition")], samples=100)
    assert len(partial) == 1
    assert partial[0].seeds == rows[0].seeds - 1
    assert 2 not in partial[0].per_seed
    # A task missing *within* a shared seed is unpaired data and stays an error.
    broken = [
        e
        for e in panel
        if not (
            e.condition == "transition"
            and e.training_seed == 2
            and e.task_id == min(x.task_id for x in panel)
        )
    ]
    with pytest.raises(ResultValidationError, match="paired roster"):
        paired_contrasts(broken, [("transition_dat", "transition")], samples=10)


def test_curves_follow_the_event_index() -> None:
    study, _, events = fixture_results()
    protocol = study.contracts[0].protocol
    points = curves([e for e in events if e.protocol == protocol], samples=50)
    indices = sorted({p.event_index for p in points})
    assert indices == list(range(1, study.contracts[0].environment.attempts + 1))
    assert all(p.estimate in (0.0, 1.0) for p in points)


def test_collect_reads_complete_and_partial_evaluations(tmp_path: Path) -> None:
    study, runs, events = fixture_results()
    contract = study.contracts[2]
    for run in runs:
        if run.protocol != contract.protocol:
            continue
        directory = (
            tmp_path / contract.protocol / run.condition / f"seed-{run.training_seed}"
        )
        evaluation = directory / "eval" / "final-retained"
        evaluation.mkdir(parents=True)
        (directory / "checkpoint.pt").write_bytes(b"")
        selected = [
            e
            for e in events
            if e.protocol == run.protocol
            and e.condition == run.condition
            and e.training_seed == run.training_seed
        ]
        if run.training_seed == 2:
            (evaluation / "benchmark_results.json").write_text(
                json.dumps(
                    {
                        "partial_task_cap": 1,
                        "runs": [asdict(run)],
                        "events": [asdict(e) for e in selected[:1]],
                    }
                )
            )
        else:
            write_benchmark_results(
                evaluation / "benchmark_results.json", [contract], [run], selected
            )
        (evaluation / "benchmark_secondary.json").write_text(
            json.dumps({"goal_fraction": 0.5})
        )
    records = collect(study, contract, tmp_path, split="final", history="retained")
    assert {(r.condition, r.seed) for r in records} == {
        (c, s) for c in study.conditions for s in study.training_seeds
    }
    assert sum(r.partial for r in records) == 2
    assert all(r.secondary == {"goal_fraction": 0.5} for r in records)
    full = complete_events(records)
    assert {e.training_seed for e in full} == {0, 1}
    assert collect(study, contract, tmp_path, split="development") and all(
        r.run is None for r in collect(study, contract, tmp_path, split="development")
    )
    assert "| condition |" in markdown_table([{"condition": "raw", "estimate": 0.5}])
    assert markdown_table([]) == "_no rows_"
    assert training_curve(tmp_path) == []


def test_fixture_events_are_frozen_dataclasses() -> None:
    _, _, events = fixture_results()
    with pytest.raises((AttributeError, TypeError)):
        events[0].numerator = 3  # type: ignore[misc]
    assert replace(events[0], numerator=1).numerator == 1
