"""R4/R6 acceptance: the tier report regenerates from saved records.

Fixture panels of the 8M Key-to-Door tier are written under a temporary
study root exactly as the evaluator writes them; the report lists every
declared cell and seed in its completeness ledger, scores only saved panels,
marks the pending cells, classifies the declared contrasts and draws the
complete-record curves.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from reasoned_icrl.analysis.statistics import CurvePoint
from reasoned_icrl.analysis.tier import (
    SUPPORTED_RISK_FRACTION,
    TIER_TABLES,
    budget_attempt_cutoff,
    completeness_rows,
    supported_attempt_cutoff,
    write_tier_report,
)
from reasoned_icrl.experiments.benchmarks import EvaluationSplit
from reasoned_icrl.experiments.evaluation import evaluation_directory
from reasoned_icrl.experiments.records import BenchmarkRun, write_benchmark_results
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from tests.analysis.test_revised_statistics import SEEDS, _panel

SPLIT = "development"


def _study() -> tuple[Any, Any]:
    study = load_summary_memory_study()
    contract = study.contract("dark_key_to_door")
    contract = replace(
        contract,
        evaluation=replace(
            contract.evaluation,
            splits={
                name: EvaluationSplit(split.source, 2, split.offset)
                for name, split in contract.evaluation.splits.items()
            },
        ),
    )
    return study, contract


def _write_panel(
    root: Path,
    contract: Any,
    doors: dict[str, int],
    *,
    history: str,
    rule: str,
    checkpoint: str,
) -> None:
    events = [
        replace(e, history=history, checkpoint_rule=rule, checkpoint=checkpoint)  # type: ignore[arg-type]
        for e in _panel(doors)
    ]
    for condition in doors:
        for seed in SEEDS:
            directory = root / contract.protocol / condition / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "checkpoint.pt").write_bytes(b"")
            weights = directory / "ckpts" / "policy_weights"
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "policy_epoch_999.pt").write_bytes(b"")
            (directory / "metrics.json").write_text(
                json.dumps({"runtime_seconds": 36000.0, "gradient_steps": 127_872})
            )
            (directory / "provenance.json").write_text(
                json.dumps(
                    {
                        "gpu": "NVIDIA RTX A6000"
                        if seed == 42
                        else "NVIDIA RTX PRO 6000",
                        "host": "node-5",
                        "slurm_job_id": "140003",
                        "wandb": {"url": "https://wandb.ai/x/y/runs/z"},
                    }
                )
            )
            (directory / "systems.json").write_text(
                json.dumps(
                    {
                        "decision_latency_seconds": 0.002,
                        "peak_gpu_bytes": 1_400_000_000,
                        "measured": {
                            "charged_calls": 8_000_000,
                            "physical_actions": 7_200_000,
                            "reset_only_steps": 800_000,
                            "validation": {"charged_calls": 160_000},
                        },
                    }
                )
            )
            run = BenchmarkRun(
                contract.protocol,
                "dark_key_to_door",
                condition,
                seed,
                checkpoint,
                SPLIT,
                history,
                "completed",
                checkpoint_rule=rule,  # type: ignore[arg-type]
                retention="complete",
                metric="doors_completed",
                charged_calls=1000,
                physical_actions=900,
                reset_only_steps=100,
            )
            rows = [
                e
                for e in events
                if e.condition == condition and e.training_seed == seed
            ]
            panel = directory / "eval" / evaluation_directory(SPLIT, history, rule)
            write_benchmark_results(
                panel / "benchmark_results.json", [contract], [run], rows
            )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_the_tier_report_scores_saved_panels_and_marks_the_rest(tmp_path: Path) -> None:
    study, contract = _study()
    doors = {"full_context": 6, "full_dual_relational": 8, "full_dual_content": 5}
    _write_panel(
        tmp_path,
        contract,
        doors,
        history="retained",
        rule="endpoint",
        checkpoint="policy_epoch_999",
    )
    _write_panel(
        tmp_path,
        contract,
        {k: v - 2 for k, v in doors.items()},
        history="attempt-cleared",
        rule="endpoint",
        checkpoint="policy_epoch_999",
    )
    _write_panel(
        tmp_path,
        contract,
        {k: v - 1 for k, v in doors.items()},
        history="retained",
        rule="selected",
        checkpoint="policy_epoch_900",
    )
    report = write_tier_report(
        study, contract, tmp_path, split=SPLIT, samples=20, seed=7, qualified=True
    )
    assert report.root == tmp_path / "reports" / contract.protocol / SPLIT / "tier"
    assert set(report.tables) == set(TIER_TABLES)
    assert set(report.figures) == {"cumulative", "attempts"}
    for path in (*report.tables.values(), *report.figures.values()):
        assert path.is_file() and path.stat().st_size > 0
    assert report.complete_primary_cells == (
        "full_context",
        "full_dual_relational",
        "full_dual_content",
    )
    assert report.pending_primary_cells == (
        "fixed_summary",
        "fixed_segment",
        "fixed_window",
    )
    assert any("pending endpoint records" in note for note in report.notes)
    ledger = _read(report.root / "completeness.csv")
    assert {(r["condition"], int(r["seed"])) for r in ledger} == {
        (condition, seed)
        for condition in study.tier(contract.name).compared_cells
        for seed in study.training_seeds
    }
    complete = [r for r in ledger if r["status"] == "complete"]
    assert len(complete) == 9 and all(r["endpoint_reached"] == "True" for r in complete)
    assert {r["group"] for r in ledger if r["condition"] == "full_gru"} == {
        "supplementary"
    }
    assert all(
        r["status"] == "missing" for r in ledger if r["condition"] == "fixed_window"
    )
    row = next(r for r in complete if r["condition"] == "full_context")
    assert (
        row["measured_charged_calls"] == "8000000"
        and row["validation_charged_calls"] == "160000"
    )
    assert (
        row["panel_endpoint_retained"] == "True"
        and row["panel_selected_retained"] == "True"
    )
    cells = _read(report.root / "cells.csv")
    endpoint = {
        r["condition"]: r
        for r in cells
        if r["history"] == "retained" and r["checkpoint_rule"] == "endpoint"
    }
    assert float(endpoint["full_dual_relational"]["estimate"]) == 8.0
    assert endpoint["full_dual_relational"]["metric"] == "doors_completed"
    assert endpoint["full_dual_relational"]["status"] == "complete"
    assert endpoint["fixed_summary"]["status"] == "pending"
    assert endpoint["full_context"]["per_seed"].count(":") == 3
    contrasts = _read(report.root / "contrasts.csv")
    primary = {
        r["contrast"]: r for r in contrasts if r["checkpoint_rule"] == "endpoint"
    }
    assert primary["DAT full - ordinary full"]["estimate"] == "2.0"
    assert (
        primary["DAT full - ordinary full"]["disposition"]
        == "consistent practical gain"
    )
    assert (
        primary["DAT full - dual content"]["disposition"] == "consistent practical gain"
    )
    assert primary["summary - segment"]["disposition"] == "pending"
    assert primary["full DAT - summary"]["disposition"] == "pending"
    assert {r["checkpoint_rule"] for r in contrasts} == {"endpoint", "selected"}
    dependence = _read(report.root / "interventions.csv")
    assert {r["dependence"] for r in dependence} == {"retained - attempt-cleared"}
    assert (
        float(
            next(r for r in dependence if r["condition"] == "full_context")["estimate"]
        )
        == 2.0
    )
    cumulative = _read(report.root / "cumulative.csv")
    assert {r["condition"] for r in cumulative} == set(doors)
    last = [r for r in cumulative if r["condition"] == "full_dual_relational"][-1]
    assert last["step"] == "500" and float(last["estimate"]) == 8.0
    attempts = _read(report.root / "attempts.csv")
    assert all(int(r["risk_set"]) == 6 for r in attempts if r["event_index"] == "1")
    intervals = _read(report.root / "intervals.csv")
    assert {r["condition"] for r in intervals} == set(doors)
    first = _read(report.root / "first_success.csv")
    assert all(
        r["median_step"] == "25" for r in first if r["condition"] == "full_context"
    )
    costs = _read(report.root / "costs.csv")
    assert {(r["condition"], r["gpu"]) for r in costs} == {
        (c, g) for c in doors for g in ("NVIDIA RTX A6000", "NVIDIA RTX PRO 6000")
    }
    notes = json.loads((report.root / "notes.json").read_text())
    assert notes["resampling_seed"] == 7 and notes["delta"] == 1.0
    assert notes["pending_primary_cells"] == [
        "fixed_summary",
        "fixed_segment",
        "fixed_window",
    ]


def test_an_empty_root_yields_pending_rows_and_a_full_ledger(tmp_path: Path) -> None:
    study, contract = _study()
    report = write_tier_report(study, contract, tmp_path, split=SPLIT, samples=5)
    assert report.complete_primary_cells == ()
    assert len(report.pending_primary_cells) == 6
    assert report.figures == {}
    ledger = completeness_rows(study, contract, tmp_path, split=SPLIT)
    assert {(r["condition"], int(r["seed"])) for r in ledger} == {
        (condition, seed)
        for condition in study.tier(contract.name).compared_cells
        for seed in study.training_seeds
    }
    assert all(r["status"] == "missing" for r in ledger)
    cells = _read(report.root / "cells.csv")
    assert {r["status"] for r in cells} == {"pending"}
    contrasts = _read(report.root / "contrasts.csv")
    assert {r["disposition"] for r in contrasts} == {"pending"}
    assert {r["contrast"] for r in contrasts} == {
        contrast.name for contrast in study.tier(contract.name).contrasts
    }


def _curve(condition: str, risk_sets: list[int]) -> list[CurvePoint]:
    return [
        CurvePoint(
            protocol="p",
            condition=condition,
            split="development",
            history="retained",
            event_index=index,
            estimate=1.0,
            lower=1.0,
            upper=1.0,
            checkpoint_rule="endpoint",
            risk_set=risk,
        )
        for index, risk in enumerate(risk_sets, start=1)
    ]


def test_the_attempt_figure_stops_where_half_the_cells_no_longer_reach() -> None:
    """Late attempts exist only for cells that already finished many short,
    successful attempts, so the success panel stops at the last attempt that
    at least half of every condition's cells finished; the risk-set panel and
    the CSV keep every attempt."""
    assert SUPPORTED_RISK_FRACTION == 0.5
    # 192 cells; attempts 1-10 by everyone, then a shrinking tail.
    full = _curve("full_context", [192] * 10 + [150, 120, 96, 95, 40, 10, 2, 1])
    assert supported_attempt_cutoff(full) == 13
    # The window's cells drop below half two attempts earlier: the figure
    # follows the least supported condition.
    window = _curve("fixed_window", [192] * 10 + [90, 80, 70, 60, 5])
    assert supported_attempt_cutoff(full + window) == 10
    # A curve whose first attempt is the only supported one still shows it.
    assert supported_attempt_cutoff(_curve("fixed_segment", [64, 3, 1])) == 1
    assert supported_attempt_cutoff([]) == 1


def test_the_attempt_figure_stops_where_a_failure_can_no_longer_finish() -> None:
    """A failed attempt costs 50 steps plus a reset, so a task that fails every
    attempt finishes attempt 9 at call 458 and is cut off inside attempt 10;
    the risk-set rule alone would keep attempt 10, where every finished attempt
    is a success by construction."""
    _, contract = _study()
    assert contract.environment.horizon == 50
    assert contract.environment.outer_length == 500
    assert budget_attempt_cutoff(contract) == 9


def test_the_ledger_names_every_resume_and_its_dropped_replay_files(
    tmp_path: Path,
) -> None:
    """A fit resumed after its pack died shows the label it resumed from and,
    for the recorded lenient resume, how many replay files the FIFO had
    evicted; uninterrupted fits show nothing (R6)."""
    study, contract = _study()
    run = tmp_path / contract.protocol / "fixed_summary" / "seed-42"
    run.mkdir(parents=True)
    records = [
        {
            "schema": "reasoned-icrl-resume.v1",
            "resumed_label": 800,
            "next_epoch": 801,
            "allow_missing_replay": True,
            "replay_deviation": {
                "schema": "replay-resume-deviation.v1",
                "expected_files": 10000,
                "missing_files": 800,
                "missing": ["a.npz"],
            },
            "gpu": "NVIDIA RTX 6000 Ada Generation",
            "slurm_job_id": "140804",
        },
        {
            "schema": "reasoned-icrl-resume.v1",
            "resumed_label": 900,
            "next_epoch": 901,
            "allow_missing_replay": False,
            "replay_deviation": None,
            "gpu": None,
            "slurm_job_id": None,
        },
    ]
    (run / "resumes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    ledger = completeness_rows(study, contract, tmp_path, split=SPLIT)
    by_cell = {(r["condition"], r["seed"]): r for r in ledger}
    assert by_cell[("fixed_summary", 42)]["resumes"] == (
        "label 800: 800 of 10000 replay files evicted and dropped "
        "(NVIDIA RTX 6000 Ada Generation, job 140804); label 900: exact"
    )
    assert by_cell[("fixed_summary", 100)]["resumes"] == ""
    assert by_cell[("full_context", 42)]["resumes"] == ""
