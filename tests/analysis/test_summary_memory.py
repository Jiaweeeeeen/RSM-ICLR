"""The summary-memory estimators on fixture records (M7.1).

Every estimator is checked against hand-computed values on a constant fixture
of the Key-to-Door matrix (two final tasks, three seeds, both checkpoint
rules) and on a small CountRecallHard query fixture, so a regression in the
arithmetic, the pairing or the panel keys cannot pass unnoticed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reasoned_icrl.analysis import (
    ATTENTION_REGIME_INTERACTION,
    PLAN_CONTRASTS,
    STRATUM_FIELDS,
    PlanContrast,
    checkpoint_rule_comparison,
    cost_table,
    curves,
    estimates,
    interaction,
    interval_successes,
    paired_contrasts,
    plan_contrasts,
    recovery_fraction,
    secondary_table,
    stratified_estimates,
    writes_table,
)
from reasoned_icrl.analysis.runs import RunRecords
from reasoned_icrl.experiments.benchmarks import EvaluationSplit
from reasoned_icrl.experiments.contracts import ResultValidationError
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    BenchmarkRun,
    validate_benchmark_results,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
)

SEEDS = (0, 1, 2)
ATTEMPTS = 8
SUCCESSES = {
    "raw": 6,
    "raw_dat": 7,
    "raw_dual_content": 6,
    "raw_gru": 3,
    "raw_segment": 4,
    "raw_summary": 5,
    "raw_dat_segment": 4,
    "raw_dat_summary": 6,
    "raw_dat_summary_relational_write_off": 5,
    "raw_dual_content_summary": 5,
    "raw_window": 4,
}
"""Successful attempts out of eight per task, identical for every seed and task."""
FINAL_EPOCH_SUCCESSES = {**SUCCESSES, "raw": 5, "raw_segment": 3}
"""The fixed-final-checkpoint supplement: two cells lower by one attempt."""
BOUNDED = {
    c for c in SUCCESSES if c not in ("raw", "raw_dat", "raw_dual_content", "raw_gru")
}
WRITES = [(index - 1) // 2 for index in range(1, ATTEMPTS + 1)]  # 0,0,1,1,2,2,3,3
SPAN = 11  # ten-step attempts separated by one reset-only step


def _study() -> tuple[Any, Any]:
    study = load_retired_summary_memory_study()
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


def _attempt(
    contract: Any,
    condition: str,
    seed: int,
    task: int,
    index: int,
    success: bool,
    *,
    rule: str = "selected",
) -> BenchmarkEvent:
    writes = WRITES[index - 1] if condition in BOUNDED else None
    age = None if writes is None or index == 1 else writes
    recent = None if age is None else min(age, index % 2)
    return BenchmarkEvent(
        protocol=contract.protocol,
        benchmark="dark_key_to_door",
        condition=condition,
        training_seed=seed,
        checkpoint="policy_epoch_400" if rule == "selected" else "policy_epoch_499",
        split="final",
        history="retained",
        task_id=task,
        cluster_id=task,
        rollout_seed=0,
        kind="attempt",
        event_index=index,
        step=10,
        numerator=int(success),
        denominator=1,
        native_return=2.0 if success else 0.0,
        writes_before_decision=writes,
        evidence_age_writes=age,
        evidence_age_writes_recent=recent,
        evidence_in_current_segment=None if recent is None else recent == 0,
        start_step=1 + SPAN * (index - 1),
        end_step=10 + SPAN * (index - 1),
        checkpoint_rule=rule,  # type: ignore[arg-type]
    )


def _run(contract: Any, condition: str, seed: int, rule: str) -> BenchmarkRun:
    return BenchmarkRun(
        contract.protocol,
        "dark_key_to_door",
        condition,
        seed,
        "policy_epoch_400" if rule == "selected" else "policy_epoch_499",
        "final",
        "retained",
        "completed",
        checkpoint_rule=rule,  # type: ignore[arg-type]
    )


def fixture(
    successes: dict[str, int] = SUCCESSES,
    *,
    rule: str = "selected",
    conditions: tuple[str, ...] | None = None,
) -> tuple[list[BenchmarkRun], list[BenchmarkEvent]]:
    """Constant per-condition success counts on the two-task final roster."""
    _, contract = _study()
    runs, events = [], []
    for condition in conditions or tuple(successes):
        for seed in SEEDS:
            runs.append(_run(contract, condition, seed, rule))
            for task in contract.roster("final"):
                for index in range(1, ATTEMPTS + 1):
                    events.append(
                        _attempt(
                            contract,
                            condition,
                            seed,
                            task,
                            index,
                            index <= successes[condition],
                            rule=rule,
                        )
                    )
    validate_benchmark_results([contract], runs, events)
    return runs, events


def test_the_plan_contrasts_name_effects_and_keep_seed_effects() -> None:
    _, events = fixture()
    rows = plan_contrasts(events, samples=50)
    by_name = {row.name: row for row in rows}
    assert [row.name for row in rows] == [row.name for row in PLAN_CONTRASTS]
    expected = {
        "Δ_relation|full": 1 / 8,
        "grounding: raw - raw_gru": 3 / 8,
        "Δ_budget": 2 / 8,
        "Δ_summary": 1 / 8,
        "Δ_window": 1 / 8,
        "Δ_relation|summary": 1 / 8,
        "Δ_relation|segment": 0.0,
        "Δ_write_branch": 1 / 8,
        "Δ_capacity|summary": 1 / 8,
        "grounding: raw_summary - raw_gru": 2 / 8,
    }
    for name, value in expected.items():
        row = by_name[name]
        assert row.contrast.estimate == pytest.approx(value)
        assert row.contrast.lower == row.contrast.upper == pytest.approx(value)
        assert set(row.contrast.per_seed) == set(SEEDS)
        assert all(v == pytest.approx(value) for v in row.contrast.per_seed.values())
        assert row.contrast.tasks == 2 and row.contrast.seeds == 3
        assert row.contrast.checkpoint_rule == "selected"
    assert by_name["Δ_budget"].effect_of_interest == 0.10
    assert by_name["Δ_budget"].meets_effect and by_name["Δ_window"].meets_effect
    assert not by_name["Δ_relation|segment"].meets_effect
    assert by_name["Δ_relation|segment"].contrast.excludes_zero is False
    assert by_name["Δ_budget"].contrast.excludes_zero
    # A missing cell yields no row, never a zero.
    _, partial = fixture(conditions=("raw", "raw_segment"))
    assert [row.name for row in plan_contrasts(partial, samples=10)] == ["Δ_budget"]
    assert isinstance(PLAN_CONTRASTS[0], PlanContrast)


def test_the_recovery_fraction_follows_its_reporting_rule() -> None:
    _, events = fixture()
    reported = recovery_fraction(events, deficit_present=True, samples=50)
    assert len(reported) == 1
    row = reported[0]
    assert row.interpretable and row.fraction == pytest.approx(0.5)
    assert row.budget.estimate == pytest.approx(0.25)
    assert row.summary.estimate == pytest.approx(0.125)
    assert row.per_seed == {seed: pytest.approx(0.5) for seed in SEEDS}
    assert row.reason.startswith("reported")
    # Never clipped: a summary above the full-prefix reference reports > 1.
    _, above = fixture({**SUCCESSES, "raw_summary": 8})
    assert recovery_fraction(above, deficit_present=True, samples=50)[
        0
    ].fraction == pytest.approx(2.0)
    # The development screen failed: both differences stand, no fraction.
    withheld = recovery_fraction(events, deficit_present=False, samples=50)[0]
    assert withheld.fraction is None and not withheld.interpretable
    assert "deficit screen" in withheld.reason
    assert withheld.budget.estimate == pytest.approx(0.25)
    # The budget interval includes zero: seeds disagree on the sign.
    _, contract = _study()
    wins = {"raw": (2, 6, 6), "raw_segment": (6, 2, 6), "raw_summary": (5, 5, 5)}
    mixed = [
        _attempt(contract, condition, seed, task, index, index <= wins[condition][seed])
        for condition in wins
        for seed in SEEDS
        for task in contract.roster("final")
        for index in range(1, ATTEMPTS + 1)
    ]
    undecided = recovery_fraction(mixed, deficit_present=True, samples=400)[0]
    assert undecided.fraction is None and "includes zero" in undecided.reason
    assert undecided.budget.lower < 0.0 < undecided.budget.upper


def test_the_interaction_is_the_paired_double_difference() -> None:
    _, events = fixture()
    rows = interaction(events, samples=50)
    assert len(rows) == 1
    row = rows[0]
    assert (row.first, row.second) == ATTENTION_REGIME_INTERACTION
    # (6 - 5) - (4 - 4) attempts of eight.
    assert row.estimate == pytest.approx(1 / 8)
    assert row.lower == row.upper == pytest.approx(1 / 8)
    assert row.per_seed == {seed: pytest.approx(1 / 8) for seed in SEEDS}
    assert row.tasks == 2 and row.seeds == 3
    _, partial = fixture(conditions=("raw_dat_summary", "raw_summary", "raw_segment"))
    assert interaction(partial, samples=10) == ()
    assert interaction(
        events, ("raw_dat", "raw"), ("raw_dat_summary", "raw_summary"), samples=10
    )[0].estimate == pytest.approx(0.0)


def test_the_writes_table_counts_decisions_after_a_write() -> None:
    _, events = fixture()
    rows = writes_table(events)
    assert {row.condition for row in rows} == BOUNDED  # full prefix: no counter
    summary = [row for row in rows if row.condition == "raw_summary"]
    assert [(row.first_index, row.last_index) for row in summary] == [
        (index, index) for index in range(1, ATTEMPTS + 1)
    ]
    assert [row.mean_writes for row in summary] == WRITES
    assert [row.fraction_after_write for row in summary] == [
        float(w >= 1) for w in WRITES
    ]
    assert all(row.events == 6 for row in summary)  # three seeds x two tasks
    headline = [
        row for row in writes_table(events, bins=1) if row.condition == "raw_summary"
    ]
    assert len(headline) == 1
    assert (headline[0].first_index, headline[0].last_index) == (1, ATTEMPTS)
    assert headline[0].fraction_after_write == pytest.approx(6 / 8)
    assert headline[0].mean_writes == pytest.approx(np.mean(WRITES))
    assert headline[0].per_seed_fraction_after_write == {
        seed: pytest.approx(6 / 8) for seed in SEEDS
    }
    halves = [
        row for row in writes_table(events, bins=2) if row.condition == "raw_summary"
    ]
    assert [(r.first_index, r.last_index) for r in halves] == [(1, 4), (5, 8)]
    assert [r.fraction_after_write for r in halves] == [0.5, 1.0]
    with pytest.raises(ResultValidationError, match="positive"):
        writes_table(events, bins=0)


def _query(
    seed: int, task: int, index: int, *, correct: bool, before: int, inside: int
) -> BenchmarkEvent:
    return BenchmarkEvent(
        protocol="count-recall-hard",
        benchmark="count_recall",
        condition="raw_summary",
        training_seed=seed,
        checkpoint="policy_epoch_1",
        split="final",
        history="retained",
        task_id=task,
        cluster_id=task,
        rollout_seed=0,
        kind="query",
        event_index=index,
        step=index,
        numerator=int(correct),
        denominator=1,
        native_return=(1.0 if correct else -1.0) / 207,
        true_count=before + inside,
        writes_before_decision=1 if before else 0,
        evidence_age_writes=1 if before else 0,
        count_before_current_segment=before,
        count_in_current_segment=inside,
        start_step=index,
        end_step=index,
    )


def test_stratified_estimates_group_by_the_retention_fields() -> None:
    _, events = fixture()
    by_flag = stratified_estimates(events, "evidence_in_current_segment", samples=50)
    rows = [r for r in by_flag if r.condition == "raw_summary"]
    assert [r.stratum for r in rows] == ["False", "True"]
    # Attempts 3, 5, 7 have their most recent evidence one write back; attempts
    # 2, 4, 6, 8 have it in the open segment; attempt 1 has no evidence.
    outside = next(r for r in rows if r.stratum == "False")
    inside = next(r for r in rows if r.stratum == "True")
    assert outside.events == 3 * 6 and inside.events == 4 * 6
    assert outside.estimate == pytest.approx(np.mean([1.0, 1.0, 0.0]))  # 3, 5 of 5
    assert inside.estimate == pytest.approx(np.mean([1.0, 1.0, 0.0, 0.0]))
    assert outside.tasks == 6 and outside.seeds == 3
    assert outside.lower <= outside.estimate <= outside.upper
    assert set(outside.per_seed) == set(SEEDS)
    assert all(r.field == "evidence_in_current_segment" for r in rows)
    # Full-prefix carriers carry None and contribute no stratum.
    assert not [r for r in by_flag if r.condition == "raw"]
    binned = stratified_estimates(
        events, "writes_before_decision", bins=[(0, 0), (1, 3)], samples=50
    )
    rows = [r for r in binned if r.condition == "raw_segment"]
    assert [(r.stratum, r.low, r.high) for r in rows] == [
        ("0", 0.0, 0.0),
        ("1-3", 1.0, 3.0),
    ]
    assert rows[0].estimate == pytest.approx(1.0)  # attempts 1, 2 of 4 successes
    assert rows[1].estimate == pytest.approx(2 / 6)  # attempts 3, 4 of 3..8
    exact = stratified_estimates(events, "writes_before_decision", samples=10)
    assert [r.stratum for r in exact if r.condition == "raw_segment"] == [
        "0",
        "1",
        "2",
        "3",
    ]
    # The count mass outside the current segment on CountRecallHard queries.
    queries = [
        _query(seed, task, index, correct=correct, before=before, inside=inside)
        for seed in (0, 1)
        for task in (2_000_000, 2_000_001)
        for index, correct, before, inside in (
            (1, True, 0, 1),
            (2, True, 0, 2),
            (3, False, 3, 0),
            (4, True, 2, 1),
            (5, False, 4, 0),
        )
    ]
    mass = stratified_estimates(
        queries, "count_before_current_segment", bins=[(0, 0), (1, 16)], samples=50
    )
    assert [(r.stratum, r.estimate) for r in mass] == [("0", 1.0), ("1-16", 1 / 3)]
    assert mass[1].events == 12 and mass[1].tasks == 4 and mass[1].seeds == 2
    assert stratified_estimates(queries, "true_count", samples=10)[0].stratum == "1"
    with pytest.raises(ResultValidationError, match="Unknown stratum field"):
        stratified_estimates(events, "numerator", samples=10)  # type: ignore[arg-type]
    assert "count_before_current_segment" in STRATUM_FIELDS


def test_interval_successes_bin_events_by_their_end_step() -> None:
    _, events = fixture()
    rows = interval_successes(events, interval=50, outer_length=100, samples=50)
    raw = [row for row in rows if row.condition == "raw"]
    assert [(row.start_step, row.end_step) for row in raw] == [(1, 50), (51, 100)]
    # Attempts 1-4 end at 10, 21, 32, 43; attempts 5-8 at 54, 65, 76, 87.
    assert raw[0].successes_per_task == pytest.approx(4.0)
    assert raw[1].successes_per_task == pytest.approx(2.0)
    assert raw[0].events_per_task == pytest.approx(4.0)
    assert raw[0].lower == raw[0].upper == pytest.approx(4.0)
    assert raw[0].per_seed == {seed: pytest.approx(4.0) for seed in SEEDS}
    assert raw[0].tasks == 2 and raw[0].seeds == 3
    # Without an outer length the windows run to the last end step seen.
    assert [
        (row.start_step, row.end_step)
        for row in interval_successes(events, interval=30, samples=10)
        if row.condition == "raw"
    ] == [(1, 30), (31, 60), (61, 87)]
    legacy = [replace(event, start_step=None, end_step=None) for event in events]
    with pytest.raises(ResultValidationError, match="start_step/end_step"):
        interval_successes(legacy, interval=50)
    with pytest.raises(ResultValidationError, match="positive"):
        interval_successes(events, interval=0)


def test_banded_curves_average_each_band_before_the_bootstrap() -> None:
    _, events = fixture(conditions=("raw", "raw_segment"))
    whole = [p for p in curves(events, samples=10) if p.condition == "raw"]
    assert [p.event_index for p in whole] == list(range(1, ATTEMPTS + 1))
    assert all(p.last_index is None for p in whole)
    halves = [p for p in curves(events, bins=2, samples=10) if p.condition == "raw"]
    assert [(p.event_index, p.last_index) for p in halves] == [(1, 4), (5, 8)]
    # raw succeeds on attempts 1-6: the first band is perfect, the second half.
    assert [p.estimate for p in halves] == [pytest.approx(1.0), pytest.approx(0.5)]
    assert halves[1].lower == halves[1].upper == pytest.approx(0.5)


def test_panels_separate_the_two_checkpoint_rules() -> None:
    _, selected = fixture()
    _, final = fixture(FINAL_EPOCH_SUCCESSES, rule="final-epoch")
    both = [*selected, *final]
    rows = estimates(both, samples=10)
    raw = {row.checkpoint_rule: row.estimate for row in rows if row.condition == "raw"}
    assert raw == {
        "selected": pytest.approx(6 / 8),
        "final-epoch": pytest.approx(5 / 8),
    }
    assert {row.checkpoint_rule for row in curves(both, samples=10)} == {
        "selected",
        "final-epoch",
    }
    budget = paired_contrasts(both, [("raw", "raw_segment")], samples=10)
    assert {row.checkpoint_rule: row.estimate for row in budget} == {
        "selected": pytest.approx(2 / 8),
        "final-epoch": pytest.approx(2 / 8),
    }
    comparison = checkpoint_rule_comparison(both, samples=50)
    by_name = {(row.kind, row.name): row for row in comparison}
    raw_row = by_name[("condition", "raw")]
    assert (raw_row.selected, raw_row.final_epoch) == (
        pytest.approx(6 / 8),
        pytest.approx(5 / 8),
    )
    assert raw_row.difference == pytest.approx(1 / 8)
    assert raw_row.lower == raw_row.upper == pytest.approx(1 / 8)
    assert raw_row.per_seed == {seed: pytest.approx(1 / 8) for seed in SEEDS}
    assert by_name[("condition", "raw_summary")].difference == pytest.approx(0.0)
    assert by_name[("contrast", "Δ_budget")].difference == pytest.approx(0.0)
    assert by_name[("contrast", "Δ_summary")].difference == pytest.approx(-1 / 8)
    assert {row.kind for row in comparison} == {"condition", "contrast"}
    # One rule alone compares nothing.
    assert checkpoint_rule_comparison(selected, samples=10) == ()


def test_the_cost_table_reads_systems_and_metrics(tmp_path: Path) -> None:
    study, contract = _study()
    root = tmp_path / contract.protocol
    measured: dict[str, dict[str, Any]] = {
        "raw": {"bytes": 3_078_144, "decision": (0.010, 0.012), "boundary": None},
        "raw_summary": {
            "bytes": 249_856,
            "decision": (0.004, 0.005),
            "boundary": (0.020, 0.030, 0.040),
        },
    }
    for condition, values in measured.items():
        for seed in SEEDS:
            run = root / condition / f"seed-{seed}"
            run.mkdir(parents=True)
            (run / "checkpoint.pt").write_bytes(b"")
            boundary = values["boundary"]
            systems = {
                "persistent_state_bytes": values["bytes"],
                "cache_bytes": values["bytes"],
                "decision_latency_seconds": values["decision"][0] + seed * 0.001,
                "decision_latency_p95_seconds": values["decision"][1] + seed * 0.001,
                "boundary_latency_seconds": None if boundary is None else boundary[0],
                "boundary_latency_p95_seconds": (
                    None if boundary is None else boundary[1]
                ),
                "boundary_latency_max_seconds": (
                    None if boundary is None else boundary[2]
                ),
                "peak_gpu_bytes": 1_000 * (seed + 1),
            }
            (run / "systems.json").write_text(json.dumps(systems))
            (run / "metrics.json").write_text(
                json.dumps(
                    {"runtime_seconds": 3600.0 * (seed + 1), "parameters": 1_234_567}
                )
            )
    reference = root / "feedforward" / "seed-0"
    reference.mkdir(parents=True)
    (reference / "checkpoint.pt").write_bytes(b"")
    (reference / "metrics.json").write_text(
        json.dumps({"runtime_seconds": 1800.0, "parameters": 500})
    )
    rows = cost_table(study, contract, tmp_path)
    assert [row.condition for row in rows] == ["raw", "raw_summary", "feedforward"]
    raw, summary, feedforward = rows
    assert raw.seeds == SEEDS and raw.persistent_state_bytes == 3_078_144
    assert raw.decision_latency_seconds == pytest.approx(0.011)
    assert raw.decision_latency_p95_seconds == pytest.approx(0.013)
    assert raw.boundary_latency_seconds is None
    assert raw.boundary_latency_max_seconds is None
    assert raw.training_hours == pytest.approx(2.0)
    assert raw.per_seed_training_hours == {0: 1.0, 1: 2.0, 2: 3.0}
    assert raw.parameters == 1_234_567 and raw.peak_gpu_bytes == 3_000
    assert summary.persistent_state_bytes == 249_856
    assert summary.boundary_latency_seconds == pytest.approx(0.020)
    assert summary.boundary_latency_p95_seconds == pytest.approx(0.030)
    assert summary.boundary_latency_max_seconds == pytest.approx(0.040)
    assert feedforward.persistent_state_bytes is None
    assert feedforward.decision_latency_seconds is None
    assert feedforward.training_hours == pytest.approx(0.5)
    assert feedforward.parameters == 500
    # Allocated bytes are an identity of the cell: seeds must agree.
    (root / "raw" / "seed-2" / "systems.json").write_text(
        json.dumps({"persistent_state_bytes": 1})
    )
    with pytest.raises(ResultValidationError, match="differs across seeds"):
        cost_table(study, contract, tmp_path)
    assert cost_table(study, contract, tmp_path / "empty") == ()


def test_the_secondary_table_averages_named_summaries_over_seeds() -> None:
    _, contract = _study()
    run = _run(contract, "raw", 0, "selected")
    records = [
        RunRecords(
            Path("raw/seed-0"),
            "raw",
            0,
            run,
            (),
            {"goal_0_steps_fixed_budget_mean": 120.0, "full_sequence_rate": 0.5},
            False,
        ),
        RunRecords(
            Path("raw/seed-1"),
            "raw",
            1,
            replace(run, training_seed=1),
            (),
            {"goal_0_steps_fixed_budget_mean": 80.0, "full_sequence_rate": None},
            False,
        ),
        RunRecords(Path("raw/seed-2"), "raw", 2, None, (), {}, False),
        RunRecords(
            Path("raw_summary/seed-0"),
            "raw_summary",
            0,
            replace(run, condition="raw_summary"),
            (),
            {"goal_0_steps_fixed_budget_mean": 200.0},
            True,  # partial: never averaged
        ),
    ]
    rows = secondary_table(
        records, ("goal_0_steps_fixed_budget_mean", "full_sequence_rate", "absent")
    )
    assert [(row.condition, row.key) for row in rows] == [
        ("raw", "goal_0_steps_fixed_budget_mean"),
        ("raw", "full_sequence_rate"),
    ]
    assert rows[0].mean == pytest.approx(100.0)
    assert rows[0].per_seed == {0: 120.0, 1: 80.0}
    assert rows[1].mean == pytest.approx(0.5) and rows[1].per_seed == {0: 0.5}
