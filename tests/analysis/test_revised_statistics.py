"""R4 acceptance for the revised estimators on synthetic complete records.

Door counts and the first-eight rate read the same events differently; the
per-seed task bootstrap is conditional on each trained seed; attempt curves
carry risk sets; cumulative curves count doors by charged call; first-door
times are right-censored; the plateau screen and the tier contrasts follow
EXPERIMENTS sections 3 and 7.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from reasoned_icrl.analysis.statistics import (
    SIGN_FLIP_MINIMUM_P,
    attempt_curves,
    cumulative_curves,
    disposition,
    estimates,
    first_event_times,
    paired_contrasts,
    plateau_screen,
    tier_contrasts,
)
from reasoned_icrl.experiments.benchmarks import TierContrast
from reasoned_icrl.experiments.contracts import ResultValidationError
from reasoned_icrl.experiments.records import BenchmarkEvent, BenchmarkRun, cell_values

PROTOCOL = "native-keydoor-fixed500-first8"
SEEDS = (42, 100, 2026)
TASKS = (1_000_000, 1_000_001)


def _attempt(
    condition: str,
    seed: int,
    task: int,
    index: int,
    *,
    start: int,
    steps: int,
    success: bool,
    complete: bool = True,
) -> BenchmarkEvent:
    end = start + steps - 1
    return BenchmarkEvent(
        protocol=PROTOCOL,
        benchmark="dark_key_to_door",
        condition=condition,
        training_seed=seed,
        checkpoint="policy_epoch_1000",
        split="development",
        history="retained",
        task_id=task,
        cluster_id=task,
        rollout_seed=0,
        kind="attempt",
        event_index=index,
        step=steps,
        numerator=int(success),
        denominator=1,
        native_return=2.0 if success else 0.0,
        start_step=start,
        end_step=end,
        checkpoint_rule="endpoint",
        complete=complete,
        key_step=start if success else None,
    )


def _task(condition: str, seed: int, task: int, doors: int) -> list[BenchmarkEvent]:
    """``doors`` quick successes (10 steps + reset), then time-outs to the budget,
    with a partial attempt at the end when the budget cuts one short."""
    rows: list[BenchmarkEvent] = []
    step = 1
    index = 1
    while step <= 500:
        steps = 10 if index <= doors else 50
        if step + steps - 1 > 500:
            rows.append(
                _attempt(
                    condition,
                    seed,
                    task,
                    index,
                    start=step,
                    steps=500 - step + 1,
                    success=False,
                    complete=False,
                )
            )
            break
        rows.append(
            _attempt(
                condition,
                seed,
                task,
                index,
                start=step,
                steps=steps,
                success=index <= doors,
            )
        )
        step += steps + 1  # the reset-only call
        index += 1
    return rows


def _panel(doors: dict[str, int]) -> list[BenchmarkEvent]:
    events: list[BenchmarkEvent] = []
    for condition, count in doors.items():
        for seed in SEEDS:
            for task in TASKS:
                events.extend(_task(condition, seed, task, count))
    return events


def test_counts_and_scored_band_rates_read_the_same_events_differently() -> None:
    events = _panel({"full_context": 12, "fixed_segment": 3})
    full = [e for e in events if e.condition == "full_context"]
    cells = cell_values(full, "doors_completed")
    assert cells[(42, TASKS[0], 0)] == 12.0
    assert cell_values(full, "door_success_first8")[(42, TASKS[0], 0)] == 1.0
    segment = [e for e in events if e.condition == "fixed_segment"]
    assert cell_values(segment, "door_success_first8")[(42, TASKS[0], 0)] == 3 / 8
    assert cell_values(segment, "doors_completed")[(42, TASKS[0], 0)] == 3.0
    rows = estimates(events, metric="doors_completed", samples=50)
    by = {row.condition: row for row in rows}
    assert by["full_context"].estimate == 12.0 and by["full_context"].metric == (
        "doors_completed"
    )
    assert by["fixed_segment"].estimate == 3.0
    assert set(by["full_context"].per_seed_lower) == set(SEEDS)
    assert all(
        by["full_context"].per_seed_lower[s]
        <= 12.0
        <= by["full_context"].per_seed_upper[s]
        for s in SEEDS
    )
    legacy = {row.condition: row for row in estimates(events, samples=10)}
    assert legacy["fixed_segment"].estimate < 1.0  # the mean rate over every event


def test_paired_contrasts_carry_per_seed_intervals_and_metric() -> None:
    events = _panel({"fixed_summary": 5, "fixed_segment": 3})
    (row,) = paired_contrasts(
        events,
        [("fixed_summary", "fixed_segment")],
        metric="doors_completed",
        samples=50,
    )
    assert row.estimate == 2.0 and row.metric == "doors_completed"
    assert row.per_seed == {42: 2.0, 100: 2.0, 2026: 2.0}
    assert row.positive_seeds == 3
    assert all(row.per_seed_lower[s] == 2.0 == row.per_seed_upper[s] for s in SEEDS)


def test_attempt_curves_report_risk_sets_and_exclude_partial_attempts() -> None:
    events = _panel({"full_context": 12})
    points = attempt_curves(events, samples=20)
    by_index = {p.event_index: p for p in points}
    assert by_index[1].estimate == 1.0 and by_index[1].risk_set == 6
    assert by_index[12].estimate == 1.0 and by_index[13].estimate == 0.0
    assert set(by_index[1].risk_set_per_seed) == set(SEEDS)
    # Twelve 10-step doors (11 calls each = 132) then 50-step time-outs: the
    # 20th attempt is the partial one and never enters the curve.
    partial = [e for e in events if not e.complete]
    assert partial and all(e.event_index == max(by_index) + 1 for e in partial)


def test_cumulative_curves_count_doors_by_charged_call() -> None:
    events = _panel({"full_context": 12, "fixed_segment": 3})
    points = cumulative_curves(events, grid=(10, 11, 22, 132, 500), samples=20)
    full = {p.step: p for p in points if p.condition == "full_context"}
    assert full[10].estimate == 1.0 and full[11].estimate == 1.0
    assert full[22].estimate == 2.0 and full[132].estimate == 12.0
    assert full[500].estimate == 12.0 and full[500].tasks == 2 and full[500].seeds == 3
    segment = {p.step: p for p in points if p.condition == "fixed_segment"}
    assert segment[500].estimate == 3.0
    with pytest.raises(ResultValidationError, match="positive steps"):
        cumulative_curves(events, grid=(0,), samples=1)
    episode = replace(events[0], kind="episode", complete=None)
    with pytest.raises(ResultValidationError, match="per-flip retention"):
        cumulative_curves([episode], grid=(1,), samples=1)


def test_first_event_times_are_censored_at_the_budget() -> None:
    events = _panel({"full_context": 12, "fixed_segment": 0})
    rows = first_event_times(
        events, grid=(5, 10, 250, 500), outer_length=500, samples=20
    )
    full = {r.step: r for r in rows if r.condition == "full_context"}
    assert full[5].fraction_reached == 0.0 and full[10].fraction_reached == 1.0
    assert full[10].median_step == 10 and full[10].censored_fraction == 0.0
    none = {r.step: r for r in rows if r.condition == "fixed_segment"}
    assert none[500].fraction_reached == 0.0
    assert none[500].censored_fraction == 1.0 and none[500].median_step is None
    with pytest.raises(ResultValidationError, match="within the outer task"):
        first_event_times(events, grid=(501,), outer_length=500, samples=1)


def test_the_plateau_screen_flags_improvement_and_variability() -> None:
    late = (850, 900, 950, 1000)
    early = (650, 700, 750, 800)
    rising = {s: {**{e: 5.0 for e in early}, **{e: 5.8 for e in late}} for s in SEEDS}
    verdict = plateau_screen("full_context", rising, late=late, early=early, delta=1.0)
    assert verdict.still_improving and not verdict.variable and verdict.complete
    assert verdict.across_seed_difference == pytest.approx(0.8)
    flat = {s: {**{e: 5.0 for e in early}, **{e: 5.2 for e in late}} for s in SEEDS}
    assert not plateau_screen(
        "c", flat, late=late, early=early, delta=1.0
    ).still_improving
    noisy = {
        s: {**{e: 5.0 for e in early}, 850: 4.0, 900: 6.0, 950: 4.0, 1000: 6.0}
        for s in SEEDS
    }
    assert plateau_screen("c", noisy, late=late, early=early, delta=1.0).variable
    partial = {42: rising[42], 100: rising[100]}
    verdict = plateau_screen("c", partial, late=late, early=early, delta=1.0)
    assert not verdict.complete and not verdict.still_improving
    unscored = {s: {650: 5.0} for s in SEEDS}
    verdict = plateau_screen("c", unscored, late=late, early=early, delta=1.0)
    assert verdict.across_seed_difference is None and verdict.late_range is None
    assert not verdict.complete and not verdict.variable
    with pytest.raises(ResultValidationError):
        plateau_screen("c", rising, late=late, early=early, delta=0.0)


def test_tier_contrasts_classify_every_declared_row() -> None:
    events = _panel({"full_dual_relational": 8, "full_context": 6, "fixed_summary": 4})
    contrasts = (
        TierContrast(
            "DAT full - ordinary full", "full_dual_relational", "full_context"
        ),
        TierContrast("summary - segment", "fixed_summary", "fixed_segment"),
    )
    companions = (
        TierContrast("full DAT - summary", "full_dual_relational", "fixed_summary"),
    )
    rows = tier_contrasts(
        events,
        contrasts,
        companions=companions,
        delta=1.0,
        metric="doors_completed",
        expected_seeds=3,
        samples=20,
    )
    assert [(r.name, r.role) for r in rows] == [
        ("DAT full - ordinary full", "primary"),
        ("full DAT - summary", "companion"),
    ]  # summary - segment is pending: fixed_segment has no records
    gain = rows[0]
    assert gain.contrast.estimate == 2.0 and gain.meets_effect
    assert gain.disposition == "consistent practical gain"
    assert rows[1].contrast.estimate == 4.0
    base = gain.contrast
    assert disposition(base, delta=1.0, expected_seeds=3, qualified=False) == (
        "baseline-unqualified"
    )
    assert (
        disposition(base, delta=1.0, expected_seeds=4, qualified=True) == "incomplete"
    )
    assert disposition(
        base, delta=1.0, expected_seeds=3, qualified=None, valid=False
    ) == ("engineering invalid")
    small = replace(base, estimate=0.5, lower=0.2, upper=0.8)
    assert (
        disposition(small, delta=1.0, expected_seeds=3, qualified=True)
        == "inconclusive"
    )
    negative = replace(base, estimate=-2.0, lower=-2.5, upper=-1.5)
    assert (
        disposition(negative, delta=1.0, expected_seeds=3, qualified=True) == "negative"
    )
    mixed = replace(base, per_seed={42: 2.0, 100: -0.5, 2026: 3.0})
    assert (
        disposition(mixed, delta=1.0, expected_seeds=3, qualified=True)
        == "inconclusive"
    )
    assert SIGN_FLIP_MINIMUM_P == 0.25


def test_the_fixture_is_a_valid_complete_record() -> None:
    from dataclasses import replace as dc_replace

    from reasoned_icrl.experiments.benchmarks import EvaluationSplit
    from reasoned_icrl.experiments.summary_memory.configs import (
        load_summary_memory_study,
    )

    study = load_summary_memory_study()
    contract = study.contract("dark_key_to_door")
    contract = dc_replace(
        contract,
        evaluation=dc_replace(
            contract.evaluation,
            splits={
                name: EvaluationSplit(split.source, 2, split.offset)
                for name, split in contract.evaluation.splits.items()
            },
        ),
    )
    from reasoned_icrl.experiments.records import validate_benchmark_results

    events = _panel({"full_context": 12})
    runs = [
        BenchmarkRun(
            PROTOCOL,
            "dark_key_to_door",
            "full_context",
            seed,
            "policy_epoch_1000",
            "development",
            "retained",
            "completed",
            checkpoint_rule="endpoint",
            retention="complete",
            metric="doors_completed",
        )
        for seed in SEEDS
    ]
    validate_benchmark_results([contract], runs, events)
