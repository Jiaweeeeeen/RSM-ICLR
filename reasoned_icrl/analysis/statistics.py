"""Estimates over evaluation events with paired hierarchical uncertainty.

The sampling units are training seeds and evaluation tasks. Every estimator
first averages the events of one (seed, task, rollout) cell, then bootstraps
seeds and tasks jointly, so thousands of queries never masquerade as thousands
of independent training replications. Paired contrasts difference the two
conditions inside each cell before resampling, which is only valid when both
conditions were evaluated on the identical roster; the estimators refuse
anything else rather than silently pooling.

A *panel* is one (protocol, split, history, checkpoint rule); the
development-selected panel and the fixed-final-checkpoint supplement are never
averaged together. The summary-memory study's estimators (plan §2, §5.3) sit
below the generic ones: the named plan contrasts with their practical effect
thresholds, the recovery fraction under its reporting rule, the attention x
regime interaction, the writes-before-decision table, performance stratified by
the retention diagnostics, the late-task interval measure and the comparison of
the two checkpoint rules. Every table shows seed-level effects: three training
seeds are three seeds, and resampling tasks does not create more of them.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np

from reasoned_icrl.experiments.benchmarks import TierContrast
from reasoned_icrl.experiments.contracts import ResultValidationError
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    Cell,
    cell_values as metric_cell_values,
)

SIGN_FLIP_MINIMUM_P = 0.25
"""With three paired training seeds an exact two-sided sign-flip test has a
minimum p of 0.25, so no contrast here can support a conventional p < 0.05
claim; seed effects are shown, not tested (EXPERIMENTS section 7)."""

Panel = tuple[str, str, str, str]
"""(protocol, split, history, checkpoint rule)."""

MINIMUM_DEPENDENCE = 0.05
"""The C3 margin and the effect of interest for every dual-attention contrast
and for ``Δ_window`` (plan §2)."""
DEFICIT_TRIGGER = 0.10
"""The effect of interest for ``Δ_budget`` (the deficit screen) and ``Δ_summary``."""


class PlanContrast(NamedTuple):
    """One named contrast of plan §2: ``left - right`` and its effect of interest."""

    name: str
    left: str
    right: str
    effect_of_interest: float


PLAN_CONTRASTS: tuple[PlanContrast, ...] = (
    PlanContrast("Δ_relation|full", "raw_dat", "raw", MINIMUM_DEPENDENCE),
    PlanContrast("grounding: raw - raw_gru", "raw", "raw_gru", 0.0),
    PlanContrast("Δ_budget", "raw", "raw_segment", DEFICIT_TRIGGER),
    PlanContrast("Δ_summary", "raw_summary", "raw_segment", DEFICIT_TRIGGER),
    PlanContrast("Δ_window", "raw_summary", "raw_window", MINIMUM_DEPENDENCE),
    PlanContrast(
        "Δ_relation|summary", "raw_dat_summary", "raw_summary", MINIMUM_DEPENDENCE
    ),
    PlanContrast(
        "Δ_relation|segment", "raw_dat_segment", "raw_segment", MINIMUM_DEPENDENCE
    ),
    PlanContrast(
        "Δ_write_branch",
        "raw_dat_summary",
        "raw_dat_summary_relational_write_off",
        MINIMUM_DEPENDENCE,
    ),
    PlanContrast(
        "Δ_capacity|summary",
        "raw_dat_summary",
        "raw_dual_content_summary",
        MINIMUM_DEPENDENCE,
    ),
    PlanContrast("grounding: raw_summary - raw_gru", "raw_summary", "raw_gru", 0.0),
)
"""The plan's §2 contrasts as (name, left, right, effect of interest)."""

HEADROOM_CONTRASTS: tuple[PlanContrast, ...] = (
    PlanContrast(
        "headroom: raw - raw_summary", "raw", "raw_summary", MINIMUM_DEPENDENCE
    ),
    PlanContrast(
        "headroom: raw_gru - raw_summary", "raw_gru", "raw_summary", MINIMUM_DEPENDENCE
    ),
)
"""The H2 headroom screen: the compression loss
an ordinary summary leaves against the best full-history reference,
``max(raw, raw_gru) - raw_summary`` on the development split. Below
``MINIMUM_DEPENDENCE`` there is no loss for a relational summary to recover, and
``Δ_relation|summary`` is reported on that environment as *not testable* — still
computed and shown, never claimed. Key-to-Door: 0.010 at seed 0 (a local RTX 3090)."""

ATTENTION_REGIME_INTERACTION = (
    ("raw_dat_summary", "raw_summary"),
    ("raw_dat_segment", "raw_segment"),
)
"""H0xH2: the dual-attention benefit under the summary regime minus the benefit
under the segment regime, ``(dat_summary - summary) - (dat_segment - segment)``."""


@dataclass(frozen=True, slots=True)
class Estimate:
    """One condition's primary metric with a seed/task bootstrap interval.

    ``per_seed_lower``/``per_seed_upper`` (R4) are the paired task-bootstrap
    intervals conditional on each trained seed: tasks resampled, the seed
    fixed. The joint interval resamples seeds and tasks together.
    """

    protocol: str
    condition: str
    split: str
    history: str
    estimate: float
    lower: float
    upper: float
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]
    checkpoint_rule: str = "selected"
    metric: str | None = None
    per_seed_lower: Mapping[int, float] = field(default_factory=dict)
    per_seed_upper: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PairedContrast:
    """``left`` minus ``right`` on an identical seed/task roster.

    ``per_seed_lower``/``per_seed_upper`` (R4) are the paired task-bootstrap
    intervals of each seed's own difference, conditional on that trained seed.
    """

    protocol: str
    left: str
    right: str
    split: str
    history: str
    estimate: float
    lower: float
    upper: float
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]
    checkpoint_rule: str = "selected"
    metric: str | None = None
    per_seed_lower: Mapping[int, float] = field(default_factory=dict)
    per_seed_upper: Mapping[int, float] = field(default_factory=dict)

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero."""
        return self.lower > 0.0 or self.upper < 0.0

    @property
    def positive_seeds(self) -> int:
        """How many paired seed differences are strictly positive."""
        return sum(value > 0.0 for value in self.per_seed.values())


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """The primary metric at one event index (one attempt or query position) or,
    when the curve is banded, over the inclusive band ``event_index .. last_index``.

    ``risk_set`` (R4) is the number of (seed, task, rollout) cells whose task
    reached that attempt and finished it; a complete record's late attempts
    are conditional on the tasks that got there, which the risk set shows."""

    protocol: str
    condition: str
    split: str
    history: str
    event_index: int
    estimate: float
    lower: float
    upper: float
    checkpoint_rule: str = "selected"
    last_index: int | None = None
    risk_set: int | None = None
    risk_set_per_seed: Mapping[int, int] = field(default_factory=dict)


def cell_means(
    events: Sequence[BenchmarkEvent], metric: str | None = None
) -> dict[Cell, float]:
    """The metric's value inside each (seed, task, rollout) cell.

    With no metric this is the legacy mean scored fraction over every event;
    a named metric follows its own membership and aggregation (R4), so a
    complete Key-to-Door record yields door counts under ``doors_completed``
    and the first-eight rate under ``door_success_first8``.
    """
    return metric_cell_values(events, metric)


def _matrix(cells: Mapping[Cell, float]) -> tuple[tuple[int, ...], np.ndarray]:
    """Rows are seeds, columns are (task, rollout) units; the roster must be exact."""
    seeds = tuple(sorted({seed for seed, _, _ in cells}))
    units = tuple(sorted({(task, rollout) for _, task, rollout in cells}))
    expected = {(seed, *unit) for seed in seeds for unit in units}
    if not seeds or not units or set(cells) != expected:
        raise ResultValidationError("Estimate lacks an exact seed/task roster.")
    matrix = np.asarray(
        [[cells[(seed, *unit)] for unit in units] for seed in seeds], dtype=np.float64
    )
    return seeds, matrix


def _bootstrap(
    matrix: np.ndarray, *, samples: int, rng: np.random.Generator
) -> np.ndarray:
    if samples == 0:
        return np.empty(0, dtype=np.float64)
    rows, columns = matrix.shape
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        end = min(start + 256, samples)
        seed_indices = rng.integers(0, rows, size=(end - start, rows))
        unit_indices = rng.integers(0, columns, size=(end - start, columns))
        sampled = matrix[seed_indices[:, :, None], unit_indices[:, None, :]]
        draws[start:end] = sampled.mean(axis=(1, 2))
    return draws


def _ragged_bootstrap(
    rows: Sequence[np.ndarray], *, samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Seed/unit bootstrap of a cell mean when seeds hold unequal unit counts.

    Used for strata, where not every seed x task cell has an event. Seeds are
    drawn with replacement; every drawn seed resamples its own units with
    replacement (independently for repeated draws of one seed); the statistic
    is the mean over every sampled cell, matching the point estimate.
    """
    if samples == 0:
        return np.empty(0, dtype=np.float64)
    count = len(rows)
    sizes = np.asarray([len(row) for row in rows], dtype=np.float64)
    means = np.empty((count, samples, count), dtype=np.float64)
    for index, values in enumerate(rows):
        picks = rng.integers(0, len(values), size=(samples, count, len(values)))
        means[index] = values[picks].mean(axis=2)
    chosen = rng.integers(0, count, size=(samples, count))
    draw_axis = np.arange(samples)[:, None]
    slot_axis = np.arange(count)[None, :]
    picked_means = means[chosen, draw_axis, slot_axis]
    picked_sizes = sizes[chosen]
    return (picked_means * picked_sizes).sum(axis=1) / picked_sizes.sum(axis=1)


def _interval(
    draws: np.ndarray, estimate: float, confidence: float
) -> tuple[float, float]:
    if len(draws) == 0:
        return estimate, estimate
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def seed_task_interval(
    matrix: np.ndarray,
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and joint seed/task bootstrap interval of a ``[seeds, tasks]`` matrix.

    The resampling of every tier report: seeds drawn with replacement, and for
    each drawn seed the tasks drawn with replacement, conditional on the
    trained seeds. Paired contrasts pass the per-task differences.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape:
        raise ResultValidationError(
            "A seed/task interval needs a non-empty 2-D matrix."
        )
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    lower, upper = _interval(
        _bootstrap(values, samples=samples, rng=rng), estimate, confidence
    )
    return estimate, lower, upper


def _panels(events: Sequence[BenchmarkEvent]) -> dict[Panel, list[BenchmarkEvent]]:
    panels: defaultdict[Panel, list[BenchmarkEvent]] = defaultdict(list)
    for event in events:
        panels[
            (event.protocol, event.split, event.history, event.checkpoint_rule)
        ].append(event)
    return dict(sorted(panels.items()))


def _exact_estimate(
    cells: Mapping[Cell, float],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, tuple[int, ...], dict[int, float]]:
    seeds, matrix = _matrix(cells)
    estimate = float(matrix.mean())
    lower, upper = _interval(
        _bootstrap(matrix, samples=samples, rng=rng), estimate, confidence
    )
    per_seed = {seed: float(matrix[i].mean()) for i, seed in enumerate(seeds)}
    return estimate, lower, upper, matrix.shape[1], seeds, per_seed


def _per_seed_intervals(
    cells: Mapping[Cell, float],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[dict[int, float], dict[int, float]]:
    """Task-bootstrap intervals conditional on each trained seed (R4): the
    seed is fixed and only its (task, rollout) units are resampled."""
    seeds, matrix = _matrix(cells)
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    for index, seed in enumerate(seeds):
        row = matrix[index]
        if samples == 0:
            lower[seed] = upper[seed] = float(row.mean())
            continue
        picks = rng.integers(0, row.shape[0], size=(samples, row.shape[0]))
        draws = row[picks].mean(axis=1)
        lower[seed], upper[seed] = _interval(draws, float(row.mean()), confidence)
    return lower, upper


def estimates(
    events: Sequence[BenchmarkEvent],
    *,
    metric: str | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[Estimate, ...]:
    """Primary-metric estimates per protocol/condition/split/history/rule panel.

    ``metric`` names the contract's primary metric (R4); ``None`` keeps the
    legacy mean scored fraction. The resampling seed is ``seed``; the
    construction is a joint seed/task bootstrap plus per-seed task bootstraps.
    """
    rng = np.random.default_rng(seed)
    # The per-seed intervals draw from their own stream so the joint draws,
    # and every interval recorded before R4, reproduce from the same seed.
    per_seed_rng = np.random.default_rng((seed, 1))
    output: list[Estimate] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for condition in sorted({event.condition for event in panel}):
            cells = cell_means([e for e in panel if e.condition == condition], metric)
            estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                cells, samples=samples, confidence=confidence, rng=rng
            )
            seed_lower, seed_upper = _per_seed_intervals(
                cells, samples=samples, confidence=confidence, rng=per_seed_rng
            )
            output.append(
                Estimate(
                    protocol,
                    condition,
                    split,
                    history,
                    estimate,
                    lower,
                    upper,
                    tasks,
                    len(seeds),
                    per_seed,
                    rule,
                    metric,
                    seed_lower,
                    seed_upper,
                )
            )
    return tuple(output)


def _paired_cells(
    panel: Sequence[BenchmarkEvent],
    left: str,
    right: str,
    *,
    protocol: str,
    metric: str | None = None,
) -> dict[Cell, float] | None:
    """``left - right`` per cell; None when a side is absent.

    The two conditions are paired on the training seeds they **share**, so a
    contrast can be read while a matrix is still filling (a condition fit for
    one seed against a reference fit for three pairs on that one seed, and the
    result reports one seed). Within a shared seed the task rosters must match
    exactly: that is a corrupt or partial evaluation, not an incomplete matrix,
    and it stays an error. Callers that require every declared seed — the
    deficit screen, the paper's tables — check the seed count themselves.
    """
    left_cells = cell_means([e for e in panel if e.condition == left], metric)
    right_cells = cell_means([e for e in panel if e.condition == right], metric)
    if not left_cells or not right_cells:
        return None
    shared = {seed for seed, _, _ in left_cells} & {seed for seed, _, _ in right_cells}
    if not shared:
        return None
    left_cells = {key: value for key, value in left_cells.items() if key[0] in shared}
    right_cells = {key: value for key, value in right_cells.items() if key[0] in shared}
    if set(left_cells) != set(right_cells):
        raise ResultValidationError(
            f"{left} and {right} do not share a paired roster on {protocol}."
        )
    return {key: left_cells[key] - right_cells[key] for key in left_cells}


def paired_contrasts(
    events: Sequence[BenchmarkEvent],
    pairs: Sequence[tuple[str, str]],
    *,
    metric: str | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[PairedContrast, ...]:
    """``left - right`` for each declared pair, on exactly paired rosters.

    A pair whose two conditions are not both present in a panel is skipped, and
    a pair is estimated on the training seeds its two conditions share, so a
    matrix that is still filling yields an honest partial read whose
    ``seeds``/``per_seed`` say how many seeds it rests on. A roster that differs
    *within* a shared seed is an error, never a silently unpaired estimate.
    ``metric`` names the contract's primary metric (R4).
    """
    rng = np.random.default_rng(seed)
    per_seed_rng = np.random.default_rng((seed, 1))
    output: list[PairedContrast] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for left, right in pairs:
            cells = _paired_cells(panel, left, right, protocol=protocol, metric=metric)
            if cells is None:
                continue
            estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                cells, samples=samples, confidence=confidence, rng=rng
            )
            seed_lower, seed_upper = _per_seed_intervals(
                cells, samples=samples, confidence=confidence, rng=per_seed_rng
            )
            output.append(
                PairedContrast(
                    protocol,
                    left,
                    right,
                    split,
                    history,
                    estimate,
                    lower,
                    upper,
                    tasks,
                    len(seeds),
                    per_seed,
                    rule,
                    metric,
                    seed_lower,
                    seed_upper,
                )
            )
    return tuple(output)


def _bands(indices: Sequence[int], bins: int | None) -> list[tuple[int, int]]:
    """Inclusive event-index bands: one per index, or ``bins`` equal bands."""
    low, high = min(indices), max(indices)
    if bins is None:
        return [(index, index) for index in sorted(set(indices))]
    if bins < 1:
        raise ResultValidationError("bins must be a positive integer.")
    edges = np.linspace(low - 1, high, bins + 1)
    bands = []
    for index in range(bins):
        start, end = int(edges[index]) + 1, int(edges[index + 1])
        if end >= start:
            bands.append((start, end))
    return bands


def curves(
    events: Sequence[BenchmarkEvent],
    *,
    bins: int | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[CurvePoint, ...]:
    """The primary metric by event index: attempt curves and query-position curves.

    ``bins`` groups the event indices of every panel into that many equal
    inclusive bands (a 207-query stream in eight bands), each band's cell mean
    taken over its events before the seed/task bootstrap. The area under an
    attempt curve, normalised by the number of attempts, is the adaptation AUC
    used by the Stage-1 study; it is the panel estimate of :func:`estimates`
    when every task contributes every attempt.
    """
    rng = np.random.default_rng(seed)
    output: list[CurvePoint] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for condition in sorted({event.condition for event in panel}):
            selected = [e for e in panel if e.condition == condition]
            for first, last in _bands([e.event_index for e in selected], bins):
                cells = cell_means(
                    [e for e in selected if first <= e.event_index <= last]
                )
                estimate, lower, upper, _, _, _ = _exact_estimate(
                    cells, samples=samples, confidence=confidence, rng=rng
                )
                output.append(
                    CurvePoint(
                        protocol,
                        condition,
                        split,
                        history,
                        first,
                        estimate,
                        lower,
                        upper,
                        rule,
                        None if bins is None else last,
                    )
                )
    return tuple(output)


def training_curve(run_directory: str | Path) -> list[dict[str, float]]:
    """Read the JSONL training telemetry mirror of one run, if present.

    A line that does not parse is skipped rather than raised on: a run whose
    filesystem fills mid-write leaves one truncated record and then appends
    after it once resumed (every MazeRunner run carries one such line). The mirror is
    telemetry, not a scored record,
    so a lost row changes a curve's resolution and nothing else; scored
    evidence lives in the evaluation panels, which are written whole or not
    at all."""
    path = Path(run_directory) / "training_metrics.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ----------------------------------------------------------------------
# Complete-record curves (R4): attempts with risk sets, calls, first events
# ----------------------------------------------------------------------


def _ragged_cells(
    cells: Mapping[Cell, float],
) -> tuple[tuple[int, ...], list[np.ndarray], dict[int, float], dict[int, int]]:
    by_seed: defaultdict[int, list[float]] = defaultdict(list)
    for (training_seed, _, _), value in cells.items():
        by_seed[training_seed].append(value)
    seeds = tuple(sorted(by_seed))
    rows = [np.asarray(by_seed[s], dtype=np.float64) for s in seeds]
    means = {s: float(np.mean(by_seed[s])) for s in seeds}
    counts = {s: len(by_seed[s]) for s in seeds}
    return seeds, rows, means, counts


def attempt_curves(
    events: Sequence[BenchmarkEvent],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[CurvePoint, ...]:
    """Success by attempt on complete records, with the risk set at each index.

    Only finished attempts enter the success rate at an index (a partial
    attempt is censored, not a failure); the risk set counts the cells that
    finished that attempt, so a late index whose few survivors succeed is not
    read as a population rate. Strata are ragged over seeds and tasks, so the
    interval is the ragged seed/unit bootstrap.
    """
    rng = np.random.default_rng(seed)
    output: list[CurvePoint] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for condition in sorted({event.condition for event in panel}):
            selected = [
                e
                for e in panel
                if e.condition == condition and e.kind == "attempt" and e.complete
            ]
            for index in sorted({e.event_index for e in selected}):
                cells = cell_means([e for e in selected if e.event_index == index])
                seeds, rows, _, counts = _ragged_cells(cells)
                estimate = float(np.mean(list(cells.values())))
                lower, upper = _interval(
                    _ragged_bootstrap(rows, samples=samples, rng=rng),
                    estimate,
                    confidence,
                )
                output.append(
                    CurvePoint(
                        protocol,
                        condition,
                        split,
                        history,
                        index,
                        estimate,
                        lower,
                        upper,
                        rule,
                        None,
                        len(cells),
                        {s: counts[s] for s in seeds},
                    )
                )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class CumulativePoint:
    """Cumulative successes per task by a charged-call step of the outer task."""

    protocol: str
    condition: str
    split: str
    history: str
    checkpoint_rule: str
    step: int
    estimate: float
    lower: float
    upper: float
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]


def cumulative_curves(
    events: Sequence[BenchmarkEvent],
    *,
    grid: Sequence[int],
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[CumulativePoint, ...]:
    """The main adaptation view (EXPERIMENTS section 4): cumulative completed
    successes versus charged evaluation calls, every attempt included.

    A success counts at the step its attempt ended (``end_step``), so resets
    are charged inside the clock; a partial attempt adds nothing. Every task
    contributes to every grid step (zero before its first success), so the
    roster stays exact and the seed/task bootstrap applies. Attempt records
    only: an episode record carries one terminal value, and the within-board
    curve needs per-flip retention (Concentration, R7).
    """
    if not grid or any(step < 1 for step in grid):
        raise ResultValidationError("The cumulative grid needs positive steps.")
    rng = np.random.default_rng(seed)
    output: list[CumulativePoint] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        if any(e.kind != "attempt" for e in panel):
            raise ResultValidationError(
                "Cumulative call curves need attempt records; episode records "
                "carry one terminal value (per-flip retention is R7)."
            )
        if any(e.end_step is None for e in panel):
            raise ResultValidationError(
                "The cumulative curve needs end_step on every event."
            )
        for condition in sorted({event.condition for event in panel}):
            selected = [e for e in panel if e.condition == condition]
            units = {(e.training_seed, e.task_id, e.rollout_seed) for e in selected}
            for step in grid:
                totals: dict[Cell, float] = dict.fromkeys(units, 0.0)
                for event in selected:
                    if int(event.end_step or 0) <= step:
                        key = (event.training_seed, event.task_id, event.rollout_seed)
                        totals[key] += event.rate
                estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                    totals, samples=samples, confidence=confidence, rng=rng
                )
                output.append(
                    CumulativePoint(
                        protocol,
                        condition,
                        split,
                        history,
                        rule,
                        int(step),
                        estimate,
                        lower,
                        upper,
                        tasks,
                        len(seeds),
                        per_seed,
                    )
                )
    return tuple(output)


def concentration_flip_curves(
    events: Sequence[BenchmarkEvent],
    *,
    grid: Sequence[int] = tuple(range(105)),
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[CumulativePoint, ...]:
    """Pair fraction by flip; completed boards carry forward without new calls.

    A terminal-only legacy record cannot reconstruct this curve. Every board
    stays in the denominator and rollouts/tasks/seeds retain equal weighting.
    """
    if not grid or any(type(step) is not int or not 0 <= step <= 104 for step in grid):
        raise ResultValidationError("Easy flip grid must lie in [0, 104].")
    rng = np.random.default_rng(seed)
    output: list[CumulativePoint] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        if any(e.benchmark != "concentration" or e.flips is None for e in panel):
            raise ResultValidationError(
                "Pair-fraction curves require complete flip traces."
            )
        for condition in sorted({e.condition for e in panel}):
            selected = [e for e in panel if e.condition == condition]
            for step in grid:
                values = {
                    (e.training_seed, e.task_id, e.rollout_seed): sum(
                        f.matched for f in e.flips or () if f.index <= step
                    )
                    / e.denominator
                    for e in selected
                }
                estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                    values, samples=samples, confidence=confidence, rng=rng
                )
                output.append(
                    CumulativePoint(
                        protocol,
                        condition,
                        split,
                        history,
                        rule,
                        step,
                        estimate,
                        lower,
                        upper,
                        tasks,
                        len(seeds),
                        per_seed,
                    )
                )
    return tuple(output)


def concentration_retrieval_rows(
    events: Sequence[BenchmarkEvent],
) -> list[dict[str, object]]:
    """Per-seed opportunity-weighted retrieval by public binding age.

    Opportunities are on-policy and can differ across cells. A binding's age
    is the pre-action observation index minus its most recent partner reveal.
    Zero opportunities produce a missing rate, never a zero success score.
    """
    rows: list[dict[str, object]] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for condition, training_seed in sorted(
            {(e.condition, e.training_seed) for e in panel}
        ):
            selected = [
                e
                for e in panel
                if (e.condition, e.training_seed) == (condition, training_seed)
            ]
            if any(e.flips is None for e in selected):
                raise ResultValidationError(
                    "Retrieval strata require complete flip traces."
                )
            flips = [f for e in selected for f in e.flips or ()]
            for label, low, high in (
                ("all", 0, 104),
                ("0-31", 0, 31),
                ("32-39", 32, 39),
                ("40-63", 40, 63),
                ("64-103", 64, 103),
            ):
                opportunities = [
                    f
                    for f in flips
                    if f.partner_last_reveal is not None
                    and low <= f.index - 1 - f.partner_last_reveal <= high
                ]
                hits = sum(f.hit for f in opportunities)
                rows.append(
                    dict(
                        protocol=protocol,
                        split=split,
                        history=history,
                        checkpoint_rule=rule,
                        condition=condition,
                        seed=training_seed,
                        age_flips=label,
                        opportunities=len(opportunities),
                        hits=hits,
                        retrieval_rate=hits / len(opportunities)
                        if opportunities
                        else None,
                        native_return_per_board=float(
                            np.mean([e.native_return for e in selected])
                        ),
                        board_completion_fraction=float(
                            np.mean([e.numerator == e.denominator for e in selected])
                        ),
                        boards=len(selected),
                        flips=len(flips),
                        matched_pairs=sum(f.matched for f in flips),
                        matching_efficiency=2
                        * sum(f.matched for f in flips)
                        / len(flips),
                        invalid_flips=sum(f.invalid for f in flips),
                        redundant_flips=sum(f.redundant for f in flips),
                    )
                )
    return rows


@dataclass(frozen=True, slots=True)
class FirstEventRow:
    """The fraction of tasks whose first success ended by ``step``, with the
    right-censoring at the outer budget made explicit."""

    protocol: str
    condition: str
    split: str
    history: str
    checkpoint_rule: str
    step: int
    fraction_reached: float
    lower: float
    upper: float
    censored_fraction: float
    median_step: int | None
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]


def first_event_times(
    events: Sequence[BenchmarkEvent],
    *,
    grid: Sequence[int],
    outer_length: int,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[FirstEventRow, ...]:
    """Time to the first success with right-censoring (EXPERIMENTS section 4).

    Every task is observed for exactly ``outer_length`` charged calls, so the
    only censoring is administrative at the budget: the empirical fraction
    reached by each grid step is the Kaplan-Meier estimate on this design.
    ``median_step`` is the first grid step at which half the tasks have
    succeeded, or ``None`` when fewer than half ever do (the median is then
    censored, not invented). Unfinished attempts never count as successes.
    """
    if not grid or any(not 1 <= step <= outer_length for step in grid):
        raise ResultValidationError("The grid must lie within the outer task.")
    rng = np.random.default_rng(seed)
    output: list[FirstEventRow] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        if any(e.end_step is None for e in panel):
            raise ResultValidationError("First-event times need end_step on events.")
        for condition in sorted({event.condition for event in panel}):
            selected = [e for e in panel if e.condition == condition]
            first: dict[Cell, int | None] = {}
            for event in selected:
                key = (event.training_seed, event.task_id, event.rollout_seed)
                first.setdefault(key, None)
                if event.numerator >= 1 and (event.complete is None or event.complete):
                    step = int(event.end_step or 0)
                    current = first[key]
                    first[key] = step if current is None else min(current, step)
            censored = float(np.mean([value is None for value in first.values()]))
            median: int | None = None
            for step in grid:
                reached = {
                    key: float(value is not None and value <= step)
                    for key, value in first.items()
                }
                estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                    reached, samples=samples, confidence=confidence, rng=rng
                )
                if median is None and estimate >= 0.5:
                    median = int(step)
                output.append(
                    FirstEventRow(
                        protocol,
                        condition,
                        split,
                        history,
                        rule,
                        int(step),
                        estimate,
                        lower,
                        upper,
                        censored,
                        None,
                        tasks,
                        len(seeds),
                        per_seed,
                    )
                )
            # The median is a property of the whole curve; stamp it on every row.
            output[-len(grid) :] = [
                replace_median(row, median) for row in output[-len(grid) :]
            ]
    return tuple(output)


def replace_median(row: FirstEventRow, median: int | None) -> FirstEventRow:
    return FirstEventRow(
        row.protocol,
        row.condition,
        row.split,
        row.history,
        row.checkpoint_rule,
        row.step,
        row.fraction_reached,
        row.lower,
        row.upper,
        row.censored_fraction,
        median,
        row.tasks,
        row.seeds,
        row.per_seed,
    )


# ----------------------------------------------------------------------
# The development plateau screen and the tier contrasts (R4)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlateauVerdict:
    """EXPERIMENTS section 3: the operational plateau screen of one cell.

    ``still_improving``: the across-seed late-window mean exceeds the early
    window by at least delta/2 with at least two of three seed differences
    positive. ``variable``: the range of the across-seed means over the late
    checkpoints exceeds delta (a descriptive flag). Neither is a convergence
    test; a flat low curve is unqualified, not converged.
    """

    condition: str
    late: tuple[int, ...]
    early: tuple[int, ...]
    delta: float
    per_seed_late: Mapping[int, float]
    per_seed_early: Mapping[int, float]
    per_seed_difference: Mapping[int, float]
    across_seed_difference: float | None
    late_range: float | None
    per_seed_late_range: Mapping[int, float]
    still_improving: bool
    variable: bool
    complete: bool


def plateau_screen(
    condition: str,
    series: Mapping[int, Mapping[int, float]],
    *,
    late: Sequence[int],
    early: Sequence[int],
    delta: float,
    expected_seeds: int = 3,
) -> PlateauVerdict:
    """Apply the plateau screen to one condition's per-seed development series
    (``seed -> {epoch: primary}``). A seed missing a checkpoint of either
    window leaves the verdict incomplete and never improving."""
    if delta <= 0 or not late or not early:
        raise ResultValidationError("The plateau screen needs windows and delta > 0.")
    per_late: dict[int, float] = {}
    per_early: dict[int, float] = {}
    per_range: dict[int, float] = {}
    for seed_id, points in series.items():
        if all(epoch in points for epoch in (*late, *early)):
            late_values = [points[epoch] for epoch in late]
            per_late[seed_id] = float(np.mean(late_values))
            per_early[seed_id] = float(np.mean([points[epoch] for epoch in early]))
            per_range[seed_id] = float(max(late_values) - min(late_values))
    complete = len(per_late) >= expected_seeds
    difference = {s: per_late[s] - per_early[s] for s in per_late}
    # No seed with both windows scored: the verdict is incomplete and its
    # across-seed figures are absent (None), never NaN.
    across = float(np.mean(list(difference.values()))) if difference else None
    positives = sum(value > 0.0 for value in difference.values())
    late_means = (
        [float(np.mean([series[s][epoch] for s in per_late])) for epoch in late]
        if per_late
        else []
    )
    late_range = float(max(late_means) - min(late_means)) if late_means else None
    return PlateauVerdict(
        condition=condition,
        late=tuple(int(e) for e in late),
        early=tuple(int(e) for e in early),
        delta=float(delta),
        per_seed_late=per_late,
        per_seed_early=per_early,
        per_seed_difference=difference,
        across_seed_difference=across,
        late_range=late_range,
        per_seed_late_range=per_range,
        still_improving=bool(
            complete
            and across is not None
            and across >= delta / 2
            and positives >= min(2, expected_seeds)
        ),
        variable=bool(complete and late_range is not None and late_range > delta),
        complete=complete,
    )


Disposition = Literal[
    "engineering invalid",
    "incomplete",
    "baseline-unqualified",
    "negative",
    "inconclusive",
    "consistent practical gain",
]


@dataclass(frozen=True, slots=True)
class TierContrastEstimate:
    """One predeclared tier contrast with its practical threshold, seed
    differences and the EXPERIMENTS section 7 disposition."""

    name: str
    role: Literal["primary", "companion"]
    effect_of_interest: float
    contrast: PairedContrast
    expected_seeds: int
    disposition: Disposition

    @property
    def meets_effect(self) -> bool:
        return self.contrast.estimate >= self.effect_of_interest


def disposition(
    contrast: PairedContrast,
    *,
    delta: float,
    expected_seeds: int,
    qualified: bool | None,
    valid: bool = True,
) -> Disposition:
    """Classify one contrast (EXPERIMENTS section 7).

    A consistent practical gain needs every expected seed, the point estimate
    at or above delta, every paired seed difference positive and the joint
    interval above zero; negative needs the joint interval below zero; an
    unqualified reference or a missing seed is named as such; anything else
    is inconclusive, with its actual effect reported rather than relabelled.
    """
    if not valid:
        return "engineering invalid"
    if contrast.seeds < expected_seeds:
        return "incomplete"
    if qualified is False:
        return "baseline-unqualified"
    if (
        contrast.estimate >= delta
        and contrast.positive_seeds == expected_seeds
        and contrast.lower > 0.0
    ):
        return "consistent practical gain"
    if contrast.upper < 0.0:
        return "negative"
    return "inconclusive"


def tier_contrasts(
    events: Sequence[BenchmarkEvent],
    contrasts: Sequence[TierContrast],
    *,
    delta: float,
    metric: str | None,
    expected_seeds: int,
    companions: Sequence[TierContrast] = (),
    qualified: bool | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[TierContrastEstimate, ...]:
    """The tier's predeclared family: primary contrasts then companions, one
    row per panel, each with delta, seed effects and its disposition. A
    contrast whose cells are absent from a panel yields no row (it is
    pending), never a fabricated zero.
    """
    declared = [(row, "primary") for row in contrasts] + [
        (row, "companion") for row in companions
    ]
    found = paired_contrasts(
        events,
        [(row.left, row.right) for row, _ in declared],
        metric=metric,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    output: list[TierContrastEstimate] = []
    for row, role in declared:
        for contrast in found:
            if (contrast.left, contrast.right) == (row.left, row.right):
                output.append(
                    TierContrastEstimate(
                        row.name,
                        cast(Literal["primary", "companion"], role),
                        float(delta),
                        contrast,
                        expected_seeds,
                        disposition(
                            contrast,
                            delta=delta,
                            expected_seeds=expected_seeds,
                            qualified=qualified,
                        ),
                    )
                )
    return tuple(output)


# ----------------------------------------------------------------------
# The plan's contrasts (Result 1 and Result 2)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContrastEstimate:
    """A named plan contrast beside its practical effect threshold."""

    name: str
    effect_of_interest: float
    contrast: PairedContrast

    @property
    def meets_effect(self) -> bool:
        """Whether the point estimate reaches the effect of interest."""
        return self.contrast.estimate >= self.effect_of_interest


def plan_contrasts(
    events: Sequence[BenchmarkEvent],
    *,
    contrasts: Sequence[PlanContrast] = PLAN_CONTRASTS,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[ContrastEstimate, ...]:
    """Every named contrast whose two cells share a panel, with seed effects.

    One row per contrast and panel, in the order of ``contrasts``; a contrast
    with a missing cell is absent, never a fabricated zero.
    """
    found = paired_contrasts(
        events,
        [(row.left, row.right) for row in contrasts],
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    output: list[ContrastEstimate] = []
    for row in contrasts:
        for contrast in found:
            if (contrast.left, contrast.right) == (row.left, row.right):
                output.append(
                    ContrastEstimate(row.name, row.effect_of_interest, contrast)
                )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class RecoveryFraction:
    """``Δ_summary / Δ_budget`` under the plan's reporting rule (plan §2).

    The fraction is reported only when the development deficit screen passed
    and the paired interval of ``Δ_budget`` on this panel excludes zero; it is
    never clipped (the full-prefix cell is a reference, not an upper bound).
    Otherwise it is ``None`` with the reason, and both differences stand.
    """

    protocol: str
    split: str
    history: str
    checkpoint_rule: str
    budget: PairedContrast
    summary: PairedContrast
    fraction: float | None
    interpretable: bool
    reason: str
    per_seed: Mapping[int, float]


def recovery_fraction(
    events: Sequence[BenchmarkEvent],
    *,
    deficit_present: bool | None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[RecoveryFraction, ...]:
    """The recovery fraction on every panel holding ``raw``, ``raw_segment``,
    ``raw_summary``, given the development deficit screen's verdict (``None``
    when the screen has not been run: not interpretable, both differences shown).
    """
    rows = plan_contrasts(
        events,
        contrasts=[
            row for row in PLAN_CONTRASTS if row.name in ("Δ_budget", "Δ_summary")
        ],
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    by_panel: dict[Panel, dict[str, PairedContrast]] = defaultdict(dict)
    for row in rows:
        contrast = row.contrast
        key = (
            contrast.protocol,
            contrast.split,
            contrast.history,
            contrast.checkpoint_rule,
        )
        by_panel[key][row.name] = contrast
    output: list[RecoveryFraction] = []
    for (protocol, split, history, rule), found in sorted(by_panel.items()):
        if "Δ_budget" not in found or "Δ_summary" not in found:
            continue
        budget, summary = found["Δ_budget"], found["Δ_summary"]
        if deficit_present is None:
            reason = (
                "not interpretable: the development deficit screen has not been run"
            )
        elif not deficit_present:
            reason = "not interpretable: the development deficit screen did not pass"
        elif not budget.excludes_zero:
            reason = "not interpretable: the paired interval of Δ_budget includes zero"
        elif budget.estimate == 0.0 or not math.isfinite(budget.estimate):
            reason = "not interpretable: Δ_budget is zero"
        else:
            reason = "reported: deficit screen passed and Δ_budget excludes zero"
        interpretable = reason.startswith("reported")
        fraction = summary.estimate / budget.estimate if interpretable else None
        per_seed = (
            {
                s: summary.per_seed[s] / budget.per_seed[s]
                for s in budget.per_seed
                if s in summary.per_seed and budget.per_seed[s] != 0.0
            }
            if interpretable
            else {}
        )
        output.append(
            RecoveryFraction(
                protocol,
                split,
                history,
                rule,
                budget,
                summary,
                fraction,
                interpretable,
                reason,
                per_seed,
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class InteractionEstimate:
    """``(first.left - first.right) - (second.left - second.right)`` per cell."""

    protocol: str
    split: str
    history: str
    checkpoint_rule: str
    first: tuple[str, str]
    second: tuple[str, str]
    estimate: float
    lower: float
    upper: float
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]


def interaction(
    events: Sequence[BenchmarkEvent],
    first: tuple[str, str] = ATTENTION_REGIME_INTERACTION[0],
    second: tuple[str, str] = ATTENTION_REGIME_INTERACTION[1],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[InteractionEstimate, ...]:
    """The double difference of two paired contrasts on one roster (H0xH2).

    Default: the dual-attention benefit under the summary regime minus the
    benefit under the segment regime. All four cells must share the panel's
    seed/task roster; a panel missing a cell yields no row.
    """
    rng = np.random.default_rng(seed)
    output: list[InteractionEstimate] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        one = _paired_cells(panel, first[0], first[1], protocol=protocol)
        two = _paired_cells(panel, second[0], second[1], protocol=protocol)
        if one is None or two is None:
            continue
        if set(one) != set(two):
            raise ResultValidationError(
                f"The interaction's four cells do not share a roster on {protocol}."
            )
        cells = {key: one[key] - two[key] for key in one}
        estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
            cells, samples=samples, confidence=confidence, rng=rng
        )
        output.append(
            InteractionEstimate(
                protocol,
                split,
                history,
                rule,
                first,
                second,
                estimate,
                lower,
                upper,
                tasks,
                len(seeds),
                per_seed,
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class HistoryContrast:
    """One condition under ``left_history`` minus the same condition under
    ``right_history``: the C3 dependence (``retained - attempt-cleared``) and the
    summary dependence (``retained - summary-cleared``), paired per cell."""

    protocol: str
    condition: str
    split: str
    checkpoint_rule: str
    left_history: str
    right_history: str
    estimate: float
    lower: float
    upper: float
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]


def history_contrasts(
    events: Sequence[BenchmarkEvent],
    left: str,
    right: str,
    *,
    metric: str | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[HistoryContrast, ...]:
    """``left`` minus ``right`` history for every condition evaluated under both.

    Both evaluations must cover one seed/task roster (they do: same checkpoint,
    same tasks, only the cache intervention differs); a condition missing one
    history yields no row.
    """
    rng = np.random.default_rng(seed)
    panels = _panels(events)
    output: list[HistoryContrast] = []
    keys = sorted({(protocol, split, rule) for protocol, split, _, rule in panels})
    for protocol, split, rule in keys:
        one = panels.get((protocol, split, left, rule))
        two = panels.get((protocol, split, right, rule))
        if one is None or two is None:
            continue
        for condition in sorted({e.condition for e in one}):
            left_cells = cell_means(
                [e for e in one if e.condition == condition], metric
            )
            right_cells = cell_means(
                [e for e in two if e.condition == condition], metric
            )
            if not right_cells:
                continue
            if set(left_cells) != set(right_cells):
                raise ResultValidationError(
                    f"{condition} was not evaluated on one roster under {left} and "
                    f"{right} on {protocol}."
                )
            cells = {key: left_cells[key] - right_cells[key] for key in left_cells}
            estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                cells, samples=samples, confidence=confidence, rng=rng
            )
            output.append(
                HistoryContrast(
                    protocol,
                    condition,
                    split,
                    rule,
                    left,
                    right,
                    estimate,
                    lower,
                    upper,
                    tasks,
                    len(seeds),
                    per_seed,
                )
            )
    return tuple(output)


# ----------------------------------------------------------------------
# Retention diagnostics (Result 3)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WritesRow:
    """Writes before the scored decisions of one event-index band."""

    protocol: str
    condition: str
    split: str
    history: str
    checkpoint_rule: str
    first_index: int
    last_index: int
    events: int
    mean_writes: float
    fraction_after_write: float
    per_seed_fraction_after_write: Mapping[int, float]


def writes_table(
    events: Sequence[BenchmarkEvent], *, bins: int | None = None
) -> tuple[WritesRow, ...]:
    """Writes before each scored decision, per condition and event-index band.

    ``fraction_after_write`` is the share of events decided after at least one
    boundary — with ``bins=1`` on Key-to-Door, the fraction of the first eight
    attempts that start after a write. Conditions whose events carry no write
    counter (full-prefix carriers) contribute no rows.
    """
    output: list[WritesRow] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for condition in sorted({event.condition for event in panel}):
            selected = [
                e
                for e in panel
                if e.condition == condition and e.writes_before_decision is not None
            ]
            if not selected:
                continue
            for first, last in _bands([e.event_index for e in selected], bins):
                band = [e for e in selected if first <= e.event_index <= last]
                writes = [int(e.writes_before_decision or 0) for e in band]
                per_seed: defaultdict[int, list[float]] = defaultdict(list)
                for event in band:
                    per_seed[event.training_seed].append(
                        float((event.writes_before_decision or 0) >= 1)
                    )
                output.append(
                    WritesRow(
                        protocol,
                        condition,
                        split,
                        history,
                        rule,
                        first,
                        last,
                        len(band),
                        float(np.mean(writes)),
                        float(np.mean([w >= 1 for w in writes])),
                        {s: float(np.mean(v)) for s, v in sorted(per_seed.items())},
                    )
                )
    return tuple(output)


StratumField = Literal[
    "writes_before_decision",
    "evidence_age_writes",
    "evidence_age_writes_recent",
    "evidence_in_current_segment",
    "count_before_current_segment",
    "count_in_current_segment",
    "true_count",
]
STRATUM_FIELDS: tuple[StratumField, ...] = (
    "writes_before_decision",
    "evidence_age_writes",
    "evidence_age_writes_recent",
    "evidence_in_current_segment",
    "count_before_current_segment",
    "count_in_current_segment",
    "true_count",
)
"""Event fields the primary metric can be stratified by (spec §9, plan §5.3)."""


@dataclass(frozen=True, slots=True)
class StratumEstimate:
    """The primary metric over the events whose ``field`` falls in one stratum.

    Strata are ragged over seeds and tasks, so the interval is the ragged
    seed/unit bootstrap; ``events`` counts the scored events, ``tasks`` the
    (task, rollout) units and ``seeds`` the seeds that reach the stratum.
    """

    protocol: str
    condition: str
    split: str
    history: str
    checkpoint_rule: str
    field: str
    stratum: str
    low: float | None
    high: float | None
    estimate: float
    lower: float
    upper: float
    events: int
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]


def _stratum_of(
    value: int | bool, bins: Sequence[tuple[int, int]] | None
) -> tuple[str, float | None, float | None] | None:
    if isinstance(value, bool):
        return str(value), None, None
    if bins is None:
        return str(value), float(value), float(value)
    for low, high in bins:
        if low <= value <= high:
            label = str(low) if low == high else f"{low}-{high}"
            return label, float(low), float(high)
    return None


def stratified_estimates(
    events: Sequence[BenchmarkEvent],
    field: StratumField,
    *,
    bins: Sequence[tuple[int, int]] | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[StratumEstimate, ...]:
    """The primary metric by the value of one retention field, per condition.

    Events whose field is ``None`` (carriers without the diagnostic, or an
    attempt with no earlier evidence) are left out; a stratum is reported when
    at least one event reaches it. ``bins`` groups integer values into
    inclusive ranges; a value outside every bin is left out.
    """
    if field not in STRATUM_FIELDS:
        raise ResultValidationError(f"Unknown stratum field: {field!r}.")
    rng = np.random.default_rng(seed)
    output: list[StratumEstimate] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for condition in sorted({event.condition for event in panel}):
            grouped: defaultdict[
                tuple[str, float | None, float | None], list[BenchmarkEvent]
            ] = defaultdict(list)
            for event in panel:
                if event.condition != condition:
                    continue
                value = getattr(event, field)
                if value is None:
                    continue
                stratum = _stratum_of(value, bins)
                if stratum is not None:
                    grouped[stratum].append(event)
            for (label, low, high), selected in sorted(
                grouped.items(),
                key=lambda item: (item[0][1] is None, item[0][1], item[0][0]),
            ):
                cells = cell_means(selected)
                by_seed: defaultdict[int, list[float]] = defaultdict(list)
                for (training_seed, _, _), value in cells.items():
                    by_seed[training_seed].append(value)
                seeds = tuple(sorted(by_seed))
                rows = [np.asarray(by_seed[s], dtype=np.float64) for s in seeds]
                estimate = float(np.mean(list(cells.values())))
                lower, upper = _interval(
                    _ragged_bootstrap(rows, samples=samples, rng=rng),
                    estimate,
                    confidence,
                )
                output.append(
                    StratumEstimate(
                        protocol,
                        condition,
                        split,
                        history,
                        rule,
                        field,
                        label,
                        low,
                        high,
                        estimate,
                        lower,
                        upper,
                        len(selected),
                        len(cells),
                        len(seeds),
                        {s: float(np.mean(by_seed[s])) for s in seeds},
                    )
                )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class IntervalRow:
    """Successes per task inside one fixed step interval of the outer task."""

    protocol: str
    condition: str
    split: str
    history: str
    checkpoint_rule: str
    start_step: int
    end_step: int
    successes_per_task: float
    lower: float
    upper: float
    events_per_task: float
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]


def interval_successes(
    events: Sequence[BenchmarkEvent],
    *,
    interval: int,
    outer_length: int | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[IntervalRow, ...]:
    """The late-task measure: successes per task in each ``interval``-step window.

    An event counts in the window holding its ``end_step`` (a door opened, a
    query answered), weighted by its scored fraction. Every task contributes
    to every window (zero when nothing ended there), so the roster stays exact
    and the seed/task bootstrap applies. Windows run to ``outer_length`` when
    given, else to the last ``end_step`` seen.
    """
    if interval < 1:
        raise ResultValidationError("interval must be a positive number of steps.")
    rng = np.random.default_rng(seed)
    output: list[IntervalRow] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        if any(e.start_step is None or e.end_step is None for e in panel):
            raise ResultValidationError(
                "The interval measure needs start_step/end_step on every event."
            )
        last = outer_length or max(int(e.end_step or 0) for e in panel)
        windows = [
            (start, min(start + interval - 1, last))
            for start in range(1, last + 1, interval)
        ]
        for condition in sorted({event.condition for event in panel}):
            selected = [e for e in panel if e.condition == condition]
            units = {(e.training_seed, e.task_id, e.rollout_seed) for e in selected}
            for start, end in windows:
                successes: dict[Cell, float] = dict.fromkeys(units, 0.0)
                counts: dict[Cell, float] = dict.fromkeys(units, 0.0)
                for event in selected:
                    if start <= int(event.end_step or 0) <= end:
                        key = (event.training_seed, event.task_id, event.rollout_seed)
                        successes[key] += event.rate
                        counts[key] += 1.0
                estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                    successes, samples=samples, confidence=confidence, rng=rng
                )
                output.append(
                    IntervalRow(
                        protocol,
                        condition,
                        split,
                        history,
                        rule,
                        start,
                        end,
                        estimate,
                        lower,
                        upper,
                        float(np.mean(list(counts.values()))),
                        tasks,
                        len(seeds),
                        per_seed,
                    )
                )
    return tuple(output)


# ----------------------------------------------------------------------
# The fixed-final-checkpoint supplement
# ----------------------------------------------------------------------


_Reader = Callable[[Sequence[BenchmarkEvent]], "dict[Cell, float] | None"]


@dataclass(frozen=True, slots=True)
class RetrievalEstimate:
    """One condition's Concentration retrieval rate — hits over opportunities,
    pooled over its boards — with a seed/task bootstrap interval, beside the
    same rate on the opportunities whose partner evidence had been evicted
    from the current segment (None off the summary carriers)."""

    protocol: str
    condition: str
    split: str
    history: str
    checkpoint_rule: str
    rate: float
    lower: float
    upper: float
    opportunities: int
    per_seed: dict[int, float]
    evicted_rate: float | None
    evicted_lower: float | None
    evicted_upper: float | None
    evicted_opportunities: int | None
    seeds: int
    tasks: int


def _ratio_bootstrap(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Seed/unit bootstrap of a ratio of sums over the seeds x units matrices."""
    if samples == 0:
        return np.empty(0, dtype=np.float64)
    rows, columns = numerators.shape
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        end = min(start + 256, samples)
        seed_indices = rng.integers(0, rows, size=(end - start, rows))
        unit_indices = rng.integers(0, columns, size=(end - start, columns))
        top = numerators[seed_indices[:, :, None], unit_indices[:, None, :]]
        bottom = denominators[seed_indices[:, :, None], unit_indices[:, None, :]]
        total = bottom.sum(axis=(1, 2))
        draws[start:end] = np.where(
            total > 0, top.sum(axis=(1, 2)) / np.maximum(total, 1), np.nan
        )
    return draws[np.isfinite(draws)]


def retrieval_rates(
    events: Sequence[BenchmarkEvent],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[RetrievalEstimate, ...]:
    """Concentration retrieval rates per panel and condition (plan §4, decision 14).

    The rate is the ratio of summed hits to summed opportunities over every
    seed x board cell (a per-board rate is undefined on boards without an
    opportunity); the bootstrap resamples seeds and boards and recomputes the
    ratio. Events without the ledger fields are skipped.
    """
    rng = np.random.default_rng(seed)
    output: list[RetrievalEstimate] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        for condition in sorted({event.condition for event in panel}):
            rows = [
                e
                for e in panel
                if e.condition == condition
                and e.retrieval_opportunities is not None
                and e.retrieval_successes is not None
            ]
            if not rows:
                continue
            hits = {
                (e.training_seed, e.task_id, e.rollout_seed): float(
                    cast(int, e.retrieval_successes)
                )
                for e in rows
            }
            opportunities = {
                (e.training_seed, e.task_id, e.rollout_seed): float(
                    cast(int, e.retrieval_opportunities)
                )
                for e in rows
            }
            seeds, top = _matrix(hits)
            _, bottom = _matrix(opportunities)
            total = float(bottom.sum())
            rate = float(top.sum() / total) if total else float("nan")
            lower, upper = _interval(
                _ratio_bootstrap(top, bottom, samples=samples, rng=rng),
                rate,
                confidence,
            )
            per_seed = {
                s: float(top[i].sum() / bottom[i].sum())
                if bottom[i].sum()
                else float("nan")
                for i, s in enumerate(seeds)
            }
            evicted_rate = evicted_lower = evicted_upper = None
            evicted_total: int | None = None
            if all(
                e.evicted_opportunities is not None and e.evicted_successes is not None
                for e in rows
            ):
                evicted_hits = {
                    key: float(cast(int, e.evicted_successes))
                    for key, e in zip(hits, rows, strict=True)
                }
                evicted = {
                    key: float(cast(int, e.evicted_opportunities))
                    for key, e in zip(hits, rows, strict=True)
                }
                _, etop = _matrix(evicted_hits)
                _, ebottom = _matrix(evicted)
                evicted_total = int(ebottom.sum())
                evicted_rate = (
                    float(etop.sum() / ebottom.sum()) if evicted_total else float("nan")
                )
                evicted_lower, evicted_upper = _interval(
                    _ratio_bootstrap(etop, ebottom, samples=samples, rng=rng),
                    evicted_rate,
                    confidence,
                )
            output.append(
                RetrievalEstimate(
                    protocol,
                    condition,
                    split,
                    history,
                    rule,
                    rate,
                    lower,
                    upper,
                    int(total),
                    per_seed,
                    evicted_rate,
                    evicted_lower,
                    evicted_upper,
                    evicted_total,
                    len(seeds),
                    top.shape[1],
                )
            )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class RuleComparison:
    """One quantity under the selected panel and the final-epoch supplement.

    ``difference`` is ``selected - final-epoch`` paired inside every seed/task
    cell, with its bootstrap interval, so a reported gain that depends on an
    isolated development peak shows as a difference that excludes zero.
    """

    protocol: str
    split: str
    history: str
    name: str
    kind: Literal["condition", "contrast"]
    selected: float
    final_epoch: float
    difference: float
    lower: float
    upper: float
    tasks: int
    seeds: int
    per_seed: Mapping[int, float]


def checkpoint_rule_comparison(
    events: Sequence[BenchmarkEvent],
    *,
    contrasts: Sequence[PlanContrast] = PLAN_CONTRASTS,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[RuleComparison, ...]:
    """Every condition and plan contrast evaluated under both checkpoint rules."""
    rng = np.random.default_rng(seed)
    panels = _panels(events)
    output: list[RuleComparison] = []
    keys = sorted(
        {(protocol, split, history) for protocol, split, history, _ in panels}
    )
    for protocol, split, history in keys:
        chosen = panels.get((protocol, split, history, "selected"))
        final = panels.get((protocol, split, history, "final-epoch"))
        if chosen is None or final is None:
            continue
        quantities: list[tuple[str, Literal["condition", "contrast"], _Reader]] = []
        for condition in sorted({e.condition for e in chosen}):

            def _cells(
                panel: Sequence[BenchmarkEvent], name: str = condition
            ) -> dict[Cell, float] | None:
                cells = cell_means([e for e in panel if e.condition == name])
                return cells or None

            quantities.append((condition, "condition", _cells))
        for row in contrasts:

            def _paired(
                panel: Sequence[BenchmarkEvent],
                row: PlanContrast = row,
                protocol: str = protocol,
            ) -> dict[Cell, float] | None:
                return _paired_cells(panel, row.left, row.right, protocol=protocol)

            quantities.append((row.name, "contrast", _paired))
        for name, kind, reader in quantities:
            one, two = reader(chosen), reader(final)
            if one is None or two is None:
                continue
            if set(one) != set(two):
                raise ResultValidationError(
                    f"{name} was not evaluated on one roster under both rules."
                )
            cells = {key: one[key] - two[key] for key in one}
            estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                cells, samples=samples, confidence=confidence, rng=rng
            )
            output.append(
                RuleComparison(
                    protocol,
                    split,
                    history,
                    name,
                    kind,
                    float(np.mean(list(one.values()))),
                    float(np.mean(list(two.values()))),
                    estimate,
                    lower,
                    upper,
                    tasks,
                    len(seeds),
                    per_seed,
                )
            )
    return tuple(output)


__all__ = [
    "ATTENTION_REGIME_INTERACTION",
    "DEFICIT_TRIGGER",
    "MINIMUM_DEPENDENCE",
    "PLAN_CONTRASTS",
    "SIGN_FLIP_MINIMUM_P",
    "STRATUM_FIELDS",
    "ContrastEstimate",
    "CumulativePoint",
    "CurvePoint",
    "Disposition",
    "Estimate",
    "FirstEventRow",
    "HistoryContrast",
    "InteractionEstimate",
    "IntervalRow",
    "PairedContrast",
    "PlanContrast",
    "PlateauVerdict",
    "RecoveryFraction",
    "RetrievalEstimate",
    "RuleComparison",
    "StratumEstimate",
    "StratumField",
    "TierContrastEstimate",
    "WritesRow",
    "attempt_curves",
    "cell_means",
    "checkpoint_rule_comparison",
    "cumulative_curves",
    "curves",
    "disposition",
    "estimates",
    "first_event_times",
    "history_contrasts",
    "interaction",
    "interval_successes",
    "paired_contrasts",
    "plan_contrasts",
    "plateau_screen",
    "recovery_fraction",
    "retrieval_rates",
    "seed_task_interval",
    "stratified_estimates",
    "tier_contrasts",
    "training_curve",
    "writes_table",
]


def count_recall_query_curves(
    events: Sequence[BenchmarkEvent],
    *,
    measure: str = "cumulative_accuracy",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[CumulativePoint, ...]:
    """Medium curves with whole-stream bootstrap and equal task/seed weights."""
    if measure not in ("cumulative_accuracy", "query_accuracy", "absolute_error"):
        raise ResultValidationError("Unknown CountRecall curve measure.")
    rng = np.random.default_rng(seed)
    output: list[CumulativePoint] = []
    for (protocol, split, history, rule), panel in _panels(events).items():
        if protocol != "count-recall-medium" or any(
            e.kind != "query" or e.answer is None for e in panel
        ):
            raise ResultValidationError("Medium curves require complete query answers.")
        for condition in sorted({e.condition for e in panel}):
            streams: dict[Cell, list[BenchmarkEvent]] = defaultdict(list)
            for event in panel:
                if event.condition == condition:
                    streams[
                        (event.training_seed, event.task_id, event.rollout_seed)
                    ].append(event)
            for rows in streams.values():
                rows.sort(key=lambda e: e.event_index)
                if [e.event_index for e in rows] != list(range(1, 104)):
                    raise ResultValidationError(
                        "Medium curves require all 103 queries per stream."
                    )
            for index in range(1, 104):
                values: dict[Cell, float] = {}
                for cell, rows in streams.items():
                    if measure == "cumulative_accuracy":
                        values[cell] = sum(e.numerator for e in rows[:index]) / index
                    elif measure == "query_accuracy":
                        values[cell] = float(rows[index - 1].numerator)
                    else:
                        event = rows[index - 1]
                        assert event.answer is not None and event.true_count is not None
                        values[cell] = float(abs(event.answer - event.true_count))
                estimate, lower, upper, tasks, seeds, per_seed = _exact_estimate(
                    values, samples=samples, confidence=confidence, rng=rng
                )
                output.append(
                    CumulativePoint(
                        protocol,
                        condition,
                        split,
                        history,
                        rule,
                        index,
                        estimate,
                        lower,
                        upper,
                        tasks,
                        len(seeds),
                        per_seed,
                    )
                )
    return tuple(output)


def count_recall_segment_rows(
    events: Sequence[BenchmarkEvent],
) -> list[dict[str, object]]:
    """Per-seed accuracy/error and pre-segment opportunities."""
    groups: dict[tuple[str, str, str, int], list[BenchmarkEvent]] = defaultdict(list)
    for event in events:
        if event.protocol != "count-recall-medium" or event.answer is None:
            raise ResultValidationError("Medium strata require recorded query answers.")
        groups[
            (event.condition, event.history, event.checkpoint_rule, event.training_seed)
        ].append(event)
    output: list[dict[str, object]] = []
    for (condition, history, rule, training_seed), rows in sorted(groups.items()):
        for low, high in ((1, 32), (33, 64), (65, 96), (97, 103)):
            selected = [e for e in rows if low <= e.event_index <= high]
            if not selected:
                continue
            older = [
                e
                for e in selected
                if e.count_before_current_segment is not None
                and e.count_before_current_segment > 0
            ]
            output.append(
                {
                    "condition": condition,
                    "history": history,
                    "checkpoint_rule": rule,
                    "seed": training_seed,
                    "first_query": low,
                    "last_query": high,
                    "streams": len({e.task_id for e in selected}),
                    "queries": len(selected),
                    "accuracy": float(np.mean([e.rate for e in selected])),
                    "absolute_error": float(
                        np.mean(
                            [
                                abs(cast(int, e.answer) - cast(int, e.true_count))
                                for e in selected
                            ]
                        )
                    ),
                    "pre_segment_opportunities": len(older)
                    if all(e.count_before_current_segment is not None for e in selected)
                    else None,
                    "pre_segment_accuracy": float(np.mean([e.rate for e in older]))
                    if older
                    else None,
                }
            )
    return output
