"""Evaluation records: one run row per evaluated checkpoint, one event per unit.

An event is an attempt (DarkRoom, Dark Key-to-Door, XLand), a query
(CountRecall) or an episode (MazeRunner, Concentration). Completed runs must
contain their whole declared roster; failed or missing runs carry status and
cost, never a fabricated score.

Since R4 a run says how its events were *retained* and which *metric* scores
them, and the two are separate: a ``complete`` Key-to-Door record keeps every
attempt of the 500-call task (a partial final attempt included) while the
legacy ``door_success_first8`` metric still reads only the first eight
completed attempts of such a record, and the new ``doors_completed`` count
reads them all. Legacy files carry neither field and load with the defaults
that describe them.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from reasoned_icrl.experiments.contracts import ALL_CONDITIONS, ResultValidationError

if TYPE_CHECKING:
    from reasoned_icrl.environments.concentration import ConcentrationFlip
    from reasoned_icrl.experiments.benchmarks import BenchmarkContract


CheckpointRule = Literal["selected", "final-epoch", "endpoint"]
"""Which rule chose a record's checkpoint: the contract's development-selected
checkpoint (``selected``), the fixed final training checkpoint of the legacy
supplement (``final-epoch``), or the exact common collection endpoint of the 8M
study (``endpoint``: the weights at the declared final epoch, written only for a
run that reached it, R4). Each rule is its own panel, so the one-checkpoint-
per-panel rule holds within each, and a selected checkpoint that happens to be
the endpoint is not a duplicate."""
CHECKPOINT_RULES: tuple[CheckpointRule, ...] = ("selected", "final-epoch", "endpoint")

Retention = Literal["scored-band", "complete"]
"""How a run's events were retained: the legacy scored band only (the first
eight Key-to-Door attempts, XLand's episodes 4-5), or every event of the task
(R4: every attempt, the unfinished one at the outer budget included and marked
``complete=False``)."""
RETENTIONS: tuple[Retention, ...] = ("scored-band", "complete")

PRIMARY_METRICS: dict[str, tuple[str, ...]] = {
    "darkroom": ("attempt_success",),
    "dark_key_to_door": ("door_success_first8", "doors_completed"),
    "count_recall": ("exact_accuracy",),
    "mazerunner": ("goal_fraction",),
    "concentration": ("pair_fraction",),
    "xland_minigrid": ("success_last2",),
    "xland_one_rule": ("success_last2",),
    "match_pattern": ("exact_accuracy",),
    "tmaze": ("goal_success",),
}
"""The declared primary endpoints each environment may name; results are never
pooled across them. Key-to-Door names two identities: the legacy
``door_success_first8`` (mean door success over the first eight attempts, the
4M study's metric and now the historical secondary) and the 8M study's
``doors_completed`` (R4: completed doors in the fixed 500-call task, a count
over every retained attempt, averaged equally over tasks)."""

COUNT_METRICS = frozenset({"doors_completed"})
"""Primary metrics that are counts per task (sums over a task's events), not
mean scored fractions; their practical-effect thresholds are in count units."""

SCORED_BAND_METRICS = frozenset({"door_success_first8"})
"""Rate metrics that read only the scored band of a complete record."""

Cell = tuple[int, int, int]
"""(training seed, task id, rollout seed): the unit the estimators average first."""


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    protocol: str
    benchmark: str
    condition: str
    training_seed: int
    checkpoint: str
    split: str
    history: str
    status: Literal["completed", "failed", "missing"]
    training_seconds: float | None = None
    evaluation_seconds: float | None = None
    peak_gpu_bytes: int | None = None
    peak_cpu_bytes: int | None = None
    parameter_count: int | None = None
    hardware: str = "unmeasured"
    note: str = ""
    checkpoint_rule: CheckpointRule = "selected"
    retention: Retention = "scored-band"
    metric: str | None = None
    """The contract's primary metric the panel is scored with (R4); ``None`` in
    records written before the field, which score the legacy metric."""
    charged_calls: int | None = None
    physical_actions: int | None = None
    reset_only_steps: int | None = None
    outer_length: int | None = None
    """Charged calls in the evaluated outer task: the
    contract's budget for a trained-horizon panel, the declared horizon for an
    extended-horizon panel; ``None`` on records written before the field."""
    layout_period: int | None = None
    """Layout-change continuation (evaluation only): a new
    hidden layout at the first attempt boundary after every this many charged
    calls; ``None`` on every ordinary panel."""
    checkpoint_sha256: str | None = None
    corpus_sha256: str | None = None
    """The evaluation's own interaction clocks over the whole roster (R4):
    every call the evaluator paid for, the ones the environment executed, and
    the reset-only remainder. ``None`` in records written before the fields."""

    @property
    def identity(self) -> tuple[str, str, int, str, str, str, str]:
        return (
            self.protocol,
            self.condition,
            self.training_seed,
            self.checkpoint,
            self.split,
            self.history,
            self.checkpoint_rule,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkEvent:
    protocol: str
    benchmark: str
    condition: str
    training_seed: int
    checkpoint: str
    split: str
    history: str
    task_id: int
    cluster_id: int
    rollout_seed: int
    kind: Literal["attempt", "query", "episode", "decision"]
    event_index: int
    step: int
    numerator: int
    denominator: int
    native_return: float
    true_count: int | None = None
    writes_before_decision: int | None = None
    evidence_age_writes: int | None = None
    evidence_age_writes_recent: int | None = None
    evidence_in_current_segment: bool | None = None
    count_before_current_segment: int | None = None
    count_in_current_segment: int | None = None
    start_step: int | None = None
    end_step: int | None = None
    checkpoint_rule: CheckpointRule = "selected"
    retrieval_opportunities: int | None = None
    retrieval_successes: int | None = None
    evicted_opportunities: int | None = None
    evicted_successes: int | None = None
    invalid_flips: int | None = None
    redundant_flips: int | None = None
    board_complete: bool | None = None
    """The Concentration retrieval ledger (decision 14): None everywhere else."""
    complete: bool | None = None
    """Attempt records of a ``complete`` retention: whether the attempt reached
    a physical boundary (door or step limit); ``False`` marks the one attempt
    the outer budget cut short, which scores zero and is censored for
    time-to-event statistics. ``None`` in scored-band records."""
    key_step: int | None = None
    """Key-to-Door: the global step (1-based, within the outer task) at which
    this attempt acquired the key, or ``None``; absent elsewhere."""

    left_pattern: int | None = None
    right_pattern: int | None = None
    answer: int | None = None
    """R8: answer required for complete query records; absent in legacy files."""

    flips: tuple[ConcentrationFlip, ...] | None = None
    outer_length: int | None = None
    """The outer budget of the task this event belongs to (see the run)."""
    layout_index: int | None = None
    """Attempt records of a layout-change panel: the hidden layout the attempt
    ran in (0 = the native task); ``None`` elsewhere."""
    goal_steps: tuple[int, ...] | None = None
    """MazeRunner lap records (the repeated-laps axis): the
    one-based charged call of the outer task at which each goal of the lap was
    reached, in order, ``numerator`` of them; ``None`` elsewhere."""
    """R7: complete public flip trace inside a Concentration episode; legacy
    terminal-only records keep None and cannot produce a within-board curve."""

    @property
    def run_identity(self) -> tuple[str, str, int, str, str, str, str]:
        return (
            self.protocol,
            self.condition,
            self.training_seed,
            self.checkpoint,
            self.split,
            self.history,
            self.checkpoint_rule,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            *self.run_identity,
            self.task_id,
            self.rollout_seed,
            self.kind,
            self.event_index,
        )

    @property
    def rate(self) -> float:
        return self.numerator / self.denominator


def metric_events(
    events: Sequence[BenchmarkEvent], metric: str | None, *, scored_band: int = 8
) -> list[BenchmarkEvent]:
    """The events a metric reads: every event for a count or a whole-roster
    rate, only the first ``scored_band`` completed attempts for the legacy
    scored-band rates. Metric membership is separate from retention."""
    if metric in SCORED_BAND_METRICS:
        return [
            e
            for e in events
            if e.event_index <= scored_band and (e.complete is None or e.complete)
        ]
    return list(events)


def cell_values(
    events: Sequence[BenchmarkEvent], metric: str | None = None
) -> dict[Cell, float]:
    """The metric's value in every (seed, task, rollout) cell.

    A count metric sums the events' successes (a partial attempt adds zero); a
    rate metric averages ``numerator / denominator`` over the events it reads.
    ``None`` is the legacy behaviour: the mean rate over every event given.
    """
    groups: defaultdict[Cell, list[float]] = defaultdict(list)
    for event in metric_events(events, metric):
        key = (event.training_seed, event.task_id, event.rollout_seed)
        groups[key].append(
            float(event.numerator) if metric in COUNT_METRICS else event.rate
        )
    if metric in COUNT_METRICS:
        return {key: float(sum(values)) for key, values in groups.items()}
    return {key: float(np.mean(values)) for key, values in groups.items()}


def primary_value(events: Sequence[BenchmarkEvent], metric: str | None = None) -> float:
    """The metric averaged equally over its cells (tasks, then rollouts)."""
    cells = cell_values(events, metric)
    if not cells:
        raise ResultValidationError("The primary metric needs at least one event.")
    return float(np.mean(list(cells.values())))


def _validate_ledger(event: BenchmarkEvent, name: str) -> None:
    """The Concentration retrieval ledger: present on its boards, absent elsewhere."""
    fields = (
        event.retrieval_opportunities,
        event.retrieval_successes,
        event.invalid_flips,
        event.redundant_flips,
        event.board_complete,
    )
    if name != "concentration":
        if any(value is not None for value in fields) or any(
            value is not None
            for value in (event.evicted_opportunities, event.evicted_successes)
        ):
            raise ResultValidationError(
                "Only Concentration records carry the retrieval ledger."
            )
        return
    if any(value is None for value in fields):
        return  # a legacy or partial record without the ledger
    opportunities, successes, invalid, redundant, complete = fields
    if any(
        type(value) is not int or value < 0
        for value in (opportunities, successes, invalid, redundant)
    ):
        raise ResultValidationError("Ledger counts are ints >= 0.")
    assert isinstance(opportunities, int) and isinstance(successes, int)
    if successes > opportunities:
        raise ResultValidationError("Retrieval successes exceed opportunities.")
    if type(complete) is not bool or complete != (event.numerator == event.denominator):
        raise ResultValidationError("board_complete must equal a full match.")
    evicted, evicted_hits = event.evicted_opportunities, event.evicted_successes
    if (evicted is None) != (evicted_hits is None):
        raise ResultValidationError("The evicted ledger needs both counts.")
    if evicted is not None and evicted_hits is not None:
        if event.writes_before_decision is None:
            raise ResultValidationError(
                "Evicted opportunities need the summary write counters."
            )
        if (
            type(evicted) is not int
            or type(evicted_hits) is not int
            or not 0 <= evicted <= opportunities
            or not 0 <= evicted_hits <= min(evicted, successes)
        ):
            raise ResultValidationError("Evicted ledger counts are inconsistent.")


def validate_benchmark_results(
    contracts: Sequence[BenchmarkContract],
    runs: Sequence[BenchmarkRun],
    events: Sequence[BenchmarkEvent],
) -> None:
    """Completed evaluations must contain their whole declared roster.

    Failed/missing evaluations carry status and cost, never a fabricated zero
    score. Zero behavioral success on a completed evaluation is valid evidence.
    """
    by_protocol = {c.protocol: c for c in contracts}
    if len(by_protocol) != len(contracts):
        raise ResultValidationError("Duplicate contract protocol.")
    by_run: dict[tuple[str, str, int, str, str, str, str], BenchmarkRun] = {}
    panels: set[tuple[object, ...]] = set()
    for run in runs:
        contract = by_protocol.get(run.protocol)
        if contract is None or run.benchmark != contract.environment.name:
            raise ResultValidationError("Unknown benchmark/protocol pair.")
        if run.condition not in ALL_CONDITIONS:
            raise ResultValidationError("Unknown result condition.")
        if run.status not in ("completed", "failed", "missing"):
            raise ResultValidationError("Unknown run status.")
        if run.checkpoint_rule not in CHECKPOINT_RULES:
            raise ResultValidationError("Unknown checkpoint rule.")
        if run.retention not in RETENTIONS:
            raise ResultValidationError("Unknown event retention.")
        if run.retention != contract.evaluation.retention:
            raise ResultValidationError(
                f"Run retention {run.retention!r} disagrees with the contract's "
                f"{contract.evaluation.retention!r}."
            )
        if run.metric is not None and run.metric not in PRIMARY_METRICS.get(
            contract.environment.name, ()
        ):
            raise ResultValidationError("Run metric is not one the environment names.")
        clocks = (run.charged_calls, run.physical_actions, run.reset_only_steps)
        if any(value is not None for value in clocks):
            if any(type(value) is not int or value < 0 for value in clocks):
                raise ResultValidationError(
                    "Evaluation clocks travel together as ints >= 0."
                )
            charged, physical, reset_only = clocks
            assert charged is not None and physical is not None
            assert reset_only is not None
            if charged != physical + reset_only:
                raise ResultValidationError(
                    "charged_calls must equal physical_actions + reset_only_steps."
                )
        if run.split not in contract.evaluation.splits or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", run.history
        ):
            raise ResultValidationError("Invalid evaluation split/history.")
        if (
            not run.checkpoint
            or type(run.training_seed) is not int
            or run.training_seed < 0
        ):
            raise ResultValidationError("Invalid checkpoint/seed.")
        if run.identity in by_run:
            raise ResultValidationError("Duplicate run.")
        panel = (
            run.protocol,
            run.condition,
            run.training_seed,
            run.split,
            run.history,
            run.checkpoint_rule,
        )
        if panel in panels:
            raise ResultValidationError(
                "Select one checkpoint per seed/panel before reporting."
            )
        panels.add(panel)
        by_run[run.identity] = run
        for value in (
            run.training_seconds,
            run.evaluation_seconds,
            run.peak_gpu_bytes,
            run.peak_cpu_bytes,
            run.parameter_count,
        ):
            if value is not None and (
                type(value) not in (float, int) or not math.isfinite(value) or value < 0
            ):
                raise ResultValidationError(
                    "Resource measurements must be finite/nonnegative."
                )
        for name, value in (
            ("outer_length", run.outer_length),
            ("layout_period", run.layout_period),
        ):
            if value is not None and (
                isinstance(value, bool) or type(value) is not int or value < 1
            ):
                raise ResultValidationError(
                    f"{name} must be a positive integer when set."
                )
    identities: set[tuple[object, ...]] = set()
    grouped: defaultdict[
        tuple[str, str, int, str, str, str, str], list[BenchmarkEvent]
    ] = defaultdict(list)
    for event in events:
        event_run = by_run.get(event.run_identity)
        if event_run is None or event_run.status != "completed":
            raise ResultValidationError(
                "Events require a completed, declared evaluation run."
            )
        if event.benchmark != event_run.benchmark:
            raise ResultValidationError("Event benchmark disagrees with run.")
        if event.identity in identities:
            raise ResultValidationError("Duplicate evaluation event.")
        identities.add(event.identity)
        for value in (
            event.task_id,
            event.cluster_id,
            event.rollout_seed,
            event.event_index,
            event.step,
            event.numerator,
            event.denominator,
        ):
            if type(value) is not int or value < 0:
                raise ResultValidationError("Invalid event integer counter.")
        if event.event_index < 1 or event.step < 1 or event.denominator < 1:
            raise ResultValidationError("Invalid event denominator/index/step.")
        if (
            event.numerator > event.denominator
            or type(event.native_return) not in (int, float)
            or not math.isfinite(event.native_return)
        ):
            raise ResultValidationError("Invalid result fraction/return.")
        contract = by_protocol[event.protocol]
        env = contract.environment
        if event.kind != contract.event_kind:
            raise ResultValidationError("Wrong event kind for benchmark.")
        if env.name == "mazerunner":
            denominator = env.goals
        elif env.name == "concentration":
            denominator = env.size // 2
        else:
            denominator = 1
        if event.denominator != denominator:
            raise ResultValidationError("Wrong benchmark metric denominator.")
        if event.step > env.horizon * env.attempts:
            raise ResultValidationError("Event exceeds native horizon.")
        if event.kind == "decision":
            if (
                event.step,
                event.event_index,
                event.start_step,
                event.end_step,
                event.complete,
            ) != (7, 1, 7, 7, True):
                raise ResultValidationError(
                    "A match-pattern decision is the complete seventh call."
                )
            left, right = event.left_pattern, event.right_pattern
            if (
                type(left) is not int
                or type(right) is not int
                or not (0 <= left < 4 and 0 <= right < 4)
            ):
                raise ResultValidationError("Missing/invalid match-pattern strata.")
            if type(event.answer) is not int or event.answer not in (0, 1):
                raise ResultValidationError("Invalid match-pattern binary answer.")
            if (
                event.numerator != int(event.answer == int(left == right))
                or event.native_return != 2 * event.numerator - 1
            ):
                raise ResultValidationError(
                    "Match-pattern answer/reward/accuracy disagree."
                )
            if event.writes_before_decision is not None:
                raise ResultValidationError("Match-pattern has no summary writes.")
        if event.kind == "query":
            # The native deck deals every category (deck + 1) / size times
            # (26 for Easy and Medium, 16 for Hard), the largest possible count;
            # the query-only tail of an extended panel deals nothing and scores
            # at the native reward scale, so both bounds follow the deck.
            # Imported here, not at module level: the environments package
            # loads the AMAGO adapters, and the record module must load
            # without amago or torch (tests/test_scripts.py, the layering test).
            from reasoned_icrl.environments.count_recall import (
                COUNT_RECALL_HORIZONS,
                count_recall_variant,
            )

            native = COUNT_RECALL_HORIZONS[count_recall_variant(env.benchmark)]
            largest = (native + 1) // env.size
            if (
                type(event.true_count) is not int
                or not 0 <= event.true_count <= largest
            ):
                raise ResultValidationError(
                    "CountRecall query requires valid true_count."
                )
            if (
                event.answer is not None or contract.evaluation.retention == "complete"
            ) and (
                type(event.answer) is not int
                or not 0 <= event.answer <= largest
                or int(event.answer == event.true_count) != event.numerator
            ):
                raise ResultValidationError(
                    "CountRecall query answer/correctness mismatch."
                )
            if event.step != event.event_index or not math.isclose(
                event.native_return,
                (2 * event.numerator - 1) / native,
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ResultValidationError("CountRecall query timing/reward mismatch.")
        elif event.true_count is not None:
            raise ResultValidationError("Only query records accept true_count.")
        if (
            event.kind == "episode"
            and env.name == "mazerunner"
            and not math.isclose(event.native_return, event.numerator)
        ):
            raise ResultValidationError(
                "MazeRunner native return must equal goals completed."
            )
        # Matches pay 1 / pairs and every penalty is negative, so a board's
        # return never exceeds its pair fraction; every flip costs at most
        # 2 / horizon, so it never falls below -2.
        if env.name == "concentration" and not (
            -2.0 <= event.native_return <= event.rate + 1e-9
        ):
            raise ResultValidationError(
                "Concentration native return exceeds its matched pairs."
            )
        _validate_ledger(event, env.name)
        if event.flips is not None:
            if env.name != "concentration":
                raise ResultValidationError("Only Concentration carries flips.")
            if tuple(f.index for f in event.flips) != tuple(range(1, event.step + 1)):
                raise ResultValidationError("The flip trace must cover every decision.")
            if sum(
                f.matched for f in event.flips
            ) != event.numerator or not math.isclose(
                sum(f.native_reward for f in event.flips),
                event.native_return,
                abs_tol=1e-9,
            ):
                raise ResultValidationError(
                    "The flip trace disagrees with pairs/return."
                )
            if event.step < env.horizon and not event.board_complete:
                raise ResultValidationError("An early board must be complete.")
            from reasoned_icrl.environments.concentration import (
                CONCENTRATION_RANKS,
                concentration_variant,
            )

            ranks = CONCENTRATION_RANKS[concentration_variant(contract.protocol)]
            for flip in event.flips:
                if (
                    not 0 <= flip.position < env.size
                    or not 0 <= flip.rank < ranks
                    or (
                        flip.partner_last_reveal is not None
                        and not 0 < flip.partner_last_reveal < flip.index
                    )
                ):
                    raise ResultValidationError(
                        "Invalid public position/rank/reveal in flip trace."
                    )
                expected_reward = (
                    1.0 / event.denominator
                    if flip.matched
                    else -(2 if flip.second else 1) / env.horizon
                    if flip.invalid
                    else -2.0 / env.horizon
                    if flip.second
                    else 0.0
                )
                if not math.isclose(flip.native_reward, expected_reward, abs_tol=1e-9):
                    raise ResultValidationError(
                        "Flip reward disagrees with its native outcome."
                    )
                if flip.hit and not flip.matched:
                    raise ResultValidationError("An eligible-partner hit must match.")
            for attribute, count in (
                ("opportunity", event.retrieval_opportunities),
                ("hit", event.retrieval_successes),
                ("invalid", event.invalid_flips),
                ("redundant", event.redundant_flips),
            ):
                if count != sum(getattr(f, attribute) for f in event.flips):
                    raise ResultValidationError(
                        "The flip trace disagrees with its ledger."
                    )
        elif env.name == "concentration" and event_run.retention == "complete":
            raise ResultValidationError("Complete Concentration records require flips.")
        if event.kind == "attempt" and env.name in ("xland_minigrid", "xland_one_rule"):
            # Success pays 1 - 0.9 * step / limit, once; a failure pays nothing.
            if (event.numerator == 1) != (0.0 < event.native_return <= 1.0) or (
                event.numerator == 0 and event.native_return != 0.0
            ):
                raise ResultValidationError("Attempt success/return mismatch.")
        elif event.kind == "attempt" and env.name == "mazerunner":
            # A lap of the repeated-laps axis: goals reached of the ordered
            # sequence, one unit each, dated by their calls.
            steps = event.goal_steps
            if event.native_return != float(event.numerator):
                raise ResultValidationError("Lap goals/return mismatch.")
            if (
                not isinstance(steps, tuple | list)
                or len(steps) != event.numerator
                or any(type(step) is not int for step in steps)
                or any(b <= a for a, b in pairwise(steps))
                or (
                    steps
                    and (
                        event.start_step is None
                        or event.end_step is None
                        or not event.start_step <= steps[0]
                        or not steps[-1] <= event.end_step
                    )
                )
            ):
                raise ResultValidationError(
                    "A lap dates every goal it reached, in order, inside its span."
                )
        elif event.kind == "attempt":
            full = 2.0 if env.name == "dark_key_to_door" else 1.0
            if event.native_return not in (0.0, 1.0, full) or (
                event.numerator == 1
            ) != (event.native_return == full):
                raise ResultValidationError("Attempt success/return mismatch.")
        if event.goal_steps is not None and (
            event.kind != "attempt" or env.name != "mazerunner"
        ):
            raise ResultValidationError("Only MazeRunner laps carry goal steps.")
        # R4: complete retention marks every attempt; a scored-band record
        # carries no flag. Only attempts are ever partial.
        if event.kind != "attempt" and event.complete is not None:
            raise ResultValidationError("Only attempt records carry a complete flag.")
        if event_run.retention == "complete" and event.kind == "attempt":
            if type(event.complete) is not bool:
                raise ResultValidationError(
                    "A complete-retention attempt says whether it finished."
                )
            if not event.complete and (
                event.numerator >= event.denominator
                if env.name == "mazerunner"
                else event.numerator != 0
            ):
                raise ResultValidationError("A partial attempt cannot succeed.")
        elif event.complete is False:
            raise ResultValidationError(
                "A scored-band record holds completed attempts only."
            )
        key_step = event.key_step
        if key_step is not None:
            if event.kind != "attempt" or env.name != "dark_key_to_door":
                raise ResultValidationError(
                    "Only Key-to-Door attempts carry a key step."
                )
            if type(key_step) is not int or event.native_return < 1.0:
                raise ResultValidationError(
                    "key_step needs a key acquisition in the attempt's return."
                )
            if event.start_step is None or event.end_step is None:
                raise ResultValidationError("key_step needs the attempt's span.")
            if not event.start_step <= key_step <= event.end_step:
                raise ResultValidationError("key_step lies outside the attempt.")
        elif (
            event_run.retention == "complete"
            and env.name == "dark_key_to_door"
            and event.native_return >= 1.0
        ):
            raise ResultValidationError(
                "A complete Key-to-Door record dates every key acquisition."
            )
        # Current M0 rosters sample one independent task/stream/map per ID.
        if event.cluster_id != event.task_id:
            raise ResultValidationError(
                "Current protocols require cluster_id == task_id."
            )
        # Summary-carrier bookkeeping: boundaries crossed before the event's
        # decision, and since its evidence entered the model. Absent (None) for
        # every other carrier and in result files written before the fields.
        writes, age = event.writes_before_decision, event.evidence_age_writes
        if writes is not None and (type(writes) is not int or writes < 0):
            raise ResultValidationError("writes_before_decision must be an int >= 0.")
        if age is not None:
            if writes is None or type(age) is not int or not 0 <= age <= writes:
                raise ResultValidationError(
                    "evidence_age_writes needs writes_before_decision and lies in "
                    "[0, writes_before_decision]."
                )
            if event.kind == "episode":
                raise ResultValidationError(
                    "Episode records carry no evidence age; attempts and queries do."
                )
        # Retention diagnostics (spec §9): the most recent evidence's age and
        # whether it is still in the open segment (attempts); the queried
        # count split across the open segment's boundary (queries); and the
        # physical span of every event, for late-task measures.
        recent = event.evidence_age_writes_recent
        flag = event.evidence_in_current_segment
        if recent is not None:
            if age is None or type(recent) is not int or not 0 <= recent <= age:
                raise ResultValidationError(
                    "evidence_age_writes_recent needs evidence_age_writes and lies "
                    "in [0, evidence_age_writes]."
                )
            if event.kind != "attempt":
                raise ResultValidationError(
                    "Only attempt records carry the recent evidence age."
                )
            if type(flag) is not bool or flag != (recent == 0):
                raise ResultValidationError(
                    "evidence_in_current_segment must equal "
                    "evidence_age_writes_recent == 0."
                )
        elif flag is not None:
            raise ResultValidationError(
                "evidence_in_current_segment needs evidence_age_writes_recent."
            )
        before = event.count_before_current_segment
        inside = event.count_in_current_segment
        if (before is None) != (inside is None):
            raise ResultValidationError("The segment count split needs both parts.")
        if before is not None and inside is not None:
            if event.kind != "query":
                raise ResultValidationError(
                    "Only query records carry the segment count split."
                )
            if writes is None or any(
                type(part) is not int or part < 0 for part in (before, inside)
            ):
                raise ResultValidationError(
                    "count_before/in_current_segment need writes_before_decision "
                    "and are ints >= 0."
                )
            if before + inside != event.true_count:
                raise ResultValidationError(
                    "count_before_current_segment + count_in_current_segment must "
                    "equal true_count."
                )
            if writes == 0 and before:
                raise ResultValidationError("No count precedes the first segment.")
        # The event's spans are read against its run's outer budget: the
        # declared horizon of an extended-horizon panel, the contract's
        # budget otherwise (records written before the field carry None).
        budget = (
            env.outer_length
            if event_run.outer_length is None
            else event_run.outer_length
        )
        if event.outer_length not in (None, budget):
            raise ResultValidationError(
                "An event's outer budget disagrees with its run's."
            )
        layout = event.layout_index
        if layout is not None:
            if event.kind != "attempt" or event_run.layout_period is None:
                raise ResultValidationError(
                    "Only attempt records of a layout-change panel carry a layout."
                )
            if isinstance(layout, bool) or type(layout) is not int or layout < 0:
                raise ResultValidationError("layout_index must be an int >= 0.")
        elif event_run.layout_period is not None and event.kind == "attempt":
            raise ResultValidationError(
                "Every attempt of a layout-change panel names its layout."
            )
        start, end = event.start_step, event.end_step
        if (start is None) != (end is None):
            raise ResultValidationError("start_step and end_step travel together.")
        if start is not None and end is not None:
            if (
                type(start) is not int
                or type(end) is not int
                or not 1 <= start <= end <= budget
            ):
                raise ResultValidationError(
                    "start_step/end_step must satisfy 1 <= start <= end <= the "
                    "outer task length."
                )
            if event.kind == "attempt" and end - start + 1 != event.step:
                raise ResultValidationError(
                    "An attempt's step span must equal its steps."
                )
            if event.kind == "query" and (start != event.event_index or start != end):
                raise ResultValidationError("A query spans its own decision.")
            if event.kind == "episode" and (start != 1 or end != event.step):
                raise ResultValidationError(
                    "An episode spans decisions 1 .. decisions."
                )
        grouped[event.run_identity].append(event)
    for identity, run in by_run.items():
        if run.status != "completed":
            continue
        contract = by_protocol[run.protocol]
        env = contract.environment
        count = {
            "attempt": env.attempts,
            "query": env.horizon * env.attempts,
            "episode": 1,
            "decision": 1,
        }[contract.event_kind]
        first = env.scored_from if contract.event_kind == "attempt" else 1
        budget = env.outer_length if run.outer_length is None else run.outer_length
        if run.outer_length is not None and any(
            row.outer_length not in (None, run.outer_length)
            for row in grouped[identity]
        ):
            raise ResultValidationError(
                "An event's outer budget disagrees with its run's."
            )
        if run.retention == "complete" and contract.event_kind == "attempt":
            _validate_complete_attempts(
                grouped[identity],
                tasks=contract.roster(run.split),
                rollouts=contract.evaluation.rollout_seeds,
                guaranteed=count,
                outer_length=budget,
            )
            continue
        expected = {
            (task, rollout, index)
            for task in contract.roster(run.split)
            for rollout in contract.evaluation.rollout_seeds
            for index in range(first, count + 1)
        }
        actual = {
            (row.task_id, row.rollout_seed, row.event_index)
            for row in grouped[identity]
        }
        if actual != expected:
            raise ResultValidationError(
                "Completed evaluation has incomplete/unexpected event roster."
            )


def _validate_complete_attempts(
    events: Sequence[BenchmarkEvent],
    *,
    tasks: Sequence[int],
    rollouts: Sequence[int],
    guaranteed: int,
    outer_length: int,
) -> None:
    """Every (task, rollout) holds attempts 1..n, contiguous, with at least the
    guaranteed count finished, at most the last one partial, and the record
    spanning the whole outer task (R4): its last attempt ends at the budget or
    one call before it (the reset-only call that ends a task)."""
    by_unit: defaultdict[tuple[int, int], list[BenchmarkEvent]] = defaultdict(list)
    for event in events:
        by_unit[(event.task_id, event.rollout_seed)].append(event)
    expected = {(int(task), int(rollout)) for task in tasks for rollout in rollouts}
    if set(by_unit) != expected:
        raise ResultValidationError(
            "Completed evaluation has incomplete/unexpected event roster."
        )
    for unit, rows in by_unit.items():
        ordered = sorted(rows, key=lambda row: row.event_index)
        if [row.event_index for row in ordered] != list(range(1, len(ordered) + 1)):
            raise ResultValidationError(
                f"Task {unit[0]}: attempts must be contiguous from 1."
            )
        finished = [row for row in ordered if row.complete]
        if len(finished) < guaranteed:
            raise ResultValidationError(
                f"Task {unit[0]} finished {len(finished)} attempts; the budget "
                f"guarantees {guaranteed}."
            )
        if any(not row.complete for row in ordered[:-1]):
            raise ResultValidationError(
                f"Task {unit[0]}: only the last attempt may be partial."
            )
        steps = [row.end_step for row in ordered]
        if any(value is None for value in steps) or any(
            a is not None and b is not None and a >= b for a, b in pairwise(steps)
        ):
            raise ResultValidationError(
                f"Task {unit[0]}: retained attempts must carry increasing spans."
            )
        last = steps[-1]
        if last is None or not outer_length - 1 <= last <= outer_length:
            raise ResultValidationError(
                f"Task {unit[0]}: the retained attempts do not span the outer task "
                f"(last attempt ends at {last}, budget {outer_length})."
            )


def write_benchmark_results(
    path: str | Path,
    contracts: Sequence[BenchmarkContract],
    runs: Sequence[BenchmarkRun],
    events: Sequence[BenchmarkEvent],
) -> Path:
    validate_benchmark_results(contracts, runs, events)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"runs": [asdict(r) for r in runs], "events": [asdict(e) for e in events]},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return output


COMPACT_SUFFIX = ".gz"
"""A panel's per-record file may be kept gzip-compacted beside its plain name
(``benchmark_results.json.gz``): the readers accept either,
so a panel read once can give its disk back (a 32-pair CountRecall panel is
1.1 GB per seed plain and about 60 MB compacted)."""


def results_file(path: str | Path) -> Path | None:
    """The plain per-record file, or its compacted twin, or ``None``."""
    plain = Path(path)
    if plain.is_file():
        return plain
    packed = plain.with_name(plain.name + COMPACT_SUFFIX)
    return packed if packed.is_file() else None


def read_results_text(path: str | Path) -> str:
    """The per-record JSON text of a panel, plain or compacted."""
    found = results_file(path)
    if found is None:
        raise FileNotFoundError(str(path))
    if found.name.endswith(COMPACT_SUFFIX):
        with gzip.open(found, "rt", encoding="utf-8") as handle:
            return handle.read()
    return found.read_text()


def compact_benchmark_results(path: str | Path) -> Path:
    """Replace a plain per-record file by its gzip twin (idempotent). The plain
    file is removed only after the twin has been written and read back to the
    same length, so an interrupted compaction leaves the plain file in place."""
    plain = Path(path)
    packed = plain.with_name(plain.name + COMPACT_SUFFIX)
    if not plain.is_file():
        if packed.is_file():
            return packed
        raise FileNotFoundError(str(plain))
    text = plain.read_text()
    with gzip.open(packed, "wt", encoding="utf-8", compresslevel=6) as handle:
        handle.write(text)
    with gzip.open(packed, "rt", encoding="utf-8") as handle:
        if len(handle.read()) != len(text):
            packed.unlink()
            raise OSError(
                f"Compacting {plain} did not round-trip; the plain file is kept."
            )
    plain.unlink()
    return packed


def read_benchmark_results(
    path: str | Path, contracts: Sequence[BenchmarkContract]
) -> tuple[tuple[BenchmarkRun, ...], tuple[BenchmarkEvent, ...]]:
    from reasoned_icrl.environments.concentration import (
        ConcentrationFlip,
    )

    try:
        raw = json.loads(read_results_text(path))
        if not isinstance(raw, Mapping) or set(raw) != {"runs", "events"}:
            raise ValueError("Expected runs and events")
        runs = tuple(BenchmarkRun(**row) for row in raw["runs"])
        events = tuple(
            BenchmarkEvent(
                **(
                    row
                    | {
                        "flips": tuple(
                            ConcentrationFlip(**flip) for flip in row["flips"]
                        )
                        if row.get("flips") is not None
                        else None,
                        "goal_steps": tuple(int(s) for s in row["goal_steps"])
                        if row.get("goal_steps") is not None
                        else None,
                    }
                )
            )
            for row in raw["events"]
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ResultValidationError("Malformed benchmark result file.") from error
    validate_benchmark_results(contracts, runs, events)
    return runs, events
