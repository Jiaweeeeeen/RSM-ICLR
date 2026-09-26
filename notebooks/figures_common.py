"""Shared plumbing for the paper figure notebooks.

What every figure notebook needs and nothing figure-specific: where the saved
records and the confirmation tier reports are, the environment table (the
paper's three main benchmarks in column order and the optional T-Maze of the
appendix), the roster rule, the palette
and legend names of the figure plan, the live-state projection to a horizon,
the CountRecall write strata, the Matplotlib style, and how a built figure is
saved with the data behind each panel. Estimators and drawing live inside
each notebook.

Roster rule (study protocol, section 3): every main-text panel of one
environment sits on one roster. :func:`choose_split` takes the first split of
the preference list on which every cell has records, otherwise the split on
which the most cells have them, so the competing methods always share one
panel; cells missing there stay blank and are never filled from another split.
A fit whose development series has not reached the endpoint is reported as in
flight, never read.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from reasoned_icrl.experiments.benchmarks import BenchmarkContract, load_contract
from reasoned_icrl.experiments.horizon import horizon_kind, maze_timer
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    read_benchmark_results,
    read_results_text,
    results_file,
)

SEEDS: tuple[int, ...] = (42, 100, 2026)
"""The three training seeds of every 8M cell."""
FULL_BUDGET = 8_000_000
"""Charged training calls of a complete 8M fit."""
ENDPOINT_CHECKPOINT = "policy_epoch_999"
"""The collection endpoint's saved label (AMAGO labels epochs from zero)."""
SEGMENT = 32
"""Records per summary segment (C): the lazy boundary opens at record 33, 65, ..."""

# ------------------------------------------------------------------ records


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    key: str
    protocol: str
    title: str
    contract: str
    """Contract YAML, relative to the repository root."""
    metric: str
    intervention: str
    """The environment's declared cache-boundary intervention."""
    unit: str
    """The endpoint's unit, as an axis label."""
    delta: float
    """The declared practical margin in that unit (EXPERIMENTS §7)."""
    horizon_kind: Literal["calls", "stream", "corridor", "maze"]
    """What ``horizon`` means: charged calls (Key-to-Door), the fixed query
    stream (CountRecall, whose beyond-horizon axis is the continued streams,
    not a longer horizon), the trained corridor length (T-Maze, evaluated at
    that length only) or the maze size (MazeRunner, under the area-scaled
    step budget)."""
    report: str
    """The confirmation tier report directory, relative to the study root
    that holds the environment's Memo comparison (Key-to-Door, CountRecall)
    or its own study (T-Maze)."""

    def records_at(self, horizon: int, contract: BenchmarkContract) -> int:
        """Records one actor holds after a task of this ``horizon``: the
        reset record plus one per charged call (Key-to-Door), the reset plus
        every observation of the stream (CountRecall, horizon ignored), the
        reset plus ``L + 1`` decisions (T-Maze), the reset plus the
        area-scaled step budget of a maze of that size (MazeRunner)."""
        if self.horizon_kind == "calls":
            return int(horizon) + 1
        if self.horizon_kind == "corridor":
            return int(horizon) + 2
        if self.horizon_kind == "maze":
            environment = contract.environment
            return maze_budget(int(horizon), environment) + 1
        return int(contract.environment.horizon) + 1

    def native_horizon(self, contract: BenchmarkContract) -> int:
        if self.horizon_kind in ("corridor", "maze"):
            return int(contract.environment.size)
        if self.horizon_kind == "calls":
            return int(contract.environment.outer_length)
        return int(contract.environment.horizon)


def maze_budget(size: int, environment: Any) -> int:
    """The step budget of a MazeRunner task at ``size``: the trained budget
    at the trained size, else the area-scaled timer of the larger-maze axis
    (EXPERIMENTS section 5)."""
    trained = int(
        environment.size
        if environment.protocol_size is None
        else environment.protocol_size
    )
    if int(size) == trained:
        return int(environment.horizon)
    return maze_timer(
        int(size), trained_size=trained, trained_horizon=int(environment.horizon)
    )


ENVIRONMENTS: Mapping[str, EnvironmentSpec] = {
    "keydoor": EnvironmentSpec(
        "keydoor",
        "native-keydoor-fixed500-first8",
        "Dark Key-to-Door",
        "configs/environments/8m/dark_key_to_door.yaml",
        "doors_completed",
        "attempt-cleared",
        "Completed doors per 500-call task",
        1.0,
        "calls",
        "reports/native-keydoor-fixed500-first8/confirmation/tier",
    ),
    "countrecall": EnvironmentSpec(
        "countrecall",
        "count-recall-medium",
        "CountRecall",
        "configs/environments/8m/count_recall.yaml",
        "exact_accuracy",
        "current-token",
        "Exact accuracy",
        0.05,
        "stream",
        "reports/count-recall-medium/confirmation/tier",
    ),
    "mazerunner": EnvironmentSpec(
        "mazerunner",
        "mazerunner-15-randomized-actions",
        "MazeRunner",
        "configs/environments/8m/mazerunner.yaml",
        "goal_fraction",
        "summary-cleared",
        "Goal fraction",
        0.1,
        "maze",
        "reports/mazerunner-15-randomized-actions/confirmation/tier",
    ),
    "tmaze": EnvironmentSpec(
        "tmaze",
        "tmaze-passive-l32-256-v3",
        "Passive T-Maze",
        "configs/environments/8m/tmaze_v3.yaml",
        "goal_success",
        "current-token",
        "Greedy success",
        0.25,
        "corridor",
        "reports/tmaze-passive-l32-256-v3/confirmation/tier",
    ),
}
"""Every environment the notebooks read, keyed for the figures: the paper's
three main benchmarks — Dark Key-to-Door,
CountRecallMedium and MazeRunner 15 x 15 with randomised actions — and the
passive T-Maze v3 of the appendix (the training corridor drawn from 32 to 256,
evaluation at 128)."""

TMAZE_STUDY = "tmaze-v3-8m"
"""The passive T-Maze's study root. Only v3 is kept; the earlier
single-corridor versions are gone."""

PAPER_ENVIRONMENTS: tuple[str, ...] = ("keydoor", "countrecall", "mazerunner")
"""The main text's three benchmarks, in column order (Dark Key-to-Door, CountRecall and
MazeRunner as the main result, the T-Maze in the appendix)."""
APPENDIX_ENVIRONMENTS: tuple[str, ...] = ("tmaze",)
"""The optional passive T-Maze: every display of it belongs to the appendix."""
ALL_ENVIRONMENTS: tuple[str, ...] = (*PAPER_ENVIRONMENTS, *APPENDIX_ENVIRONMENTS)

PAPER_CONDITIONS: tuple[str, ...] = (
    "raw_summary",
    "raw_segment",
    "full_context",
    "full_gru",
    "memo",
    "memo_fixed",
)
"""The six cells declared (study protocol, section 1)."""
METHOD_CONDITION = "raw_summary"
"""The paper's method, RSM-O: the replacing
rewrite, the boundary write replaces the carried memory."""
ABLATION_CONDITION = "raw_summary_residual"
"""RSM-R, the residual rewrite (the boundary write added to the carried
memory): the method's rewrite ablation, drawn beside
it wherever it has panels. It stands in as the CountRecall strata reference
while the method's panel is absent: the strata depend on the task and the
fixed C32 schedule."""
OVERWRITE_CONDITION = METHOD_CONDITION
RESIDUAL_CONDITION = ABLATION_CONDITION
DRAWN_CONDITIONS: tuple[str, ...] = (
    METHOD_CONDITION,
    ABLATION_CONDITION,
    *(c for c in PAPER_CONDITIONS if c != METHOD_CONDITION),
)
"""Every cell a figure draws where its records exist, the method first."""


def find_repository(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Run from inside the repository.")


@dataclass(frozen=True, slots=True)
class StudyLayout:
    """Where each cell's runs live: ``<root>/<protocol>/<condition>/seed-<s>``."""

    repository: Path
    roots: Mapping[tuple[str, str], Path]
    """Study root by (environment key, condition)."""
    contracts: Mapping[str, BenchmarkContract] = field(default_factory=dict)
    """Contract overrides by environment key (tests use two-task rosters)."""
    reports: Mapping[str, Path] = field(default_factory=dict)
    """The confirmation tier report directory by environment key."""
    specs: Mapping[str, EnvironmentSpec] = field(
        default_factory=lambda: dict(ENVIRONMENTS)
    )
    """The environment spec by key (the T-Maze's version is the layout's)."""

    def spec(self, environment: str) -> EnvironmentSpec:
        """The layout's spec of one environment (``ENVIRONMENTS`` unless the
        layout was built for another T-Maze version)."""
        return self.specs.get(environment, ENVIRONMENTS[environment])

    def report_directory(self, environment: str) -> Path | None:
        """Where the environment's confirmation tier report lives, or ``None``
        when the layout names no report for it."""
        return self.reports.get(environment)

    def run_directory(self, environment: str, condition: str, seed: int) -> Path:
        spec = self.spec(environment)
        return (
            self.roots[(environment, condition)]
            / spec.protocol
            / condition
            / f"seed-{seed}"
        )

    def contract(self, environment: str) -> BenchmarkContract:
        if environment in self.contracts:
            return self.contracts[environment]
        return load_contract(self.repository / self.spec(environment).contract)

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repository.resolve()))
        except ValueError:
            return str(path)


def default_layout(
    repository: Path,
    *,
    outputs: Path | None = None,
    core_study: str = "summary-memory-8m",
    memo_studies: Mapping[str, str] | None = None,
    tmaze_study: str = TMAZE_STUDY,
    mazerunner_study: str = "mazerunner-8m",
    conditions: Sequence[str] = DRAWN_CONDITIONS,
) -> StudyLayout:
    """Where every cell's runs and every environment's report live, all under
    ``outputs`` (default ``<repository>/outputs``). Key-to-Door: the core 8M
    study (the RSM pair, the full-history and GRU references, the residual
    cell) plus the Memo study, which holds the confirmation tier report;
    CountRecall the same. MazeRunner: its own study holds every cell and its
    report. The T-Maze (appendix): its own v3 study, the only version kept.
    An explicit ``*_study`` overrides a study root only."""
    memo = dict(memo_studies or {})
    specs = dict(ENVIRONMENTS)
    memo.setdefault("keydoor", "memo-key-to-door-8m")
    memo.setdefault("countrecall", "memo-count-recall-8m")
    base = outputs if outputs is not None else repository / "outputs"
    roots: dict[tuple[str, str], Path] = {}
    reports: dict[str, Path] = {}
    for environment, spec in specs.items():
        own_study = {"tmaze": tmaze_study, "mazerunner": mazerunner_study}.get(
            environment
        )
        for condition in conditions:
            if own_study is not None:
                study = own_study
            elif condition.startswith("memo"):
                study = memo[environment]
            else:
                study = core_study
            roots[(environment, condition)] = base / study
        report_study = own_study if own_study is not None else memo[environment]
        reports[environment] = base / report_study / spec.report
    return StudyLayout(repository, roots, reports=reports, specs=specs)


def fit_complete(
    layout: StudyLayout, environment: str, condition: str, seed: int
) -> bool:
    """The run measured the full 8M budget and its development series reached
    the endpoint label; a fit still training or still being scored is not."""
    run = layout.run_directory(environment, condition, seed)
    systems, development = run / "systems.json", run / "development.json"
    if not systems.is_file() or not development.is_file():
        return False
    measured = json.loads(systems.read_text()).get("measured", {})
    scored = json.loads(development.read_text())
    return bool(
        measured.get("charged_calls") == FULL_BUDGET
        and scored.get("endpoint_reached") is True
    )


def endpoint_panel(
    layout: StudyLayout,
    environment: str,
    condition: str,
    seed: int,
    split: str,
    history: str,
    *,
    contract: BenchmarkContract,
    sources: list[str] | None = None,
) -> list[BenchmarkEvent] | None:
    """One seed's endpoint panel on one split and history, or ``None``."""
    directory = layout.run_directory(environment, condition, seed) / "eval"
    path = directory / f"{split}-{history}-endpoint" / "benchmark_results.json"
    if results_file(path) is None:  # the plain file or its compacted twin
        return None
    runs, events = read_benchmark_results(path, [contract])
    (run,) = runs
    if (
        run.status != "completed"
        or run.checkpoint != ENDPOINT_CHECKPOINT
        or run.split != split
        or run.history != history
    ):
        return None
    if sources is not None:
        sources.append(layout.relative(path))
    return list(events)


def horizon_contract(contract: BenchmarkContract, horizon: int) -> BenchmarkContract:
    """The evaluation-only contract of an extended-horizon panel: the same
    environment with its outer budget set to ``horizon`` charged calls
    (Key-to-Door) or its maze enlarged to ``horizon`` under the area-scaled
    budget with the trained protocol identity kept (MazeRunner); rosters,
    recipe and identity untouched, as the adapter derives them."""
    horizon = int(horizon)
    kind = horizon_kind(contract)
    if kind == "maze":
        native = contract.environment
        if horizon == int(native.size):
            return contract
        environment = replace(
            native,
            size=horizon,
            horizon=maze_budget(horizon, native),
            protocol_size=int(native.size),
        )
    else:
        environment = replace(contract.environment, meta_horizon=horizon)
    return replace(contract, environment=environment)


def streams_contract(contract: BenchmarkContract, streams: int) -> BenchmarkContract:
    """The evaluation-only contract of a CountRecall task continued over
    ``streams`` deck pairs (the continued pairs of C3): ``attempts`` names the
    pairs, as :func:`continued_streams` derives it."""
    environment = replace(contract.environment, attempts=int(streams))
    return replace(contract, environment=environment)


def streams_panel(
    layout: StudyLayout,
    environment: str,
    condition: str,
    seed: int,
    split: str,
    history: str,
    streams: int,
    *,
    contract: BenchmarkContract,
    sources: list[str] | None = None,
) -> list[BenchmarkEvent] | None:
    """One seed's frozen-endpoint panel of a CountRecall task continued over
    ``streams`` pairs (``<split>-<history>-endpoint-s<N>``), or ``None``: the
    panel must be complete, at the endpoint label, on the given split and
    history, with the run's recorded outer length equal to the continued
    task's."""
    extended = streams_contract(contract, streams)
    directory = layout.run_directory(environment, condition, seed) / "eval"
    path = (
        directory
        / f"{split}-{history}-endpoint-s{int(streams)}"
        / "benchmark_results.json"
    )
    if results_file(path) is None:
        return None
    raw = json.loads(read_results_text(path))
    if not isinstance(raw, dict) or "partial_task_cap" in raw:
        return None
    runs, events = read_benchmark_results(path, [extended])
    (run,) = runs
    if (
        run.status != "completed"
        or run.checkpoint != ENDPOINT_CHECKPOINT
        or run.split != split
        or run.history != history
        or run.outer_length != extended.environment.outer_length
    ):
        return None
    if sources is not None:
        sources.append(layout.relative(path))
    return list(events)


def load_streams_cell(
    layout: StudyLayout,
    environment: str,
    condition: str,
    split: str,
    history: str,
    streams: int,
    *,
    contract: BenchmarkContract,
    seeds: Sequence[int] = SEEDS,
    sources: list[str] | None = None,
) -> list[BenchmarkEvent] | None:
    """Every seed's continued-stream panel at ``streams`` pairs, or ``None``
    when any seed lacks one."""
    collected: list[BenchmarkEvent] = []
    staged: list[str] = []
    for seed in seeds:
        events = streams_panel(
            layout,
            environment,
            condition,
            seed,
            split,
            history,
            streams,
            contract=contract,
            sources=staged,
        )
        if events is None:
            return None
        collected.extend(events)
    if sources is not None:
        sources.extend(staged)
    return collected


def horizon_panel(
    layout: StudyLayout,
    environment: str,
    condition: str,
    seed: int,
    split: str,
    history: str,
    horizon: int,
    *,
    contract: BenchmarkContract,
    sources: list[str] | None = None,
) -> list[BenchmarkEvent] | None:
    """One seed's frozen-endpoint panel at an extended horizon
    (``<split>-<history>-endpoint-h<H>``), or ``None``: the panel must be
    complete, at the endpoint label, on the given split and history, with
    the run's recorded outer length equal to the horizon's, and carry no
    layout change."""
    extended = horizon_contract(contract, horizon)
    directory = layout.run_directory(environment, condition, seed) / "eval"
    path = (
        directory
        / f"{split}-{history}-endpoint-h{int(horizon)}"
        / "benchmark_results.json"
    )
    if results_file(path) is None:  # the plain file or its compacted twin
        return None
    raw = json.loads(read_results_text(path))
    if not isinstance(raw, dict) or "partial_task_cap" in raw:
        return None
    runs, events = read_benchmark_results(path, [extended])
    (run,) = runs
    if (
        run.status != "completed"
        or run.checkpoint != ENDPOINT_CHECKPOINT
        or run.split != split
        or run.history != history
        or run.outer_length != extended.environment.outer_length
        or run.layout_period is not None
    ):
        return None
    if sources is not None:
        sources.append(layout.relative(path))
    return list(events)


def laps_contract(contract: BenchmarkContract, budget: int) -> BenchmarkContract:
    """The evaluation-only contract of a MazeRunner repeated-laps panel: the
    trained task replayed on one map until ``budget`` charged calls, one
    attempt record per lap under complete retention, as
    :func:`~reasoned_icrl.experiments.horizon.continued_laps` derives it."""
    return replace(
        contract,
        environment=replace(contract.environment, meta_horizon=int(budget)),
        evaluation=replace(contract.evaluation, retention="complete"),
    )


def laps_panel(
    layout: StudyLayout,
    condition: str,
    seed: int,
    split: str,
    history: str,
    budget: int,
    *,
    contract: BenchmarkContract,
    sources: list[str] | None = None,
    layout_period: int | None = None,
) -> list[BenchmarkEvent] | None:
    """One seed's frozen-endpoint MazeRunner repeated-laps panel
    (``<split>-<history>-endpoint[-relayout<P>]-calls<B>``; ``layout_period``
    names the map-change continuation), or ``None`` unless it is
    complete, at the endpoint label, on the given split and history, with the
    run's outer length equal to the budget."""
    longer = laps_contract(contract, budget)
    directory = layout.run_directory("mazerunner", condition, seed) / "eval"
    path = (
        directory
        / (
            f"{split}-{history}-endpoint"
            + ("" if layout_period is None else f"-relayout{int(layout_period)}")
            + f"-calls{int(budget)}"
        )
        / "benchmark_results.json"
    )
    if results_file(path) is None:
        return None
    raw = json.loads(read_results_text(path))
    if not isinstance(raw, dict) or "partial_task_cap" in raw:
        return None
    runs, events = read_benchmark_results(path, [longer])
    (run,) = runs
    if (
        run.status != "completed"
        or run.checkpoint != ENDPOINT_CHECKPOINT
        or run.split != split
        or run.history != history
        or run.outer_length != int(budget)
        or run.layout_period != layout_period
    ):
        return None
    if sources is not None:
        sources.append(layout.relative(path))
    return list(events)


def load_cell(
    layout: StudyLayout,
    environment: str,
    condition: str,
    split: str,
    history: str,
    *,
    contract: BenchmarkContract,
    seeds: Sequence[int] = SEEDS,
    sources: list[str] | None = None,
) -> list[BenchmarkEvent] | None:
    """Every seed's endpoint panel, or ``None`` when any seed lacks one."""
    collected: list[BenchmarkEvent] = []
    staged: list[str] = []
    for seed in seeds:
        events = endpoint_panel(
            layout,
            environment,
            condition,
            seed,
            split,
            history,
            contract=contract,
            sources=staged,
        )
        if events is None:
            return None
        collected.extend(events)
    if sources is not None:
        sources.extend(staged)
    return collected


def secondary_panel(
    layout: StudyLayout,
    environment: str,
    condition: str,
    seed: int,
    split: str,
    history: str,
    *,
    sources: list[str] | None = None,
) -> Mapping[str, float] | None:
    """One seed's ``benchmark_secondary.json`` of an endpoint panel (the
    evaluator's per-panel summary: T-Maze success by cue side, junction
    arrival, penalised moves), or ``None`` when the panel has none."""
    directory = layout.run_directory(environment, condition, seed) / "eval"
    path = directory / f"{split}-{history}-endpoint" / "benchmark_secondary.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        return None
    if sources is not None:
        sources.append(layout.relative(path))
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


# ------------------------------------------------------------ write strata
# CountRecall's retention axis: every
# scored query is placed in a stratum by the number of C32 segments between the
# observation that first dealt the queried suit and the segment open at the
# query. The summary carriers record it on every query (`evidence_age_writes`,
# on the fixed lazy schedule: writes before queries 33, 65, 97); the stream and
# the schedule depend on the task alone, so the stratum of a query is borrowed
# from the reference cell's panel for the cells that record no writes (full
# history, GRU) and for Memo, whose own jittered boundaries are another clock.

StratumKey = tuple[int, int, int]
"""(task_id, rollout_seed, event_index) of one scored query."""


def write_strata(reference: Iterable[BenchmarkEvent]) -> dict[StratumKey, int | None]:
    """Stratum per query of the reference panel: ``evidence_age_writes``
    (``None`` for a suit never dealt before the query)."""
    strata: dict[StratumKey, int | None] = {}
    for event in reference:
        if event.kind != "query":
            continue
        strata[(event.task_id, event.rollout_seed, event.event_index)] = (
            event.evidence_age_writes
        )
    return strata


def stratify(
    events: Iterable[BenchmarkEvent], strata: Mapping[StratumKey, int | None]
) -> dict[int, list[BenchmarkEvent]]:
    """The queries of one panel grouped by borrowed stratum, in stratum order;
    a query the reference does not know is an error (different roster), a
    query in no stratum (suit never dealt) is dropped."""
    grouped: dict[int, list[BenchmarkEvent]] = {}
    for event in events:
        if event.kind != "query":
            continue
        key = (event.task_id, event.rollout_seed, event.event_index)
        if key not in strata:
            raise KeyError(f"Query {key} is not on the reference panel's roster.")
        stratum = strata[key]
        if stratum is None:
            continue
        grouped.setdefault(stratum, []).append(event)
    return dict(sorted(grouped.items()))


# -------------------------------------------------------------- live state
# Peak live inference state per actor over a task, from each cell's cost
# record (`systems.json`), as Table 2 and Figure 3's second row report it.
# One accounting for every cell: the most state the actor holds at any record of a task
# with
# ``records`` records. RSM's measured persistent state is its allocated
# buffer (the C-record segment, the M memory slots and the write slots),
# which it fills to capacity before every write, so its peak is that
# measurement at every horizon. Memo's recorded schedule oscillates (S summary
# slots replace L records at each write); its peak within the task is the
# record before the last write. The full history peaks at the last record.
# What a cell *allocates* for a declared horizon (Memo's capacity for the
# horizon, the full history's table) is reported beside the peak, never mixed
# with it.


def memo_live_slots(prefix: int, segment_length: int, summary_tokens: int) -> int:
    """Filled cache slots after ``prefix`` records under the lazy boundary:
    ``(prefix - 1) // L`` completed segments of ``S`` summary slots plus the
    open segment's records."""
    if prefix <= 0:
        return 0
    written = (prefix - 1) // segment_length
    return written * summary_tokens + prefix - written * segment_length


def memo_peak_slots(records: int, segment_length: int, summary_tokens: int) -> int:
    """The most filled slots at any record of a task of ``records`` records:
    the schedule peaks on the last record of a full segment (``L`` records
    plus every earlier summary), so the candidates are the segment ends and
    the task's last record."""
    if records <= 0:
        return 0
    ends = range(segment_length, records + 1, segment_length)
    return max(
        memo_live_slots(p, segment_length, summary_tokens) for p in (*ends, records)
    )


def live_state_bytes(
    systems: Mapping[str, Any],
    condition: str,
    records: int,
    *,
    segment_length: int,
    summary_tokens: int,
) -> tuple[int, str]:
    """Peak bytes of live inference state one actor holds over a task of
    ``records`` records, and how the number was obtained.

    A cell with a recorded ``state_growth`` schedule (Memo) follows the lazy
    boundary at the probe's bytes per slot, after the schedule is checked
    against every recorded prefix, and takes the peak of that schedule within
    the task. A cell whose state tensors include caches (the full history)
    holds one cache slot per record and peaks at the last record, projected
    linearly from the trained-length measurement. Every other cell holds its
    measured persistent state, which is its peak, whatever the horizon."""
    growth = systems.get("state_growth")
    if growth:
        recorded = {int(k): int(v) for k, v in growth["live_slots_by_prefix"].items()}
        for prefix, slots in recorded.items():
            if memo_live_slots(prefix, segment_length, summary_tokens) != slots:
                raise ValueError(
                    f"{condition}: the lazy-boundary schedule does not reproduce "
                    f"the recorded live slots at prefix {prefix}."
                )
        slots = memo_peak_slots(records, segment_length, summary_tokens)
        total = int(growth["bytes_per_slot"]) * slots + int(growth["counter_bytes"])
        return total, "peak of the recorded per-slot schedule within the task"
    tensors = systems["state_tensor_bytes"]
    caches = {k: v for k, v in tensors.items() if k.endswith("_cache")}
    if not caches:
        return int(systems["persistent_state_bytes"]), "measured (fixed state)"
    fixed = sum(tensors.values()) - sum(caches.values())
    native_records = int(systems["context_actions"]) + 1
    total = round(fixed + sum(caches.values()) * records / native_records)
    return total, "one cache slot per record, peak at the last record"


def mean_live_state_bytes(
    systems: Mapping[str, Any],
    condition: str,
    records: int,
    *,
    segment_length: int,
    summary_tokens: int,
) -> tuple[int, str]:
    """Bytes of live inference state one actor holds on average over the
    records of a task of ``records`` records, and how the number was obtained:
    the same three growth rules as :func:`live_state_bytes`, averaged over the
    task instead of taken at their peak (Memo the mean of its recorded per-slot
    schedule, the full history ``(records + 1) / 2`` cache slots, the bounded
    cells their fixed buffers)."""
    growth = systems.get("state_growth")
    if growth:
        recorded = {int(k): int(v) for k, v in growth["live_slots_by_prefix"].items()}
        for prefix, slots in recorded.items():
            if memo_live_slots(prefix, segment_length, summary_tokens) != slots:
                raise ValueError(
                    f"{condition}: the lazy-boundary schedule does not reproduce "
                    f"the recorded live slots at prefix {prefix}."
                )
        slots = sum(
            memo_live_slots(p, segment_length, summary_tokens)
            for p in range(1, records + 1)
        ) / max(records, 1)
        total = round(
            int(growth["bytes_per_slot"]) * slots + int(growth["counter_bytes"])
        )
        return total, "mean of the recorded per-slot schedule over the task"
    tensors = systems["state_tensor_bytes"]
    caches = {k: v for k, v in tensors.items() if k.endswith("_cache")}
    if not caches:
        return int(systems["persistent_state_bytes"]), "measured (fixed state)"
    fixed = sum(tensors.values()) - sum(caches.values())
    native_records = int(systems["context_actions"]) + 1
    mean_slots = (records + 1) / 2
    total = round(fixed + sum(caches.values()) * mean_slots / native_records)
    return total, "one cache slot per record, averaged over the task"


def allocated_state_bytes(systems: Mapping[str, Any]) -> tuple[int, str]:
    """Bytes the cell allocated for its trained horizon (the cost probe's
    measured persistent state): Memo's capacity for that horizon, the full
    history's table, and the fixed buffers of the bounded cells."""
    growth = systems.get("state_growth")
    if growth and "capacity_slots" in growth:
        total = int(growth["bytes_per_slot"]) * int(growth["capacity_slots"]) + int(
            growth["counter_bytes"]
        )
        return total, "capacity allocated for the trained horizon"
    return int(systems["persistent_state_bytes"]), "measured persistent state"


def cell_state_rows(
    layout: StudyLayout,
    environment: str,
    conditions: Iterable[str],
    horizon: int,
    *,
    contract: BenchmarkContract,
    seeds: Sequence[int] = SEEDS,
    sources: list[str] | None = None,
    accounting: Literal["peak", "mean"] = "peak",
    records: int | None = None,
) -> list[dict[str, object]]:
    """One live-state row per condition for a task of ``horizon`` (``records``
    overrides the environment's record count, as the MazeRunner laps axis
    needs: its ``horizon`` is a budget of calls, not a maze size; mean over
    the seeds whose ``systems.json`` exists): ``bytes`` under the requested
    accounting (the peak an actor holds within the task, or the mean over its
    records), ``peak_bytes`` and ``mean_bytes`` both, the bytes allocated for
    the trained horizon beside them, with the accounting rule printed."""
    spec = layout.spec(environment)
    summary = contract.memory["summary"]
    if records is None:
        records = spec.records_at(horizon, contract)
    rows: list[dict[str, object]] = []
    for condition in conditions:
        peaks, means, methods, allocated = [], [], set(), []
        for seed in seeds:
            path = layout.run_directory(environment, condition, seed) / "systems.json"
            if not path.is_file():
                continue
            systems = json.loads(path.read_text())
            growth = {
                "segment_length": int(summary["segment_length"]),
                "summary_tokens": int(summary["memory_tokens"]),
            }
            peak, peak_method = live_state_bytes(systems, condition, records, **growth)
            mean, mean_method = mean_live_state_bytes(
                systems, condition, records, **growth
            )
            peaks.append(peak)
            means.append(mean)
            methods.add(peak_method if accounting == "peak" else mean_method)
            allocated.append(allocated_state_bytes(systems)[0])
            if sources is not None:
                sources.append(layout.relative(path))
        if peaks:
            chosen = peaks if accounting == "peak" else means
            rows.append(
                {
                    "environment": environment,
                    "condition": condition,
                    "horizon": horizon,
                    "records": records,
                    "accounting": accounting,
                    "bytes": round(sum(chosen) / len(chosen)),
                    "peak_bytes": round(sum(peaks) / len(peaks)),
                    "mean_bytes": round(sum(means) / len(means)),
                    "allocated_bytes": round(sum(allocated) / len(allocated)),
                    "seeds_measured": len(peaks),
                    "method": "; ".join(sorted(methods)),
                }
            )
    return rows


@dataclass(frozen=True, slots=True)
class RosterSelection:
    """Which split a column is drawn on and which cells have records there."""

    environment: str
    split: str | None
    paper_split: str
    panels: Mapping[str, list[BenchmarkEvent] | None]

    @property
    def provisional(self) -> bool:
        return self.split != self.paper_split

    @property
    def present(self) -> tuple[str, ...]:
        return tuple(c for c, rows in self.panels.items() if rows is not None)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(c for c, rows in self.panels.items() if rows is None)

    @property
    def tasks(self) -> int:
        units = {
            (e.task_id, e.rollout_seed)
            for rows in self.panels.values()
            for e in rows or ()
        }
        return len(units)

    def title(self, name: str) -> str:
        """The panel title: the environment's name alone on the paper roster;
        the roster named when the column is provisional. Cells without
        records are named in the legend (:func:`pending_handles`), never on
        the panel."""
        if self.split is not None and self.provisional:
            return f"{name} ({self.split} roster, provisional)"
        return name

    def notes(self) -> dict[str, object]:
        return {
            "split": self.split,
            "provisional": self.provisional,
            "present": list(self.present),
            "missing": list(self.missing),
            "tasks": self.tasks,
        }


def choose_split(
    layout: StudyLayout,
    environment: str,
    conditions: Sequence[str],
    *,
    contract: BenchmarkContract,
    preference: Sequence[str] = ("confirmation", "development"),
    history: str = "retained",
    seeds: Sequence[int] = SEEDS,
    sources: list[str] | None = None,
    required: Sequence[str] | None = None,
) -> RosterSelection:
    """The first split in ``preference`` on which every required cell has
    complete records; failing that, the split on which the most required cells
    have them (ties go to the earlier preference), so every panel shows the
    competing methods together on one roster. The required cells are the six
    cells among ``conditions`` (all of them when none is one): a
    cell still being evaluated, the method included until its panels land, is
    drawn where it has records on the chosen split and listed as pending
    otherwise, and never moves the frozen cells off their roster. A column
    drawn on a split other than the first preference is provisional."""
    if required is None:
        required = [c for c in conditions if c in PAPER_CONDITIONS] or list(conditions)
    candidates: list[
        tuple[int, str, dict[str, list[BenchmarkEvent] | None], list[str]]
    ] = []
    for split in preference:
        staged: list[str] = []
        panels = {
            condition: load_cell(
                layout,
                environment,
                condition,
                split,
                history,
                contract=contract,
                seeds=seeds,
                sources=staged,
            )
            for condition in conditions
        }
        present = sum(panels[c] is not None for c in required)
        if present == len(required):
            if sources is not None:
                sources.extend(staged)
            return RosterSelection(environment, split, preference[0], panels)
        candidates.append((present, split, panels, staged))
    present, split, panels, staged = max(candidates, key=lambda c: c[0])
    if present == 0:
        return RosterSelection(
            environment, None, preference[0], dict.fromkeys(conditions)
        )
    if sources is not None:
        sources.extend(staged)
    return RosterSelection(environment, split, preference[0], panels)


# ------------------------------------------------------------------- style

METHOD_COLOUR = "#D55E00"
"""Vermillion: the paper's method, RSM-O."""
ABLATION_COLOUR = "#E69F00"
"""Amber: the rewrite ablation, RSM-R — the same family, told apart at a glance."""
CONTROL_COLOUR = "#7F7F7F"
"""Grey: the control that carries nothing between segments."""

INK = "#1a1a1a"
INK_MUTED = "#5a5a5a"
INK_FAINT = "#8c8c8c"
GRID = "#e6e6e6"
BAND_ALPHA = 0.12
"""Fill opacity of a 95 % interval band."""

LineStyle = str | tuple[float, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class CellStyle:
    """How one condition is drawn everywhere (figure plan, palette table)."""

    condition: str
    label: str
    color: str
    linestyle: LineStyle = "-"
    linewidth: float = 1.0
    marker: str = "o"
    open_marker: bool = False
    zorder: int = 1

    @property
    def marker_face(self) -> str:
        return "white" if self.open_marker else self.color

    def line(self, **overrides: Any) -> dict[str, Any]:
        """Keyword arguments for ``Axes.plot`` of this cell's curve."""
        base: dict[str, Any] = {
            "color": self.color,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "zorder": self.zorder + 10,
        }
        base.update(overrides)
        return base

    def markers(self, size: float = 3.2, **overrides: Any) -> dict[str, Any]:
        """Keyword arguments for a marked curve; ``overrides`` win."""
        base = self.line(
            marker=self.marker,
            markersize=size,
            markeredgewidth=0.7,
            markerfacecolor=self.marker_face,
        )
        base.update(overrides)
        return base

    def handle(self) -> Line2D:
        return Line2D([], [], label=self.label, **self.markers(4.0))


METHOD = CellStyle(
    METHOD_CONDITION, "RSM-O (ours)", METHOD_COLOUR, "-", 1.4, "o", zorder=6
)
"""RSM-O, the replacing write and the paper's method: vermillion, solid, the thickest
line."""
ABLATION = CellStyle(
    ABLATION_CONDITION,
    "RSM-R (residual ablation)",
    ABLATION_COLOUR,
    (0, (3, 1, 1, 1)),
    1.2,
    "s",
    zorder=5,
)
"""RSM-R, the residual write and the method's rewrite ablation: amber,
dash-dotted, square markers — the same family, told apart at a glance."""

PAPER_CELLS: tuple[CellStyle, ...] = (
    METHOD,
    CellStyle(
        "raw_segment", "w/o memory", CONTROL_COLOUR, (0, (4, 2)), 1.0, "o", True, 3
    ),
    CellStyle(
        "full_context", "Full-history Transformer", "#000000", "-", 1.0, "^", zorder=5
    ),
    CellStyle("full_gru", "GRU", "#009E73", "-", 1.0, "D", zorder=2),
    CellStyle("memo", "Memo (accumulating)", "#CC79A7", "-", 1.0, "p", zorder=4),
    CellStyle(
        "memo_fixed",
        "Memo, fixed segments",
        "#CC79A7",
        (0, (5, 1.5, 1, 1.5)),
        1.0,
        "p",
        True,
        1,
    ),
)
"""The six cells declared, in drawing and legend order,
the method (RSM-O, the cell those declarations named RSM) first; the carry
control, which keeps nothing between segments, is grey (*w/o memory*)."""

RESIDUAL = ABLATION

DRAWN_CELLS: tuple[CellStyle, ...] = (METHOD, ABLATION, *PAPER_CELLS[1:])
"""Every cell a figure draws where its records exist, in legend order."""

MAIN_CONDITIONS: tuple[str, ...] = (
    METHOD_CONDITION,
    ABLATION_CONDITION,
    "raw_segment",
    "full_context",
    "full_gru",
    "memo",
)
"""The cells of the main text: the RSM family,
its carry control and the published comparators. Memo fixed, the segmentation
control, stays in every contrast family, report and record and is drawn and
tabulated in the appendix only."""
APPENDIX_ONLY_CONDITIONS: tuple[str, ...] = ("memo_fixed",)
MAIN_CELLS: tuple[CellStyle, ...] = tuple(
    c for c in DRAWN_CELLS if c.condition in MAIN_CONDITIONS
)
"""``DRAWN_CELLS`` restricted to the main text's cells, in the same order."""

SUMMARY_CLEARED = CellStyle(
    f"{METHOD_CONDITION}/summary-cleared",
    "RSM, summary cleared",
    METHOD_COLOUR,
    (0, (1, 1.5)),
    0.9,
    "x",
    zorder=20,
)
"""The intervention trace: the method's colour, dotted, with x markers."""


def select_cells(conditions: Sequence[str] | None = None) -> tuple[CellStyle, ...]:
    """The drawn cells (the method and the six cells),
    optionally restricted, always in the fixed order."""
    if conditions is None:
        return DRAWN_CELLS
    unknown = sorted(set(conditions) - {c.condition for c in DRAWN_CELLS})
    if unknown:
        raise KeyError(f"Not drawn cells: {unknown}")
    return tuple(c for c in DRAWN_CELLS if c.condition in conditions)


def style_of(condition: str) -> CellStyle:
    """The drawn style of one condition."""
    for cell in DRAWN_CELLS:
        if cell.condition == condition:
            return cell
    raise KeyError(f"Not a drawn cell: {condition}")


FULL_WIDTH = 11.0
"""Width in inches of a full-width composite (larger
figures, thinner lines). Every panel is drawn at its natural size and the
figure is scaled down to the column width at typesetting time; the type is set
so that it stays legible after that reduction."""
PANEL = (4.6, 3.4)
"""Default size of a standalone panel, in the same proportions."""
SMALL = 8.5
"""Point size of the in-panel annotations (end labels, thresholds, notes)."""


def paper_rc() -> dict[str, Any]:
    """Matplotlib settings for a figure drawn at :data:`FULL_WIDTH`: 10-point
    type, hairline axes, round joins so a polyline reads as a smooth curve."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        # Round joins and caps: at these line widths a corner between two
        # samples is what makes a curve look broken rather than drawn.
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.dash_joinstyle": "round",
        "lines.antialiased": True,
        "patch.antialiased": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    }


@contextmanager
def paper_style() -> Iterator[None]:
    """Apply :func:`paper_rc` for the duration of a ``with`` block."""
    with plt.rc_context(paper_rc()):
        yield


def style_axes(ax: Axes, *, grid_axis: Literal["both", "x", "y"] = "y") -> None:
    """Recessive grid on one axis, no top or right spine."""
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend_handles(cells: Iterable[CellStyle]) -> list[Line2D]:
    return [cell.handle() for cell in cells]


def line_handles(cells: Iterable[CellStyle]) -> list[Line2D]:
    """Legend entries as bare lines, for a figure whose curves carry no
    markers: the legend shows what the panel draws."""
    return [Line2D([], [], label=cell.label, **cell.line()) for cell in cells]


def band(
    ax: Axes,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    cell: CellStyle,
    *,
    alpha: float = BAND_ALPHA,
) -> None:
    """A 95 % interval band in the cell's colour; ``alpha`` lightens it where
    many bands overlap on one panel."""
    ax.fill_between(
        x,
        lower,
        upper,
        color=cell.color,
        alpha=alpha,
        linewidth=0,
        zorder=cell.zorder,
    )


def pending_handles(
    pending: Mapping[str, Sequence[CellStyle]],
    cells: Sequence[CellStyle] | None = None,
    *,
    reason: str = "pending",
) -> list[Line2D]:
    """Legend entries naming the cells without records, one per panel title
    (a missing cell is mentioned in the legend and
    never written on the chart). ``pending`` maps a panel's title to the
    cells missing there; when every cell of ``cells`` is missing the entry
    says so instead of listing them."""
    handles: list[Line2D] = []
    for title, missing in pending.items():
        if not missing:
            continue
        names = [c.label for c in missing]
        if cells is not None and {c.condition for c in missing} >= {
            c.condition for c in cells
        }:
            text = f"{title}: every cell {reason}"
        else:
            text = f"{title}: {', '.join(names)} {reason}"
        handles.append(
            Line2D([], [], linestyle="none", marker=None, color="none", label=text)
        )
    return handles


def composite_legend(
    fig: Figure,
    handles: Sequence[Line2D],
    *,
    ncol: int | None = None,
    title: str | None = None,
) -> None:
    """One shared legend above a constrained-layout composite, in the figure's
    margin so it never covers a panel. Its title line carries ``title`` (what
    the figure as a whole shows, when its panels do not each name it) and then
    what is missing, which is never written on the chart. A figure-level title and an
    outside legend would otherwise compete
    for the same margin."""
    drawn = [h for h in handles if h.get_color() != "none"]
    notes = [str(h.get_label()) for h in handles if h.get_color() == "none"]
    if title:
        notes = [title, *notes]
    columns = ncol or min(len(drawn), 4) or 1
    legend = fig.legend(
        handles=drawn,
        loc="outside upper center",
        ncol=columns,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.4,
        handletextpad=0.5,
        borderaxespad=0.2,
        title="\n".join(notes) or None,
        alignment="center",
    )
    if notes:
        legend.get_title().set_fontsize(11 if title else 9)
        legend.get_title().set_color(INK if title else INK_MUTED)
        if title:
            legend.get_title().set_fontweight("bold")


# ------------------------------------------------------------------ export

Row = dict[str, object]


@dataclass
class BuiltFigure:
    """A rendered figure, the plotted data behind each panel, and what was read."""

    name: str
    figure: Figure
    tables: dict[str, list[Row]] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    panels: dict[str, Figure] = field(default_factory=dict)
    """Each panel of the composite as its own figure with its own legend."""

    def save(
        self,
        figures_dir: Path,
        data_dir: Path | None = None,
        *,
        formats: Sequence[str] = ("pdf", "png"),
        panel_formats: Sequence[str] = ("png",),
    ) -> list[Path]:
        """The composite as vector PDF and 300-dpi PNG and every standalone
        panel as PNG under ``figures_dir``; one CSV per table and the notes
        (with every record path read) under ``data_dir`` (default: the same)."""
        data_dir = data_dir if data_dir is not None else figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for suffix in formats:
            path = figures_dir / f"{self.name}.{suffix}"
            self.figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
            written.append(path)
        for suffix in panel_formats:
            for panel, figure in self.panels.items():
                path = figures_dir / f"{self.name}_panel_{panel}.{suffix}"
                figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
                written.append(path)
        for table, rows in self.tables.items():
            path = data_dir / f"{self.name}_{table}.csv"
            write_csv(path, rows)
            written.append(path)
        path = data_dir / f"{self.name}_notes.json"
        payload = {"notes": self.notes, "sources": sorted(set(self.sources))}
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        written.append(path)
        return written

    def close(self) -> None:
        """Release the composite and every standalone panel."""
        plt.close(self.figure)
        for figure in self.panels.values():
            plt.close(figure)


def standalone(
    draw: Callable[[Axes], None],
    *,
    size: tuple[float, float] = PANEL,
    handles: Sequence[Line2D] | None = None,
) -> Figure:
    """One panel on its own figure: ``draw`` receives the axes; the legend of
    the drawn cells is placed below the axes so it never covers a curve (a
    composite carries one shared legend, a standalone panel its own)."""
    with paper_style():
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        if handles:
            # Anchor the legend a fixed distance in inches under everything the
            # axes draw (two-line tick labels and the axis label included), so
            # the gap survives tight_layout shrinking the axes to fit it.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            below = ax.get_window_extent(renderer).y0 - ax.get_tightbbox(renderer).y0
            drop = ScaledTranslation(
                0.0, -(below / fig.dpi + 0.08), fig.dpi_scale_trans
            )
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                bbox_transform=ax.transAxes + drop,
                ncol=2 if len(handles) > 3 else len(handles),
                frameon=False,
                handlelength=2.2,
                columnspacing=1.2,
            )
        fig.tight_layout(pad=0.3)
    return fig


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """One CSV whose header is the union of every row's keys, in first-seen order."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or ["status"])
        writer.writeheader()
        writer.writerows(rows if rows else [{"status": "empty"}])


def markdown_preview(rows: Sequence[Mapping[str, object]], limit: int = 12) -> str:
    """The first rows of a table as Markdown, for a notebook cell."""
    if not rows:
        return "_empty_"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows[:limit]:
        cells = [
            f"{v:.4g}" if isinstance(v := row.get(c, ""), float) else str(v)
            for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        lines.append(f"| … {len(rows) - limit} more rows |")
    return "\n".join(lines)
