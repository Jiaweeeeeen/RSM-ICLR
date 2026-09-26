"""Length extrapolation on Key-to-Door: the extended-horizon evaluation adapter.

EXPERIMENTS section 5 and the study protocol evaluate
the frozen 8M endpoints of the bounded-summary paper's six cells at H = 500,
1,000 and 2,000 charged calls (4,000 optional) *in the same task*: the same
start, key and door for the whole outer task, attempts still ending on a door
or after ``horizon`` physical actions, prefix-consistent environment
randomness, every attempt counted. Nothing here trains, relabels or changes a
saved identity; the adapter derives an evaluation-only contract and config
whose outer budget is ``horizon`` and whose carrier is allocated for it.

The full-history carrier receives the complete prefix with AMAGO's sinusoidal
position table (no parameters) and a cache sized for the horizon; the Memo
carrier's slot allocation follows its ``max_index`` and grows with the declared
horizon; the summary carrier and the GRU change nothing. Panels are written
beside the trained-horizon panels under ``<split>-<history>-endpoint-h<H>/``
with ``outer_length`` on every run and event record, and the adapter's
acceptance check is that the first ``native`` calls of every longer panel
reproduce the trained-horizon panel exactly (:func:`check_horizon_prefix`).

CountRecall extends the stream (and, on the continued-stream axis, the
number of deck pairs) and MazeRunner the maze size
(func:`extended_horizon`) or, on the repeated-laps axis,
the number of charged calls one map is replayed for (:func:`continued_laps`):
the same trained task, lap after lap, so that only the memory has to last
longer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.records import BenchmarkEvent

HORIZON_ENVIRONMENTS = ("dark_key_to_door", "count_recall", "mazerunner")
"""Environments whose outer budget can be extended without changing the task
family: the native Key-to-Door meta task keeps one layout for the whole
budget; CountRecall keeps its dealt deck and continues with query-only
records (the third axis);
MazeRunner keeps its trained protocol, goals and packet and plays a larger
maze under the area-scaled step budget (the fourth axis, EXPERIMENTS section 5)."""

DECLARED_HORIZONS = (500, 1000, 2000)
"""The paper's declared Key-to-Door horizons (the trained 500-call budget, 2x
and 4x)."""

STREAM_HORIZONS = (104, 208, 416, 832)
"""The paper's declared CountRecall stream lengths in records: the trained
104-record stream (three lazy C32 writes), then 6, 12 and 24 writes with a
query-only tail. The ``horizon`` argument names records on this benchmark."""

MAZE_LADDER_OFFSETS = (0, 2, 4, 6, 10)
"""The declared ladder of a MazeRunner study as offsets from its trained
maze size (before any MazeRunner endpoint existed): the
trained size and four larger odd sizes. The ``horizon`` argument names the
maze size on this benchmark; the step budget follows the area
(:func:`maze_timer`), so the budget per cell of the maze is the trained one.
The development roster is read first; the confirmation roster is read at
every declared size whatever it showed."""


def maze_size_ladder(trained_size: int) -> tuple[int, ...]:
    """The declared maze sizes of a study trained at ``trained_size``: 15 ->
    15, 17, 19, 21, 25 (the paper's protocol); the 11x11 fallback -> 11, 13,
    15, 17, 21."""
    if trained_size < 7 or trained_size % 2 != 1:
        raise ContractError(
            "A MazeRunner ladder starts from an odd maze of at least 7."
        )
    return tuple(trained_size + offset for offset in MAZE_LADDER_OFFSETS)


MAZE_SIZES = maze_size_ladder(15)
"""The paper's declared ladder for the 15x15 protocol."""


def maze_timer(size: int, *, trained_size: int, trained_horizon: int) -> int:
    """The larger maze's step budget: the trained budget scaled by the area
    ratio, rounded to the nearest whole step (500 at 15; 642, 802, 980 and
    1,389 at 17, 19, 21 and 25)."""
    if size < trained_size or trained_size < 1 or trained_horizon < 1:
        raise ContractError("The maze budget scales a trained budget up, never down.")
    return round(trained_horizon * size * size / (trained_size * trained_size))


WINDOW = 500
"""The window of the pre-declared Key-to-Door estimand: completed doors per
500 calls."""


def horizon_kind(contract: BenchmarkContract) -> str:
    """What the ``horizon`` argument means for this contract: the outer
    budget in charged calls (Key-to-Door), the stream length in records
    (CountRecall) or the maze size (MazeRunner)."""
    name = contract.environment.name
    if name not in HORIZON_ENVIRONMENTS:
        raise ContractError(
            f"{name!r} has no extended-horizon adapter; only "
            f"{', '.join(HORIZON_ENVIRONMENTS)} keep one task over a longer budget."
        )
    if name == "mazerunner":
        return "maze"
    return "stream" if name == "count_recall" else "calls"


def native_horizon(contract: BenchmarkContract) -> int:
    """The trained value of the ``horizon`` argument for this contract."""
    kind = horizon_kind(contract)
    if kind == "maze":
        return int(contract.environment.size)
    if kind == "stream":
        # Records: the scored decisions plus the reset observation.
        return int(contract.environment.outer_length) + 1
    return int(contract.environment.outer_length)


def extended_horizon(
    contract: BenchmarkContract, config: ExperimentConfig, horizon: int
) -> tuple[BenchmarkContract, ExperimentConfig]:
    """Derive the evaluation-only contract and config for ``horizon``.

    On Key-to-Door ``horizon`` is the outer budget in charged calls and the
    environment's meta budget becomes it; on CountRecall ``horizon`` is the
    stream length in records, the decisions are one fewer and every record
    past the native deck is a query-only tail record; on MazeRunner
    ``horizon`` is the maze size, the trained protocol is kept through
    ``protocol_size``, the budget is the area-scaled timer and the trained
    size returns the contract unchanged. The carrier's context
    (``max_sequence_length``) and the replay file length grow with the
    outer length so the complete prefix fits, exactly as the contract loader
    requires of the trained horizon. Every other field, including the model
    identity, the training recipe and the rosters, is untouched.
    """
    if isinstance(horizon, bool) or int(horizon) < 1:
        raise ContractError("The evaluation horizon must be a positive integer.")
    horizon = int(horizon)
    environment = contract.environment
    kind = horizon_kind(contract)
    native = native_horizon(contract)
    if horizon < native:
        raise ContractError(
            f"The evaluation horizon {horizon} is below the trained budget {native}."
        )
    if config.environment != environment:
        raise ContractError("The resolved config's environment is not the contract's.")
    if kind == "stream":
        extended = replace(environment, horizon=horizon - 1)
    elif kind == "maze":
        if (
            environment.protocol_size is not None
            or environment.meta_horizon is not None
        ):
            raise ContractError(
                "The larger-maze adapter starts from the trained contract."
            )
        if horizon % 2 != 1:
            raise ContractError(
                f"A MazeRunner maze is odd; {horizon} is not (the native task would "
                "round it up)."
            )
        extended = (
            environment
            if horizon == native
            else replace(
                environment,
                size=horizon,
                horizon=maze_timer(
                    horizon, trained_size=native, trained_horizon=environment.horizon
                ),
                protocol_size=native,
            )
        )
    else:
        if environment.meta_horizon is None:
            raise ContractError("The extended horizon needs a native meta budget.")
        extended = replace(environment, meta_horizon=horizon)
    outer = extended.outer_length
    training = replace(
        config.training,
        max_sequence_length=max(config.training.max_sequence_length, outer),
        trajectory_length=max(config.training.trajectory_length, outer + 1),
    )
    return (
        replace(contract, environment=extended, training=training),
        replace(config, environment=extended, training=training),
    )


STREAM_COUNTS = (2, 4, 8)
"""Deck pairs per outer task on the continued-stream CountRecall axis (C3): 208, 416 and
832 records."""


def continued_streams(
    contract: BenchmarkContract, config: ExperimentConfig, streams: int
) -> tuple[BenchmarkContract, ExperimentConfig]:
    """Derive the evaluation-only contract and config for a CountRecall task
    continued over ``streams`` consecutive deck pairs: stream 1 is the trained
    stream, every later pair is dealt from its own seed after a reset-only
    call, counts and the native timer restart, RL2 and the carried memory
    continue. The outer length is ``streams * 104 - 1`` charged calls; the
    carrier's context and the replay length grow with it as for
    :func:`extended_horizon`. Everything else is untouched."""
    if isinstance(streams, bool) or int(streams) < 2:
        raise ContractError("A continued task needs at least two streams.")
    if horizon_kind(contract) != "stream":
        raise ContractError("Only CountRecall continues over deck pairs.")
    environment = contract.environment
    from reasoned_icrl.environments.count_recall import (
        COUNT_RECALL_HORIZONS,
        count_recall_variant,
    )

    deck = COUNT_RECALL_HORIZONS[count_recall_variant(environment.benchmark)]
    if environment.attempts != 1 or environment.horizon != deck:
        raise ContractError("The continued task starts from the trained stream.")
    if config.environment != environment:
        raise ContractError("The resolved config's environment is not the contract's.")
    # The continued task is the plain roster task over ``streams`` pairs; its
    # outer length follows ``attempts``.
    extended = replace(environment, attempts=int(streams))
    outer = extended.outer_length
    training = replace(
        config.training,
        max_sequence_length=max(config.training.max_sequence_length, outer),
        trajectory_length=max(config.training.trajectory_length, outer + 1),
    )
    return (
        replace(contract, environment=extended, training=training),
        replace(config, environment=extended, training=training),
    )


LAP_BUDGETS = (1000, 2000, 4000)
"""Charged calls per outer task on the MazeRunner repeated-laps axis: two,
four and eight times the trained 500-call episode."""


def continued_laps(
    contract: BenchmarkContract, config: ExperimentConfig, budget: int
) -> tuple[BenchmarkContract, ExperimentConfig]:
    """Derive the evaluation-only contract and config for a MazeRunner map
    replayed in laps until ``budget`` charged calls: every lap is one native
    episode of the trained task on the map's own seed (same maze, goals and
    hidden action permutation, the agent at the start, the native timer per
    lap), a reset-only call separates consecutive laps and the carried memory
    continues across it; the last lap is cut by the budget and marked partial.
    The contract scores one attempt record per lap under complete retention;
    the carrier's context and the replay length grow with the budget as for
    :func:`extended_horizon`. Everything else is untouched."""
    if isinstance(budget, bool) or int(budget) < 1:
        raise ContractError("The repeated-laps budget is a positive number of calls.")
    if contract.environment.name != "mazerunner":
        raise ContractError("Only MazeRunner replays a map in laps.")
    environment = contract.environment
    if environment.protocol_size is not None or environment.meta_horizon is not None:
        raise ContractError(
            "The repeated-laps adapter starts from the trained one-episode task on "
            "the trained maze."
        )
    if int(budget) <= environment.horizon:
        raise ContractError(
            f"The repeated-laps budget {budget} does not exceed the trained "
            f"episode of {environment.horizon} calls."
        )
    if config.environment != environment:
        raise ContractError("The resolved config's environment is not the contract's.")
    extended = replace(environment, meta_horizon=int(budget))
    outer = extended.outer_length
    training = replace(
        config.training,
        max_sequence_length=max(config.training.max_sequence_length, outer),
        trajectory_length=max(config.training.trajectory_length, outer + 1),
    )
    evaluation = replace(contract.evaluation, retention="complete")
    return (
        replace(
            contract, environment=extended, evaluation=evaluation, training=training
        ),
        replace(config, environment=extended, training=training),
    )


def _lap_signature(event: BenchmarkEvent) -> tuple[object, ...]:
    return (event.numerator, event.step, round(float(event.native_return), 9))


def check_laps_prefix(
    base: Sequence[BenchmarkEvent], laps: Sequence[BenchmarkEvent]
) -> dict[str, int]:
    """The repeated-laps adapter's acceptance check: on every map and rollout
    seed the first lap of the laps panel reproduces the plain one-episode panel
    (same goals reached, same calls taken, same return) and starts at call 1.
    Returns the counts compared; raises :class:`ContractError` on the first
    disagreement."""
    base_by_key = {
        (int(e.task_id), int(e.rollout_seed)): e for e in base if e.kind == "episode"
    }
    first_laps = {
        (int(e.task_id), int(e.rollout_seed)): e
        for e in laps
        if e.kind == "attempt" and int(e.event_index) == 1
    }
    if not base_by_key or not first_laps:
        raise ContractError(
            "The laps check needs the plain episode panel and lap records."
        )
    if set(base_by_key) != set(first_laps):
        raise ContractError("The two panels cover different maps or rollout seeds.")
    for key, event in base_by_key.items():
        lap = first_laps[key]
        if _lap_signature(lap) != _lap_signature(event) or (
            lap.start_step,
            lap.end_step,
        ) != (1, event.step):
            raise ContractError(
                f"Map {key[0]}: the first lap differs from the plain episode: base "
                f"{_lap_signature(event)}, lap {_lap_signature(lap)} spanning "
                f"{lap.start_step}-{lap.end_step}."
            )
    return {"compared": len(base_by_key), "units": len(base_by_key)}


def window_goal_counts(
    events: Sequence[BenchmarkEvent], *, outer_length: int, window: int = WINDOW
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Goals reached per fixed window of calls, per (task, rollout seed), on
    the repeated-laps axis: every goal of every lap counts in the window
    holding the call that reached it (``goal_steps``), the partial lap
    included, because a goal reached is reached. A lap record without goal
    steps (none is written without them) dates its goals at the lap's end."""
    if outer_length % window:
        raise ContractError("The outer budget must be a whole number of windows.")
    windows = outer_length // window
    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0] * windows)
    for event in events:
        if event.kind != "attempt" or not event.numerator:
            continue
        steps: tuple[int, ...]
        if event.goal_steps is not None:
            steps = tuple(int(step) for step in event.goal_steps)
        elif event.end_step is not None:
            steps = (int(event.end_step),) * int(event.numerator)
        else:
            raise ContractError("A lap record dates its goals or its end.")
        unit = (int(event.task_id), int(event.rollout_seed))
        for step in steps:
            index = (step - 1) // window
            if not 0 <= index < windows:
                raise ContractError(
                    f"Goal call {step} lies outside the {outer_length}-call budget."
                )
            counts[unit][index] += 1
    return {unit: tuple(values) for unit, values in counts.items()}


def window_goal_matrix(
    events: Sequence[BenchmarkEvent],
    *,
    units: Sequence[tuple[int, int]],
    outer_length: int,
    window: int = WINDOW,
) -> np.ndarray:
    """Goals per window as a ``[len(units), windows]`` array in the given
    (task, rollout seed) order, zero where a unit reached no goal: the paired
    form of :func:`window_goal_counts`, so panels of different cells subtract
    unit by unit."""
    counts = window_goal_counts(events, outer_length=outer_length, window=window)
    index = {
        (int(task), int(rollout)): row for row, (task, rollout) in enumerate(units)
    }
    if len(index) != len(units):
        raise ContractError("The roster units must be distinct.")
    unknown = set(counts) - set(index)
    if unknown:
        raise ContractError(
            f"Events for units outside the roster: {sorted(unknown)[:3]}."
        )
    matrix = np.zeros((len(units), outer_length // window), dtype=np.float64)
    for unit, values in counts.items():
        matrix[index[unit]] = values
    return matrix


def laps_per_unit(events: Sequence[BenchmarkEvent]) -> dict[tuple[int, int], int]:
    """Finished laps per (task, rollout seed) on a laps panel."""
    finished: dict[tuple[int, int], int] = defaultdict(int)
    for event in events:
        if event.kind == "attempt" and event.complete:
            finished[(int(event.task_id), int(event.rollout_seed))] += 1
    return dict(finished)


def streams_matrix(
    events: Sequence[BenchmarkEvent],
    *,
    units: Sequence[tuple[int, int]],
    decisions: int,
    streams: int,
) -> np.ndarray:
    """``[units, streams]`` exact accuracy per deck pair for every (task,
    rollout seed) unit: decision ``i`` (1-based over the outer task) belongs
    to stream ``(i - 1) // decisions``. Every unit must hold exactly
    ``decisions`` scored decisions per stream."""
    if len(set(units)) != len(units):
        raise ContractError("Units must be distinct.")
    index = {unit: row for row, unit in enumerate(units)}
    correct = np.zeros((len(units), streams), dtype=np.float64)
    counted = np.zeros((len(units), streams), dtype=np.int64)
    for event in events:
        if event.kind != "query":
            continue
        key = (int(event.task_id), int(event.rollout_seed))
        if key not in index:
            raise ContractError(f"Unit {key} lies outside the roster.")
        stream = (int(event.step) - 1) // decisions
        if not 0 <= stream < streams:
            raise ContractError("A decision lies outside the declared streams.")
        correct[index[key], stream] += float(event.numerator)
        counted[index[key], stream] += 1
    if (counted != decisions).any():
        raise ContractError("Every unit needs exactly the stream's decisions per pair.")
    return correct / decisions


def streams_accuracy(
    events: Sequence[BenchmarkEvent], *, decisions: int, streams: int
) -> list[float]:
    """Mean exact accuracy per deck pair over every scored unit of the panel."""
    units = sorted(
        {(int(e.task_id), int(e.rollout_seed)) for e in events if e.kind == "query"}
    )
    if not units:
        raise ContractError("The stream estimand needs query events.")
    matrix = streams_matrix(events, units=units, decisions=decisions, streams=streams)
    return [float(value) for value in matrix.mean(axis=0)]


def _episode_signature(event: BenchmarkEvent) -> tuple[object, ...]:
    return (
        event.kind,
        event.step,
        event.numerator,
        event.denominator,
        round(float(event.native_return), 9),
        event.start_step,
        event.end_step,
    )


def check_episode_panel(
    base: Sequence[BenchmarkEvent], extended: Sequence[BenchmarkEvent]
) -> dict[str, int]:
    """The episode adapter's acceptance check: at the trained size the
    adapter's panel reproduces the trained panel episode for episode (same
    success, steps, return and span on every task and rollout seed). A larger
    size changes every episode by design, so the check applies only at the
    trained size; the driver records the others as not applicable. Returns
    the counts compared; raises :class:`ContractError` on the first
    disagreement.
    """
    base_by_key = {
        (int(e.task_id), int(e.rollout_seed)): e for e in base if e.kind == "episode"
    }
    ext_by_key = {
        (int(e.task_id), int(e.rollout_seed)): e
        for e in extended
        if e.kind == "episode"
    }
    if not base_by_key or not ext_by_key:
        raise ContractError("The episode check needs episode events on both panels.")
    if set(base_by_key) != set(ext_by_key):
        raise ContractError("The two panels cover different tasks or rollout seeds.")
    for key, event in base_by_key.items():
        other = ext_by_key[key]
        if _episode_signature(other) != _episode_signature(event):
            raise ContractError(
                f"Task {key[0]} differs at the trained size: base "
                f"{_episode_signature(event)}, adapter {_episode_signature(other)}."
            )
    return {"compared": len(base_by_key), "units": len(base_by_key)}


STREAM_FLIP_TOLERANCE = 0.001
"""Largest fraction of the native stream's decisions whose answer may differ
between the trained panel and the adapter's panel with the same query and
true count (numerical near-tie flips across GPU kernels); at least one is
always allowed. Any difference in the query itself fails the check."""


def _query_signature(event: BenchmarkEvent) -> tuple[object, ...]:
    return (
        event.kind,
        event.step,
        event.numerator,
        event.denominator,
        round(float(event.native_return), 9),
        event.true_count,
        event.answer,
    )


def check_stream_prefix(
    base: Sequence[BenchmarkEvent],
    extended: Sequence[BenchmarkEvent],
    *,
    native: int,
) -> dict[str, int]:
    """The CountRecall adapter's acceptance check: the extended panel's first
    ``native`` records (``native - 1`` scored decisions) reproduce the trained
    panel query for query (same correctness, reward, true count and answer on
    every task and rollout seed), and the extended panel holds no other
    decision inside them. Returns the counts compared; raises
    :class:`ContractError` on the first disagreement.
    """
    decisions = native - 1
    base_by_key = {
        (int(e.task_id), int(e.rollout_seed), int(e.step)): e
        for e in base
        if e.kind == "query"
    }
    ext_by_key = {
        (int(e.task_id), int(e.rollout_seed), int(e.step)): e
        for e in extended
        if e.kind == "query" and int(e.step) <= decisions
    }
    if not base_by_key or not ext_by_key:
        raise ContractError("The stream check needs query events on both panels.")
    if {k[:2] for k in base_by_key} != {k[:2] for k in ext_by_key}:
        raise ContractError("The two panels cover different tasks or rollout seeds.")
    if set(base_by_key) != set(ext_by_key):
        raise ContractError(
            "The extended panel's decisions inside the native stream are not "
            "the trained panel's."
        )
    flips = 0
    for key, event in base_by_key.items():
        other = ext_by_key[key]
        if _query_signature(other) == _query_signature(event):
            continue
        if (other.denominator, other.true_count) != (
            event.denominator,
            event.true_count,
        ):
            raise ContractError(
                f"Task {key[0]} decision {key[2]} differs inside the native stream: "
                f"base {_query_signature(event)}, extended {_query_signature(other)}."
            )
        # Same query and true count, a different answer: a numerical flip of a
        # near-tie argmax between two GPU kernels (seen on the GRU carrier, two of
        # 26,368 decisions on a different card); counted and bounded.
        flips += 1
    compared = len(base_by_key)
    if flips > max(1, int(compared * STREAM_FLIP_TOLERANCE)):
        raise ContractError(
            f"{flips} of {compared} decisions inside the native stream differ "
            f"between the panels (tolerance {STREAM_FLIP_TOLERANCE:.0%} for "
            "numerical flips of the same query)."
        )
    return {
        "compared": compared,
        "units": len({k[:2] for k in base_by_key}),
        "decision_flips": flips,
    }


def stream_window_index(decision: int, *, native: int) -> int:
    """The window of a scored decision (1-based): window 0 is the trained
    stream's ``native - 1`` decisions; every later window holds ``native``
    query-only decisions, so a panel of ``k * native`` records ends exactly at
    window ``k - 1``'s last decision."""
    if decision < 1:
        raise ContractError("A decision index is positive.")
    if decision < native:
        return 0
    return 1 + (decision - native) // native


def stream_window_matrix(
    events: Sequence[BenchmarkEvent],
    *,
    units: Sequence[tuple[int, int]],
    native: int,
    horizon: int,
) -> np.ndarray:
    """Exact accuracy per window as a ``[len(units), windows]`` array in the
    given (task, rollout seed) order: the paired form of the pre-declared
    CountRecall estimand, one
    row per roster unit, every window of every unit fully scored."""
    windows = stream_window_index(horizon - 1, native=native) + 1
    index = {
        (int(task), int(rollout)): row for row, (task, rollout) in enumerate(units)
    }
    if len(index) != len(units):
        raise ContractError("The roster units must be distinct.")
    correct = np.zeros((len(units), windows), dtype=np.float64)
    scored = np.zeros((len(units), windows), dtype=np.float64)
    for event in events:
        if event.kind != "query":
            continue
        unit = (int(event.task_id), int(event.rollout_seed))
        if unit not in index:
            raise ContractError(f"Events for a unit outside the roster: {unit}.")
        if not 1 <= int(event.step) <= horizon - 1:
            raise ContractError(
                f"Decision {event.step} lies outside the {horizon}-record stream."
            )
        column = stream_window_index(int(event.step), native=native)
        correct[index[unit], column] += int(event.numerator)
        scored[index[unit], column] += 1
    expected = np.array([native - 1] + [native] * (windows - 1), dtype=np.float64)
    if not np.array_equal(scored, np.broadcast_to(expected, scored.shape)):
        raise ContractError("Every window of every unit must be fully scored.")
    return correct / scored


def stream_window_accuracy(
    events: Sequence[BenchmarkEvent], *, native: int, horizon: int
) -> list[float]:
    """Mean exact accuracy per window over every scored decision of the panel
    (the roster read; the paired reading uses :func:`stream_window_matrix`)."""
    units = sorted(
        {(int(e.task_id), int(e.rollout_seed)) for e in events if e.kind == "query"}
    )
    if not units:
        raise ContractError("The window estimand needs query events.")
    matrix = stream_window_matrix(events, units=units, native=native, horizon=horizon)
    return [float(value) for value in matrix.mean(axis=0)]


def maze_goal_fraction(events: Sequence[BenchmarkEvent]) -> dict[str, float]:
    """The MazeRunner estimand of one panel: the goal fraction (goals reached
    of the ordered sequence) averaged equally over every (task, rollout seed)
    unit, beside the full-sequence rate, the mean decisions taken and the
    pooled goals per 500 steps (goals reached over decisions taken, scaled to
    the trained budget), with the unit count."""
    units = [e for e in events if e.kind == "episode"]
    if not units:
        raise ContractError("The goal-fraction estimand needs episode events.")
    decisions = [int(e.step) for e in units]
    if any(d < 1 for d in decisions) or any(e.denominator < 1 for e in units):
        raise ContractError(
            "Every MazeRunner episode takes a decision and names goals."
        )
    fractions = [float(e.numerator) / float(e.denominator) for e in units]
    return {
        "goal_fraction": float(sum(fractions) / len(units)),
        "full_sequence": float(
            sum(e.numerator == e.denominator for e in units) / len(units)
        ),
        "mean_decisions": float(sum(decisions) / len(units)),
        "goals_per_500_steps": 500.0
        * float(sum(e.numerator for e in units))
        / float(sum(decisions)),
        "units": float(len(units)),
    }


def _attempt_key(event: BenchmarkEvent) -> tuple[int, int, int]:
    return (int(event.task_id), int(event.rollout_seed), int(event.event_index))


def _attempt_signature(event: BenchmarkEvent) -> tuple[object, ...]:
    return (
        event.kind,
        event.step,
        event.numerator,
        event.denominator,
        round(float(event.native_return), 9),
        event.start_step,
        event.end_step,
        event.key_step,
    )


def check_horizon_prefix(
    base: Sequence[BenchmarkEvent],
    extended: Sequence[BenchmarkEvent],
    *,
    native: int,
) -> dict[str, int]:
    """The adapter's acceptance check: the extended panel's first ``native``
    calls reproduce the trained-horizon panel.

    Every attempt of the base panel that finished within the budget must appear
    in the extended panel with the same span, steps, success and key step, and
    the extended panel must hold no other attempt ending inside the budget.
    The base panel's last attempt is partial when the budget cut it; the
    extended panel continues that attempt, so it is compared on its start
    alone. Returns the counts compared; raises :class:`ContractError` on the
    first disagreement.
    """
    base_by_key = {_attempt_key(e): e for e in base if e.kind == "attempt"}
    ext_by_key = {_attempt_key(e): e for e in extended if e.kind == "attempt"}
    if not base_by_key or not ext_by_key:
        raise ContractError("The prefix check needs attempt events on both panels.")
    base_units = {key[:2] for key in base_by_key}
    ext_units = {key[:2] for key in ext_by_key}
    if base_units != ext_units:
        raise ContractError("The two panels cover different tasks or rollout seeds.")
    compared = partial = 0
    for key, event in base_by_key.items():
        other = ext_by_key.get(key)
        if other is None:
            raise ContractError(
                f"Attempt {key[2]} of task {key[0]} is missing from the extended panel."
            )
        if event.complete is False or (
            event.end_step is not None and event.end_step >= native
        ):
            # The budget cut this attempt; the extended task keeps playing it.
            if other.start_step != event.start_step:
                raise ContractError(
                    f"Task {key[0]}: the attempt cut at the budget starts at "
                    f"{event.start_step} in the base panel and {other.start_step} "
                    "in the extended one."
                )
            partial += 1
            continue
        if _attempt_signature(other) != _attempt_signature(event):
            raise ContractError(
                f"Task {key[0]} attempt {key[2]} differs inside the first {native} "
                f"calls: base {_attempt_signature(event)}, extended "
                f"{_attempt_signature(other)}."
            )
        compared += 1
    for key, other in ext_by_key.items():
        if key not in base_by_key and (
            other.end_step is not None and other.end_step <= native
        ):
            raise ContractError(
                f"Task {key[0]}: the extended panel holds attempt {key[2]} inside "
                f"the first {native} calls that the base panel lacks."
            )
    return {"compared": compared, "partial": partial, "units": len(base_units)}


def window_door_counts(
    events: Sequence[BenchmarkEvent], *, outer_length: int, window: int = WINDOW
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Completed doors per fixed window of calls, per (task, rollout seed).

    A door counts in the window holding the call that completed it (the
    attempt's ``end_step``, one-based); an attempt cut by the budget counts
    nowhere. This is the pre-declared estimand of the study protocol
    (section 5): doors in calls 1-500, 501-1,000, 1,001-1,500, 1,501-2,000.
    """
    if outer_length % window:
        raise ContractError("The outer budget must be a whole number of windows.")
    windows = outer_length // window
    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0] * windows)
    for event in events:
        if event.kind != "attempt" or not event.numerator:
            continue
        if event.end_step is None or event.complete is False:
            continue
        index = (int(event.end_step) - 1) // window
        if not 0 <= index < windows:
            raise ContractError(
                f"Attempt end {event.end_step} lies outside the {outer_length}-call "
                "budget."
            )
        counts[(int(event.task_id), int(event.rollout_seed))][index] += 1
    return {unit: tuple(values) for unit, values in counts.items()}


def window_matrix(
    events: Sequence[BenchmarkEvent],
    *,
    units: Sequence[tuple[int, int]],
    outer_length: int,
    window: int = WINDOW,
) -> np.ndarray:
    """Completed doors per window as a ``[len(units), windows]`` array in the
    given (task, rollout seed) order, zero where a unit completed no door.

    The paired form of :func:`window_door_counts`: every unit of the roster
    has a row, so panels of different cells subtract unit by unit and the
    per-window rate is a mean over the whole roster.
    """
    counts = window_door_counts(events, outer_length=outer_length, window=window)
    index = {
        (int(task), int(rollout)): row for row, (task, rollout) in enumerate(units)
    }
    if len(index) != len(units):
        raise ContractError("The roster units must be distinct.")
    unknown = set(counts) - set(index)
    if unknown:
        raise ContractError(
            f"Events for units outside the roster: {sorted(unknown)[:3]}."
        )
    matrix = np.zeros((len(units), outer_length // window), dtype=np.float64)
    for unit, values in counts.items():
        matrix[index[unit]] = values
    return matrix


def layout_matrix(
    events: Sequence[BenchmarkEvent],
    *,
    units: Sequence[tuple[int, int]],
    layouts: int,
) -> np.ndarray:
    """Completed doors per hidden layout as a ``[len(units), layouts]`` array
    in the given (task, rollout seed) order, zero where a unit completed no
    door in that layout.

    The layout-change continuation swaps the layout at the first attempt
    boundary at or after every k*P calls, so the old layout persists up to
    one attempt past the period boundary; a door is therefore binned by the
    layout its attempt records (``layout_index``), not by the window holding
    its completing call. Every attempt of a layout-change panel names its
    layout; an attempt without one, or outside ``layouts``, is an error.
    """
    if layouts <= 0:
        raise ContractError("A layout-change read needs at least one layout.")
    index = {
        (int(task), int(rollout)): row for row, (task, rollout) in enumerate(units)
    }
    if len(index) != len(units):
        raise ContractError("The roster units must be distinct.")
    matrix = np.zeros((len(units), layouts), dtype=np.float64)
    for event in events:
        if event.kind != "attempt" or not event.numerator:
            continue
        if event.end_step is None or event.complete is False:
            continue
        if event.layout_index is None:
            raise ContractError(
                f"Task {event.task_id}: attempt {event.event_index} of a "
                "layout-change panel names no layout."
            )
        layout = int(event.layout_index)
        if not 0 <= layout < layouts:
            raise ContractError(
                f"Task {event.task_id}: layout {layout} lies outside the "
                f"{layouts} layouts of the panel."
            )
        unit = (int(event.task_id), int(event.rollout_seed))
        if unit not in index:
            raise ContractError(f"Events for a unit outside the roster: {unit}.")
        matrix[index[unit], layout] += 1
    return matrix


__all__ = [
    "DECLARED_HORIZONS",
    "HORIZON_ENVIRONMENTS",
    "LAP_BUDGETS",
    "MAZE_LADDER_OFFSETS",
    "MAZE_SIZES",
    "STREAM_COUNTS",
    "STREAM_FLIP_TOLERANCE",
    "STREAM_HORIZONS",
    "WINDOW",
    "check_episode_panel",
    "check_horizon_prefix",
    "check_laps_prefix",
    "check_stream_prefix",
    "continued_laps",
    "continued_streams",
    "extended_horizon",
    "horizon_kind",
    "laps_per_unit",
    "layout_matrix",
    "maze_goal_fraction",
    "maze_size_ladder",
    "maze_timer",
    "native_horizon",
    "stream_window_accuracy",
    "stream_window_index",
    "stream_window_matrix",
    "streams_accuracy",
    "streams_matrix",
    "window_door_counts",
    "window_goal_counts",
    "window_goal_matrix",
    "window_matrix",
]
