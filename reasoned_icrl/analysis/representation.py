"""Representation displays and the summary-transplant readings.

Pure functions over the files ``scripts/representation_read.py`` writes: per
evaluation the summaries each segment read (``summaries-*.npz``), the decisions'
public inputs and actions (``inputs-*.npz``), their attention masses per block
(``decisions-*.npz``) and the evaluator events (``labels-*.csv``); on
Key-to-Door the hidden cells of every task (``layouts.csv``). Decision steps are
zero-based; event steps are one-based (decision ``s`` is event step ``s + 1``).
Intervals are the tier reports' joint seed/task bootstrap
(:func:`reasoned_icrl.analysis.statistics.seed_task_interval`) over paired
per-task values, conditional on the three trained seeds.

P1 projects the summaries one segment reads onto principal components fitted on
the development captures and reads the hidden variables out of them by ridge
regression fitted there; P2 is the centred cosine similarity of the summaries
across rewrites; P3 is the position-adjusted summary share by room cell; P4 the
summary mass per 500-call window. T1-T4 are the transplant's paired readings.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.analysis.attention import load_labels, residual_mass
from reasoned_icrl.analysis.statistics import seed_task_interval
from reasoned_icrl.experiments.contracts import ResultValidationError

METHOD, ABLATION, REFERENCE = "raw_summary", "raw_summary_residual", "raw_segment"
WINDOW = 500
"""Calls per window of the long-running displays (P4)."""
GRID_MINIMUM = 50
"""Decisions a room cell needs, over the three seeds, to be drawn (P3)."""
SEGMENTS = {"dark_key_to_door": 8, "count_recall": 2}
"""The segment whose summary P1 projects and the transplant replaces."""
RIDGE_ALPHAS = tuple(float(value) for value in np.logspace(-3, 5, 17))
_FILE = re.compile(
    r"(?P<kind>[a-z]+)-(?P<cell>.+)-seed(?P<seed>\d+)-(?P<label>.+)\.npz"
)

Row = dict[str, Any]
Columns = dict[str, NDArray[Any]]
Decoder = Callable[[NDArray[np.float32]], tuple[int, int]]
"""A CountRecall packet to its (dealt value, queried category)."""


# ------------------------------------------------------------------ files


def load_files(directory: Path, kind: str) -> dict[tuple[str, int, str], Columns]:
    """Every ``<kind>-<cell>-seed<seed>-<label>.npz``, keyed by (cell, seed, label)."""
    found: dict[tuple[str, int, str], Columns] = {}
    for path in sorted(directory.glob(f"{kind}-*.npz")):
        match = _FILE.fullmatch(path.name)
        if match is None or match["kind"] != kind:
            continue
        with np.load(path) as data:
            found[(match["cell"], int(match["seed"]), match["label"])] = {
                name: data[name] for name in data.files
            }
    return found


def load_layouts(path: Path) -> dict[int, dict[str, tuple[int, int]]]:
    """``layouts.csv``: task -> {start, key, door} as (row, column)."""
    layouts: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            layouts[int(row["task_id"])][row["cell"]] = (
                int(row["row"]),
                int(row["column"]),
            )
    return dict(layouts)


def _matrix(
    per_seed: Mapping[int, Mapping[int, float]],
) -> tuple[list[int], NDArray[np.float64]]:
    """A ``[seeds, tasks]`` matrix over the tasks every seed has."""
    seeds = sorted(per_seed)
    tasks = sorted(set.intersection(*(set(per_seed[seed]) for seed in seeds)))
    if not tasks:
        raise ResultValidationError("No task is shared by every seed.")
    return tasks, np.asarray(
        [[per_seed[seed][task] for task in tasks] for seed in seeds], dtype=np.float64
    )


def _interval_row(name: str, per_seed: Mapping[int, Mapping[int, float]]) -> Row:
    tasks, matrix = _matrix(per_seed)
    estimate, lower, upper = seed_task_interval(matrix)
    seed_means = matrix.mean(axis=1)
    return {
        "reading": name,
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "per_seed": "; ".join(
            f"{seed}: {value:+.4f}"
            for seed, value in zip(sorted(per_seed), seed_means, strict=True)
        ),
        "positive_seeds": int((seed_means > 0).sum()),
        "tasks": len(tasks),
    }


def _holds(row: Row) -> bool:
    return bool(
        row["lower"] > 0 and row["positive_seeds"] == len(row["per_seed"].split(";"))
    )


def _keyed(columns: Columns) -> dict[tuple[int, int], int]:
    """(task, step) -> row index of a decisions or inputs file."""
    return {
        (int(task), int(step)): index
        for index, (task, step) in enumerate(
            zip(columns["task_id"], columns["step"], strict=True)
        )
    }


# ------------------------------------------------------------------ P1


def principal_components(
    features: NDArray[np.floating[Any]], count: int = 2
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Mean, the first ``count`` components (rows) and their variance shares."""
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    _, singular, vectors = np.linalg.svd(values - mean, full_matrices=False)
    shares = singular**2 / max(float((singular**2).sum()), 1e-12)
    return mean, vectors[:count], shares[:count]


def ridge_readout(
    train: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    test: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    folds: int = 5,
    seed: int = 0,
) -> tuple[float, NDArray[np.float64]]:
    """Ridge regression fitted on ``train`` (penalty by ``folds``-fold
    cross-validation on ``train`` alone), scored on ``test`` as one R² per
    target column. Features and targets are centred by the training means."""
    x = np.asarray(train, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    order = np.random.default_rng(seed).permutation(len(x))
    parts = np.array_split(order, folds)

    def fit(
        features: NDArray[np.float64], response: NDArray[np.float64], alpha: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        mean_x, mean_y = features.mean(0), response.mean(0)
        u, s, vt = np.linalg.svd(features - mean_x, full_matrices=False)
        weights = vt.T @ ((s / (s**2 + alpha))[:, None] * (u.T @ (response - mean_y)))
        return mean_x, mean_y, weights

    errors = []
    for alpha in alphas:
        error = 0.0
        for part in parts:
            keep = np.setdiff1d(order, part)
            mean_x, mean_y, weights = fit(x[keep], y[keep], alpha)
            predicted = (x[part] - mean_x) @ weights + mean_y
            error += float(((predicted - y[part]) ** 2).sum())
        errors.append(error)
    alpha = float(alphas[int(np.argmin(errors))])
    mean_x, mean_y, weights = fit(x, y, alpha)
    observed = np.asarray(truth, dtype=np.float64)
    if observed.ndim == 1:
        observed = observed[:, None]
    predicted = (np.asarray(test, dtype=np.float64) - mean_x) @ weights + mean_y
    residual = ((predicted - observed) ** 2).sum(0)
    total = ((observed - observed.mean(0)) ** 2).sum(0)
    return alpha, 1.0 - residual / np.maximum(total, 1e-12)


def _flat(memory: NDArray[Any]) -> NDArray[np.float64]:
    return np.asarray(memory, dtype=np.float64).reshape(len(memory), -1)


def _segment_rows(
    summaries: Columns, segment: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    chosen = summaries["segment"] == segment
    order = np.argsort(summaries["task_id"][chosen], kind="stable")
    return (
        summaries["task_id"][chosen][order].astype(np.int64),
        _flat(summaries["memory"][chosen][order]),
    )


def keydoor_targets(
    tasks: Sequence[int], layouts: Mapping[int, Mapping[str, tuple[int, int]]]
) -> tuple[list[str], NDArray[np.float64]]:
    names = ["key_row", "key_column", "door_row", "door_column"]
    values = [
        [*layouts[int(task)]["key"], *layouts[int(task)]["door"]] for task in tasks
    ]
    return names, np.asarray(values, dtype=np.float64)


def count_targets(
    tasks: Sequence[int],
    inputs: Columns,
    decode: Decoder,
    *,
    before_step: int,
    categories: int,
) -> tuple[list[str], NDArray[np.float64]]:
    """Cards of each category dealt at decisions ``0 .. before_step - 1``."""
    dealt = _dealt(inputs, decode)
    counts = []
    for task in tasks:
        values = [value for step, value in dealt[int(task)] if step < before_step]
        counts.append([values.count(category) for category in range(categories)])
    return [f"count_{category}" for category in range(categories)], np.asarray(
        counts, dtype=np.float64
    )


def _dealt(inputs: Columns, decode: Decoder) -> dict[int, list[tuple[int, int]]]:
    """task -> [(step, dealt value)] in step order (blank values dropped)."""
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for task, step, current in zip(
        inputs["task_id"], inputs["step"], inputs["current"], strict=True
    ):
        value, _ = decode(current)
        if value >= 0:
            out[int(task)].append((int(step), value))
    for values in out.values():
        values.sort()
    return dict(out)


def projection_rows(
    summaries: Mapping[int, Columns],
    development: Mapping[int, Columns],
    targets: Callable[[Sequence[int], bool], tuple[list[str], NDArray[np.float64]]],
    *,
    segment: int,
) -> tuple[list[Row], list[Row]]:
    """P1 per seed: the confirmation summaries of ``segment`` on the first two
    components fitted on every development summary, with their hidden
    variables; and the ridge read-out (fitted on the development summaries of
    the same segment) as one R² per target."""
    points: list[Row] = []
    readout: list[Row] = []
    for seed in sorted(summaries):
        if seed not in development:
            continue
        mean, components, shares = principal_components(
            _flat(development[seed]["memory"])
        )
        tasks, features = _segment_rows(summaries[seed], segment)
        names, values = targets([int(task) for task in tasks], False)
        projected = (features - mean) @ components.T
        for index, task in enumerate(tasks):
            points.append(
                {
                    "seed": seed,
                    "task_id": int(task),
                    "segment": segment,
                    "pc1": float(projected[index, 0]),
                    "pc2": float(projected[index, 1]),
                    "pc1_share": float(shares[0]),
                    "pc2_share": float(shares[1]),
                    **{name: float(values[index, k]) for k, name in enumerate(names)},
                }
            )
        train_tasks, train = _segment_rows(development[seed], segment)
        _, train_targets = targets([int(task) for task in train_tasks], True)
        alpha, scores = ridge_readout(train, train_targets, features, values)
        for name, score in zip(names, scores, strict=True):
            readout.append(
                {
                    "seed": seed,
                    "segment": segment,
                    "target": name,
                    "r2": float(score),
                    "alpha": alpha,
                    "train_tasks": len(train_tasks),
                    "test_tasks": len(tasks),
                }
            )
    return points, readout


# ------------------------------------------------------------------ P2


def similarity_rows(
    summaries: Mapping[str, Mapping[int, Columns]], *, first: int = 1
) -> tuple[list[Row], list[Row]]:
    """P2: per cell, the centred cosine similarity between the summaries read
    by segments b and b' (from ``first``), averaged over tasks then seeds, and
    the mean centred norm per segment."""
    matrix_rows: list[Row] = []
    norm_rows: list[Row] = []
    for cell, by_seed in summaries.items():
        matrices, norms = [], []
        segments_seen: list[int] = []
        for seed in sorted(by_seed):
            columns = by_seed[seed]
            chosen = columns["segment"] >= first
            tasks = np.unique(columns["task_id"][chosen])
            segments = np.unique(columns["segment"][chosen])
            grid = np.full(
                (len(tasks), len(segments), columns["memory"][0].size), np.nan
            )
            task_index = {int(task): k for k, task in enumerate(tasks)}
            segment_index = {int(segment): k for k, segment in enumerate(segments)}
            for task, segment, memory in zip(
                columns["task_id"][chosen],
                columns["segment"][chosen],
                columns["memory"][chosen],
                strict=True,
            ):
                grid[task_index[int(task)], segment_index[int(segment)]] = (
                    memory.reshape(-1)
                )
            complete = ~np.isnan(grid).any(axis=(1, 2))
            grid = grid[complete]
            centred = grid - grid.reshape(-1, grid.shape[-1]).mean(axis=0)
            length = np.linalg.norm(centred, axis=-1)
            unit = centred / np.maximum(length, 1e-12)[..., None]
            matrices.append(np.einsum("tsd,tud->su", unit, unit) / len(unit))
            norms.append(length.mean(axis=0))
            segments_seen = [int(segment) for segment in segments]
        mean_matrix = np.mean(matrices, axis=0)
        mean_norm = np.mean(norms, axis=0)
        for i, b in enumerate(segments_seen):
            norm_rows.append({"cell": cell, "segment": b, "norm": float(mean_norm[i])})
            for j, c in enumerate(segments_seen):
                matrix_rows.append(
                    {
                        "cell": cell,
                        "segment": b,
                        "other": c,
                        "similarity": float(mean_matrix[i, j]),
                    }
                )
    return matrix_rows, norm_rows


# ------------------------------------------------------------------ P3


def _attempt_of(labels: Sequence[Row]) -> dict[int, list[tuple[int, int, int]]]:
    """task -> [(first decision, last decision, attempt index)] (zero-based)."""
    attempts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for row in labels:
        if row["kind"] != "attempt":
            continue
        attempts[int(row["task_id"])].append(
            (
                int(row["start_step"]) - 1,
                int(row["end_step"]) - 1,
                int(row["event_index"]),
            )
        )
    return dict(attempts)


def keydoor_cells(
    current: NDArray[np.floating[Any]], size: int = 8
) -> NDArray[np.int64]:
    """Key-to-Door packets to (row, column) cells: the native observation is
    ``cell / size`` and the packet ``2 * observation - 1``."""
    raw = (np.asarray(current, dtype=np.float64)[:, :2] + 1.0) / 2.0
    return np.rint(raw * size).astype(np.int64)


def grid_rows(
    decisions: Mapping[int, Columns],
    inputs: Mapping[int, Columns],
    labels: Mapping[int, Sequence[Row]],
    *,
    size: int = 8,
    minimum: int = GRID_MINIMUM,
) -> list[Row]:
    """P3: the position-adjusted summary share (block mean) of the decisions at
    each room cell, attempt 1 against attempts 2 and later, seeds pooled."""
    sums: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for seed in sorted(decisions):
        found = decisions[seed]
        layers = found["summary_layer"].shape[1]
        residual = np.mean(
            [residual_mass(found, layer) for layer in range(layers)], axis=0
        )
        index = _keyed(inputs[seed])
        cells = keydoor_cells(inputs[seed]["current"], size)
        spans = _attempt_of(labels[seed])
        for row, (task, step) in enumerate(
            zip(found["task_id"], found["step"], strict=True)
        ):
            where = index.get((int(task), int(step)))
            if where is None:
                continue
            group = None
            for first, last, attempt in spans.get(int(task), ()):
                if first <= int(step) <= last:
                    group = "first" if attempt == 1 else "later"
                    break
            if group is None:
                continue
            cell_row, cell_column = cells[where]
            sums[(group, int(cell_row), int(cell_column))].append(float(residual[row]))
    rows = []
    for (group, cell_row, cell_column), values in sorted(sums.items()):
        rows.append(
            {
                "attempts": group,
                "row": cell_row,
                "column": cell_column,
                "decisions": len(values),
                "share": float(np.mean(values)) if len(values) >= minimum else "",
            }
        )
    return rows


# ------------------------------------------------------------------ P4


def window_rows(
    decisions: Mapping[str, Mapping[int, Columns]], *, window: int = WINDOW
) -> list[Row]:
    """P4: per cell, block and 500-call window, the decisions' mean summary mass
    (per task, then the joint seed/task interval)."""
    rows = []
    for cell, by_seed in decisions.items():
        per: dict[tuple[int, int], dict[int, dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for seed, found in by_seed.items():
            tasks, task_index = np.unique(found["task_id"], return_inverse=True)
            windows = found["step"].astype(np.int64) // window
            count = int(windows.max()) + 1
            key = task_index * count + windows
            totals = np.bincount(key, minlength=len(tasks) * count)
            for layer in range(found["summary_layer"].shape[1]):
                mass = np.bincount(
                    key,
                    weights=found["summary_layer"][:, layer].astype(np.float64),
                    minlength=len(tasks) * count,
                )
                for k, task in enumerate(tasks):
                    for w in range(count):
                        cell_total = totals[k * count + w]
                        if cell_total:
                            per[(layer, w)][seed][int(task)] = float(
                                mass[k * count + w] / cell_total
                            )
        for (layer, w), per_seed in sorted(per.items()):
            tasks_shared, matrix = _matrix(per_seed)
            estimate, lower, upper = seed_task_interval(matrix)
            rows.append(
                {
                    "cell": cell,
                    "layer": layer,
                    "window": w + 1,
                    "centre": (w + 0.5) * window,
                    "estimate": estimate,
                    "lower": lower,
                    "upper": upper,
                    "tasks": len(tasks_shared),
                }
            )
    return rows


# ------------------------------------------------------------------ T1-T4


def _doors_between(labels: Sequence[Row], first: int, last: int) -> dict[int, float]:
    """Doors completed with the completing call in ``first .. last`` (one-based)."""
    doors: dict[int, float] = defaultdict(float)
    for row in labels:
        if row["kind"] != "attempt":
            continue
        doors[int(row["task_id"])] += 0.0
        end = row["end_step"]
        if row["numerator"] and end is not None and first <= int(end) <= last:
            doors[int(row["task_id"])] += 1.0
    return dict(doors)


def donor_key_visits(
    labels: Sequence[Row],
    inputs: Columns,
    layouts: Mapping[int, Mapping[str, tuple[int, int]]],
    donors: Mapping[int, int],
    *,
    boundary_step: int,
    size: int = 8,
) -> dict[int, float]:
    """T2 per recipient: 1 when, in its first attempt that starts at or after
    decision ``boundary_step``, the agent steps on the donor's key cell before
    it holds its key. Tasks whose donor key is their own key are left out."""
    cells = keydoor_cells(inputs["current"], size)
    holds = (np.asarray(inputs["current"])[:, 2] + 1.0) / 2.0 > 0.5
    index = _keyed(inputs)
    visits: dict[int, float] = {}
    for task, spans in _attempt_of(labels).items():
        donor_key = layouts[donors[task]]["key"]
        if donor_key == layouts[task]["key"]:
            continue
        later = sorted(span for span in spans if span[0] >= boundary_step)
        if not later:
            continue
        first, last, _ = later[0]
        hit = 0.0
        # Observations of decisions first .. last + 1: the cells from the
        # attempt's start through its final cell (the reset-only call's).
        for step in range(first, last + 2):
            where = index.get((task, step))
            if where is None or holds[where]:
                break
            if tuple(int(v) for v in cells[where]) == donor_key:
                hit = 1.0
                break
        visits[task] = hit
    return visits


def _streams(
    inputs: Columns, decode: Decoder, categories: int
) -> dict[int, tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]]:
    """task -> (queried category, answer, cumulative counts ``[steps, K]``) in
    step order; the cumulative count at a step includes that step's card."""
    order = np.lexsort((inputs["step"], inputs["task_id"]))
    by_task: dict[int, list[int]] = defaultdict(list)
    for row in order:
        by_task[int(inputs["task_id"][row])].append(int(row))
    streams = {}
    for task, rows in by_task.items():
        decoded = [decode(inputs["current"][row]) for row in rows]
        onehot = np.zeros((len(rows), categories), dtype=np.int64)
        for position, (value, _) in enumerate(decoded):
            if value >= 0:
                onehot[position, value] = 1
        streams[task] = (
            np.asarray([query for _, query in decoded], dtype=np.int64),
            np.asarray([int(inputs["action"][row]) for row in rows], dtype=np.int64),
            np.cumsum(onehot, axis=0),
        )
    return streams


def count_scores(
    inputs: Columns,
    decode: Decoder,
    *,
    first_step: int,
    donors: Mapping[int, int] | None = None,
    donor_inputs: Columns | None = None,
    boundary_step: int = 0,
    categories: int = 4,
) -> tuple[dict[int, float], dict[int, NDArray[np.float64]]]:
    """Exact accuracy of every task's answers at decisions ``first_step`` and
    later, against the true counts or, with ``donors``, against the
    donor-consistent counts (the donor's cards of the queried category before
    ``boundary_step`` plus the task's own since it). Returns the per-task
    accuracy and each task's per-decision correctness (NaN before
    ``first_step``)."""
    streams = _streams(inputs, decode, categories)
    donor_streams = (
        _streams(donor_inputs, decode, categories)
        if donor_inputs is not None
        else streams
    )
    per_task: dict[int, float] = {}
    per_step: dict[int, NDArray[np.float64]] = {}
    for task, (queries, answers, counts) in streams.items():
        steps = np.arange(len(queries))
        truth = counts[steps, queries]
        if donors is not None:
            own_before = (
                counts[boundary_step - 1] if boundary_step > 0 else 0 * counts[0]
            )
            donor_counts = donor_streams[donors[task]][2]
            donor_before = (
                donor_counts[boundary_step - 1] if boundary_step > 0 else 0 * counts[0]
            )
            truth = truth - own_before[queries] + donor_before[queries]
        correct = (answers == truth).astype(np.float64)
        correct[steps < first_step] = np.nan
        per_step[task] = correct
        per_task[task] = float(np.nanmean(correct))
    return per_task, per_step


def keydoor_transplant(
    labels: Mapping[str, Mapping[int, Sequence[Row]]],
    inputs: Mapping[str, Mapping[int, Columns]],
    layouts: Mapping[int, Mapping[str, tuple[int, int]]],
    donors: Mapping[int, int],
    *,
    segment: int,
    segment_length: int = 32,
    outer: int = 500,
) -> tuple[list[Row], list[Row]]:
    """T1 and T2 on Key-to-Door, with doors per 50-call bin for the display."""
    boundary_step = segment * segment_length  # the first decision it reads
    retained, transplant, cleared = (
        "retained",
        f"transplant-b{segment}",
        f"cleared-once-b{segment}",
    )
    readings: list[Row] = []
    doors = {
        label: {
            seed: _doors_between(rows, boundary_step + 1, outer)
            for seed, rows in labels[label].items()
        }
        for label in (retained, transplant, cleared)
    }
    for other, name in (
        (transplant, "T1 transplanted - retained, doors in calls 257-500"),
        (cleared, "T1 cleared once - retained, doors in calls 257-500"),
    ):
        readings.append(
            _interval_row(
                name,
                {
                    seed: {
                        task: doors[other][seed][task] - value
                        for task, value in doors[retained][seed].items()
                        if task in doors[other][seed]
                    }
                    for seed in doors[retained]
                },
            )
        )
    visits = {
        label: {
            seed: donor_key_visits(
                labels[label][seed],
                inputs[label][seed],
                layouts,
                donors,
                boundary_step=boundary_step,
            )
            for seed in labels[label]
        }
        for label in (retained, transplant, cleared)
    }
    for label in (retained, transplant, cleared):
        readings.append(
            _interval_row(f"T2 level, {label}: donor key visited first", visits[label])
        )
    for other, name in (
        (retained, "T2 transplanted - retained, donor key visited first"),
        (cleared, "T2 transplanted - cleared once, donor key visited first"),
    ):
        row = _interval_row(
            name,
            {
                seed: {
                    task: value - visits[other][seed][task]
                    for task, value in visits[transplant][seed].items()
                    if task in visits[other][seed]
                }
                for seed in visits[transplant]
            },
        )
        if other == retained:
            row["declared_rule_holds"] = _holds(row)
        readings.append(row)
    bins: list[Row] = []
    width = 50
    for label in (retained, transplant, cleared):
        for seed, rows in labels[label].items():
            for start in range(0, outer, width):
                counts = _doors_between(rows, start + 1, start + width)
                bins.append(
                    {
                        "label": label,
                        "seed": seed,
                        "first_call": start + 1,
                        "last_call": start + width,
                        "doors": float(np.mean(list(counts.values())))
                        if counts
                        else 0.0,
                    }
                )
    return readings, bins


def countrecall_transplant(
    inputs: Mapping[str, Mapping[int, Columns]],
    decode: Decoder,
    donors: Mapping[int, int],
    *,
    segment: int,
    segment_length: int = 32,
) -> tuple[list[Row], list[Row]]:
    """T3 and T4 on CountRecall, with accuracy per query for the display."""
    boundary_step = segment * segment_length
    retained, transplant, cleared = (
        "retained",
        f"transplant-b{segment}",
        f"cleared-once-b{segment}",
    )
    own: dict[str, dict[int, dict[int, float]]] = defaultdict(dict)
    steps: dict[str, dict[int, dict[int, NDArray[np.float64]]]] = defaultdict(dict)
    donor: dict[str, dict[int, dict[int, float]]] = defaultdict(dict)
    donor_steps: dict[str, dict[int, dict[int, NDArray[np.float64]]]] = defaultdict(
        dict
    )
    for label in (retained, transplant, cleared):
        for seed, columns in inputs[label].items():
            own[label][seed], steps[label][seed] = count_scores(
                columns, decode, first_step=boundary_step
            )
            donor[label][seed], donor_steps[label][seed] = count_scores(
                columns,
                decode,
                first_step=boundary_step,
                donors=donors,
                donor_inputs=inputs[retained][seed],
                boundary_step=boundary_step,
            )

    def paired(
        a: Mapping[int, Mapping[int, float]], b: Mapping[int, Mapping[int, float]]
    ) -> dict[int, dict[int, float]]:
        return {
            seed: {
                task: a[seed][task] - b[seed][task]
                for task in a[seed]
                if task in b[seed]
            }
            for seed in a
        }

    readings = [
        _interval_row(
            "T3 transplanted - retained, true-count accuracy, queries 65-103",
            paired(own[transplant], own[retained]),
        ),
        _interval_row(
            "T3 cleared once - retained, true-count accuracy, queries 65-103",
            paired(own[cleared], own[retained]),
        ),
    ]
    for label in (retained, transplant, cleared):
        readings.append(
            _interval_row(f"level, {label}: true-count accuracy", own[label])
        )
        readings.append(
            _interval_row(f"level, {label}: donor-consistent accuracy", donor[label])
        )
    within = _interval_row(
        "T4a transplanted: donor-consistent - true-count accuracy",
        paired(donor[transplant], own[transplant]),
    )
    across = _interval_row(
        "T4b transplanted - retained, donor-consistent accuracy",
        paired(donor[transplant], donor[retained]),
    )
    for row in (within, across):
        row["declared_rule_holds"] = _holds(within) and _holds(across)
    readings += [within, across]
    queries: list[Row] = []
    for label, scoring, table in (
        (retained, "true", steps[retained]),
        (transplant, "true", steps[transplant]),
        (transplant, "donor", donor_steps[transplant]),
        (cleared, "true", steps[cleared]),
        (retained, "donor", donor_steps[retained]),
    ):
        stacked = np.stack(
            [values for per_seed in table.values() for values in per_seed.values()]
        )
        for position in range(boundary_step, stacked.shape[1]):
            queries.append(
                {
                    "label": label,
                    "scoring": scoring,
                    "query": position + 1,
                    "accuracy": float(np.nanmean(stacked[:, position])),
                }
            )
    return readings, queries


# ------------------------------------------------------------------ entry


def _by(
    found: Mapping[tuple[str, int, str], Columns], cell: str, label: str
) -> dict[int, Columns]:
    return {
        seed: columns
        for (name, seed, kind), columns in found.items()
        if name == cell and kind == label
    }


def _labels(
    directory: Path, cell: str, label: str, seeds: Sequence[int]
) -> dict[int, list[Row]]:
    return {
        seed: load_labels(directory / f"labels-{cell}-seed{seed}-{label}.csv")
        for seed in seeds
        if (directory / f"labels-{cell}-seed{seed}-{label}.csv").exists()
    }


def read_representation(
    directory: Path,
    *,
    benchmark: str,
    development: Path | None = None,
    decode: Decoder | None = None,
    donors: Mapping[int, int] | None = None,
    categories: int = 4,
) -> dict[str, list[Row]]:
    """Every display table and reading the files in ``directory`` support."""
    summaries = load_files(directory, "summaries")
    inputs = load_files(directory, "inputs")
    decisions = load_files(directory, "decisions")
    tables: dict[str, list[Row]] = {}
    development_summaries = load_files(development, "summaries") if development else {}
    segment = SEGMENTS.get(benchmark)
    if benchmark == "dark_key_to_door":
        layouts = load_layouts(directory / "layouts.csv")
        dev_layouts = (
            load_layouts(development / "layouts.csv")
            if development is not None and (development / "layouts.csv").exists()
            else {}
        )
        confirmation = _by(summaries, METHOD, "retained")
        if confirmation and development_summaries:

            def targets(
                tasks: Sequence[int], train: bool
            ) -> tuple[list[str], NDArray[np.float64]]:
                return keydoor_targets(tasks, dev_layouts if train else layouts)

            tables["projection"], tables["readout"] = projection_rows(
                confirmation,
                _by(development_summaries, METHOD, "retained"),
                targets,
                segment=int(segment or 8),
            )
        retained = _by(decisions, METHOD, "retained")
        if retained:
            seeds = sorted(retained)
            tables["grid"] = grid_rows(
                retained,
                _by(inputs, METHOD, "retained"),
                _labels(directory, METHOD, "retained", seeds),
            )
        long = {
            cell: _by(decisions, cell, "retained-h4000")
            for cell in (METHOD, REFERENCE)
            if _by(decisions, cell, "retained-h4000")
        }
        if long:
            tables["windows"] = window_rows(long)
        transplant_label = f"transplant-b{segment}"
        if donors is not None and _by(inputs, METHOD, transplant_label):
            labels = {
                label: _labels(
                    directory, METHOD, label, sorted(_by(inputs, METHOD, label))
                )
                for label in ("retained", transplant_label, f"cleared-once-b{segment}")
            }
            tables["transplant"], tables["transplant_bins"] = keydoor_transplant(
                labels,
                {label: _by(inputs, METHOD, label) for label in labels},
                layouts,
                donors,
                segment=int(segment or 8),
            )
    elif benchmark == "count_recall":
        if decode is None:
            raise ResultValidationError("CountRecall readings need the packet decoder.")
        confirmation = _by(summaries, METHOD, "retained")
        dev_inputs = load_files(development, "inputs") if development else {}
        if confirmation and development_summaries:
            before = int(segment or 2) * 32
            points: list[Row] = []
            readout: list[Row] = []
            for seed in sorted(confirmation):
                if (METHOD, seed, "retained") not in development_summaries:
                    continue

                def seed_targets(
                    tasks: Sequence[int], train: bool, seed: int = seed
                ) -> tuple[list[str], NDArray[np.float64]]:
                    source = (
                        dev_inputs[(METHOD, seed, "retained")]
                        if train
                        else inputs[(METHOD, seed, "retained")]
                    )
                    return count_targets(
                        tasks, source, decode, before_step=before, categories=categories
                    )

                seed_points, seed_readout = projection_rows(
                    {seed: confirmation[seed]},
                    {seed: development_summaries[(METHOD, seed, "retained")]},
                    seed_targets,
                    segment=int(segment or 2),
                )
                points += seed_points
                readout += seed_readout
            tables["projection"], tables["readout"] = points, readout
        transplant_label = f"transplant-b{segment}"
        if donors is not None and _by(inputs, METHOD, transplant_label):
            tables["transplant"], tables["transplant_queries"] = countrecall_transplant(
                {
                    label: _by(inputs, METHOD, label)
                    for label in (
                        "retained",
                        transplant_label,
                        f"cleared-once-b{segment}",
                    )
                },
                decode,
                donors,
                segment=int(segment or 2),
            )
    elif benchmark == "mazerunner":
        long_summaries = {
            cell: _by(summaries, cell, "retained-calls4000")
            for cell in (METHOD, ABLATION)
            if _by(summaries, cell, "retained-calls4000")
        }
        if long_summaries:
            tables["similarity"], tables["norms"] = similarity_rows(long_summaries)
        long = {
            cell: _by(decisions, cell, "retained-calls4000")
            for cell in (METHOD, ABLATION, REFERENCE)
            if _by(decisions, cell, "retained-calls4000")
        }
        if long:
            tables["windows"] = window_rows(long)
    return tables


__all__ = [
    "count_scores",
    "donor_key_visits",
    "grid_rows",
    "keydoor_cells",
    "principal_components",
    "projection_rows",
    "read_representation",
    "ridge_readout",
    "similarity_rows",
    "window_rows",
]
