"""Readings of the attention declaration.

Pure functions over the files ``scripts/attention_read.py run`` writes: the
per-task primary metric of every evaluation (``tasks.csv``), every decision's
attention masses per block (``decisions-<cell>-seed<seed>-<label>.npz``) and the
capture's evaluator events (``labels-<cell>-seed<seed>.csv``). Decision steps
are zero-based; event steps are one-based (decision ``s`` is event step
``s + 1``). Intervals are the tier reports' joint seed/task bootstrap
(:func:`reasoned_icrl.analysis.statistics.seed_task_interval`) over paired
per-task values, conditional on the three trained seeds.

A1 compares RSM's summary mass with the w/o-memory policy's READ mass, pooled
over positions 2-32. A2 (CountRecall) and A3 (Key-to-Door) compare groups of
RSM's decisions after subtracting the seed's mean at each segment and position.
B1-B3 are paired per-task differences of the primary metric under a read bias.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.analysis.statistics import seed_task_interval
from reasoned_icrl.experiments.contracts import ResultValidationError

LABEL_FIELDS = (
    "task_id",
    "kind",
    "event_index",
    "step",
    "numerator",
    "denominator",
    "start_step",
    "end_step",
    "count_before_current_segment",
    "count_in_current_segment",
    "writes_before_decision",
)
"""The capture events kept as decision labels."""

DELTAS: Mapping[str, float] = {"dark_key_to_door": 1.0, "count_recall": 0.05}
"""The declared practical margins of the primary metrics."""

METHOD, REFERENCE, CAPTURE = "raw_summary", "raw_segment", "retained"
ATTEMPT_START_STEPS = 8
_DECISIONS = re.compile(r"decisions-(?P<cell>.+)-seed(?P<seed>\d+)-(?P<label>.+)\.npz")

Row = dict[str, Any]


def load_decisions(
    directory: Path,
) -> dict[tuple[str, int, str], dict[str, NDArray[Any]]]:
    """Every decisions file, keyed by (cell, seed, label)."""
    found: dict[tuple[str, int, str], dict[str, NDArray[Any]]] = {}
    for path in sorted(directory.glob("decisions-*.npz")):
        match = _DECISIONS.fullmatch(path.name)
        if match is None:
            continue
        with np.load(path) as data:
            found[(match["cell"], int(match["seed"]), match["label"])] = {
                name: data[name] for name in data.files
            }
    return found


def load_tasks(path: Path) -> list[Row]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["task_id"] = int(row["task_id"])
        row["beta"] = float(row["beta"])
        row["primary"] = float(row["primary"])
    return rows


def load_labels(path: Path) -> list[Row]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in LABEL_FIELDS:
            value = row.get(name)
            if name == "kind":
                continue
            row[name] = None if value in (None, "") else int(float(value))
    return rows


def _layers(decisions: Mapping[str, NDArray[Any]]) -> int:
    return int(decisions["summary_layer"].shape[1])


def residual_mass(
    decisions: Mapping[str, NDArray[Any]], layer: int
) -> NDArray[np.float64]:
    """Summary mass minus the mean of its (segment, position) cell."""
    mass = decisions["summary_layer"][:, layer].astype(np.float64)
    keys = decisions["segment"].astype(np.int64) * 1000 + decisions["position"].astype(
        np.int64
    )
    residual = np.empty_like(mass)
    for key in np.unique(keys):
        chosen = keys == key
        residual[chosen] = mass[chosen] - mass[chosen].mean()
    return residual


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


def _interval_row(
    name: str, per_seed: Mapping[int, Mapping[int, float]], **extra: Any
) -> Row:
    tasks, matrix = _matrix(per_seed)
    estimate, lower, upper = seed_task_interval(matrix)
    seed_means = {
        seed: float(row.mean())
        for seed, row in zip(sorted(per_seed), matrix, strict=True)
    }
    return {
        "reading": name,
        **extra,
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "per_seed": "; ".join(
            f"{seed}: {value:+.4f}" for seed, value in seed_means.items()
        ),
        "positive_seeds": sum(value > 0 for value in seed_means.values()),
        "negative_seeds": sum(value < 0 for value in seed_means.values()),
        "tasks": len(tasks),
    }


def _task_means(
    task_ids: NDArray[Any], values: NDArray[np.float64], chosen: NDArray[np.bool_]
) -> dict[int, float]:
    sums: defaultdict[int, float] = defaultdict(float)
    counts: defaultdict[int, int] = defaultdict(int)
    for task, value in zip(
        task_ids[chosen].tolist(), values[chosen].tolist(), strict=True
    ):
        sums[int(task)] += value
        counts[int(task)] += 1
    return {task: sums[task] / counts[task] for task in sums}


def profile_rows(
    decisions: Mapping[tuple[str, int, str], Mapping[str, NDArray[Any]]],
    *,
    memory_tokens: int,
) -> list[Row]:
    """A1's display: mean masses by cell, block and position, per seed."""
    rows: list[Row] = []
    for (cell, seed, label), arrays in sorted(decisions.items()):
        if label != CAPTURE:
            continue
        positions = arrays["position"]
        for layer in range(_layers(arrays)):
            for position in np.unique(positions).tolist():
                chosen = positions == position
                rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "layer": layer,
                        "position": int(position),
                        "summary": float(arrays["summary_layer"][chosen, layer].mean()),
                        "buffer": float(arrays["buffer_layer"][chosen, layer].mean()),
                        "own": float(arrays["own_layer"][chosen, layer].mean()),
                        "uniform_summary": memory_tokens / (memory_tokens + position),
                        "decisions": int(chosen.sum()),
                    }
                )
    return rows


def reading_a1(
    decisions: Mapping[tuple[str, int, str], Mapping[str, NDArray[Any]]],
) -> list[Row]:
    """RSM's summary mass minus w/o memory's READ mass, positions 2-32."""
    seeds = sorted(
        seed for cell, seed, label in decisions if cell == METHOD and label == CAPTURE
    )
    rows: list[Row] = []
    if not seeds or any((REFERENCE, seed, CAPTURE) not in decisions for seed in seeds):
        return rows
    layers = _layers(decisions[(METHOD, seeds[0], CAPTURE)])
    for layer in range(layers):
        per_seed: dict[int, dict[int, float]] = {}
        for seed in seeds:
            means = []
            for cell in (METHOD, REFERENCE):
                arrays = decisions[(cell, seed, CAPTURE)]
                chosen = arrays["position"] >= 2
                means.append(
                    _task_means(
                        arrays["task_id"],
                        arrays["summary_layer"][:, layer].astype(np.float64),
                        chosen,
                    )
                )
            per_seed[seed] = {
                task: means[0][task] - means[1][task]
                for task in means[0]
                if task in means[1]
            }
        rows.append(
            _interval_row(
                "A1 RSM - w/o memory, summary mass, positions 2-32",
                per_seed,
                layer=layer,
            )
        )
    return rows


def reading_a2(
    decisions: Mapping[tuple[str, int, str], Mapping[str, NDArray[Any]]],
    labels: Mapping[int, Sequence[Row]],
) -> list[Row]:
    """CountRecall: summary-only evidence minus buffer-present evidence."""
    rows: list[Row] = []
    seeds = sorted(
        seed for cell, seed, label in decisions if cell == METHOD and label == CAPTURE
    )
    if not seeds:
        return rows
    layers = _layers(decisions[(METHOD, seeds[0], CAPTURE)])
    for layer in range(layers):
        per_seed: dict[int, dict[int, float]] = {}
        raw: dict[str, list[float]] = {"summary_only": [], "buffer_present": []}
        for seed in seeds:
            arrays = decisions[(METHOD, seed, CAPTURE)]
            residual = residual_mass(arrays, layer)
            before: dict[tuple[int, int], int] = {}
            inside: dict[tuple[int, int], int] = {}
            for event in labels[seed]:
                if event["kind"] != "query":
                    continue
                key = (int(event["task_id"]), int(event["step"]) - 1)
                before[key] = int(event["count_before_current_segment"])
                inside[key] = int(event["count_in_current_segment"])
            keys = list(
                zip(arrays["task_id"].tolist(), arrays["step"].tolist(), strict=True)
            )
            later = arrays["segment"] >= 1
            only = (
                np.asarray(
                    [before.get(k, 0) > 0 and inside.get(k, 1) == 0 for k in keys],
                    dtype=np.bool_,
                )
                & later
            )
            present = (
                np.asarray([inside.get(k, 0) > 0 for k in keys], dtype=np.bool_) & later
            )
            mass = arrays["summary_layer"][:, layer].astype(np.float64)
            raw["summary_only"].extend(mass[only].tolist())
            raw["buffer_present"].extend(mass[present].tolist())
            first = _task_means(arrays["task_id"], residual, only)
            second = _task_means(arrays["task_id"], residual, present)
            per_seed[seed] = {
                task: first[task] - second[task] for task in first if task in second
            }
        rows.append(
            _interval_row(
                "A2 summary-only - buffer-present evidence, "
                "position-adjusted summary mass",
                per_seed,
                layer=layer,
                summary_only_mean=float(np.mean(raw["summary_only"])),
                buffer_present_mean=float(np.mean(raw["buffer_present"])),
            )
        )
    return rows


def reading_a3(
    decisions: Mapping[tuple[str, int, str], Mapping[str, NDArray[Any]]],
    labels: Mapping[int, Sequence[Row]],
) -> list[Row]:
    """Key-to-Door: the first eight steps of later attempts minus attempt 1's."""
    rows: list[Row] = []
    seeds = sorted(
        seed for cell, seed, label in decisions if cell == METHOD and label == CAPTURE
    )
    if not seeds:
        return rows
    layers = _layers(decisions[(METHOD, seeds[0], CAPTURE)])
    for layer in range(layers):
        per_seed: dict[int, dict[int, float]] = {}
        for seed in seeds:
            arrays = decisions[(METHOD, seed, CAPTURE)]
            residual = residual_mass(arrays, layer)
            first_attempt: set[tuple[int, int]] = set()
            later_attempts: set[tuple[int, int]] = set()
            for event in labels[seed]:
                if event["kind"] != "attempt":
                    continue
                task = int(event["task_id"])
                start, end = int(event["start_step"]), int(event["end_step"])
                steps = range(start - 1, min(start - 1 + ATTEMPT_START_STEPS, end))
                target = (
                    first_attempt if int(event["event_index"]) == 1 else later_attempts
                )
                target.update((task, step) for step in steps)
            keys = list(
                zip(arrays["task_id"].tolist(), arrays["step"].tolist(), strict=True)
            )
            first = np.asarray([k in first_attempt for k in keys], dtype=np.bool_)
            later = np.asarray([k in later_attempts for k in keys], dtype=np.bool_)
            one = _task_means(arrays["task_id"], residual, first)
            rest = _task_means(arrays["task_id"], residual, later)
            per_seed[seed] = {
                task: rest[task] - one[task] for task in rest if task in one
            }
        rows.append(
            _interval_row(
                "A3 later-attempt starts - attempt-1 starts, "
                "position-adjusted summary mass",
                per_seed,
                layer=layer,
            )
        )
    return rows


def _primary(
    tasks: Sequence[Row], cell: str, label: str
) -> dict[int, dict[int, float]]:
    out: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for row in tasks:
        if row["cell"] == cell and row["label"] == label:
            out[int(row["seed"])][int(row["task_id"])] = float(row["primary"])
    return dict(out)


def dose_rows(
    tasks: Sequence[Row],
    decisions: Mapping[tuple[str, int, str], Mapping[str, NDArray[Any]]],
) -> list[Row]:
    """B2/B3's display: primary and achieved masses at every read bias."""
    retained = _primary(tasks, METHOD, CAPTURE)
    labels = sorted(
        {
            (row["target"], float(row["beta"]), row["label"])
            for row in tasks
            if row["cell"] == METHOD
        },
        key=lambda item: (item[0] != "none", item[0], item[1]),
    )
    rows: list[Row] = []
    for target, beta, label in labels:
        biased = _primary(tasks, METHOD, label)
        if set(biased) != set(retained):
            continue
        _, level = _matrix(biased)
        mean, lower, upper = seed_task_interval(level)
        diff = {
            seed: {
                task: biased[seed][task] - retained[seed][task]
                for task in biased[seed]
                if task in retained[seed]
            }
            for seed in biased
        }
        row: Row = {
            "target": target,
            "beta": beta,
            "label": label,
            "primary": mean,
            "primary_lower": lower,
            "primary_upper": upper,
        }
        contrast = _interval_row("difference", diff)
        row.update(
            {
                "difference": contrast["estimate"],
                "difference_lower": contrast["lower"],
                "difference_upper": contrast["upper"],
                "difference_per_seed": contrast["per_seed"],
            }
        )
        for kind in ("summary", "buffer"):
            per_layer = [
                np.mean(
                    [
                        decisions[(METHOD, seed, label)][f"{kind}_layer"][
                            :, layer
                        ].mean()
                        for seed in biased
                    ]
                )
                for layer in range(
                    _layers(decisions[(METHOD, next(iter(biased)), label)])
                )
            ]
            for layer, value in enumerate(per_layer):
                row[f"{kind}_mass_layer{layer}"] = float(value)
        rows.append(row)
    return rows


def readings_b(tasks: Sequence[Row], *, benchmark: str) -> list[Row]:
    """B1 (summary blocked) with its declared rule, B3 (buffer blocked)."""
    delta = DELTAS[benchmark]
    retained = _primary(tasks, METHOD, CAPTURE)
    rows: list[Row] = []
    for name, label in (
        ("B1 summary reads blocked - retained", "summary-read-bias-inf"),
        ("B3 buffer reads blocked - retained", "buffer-read-bias-inf"),
    ):
        biased = _primary(tasks, METHOD, label)
        if not biased or set(biased) != set(retained):
            continue
        diff = {
            seed: {
                task: biased[seed][task] - retained[seed][task]
                for task in biased[seed]
                if task in retained[seed]
            }
            for seed in biased
        }
        row = _interval_row(name, diff, delta=delta)
        if name.startswith("B1"):
            row["declared_rule_holds"] = bool(
                row["estimate"] <= -delta
                and row["upper"] < 0
                and row["negative_seeds"] == len(biased)
            )
        rows.append(row)
    reference = _primary(tasks, REFERENCE, CAPTURE)
    if reference:
        _, level = _matrix(reference)
        mean, lower, upper = seed_task_interval(level)
        rows.append(
            {
                "reading": "w/o memory primary (reference)",
                "estimate": mean,
                "lower": lower,
                "upper": upper,
            }
        )
    return rows


def read_attention(
    directory: Path, *, benchmark: str, memory_tokens: int
) -> dict[str, list[Row]]:
    """Every declared reading and display table from one attention directory."""
    if benchmark not in DELTAS:
        raise ResultValidationError(
            f"No attention reading is declared for {benchmark!r}."
        )
    decisions = load_decisions(directory)
    tasks = load_tasks(directory / "tasks.csv")
    labels = {
        int(match["seed"]): load_labels(path)
        for path in sorted(directory.glob(f"labels-{METHOD}-seed*.csv"))
        if (
            match := re.fullmatch(rf"labels-{METHOD}-seed(?P<seed>\d+)\.csv", path.name)
        )
    }
    readings = reading_a1(decisions)
    if benchmark == "count_recall":
        readings += reading_a2(decisions, labels)
    else:
        readings += reading_a3(decisions, labels)
    readings += readings_b(tasks, benchmark=benchmark)
    return {
        "readings": readings,
        "profile": profile_rows(decisions, memory_tokens=memory_tokens),
        "dose": dose_rows(tasks, decisions),
    }
