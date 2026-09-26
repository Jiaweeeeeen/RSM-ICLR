"""Collect evaluation records, costs and secondary summaries from run directories.

Nothing here trains, evaluates or reads a checkpoint: the readers walk
``<root>/<protocol>/<condition>/seed-N`` for the saved ``benchmark_results.json``
panels, the ``systems.json``/``metrics.json`` cost measurements (Table 3) and
the per-run secondary summaries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from reasoned_icrl.experiments.benchmarks import BenchmarkContract, Study
from reasoned_icrl.experiments.contracts import ResultValidationError
from reasoned_icrl.experiments.evaluation import (
    RESULTS_FILE,
    SECONDARY_FILE,
    evaluation_directory,
)
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    BenchmarkRun,
    read_benchmark_results,
)
from reasoned_icrl.experiments.resumes import gpu_chain, session_hours


@dataclass(frozen=True, slots=True)
class RunRecords:
    """Everything one evaluated run directory contributes to a notebook."""

    directory: Path
    condition: str
    seed: int
    run: BenchmarkRun | None
    events: tuple[BenchmarkEvent, ...]
    secondary: Mapping[str, float | None]
    partial: bool


def run_directories(
    study: Study, contract: BenchmarkContract, root: str | Path
) -> list[tuple[str, int, Path]]:
    """Every ``<root>/<protocol>/<condition>/seed-N`` that holds a checkpoint.

    Only the cells the study runs on this contract are read (its tier's
    groups on a tiered study), so an excluded cell never appears as a run. A
    comparator tier's frozen reference cells are read from its
    ``reference_root`` at the study's training seeds: their panels are paired
    with the tier's fits, their fits belong to the other study."""
    found = []
    base = Path(root) / contract.protocol
    for condition in (*study.cells(contract), "feedforward"):
        for directory in sorted(base.glob(f"{condition}/seed-*")):
            if (directory / "checkpoint.pt").is_file():
                found.append((condition, int(directory.name.split("-")[1]), directory))
    if study.tiered:
        plan = study.tier(contract.name)
        for condition in plan.reference_cells:
            reference = study.cell_root(contract, condition, root) / contract.protocol
            for seed in study.training_seeds:
                directory = reference / condition / f"seed-{seed}"
                if (directory / "checkpoint.pt").is_file():
                    found.append((condition, seed, directory))
    return found


def collect(
    study: Study,
    contract: BenchmarkContract,
    root: str | Path,
    *,
    split: str = "development",
    history: str = "retained",
    checkpoint_rule: str = "selected",
) -> list[RunRecords]:
    """Read the evaluation of every run under ``root`` for one split/history.

    ``checkpoint_rule="final-epoch"`` reads the fixed-final-checkpoint
    supplement's panel instead of the development-selected one.
    """
    records = []
    for condition, seed, directory in run_directories(study, contract, root):
        evaluation = (
            directory / "eval" / evaluation_directory(split, history, checkpoint_rule)
        )
        results = evaluation / RESULTS_FILE
        if not results.is_file():
            records.append(RunRecords(directory, condition, seed, None, (), {}, False))
            continue
        raw = json.loads(results.read_text())
        partial = "partial_task_cap" in raw
        if partial:
            run = BenchmarkRun(**raw["runs"][0])
            events = tuple(BenchmarkEvent(**row) for row in raw["events"])
        else:
            runs, events = read_benchmark_results(results, [contract])
            run = runs[0]
        secondary_path = evaluation / SECONDARY_FILE
        secondary = (
            json.loads(secondary_path.read_text()) if secondary_path.is_file() else {}
        )
        records.append(
            RunRecords(directory, condition, seed, run, events, secondary, partial)
        )
    return records


def complete_events(records: Iterable[RunRecords]) -> tuple[BenchmarkEvent, ...]:
    """Events of completed, full-roster evaluations only."""
    return tuple(
        event
        for record in records
        if record.run is not None and not record.partial
        for event in record.events
    )


@dataclass(frozen=True, slots=True)
class CostRow:
    """Table 3 for one condition: allocated state, latencies and training cost.

    Allocated bytes and parameter counts are identities of the cell and must
    agree across its seeds; latencies and training hours are seed means with
    the per-seed hours kept; ``peak_gpu_bytes`` is the largest seed's. A
    field is ``None`` when no run of the condition measured it (a
    ``feedforward`` run writes no ``systems.json``).
    """

    protocol: str
    condition: str
    seeds: tuple[int, ...]
    persistent_state_bytes: int | None
    cache_bytes: int | None
    decision_latency_seconds: float | None
    decision_latency_p95_seconds: float | None
    boundary_latency_seconds: float | None
    boundary_latency_p95_seconds: float | None
    boundary_latency_max_seconds: float | None
    training_hours: float | None
    per_seed_training_hours: Mapping[int, float]
    parameters: int | None
    peak_gpu_bytes: int | None


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else None


def _shared_integer(
    source: Mapping[int, Mapping[str, object]], key: str, *, label: str
) -> int | None:
    """One integer every seed agrees on (an identity of the cell), or None."""
    values = {
        int(row[key])  # type: ignore[call-overload]
        for row in source.values()
        if row.get(key) is not None
    }
    if len(values) > 1:
        raise ResultValidationError(f"{label}: {key} differs across seeds.")
    return next(iter(values), None)


def _seed_mean(source: Mapping[int, Mapping[str, object]], key: str) -> float | None:
    values = [
        float(row[key])  # type: ignore[arg-type]
        for row in source.values()
        if row.get(key) is not None
    ]
    return float(np.mean(values)) if values else None


def cost_table(
    study: Study, contract: BenchmarkContract, root: str | Path
) -> tuple[CostRow, ...]:
    """One :class:`CostRow` per condition with a run under ``root`` (Table 3)."""
    grouped: defaultdict[str, list[tuple[int, Path]]] = defaultdict(list)
    for condition, seed, directory in run_directories(study, contract, root):
        grouped[condition].append((seed, directory))
    rows: list[CostRow] = []
    for condition in (*study.compared_cells(contract), "feedforward"):
        if condition not in grouped:
            continue
        systems: dict[int, dict[str, object]] = {}
        metrics: dict[int, dict[str, object]] = {}
        for seed, directory in grouped[condition]:
            measured = _read_json(directory / "systems.json")
            if measured is not None:
                systems[seed] = measured
            trained = _read_json(directory / "metrics.json")
            if trained is not None:
                metrics[seed] = trained

        label = f"{contract.protocol}/{condition}"
        hours = {
            seed: float(row["runtime_seconds"]) / 3600.0  # type: ignore[arg-type]
            for seed, row in sorted(metrics.items())
            if row.get("runtime_seconds") is not None
        }
        peaks = [
            int(row["peak_gpu_bytes"])  # type: ignore[call-overload]
            for row in systems.values()
            if row.get("peak_gpu_bytes") is not None
        ]
        rows.append(
            CostRow(
                protocol=contract.protocol,
                condition=condition,
                seeds=tuple(sorted(seed for seed, _ in grouped[condition])),
                persistent_state_bytes=_shared_integer(
                    systems, "persistent_state_bytes", label=label
                ),
                cache_bytes=_shared_integer(systems, "cache_bytes", label=label),
                decision_latency_seconds=_seed_mean(
                    systems, "decision_latency_seconds"
                ),
                decision_latency_p95_seconds=_seed_mean(
                    systems, "decision_latency_p95_seconds"
                ),
                boundary_latency_seconds=_seed_mean(
                    systems, "boundary_latency_seconds"
                ),
                boundary_latency_p95_seconds=_seed_mean(
                    systems, "boundary_latency_p95_seconds"
                ),
                boundary_latency_max_seconds=_seed_mean(
                    systems, "boundary_latency_max_seconds"
                ),
                training_hours=float(np.mean(list(hours.values()))) if hours else None,
                per_seed_training_hours=hours,
                parameters=_shared_integer(metrics, "parameters", label=label),
                peak_gpu_bytes=max(peaks) if peaks else None,
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class SecondaryRow:
    """One secondary-summary key per condition: the seed mean and every seed."""

    condition: str
    key: str
    mean: float
    per_seed: Mapping[int, float]


def secondary_table(
    records: Iterable[RunRecords], keys: Sequence[str]
) -> tuple[SecondaryRow, ...]:
    """Seed means of named secondary summaries (e.g. MazeRunner's steps to the
    first goal) over completed, full-roster evaluations; a key a run does not
    carry, or carries as ``None``, is left out of that run's mean."""
    grouped: defaultdict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    conditions: list[str] = []
    for record in records:
        if record.run is None or record.partial:
            continue
        if record.condition not in conditions:
            conditions.append(record.condition)
        for key in keys:
            value = record.secondary.get(key)
            if isinstance(value, int | float):
                grouped[(record.condition, key)][record.seed] = float(value)
    return tuple(
        SecondaryRow(
            condition,
            key,
            float(np.mean(list(grouped[(condition, key)].values()))),
            dict(sorted(grouped[(condition, key)].items())),
        )
        for condition in conditions
        for key in keys
        if grouped.get((condition, key))
    )


@dataclass(frozen=True, slots=True)
class GpuCostRow:
    """Table 3 by GPU model (R4): hours and latency are never pooled across
    hardware; every run records its GPU in ``provenance.json``."""

    protocol: str
    condition: str
    gpu: str
    seeds: tuple[int, ...]
    training_hours: float | None
    per_seed_training_hours: Mapping[int, float]
    decision_latency_seconds: float | None
    peak_gpu_bytes: int | None
    gradient_steps: int | None
    charged_calls: int | None
    slurm_jobs: tuple[str, ...]


def _sessions(systems: Mapping[str, object]) -> list[Mapping[str, object]]:
    """The training sessions a resumed run records in ``systems.json`` (R6)."""
    raw = systems.get("sessions")
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return []
    return [cast(Mapping[str, object], s) for s in raw if isinstance(s, Mapping)]


def cost_by_gpu(
    study: Study, contract: BenchmarkContract, root: str | Path
) -> tuple[GpuCostRow, ...]:
    """One row per condition and GPU model with a completed run under ``root``.

    Reads ``provenance.json`` (GPU model, job), ``metrics.json`` (hours,
    gradient steps) and ``systems.json`` (latency, peak memory, measured
    charged calls). Runs without a GPU name are grouped under ``unknown``.
    """
    grouped: defaultdict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    for condition, seed, directory in run_directories(study, contract, root):
        provenance = _read_json(directory / "provenance.json") or {}
        sessions = _sessions(_read_json(directory / "systems.json") or {})
        # A resumed run trained on more than one GPU model: its row is keyed
        # by the chain of models (R6) so hours are never pooled across them.
        gpu = (
            gpu_chain(sessions) if sessions else str(provenance.get("gpu") or "unknown")
        )
        grouped[(condition, gpu)].append((seed, directory))
    rows: list[GpuCostRow] = []
    for (condition, gpu), members in sorted(grouped.items()):
        hours: dict[int, float] = {}
        latencies: list[float] = []
        peaks: list[int] = []
        steps: list[int] = []
        charged: list[int] = []
        jobs: list[str] = []
        for seed, directory in sorted(members):
            metrics = _read_json(directory / "metrics.json") or {}
            systems = _read_json(directory / "systems.json") or {}
            provenance = _read_json(directory / "provenance.json") or {}
            sessions = _sessions(systems)
            total_hours = session_hours(sessions) if sessions else None
            if total_hours is not None:
                hours[seed] = total_hours
            elif metrics.get("runtime_seconds") is not None:
                hours[seed] = float(metrics["runtime_seconds"]) / 3600.0  # type: ignore[arg-type]
            if metrics.get("gradient_steps") is not None:
                steps.append(int(metrics["gradient_steps"]))  # type: ignore[call-overload]
            if systems.get("decision_latency_seconds") is not None:
                latencies.append(float(systems["decision_latency_seconds"]))  # type: ignore[arg-type]
            if systems.get("peak_gpu_bytes") is not None:
                peaks.append(int(systems["peak_gpu_bytes"]))  # type: ignore[call-overload]
            measured = systems.get("measured")
            if isinstance(measured, Mapping) and measured.get("charged_calls"):
                charged.append(int(measured["charged_calls"]))
            if provenance.get("slurm_job_id"):
                jobs.append(str(provenance["slurm_job_id"]))
        rows.append(
            GpuCostRow(
                protocol=contract.protocol,
                condition=condition,
                gpu=gpu,
                seeds=tuple(seed for seed, _ in sorted(members)),
                training_hours=float(np.mean(list(hours.values()))) if hours else None,
                per_seed_training_hours=hours,
                decision_latency_seconds=float(np.mean(latencies))
                if latencies
                else None,
                peak_gpu_bytes=max(peaks) if peaks else None,
                gradient_steps=min(steps) if steps else None,
                charged_calls=min(charged) if charged else None,
                slurm_jobs=tuple(dict.fromkeys(jobs)),
            )
        )
    return tuple(rows)


def markdown_table(rows: Sequence[Mapping[str, object]], digits: int = 3) -> str:
    """Render mappings as a Markdown table for ``IPython.display.Markdown``."""
    if not rows:
        return "_no rows_"
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(
                f"{value:.{digits}f}" if isinstance(value, float) else str(value)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = [
    "CostRow",
    "GpuCostRow",
    "RunRecords",
    "SecondaryRow",
    "collect",
    "complete_events",
    "cost_by_gpu",
    "cost_table",
    "markdown_table",
    "run_directories",
    "secondary_table",
]
