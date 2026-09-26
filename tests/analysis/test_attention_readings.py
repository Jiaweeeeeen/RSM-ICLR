"""The attention readings on synthetic records with planted effects.

A1 recovers a planted summary-mass gap to the reference; A2 recovers a planted
gap between evidence held only in the summary and evidence in the buffer after
the position adjustment, and ignores a position trend both groups share; A3
reads attempt starts from the events' one-based steps; B1 applies its declared
rule; the dose table pairs every bias with the retained capture.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reasoned_icrl.analysis.attention import (
    CAPTURE,
    METHOD,
    REFERENCE,
    dose_rows,
    load_tasks,
    reading_a1,
    reading_a2,
    reading_a3,
    readings_b,
)

SEEDS = (42, 100)
TASKS = (5, 6, 7, 8)
C = 32


def _decisions(
    summary: np.ndarray, tasks: np.ndarray, steps: np.ndarray
) -> dict[str, np.ndarray]:
    layers = np.stack([summary, summary], axis=1).astype(np.float32)
    return {
        "task_id": tasks,
        "step": steps,
        "segment": steps // C,
        "position": steps % C + 1,
        "summary_layer": layers,
        "buffer_layer": 1.0 - layers,
        "own_layer": np.zeros_like(layers),
    }


def _stream(length: int) -> tuple[np.ndarray, np.ndarray]:
    tasks = np.repeat(np.asarray(TASKS), length)
    steps = np.tile(np.arange(length), len(TASKS))
    return tasks, steps


def test_a1_recovers_a_planted_gap_to_the_reference() -> None:
    tasks, steps = _stream(2 * C)
    trend = 1.0 / (1.0 + (steps % C))
    found: dict[tuple[str, int, str], Any] = {}
    for seed in SEEDS:
        found[(METHOD, seed, CAPTURE)] = _decisions(trend + 0.1, tasks, steps)
        found[(REFERENCE, seed, CAPTURE)] = _decisions(trend, tasks, steps)
    rows = reading_a1(found)
    assert [row["layer"] for row in rows] == [0, 1]
    for row in rows:
        assert row["estimate"] == pytest.approx(0.1, abs=1e-6)
        assert row["positive_seeds"] == len(SEEDS)


def test_a2_reads_the_evidence_gap_after_the_position_adjustment() -> None:
    tasks, steps = _stream(3 * C)
    # Alternate the evidence group across tasks and steps, so every task and
    # every (segment, position) cell holds both groups; plant +0.2 on evidence
    # held only in the summary on top of a position trend both groups share.
    only = ((steps + tasks) % 2 == 0) & (steps >= C)
    summary = 0.8 / (1.0 + (steps % C)) + 0.2 * only
    labels: dict[int, list[dict[str, Any]]] = {}
    found: dict[tuple[str, int, str], Any] = {}
    for seed in SEEDS:
        found[(METHOD, seed, CAPTURE)] = _decisions(summary, tasks, steps)
        labels[seed] = [
            {
                "kind": "query",
                "task_id": int(task),
                "step": int(step) + 1,
                "count_before_current_segment": 3 if step >= C else 0,
                "count_in_current_segment": 0 if is_only else 1,
            }
            for task, step, is_only in zip(tasks, steps, only, strict=True)
        ]
    rows = reading_a2(found, labels)
    for row in rows:
        assert row["estimate"] == pytest.approx(0.2, abs=1e-6)
        assert row["tasks"] == len(TASKS)


def test_a3_reads_attempt_starts_from_one_based_event_steps() -> None:
    tasks, steps = _stream(2 * C)
    # Attempt 2 starts at a different step in every task, as attempt lengths
    # differ in the rollouts, so the position adjustment can separate it.
    offset = {task: 4 * index for index, task in enumerate(TASKS)}
    start = np.asarray([20 + offset[int(task)] for task in tasks])
    later = (steps >= start) & (steps < start + 8)
    summary = 0.3 + 0.5 * later
    labels: dict[int, list[dict[str, Any]]] = {}
    found: dict[tuple[str, int, str], Any] = {}
    for seed in SEEDS:
        found[(METHOD, seed, CAPTURE)] = _decisions(summary, tasks, steps)
        labels[seed] = [
            row
            for task in TASKS
            for row in (
                {
                    "kind": "attempt",
                    "task_id": task,
                    "event_index": 1,
                    "start_step": 1,
                    "end_step": 19 + offset[task],
                },
                {
                    "kind": "attempt",
                    "task_id": task,
                    "event_index": 2,
                    "start_step": 21 + offset[task],
                    "end_step": 50,
                },
            )
        ]
    rows = reading_a3(found, labels)
    for row in rows:
        assert row["estimate"] > 0.1
        assert row["positive_seeds"] == len(SEEDS)


def _tasks_file(tmp_path: Path, blocked: float) -> Path:
    path = tmp_path / "tasks.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell",
                "seed",
                "label",
                "target",
                "beta",
                "task_id",
                "primary",
            ],
        )
        writer.writeheader()
        for seed in SEEDS:
            for task in TASKS:
                for label, target, beta, value in (
                    ("retained", "none", 0.0, 30.0),
                    ("summary-read-bias-inf", "summary", float("inf"), blocked),
                    ("buffer-read-bias-inf", "buffer", float("inf"), 20.0),
                ):
                    writer.writerow(
                        {
                            "cell": METHOD,
                            "seed": seed,
                            "label": label,
                            "target": target,
                            "beta": beta,
                            "task_id": task,
                            "primary": value,
                        }
                    )
                writer.writerow(
                    {
                        "cell": REFERENCE,
                        "seed": seed,
                        "label": "retained",
                        "target": "none",
                        "beta": 0.0,
                        "task_id": task,
                        "primary": 2.0,
                    }
                )
    return path


@pytest.mark.parametrize("blocked,holds", [(2.0, True), (29.5, False)])
def test_b1_applies_its_declared_rule(
    tmp_path: Path, blocked: float, holds: bool
) -> None:
    rows = readings_b(
        load_tasks(_tasks_file(tmp_path, blocked)), benchmark="dark_key_to_door"
    )
    b1 = next(row for row in rows if row["reading"].startswith("B1"))
    assert b1["estimate"] == pytest.approx(blocked - 30.0)
    assert b1["declared_rule_holds"] is holds
    reference = next(row for row in rows if row["reading"].startswith("w/o memory"))
    assert reference["estimate"] == pytest.approx(2.0)


def test_the_dose_table_pairs_every_bias_with_the_capture(tmp_path: Path) -> None:
    tasks = load_tasks(_tasks_file(tmp_path, 2.0))
    steps = np.arange(C)
    found = {
        (METHOD, seed, label): _decisions(
            np.full(C, value), np.full(C, TASKS[0]), steps
        )
        for seed in SEEDS
        for label, value in (
            ("retained", 0.4),
            ("summary-read-bias-inf", 0.0),
            ("buffer-read-bias-inf", 0.6),
        )
    }
    rows = dose_rows(tasks, found)
    by_label = {row["label"]: row for row in rows}
    assert by_label["retained"]["difference"] == pytest.approx(0.0)
    assert by_label["summary-read-bias-inf"]["difference"] == pytest.approx(-28.0)
    assert by_label["summary-read-bias-inf"]["summary_mass_layer0"] == pytest.approx(
        0.0
    )
    assert by_label["buffer-read-bias-inf"]["summary_mass_layer0"] == pytest.approx(0.6)
