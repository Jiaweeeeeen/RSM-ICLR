"""The representation displays and transplant readings on synthetic records
with planted effects.

The components recover a planted direction and the ridge read-out a planted
linear target (and not noise); the centred similarity is one within a task
whose summary never changes and near zero between independent summaries; the
window means recover a planted per-window mass; T2 finds the donor's key cell
only before the key is held and skips a donor key that is the task's own; T4's
donor-consistent score adds the donor's cards before the boundary to the
task's own since it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from reasoned_icrl.analysis.representation import (
    count_scores,
    donor_key_visits,
    keydoor_cells,
    principal_components,
    ridge_readout,
    similarity_rows,
    window_rows,
)

SEEDS = (42, 100)


def test_components_and_readout_recover_planted_structure() -> None:
    rng = np.random.default_rng(0)
    direction = rng.normal(size=32)
    direction /= np.linalg.norm(direction)
    latent = rng.normal(size=400)
    features = latent[:, None] * direction * 5.0 + 0.1 * rng.normal(size=(400, 32))
    _, components, shares = principal_components(features)
    assert abs(float(components[0] @ direction)) > 0.99
    assert shares[0] > 0.9
    target = np.stack([latent, rng.normal(size=400)], axis=1)
    _, scores = ridge_readout(
        features[:200], target[:200], features[200:], target[200:]
    )
    assert scores[0] > 0.95
    assert scores[1] < 0.2


def test_similarity_is_one_for_a_fixed_summary_and_near_zero_between_draws() -> None:
    rng = np.random.default_rng(1)
    tasks, segments, width = 40, 6, 64
    fixed = np.repeat(rng.normal(size=(tasks, 1, 2, width // 2)), segments, axis=1)
    fresh = rng.normal(size=(tasks, segments, 2, width // 2))

    def columns(memory: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "task_id": np.repeat(np.arange(tasks), segments),
            "segment": np.tile(np.arange(1, segments + 1), tasks),
            "memory": memory.reshape(tasks * segments, 2, width // 2),
        }

    rows, norms = similarity_rows(
        {"fixed": {42: columns(fixed)}, "fresh": {42: columns(fresh)}}
    )
    by_cell: dict[str, dict[tuple[int, int], float]] = {"fixed": {}, "fresh": {}}
    for row in rows:
        by_cell[row["cell"]][(row["segment"], row["other"])] = row["similarity"]
    assert by_cell["fixed"][(1, 5)] == pytest.approx(1.0, abs=1e-9)
    assert by_cell["fresh"][(2, 2)] == pytest.approx(1.0, abs=1e-9)
    assert abs(by_cell["fresh"][(1, 5)]) < 0.1
    assert len(norms) == 2 * segments


def test_window_means_recover_a_planted_mass_per_window() -> None:
    steps = np.tile(np.arange(1000), 3)
    tasks = np.repeat([7, 8, 9], 1000)
    mass = np.where(steps < 500, 0.2, 0.6)
    found = {
        "task_id": tasks,
        "step": steps,
        "summary_layer": np.stack([mass, mass / 2], axis=1),
    }
    rows = window_rows({"raw_summary": {seed: found for seed in SEEDS}})
    got = {(row["layer"], row["window"]): row["estimate"] for row in rows}
    assert got[(0, 1)] == pytest.approx(0.2)
    assert got[(0, 2)] == pytest.approx(0.6)
    assert got[(1, 2)] == pytest.approx(0.3)


def _packet(row: int, column: int, has_key: bool) -> list[float]:
    return [2 * row / 8 - 1, 2 * column / 8 - 1, 1.0 if has_key else -1.0, 0.0]


def test_the_donor_key_counts_only_before_the_key_is_held() -> None:
    assert keydoor_cells(np.asarray([_packet(3, 5, False)])).tolist() == [[3, 5]]
    layouts: dict[int, dict[str, tuple[int, int]]] = {
        1: {"key": (0, 0), "door": (7, 7), "start": (4, 4)},
        2: {"key": (2, 2), "door": (7, 0), "start": (4, 4)},
        3: {"key": (0, 0), "door": (1, 1), "start": (4, 4)},
    }
    donors = {1: 2, 2: 3, 3: 1}
    # Task 1 crosses the donor's key cell (2, 2) before holding its key; task 2
    # reaches the donor's key cell (0, 0) only after holding its own; task 3's
    # donor key is its own and is skipped.
    paths: dict[int, list[tuple[int, int, bool]]] = {
        1: [(4, 4, False), (3, 3, False), (2, 2, False), (1, 1, False), (0, 0, True)],
        2: [(4, 4, False), (2, 2, True), (1, 1, True), (0, 0, True), (0, 1, True)],
        3: [(4, 4, False), (0, 0, True), (0, 0, True), (0, 0, True), (0, 0, True)],
    }
    labels: list[dict[str, Any]] = []
    rows: dict[str, list[Any]] = {
        "task_id": [],
        "step": [],
        "current": [],
        "action": [],
    }
    for task, path in paths.items():
        # Attempt 1 spans decisions 0-9, attempt 2 decisions 11-14 (one-based
        # event steps 12-15); the boundary sits at decision 10.
        labels += [
            {
                "kind": "attempt",
                "task_id": task,
                "event_index": 1,
                "start_step": 1,
                "end_step": 10,
            },
            {
                "kind": "attempt",
                "task_id": task,
                "event_index": 2,
                "start_step": 12,
                "end_step": 15,
            },
        ]
        for step in range(11):
            rows["task_id"].append(task)
            rows["step"].append(step)
            rows["current"].append(_packet(7, 7, False))
            rows["action"].append(0)
        for offset, (row, column, held) in enumerate(path):
            rows["task_id"].append(task)
            rows["step"].append(11 + offset)
            rows["current"].append(_packet(row, column, held))
            rows["action"].append(0)
    inputs = {name: np.asarray(values) for name, values in rows.items()}
    visits = donor_key_visits(labels, inputs, layouts, donors, boundary_step=10)
    assert visits == {1: 1.0, 2: 0.0}


def test_the_donor_consistent_score_adds_the_donors_prefix() -> None:
    def decode(current: np.ndarray) -> tuple[int, int]:
        return int(current[0]), int(current[1])

    # Two tasks, eight decisions, the boundary at decision 4. Task 10 deals
    # category 0 four times before the boundary, task 11 never.
    deals = {10: [0, 0, 0, 0, 1, 1, 0, 1], 11: [1, 1, 1, 1, 0, 1, 1, 0]}
    rows: dict[str, list[Any]] = {
        "task_id": [],
        "step": [],
        "current": [],
        "action": [],
    }
    for task, values in deals.items():
        source = 11 if task == 10 else 10
        for step, value in enumerate(values):
            own = values[: step + 1].count(0)
            consistent = deals[source][:4].count(0) + values[4 : step + 1].count(0)
            rows["task_id"].append(task)
            rows["step"].append(step)
            rows["current"].append([value, 0])
            # Answers follow the donor-consistent count after the boundary.
            rows["action"].append(consistent if step >= 4 else own)
    inputs = {name: np.asarray(values) for name, values in rows.items()}
    true, _ = count_scores(inputs, decode, first_step=4)
    donor, per_step = count_scores(
        inputs,
        decode,
        first_step=4,
        donors={10: 11, 11: 10},
        donor_inputs=inputs,
        boundary_step=4,
    )
    assert donor == {10: 1.0, 11: 1.0}
    assert true[10] == 0.0 and true[11] == 0.0
    assert np.isnan(per_step[10][:4]).all()
