"""Stratified uncertainty and fixed-clock learning summaries for match-pattern."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.match_pattern import BOOTSTRAP_DRAWS, GENERATOR_SEED
from reasoned_icrl.experiments.records import BenchmarkEvent, Cell


def stratified_interval(
    cells: Mapping[Cell, float],
    events: Sequence[BenchmarkEvent],
    *,
    samples: int = BOOTSTRAP_DRAWS,
    seed: int = GENERATOR_SEED,
    crossed: bool = False,
) -> tuple[float, float, float]:
    """Pair example draws across methods/seeds, preserving pattern-pair counts.

    ``cells`` may contain scores or paired differences. Conditional intervals
    fix the fitted seeds; crossed intervals also resample fitted seeds. Call
    separately for different populations: their example identities are unpaired.
    """
    seeds = sorted({s for s, _, _ in cells})
    units = sorted({(t, r) for _, t, r in cells})
    if not seeds or not units or len(cells) != len(seeds) * len(units):
        raise ContractError("Match-pattern bootstrap needs a rectangular seed panel.")
    strata = {
        (e.task_id, e.rollout_seed): (e.left_pattern, e.right_pattern) for e in events
    }
    if any(u not in strata or None in strata[u] for u in units):
        raise ContractError("Match-pattern bootstrap requires pattern-pair strata.")
    matrix = np.asarray([[cells[(s, *u)] for u in units] for s in seeds])
    groups = [
        [i for i, u in enumerate(units) if strata[u] == pair]
        for pair in sorted(set(strata[u] for u in units))
    ]
    populations = {event.split for event in events}
    if len(populations) != 1:
        raise ContractError("IID and binding examples are separate populations.")
    population = next(iter(populations))
    stream = int.from_bytes(hashlib.sha256(population.encode()).digest()[:4], "big")
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    seed_rng = np.random.default_rng(seed)  # shared seed draws across populations
    draws = np.empty(samples)
    # Bounded batches avoid samples x seeds x full-panel temporary tensors.
    for start in range(0, samples, 128):
        n = min(128, samples - start)
        picks = np.concatenate([rng.choice(g, (n, len(g))) for g in groups], axis=1)
        values = matrix[:, picks].mean(axis=2).T
        if crossed:
            selected = seed_rng.integers(len(seeds), size=(n, len(seeds)))
            values = np.take_along_axis(values, selected, axis=1)
        draws[start : start + n] = values.mean(axis=1)
    estimate = float(matrix.mean())
    if samples == 0:
        return estimate, estimate, estimate
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def learning_summary(points: Mapping[int, float]) -> dict[str, Any]:
    """Normalized AUC and observed, censored 90/95% crossings on the frozen grid."""
    expected = (0, *(8000 * (i + 1) for i in range(50, 1000, 50)), 8_000_000)
    missing = sorted(set(expected) - set(points))
    result: dict[str, Any] = {"complete": not missing, "missing_calls": missing}
    ordered = sorted((x, y) for x, y in points.items() if 0 <= x <= 8_000_000)
    for threshold in (0.9, 0.95):
        hits = [x for x, y in ordered if y >= threshold]
        result[f"crossing_{int(threshold * 100)}"] = {
            "calls": min(hits) if hits else None,
            "right_censored": not hits,
            "last_observed_calls": max(points, default=0),
            "previous_observed_calls": max(
                (x for x, _ in ordered if hits and x < min(hits)), default=None
            ),
            "resolution": "observed checkpoints; nominal 400000 calls",
        }
    for endpoint in (8_000_000, 2_000_000):
        key = f"normalized_auc_0_{endpoint}"
        if missing:
            result[key] = None
            continue
        x, y = np.asarray(ordered).T
        selected = x < endpoint
        xx = np.append(x[selected], endpoint)
        yy = np.append(y[selected], np.interp(endpoint, x, y))
        result[key] = float(np.sum((yy[1:] + yy[:-1]) * np.diff(xx) / 2) / endpoint)
    return result
