"""The summary-length read of the method on Key-to-Door.

    python scripts/capacity_read.py [--cell raw_summary] [--split confirmation] \
        [--horizons 500 1000 2000 4000] [--delta 1.0] [--out DIR]

Reads the frozen-endpoint horizon panels (``<split>-retained-endpoint-h<H>``)
of one cell at every summary length M of the ablation -- the paper's M = 4
under the 8M root and the arms M = 1, 8 and 16 under their own roots -- turns
each into completed doors per 500-call window for every (task, rollout seed)
unit of the roster, and pairs every arm with M = 4 on the same unit and seed.
Writes under ``--out`` (default ``<8M root>/reports/dark-key-to-door/<split>/
capacity/<cell>/``):

``windows.csv``
    doors per window per M and horizon: the joint seed/task bootstrap estimate
    and 95 % interval, the three seeds beside it;
``contrasts.csv``
    RSM M - RSM M4 per horizon in the first window (the trained 500 calls) and
    the last window, and in the doors summed over the horizon: the joint
    interval, the per-seed means and the declared flags;
``capacity.json``
    the declared reading per arm (kept for the method's ladder): inside +-delta at 500
    and in the last window at the
    longest horizon on every seed = M = 4 is not a tuned point; M above M4 by
    at least delta (joint interval above zero, every seed) = more summary
    helps; M below M4 by at least delta = the arm is harder to learn at
    matched experience.

An arm whose panels are incomplete at a horizon is reported and skipped;
nothing is projected. Runs are resolved through each study file, so the read
works on any copy of the study roots. Nothing here evaluates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reasoned_icrl.analysis.statistics import _bootstrap, _interval
from reasoned_icrl.experiments.evaluation import RESULTS_FILE, evaluation_directory
from reasoned_icrl.experiments.horizon import WINDOW, window_matrix
from reasoned_icrl.experiments.records import read_benchmark_results, results_file
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve
from reasoned_icrl.utils import repository_root

BENCHMARK = "dark_key_to_door"
PAPER_STUDY = Path("configs/summary_memory_8m.yaml")
"""The paper's M = 4 cells."""
ARMS = (
    Path("configs/keydoor_capacity_m1_8m.yaml"),
    Path("configs/keydoor_capacity_m8_8m.yaml"),
    Path("configs/keydoor_capacity_8m.yaml"),
)
"""The summary-length arms, M = 1, 8 and 16 (each its own root)."""
HORIZONS = (500, 1000, 2000, 4000)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument("--cell", default="raw_summary")
    result.add_argument("--split", default="confirmation")
    result.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    result.add_argument("--arms", type=Path, nargs="+", default=list(ARMS))
    result.add_argument("--paper-study", type=Path, default=PAPER_STUDY)
    result.add_argument("--delta", type=float, default=1.0)
    result.add_argument("--samples", type=int, default=2000)
    result.add_argument("--confidence", type=float, default=0.95)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--out", type=Path, default=None)
    return result


def _cube(
    study_file: Path, cell: str, split: str, horizon: int
) -> tuple[int, tuple[int, ...], np.ndarray | None]:
    """``(M, seeds, [seeds, units, windows])`` of one study's cell at one
    horizon, or ``(M, seeds, None)`` when a seed's panel is missing."""
    study = load_summary_memory_study(study_file)
    contract = study.contract(BENCHMARK)
    units = [
        (int(task), int(rollout))
        for task in contract.roster(split)
        for rollout in contract.evaluation.rollout_seeds
    ]
    rows: list[np.ndarray] = []
    memory = 0
    seeds = tuple(study.training_seeds)
    for seed in seeds:
        _, config = resolve(
            study, benchmark=BENCHMARK, condition=cell, seed=seed, device="cpu"
        )
        assert config.model.summary is not None
        memory = int(config.model.summary.memory_tokens)
        path = (
            config.run_directory
            / "eval"
            / evaluation_directory(split, "retained", "endpoint", horizon=horizon)
            / RESULTS_FILE
        )
        if results_file(path) is None:  # the plain file or its compacted twin
            return memory, seeds, None
        runs, events = read_benchmark_results(path, [contract])
        (run,) = runs
        if run.status != "completed" or run.split != split:
            return memory, seeds, None
        rows.append(window_matrix(events, units=units, outer_length=horizon))
    return memory, seeds, np.stack(rows)


def _estimate(
    matrix: np.ndarray,
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Joint seed/task interval and the per-seed means of a [seeds, units] matrix."""
    estimate = float(matrix.mean())
    lower, upper = _interval(
        _bootstrap(matrix, samples=samples, rng=rng), estimate, confidence
    )
    per_seed = {int(s): float(matrix[i].mean()) for i, s in enumerate(seeds)}
    return {"estimate": estimate, "lower": lower, "upper": upper, "per_seed": per_seed}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    delta = float(args.delta)
    paper = load_summary_memory_study(args.paper_study)
    out = args.out or (
        repository_root()
        / paper.output_root
        / "reports"
        / BENCHMARK.replace("_", "-")
        / args.split
        / "capacity"
        / args.cell
    )

    def est(matrix: np.ndarray, seeds: tuple[int, ...]) -> dict[str, Any]:
        return _estimate(
            matrix,
            samples=args.samples,
            confidence=args.confidence,
            rng=rng,
            seeds=seeds,
        )

    window_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    readings: dict[str, dict[str, Any]] = {}
    for horizon in args.horizons:
        base_m, seeds, base = _cube(args.paper_study, args.cell, args.split, horizon)
        if base is None:
            print(f"M = {base_m}: panels incomplete at h{horizon}, skipped", flush=True)
            continue
        cubes = {base_m: base}
        for arm in args.arms:
            m, arm_seeds, cube = _cube(arm, args.cell, args.split, horizon)
            if cube is None:
                print(f"M = {m} ({arm.name}): panels incomplete at h{horizon}, skipped")
                continue
            if arm_seeds != seeds:
                raise SystemExit(f"{arm}: seeds {arm_seeds} differ from {seeds}")
            cubes[m] = cube
        windows = horizon // WINDOW
        for m in sorted(cubes):
            for w in range(windows):
                e = est(cubes[m][:, :, w], seeds)
                window_rows.append(
                    {
                        "cell": args.cell,
                        "memory_tokens": m,
                        "horizon": horizon,
                        "window": w + 1,
                        "estimate": e["estimate"],
                        "lower": e["lower"],
                        "upper": e["upper"],
                        **{f"seed_{s}": v for s, v in e["per_seed"].items()},
                    }
                )
        for m in sorted(cubes):
            if m == base_m:
                continue
            for label, left, right in (
                ("first window", cubes[m][:, :, 0], base[:, :, 0]),
                ("last window", cubes[m][:, :, -1], base[:, :, -1]),
                ("doors over the horizon", cubes[m].sum(2), base.sum(2)),
            ):
                e = est(left - right, seeds)
                per = e["per_seed"]
                scale = 1 if label != "doors over the horizon" else windows
                row = {
                    "contrast": f"RSM M{m} - RSM M{base_m} ({label})",
                    "memory_tokens": m,
                    "horizon": horizon,
                    "measure": label,
                    "estimate": e["estimate"],
                    "lower": e["lower"],
                    "upper": e["upper"],
                    **{f"seed_{s}": v for s, v in per.items()},
                    "delta": delta * scale,
                    "within_delta_every_seed": all(
                        abs(v) <= delta * scale for v in per.values()
                    ),
                    "gain_every_seed": e["estimate"] >= delta * scale
                    and e["lower"] > 0
                    and all(v > 0 for v in per.values()),
                    "loss_every_seed": e["estimate"] <= -delta * scale
                    and e["upper"] < 0
                    and all(v < 0 for v in per.values()),
                }
                contrast_rows.append(row)
                if label != "doors over the horizon":
                    readings.setdefault(f"M{m}", {})[f"h{horizon} {label}"] = {
                        k: row[k]
                        for k in (
                            "estimate",
                            "lower",
                            "upper",
                            "within_delta_every_seed",
                            "gain_every_seed",
                            "loss_every_seed",
                        )
                    } | {"per_seed": per}
    if not window_rows:
        print("capacity_read: no complete panel at any horizon.", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("windows", window_rows), ("contrasts", contrast_rows)):
        if not rows:
            continue
        with (out / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    longest = max(args.horizons)
    summary: dict[str, Any] = {
        "cell": args.cell,
        "split": args.split,
        "delta": delta,
        "horizons": args.horizons,
        "reading_rule": (
            "inside +-delta at 500 calls and in the last window at the longest "
            "horizon on every seed: M = 4 is not a tuned point; above M4 by at "
            "least delta (joint interval above zero, every seed): more summary "
            "helps; below M4 by at least delta: the arm is harder to learn at "
            "matched experience"
        ),
        "arms": {},
    }
    for arm, by in readings.items():
        trained = by.get("h500 first window")
        last = by.get(f"h{longest} last window")
        verdict = "incomplete"
        if trained and last:
            if trained["within_delta_every_seed"] and last["within_delta_every_seed"]:
                verdict = "within delta: M = 4 is not a tuned point"
            elif last["gain_every_seed"] or trained["gain_every_seed"]:
                verdict = "more summary helps"
            elif last["loss_every_seed"] or trained["loss_every_seed"]:
                verdict = "harder to learn at matched experience"
            else:
                verdict = "inconclusive"
        summary["arms"][arm] = {"verdict": verdict, **by}
        print(f"{arm}: {verdict}", flush=True)
    (out / "capacity.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in ("windows.csv", "contrasts.csv", "capacity.json"):
        if (out / path).exists():
            print(f"written {out / path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
