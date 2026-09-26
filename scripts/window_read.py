"""Read the sliding-window baseline's declared MazeRunner rows.

The study protocol declares three companion rows
for `raw_window` (RSM - window, window - RSM no carry, full history - window)
and two MazeRunner readings: W1, the goal fraction of the trained task on the
confirmation roster (delta 0.1), and W3, goals per 500-call window in the last
window of the repeated-laps panels at 4,000 calls under the retained and the
lap-cleared histories (delta 1 goal). The Key-to-Door rows (W1 and W2) are
written by `scripts/horizon_read.py`.

    uv run --no-sync python scripts/window_read.py [--study configs/mazerunner_8m.yaml]

Every estimate pools the three seeds and the 256 roster units with the
repository's joint seed/unit bootstrap (`analysis.statistics._bootstrap`), and
every contrast is paired on the same (seed, unit). The method's own rows are
printed beside the tier report's and the laps reader's so the read can be
checked against them. Writes ``native.csv`` and ``laps.csv`` under
``<root>/reports/<benchmark>/<split>/window/``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reasoned_icrl.analysis.statistics import _bootstrap, _interval
from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.evaluation import RESULTS_FILE, evaluation_directory
from reasoned_icrl.experiments.records import read_benchmark_results, results_file
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.utils import repository_root

WINDOW = "raw_window"
METHOD = "raw_summary"
CELLS = (METHOD, "raw_segment", "full_context", WINDOW)
PAIRS = (
    ("RSM - window", METHOD, WINDOW),
    ("window - RSM no carry", WINDOW, "raw_segment"),
    ("full history - window", "full_context", WINDOW),
)
LAP_HISTORIES = ("retained", "attempt-cleared")


def _laps_reader() -> Any:
    """The repeated-laps reader's panel loader and config resolution, reused."""
    path = Path(__file__).with_name("laps_read.py")
    spec = importlib.util.spec_from_file_location("laps_read", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["laps_read"] = module
    spec.loader.exec_module(module)
    return module


def _native(
    contract: BenchmarkContract, config: ExperimentConfig, split: str
) -> np.ndarray:
    """Goal fraction per roster unit of one seed's plain endpoint panel."""
    directory = config.run_directory / "eval"
    directory = directory / evaluation_directory(split, "retained", "endpoint")
    if results_file(directory / RESULTS_FILE) is None:
        raise SystemExit(f"missing panel {directory}")
    _, events = read_benchmark_results(directory / RESULTS_FILE, [contract])
    fractions = {
        (int(e.task_id), int(e.rollout_seed)): float(e.numerator)
        / float(e.denominator or 1)
        for e in events
        if e.kind == "episode"
    }
    units = [
        (int(task), int(rollout))
        for task in contract.roster(split)
        for rollout in contract.evaluation.rollout_seeds
    ]
    if any(unit not in fractions for unit in units):
        raise SystemExit(f"incomplete panel {directory}")
    return np.asarray([fractions[unit] for unit in units], dtype=np.float64)


def _row(
    name: str,
    matrix: np.ndarray,
    seeds: Any,
    delta: float,
    rng: Any,
    samples: int,
) -> dict[str, object]:
    estimate = float(matrix.mean())
    lower, upper = _interval(
        _bootstrap(matrix, samples=samples, rng=rng), estimate, 0.95
    )
    per_seed = {s: float(matrix[i].mean()) for i, s in enumerate(seeds)}
    return {
        "row": name,
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "positive_seeds": sum(v > 0 for v in per_seed.values()),
        "every_seed_clears_delta": all(v >= delta for v in per_seed.values()),
        **{f"seed_{s}": v for s, v in per_seed.items()},
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {path}")


def _print(rows: list[dict[str, object]]) -> None:
    for r in rows:
        seeds = " / ".join(f"{v:.3f}" for k, v in r.items() if k.startswith("seed_"))
        print(
            f"  {r['row']:44s} {r['estimate']:+8.3f} "
            f"[{r['lower']:+8.3f}, {r['upper']:+8.3f}]  seeds {seeds}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("configs/mazerunner_8m.yaml")
    )
    parser.add_argument("--benchmark", default="mazerunner")
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    laps = _laps_reader()
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    seeds = tuple(study.training_seeds)
    reader_args = argparse.Namespace(
        study=args.study, benchmark=args.benchmark, cells=list(CELLS), device="cpu"
    )
    _, configs = laps._configs(reader_args)
    missing = [c for c in CELLS if c not in configs]
    if missing:
        raise SystemExit(f"cells without started runs: {missing}")
    ordered = {
        cell: sorted(configs[cell], key=lambda c: seeds.index(c.seed)) for cell in CELLS
    }
    out = (
        repository_root()
        / study.output_root
        / "reports"
        / contract.protocol
        / args.split
        / "window"
    )
    rng = np.random.default_rng(args.seed)

    # W1: the trained task's goal fraction.
    native = {
        cell: np.stack([_native(contract, c, args.split) for c in ordered[cell]])
        for cell in CELLS
    }
    rows = [_row(cell, native[cell], seeds, 0.1, rng, args.samples) for cell in CELLS]
    rows += [
        _row(name, native[a] - native[b], seeds, 0.1, rng, args.samples)
        for name, a, b in PAIRS
    ]
    print(f"W1 {args.split} goal fraction (delta 0.1)")
    _print(rows)
    _write(out / "native.csv", rows)

    # W3: the repeated laps, last 500-call window of the budget.
    lap_rows: list[dict[str, object]] = []
    for history in LAP_HISTORIES:
        cubes = {}
        for cell in CELLS:
            loaded = [
                laps._panel(contract, c, args.split, history, args.budget)
                for c in ordered[cell]
            ]
            if any(item is None for item in loaded):
                raise SystemExit(f"{history} {cell}: laps panel missing or incomplete")
            cubes[cell] = np.stack(
                [item[0] for item in loaded]
            )  # [seeds, units, windows]
        last = {cell: cube[:, :, -1] for cell, cube in cubes.items()}
        rows = [
            {
                "history": history,
                **_row(cell, last[cell], seeds, 1.0, rng, args.samples),
            }
            for cell in CELLS
        ]
        rows += [
            {
                "history": history,
                **_row(name, last[a] - last[b], seeds, 1.0, rng, args.samples),
            }
            for name, a, b in PAIRS
        ]
        print(f"W3 laps {args.budget}, {history}, last window (delta 1 goal)")
        _print(rows)
        lap_rows += rows
    _write(out / "laps.csv", lap_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
