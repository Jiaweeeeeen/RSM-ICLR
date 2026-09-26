"""Read the MazeRunner repeated-laps panels of a study: goals reached per
500-call window by budget and the paired contrasts of the repeated-laps
hypotheses (H9).

MazeRunner's second beyond-horizon axis (before any laps panel existed) replays every
roster map on the frozen 8M endpoints in
laps until a budget of charged calls: a lap is one native episode of the
trained task on the map's own seed (same maze, ordered goals and hidden action
permutation), a reset-only call separates laps and the carried memory
continues across it. Every panel is one greedy outer task per roster map,
``<split>-<history>-endpoint-calls<B>`` under each run's ``eval/``
(``scripts/horizon_panels.py --benchmark mazerunner --laps``). This reader
stacks those panels into one matrix per cell, history and budget
([seeds, maps, windows]) and writes, under
``<study root>/reports/<benchmark>/<split>/laps/<history>/``:

``goals_per_window.csv``
    the joint seed/map bootstrap estimate and interval of goals reached per
    500-call window per cell, budget and window (each goal binned by the call
    that reached it), the three seeds beside it, and the cumulative goals;
``laps.csv``
    finished laps per map and calls per finished lap by lap index, per cell
    and budget, seed by seed;
``contrasts.csv``
    paired last-window contrasts (the same map and seed on both sides) of the
    method against every other cell, of the overwrite rewrite against every
    other cell and of every comparator against the carry control, per budget:
    the joint estimate and interval, the per-seed means and whether every
    seed clears ``--delta``;
``h9.json``
    the pre-declared readings at the largest budget. H9a: the method's rate
    in the last window within ``delta`` of its rate in calls 501-1,000 on
    every seed. H9b: the full history's and Memo's last-window rates below
    the method's by at least ``delta``, joint interval above zero, every seed
    in the same direction. H9c (the lap-cleared companion, read from the
    ``attempt-cleared`` panels): every cell's window rates stay within
    ``delta`` of its first window, so the lap structure itself is not what
    moves a cell.

Cells come from the study's tier (primary and supplementary) plus its frozen
reference cells. A cell whose panels are incomplete for a history and budget
is reported and skipped; nothing is projected.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from reasoned_icrl.analysis.statistics import _bootstrap, _interval
from reasoned_icrl.experiments.benchmarks import BenchmarkContract, Study, saved_config
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.evaluation import RESULTS_FILE, evaluation_directory
from reasoned_icrl.experiments.horizon import (
    LAP_BUDGETS,
    WINDOW,
    continued_laps,
    laps_per_unit,
    window_goal_matrix,
)
from reasoned_icrl.experiments.records import (
    read_benchmark_results,
    read_results_text,
    results_file,
)
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve
from reasoned_icrl.experiments.summary_memory.revised import reference_configs
from reasoned_icrl.utils import repository_root

METHOD = "raw_summary"
"""RSM-O, the paper's method."""
OVERWRITE = "raw_summary_residual"
"""The other write rule, compared with every cell: RSM-R, the rewrite ablation
(the name is kept from when the replacing write was the ablation)."""
CELLS = (
    "raw_summary_residual",
    "raw_summary",
    "raw_segment",
    "full_context",
    "full_gru",
    "memo",
    "memo_fixed",
    "raw_window",
)
"""Every cell the paper's MazeRunner study fits, in report order."""
CONTROL = "raw_segment"
H9B_CELLS = ("full_context", "memo")
HISTORIES = ("retained", "attempt-cleared", "summary-cleared")
REFERENCE_WINDOW = 1
"""The window the sustained-laps reading compares against: calls 501-1,000,
the first window after the route can have been learned (window 0 holds the
exploring first lap)."""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument(
        "--study", type=Path, default=Path("configs/mazerunner_8m.yaml")
    )
    result.add_argument("--benchmark", default="mazerunner")
    result.add_argument("--split", default="confirmation")
    result.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(LAP_BUDGETS),
        help="laps budgets in charged calls to read (default the declared 1000 "
        "2000 4000)",
    )
    result.add_argument(
        "--histories", nargs="+", default=list(HISTORIES), choices=HISTORIES
    )
    result.add_argument("--cells", nargs="*", default=None, choices=CELLS)
    result.add_argument("--method", default=METHOD, choices=CELLS)
    result.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="goals per 500-call window (the pre-declared margin, 1.0)",
    )
    result.add_argument("--samples", type=int, default=2000)
    result.add_argument("--confidence", type=float, default=0.95)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report directory; default <root>/reports/<benchmark>/<split>/laps",
    )
    return result


def _configs(
    args: argparse.Namespace,
) -> tuple[Study, dict[str, list[ExperimentConfig]]]:
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    plan = study.tier(contract.name)
    fitted = tuple(plan.primary) + tuple(plan.supplementary)
    references = (
        reference_configs(
            study, contract, repository=repository_root(), device=args.device
        )
        if plan.reference_cells
        else ()
    )
    wanted = tuple(args.cells) if args.cells else CELLS
    chosen: dict[str, list[ExperimentConfig]] = {}
    for condition in CELLS:
        if condition not in wanted:
            continue
        if condition in fitted:
            resolved = [
                resolve(
                    study,
                    benchmark=args.benchmark,
                    condition=condition,
                    seed=seed,
                    device=args.device,
                )[1]
                for seed in study.training_seeds
            ]
            # A supplementary cell declared before its fits exist (raw_window) must not
            # abort the read of the others.
            if not all((c.run_directory / "config.yaml").is_file() for c in resolved):
                print(f"{condition}: no run started, skipped", flush=True)
                continue
            chosen[condition] = [saved_config(c) for c in resolved]
        elif condition in plan.reference_cells:
            chosen[condition] = [c for c in references if c.condition == condition]
    return study, chosen


def _panel(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    split: str,
    history: str,
    budget: int,
) -> tuple[np.ndarray, dict[tuple[int, int], int], dict[int, list[int]]] | None:
    """Goals per window per roster unit ([units, windows]), finished laps per
    unit and calls per finished lap by lap index, of one seed's laps panel;
    None when the panel is missing or incomplete."""
    directory = config.run_directory / "eval"
    directory = directory / evaluation_directory(
        split, history, "endpoint", laps=budget
    )
    path = results_file(directory / RESULTS_FILE)
    if path is None:
        return None
    raw = json.loads(read_results_text(directory / RESULTS_FILE))
    if not isinstance(raw, dict) or "partial_task_cap" in raw:
        return None
    # The panel's records run under the laps budget, so they validate against
    # the evaluation-only contract of that budget, not the trained one.
    longer, _ = continued_laps(contract, config, budget)
    _, events = read_benchmark_results(directory / RESULTS_FILE, [longer])
    units = [
        (int(task), int(rollout))
        for task in contract.roster(split)
        for rollout in contract.evaluation.rollout_seeds
    ]
    matrix = window_goal_matrix(events, units=units, outer_length=budget)
    calls: dict[int, list[int]] = {}
    for event in events:
        if event.kind == "attempt" and event.complete:
            calls.setdefault(int(event.event_index), []).append(int(event.step))
    return matrix, laps_per_unit(events), calls


def _estimate(
    matrix: np.ndarray, *, samples: int, confidence: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    estimate = float(matrix.mean())
    lower, upper = _interval(
        _bootstrap(matrix, samples=samples, rng=rng), estimate, confidence
    )
    return estimate, lower, upper


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    study, configs = _configs(args)
    contract = study.contract(args.benchmark)
    seeds = tuple(study.training_seeds)
    rng = np.random.default_rng(args.seed)
    budgets = sorted(int(b) for b in args.budgets)
    delta = float(args.delta)
    out_root = args.out or (
        repository_root()
        / study.output_root
        / "reports"
        / args.benchmark
        / args.split
        / "laps"
    )
    written: list[Path] = []
    # history -> cell -> budget -> [seeds, units, windows]
    panels: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    lap_rows: list[dict[str, object]] = []
    for history in args.histories:
        for cell, cell_configs in configs.items():
            ordered = sorted(cell_configs, key=lambda c: seeds.index(c.seed))
            if len(ordered) != len(seeds):
                print(
                    f"{history} {cell}: {len(ordered)} of {len(seeds)} seeds "
                    "resolved, skipped"
                )
                continue
            for budget in budgets:
                rows = [
                    _panel(contract, c, args.split, history, budget) for c in ordered
                ]
                if any(r is None for r in rows):
                    missing = [x for x, r in zip(seeds, rows, strict=True) if r is None]
                    print(
                        f"{history} {cell} calls{budget}: panels missing for seeds "
                        f"{missing}, skipped"
                    )
                    continue
                present = [r for r in rows if r is not None]
                panels.setdefault(history, {}).setdefault(cell, {})[budget] = np.stack(
                    [r[0] for r in present]
                )
                for seed, (_, finished, calls) in zip(seeds, present, strict=True):
                    lap_rows.append(
                        {
                            "history": history,
                            "cell": cell,
                            "budget": budget,
                            "seed": seed,
                            "finished_laps_mean": float(
                                np.mean([float(v) for v in finished.values()])
                            )
                            if finished
                            else 0.0,
                            **{
                                f"calls_lap_{index}": float(np.mean(values))
                                for index, values in sorted(calls.items())
                                if index <= 12
                            },
                        }
                    )
    if not panels:
        print("no complete laps panel, nothing written")
        return 0
    for history, by_cell in panels.items():
        out = out_root / history
        out.mkdir(parents=True, exist_ok=True)
        window_rows: list[dict[str, object]] = []
        for cell, by_budget in by_cell.items():
            for budget, matrix in sorted(by_budget.items()):
                cumulative = np.cumsum(matrix, axis=2)
                for index in range(matrix.shape[2]):
                    estimate, lower, upper = _estimate(
                        matrix[:, :, index],
                        samples=args.samples,
                        confidence=args.confidence,
                        rng=rng,
                    )
                    window_rows.append(
                        {
                            "cell": cell,
                            "budget": budget,
                            "window": index + 1,
                            "calls_from": index * WINDOW + 1,
                            "calls_to": (index + 1) * WINDOW,
                            "estimate": estimate,
                            "lower": lower,
                            "upper": upper,
                            **{
                                f"seed_{s}": float(matrix[i, :, index].mean())
                                for i, s in enumerate(seeds)
                            },
                            "cumulative": float(cumulative[:, :, index].mean()),
                            **{
                                f"cumulative_seed_{s}": float(
                                    cumulative[i, :, index].mean()
                                )
                                for i, s in enumerate(seeds)
                            },
                        }
                    )
        _write_csv(out / "goals_per_window.csv", window_rows)
        written.append(out / "goals_per_window.csv")
        _write_csv(out / "laps.csv", [r for r in lap_rows if r["history"] == history])
        written.append(out / "laps.csv")
        # contrasts.csv: the last window of every budget
        contrast_rows: list[dict[str, object]] = []
        pairs: list[tuple[str, str, str]] = []
        method = args.method
        if method in by_cell:
            for cell in by_cell:
                if cell != method:
                    pairs.append((f"{_label(method)} - {_label(cell)}", method, cell))
        if OVERWRITE in by_cell and method != OVERWRITE:
            for cell in by_cell:
                if cell not in (OVERWRITE, method):
                    pairs.append(
                        (f"{_label(OVERWRITE)} - {_label(cell)}", OVERWRITE, cell)
                    )
        if CONTROL in by_cell:
            for cell in by_cell:
                if cell not in (CONTROL, method, OVERWRITE):
                    pairs.append((f"{_label(cell)} - {_label(CONTROL)}", cell, CONTROL))
        for name, left, right in pairs:
            for budget in budgets:
                if budget not in by_cell[left] or budget not in by_cell[right]:
                    continue
                left_last = by_cell[left][budget][:, :, -1]
                diff = left_last - by_cell[right][budget][:, :, -1]
                estimate, lower, upper = _estimate(
                    diff, samples=args.samples, confidence=args.confidence, rng=rng
                )
                per_seed = [float(diff[i].mean()) for i in range(len(seeds))]
                contrast_rows.append(
                    {
                        "history": history,
                        "budget": budget,
                        "window": "last",
                        "name": name,
                        "left": left,
                        "right": right,
                        "estimate": estimate,
                        "lower": lower,
                        "upper": upper,
                        **{
                            f"seed_{x}": v for x, v in zip(seeds, per_seed, strict=True)
                        },
                        "every_seed_gain": all(v >= delta for v in per_seed),
                        "every_seed_loss": all(v <= -delta for v in per_seed),
                        "joint_above_zero": lower > 0.0,
                        "joint_below_zero": upper < 0.0,
                    }
                )
        _write_csv(out / "contrasts.csv", contrast_rows)
        written.append(out / "contrasts.csv")
        h9 = _hypotheses(
            history,
            by_cell,
            panels.get("attempt-cleared", {}),
            seeds=seeds,
            budgets=budgets,
            method=method,
            delta=delta,
            samples=args.samples,
            confidence=args.confidence,
            rng=rng,
        )
        (out / "h9.json").write_text(json.dumps(h9, indent=2, default=str) + "\n")
        written.append(out / "h9.json")
        _print(history, by_cell, seeds, window_rows, h9)
    for path in written:
        print(f"written {path}")
    return 0


def _hypotheses(
    history: str,
    by_cell: dict[str, dict[int, np.ndarray]],
    cleared: dict[str, dict[int, np.ndarray]],
    *,
    seeds: tuple[int, ...],
    budgets: list[int],
    method: str,
    delta: float,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    largest = budgets[-1]
    h9: dict[str, Any] = {
        "history": history,
        "method": method,
        "delta": delta,
        "largest_budget": largest,
        "window_calls": WINDOW,
    }
    if history != "retained":
        h9["note"] = "H9a and H9b are read on the retained history only."
        return h9
    if method in by_cell and largest in by_cell[method]:
        matrix = by_cell[method][largest]
        if matrix.shape[2] > REFERENCE_WINDOW:
            drop = matrix[:, :, -1] - matrix[:, :, REFERENCE_WINDOW]
            drop_by_seed = {x: float(drop[i].mean()) for i, x in enumerate(seeds)}
            estimate, lower, upper = _estimate(
                drop, samples=samples, confidence=confidence, rng=rng
            )
            h9["H9a"] = {
                "statement": (
                    f"{method}'s goals per window in the last window of {largest} "
                    f"calls within delta of its window {REFERENCE_WINDOW + 1} "
                    "(calls 501-1,000), every seed"
                ),
                "last_minus_reference": {
                    "estimate": estimate,
                    "lower": lower,
                    "upper": upper,
                },
                "per_seed": drop_by_seed,
                "holds": all(abs(v) <= delta for v in drop_by_seed.values()),
            }
    if "H9a" not in h9:
        h9["H9a"] = {"holds": None, "reason": "the method's panels are incomplete"}
    h9b: dict[str, object] = {}
    for cell in H9B_CELLS:
        if (
            method not in by_cell
            or cell not in by_cell
            or largest not in by_cell[method]
            or largest not in by_cell[cell]
        ):
            h9b[cell] = {"holds": None, "reason": "panels incomplete"}
            continue
        diff = by_cell[method][largest][:, :, -1] - by_cell[cell][largest][:, :, -1]
        estimate, lower, upper = _estimate(
            diff, samples=samples, confidence=confidence, rng=rng
        )
        diff_by_seed = {x: float(diff[i].mean()) for i, x in enumerate(seeds)}
        h9b[cell] = {
            "statement": (
                f"{method} above {cell} by at least delta goals per window in the "
                f"last window of {largest} calls, every seed, joint interval "
                "above zero"
            ),
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "per_seed": diff_by_seed,
            "holds": lower > 0.0 and all(v >= delta for v in diff_by_seed.values()),
        }
    h9["H9b"] = h9b
    h9c: dict[str, object] = {}
    for cell, by_budget in cleared.items():
        if largest not in by_budget:
            h9c[cell] = {"holds": None, "reason": "lap-cleared panel incomplete"}
            continue
        matrix = by_budget[largest]
        first = matrix[:, :, 0]
        windows = {
            index + 1: {
                x: float((matrix[i, :, index] - first[i]).mean())
                for i, x in enumerate(seeds)
            }
            for index in range(1, matrix.shape[2])
        }
        h9c[cell] = {
            "statement": (
                f"under lap-cleared {cell}'s goals per window stay within delta of "
                "its first window, every window, every seed"
            ),
            "window_minus_first_per_seed": windows,
            "holds": all(
                abs(v) <= delta
                for per_seed in windows.values()
                for v in per_seed.values()
            ),
        }
    h9["H9c"] = h9c or {"holds": None, "reason": "no lap-cleared panel"}
    h9["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return h9


def _label(cell: str) -> str:
    names = {
        "raw_summary_residual": "RSM-R",
        "raw_summary": "RSM-O",
        "raw_segment": "w/o memory",
        "full_context": "full history",
        "full_gru": "GRU",
        "memo": "Memo",
        "memo_fixed": "Memo fixed",
    }
    return names.get(cell, cell)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print(
    history: str,
    by_cell: dict[str, dict[int, np.ndarray]],
    seeds: tuple[int, ...],
    window_rows: list[dict[str, object]],
    h9: dict[str, Any],
) -> None:
    print(f"== {history}")
    for cell in by_cell:
        for budget in sorted(by_cell[cell]):
            parts = [
                f"w{row['window']}: {row['estimate']:.2f}"
                for row in window_rows
                if row["cell"] == cell and row["budget"] == budget
            ]
            print(f"{_label(cell)} calls{budget}: " + " · ".join(parts))
    a = h9.get("H9a", {})
    if a.get("holds") is not None:
        seeds_text = " / ".join(f"{v:+.2f}" for v in a["per_seed"].values())
        print(
            f"H9a {'holds' if a['holds'] else 'FAILS'}: last - reference "
            f"{a['last_minus_reference']['estimate']:+.2f} seeds {seeds_text}"
        )
    for cell, b in h9.get("H9b", {}).items():
        if b.get("holds") is None:
            print(f"H9b {_label(cell)}: {b.get('reason')}")
            continue
        seeds_text = " / ".join(f"{v:+.2f}" for v in b["per_seed"].values())
        print(
            f"H9b {_label(h9['method'])} - {_label(cell)} "
            f"{'holds' if b['holds'] else 'FAILS'}: {b['estimate']:+.2f} "
            f"[{b['lower']:+.2f}, {b['upper']:+.2f}] seeds {seeds_text}"
        )
    c = h9.get("H9c", {})
    if isinstance(c, dict) and c.get("holds", 0) is not None:
        for cell, v in c.items():
            if isinstance(v, dict) and v.get("holds") is not None:
                print(f"H9c {_label(cell)} {'flat' if v['holds'] else 'MOVES'}")


if __name__ == "__main__":
    sys.exit(main())
