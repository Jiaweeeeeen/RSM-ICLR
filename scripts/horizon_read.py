"""The paired extended-horizon read (EXPERIMENTS section 5; study protocol §5).

    python scripts/horizon_read.py --study configs/memo_key_to_door_8m.yaml \
        --benchmark dark_key_to_door --split confirmation --horizon 2000 \
        [--delta 1.0] [--samples 2000]

Reads the frozen-endpoint panels of the paper's six cells at one horizon (plus
the RSM summary-cleared panel where it exists),
turns every panel into completed doors per 500-call window for every (task,
rollout seed) unit of the roster, and reads the pre-declared hypotheses with
the repository's joint seed/task bootstrap and per-seed task bootstraps:

* H4a, as declared: RSM's rate in the last window within delta of its rate in
  the first window, every seed. Reported literally, and beside it the two
  readings the record's outcome needs: does RSM's last-window rate fall below
  its first window by more than delta, and below its second window (the
  rate once the layout is known) by more than delta?
* H4b: Memo's and the full history's last-window rate below RSM's by at least
  delta, joint interval above zero, every seed in the same direction.
* Companions: RSM against the GRU, Memo fixed and RSM summary-cleared; Memo
  fixed against Memo.

Writes ``windows.csv``, ``contrasts.csv`` and ``h4.json`` under
``<study root>/reports/<benchmark>/<split>/horizon/h<H>/`` and prints the read.
Nothing here evaluates: it reads records that exist.
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
from reasoned_icrl.experiments.benchmarks import saved_config
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import RESULTS_FILE, evaluation_directory
from reasoned_icrl.experiments.horizon import WINDOW, layout_matrix, window_matrix
from reasoned_icrl.experiments.records import (
    read_benchmark_results,
    read_results_text,
    results_file,
)
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve
from reasoned_icrl.experiments.summary_memory.revised import reference_configs
from reasoned_icrl.utils import repository_root

PAPER_CELLS = (
    "raw_summary_residual",
    "raw_summary",
    "raw_segment",
    "full_context",
    "full_gru",
    "memo",
    "memo_fixed",
    # the sliding-window baseline; skipped where it has no run
    "raw_window",
)
METHOD = "raw_summary"  # RSM-O, the method again
"""The paper's method (the residual rewrite);
``--method raw_summary`` reproduces the overwrite's reads,
whose records sit under ``h<H>/`` without a suffix; any other method's read
lands under ``h<H>-<method>/`` so no earlier record is overwritten."""
HISTORICAL_METHOD = "raw_summary"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--study", type=Path, required=True)
    result.add_argument("--benchmark", default="dark_key_to_door")
    result.add_argument("--split", default="confirmation")
    result.add_argument("--horizon", type=int, default=2000)
    result.add_argument("--delta", type=float, default=1.0)
    result.add_argument("--samples", type=int, default=2000)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--layout-period",
        type=int,
        default=None,
        metavar="P",
        help="read the layout-change continuation panels (-relayout<P>): every "
        "column is one hidden layout (doors binned by the attempt's recorded "
        "layout, since the old layout persists to the next attempt boundary), "
        "and the recovery table is added",
    )
    result.add_argument("--method", default=METHOD, choices=PAPER_CELLS)
    result.add_argument("--device", default="cpu")
    return result


def _started_configs(
    study: Any, args: argparse.Namespace, condition: str
) -> list[ExperimentConfig] | None:
    """The saved configs of ``condition``'s seeds, or ``None`` when no seed has
    started (a supplementary cell added to the study before its fits run must
    not abort the read)."""
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
    if not all((c.run_directory / "config.yaml").is_file() for c in resolved):
        return None
    return [saved_config(c) for c in resolved]


def _configs(args: argparse.Namespace) -> dict[str, list[ExperimentConfig]]:
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    plan = study.tier(contract.name)
    fitted = tuple(plan.primary) + tuple(plan.supplementary)
    references = reference_configs(
        study, contract, repository=repository_root(), device=args.device
    )
    chosen: dict[str, list[ExperimentConfig]] = {}
    for condition in PAPER_CELLS:
        if condition in fitted:
            configs = _started_configs(study, args, condition)
            if configs is None:
                print(f"{condition}: no run started, skipped", flush=True)
                continue
            chosen[condition] = configs
        elif condition in plan.reference_cells:
            chosen[condition] = [c for c in references if c.condition == condition]
    return chosen


def _load(
    contract: Any,
    config: ExperimentConfig,
    split: str,
    horizon: int,
    *,
    history: str,
    layout_period: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    panel = (
        config.run_directory
        / "eval"
        / evaluation_directory(
            split,
            history,
            "endpoint",
            horizon=horizon,
            layout_period=layout_period,
        )
    )
    path = panel / RESULTS_FILE
    if results_file(path) is None:  # the plain file or its compacted twin
        return None
    raw = json.loads(read_results_text(path))
    if not isinstance(raw, dict) or "partial_task_cap" in raw:
        return None
    _, events = read_benchmark_results(path, [contract])
    units = [
        (int(task), int(rollout))
        for task in contract.roster(split)
        for rollout in contract.evaluation.rollout_seeds
    ]
    if layout_period is not None:
        layouts = horizon // layout_period
        matrix = layout_matrix(events, units=units, layouts=layouts)
        return matrix, _layout_recovery(events, units, layouts)
    matrix = window_matrix(events, units=units, outer_length=horizon)
    return matrix, _recovery(events, units, horizon)


def _recovery(events: Any, units: Any, horizon: int) -> np.ndarray:
    """Calls from each window's start to its first completed door per unit,
    NaN where the window holds no door (right-censored at the window's end)."""
    index = {unit: row for row, unit in enumerate(units)}
    windows = horizon // WINDOW
    out = np.full((len(units), windows), np.nan)
    for e in events:
        if e.kind != "attempt" or not e.numerator or e.end_step is None:
            continue
        if e.complete is False:
            continue
        w = (int(e.end_step) - 1) // WINDOW
        row = index[(int(e.task_id), int(e.rollout_seed))]
        calls = int(e.end_step) - w * WINDOW
        if np.isnan(out[row, w]) or calls < out[row, w]:
            out[row, w] = calls
    return out


def _layout_recovery(events: Any, units: Any, layouts: int) -> np.ndarray:
    """Calls from each layout's first call (the start of its first attempt)
    to the unit's first completed door in that layout, NaN where the layout
    holds no door. The window form above would credit the old layout's last
    attempt, which runs up to one attempt cap past the period boundary."""
    index = {unit: row for row, unit in enumerate(units)}
    starts = np.full((len(units), layouts), np.inf)
    doors = np.full((len(units), layouts), np.inf)
    for e in events:
        if e.kind != "attempt" or e.layout_index is None or e.start_step is None:
            continue
        row = index[(int(e.task_id), int(e.rollout_seed))]
        k = int(e.layout_index)
        starts[row, k] = min(starts[row, k], int(e.start_step))
        if e.numerator and e.end_step is not None and e.complete is not False:
            doors[row, k] = min(doors[row, k], int(e.end_step))
    out: np.ndarray = doors - starts + 1
    out[~np.isfinite(out)] = np.nan
    return out


def _estimate(
    matrix: np.ndarray, *, samples: int, confidence: float, rng: Any, seeds: Any
) -> dict[str, Any]:
    """Joint seed/task interval and per-seed task intervals, [seeds, units] in."""
    estimate = float(matrix.mean())
    lower, upper = _interval(
        _bootstrap(matrix, samples=samples, rng=rng), estimate, confidence
    )
    per_seed = {}
    for index, seed in enumerate(seeds):
        row = matrix[index]
        picks = rng.integers(0, row.shape[0], size=(samples, row.shape[0]))
        draws = row[picks].mean(axis=1)
        low, high = _interval(draws, float(row.mean()), confidence)
        per_seed[seed] = {"estimate": float(row.mean()), "lower": low, "upper": high}
    return {"estimate": estimate, "lower": lower, "upper": upper, "per_seed": per_seed}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    horizon = int(args.horizon)
    # Columns: 500-call windows, or one hidden layout each under --layout-period.
    windows = horizon // (args.layout_period or WINDOW)
    column = "window" if args.layout_period is None else "layout"
    delta = float(args.delta)
    rng = np.random.default_rng(args.seed)
    configs = _configs(args)
    method_name = args.method
    seeds = tuple(study.training_seeds)
    # variant name -> [seeds, units, windows]
    panels: dict[str, np.ndarray] = {}
    recoveries: dict[str, np.ndarray] = {}
    for condition, cell_configs in configs.items():
        cell_configs = sorted(cell_configs, key=lambda c: seeds.index(c.seed))
        variants = [(condition, "retained")]
        if condition == method_name:
            variants.append((f"{condition} summary-cleared", "summary-cleared"))
        for name, history in variants:
            rows = [
                _load(
                    contract,
                    c,
                    args.split,
                    horizon,
                    history=history,
                    layout_period=args.layout_period,
                )
                for c in cell_configs
            ]
            if len(rows) != len(seeds) or any(r is None for r in rows):
                print(f"{name}: panels incomplete at h{horizon}, skipped", flush=True)
                continue
            panels[name] = np.stack([r[0] for r in rows])  # type: ignore[index]
            recoveries[name] = np.stack([r[1] for r in rows])  # type: ignore[index]
    if method_name not in panels:
        raise ContractError(
            f"{method_name!r} has no complete h{horizon} panel on every seed."
        )
    root = Path(study.output_root)
    out = (
        root
        / "reports"
        / contract.name.replace("_", "-")
        / args.split
        / "horizon"
        / (
            (
                f"h{horizon}"
                if args.layout_period is None
                else f"h{horizon}-relayout{args.layout_period}"
            )
            + ("" if method_name == HISTORICAL_METHOD else f"-{method_name}")
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    confidence = 0.95

    def est(matrix: np.ndarray) -> dict[str, Any]:
        return _estimate(
            matrix, samples=args.samples, confidence=confidence, rng=rng, seeds=seeds
        )

    # Per-window rates.
    window_rows = []
    for name, cube in panels.items():
        for w in range(windows):
            e = est(cube[:, :, w])
            window_rows.append(
                {
                    "variant": name,
                    column: w + 1,
                    **{k: v for k, v in e.items() if k != "per_seed"},
                    **{f"seed_{s}": e["per_seed"][s]["estimate"] for s in seeds},
                }
            )
    with (out / "windows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)

    last = windows - 1
    method = panels[method_name]

    def contrast(name: str, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        e = est(left - right)
        per = e["per_seed"]
        e.update(
            {
                "contrast": name,
                "positive_seeds": sum(v["estimate"] > 0 for v in per.values()),
                "negative_seeds": sum(v["estimate"] < 0 for v in per.values()),
                "consistent_gain": e["estimate"] >= delta
                and all(v["estimate"] > 0 for v in per.values())
                and e["lower"] > 0,
                "within_delta_every_seed": all(
                    abs(v["estimate"]) <= delta for v in per.values()
                ),
                "no_fall_every_seed": all(
                    v["estimate"] >= -delta for v in per.values()
                ),
            }
        )
        return e

    contrasts = []
    # H4a and its readings (RSM against itself, paired within unit).
    contrasts.append(
        contrast(
            "H4a RSM last window - first window", method[:, :, last], method[:, :, 0]
        )
    )
    if windows >= 2:
        contrasts.append(
            contrast(
                "RSM last window - second window", method[:, :, last], method[:, :, 1]
            )
        )
    # H4b and companions in the last window.
    for right, label in (
        ("memo", "H4b RSM - Memo"),
        ("full_context", "H4b RSM - full history"),
        ("full_gru", "RSM - GRU"),
        ("memo_fixed", "RSM - Memo fixed"),
        ("raw_segment", "RSM - RSM no carry"),
        (f"{method_name} summary-cleared", "RSM - RSM summary-cleared"),
        ("raw_summary", "RSM - RSM overwrite"),
        ("raw_summary_residual", "RSM - RSM residual"),
    ):
        if right == method_name:
            continue
        if right in panels:
            contrasts.append(
                contrast(
                    f"{label} (last window)",
                    method[:, :, last],
                    panels[right][:, :, last],
                )
            )
    if "memo_fixed" in panels and "memo" in panels:
        contrasts.append(
            contrast(
                "Memo fixed - Memo (last window)",
                panels["memo_fixed"][:, :, last],
                panels["memo"][:, :, last],
            )
        )
    for other in ("memo", "full_context", "full_gru", "memo_fixed", "raw_summary"):
        if other in panels and windows >= 2 and other != method_name:
            contrasts.append(
                contrast(
                    f"{other} last window - second window",
                    panels[other][:, :, last],
                    panels[other][:, :, 1],
                )
            )

    if args.layout_period is not None:
        # Every window is one hidden layout: H5a compares RSM's later layouts
        # with its first; H5b compares the cells within layouts 2..n, pooled
        # per unit and window by window.
        later = slice(1, windows)
        contrasts.append(
            contrast(
                "H5a RSM layouts 2-n mean - layout 1",
                method[:, :, later].mean(axis=2),
                method[:, :, 0],
            )
        )
        for w in range(1, windows):
            contrasts.append(
                contrast(
                    f"RSM layout {w + 1} - layout 1", method[:, :, w], method[:, :, 0]
                )
            )
        for right, label in (
            ("memo", "H5b RSM - Memo"),
            ("full_context", "H5b RSM - full history"),
            ("memo_fixed", "RSM - Memo fixed"),
            ("full_gru", "RSM - GRU"),
            ("raw_segment", "RSM - RSM no carry"),
            ("raw_summary", "RSM - RSM overwrite"),
        ):
            if right not in panels or right == method_name:
                continue
            contrasts.append(
                contrast(
                    f"{label} (layouts 2-n mean)",
                    method[:, :, later].mean(axis=2),
                    panels[right][:, :, later].mean(axis=2),
                )
            )
            for w in range(1, windows):
                contrasts.append(
                    contrast(
                        f"{label} (layout {w + 1})",
                        method[:, :, w],
                        panels[right][:, :, w],
                    )
                )
        recovery_rows = []
        for name, cube in recoveries.items():
            for w in range(windows):
                values = cube[:, :, w]
                found = ~np.isnan(values)
                recovery_rows.append(
                    {
                        "variant": name,
                        "layout": w + 1,
                        "units_with_a_door": int(found.sum()),
                        "units": int(values.size),
                        "median_calls_to_first_door": float(np.nanmedian(values))
                        if found.any()
                        else None,
                        "mean_calls_to_first_door": float(np.nanmean(values))
                        if found.any()
                        else None,
                    }
                )
        with (out / "recovery.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(recovery_rows[0]))
            writer.writeheader()
            writer.writerows(recovery_rows)
    # The sliding window's declared rows:
    # W1 at the first window (the trained task), W2 at the last. Appended after
    # every other row so the earlier rows' bootstrap draws are unchanged.
    window_cell = "raw_window"
    if window_cell in panels and method_name != window_cell:
        window_panel = panels[window_cell]
        for w, tag in ((0, "first window"), (last, "last window")):
            pairs = [("RSM - window", method, window_panel)]
            if "raw_segment" in panels:
                pairs.append(
                    ("window - RSM no carry", window_panel, panels["raw_segment"])
                )
            if "full_context" in panels:
                pairs.append(
                    ("full history - window", panels["full_context"], window_panel)
                )
            for label, left_panel, right_panel in pairs:
                contrasts.append(
                    contrast(
                        f"{label} ({tag})", left_panel[:, :, w], right_panel[:, :, w]
                    )
                )
    with (out / "contrasts.csv").open("w", newline="") as fh:
        fields = [
            "contrast",
            "estimate",
            "lower",
            "upper",
            "positive_seeds",
            "negative_seeds",
            "consistent_gain",
            "within_delta_every_seed",
            "no_fall_every_seed",
        ] + [f"seed_{s}" for s in seeds]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for c in contrasts:
            writer.writerow(
                {
                    **{k: c[k] for k in fields if k in c},
                    **{f"seed_{s}": c["per_seed"][s]["estimate"] for s in seeds},
                }
            )

    by_name = {c["contrast"]: c for c in contrasts}
    h4a = by_name["H4a RSM last window - first window"]
    h4b = [
        by_name[k]
        for k in (
            "H4b RSM - Memo (last window)",
            "H4b RSM - full history (last window)",
        )
        if k in by_name
    ]
    read = {
        "horizon": horizon,
        "method": method_name,
        "layout_period": args.layout_period,
        "window": WINDOW,
        "columns": column,
        "binning": "layout_index" if args.layout_period else "end_step window",
        "split": args.split,
        "delta": delta,
        "samples": args.samples,
        "bootstrap_seed": args.seed,
        "confidence": confidence,
        "seeds": list(seeds),
        "units": int(method.shape[1]),
        "variants": sorted(panels),
        "H4a_literal_within_delta": h4a["within_delta_every_seed"],
        "H4a_no_fall_below_first_window": h4a["no_fall_every_seed"],
        "H4a_no_fall_below_second_window": by_name.get(
            "RSM last window - second window", {}
        ).get("no_fall_every_seed"),
        "H4b_holds": bool(h4b)
        and all(c["consistent_gain"] for c in h4b)
        and len(h4b) == 2,
        "H4b_parts": {c["contrast"]: c["consistent_gain"] for c in h4b},
        "contrasts": contrasts,
        "read_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "h4.json").write_text(json.dumps(read, indent=2) + "\n")
    print(
        f"h{horizon} {args.split}: {method.shape[1]} units x {len(seeds)} seeds; "
        f"delta {delta}"
    )
    for row in window_rows:
        print(
            f"  {row['variant']:<32} {column[0]}{row[column]}  {row['estimate']:6.2f} "
            f"[{row['lower']:6.2f}, {row['upper']:6.2f}]  "
            + " / ".join(f"{row[f'seed_{s}']:.1f}" for s in seeds)
        )
    for c in contrasts:
        print(
            f"  {c['contrast']:<48} {c['estimate']:+7.2f} "
            f"[{c['lower']:+7.2f}, {c['upper']:+7.2f}]  seeds "
            + " / ".join(f"{c['per_seed'][s]['estimate']:+.1f}" for s in seeds)
            + f"  gain={c['consistent_gain']} within={c['within_delta_every_seed']}"
            + f" nofall={c['no_fall_every_seed']}"
        )
    print(
        f"H4a literal: {read['H4a_literal_within_delta']}; "
        f"no fall vs first: {read['H4a_no_fall_below_first_window']}; "
        f"no fall vs second: {read['H4a_no_fall_below_second_window']}; "
        f"H4b: {read['H4b_holds']} {read['H4b_parts']}"
    )
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"horizon_read: {error}", file=sys.stderr)
        raise SystemExit(2) from error
