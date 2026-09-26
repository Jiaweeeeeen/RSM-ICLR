"""Where RSM's decisions read, and what reading the summary is worth.

    python scripts/attention_read.py run --study configs/summary_memory_8m.yaml \\
        --benchmark dark_key_to_door --split confirmation --device cuda \\
        [--cells raw_summary raw_segment] [--seeds 42 100 2026] \\
        [--targets summary buffer] [--betas 1 2 4 8 inf] [--task-cap N]
    python scripts/attention_read.py read --study configs/summary_memory_8m.yaml \\
        --benchmark dark_key_to_door --split confirmation

The attention declaration. ``run``
evaluates the frozen endpoint of every requested cell and seed through the
shared evaluator with the carrier's attention probe attached: one unbiased
capture per cell (which must reproduce the saved ``<split>-retained-endpoint``
panel task for task, the acceptance check) and, for ``raw_summary`` only, one
evaluation per read-bias target and beta. It writes, under
``<study root>/reports/<protocol>/<split>/attention/`` (or ``--out``), the
per-task primary metric of every evaluation (``tasks.csv``), every decision's
attention masses (``decisions-<cell>-seed<seed>-<label>.npz``) and the
capture's evaluator events as labels (``labels-<cell>-seed<seed>.csv``).
``read`` turns those files into the declared readings A1-A3 and B1-B3
(``readings.csv``) and the display tables (``profile.csv``, ``dose.csv``) with
:mod:`reasoned_icrl.analysis.attention`. Nothing is trained and no checkpoint is
selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from reasoned_icrl.analysis.attention import (
    LABEL_FIELDS,
    read_attention,
)
from reasoned_icrl.experiments.artifacts import endpoint_training_epoch
from reasoned_icrl.experiments.benchmarks import saved_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.records import cell_values, read_benchmark_results
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve
from reasoned_icrl.runtime.attention import READ_TARGETS, ReadBias, attention_evaluation
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
    load_selected_checkpoint,
)

TASK_FIELDS = ("cell", "seed", "label", "target", "beta", "task_id", "primary")


def _beta(text: str) -> float:
    value = float(text)
    if math.isnan(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("betas are positive numbers or inf")
    return value


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    out.add_argument("mode", choices=("run", "read"))
    out.add_argument("--study", type=Path, required=True)
    out.add_argument("--benchmark", required=True)
    out.add_argument("--split", default="confirmation")
    out.add_argument("--cells", nargs="+", default=["raw_summary", "raw_segment"])
    out.add_argument("--seeds", nargs="+", type=int, default=None)
    out.add_argument(
        "--targets",
        nargs="+",
        default=["summary", "buffer"],
        choices=[target for target in READ_TARGETS if target != "none"],
    )
    out.add_argument("--betas", nargs="+", type=_beta, default=[1, 2, 4, 8, math.inf])
    out.add_argument("--device", default="auto")
    out.add_argument("--task-cap", type=int, default=None)
    out.add_argument("--batch", type=int, default=None)
    out.add_argument("--out", type=Path, default=None)
    return out


def _destination(args: argparse.Namespace, study: Any) -> tuple[Path, Any]:
    contract = study.contract(args.benchmark)
    if args.out is not None:
        return Path(args.out), contract
    _, config = resolve(
        study,
        benchmark=args.benchmark,
        condition=args.cells[0],
        seed=study.training_seeds[0],
        device="cpu",
    )
    root = Path(config.run_directory).parents[2]
    return root / "reports" / contract.protocol / args.split / "attention", contract


def _saved_panel(config: Any, contract: Any, split: str) -> dict[int, float]:
    path = (
        Path(config.run_directory)
        / "eval"
        / f"{split}-retained-endpoint"
        / "benchmark_results.json"
    )
    _, events = read_benchmark_results(path, [contract])
    cells = cell_values(events, contract.evaluation.primary_metric)
    return {int(task): value for (_, task, _), value in cells.items()}


def _write_labels(path: Path, events: Any) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for event in events:
            row = asdict(event)
            writer.writerow({name: row.get(name) for name in LABEL_FIELDS})


def _save_decisions(path: Path, decisions: dict[str, Any], *, heads: bool) -> None:
    arrays: dict[str, Any] = {
        name: decisions[name] for name in ("task_id", "step", "segment", "position")
    }
    for name in ("summary", "buffer", "own"):
        values = decisions[name]
        arrays[f"{name}_layer"] = values.mean(axis=2).astype(np.float32)
        if heads:
            arrays[f"{name}_head"] = values.astype(np.float16)
    np.savez_compressed(path, **arrays)


def run(args: argparse.Namespace) -> int:
    study = load_summary_memory_study(args.study)
    out, contract = _destination(args, study)
    out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds or study.training_seeds)
    biases = [ReadBias()] + [
        ReadBias(target, beta) for target in args.targets for beta in args.betas
    ]
    tasks_path = out / "tasks.csv"
    fresh = not tasks_path.exists()
    checks: dict[str, Any] = {}
    with tasks_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS)
        if fresh:
            writer.writeheader()
        for cell in args.cells:
            for seed in seeds:
                _, requested = resolve(
                    study,
                    benchmark=args.benchmark,
                    condition=cell,
                    seed=seed,
                    device=args.device,
                )
                config = saved_config(requested)
                run_directory = Path(config.run_directory)
                epoch = endpoint_training_epoch(
                    run_directory, epochs=config.training.epochs
                )
                checkpoint = f"policy_epoch_{epoch}"
                experiment = load_experiment(
                    config, work_directory=run_directory, persist_configuration=False
                )
                try:
                    load_selected_checkpoint(experiment, checkpoint)
                    for bias in biases if cell == "raw_summary" else biases[:1]:
                        started = time.perf_counter()
                        events, decisions = attention_evaluation(
                            contract,
                            config,
                            experiment,
                            checkpoint=checkpoint,
                            split=args.split,
                            bias=bias,
                            task_cap=args.task_cap,
                            batch_size=args.batch,
                        )
                        values = cell_values(events, contract.evaluation.primary_metric)
                        per_task = {int(task): v for (_, task, _), v in values.items()}
                        for task_id, value in sorted(per_task.items()):
                            writer.writerow(
                                {
                                    "cell": cell,
                                    "seed": seed,
                                    "label": bias.label,
                                    "target": bias.target,
                                    "beta": bias.beta,
                                    "task_id": task_id,
                                    "primary": value,
                                }
                            )
                        handle.flush()
                        stem = f"{cell}-seed{seed}-{bias.label}"
                        _save_decisions(
                            out / f"decisions-{stem}.npz",
                            decisions,
                            heads=bias.target == "none",
                        )
                        if bias.target == "none":
                            _write_labels(out / f"labels-{cell}-seed{seed}.csv", events)
                            saved = _saved_panel(config, contract, args.split)
                            mismatched = [
                                task
                                for task, value in per_task.items()
                                if task not in saved or saved[task] != value
                            ]
                            checks[f"{cell}-seed{seed}"] = {
                                "checkpoint": checkpoint,
                                "tasks": len(per_task),
                                "mismatched_tasks": mismatched[:20],
                                "status": "passed" if not mismatched else "failed",
                            }
                        mean = float(np.mean(list(per_task.values())))
                        print(
                            f"{cell} seed {seed} {bias.label}: primary {mean:.4f} "
                            f"({len(per_task)} tasks, "
                            f"{time.perf_counter() - started:.0f} s)",
                            flush=True,
                        )
                finally:
                    close_experiment(experiment)
    check_path = out / "acceptance.json"
    previous = json.loads(check_path.read_text()) if check_path.exists() else {}
    previous.update(checks)
    check_path.write_text(json.dumps(previous, indent=2) + "\n")
    failed = [name for name, check in checks.items() if check["status"] != "passed"]
    if failed and args.task_cap is None:
        raise ContractError(f"Captures disagree with their saved panels: {failed}.")
    return 0


def read(args: argparse.Namespace) -> int:
    study = load_summary_memory_study(args.study)
    out, contract = _destination(args, study)
    tables = read_attention(
        out,
        benchmark=contract.environment.name,
        memory_tokens=int(study_memory_tokens(study, args)),
    )
    for name, rows in tables.items():
        if not rows:
            continue
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (out / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out / f'{name}.csv'} ({len(rows)} rows)")
    return 0


def study_memory_tokens(study: Any, args: argparse.Namespace) -> int:
    _, config = resolve(
        study,
        benchmark=args.benchmark,
        condition="raw_summary",
        seed=study.training_seeds[0],
        device="cpu",
    )
    summary = config.model.summary
    if summary is None:
        raise ContractError("raw_summary carries no summary spec.")
    return int(summary.memory_tokens)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(args) if args.mode == "run" else read(args)


if __name__ == "__main__":
    sys.exit(main())
