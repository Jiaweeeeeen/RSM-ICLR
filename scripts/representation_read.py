"""What the summary carries, where it is read over long tasks, and a transplant.

    python scripts/representation_read.py run --study configs/summary_memory_8m.yaml \\
        --benchmark dark_key_to_door --split confirmation --cells raw_summary \\
        [--horizon 4000 | --laps 4000] [--task-cap 64] [--seeds 42 100 2026]
    python scripts/representation_read.py transplant \\
        --study configs/summary_memory_8m.yaml --benchmark dark_key_to_door \\
        --segment 8 [--offset 128]
    python scripts/representation_read.py read --study configs/summary_memory_8m.yaml \\
        --benchmark dark_key_to_door

The representation declaration. ``run`` evaluates the
frozen endpoint of every requested cell and seed through the shared evaluator
with the unbiased attention probe and the summary capture attached, at the
trained horizon, a longer Key-to-Door horizon (``--horizon``) or MazeRunner
laps (``--laps``); each capture must reproduce its saved retained panel of the
same horizon task for task (``acceptance.json``). ``transplant`` replays the
method's trained-horizon capture with, at the boundary opening ``--segment``,
every task's summary replaced by the one the task ``--offset`` roster places
later read there, and again with the initial memory (*cleared once*). Under
``<study root>/reports/<protocol>/<split>/representation/`` (or ``--out``) it
writes per evaluation the summaries (``summaries-*.npz``), the decisions'
inputs and actions (``inputs-*.npz``), their attention masses per block
(``decisions-*.npz``), the evaluator events as labels (``labels-*.csv``), the
per-task primary metric (``tasks.csv``) and, on Key-to-Door, the hidden cells
of every task (``layouts.csv``, evaluator labels). ``read`` turns them into the
display tables and the declared transplant readings with
:mod:`reasoned_icrl.analysis.representation`. Nothing is trained and no
checkpoint is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from reasoned_icrl.analysis.attention import LABEL_FIELDS
from reasoned_icrl.environments.count_recall import CountRecallEnv
from reasoned_icrl.experiments.artifacts import endpoint_training_epoch
from reasoned_icrl.experiments.benchmarks import saved_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import evaluation_environment
from reasoned_icrl.experiments.horizon import continued_laps, extended_horizon
from reasoned_icrl.experiments.records import cell_values, read_benchmark_results
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve
from reasoned_icrl.runtime.representation import (
    SummaryTransplant,
    donor_order,
    keydoor_layouts,
    representation_evaluation,
)
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
    load_selected_checkpoint,
)

TASK_FIELDS = ("cell", "seed", "label", "task_id", "primary")
METHOD = "raw_summary"


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    out.add_argument("mode", choices=("run", "transplant", "read"))
    out.add_argument("--study", type=Path, required=True)
    out.add_argument("--benchmark", required=True)
    out.add_argument("--split", default="confirmation")
    out.add_argument("--cells", nargs="+", default=[METHOD])
    out.add_argument("--seeds", nargs="+", type=int, default=None)
    horizon = out.add_mutually_exclusive_group()
    horizon.add_argument("--horizon", type=int, default=None)
    horizon.add_argument("--laps", type=int, default=None)
    out.add_argument("--segment", type=int, default=None)
    out.add_argument("--offset", type=int, default=128)
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
        condition=METHOD,
        seed=study.training_seeds[0],
        device="cpu",
    )
    root = Path(config.run_directory).parents[2]
    return (
        root / "reports" / contract.protocol / args.split / "representation",
        contract,
    )


def _suffix(args: argparse.Namespace) -> str:
    if args.horizon is not None:
        return f"-h{int(args.horizon)}"
    if args.laps is not None:
        return f"-calls{int(args.laps)}"
    return ""


def _extended(args: argparse.Namespace, contract: Any, config: Any) -> tuple[Any, Any]:
    if args.horizon is not None:
        return extended_horizon(contract, config, int(args.horizon))
    if args.laps is not None:
        return continued_laps(contract, config, int(args.laps))
    return contract, config


def _saved_panel(
    run_directory: Path, contract: Any, split: str, suffix: str
) -> dict[int, float]:
    path = (
        run_directory
        / "eval"
        / f"{split}-retained-endpoint{suffix}"
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


def _save(
    out: Path,
    stem: str,
    events: Any,
    decisions: dict[str, Any],
    summaries: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    """One evaluation's files; attention masses as block means (float32)."""
    masses: dict[str, Any] = {
        name: decisions[name] for name in ("task_id", "step", "segment", "position")
    }
    for name in ("summary", "buffer", "own"):
        masses[f"{name}_layer"] = decisions[name].mean(axis=2).astype(np.float32)
    np.savez_compressed(out / f"decisions-{stem}.npz", **masses)
    np.savez_compressed(out / f"summaries-{stem}.npz", **summaries)
    np.savez_compressed(out / f"inputs-{stem}.npz", **inputs)
    _write_labels(out / f"labels-{stem}.csv", events)


class Runner:
    """Loads each (cell, seed) endpoint once and evaluates it."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.study = load_summary_memory_study(args.study)
        self.out, self.contract = _destination(args, self.study)
        self.out.mkdir(parents=True, exist_ok=True)
        self.seeds = tuple(args.seeds or self.study.training_seeds)

    def evaluate(
        self,
        cell: str,
        seed: int,
        *,
        label: str,
        transplant: SummaryTransplant | None = None,
        acceptance: bool = True,
    ) -> dict[int, float]:
        args = self.args
        _, requested = resolve(
            self.study,
            benchmark=args.benchmark,
            condition=cell,
            seed=seed,
            device=args.device,
        )
        native_config = saved_config(requested)
        run_directory = Path(native_config.run_directory)
        epoch = endpoint_training_epoch(
            run_directory, epochs=native_config.training.epochs
        )
        checkpoint = f"policy_epoch_{epoch}"
        contract, config = _extended(args, self.contract, native_config)
        experiment = load_experiment(
            config, work_directory=run_directory, persist_configuration=False
        )
        started = time.perf_counter()
        try:
            load_selected_checkpoint(experiment, checkpoint)
            events, decisions, summaries, inputs = representation_evaluation(
                contract,
                config,
                experiment,
                checkpoint=checkpoint,
                split=args.split,
                transplant=transplant,
                task_cap=args.task_cap,
                batch_size=args.batch,
            )
        finally:
            close_experiment(experiment)
        stem = f"{cell}-seed{seed}-{label}"
        _save(self.out, stem, events, decisions, summaries, inputs)
        values = cell_values(events, contract.evaluation.primary_metric)
        per_task = {int(task): value for (_, task, _), value in values.items()}
        tasks_path = self.out / "tasks.csv"
        fresh = not tasks_path.exists()
        with tasks_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS)
            if fresh:
                writer.writeheader()
            for task_id, value in sorted(per_task.items()):
                writer.writerow(
                    {
                        "cell": cell,
                        "seed": seed,
                        "label": label,
                        "task_id": task_id,
                        "primary": value,
                    }
                )
        if acceptance:
            saved = _saved_panel(run_directory, contract, args.split, _suffix(args))
            mismatched = [
                task
                for task, value in per_task.items()
                if task not in saved or saved[task] != value
            ]
            check_path = self.out / "acceptance.json"
            checks = json.loads(check_path.read_text()) if check_path.exists() else {}
            checks[stem] = {
                "checkpoint": checkpoint,
                "tasks": len(per_task),
                "mismatched_tasks": mismatched[:20],
                "status": "passed" if not mismatched else "failed",
            }
            check_path.write_text(json.dumps(checks, indent=2) + "\n")
            if mismatched:
                raise ContractError(
                    f"{stem}: the capture disagrees with its saved panel on "
                    f"{len(mismatched)} tasks."
                )
        mean = float(np.mean(list(per_task.values())))
        print(
            f"{stem}: primary {mean:.4f} ({len(per_task)} tasks, "
            f"{time.perf_counter() - started:.0f} s)",
            flush=True,
        )
        return per_task


def _write_layouts(runner: Runner) -> None:
    args = runner.args
    _, requested = resolve(
        runner.study,
        benchmark=args.benchmark,
        condition=METHOD,
        seed=runner.seeds[0],
        device="cpu",
    )
    contract, config = _extended(args, runner.contract, saved_config(requested))
    roster = contract.roster(args.split)
    if args.task_cap is not None:
        roster = roster[: args.task_cap]
    layouts = keydoor_layouts(contract, config, split=args.split, task_ids=roster)
    path = runner.out / "layouts.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "cell", "row", "column"])
        for task, cells in layouts.items():
            for name, (row, column) in cells.items():
                writer.writerow([task, name, row, column])


def run(args: argparse.Namespace) -> int:
    runner = Runner(args)
    if args.benchmark == "dark_key_to_door" and args.horizon is None:
        _write_layouts(runner)
    label = f"retained{_suffix(args)}"
    for cell in args.cells:
        for seed in runner.seeds:
            runner.evaluate(cell, seed, label=label)
    return 0


def transplant(args: argparse.Namespace) -> int:
    if args.segment is None:
        raise ContractError("A transplant names the segment it opens (--segment).")
    if args.horizon is not None or args.laps is not None or args.cells != [METHOD]:
        raise ContractError(
            "The declared transplant replays the method's trained-horizon capture."
        )
    runner = Runner(args)
    roster = runner.contract.roster(args.split)
    if args.task_cap is not None:
        roster = roster[: args.task_cap]
    pairs = donor_order(roster, args.offset)
    for seed in runner.seeds:
        path = runner.out / f"summaries-{METHOD}-seed{seed}-retained.npz"
        if not path.exists():
            raise ContractError(f"Run the retained capture first: {path} is missing.")
        with np.load(path) as stored:
            chosen = stored["segment"] == int(args.segment)
            by_task = {
                int(task): memory
                for task, memory in zip(
                    stored["task_id"][chosen], stored["memory"][chosen], strict=True
                )
            }
        donors = {recipient: by_task[donor] for recipient, donor in pairs.items()}
        for summary in (
            SummaryTransplant(segment=args.segment, donors=donors),
            SummaryTransplant(segment=args.segment),
        ):
            runner.evaluate(
                METHOD, seed, label=summary.label, transplant=summary, acceptance=False
            )
            if sorted(summary.applied) != sorted(pairs):
                raise ContractError(
                    f"{summary.label}: not every task was transplanted."
                )
    return 0


def read(args: argparse.Namespace) -> int:
    from reasoned_icrl.analysis.representation import read_representation

    study = load_summary_memory_study(args.study)
    out, contract = _destination(args, study)
    development = out.parents[1] / "development" / "representation"
    roster = contract.roster(args.split)
    decode = None
    categories = 4
    if contract.environment.name == "count_recall":
        _, requested = resolve(
            study,
            benchmark=args.benchmark,
            condition=METHOD,
            seed=study.training_seeds[0],
            device="cpu",
        )
        environment = evaluation_environment(
            contract,
            saved_config(requested),
            split=args.split,
            seed=contract.evaluation.rollout_seeds[0],
        )
        if not isinstance(environment, CountRecallEnv):
            raise ContractError("The CountRecall contract built another environment.")
        categories = int(environment.categories)

        def decode(current: Any) -> tuple[int, int]:
            value, query, _ = environment.decode({"current": current})
            return value, query

    tables = read_representation(
        out,
        benchmark=contract.environment.name,
        development=development if development.is_dir() else None,
        decode=decode,
        donors=donor_order(roster, args.offset) if len(roster) > args.offset else None,
        categories=categories,
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


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    modes = {"run": run, "transplant": transplant, "read": read}
    return modes[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
