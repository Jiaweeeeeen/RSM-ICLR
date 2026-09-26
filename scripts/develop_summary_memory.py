"""Score the development series of every declared run that has new checkpoints.

    python scripts/develop_summary_memory.py --benchmark dark_key_to_door \
        [--conditions ...] [--seeds ...] [--device cpu] [--task-cap N] \
        [--report] [--gate] [--loop-minutes M]

For each declared cell and seed under the active study root, evaluates every
scheduled checkpoint that the run has saved and the record has not scored yet
(the 0.4M development grid, extended while training continues), selects the
best development checkpoint and writes the selected checkpoint's retained and
intervention records; a run that reached its endpoint also gets its
development-split endpoint and selected panels (``--no-panels`` skips them).
``--report`` then regenerates the development tier
report, ``--gate`` the tier qualification dashboard. With ``--loop-minutes``
the pass repeats until every declared run has reached its endpoint.
``--finalize`` evaluates the final roster (endpoint and selected panels of
every primary cell and seed) and writes the final tier report, and is refused
unless every primary fit has reached its endpoint: the held-out roster is read
only once the whole matrix is complete.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.qualification import scheduled_epochs
from reasoned_icrl.experiments.summary_memory.configs import (
    load_summary_memory_study,
)
from reasoned_icrl.experiments.summary_memory.jobs import (
    fit_status,
    reconcile_resumed_runs,
    require_primary_training_complete,
)
from reasoned_icrl.experiments.summary_memory.revised import (
    develop_run,
    finalize_revised_cell,
    gate_tier,
    read_run_development,
    reference_configs,
)
from reasoned_icrl.utils import repository_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--study", type=Path, default=None)
    result.add_argument("--benchmark", required=True)
    result.add_argument("--conditions", nargs="*", default=None)
    result.add_argument("--seeds", type=int, nargs="*", default=None)
    result.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    result.add_argument("--output-root", type=Path, default=None)
    result.add_argument("--task-cap", type=int, default=None)
    result.add_argument("--report", action="store_true")
    result.add_argument("--gate", action="store_true")
    result.add_argument("--loop-minutes", type=float, default=None)
    result.add_argument("--samples", type=int, default=2000)
    result.add_argument(
        "--no-panels",
        action="store_true",
        help="do not write the development endpoint/selected panels of runs "
        "that reached their endpoint",
    )
    result.add_argument(
        "--final-split",
        default="final",
        help="the held-out split --finalize scores and reports (default `final`; "
        "the Memo comparator's locked panel is `confirmation`)",
    )
    result.add_argument(
        "--panel-rules",
        nargs="+",
        default=None,
        choices=("endpoint", "selected"),
        help="restrict --finalize to these checkpoint panels (default: the "
        "contract's rules); `endpoint` alone when the development-selected "
        "weights of a run were pruned after its report",
    )
    result.add_argument(
        "--final-only",
        action="store_true",
        help="with --finalize: skip the development pass, gate and report and "
        "go straight to the held-out roster (the development records must "
        "already be complete, e.g. scored by a running development loop)",
    )
    result.add_argument(
        "--finalize",
        action="store_true",
        help="evaluate the final roster and write the final report; refused "
        "unless every primary fit reached its endpoint",
    )
    return result


def one_pass(args: argparse.Namespace) -> bool:
    """Develop every run with unscored checkpoints; True when all reached the end."""
    root = repository_root()
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    output_root = (
        root / study.output_root if args.output_root is None else args.output_root
    )
    cells = study.cells(contract)
    if args.conditions:
        unknown = [c for c in args.conditions if c not in cells]
        if unknown:
            raise ContractError(f"{unknown} are not cells of {contract.name}.")
        cells = tuple(c for c in cells if c in args.conditions)
    seeds = study.training_seeds if not args.seeds else tuple(args.seeds)
    # R6: a run resumed from a pre-R6 snapshot reports session-only counters
    # until reconciled; do it before any report, gate or guard reads them
    # (idempotent; the original file is kept beside the reconciled one).
    for path in reconcile_resumed_runs(study, contract, output_root):
        print(f"reconciled counters and sessions: {path}")
    if getattr(args, "final_only", False):
        if not args.finalize:
            raise ContractError("--final-only requires --finalize.")
        # R6: a second evaluator (a CPU host beside a GPU development loop)
        # must not score development labels concurrently with the loop.
        finalize(study, contract, output_root, args)
        return True
    all_done = True
    for condition in cells:
        for seed in seeds:
            config = experiment_config(
                contract,
                study,
                condition=condition,
                seed=seed,
                repository=root,
                device=args.device,
                output_root=output_root,
            )
            directory = config.run_directory
            status = fit_status(directory)
            if status == "missing":
                print(f"{condition} seed {seed}: missing (not started)")
                all_done = False
                continue
            saved = scheduled_epochs(directory)
            if (
                contract.name == "match_pattern"
                and (directory / "initial_checkpoint.pt").is_file()
            ):
                saved = [-1, *saved]
            record = read_run_development(directory)
            scored = set() if record is None else {s.epoch for s in record.series}
            pending = [e for e in saved if e not in scored]
            if not pending and record is not None:
                print(
                    f"{condition} seed {seed}: {len(saved)} checkpoints scored, "
                    f"selected e{record.selected_epoch} "
                    f"{record.selected_primary:.3f}, {status}"
                )
            elif not saved:
                print(f"{condition} seed {seed}: no checkpoint saved yet ({status})")
            else:
                started = time.perf_counter()
                record = develop_run(study, contract, config, task_cap=args.task_cap)
                print(
                    f"{condition} seed {seed}: scored {pending} in "
                    f"{time.perf_counter() - started:.0f} s; selected "
                    f"e{record.selected_epoch} {record.selected_primary:.3f}; "
                    f"{status}"
                )
            if record is None or not record.endpoint_reached:
                all_done = False
            elif not args.no_panels:
                # A completed run gets its development-split endpoint panel
                # (every history mode) and selected panel; records that exist
                # are kept, so this is idempotent across passes.
                written = finalize_revised_cell(
                    study, contract, config, split="development", task_cap=args.task_cap
                )
                print(f"  development panels: {sorted(written)}")
    if args.gate:
        gate = gate_tier(
            study,
            contract,
            output_root,
            repository=root,
            device=args.device,
            task_cap=args.task_cap,
            samples=args.samples,
        )
        print(f"gate: {gate.status}")
    if args.report:
        from reasoned_icrl.analysis.tier import write_tier_report

        report = write_tier_report(
            study, contract, output_root, split="development", samples=args.samples
        )
        print(f"report: {report.root}; pending {list(report.pending_primary_cells)}")
        for note in report.notes:
            print(f"  note: {note}")
    if args.finalize:
        finalize(study, contract, output_root, args)
    return all_done


def finalize(
    study: Any, contract: Any, output_root: Path, args: argparse.Namespace
) -> None:
    """The held-out pass, only once every primary fit reached its endpoint.

    Every primary cell is scored; a supplementary cell (a declared variant such
    as the T-Maze v3 gated write) is scored only when named in ``--conditions``
    and only if its own three fits reached their endpoints, so the report's
    contrasts that involve it fill in without the variant ever entering the
    primary set.
    """
    from reasoned_icrl.analysis.tier import completeness_rows, write_tier_report

    root = repository_root()
    for path in reconcile_resumed_runs(study, contract, output_root):
        print(f"reconciled counters and sessions: {path}")
    if contract.name in ("concentration", "count_recall", "match_pattern", "tmaze"):
        require_primary_training_complete(study, contract, output_root)
    split = str(getattr(args, "final_split", "final"))
    if split not in contract.evaluation.splits or split == "development":
        raise ContractError(
            f"{contract.protocol} declares no held-out split {split!r}."
        )
    ledger = completeness_rows(study, contract, output_root, split=split)
    pending = [
        f"{row['condition']} seed {row['seed']}: {row['status']}"
        for row in ledger
        if row["group"] == "primary" and not row["endpoint_reached"]
    ]
    if pending:
        raise ContractError(
            "The held-out roster is read only after every primary fit completes "
            f"its full budget; still pending: {'; '.join(pending)}."
        )
    plan = study.tier(contract.name)
    opted_in = tuple(
        c
        for c in plan.supplementary
        if getattr(args, "conditions", None) and c in args.conditions
    )
    not_done = [
        f"{row['condition']} seed {row['seed']}: {row['status']}"
        for row in ledger
        if row["condition"] in opted_in and not row["endpoint_reached"]
    ]
    if not_done:
        raise ContractError(
            "A supplementary cell is scored on the held-out roster only after "
            f"its fits complete; still pending: {'; '.join(not_done)}."
        )
    for condition in plan.primary + opted_in:
        for seed in study.training_seeds:
            config = experiment_config(
                contract,
                study,
                condition=condition,
                seed=seed,
                repository=root,
                device=args.device,
                output_root=output_root,
            )
            written = finalize_revised_cell(
                study,
                contract,
                config,
                split=split,
                task_cap=args.task_cap,
                rules=getattr(args, "panel_rules", None),
            )
            print(f"{split} {condition} seed {seed}: {sorted(written)}")
    # A comparator tier's frozen reference cells are scored on the same split
    # from their saved recipes under the reference root, beside their own
    # panels; nothing of the other study is re-resolved or rewritten.
    for config in reference_configs(
        study, contract, repository=root, device=args.device
    ):
        written = finalize_revised_cell(
            study,
            contract,
            config,
            split=split,
            task_cap=args.task_cap,
            rules=getattr(args, "panel_rules", None),
        )
        print(
            f"{split} reference {config.condition} seed {config.seed}: "
            f"{sorted(written)}"
        )
    report = write_tier_report(
        study, contract, output_root, split=split, samples=args.samples
    )
    print(f"{split} report: {report.root}")
    for note in report.notes:
        print(f"  note: {note}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    while True:
        try:
            done = one_pass(args)
        except ContractError as error:
            if args.loop_minutes is None:
                print(f"develop: {error}", file=sys.stderr)
                return 2
            # Several sessions edit the shared tree while a day-long series
            # runs; a pass that cannot load or evaluate the study is reported
            # and retried after the wait instead of ending the loop.
            print(
                f"{time.strftime('%FT%T%z')}: develop: {error}; retrying after "
                "the wait",
                file=sys.stderr,
                flush=True,
            )
            done = False
        if done or args.loop_minutes is None:
            return 0
        print(
            f"{time.strftime('%FT%T%z')}: waiting {args.loop_minutes:g} min for "
            "new checkpoints",
            flush=True,
        )
        time.sleep(args.loop_minutes * 60.0)


if __name__ == "__main__":
    raise SystemExit(main())
