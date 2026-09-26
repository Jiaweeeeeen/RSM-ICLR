"""Plan one tier's fits and write the pool launcher's job file.

    python scripts/generate_summary_memory_jobs.py --benchmark dark_key_to_door \
        --group primary --output outputs/summary-memory-8m/queue/tier0.jobs

Prints every planned (condition, seed) with its group and what its run
directory already holds, the counts, and the train -> evaluate dependencies.
Nothing is submitted: pass the written file to scripts/slurm/submit_summary_memory.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.summary_memory.configs import (
    load_summary_memory_study,
)
from reasoned_icrl.experiments.summary_memory.jobs import (
    evaluation_dependencies,
    plan_fits,
    write_jobs_file,
)
from reasoned_icrl.utils import repository_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--study", type=Path, default=None)
    result.add_argument("--benchmark", required=True)
    result.add_argument(
        "--group", default="primary", choices=("primary", "supplementary", "all")
    )
    result.add_argument("--seeds", type=int, nargs="*", default=None)
    result.add_argument("--conditions", nargs="*", default=None)
    result.add_argument("--output-root", type=Path, default=None)
    result.add_argument("--output", type=Path, default=None, help="job file to write")
    result.add_argument(
        "--include-complete",
        action="store_true",
        help="list completed fits as live lines (the queue still skips them)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        study = load_summary_memory_study(args.study)
        contract = study.contract(args.benchmark)
        plans = plan_fits(
            study,
            contract,
            repository=repository_root(),
            group=args.group,
            seeds=args.seeds,
            conditions=args.conditions,
            output_root=args.output_root,
        )
    except ContractError as error:
        print(f"generate-jobs: {error}", file=sys.stderr)
        return 2
    width = max((len(plan.condition) for plan in plans), default=9)
    print(f"{contract.protocol}: {len(plans)} {args.group} fits")
    for plan in plans:
        print(
            f"  {plan.condition:<{width}} seed {plan.seed:<5} {plan.group:<13} "
            f"{plan.status:<9} {plan.run_directory}"
        )
    counts = {
        status: sum(plan.status == status for plan in plans)
        for status in ("complete", "resumable", "started", "missing")
    }
    print("counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("dependencies:")
    for row in evaluation_dependencies(contract):
        print(f"  - {row}")
    if args.output is not None:
        written = write_jobs_file(
            args.output,
            plans,
            header=(
                f"{study.name} {contract.protocol} {args.group} group; seeds "
                f"{list(study.training_seeds if args.seeds is None else args.seeds)}"
            ),
            include_complete=args.include_complete,
        )
        print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
