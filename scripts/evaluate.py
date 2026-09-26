"""Evaluate one saved run of a study and write its benchmark records.

    python scripts/evaluate.py summary_memory --benchmark dark_key_to_door \
        --condition fixed_summary --seed 42 --split development --history retained
    python scripts/evaluate.py summary_memory --benchmark count_recall \
        --condition full_dual_relational --seed 42 --checkpoint-rule endpoint

The study name is a subcommand so that a second study can register beside the
active one without changing the launchers.
"""

from __future__ import annotations

import argparse
import sys

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.summary_memory import experiments as summary_memory

STUDIES = {"summary_memory": summary_memory}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    studies = result.add_subparsers(dest="study_name", required=True)
    for name, study in STUDIES.items():
        study.add_evaluate_arguments(studies.add_parser(name))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        print(STUDIES[args.study_name].evaluate_main(args))
    except ContractError as error:
        print(f"evaluate: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
