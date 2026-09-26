"""Train one run of a study.

    python scripts/train.py summary_memory --benchmark dark_key_to_door \
        --condition fixed_summary --seed 42 --device cuda
    python scripts/train.py summary_memory --benchmark count_recall \
        --condition full_dual_relational --seed 42 --device cpu --smoke

The study name is a subcommand so that a second study can register beside the
active one without changing the launchers.
"""

from __future__ import annotations

import argparse
import json
import sys

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.summary_memory import experiments as summary_memory

STUDIES = {"summary_memory": summary_memory}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    studies = result.add_subparsers(dest="study_name", required=True)
    for name, study in STUDIES.items():
        study.add_train_arguments(studies.add_parser(name))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result: object = STUDIES[args.study_name].train_main(args)
    except ContractError as error:
        print(f"train: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2) if isinstance(result, dict) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
