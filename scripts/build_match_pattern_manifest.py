"""Materialize the frozen corpus and verify its manifest."""

from __future__ import annotations

import argparse
import json

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.match_pattern import (
    CORPUS_PATH,
    MANIFEST_PATH,
    generate_corpus,
    write_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the first frozen manifest and corpus.",
    )
    args = parser.parse_args()
    corpus = generate_corpus()
    if MANIFEST_PATH.exists() and json.loads(MANIFEST_PATH.read_text()) != json.loads(
        json.dumps(corpus.manifest)
    ):
        raise ContractError(
            "Refusing to replace a different frozen match-pattern manifest."
        )
    if args.write:
        write_corpus(corpus, CORPUS_PATH, MANIFEST_PATH)
    print(
        json.dumps(
            {
                "sha256": corpus.sha256,
                "counts": corpus.manifest["counts"],
                "written": args.write,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
