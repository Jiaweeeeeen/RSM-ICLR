"""Build, verify or witness the XLand one-rule task manifest.

    python scripts/build_xland_one_rule_manifest.py             # rebuild and verify
    python scripts/build_xland_one_rule_manifest.py --write     # (re)write the file
    python scripts/build_xland_one_rule_manifest.py --witnesses <path>

Filters the pinned ``small-1m`` corpus under the one-rule protocol, selects the
goal objects and allocates every stratum with the frozen hash-sort, and prints
the ledger, census and coverage. Without ``--write`` the committed manifest
must reproduce byte for byte. ``--witnesses`` plans a native witness for
every task on every declared layout fixture, verifies it with and without the
hidden rule on the CPU simulator, and writes the record. Nothing here trains,
evaluates a policy or touches a study root other than the witness file named.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.xland_one_rule import (
    LAYOUT_FIXTURES,
    build_manifest,
    load_manifest,
    manifest_path,
    manifest_text,
    run_witnesses,
    witness_summary,
    write_manifest,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output", type=Path, default=None, help="manifest path (default: configs)"
    )
    result.add_argument(
        "--write", action="store_true", help="write the rebuilt manifest"
    )
    result.add_argument(
        "--witnesses", type=Path, default=None, help="write the witness record here"
    )
    result.add_argument(
        "--fixtures", type=int, nargs="*", default=None, help="layout indices"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    target = manifest_path() if args.output is None else args.output
    try:
        manifest = build_manifest()
        text = manifest_text(manifest)
        if args.write:
            write_manifest(manifest, target)
            print(f"wrote {target} ({len(text)} bytes)")
        elif not target.is_file():
            print(f"manifest: {target} does not exist; pass --write", file=sys.stderr)
            return 2
        elif target.read_text(encoding="utf-8") != text:
            print(f"manifest: {target} does not reproduce", file=sys.stderr)
            return 2
        else:
            print(f"{target} reproduces from the pinned corpus")
        saved = load_manifest(target)
    except ContractError as error:
        print(f"manifest: {error}", file=sys.stderr)
        return 2
    print(f"protocol {saved['protocol']}")
    print(f"corpus {saved['corpus']['file']} sha256 {saved['corpus']['sha256']}")
    print("counts " + json.dumps(saved["counts"], sort_keys=True))
    print("ledger " + json.dumps(saved["ledger"], sort_keys=True))
    census = {k: v for k, v in saved["census"].items() if k != "qualifying_goals"}
    print("census " + json.dumps(census, sort_keys=True))
    print("goals " + json.dumps([g["object"] for g in saved["goals"]]))
    print("coverage " + json.dumps(saved["coverage"], sort_keys=True))
    if args.witnesses is not None:
        fixtures = LAYOUT_FIXTURES if args.fixtures is None else tuple(args.fixtures)
        witnesses = run_witnesses(saved, fixtures=fixtures)
        summary = witness_summary(witnesses)
        record = {
            **summary,
            "protocol": saved["protocol"],
            "manifest_corpus_sha256": saved["corpus"]["sha256"],
            "fixtures": list(fixtures),
            "rows": [w.as_dict() for w in witnesses],
        }
        args.witnesses.parent.mkdir(parents=True, exist_ok=True)
        args.witnesses.write_text(
            json.dumps(record, sort_keys=True, indent=1) + "\n", encoding="utf-8"
        )
        print("witnesses " + json.dumps(summary, sort_keys=True))
        print(f"wrote {args.witnesses}")
        if summary["passed"] != summary["witnesses"]:
            print("witnesses: not every task x fixture passed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
