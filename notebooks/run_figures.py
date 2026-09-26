"""Execute each paper figure and table notebook from a fresh kernel, output-free
on disk.

    python notebooks/run_figures.py [--only figure2 countrecall figure3 mazerunner
                                     figure4 figure_attention figure_summary tables
                                     paper]
                                    [--outputs DIR] [--out DIR]

The executed notebook is validated and discarded; the committed notebook stays
free of outputs. The figures land under ``--out`` (default
``notebooks/outputs/figures``), the tables and notes under its ``data/``
sibling. The tables notebook runs after the figures so that it can read their
saved tables (the horizon contrasts and the first crossings),
and the paper figures notebook last, since it redraws those saved tables at
print size.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = (
    "figure2_in_context_adaptation",
    "countrecall_boundary_segments",
    "figure3_beyond_the_training_horizon",
    "mazerunner_repeated_laps",
    "figure4_learning_matched_experience",
    "figure_attention_summary",
    "figure_summary_length",
    "tables_three_benchmarks",
    "paper_figures",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=(
            "figure2",
            "countrecall",
            "figure3",
            "mazerunner",
            "figure4",
            "figure_attention",
            "figure_summary",
            "tables",
            "paper",
        ),
    )
    parser.add_argument("--outputs", type=Path, help="study roots (default outputs/)")
    parser.add_argument("--out", type=Path, help="figure destination")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.outputs is not None:
        os.environ["REASONED_ICRL_OUTPUTS"] = str(args.outputs.resolve())
    if args.out is not None:
        os.environ["FIGURE_OUTPUT_ROOT"] = str(args.out.resolve())
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["PATH"] = str(repo / ".venv/bin") + os.pathsep + os.environ["PATH"]
    for stem in NOTEBOOKS:
        if args.only and not any(stem.startswith(prefix) for prefix in args.only):
            continue
        path = repo / "notebooks" / f"{stem}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        print(f"Fresh kernel: {path.name}", flush=True)
        NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(repo)}},
        ).execute()
        nbformat.validate(notebook)
        print(f"Executed: {path.name}", flush=True)


if __name__ == "__main__":
    main()
