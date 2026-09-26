# Figure and table notebooks

Every display of the paper is built from saved records only; nothing here trains or evaluates a model. Each notebook holds its own estimators and drawing code (the cell tagged `definitions`), and [`figures_common.py`](figures_common.py) carries what the figures share: where the study roots and reports are, the benchmark table, the roster rule, the palette and legend names, the deployment-state projection and the save step.

| Notebook | Builds |
|---|---|
| [paper_figures](paper_figures.ipynb) | The paper's print-size figures (5.5 in, 7-pt type): results at a glance, deployment state, interventions, retention by rewrite age, room attention, in-context adaptation, new tasks, transplant, what the summary holds, attention, ablations and learning curves. It redraws the tables the notebooks below save, so it runs last |
| [tables_three_benchmarks](tables_three_benchmarks.ipynb) | The results table (endpoints, interventions, the contrast family per benchmark, quoted from the tier reports), the deployment-state table, the cost table and the learning-threshold table, as CSV, Markdown and LaTeX |
| [figure2_in_context_adaptation](figure2_in_context_adaptation.ipynb) | In-context adaptation at the training horizon and the intervention views |
| [countrecall_boundary_segments](countrecall_boundary_segments.ipynb) | CountRecall accuracy within each C = 32 segment, by the number of rewrites since the evidence |
| [figure3_beyond_the_training_horizon](figure3_beyond_the_training_horizon.ipynb) | The 4,000-call Key-to-Door windows and CountRecall retention, with deployment state against task length |
| [mazerunner_repeated_laps](mazerunner_repeated_laps.ipynb) | MazeRunner same-map laps and map changes over 4,000 calls, memory kept or wiped |
| [figure4_learning_matched_experience](figure4_learning_matched_experience.ipynb) | Learning curves against charged training calls |
| [figure_attention_summary](figure_attention_summary.ipynb) | Summary attention per segment position and the read-bias dose |
| [figure_summary_length](figure_summary_length.ipynb) | The summary-length ablation (M = 1, 4, 8, 16) on Key-to-Door |

## Running

Headless, each notebook from a fresh kernel:

```bash
uv run python notebooks/run_figures.py                 # every notebook, tables and print-size figures last
uv run python notebooks/run_figures.py --only figure2  # one of them
```

`REASONED_ICRL_OUTPUTS` points at the study roots (default `outputs/`) and `FIGURE_OUTPUT_ROOT` at the figure destination. Each composite is saved as PDF and 300-dpi PNG to `notebooks/outputs/figures/`, with every panel on its own, and each plotted table plus a notes file (roster per column, seeds, bootstrap settings, every record path read) to `notebooks/outputs/data/`; the copies used by the paper are in [`results/figure_data/`](../results/README.md). Figure 3 reads about 2 GB of 4,000-call panels; `--timeout` raises the per-cell limit on a slow file system.

## Rules the code enforces

- One roster per column: the confirmation roster when every cell has records there. A cell without records is named in the legend and its panel stays empty; nothing missing is drawn.
- Completed fits only: a run enters the learning figure only when its measured budget is the full 8M.
- Smoothing is for drawing only; every saved number uses the raw series.
- Tables quote the tier reports as written; every contrast is matched to the reports by its cell pair.
- Deployment state follows one rule per cell (Memo its per-slot schedule, the full history one cache slot per record, the bounded cells their fixed buffers) and is drawn as the peak an actor holds within the task.

Tests: `tests/analysis/test_paper_notebooks.py` executes the `definitions` cells on synthetic records; `tests/test_scripts.py` executes each notebook from a fresh kernel against an empty study root.
