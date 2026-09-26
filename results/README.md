# Results

The saved records behind the paper's figures and tables, copied from the study roots (`outputs/<root>/`) with their directory layout kept. They were written by the readers and notebooks of this repository from the per-task evaluation panels, which are not included (about 15 GB), nor are the trained weights. Rows of cells outside the paper are omitted from the tier reports; hosts, paths and account names are replaced by placeholders (`node-N`, `<user>`, `<entity>`); no value is edited. Statistics are defined in [EXPERIMENTS](../docs/EXPERIMENTS.md#statistics).

## Figure and table data

`figure_data/` holds the tables the figure and table notebooks save:

- `paper_<figure>_plotted.csv` and `paper_<figure>_notes.json`: exactly what each print-size figure draws, and the roster, seeds, bootstrap settings and record paths behind it (results at a glance, state, interventions, retention age, in-context adaptation, new tasks, transplant, representation detail, attention, ablations, learning).
- `tables_three_benchmarks_*.{csv,md}`: the results table's endpoints and interventions (`table1_endpoints`), every contrast of the family with its estimate, joint interval, per-seed differences, δ and disposition (`table1_contrasts`), deployment state (`table_state_memory*`), cost per GPU model (`table2_costs`) and learning thresholds (`table_a1_learning`).
- The upstream tables those figures redraw (`figure2_*`, `figure3_*`, `figureA1_*`, `figure_attention_summary_*`, `figure_countrecall_boundary_segments_*`, `figure_mazerunner_*`, `figure_summary_length_*`).

## Reports

| Path | Contents |
|---|---|
| `<root>/reports/<protocol>/confirmation/tier/` | Tier report per benchmark: `cells.csv` (estimate, interval, per-seed values per cell and history), `contrasts.csv` (paired estimate, joint interval, per-seed differences and per-seed intervals, δ, disposition), `interventions.csv`, `costs.csv` |
| `memo-key-to-door-8m/reports/dark-key-to-door/confirmation/horizon/h4000/` | Key-to-Door same task to 4,000 calls: doors per 500-call window and contrasts for every cell |
| `…/horizon/h4000-relayout500[-raw_summary_residual]/` | Key-to-Door with a new layout every 500 calls |
| `summary-memory-8m/reports/dark-key-to-door/confirmation/horizon/h4000/` | The same 4,000-call read with the sliding window |
| `summary-memory-8m/reports/dark-key-to-door/confirmation/capacity/raw_summary/` | Summary length M = 1, 8, 16 against M = 4 |
| `mazerunner-8m/reports/mazerunner/confirmation/laps/{retained,attempt-cleared,summary-cleared}/` | MazeRunner same-map laps to 4,000 calls: goals per 500-call window with memory kept, wiped at every lap, or with the summary cleared |
| `mazerunner-8m/reports/mazerunner-15-randomized-actions/confirmation/window/` | The sliding window on MazeRunner |
| `…/confirmation/attention/` | Attention masses, read-bias dose tables and readings (Key-to-Door, CountRecall) |
| `…/confirmation/representation/` | Projections, probe R², the room map, per-window attention, centred similarity and norms, and the transplant readings |
| `summary-memory-8m/reports/count-recall-medium/confirmation/tier/` | The CountRecall tier report of the main study, holding the RSM detach contrasts |
| `<root>/<protocol>/<cell>/seed-<N>/{systems,development}.json` | Per-run measured calls, state bytes, FLOPs and latency, and the development series |

Study roots: `summary-memory-8m` (Key-to-Door and CountRecall), `memo-key-to-door-8m` and `memo-count-recall-8m` (Memo, and the held-out passes that re-score every cell on one roster), `tmaze-v3-8m`, `mazerunner-8m`, and `keydoor-capacity-{m1,m8,}-8m-*` (summary length). Contrast names in the T-Maze tier report follow its study file, where "RSM" names `raw_summary_residual`; read the `left` and `right` columns, or `figure_data/tables_three_benchmarks_table1_contrasts.csv`, which uses the paper's names.
