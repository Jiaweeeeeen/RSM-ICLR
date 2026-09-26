# Configurations

A **study** file (`configs/*.yaml`) names its output root, environment contracts, training seeds, conditions, the shared model block and, per benchmark, the cells, the practical margin δ and the contrasts. An **environment contract** (`configs/environments/8m/*.yaml`) fixes one task: public packet, horizon, training recipe, memory geometry and task rosters. Identity hashes are computed from the parsed specifications, so comments never change a run's identity. The files are kept as the fits were run; in `tmaze_v3_8m.yaml` the contrast names follow the declaration in force at the time (a header note explains).

## Studies of the paper

| Study | Output root | Benchmarks | Cells fitted under this root |
|---|---|---|---|
| [`summary_memory_8m.yaml`](summary_memory_8m.yaml) | `outputs/summary-memory-8m` | Key-to-Door, CountRecallMedium | `full_context`, `full_gru`, `raw_summary`, `raw_segment`, `raw_summary_residual`; `raw_window` (Key-to-Door); `raw_summary_detach` (CountRecall) |
| [`memo_key_to_door_8m.yaml`](memo_key_to_door_8m.yaml) | `outputs/memo-key-to-door-8m` | Key-to-Door | `memo`, `memo_fixed`; re-scores the main root's endpoints on the same roster |
| [`memo_count_recall_8m.yaml`](memo_count_recall_8m.yaml) | `outputs/memo-count-recall-8m` | CountRecallMedium | `memo`, `memo_fixed`; re-scores the main root's endpoints |
| [`tmaze_v3_8m.yaml`](tmaze_v3_8m.yaml) | `outputs/tmaze-v3-8m` | Passive T-Maze | seven paper cells (the file also lists a gated-write variant the paper does not use) |
| [`mazerunner_8m.yaml`](mazerunner_8m.yaml) | `outputs/mazerunner-8m` | MazeRunner 15×15, randomized actions | eight paper cells |
| [`keydoor_capacity_m1_8m.yaml`](keydoor_capacity_m1_8m.yaml), [`keydoor_capacity_m8_8m.yaml`](keydoor_capacity_m8_8m.yaml), [`keydoor_capacity_8m.yaml`](keydoor_capacity_8m.yaml) | `outputs/keydoor-capacity-{m1,m8,}-8m-*` | Key-to-Door, M = 1, 8, 16 | `raw_summary` |

Environment contracts of the paper: `environments/8m/dark_key_to_door.yaml` (with `_m1`, `_m8`, `_m16` for the summary-length arms), `count_recall.yaml`, `mazerunner.yaml` and `tmaze_v3.yaml`.

## Other files

`summary_memory.yaml`, `xland_one_rule_8m.yaml`, `match_pattern_8m.yaml`, `manifests/`, `environments/8m/{concentration,xland_minigrid,xland_one_rule,match_pattern}.yaml` and the contracts directly under `environments/` belong to an earlier study and to the retained relational-attention code. The paper does not use them; they are kept because the shared code and its tests resolve them.
