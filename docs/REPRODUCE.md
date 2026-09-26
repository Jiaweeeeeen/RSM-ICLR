# Reproducing the paper

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) with the pinned lockfile:

```bash
make sync                        # pinned environment; the XLand extra is needed by part of the test suite
make cuda-sync                   # Linux with CUDA
make check                       # format, lint, strict typing, non-slow tests
ACCELERATE_USE_CPU=true uv run --locked pytest -m "not cuda"   # adds the CPU lifecycle tests
```

## Studies and run directories

| Study | Benchmarks and cells |
|---|---|
| `configs/summary_memory_8m.yaml` | Key-to-Door and CountRecall: full history, GRU, RSM, RSM-R, w/o memory; the window (Key-to-Door); RSM detach (CountRecall) |
| `configs/memo_key_to_door_8m.yaml`, `configs/memo_count_recall_8m.yaml` | Memo and Memo fixed; their held-out passes re-score every cell of the main study on the same roster |
| `configs/tmaze_v3_8m.yaml` | Passive T-Maze: the seven cells without the window and detach |
| `configs/mazerunner_8m.yaml` | MazeRunner: eight cells including the window |
| `configs/keydoor_capacity_m1_8m.yaml`, `keydoor_capacity_m8_8m.yaml`, `keydoor_capacity_8m.yaml` | RSM with M = 1, 8, 16 on Key-to-Door |

A run lands in `<output_root>/<protocol>/<condition>/seed-<N>/` (`output_root` from the study file): `config.yaml`, `systems.json` (measured calls, state bytes, FLOPs, latency), `development.json`, `training_metrics.jsonl`, `ckpts/policy_weights/policy_epoch_999.pt` (the 8M checkpoint) and `eval/<split>-<history>-endpoint[-suffix]/` panels. Readers write CSV/JSON under `<output_root>/reports/`.

## Pipeline

```bash
STUDY=configs/summary_memory_8m.yaml; ENV=dark_key_to_door   # or count_recall, tmaze, mazerunner with their study

# 1. Train (8M charged calls per cell and seed; --smoke for a short lifecycle check)
uv run python scripts/train.py summary_memory --study $STUDY --benchmark $ENV --condition raw_summary --seed 42 --device cuda

# 2. Development series, then the held-out confirmation pass and tier report (retained and intervention panels)
uv run python scripts/develop_summary_memory.py --study $STUDY --benchmark $ENV \
    --finalize --final-split confirmation --panel-rules endpoint

# 3. Long-running panels on the frozen endpoints (each checked against the training-horizon prefix)
uv run python scripts/horizon_panels.py --study $STUDY --benchmark dark_key_to_door --split confirmation --horizons 4000
uv run python scripts/horizon_panels.py --study $STUDY --benchmark dark_key_to_door --split confirmation --horizons 4000 --layout-period 500
uv run python scripts/horizon_panels.py --study configs/mazerunner_8m.yaml --benchmark mazerunner --split confirmation \
    --laps 4000 [--history attempt-cleared | --history summary-cleared | --layout-period 1000]

# 4. Reads
uv run python scripts/horizon_read.py --study configs/memo_key_to_door_8m.yaml --horizon 4000 [--layout-period 500]
uv run python scripts/laps_read.py --study configs/mazerunner_8m.yaml --histories retained attempt-cleared summary-cleared
uv run python scripts/window_read.py --study configs/mazerunner_8m.yaml
uv run python scripts/capacity_read.py
uv run python scripts/attention_read.py {run,read} --study $STUDY --benchmark $ENV --split confirmation
uv run python scripts/representation_read.py {run,transplant,read} --study $STUDY --benchmark $ENV

# 5. Figures and tables from the saved records (REASONED_ICRL_OUTPUTS points at the study roots)
uv run python notebooks/run_figures.py
```

Every command accepts `--help`. `scripts/evaluate.py summary_memory` writes a single panel (`--history`, `--horizon`, `--layout-period`), and `scripts/play.py summary_memory` steps one task of a saved run with rendered observer frames.

## Where each figure and table comes from

| Paper display | Built by | Saved tables in `results/` |
|---|---|---|
| Results at a glance; results table | `notebooks/paper_figures.ipynb`, `notebooks/tables_three_benchmarks.ipynb` | `figure_data/paper_results_at_a_glance_*`, `figure_data/tables_three_benchmarks_*`, the tier reports, the Key-to-Door 4,000-call windows |
| Deployment state against task length | `paper_figures` from `mazerunner_repeated_laps` | `figure_data/paper_state_*` and each run's `systems.json` |
| Sliding-window table | `scripts/horizon_read.py`, `scripts/window_read.py` | `summary-memory-8m/reports/dark-key-to-door/confirmation/horizon/h4000/`, `mazerunner-8m/reports/.../window/` |
| Interventions; retention by rewrite age; room attention | `paper_figures` | `figure_data/paper_interventions_*`, `paper_retention_age_*`, the representation `grid.csv` |
| In-context adaptation; new tasks; memory kept or wiped | `paper_figures`, `scripts/laps_read.py` | `figure_data/paper_in_context_*`, `paper_new_tasks_*`, `mazerunner-8m/reports/mazerunner/confirmation/laps/` |
| Transplant; what the summary holds; attention | `scripts/representation_read.py`, `scripts/attention_read.py`, `paper_figures` | the `representation/` and `attention/` reports, `figure_data/paper_{transplant,representation_detail,attention}_*` |
| Ablations; summary-length table | `paper_figures`, `scripts/capacity_read.py` | `figure_data/paper_ablations_*`, `summary-memory-8m/reports/dark-key-to-door/confirmation/capacity/raw_summary/` |
| Learning curves | `notebooks/figure4_learning_matched_experience.ipynb`, `paper_figures` | `figure_data/paper_learning_*`, `figureA1_*` |

## Running at scale

`scripts/generate_summary_memory_jobs.py --study <config> --benchmark <env> --output <jobs>` lists the pending fits of a study. `scripts/slurm/submit_summary_memory.sh <jobs> [GPUS_PER_PACK] [SLOTS]` shards the job file into Slurm packs that run several fits per GPU through `scripts/run_summary_memory_queue.sh`, which also runs on a standalone GPU host (`GPUS=0 SLOTS=3 STUDY=<config> bash scripts/run_summary_memory_queue.sh <jobs>`). A fit uses about 1.8 GB RAM and 1.4 GB GPU memory; three fits share an RTX 3090 at about 2.45× the throughput of one. An interrupted fit resumes from its latest training state (`--resume`).

## Code map

| Package | Contents |
|---|---|
| `reasoned_icrl/environments/` | Task implementations, public packets, rosters and restorable state |
| `reasoned_icrl/model/` | Timestep encoder; summary (`summary_transformer.py`), Memo, window, full-history and GRU carriers; shared attention operators and caches (`dat_transformer.py`) |
| `reasoned_icrl/experiments/` | Condition identities (`contracts.py`), configuration resolution, rosters, records, evaluation summaries, the long-running adapters (`horizon.py`) and the study CLI (`summary_memory/`) |
| `reasoned_icrl/runtime/` | AMAGO integration: bindings, replay, checkpoints, training, rollout, attention probe and summary capture |
| `reasoned_icrl/analysis/` | Statistics, tier reports, attention and representation readings |

The package also retains environments and carriers of an earlier relational-attention study (dual-attention cells, ConcentrationEasy, XLand-MiniGrid, match-pattern), which the paper does not use; their configurations and tests are kept so the shared code stays tested.
