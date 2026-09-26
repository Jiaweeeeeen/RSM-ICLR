# Recurrent Summary Memory for bounded-state in-context RL

Code, configurations and summary results for *Remember Less, Adapt Longer: Bounded-State In-Context Reinforcement Learning with Recurrent Summary Memory*.

Recurrent Summary Memory (RSM) is a Transformer policy for in-context reinforcement learning. It attends exactly over a short working buffer of the last C = 32 records and carries a fixed set of M = 4 summary tokens that is rewritten from the previous summary and the buffer whenever the buffer fills, after which the buffer is cleared. Deployment memory and per-decision cost are therefore independent of task length. The agent is trained with AMAGO's off-policy actor–critic on complete tasks, with gradients through every summary write.

## Contents

| Path | What it holds |
|---|---|
| [docs/METHOD.md](docs/METHOD.md) | The operator, training rule, deployment state, the compared memories and the interventions |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Benchmarks, budget, task splits, long-running evaluations, diagnostics and statistics |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Setup, the training-to-figures pipeline, which script builds each figure and table, running at scale, code map |
| [results/](results/README.md) | The saved records behind every figure and table of the paper |
| [configs/](configs/README.md) | Study rosters and environment contracts |
| [notebooks/](notebooks/README.md) | Figure and table notebooks over the saved records |
| `reasoned_icrl/`, `scripts/`, `tests/` | The package, entry points and readers, and the test suite |

## Compared memories

| Paper name | Condition |
|---|---|
| **RSM** (summary replaced at every boundary) | `raw_summary` |
| RSM-R (residual rewrite, ablation) | `raw_summary_residual` |
| RSM detach (truncated gradients, ablation) | `raw_summary_detach` |
| w/o memory (summary discarded at every boundary) | `raw_segment` |
| Full-history Transformer | `full_context` |
| Memo / Memo, fixed segments | `memo` / `memo_fixed` |
| GRU | `full_gru` |
| Sliding window, W = 40 | `raw_window` |

Benchmarks: Dark Key-to-Door, CountRecallMedium, MazeRunner 15×15 with randomized actions, and the passive T-Maze. Every policy is trained for 8M environment calls with seeds 42, 100 and 2026 and evaluated on 256 held-out confirmation tasks per benchmark, at the training horizon and in 4,000-step tasks.

## Quick start

```bash
make sync                   # pinned Python 3.12 environment (uv sync --frozen --extra xland)
make check                  # format, lint, strict typing, non-slow tests
uv run python scripts/train.py summary_memory --study configs/summary_memory_8m.yaml \
    --benchmark dark_key_to_door --condition raw_summary --seed 42 --smoke
```

The full pipeline, from training to the paper's figures, is in [docs/REPRODUCE.md](docs/REPRODUCE.md). The per-task evaluation panels (about 15 GB) and the trained weights are not included in this repository.
