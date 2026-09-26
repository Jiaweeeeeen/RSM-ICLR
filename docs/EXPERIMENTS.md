# Experiments

The benchmarks, training budget, evaluation protocol, statistics and diagnostics behind the paper. The compared memories are defined in [METHOD](METHOD.md); the commands are in [REPRODUCE](REPRODUCE.md); the saved results are in [`results/`](../results/README.md).

## Benchmarks

| Benchmark | Task | Training task | Primary endpoint | Margin δ |
|---|---|---|---|---|
| Dark Key-to-Door | AMAGO v3.4.0 `RoomKeyDoor`: hidden start, key and door cells fixed for the task; the public observation is position, key possession and the attempt clock; an attempt ends at the door or after 50 physical actions | 500 charged calls | Completed doors in the 500-call task | 1 door |
| CountRecallMedium | POPGym 1.0.7 through AMAGO's wrapper: 104 cards from four categories; each step answers how many cards of the queried category have been dealt, the current card included; reward ±1/103 | One 104-card stream | Exact accuracy over the 103 scored queries | 0.05 |
| MazeRunner 15×15 | AMAGO `MazeRunner` with randomized actions: hidden walls and a hidden per-map action permutation; the observation is position, four wall distances, the timer and the coordinates of three ordered goals | 500 steps | Goal fraction (goals reached of three) | 0.1 |
| Passive T-Maze | AMAGO v3.4.0 `TMazeAltPassive`: the rewarded turn is shown once at the first step; the corridor observation is constant and the junction observable; exactly L + 1 decisions, reward 1 for the correct turn, a −1/128 penalty on non-forward moves | Corridor L drawn uniformly from {32, 64, …, 256} per task | Greedy success at L = 128 | 0.25 |

The environment contracts are `configs/environments/8m/{dark_key_to_door,count_recall,mazerunner,tmaze_v3}.yaml`. On the T-Maze a deterministic policy sees only two distinct input streams at a fixed corridor length (one per cue side), so the per-seed reading is primary there.

## Training and evaluation protocol

**Budget.** Every cell and seed is trained for 8,000,000 charged environment calls (1,000 epochs × 16 actors × 500 calls), reset-only calls included, with seeds 42, 100 and 2026. Batch size, update schedule, learning rate, reward scale, replay and exploration are identical across cells within a benchmark. All fits run in FP32.

**Tasks.** Each task is generated from an integer identity. Training draws identities from 0–999,999; development tasks are 1,000,000–1,000,063; every reported held-out number uses the 256 **confirmation** tasks 5,000,000–5,000,255. The ranges are disjoint as generating identities, which gives fresh draws from the task distribution; two identities can still produce the same configuration (for example the same key and door cells), and no content-level overlap check is made, so generalization is to fresh draws rather than to configurations guaranteed unseen.

**Checkpoints.** Development performance is scored every 50 epochs. Every reported number uses the final (8M) checkpoint, evaluated greedily with frozen weights; a deterministic task gets one rollout.

## Long-running evaluations

Frozen policies keep acting in one task for 4,000 charged calls (eight times the Key-to-Door and MazeRunner training horizon) with their memory carried throughout. Successes are counted per 500-call window, so a longer task gives no advantage from extra opportunity alone. Every long-running panel's first native-length prefix reproduces the training-horizon evaluation exactly.

- **Same task.** Key-to-Door keeps its layout while attempts continue. On MazeRunner one map is replayed lap after lap (a lap ends at the third goal or the 500-step timer), with a reset-only call between laps.
- **New task inside the run.** A new Key-to-Door layout at the first attempt boundary after every 500 calls; a new MazeRunner maze and goal sequence (same action permutation) at the first lap boundary after every 1,000 calls, drawn from identities 6,000,000–6,999,999. Estimands: successes per layout or per 500 calls on each map, and calls to the first door (censored at the layout's end) or to the first completed lap.
- **Memory wiped at every attempt.** The same MazeRunner laps with the memory reset at each lap boundary separate what a memory carries between attempts.

## Diagnostics

**Attention and read bias** (`scripts/attention_read.py`). An unbiased capture records each decision row's attention mass on the summary (READ slots), on earlier buffer records and on its own record, and must reproduce the saved retained panel task for task. The read-bias sweep adds −β (β ∈ {1, 2, 4, 8, ∞}) to the decision rows' scores on the summary keys (companion: on earlier buffer keys) and reads the paired change in the primary metric.

**Representation and transplant** (`scripts/representation_read.py`, `reasoned_icrl/analysis/representation.py`). The capture keeps, per task, the summary each segment reads (the 4 × 256 memory at the segment's first decision, flattened to 1,024 features).

- *Projection.* Per seed, principal components are fitted on all development summaries after subtracting their mean; the confirmation summaries of one segment (segment 8 on Key-to-Door, segment 2 on CountRecall) are centred by that mean and projected.
- *Probe.* Per seed, ridge regression from the flattened summary of that segment to the hidden variables (Key-to-Door: key and door row and column; CountRecall: the count of each category dealt before the segment). Features and targets are centred by the training means and not otherwise scaled; the penalty is chosen from 17 values log-spaced over [10⁻³, 10⁵] by 5-fold cross-validation on the development tasks (squared error, fold permutation seed 0); the model refitted on all development tasks is scored on the confirmation tasks as R² per target, one row per task.
- *Similarity.* Per cell and seed, summaries of every complete task at segments b ≥ 1 are centred by their pooled mean over tasks and segments; the cosine similarity between segments b and b′ is averaged over tasks, then seeds, beside the mean centred norm. Centring removes the common trend, so a centred anticorrelation is not a sign reversal of the uncentred summaries, and the centred norm is not the uncentred row norm of the paper's bound.
- *Transplant.* At the boundary opening one segment, each task's summary is replaced by a donor task's summary (or by the initial memory, once); the paired change in behaviour toward the donor's hidden variables is read against the recipient's own.

## Statistics

A cell's estimate averages rollouts within a task, tasks within a seed, then seeds. A contrast is the paired per-task, per-seed difference between two cells on the same roster. Its 95 % interval is a percentile bootstrap with 2,000 replicates (resampling seed 0): each replicate draws the three seed indices with replacement and one task-index vector with replacement shared by every drawn seed, and averages the paired differences over the drawn cells (`reasoned_icrl/analysis/statistics.py`). Every contrast also reports the three per-seed differences and a task bootstrap within each seed. Intervals are conditional on the three trained seeds; 256 tasks do not remove training-seed uncertainty.

A contrast is *consistent* when its estimate is at least δ, its joint interval excludes zero and every seed agrees in sign; *comparable* when the observed mean difference lies within ±δ (a descriptive closeness, not an equivalence test); *inconclusive* otherwise. The interval is tested against zero, not against δ. The contrast family per benchmark is RSM − w/o memory (primary), full history − RSM, RSM − Memo, RSM − Memo fixed, RSM − GRU and Memo − Memo fixed, with the same family for RSM-R and RSM − RSM-R; RSM − window on Key-to-Door and MazeRunner; RSM − RSM detach and RSM detach − w/o memory on CountRecall.
