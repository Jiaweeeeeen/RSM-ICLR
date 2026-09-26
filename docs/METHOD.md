# Method

## Recurrent Summary Memory

The agent is AMAGO's off-policy actor–critic with a Transformer policy that processes its history in segments of C = 32 admitted records. Each segment is one causal attention problem over M + C + M slots in the fixed order `[READ | RECORD | WRITE]`:

- **READ** (M = 4 slots) carries the summary written at the previous boundary; the first segment of a task reads a learned, task-independent initial memory.
- **RECORD** (C = 32 slots) holds the segment's timestep records and produces the actor and critic inputs. This is the **working buffer**: exact, recent, cleared at every boundary.
- **WRITE** (M = 4 slots) is driven by learned write queries that attend to the carried summary and the whole segment. At the boundary the final-normalized WRITE outputs pass through a learned projection P and **replace** the carried summary, Z ← P(W). This is the **long-term summary**: fixed size, rewritten at every boundary.

READ and WRITE rows produce no actions, rewards or losses, and a segment's write reaches the policy only through the next segment's reads. Positions are segment-local. Boundaries count admitted records (never attempts, rewards or hidden task data) and are lazy: after the 32nd record the segment stays open until the next record arrives, so a segment never crosses an outer-task reset.

**Training.** Complete outer tasks are replayed from raw public trajectories with gradients through every carried write (no truncation). After each learner update every actor's state is rebuilt exactly from its true prefix with the current weights. Training memory therefore grows with task length, while **deployment state** does not: per actor it is the summary, one M + C + M-slot key/value cache per layer and two counters (249,868 B at C = 32, M = 4, d = 256, three layers).

**Inputs.** Every compared memory receives the same five-key public packet (`current`, `previous`, `outcome`, `event`, `valid`); the stateless timestep encoder reads the current observation and AMAGO's RL2 previous action and reward. Hidden layouts, task identities, evaluator labels and `info` never reach the policy. Timestep history is kept across attempts within an outer task; every actor's memory is reset at an outer-task reset.

## Compared memories

Every cell shares the timestep encoder, width 256, three layers, eight heads (the GRU matches width and depth), actor and critic heads, optimizer, reward scaling, replay and exploration schedule within an environment.

| Paper name | Condition | Memory |
|---|---|---|
| **RSM** | `raw_summary` | As above: working buffer plus a four-token summary replaced at every boundary |
| RSM-R (ablation) | `raw_summary_residual` | The write is added to the summary the segment read, Z ← Z + P(W) |
| RSM detach (ablation, CountRecall) | `raw_summary_detach` | The carried summary is detached at every boundary in training (truncated gradients); forward values unchanged |
| RSM, M ∈ {1, 8, 16} (ablation, Key-to-Door) | `raw_summary` | The same carrier with M summary tokens |
| w/o memory | `raw_segment` | The write is discarded: every segment reads the initial memory |
| Full-history Transformer | `full_context` | Causal attention over the entire task prefix |
| Memo | `memo` | Accumulating summaries (Gupta et al., 2025): four tokens appended at every boundary and never rewritten, segments of L = 32 with ±20 % training jitter, full gradients |
| Memo, fixed segments | `memo_fixed` | Memo with training-segment jitter 0 |
| GRU | `full_gru` | AMAGO's GRU; one recurrent state carried through the task |
| Sliding window | `raw_window` | A per-layer band of W = 40 cached records (Key-to-Door and MazeRunner) |

`raw_summary`, `raw_summary_residual` and `raw_segment` share the entire model and boundary schedule and differ only in the boundary rule; each rule and each gradient rule is a separately hashed identity, so no checkpoint or rollout state loads into another cell.

**Memo.** The accumulated-summary Transformer is ported from the author-linked `Memory-icrl/memo` source (commit `9e7044f`) onto the shared backbone: each boundary appends four summary tokens, positions are the slot index of the concatenated block, and training segments are drawn from [26, 38] per learner forward (32 at rollout). Its live state grows by four tokens per boundary (6,156 B to 497,676 B per actor over a 501-record task, larger than RSM's from the ninth boundary).

**Sliding window.** W is the largest window whose per-actor state (W cache slots per layer, source times, length) does not exceed RSM's: W = 40 gives 246,084 B (W = 41 would exceed 249,868 B). The window keeps absolute trajectory positions, so beyond the training horizon it meets positions unseen in training, whereas RSM resets positions in every segment; a window–RSM gap there mixes retained information with positional extrapolation. Its contextualized deeper-layer caches give a structural receptive field of up to 1 + 3 × (40 − 1) = 118 records, so older records are not directly accessible but are not strictly cut off at 40.

## Relation to recurrent memory Transformers

RSM belongs to the recurrent-memory-Transformer family: fixed memory tokens read at the start of a segment and written at its end, as in RMT (Bulatov et al., 2022), RATE's offline agent, and the non-accumulating RMT variant that Memo evaluates in online RL, including on Dark Key-to-Door. The mechanism is not claimed as new. The recipe choices of this carrier are a learned initial memory, dedicated write queries over the carried summary and the whole segment, a learned projection of the normalized writes, segment-local positions, lazy boundaries, complete-task gradients with raw-trajectory replay and exact post-update rebuild, and M = 4. The residual and detach ablations test the rewrite and gradient choices; the other choices are not isolated by controlled comparisons.

## Interventions

All interventions use the same frozen checkpoint and roster as the retained evaluation. *Summary-cleared* replaces the carried summary by the initial memory at every boundary and keeps the working buffer. *Current-token* removes both. *Attempt-cleared* resets the memory at every attempt (lap) boundary; *goal-cleared* at every MazeRunner goal. *Summary unreadable* blocks the decision rows' attention to the summary keys (the write still reads them). The *transplant* replaces every task's summary at one boundary with a donor task's summary, or with the initial memory once.
