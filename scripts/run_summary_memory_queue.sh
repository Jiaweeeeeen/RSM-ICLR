#!/usr/bin/env bash
# Pool launcher for the summary-memory matrix: keep SLOTS fits running on every
# GPU of the allocation, drawing `benchmark condition seed` lines from a job file.
#   [GPUS=0,1] [SLOTS=3] [WANDB=1] bash scripts/run_summary_memory_queue.sh JOBFILE
# - GPUS: comma-separated device indices; defaults to CUDA_VISIBLE_DEVICES (what
#   Slurm allocated) or "0". Each fit is started with CUDA_VISIBLE_DEVICES set to
#   one device, so `--device cuda` inside the fit always means "my GPU".
# - SLOTS: fits per GPU (measured on the RTX 3090: three concurrent recurrent-carrier fits give ~2.45x the aggregate throughput of one and keep the GPU at 94-99 %). The next fit starts the moment a slot frees on any GPU,
#   on the least-loaded GPU.
# - The job file is re-read every cycle, so it can be appended to while the
#   queue runs; `#` starts a comment. A job whose run directory already holds
#   checkpoint.pt is skipped, and every started line is recorded under
#   $root/logs/queue-started.txt so a restarted queue never doubles a fit. A
#   line whose run directory was started but not finished (a pack that died)
#   is resumed from its latest AMAGO training checkpoint (`--resume`), or
#   restarted with `--overwrite` when no training checkpoint was written yet;
#   remove such a line from queue-started.txt (or submit a fresh job file)
#   before resubmitting, since the ledger otherwise skips it as started.
#   A pack that died well after its last training state has usually let the
#   FIFO evict that state's oldest replay files, which an exact resume
#   refuses; RESUME_ALLOW_EVICTED_REPLAY=1 adds --resume-allow-evicted-replay
#   so the resume drops those files and records the deviation in the run's
#   resumes.jsonl and systems.json (R6). WANDB_ATTACH=1 makes a resumed fit
#   continue its original W&B run (the id in provenance.json, WANDB_RESUME=must)
#   instead of opening a new run.
# - Portability: the reap loop uses `wait -n` (bash >= 4.3) without `-p` and
#   never indexes an array with an empty key, so it runs on older node shells.
#   The header line records the bash.
# - WANDB=1 adds --wandb to every fit (run `wandb login` first, or export
#   WANDB_API_KEY; both are inherited by sbatch's default --export=ALL).
# - TREE names the checkout to run from (pin a worktree for a long queue); the
#   runs land under OUTPUT_ROOT (default:
#   outputs/summary-memory-8m)
#   regardless. STUDY names an alternative study YAML. UV_LOCKED=0 drops --locked.
# - DRY_RUN=1 prints the GPU assignment of every pending job and exits.
set -Eeuo pipefail
jobfile="${1:?usage: run_summary_memory_queue.sh JOBFILE}"
jobfile="$(cd -- "$(dirname -- "$jobfile")" && pwd)/$(basename -- "$jobfile")"
# The study root is resolved against this script's repository before any cd,
# so a queue run from a pinned worktree still lands its runs in one place.
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${OUTPUT_ROOT:-$here/outputs/summary-memory-8m}"
root="$(cd -- "${OUTPUT_ROOT:-$here/outputs/summary-memory-8m}" && pwd)"
cd -- "${TREE:-$here}"
export ACCELERATE_USE_CPU=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# XLand-MiniGrid (decision 15): the JAX simulator stays on the CPU, single-
# threaded, and reads the pinned benchmark from the study's data directory.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"
export XLAND_MINIGRID_DATA="${XLAND_MINIGRID_DATA:-$here/outputs/xland-data}"
unset VIRTUAL_ENV
slots="${SLOTS:-3}"
IFS=',' read -r -a gpus <<< "${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
(( ${#gpus[@]} > 0 )) || { echo "no GPU listed" >&2; exit 1; }
capacity=$(( slots * ${#gpus[@]} ))
mkdir -p "$root/logs"
study_args=()
if [[ -n "${STUDY:-}" ]]; then study_args=(--study "$STUDY"); fi
fit_args=()
if [[ "${WANDB:-0}" == 1 ]]; then fit_args+=(--wandb); fi
uv_run=(uv run)
if [[ "${UV_LOCKED:-1}" == 1 ]]; then uv_run+=(--locked); fi
started="$root/logs/queue-started.txt"; touch "$started"
dry="${DRY_RUN:-0}"

if [[ "$dry" != 1 ]]; then
  CUDA_VISIBLE_DEVICES="${gpus[0]}" "${uv_run[@]}" python -c 'import torch; assert torch.cuda.is_available(); print("PyTorch:", torch.cuda.get_device_name(0))'
fi
printf '%s queue %s from %s, %s slots on GPUs %s (%s fits at once), root %s, bash %s\n' \
  "$(date +%FT%T%z)" "$jobfile" "$PWD" "$slots" "${gpus[*]}" "$capacity" "$root" "$BASH_VERSION"

run_directory() {  # benchmark condition seed -> run directory (protocol resolved by the study)
  # stderr goes to the queue's own error log rather than nowhere: a resolver
  # that fails takes the pack down with it (set -e on the assignment) and used
  # to leave no trace at all.
  "${uv_run[@]}" python - "$@" "${study_args[@]}" <<'PY' 2>>"$root/logs/run_directory.err" | tail -1
import sys
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve
args = sys.argv[1:]
study_path = None
if "--study" in args:
    i = args.index("--study"); study_path = args[i + 1]; args = args[:i] + args[i + 2:]
benchmark, condition, seed, root = args
_, config = resolve(load_summary_memory_study(study_path), benchmark=benchmark, condition=condition, seed=int(seed), output_root=root)
print(config.run_directory)
PY
}

declare -A pid_job pid_gpu gpu_load
for gpu in "${gpus[@]}"; do gpu_load["$gpu"]=0; done
running=0
failures=()
next_job() {  # first job line not yet started
  local line
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(printf '%s' "$line" | xargs || true)"
    [[ -z "$line" ]] && continue
    if ! grep -qxF -- "$line" "$started"; then printf '%s' "$line"; return 0; fi
  done < "$jobfile"
  return 1
}
least_loaded_gpu() {  # the GPU with the fewest running fits (first wins ties)
  local best="" gpu
  for gpu in "${gpus[@]}"; do
    if [[ -z "$best" || ${gpu_load[$gpu]} -lt ${gpu_load[$best]} ]]; then best="$gpu"; fi
  done
  printf '%s' "$best"
}

if [[ "$dry" == 1 ]]; then  # print the assignment the live queue would make, then exit
  declare -A dry_load
  for gpu in "${gpus[@]}"; do dry_load["$gpu"]=0; done
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(printf '%s' "$line" | xargs || true)"
    [[ -z "$line" ]] && continue
    grep -qxF -- "$line" "$started" && { printf 'started  %s\n' "$line"; continue; }
    best=""
    for gpu in "${gpus[@]}"; do
      if [[ -z "$best" || ${dry_load[$gpu]} -lt ${dry_load[$best]} ]]; then best="$gpu"; fi
    done
    dry_load["$best"]=$(( dry_load[$best] + 1 ))
    printf 'gpu %-3s %s%s\n' "$best" "$line" "$( (( dry_load[$best] > slots )) && printf ' (waits for a slot)' )"
  done < "$jobfile"
  exit 0
fi

while :; do
  while (( running < capacity )) && job="$(next_job)"; do
    printf '%s\n' "$job" >> "$started"
    read -r benchmark condition seed <<< "$job"
    directory="$(run_directory "$benchmark" "$condition" "$seed" "$root")"
    if [[ -f "$directory/checkpoint.pt" ]]; then
      printf '%s skipping %s (checkpoint.pt exists at %s)\n' "$(date +%FT%T%z)" "$job" "$directory"; continue
    fi
    restart_args=()
    if [[ -f "$directory/config.yaml" ]]; then  # started by an earlier pack that did not finish
      if compgen -G "$directory/ckpts/training_states/*_epoch_*" > /dev/null; then
        restart_args=(--resume)
        if [[ "${RESUME_ALLOW_EVICTED_REPLAY:-0}" == 1 ]]; then restart_args+=(--resume-allow-evicted-replay); fi
        printf '%s resuming %s from its latest training checkpoint%s\n' "$(date +%FT%T%z)" "$job" "$( [[ "${RESUME_ALLOW_EVICTED_REPLAY:-0}" == 1 ]] && printf ' (evicted replay files allowed, recorded in resumes.jsonl)' )"
      else
        restart_args=(--overwrite)
        printf '%s restarting %s (started earlier, no training checkpoint yet)\n' "$(date +%FT%T%z)" "$job"
      fi
    fi
    gpu="$(least_loaded_gpu)"
    log="$root/logs/$benchmark-$condition-seed-$seed.log"
    wandb_env=()
    if [[ "${WANDB_ATTACH:-0}" == 1 && " ${restart_args[*]} " == *" --resume "* && -f "$directory/provenance.json" ]]; then
      # Continue the fit's original W&B run (one URL per fit) instead of opening a new one.
      old_id="$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("wandb") or {}).get("id") or "")' "$directory/provenance.json")"
      if [[ -n "$old_id" ]]; then wandb_env=(WANDB_RUN_ID="$old_id" WANDB_RESUME=must); printf '%s attaching %s to W&B run %s\n' "$(date +%FT%T%z)" "$job" "$old_id"; fi
    fi
    printf '%s starting %s on GPU %s -> %s\n' "$(date +%FT%T%z)" "$job" "$gpu" "$log"
    env "${wandb_env[@]}" CUDA_VISIBLE_DEVICES="$gpu" "${uv_run[@]}" python scripts/train.py summary_memory "${study_args[@]}" \
      --benchmark "$benchmark" --condition "$condition" --seed "$seed" --device cuda \
      --output-root "$root" "${fit_args[@]}" "${restart_args[@]}" >> "$log" 2>&1 &
    pid_job["$!"]="$job"; pid_gpu["$!"]="$gpu"
    gpu_load["$gpu"]=$(( gpu_load[$gpu] + 1 )); running=$((running + 1))
  done
  (( running == 0 )) && break
  # wait for any one fit to exit; then reap every finished pid. bash keeps the
  # exit status of a child already reaped by `wait -n`, so `wait "$pid"` below
  # returns it; `-p` (bash 5.1) is deliberately not used.
  wait -n ${pid_job[@]+"${!pid_job[@]}"} || true
  for pid in ${pid_job[@]+"${!pid_job[@]}"}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" && code=0 || code=$?
      job="${pid_job[$pid]:-unknown job}"
      if (( code == 0 )); then
        printf '%s finished %s\n' "$(date +%FT%T%z)" "$job"
      else
        printf '%s FAILED %s (exit %s; see its log)\n' "$(date +%FT%T%z)" "$job" "$code"
        failures+=("$job")
      fi
      gpu="${pid_gpu[$pid]:-}"
      if [[ -n "$gpu" ]]; then gpu_load["$gpu"]=$(( ${gpu_load[$gpu]:-1} - 1 )); fi
      unset "pid_job[$pid]" "pid_gpu[$pid]"; running=$((running - 1))
    fi
  done
done

if (( ${#failures[@]} > 0 )); then
  printf '\nFits with failures: %s\n' "${failures[*]}"; exit 1
fi
printf '\n%s queue drained.\n' "$(date +%FT%T%z)"
