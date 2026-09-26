#!/bin/bash
#SBATCH --job-name=sm-pack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/summary-memory-8m/logs/slurm-%x-%j.log
# One "pack": SLOTS concurrent summary-memory fits on EVERY GPU of the
# allocation, driven by the pool launcher over the lines of a job file.
# Submit from the repository root (SLURM_SUBMIT_DIR):
#   sbatch [--gres=gpu:3090:4 --cpus-per-task=24 --mem=36G] \
#     scripts/slurm/summary_memory_pack.slurm JOBFILE [SLOTS]
# or let scripts/slurm/submit_summary_memory.sh size and submit the packs.
# The runs and logs land under the study root (OUTPUT_ROOT, default
# outputs/summary-memory-8m). Generate the job file with
# scripts/generate_summary_memory_jobs.py so only pending fits are listed.
# The #SBATCH lines above are one-GPU defaults; the submit script overrides
# them per pack, and the partition name is site-specific. Sizing (RTX 3090): a
# fit uses about 1.8 GB RAM, 1.4 GB GPU memory and one CPU thread of compute;
# the recurrent carriers are launch-bound, so three fits per GPU give about
# 2.45x the aggregate of one. OMP_NUM_THREADS is cpus-per-task / (GPUs x SLOTS).
# Slurm opens --output before this script runs, so the log directory must
# exist: the submit script creates it.
# WANDB=1 (inherited from the submitting shell) turns on Weights & Biases,
# online unless WANDB_MODE says otherwise.
# RETRIES=N (default 0) survives transient "No space left on device" failures
# on a shared filesystem. When the queue exits with failures and either a fit
# wrote ENOSPC during that attempt or the study root has at most ENOSPC_GB
# (default 2) GB free when it fails, the pack waits until FREE_GB (default 20)
# GB are free, removes its own lines from logs/queue-started.txt and runs the
# queue again, so the died fits resume from their latest training state while
# finished fits are skipped (checkpoint.pt); up to N times. Any other failure
# still stops the pack at once. The retried resumes set
# RESUME_ALLOW_EVICTED_REPLAY=1, because the FIFO kept collecting after the
# state was saved; each such resume is recorded in the run's resumes.jsonl.
set -Eeuo pipefail
jobfile="${1:?usage: sbatch summary_memory_pack.slurm JOBFILE [SLOTS]}"
slots="${2:-3}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"   # uv
export WANDB_MODE="${WANDB_MODE:-online}"
study_root="${OUTPUT_ROOT:-$PWD/outputs/summary-memory-8m}"
mkdir -p "$study_root/logs"
IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "node $(hostname) job $SLURM_JOB_ID gpus ${CUDA_VISIBLE_DEVICES:-?} ($(nvidia-smi -L | head -1)) cpus ${SLURM_CPUS_PER_TASK:-?} slots/gpu $slots wandb ${WANDB:-0} bash $BASH_VERSION"
threads=$(( ${SLURM_CPUS_PER_TASK:-8} / ( ${#gpus[@]} * slots ) )); (( threads < 1 )) && threads=1
run_queue() {
  SLOTS="$slots" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
    OUTPUT_ROOT="$study_root" \
    bash scripts/run_summary_memory_queue.sh "$jobfile"
}
pack_lines() {  # the job lines of this pack, comments and blanks removed
  sed 's/#.*//' "$jobfile" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF'
}
declare -A log_size  # bytes of each fit log before the attempt: only output appended by this attempt is inspected
mark_logs() {
  local benchmark condition seed log
  while read -r benchmark condition seed; do
    log="$study_root/logs/$benchmark-$condition-seed-$seed.log"
    log_size["$log"]=$(stat -c %s "$log" 2>/dev/null || echo 0)
  done < <(pack_lines)
}
pack_lines_idle() {  # this pack's lines whose run shows no training write for 10 min: safe to re-take
  # When several packs draw from one job file, a line this pack once started can
  # already be running again in another pack (its resubmission or a retry there); the shared ledger carries
  # no job id, so a live copy is recognised by its run writing metrics every epoch. A line whose run wrote
  # within the last 10 minutes is left in the ledger and is not resumed here.
  local benchmark condition seed directory metrics now
  now=$(date +%s)
  while read -r benchmark condition seed; do
    directory="$(run_directory "$benchmark" "$condition" "$seed" "$study_root" 2>/dev/null || echo "")"
    [[ -n "$directory" ]] || directory="$(ls -d "$study_root"/*/"$condition"/seed-"$seed" 2>/dev/null | head -n1)"
    metrics="$directory/training_metrics.jsonl"
    if [[ -f "$metrics" ]] && (( now - $(stat -c %Y "$metrics") < 600 )); then
      echo "$(date +%FT%T%z) not re-taking $benchmark $condition $seed: its run wrote metrics within 10 min (alive elsewhere)" >&2
      continue
    fi
    echo "$benchmark $condition $seed"
  done < <(pack_lines)
}
died_of_enospc() {  # a fit of this pack wrote ENOSPC during this attempt
  local benchmark condition seed log
  while read -r benchmark condition seed; do
    log="$study_root/logs/$benchmark-$condition-seed-$seed.log"
    [[ -f "$log" ]] || continue
    tail -c +$(( ${log_size["$log"]:-0} + 1 )) "$log" | grep "No space left on device" > /dev/null && return 0
  done < <(pack_lines)
  # A full filesystem cannot record that it is full: at
  # 23:21 every fit of all four MazeRunner packs died with its log truncated
  # mid-traceback, so no message was ever written and this pack exited instead
  # of waiting. A study root with almost nothing free at the moment of failure
  # is therefore read as ENOSPC in its own right.
  local avail
  avail="$(df -BG --output=avail "$study_root" | tail -1 | tr -dc '0-9')"
  if (( ${avail:-99} <= ${ENOSPC_GB:-2} )); then
    echo "$(date +%FT%T%z) no fit logged ENOSPC, but $study_root has ${avail}G free: treating the failure as a full filesystem"
    return 0
  fi
  return 1
}
retries="${RETRIES:-0}"; attempt=0
while :; do
  mark_logs
  run_queue && exit 0
  code=$?
  (( attempt < retries )) || exit "$code"
  died_of_enospc || { echo "$(date +%FT%T%z) queue failed without ENOSPC; not retried"; exit "$code"; }
  attempt=$(( attempt + 1 ))
  while (( $(df -BG --output=avail "$study_root" | tail -1 | tr -dc '0-9') < ${FREE_GB:-20} )); do
    echo "$(date +%FT%T%z) waiting for ${FREE_GB:-20} GB free on $(df --output=source "$study_root" | tail -1)"; sleep 120
  done
  ledger="$study_root/logs/queue-started.txt"
  grep -vxF -f <(pack_lines_idle) "$ledger" > "$ledger.tmp" || true; mv "$ledger.tmp" "$ledger"
  export RESUME_ALLOW_EVICTED_REPLAY=1
  echo "$(date +%FT%T%z) retry $attempt/$retries after ENOSPC: resuming this pack's died fits"
done
