#!/usr/bin/env bash
# Shard a job file into packs sized by GPU count and submit one pack job each.
#   bash scripts/slurm/submit_summary_memory.sh JOBFILE [GPUS_PER_PACK] [SLOTS] [extra sbatch args...]
# GPUS_PER_PACK is a comma-separated list, one entry per pack, e.g. "2,4": the
# first pack asks for two GPUs, the second for four, and the job lines are
# dealt to the packs in proportion to their GPUs (a 2,4 split sends a third of
# the lines to the first pack). Each pack runs SLOTS fits per GPU; the packs run
# in parallel as Slurm grants GPUs.
#   GRES_TYPE (default 3090) picks the GPU model, GRES_TYPE=any asks for any GPU;
#   TIME (default 2-00:00:00) the limit; WANDB=1 turns on Weights & Biases inside
#   every fit; DRY_RUN=1 prints the sbatch commands without submitting; JOB_NAME
#   (default sm-pack) names the job; STUDY_ROOT (default
#   outputs/summary-memory-8m) receives the shards, the Slurm logs and
#   the runs. Extra arguments after SLOTS are passed to sbatch (for example
#   --partition or --exclude for your cluster).
# Shards are written under $STUDY_ROOT/queue/packs/.
# e.g. WANDB=1 bash scripts/slurm/submit_summary_memory.sh my.jobs 2,4 3
# Generate the job file with scripts/generate_summary_memory_jobs.py for the
# roster you are running.
set -Eeuo pipefail
jobfile="${1:?usage: submit_summary_memory.sh JOBFILE [GPUS_PER_PACK] [SLOTS] [sbatch args]}"
gpuspec="${2:-1}"; slots="${3:-3}"; shift $(( $# >= 3 ? 3 : $# ))
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
study_root="${STUDY_ROOT:-outputs/summary-memory-8m}"
mkdir -p "$study_root/queue/packs" "$study_root/logs"
stamp="$(date +%Y%m%d-%H%M%S)"
IFS=',' read -r -a weights <<< "$gpuspec"
packs=${#weights[@]}
mapfile -t jobs < <(grep -v '^\s*#' "$jobfile" | grep -v '^\s*$')
(( ${#jobs[@]} > 0 )) || { echo "no jobs in $jobfile" >&2; exit 1; }
declare -a count shard
for (( p = 0; p < packs; p++ )); do
  count[p]=0
  shard[p]="$study_root/queue/packs/$(basename "${jobfile%.jobs}")-$stamp-pack$p.jobs"
  : > "${shard[p]}"
done
for line in "${jobs[@]}"; do  # weighted round robin: the pack with the smallest fill ratio
  best=0
  for (( p = 1; p < packs; p++ )); do
    if (( count[p] * weights[best] < count[best] * weights[p] )); then best=$p; fi
  done
  printf '%s\n' "$line" >> "${shard[best]}"; count[best]=$(( count[best] + 1 ))
done
for (( p = 0; p < packs; p++ )); do
  (( count[p] > 0 )) || { printf 'pack %d: empty, not submitted\n' "$p"; continue; }
  n=${weights[p]}
  printf 'pack %d (%s GPUs, %s fits at once): %s\n' "$p" "$n" "$(( n * slots ))" "$(tr '\n' ';' < "${shard[p]}")"
  gres="gpu:${GRES_TYPE:-3090}:$n"; [[ "${GRES_TYPE:-}" == any ]] && gres="gpu:$n"
  cmd=(sbatch --gres="$gres" --cpus-per-task=$(( n * slots * 2 )) \
       --mem="$(( n * slots * 3 ))G" --time="${TIME:-2-00:00:00}" \
       --job-name="${JOB_NAME:-sm-pack}" --output="$study_root/logs/slurm-%x-%j.log" \
       --export="ALL,OUTPUT_ROOT=$(cd -- "$study_root" && pwd)" "$@" \
       scripts/slurm/summary_memory_pack.slurm "${shard[p]}" "$slots")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '  %q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
done
