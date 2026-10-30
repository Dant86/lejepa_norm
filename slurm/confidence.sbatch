#!/bin/bash
# One run of the radius-vs-confidence study per array task (CPU only), preemption
# safe: a finished run's unit file is recomputed from scratch on requeue (minutes).
#
# Usage (from the repo root; the array size is the number of runs given):
#   sbatch --export=ALL --array=0-8 slurm/confidence.sbatch CONFIG SPACE RUN [RUN ...]
# SPACE is z (LeJEPA) or h (the E2 baselines only). Then, on any node, aggregate:
#   python apps/analyze.py --config CONFIG --confidence PREFIX \
#       --confidence-space SPACE --note radius_confidence/confidence.md
# Use --export=ALL and submit from the checkout (a worktree works: PYTHONPATH is set
# to <checkout>/src below). Logs go to <LEJEPA_NORM_RUNS_DIR>/confidence/.
#
#SBATCH --job-name=lejepa-confidence
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

if [[ $# -lt 3 ]]; then
  echo "usage: sbatch --array=0-N slurm/confidence.sbatch CONFIG SPACE RUN [RUN ...]" >&2
  exit 2
fi
CONFIG="$1"
SPACE="$2"
shift 2
RUNS=("$@")

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir confidence confidence
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

INDEX="${SLURM_ARRAY_TASK_ID:?submit with --array}"
if (( INDEX >= ${#RUNS[@]} )); then
  echo "error: array index ${INDEX} but only ${#RUNS[@]} runs" >&2
  exit 2
fi

set -x
run_preemptable "${PYTHON}" apps/analyze.py --config "${CONFIG}" \
  --run-name "${RUNS[INDEX]}" --confidence-unit --confidence-space "${SPACE}"
