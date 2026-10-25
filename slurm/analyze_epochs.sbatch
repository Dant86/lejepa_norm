#!/bin/bash
# Epoch sweep of the redundancy as ONE job array: every array task is one
# (run, epoch) unit (apps/analyze.py --epoch-unit), independent and resumable (a
# finished unit's JSON shard is reused, so resubmitting recomputes only the
# missing or failed tasks). Aggregate afterwards, on the login node or in a short
# job, with `apps/analyze.py --epoch-sweep RUN... --note NOTE` (reads the shards).
#
# Usage (from the repo root; the array range must be 0..(n_runs * n_epochs - 1),
# index = run_idx * n_epochs + epoch_idx; an out-of-range task fails at once):
#   RUNS="a_s0 a_s1" EPOCHS="10 20 30" \
#     sbatch --array=0-5%20 slurm/analyze_epochs.sbatch CONFIG [--set KEY=VALUE ...]
# `%N` caps the concurrently running tasks (24 running jobs per user are allowed).
# Use the same --set overrides (eval.num_permutations, ...) in the aggregation:
# shards computed with other settings are ignored.
#
# Each task's log goes to <LEJEPA_NORM_RUNS_DIR>/<run>/slurm-analyze-epoch-<jobid>.log.
# When submitting from a git worktree use `sbatch --export=ALL` with
# PYTHONPATH=$PWD/src (the interpreter's venv belongs to the main checkout).
#
#SBATCH --job-name=lejepa-dynamics
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

if [[ $# -lt 1 ]]; then
  echo "usage: RUNS=\"...\" EPOCHS=\"...\" sbatch --array=0-N" \
    "slurm/analyze_epochs.sbatch CONFIG [--set KEY=VALUE ...]" >&2
  exit 2
fi
CONFIG="$1"
shift 1

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
: "${SLURM_ARRAY_TASK_ID:?run this script as a job array (sbatch --array=0-N)}"
: "${RUNS:?set RUNS, e.g. RUNS=\"a_s0 a_s1\"}"
: "${EPOCHS:?set EPOCHS, e.g. EPOCHS=\"10 20 30\"}"

PARAMS="$(sweep_params "${SLURM_ARRAY_TASK_ID}" "${RUNS}" "${EPOCHS}")" || exit 2
read -r RUN_NAME EPOCH <<<"${PARAMS}"
log_to_run_dir "${RUN_NAME}" analyze-epoch
echo "task ${SLURM_ARRAY_TASK_ID}: run=${RUN_NAME} epoch=${EPOCH}"

# One BLAS thread per allocated CPU (an oversubscribed pool is much slower).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

set -x
run_preemptable "${PYTHON}" apps/analyze.py --config "${CONFIG}" \
  --epoch-unit "${RUN_NAME}" "${EPOCH}" "$@"
