#!/bin/bash
# Variant A lambda x seed sweep as ONE job array: every array task is its own
# independent, preemptable training run (a requeued task keeps its task ID and
# resumes in its own run directory, exactly like slurm/train.sbatch).
#
# Usage (from the repo root; pass the array range on the command line, it must be
# 0..(n_lambdas * n_seeds - 1); an out-of-range task fails immediately):
#   LAMBDAS="0.02 0.05 0.1" SEEDS="0 1 2" \
#     sbatch --array=0-8 slurm/sweep.sbatch CONFIG RUN_PREFIX [--set KEY=VALUE ...]
# Example:
#   LAMBDAS="0.02 0.05 0.1" SEEDS="0 1 2" \
#     sbatch --array=0-8 slurm/sweep.sbatch configs/base.yaml radius_marginal
# runs radius_marginal_lam0.02_s0 ... radius_marginal_lam0.1_s2. Add `%N` to the
# range (e.g. --array=0-8%3) to cap how many tasks run at once; the DSI guide
# suggests %5-%10 for GPU-heavy arrays. To use fewer CPUs: --cpus-per-task=4.
#
# Each task's log goes to <LEJEPA_NORM_RUNS_DIR>/<run name>/slurm-train-<jobid>.log;
# resubmitting the same command for a failed task resumes it (finished runs exit
# at once). See slurm/README.md.
#
#SBATCH --job-name=lejepa-sweep
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=general
# Pin the GPU model so run times are comparable (see slurm/train.sbatch).
#SBATCH --gres=gpu:a100:1
# 12:00:00 is the maximum wall time of the general QoS.
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

if [[ $# -lt 2 ]]; then
  echo "usage: LAMBDAS=\"...\" SEEDS=\"...\" sbatch --array=0-N slurm/sweep.sbatch" \
    "CONFIG RUN_PREFIX [--set KEY=VALUE ...]" >&2
  exit 2
fi
CONFIG="$1"
RUN_PREFIX="$2"
shift 2

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
: "${SLURM_ARRAY_TASK_ID:?run this script as a job array (sbatch --array=0-N)}"
: "${LAMBDAS:?set LAMBDAS, e.g. LAMBDAS=\"0.02 0.05 0.1\"}"
: "${SEEDS:?set SEEDS, e.g. SEEDS=\"0 1 2\"}"

PARAMS="$(sweep_params "${SLURM_ARRAY_TASK_ID}" "${LAMBDAS}" "${SEEDS}")" || exit 2
read -r LAMBDA SEED <<<"${PARAMS}"
RUN_NAME="${RUN_PREFIX}_lam${LAMBDA}_s${SEED}"
log_to_run_dir "${RUN_NAME}" train
echo "sweep task ${SLURM_ARRAY_TASK_ID}: lambda=${LAMBDA} seed=${SEED} run=${RUN_NAME}"

set -x
run_preemptable "${PYTHON}" apps/train.py --config "${CONFIG}" \
  --run-name "${RUN_NAME}" --env-file "${ENV_FILE}" --resume auto \
  --set "data.num_workers=${SLURM_CPUS_PER_TASK:-8}" \
  --set "loss.lambda=${LAMBDA}" --set "seed=${SEED}" "$@"
