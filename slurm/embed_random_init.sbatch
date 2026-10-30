#!/bin/bash
# E2 random-init baseline: one untrained network per array task (BatchNorm mode x
# seed), embedded with apps/embed.py --random-init on the STL-10 train and test
# sets (K augmented views + center), single GPU, preemption-safe.
#
# Usage (from the repo root; 6 tasks = 2 BN modes x 3 seeds):
#   sbatch --export=ALL --array=0-5 slurm/embed_random_init.sbatch configs/base.yaml
# Optional second/third argument: space-separated BN modes and seeds
# (default "calibrated identity" and "0 1 2"). Task INDEX = mode_idx * n_seeds +
# seed_idx (as `sweep_params` in common.sh). Run names: random_init_bncal_s<S>
# (calibrated) and random_init_bnid_s<S> (identity). The untrained checkpoint is
# written once to <LEJEPA_NORM_RUNS_DIR>/<run>/last.pt and reused after a requeue.
# Use --export=ALL and submit from the checkout (a worktree works: PYTHONPATH is
# set to <checkout>/src below).
#
#SBATCH --job-name=lejepa-embed-rand
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=general
# Any GPU type (the embedding of 13k images is cheap); the type is logged by the job.
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch --array=0-5 slurm/embed_random_init.sbatch CONFIG [MODES] [SEEDS]" >&2
  exit 2
fi
CONFIG="$1"
MODES="${2:-calibrated identity}"
SEEDS="${3:-0 1 2}"

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
read -r MODE SEED < <(sweep_params "${SLURM_ARRAY_TASK_ID:?submit with --array}" \
  "${MODES}" "${SEEDS}")
case "${MODE}" in
  calibrated) TAG=bncal ;;
  identity) TAG=bnid ;;
  *) echo "error: unknown BN mode ${MODE}" >&2; exit 2 ;;
esac
RUN_NAME="random_init_${TAG}_s${SEED}"
log_to_run_dir "${RUN_NAME}" embed
nvidia-smi -L || true

set -x
run_preemptable "${PYTHON}" apps/embed.py --config "${CONFIG}" \
  --run-name "${RUN_NAME}" --random-init "${SEED}" --bn-mode "${MODE}" \
  --env-file "${ENV_FILE}" --sets stl10_train,stl10_test \
  --resume --set "data.num_workers=${SLURM_CPUS_PER_TASK:-8}"
