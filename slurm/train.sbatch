#!/bin/bash
# Pretrains one model with apps/train.py (single GPU).
#
# Usage (from the repo root):
#   sbatch slurm/train.sbatch CONFIG RUN_NAME [--set KEY=VALUE ...]
# Example:
#   sbatch slurm/train.sbatch configs/base.yaml a_l0.05_s0 \
#       --set loss.lambda=0.05 --set seed=0
#
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-train-<jobid>.log; the
# --output file below only catches failures before the run directory is known.
#
#SBATCH --job-name=lejepa-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# TODO(cluster docs): partition name; not yet looked up, do not guess.
#SBATCH --partition=TODO_PARTITION
# TODO(cluster docs): GPU request syntax and type (planned: 1x A100).
#SBATCH --gres=gpu:1
# TODO(cluster docs): time limit allowed on the chosen partition.
#SBATCH --time=TODO_TIME_LIMIT

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch slurm/train.sbatch CONFIG RUN_NAME [--set KEY=VALUE ...]" >&2
  exit 2
fi
CONFIG="$1"
RUN_NAME="$2"
shift 2

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir "${RUN_NAME}" train

set -x
"${PYTHON}" apps/train.py --config "${CONFIG}" --run-name "${RUN_NAME}" \
  --env-file "${ENV_FILE}" \
  --set "data.num_workers=${SLURM_CPUS_PER_TASK:-8}" "$@"
