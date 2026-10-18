#!/bin/bash
# Dumps multi-view embeddings of one checkpoint with apps/embed.py (single GPU).
#
# Usage (from the repo root):
#   sbatch slurm/embed.sbatch CONFIG RUN_NAME CHECKPOINT [extra apps/embed.py args]
# Example:
#   sbatch slurm/embed.sbatch configs/base.yaml a_l0.05_s0 last.pt \
#       --sets stl10_train,stl10_test
# (A relative CHECKPOINT is resolved against <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/; an
# absolute path is used as is.)
#
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-embed-<jobid>.log; the
# --output file below only catches failures before the run directory is known.
#
#SBATCH --job-name=lejepa-embed
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

if [[ $# -lt 3 ]]; then
  echo "usage: sbatch slurm/embed.sbatch CONFIG RUN_NAME CHECKPOINT [args ...]" >&2
  exit 2
fi
CONFIG="$1"
RUN_NAME="$2"
CHECKPOINT="$3"
shift 3

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir "${RUN_NAME}" embed
if [[ "${CHECKPOINT}" != /* ]]; then
  CHECKPOINT="${LEJEPA_NORM_RUNS_DIR}/${RUN_NAME}/${CHECKPOINT}"
fi

set -x
"${PYTHON}" apps/embed.py --config "${CONFIG}" --run-name "${RUN_NAME}" \
  --checkpoint "${CHECKPOINT}" --env-file "${ENV_FILE}" \
  --set "data.num_workers=${SLURM_CPUS_PER_TASK:-8}" "$@"
