#!/bin/bash
# Dumps multi-view embeddings of one checkpoint with apps/embed.py (single GPU),
# preemption-safe.
#
# Usage (from the repo root):
#   sbatch slurm/embed.sbatch CONFIG RUN_NAME CHECKPOINT [extra apps/embed.py args]
# Example:
#   sbatch slurm/embed.sbatch configs/base.yaml a_l0.05_s0 last.pt \
#       --sets stl10_train,stl10_test
# (A relative CHECKPOINT is resolved against <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/; an
# absolute path is used as is.)
#
# Runs with `--resume`: shards are written atomically, so a preempted (killed and
# requeued) job writes only the shards that are still missing. To replace an
# existing dump, delete it first (`--overwrite` cannot be combined with
# `--resume`). See slurm/README.md.
#
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-embed-<jobid>.log
# (appended across requeues); the --output file below only catches failures
# before the run directory is known.
#
#SBATCH --job-name=lejepa-embed
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=general
# TODO(cluster docs): GPU model selection syntax is not documented (nodes have
# A40/A100/L40S/H100/H200); add a type here only once confirmed.
#SBATCH --gres=gpu:1
# 12:00:00 is the maximum wall time of the general QoS.
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

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
run_preemptable "${PYTHON}" apps/embed.py --config "${CONFIG}" \
  --run-name "${RUN_NAME}" --checkpoint "${CHECKPOINT}" --env-file "${ENV_FILE}" \
  --resume --set "data.num_workers=${SLURM_CPUS_PER_TASK:-8}" "$@"
