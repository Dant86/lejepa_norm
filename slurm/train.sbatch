#!/bin/bash
# Pretrains one model with apps/train.py (single GPU), preemption-safe.
#
# Usage (from the repo root):
#   sbatch slurm/train.sbatch CONFIG RUN_NAME [--set KEY=VALUE ...]
# Example:
#   sbatch slurm/train.sbatch configs/base.yaml a_l0.05_s0 \
#       --set loss.lambda=0.05 --set seed=0
#
# Runs with `--resume auto`: on preemption (or 300 s before the wall-time limit)
# training saves <run_dir>/resume.pt and the job is requeued; the requeued job
# continues from it. Resubmitting the same command after a failure also resumes,
# and exits at once if the run already finished. See slurm/README.md.
#
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-train-<jobid>.log
# (appended across requeues); the --output file below only catches failures
# before the run directory is known.
#
#SBATCH --job-name=lejepa-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=general
# Pin the GPU model so run times are comparable: the general partition mixes
# A40/A100/L40S/H100/H200 nodes (`sinfo -o "%P %G"`); gres syntax is type:count.
#SBATCH --gres=gpu:a100:1
# 12:00:00 is the maximum wall time of the general QoS.
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

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
run_preemptable "${PYTHON}" apps/train.py --config "${CONFIG}" \
  --run-name "${RUN_NAME}" --env-file "${ENV_FILE}" --resume auto \
  --set "data.num_workers=${SLURM_CPUS_PER_TASK:-8}" "$@"
