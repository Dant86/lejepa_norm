#!/bin/bash
# Dumps the stl10_test embeddings of every epoch_XXXX.pt checkpoint of one run with
# apps/embed.py (single GPU, one job per run), preemption-safe.
#
# Usage (from the repo root):
#   sbatch slurm/embed_epochs.sbatch CONFIG RUN_NAME [extra apps/embed.py args]
# Example:
#   sbatch slurm/embed_epochs.sbatch configs/base.yaml a_l0.05_s0
#
# The dump of epoch E goes to
#   <LEJEPA_NORM_EMBED_DIR>/radius_dynamics/RUN_NAME/epoch_E/stl10_test
# (name from lejepa_norm.eval.dynamics.dump_name), leaving the dumps of the final
# checkpoint untouched. Runs with `--resume`: a preempted and requeued job skips
# the shards that exist. See slurm/README.md.
#
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-embed-<jobid>.log.
#
#SBATCH --job-name=lejepa-embed-epochs
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch slurm/embed_epochs.sbatch CONFIG RUN_NAME [args ...]" >&2
  exit 2
fi
CONFIG="$1"
RUN_NAME="$2"
shift 2

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir "${RUN_NAME}" embed

loop() {
  local ckpt epoch
  for ckpt in "${LEJEPA_NORM_RUNS_DIR}/${RUN_NAME}"/epoch_[0-9][0-9][0-9][0-9].pt; do
    epoch="$(basename "${ckpt}" .pt)"
    echo "[$(date -u +%FT%TZ)] embedding ${epoch}"
    "${PYTHON}" apps/embed.py --config "${CONFIG}" \
      --run-name "radius_dynamics/${RUN_NAME}/${epoch}" --checkpoint "${ckpt}" \
      --env-file "${ENV_FILE}" --sets stl10_test --resume \
      --set "data.num_workers=${SLURM_CPUS_PER_TASK:-4}" "$@" || return $?
  done
}

set -x
export -f loop
export CONFIG RUN_NAME PYTHON ENV_FILE LEJEPA_NORM_RUNS_DIR
run_preemptable bash -c 'loop "$@"' loop "$@"
