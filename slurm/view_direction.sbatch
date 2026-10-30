#!/bin/bash
# One run of the view-direction study (apps/analyze.py --view-direction
# PREFIX --view-direction-run RUN_NAME), CPU only. Reads the re-embedded
# `augreg_<RUN_NAME>` stl10_test dump and the clean stl10_train bank of RUN_NAME; the
# bootstrap and the boosted trees take a few minutes. Preemption-safe: the unit file
# is written at the end, a requeued job recomputes the run.
#
# Usage (from the repo root; one job per run):
#   for run in radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}; do
#     sbatch --export=ALL slurm/view_direction.sbatch configs/base.yaml \
#       radius_marginal "${run}"
#   done
# Then write the note (cheap, any node):
#   python apps/analyze.py --config CONFIG --view-direction PREFIX \
#     --note radius_view_direction/view_direction.md
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-view_direction-<jobid>.log.
#
#SBATCH --job-name=lejepa-view-direction
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

if [[ $# -lt 3 ]]; then
  echo "usage: sbatch slurm/view_direction.sbatch CONFIG PREFIX RUN_NAME [args ...]" >&2
  exit 2
fi
CONFIG="$1"
PREFIX="$2"
RUN_NAME="$3"
shift 3

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir "${RUN_NAME}" view_direction
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

set -x
run_preemptable "${PYTHON}" apps/analyze.py --config "${CONFIG}" \
  --view-direction "${PREFIX}" --view-direction-run "${RUN_NAME}" "$@"
