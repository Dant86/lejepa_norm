#!/bin/bash
# One run of the radius view-signal study (apps/analyze.py --view-signal PREFIX
# --run-name RUN_NAME), CPU only; STL-10-C is read with all views (about 4 GB of z),
# so the job requests 64 GB. Preemption-safe: the unit file is written atomically at
# the end, a requeued job recomputes the run.
#
# Usage (from the repo root; one job per run):
#   for run in radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}; do
#     sbatch --export=ALL slurm/view_signal.sbatch configs/base.yaml radius_marginal \
#       "${run}"
#   done
# Then aggregate (cheap, any node; the bootstrap of the severity part takes about
# 15 min per lambda):
#   python apps/analyze.py --config CONFIG --view-signal PREFIX \
#     --note radius_view_signal/view_signal.md
# Use --export=ALL and submit from the checkout (a worktree works: PYTHONPATH is set
# to <checkout>/src below). The job log goes to
# <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-view_signal-<jobid>.log.
#
#SBATCH --job-name=lejepa-view-signal
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

if [[ $# -lt 3 ]]; then
  echo "usage: sbatch slurm/view_signal.sbatch CONFIG PREFIX RUN_NAME [args ...]" >&2
  exit 2
fi
CONFIG="$1"
PREFIX="$2"
RUN_NAME="$3"
shift 3

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir "${RUN_NAME}" view_signal
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

set -x
run_preemptable "${PYTHON}" apps/analyze.py --config "${CONFIG}" \
  --view-signal "${PREFIX}" --view-signal-run "${RUN_NAME}" "$@"
