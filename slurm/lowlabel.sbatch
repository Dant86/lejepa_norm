#!/bin/bash
# One run of the low-label E1 study (apps/analyze.py --lowlabel PREFIX
# --lowlabel-run RUN_NAME), CPU only, preemption-safe: the shard is written
# atomically at the end, a requeued job recomputes the run.
#
# Usage (from the repo root; one job per run):
#   for run in radius_marginal_lam{0.02,0.05,0.1}_s{0,1,2}; do
#     sbatch --export=ALL slurm/lowlabel.sbatch configs/base.yaml radius_marginal \
#       "${run}" [extra apps/analyze.py args, e.g. --lowlabel-shards TAG]
#   done
# Then aggregate (cheap, any node), once per lambda and once for all:
#   python apps/analyze.py --config CONFIG --lowlabel PREFIX \
#     --lowlabel-lambda 0.05 --note radius_usage/lowlabel_lam0.05.md
# Use --export=ALL and submit from the checkout (a worktree works: PYTHONPATH is set
# to <checkout>/src below). The job log goes to
# <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-lowlabel-<jobid>.log.
#
# Threads: the fits are small (at most 5000 x 256), so few BLAS threads are
# faster than many (`LOWLABEL_THREADS`, default 2).
#
#SBATCH --job-name=lejepa-lowlabel
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

if [[ $# -lt 3 ]]; then
  echo "usage: sbatch slurm/lowlabel.sbatch CONFIG PREFIX RUN_NAME [args ...]" >&2
  exit 2
fi
CONFIG="$1"
PREFIX="$2"
RUN_NAME="$3"
shift 3

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir "${RUN_NAME}" lowlabel
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${LOWLABEL_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${LOWLABEL_THREADS:-2}"

set -x
run_preemptable "${PYTHON}" apps/analyze.py --config "${CONFIG}" \
  --lowlabel "${PREFIX}" --lowlabel-run "${RUN_NAME}" "$@"
