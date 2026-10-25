#!/bin/bash
# One run of the frozen-probe study per array task (CPU only, a few minutes each),
# preemption-safe: a finished run's shard is skipped on requeue.
#
# Usage (from the repo root; the array size is the number of runs):
#   N=$(python apps/analyze.py --frozen-linear PREFIX --frozen-list-runs \
#         --config CONFIG | wc -l)
#   sbatch --export=ALL --array=0-$((N - 1)) slurm/frozen_linear.sbatch \
#       CONFIG PREFIX [extra apps/analyze.py args, e.g. --frozen-shards TAG]
# Then, on any node (cheap, CPU only), aggregate into the note:
#   python apps/analyze.py --frozen-linear PREFIX --config CONFIG \
#       --note radius_usage/frozen_linear_summary.md
# Use --export=ALL and submit from the checkout (a worktree works: PYTHONPATH is set
# to <checkout>/src below).
#
# Runs are listed by `--frozen-list-runs` in a fixed order (runs PREFIX_* with both
# the stl10_train and stl10_test dumps); every task picks its run by
# SLURM_ARRAY_TASK_ID. Logs go to <LEJEPA_NORM_RUNS_DIR>/frozen_linear/.
#
#SBATCH --job-name=lejepa-frozen-linear
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch --array=0-N slurm/frozen_linear.sbatch CONFIG PREFIX [args ...]" >&2
  exit 2
fi
CONFIG="$1"
PREFIX="$2"
shift 2

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir frozen_linear frozen_linear
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mapfile -t RUNS < <("${PYTHON}" apps/analyze.py --frozen-linear "${PREFIX}" \
  --frozen-list-runs --config "${CONFIG}" "$@")
INDEX="${SLURM_ARRAY_TASK_ID:?submit with --array}"
if (( INDEX >= ${#RUNS[@]} )); then
  echo "error: array index ${INDEX} but only ${#RUNS[@]} runs" >&2
  exit 2
fi

set -x
run_preemptable "${PYTHON}" apps/analyze.py --frozen-linear "${PREFIX}" \
  --config "${CONFIG}" --frozen-run "${RUNS[INDEX]}" "$@"
