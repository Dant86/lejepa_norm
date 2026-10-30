#!/bin/bash
# One run of the E2 controls per array task (CPU only), preemption-safe: a finished
# run's shard is skipped on requeue.
#
# Usage (from the repo root; the array size is the number of runs):
#   N=$(python apps/analyze.py --baseline-controls radius_marginal random_init \
#         --bc-list-runs --config CONFIG | wc -l)
#   sbatch --export=ALL --array=0-$((N - 1)) slurm/baseline_controls.sbatch \
#       CONFIG "radius_marginal random_init" [extra apps/analyze.py args]
# Then, on any node (cheap, CPU only), aggregate into the notes:
#   python apps/analyze.py --config CONFIG \
#       --baseline-controls radius_marginal random_init \
#       --note radius_baselines/controls.md
# Use --export=ALL and submit from the checkout (a worktree works: PYTHONPATH is set
# to <checkout>/src below). Runs are listed by `--bc-list-runs` in a fixed order;
# every task picks its run by SLURM_ARRAY_TASK_ID. Logs go to
# <LEJEPA_NORM_RUNS_DIR>/baseline_controls/.
#
#SBATCH --job-name=lejepa-baseline-controls
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch --array=0-N slurm/baseline_controls.sbatch CONFIG \"PREFIX ...\" [args ...]" >&2
  exit 2
fi
CONFIG="$1"
read -r -a PREFIXES <<<"$2"
shift 2

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir baseline_controls baseline_controls
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mapfile -t RUNS < <("${PYTHON}" apps/analyze.py --baseline-controls "${PREFIXES[@]}" \
  --bc-list-runs --config "${CONFIG}" "$@")
INDEX="${SLURM_ARRAY_TASK_ID:?submit with --array}"
if (( INDEX >= ${#RUNS[@]} )); then
  echo "error: array index ${INDEX} but only ${#RUNS[@]} runs" >&2
  exit 2
fi

set -x
run_preemptable "${PYTHON}" apps/analyze.py --baseline-controls "${PREFIXES[@]}" \
  --config "${CONFIG}" --bc-run "${RUNS[INDEX]}" "$@"
