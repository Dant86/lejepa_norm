#!/bin/bash
# One unit of the SIGReg power study per array task (single GPU, a few seconds to
# minutes each), preemption-safe: a finished unit's shard is skipped on requeue.
#
# Usage (from the repo root; the array size is the number of units):
#   N=$(python apps/analyze.py --sigreg-power --power-list-units \
#         --config CONFIG --runs-prefix PREFIX | wc -l)
#   sbatch --export=ALL --array=0-$((N - 1)) slurm/sigreg_power.sbatch \
#       CONFIG PREFIX [extra apps/analyze.py args, e.g. --power-shards TAG]
# Then, on any node (cheap, CPU only), aggregate into the note:
#   python apps/analyze.py --sigreg-power --config CONFIG --runs-prefix PREFIX \
#       --note sigreg_power/power.md
# Use --export=ALL and submit from the checkout (a worktree works: PYTHONPATH is set
# to <checkout>/src below).
#
# Units are listed by `--power-list-units` in a fixed order (synthetic cells, then
# one per run PREFIX_* with an stl10_test dump); every task picks its unit by
# SLURM_ARRAY_TASK_ID. Logs go to <LEJEPA_NORM_RUNS_DIR>/sigreg_power/.
#
#SBATCH --job-name=lejepa-sigreg-power
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --qos=general
# The matmul / cos / sin of SIGReg is what the GPU is for; any GPU model will do
# (the results do not depend on it beyond float rounding).
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch --array=0-N slurm/sigreg_power.sbatch CONFIG PREFIX [args ...]" >&2
  exit 2
fi
CONFIG="$1"
PREFIX="$2"
shift 2

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir sigreg_power sigreg_power
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mapfile -t UNITS < <("${PYTHON}" apps/analyze.py --sigreg-power --power-list-units \
  --config "${CONFIG}" --runs-prefix "${PREFIX}" "$@")
INDEX="${SLURM_ARRAY_TASK_ID:?submit with --array}"
if (( INDEX >= ${#UNITS[@]} )); then
  echo "error: array index ${INDEX} but only ${#UNITS[@]} units" >&2
  exit 2
fi

set -x
run_preemptable "${PYTHON}" apps/analyze.py --sigreg-power --config "${CONFIG}" \
  --runs-prefix "${PREFIX}" --power-unit "${UNITS[INDEX]}" "$@"
