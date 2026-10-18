#!/bin/bash
# Writes all analyses of one run into an experiment note with apps/analyze.py
# (CPU only; PNG export needs the Chrome binary in LEJEPA_NORM_KALEIDO_CHROME).
#
# Usage (from the repo root):
#   sbatch slurm/analyze.sbatch CONFIG RUN_NAME NOTE [extra apps/analyze.py args]
# Example:
#   sbatch slurm/analyze.sbatch configs/base.yaml a_l0.05_s0 \
#       radius-conditional/variant_a.md --set eval.num_bootstrap=200
# (NOTE is relative to LEJEPA_NORM_NOTES_DIR. apps/analyze.py takes one KEY=VALUE
# per --set.)
#
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-analyze-<jobid>.log;
# the --output file below only catches failures before the run directory is known.
#
#SBATCH --job-name=lejepa-analyze
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# TODO(cluster docs): partition name (CPU); not yet looked up, do not guess.
#SBATCH --partition=TODO_PARTITION
# TODO(cluster docs): time limit allowed on the chosen partition.
#SBATCH --time=TODO_TIME_LIMIT

if [[ $# -lt 3 ]]; then
  echo "usage: sbatch slurm/analyze.sbatch CONFIG RUN_NAME NOTE [args ...]" >&2
  exit 2
fi
CONFIG="$1"
RUN_NAME="$2"
NOTE="$3"
shift 3

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir "${RUN_NAME}" analyze

set -x
"${PYTHON}" apps/analyze.py --config "${CONFIG}" --run-name "${RUN_NAME}" \
  --note "${NOTE}" "$@"
