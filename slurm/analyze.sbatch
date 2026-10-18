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
# Preemption: analyze has no checkpoint; it is killed and the requeued job simply
# recomputes everything (re-running updates the note's generated sections in
# place). See slurm/README.md.
#
# The job log goes to <LEJEPA_NORM_RUNS_DIR>/RUN_NAME/slurm-analyze-<jobid>.log
# (appended across requeues); the --output file below only catches failures
# before the run directory is known.
#
#SBATCH --job-name=lejepa-analyze
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only jobs on GPU nodes may use at most (node cores - 2 * GPUs) CPUs; keep
# this modest so it fits under that cap on every node.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --qos=general
# 12:00:00 is the maximum wall time of the general QoS.
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

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
run_preemptable "${PYTHON}" apps/analyze.py --config "${CONFIG}" \
  --run-name "${RUN_NAME}" --note "${NOTE}" "$@"
