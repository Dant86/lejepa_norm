#!/bin/bash
# Builds the prepared datasets and STL-10-C with `apps/prepare_data.py --stage
# build` (CPU only, offline). Run `--stage download` on the login node first; it
# needs network access, which compute nodes may not have.
#
# Usage (from the repo root):
#   sbatch slurm/prepare_data_build.sbatch [--set KEY=VALUE ...]
# Example:
#   sbatch slurm/prepare_data_build.sbatch --set data.full_corruption_set=true
#
# Preemption: the build has no checkpoint; it is killed and the requeued job
# skips every array that is already complete (each array is written atomically).
# See slurm/README.md.
#
# The job log goes to
# <LEJEPA_NORM_RUNS_DIR>/_prepare_data/slurm-prepare_data-<jobid>.log (appended
# across requeues); the --output file below only catches failures before that.
#
#SBATCH --job-name=lejepa-prepare-data
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

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
log_to_run_dir _prepare_data prepare_data

set -x
run_preemptable "${PYTHON}" apps/prepare_data.py --stage build \
  --env-file "${ENV_FILE}" --set "data.num_workers=${SLURM_CPUS_PER_TASK:-8}" "$@"
