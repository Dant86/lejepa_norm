# Shared setup for the sbatch templates in this directory (sourced, not run).
#
# Expects to be sourced from the repo checkout (the templates `cd` to
# $SLURM_SUBMIT_DIR, so submit jobs from the repo root). Loads every path from the
# .env file (default `.env`; override with LEJEPA_NORM_ENV_FILE) and never uses
# the network.
#
# Variables honoured:
#   LEJEPA_NORM_ENV_FILE  .env file to load (default: .env in the repo root)
#   LEJEPA_NORM_PYTHON    Python interpreter (default: .venv/bin/python, created
#                         on the login node with `uv sync --locked`)

set -euo pipefail

ENV_FILE="${LEJEPA_NORM_ENV_FILE:-.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "error: ${ENV_FILE} not found; copy .env.example to .env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
: "${LEJEPA_NORM_RUNS_DIR:?LEJEPA_NORM_RUNS_DIR must be set in ${ENV_FILE}}"

PYTHON="${LEJEPA_NORM_PYTHON:-.venv/bin/python}"

# Compute nodes are assumed to have no network: fail fast instead of hanging on a
# download (timm builds ResNet-18 from scratch, so nothing should be fetched).
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# Sends all further output of the job into <runs_dir>/<run_name>/.
# Usage: log_to_run_dir RUN_NAME APP
log_to_run_dir() {
  local run_dir="${LEJEPA_NORM_RUNS_DIR}/$1"
  mkdir -p "${run_dir}"
  exec >>"${run_dir}/slurm-$2-${SLURM_JOB_ID:-local}.log" 2>&1
  echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
}
