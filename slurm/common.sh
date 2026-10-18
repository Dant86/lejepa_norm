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
  echo "restart #${SLURM_RESTART_COUNT:-0}"
}

# --- Preemption (adapted from the DSI batch-jobs guide's wrapper) ---------------
#
# The general QoS is preemptable. With `#SBATCH --signal=B:USR1@300` Slurm sends
# USR1 to this batch shell on preemption and 300 s before the wall-time limit,
# kills the job after the grace period and, with `#SBATCH --requeue`, requeues it
# under the same job ID when the job exits with code 99.
#
# run_preemptable CMD [ARGS ...]
#   Runs CMD in its own process group (setsid), waits for it and then exits the
#   job script (it never returns, so call it last):
#   * USR1: forwarded to CMD's process group; once CMD exits, this shell exits 99
#     (requeue). Apps with a USR1 handler (apps/train.py) checkpoint first; apps
#     without one die, which is fine because they are re-runnable and skip
#     finished work (embed --resume, prepare_data build, analyze).
#   * TERM: forwarded; this shell exits 143 without requeue.
#   * Otherwise the script exits with CMD's exit code (so a CMD that exits 99 by
#     itself also requeues; the defensive path of the guide).
_preempt_child_pid=""

_signal_child_group() {
  local signal="$1"
  if [[ -n "${_preempt_child_pid}" ]] && kill -0 "${_preempt_child_pid}" 2>/dev/null
  then
    echo "[$(date -u +%FT%TZ)] sending ${signal} to the process group"
    # With setsid the child PID is also its process-group ID; fall back to the PID.
    kill -s "${signal}" -- "-${_preempt_child_pid}" 2>/dev/null ||
      kill -s "${signal}" "${_preempt_child_pid}" 2>/dev/null ||
      true
  fi
}

_on_preempt() {
  echo "[$(date -u +%FT%TZ)] USR1: preemption or time-limit warning"
  _signal_child_group USR1
  # The grace period is the hard deadline; Slurm kills everything after it.
  if [[ -n "${_preempt_child_pid}" ]]; then
    wait "${_preempt_child_pid}" || true
  fi
  echo "[$(date -u +%FT%TZ)] exiting 99 to request a requeue"
  exit 99
}

_on_term() {
  echo "[$(date -u +%FT%TZ)] TERM: terminating without requeue"
  _signal_child_group TERM
  if [[ -n "${_preempt_child_pid}" ]]; then
    wait "${_preempt_child_pid}" || true
  fi
  exit 143
}

run_preemptable() {
  trap _on_preempt USR1
  trap _on_term TERM
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
  else
    # Not on the cluster (e.g. macOS for local tests): signal the PID only.
    "$@" &
  fi
  _preempt_child_pid=$!
  local return_code=0
  wait "${_preempt_child_pid}" || return_code=$?
  echo "[$(date -u +%FT%TZ)] $1 exited with code ${return_code}"
  if [[ "${return_code}" -eq 99 ]]; then
    echo "[$(date -u +%FT%TZ)] requeue requested via exit code 99"
    exit 99
  fi
  exit "${return_code}"
}
