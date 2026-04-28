#!/bin/bash
# ── Common environment setup for neuropaths SLURM jobs ───────────────────
#
# Source with absolute path (SLURM copies scripts to /var/spool):
#   source /springbrook/share/maths/maskbg/neuropaths/slurm/env.sh

# ── Project paths ─────────────────────────────────────────────────────
export PROJECT_DIR="/springbrook/share/maths/maskbg/neuropaths"
export VENV_DIR="${PROJECT_DIR}/.venv"

# ── Cache directories (redirect away from $HOME) ─────────────────────
CACHE_ROOT="${PROJECT_DIR}/.cache"

export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_PYTHON_INSTALL_DIR="${CACHE_ROOT}/uv-python"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"

export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/wandb"

# ── Temp directory ────────────────────────────────────────────────────
if [ -n "${SLURM_JOB_ID}" ]; then
    export TMPDIR="${CACHE_ROOT}/tmp/${SLURM_JOB_ID}"
else
    export TMPDIR="${CACHE_ROOT}/tmp/local_$$"
fi

# ── Create all directories ────────────────────────────────────────────
mkdir -p \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${PIP_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${MPLCONFIGDIR}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${WANDB_DIR}" \
    "${TMPDIR}" \
    2>/dev/null

# ── MLflow (shared tracking directory for all projects) ──────────────
export MLFLOW_TRACKING_URI="file:///springbrook/share/maths/maskbg/mlruns"

# ── Print summary ─────────────────────────────────────────────────────
echo "=== neuropaths environment ==="
echo "  PROJECT_DIR: ${PROJECT_DIR}"
echo "  CACHE_ROOT:  ${CACHE_ROOT}"
echo "  TMPDIR:      ${TMPDIR}"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "=========================="
