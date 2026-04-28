#!/bin/bash
# SLURM job script: end-to-end smoke test for the neuralop-backed pipeline.
#
# Generates a tiny dataset (20 train + 5 test PDE solutions on a 311x311
# grid, subsampled to 32x32) and trains the FNO for 5 epochs on a single
# L40. Whole job should finish in well under 10 minutes; the walltime
# below is generous.
#
# Submit:
#     sbatch slurm/smoke_test.slurm
#
# Override config:
#     CONFIG=configs/smoke_test.yaml sbatch slurm/smoke_test.slurm
#
# GPU directives match scrtp-docs ("GPU jobs on Blythe").

#SBATCH --job-name=neuropaths-smoke
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5960
#SBATCH --time=00:30:00

# ── Environment ───────────────────────────────────────────────────────
source /springbrook/share/maths/maskbg/neuropaths/slurm/env.sh
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

set -euo pipefail

CONFIG="${CONFIG:-configs/smoke_test.yaml}"

cd "${SLURM_SUBMIT_DIR}"

export PYTHONUNBUFFERED=1

# Match generate.slurm: pin BLAS to one thread per worker so the parallel
# data-gen step doesn't oversubscribe the allocated CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

uv sync --frozen || uv sync

echo "[$(date -Is)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} gpu=$(nvidia-smi -L | head -1) config=${CONFIG}"

echo "[$(date -Is)] === generate train split ==="
uv run neuropaths-generate \
    --config "${CONFIG}" \
    --split  train \
    --num-workers "${SLURM_CPUS_PER_TASK}"

echo "[$(date -Is)] === generate test split ==="
uv run neuropaths-generate \
    --config "${CONFIG}" \
    --split  test \
    --num-workers "${SLURM_CPUS_PER_TASK}"

echo "[$(date -Is)] === train FNO ==="
uv run neuropaths-train --config "${CONFIG}"

echo "[$(date -Is)] === smoke test done ==="
