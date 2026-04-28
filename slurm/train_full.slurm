#!/bin/bash
# SLURM job script: train the FNO on already-generated data.
#
# Submit:
#     sbatch slurm/train_full.slurm
#
# Override config:
#     CONFIG=configs/square_32.yaml sbatch slurm/train_full.slurm
#
# Chain after generation:
#     gen=$(sbatch --parsable slurm/generate_full.slurm)
#     sbatch --dependency=afterok:$gen slurm/train_full.slurm
#
# Output:
#   runs/<name>/<stem>_state_dict.pt  (clean state_dict)
#   runs/<name>/<stem>_metadata.pkl   (init kwargs; loaded via FNO.from_checkpoint)
#
# GPU directives match scrtp-docs ("GPU jobs on Blythe").

#SBATCH --job-name=neuropaths-train
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5960
#SBATCH --time=04:00:00

# ── Environment ───────────────────────────────────────────────────────
source /springbrook/share/maths/maskbg/neuropaths/slurm/env.sh
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

set -euo pipefail

CONFIG="${CONFIG:-configs/square_32.yaml}"

cd "${SLURM_SUBMIT_DIR}"

export PYTHONUNBUFFERED=1
# Pin BLAS even though training is GPU-bound; prevents PyTorch dataloader
# workers from accidentally fanning out via numpy ops.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

uv sync --frozen || uv sync

echo "[$(date -Is)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} gpu=$(nvidia-smi -L | head -1) config=${CONFIG}"

uv run neuropaths-train --config "${CONFIG}"

echo "[$(date -Is)] === training done ==="
