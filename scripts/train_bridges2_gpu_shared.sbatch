#!/bin/bash
# Bridges-2: GPU-shared partition, single V100-32 GPU (adjust types/count per User Guide).
# Submit from the repository root (so SLURM_SUBMIT_DIR is the project):
#   sbatch scripts/train_bridges2_gpu_shared.sbatch
# Docs:    https://www.psc.edu/resources/bridges-2/user-guide/
# Replace MODULE_LINE and paths before first use.

#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 8:00:00
#SBATCH -J ddpm-cifar
##SBATCH -A YOUR_ALLOC_ID

set -euo pipefail

# Project root (clone repo here or use your Ocean path)
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${ROOT}"

# --- PyTorch from PSC AI module (pick a version from: module spider AI) ---
module load AI/pytorch_23.02-1.13.1-py3
source activate "${AI_ENV}"

export DDPM_RUNTIME=psc
# Optional: put CIFAR / checkpoints on Ocean
# export DDPM_DATA_DIR=/ocean/projects/XXXXX/USER/ddpm_data
# export DDPM_OUTPUT_DIR=/ocean/projects/XXXXX/USER/ddpm_experiments

pip install -q -r requirements-psc.txt

python train.py --config configs/ddpm_psc.yaml --runtime psc
