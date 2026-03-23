#!/usr/bin/env bash
# Start an interactive GPU session on Bridges-2, then run this script's printed commands.
# Example (see Bridges-2 User Guide — GPU-shared / interact):
#   interact -p GPU-shared --gres=gpu:v100-32:1 -t 2:00:00
#
# Then on the compute node:
#   module load AI/pytorch_23.02-1.13.1-py3   # use: module spider AI
#   source activate "$AI_ENV"
#   cd /path/to/11685-Diffusion-Project
#   export DDPM_RUNTIME=psc
#   pip install -r requirements-psc.txt
#   python train.py --config configs/ddpm_psc.yaml --runtime psc

cat <<'EOF'
1) Request GPU (adjust walltime / GPU type per PSC User Guide):
   interact -p GPU-shared --gres=gpu:v100-32:1 -t 2:00:00

2) After the session starts:
   module spider AI
   module load AI/pytorch_<version>-py3
   source activate "$AI_ENV"
   cd "$(dirname "$0")/.."
   export DDPM_RUNTIME=psc
   pip install -r requirements-psc.txt
   python train.py --config configs/ddpm_psc.yaml --runtime psc
EOF
