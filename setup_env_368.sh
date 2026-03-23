#!/usr/bin/env bash
# Create an environment with Python 3.6.8 (course VM alignment).
# On Apple Silicon: arm64 conda has NO python 3.6 — we retry with CONDA_SUBDIR=osx-64 (x86_64; needs Rosetta to run),
# or use setup_env_modern.sh for native arm64 Python 3.9+.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="${ROOT}/.venv-cifar"

install_deps() {
  local py="$1"
  "${py}" -m pip install -U pip setuptools wheel
  "${py}" -m pip install -r requirements-py36.txt
}

echo "=== Diffusion project: Python 3.6.8 env ==="

if command -v python3.6 >/dev/null 2>&1; then
  PY="$(command -v python3.6)"
  echo "Found: ${PY}"
  "${PY}" -m venv "${VENV}"
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
  install_deps python
  echo ""
  echo "Done. Activate with:"
  echo "  source ${VENV}/bin/activate"
  echo "Train:"
  echo "  python train.py --config configs/ddpm.yaml"
  exit 0
fi

if command -v pyenv >/dev/null 2>&1; then
  echo "Using pyenv to install 3.6.8 (may take several minutes)..."
  pyenv install -s 3.6.8
  PY="$(pyenv root)/versions/3.6.8/bin/python3.6"
  "${PY}" -m venv "${VENV}"
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
  install_deps python
  echo ""
  echo "Done. Activate with:"
  echo "  source ${VENV}/bin/activate"
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  echo "No python3.6/pyenv; using conda to create: ${VENV}"
  if [[ -x "${VENV}/bin/python" ]]; then
    echo "Reusing existing env at ${VENV}"
  else
    conda_ok=0
    if conda create -y -p "${VENV}" python=3.6.8 pip setuptools wheel; then
      conda_ok=1
    else
      echo ""
      echo "arm64 conda has no Python 3.6. Retrying with CONDA_SUBDIR=osx-64 (Intel/x86_64 packages)..."
      if CONDA_SUBDIR=osx-64 conda create -y -p "${VENV}" python=3.6.8 pip setuptools wheel; then
        conda_ok=1
        echo ""
        echo "Note: This Python is x86_64. On M1/M2/M3, run Terminal under Rosetta, or use native env:"
        echo "  bash setup_env_modern.sh"
      fi
    fi
    if [[ "${conda_ok}" -ne 1 ]]; then
      echo ""
      echo "Could not create 3.6.8 with conda. Use native Apple Silicon stack (recommended):"
      echo "  bash setup_env_modern.sh"
      exit 1
    fi
  fi
  install_deps "${VENV}/bin/python"
  echo ""
  echo "Done. Activate with:"
  echo "  conda activate \"${VENV}\""
  echo "Train:"
  echo "  python train.py --config configs/ddpm.yaml"
  exit 0
fi

echo "Could not find any of: python3.6, pyenv, conda"
echo "Easiest on Mac:  bash setup_env_modern.sh"
exit 1
