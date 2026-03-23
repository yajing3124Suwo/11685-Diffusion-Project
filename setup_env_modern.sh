#!/usr/bin/env bash
# Apple Silicon / any machine: Python 3.9+ venv (no conda, no 3.6).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OUT="${ROOT}/.venv-ddpm"

if [[ ! -x "${OUT}/bin/python" ]]; then
  python3 -m venv "${OUT}"
fi
# shellcheck disable=SC1090
source "${OUT}/bin/activate"
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "Done. Activate:"
echo "  source \"${OUT}/bin/activate\""
echo "Train:"
echo "  python train.py --config configs/ddpm.yaml"
