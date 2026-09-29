#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"
[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple -r requirements.txt
python -m pip install --no-deps -e .
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
