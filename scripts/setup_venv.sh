#!/usr/bin/env bash
# Install the two Python packages of this repo into .venv:
#   triton      the compiler, built from third_party/triton (libtriton.so)
#   triton_llm  this repo's python/ directory
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Creates .py-triton, installs `triton` editable, and points .venv at .py-triton.
"$ROOT/scripts/build_libtriton.sh"
"$ROOT/.venv/bin/pip" install --no-deps -e "$ROOT"
"$ROOT/.venv/bin/python" -c "import os, torch, triton, triton_llm; print(torch.__version__, triton.__version__, os.path.realpath(triton._C.libtriton.__file__), triton_llm.__file__, torch.cuda.is_available())"
