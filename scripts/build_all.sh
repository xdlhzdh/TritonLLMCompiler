#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -f third_party/triton/include/triton/Dialect/Triton/IR/TritonOps.td ]]; then
  git submodule update --init third_party/triton
fi
cmake -B build -DTRITON_LLM_CUDA_ARCH=70 -DTRITON_LLM_ENABLE_SM90=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
if [[ -x "$ROOT/.venv/bin/pytest" ]]; then
  source "$ROOT/.venv/bin/activate"
  PYTHONPATH="$ROOT/python" pytest tests/python -q
else
  echo "NOTE: skip pytest (run scripts/setup_venv.sh)"
fi
echo "build_all: OK"
