#!/usr/bin/env bash
# FileCheck tests. Each tests/tt/*.mlir carries a // RUN: line.
# TT_OPT_CPP is build/bin/triton-opt (same LLVM as libtriton.so).
# The word tt-opt in a RUN line is the Python driver, not that binary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
FILECHECK="${FILECHECK:-/opt/torch-mlir/externals/llvm-project/build/bin/FileCheck}"
if [[ ! -x "$FILECHECK" ]]; then
  echo "FileCheck not found at $FILECHECK" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
TT_OPT_BIN="$ROOT/build/bin/triton-opt"
if [[ ! -x "$TT_OPT_BIN" ]]; then
  echo "missing $TT_OPT_BIN (scripts/build_libtriton.sh)" >&2
  exit 1
fi
shopt -s nullglob
files=("$ROOT"/tests/tt/*.mlir)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no tests/tt/*.mlir" >&2
  exit 1
fi
for src in "${files[@]}"; do
  run=$(grep -m1 '^// RUN:' "$src" || true)
  if [[ -z "$run" ]]; then
    echo "FAIL: $src has no // RUN: line" >&2
    exit 1
  fi
  cmd=${run#// RUN: }
  cmd=${cmd//tt-opt/$PYTHON -m triton_llm.tt_opt}
  cmd=${cmd//TT_OPT_CPP/$TT_OPT_BIN}
  cmd=${cmd//%s/$src}
  cmd=${cmd//FileCheck/$FILECHECK}
  echo "== $(basename "$src") =="
  eval "$cmd"
  echo "PASS: $(basename "$src")"
done
