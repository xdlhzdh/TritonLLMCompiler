#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="$(find "$ROOT/build" -name '*cutlass_sm90*' 2>/dev/null | head -1 || true)"
[[ -n "$LIB" ]] || { echo "Sm90 artifact missing"; exit 1; }
echo "PASS: $LIB"
