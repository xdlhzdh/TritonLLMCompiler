#!/usr/bin/env bash
set -euo pipefail
command -v ncu >/dev/null || { echo "ncu not found — skip"; exit 0; }
echo "Soft profiling only; see docs/acceptance.md"
