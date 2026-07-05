#!/usr/bin/env bash
# Compatibility wrapper for the current Vietnamese v2 protocol.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 run_scripts/run_vi_epoch_smatch_grid.py "$@"
