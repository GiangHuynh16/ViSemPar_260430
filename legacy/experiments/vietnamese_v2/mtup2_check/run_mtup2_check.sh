#!/usr/bin/env bash
# Run the Vietnamese v2 MTUP2_CHECK experiment.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods MTUP2_CHECK \
  "$@"
