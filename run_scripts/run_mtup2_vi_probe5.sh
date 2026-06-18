#!/bin/bash
# Backward-compatible wrapper for the standard five-epoch MTUP2 probe.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash run_scripts/run_vi_probe5.sh MTUP2
