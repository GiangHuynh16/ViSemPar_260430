#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======================================================"
echo "  BASELINE B1 – Direct AMR (all seeds)"
echo "======================================================"

for seed in 42 123 456; do
    bash run_scripts/run_one.sh B1 $seed
done

python src/aggregate_scores.py
