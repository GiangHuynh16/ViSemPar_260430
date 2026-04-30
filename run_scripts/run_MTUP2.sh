#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======================================================"
echo "  MTUP2 – 2-task (all seeds)"
echo "======================================================"

for seed in 42 123 456; do
    bash run_scripts/run_one.sh MTUP2 $seed
done

python src/aggregate_scores.py
