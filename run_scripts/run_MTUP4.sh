#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======================================================"
echo "  MTUP4 – 4-task concepts→relations→no-var→AMR (all seeds)"
echo "======================================================"

for seed in 42 123 456; do
    bash run_scripts/run_one.sh MTUP4 $seed
done

python src/aggregate_scores.py
