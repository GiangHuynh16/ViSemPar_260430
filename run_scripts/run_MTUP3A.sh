#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======================================================"
echo "  MTUP3A – 3-task concepts→no-var→AMR (all seeds)"
echo "======================================================"

for seed in 42 123 456; do
    bash run_scripts/run_one.sh MTUP3A $seed
done

python src/aggregate_scores.py
