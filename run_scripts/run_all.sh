#!/bin/bash
# Run ALL methods sequentially.
# Follows plan execution order: B1 → B2 → MTUP2 → MTUP3A → MTUP3B → MTUP4
#
# Usage: bash run_scripts/run_all.sh
# To run only baselines: bash run_scripts/run_all.sh baselines
# To run only mtup:      bash run_scripts/run_all.sh mtup

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE=${1:-all}

echo "======================================================"
echo "  ViSemPar_260430 – Full Experimental Run"
echo "  Mode: $MODE"
echo "======================================================"

# Step 0: Build all formatted datasets (only needed once)
echo ""
echo "[Step 0] Building formatted datasets..."
python src/data_pipeline.py

if [ "$MODE" = "baselines" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[Step 1] Running Baseline B1..."
    bash run_scripts/run_B1.sh

    echo ""
    echo "[Step 2] Running Baseline B2..."
    bash run_scripts/run_B2.sh
fi

if [ "$MODE" = "mtup" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "[Step 3] Running MTUP2..."
    bash run_scripts/run_MTUP2.sh

    echo ""
    echo "[Step 4] Running MTUP3A..."
    bash run_scripts/run_MTUP3A.sh

    echo ""
    echo "[Step 5] Running MTUP3B..."
    bash run_scripts/run_MTUP3B.sh

    echo ""
    echo "[Step 6] Running MTUP4..."
    bash run_scripts/run_MTUP4.sh
fi

echo ""
echo "[Final] Aggregating all scores..."
python src/aggregate_scores.py

echo ""
echo "======================================================"
echo "  ALL DONE. Results in results/summary/all_scores.csv"
echo "======================================================"
