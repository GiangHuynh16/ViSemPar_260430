#!/bin/bash
# Run inference + evaluation for B2 using existing checkpoints.
# This intentionally does NOT train.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="$ROOT/configs/B2.yaml"
GROUP="baselines"
GOLD="$ROOT/data/raw/public_test_ground_truth.txt"

echo "======================================================"
echo "  BASELINE B2 – inference/eval only (all seeds)"
echo "======================================================"

for seed in 42 123 456; do
    ADAPTER="$ROOT/checkpoints/$GROUP/B2/s${seed}/final"
    if [ ! -d "$ADAPTER" ]; then
        echo "ERROR: missing checkpoint: $ADAPTER"
        exit 1
    fi

    echo "======================================================"
    echo "  B2 seed=$seed"
    echo "======================================================"

    echo "[1/2] Inference..."
    python src/inference.py --config "$CONFIG" --seed "$seed" --split test

    echo "[2/2] Evaluate..."
    python src/evaluate.py \
        --pred  "$ROOT/results/$GROUP/B2/s${seed}_pred.txt" \
        --gold  "$GOLD" \
        --method B2 \
        --seed   "$seed" \
        --group  "$GROUP"
done

python src/aggregate_scores.py
