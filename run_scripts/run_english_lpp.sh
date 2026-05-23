#!/bin/bash
# Run train + inference + evaluation on Little Prince English AMR.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

METHODS=${METHODS:-"B1 B2 MTUP2 MTUP3A MTUP3B MTUP4"}
SEEDS=${SEEDS:-"42"}
GOLD="$ROOT/data/english/little_prince/raw/test_ground_truth.txt"

if [ ! -f "$GOLD" ]; then
    echo "ERROR: missing Little Prince data. Run: bash run_scripts/download_little_prince.sh"
    exit 1
fi

for method in $METHODS; do
    CONFIG="$ROOT/configs/english_lpp/${method}.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: config not found: $CONFIG"
        exit 1
    fi

    GROUP=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
base = yaml.safe_load(open('$ROOT/configs/base.yaml'))
base.update(cfg)
print(base['group'])
")

    for seed in $SEEDS; do
        echo "======================================================"
        echo "  ENGLISH LPP METHOD=$method SEED=$seed"
        echo "======================================================"

        echo "[1/3] Training..."
        python3 src/train.py --config "$CONFIG" --seed "$seed"

        echo "[2/3] Inference..."
        python3 src/inference.py --config "$CONFIG" --seed "$seed" --split test

        echo "[3/3] Evaluate..."
        python3 src/evaluate.py \
            --pred  "$ROOT/results/$GROUP/$method/s${seed}_pred.txt" \
            --gold  "$GOLD" \
            --method "$method" \
            --seed   "$seed" \
            --group  "$GROUP"
    done
done

python3 src/aggregate_scores.py
