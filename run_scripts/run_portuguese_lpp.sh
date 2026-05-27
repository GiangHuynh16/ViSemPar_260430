#!/bin/bash
# Run train + inference + evaluation on Little Prince Brazilian Portuguese AMR.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

METHODS=${METHODS:-"B1 B2 MTUP2 MTUP3A MTUP3B MTUP4"}
SEEDS=${SEEDS:-"42 123 456"}
GOLD="$ROOT/data/portuguese/little_prince/raw/test_ground_truth.txt"

if [ ! -f "$GOLD" ]; then
    echo "ERROR: missing Portuguese Little Prince data. Run: python3 src/prepare_portuguese.py"
    exit 1
fi

# Generate formatted training data if not present
if [ ! -f "$ROOT/data/formatted/portuguese_lpp/B1_train.txt" ]; then
    echo "[0/3] Generating formatted training data..."
    python3 src/data_pipeline.py \
        --language pt \
        --no_var_mode clean \
        --raw_dir data/portuguese/little_prince/raw \
        --formatted_dir data/formatted/portuguese_lpp \
        --amr1_weight 1 \
        --amr2_weight 1 \
        --no_dedup
fi

for method in $METHODS; do
    CONFIG="$ROOT/configs/portuguese_lpp/${method}.yaml"
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
        echo "  PORTUGUESE LPP METHOD=$method SEED=$seed"
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
