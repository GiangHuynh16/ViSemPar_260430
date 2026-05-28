#!/bin/bash
# Run train + inference + evaluation for Vietnamese v2 (clean config).
#
# Changes vs v1:
#   - epochs: 10 (vs 25), no data duplication (amr2_weight=1)
#   - dev split: 10% held out for eval + load_best_model_at_end
#   - no_var_mode: clean (Penman format, no v13 triple format)
#   - save_total_limit: 1 (saves ~20GB disk per run)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

METHODS=${METHODS:-"B1 B2 MTUP2 MTUP3A MTUP3B MTUP4"}
SEEDS=${SEEDS:-"42 123 456"}
GOLD="$ROOT/data/raw/public_test_ground_truth.txt"

if [ ! -f "$GOLD" ]; then
    echo "ERROR: missing Vietnamese test ground truth: $GOLD"
    exit 1
fi

# Generate formatted training data with 10% dev split
if [ ! -f "$ROOT/data/formatted/vietnamese_v2/MTUP2_train.txt" ]; then
    echo "[0/3] Generating formatted training data (v2: amr2_weight=1, dev_split=0.1)..."
    python3 src/data_pipeline.py \
        --language vi \
        --no_var_mode clean \
        --raw_dir data/raw \
        --formatted_dir data/formatted/vietnamese_v2 \
        --amr1_weight 1 \
        --amr2_weight 1 \
        --dev_split 0.1
fi

for method in $METHODS; do
    CONFIG="$ROOT/configs/vietnamese_v2/${method}.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: config not found: $CONFIG"
        exit 1
    fi

    GROUP=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
base = yaml.safe_load(open('$ROOT/configs/base_v2.yaml'))
base.update(cfg)
print(base['group'])
")

    for seed in $SEEDS; do
        echo "======================================================"
        echo "  VIETNAMESE V2  METHOD=$method SEED=$seed"
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
