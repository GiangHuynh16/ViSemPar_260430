#!/bin/bash
# Retrain Vietnamese MTUP2 using current code (Penman format, not v13 triple format).
#
# Background: the original MTUP2 checkpoints were trained with old v13 code that
# used "Task 1: root: X / X -> :role -> Y" triple format. Current code uses
# "[STEP 1: AMR_NO_VAR] (leave-01 :ARG0 (boy))" Penman format.
# This script regenerates the training data and retrains all 3 seeds.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=${SEEDS:-"42 123 456"}
CONFIG="$ROOT/configs/MTUP2.yaml"
GOLD="$ROOT/data/raw/public_test_ground_truth.txt"

if [ ! -f "$GOLD" ]; then
    echo "ERROR: missing Vietnamese test ground truth: $GOLD"
    exit 1
fi

# Step 0: Regenerate MTUP2 training data with current code
echo "======================================================"
echo "  Regenerating MTUP2_train.txt (Penman format)"
echo "======================================================"
python3 src/data_pipeline.py \
    --language vi \
    --no_var_mode legacy \
    --methods MTUP2 \
    --amr1_weight 1 \
    --amr2_weight 3

echo "Done. Sample of regenerated data:"
head -20 data/formatted/MTUP2_train.txt
echo "..."

GROUP=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
base = yaml.safe_load(open('$ROOT/configs/base.yaml'))
base.update(cfg)
print(base['group'])
")

for seed in $SEEDS; do
    echo "======================================================"
    echo "  RETRAIN VI MTUP2 (current code)  SEED=$seed"
    echo "======================================================"

    echo "[1/3] Training..."
    python3 src/train.py --config "$CONFIG" --seed "$seed"

    echo "[2/3] Inference..."
    python3 src/inference.py --config "$CONFIG" --seed "$seed" --split test

    echo "[3/3] Evaluate..."
    python3 src/evaluate.py \
        --pred  "$ROOT/results/$GROUP/MTUP2/s${seed}_pred.txt" \
        --gold  "$GOLD" \
        --method MTUP2 \
        --seed   "$seed" \
        --group  "$GROUP"
done

python3 src/aggregate_scores.py

echo ""
echo "======================================================"
echo "  DONE. Check results/mtup_2tasks/MTUP2/scores.json"
echo "======================================================"
