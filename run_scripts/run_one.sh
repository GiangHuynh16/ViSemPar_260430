#!/bin/bash
# Run train + inference + evaluate for ONE method × ONE seed.
# Usage: bash run_scripts/run_one.sh B1 42
#
# Called by individual run_METHOD.sh scripts.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
METHOD=$1
SEED=$2

if [ -z "$METHOD" ] || [ -z "$SEED" ]; then
    echo "Usage: $0 <METHOD> <SEED>"
    exit 1
fi

CONFIG="$ROOT/configs/${METHOD}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

echo "======================================================"
echo "  METHOD=$METHOD  SEED=$SEED"
echo "======================================================"

cd "$ROOT"

echo "[1/3] Training..."
python src/train.py --config "$CONFIG" --seed "$SEED"

echo "[2/3] Inference..."
python src/inference.py --config "$CONFIG" --seed "$SEED" --split test

echo "[3/3] Evaluate..."
# Determine group and result path from config
GROUP=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
base = yaml.safe_load(open('configs/' + cfg.get('base','base.yaml'))) if 'base' in cfg else {}
base.update(cfg)
print(base['group'])
")
PRED="$ROOT/results/${GROUP}/${METHOD}/s${SEED}_pred.txt"
GOLD="$ROOT/data/raw/public_test_ground_truth.txt"

python src/evaluate.py \
    --pred  "$PRED" \
    --gold  "$GOLD" \
    --method "$METHOD" \
    --seed   "$SEED" \
    --group  "$GROUP"

echo "======================================================"
echo "  DONE: $METHOD seed=$SEED"
echo "======================================================"
