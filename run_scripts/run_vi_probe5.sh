#!/bin/bash
# Five-epoch Vietnamese v2 probe runner.
#
# Usage:
#   bash run_scripts/run_vi_probe5.sh B1
#   bash run_scripts/run_vi_probe5.sh B2
#   bash run_scripts/run_vi_probe5.sh MTUP2
#   bash run_scripts/run_vi_probe5.sh MTUP2_LOW

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

METHOD="${1:-B1}"
SEED="${SEED:-42}"
GOLD="data/raw/public_test_ground_truth.txt"

case "$METHOD" in
  B1)
    CONFIG="configs/vietnamese_v2/B1_probe5.yaml"
    ;;
  B2)
    CONFIG="configs/vietnamese_v2/B2_probe5.yaml"
    ;;
  MTUP2)
    CONFIG="configs/vietnamese_v2/MTUP2_probe5.yaml"
    ;;
  MTUP2_LOW)
    CONFIG="configs/vietnamese_v2/MTUP2_probe5_lr_low.yaml"
    ;;
  *)
    echo "ERROR: unknown probe '$METHOD'. Use B1, B2, MTUP2, or MTUP2_LOW."
    exit 1
    ;;
esac

echo "== Regenerate clean B1 + B2 + MTUP2 Vietnamese v2 data =="
python3 src/data_pipeline.py \
  --raw_dir data/raw \
  --formatted_dir data/formatted/vietnamese_v2 \
  --language vi \
  --no_var_mode clean \
  --amr1_weight 1 \
  --amr2_weight 1 \
  --syn_weight 0 \
  --dev_split 0.1 \
  --dev_seed 42 \
  --methods B1 B2 MTUP2

echo "== Audit B2 against MTUP2 step 1 =="
python3 src/audit_b2.py \
  --b2 data/formatted/vietnamese_v2/B2_train.txt \
  --mtup2 data/formatted/vietnamese_v2/MTUP2_train.txt \
  --b2_dev data/formatted/vietnamese_v2/B2_dev.txt \
  --mtup2_dev data/formatted/vietnamese_v2/MTUP2_dev.txt

echo "== Audit MTUP2 setup =="
python3 src/audit_mtup2_setup.py --config configs/vietnamese_v2/MTUP2_probe5.yaml

echo "== Train probe: $METHOD =="
python3 src/train.py --config "$CONFIG" --seed "$SEED"

echo "== Inference probe: $METHOD =="
python3 src/inference.py --config "$CONFIG" --seed "$SEED" --split test

RESULT_INFO="$(python3 - "$CONFIG" "$SEED" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from config_utils import load_yaml_config

cfg = load_yaml_config(sys.argv[1])
seed = sys.argv[2]
run_name = cfg.get("run_name", cfg["method"])
pred = Path("results") / cfg["group"] / run_name / f"s{seed}_pred.txt"
print(cfg["method"], cfg["group"], run_name, pred)
PY
)"

EVAL_METHOD="$(echo "$RESULT_INFO" | awk '{print $1}')"
GROUP="$(echo "$RESULT_INFO" | awk '{print $2}')"
RUN_NAME="$(echo "$RESULT_INFO" | awk '{print $3}')"
PRED="$(echo "$RESULT_INFO" | awk '{print $4}')"

echo "== Evaluate probe: $METHOD =="
python3 src/evaluate.py \
  --pred "$PRED" \
  --gold "$GOLD" \
  --method "$EVAL_METHOD" \
  --seed "$SEED" \
  --group "$GROUP"

echo "Done."
echo "Train meta: checkpoints/$GROUP/$RUN_NAME/s${SEED}/train_meta.json"
echo "Scores:     results/$GROUP/$RUN_NAME/scores.json"
