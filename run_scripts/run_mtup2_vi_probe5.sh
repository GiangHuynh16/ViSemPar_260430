#!/bin/bash
# One MTUP2 Vietnamese probe: one mid-range LR, five epochs, seed 42.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-42}"
CONFIG="configs/vietnamese_v2/MTUP2_probe5.yaml"

echo "== Audit MTUP2 setup =="
python3 src/audit_b2.py \
  --b2 data/formatted/vietnamese_v2/B2_train.txt \
  --mtup2 data/formatted/vietnamese_v2/MTUP2_train.txt \
  --b2_dev data/formatted/vietnamese_v2/B2_dev.txt \
  --mtup2_dev data/formatted/vietnamese_v2/MTUP2_dev.txt
python3 src/audit_mtup2_setup.py --config "$CONFIG"

echo "== Train one MTUP2 probe: LR=1.752410111812816e-5, epochs=5 =="
python3 src/train.py --config "$CONFIG" --seed "$SEED"

echo "Probe training complete."
echo "Inspect:"
echo "  checkpoints/vietnamese_v2_mtup_2tasks_probe/MTUP2_lr_mid_probe5/s${SEED}/train_meta.json"
echo "  checkpoints/vietnamese_v2_mtup_2tasks_probe/MTUP2_lr_mid_probe5/s${SEED}/loss_curve.png"
