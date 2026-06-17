#!/bin/bash
# Clean MTUP2-only Vietnamese v2 protocol.
#
# Default mode regenerates data, runs audits, and writes LR-search configs.
# Set RUN_TRAIN=1 on a CUDA machine to train the three LR candidates, select by
# dev loss, then run one public-test inference/evaluation for seed 42.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-42}"
CONFIG="configs/vietnamese_v2/MTUP2.yaml"
TUNE_DIR="configs/tuning/mtup2_vi_lr"
MANIFEST="$TUNE_DIR/MTUP2_lrsearch_manifest.json"
GOLD="data/raw/public_test_ground_truth.txt"

echo "== Regenerate clean B2 + MTUP2 Vietnamese v2 data =="
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
  --methods B2 MTUP2

echo "== Audit B2 against MTUP2 step 1 =="
python3 src/audit_b2.py \
  --b2 data/formatted/vietnamese_v2/B2_train.txt \
  --mtup2 data/formatted/vietnamese_v2/MTUP2_train.txt \
  --b2_dev data/formatted/vietnamese_v2/B2_dev.txt \
  --mtup2_dev data/formatted/vietnamese_v2/MTUP2_dev.txt

echo "== Audit MTUP2 setup =="
python3 src/audit_mtup2_setup.py --config "$CONFIG"

echo "== Generate MTUP2-only LR-search configs =="
python3 src/tune_lr.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  --n_iter 3 \
  --lr_min 1e-6 \
  --lr_max 5e-5 \
  --epochs 30 \
  --out_dir "$TUNE_DIR"

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; stopping before GPU training."
  echo "On a CUDA machine, rerun with: RUN_TRAIN=1 bash run_scripts/run_mtup2_vi_clean.sh"
  exit 0
fi

echo "== Train exactly the MTUP2 LR candidates =="
python3 - "$MANIFEST" "$SEED" <<'PY'
import json
import subprocess
import sys

manifest_path, seed = sys.argv[1], sys.argv[2]
manifest = json.load(open(manifest_path))
for config in manifest["configs"]:
    subprocess.run(
        ["python3", "src/train.py", "--config", config, "--seed", seed],
        check=True,
    )
PY

echo "== Select LR by lowest dev loss =="
python3 src/tune_lr.py --config "$CONFIG" --seed "$SEED" --summarize "$MANIFEST"

SELECTED_CONFIG="$(python3 - "$TUNE_DIR/MTUP2_lrsearch_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
selected = summary.get("selected")
if not selected:
    raise SystemExit("No selected LR run found. Check trainer_log_history.json files.")
print(selected["config"])
PY
)"

echo "== Public-test inference using selected validation checkpoint =="
python3 src/inference.py --config "$SELECTED_CONFIG" --seed "$SEED" --split test

RESULT_INFO="$(python3 - "$SELECTED_CONFIG" "$SEED" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from config_utils import load_yaml_config

cfg = load_yaml_config(sys.argv[1])
seed = sys.argv[2]
run_name = cfg.get("run_name", cfg["method"])
pred = Path("results") / cfg["group"] / run_name / f"s{seed}_pred.txt"
print(cfg["group"], run_name, pred)
PY
)"

GROUP="$(echo "$RESULT_INFO" | awk '{print $1}')"
RUN_NAME="$(echo "$RESULT_INFO" | awk '{print $2}')"
PRED="$(echo "$RESULT_INFO" | awk '{print $3}')"

echo "== Evaluate selected MTUP2 run =="
python3 src/evaluate.py \
  --pred "$PRED" \
  --gold "$GOLD" \
  --method MTUP2 \
  --seed "$SEED" \
  --group "$GROUP"

echo "Done. Selected run: $RUN_NAME"
