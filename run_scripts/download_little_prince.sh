#!/bin/bash
# Download and prepare the public Little Prince AMR corpus for English runs.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="${TMPDIR:-/tmp}/visempar_little_prince"
RAW_OUT="$ROOT/data/english/little_prince/raw"

mkdir -p "$TMP_DIR"
cd "$ROOT"

BASE_URL="https://raw.githubusercontent.com/flipz357/AMR-World/main/data/reference_amrs"

curl -L --fail --max-time 60 "$BASE_URL/amr-bank-struct-v1.6.txt" \
    -o "$TMP_DIR/amr-bank-struct-v1.6.txt"
curl -L --fail --max-time 60 "$BASE_URL/amr-bank-struct-v1.6-training.txt" \
    -o "$TMP_DIR/amr-bank-struct-v1.6-training.txt"
curl -L --fail --max-time 60 "$BASE_URL/amr-bank-struct-v1.6-test.txt" \
    -o "$TMP_DIR/amr-bank-struct-v1.6-test.txt"

python3 src/prepare_little_prince.py \
    --full  "$TMP_DIR/amr-bank-struct-v1.6.txt" \
    --train "$TMP_DIR/amr-bank-struct-v1.6-training.txt" \
    --test  "$TMP_DIR/amr-bank-struct-v1.6-test.txt" \
    --out_dir "$RAW_OUT"

python3 src/data_pipeline.py \
    --raw_dir data/english/little_prince/raw \
    --formatted_dir data/formatted/english_lpp \
    --methods B1 B2 MTUP2 \
    --amr1_weight 1 \
    --amr2_weight 0 \
    --syn_weight 0 \
    --no_var_mode clean \
    --no_dedup \
    --language en
