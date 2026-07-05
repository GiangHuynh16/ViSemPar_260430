#!/usr/bin/env bash
# Install FlashAttention in the CUDA Conda environment used for training.
#
# Run this on the RTX/CUDA server, after CUDA PyTorch is installed:
#   bash run_scripts/install_flash_attn.sh
#
# Optional overrides:
#   ENV_NAME=visempar-mtup2 bash run_scripts/install_flash_attn.sh
#   FLASH_ATTN_VERSION=2.7.4.post1 bash run_scripts/install_flash_attn.sh
#   FLASH_ATTN_WHEEL_URL=https://.../flash_attn-...whl bash run_scripts/install_flash_attn.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-visempar-mtup2}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-}"
MAX_JOBS="${MAX_JOBS:-4}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.cache/visempar_flash_attn_install}"
WORK_DIR="$INSTALL_ROOT/work"
TMP_DIR="$INSTALL_ROOT/tmp"
PIP_CACHE_DIR="$INSTALL_ROOT/pip-cache"

export MAX_JOBS
export TMPDIR="$TMP_DIR"
export PIP_CACHE_DIR

mkdir -p "$WORK_DIR" "$TMPDIR" "$PIP_CACHE_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
  else
    echo "ERROR: Conda env '$ENV_NAME' does not exist."
    echo "Create it first with: bash run_scripts/setup_conda_env.sh"
    exit 1
  fi
fi

python - <<'PY'
import platform
import sys

if platform.system() != "Linux":
    raise SystemExit(
        "ERROR: flash-attn should be installed on the Linux CUDA server, "
        f"not on {platform.platform()}."
    )

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: PyTorch is not installed/importable: {exc}")

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)

if not torch.cuda.is_available() or torch.version.cuda is None:
    raise SystemExit(
        "ERROR: CUDA PyTorch is not available. Install CUDA PyTorch first with "
        "run_scripts/setup_conda_env.sh, then rerun this script."
    )

print("gpu:", torch.cuda.get_device_name(0))
PY

cd "$WORK_DIR"

echo
echo "Using FlashAttention install workspace:"
echo "  work:      $WORK_DIR"
echo "  tmp:       $TMPDIR"
echo "  pip cache: $PIP_CACHE_DIR"
echo

python -m pip install --cache-dir "$PIP_CACHE_DIR" --upgrade pip
python -m pip install --cache-dir "$PIP_CACHE_DIR" packaging ninja setuptools wheel

if [[ -n "$FLASH_ATTN_WHEEL_URL" ]]; then
  WHEEL_PATH="$WORK_DIR/${FLASH_ATTN_WHEEL_URL##*/}"
  python - "$FLASH_ATTN_WHEEL_URL" "$WHEEL_PATH" <<'PY'
import sys
import urllib.request

url, out = sys.argv[1], sys.argv[2]
print(f"Downloading wheel:\n  {url}\n  -> {out}")
urllib.request.urlretrieve(url, out)
PY
  python -m pip install --no-cache-dir "$WHEEL_PATH"
elif [[ -n "$FLASH_ATTN_VERSION" ]]; then
  python -m pip install --no-cache-dir "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
else
  python -m pip install --no-cache-dir flash-attn --no-build-isolation
fi

python - <<'PY'
import flash_attn

print("flash_attn:", flash_attn.__version__)
print("FlashAttention install OK.")
PY

echo
echo "Next, verify training logs show:"
echo "  Attention implementation: flash_attention_2"
