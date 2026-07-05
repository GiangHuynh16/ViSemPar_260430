#!/usr/bin/env bash
# Install FlashAttention in the CUDA Conda environment used for training.
#
# Run this on the RTX/CUDA server, after CUDA PyTorch is installed:
#   bash run_scripts/install_flash_attn.sh
#
# Optional overrides:
#   ENV_NAME=visempar-mtup2 bash run_scripts/install_flash_attn.sh
#   FLASH_ATTN_VERSION=2.7.4.post1 bash run_scripts/install_flash_attn.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-visempar-mtup2}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-}"
MAX_JOBS="${MAX_JOBS:-4}"

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

python -m pip install --upgrade pip
python -m pip install packaging ninja setuptools wheel

export MAX_JOBS
if [[ -n "$FLASH_ATTN_VERSION" ]]; then
  python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
else
  python -m pip install flash-attn --no-build-isolation
fi

python - <<'PY'
import flash_attn

print("flash_attn:", flash_attn.__version__)
print("FlashAttention install OK.")
PY

echo
echo "Next, verify training logs show:"
echo "  Attention implementation: flash_attention_2"
