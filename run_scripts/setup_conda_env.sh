#!/bin/bash
# Create/update the Conda environment used for CUDA server runs.
#
# Override ENV_NAME or TORCH_INDEX_URL if needed:
#   ENV_NAME=visempar-mtup2 bash run_scripts/setup_conda_env.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash run_scripts/setup_conda_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-visempar-mtup2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available on PATH."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

python --version
python -m pip install --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available in this Conda environment.")
PY

echo
echo "Environment is ready."
echo "Activate it in each server shell with:"
echo "  conda activate $ENV_NAME"
echo
echo "Then run:"
echo "  RUN_TRAIN=1 bash run_scripts/run_mtup2_vi_clean.sh"
