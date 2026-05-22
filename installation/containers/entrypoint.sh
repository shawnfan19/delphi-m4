#!/bin/bash
set -e

DELPHI_REPO="${DELPHI_REPO:-https://github.com/gerstung-lab/Delphi.git}"
DELPHI_BRANCH="${DELPHI_BRANCH:-main}"

if [ ! -d /workspace/Delphi ]; then
    cd /workspace
    git clone --branch "$DELPHI_BRANCH" "$DELPHI_REPO"
fi
cd /workspace/Delphi
pip install --no-cache-dir -e .

echo "=========================================="
echo "GPU Status"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU detected"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" 2>&1 || true
echo "=========================================="

if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi
