#!/bin/bash
set -e

cd /workspace
git clone --branch delphi-m4 https://github.com/gerstung-lab/Delphi.git
cd /workspace/Delphi
# pip install -r Delphi/installation/requirements.txt
pip install -e .

export DELPHI_CKPT_DIR="/tmp/ckpt"
mkdir -p /tmp/ckpt
export DELPHI_DATA_DIR="/mnt/project/data"

# ========================================
# GPU Detection
# ========================================
echo "=========================================="
echo "GPU Status"
echo "=========================================="
# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed - no GPU available"
else
    echo "nvidia-smi not found"
fi
# Check CUDA visibility
echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
# Quick Python GPU check
echo ""
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: No GPU detected!')
" 2>/dev/null || echo "PyTorch GPU check failed"
echo "=========================================="
echo ""

if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi
