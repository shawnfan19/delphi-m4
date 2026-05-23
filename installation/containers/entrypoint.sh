#!/bin/bash
set -e

DELPHI_REPO="${DELPHI_REPO:-https://github.com/gerstung-lab/Delphi.git}"
DELPHI_BRANCH="${DELPHI_BRANCH:-main}"

if [ ! -d /workspace/Delphi ]; then
    if [ -n "$GH_TOKEN" ]; then
        AUTH_REPO="${DELPHI_REPO/https:\/\//https://x-access-token:$GH_TOKEN@}"
    else
        AUTH_REPO="$DELPHI_REPO"
    fi
    git clone --branch "$DELPHI_BRANCH" "$AUTH_REPO" /workspace/Delphi
    unset GH_TOKEN AUTH_REPO
    # The just-cloned tree isn't pip-installed yet; install it.
    # (When /workspace/Delphi was pre-baked into the image, the install
    # already happened at build time, so this branch is skipped — that
    # matters for AoU Batch VMs where PyPI is unreachable from inside
    # the perimeter.)
    cd /workspace/Delphi
    pip install --no-cache-dir -e .
else
    cd /workspace/Delphi
fi

echo "=========================================="
echo "GPU Status"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU detected"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" 2>&1 || true
echo "=========================================="

# Only exec when this file is run directly (e.g. as the SAK ENTRYPOINT).
# When sourced (e.g. from a dsub --script), skip exec so the caller can
# continue after the source line.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -eq 0 ]; then
        exec /bin/bash
    else
        exec "$@"
    fi
fi
