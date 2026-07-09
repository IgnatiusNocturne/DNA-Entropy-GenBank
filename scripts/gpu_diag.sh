#!/usr/bin/env bash
# Probe the GPU box environment so we plan the Evo install precisely.
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh; conda activate base 2>/dev/null
fi
echo "=== user/host ==="; whoami; hostname
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
echo "=== python ==="; command -v python; python --version 2>&1
echo "=== torch ==="
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available(), "| cuda", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch failed:", e)
PY
echo "=== pip pkgs ==="
pip show flash-attn 2>/dev/null | head -2
pip show evo2 2>/dev/null | head -2
pip show transformer-engine 2>/dev/null | head -2
echo "=== disk(root) ==="; df -h / | tail -1
echo "=== mem(GB)/cpu ==="; free -g | head -2; echo "nproc=$(nproc)"
echo "=== build tools ==="; gcc --version 2>/dev/null | head -1; nvcc --version 2>/dev/null | tail -1
echo "=== DONE ==="
