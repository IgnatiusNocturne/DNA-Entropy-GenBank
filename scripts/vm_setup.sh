#!/usr/bin/env bash
# Install the Evo 2 stack on a fresh Google Deep Learning VM, from PUBLIC sources only
# (no private image, nothing hosted by us). Idempotent: safe to re-run; on a reused box
# it detects the stack is already present and exits fast.
set -e

if python3 -c 'import evo2, flash_attn' 2>/dev/null; then
  echo "EVO_STACK_PRESENT"
  exit 0
fi

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | head -1 || echo /usr/local/cuda)"
# Build flash-attn only for THIS VM's GPU arch (L4=8.9, A100=8.0) -> fast, and works on
# whichever GPU we landed on after the stockout fallback.
ARCH="$(python3 -c 'import torch;cc=torch.cuda.get_device_capability();print(f"{cc[0]}.{cc[1]}")')"
export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS=4

echo "===== installing Evo stack (arch sm_$ARCH) $(date) ====="
python3 -m pip install --break-system-packages -q typer pyrodigal evo2 ninja
python3 -m pip install --break-system-packages --no-build-isolation flash-attn==2.8.3
python3 -c 'import evo2, flash_attn; print("EVO_STACK_OK")'
echo "===== done $(date) ====="
