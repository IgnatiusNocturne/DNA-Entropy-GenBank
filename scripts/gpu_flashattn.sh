#!/usr/bin/env bash
# Build flash-attn from source against the box's torch 2.9.1+cu129.
# Restricted to the L4 architecture (sm_89) to keep the build fast.
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export TORCH_CUDA_ARCH_LIST=8.9
export FLASH_ATTN_CUDA_ARCHS=89
export MAX_JOBS=4
echo "===== flash-attn build START $(date) ====="
nvcc --version | tail -1
python3 -m pip install --break-system-packages ninja 2>&1 | tail -1
echo "----- building flash-attn 2.8.3 (sm_89 only) -----"
python3 -m pip install --break-system-packages --no-build-isolation flash-attn==2.8.3 2>&1
echo "FA_EXIT=$?"
echo "----- flash_attn import -----"
python3 -c 'import flash_attn; print("flash_attn", flash_attn.__version__)' 2>&1 | tail -2
echo "----- evo2 import -----"
python3 -c 'from evo2 import Evo2; print("evo2 import OK")' 2>&1 | tail -8
echo "===== flash-attn DONE $(date) ====="
