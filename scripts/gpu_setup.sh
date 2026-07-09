#!/usr/bin/env bash
# Install the Evo 2 stack into the system Python (torch 2.9.1+cu129 already present).
export PATH=/usr/local/cuda/bin:$PATH
echo "===== evo2 install START $(date) ====="
echo "nvcc: $(which nvcc 2>/dev/null || echo NONE)"
echo "torch (pre): $(python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"
echo "----- pip install evo2 -----"
python3 -m pip install --break-system-packages evo2
echo "PIP_EVO2_EXIT=$?"
echo "----- torch (post, must still be 2.9.1) -----"
python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1
echo "----- import evo2 -----"
python3 -c 'from evo2 import Evo2; print("evo2 import OK")' 2>&1
echo "----- flash-attn present? -----"
python3 -c 'import flash_attn; print("flash_attn", flash_attn.__version__)' 2>&1 | tail -1
echo "===== ALL DONE $(date) ====="
