#!/usr/bin/env bash
echo "=== /opt ==="; ls -la /opt 2>/dev/null
echo "=== which python/python3/conda ==="; which python python3 conda 2>/dev/null
echo "=== /usr/bin/python* ==="; ls /usr/bin/python* 2>/dev/null
echo "=== python3 version ==="; python3 --version 2>&1
echo "=== torch via python3 ==="; python3 -c 'import torch; print("torch", torch.__version__, torch.cuda.is_available(), torch.version.cuda)' 2>&1 | tail -3
echo "=== /etc/profile.d ==="; ls /etc/profile.d/ 2>/dev/null
echo "=== venvs/activate under /opt ==="; find /opt -maxdepth 4 -name activate 2>/dev/null | head
echo "=== torch install location ==="; find /opt /usr -maxdepth 7 -type d -name torch 2>/dev/null | head
echo "=== pip (python3) ==="; python3 -m pip --version 2>&1 | head -1
echo "=== DONE ==="
