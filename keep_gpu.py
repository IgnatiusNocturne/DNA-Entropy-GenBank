#!/usr/bin/env python3
"""Standalone always-on GPU keeper for DNA-Entropy.

Run this ONCE at the start of your demo window and leave the window open:

    python keep_gpu.py

It secures a GPU VM in YOUR Google Cloud project and keeps it RUNNING 24/7 (retrying
forever if GPUs are momentarily out of stock), pre-installs the Evo stack, and restarts
the VM if cloud maintenance ever stops it. It never deletes or stops the VM.

  *** The VM keeps billing (~$0.74/h) until you delete it yourself. ***
  When finished:  gcloud compute instances delete dna-entropy-box --zone=<zone>

Requires only the Google Cloud CLI (`gcloud`) installed and authenticated:
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID
"""

from __future__ import annotations

import os
import sys

# Allow running straight from a checkout without `pip install` (src/ layout).
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dna_entropy.cloud.keeper import main

_USAGE = """DNA-Entropy GPU keeper

Secures a GPU VM in your Google Cloud project and keeps it RUNNING 24/7 until you stop
this program and delete the VM yourself. Just run it with no arguments (or double-click
the .exe) and leave the window open:

    python keep_gpu.py

One-time setup: install the Google Cloud CLI, then
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID

The VM bills continuously (~$0.74/h) while it is up. When finished:
    gcloud compute instances delete dna-entropy-box --zone=<zone>
"""

if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(_USAGE)
    else:
        main()
