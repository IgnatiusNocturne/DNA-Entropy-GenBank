# EVO_SETUP — running Evo 2 (7B) on a GPU

The real Evo predictor needs an NVIDIA GPU with ~24 GB VRAM. There are two ways to get one:

- **Cloud (automated):** the **GPU keeper** (`keep_gpu.py` / `keep-gpu.exe`) does everything
  in this doc for you — it provisions the VM, installs the Evo stack, and keeps it running.
  You don't run these commands by hand; see [DISTRIBUTION.md](DISTRIBUTION.md). This file is
  the *reference* for what the keeper automates (and how to debug it).
- **Local:** if your own machine has a 24 GB NVIDIA GPU, install the Evo stack (below) and
  run `--predictor evo` directly — no cloud needed.

Everything else (readers, validation, entropy, writers) develops against `MockPredictor`,
which needs no GPU.

## Target hardware

- **GPU:** any NVIDIA card with **≥ 24 GB VRAM**. Concretely: **GCP L4** (Ada) or
  **AWS A10G** (Ampere) — both 24 GB.
- **Why 7B + a 24 GB card:** Evo 2 **7B runs in bfloat16 with no FP8**, so it works on
  Ampere *or* Ada (no Hopper needed). 7B weights are ~14 GB in bf16, comfortable in 24 GB.
  *(The 1B model would be smaller but requires FP8 → Ada/Hopper only; H100 is overkill
  and expensive. So: 7B in bf16 on the cheapest 24 GB card.)*
- **Cloud instance — pick one:**
  - **GCP:** `g2-standard-8` (1× **L4** 24 GB, 32 GB RAM). ~$0.70–0.85/hr on-demand,
    less on Spot. Use `-8` (not `-4`) so host RAM has headroom to load the checkpoint.
    *(GCP has no A10; its lineup is T4 / L4 / A100 / H100 — L4 is the right pick.)*
  - **AWS:** `g5.xlarge` (1× **A10G** 24 GB). ~$1/hr on-demand, cheaper on spot.
  - **Billing is continuous while the VM exists.** For the always-on cloud model the keeper
    keeps the box running on purpose; **you delete it by hand when done**. For a manual box,
    stop it when idle.

## Provision the VM manually (reference — the keeper does this for you)

The keeper creates the VM automatically. This is the equivalent manual command if you want
to stand one up yourself (or run locally, skip this). Single L4, sized for Evo 2 7B. **First
request quota** for "NVIDIA L4 GPUs" (≥1, in your zone) under IAM & Admin -> Quotas — new
projects have 0 GPU quota and `create` will fail without it.

```bash
gcloud compute instances create evo-7b \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \            # 1x L4 (24 GB), 8 vCPU, 32 GB RAM
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu123-debian-11 \   # Deep Learning VM: CUDA + driver + conda
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \                  # ~50 GB used (DLVM + env + 13.8 GB weights); 100 = headroom
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \          # required: GPU VMs can't live-migrate
  --restart-on-failure
  # cheaper (reclaimable; fine for short runs):
  #   --provisioning-model=SPOT --instance-termination-action=STOP

gcloud compute ssh evo-7b --zone=us-central1-a    # handles keys/firewall

# A manual box like this is billed while it exists — stop or delete it when idle:
#   gcloud compute instances stop evo-7b --zone=us-central1-a
# (The keeper's always-on box is meant to stay running; you delete it by hand when done.)
```

*AWS equivalent: `g5.xlarge` (A10G 24 GB) with a Deep Learning AMI; `ssh` in normally.*
*If GCP rejects the DLVM image on G2, use `--image-family=ubuntu-2204-lts
--image-project=ubuntu-os-cloud` and install the GPU driver per GCP's docs.*

## One-time setup on the GPU box

**Verified 2026-06-18** on the GCP `pytorch-2-9-cu129-ubuntu-2404` DLVM image. That image
ships **system Python 3.12 + torch 2.9.1+cu129 preinstalled** (no conda/venv), so we
install into the system interpreter with `--break-system-packages`.

```bash
# 1. SSH in; confirm the GPU + the preinstalled torch
nvidia-smi
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'   # 2.9.1+cu129 True

# 2. Evo 2 (pulls vortex/vtx, biopython, huggingface_hub; leaves torch untouched)
python3 -m pip install --break-system-packages evo2

# 3. flash-attn — REQUIRED by vortex (import fails without flash_attn_2_cuda).
#    There is NO cu12 prebuilt wheel for torch 2.9 (only cu13), and our torch is cu12.9,
#    so build from source against the installed torch, restricted to the L4 arch (sm_89)
#    to keep the build to a couple of minutes.
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
python3 -m pip install --break-system-packages ninja
python3 -m pip install --break-system-packages --no-build-isolation flash-attn==2.8.3

# 4. Sanity: evo2 must import
python3 -c 'from evo2 import Evo2; print("evo2 import OK")'

# NOTE: Transformer Engine / FP8 is only for the 1B/40B FP8 path. For 7B in bf16 it is
# not needed (you'll see a harmless "Transformer Engine not installed" warning).
```

Verify against the canonical instructions, which can change:
[ArcInstitute/evo2 README](https://github.com/ArcInstitute/evo2/blob/main/README.md).

## Smoke test (confirms weights load + a forward pass works)

```python
from evo2 import Evo2
model = Evo2("evo2_7b")          # downloads weights on first run (large; needs disk + HF)
out = model("ACGTACGTACGT")       # returns logits; confirm shape & no OOM
print("ok")
```

If this runs without OOM, the box can serve `EvoPredictor`.

## Running this tool on the GPU box

Copy the project up (no git remote needed) and install it. torch/evo2 are already
present, so just install the package (+ pytest):

```bash
# from the laptop:
gcloud compute scp --recurse src tests pyproject.toml README.md dna-entropy:DNA-Entropy/ \
    --zone=us-east1-c --ssh-key-file=<key>

# on the box:
cd ~/DNA-Entropy
python3 -m pip install --break-system-packages -e ".[dev]"
python3 -m pytest -m gpu            # validates EvoPredictor vs the real model (3 tests)

# the console script lands in ~/.local/bin (often off PATH); invoke as a module:
python3 -m dna_entropy.cli run -i locus.fa --predictor evo --name demo --out ~/out
```

Copy the `~/out/` files back to your laptop (`gcloud compute scp --recurse dna-entropy:out ...`)
and load them into IGV (see [DESIGN.md §7](DESIGN.md#7-igv-output)).

## Notes & troubleshooting

- **First run downloads weights** (several GB) from Hugging Face — ensure disk space and,
  if needed, `huggingface-cli login`.
- **`transformer-engine` build fails:** skip it; it's not needed for 7B bf16.
- **OOM:** lower `--max-len` (the single-pass context cap) or use a bigger-VRAM instance.
- **flash-attn version mismatch:** pin to the version the current evo2 README specifies.
- **Keeper "no quota" vs "no capacity":** before probing zones, the keeper reads your
  project's GPU quota per region (one `regions list` call) and only chases zones in regions
  that actually have quota — so `no <GPU> quota in this region` spam disappears and the
  remaining `no capacity` messages are genuine, transient stockouts (retry succeeds). If
  **no region has any GPU quota**, that is the fixable case: the keeper prints the one-time
  quota-request steps (IAM & Admin -> Quotas, request `NVIDIA_L4_GPUS` >= 1 in your region)
  and keeps retrying so it picks up automatically once the grant lands. It also runs
  `nvidia-smi` on a freshly-acquired box, so a VM that is SSH-reachable but has no live GPU
  is caught and re-checked instead of being treated as ready.
- **Keeper setup health check ("nothing ever provisions"):** on startup the keeper prints a
  `GCP setup health check` block that verifies, with the **exact gcloud error** and the one
  command that fixes each, that: the project is reachable/ACTIVE, **billing is enabled**
  (a project with no billing account cannot create *any* VM — the classic "creates are
  attempted but nothing appears" symptom), the **Compute Engine API is enabled**, and GPU
  quota exists. If the **Compute Engine API is off**, the keeper **enables it automatically**
  (`gcloud services enable compute.googleapis.com`, ~30-60s; safe and idempotent) and only
  reports an error if that fails (e.g. the account lacks `serviceusage.services.enable`),
  printing the exact gcloud text and the manual command. Billing-off and API-off are
  project-wide, so during acquisition the keeper stops on the first zone and prints the
  remediation verbatim instead of mislabelling them as "no capacity" across every zone. If a
  check can't be run (e.g. the Cloud Billing API itself is off) it prints `WARN:` and
  continues rather than blocking.
- **Keep `EvoPredictor` the only Evo-aware module** — see [CLAUDE.md](../CLAUDE.md) hard
  rule #1. This file documents the *environment*; the code lives in
  `src/dna_entropy/predictors/evo.py`.
