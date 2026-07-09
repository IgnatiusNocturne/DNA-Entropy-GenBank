# DISTRIBUTION — how DNA-Entropy ships

Goal: a user double-clicks an `.exe`, gives a locus, and gets their results — running on
**their own** Google Cloud (or a local GPU), with **nothing hosted or paid for by us**.

## Two executables

Built by `packaging/build_exe.ps1` (distribute via a GitHub Release):

| Executable | Role |
|---|---|
| **`keep-gpu.exe`** | The always-on **keeper**. Double-click once, leave open. Secures a GPU VM and keeps it running 24/7. |
| **`dna-entropy.exe`** | The **client**. Double-click, drop a GenBank/FASTA file (or paste a sequence). Runs Evo on the box and saves results to `Downloads\<name>\`. |

Both are lightweight PyInstaller builds that bundle the package source (`--add-data
src/dna_entropy;_pkgsrc`) and Biopython, but **not** torch/evo2/flash-attn — those run on
the GPU, not on the user's laptop.

## Architecture: decentralized, public sources only

```
  user's laptop                         user's OWN gcloud project
  ─────────────                         ─────────────────────────
  keep-gpu.exe  ───────────────────▶    create a STOCK GPU VM (Google PUBLIC DLVM image,
   (run once, leave open)                L4 -> A100 fallback, multi-zone, retry forever)
        │                                install Evo from PUBLIC pip + HF weights
        │                                keep it RUNNING 24/7 (restart on maintenance)
        │
  dna-entropy.exe                        find the box, wake if stopped
   drop GenBank/FASTA/seq  ──upload──▶   run: dna-entropy run --predictor evo
   validate locally        ◀─download──  <name>.gb / .entropy.wig / stats.txt (etc.)
  save Downloads\<name>\                 box LEFT RUNNING for the next run
```

- **We host nothing.** No private image, no server. The base OS is **Google's public Deep
  Learning image**; Evo 2 + flash-attn come from **public pip**; weights from **Hugging
  Face** (Apache-2.0). The tool uploads its own source to the box.
- **The author pays $0 ongoing.** The user pays only their own GPU time in their own
  account.
- **Data stays in the user's project** — nothing transits anything of ours.

## The keeper / client split (why it's decoupled)

Provisioning a GPU and analyzing a locus are separate concerns:

- **`keep-gpu.exe` owns the VM lifecycle.** It creates the box (`dna-entropy-box`), retries
  across every zone/region forever if GPUs are out of stock (L4 → A100 fallback, last-good
  zone first), pre-installs Evo, and restarts the VM if host maintenance stops it. It
  **never stops or deletes** the box. Cost is continuous (~$0.74/h for an L4) — intentional,
  so each run is instant. The user deletes the VM by hand when finished.
- **`dna-entropy.exe` never touches the VM lifecycle.** It finds the keeper's box, wakes it
  if stopped, runs, downloads, and **leaves it running**. If no box exists it errors with
  "start the keeper first" — it will not create one.

This makes runs fast and predictable, and keeps the billing decision explicit and in the
user's hands.

## Local GPU (no cloud)

If the user's machine has an NVIDIA GPU with the Evo stack installed, the client can run
Evo **locally**: `cloudrun --prefer-local` (or the wizard's prompt) tries the local GPU
first and falls back to the cloud on any failure. The frozen `.exe` bundles no torch, so it
is cloud-only; local runs use a source install (`pip install -e ".[evo]"`).

## One-time user setup (the app guides each step)

1. **Google Cloud account** + `gcloud auth login`.
2. **A project with billing enabled** (`gcloud config set project ...`).
3. **GPU quota** (new projects have 0) — request "NVIDIA L4 GPUs" once; usually approved in
   minutes. The keeper points to the exact page if quota is missing.

## Components

| Piece | What |
|---|---|
| `keep_gpu.py` / `cloud/keeper.py` | the always-on keeper (acquire + watchdog loop) |
| `cloud/orchestrator.py` | connect / wake / run / download; provisioning helpers the keeper reuses |
| `cloud/gcloud.py` | thin wrappers over the user's `gcloud` |
| `cloud/ui.py` | ASCII spinner / progress |
| `packaging/launcher.py` | the double-click wizard (asks local vs cloud, forwards to the CLI) |
| config | `%APPDATA%\dna-entropy\config.json` (last-good zone; optional `ssh_key_file`) |
