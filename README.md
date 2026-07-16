# DNA-Entropy

Compute the **per-position information entropy** of a DNA sequence with the **Evo 2 (7B)**
genomic language model, and export it as a track for **IGV**, **GenBank**-aware editors
(SnapGene / Benchling / Geneious), and genome browsers (UCSC / JBrowse). Runs Evo 2 on an
**always-on GPU in your own Google Cloud**, or on a **local NVIDIA GPU** if you have one.

![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-proof--of--concept-orange)

> Proof of concept for **Prof. Meers, Vanderbilt University Medical Center**.
> Project rules live in [CLAUDE.md](CLAUDE.md); deeper detail in [docs/](docs/).

---

## What it does

```
input (GenBank / FASTA / pasted DNA)  ->  validate  ->  Evo 2 (one forward pass)
      ->  Shannon entropy per base  ->  export (entropy track + GenBank + stats)
```

For every position, Evo 2 predicts the probability of each nucleotide. We turn those four
probabilities into a single **Shannon entropy** value in bits:

```
H = -Σ pᵢ · log₂(pᵢ)      p = [P(A), P(C), P(G), P(T)]
```

- All four bases at 25% → **2.0 bits** (maximum uncertainty).
- One base at 100%       → **0.0 bits** (fully predictable).

**Low-entropy** regions are highly predictable to the model (often conserved / structured);
**high-entropy** regions are uncertain. That per-base "surprise" track is the scientific
deliverable.

## Inputs and outputs

The output set depends on what you feed in:

| Input | What we do | Files produced |
|-------|-----------|----------------|
| **GenBank** (`.gb`/`.gbk`) | Use the genes **already in the file** (never re-annotated); compute entropy. **Multi-record files are fully supported** — every record is analyzed | `<name>.gb` (all records, genes + per-gene `/note="mean_entropy=..."`), `<name>.entropy.wig` (one block per record), `<name>.entropy.geneious.gff3` (per-position track for Geneious), `stats.txt` |
| **FASTA** (`.fa`/`.fasta`) or **pasted sequence** | Compute entropy; call genes with Prodigal when possible | `<name>.fasta`, `<name>.entropy.bedgraph`, `<name>.entropy.geneious.gff3`, `<name>.summary.txt`, `<name>.genes.gff3` *(with `--genes`)*, plus a bonus `<name>.gb` |

Why several entropy-track formats? A GenBank file has **no channel for a per-base numeric
graph**, so the full-resolution entropy graph always ships as a companion track: **WIG /
bedGraph** for IGV/UCSC/JBrowse, and **GFF3** (`<name>.entropy.geneious.gff3`) for
**Geneious Prime**, which imports GFF3 but *not* WIG/bedGraph as graphs. The GenBank
itself carries only a **coarse per-gene mean** as a note (so it shows in
SnapGene/Benchling/Geneious). Between them, whatever tool you open, you see something
meaningful.

## Architecture

```
readers/ ─▶ validation/ ─▶ predictors/ ─▶ analysis/ ─▶ writers/
 (genbank,     (A/C/G/T,      (Evo 2 |      (Shannon    (genbank, wig,
  fasta,        N-tolerant,    Mock)         entropy)     bedgraph, fasta,
  paste)        RNA, length)                              gff3, summary)
```

One direction, no back-edges; each stage is swappable. **The model is the only swap point
that matters**: all Evo-specific code lives in `predictors/evo.py`, and everything
downstream depends only on an `(L, 4)` probability array — so Evo can later be replaced by
a custom ANN with zero downstream changes.

## Fastest start: the Windows `.exe` (no Python)

Two double-click executables (build them with `packaging/build_exe.ps1`, or grab them from
a GitHub Release):

1. **`keep-gpu.exe`** — double-click once and leave the window open. It brings up your GPU
   box and keeps it running. (One-time GCP setup below.)
2. **`dna-entropy.exe`** — double-click, then drop a GenBank/FASTA file (or paste a
   sequence) onto the window. It runs Evo on the box and saves the results to your
   `Downloads\<name>\` folder.

The `.exe` is a lightweight client — it does **not** bundle Evo/torch, so it always uses
the cloud box. To run on a **local** NVIDIA GPU, use the Python install below.

## The GPU: an always-on box in *your* Google Cloud

Evo 2 needs a ~24 GB GPU, so it runs on a VM **in your own Google Cloud project** — your
sequence never leaves it. This repo splits GPU management from analysis:

1. **`keep_gpu.py`** — the **keeper**. Run it once and leave it open. It secures a GPU VM
   (retrying across every zone forever if GPUs are momentarily out of stock — it never
   errors out), pre-installs the Evo stack, and **keeps the VM running 24/7**, restarting
   it if cloud maintenance ever stops it. It never stops or deletes the VM.
2. **`dna-entropy cloudrun`** — the **app**. It connects to the keeper's box, runs Evo on
   your locus, downloads the results, and **leaves the box running** for the next run.

> 💸 **Cost:** the VM bills continuously (~**$0.74/h** for an L4 ≈ **$18/day**) for as long
> as the keeper keeps it up. This is intentional — it makes each run instant. **You must
> delete the VM yourself when finished:**
> ```
> gcloud compute instances delete dna-entropy-box --zone=<zone>
> ```

### One-time cloud setup

```bash
# 1. Install the Google Cloud CLI:  https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID          # billing enabled
# 2. Request NVIDIA L4 GPU quota once (the keeper points you to the page if you have none).
```

### Running

```bash
# Terminal 1 — start the keeper and leave it open for your demo window (~days):
python keep_gpu.py

# Terminal 2 — once it prints "GPU is UP", run as many loci as you like:
dna-entropy cloudrun -i my_locus.gb            # GenBank in -> GenBank + entropy + stats
dna-entropy cloudrun -i my_locus.fasta         # FASTA in   -> full IGV set + GenBank
```

When you are done for good, stop the keeper (Ctrl-C) **and delete the VM** with the command
above.

### Or run on a local NVIDIA GPU

If this machine has an NVIDIA GPU (~24 GB) with the Evo stack installed (`pip install
-e ".[evo]"`, see [docs/EVO_SETUP.md](docs/EVO_SETUP.md)), you can skip the cloud entirely:

```bash
dna-entropy run -i my_locus.gb --name my_locus --predictor evo     # runs here, on your GPU
```

The double-click wizard also **asks** whether to use the cloud or this computer, and if you
pick local but it isn't available, it **falls back to the cloud automatically**
(`cloudrun --prefer-local`).

## Install (local dev, no GPU)

The mock-predictor pipeline installs and runs anywhere — no GPU, no Evo:

```bash
pip install -e ".[dev]"      # numpy, typer, biopython, pytest
```

Then run everything locally against the seeded mock predictor:

```bash
# GenBank in -> genes preserved, entropy computed
dna-entropy run -i tests/data/sample.gb --name toy --predictor mock

# FASTA / pasted sequence
dna-entropy run -i locus.fasta --name demo --predictor mock
echo "ATGCGTACGTTAGC" | dna-entropy run --name demo --predictor mock
```

Optional local gene calling (prokaryotic, CPU-only):

```bash
pip install -e ".[genes]"    # pyrodigal
```

The real Evo predictor (`--predictor evo`) needs an NVIDIA GPU with ~24 GB VRAM (GCP **L4**
/ AWS **A10G**); Evo 2 7B runs in bf16, so no FP8 hardware is required. See
[docs/EVO_SETUP.md](docs/EVO_SETUP.md).

## Viewing the results

**In IGV** (full per-base graph):
1. *Genomes → Load Genome from File…* → `<name>.fasta` (or load the `.gb`).
2. *File → Load from File…* → `<name>.entropy.bedgraph` / `.wig` (and `.genes.gff3`).
3. The entropy track renders as a graph aligned to the locus.

**In Geneious Prime** (full per-base entropy track):
1. Import the sequence: *File → Import → From File…* → `<name>.gb` (or `<name>.fasta`).
2. Import the track onto it: *File → Import → From File…* → `<name>.entropy.geneious.gff3`.
   When Geneious asks, import the annotations **onto the existing `<name>` sequence** (not
   as a new document) so the coordinates line up. The entropy lands as an `entropy`
   annotation track.
3. Shade it by value: right-click the track (or use the track's popup menu) → *Color by /
   Heatmap* → choose the **`entropy`** qualifier (or **score**). Each position is now
   colored by its Shannon entropy (0–2 bits).

Why not `.wig`/`.bedgraph` in Geneious? Geneious Prime imports GFF3 but treats WIG/bedGraph
as *export-only* graph formats — it won't load them as tracks. The
`<name>.entropy.geneious.gff3` file is the Geneious-native equivalent.

**In SnapGene / Benchling / Geneious (coarse per-gene view)**: open `<name>.gb` — genes
carry a `mean_entropy=… bits` note.

## Testing

```bash
pytest -m "not gpu"          # everything that runs without a GPU (use this on a laptop)
pytest                       # includes Evo GPU integration tests (CUDA box only)
pytest tests/test_genbank.py # a single file
```

GPU/Evo tests are marked `@pytest.mark.gpu` and skip themselves without CUDA. Every
behavior change ships with a test.

## Project layout

```
keep_gpu.py                    # standalone always-on GPU keeper
src/dna_entropy/
├── cli.py                     # run · validate · cloudrun · keep-gpu
├── pipeline.py                # wires the stages; branches on input kind
├── readers/                   # detect · genbank · fasta · paste · input
├── validation/                # normalize + validate (N-tolerant for GenBank/FASTA)
├── predictors/                # base (L,4) contract · mock · evo · logits
├── analysis/                  # Shannon entropy + summary stats
├── annotators/                # Prodigal gene calling (optional)
├── writers/                   # genbank · wig · bedgraph · geneious · fasta · gff · summary
└── cloud/                     # gcloud wrappers · orchestrator · keeper
tests/                         # mirrors src/, plus data/ fixtures
```

## Documentation

- [CLAUDE.md](CLAUDE.md) — project rules & architecture summary.
- [docs/DESIGN.md](docs/DESIGN.md) — architecture, interfaces, CLI, validation, entropy math.
- [docs/EVO_SETUP.md](docs/EVO_SETUP.md) — Evo 2 install & cloud-GPU setup.
- [docs/ROADMAP.md](docs/ROADMAP.md) — sprint plan & backlog.

## License

Released under the [MIT License](LICENSE).
