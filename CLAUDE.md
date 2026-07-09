# CLAUDE.md — DNA-Entropy

Guidance for Claude (and humans) working in this repo. Read this first.

## What this project is

A **command-line tool** that takes a DNA sequence, runs the **Evo 2 (7B)** genomic
language model to get the per-position probability of each nucleotide, computes the
**Shannon entropy** at every position, and exports an **entropy track** plus a
companion **FASTA** so the result can be viewed in **IGV (Integrated Genomics Viewer)**.

This is a **proof of concept** for Prof. Meers (Vanderbilt University Medical Center),
intended to run for ~1 week to test whether the idea is useful. If it is, it grows into
a year-long project with a GUI, where Evo is replaced by a custom ANN trained on Evo
embeddings + BLAST. **Therefore: build for modularity and a clean swap-out of the model,
but do not over-engineer a one-week demo.**

## Documentation map

Read these before working on the relevant area:

- **[README.md](README.md)** — what it does, install, quickstart, demo walkthrough.
- **[docs/DESIGN.md](docs/DESIGN.md)** — architecture, module layout, the `Predictor`
  interface and the `(L, 4)` data contract, entropy math, CLI spec, validation rules.
- **[docs/EVO_SETUP.md](docs/EVO_SETUP.md)** — Evo 2 install, GPU requirements, the
  always-on GPU keeper, and local-GPU setup for real runs.
- **[docs/DISTRIBUTION.md](docs/DISTRIBUTION.md)** — how it ships (double-click `.exe`s,
  the decentralized "we host nothing" cloud model).
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — the sprint plan with tickable checklists and
  the future backlog. **Update the checkboxes as work completes.**

## Architecture in one line

```
input → validate → predict → analyze → export
```

Each stage is an independent module behind an interface. The **`Predictor`** is the
single swap point: today `EvoPredictor` (Evo 2 7B) and `MockPredictor` (for GPU-less
dev); tomorrow `AnnPredictor`. Everything downstream depends only on the `(L, 4)`
probability array, never on Evo. Full detail in [docs/DESIGN.md](docs/DESIGN.md).

## Hard rules (non-negotiable)

1. **The model is the only swappable detail that matters.** All Evo-specific code lives
   *only* in `src/dna_entropy/predictors/evo.py`. Nothing else may import `torch`,
   `evo2`, or reference Evo token IDs. Downstream code depends only on the `(L, 4)`
   probability contract.
2. **Validate before predict, always.** Never feed unvalidated input to a predictor.
3. **Respect the data contract.** `Predictor.predict(seq) -> np.ndarray` of shape
   `(len(seq), 4)`, dtype `float32`, columns ordered **`[A, C, G, T]`**, each row sums
   to `1.0` (assert/normalize). Entropy output is shape `(len(seq),)` with every value
   in `[0.0, 2.0]` (assert).
4. **Every module ships with tests.** No feature is "done" until `pytest -m "not gpu"`
   passes on the GPU-less dev laptop. See "Testing" below.
5. **Keep the laptop install light.** Heavy/GPU deps (`torch`, `evo2`, `pyrodigal`) are
   *optional extras*, never core dependencies. The mock-predictor pipeline must install
   and run with no GPU and no Evo.
6. **Work sprint by sprint.** Do not start a sprint's code until the previous sprint's
   checklist is ticked and its tests pass. Tick the boxes in `docs/ROADMAP.md`.
7. **Determinism in tests.** `MockPredictor` is seeded; tests must not depend on network
   or GPU.
8. **No secrets, no weights in git.** Model weights, API keys, and large files stay out
   of the repo (`.gitignore` them).
9. **Type hints + docstrings** on every public function/class.
10. **One forward pass.** Evo is autoregressive — a single forward pass yields every
    position's next-token distribution. Never loop the model per position. Loop/window
    *only* when a locus exceeds the configured context cap (see DESIGN.md §5).
11. **ASCII-safe console output.** Lab users run Windows, whose console is cp1252 and
    crashes on Unicode glyphs (`✓ ✗ •`). Keep CLI text ASCII (`OK:` / `ERROR:` / `-`).
    Files we *write* use explicit `encoding="utf-8"`.

## Project status

Sprints 0-5 complete. The full pipeline runs on the mock predictor (GPU-less) and on real
Evo 2 (local NVIDIA GPU or the always-on cloud box). GenBank/FASTA I/O, multi-record
GenBank, the GPU keeper, and the double-click `.exe`s are all done and tested
(`pytest -m "not gpu"` green). Verified end-to-end on a real GCP L4. See
[docs/ROADMAP.md](docs/ROADMAP.md) for the per-sprint checklist and the future backlog.

## Environment notes

- Dev laptop: Windows 11, Intel Core Ultra 7 255H, 32 GB RAM, **Intel Arc GPU (no CUDA)**.
  → Evo **cannot** run here. Develop against `MockPredictor`; run real Evo (Evo 2 7B,
  bf16) on a cloud **24 GB GPU** (GCP L4 / AWS A10G) over SSH (see
  [docs/EVO_SETUP.md](docs/EVO_SETUP.md)).
- Python: this repo uses a venv at **`.venv`** built from **standard CPython 3.12**
  (`%LOCALAPPDATA%\Programs\Python\Python312\python.exe`). The Evo environment also
  targets 3.10–3.12 (the Evo stack has no 3.14 wheels).
- ⚠️ **Do not use the `python` on PATH** — it's an MSYS2/UCRT64 build for which PyPI has
  no wheels, so `numpy`/etc. try to compile from source and fail. Always use the `.venv`
  interpreter: on Windows PowerShell, `.\.venv\Scripts\python.exe` (this venv uses the
  standard `Scripts\` layout, not `bin\`).

## Testing (summary — full detail in docs/DESIGN.md)

Commands below assume an activated venv. On this laptop, either activate it
(`.\.venv\Scripts\Activate.ps1`) or prefix with `.\.venv\Scripts\python.exe -m`.

```bash
pip install -e ".[dev]"      # dev install (no GPU needed)
pytest -m "not gpu"          # all tests that run without a GPU  (use this on the laptop)
pytest                       # everything, incl. GPU tests (only on a CUDA box)
pytest tests/test_entropy.py # a single file
pytest -k entropy            # tests matching a keyword
```

- Tests live in `tests/`, files named `test_*.py`, functions `test_*`.
- Shared fixtures in `tests/conftest.py`; sample data in `tests/data/`.
- GPU/Evo integration tests are marked `@pytest.mark.gpu` and are **skipped by default**.
- Every PR/commit that adds behavior must add or update a test in the same change.
