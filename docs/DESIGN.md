# DESIGN — DNA-Entropy

Architecture and contracts. If you change an interface here, update this file in the
same change.

## 1. Goals & non-goals

**Goals (demo):**
- CLI, stateless, modular.
- Paste/stdin DNA input → validation → Evo 2 (7B) probabilities → entropy → IGV output.
- Clean swap point for the model (Evo today, ANN later).
- Optional gene-boundary annotation.

**Non-goals (demo):**
- No GUI, no persistence/state, no user accounts.
- No hosted API for inference — Evo runs locally on a GPU box (cloud over SSH for the demo).
- No eukaryotic gene prediction (Pyrodigal is prokaryotic).
- No reference-genome mapping (sequence is treated as its own contig).

## 2. Pipeline

```
                 ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐
  raw text  ───▶ │  reader  │─▶ │ validator  │─▶ │ predictor │─▶ │ analysis │─▶ │ writer │ ──▶ files
                 └──────────┘   └────────────┘   └───────────┘   └──────────┘   └────────┘
                                                       │
                                                 (optional)
                                                 ┌───────────┐
                                                 │ annotator │ ──▶ GFF3
                                                 └───────────┘
```

One direction, no back-edges. Each stage is replaceable without touching the others.
`pipeline.py` wires them together; `cli.py` only parses args and calls the pipeline.

## 3. Module layout

```
DNA-Entropy/
├── CLAUDE.md
├── README.md
├── pyproject.toml                 # core deps light; torch/evo2/pyrodigal are extras
├── docs/{DESIGN,EVO_SETUP,ROADMAP}.md
├── src/dna_entropy/
│   ├── __init__.py
│   ├── cli.py                     # typer entrypoint, arg parsing only
│   ├── config.py                  # dataclass run-config (no third-party dep)
│   ├── pipeline.py                # orchestrates the stages
│   ├── readers/                   # detect.py, paste.py, fasta.py, genbank.py, input.py
│   ├── validation/                # validators.py — normalize + check (N-tolerant mode)
│   ├── predictors/                # base.py (Predictor + (L,4) contract), mock.py, evo.py
│   ├── analysis/                  # entropy.py
│   ├── annotators/                # base.py (Annotator), prodigal.py  (optional)
│   ├── writers/                   # base.py (Writer), bedgraph.py, wig.py, fasta.py, gff.py, genbank.py
│   └── cloud/                     # gcloud.py, orchestrator.py, keeper.py, ui.py
├── keep_gpu.py                    # standalone always-on GPU keeper
└── tests/                         # mirrors src/, plus data/ and conftest.py
```

## 4. Core contracts (the heart of the modularity)

### Predictor

```python
from typing import Protocol
import numpy as np

NUCLEOTIDES = ("A", "C", "G", "T")   # column order is FIXED

class Predictor(Protocol):
    def predict(self, seq: str) -> np.ndarray:
        """Return per-position nucleotide probabilities.

        Args:
            seq: validated uppercase A/C/G/T string of length L.
        Returns:
            np.ndarray, shape (L, 4), dtype float32, columns [A, C, G, T],
            each row summing to 1.0.
        """
```

Implementations:
- **`MockPredictor`** — seeded random / parameterizable distributions. No GPU. Used for
  all non-GPU dev and tests. Same output contract as Evo.
- **`EvoPredictor`** — wraps Evo 2 (7B). **The only place** `torch`/`evo2` may be imported.
- *(future)* **`AnnPredictor`** — the trained model that replaces Evo.

### Position semantics (read carefully — Evo sprint)

Evo predicts the *next* token. Feeding `[BOS, s₁, …, s_L]`, the logits at the BOS slot
predict `s₁`, logits at `s₁` predict `s₂`, etc. So **row `i` of the returned array is the
model's predicted distribution for position `i+1` (1-indexed) given positions `1..i`**.
We align outputs so row `i` corresponds to the entropy *of* base `i`:

- Row 0 (base 1) is predicted from BOS/no context → entropy may be high/less meaningful.
  Document it; do not special-case unless asked.
- Confirm whether Evo's tokenizer adds BOS; adjust the off-by-one alignment accordingly.
  `MockPredictor` simply returns `L` rows directly.

### Analysis

```python
def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """probs: (L, 4) → entropy (L,) in bits, each value in [0.0, 2.0].
    H = -Σ p·log2(p), with 0·log2(0) := 0."""
```

### Writer / Annotator

```python
class Writer(Protocol):
    def write(self, *, name: str, values: np.ndarray, seq: str, start: int, out_dir: str) -> str: ...

class Annotator(Protocol):
    def annotate(self, seq: str) -> list[GeneFeature]: ...   # → GFF3 via gff writer
```

## 5. Validation rules

Applied in `validation/validators.py`, **before** any predictor runs.

**Normalize:** strip surrounding whitespace; remove internal whitespace, newlines, tabs,
and digits (handles pasted line numbers/spacing); uppercase.

**Checks (fail fast, clear message):**
- **Alphabet:** only `A C G T` after normalization. On failure, report the **first
  offending character and its position**, plus a count of bad characters.
- **RNA:** if `U` present → error suggesting it's RNA; `--rna` converts `U→T` then proceeds.
- **Ambiguity codes** (`N R Y …`): rejected for the demo (future: handle `N`).
- **Empty / too short:** reject empty; warn below a small minimum (e.g. < 10 nt).
- **Length vs context:** there is a **configurable context cap** (`--max-len`, default
  **8192 nt**). Evo 2 7B supports much longer contexts in principle, but a single pass is
  bounded by GPU VRAM (≈24 GB on an L4/A10G), so we cap conservatively and window beyond it
  (future work). Over the cap → error. (Mock predictor has no limit.)
- **Header lines:** a single leading `>` FASTA-style header line in pasted text is
  stripped with a notice (full FASTA parsing is a future reader).

## 6. CLI spec

`typer` app, primary command `run`:

```
dna-entropy run [OPTIONS]

Input
  -i, --input PATH         Read sequence from file (default: stdin/paste).
      --name TEXT          Output base name + contig id. PROMPTS if omitted
                           (pass explicitly when piping the sequence via stdin).
                           Sanitized to [A-Za-z0-9._-] for safe folder/file/contig names.
      --rna                Convert U→T before processing.

Model
      --predictor [mock|evo]   Predictor backend (default: mock).
      --model TEXT             Evo model id (default: evo2_7b).
      --device TEXT            cuda|cpu (default: cuda for evo).

Output
  -o, --out DIR            Base folder for outputs (default: the user's Downloads
                           folder). Each run writes into a subfolder <out>/<name>/.
      --format [bedgraph|wig]  Entropy track format (default: bedgraph).
      --start INT          Genomic start coordinate for the track (default: 1).
      --genes/--no-genes   Run gene annotation (default: --no-genes).
```

`run` also takes `--informat genbank|fasta|paste` to override auto-detection.

Secondary commands: `dna-entropy validate -i FILE` (validate only), `dna-entropy version`,
`dna-entropy cloudrun` (run Evo on the keeper's always-on GPU box), and
`dna-entropy keep-gpu` (the keeper itself; also available as the standalone `keep_gpu.py`).
The app never creates or deletes a VM — the keeper provisions one and keeps it running;
the app connects, runs, and leaves it up (delete it manually when done).

**Input routing** — `readers/detect.py` picks the reader by extension (then a content
sniff): `.gb/.gbk/.genbank` → GenBank, `.fa/.fasta/.fna` → FASTA, else paste/stdin.
`readers/input.py` returns a `LoadedInput(seq, notices, features, source_kind)`. GenBank
and FASTA use the **N-tolerant** validation mode (`allow_ambiguity=True`); paste stays
strict A/C/G/T.

**Outputs** — a per-run folder `<out>/<name>/` (default `~/Downloads/<name>/`); the set
depends on the input kind:

- **GenBank input** → `<name>.gb` (its **existing** genes, each with a
  `/note="mean_entropy=… bits"`), `<name>.entropy.wig` (full-res graph), `stats.txt`.
  Prodigal is **not** run — the input's gene annotations are authoritative.
- **FASTA / pasted input** → `<name>.fasta`, `<name>.entropy.bedgraph|wig`,
  `<name>.summary.txt`, `<name>.genes.gff3` (if `--genes`), plus a bonus `<name>.gb`
  (genes from Prodigal when available, else sequence + entropy only).

GenBank has no per-base numeric channel, so the full-resolution entropy graph is always a
companion WIG/bedGraph track; the GenBank carries only the coarse per-gene mean.

## 7. IGV output

A pasted sequence has no genome coordinates, so we make it **self-contained**:

- **`<name>.fasta`** — the sequence as a single contig named `<name>`. Load via
  *Genomes → Load Genome from File*.
- **`<name>.entropy.bedgraph`** — track whose `chrom` equals `<name>`, so coordinates
  line up. bedGraph is plain text, 0-based half-open: `chrom  start  end  value`.
  WIG (fixedStep, 1-based) is an alternate writer.
- **`<name>.genes.gff3`** — optional gene features on the same contig.

`--start` lets the track be offset to a real genomic coordinate later (mapping onto an
existing reference in IGV is future work).

## 8. Testing

**Install & run:**
```bash
pip install -e ".[dev]"
pytest -m "not gpu"     # laptop / CI — no GPU, no Evo
pytest                  # GPU box only — includes Evo integration tests
pytest tests/test_entropy.py        # one file
pytest -k "entropy or writer"       # by keyword
```

**Layout & conventions:**
- `tests/` mirrors `src/dna_entropy/`; files `test_*.py`, functions `test_*`.
- Fixtures in `tests/conftest.py`; sample sequences in `tests/data/`.
- GPU/Evo tests: decorate with `@pytest.mark.gpu` (registered in `pyproject.toml`);
  skipped unless run on a CUDA machine.

**Adding a test (required with every behavior change):**
1. Put it in the file mirroring the module you changed (create it if absent).
2. Use a fixture or `tests/data/` sample, not inline mega-strings.
3. Assert the **contract**, not the implementation (shapes, sums, ranges, file format).
4. Make it deterministic (seed the mock; no network/GPU in `not gpu` tests).

**Must-have test cases per area:**
- *Validation:* valid seq passes; lowercase/whitespace/newlines normalized; non-ACGT
  reports correct first-bad position; RNA detected; over-context length rejected.
- *Entropy:* uniform `[.25,.25,.25,.25] → 2.0`; one-hot `→ 0.0`; output range `[0,2]`;
  `0·log0` handled.
- *Predictor contract:* mock output is `(L,4)`, float32, rows sum to 1, deterministic.
- *Writers:* bedGraph/WIG/FASTA/GFF3 produce spec-correct, IGV-loadable text.
- *Pipeline:* end-to-end on mock yields all expected output files.

## 9. Future / swap points (don't build now)

- `AnnPredictor` replaces `EvoPredictor` behind the same `(L,4)` contract.
- FASTA/GenBank readers; IUPAC/`N` handling; long-locus windowing; reverse strand;
  bigWig; reference-genome mapping; GUI. Tracked in [ROADMAP.md](ROADMAP.md).
