# ROADMAP — DNA-Entropy

Sprint plan with tickable checklists. **Tick boxes (`[ ]` → `[x]`) as work completes.**
Do not start a sprint until the previous sprint's boxes are ticked and its tests pass
(CLAUDE.md hard rule #6).

**Definition of Done (every sprint):** code + type hints + docstrings, tests added,
`pytest -m "not gpu"` green on the laptop, docs updated if an interface changed.

Legend: ⭐ = critical path · 🧪 = has required tests · 🔁 = swap point for the future ANN.

---

## Sprint 0 — Scaffold & contracts ⭐

Goal: a runnable skeleton with the core interfaces and the mock predictor, no real logic.

- [x] Create `pyproject.toml` — core deps light (`numpy`, `typer`);
      extras `[dev]` (`pytest`), `[evo]` (`torch`, `evo2`), `[genes]` (`pyrodigal`).
- [x] Package skeleton under `src/dna_entropy/` (see DESIGN.md §3).
- [x] `predictors/base.py` — `Predictor` Protocol + the `(L, 4)` contract docstring +
      `check_probability_matrix` guard. 🔁
- [x] `predictors/mock.py` — `MockPredictor` (seeded; sequence-dependent via CRC32).
- [x] `config.py` — dataclass run-config object (no third-party dep).
- [x] `cli.py` — `typer` app with `run`, `validate`, `version` (`version` live; others stub).
- [x] Register pytest markers (`gpu`) in `pyproject.toml`; add `tests/conftest.py`.
- [x] 🧪 `tests/test_mock_predictor.py` — output is `(L,4)`, float32, rows sum to 1,
      deterministic for a fixed seed (12 tests, all passing).

## Sprint 1 — Input & validation ⭐

Goal: turn pasted text into a clean, validated A/C/G/T string with great error messages.

- [x] `readers/base.py` — `Reader` Protocol.
- [x] `readers/paste.py` — read from stdin/file.
- [x] `validation/validators.py` — normalize (strip whitespace/newlines/digits, uppercase);
      checks per DESIGN.md §5.
- [x] Clear errors: first offending char + position + count; RNA (`U`) detection +
      `--rna` convert; over-`--max-len` rejection; leading `>` header strip-with-notice.
- [x] Wire reader+validation into `pipeline.py`; `validate` CLI command is live.
- [x] 🧪 `tests/test_validation.py` + `tests/test_readers.py` — valid passes;
      lowercase/whitespace/newlines/digits normalized; non-ACGT reports correct position
      & count; ambiguity hint; RNA detected/converted; too-long rejected; empty rejected
      (32 tests total, all passing).

## Sprint 2 — Analysis & export (end-to-end on mock) ⭐

Goal: a fully working tool on mock data — the demo skeleton the professor could click.

- [x] `analysis/entropy.py` — `shannon_entropy((L,4)) -> (L,)`, values in `[0,2]`,
      `0·log0 := 0`; plus `summarize()` stats.
- [x] `writers/base.py` — `Writer` Protocol + LF/UTF-8 write helper.
- [x] `writers/bedgraph.py` (default), `writers/wig.py` (fixedStep).
- [x] `writers/fasta.py` — sequence as a single contig named `<name>`.
- [x] `<name>.summary.txt` — length, mean/min/max entropy + extrema positions.
- [x] Complete `pipeline.py` + `cli.py run` so the full chain runs on `--predictor mock`.
- [x] 🧪 `tests/test_entropy.py` — uniform → 2.0; one-hot → 0.0; range `[0,2]`.
- [x] 🧪 `tests/test_writers.py` — bedGraph/WIG/FASTA produce spec-correct text.
- [x] 🧪 `tests/test_pipeline.py` — end-to-end on mock writes all expected files
      (50 tests total, all passing; output files verified well-formed).
- [ ] Manual: load `demo.fasta` + `demo.entropy.bedgraph` in IGV; confirm track renders.
      *(Needs IGV on your machine — files generated & format-verified; see note below.)*

## Sprint 3 — Real Evo predictor ⭐ 🔁

Goal: swap the mock for Evo 2 7B with zero downstream changes. Runs on the cloud GPU box
(24 GB: GCP L4 / AWS A10G — see EVO_SETUP.md). You set up the SSH box at the start of this sprint.

- [x] `predictors/evo.py` — `EvoPredictor` wrapping Evo 2 7B. **Only Evo-aware module.**
- [x] Logit → `(L,4)`: select A/C/G/T token logits (ids derived from the tokenizer),
      softmax over the four (== renormalize). Pure helper in `predictors/logits.py`.
- [x] Next-token position alignment (DESIGN.md §4); robust `_extract_logits` for evo2
      return-shape variations; tokenizer-derived nucleotide ids (no hard-coded ASCII).
- [x] `--max-len` context guard; `--device` config (bf16 handled by the Evo2 loader).
- [x] 🧪 `tests/test_evo_logits.py` — torch-free alignment math (softmax, position shift,
      uniform row 0, single-base): runs on the laptop (5 tests, passing).
- [x] 🧪 `tests/test_evo_predictor.py` marked `gpu` — contract holds on GPU; skips
      cleanly on the laptop (verified via `pytest -rs`).
- [x] Ran `pytest -m gpu` on a GCP L4 (g2-standard-8): **3 passed**. Confirmed the `(L,4)`
      contract, drop-in swap with mock, and that the tokenizer adds **no BOS** (L == len(seq)).
      Two real-model fixes the contract guard caught: unwrap evo2's nested return tuple, and
      cast uint8 tokenizer ids to int. Position 1 is uniform (2.0 bits) as designed.
- [x] Validated the real pipeline on 5 TnpB loci (`--predictor evo`): meaningful entropy
      (mean ~1.2-1.4 bits, conserved bases near 0). EVO_SETUP.md updated with the verified
      install (system Python + preinstalled torch; flash-attn built from source for cu12.9).

## Sprint 4 — Gene boundaries (optional) + polish

Goal: nice-to-haves and a clean demo. Gene calling is off by default and must never
block the core flow.

- [x] `annotators/base.py` — `Annotator` Protocol + `GeneFeature` + `AnnotatorError`.
- [x] `annotators/prodigal.py` — Pyrodigal gene calling (meta mode; prokaryotic caveat
      documented; lazy import of the `[genes]` extra).
- [x] `writers/gff.py` — GFF3 on the same contig as the entropy track, start-offset aware.
- [x] Wire `--genes` flag (default off) through pipeline + CLI; clean `AnnotatorError` UX.
- [x] 🧪 `tests/test_gff.py` (synthetic features) + `tests/test_annotator.py` (real
      Pyrodigal) — 61 tests total, all passing.
- [x] Polish: friendly error UX (Validation/Predictor/Annotator -> red `ERROR:`),
      `tests/data/` sample loci (generic + a prokaryotic ORF), `scripts/demo.ps1`.
- [x] End-to-end dry run: mock pipeline + `--genes` verified; GFF3 calls the ORF at
      25–690 on `prok_demo`. (Real Evo dry run still pending the GPU box — Sprint 3.)

## Sprint 5 — GenBank I/O + always-on GPU ⭐

Goal: accept GenBank/FASTA input (GenBank genes preserved, never re-annotated), return a
GenBank result, and split GPU provisioning from analysis into an always-on keeper.

- [x] Standalone GPU **keeper** (`keep_gpu.py` / `dna-entropy keep-gpu`,
      `cloud/keeper.py`): secures a GPU VM and keeps it RUNNING 24/7, retrying forever on
      stockout/quota (never errors out), pre-installs Evo, restarts on maintenance TERMINATE.
- [x] Slimmed the app: `cloudrun` no longer creates or tears down a VM — it finds the
      keeper's box, wakes it if stopped, errors if none exists, and leaves it running.
- [x] `readers/detect.py`, `readers/fasta.py`, `readers/genbank.py`, `readers/input.py` —
      input routing by extension/sniff; GenBank genes -> `GeneFeature` (gene preferred over CDS).
- [x] Lenient `N`/IUPAC validation for GenBank/FASTA (`allow_ambiguity`); paste stays strict.
- [x] `writers/genbank.py` — sequence + genes + per-gene `mean_entropy` note (Biopython, core dep).
- [x] `pipeline.py` branches on `source_kind`: GenBank -> `.gb` + `.entropy.wig` + `stats.txt`;
      FASTA/paste -> existing IGV files + bonus `.gb`. Cloud path uploads the original file.
- [x] 🧪 `tests/test_keeper.py` + `tests/test_genbank.py` (30 new tests; `pytest -m "not gpu"` green).

---

## Backlog (future project — do NOT build during the demo)

- [x] FASTA reader · [x] GenBank reader
- [x] IUPAC ambiguity / `N` handling (tolerant on the GenBank/FASTA path)
- [ ] Long-locus windowing (overlap-tile beyond the context cap)
- [ ] Reverse-strand / both-strand entropy
- [ ] Map track onto a reference genome (`chrom` + `--start`) instead of self-contained
- [ ] bigWig output (needs `bedGraphToBigWig` + chrom.sizes)
- [ ] 🔁 Replace Evo with the trained ANN (Evo embeddings + BLAST → ANN) behind the same
      `Predictor` contract
- [ ] GUI
