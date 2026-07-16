"""Orchestrates the pipeline stages. The CLI calls these; stages stay decoupled.

  input -> validate -> predict -> analyze -> export

Each stage is swappable; this module is the only place that knows the order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis.entropy import shannon_entropy
from .annotators.base import GeneFeature
from .annotators.prodigal import ProdigalAnnotator
from .config import PredictorKind, RunConfig, TrackFormat
from .predictors.base import Predictor, PredictorError, check_probability_matrix
from .predictors.mock import MockPredictor
from .readers import detect
from .readers.input import LoadedInput, load_input
from .readers.paste import PasteReader
from .validation.validators import ValidatedSequence, validate_sequence
from .writers.base import Writer
from .writers.bedgraph import BedGraphWriter
from .writers.fasta import FastaWriter
from .writers.genbank import GenBankWriter
from .writers.geneious import GeneiousWriter
from .writers.gff import GffWriter
from .writers.summary import SummaryWriter
from .writers.wig import WigWriter


@dataclass
class RunResult:
    """Outcome of a full run: the clean sequence, entropy track, and written files.

    ``seq``/``values`` refer to the first contig (single-sequence inputs have exactly one).
    ``all_values`` concatenates every contig's entropy for aggregate reporting; ``contigs``
    and ``total_nt`` describe multi-record GenBank runs.
    """

    seq: str
    values: np.ndarray
    notices: list[str]
    outputs: list[str]
    genes: list[GeneFeature] = field(default_factory=list)
    contigs: int = 1
    total_nt: int = 0
    all_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.all_values is None:
            self.all_values = self.values
        if not self.total_nt:
            self.total_nt = len(self.seq)


def read_raw(cfg: RunConfig) -> str:
    """Read raw text from the configured input (file or stdin)."""
    return PasteReader(cfg.input_path).read()


def load_and_validate(cfg: RunConfig, raw: str | None = None) -> ValidatedSequence:
    """Read (unless ``raw`` is supplied) and validate into a clean sequence."""
    if raw is None:
        raw = read_raw(cfg)
    return validate_sequence(raw, max_len=cfg.max_len, rna=cfg.rna)


def build_predictor(cfg: RunConfig) -> Predictor:
    """Construct the predictor backend named by the config."""
    if cfg.predictor is PredictorKind.MOCK:
        return MockPredictor(seed=cfg.seed)
    if cfg.predictor is PredictorKind.EVO:
        # Lazy import: the Evo stack is heavy and GPU-only, so it must not be imported
        # on the mock path (CLAUDE.md hard rule #5).
        try:
            from .predictors.evo import EvoPredictor
        except ImportError as exc:  # not built / deps absent
            raise PredictorError(
                "Evo predictor is not available yet (arrives in Sprint 3, and needs "
                "the [evo] extra on a GPU box). Use --predictor mock for now."
            ) from exc
        return EvoPredictor(model=cfg.model, device=cfg.device)
    raise PredictorError(f"Unknown predictor: {cfg.predictor!r}")


def _select_track_writer(cfg: RunConfig) -> Writer:
    return WigWriter() if cfg.track_format is TrackFormat.WIG else BedGraphWriter()


def _try_annotate(seq: str) -> list[GeneFeature]:
    """Best-effort Prodigal call for a bonus GenBank; never raises (returns [] on failure)."""
    try:
        return ProdigalAnnotator().annotate(seq)
    except Exception:  # missing [genes] extra, or Prodigal failure — GenBank genes are optional
        return []


def _write_genbank_outputs(cfg: RunConfig, processed: list[tuple]) -> list[str]:
    """GenBank input -> ONE GenBank (all records, genes preserved + entropy notes) + ONE
    WIG track (a block per record) + ONE stats.txt (per-record + overall).

    ``processed`` is a list of ``(Contig, values)``.
    """
    gb = GenBankWriter().write_multi(
        name=cfg.name,
        records=[(c.name, c.seq, c.features, values, c.source_id) for c, values in processed],
        out_dir=cfg.out_dir,
    )
    wig = WigWriter().write_multi(
        name=cfg.name,
        blocks=[(c.name, values) for c, values in processed],
        start=cfg.start,
        out_dir=cfg.out_dir,
    )
    geneious = GeneiousWriter().write_multi(
        name=cfg.name,
        blocks=[(c.name, values) for c, values in processed],
        start=cfg.start,
        out_dir=cfg.out_dir,
    )
    stats = SummaryWriter().write_multi(
        name=cfg.name,
        sections=[(c.name, values) for c, values in processed],
        start=cfg.start,
        out_dir=cfg.out_dir,
        filename="stats.txt",
    )
    return [gb, wig, geneious, stats]


def _write_standard_outputs(
    cfg: RunConfig, values: np.ndarray, seq: str
) -> tuple[list[str], list[GeneFeature]]:
    """FASTA/paste input -> the existing files, plus a bonus GenBank when possible."""
    writers: list[Writer] = [
        FastaWriter(),
        _select_track_writer(cfg),
        GeneiousWriter(),
        SummaryWriter(),
    ]
    outputs = [
        w.write(name=cfg.name, values=values, seq=seq, start=cfg.start, out_dir=cfg.out_dir)
        for w in writers
    ]

    genes: list[GeneFeature] = []
    if cfg.genes:
        genes = ProdigalAnnotator().annotate(seq)  # may raise AnnotatorError (explicit opt-in)
        outputs.append(
            GffWriter().write(
                name=cfg.name, features=genes, length=len(seq),
                start=cfg.start, out_dir=cfg.out_dir,
            )
        )

    # Bonus GenBank "if possible": reuse --genes features, else best-effort Prodigal.
    gb_features = genes or _try_annotate(seq)
    try:
        outputs.append(
            GenBankWriter().write(
                name=cfg.name, values=values, seq=seq, start=cfg.start,
                features=gb_features, out_dir=cfg.out_dir,
            )
        )
    except Exception:
        pass  # GenBank is a bonus on this path; never fail the core run over it
    return outputs, genes


def run(cfg: RunConfig, raw: str | None = None) -> RunResult:
    """Run the full pipeline and write all output files (output set depends on input kind).

    Every contig gets one Evo forward pass (GenBank inputs may carry several records).
    """
    loaded = load_input(cfg, raw)
    predictor = build_predictor(cfg)

    processed: list[tuple] = []  # (Contig, entropy values)
    for contig in loaded.contigs:
        probs = predictor.predict(contig.seq)
        check_probability_matrix(probs, len(contig.seq))  # guard the predictor boundary
        processed.append((contig, shannon_entropy(probs)))

    if loaded.source_kind == detect.GENBANK:
        outputs = _write_genbank_outputs(cfg, processed)
        genes = [f for c, _ in processed for f in c.features]
    else:
        contig, values = processed[0]
        outputs, genes = _write_standard_outputs(cfg, values, contig.seq)

    first_contig, first_values = processed[0]
    all_values = np.concatenate([v for _, v in processed])
    return RunResult(
        seq=first_contig.seq,
        values=first_values,
        notices=loaded.notices,
        outputs=outputs,
        genes=genes,
        contigs=len(processed),
        total_nt=sum(len(c.seq) for c, _ in processed),
        all_values=all_values,
    )
