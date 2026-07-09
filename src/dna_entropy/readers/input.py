"""Unified input loading: route a file (or paste) to the right reader + validation.

Produces a :class:`LoadedInput` — one or more :class:`Contig`s to analyze, plus any
non-fatal notices and the source kind (which the pipeline uses to pick the output set).

A GenBank may hold multiple records; we return **all** of them as separate contigs so the
pipeline computes entropy for each. FASTA/paste yield a single contig.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..annotators.base import GeneFeature
from ..config import RunConfig
from ..validation.validators import validate_sequence
from . import detect
from .fasta import read_fasta
from .genbank import read_genbank
from .paste import PasteReader


@dataclass
class Contig:
    """One sequence to analyze: an output-safe name, the bases, and any known genes."""

    name: str                                   # safe for a filename / IGV chrom / GenBank LOCUS
    seq: str
    features: list[GeneFeature] = field(default_factory=list)
    source_id: str = ""                         # original record id (kept for provenance)


@dataclass
class LoadedInput:
    """One or more validated contigs plus input-derived context for the pipeline."""

    contigs: list[Contig]
    notices: list[str] = field(default_factory=list)
    source_kind: str = detect.PASTE

    @property
    def seq(self) -> str:
        """First contig's sequence (convenience for single-contig callers)."""
        return self.contigs[0].seq

    @property
    def features(self) -> list[GeneFeature]:
        """First contig's features (convenience for single-contig callers)."""
        return self.contigs[0].features


def _safe_contig_name(base: str, index: int, total: int) -> str:
    """Make an output-safe contig name (filenames / IGV chrom / GenBank LOCUS).

    Single record -> just ``base``; multiple -> ``base_1``, ``base_2``, ... . Geneious-style
    ids (``geneious|urn:local:...``) are unsafe, so we index off the run name instead.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._-") or "seq"
    return safe if total == 1 else f"{safe}_{index + 1}"


def load_input(cfg: RunConfig, raw: str | None = None) -> LoadedInput:
    """Read + validate the configured input into a :class:`LoadedInput`.

    GenBank/FASTA files are read from ``cfg.input_path``; ``raw`` (if given) is used only
    for the paste path. GenBank/FASTA tolerate IUPAC ambiguity codes (real files contain
    ``N``); the paste path stays strict A/C/G/T.
    """
    kind = cfg.informat or detect.detect_kind(cfg.input_path)

    if kind == detect.GENBANK:
        records, notices = read_genbank(cfg.input_path)
        contigs: list[Contig] = []
        for i, rec in enumerate(records):
            v = validate_sequence(rec.seq, max_len=cfg.max_len, rna=cfg.rna, allow_ambiguity=True)
            notices += v.notices
            contigs.append(
                Contig(
                    name=_safe_contig_name(cfg.name, i, len(records)),
                    seq=v.seq,
                    features=rec.features,
                    source_id=rec.record_id,
                )
            )
        return LoadedInput(contigs=contigs, notices=notices, source_kind=kind)

    if kind == detect.FASTA:
        raw_seq, notices = read_fasta(cfg.input_path)
        v = validate_sequence(raw_seq, max_len=cfg.max_len, rna=cfg.rna, allow_ambiguity=True)
        contig = Contig(name=_safe_contig_name(cfg.name, 0, 1), seq=v.seq)
        return LoadedInput(contigs=[contig], notices=notices + v.notices, source_kind=kind)

    text = raw if raw is not None else PasteReader(cfg.input_path).read()
    v = validate_sequence(text, max_len=cfg.max_len, rna=cfg.rna)
    contig = Contig(name=_safe_contig_name(cfg.name, 0, 1), seq=v.seq)
    return LoadedInput(contigs=[contig], notices=v.notices, source_kind=detect.PASTE)
