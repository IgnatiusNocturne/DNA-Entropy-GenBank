"""Read a GenBank file into its records: sequence + EXISTING gene features, per record.

Key rule (professor's brief): when the input is a GenBank, we **use its genes as-is** and
never re-annotate. A GenBank may hold **multiple records** (the ``SetTnpB-Evo.gb`` test file
has five); we return **all** of them so the pipeline processes each. This reader extracts
each record's sequence and maps its ``gene`` (or, failing that, ``CDS``) features onto our
:class:`GeneFeature` type. Biopython does the parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..annotators.base import GeneFeature


class GenBankReadError(ValueError):
    """Raised when a GenBank file has no usable sequence record."""


@dataclass
class GenBankRecord:
    """One record from a GenBank file: its id, sequence, and pre-existing gene features."""

    record_id: str
    seq: str
    features: list[GeneFeature] = field(default_factory=list)


def _feature_id(feature) -> str:
    """Pick a stable label for a feature from its qualifiers."""
    for key in ("gene", "locus_tag", "product"):
        vals = feature.qualifiers.get(key)
        if vals:
            return str(vals[0])
    return feature.type


def _features_of(record) -> list[GeneFeature]:
    """Map a record's ``gene`` (preferred) or ``CDS`` features to :class:`GeneFeature`."""
    gene_feats = [f for f in record.features if f.type == "gene"]
    cds_feats = [f for f in record.features if f.type == "CDS"]
    source = gene_feats or cds_feats

    features: list[GeneFeature] = []
    for f in source:
        loc = f.location
        if loc is None:
            continue
        begin = int(loc.start) + 1  # Biopython is 0-based half-open -> our 1-based inclusive
        end = int(loc.end)
        strand = "-" if loc.strand == -1 else "+"
        partial = "<" in str(loc.start) or ">" in str(loc.end)
        features.append(
            GeneFeature(begin=begin, end=end, strand=strand, partial=partial, gene_id=_feature_id(f))
        )
    return features


def read_genbank(path: str) -> tuple[list[GenBankRecord], list[str]]:
    """Return ``(records, notices)`` for **every** record in ``path``.

    Features are 1-based inclusive, sequence-relative (matching :class:`GeneFeature`).
    Records whose ORIGIN block is empty (no nucleotides) are skipped with a notice.
    """
    from Bio import SeqIO  # local import keeps import cost off unrelated paths

    parsed = list(SeqIO.parse(path, "genbank"))
    if not parsed:
        raise GenBankReadError("No GenBank records found in the file.")

    notices: list[str] = []
    records: list[GenBankRecord] = []
    total_features = 0
    for rec in parsed:
        seq = str(rec.seq)
        if not seq or set(seq.upper()) <= {"N"}:
            notices.append(f"Skipped record {rec.id!r}: no nucleotide sequence.")
            continue
        feats = _features_of(rec)
        total_features += len(feats)
        records.append(GenBankRecord(record_id=rec.id, seq=seq, features=feats))

    if not records:
        raise GenBankReadError("The GenBank file has no records with a nucleotide sequence.")

    kind = "gene" if any(r.features for r in records) else "no gene"
    notices.append(
        f"Read {len(records)} record(s) with {total_features} {kind} feature(s) from the "
        "GenBank (not re-annotated)."
    )
    return records, notices
