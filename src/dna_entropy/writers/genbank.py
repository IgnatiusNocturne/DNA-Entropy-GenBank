"""GenBank writer — sequence + gene features, each annotated with its mean entropy.

Self-contained result for sequence editors (SnapGene / Benchling / Geneious): every gene
carries a ``/note="mean_entropy=... bits"`` qualifier, so the coarse per-gene value shows
right on the feature. The full per-base entropy graph lives in the companion WIG track for
genome browsers (IGV/UCSC/JBrowse) — GenBank has no per-base numeric channel.

Supports a single sequence (``write``) or many (``write_multi``, one GenBank record per
contig in one file) so a multi-record GenBank input round-trips to a multi-record output.

Unlike the uniform :class:`Writer` protocol, this writer also takes ``features``, so the
pipeline calls it directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..annotators.base import GeneFeature


def _build_record(
    contig_name: str, seq: str, features: Sequence[GeneFeature], values: np.ndarray, source_id: str
):
    """Build one Biopython ``SeqRecord`` with per-gene mean-entropy notes."""
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    locus = (contig_name[:16] or "locus")  # GenBank LOCUS ids are short; avoid a warning
    desc = f"DNA-Entropy per-position Shannon entropy for {contig_name}"
    if source_id and source_id != contig_name:
        desc += f" (source: {source_id})"
    record = SeqRecord(Seq(seq), id=locus, name=locus, description=desc)
    record.annotations["molecule_type"] = "DNA"
    record.annotations["source"] = "DNA-Entropy"

    for f in features:
        begin0 = max(f.begin - 1, 0)               # 1-based inclusive -> 0-based half-open
        end = min(f.end, len(seq))
        if end <= begin0:
            continue
        segment = np.asarray(values[begin0:end], dtype=float)
        mean_h = float(segment.mean()) if segment.size else 0.0
        quals: dict[str, list[str]] = {"note": [f"mean_entropy={mean_h:.3f} bits"]}
        if f.gene_id:
            quals["gene"] = [f.gene_id]
        if f.partial:
            quals["note"].append("partial")
        record.features.append(
            SeqFeature(
                FeatureLocation(begin0, end, strand=1 if f.strand == "+" else -1),
                type="gene",
                qualifiers=quals,
            )
        )
    return record


class GenBankWriter:
    """Writes ``<name>.gb`` (GenBank flat file with per-gene entropy notes)."""

    def write(
        self,
        *,
        name: str,
        values: np.ndarray,
        seq: str,
        start: int,
        features: Sequence[GeneFeature],
        out_dir: str,
    ) -> str:
        """Write a single-record GenBank."""
        return self.write_multi(
            name=name, records=[(name, seq, features, values, "")], out_dir=out_dir
        )

    def write_multi(
        self,
        *,
        name: str,
        records: Sequence[tuple[str, str, Sequence[GeneFeature], np.ndarray, str]],
        out_dir: str,
    ) -> str:
        """Write one GenBank file holding one record per contig.

        Each ``records`` item is ``(contig_name, seq, features, values, source_id)``.
        """
        from Bio.SeqIO import write as _seqio_write

        seq_records = [
            _build_record(contig_name, seq, features, values, source_id)
            for contig_name, seq, features, values, source_id in records
        ]
        out_path = Path(out_dir) / f"{name}.gb"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _seqio_write(seq_records, str(out_path), "genbank")
        return str(out_path)
