"""GFF3 writer for gene boundaries (IGV feature track).

Emits features on the same contig(s) as the entropy track, offset by the same ``start``
coordinate, so genes and entropy line up in IGV. Two sources feed this:

- **Prodigal** (FASTA/paste path): predicted CDS emitted as ``gene`` features via ``write``.
- **GenBank** (GenBank path): the record's **existing** genes via ``write_multi`` — one
  block per record, keyed by contig name — so IGV shows the curated genes, not Prodigal's.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..annotators.base import GeneFeature
from .base import write_text_lf

# GFF3 column-9 reserved characters that must be percent-encoded in attribute values.
_GFF3_RESERVED = {";": "%3B", "=": "%3D", "&": "%26", ",": "%2C", "\t": "%09", "\n": "%0A"}


def _escape(value: str) -> str:
    for ch, code in _GFF3_RESERVED.items():
        value = value.replace(ch, code)
    return value


def _feature_line(chrom: str, source: str, f: GeneFeature, offset: int) -> str:
    gid = _escape(f.gene_id)
    attrs = f"ID={gid};Name={gid}"
    if f.partial:
        attrs += ";partial=true"
    return "\t".join(
        [
            chrom,
            source,
            "gene",
            str(f.begin + offset),
            str(f.end + offset),
            ".",
            f.strand,
            ".",
            attrs,
        ]
    )


class GffWriter:
    """Writes ``<name>.genes.gff3``."""

    def write(
        self,
        *,
        name: str,
        features: Sequence[GeneFeature],
        length: int,
        start: int,
        out_dir: str,
        source: str = "pyrodigal",
    ) -> str:
        return self.write_multi(
            name=name,
            blocks=[(name, features, length)],
            start=start,
            out_dir=out_dir,
            source=source,
        )

    def write_multi(
        self,
        *,
        name: str,
        blocks: Sequence[tuple[str, Sequence[GeneFeature], int]],
        start: int,
        out_dir: str,
        source: str = "genbank",
    ) -> str:
        """Write one GFF3 with a ``(chrom, features, length)`` block per record in ``blocks``.

        ``offset`` keeps genes aligned with the entropy track's coordinates; each block's
        ``chrom`` matches its FASTA contig so IGV places the genes on the right sequence.
        """
        offset = start - 1
        lines = ["##gff-version 3"]
        for chrom, _features, length in blocks:
            lines.append(f"##sequence-region {chrom} {start} {start + length - 1}")
        for chrom, features, _length in blocks:
            for f in features:
                lines.append(_feature_line(chrom, source, f, offset))
        text = "\n".join(lines) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.genes.gff3", text)
