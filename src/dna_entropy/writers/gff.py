"""GFF3 writer for predicted gene boundaries (IGV feature track).

Emits features on the same contig (``name``) as the entropy track, offset by the same
``start`` coordinate, so genes and entropy line up in IGV. Prodigal predicts CDS; we
emit them as ``gene`` features (no GFF3 phase bookkeeping needed) which render cleanly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..annotators.base import GeneFeature
from .base import write_text_lf


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
    ) -> str:
        offset = start - 1  # keep genes aligned with the entropy track's coordinates
        lines = [
            "##gff-version 3",
            f"##sequence-region {name} {start} {start + length - 1}",
        ]
        for f in features:
            attrs = f"ID={f.gene_id};Name={f.gene_id}"
            if f.partial:
                attrs += ";partial=true"
            lines.append(
                "\t".join(
                    [
                        name,
                        "pyrodigal",
                        "gene",
                        str(f.begin + offset),
                        str(f.end + offset),
                        ".",
                        f.strand,
                        ".",
                        attrs,
                    ]
                )
            )
        text = "\n".join(lines) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.genes.gff3", text)
