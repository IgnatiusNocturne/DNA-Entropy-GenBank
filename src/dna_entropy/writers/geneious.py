"""GFF3 entropy-track writer for Geneious Prime.

Geneious Prime does **not** import WIG or bedGraph as graph tracks (in Geneious those
are *export*-only from the Graphs tab). It *does* import GFF3, and it can shade an
annotation track by a numeric qualifier via *Color by / Heatmap*. So we emit the
per-position entropy as a GFF3 feature track: one 1 bp feature per position carrying the
entropy in both the score column and an ``entropy`` qualifier, on the same contig
(``name``) and coordinate frame as the FASTA/GenBank we also write, so it lines up.

See docs/DISTRIBUTION.md / README.md for the load-into-Geneious instructions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import write_text_lf


def _feature_lines(chrom: str, values: np.ndarray, start: int) -> list[str]:
    offset = start - 1  # 1-based genomic coord of position i (0-based) is (start + i)
    lines = []
    for i, v in enumerate(values):
        pos = offset + i + 1  # GFF3 is 1-based, inclusive
        val = f"{float(v):.4f}"
        lines.append(
            "\t".join(
                [
                    chrom,
                    "dna-entropy",
                    "entropy",
                    str(pos),
                    str(pos),
                    val,  # score column: Geneious can Color by / Heatmap on this
                    ".",
                    ".",
                    f"Name=H={val};entropy={val}",
                ]
            )
        )
    return lines


class GeneiousWriter:
    """Writes ``<name>.entropy.geneious.gff3`` (a per-position entropy heatmap track)."""

    def write(
        self,
        *,
        name: str,
        values: np.ndarray,
        seq: str,
        start: int,
        out_dir: str,
    ) -> str:
        return self.write_multi(
            name=name, blocks=[(name, values)], start=start, out_dir=out_dir
        )

    def write_multi(
        self,
        *,
        name: str,
        blocks: Sequence[tuple[str, np.ndarray]],
        start: int,
        out_dir: str,
    ) -> str:
        """Write one GFF3 with a per-position block per ``(chrom, values)`` in ``blocks``."""
        lines = ["##gff-version 3"]
        for chrom, values in blocks:
            end = start + len(values) - 1
            lines.append(f"##sequence-region {chrom} {start} {end}")
        for chrom, values in blocks:
            lines.extend(_feature_lines(chrom, values, start))
        text = "\n".join(lines) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.entropy.geneious.gff3", text)
