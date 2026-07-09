"""FASTA writer — emits the sequence as its own contig so IGV needs no reference.

Load this via 'Genomes -> Load Genome from File' in IGV; the entropy track's ``chrom``
matches this contig's name, so they align.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import write_text_lf

LINE_WIDTH = 60


class FastaWriter:
    """Writes ``<name>.fasta`` (``values``/``start`` are unused)."""

    def write(
        self,
        *,
        name: str,
        values: np.ndarray,
        seq: str,
        start: int,
        out_dir: str,
    ) -> str:
        wrapped = [seq[i : i + LINE_WIDTH] for i in range(0, len(seq), LINE_WIDTH)]
        text = f">{name}\n" + "\n".join(wrapped) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.fasta", text)
