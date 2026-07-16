"""FASTA writer — emits the sequence as its own contig so IGV needs no reference.

Load this via 'Genomes -> Load Genome from File' in IGV; the entropy track's ``chrom``
matches this contig's name, so they align.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import write_text_lf

LINE_WIDTH = 60


def _record(chrom: str, seq: str) -> str:
    wrapped = [seq[i : i + LINE_WIDTH] for i in range(0, len(seq), LINE_WIDTH)]
    return f">{chrom}\n" + "\n".join(wrapped)


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
        text = _record(name, seq) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.fasta", text)

    def write_multi(
        self,
        *,
        name: str,
        blocks: Sequence[tuple[str, str]],
        out_dir: str,
    ) -> str:
        """Write one FASTA with a ``>chrom`` record per ``(chrom, seq)`` in ``blocks``.

        The contig names match the entropy track's ``chrom`` values, so IGV lines them up.
        """
        text = "\n".join(_record(chrom, seq) for chrom, seq in blocks) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.fasta", text)
