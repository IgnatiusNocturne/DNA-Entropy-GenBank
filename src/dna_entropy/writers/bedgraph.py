"""bedGraph writer for the per-position entropy track (IGV default).

bedGraph is plain text, 0-based half-open: ``chrom  start  end  value``. The ``chrom``
equals the contig name we also emit as FASTA, so coordinates line up in IGV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import write_text_lf


class BedGraphWriter:
    """Writes ``<name>.entropy.bedgraph``."""

    def write(
        self,
        *,
        name: str,
        values: np.ndarray,
        seq: str,
        start: int,
        out_dir: str,
    ) -> str:
        lines = [
            f'track type=bedGraph name="{name} entropy" '
            'description="Shannon entropy (bits)" visibility=full'
        ]
        # genomic 1-based coord of base i (0-based) is (start + i);
        # bedGraph is 0-based half-open => [start-1+i, start+i).
        base0 = start - 1
        for i, v in enumerate(values):
            lines.append(f"{name}\t{base0 + i}\t{base0 + i + 1}\t{float(v):.4f}")
        text = "\n".join(lines) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.entropy.bedgraph", text)
