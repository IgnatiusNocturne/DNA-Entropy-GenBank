"""WIG (fixedStep) writer for the per-position entropy track.

fixedStep WIG is 1-based and very compact (one value per line). Alternate to bedGraph.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import write_text_lf


def _fixed_step_block(chrom: str, values: np.ndarray, start: int) -> list[str]:
    lines = [f"fixedStep chrom={chrom} start={start} step=1 span=1"]
    lines.extend(f"{float(v):.4f}" for v in values)
    return lines


class WigWriter:
    """Writes ``<name>.entropy.wig`` in fixedStep format (one block per contig)."""

    def write(
        self,
        *,
        name: str,
        values: np.ndarray,
        seq: str,
        start: int,
        out_dir: str,
    ) -> str:
        return self.write_multi(name=name, blocks=[(name, values)], start=start, out_dir=out_dir)

    def write_multi(
        self,
        *,
        name: str,
        blocks: Sequence[tuple[str, np.ndarray]],
        start: int,
        out_dir: str,
    ) -> str:
        """Write one WIG file with a fixedStep block per ``(chrom, values)`` in ``blocks``."""
        lines = [
            f'track type=wiggle_0 name="{name} entropy" '
            'description="Shannon entropy (bits)" visibility=full'
        ]
        for chrom, values in blocks:
            lines.extend(_fixed_step_block(chrom, values, start))
        text = "\n".join(lines) + "\n"
        return write_text_lf(Path(out_dir) / f"{name}.entropy.wig", text)
