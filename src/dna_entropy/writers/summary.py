"""Plain-text summary of an entropy run."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..analysis.entropy import summarize
from .base import write_text_lf


class SummaryWriter:
    """Writes ``<name>.summary.txt`` (length + entropy stats)."""

    def write(
        self,
        *,
        name: str,
        values: np.ndarray,
        seq: str,
        start: int,
        out_dir: str,
        filename: str | None = None,
    ) -> str:
        """Write the summary. ``filename`` overrides the default ``<name>.summary.txt``
        (GenBank runs use ``stats.txt`` per the professor's spec)."""
        s = summarize(values)
        lines = [
            "DNA-Entropy summary",
            f"name:               {name}",
            f"length:             {s.length} nt",
            f"coordinate start:   {start}",
            f"entropy mean:       {s.mean:.4f} bits",
            f"entropy min:        {s.minimum:.4f} bits (position {start + s.argmin})",
            f"entropy max:        {s.maximum:.4f} bits (position {start + s.argmax})",
        ]
        text = "\n".join(lines) + "\n"
        return write_text_lf(Path(out_dir) / (filename or f"{name}.summary.txt"), text)

    def write_multi(
        self,
        *,
        name: str,
        sections: Sequence[tuple[str, np.ndarray]],
        start: int,
        out_dir: str,
        filename: str | None = None,
    ) -> str:
        """Write a summary covering several contigs: an overall block, then one per contig."""
        all_values = np.concatenate([v for _, v in sections]) if sections else np.array([0.0])
        overall = summarize(all_values)
        lines = [
            "DNA-Entropy summary",
            f"name:               {name}",
            f"records:            {len(sections)}",
            f"total length:       {overall.length} nt",
            f"entropy mean (all): {overall.mean:.4f} bits",
            f"entropy min (all):  {overall.minimum:.4f} bits",
            f"entropy max (all):  {overall.maximum:.4f} bits",
            "",
        ]
        for chrom, values in sections:
            s = summarize(values)
            lines += [
                f"[{chrom}]",
                f"  length:           {s.length} nt",
                f"  coordinate start: {start}",
                f"  entropy mean:     {s.mean:.4f} bits",
                f"  entropy min:      {s.minimum:.4f} bits (position {start + s.argmin})",
                f"  entropy max:      {s.maximum:.4f} bits (position {start + s.argmax})",
                "",
            ]
        text = "\n".join(lines).rstrip("\n") + "\n"
        return write_text_lf(Path(out_dir) / (filename or f"{name}.summary.txt"), text)
