"""The Annotator contract — optional gene-boundary calling.

Off by default and never on the critical path (CLAUDE.md): if an annotator is missing
or fails, the entropy track is still produced. A different gene caller can slot in behind
this Protocol later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class AnnotatorError(RuntimeError):
    """Raised when an annotator backend is unavailable or fails."""


@dataclass
class GeneFeature:
    """A predicted gene/CDS. Coordinates are 1-based inclusive, sequence-relative."""

    begin: int
    end: int
    strand: str  # "+" or "-"
    partial: bool = False
    gene_id: str = ""


@runtime_checkable
class Annotator(Protocol):
    """Predicts gene boundaries from a DNA sequence."""

    def annotate(self, seq: str) -> list[GeneFeature]:
        """Return predicted genes as a list of :class:`GeneFeature`."""
        ...
