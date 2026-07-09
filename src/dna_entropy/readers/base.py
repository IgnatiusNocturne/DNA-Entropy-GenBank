"""The Reader contract — turns some input source into raw text.

Paste/stdin today; FASTA and GenBank readers slot in behind the same Protocol later
(docs/ROADMAP.md backlog). A reader returns *raw* text; cleaning/validation is a
separate stage (CLAUDE.md hard rule #2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reader(Protocol):
    """Reads raw, unvalidated sequence text from some source."""

    def read(self) -> str:
        """Return the raw text (may contain whitespace, headers, line numbers, etc.)."""
        ...
