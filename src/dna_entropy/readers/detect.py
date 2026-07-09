"""Detect the input file kind so the pipeline can route it to the right reader.

Order of evidence: file extension first, then a content sniff of the first non-blank line.
``None`` (stdin/paste) is always ``"paste"``.
"""

from __future__ import annotations

from pathlib import Path

# Source kinds used throughout the pipeline.
GENBANK = "genbank"
FASTA = "fasta"
PASTE = "paste"

_GENBANK_EXT = frozenset({".gb", ".gbk", ".genbank", ".gbff"})
_FASTA_EXT = frozenset({".fa", ".fasta", ".fna", ".ffn"})


def detect_kind(path: str | None) -> str:
    """Return ``"genbank"``, ``"fasta"``, or ``"paste"`` for ``path`` (None => paste)."""
    if path is None:
        return PASTE
    ext = Path(path).suffix.lower()
    if ext in _GENBANK_EXT:
        return GENBANK
    if ext in _FASTA_EXT:
        return FASTA
    # Unknown extension: sniff the first non-blank line.
    try:
        head = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return PASTE
    for line in head.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("LOCUS"):
            return GENBANK
        if s.startswith(">"):
            return FASTA
        break
    return PASTE
