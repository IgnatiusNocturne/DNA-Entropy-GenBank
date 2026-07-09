"""Read a FASTA file into a raw sequence.

Hand-rolled (no third-party dep) so the FASTA path works even without Biopython. For the
demo we analyze a single contig: if the file has multiple records, we use the first and
add a notice. The returned sequence is *raw* (not yet validated/uppercased) — validation
runs downstream like every other input.
"""

from __future__ import annotations

from pathlib import Path


class FastaReadError(ValueError):
    """Raised when a FASTA file has no sequence records."""


def read_fasta(path: str) -> tuple[str, list[str]]:
    """Return ``(raw_sequence, notices)`` for the first record in ``path``."""
    notices: list[str] = []
    records: list[tuple[str, list[str]]] = []  # (header, sequence-lines)
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    header: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, lines))
            header = line[1:].strip()
            lines = []
        elif header is not None:
            lines.append(line.strip())
    if header is not None:
        records.append((header, lines))

    if not records:
        raise FastaReadError("No FASTA records found (expected a '>' header line).")
    if len(records) > 1:
        notices.append(
            f"FASTA has {len(records)} records; using the first ({records[0][0][:40]!r})."
        )

    seq = "".join(records[0][1])
    return seq, notices
