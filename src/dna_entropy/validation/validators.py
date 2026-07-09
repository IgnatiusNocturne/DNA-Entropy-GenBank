"""Normalize and validate a raw pasted sequence into clean A/C/G/T.

Runs *before* any predictor (CLAUDE.md hard rule #2). Fails fast with messages that
point at the exact problem; non-fatal observations come back as ``notices``.

Rules are specified in docs/DESIGN.md §5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import DEFAULT_MAX_LEN

_ACGT: frozenset[str] = frozenset("ACGT")
# IUPAC nucleotide ambiguity codes — recognized so we can give a helpful message,
# but not supported yet (future work).
_AMBIGUITY: frozenset[str] = frozenset("NRYSWKMBDHV")

# Below this length, entropy is dominated by the model's prior near the start; warn only.
DEFAULT_MIN_LEN: int = 10


class ValidationError(ValueError):
    """Raised when input cannot be turned into a valid A/C/G/T sequence."""


@dataclass
class ValidatedSequence:
    """A clean, model-ready sequence plus any non-fatal notices for the user."""

    seq: str
    notices: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.seq)


def _strip_leading_header(raw: str) -> tuple[str, list[str]]:
    """Drop a single leading FASTA-style header line (``>...``) if present."""
    notices: list[str] = []
    lines = raw.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "":
            continue  # skip blank lines before the first content line
        if line.lstrip().startswith(">"):
            notices.append(f"Ignored FASTA header line: {line.strip()[:60]!r}")
            del lines[idx]
        break
    return "\n".join(lines), notices


def _normalize(text: str) -> tuple[str, list[str]]:
    """Remove whitespace and digits (e.g. line numbers); uppercase. No U->T here."""
    notices: list[str] = []
    no_ws = re.sub(r"\s", "", text)
    n_digits = sum(c.isdigit() for c in no_ws)
    if n_digits:
        notices.append(f"Removed {n_digits} digit character(s) (e.g. line numbers).")
    seq = re.sub(r"\d", "", no_ws).upper()
    return seq, notices


def validate_sequence(
    raw: str,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    rna: bool = False,
    min_len: int = DEFAULT_MIN_LEN,
    allow_ambiguity: bool = False,
) -> ValidatedSequence:
    """Clean and validate ``raw`` into a :class:`ValidatedSequence`.

    Args:
        raw: raw pasted text (may contain a header, whitespace, line numbers).
        max_len: single-pass context cap; longer sequences are rejected.
        rna: if True, convert ``U`` -> ``T`` instead of rejecting RNA input.
        min_len: warn (do not fail) below this length.
        allow_ambiguity: if True, IUPAC ambiguity codes (``N`` etc.) are kept with a
            notice instead of failing. Used for real GenBank/FASTA files, which routinely
            contain ``N``; the pasted-sequence path stays strict A/C/G/T.

    Raises:
        ValidationError: on RNA without ``rna=True``, empty input, non-ACGT characters
            (ambiguity codes allowed only when ``allow_ambiguity``), or length over
            ``max_len``.
    """
    notices: list[str] = []

    text, n = _strip_leading_header(raw)
    notices += n
    seq, n = _normalize(text)
    notices += n

    # RNA handling must come before the strict A/C/G/T check, since U is not in ACGT.
    if "U" in seq:
        if rna:
            count = seq.count("U")
            seq = seq.replace("U", "T")
            notices.append(f"Converted {count} U->T (RNA input).")
        else:
            pos = seq.index("U") + 1
            raise ValidationError(
                f"Found 'U' at position {pos}: this looks like RNA. "
                "Re-run with --rna to convert U->T, or paste a DNA sequence."
            )

    if not seq:
        raise ValidationError(
            "No nucleotides found after cleaning the input (empty sequence)."
        )

    bad = [i for i, c in enumerate(seq) if c not in _ACGT]
    if bad:
        non_iupac = [i for i in bad if seq[i] not in _AMBIGUITY]
        if allow_ambiguity and not non_iupac:
            # Every offending char is a recognized IUPAC ambiguity code — keep them.
            codes = sorted({seq[i] for i in bad})
            notices.append(
                f"Kept {len(bad)} ambiguity code(s) ({', '.join(codes)}); entropy at those "
                "positions reflects the model's prediction, not a definite base."
            )
        else:
            i = (non_iupac or bad)[0]
            c = seq[i]
            hint = ""
            if c in _AMBIGUITY:
                hint = f" '{c}' is an IUPAC ambiguity code, which isn't supported yet."
            raise ValidationError(
                f"Invalid character {c!r} at position {i + 1} "
                f"({len(bad)} non-ACGT character(s) total). "
                f"Only A, C, G, T are allowed.{hint}"
            )

    if len(seq) > max_len:
        raise ValidationError(
            f"Sequence length {len(seq)} exceeds the single-pass cap of {max_len} nt. "
            "Longer loci need windowing (future work); raise --max-len only if the GPU "
            "has the VRAM."
        )

    if len(seq) < min_len:
        notices.append(
            f"Warning: sequence is short ({len(seq)} nt); entropy near the start is "
            "dominated by the model's prior."
        )

    return ValidatedSequence(seq=seq, notices=notices)
