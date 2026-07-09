"""Per-position Shannon entropy from nucleotide probabilities.

The scientific core: given the predicted distribution over A/C/G/T at each position,
how uncertain is the model? Uniform (0.25 each) -> 2.0 bits; one-hot -> 0.0 bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..predictors.base import NUM_NUCLEOTIDES

# Maximum entropy for a 4-symbol alphabet: log2(4) = 2.0 bits.
MAX_ENTROPY_BITS: float = float(np.log2(NUM_NUCLEOTIDES))


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute per-position Shannon entropy in bits.

    Args:
        probs: ``(L, 4)`` array of per-position probabilities (rows sum to 1).

    Returns:
        ``(L,)`` float32 array, each value in ``[0.0, 2.0]``. Uses the convention
        ``0 * log2(0) = 0``.
    """
    # log2(0) is -inf; multiplied by p=0 it should contribute 0. Compute log2 only
    # where p > 0, leaving zeros elsewhere (the 0*log0 = 0 convention).
    safe_log = np.zeros_like(probs)
    np.log2(probs, out=safe_log, where=probs > 0.0)
    h = -np.sum(probs * safe_log, axis=1)
    # Clip away tiny floating-point excursions outside [0, MAX].
    return np.clip(h, 0.0, MAX_ENTROPY_BITS).astype(np.float32)


@dataclass
class EntropySummary:
    """Summary statistics for an entropy track."""

    length: int
    mean: float
    minimum: float
    maximum: float
    argmin: int  # 0-based index of the lowest-entropy position
    argmax: int  # 0-based index of the highest-entropy position


def summarize(values: np.ndarray) -> EntropySummary:
    """Compute summary statistics over a ``(L,)`` entropy array."""
    return EntropySummary(
        length=int(values.shape[0]),
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        argmin=int(values.argmin()),
        argmax=int(values.argmax()),
    )
