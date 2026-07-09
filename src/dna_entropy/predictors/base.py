"""The Predictor contract — the single swap point of the whole project.

Every predictor (mock today, Evo 2 7B today, a trained ANN tomorrow) returns the SAME
thing: per-position nucleotide probabilities. Everything downstream (entropy, export)
depends only on this `(L, 4)` array, never on how it was produced.

See CLAUDE.md hard rules #1 and #3, and docs/DESIGN.md §4.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# Column order is FIXED across the whole codebase. Do not reorder.
NUCLEOTIDES: tuple[str, str, str, str] = ("A", "C", "G", "T")
NUM_NUCLEOTIDES: int = 4

# Numerical tolerance for the per-row "sums to 1.0" invariant.
PROB_SUM_TOL: float = 1e-4


class PredictorError(RuntimeError):
    """Raised when a predictor backend is unavailable or fails to run."""


@runtime_checkable
class Predictor(Protocol):
    """Maps a validated DNA string to per-position nucleotide probabilities."""

    def predict(self, seq: str) -> np.ndarray:
        """Predict per-position nucleotide probabilities.

        Args:
            seq: A validated, uppercase ``A``/``C``/``G``/``T`` string of length ``L``.

        Returns:
            ``np.ndarray`` of shape ``(L, 4)``, dtype ``float32``, columns ordered
            ``[A, C, G, T]`` (see :data:`NUCLEOTIDES`), with each row summing to ``1.0``.
            Row ``i`` is the model's predicted distribution for base ``i``.
        """
        ...


def check_probability_matrix(probs: np.ndarray, seq_len: int) -> np.ndarray:
    """Assert that ``probs`` satisfies the Predictor contract; return it unchanged.

    Use this to guard the boundary between any predictor and the rest of the pipeline
    (CLAUDE.md hard rule #3).

    Raises:
        ValueError: if shape, dtype, value range, or row-sum invariants are violated.
    """
    if probs.ndim != 2 or probs.shape != (seq_len, NUM_NUCLEOTIDES):
        raise ValueError(
            f"probabilities must have shape ({seq_len}, {NUM_NUCLEOTIDES}), "
            f"got {probs.shape}"
        )
    if probs.dtype != np.float32:
        raise ValueError(f"probabilities must be float32, got {probs.dtype}")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=PROB_SUM_TOL):
        worst = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"each row must sum to 1.0 (worst deviation {worst:.2e})")
    return probs
