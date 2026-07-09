"""Torch-free helpers to turn a model's next-token logits into the (L, 4) contract.

This is the trickiest part of the Evo integration (softmax + position alignment), so it
lives here as pure NumPy and is unit-tested on any machine (no GPU/torch needed). The
GPU-only model call lives in ``evo.py`` and feeds this function.
"""

from __future__ import annotations

import numpy as np

from .base import NUM_NUCLEOTIDES


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def aligned_acgt_probs(nuc_logits: np.ndarray) -> np.ndarray:
    """Align next-token nucleotide logits to per-base probabilities.

    Evo is autoregressive: ``nuc_logits[i]`` is the model's prediction for the base at
    position ``i + 1`` (given bases ``0..i``). We return an array where row ``i`` is the
    distribution *for* base ``i`` (CLAUDE.md / DESIGN.md §4 "Position semantics").

    Args:
        nuc_logits: ``(L, 4)`` logits over ``[A, C, G, T]`` (already restricted to the
            four nucleotide token ids), one row per input position.

    Returns:
        ``(L, 4)`` float32 probabilities. Row 0 has no preceding context, so it is set to
        the uniform distribution (maximum entropy, 2.0 bits).
    """
    if nuc_logits.ndim != 2 or nuc_logits.shape[1] != NUM_NUCLEOTIDES:
        raise ValueError(
            f"nuc_logits must be (L, {NUM_NUCLEOTIDES}), got {nuc_logits.shape}"
        )
    length = nuc_logits.shape[0]
    # next_probs[i] = distribution for base i+1.
    next_probs = _softmax_rows(nuc_logits.astype(np.float32))
    aligned = np.empty((length, NUM_NUCLEOTIDES), dtype=np.float32)
    aligned[0] = 1.0 / NUM_NUCLEOTIDES  # no context for the first base
    if length > 1:
        aligned[1:] = next_probs[:-1]
    return aligned
