"""A GPU-free stand-in for a real predictor.

Produces deterministic, contract-correct ``(L, 4)`` probabilities so the entire
pipeline (validation -> analysis -> export) can be developed and tested on a machine
with no GPU and no Evo install. Same output contract as :class:`EvoPredictor`, so it is
a true drop-in (CLAUDE.md hard rule #5).
"""

from __future__ import annotations

import zlib

import numpy as np

from .base import NUM_NUCLEOTIDES, check_probability_matrix


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class MockPredictor:
    """Deterministic fake predictor.

    Output depends on ``seed`` and the sequence content (via a stable CRC32), so it is
    reproducible across processes and runs, while different sequences still differ.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)

    def predict(self, seq: str) -> np.ndarray:
        length = len(seq)
        # Stable, process-independent per-sequence seed (unlike builtin hash()).
        seq_seed = zlib.crc32(seq.encode("ascii")) & 0xFFFFFFFF
        rng = np.random.default_rng(self._seed + seq_seed)
        logits = rng.standard_normal((length, NUM_NUCLEOTIDES)).astype(np.float32)
        probs = _softmax(logits).astype(np.float32)
        return check_probability_matrix(probs, length)
