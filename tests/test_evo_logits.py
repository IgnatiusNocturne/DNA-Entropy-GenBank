"""Tests for the torch-free logit->(L,4) alignment used by EvoPredictor.

These run on any machine (no GPU/torch) and cover the trickiest part of the Evo
integration: softmax over the four nucleotide logits and the next-token position shift.
"""

from __future__ import annotations

import numpy as np

from dna_entropy.predictors import aligned_acgt_probs


def _softmax(row: np.ndarray) -> np.ndarray:
    e = np.exp(row - row.max())
    return e / e.sum()


def test_shape_and_dtype() -> None:
    logits = np.zeros((5, 4), dtype=np.float32)
    out = aligned_acgt_probs(logits)
    assert out.shape == (5, 4)
    assert out.dtype == np.float32


def test_rows_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((20, 4)).astype(np.float32)
    out = aligned_acgt_probs(logits)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_first_row_is_uniform() -> None:
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((10, 4)).astype(np.float32)
    out = aligned_acgt_probs(logits)
    assert np.allclose(out[0], 0.25, atol=1e-6)


def test_position_alignment_is_next_token_shifted() -> None:
    # Distinct rows so the shift is unambiguous.
    logits = np.array(
        [[5.0, 0.0, 0.0, 0.0],   # predicts base 1
         [0.0, 5.0, 0.0, 0.0],   # predicts base 2
         [0.0, 0.0, 5.0, 0.0]],  # predicts base 3 (beyond -> dropped)
        dtype=np.float32,
    )
    out = aligned_acgt_probs(logits)
    # base 1 distribution == softmax(logits[0]); base 2 == softmax(logits[1])
    assert np.allclose(out[1], _softmax(logits[0]), atol=1e-6)
    assert np.allclose(out[2], _softmax(logits[1]), atol=1e-6)


def test_single_base_sequence() -> None:
    out = aligned_acgt_probs(np.zeros((1, 4), dtype=np.float32))
    assert out.shape == (1, 4)
    assert np.allclose(out[0], 0.25, atol=1e-6)
