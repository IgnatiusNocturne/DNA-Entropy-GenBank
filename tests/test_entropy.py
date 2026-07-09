"""Tests for Shannon entropy computation (the scientific core)."""

from __future__ import annotations

import numpy as np

from dna_entropy.analysis import MAX_ENTROPY_BITS, shannon_entropy, summarize


def test_uniform_distribution_is_max_entropy() -> None:
    probs = np.full((3, 4), 0.25, dtype=np.float32)
    h = shannon_entropy(probs)
    assert np.allclose(h, 2.0, atol=1e-6)


def test_one_hot_is_zero_entropy() -> None:
    probs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    h = shannon_entropy(probs)
    assert np.allclose(h, 0.0, atol=1e-6)  # 0*log0 handled, no NaN


def test_two_way_even_split_is_one_bit() -> None:
    probs = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
    h = shannon_entropy(probs)
    assert np.allclose(h, 1.0, atol=1e-6)


def test_output_shape_and_dtype() -> None:
    probs = np.full((10, 4), 0.25, dtype=np.float32)
    h = shannon_entropy(probs)
    assert h.shape == (10,)
    assert h.dtype == np.float32


def test_values_within_bounds_for_random_input() -> None:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((100, 4))
    probs = (np.exp(logits) / np.exp(logits).sum(1, keepdims=True)).astype(np.float32)
    h = shannon_entropy(probs)
    assert h.min() >= 0.0
    assert h.max() <= MAX_ENTROPY_BITS + 1e-6


def test_max_entropy_constant_is_two_bits() -> None:
    assert np.isclose(MAX_ENTROPY_BITS, 2.0)


def test_summarize_reports_extrema_positions() -> None:
    # position 1 uniform (2.0 bits), position 0 one-hot (0.0 bits)
    probs = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32
    )
    h = shannon_entropy(probs)
    s = summarize(h)
    assert s.length == 2
    assert s.argmin == 0
    assert s.argmax == 1
    assert np.isclose(s.minimum, 0.0, atol=1e-6)
    assert np.isclose(s.maximum, 2.0, atol=1e-6)
