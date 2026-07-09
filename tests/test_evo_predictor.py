"""GPU integration test for EvoPredictor (Evo 2 7B).

Marked ``gpu`` and skipped automatically where torch/evo2 aren't installed, so it never
runs (or errors) on the dev laptop. Run it on the A10 box: ``pytest -m gpu``.
"""

from __future__ import annotations

import pytest

# Skip the whole module cleanly if the Evo stack isn't present.
pytest.importorskip("torch")
pytest.importorskip("evo2")

import numpy as np  # noqa: E402

from dna_entropy.predictors.base import NUM_NUCLEOTIDES, check_probability_matrix  # noqa: E402
from dna_entropy.predictors.evo import EvoPredictor  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def evo() -> EvoPredictor:
    # Loads weights (large, first run downloads). Module-scoped to do it once.
    return EvoPredictor(model="evo2_7b", device="cuda")


def test_evo_output_satisfies_contract(evo: EvoPredictor) -> None:
    seq = "ACGTACGTACGTACGTACGT"
    probs = evo.predict(seq)
    check_probability_matrix(probs, len(seq))  # shape (L,4), float32, rows sum to 1
    assert probs.shape == (len(seq), NUM_NUCLEOTIDES)


def test_evo_first_position_is_uniform(evo: EvoPredictor) -> None:
    probs = evo.predict("ACGTACGT")
    assert np.allclose(probs[0], 1.0 / NUM_NUCLEOTIDES, atol=1e-5)


def test_evo_matches_mock_contract_shape(evo: EvoPredictor) -> None:
    # The whole point of the contract: Evo is a drop-in for the mock.
    from dna_entropy.predictors import MockPredictor

    seq = "ACGTACGTACGT"
    evo_probs = evo.predict(seq)
    mock_probs = MockPredictor().predict(seq)
    assert evo_probs.shape == mock_probs.shape
    assert evo_probs.dtype == mock_probs.dtype
