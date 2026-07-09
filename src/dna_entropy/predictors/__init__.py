"""Prediction backends. The Predictor contract is the project's single swap point."""

from .base import (
    NUCLEOTIDES,
    NUM_NUCLEOTIDES,
    Predictor,
    PredictorError,
    check_probability_matrix,
)
from .logits import aligned_acgt_probs
from .mock import MockPredictor

# NB: EvoPredictor is intentionally NOT imported here — it pulls in torch/evo2, which
# must stay off the mock/dev path. Import it lazily from `.evo` (the pipeline does this).

__all__ = [
    "NUCLEOTIDES",
    "NUM_NUCLEOTIDES",
    "Predictor",
    "PredictorError",
    "check_probability_matrix",
    "aligned_acgt_probs",
    "MockPredictor",
]
