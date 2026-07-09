"""EvoPredictor — the real Evo 2 (7B) backend.

THE ONLY module allowed to import ``torch`` / ``evo2`` (CLAUDE.md hard rule #1). Runs on
an NVIDIA GPU; see docs/EVO_SETUP.md. Everything here funnels into the same ``(L, 4)``
probability contract as MockPredictor, so it is a drop-in swap.

Untested on the dev laptop (no GPU). Validated by ``tests/test_evo_predictor.py``
(marked ``gpu``) on the A10 box. The pure alignment math is in ``logits.py`` and is
tested everywhere.
"""

from __future__ import annotations

import numpy as np
import torch  # noqa: F401  (GPU-only dep; isolated to this module)
from evo2 import Evo2

from .base import PredictorError, check_probability_matrix
from .logits import aligned_acgt_probs


class EvoPredictor:
    """Wraps Evo 2 to produce per-position A/C/G/T probabilities."""

    def __init__(
        self,
        model: str = "evo2_7b",
        device: str = "cuda",
        max_context: int = 8192,
    ) -> None:
        self.model_name = model
        self.device = device
        self.max_context = max_context
        try:
            self._model = Evo2(model)
        except Exception as exc:  # weights missing, OOM, etc.
            raise PredictorError(
                f"Failed to load Evo model {model!r}: {exc}"
            ) from exc
        # Ask the tokenizer for the exact ids of A, C, G, T rather than hard-coding
        # ASCII — robust to whatever scheme the model uses.
        nuc_ids = list(self._model.tokenizer.tokenize("ACGT"))
        if len(nuc_ids) != 4:
            raise PredictorError(
                f"Unexpected tokenization of 'ACGT' -> {nuc_ids}; "
                "the alignment in evo.py assumes one token per nucleotide."
            )
        # Cast to plain ints: the tokenizer returns uint8, which torch would otherwise
        # interpret as a boolean mask during column selection.
        self._nuc_ids = [int(x) for x in nuc_ids]

    @staticmethod
    def _extract_logits(raw: object) -> "torch.Tensor":
        """Normalize the model's return into a 2D ``(L, vocab)`` logits tensor.

        Tolerates evo2 versions that return a tuple, an object with ``.logits``, or a
        bare tensor, with or without a batch dimension.
        """
        out = raw
        # Evo2 returns a nested tuple like ((logits, inference_params), ...); unwrap
        # to the first tensor regardless of nesting depth.
        while isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits
        if out.ndim == 3:  # (batch, L, vocab)
            out = out[0]
        return out

    def predict(self, seq: str) -> np.ndarray:
        if self.max_context and len(seq) > self.max_context:
            raise PredictorError(
                f"Sequence length {len(seq)} exceeds Evo single-pass context "
                f"{self.max_context}; windowing is future work."
            )

        token_ids = self._model.tokenizer.tokenize(seq)
        input_ids = torch.tensor(token_ids, dtype=torch.int).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw = self._model(input_ids)
        logits = self._extract_logits(raw)  # (L, vocab)

        nuc_logits = logits[:, self._nuc_ids].float().cpu().numpy()  # (L, 4)
        probs = aligned_acgt_probs(nuc_logits)
        # Guards the predictor boundary; also catches any tokenizer length surprise
        # (e.g. an unexpected BOS token) by failing the shape check loudly.
        return check_probability_matrix(probs, len(seq))
