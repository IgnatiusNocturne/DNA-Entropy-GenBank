"""Tests for the local-GPU-first path (`--prefer-local`) and its cloud fallback.

On a GPU-less dev machine the Evo predictor is unavailable, so a local attempt must fail
*gracefully* (return False) rather than raise — that is exactly what triggers the cloud
fallback in `cloudrun`.
"""

from __future__ import annotations

from pathlib import Path

from dna_entropy.cli import _try_local_evo

DATA = Path(__file__).parent / "data"
SAMPLE_FA = str(DATA / "sample.fasta")


def test_try_local_evo_returns_false_without_gpu(tmp_path: Path) -> None:
    # Evo isn't importable on the laptop -> the helper must catch it and report failure.
    ok = _try_local_evo(
        name="x", input=SAMPLE_FA, informat=None, out_dir=str(tmp_path),
        genes=False, rna=False,
    )
    assert ok is False


def test_try_local_evo_success_with_mock(monkeypatch, tmp_path: Path) -> None:
    # Force the "evo" path to actually build a MockPredictor so we exercise the success
    # branch (writes files, returns True) without a GPU.
    from dna_entropy import pipeline
    from dna_entropy.predictors.mock import MockPredictor

    monkeypatch.setattr(pipeline, "build_predictor", lambda cfg: MockPredictor(seed=0))
    ok = _try_local_evo(
        name="x", input=SAMPLE_FA, informat=None, out_dir=str(tmp_path),
        genes=False, rna=False,
    )
    assert ok is True
    assert (tmp_path / "x.fasta").exists()
