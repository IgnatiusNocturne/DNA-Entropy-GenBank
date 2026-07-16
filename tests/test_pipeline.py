"""End-to-end pipeline tests on the mock predictor (no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dna_entropy.analysis import MAX_ENTROPY_BITS
from dna_entropy.config import PredictorKind, RunConfig, TrackFormat
from dna_entropy.predictors.base import PredictorError
from dna_entropy import pipeline

RAW = ">demo header\nATGC ATGC ATGC\nACGTACGTACGT"
CLEAN_LEN = 24


def test_run_writes_all_outputs(tmp_path: Path) -> None:
    cfg = RunConfig(name="demo", out_dir=str(tmp_path))
    result = pipeline.run(cfg, raw=RAW)

    assert len(result.seq) == CLEAN_LEN
    assert result.values.shape == (CLEAN_LEN,)
    assert result.values.min() >= 0.0
    assert result.values.max() <= MAX_ENTROPY_BITS + 1e-6

    # Paste/FASTA input yields the IGV set plus a bonus self-contained GenBank.
    expected = {
        "demo.fasta",
        "demo.entropy.bedgraph",
        "demo.entropy.geneious.gff3",
        "demo.summary.txt",
        "demo.gb",
    }
    written = {Path(p).name for p in result.outputs}
    assert expected == written
    for p in result.outputs:
        assert Path(p).exists()


def test_run_propagates_validation_notices(tmp_path: Path) -> None:
    cfg = RunConfig(name="demo", out_dir=str(tmp_path))
    result = pipeline.run(cfg, raw=RAW)
    assert any("header" in n.lower() for n in result.notices)


def test_run_wig_format(tmp_path: Path) -> None:
    cfg = RunConfig(name="demo", out_dir=str(tmp_path), track_format=TrackFormat.WIG)
    result = pipeline.run(cfg, raw="ATGCATGCATGC")
    assert any(Path(p).name == "demo.entropy.wig" for p in result.outputs)


def test_run_is_deterministic(tmp_path: Path) -> None:
    cfg = RunConfig(name="demo", out_dir=str(tmp_path), seed=7)
    a = pipeline.run(cfg, raw="ATGCATGCATGC")
    b = pipeline.run(cfg, raw="ATGCATGCATGC")
    assert np.array_equal(a.values, b.values)


def test_evo_predictor_unavailable_is_clean_error(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="demo", out_dir=str(tmp_path), predictor=PredictorKind.EVO
    )
    with pytest.raises(PredictorError):
        pipeline.run(cfg, raw="ATGCATGCATGC")
