"""Tests for the IGV output writers (bedGraph, WIG, FASTA, summary)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dna_entropy.writers import (
    BedGraphWriter,
    FastaWriter,
    SummaryWriter,
    WigWriter,
    Writer,
)

VALUES = np.array([0.0, 1.0, 2.0], dtype=np.float32)
SEQ = "ATG"


def test_writers_satisfy_protocol() -> None:
    for w in (BedGraphWriter(), WigWriter(), FastaWriter(), SummaryWriter()):
        assert isinstance(w, Writer)


def test_bedgraph_coordinates_and_values(tmp_path: Path) -> None:
    path = BedGraphWriter().write(
        name="locus", values=VALUES, seq=SEQ, start=1, out_dir=str(tmp_path)
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("track type=bedGraph")
    # 0-based half-open, start=1 -> first base is [0,1)
    assert lines[1] == "locus\t0\t1\t0.0000"
    assert lines[2] == "locus\t1\t2\t1.0000"
    assert lines[3] == "locus\t2\t3\t2.0000"


def test_bedgraph_respects_start_offset(tmp_path: Path) -> None:
    path = BedGraphWriter().write(
        name="locus", values=VALUES, seq=SEQ, start=100, out_dir=str(tmp_path)
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[1] == "locus\t99\t100\t0.0000"


def test_wig_header_and_values(tmp_path: Path) -> None:
    path = WigWriter().write(
        name="locus", values=VALUES, seq=SEQ, start=1, out_dir=str(tmp_path)
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("track type=wiggle_0")
    assert lines[1] == "fixedStep chrom=locus start=1 step=1 span=1"
    assert lines[2:] == ["0.0000", "1.0000", "2.0000"]


def test_fasta_roundtrip_and_wrapping(tmp_path: Path) -> None:
    seq = "ACGT" * 40  # 160 nt -> wraps at 60
    path = FastaWriter().write(
        name="locus", values=np.zeros(len(seq), dtype=np.float32),
        seq=seq, start=1, out_dir=str(tmp_path),
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == ">locus"
    assert all(len(line) <= 60 for line in lines[1:])
    assert "".join(lines[1:]) == seq


def test_summary_contains_stats(tmp_path: Path) -> None:
    path = SummaryWriter().write(
        name="locus", values=VALUES, seq=SEQ, start=1, out_dir=str(tmp_path)
    )
    text = Path(path).read_text(encoding="utf-8")
    assert "length:" in text
    assert "3 nt" in text
    assert "entropy mean:" in text
