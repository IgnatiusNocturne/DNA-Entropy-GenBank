"""Tests for the GFF3 gene writer (runs without Pyrodigal: uses synthetic features)."""

from __future__ import annotations

from pathlib import Path

from dna_entropy.annotators.base import GeneFeature
from dna_entropy.writers import GffWriter

FEATURES = [
    GeneFeature(begin=10, end=60, strand="+", gene_id="gene_1"),
    GeneFeature(begin=80, end=120, strand="-", partial=True, gene_id="gene_2"),
]


def test_gff_header_and_features(tmp_path: Path) -> None:
    path = GffWriter().write(
        name="locus", features=FEATURES, length=200, start=1, out_dir=str(tmp_path)
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "##gff-version 3"
    assert lines[1] == "##sequence-region locus 1 200"
    assert lines[2] == "locus\tpyrodigal\tgene\t10\t60\t.\t+\t.\tID=gene_1;Name=gene_1"
    assert lines[3].endswith("partial=true")
    assert "\t80\t120\t.\t-\t." in lines[3]


def test_gff_respects_start_offset(tmp_path: Path) -> None:
    path = GffWriter().write(
        name="locus", features=FEATURES, length=200, start=101, out_dir=str(tmp_path)
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[1] == "##sequence-region locus 101 300"
    # gene_1 begin 10 -> 110, end 60 -> 160
    assert "\t110\t160\t" in lines[2]


def test_gff_empty_features(tmp_path: Path) -> None:
    path = GffWriter().write(
        name="locus", features=[], length=50, start=1, out_dir=str(tmp_path)
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines == ["##gff-version 3", "##sequence-region locus 1 50"]
