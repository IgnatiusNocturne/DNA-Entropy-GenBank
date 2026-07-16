"""Tests for GenBank/FASTA input routing, reading, writing, and the pipeline branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from dna_entropy.config import RunConfig
from dna_entropy.readers import detect
from dna_entropy.readers.fasta import read_fasta
from dna_entropy.readers.genbank import read_genbank
from dna_entropy.readers.input import load_input
from dna_entropy.validation.validators import ValidationError, validate_sequence
from dna_entropy.writers.genbank import GenBankWriter
from dna_entropy import pipeline

DATA = Path(__file__).parent / "data"
SAMPLE_GB = str(DATA / "sample.gb")
MULTI_GB = str(DATA / "multi.gb")
SAMPLE_FA = str(DATA / "sample.fasta")


# --- detection ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "path, expected",
    [
        (None, detect.PASTE),
        ("x.gb", detect.GENBANK),
        ("x.gbk", detect.GENBANK),
        ("x.genbank", detect.GENBANK),
        ("x.fasta", detect.FASTA),
        ("x.fa", detect.FASTA),
        ("x.fna", detect.FASTA),
    ],
)
def test_detect_by_extension(path, expected) -> None:
    assert detect.detect_kind(path) == expected


def test_detect_by_content_sniff(tmp_path: Path) -> None:
    gb = tmp_path / "mystery.dat"
    gb.write_text("LOCUS       foo    10 bp\n//\n", encoding="utf-8")
    assert detect.detect_kind(str(gb)) == detect.GENBANK
    fa = tmp_path / "mystery2.dat"
    fa.write_text(">seq1\nACGT\n", encoding="utf-8")
    assert detect.detect_kind(str(fa)) == detect.FASTA
    plain = tmp_path / "mystery3.dat"
    plain.write_text("ACGTACGT\n", encoding="utf-8")
    assert detect.detect_kind(str(plain)) == detect.PASTE


# --- GenBank reader (genes preserved, gene preferred over CDS) -----------------------

def test_read_genbank_extracts_seq_and_genes() -> None:
    records, notices = read_genbank(SAMPLE_GB)
    assert len(records) == 1
    rec = records[0]
    assert len(rec.seq) == 126
    assert len(rec.features) == 2  # two 'gene' features; the duplicate CDS is ignored
    a, b = rec.features
    assert (a.begin, a.end, a.strand, a.gene_id) == (1, 42, "+", "geneA")
    assert (b.begin, b.end, b.strand, b.gene_id) == (85, 126, "-", "geneB")
    assert any("not re-annotated" in n for n in notices)


# --- FASTA reader -------------------------------------------------------------------

def test_read_fasta_single_record() -> None:
    seq, notices = read_fasta(SAMPLE_FA)
    assert seq and set(seq.upper()) <= set("ACGTN")


def test_read_fasta_multi_record_uses_first(tmp_path: Path) -> None:
    p = tmp_path / "multi.fasta"
    p.write_text(">one\nACGT\nACGT\n>two\nTTTT\n", encoding="utf-8")
    seq, notices = read_fasta(str(p))
    assert seq == "ACGTACGT"
    assert any("using the first" in n for n in notices)


# --- lenient N validation -----------------------------------------------------------

def test_validate_tolerates_n_when_allowed() -> None:
    v = validate_sequence("ACGTNNNACGT", allow_ambiguity=True)
    assert v.seq == "ACGTNNNACGT"
    assert any("ambiguity" in n.lower() for n in v.notices)


def test_validate_rejects_n_by_default() -> None:
    with pytest.raises(ValidationError):
        validate_sequence("ACGTNNNACGT")


def test_validate_still_rejects_true_garbage_even_when_lenient() -> None:
    with pytest.raises(ValidationError):
        validate_sequence("ACGT@@@ACGT", allow_ambiguity=True)


# --- GenBank writer round-trip ------------------------------------------------------

def test_genbank_writer_embeds_mean_entropy(tmp_path: Path) -> None:
    import numpy as np
    from dna_entropy.readers.genbank import read_genbank

    records, _ = read_genbank(SAMPLE_GB)
    rec = records[0]
    seq, features = rec.seq, rec.features
    values = np.linspace(0.0, 2.0, len(seq), dtype="float32")
    out = GenBankWriter().write(
        name="tl", values=values, seq=seq, start=1, features=features, out_dir=str(tmp_path)
    )
    text = Path(out).read_text(encoding="utf-8")
    assert out.endswith("tl.gb")
    assert "mean_entropy=" in text
    assert "geneA" in text and "geneB" in text


# --- pipeline: GenBank input -> genbank + wig + stats, genes preserved ---------------

def test_pipeline_genbank_input(tmp_path: Path) -> None:
    cfg = RunConfig(name="tl", input_path=SAMPLE_GB, out_dir=str(tmp_path))
    result = pipeline.run(cfg)
    names = {Path(p).name for p in result.outputs}
    assert names == {"tl.gb", "tl.entropy.wig", "tl.entropy.geneious.gff3", "stats.txt"}
    # No FASTA / bedGraph, and no genes GFF, on the GenBank path (the only .gff3 is the
    # Geneious entropy track).
    assert not any(n.endswith((".fasta", ".bedgraph")) for n in names)
    assert not any(n.endswith(".genes.gff3") for n in names)
    assert len(result.genes) == 2  # preserved from the input, not re-called


def test_pipeline_fasta_input_adds_genbank(tmp_path: Path) -> None:
    cfg = RunConfig(name="fa", input_path=SAMPLE_FA, out_dir=str(tmp_path))
    result = pipeline.run(cfg)
    names = {Path(p).name for p in result.outputs}
    assert "fa.fasta" in names
    assert "fa.entropy.bedgraph" in names
    assert "fa.summary.txt" in names
    assert "fa.gb" in names  # bonus GenBank on the FASTA path


def test_load_input_paste_stays_strict(tmp_path: Path) -> None:
    cfg = RunConfig(name="p", input_path=None)
    loaded = load_input(cfg, raw="ACGTACGTACGT")
    assert loaded.source_kind == detect.PASTE
    assert loaded.features == []


# --- multi-record GenBank: ALL records processed ------------------------------------

def test_read_genbank_returns_all_records() -> None:
    records, notices = read_genbank(MULTI_GB)
    assert len(records) == 2
    assert [len(r.features) for r in records] == [2, 1]
    assert any("2 record" in n for n in notices)


def test_load_input_multi_record_contigs() -> None:
    cfg = RunConfig(name="mt", input_path=MULTI_GB)
    loaded = load_input(cfg)
    assert len(loaded.contigs) == 2
    # Geneious-style ids must be sanitized to safe, indexed contig names.
    assert [c.name for c in loaded.contigs] == ["mt_1", "mt_2"]
    assert all(set(c.name) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
               for c in loaded.contigs)


def test_pipeline_multi_record_genbank(tmp_path: Path) -> None:
    cfg = RunConfig(name="mt", input_path=MULTI_GB, out_dir=str(tmp_path))
    result = pipeline.run(cfg)
    names = {Path(p).name for p in result.outputs}
    assert names == {"mt.gb", "mt.entropy.wig", "mt.entropy.geneious.gff3", "stats.txt"}
    assert result.contigs == 2
    assert len(result.genes) == 3  # 2 + 1 genes preserved across both records

    # One GenBank file holds BOTH records.
    from Bio import SeqIO
    gb_path = next(p for p in result.outputs if p.endswith(".gb"))
    recs = list(SeqIO.parse(gb_path, "genbank"))
    assert len(recs) == 2

    # The WIG has a fixedStep block per record (chrom = each contig name).
    wig_text = Path(next(p for p in result.outputs if p.endswith(".wig"))).read_text()
    assert wig_text.count("fixedStep") == 2
    assert "chrom=mt_1" in wig_text and "chrom=mt_2" in wig_text

    # stats.txt reports each record.
    stats = Path(next(p for p in result.outputs if p.endswith("stats.txt"))).read_text()
    assert "records:            2" in stats
    assert "[mt_1]" in stats and "[mt_2]" in stats
