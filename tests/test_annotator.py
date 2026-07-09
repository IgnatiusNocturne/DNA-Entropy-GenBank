"""Tests for the Pyrodigal annotator (skipped if the [genes] extra isn't installed)."""

from __future__ import annotations

import pytest

pytest.importorskip("pyrodigal")

from dna_entropy.annotators import (  # noqa: E402
    Annotator,
    GeneFeature,
    ProdigalAnnotator,
)


def test_annotator_satisfies_protocol() -> None:
    assert isinstance(ProdigalAnnotator(), Annotator)


def test_annotate_returns_gene_features() -> None:
    # A pseudo-random but valid sequence; result may be empty, but must be a valid list.
    seq = "ATGGCAAAACGT" * 50
    genes = ProdigalAnnotator().annotate(seq)
    assert isinstance(genes, list)
    for g in genes:
        assert isinstance(g, GeneFeature)
        assert 1 <= g.begin <= g.end <= len(seq)
        assert g.strand in ("+", "-")


def test_annotate_empty_on_trivial_sequence() -> None:
    genes = ProdigalAnnotator().annotate("ACGT")
    assert genes == []
