"""Gene-boundary calling via Pyrodigal (a binding to Prodigal).

CAVEAT: Prodigal is a *prokaryotic* gene finder (bacteria/archaea). It is not accurate
for eukaryotic genomes. Uses metagenomic/"single-sequence" mode so it works on loci of
any length without a separate training pass.
"""

from __future__ import annotations

from .base import Annotator, AnnotatorError, GeneFeature


class ProdigalAnnotator:
    """Prokaryotic gene caller. ``pyrodigal`` is imported lazily (the ``[genes]`` extra)."""

    def __init__(self) -> None:
        try:
            import pyrodigal
        except ImportError as exc:
            raise AnnotatorError(
                "Gene calling needs Pyrodigal. Install it with: pip install -e \".[genes]\""
            ) from exc
        # meta=True uses Prodigal's pre-trained models, so no training pass is needed
        # and short loci work.
        self._finder = pyrodigal.GeneFinder(meta=True)

    def annotate(self, seq: str) -> list[GeneFeature]:
        try:
            genes = self._finder.find_genes(seq.encode("ascii"))
        except Exception as exc:
            raise AnnotatorError(f"Pyrodigal failed: {exc}") from exc

        features: list[GeneFeature] = []
        for i, g in enumerate(genes, start=1):
            features.append(
                GeneFeature(
                    begin=int(g.begin),
                    end=int(g.end),
                    strand="+" if g.strand > 0 else "-",
                    partial=bool(g.partial_begin or g.partial_end),
                    gene_id=f"gene_{i}",
                )
            )
        return features
