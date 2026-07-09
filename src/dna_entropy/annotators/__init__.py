"""Optional gene-boundary annotators (off by default)."""

from .base import Annotator, AnnotatorError, GeneFeature
from .prodigal import ProdigalAnnotator

__all__ = ["Annotator", "AnnotatorError", "GeneFeature", "ProdigalAnnotator"]
