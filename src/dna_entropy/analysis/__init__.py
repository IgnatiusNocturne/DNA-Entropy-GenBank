"""Analysis stage: turn probabilities into entropy (and, later, other metrics)."""

from .entropy import MAX_ENTROPY_BITS, EntropySummary, shannon_entropy, summarize

__all__ = ["MAX_ENTROPY_BITS", "EntropySummary", "shannon_entropy", "summarize"]
