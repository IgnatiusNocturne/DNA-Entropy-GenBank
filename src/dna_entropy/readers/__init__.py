"""Input readers. Paste/stdin today; FASTA/GenBank later, behind the same Protocol."""

from .base import Reader
from .paste import PasteReader

__all__ = ["Reader", "PasteReader"]
