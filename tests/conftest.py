"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_seq() -> str:
    """A short, valid uppercase A/C/G/T sequence."""
    return "ATGCGTACGTTAGCAACGTACGATCGATCGTAGCTAGCTAGCAT"
