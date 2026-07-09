"""Run configuration for a single DNA-Entropy invocation.

Uses stdlib dataclasses (no third-party dep) to keep the laptop/mock install light
per CLAUDE.md hard rule #5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PredictorKind(str, Enum):
    """Which prediction backend to use."""

    MOCK = "mock"
    EVO = "evo"


class TrackFormat(str, Enum):
    """Output format for the per-position entropy track."""

    BEDGRAPH = "bedgraph"
    WIG = "wig"


# Default Evo context cap for a single forward pass. The 7B model supports longer
# contexts, but a single pass is bounded by GPU VRAM (~24 GB on an A10); see
# docs/DESIGN.md §5. Sequences longer than this are rejected for now (windowing is
# future work).
DEFAULT_MAX_LEN = 8192


@dataclass
class RunConfig:
    """All knobs for one run. Populated by the CLI; consumed by the pipeline."""

    name: str = "user_locus"
    input_path: str | None = None  # None => read from stdin
    informat: str | None = None  # "genbank"|"fasta"|"paste" override; None => auto-detect
    predictor: PredictorKind = PredictorKind.MOCK
    model: str = "evo2_7b"
    device: str = "cuda"
    out_dir: str = "out"
    track_format: TrackFormat = TrackFormat.BEDGRAPH
    start: int = 1
    max_len: int = DEFAULT_MAX_LEN
    genes: bool = False
    rna: bool = False
    seed: int = 0
