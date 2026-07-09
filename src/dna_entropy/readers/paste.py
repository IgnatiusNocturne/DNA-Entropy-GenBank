"""Read a pasted sequence from a file or from stdin."""

from __future__ import annotations

import sys
from pathlib import Path


class PasteReader:
    """Reads raw text from a file path, or from stdin when ``path`` is ``None``.

    This is the "copy-paste" input for the demo. It does no cleaning — it just hands the
    raw text to the validation stage.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path

    def read(self) -> str:
        if self.path is not None:
            return Path(self.path).read_text(encoding="utf-8")
        return sys.stdin.read()
