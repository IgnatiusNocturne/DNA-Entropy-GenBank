"""Tiny ASCII progress UI for the cloud orchestrator.

ASCII-only (CLAUDE.md hard rule #11) so it never crashes a cp1252 Windows console.
The spinner runs in a background thread and shows a live frame + elapsed seconds; if
stdout isn't a TTY it degrades to plain start/end lines.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Optional

# Classic 4-frame ASCII spinner — flips quickly, safe everywhere.
_FRAMES = ["-", "\\", "|", "/"]


class Spinner:
    """Context manager: shows ``  / message (3s)`` while a step runs."""

    def __init__(self, message: str, stream=None) -> None:
        self.message = message
        self.stream = stream or sys.stdout
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start = 0.0
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def __enter__(self) -> "Spinner":
        self._start = time.monotonic()
        if self._tty:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            self.stream.write(f"  .. {self.message}\n")
            self.stream.flush()
        return self

    def _run(self) -> None:
        for frame in itertools.cycle(_FRAMES):
            if self._stop.is_set():
                break
            elapsed = int(time.monotonic() - self._start)
            self.stream.write(f"\r  {frame} {self.message} ({elapsed}s)    ")
            self.stream.flush()
            time.sleep(0.12)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        elapsed = int(time.monotonic() - self._start)
        mark = "OK" if exc_type is None else "!!"
        line = f"  [{mark}] {self.message} ({elapsed}s)"
        if self._tty:
            self.stream.write("\r" + line + " " * 8 + "\n")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        return False  # never suppress exceptions
