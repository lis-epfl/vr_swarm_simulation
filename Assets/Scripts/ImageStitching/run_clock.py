"""
Process-wide run clock for the stitcher's console lines.

One module so every periodic line (StitcherThreading, StabStitcher, PlanarStitcher) is
stamped from the same origin. The clock starts on first import, which StitcherThreading
does before anything heavy, so it reads as time since the script was launched.
"""
import time

_START = time.perf_counter()


def uptime() -> str:
    """Time since launch as H:MM:SS."""
    s = int(time.perf_counter() - _START)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def stamp() -> str:
    """Prefix for a periodic console line: a blank line, then the run time."""
    return f"\n[up {uptime()}]"
