"""Shared test setup: bytecode off, citysim and the bench modules importable, and a skip that works with or
without pytest (run_tests.py swaps SKIP_EXC for its own exception)."""
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
BENCH_DIR = EXPERIMENT_DIR.parent / "swarm" / "Bench"
for p in (str(EXPERIMENT_DIR), str(BENCH_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import pytest
    SKIP_EXC = None
except ImportError:          # run_tests.py
    pytest = None
    SKIP_EXC = RuntimeError


def skip(reason):
    if pytest is not None and SKIP_EXC is None:
        pytest.skip(reason)
    raise SKIP_EXC(reason)


def importorskip(name):
    try:
        return __import__(name)
    except ImportError:
        skip(f"{name} is not installed")
