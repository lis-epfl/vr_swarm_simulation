"""Run citysim's tests with or without pytest.

    python -B -m pytest -p no:cacheprovider Assets/Scripts/Experiment/citysim/tests     # where pytest is installed
    python -B Assets/Scripts/Experiment/citysim/tests/run_tests.py [-k substring]         # the stitching env has none

The fallback collects test_*.py here, calls every module-level test_* function, hands `tmp_path` a fresh
temporary directory when a test asks for one, and treats pytest.skip(...) (via the shim in conftest) as a skip.
"""
import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import inspect  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))      # Assets/Scripts/Experiment: import citysim
sys.path.insert(0, str(HERE))                 # conftest


class Skipped(Exception):
    pass


def main():
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    import conftest
    conftest.SKIP_EXC = Skipped
    passed, failed, skipped = 0, [], 0
    t0 = time.time()
    for f in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Skipped as e:
            print(f"SKIP {f.name}: {e}")
            skipped += 1
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__ or sel not in f"{f.stem}::{name}":
                continue
            kwargs = {}
            tmp = None
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmp.name)
            t1 = time.time()
            try:
                fn(**kwargs)
                passed += 1
                print(f"ok   {f.stem}::{name} ({time.time() - t1:.1f} s)")
            except Skipped as e:
                skipped += 1
                print(f"SKIP {f.stem}::{name}: {e}")
            except Exception:  # noqa: BLE001
                failed.append(f"{f.stem}::{name}")
                print(f"FAIL {f.stem}::{name}")
                traceback.print_exc()
            finally:
                if tmp is not None:
                    tmp.cleanup()
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped in {time.time() - t0:.0f} s")
    for n in failed:
        print("  FAILED", n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
