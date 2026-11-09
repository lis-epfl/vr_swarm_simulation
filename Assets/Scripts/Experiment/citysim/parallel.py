"""A process pool that survives a worker dying.

A pool worker that dies takes its job with it, and Pool.map then waits for that job forever with every other worker
idle -- it has happened here several times, with no error anywhere (this machine's CPU throws the odd access
violation under load). run_jobs gives up on a pool that has produced nothing for `stall_s`, tears it down, and runs
the jobs still outstanding again in a fresh one, up to `retries` times; only then does it fail, naming them.
"""
import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool, TimeoutError as PoolTimeout  # noqa: E402


def default_jobs():
    return max(1, (os.cpu_count() or 2) - 2)


def run_jobs(fn, jobs, label="jobs", processes=None, stall_s=600.0, retries=2, progress=None):
    """[fn(job) for job in jobs] computed in a process pool, in job order. `progress(done, total, result)` is called
    as results arrive."""
    jobs = list(jobs)
    out = [None] * len(jobs)
    todo = list(range(len(jobs)))
    done = 0
    for attempt in range(retries + 1):
        if not todo:
            break
        n = max(1, min(processes or default_jobs(), len(todo)))
        pool = Pool(n)
        try:
            pending = {k: pool.apply_async(fn, (jobs[k],)) for k in todo}
            last = time.time()
            while pending:
                finished = [k for k, r in pending.items() if r.ready()]
                for k in finished:
                    out[k] = pending.pop(k).get()
                    done += 1
                    last = time.time()
                    if progress is not None:
                        progress(done, len(jobs), out[k])
                if not finished:
                    if time.time() - last > stall_s:
                        break
                    try:
                        next(iter(pending.values())).wait(timeout=0.2)
                    except PoolTimeout:
                        pass
            todo = sorted(pending)
        finally:
            pool.terminate()
            pool.join()
        if todo:
            print(f"[parallel] {label}: no result for {stall_s:.0f} s with {len(todo)} job(s) outstanding "
                  f"(a worker probably died); {'retrying them' if attempt < retries else 'giving up'}", flush=True)
    if todo:
        raise SystemExit(f"[parallel] {label}: {len(todo)} job(s) never finished: {todo[:10]}")
    return out
