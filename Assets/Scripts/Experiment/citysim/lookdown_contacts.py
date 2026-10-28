"""S4 Phase 3: which building each of Part A's building contacts touched (lessons.md 32: locate a mechanism's events
before writing it down).

    python -B -m citysim.lookdown_contacts [--designs 40 60] [--d 2.6] [--trials 10] [--jobs 4]
        -> experiment_results/split_study/lookdown_A/contacts.csv, and the counts on the console

Each Part A job named is flown again exactly as its run planned it (lookdown.run_job), with a tape on every flight
(citysim.playback), which records each contact as it happens and where. A contact is put on the nearest building
footprint: the block's own building, one across one of the block's streets, or another. Read-only: nothing the run
stored is changed, and the tapes are not kept.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, lookdown as LD, split_probe as S  # noqa: E402


def _seg_dist(q, a, b):
    ab = b - a
    t = np.clip(((q - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(q - (a + t * ab)))


def _poly_dist(q, poly):
    return min(_seg_dist(q, poly[i - 1], poly[i]) for i in range(len(poly)))


def locate(args):
    """One Part A job's contacts: [dict(p, d, cond, mode, T, kind, dist, t, t_J, t_commit)]."""
    from citysim.split_replays import taped
    p, s, c, t, d = args
    j = dict(p=p, s=s, city_seed=c, trial=t, d=d, T_grid=list(LD.T_GRID))
    res, made = taped(lambda: LD.run_job(j), 0.2)
    if "rows" not in res:
        return []
    C = LD.city_targets(S.city("R", p, s, c))
    net, L = C["net"], C["L"]
    polys = [blockcity.building_poly(b) for b in L.boxes]
    own = net.blocks[res["block"]]["building"]
    across = {net.blocks[b]["building"] for sid in net.blocks[res["block"]]["streets"]
              for b in net.streets[sid]["blocks"]
              if b != res["block"] and b in net.blocks and net.blocks[b]["building"] >= 0}
    out = []
    for (cond, mode, T), (f, tape), row in zip(LD.conditions(LD.T_GRID), made, res["rows"]):
        for ct in tape.record()["contacts"]:
            q = np.array([ct["x"], ct["z"]], float)
            dists = [_poly_dist(q, P) for P in polys]
            k = int(np.argmin(dists))
            out.append(dict(p=p, d=d, city_seed=c, trial=t, cond=cond, mode=mode, T=T,
                            kind="own" if k == own else "across" if k in across else "other",
                            dist=round(dists[k], 2), t=ct["t"], t_J=row["t_J"], t_commit=row["t_commit"]))
    return out


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--designs", type=float, nargs="+", default=[40.0, 60.0], help="block pitches p (s 18)")
    ap.add_argument("--d", type=float, nargs="+", default=[2.6], help="block spreads")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(p, 18.0, LD.CITY_SEEDS[t % len(LD.CITY_SEEDS)], t, d) for p in a.designs for d in a.d
            for t in range(a.trials)]
    rows = []
    with Pool(min(a.jobs, len(jobs))) as pool:
        for r in pool.imap_unordered(locate, jobs):
            rows += r
    df = pd.DataFrame(rows)
    out = LD.OUT_ROOT / "lookdown_A" / "contacts.csv"
    df.to_csv(out, index=False)
    print(f"[lookdown_contacts] {len(df)} contacts in {len(jobs)} jobs -> {out}")
    if len(df):
        df["phase"] = np.where(df.t < df.t_J, "before arriving",
                               np.where(df.t <= df.t_commit, "arriving to the decision", "after the decision"))
        print(df.groupby(["p", "d", "kind"]).size().unstack(fill_value=0))
        print(df.groupby(["cond", "kind"]).size().unstack(fill_value=0))
        print(df.groupby(["kind", "phase"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
