"""S11 Wrap Hold: the search flight's calibration against ERKS's wrapped moments (internal_4).

    python -B -m citysim.swarmwrap_calib [--n 24]
    -> experiment_results/swarm_wrap/calibration.json

For ERKS's three swarm flights (from the start line to the last find) and for n scripted searches flown under the
scene's rule (swarmwrap.search_route, the screen's route indices), while a building was wrapped: the share of the time
narrowed (scale < 0.95), the share with the stick held (|stick| > 0.1), and the centroid's offset from the wrapped
building's centre in its ellipse's unit space (the rule's 0.8 test reads it): its median and the share beyond 0.8.
A separate module from swarmwrap.py so it can run beside a pool importing that one (lesson 18).
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

from citysim import swarmtune as T, swarmwrap as W  # noqa: E402

rep = T.rep


def unit_offset(E, cen, P, cent):
    """The centroid's offset in the unit space of the nearest wrapped building's ellipse (NaN when none is wrapped)."""
    rmax = float(np.linalg.norm(P - cen, axis=1).max())
    near = np.flatnonzero(np.linalg.norm(cent - cen, axis=1) <= rmax)
    if not len(near):
        return math.nan
    inside = near[rep.inside_formation(P, cent[near])]
    if not len(inside):
        return math.nan
    rel = cen - E.c[inside]
    u = (rel * E.e[inside]).sum(1) / E.a[inside]
    v = (rel * E.ep[inside]).sum(1) / E.b[inside]
    return float(np.hypot(u, v).min())


def summarise(narrowed, held, off):
    off = np.asarray([x for x in off if math.isfinite(x)])
    return dict(n=len(narrowed), narrowed=float(np.mean(narrowed)), held=float(np.mean(held)),
                off_median=float(np.median(off)) if len(off) else None,
                beyond=float(np.mean(off > 0.8)) if len(off) else None)


def erks():
    import pandas as pd
    runs = pd.read_csv(os.path.join(W.INTERNAL_4, "results", "runs.csv"), sep=";")
    goals = pd.read_csv(os.path.join(W.INTERNAL_4, "results", "goals.csv"), sep=";")
    air = dict(T.sp.resolve_airframe())
    nar, held, off = [], [], []
    for stem in sorted(runs[runs.condition == "Swarm"].run):
        ses = json.loads(open(os.path.join(W.INTERNAL_4, stem + "_session.json"), encoding="utf-8").read())
        L = json.loads(open(ses["layout"]["path"], encoding="utf-8-sig").read())
        rc = rep.PolyCity([np.reshape(b["poly_xz"], (-1, 2)) for b in L["boxes"]])
        m = rep.Model(dict(T.sp.resolve_swarm(T.SCENE, W.INTERNAL_4, stem)), air)
        E = rep.ell_rules(m, rc)
        cent = W.building_centres(m, rc)
        dr = pd.read_csv(os.path.join(W.INTERNAL_4, stem + "_drones.csv"), sep=";")
        hd = pd.read_csv(os.path.join(W.INTERNAL_4, stem + "_head.csv"), sep=";")
        sh = pd.read_csv(os.path.join(W.INTERNAL_4, stem + "_shape.csv"), sep=";")
        t0 = float(runs[runs.run == stem].startSec.iloc[0])
        t1 = float(goals[goals.run == stem].decisionTimeSec.max())
        for t, g in dr[(dr.alive == 1) & (dr.t >= t0) & (dr.t <= t1)].groupby("t"):
            P = g[["gtX", "gtZ"]].to_numpy()
            if not W.wrapped_at(P, cent):
                continue
            h = hd.iloc[int(np.abs(hd.t.to_numpy() - t).argmin())]
            nar.append(float(np.interp(t, sh.t.to_numpy(), sh.spreadScale.to_numpy())) < 0.95)
            held.append(float(np.hypot(h.inRoll, h.inPitch)) > 0.1)
            off.append(unit_offset(E, P.mean(0), P, cent))
    return summarise(nar, held, off)


class _Log(W.WrapLog):
    def __init__(self, m, rc):
        super().__init__(m, rc)
        self.E, self.off = rep.ell_rules(m, rc), []

    def step(self, sim, t, cmd, wp):
        n = len(self.t)
        super().step(sim, t, cmd, wp)
        if len(self.t) > n:
            P = sim.P[sim.alive]
            self.off.append(unit_offset(self.E, P.mean(0), P, self.cent) if self.wrap[-1] else math.nan)


def scripted(n):
    over = json.loads((W.OUT / "screen_cands.json").read_text(encoding="utf-8"))[0]["over"]
    m = T.model(over)
    nar, held, off = [], [], []
    for k in range(W.K_SCREEN, W.K_SCREEN + n):
        C = T.city(T.TUNE_SEED)
        rng = np.random.default_rng(T.route_seed("search", T.TUNE_SEED, k))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly, info = W.search_route(C, T.TUNE_SEED, k, rng)
        P0 = T.start_positions(C, 2.0, fseed, over, start)
        log = _Log(m, C["rcity"])
        W.fly(m, C, P0, wps, 2.0, fseed, hook=log)
        wp = np.array(log.wp)
        win = (wp >= info["hold0"]) & (wp <= info["hold1"]) & np.array(log.wrap)
        nar += list(np.array(log.sig)[win] < 0.95)
        held += list(np.array(log.stick)[win] > 0.1)
        off += list(np.array(log.off)[win])
    return summarise(nar, held, off)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    out = dict(erks=erks(), scripted=scripted(a.n), scripted_n=a.n, k0=W.K_SCREEN,
               note="While a building was wrapped (ERKS: start line to last find; scripted: the search window). "
                    "narrowed: scale < 0.95; held: |stick| > 0.1; off: the centroid's offset from the wrapped "
                    "building's centre in its ellipse's unit space (the rule's aim test reads 0.8).")
    W.OUT.mkdir(parents=True, exist_ok=True)
    (W.OUT / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
