"""S11 Wrap Hold: the page's flights tab data (lessons 30-31).

    python -B -m citysim.swarmwrap_replays [--stage all|refly|assemble] [--jobs 10]
    -> experiment_results/swarm_wrap/replay/ (cities/, flights/, picks/, index.json)

The city is BlockShapedPatchCityWorld's (seed 1) drawn with the ellipses its law avoids; the held-out cities are in the
gallery. The flights are the Unity bench's routes (swarmwrap_bench1), each flown every way the bench flew it (CONDS):
the scene's rule (Off), Hold, HoldUntilLeaving and the spread held. Each is shown twice, Unity's own trajectory and the
replica's twin flown from Unity's recorded start under the values the bench recorded, re-flown here with a tape and
kept only if it scores identically to the gate's row. Every flight carries the spread scale as a series and the
stretches wrapped round a building and leaving one as timeline bands.

The picks follow rules written before the bench flew (select):
  s11_narrow   the user's observation: the search where the scene's rule (Unity) was narrowed for the largest share
               of its hold window while wrapped
  s11_search   the typical search: its difference in wrapped-at-full share through the hold window, Hold minus the
               scene (Unity), nearest the median
  s11_leave    the cost: the search whose scale on leaving, Hold minus the scene (Unity), is the largest
  s11_street   the typical street route: its difference in cohesive speed, Hold minus the scene (Unity), nearest the
               median
  s11_loss     the route where Hold lost the most drones in Unity, if it lost any
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, playback, swarmshape as SH, swarmtune as T, swarmwrap as W  # noqa: E402
from citysim import swarmellipse_replays as ER  # noqa: E402
from citysim.split_replays import apart_series  # noqa: E402
from citysim.swarmtune_replays import DT, Hook, _bearing, _same, SUMMARY  # noqa: E402
from citysim.swarmwrap_bench import route_of  # noqa: E402

OUT = W.OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")
BENCH = "bench1"
RUN_TAG = "swarmwrap_bench1"
# condition: (its parameter set, label)
CONDS = {
    "off": ("E15_35a", "the scene's rule: narrows once the stick's line or the centre is off the building"),
    "hold": ("H15_35a", "Hold: full spread while a building lies inside the formation"),
    "leave": ("L15_35a", "HoldUntilLeaving: Hold, released while steering away from the building"),
    "f20": ("F20", "the spread held at 2.0 everywhere"),
}
ALL = tuple(CONDS)
KIND_TITLE = {"cross": "A straight line across the city", "pass": "Through one block", "street": "Along the streets",
              "face": "Square on at a wall", "search": "A goal search round one building"}
WRAP_KEYS = ("wrap_s", "wrap_full", "wrap_sigma", "wrap_ring", "exit_s", "exit_sigma", "sigma_mean", "loss_events",
             "loss_wrap", "loss_exit", "loss_other", "hold_s", "hold_wrapped", "hold_full", "hold_sigma", "leave_v",
             "leave_sigma")


def gate():
    import pandas as pd
    return pd.read_json(W.OUT / BENCH / "bench" / "gate.jsonl", lines=True)


def select():
    """The picks (see the docstring): [dict(key, kind, k, title, why, conds, first)]."""
    g = gate()
    u = {c: g[g.set == CONDS[c][0]].set_index(["kind", "k"]) for c in ALL}
    picks = []
    s_off, s_hold = u["off"].loc["search"], u["hold"].loc["search"]
    narrowed = (s_off.u_hold_wrapped - s_off.u_hold_full).astype(float)
    k = int(narrowed.idxmax())
    picks.append(dict(key="s11_narrow", kind="search", k=k, first="off",
                      title="Where the scene's rule narrowed while wrapped round the building",
                      why=f"the search where the scene's rule held a narrowed spread for the largest share of its hold "
                          f"window while wrapped ({narrowed.max():.0%})"))
    dfull = (s_hold.u_hold_full - s_off.u_hold_full).astype(float).dropna()
    k = int((dfull - dfull.median()).abs().idxmin())
    picks.append(dict(key="s11_search", kind="search", k=k, first="hold", title="A typical goal search",
                      why=f"the typical search: Hold's gain in the hold window's share at full spread nearest the "
                          f"median ({dfull.median():+.0%})"))
    dleave = (s_hold.u_leave_sigma - s_off.u_leave_sigma).astype(float).dropna()
    k = int(dleave.idxmax())
    picks.append(dict(key="s11_leave", kind="search", k=k, first="hold", title="Leaving wide: the cost",
                      why=f"the search Hold leaves widest against the scene (scale on leaving +{dleave.max():.2f})"))
    st_off, st_hold = u["off"].loc["street"], u["hold"].loc["street"]
    dv = (st_hold.u_v_coh - st_off.u_v_coh).astype(float).dropna()
    k = int((dv - dv.median()).abs().idxmin())
    picks.append(dict(key="s11_street", kind="street", k=k, first="hold", title="A typical street route",
                      why=f"the typical street route: Hold's difference in cohesive speed nearest the median "
                          f"({dv.median():+.2f} m/s)"))
    lost = u["hold"].u_lost.astype(float)
    if lost.max() > 0:
        kind, k = lost.idxmax()
        picks.append(dict(key="s11_loss", kind=kind, k=int(k), first="hold", title="Hold's worst loss in Unity",
                          why=f"the route where Hold lost the most drones in Unity ({int(lost.max())})"))
    for p in picks:
        p["conds"] = list(ALL)
    return picks


def bench_flight(cond, kind, k):
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    label = CONDS[cond][0]
    run = RUNS / RUN_TAG
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    idx = next(i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{kind}_{k}")
    _, _, params, flights, _, _ = bench.read_results(run)
    fl = next(x for x in flights if x["set"] == label and x["scenario"] == idx)
    eff = {kk: float(f"{v:.7g}") if isinstance(v, float) else v for kk, v in params[label]["effective"].items()}
    return run, label, idx, fl, eff, cfg["droneCount"]


def summary(row, d, eff):
    s = {k: row.get(k) for k in SUMMARY + WRAP_KEYS}
    s.update(d=d, rule=W.WRAP_RULES[int(eff.get("spreadWrapRule", 0))], adaptive=bool(eff.get("adaptiveSpread")),
             deaths=row.get("deaths"))
    return playback.rounded(s, 3)


def spans(t, flag):
    """[(start, end)] of the runs of True in flag over the times t."""
    t, flag = np.asarray(t, float), np.asarray(flag, bool)
    out, k = [], 0
    while k < len(flag):
        if not flag[k]:
            k += 1
            continue
        k1 = k
        while k1 < len(flag) and flag[k1]:
            k1 += 1
        out.append([round(float(t[k]), 2), round(float(t[min(k1, len(t) - 1)]), 2)])
        k = k1
    return out


def wrap_overlay(t, sig, wrap):
    """The flight's series (the spread scale) and bands (wrapped, leaving) for replay_view.js."""
    t = np.asarray(t, float)
    ph = W.phases(t, wrap)
    q = np.clip(np.round((np.asarray(sig, float) - 0.8) / 0.2 * 255), 0, 255).astype("u1")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    series = [dict(label="spread scale (x d_ref 2.0)", t0=round(float(t[0]), 2), dt=round(dt, 3),
                   q=playback.b64(q, "u1"), lo=0.8, hi=1.0, digits=2)]
    bands = [dict(label="wrapped round a building", tone="good", spans=spans(t, ph == 0)),
             dict(label="leaving one (10 s)", tone="warn", spans=spans(t, ph == 1))]
    return series, bands


class Both:
    """The tape (Hook) and the WrapLog on one flight; both only read."""

    def __init__(self, hook, log):
        self.hook, self.log = hook, log

    def start(self, sim):
        self.hook.start(sim)
        self.log.start(sim)

    def step(self, sim, t, cmd, wp):
        self.hook.step(sim, t, cmd, wp)
        self.log.step(sim, t, cmd, wp)


def worker(args):
    pick, cond = args
    from citysim import tierc
    t0 = time.time()
    try:
        kind, k = pick["kind"], pick["k"]
        run, label, idx, fl, eff, n = bench_flight(cond, kind, k)
        C = T.city(T.TUNE_SEED)
        rng = np.random.default_rng(T.route_seed(kind, T.TUNE_SEED, k))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly, info = route_of(C, kind, k, rng)
        g = gate()
        grow = g[(g.set == label) & (g.kind == kind) & (g.k == k)].iloc[0].to_dict()
        u = {kk[2:]: v for kk, v in grow.items() if kk.startswith("u_")}
        r_stored = {kk[2:]: v for kk, v in grow.items() if kk.startswith("r_")}
        tr = tierc.read_traj(run / f"traj_{label}_{idx}.csv")
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        over = dict(eff)
        d = float(over.pop("d_ref"))
        m = T.model(over)
        hook = Both(Hook(len(P0), _bearing(np.array(wps[0][:2]) - P0.mean(0))), W.WrapLog(m, C["rcity"]))
        fly = W.fly if kind == "search" else T.fly
        sim, rs, rdone = fly(m, C, P0, wps, d, fseed, settle=0.0, hook=hook)
        r = T.score_flight(C, start, wps, poly, L, sim, rs, rdone, sim.n0)
        r.update(W.log_measures(hook.log))
        if kind == "face":
            r.update(SH.face_measures(C, info, rs, rdone))
        elif kind == "search":
            r.update(W.search_measures(info, rdone, hook.log.t, hook.log.sig, hook.log.wrap, hook.log.wp, rs))
        same, bad = _same(r, r_stored)
        city = Path(blockcity.path_for(T.params(T.TUNE_SEED))).stem + "__ell"
        net = C["net"]
        wj = [[round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 2), d] for w in wps]
        common = dict(group=pick["key"], city=city, kind="route", unit="route", cond=cond, wps=wj, span=200.0,
                      title=CONDS[cond][1], route=dict(kind=kind, k=k, L=round(L, 1)))
        r_series, r_bands = wrap_overlay(hook.log.t, hook.log.sig, hook.log.wrap)
        cent = W.building_centres(m, C["rcity"])
        uw = [W.wrapped_at(np.c_[tr["X"][i], tr["Z"][i]][tr["alive"][i]], cent) for i in range(len(tr["t"]))]
        u_series, u_bands = wrap_overlay(tr["t"], tr["spread"], uw)
        rep_apart = apart_series(net, [dict(t=t_, P=P_) for t_, _, P_ in rs])
        usmp = [dict(t=float(tr["t"][i]), P=np.where(tr["alive"][i][:, None], np.c_[tr["X"][i], tr["Z"][i]], np.nan))
                for i in range(0, len(tr["t"]), 5)]
        f_rep = playback.flight_json(hook.hook.tape.record(), id=f"{pick['key']}_{cond}_replica", source="replica",
                                     identical=bool(same), apart=rep_apart, summary=summary(r, d, eff),
                                     series=r_series, bands=r_bands, **common)
        f_uni = playback.flight_json(playback.unity_tape(tr, DT, False), id=f"{pick['key']}_{cond}_unity",
                                     source="unity", run=run.name, scenario=int(idx), apart=apart_series(net, usmp),
                                     summary=summary(dict(u, deaths=fl.get("deaths")), d, eff), series=u_series,
                                     bands=u_bands, **common)
        return dict(pick=pick, cond=cond, flights=[f_rep, f_uni], identical=bool(same), mismatch=bad,
                    wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, cond=cond, error=repr(e), trace=traceback.format_exc()[-3000:])


def refly(jobs=10):
    from multiprocessing import Pool
    for dd in ("cities", "flights", "picks"):
        (OUT / dd).mkdir(parents=True, exist_ok=True)
    picks = select()
    work = [(p, c) for p in picks for c in p["conds"]]
    got = {p["key"]: dict(pick=p, flights={}, identical=True, mismatch={}) for p in picks}
    with Pool(min(jobs, len(work))) as pool:
        for res in pool.imap_unordered(worker, work):
            key = res["pick"]["key"]
            if "error" in res:
                print(f"[swarmwrap_replays] {key} {res['cond']}: ERROR {res['error']}\n{res['trace']}", flush=True)
                got[key]["identical"] = False
                continue
            for f in res["flights"]:
                (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":"), allow_nan=False),
                                                                 encoding="utf-8")
            got[key]["flights"][res["cond"]] = [f["id"] for f in res["flights"]]
            got[key]["identical"] = got[key]["identical"] and res["identical"]
            if res["mismatch"]:
                got[key]["mismatch"][res["cond"]] = res["mismatch"]
            print(f"[swarmwrap_replays] {key} {res['cond']}: identical to the gate: {res['identical']}"
                  f"{'' if res['identical'] else ' ' + str(res['mismatch'])}", flush=True)
    for key, m in got.items():
        (OUT / "picks" / f"{key}.json").write_text(json.dumps(m, indent=1, allow_nan=False), encoding="utf-8")


def assemble():
    index = dict(built=time.strftime("%Y-%m-%d %H:%M"), conds={c: v[1] for c, v in CONDS.items()}, groups=[],
                 gallery=[])
    for p in select():
        path = OUT / "picks" / f"{p['key']}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if not m["identical"] or len(m["flights"]) < len(p["conds"]):
            print(f"[swarmwrap_replays] {p['key']}: not every twin identical to its gate row; left out "
                  f"{m.get('mismatch')}", flush=True)
            continue
        fls = [json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8"))
               for c in p["conds"] for fid in m["flights"][c]]
        ref = next(f for f in fls if f["cond"] == p["first"] and f["source"] == "unity")
        rt = ref["route"]
        about = (f"Route {rt['k']} ({KIND_TITLE[rt['kind']].lower()}, {rt['L']:.0f} m of route) in the scene's city, "
                 f"picked as {p['why']}. Each way starts from the same spot after Unity's 15 s settle; switch between "
                 f"Unity's flight and the replica's twin flown from Unity's recorded start (the other's drones are drawn "
                 f"as outlines). The timeline's bands mark the stretches wrapped round a building and the 10 s after "
                 f"leaving one, and its line is the spread scale.")
        index["groups"].append(dict(id=p["key"], kind="route", title=p["title"], about=about, first=p["first"],
                                    city=ref["city"], route=rt, why=p["why"], flights=[ER._entry(f) for f in fls]))
    (OUT / "cities").mkdir(parents=True, exist_ok=True)
    for seed in (T.TUNE_SEED,) + T.HELD_SEEDS:
        stem = Path(blockcity.path_for(T.params(seed))).stem
        pays = ER.shape_payloads(seed)
        c = pays["__ell"]
        (OUT / "cities" / f"{stem}__ell.json").write_text(json.dumps(c, separators=(",", ":"), allow_nan=False),
                                                          encoding="utf-8")
        index["gallery"].append(dict(seed=seed, city=f"{stem}__ell", stats={"ellipse": c["stats"]},
                                     what="the scene's city (BlockShapedPatchCityWorld)" if seed == T.TUNE_SEED
                                     else f"held out, the confirmation only (seed {seed})"))
    (OUT / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[swarmwrap_replays] {len(index['groups'])} groups -> {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="all", choices=("all", "refly", "assemble"))
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    if a.stage in ("all", "refly"):
        refly(a.jobs)
    if a.stage in ("all", "assemble"):
        assemble()


if __name__ == "__main__":
    main()
