"""S10 Ellipse Avoidance: the page's flights tab data (lessons 30-31).

    python -B -m citysim.swarmellipse_replays [--stage all|refly|assemble] [--jobs 10]
    -> experiment_results/swarm_ellipse/replay/ (cities/, flights/, picks/, index.json)

The cities: BlockShapedPatchCityWorld's (seed 1) and the three held out (seeds 2-4), each drawn twice with the shape
each law avoids: the enclosing circles (`<city>`) and the ellipses at cap 1.5 (`<city>__ell`). The flights are the
Unity bench's routes (swarmellipse_bench1 for the formation, swarmellipse_bench1d for one drone), each flown every way
its bench flew it (CONDS): one drone avoiding the circle or the ellipse; the formation with the circle or the ellipse,
the spread adaptive (the scene) or held. Each is shown twice, Unity's own trajectory and the replica's twin flown from
Unity's recorded start under the values the bench recorded, re-flown here with a tape and kept only if it scores
identically to the gate's row.

The picks follow rules written in the plan, before the bench flew (select):
  s10_wall    the user's question: the along probe 6 m off a wall where the circle stands furthest off that wall
  s10_pinch   the street route through the narrowest gap the circles leave
  s10_narrow  the street route through the street the ellipse narrows most (its cost)
  s10_square  the square-on approach (offset 0) at the longest building, by its footprint's aspect
  s10_angle   the 60 deg run-in at the longest building, flown by the formation and by one drone
  s10_cross   the typical straight line: its difference in cohesive speed, ellipse minus circle (both adaptive), nearest
              the median, of those meeting three buildings
  s10_loss    the route where the ellipse (adaptive) lost the most drones in Unity, if it lost any
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
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, playback, swarmellipse as E, swarmshape as SH, swarmtune as T  # noqa: E402
from citysim.shaped_replays import city_payload  # noqa: E402
from citysim.split_replays import apart_series  # noqa: E402
from citysim.swarmtune_replays import DT, Hook, _bearing, _same, SUMMARY  # noqa: E402

OUT = E.OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")
GATE_RUNS = {"bench1": ("swarmellipse_bench1",), "bench1d": ("swarmellipse_bench1d",)}
CAP = 1.5
# condition: (bench, its parameter set, label, the city drawing its shape)
CONDS = {
    "single_c": ("bench1d", "C35", "one drone, enclosing circles", ""),
    "single_e": ("bench1d", "E15_35", "one drone, ellipses (cap 1.5)", "__ell"),
    "c35a": ("bench1", "C35a", "enclosing circles, adaptive spread (the scene)", ""),
    "e35a": ("bench1", "E15_35a", "ellipses (cap 1.5), adaptive spread", "__ell"),
    "c35": ("bench1", "C35", "enclosing circles, spread held", ""),
    "e35": ("bench1", "E15_35", "ellipses (cap 1.5), spread held", "__ell"),
}
SWARM = ("c35a", "e35a", "c35", "e35")
SINGLE = ("single_c", "single_e")
KIND_TITLE = {"cross": "A straight line across the city", "pass": "Through one block", "street": "Along the streets",
              "face": "Square on at a wall", "angle": "At a wall, at an angle", "along": "One drone along a wall"}
EXTRA = ("face_off", "face_L", "arrived", "t_side", "t_split", "t_wrap", "t_through", "angle", "onset", "closest",
         "along_off", "shell", "standoff", "push", "v_along")


def gate(name):
    import pandas as pd
    return pd.read_json(E.OUT / name / "bench" / "gate.jsonl", lines=True)


def _street_gap_on(poly_pts, rc, E_):
    """The narrowest gap a shape leaves on any interior street the route passes within 15 m of the middle of."""
    best = math.inf
    for ia, ib, pts, _ in E._pairs(T.TUNE_SEED):
        mid = pts[len(pts) // 2]
        if np.min(np.linalg.norm(poly_pts - mid, axis=1)) > 15.0:
            continue
        sd = E.rep.ell_frame(pts, E_, np.array([ia, ib]))[5]
        best = min(best, float(sd.sum(1).min()))
    return best


def select():
    """[pick dict] by the rules in the module docstring."""
    g1, gd = gate("bench1"), gate("bench1d")
    C = T.city(T.TUNE_SEED)
    rc = C["rcity"]
    sh = E.shapes(rc, (CAP,))                       # fits the ellipses, so rc._mvee holds each footprint's aspect
    circ, ell = sh["enclosing circle"], sh[f"ellipse {CAP:g}"]
    aspect = np.array([k for _, k in rc._mvee])
    picks = []
    # s10_wall: the along probe at 6 m where the circle stands furthest off the wall
    a = gd[(gd.set == "C35") & (gd.kind == "along") & (gd.u_along_off == 6.0)].sort_values("u_shell", ascending=False)
    if len(a):
        r = a.iloc[0]
        picks.append(dict(key="s10_wall", kind="along", k=int(r.k), title="The question: one drone along a wall",
                          conds=SINGLE, first="single_c", met=0,
                          why=f"the along probe 6 m off a wall where the circle stands furthest off that wall "
                              f"({float(r.u_shell):.1f} m at the wall's middle; the ellipse "
                              f"{float(gd[(gd.set == 'E15_35') & (gd.k == r.k)].u_shell.iloc[0]):.1f} m)"))
    # street routes: the narrowest circle gap, and the street the ellipse narrows most
    st = sorted(set(int(k) for k in g1[g1.kind == "street"].k))
    rows = []
    for k in st:
        rng = np.random.default_rng(T.route_seed("street", T.TUNE_SEED, k))
        rng.integers(1 << 30)
        start, wps, L, poly = T.route(C, "street", k, rng)
        pts = np.vstack([poly[i] + np.linspace(0, 1, 20)[:, None] * (poly[i + 1] - poly[i]) for i in range(len(poly) - 1)])
        rows.append((k, _street_gap_on(pts, rc, circ), _street_gap_on(pts, rc, ell)))
    rows = [r for r in rows if math.isfinite(r[1])]
    k, gc, ge = min(rows, key=lambda r: r[1])
    picks.append(dict(key="s10_pinch", kind="street", k=k, title="The streets the circles pinch", conds=SWARM,
                      first="c35a", met=0,
                      why=f"the street route through the narrowest gap the circles leave ({gc:.1f} m between the two "
                          f"circles; {ge:.1f} m between the ellipses)"))
    k2, gc2, ge2 = min((r for r in rows if r[0] != k), key=lambda r: r[2] - r[1])
    picks.append(dict(key="s10_narrow", kind="street", k=k2, title="Cost: a street the ellipse narrows", conds=SWARM,
                      first="e35a", met=0,
                      why=f"the street route through the street the ellipse narrows most ({gc2:.1f} m between the circles, "
                          f"{ge2:.1f} m between the ellipses)"))
    # square on and at 60 deg, at the longest building
    f = g1[(g1.set == "C35a") & (g1.kind == "face")].copy()
    f = f[f.u_face_off.abs() < 1e-6]
    f["asp"] = [aspect[int(b)] for b in f.u_face_b]
    r = f.sort_values("asp", ascending=False).iloc[0]
    picks.append(dict(key="s10_square", kind="face", k=int(r.k), title="Square on at a long building", conds=SWARM,
                      first="e35a", met=1,
                      why=f"the square-on approach at the longest building met square on (its footprint's aspect "
                          f"{float(r.asp):.2f})"))
    an = g1[(g1.set == "C35a") & (g1.kind == "angle")].copy()
    an = an[an.u_angle.abs() > 50]
    an["asp"] = [aspect[int(b)] for b in an.u_face_b]
    an = an[an.k.isin(set(gd[gd.kind == "angle"].k))]
    if len(an):
        r = an.sort_values("asp", ascending=False).iloc[0]
        picks.append(dict(key="s10_angle", kind="angle", k=int(r.k), title="At 60° to a long building",
                          conds=SWARM + SINGLE, first="e35a", met=1,
                          why=f"the 60° run-in at the longest building met that way (aspect {float(r.asp):.2f}), flown "
                              f"by the formation and by one drone"))
    # the typical straight line
    w = g1[g1.kind == "cross"].pivot_table(index="k", columns="set", values=["u_v_coh", "u_met"], aggfunc="first")
    diff = (w["u_v_coh"]["E15_35a"] - w["u_v_coh"]["C35a"])
    el = w["u_met"]["C35a"] >= 3
    med = float(diff.median())
    k = int((diff[el] - med).abs().sort_values(kind="stable").index[0])
    picks.append(dict(key="s10_cross", kind="cross", k=k, title=KIND_TITLE["cross"], conds=SWARM, first="e35a",
                      met=int(w["u_met"]["C35a"][k]),
                      why=f"the typical straight line (of those meeting three buildings): its difference in cohesive "
                          f"speed, ellipse minus circle (both adaptive), {float(diff[k]):+.2f} m/s, is the nearest to the "
                          f"median ({med:+.2f} m/s, {len(diff)} routes)"))
    # the pick's worst loss
    lo = g1[(g1.set == "E15_35a") & (g1.u_lost > 0) & ~g1.k.isin([p["k"] for p in picks if p["kind"] != "along"])]
    if len(lo):
        r = lo.sort_values(["u_lost", "u_contacts"], ascending=False).iloc[0]
        picks.append(dict(key="s10_loss", kind=str(r.kind), k=int(r.k), title="Cost: the ellipse's worst loss in Unity",
                          conds=SWARM, first="e35a", met=int(r.u_met),
                          why=f"the route where the ellipse (adaptive) lost the most drones in Unity ({int(r.u_lost)})"))
    return picks


def bench_flight(cond, kind, k):
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    name, label, _, _ = CONDS[cond]
    run = RUNS / GATE_RUNS[name][0]
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    idx = next(i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{kind}_{k}")
    _, _, params, flights, _, _ = bench.read_results(run)
    fl = next(x for x in flights if x["set"] == label and x["scenario"] == idx)
    eff = {kk: float(f"{v:.7g}") if isinstance(v, float) else v for kk, v in params[label]["effective"].items()}
    return run, label, idx, fl, eff, cfg["droneCount"]


def summary(row, d, eff):
    s = {k: row.get(k) for k in SUMMARY + EXTRA}
    bp, bf = row.get("bld_pass_s") or [], row.get("bld_frag_s") or []
    shape = SH.SHAPES[int(eff.get("obstacleShape", 0))]
    s.update(d=d, shape=shape, cap=eff.get("ellipseMaxAspect") if shape == "Ellipse" else None,
             adaptive=bool(eff.get("adaptiveSpread")), d_obs=eff.get("d_obs"),
             bld_pass=float(np.mean(bp)) if len(bp) else None, bld_frag=float(np.mean(bf)) if len(bf) else None,
             deaths=row.get("deaths"))
    return playback.rounded(s, 3)


def route_of(C, kind, k):
    rng = np.random.default_rng(T.route_seed(kind, T.TUNE_SEED, k))
    fseed = int(rng.integers(1 << 30))
    if kind == "face":
        start, wps, L, poly, info = SH.face_route(C, T.TUNE_SEED, k, rng)
    elif kind == "angle":
        start, wps, L, poly, info = E.angle_route(C, T.TUNE_SEED, k, rng)
    elif kind == "along":
        start, wps, L, poly, info = E.along_route(C, T.TUNE_SEED, k, rng)
    else:
        start, wps, L, poly = T.route(C, kind, k, rng)
        info = None
    return fseed, start, wps, L, poly, info


class Both:
    """The tape (Hook) and S10's track (swarmellipse.TrackLog) on one flight; both only read."""

    def __init__(self, hook):
        self.hook, self.log = hook, E.TrackLog()

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
        fseed, start, wps, L, poly, info = route_of(C, kind, k)
        g = gate(CONDS[cond][0])
        grow = g[(g.set == label) & (g.kind == kind) & (g.k == k)].iloc[0].to_dict()
        u = {kk[2:]: v for kk, v in grow.items() if kk.startswith("u_")}
        r_stored = {kk[2:]: v for kk, v in grow.items() if kk.startswith("r_")}
        tr = tierc.read_traj(run / f"traj_{label}_{idx}.csv")
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        over = dict(eff)
        d = float(over.pop("d_ref"))
        m = T.model(over)
        hook = Both(Hook(len(P0), _bearing(np.array(wps[0][:2]) - P0.mean(0))))
        sim, rs, rdone = T.fly(m, C, P0, wps, d, fseed, settle=0.0, hook=hook)
        r = T.score_flight(C, start, wps, poly, L, sim, rs, rdone, sim.n0)
        if kind == "face":
            r.update(SH.face_measures(C, info, rs, rdone))
        elif kind == "angle":
            r.update(SH.face_measures(C, info, rs, rdone), angle=round(info["angle"], 1))
            if sim.n0 == 1:
                r.update(E.approach_measures(C, info, hook.log))
        elif kind == "along":
            r.update(E.along_measures(C, info, hook.log, E.model_shape(m, C["rcity"])))
        same, bad = _same(r, r_stored)
        city = Path(blockcity.path_for(T.params(T.TUNE_SEED))).stem + CONDS[cond][3]
        net = C["net"]
        wj = [[round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 2), d] for w in wps]
        single = n == 1
        common = dict(group=pick["key"], city=city, kind="route", unit="route", cond=cond, wps=wj, span=200.0,
                      title=CONDS[cond][2], route=dict(kind=kind, k=k, L=round(L, 1)))
        rep_apart = None if single else apart_series(net, [dict(t=t_, P=P_) for t_, _, P_ in rs])
        usmp = [dict(t=float(tr["t"][i]), P=np.where(tr["alive"][i][:, None], np.c_[tr["X"][i], tr["Z"][i]], np.nan))
                for i in range(0, len(tr["t"]), 5)]
        uni_apart = None if single else apart_series(net, usmp)
        f_rep = playback.flight_json(hook.hook.tape.record(), id=f"{pick['key']}_{cond}_replica", source="replica",
                                     identical=bool(same), apart=rep_apart, summary=summary(r, d, eff), **common)
        f_uni = playback.flight_json(playback.unity_tape(tr, DT, single), id=f"{pick['key']}_{cond}_unity",
                                     source="unity", run=run.name, scenario=int(idx), apart=uni_apart,
                                     summary=summary(dict(u, deaths=fl.get("deaths")), d, eff), **common)
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
                print(f"[swarmellipse_replays] {key} {res['cond']}: ERROR {res['error']}\n{res['trace']}", flush=True)
                got[key]["identical"] = False
                continue
            for f in res["flights"]:
                (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":"), allow_nan=False),
                                                                 encoding="utf-8")
            got[key]["flights"][res["cond"]] = [f["id"] for f in res["flights"]]
            got[key]["identical"] = got[key]["identical"] and res["identical"]
            if res["mismatch"]:
                got[key]["mismatch"][res["cond"]] = res["mismatch"]
            print(f"[swarmellipse_replays] {key} {res['cond']}: identical to the gate: {res['identical']}"
                  f"{'' if res['identical'] else ' ' + str(res['mismatch'])}", flush=True)
    for key, m in got.items():
        (OUT / "picks" / f"{key}.json").write_text(json.dumps(m, indent=1, allow_nan=False), encoding="utf-8")


def shape_payloads(seed):
    """{suffix: city JSON} the city of `seed` drawn with each shape: the enclosing circles, and the ellipses at CAP."""
    P = T.params(seed)
    base = city_payload(P)
    rc = T.city(seed)["rcity"]
    enc = dict(base, cylinders=playback.city_json("x", blockcity.ARENA, [], [],
                                                   cylinders=np.c_[rc.mec_c, rc.mec_R])["cylinders"])
    Ec = rc.ellipses(CAP)
    ang = np.degrees(np.arctan2(Ec.e[:, 1], Ec.e[:, 0]))
    ell = {k: v for k, v in base.items() if k != "cylinders"}
    ell["ellipses"] = playback.city_json("x", blockcity.ARENA, [], [],
                                         ellipses=np.c_[Ec.c, Ec.a, Ec.b, ang])["ellipses"]
    pairs = E._pairs(seed)
    sh = E.shapes(rc, (CAP,))
    for c, key in ((enc, "enclosing circle"), (ell, f"ellipse {CAP:g}")):
        g = E.street_gaps(rc, pairs, sh[key])
        wd = E.wall_distances(rc, sh[key])
        spread = np.array([max(w) - min(w) for w in wd if w])
        c["stats"] = dict(base["stats"], gap_median=round(float(np.median(g)), 1), under10=int((g < 10).sum()),
                          pairs=len(g), wall_spread=round(float(np.median(spread)), 2),
                          far6=int(sum(1 for w in wd if w and max(w) > 6.0)))
    return {"": enc, "__ell": ell}


def assemble():
    index = dict(built=time.strftime("%Y-%m-%d %H:%M"), conds={c: v[2] for c, v in CONDS.items()}, groups=[],
                 gallery=[])
    for p in select():
        path = OUT / "picks" / f"{p['key']}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if not m["identical"] or len(m["flights"]) < len(p["conds"]):
            print(f"[swarmellipse_replays] {p['key']}: not every twin identical to its gate row; left out "
                  f"{m.get('mismatch')}", flush=True)
            continue
        fls = [json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8"))
               for c in p["conds"] for fid in m["flights"][c]]
        ref = next(f for f in fls if f["cond"] == p["first"] and f["source"] == "unity")
        rt = ref["route"]
        about = (f"Route {rt['k']} ({KIND_TITLE[rt['kind']].lower()}, {rt['L']:.0f} m) in the scene's city, picked as "
                 f"{p['why']}. Each way starts from the same spot after Unity's 15 s settle; switch between Unity's "
                 f"flight and the replica's twin flown from Unity's recorded start (the other's drones are drawn as "
                 f"outlines). With the avoidance shapes on, the map draws the shape the selected way avoids.")
        index["groups"].append(dict(id=p["key"], kind="route", title=p["title"], about=about, first=p["first"],
                                    city=ref["city"], route=rt, why=p["why"], flights=[_entry(f) for f in fls]))
    (OUT / "cities").mkdir(parents=True, exist_ok=True)
    for seed in (T.TUNE_SEED,) + T.HELD_SEEDS:
        stem = Path(blockcity.path_for(T.params(seed))).stem
        pays = shape_payloads(seed)
        for suf, c in pays.items():
            (OUT / "cities" / f"{stem}{suf}.json").write_text(json.dumps(c, separators=(",", ":"), allow_nan=False),
                                                              encoding="utf-8")
        index["gallery"].append(dict(seed=seed, city=stem, stats={s or "circle": c["stats"] for s, c in pays.items()},
                                     what="the scene's city (BlockShapedPatchCityWorld)" if seed == T.TUNE_SEED
                                     else f"held out, the confirmation only (seed {seed})"))
    (OUT / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[swarmellipse_replays] {len(index['groups'])} groups -> {OUT}", flush=True)


def _entry(f):
    e = {k: f.get(k) for k in ("id", "title", "city", "source", "kind", "cond", "n", "identical")}
    e["duration"] = round(f["t0"] + (f["frames"] - 1) * f["dt"], 1)
    e["summary"] = f.get("summary")
    return e


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
