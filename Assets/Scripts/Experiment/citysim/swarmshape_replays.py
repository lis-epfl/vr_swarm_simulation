"""S8 Footprint Avoidance: the page's flights tab data (lessons 30-31).

    python -B -m citysim.swarmshape_replays [--stage all|refly|assemble] [--jobs 10]
    -> experiment_results/swarm_shape/replay/ (cities/, flights/, picks/, index.json)

The cities: BlockShapedPatchCityWorld's (seed 1) and the three held out (seeds 2-4, S6's), each drawn three ways, with
the shape each law avoids: the circles round the buildings' bounds (`<city>`), the smallest circles round their
footprints (`<city>__enc`) and the footprints alone (`<city>__fp`). Each flight points at the city drawn with its own
shape. The flights: routes from the Unity bench's set (swarmshape_bench_shape, swarmshape_bench_single), each flown
seven ways (CONDS): one drone avoiding the bounds circles and the enclosing circles, and the swarm with the bounds
circles and no slide (the scene today), the bounds circles with the slide (S7's pick), the footprint with and without
the slide, and the enclosing circle with the slide (the rule's pick, or C25 without one). Each is shown twice: Unity's
own trajectory and the replica's twin, flown from Unity's recorded (settled) start under the values the bench
recorded for that set, re-flown here with a tape and kept only if it scores identically to the gate's row.

The picks follow rules written before the bench flew (select): the typical straight line and the typical street route
-- the one whose enclosing-circle-minus-cylinder difference in cohesive speed (Unity) is nearest the median of its
kind, among straight lines meeting at least three buildings -- and the two problems the shapes are about: the
square-on approach (offset 0) where the footprint without the slide took longest to get past its building in Unity
(the stall the cylinder was chosen to avoid), and the route where the footprint with the slide lost the most drones in
Unity (drones in a lane along a wall), or touched buildings most if it lost none.
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

from citysim import blockcity, playback, swarmshape as X, swarmtune as T  # noqa: E402
from citysim.shaped_replays import city_payload  # noqa: E402
from citysim.split_replays import apart_series  # noqa: E402
from citysim.swarmtune_replays import DT, Hook, _bearing, _same, SUMMARY  # noqa: E402

OUT = X.OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")         # the bench copy's runs (run_bench.ps1)
# each gate's Unity runs: the rule picked C35a after the first bench had flown, so it and the one drone at C35 flew the
# same scenarios in a second run each, gated together with the first
GATE_RUNS = {"bench_shape": ("swarmshape_bench_shape", "swarmshape_bench_extra"),
             "bench_single": ("swarmshape_bench_single", "swarmshape_bench_single_extra"),
             "bench_face12": ("swarmshape_bench_face12", "swarmshape_bench_face12_extra")}
FACE_KEYS = ("face_off", "face_L", "arrived") + X.FACE_KEYS


def pick_set():
    """The enclosing-circle set the replays fly: the rule's pick, or C25 without one."""
    v = X.OUT / "confirm" / "verdicts.json"
    if v.exists():
        p = json.loads(v.read_text(encoding="utf-8")).get("pick")
        if p and p.split()[0].startswith("C"):
            return p.split()[0]
    return "C25"


def conds():
    """condition: (bench run, its parameter set, label, the city drawing its shape)."""
    c = pick_set()
    return {
        "single_cyl": ("bench_single", "A1", "one drone, bounds circles", ""),
        "single_enc": ("bench_single", "C35", "one drone, enclosing circles (d_obs 0.35)", "__enc"),
        "a0": ("bench_shape", "A0", "bounds circles, no slide (the scene today)", ""),
        "a1": ("bench_shape", "A1", "bounds circles + slide (S7's pick)", ""),
        "b0": ("bench_shape", "B0", "footprint, no slide", "__fp"),
        "b2": ("bench_shape", "B2", "footprint + slide", "__fp"),
        "c": ("bench_shape", c, f"enclosing circles + slide ({c})", "__enc"),
    }


KIND_TITLE = {"cross": "A straight line across the city", "pass": "Through one block", "street": "Along the streets",
              "face": "Square on at a wall"}


def gate(name):
    import pandas as pd
    return pd.read_json(X.OUT / name / "bench" / "gate.jsonl", lines=True)


def wide():
    """Per bench route (kind, k): the Unity measures the picks are chosen by."""
    import pandas as pd
    t = gate("bench_shape")
    c = pick_set()
    key = ["kind", "k"]
    t = t.assign(u_t_through_c=t["u_t_through"].astype(float).fillna(X.FACE_CAP_S) if "u_t_through" in t else np.nan)
    vt = t.pivot_table(index=key, columns="set", values=["u_v_coh", "u_contacts", "u_met", "u_lost", "u_t_through_c",
                                                         "u_face_off"], aggfunc="first")
    return pd.DataFrame({"diff": vt["u_v_coh"][c] - vt["u_v_coh"]["A1"], "met": vt["u_met"]["A1"],
                         "b0_through": vt["u_t_through_c"]["B0"], "off": vt["u_face_off"]["A1"],
                         "b2_lost": vt["u_lost"]["B2"], "b2_contacts": vt["u_contacts"]["B2"]})


def select():
    """[pick dict] by the rules in the module docstring."""
    w = wide()
    picks, used = [], set()

    def typical(kind, eligible, why):
        g = w.loc[kind]
        med = float(g["diff"].median())
        cand = g[eligible(g)].assign(dev=lambda x: (x["diff"] - med).abs()).sort_values(["dev"], kind="stable")
        k = int(cand.index[0])
        used.add((kind, k))
        return dict(key=f"s8_{kind}", kind=kind, k=k, title=KIND_TITLE[kind], first="c",
                    why=f"{why}: its difference in cohesive speed, enclosing circles minus bounds circles (both with the "
                        f"slide), {float(g.loc[k, 'diff']):+.2f} m/s, is the nearest to the median of its kind "
                        f"({med:+.2f} m/s, {len(g)} routes)", met=int(g.loc[k, "met"]))

    picks.append(typical("cross", lambda g: g.met >= 3, "the typical straight line (of those meeting three buildings)"))
    picks.append(typical("street", lambda g: g.met >= 0, "the typical street route"))
    f = w.loc["face"]
    f = f[f.off.abs() < 1e-6].sort_values("b0_through", ascending=False, kind="stable")
    k = int(f.index[0])
    used.add(("face", k))
    picks.append(dict(key="s8_face", kind="face", k=k, title="Problem: square on at a wall", first="b0", met=1,
                      why=f"the square-on approach where the footprint without the slide took longest to get past its "
                          f"building in Unity ({float(f.iloc[0].b0_through):.0f} s from arriving, censored at "
                          f"{X.FACE_CAP_S:.0f})"))
    g = w.drop(index=list(used)).sort_values(["b2_lost", "b2_contacts"], ascending=False, kind="stable")
    kind, k = g.index[0]
    lost, con = int(g.iloc[0].b2_lost), int(g.iloc[0].b2_contacts)
    picks.append(dict(key="s8_lane", kind=kind, k=int(k), title="Problem: the footprint's wall lanes", first="b2",
                      met=int(g.iloc[0].met),
                      why=f"the route where the footprint with the slide lost the most drones in Unity ({lost}"
                          f"{'' if lost else f'; touching buildings {con} times'})"))
    return picks


def bench_flight(cond, kind, k):
    """(run dir, its parameter-set label, scenario index, the flight line, the set's recorded SwarmManager values)."""
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    name, label, _, _ = conds()[cond]
    for rn in GATE_RUNS[name]:
        run = RUNS / rn
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if any(ps["label"] == label for ps in cfg["paramSets"]):
            break
    idx = next(i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{kind}_{k}")
    _, _, params, flights, _, _ = bench.read_results(run)
    fl = next(x for x in flights if x["set"] == label and x["scenario"] == idx)
    eff = {kk: float(f"{v:.7g}") if isinstance(v, float) else v for kk, v in params[label]["effective"].items()}
    return run, label, idx, fl, eff, cfg["droneCount"]


def summary(row, d, eff):
    s = {k: row.get(k) for k in SUMMARY + FACE_KEYS}
    bp, bf = row.get("bld_pass_s") or [], row.get("bld_frag_s") or []
    shape = X.SHAPES[int(eff.get("obstacleShape", 0))]
    s.update(d=d, shape=shape, slide=bool(eff.get("slideRoundBuildings")), d_obs=eff.get("d_obs"),
             bld_pass=float(np.mean(bp)) if len(bp) else None, bld_frag=float(np.mean(bf)) if len(bf) else None,
             deaths=row.get("deaths"))
    return playback.rounded(s, 3)


def route_of(C, kind, k):
    rng = np.random.default_rng(T.route_seed(kind, T.TUNE_SEED, k))
    fseed = int(rng.integers(1 << 30))
    if kind == "face":
        start, wps, L, poly, face = X.face_route(C, T.TUNE_SEED, k, rng)
        return fseed, start, wps, L, poly, face
    start, wps, L, poly = T.route(C, kind, k, rng)
    return fseed, start, wps, L, poly, None


def worker(args):
    pick, cond = args
    from citysim import tierc
    t0 = time.time()
    try:
        kind, k = pick["kind"], pick["k"]
        run, label, idx, fl, eff, n = bench_flight(cond, kind, k)
        C = T.city(T.TUNE_SEED)
        fseed, start, wps, L, poly, face = route_of(C, kind, k)
        g = gate(conds()[cond][0])
        grow = g[(g.set == label) & (g.kind == kind) & (g.k == k)].iloc[0].to_dict()
        u = {kk[2:]: v for kk, v in grow.items() if kk.startswith("u_")}
        r_stored = {kk[2:]: v for kk, v in grow.items() if kk.startswith("r_")}
        tr = tierc.read_traj(run / f"traj_{label}_{idx}.csv")
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        over = dict(eff)
        d = float(over.pop("d_ref"))
        hook = Hook(len(P0), _bearing(np.array(wps[0][:2]) - P0.mean(0)))
        sim, rs, rdone = T.fly(T.model(over), C, P0, wps, d, fseed, settle=0.0, hook=hook)
        r = T.score_flight(C, start, wps, poly, L, sim, rs, rdone, sim.n0)
        if face is not None:
            r.update(X.face_measures(C, face, rs, rdone))
        same, bad = _same(r, r_stored)
        city = Path(blockcity.path_for(T.params(T.TUNE_SEED))).stem + conds()[cond][3]
        net = C["net"]
        wj = [[round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 2), d] for w in wps]
        single = n == 1
        common = dict(group=pick["key"], city=city, kind="route", unit="route", cond=cond, wps=wj, span=200.0,
                      title=conds()[cond][2], route=dict(kind=kind, k=k, L=round(L, 1)))
        rep_apart = None if single else apart_series(net, [dict(t=t_, P=P_) for t_, _, P_ in rs])
        usmp = [dict(t=float(tr["t"][i]), P=np.where(tr["alive"][i][:, None], np.c_[tr["X"][i], tr["Z"][i]], np.nan))
                for i in range(0, len(tr["t"]), 5)]
        uni_apart = None if single else apart_series(net, usmp)
        f_rep = playback.flight_json(hook.tape.record(), id=f"{pick['key']}_{cond}_replica", source="replica",
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
    work = [(p, c) for p in picks for c in conds()]
    got = {p["key"]: dict(pick=p, flights={}, identical=True, mismatch={}) for p in picks}
    with Pool(min(jobs, len(work))) as pool:
        for res in pool.imap_unordered(worker, work):
            key = res["pick"]["key"]
            if "error" in res:
                print(f"[swarmshape_replays] {key} {res['cond']}: ERROR {res['error']}\n{res['trace']}", flush=True)
                got[key]["identical"] = False
                continue
            for f in res["flights"]:
                (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":"), allow_nan=False),
                                                                 encoding="utf-8")
            got[key]["flights"][res["cond"]] = [f["id"] for f in res["flights"]]
            got[key]["identical"] = got[key]["identical"] and res["identical"]
            if res["mismatch"]:
                got[key]["mismatch"][res["cond"]] = res["mismatch"]
            print(f"[swarmshape_replays] {key} {res['cond']}: identical to the gate: {res['identical']}"
                  f"{'' if res['identical'] else ' ' + str(res['mismatch'])}", flush=True)
    for key, m in got.items():
        (OUT / "picks" / f"{key}.json").write_text(json.dumps(m, indent=1, allow_nan=False), encoding="utf-8")


def shape_payloads(seed):
    """{suffix: city JSON} the city of `seed` drawn with each law's shape: the bounds circles (S7's payload), the
    enclosing circles, and the footprints alone."""
    P = T.params(seed)
    base = city_payload(P)
    rc = T.city(seed)["rcity"]
    enc = dict(base, cylinders=playback.city_json("x", blockcity.ARENA, [], [],
                                                   cylinders=np.c_[rc.mec_c, rc.mec_R])["cylinders"])
    enc["stats"] = dict(base["stats"], cyl_r=round(float(np.median(rc.mec_R)), 1))
    fp = {k: v for k, v in base.items() if k != "cylinders"}
    gaps = X.street_gaps(seed)
    for c, key in ((base, "bounds circle"), (enc, "enclosing circle"), (fp, "facades")):
        g = gaps[key]
        c["stats"] = dict(c["stats"], gap_median=round(float(np.median(g)), 1), closed=int((g < 0).sum()),
                          narrow=int((g < 6).sum()), pairs=len(g))
    return {"": base, "__enc": enc, "__fp": fp}


def assemble():
    cd = conds()
    index = dict(built=time.strftime("%Y-%m-%d %H:%M"), conds={c: v[2] for c, v in cd.items()}, groups=[], gallery=[])
    for p in select():
        path = OUT / "picks" / f"{p['key']}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if not m["identical"] or len(m["flights"]) < len(cd):
            print(f"[swarmshape_replays] {p['key']}: not every twin identical to its gate row; left out "
                  f"{m.get('mismatch')}", flush=True)
            continue
        fls = []
        for c in cd:
            for fid in m["flights"][c]:
                fls.append(json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8")))
        ref = next(f for f in fls if f["cond"] == "a1" and f["source"] == "unity")
        rt = ref["route"]
        about = (f"Route {rt['k']} ({KIND_TITLE[rt['kind']].lower()}, {rt['L']:.0f} m) in the scene's city, picked as "
                 f"{p['why']}. Each way starts from the same spot after Unity's 15 s settle; switch between Unity's "
                 f"flight and the replica's twin flown from Unity's recorded start (the other's drones are drawn as "
                 f"outlines). The map draws the shape the selected way's law avoids.")
        index["groups"].append(dict(id=p["key"], kind="route", title=p["title"], about=about, first=p["first"],
                                    city=ref["city"], route=rt, why=p["why"], flights=[_entry(f) for f in fls]))
    (OUT / "cities").mkdir(parents=True, exist_ok=True)
    for seed in (T.TUNE_SEED,) + T.HELD_SEEDS:
        stem = Path(blockcity.path_for(T.params(seed))).stem
        pays = shape_payloads(seed)
        for suf, c in pays.items():
            (OUT / "cities" / f"{stem}{suf}.json").write_text(json.dumps(c, separators=(",", ":"), allow_nan=False),
                                                              encoding="utf-8")
        index["gallery"].append(dict(seed=seed, city=stem, stats={s or "bounds": c["stats"] for s, c in pays.items()},
                                     what="the scene's city (BlockShapedPatchCityWorld)" if seed == T.TUNE_SEED
                                     else f"held out, the confirmation only (seed {seed})"))
    (OUT / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[swarmshape_replays] {len(index['groups'])} groups -> {OUT}", flush=True)


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
