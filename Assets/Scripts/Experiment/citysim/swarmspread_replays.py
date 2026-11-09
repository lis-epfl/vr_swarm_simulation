"""S9 Street Spread: the page's flights tab data (lessons 30-31).

    python -B -m citysim.swarmspread_replays [--stage all|refly|assemble] [--jobs 12]
    -> experiment_results/swarm_spread/replay/ (cities/, flights/, picks/, index.json)

The cities: BlockShapedPatchCityWorld's (seed 1) and the three held out (seeds 2-4), drawn as S8's page draws them
(swarmshape_replays.shape_payloads): the bounds circles, the enclosing circles every S9 law avoids, the footprints. The
flights: routes from the Unity bench's set (swarmspread_bench_spread), each flown the six ways the bench flew (CONDS),
Unity's own trajectory and the replica's twin from Unity's recorded start, re-flown here with a tape and kept only if
it scores identically to the gate's row.

The picks follow rules written before the bench flew: the typical street route and the typical straight line (of those
meeting three buildings) -- the one whose difference in cohesive speed, adaptive 0.95 minus 0.85, in Unity is nearest
the median of its kind -- and two problems: the route where the scene's setting (A85) lost the most drones in Unity
(then touched buildings most), and the same for the dial at its lowest with the adaptive spread (L85).
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

from citysim import blockcity, playback, swarmspread as X, swarmtune as T  # noqa: E402
from citysim.split_replays import apart_series  # noqa: E402
from citysim.swarmshape_replays import shape_payloads, summary  # noqa: E402
from citysim.swarmtune_replays import DT, Hook, _bearing, _same  # noqa: E402

OUT = X.OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")
GATE = "bench_spread"
RUN = "swarmspread_bench_spread"
# condition: (the bench's parameter set, label)
CONDS = {"f20": ("F20", "d_ref 2.0 held (adaptive off, C35)"),
         "a85": ("A85", "adaptive 0.85: 1.7 in the streets (the scene)"),
         "a90": ("A90", "adaptive 0.90: 1.8 in the streets"),
         "a95": ("A95", "adaptive 0.95: 1.9 in the streets"),
         "f17": ("F17", "d_ref 1.7 held"),
         "l85": ("L85", "dial at 1.4, adaptive 0.85: 1.19 in the streets")}
KIND_TITLE = {"cross": "A straight line across the city", "pass": "Through one block", "street": "Along the streets"}


def gate():
    import pandas as pd
    return pd.read_json(X.OUT / GATE / "bench" / "gate.jsonl", lines=True)


def select():
    """[pick dict] by the rules in the module docstring."""
    t = gate()
    w = t.pivot_table(index=["kind", "k"], columns="set", values=["u_v_coh", "u_lost", "u_contacts", "u_met"],
                      aggfunc="first")
    diff = w["u_v_coh"]["A95"] - w["u_v_coh"]["A85"]
    picks, used = [], set()

    def typical(kind, min_met, why):
        g = diff.loc[kind].dropna()
        met = w["u_met"]["A85"].loc[kind]
        med = float(g.median())
        el = g[met.reindex(g.index) >= min_met]
        k = int((el - med).abs().sort_values(kind="stable").index[0])
        used.add((kind, k))
        return dict(key=f"s9_{kind}", kind=kind, k=k, title=KIND_TITLE[kind], first="a85",
                    why=f"{why}: its difference in cohesive speed, adaptive 0.95 minus 0.85, {float(g.loc[k]):+.2f} m/s, "
                        f"is the nearest to the median of its kind ({med:+.2f} m/s, {len(g)} routes)")

    picks.append(typical("street", 0, "the typical street route"))
    picks.append(typical("cross", 3, "the typical straight line (of those meeting three buildings)"))
    for setl, cond, title in (("A85", "a85", "Problem: the scene's setting"),
                              ("L85", "l85", "Problem: the dial at its lowest")):
        g = (w["u_lost"][setl].to_frame("lost").join(w["u_contacts"][setl].to_frame("con"))
             .drop(index=list(used), errors="ignore").sort_values(["lost", "con"], ascending=False, kind="stable"))
        kind, k = g.index[0]
        used.add((kind, int(k)))
        lost, con = int(g.iloc[0].lost), int(g.iloc[0].con)
        picks.append(dict(key=f"s9_{cond}", kind=kind, k=int(k), title=title, first=cond,
                          why=f"the route where {CONDS[cond][1]} lost the most drones in Unity ({lost}"
                              f"{'' if lost else f'; none anywhere, so the most building contacts, {con}'})"))
    return picks


def bench_flight(cond, kind, k):
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    label = CONDS[cond][0]
    run = RUNS / RUN
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    idx = next(i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{kind}_{k}")
    _, _, params, flights, _, _ = bench.read_results(run)
    fl = next(x for x in flights if x["set"] == label and x["scenario"] == idx)
    eff = {kk: float(f"{v:.7g}") if isinstance(v, float) else v for kk, v in params[label]["effective"].items()}
    return run, label, idx, fl, eff, cfg["droneCount"]


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
        start, wps, L, poly = T.route(C, kind, k, rng)
        g = gate()
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
        same, bad = _same(r, r_stored)
        city = Path(blockcity.path_for(T.params(T.TUNE_SEED))).stem + "__enc"
        net = C["net"]
        wj = [[round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 2), d] for w in wps]
        common = dict(group=pick["key"], city=city, kind="route", unit="route", cond=cond, wps=wj, span=200.0,
                      title=CONDS[cond][1], route=dict(kind=kind, k=k, L=round(L, 1)))
        rep_apart = apart_series(net, [dict(t=t_, P=P_) for t_, _, P_ in rs])
        usmp = [dict(t=float(tr["t"][i]), P=np.where(tr["alive"][i][:, None], np.c_[tr["X"][i], tr["Z"][i]], np.nan))
                for i in range(0, len(tr["t"]), 5)]
        s_rep, s_uni = summary(r, d, eff), summary(dict(u, deaths=fl.get("deaths")), d, eff)
        for s in (s_rep, s_uni):
            s["adaptive"] = bool(eff.get("adaptiveSpread"))
            s["street_factor"] = eff.get("streetSpreadFactor")
        f_rep = playback.flight_json(hook.tape.record(), id=f"{pick['key']}_{cond}_replica", source="replica",
                                     identical=bool(same), apart=rep_apart, summary=s_rep, **common)
        f_uni = playback.flight_json(playback.unity_tape(tr, DT, n == 1), id=f"{pick['key']}_{cond}_unity",
                                     source="unity", run=run.name, scenario=int(idx), apart=apart_series(net, usmp),
                                     summary=s_uni, **common)
        return dict(pick=pick, cond=cond, flights=[f_rep, f_uni], identical=bool(same), mismatch=bad,
                    wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, cond=cond, error=repr(e), trace=traceback.format_exc()[-3000:])


def refly(jobs=12):
    from multiprocessing import Pool
    for dd in ("cities", "flights", "picks"):
        (OUT / dd).mkdir(parents=True, exist_ok=True)
    picks = select()
    work = [(p, c) for p in picks for c in CONDS]
    got = {p["key"]: dict(pick=p, flights={}, identical=True, mismatch={}) for p in picks}
    with Pool(min(jobs, len(work))) as pool:
        for res in pool.imap_unordered(worker, work):
            key = res["pick"]["key"]
            if "error" in res:
                print(f"[swarmspread_replays] {key} {res['cond']}: ERROR {res['error']}\n{res['trace']}", flush=True)
                got[key]["identical"] = False
                continue
            for f in res["flights"]:
                (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":"), allow_nan=False),
                                                                 encoding="utf-8")
            got[key]["flights"][res["cond"]] = [f["id"] for f in res["flights"]]
            got[key]["identical"] = got[key]["identical"] and res["identical"]
            if res["mismatch"]:
                got[key]["mismatch"][res["cond"]] = res["mismatch"]
            print(f"[swarmspread_replays] {key} {res['cond']}: identical to the gate: {res['identical']}"
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
        if not m["identical"] or len(m["flights"]) < len(CONDS):
            print(f"[swarmspread_replays] {p['key']}: not every twin identical to its gate row; left out "
                  f"{m.get('mismatch')}", flush=True)
            continue
        fls = [json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8"))
               for c in CONDS for fid in m["flights"][c]]
        ref = next(f for f in fls if f["cond"] == "a85" and f["source"] == "unity")
        rt = ref["route"]
        about = (f"Route {rt['k']} ({KIND_TITLE[rt['kind']].lower()}, {rt['L']:.0f} m) in the scene's city, picked as "
                 f"{p['why']}. Each way starts from the same spot after Unity's 15 s settle; switch between Unity's "
                 f"flight and the replica's twin flown from Unity's recorded start (the other's drones are drawn as "
                 f"outlines). Every way avoids the enclosing circles.")
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
                                     else f"held out (seed {seed})"))
    (OUT / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[swarmspread_replays] {len(index['groups'])} groups -> {OUT}", flush=True)


def _entry(f):
    e = {k: f.get(k) for k in ("id", "title", "city", "source", "kind", "cond", "n", "identical")}
    e["duration"] = round(f["t0"] + (f["frames"] - 1) * f["dt"], 1)
    e["summary"] = f.get("summary")
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="all", choices=("all", "refly", "assemble"))
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    if a.stage in ("all", "refly"):
        refly(a.jobs)
    if a.stage in ("all", "assemble"):
        assemble()


if __name__ == "__main__":
    main()
