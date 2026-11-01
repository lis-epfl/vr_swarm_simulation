"""S7 Building Flow: the page's flights tab data (lessons 30-31).

    python -B -m citysim.swarmflow_replays [--stage all|refly|assemble] [--jobs 10]
    -> experiment_results/swarm_flow/replay/ (cities/, flights/, picks/, index.json)

The cities: BlockShapedPatchCityWorld's (seed 1) and the three held out (seeds 2-4, S6's), with the avoidance circles
Unity's law draws round each prism. The flights: routes from the Unity bench's set (swarmflow_bench_flow, swarmflow_bench_single), each flown seven ways
(CONDS): one drone with the switches off and with the slide on, and the swarm with all three off (T3, S6's scene),
each switch alone, and all three. The adaptive spread flew the confirmation's setting (0.8 of d_ref, a 1 s filter); it
ships at 0.85 and 3 s since the tuning that followed (docs/swarm_flow/README.md, section 6). Each is shown twice: Unity's own trajectory and the replica's twin, flown from Unity's
recorded (settled) start under the values the bench recorded for that set, re-flown here with a tape and kept only if
it scores identically to the gate's row (swarmtune.bench_job wrote those).

The picks follow rules written before the bench flew (select): the typical route of each kind -- the one whose
all-three-minus-T3 difference in cohesive speed (Unity) is nearest the median of its kind, among the cross routes that
meet at least three buildings and the pass routes that meet one -- and the two problems the switches are for: the
route where T3 touched buildings most in Unity (a straggler pulled through a facade), and the one where one drone with
the switches off stalled longest (flown at a building, the shield stops it).
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

from citysim import blockcity, playback, swarmflow as F, swarmtune as T  # noqa: E402
from citysim.shaped_replays import city_payload  # noqa: E402
from citysim.split_replays import apart_series  # noqa: E402
from citysim.swarmtune_replays import DT, Hook, _bearing, _same, SUMMARY  # noqa: E402

OUT = F.OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")         # the bench copy's runs (run_bench.ps1)
# condition: (bench run, its parameter set, label)
CONDS = {
    "single_off": ("bench_single", "T3", "one drone, switches off"),
    "single_slide": ("bench_single", "slide", "one drone, slide on"),
    "t3": ("bench_flow", "T3", "the swarm, switches off (T3)"),
    "slide": ("bench_flow", "slide", "slide round buildings"),
    "route": ("bench_flow", "route", "route stragglers"),
    "adaptive": ("bench_flow", "adaptive", "adaptive spread (0.8 of d_ref, 1 s filter)"),
    "all": ("bench_flow", "all_three", "all three on (adaptive at 0.8, 1 s)"),
}
KIND_TITLE = {"cross": "A straight line across the city", "pass": "Through one block", "street": "Along the streets"}


def gate(name):
    import pandas as pd
    return pd.read_json(F.OUT / name / "bench" / "gate.jsonl", lines=True)


def wide():
    """Per bench route (kind, k): the Unity measures the picks are chosen by."""
    import pandas as pd
    t, s = gate("bench_flow"), gate("bench_single")
    key = ["kind", "k"]
    vt = t.pivot_table(index=key, columns="set", values=["u_v_coh", "u_contacts", "u_met"], aggfunc="first")
    vs = s.pivot_table(index=key, columns="set", values=["u_stall_s"], aggfunc="first")
    return pd.DataFrame({"diff": vt["u_v_coh"]["all_three"] - vt["u_v_coh"]["T3"], "met": vt["u_met"]["T3"],
                         "contacts": vt["u_contacts"]["T3"], "stall": vs["u_stall_s"]["T3"]})


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
        return dict(key=f"s7_{kind}", kind=kind, k=k, title=KIND_TITLE[kind], first="all",
                    why=f"{why}: its all-three-minus-T3 difference ({float(g.loc[k, 'diff']):.2f} m/s) is the nearest "
                        f"to the median of its kind ({med:.2f} m/s, {len(g)} routes)", met=int(g.loc[k, "met"]))

    picks.append(typical("cross", lambda g: g.met >= 3, "the typical straight line (of those meeting three buildings)"))
    picks.append(typical("pass", lambda g: g.met >= 1, "the typical block (of those where a building is met)"))
    picks.append(typical("street", lambda g: g.met >= 0, "the typical street route"))
    f = w.drop(index=list(used)).sort_values(["contacts", "met"], ascending=False, kind="stable")
    kind, k = f.index[0]
    used.add((kind, int(k)))
    picks.append(dict(key="s7_contact", kind=kind, k=int(k), title="Problem: a drone pulled into a facade",
                      first="t3", met=int(f.iloc[0].met),
                      why=f"the route where the swarm with the switches off touched buildings most in Unity "
                          f"({int(f.iloc[0].contacts)} contact{'' if int(f.iloc[0].contacts) == 1 else 's'}; a "
                          f"straggler, its nearest neighbour 27 m away, pushed into a facade by cohesion against the "
                          f"saturated obstacle field)"))
    f = w.drop(index=list(used)).sort_values(["stall"], ascending=False, kind="stable")
    kind, k = f.index[0]
    picks.append(dict(key="s7_stall", kind=kind, k=int(k), title="Problem: one drone stalls at a building",
                      first="single_off", met=int(f.iloc[0].met),
                      why=f"the route where one drone with the switches off stalled longest in Unity "
                          f"({float(f.iloc[0].stall):.0f} s under 1 m/s)"))
    return picks


def bench_flight(cond, kind, k):
    """(run dir, its parameter-set label, scenario index, the flight line, the set's recorded SwarmManager values)."""
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    name, label, _ = CONDS[cond]
    run = RUNS / f"swarmflow_{name}"
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    idx = next(i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{kind}_{k}")
    _, _, params, flights, _, _ = bench.read_results(run)
    fl = next(x for x in flights if x["set"] == label and x["scenario"] == idx)
    eff = {kk: float(f"{v:.7g}") if isinstance(v, float) else v for kk, v in params[label]["effective"].items()}
    return run, label, idx, fl, eff, cfg["droneCount"]


def summary(row, d, cond, eff):
    s = {k: row.get(k) for k in SUMMARY}
    bp, bf = row.get("bld_pass_s") or [], row.get("bld_frag_s") or []
    on = [n for n, f in (("slide", "slideRoundBuildings"), ("route", "routeStragglers"), ("adaptive", "adaptiveSpread"))
          if eff.get(f)]
    s.update(d=d, switches=" + ".join(on) or "off", bld_pass=float(np.mean(bp)) if len(bp) else None,
             bld_frag=float(np.mean(bf)) if len(bf) else None, deaths=row.get("deaths"))
    return playback.rounded(s, 3)


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
        g = gate(CONDS[cond][0])
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
        city = Path(blockcity.path_for(T.params(T.TUNE_SEED))).stem
        net = C["net"]
        wj = [[round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 2), d] for w in wps]
        single = n == 1
        common = dict(group=pick["key"], city=city, kind="route", unit="route", cond=cond, wps=wj, span=200.0,
                      title=CONDS[cond][2], route=dict(kind=kind, k=k, L=round(L, 1)))
        rep_apart = None if single else apart_series(net, [dict(t=t_, P=P_) for t_, _, P_ in rs])
        usmp = [dict(t=float(tr["t"][i]), P=np.where(tr["alive"][i][:, None], np.c_[tr["X"][i], tr["Z"][i]], np.nan))
                for i in range(0, len(tr["t"]), 5)]
        uni_apart = None if single else apart_series(net, usmp)
        f_rep = playback.flight_json(hook.tape.record(), id=f"{pick['key']}_{cond}_replica", source="replica",
                                     identical=bool(same), apart=rep_apart, summary=summary(r, d, cond, eff), **common)
        f_uni = playback.flight_json(playback.unity_tape(tr, DT, single), id=f"{pick['key']}_{cond}_unity",
                                     source="unity", run=run.name, scenario=int(idx), apart=uni_apart,
                                     summary=summary(dict(u, deaths=fl.get("deaths")), d, cond, eff), **common)
        return dict(pick=pick, cond=cond, flights=[f_rep, f_uni], identical=bool(same), mismatch=bad,
                    wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, cond=cond, error=repr(e), trace=traceback.format_exc()[-3000:])


def refly(jobs=10):
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
                print(f"[swarmflow_replays] {key} {res['cond']}: ERROR {res['error']}\n{res['trace']}", flush=True)
                got[key]["identical"] = False
                continue
            for f in res["flights"]:
                (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":"), allow_nan=False),
                                                                 encoding="utf-8")
            got[key]["flights"][res["cond"]] = [f["id"] for f in res["flights"]]
            got[key]["identical"] = got[key]["identical"] and res["identical"]
            if res["mismatch"]:
                got[key]["mismatch"][res["cond"]] = res["mismatch"]
            print(f"[swarmflow_replays] {key} {res['cond']}: identical to the gate: {res['identical']}"
                  f"{'' if res['identical'] else ' ' + str(res['mismatch'])}", flush=True)
    for key, m in got.items():
        (OUT / "picks" / f"{key}.json").write_text(json.dumps(m, indent=1, allow_nan=False), encoding="utf-8")


def assemble():
    index = dict(built=time.strftime("%Y-%m-%d %H:%M"), conds={c: v[2] for c, v in CONDS.items()}, groups=[],
                 gallery=[])
    for p in select():
        path = OUT / "picks" / f"{p['key']}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if not m["identical"] or len(m["flights"]) < len(CONDS):
            print(f"[swarmflow_replays] {p['key']}: not every twin identical to its gate row; left out", flush=True)
            continue
        fls = []
        for c in CONDS:
            for fid in m["flights"][c]:
                fls.append(json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8")))
        ref = next(f for f in fls if f["cond"] == "all" and f["source"] == "unity")
        rt = ref["route"]
        about = (f"Route {rt['k']} ({rt['kind']}, {rt['L']:.0f} m) in the scene's city, picked as {p['why']}. "
                 f"Each way starts from the same spot after Unity's 15 s settle; switch between Unity's flight and the "
                 f"replica's twin flown from Unity's recorded start (the other's drones are drawn as outlines).")
        index["groups"].append(dict(id=p["key"], kind="route", title=p["title"], about=about, first=p["first"],
                                    city=ref["city"], route=rt, why=p["why"], flights=[_entry(f) for f in fls]))
    (OUT / "cities").mkdir(parents=True, exist_ok=True)
    for seed in (T.TUNE_SEED,) + T.HELD_SEEDS:
        P = T.params(seed)
        c = city_payload(P)
        stem = Path(blockcity.path_for(P)).stem
        (OUT / "cities" / f"{stem}.json").write_text(json.dumps(c, separators=(",", ":"), allow_nan=False),
                                                     encoding="utf-8")
        index["gallery"].append(dict(seed=seed, city=stem, stats=c["stats"],
                                     what="the scene's city (BlockShapedPatchCityWorld)" if seed == T.TUNE_SEED
                                     else f"held out, the confirmation only (seed {seed})"))
    (OUT / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[swarmflow_replays] {len(index['groups'])} groups -> {OUT}", flush=True)


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
