"""S6 Patch City Tuning: the page's Cities tab data (lessons 30-31).

    python -B -m citysim.swarmtune_replays [--stage all|refly|assemble] [--jobs 10]
    -> experiment_results/swarm_tune/replay/ (cities/, flights/, picks/, index.json)

The cities: BlockShapedPatchCityWorld's (seed 1) and the three held out of the tuning (seeds 2-4), with the avoidance
circles Unity's law draws round each prism. The flights: routes from the Unity bench's set in the scene's city, each
flown five ways (CONDS): one drone under the old gains and under the new, the swarm as the scene was (old gains,
d_ref 1.08), the old gains at d_ref 2.0 and the tuned swarm (T3). Each is shown twice: Unity's own trajectory
(swarmtune_bench_tuned / swarmtune_bench_single) and the replica's twin, flown from Unity's recorded (settled) start
under the values the bench recorded for that set, re-flown here with a tape and kept only if it scores identically to
the gate's row (swarmtune.bench_job wrote those).

The picks follow rules written before they were drawn (select): the typical route of each kind -- the one whose
tuned-minus-old difference in cohesive speed (Unity, both at d_ref 2.0) is nearest the median of its kind, among the
cross routes that meet at least three buildings and the pass routes that meet one -- and the two failures the findings
name: the route where the scene as it was lost the most drones, and the one where one drone under the old gains stalled
longest.
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

from citysim import blockcity, camera, display, pilots, playback, swarmtune as T  # noqa: E402
from citysim.shaped_replays import city_payload  # noqa: E402
from citysim.split_replays import apart_series  # noqa: E402

OUT = T.OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")         # the bench copy's runs (run_bench.ps1)
DT = 0.2
LATCH_S = 0.1                                         # split_probe's: the feed latch's step
SEEDS = (T.TUNE_SEED,) + T.HELD_SEEDS
# condition: (bench run, its parameter set, colour slot on the page, label)
CONDS = {
    "single_old": ("bench_single", "scene_gains", "one drone, old gains"),
    "single_new": ("bench_single", "T3_gains", "one drone, new gains"),
    "old108": ("bench_tuned", "scene_d1.08", "the swarm as it was (old gains, d_ref 1.08)"),
    "old20": ("bench_tuned", "scene_d2.0", "old gains at d_ref 2.0"),
    "tuned": ("bench_tuned", "T3_d2.0", "the tuned swarm (d_ref 2.0)"),
}
KIND_TITLE = {"cross": "A straight line across the city", "pass": "Through one block", "street": "Along the streets"}


# ============================================================================= the gate's rows and the bench's runs

def gate(name):
    import pandas as pd
    return pd.read_json(T.OUT / name / "bench" / "gate.jsonl", lines=True)


def wide():
    """Per bench route (kind, k): the Unity measures the picks are chosen by."""
    import pandas as pd
    t, s = gate("bench_tuned"), gate("bench_single")
    key = ["kind", "k"]
    vt = t.pivot_table(index=key, columns="set", values=["u_v_coh", "u_lost", "u_met"], aggfunc="first")
    vs = s.pivot_table(index=key, columns="set", values=["u_stall_s"], aggfunc="first")
    return pd.DataFrame({"diff": vt["u_v_coh"]["T3_d2.0"] - vt["u_v_coh"]["scene_d2.0"],
                         "met": vt["u_met"]["T3_d2.0"], "lost108": vt["u_lost"]["scene_d1.08"],
                         "stall": vs["u_stall_s"]["scene_gains"]})


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
        return dict(key=f"s6_{kind}", kind=kind, k=k, title=KIND_TITLE[kind], first="tuned",
                    why=f"{why}: its tuned-minus-old difference ({float(g.loc[k, 'diff']):.2f} m/s) is the nearest to "
                        f"the median of its kind ({med:.2f} m/s, {len(g)} routes)", met=int(g.loc[k, "met"]))

    picks.append(typical("cross", lambda g: g.met >= 3, "the typical straight line (of those meeting three buildings)"))
    picks.append(typical("pass", lambda g: g.met >= 1, "the typical block (of those where a building is met)"))
    picks.append(typical("street", lambda g: g.met >= 0, "the typical street route"))
    f = w.drop(index=list(used)).sort_values(["lost108", "met"], ascending=False, kind="stable")
    kind, k = f.index[0]
    used.add((kind, int(k)))
    picks.append(dict(key="s6_collide", kind=kind, k=int(k), title="Failure: drones collide at the old spread",
                      first="old108", met=int(f.iloc[0].met),
                      why=f"the route where the swarm as it was lost the most drones in Unity ({int(f.iloc[0].lost108)})"))
    f = w.drop(index=list(used)).sort_values(["stall"], ascending=False, kind="stable")
    kind, k = f.index[0]
    picks.append(dict(key="s6_stall", kind=kind, k=int(k), title="Failure: one drone stalls under the old shield",
                      first="single_old", met=int(f.iloc[0].met),
                      why=f"the route where one drone under the old gains stalled longest in Unity "
                          f"({float(f.iloc[0].stall):.0f} s under 1 m/s)"))
    return picks


def bench_flight(cond, kind, k):
    """(run dir, its parameter-set label, scenario index, the flight line, the set's recorded SwarmManager values)."""
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    name, label, _ = CONDS[cond]
    run = RUNS / f"swarmtune_{name}"
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    idx = next(i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{kind}_{k}")
    _, _, params, flights, _, _ = bench.read_results(run)
    fl = next(x for x in flights if x["set"] == label and x["scenario"] == idx)
    eff = {kk: float(f"{v:.7g}") if isinstance(v, float) else v for kk, v in params[label]["effective"].items()}
    return run, label, idx, fl, eff, cfg["droneCount"]


# ============================================================================= the replica's twin, taped

class Hook:
    """swarmtune.fly's read-only hook: a tape, and the feed latch the replica has no part in (headings never feed the
    translation, so the flight is the same flight). The pilot's heading turns toward its waypoint as the bench's
    does; the latch steps every LATCH_S, as split_probe.Flight steps it."""

    def __init__(self, n, body0):
        self.latch = display.FeedLatch(n, look_gap=True, pitch_deg=0.0)
        self.body, self.feeds, self.wp, self.tape = float(body0), None, 0, None

    def start(self, sim):
        self.every = max(1, int(round(LATCH_S / sim.m.dt)))
        self.tape = playback.Tape(sim, DT, heading=lambda: self.latch.yaw.copy(),
                                  shown=lambda: (np.asarray(self.feeds.drones, int) if self.feeds is not None
                                                 else np.zeros(0, int)),
                                  body=lambda: float(self.body), wp=lambda: int(self.wp))

    def step(self, sim, t, cmd, wp):
        self.wp = wp
        if float(np.linalg.norm(cmd)) > 1e-9:
            self.body = pilots.turn_toward(self.body, _bearing(cmd), camera.BODY_YAW_RATE_DEG, sim.m.dt)
        if self.feeds is None or sim.tick % self.every == 0:
            self.feeds = self.latch.step(sim.P, sim.alive, self.body, LATCH_S)


def _bearing(v):
    return float(math.degrees(math.atan2(v[0], v[1])))


def _same(a, b):
    """The replica's scored row against the gate's (both from score_flight; the gate's went through pandas JSON)."""
    def norm(x):
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return None
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return round(float(x), 6)
        return x
    keys = [k for k in a if k != "deaths"]
    bad = [k for k in keys if norm(a[k]) != norm(b.get(k))]
    return not bad, bad


SUMMARY = ("v_prog", "one_frac", "v_coh", "T", "L", "finished", "n_eps", "ep_max", "open_end", "reform_end", "met",
           "wrapped", "coverage", "lost", "contacts", "min_pair", "stall_s", "hull_frac")


def summary(row, d, cond):
    s = {k: row.get(k) for k in SUMMARY}
    bp, bf = row.get("bld_pass_s") or [], row.get("bld_frag_s") or []
    s.update(d=d, gains="new" if cond in ("tuned", "single_new") else "old",
             bld_pass=float(np.mean(bp)) if len(bp) else None, bld_frag=float(np.mean(bf)) if len(bf) else None,
             deaths=row.get("deaths"))
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
        # the replica's twin from Unity's recorded start, taped
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
                                     identical=bool(same), apart=rep_apart, summary=summary(r, d, cond), **common)
        f_uni = playback.flight_json(playback.unity_tape(tr, DT, single), id=f"{pick['key']}_{cond}_unity",
                                     source="unity", run=run.name, scenario=int(idx), apart=uni_apart,
                                     summary=summary(dict(u, deaths=fl.get("deaths")), d, cond), **common)
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
                print(f"[swarmtune_replays] {key} {res['cond']}: ERROR {res['error']}\n{res['trace']}", flush=True)
                got[key]["identical"] = False
                continue
            for f in res["flights"]:
                (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":"), allow_nan=False),
                                                                 encoding="utf-8")
            got[key]["flights"][res["cond"]] = [f["id"] for f in res["flights"]]
            got[key]["identical"] = got[key]["identical"] and res["identical"]
            if res["mismatch"]:
                got[key]["mismatch"][res["cond"]] = res["mismatch"]
            print(f"[swarmtune_replays] {key} {res['cond']}: identical to the gate: {res['identical']}"
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
            print(f"[swarmtune_replays] {p['key']}: not every twin identical to its gate row; left out", flush=True)
            continue
        fls = []
        for c in CONDS:
            for fid in m["flights"][c]:
                fls.append(json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8")))
        tuned = next(f for f in fls if f["cond"] == "tuned" and f["source"] == "unity")
        rt = tuned["route"]
        about = (f"Route {rt['k']} ({rt['kind']}, {rt['L']:.0f} m) in the scene's city, picked as {p['why']}. "
                 f"Each way starts from the same spot after Unity's 15 s settle; switch between Unity's flight and the "
                 f"replica's twin flown from Unity's recorded start (the other's drones are drawn as outlines).")
        index["groups"].append(dict(id=p["key"], kind="route", title=p["title"], about=about, first=p["first"],
                                    city=tuned["city"], route=rt, why=p["why"],
                                    flights=[dict(_entry(f)) for f in fls]))
    (OUT / "cities").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        P = T.params(seed)
        c = city_payload(P)
        stem = Path(blockcity.path_for(P)).stem
        (OUT / "cities" / f"{stem}.json").write_text(json.dumps(c, separators=(",", ":"), allow_nan=False),
                                                     encoding="utf-8")
        index["gallery"].append(dict(seed=seed, city=stem, stats=c["stats"],
                                     what="the scene's city (BlockShapedPatchCityWorld)" if seed == T.TUNE_SEED
                                     else f"held out of the tuning (seed {seed})"))
    (OUT / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[swarmtune_replays] {len(index['groups'])} groups, {len(index['gallery'])} cities -> {OUT}", flush=True)


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
