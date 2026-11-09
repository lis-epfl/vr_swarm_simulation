"""S7 Building Flow: three changes to how the swarm moves among BlockShapedPatchCityWorld's buildings, each behind its
own SwarmManager flag, flown on S6's flights and measures (citysim.swarmtune) against S6's tuned gains (T3).

    python -B -m citysim.swarmflow screen --name N --cand FILE [--cross 24 --pass 16 --street 12] [--k0 3000] [--single]
    python -B -m citysim.swarmflow confirm --name N --cand FILE [--single]   # fresh routes, the scene's city + held out
    python -B -m citysim.swarmflow table --name N
    python -B -m citysim.swarmflow choose --name confirm                     # rule.json applied: verdicts.json
    python -B -m citysim.swarmflow steps --cand FILE --out FILE              # spread steps at hover (open sky)
    python -B -m citysim.swarmflow bench-plan --name N --cand FILE [--drones 1]
    python -B -m citysim.swarmflow bench-gate --name N --runs DIR [DIR ...]
    -> experiment_results/swarm_flow/<name>/

The three (swarm_replica.slide_round, route_round, spread_target; OlfatiSaber and SwarmPlaneController in Unity):
  slideRoundBuildings  the part of the pilot's stick the shield removes near a building is turned along the building
                       instead of deleted, for head-on approaches, so a drone flown at a building flows round it
  routeStragglers      cohesion's pull into a building is turned along the building the same way, so a drone left
                       behind one is drawn round it to the swarm rather than through it
  adaptiveSpread       the formation narrows to streetSpreadFactor x d_ref among buildings and widens back to d_ref
                       when the swarm flies at one (surroundLookAhead, surroundAimFraction), to fit the streets and
                       still surround

A candidate is S6's: {"label", "d", "over": {SwarmManager field: value, "x:<extra>": value}}. The scene's SwarmManager
is T3 since S6, so an empty `over` is T3 and every candidate is T3 plus its flags. Route indices: screens from K_SCREEN,
the confirmation from K_CONFIRM, the bench from K_BENCH (S6 used 0, 500, 1000 and 2000), so nothing here re-flies one
of S6's routes. A flight's measures are S6's (swarmtune.score_flight) plus the adaptive spread's share of d_ref over
the flight (`sigma_mean`, and `narrow_frac`, the share of the flight below 0.95).
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

import numpy as np  # noqa: E402

from citysim import REPO  # noqa: E402
from citysim import swarmtune as T  # noqa: E402

OUT = REPO / "experiment_results" / "swarm_flow"
K_SCREEN, K_CONFIRM, K_BENCH, K_TUNE2 = 3000, 4000, 5000, 6000
FLAGS = ("slideRoundBuildings", "routeStragglers", "adaptiveSpread")


class SpreadLog:
    """A read-only fly() hook: the adaptive spread's share of d_ref at every step of the flight."""

    def __init__(self):
        self.sig = []

    def start(self, sim):
        pass

    def step(self, sim, t, cmd, wp):
        self.sig.append(1.0 if sim.sigma is None else float(sim.sigma))


def run_job(j):
    """swarmtune.run_job with the spread logged."""
    t0 = time.time()
    base = dict(job_id=T.job_id(j), cand=j["cand"], kind=j["kind"], seed=j["seed"], k=j["k"], d=j["d"])
    try:
        C = T.city(j["seed"])
        rng = np.random.default_rng(T.route_seed(j["kind"], j["seed"], j["k"]))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly = T.route(C, j["kind"], j["k"], rng)
        over = dict(j["over"])
        m = T.model(over)
        P0 = start[None].copy() if j.get("single") else T.start_positions(C, j["d"], fseed, over, start)
        log = SpreadLog()
        sim, samples, done = T.fly(m, C, P0, wps, j["d"], fseed, hook=log)
        sig = np.array(log.sig) if log.sig else np.ones(1)
        return dict(base, **T.score_flight(C, start, wps, poly, L, sim, samples, done, sim.n0),
                    sigma_mean=round(float(sig.mean()), 3), narrow_frac=round(float((sig < 0.95).mean()), 3),
                    wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


def load_cands(path):
    return json.loads(open(path, encoding="utf-8").read())


def jobs_for(spec, seeds, flights, k0, single):
    cands, jobs = {}, []
    for c in spec:
        cid = T.cand_id(c["over"], c["d"])
        cands[cid] = dict(over=c["over"], d=c["d"], label=c.get("label", ""))
        jobs += T.jobs_for(cid, c["over"], c["d"], seeds=seeds, flights=flights, k0=k0)
        if single:
            sid = "single-" + T.cand_id(c["over"], 0.0)
            if sid not in cands:
                cands[sid] = dict(over=c["over"], d=c["d"], label=f"one drone, {c.get('label', '')}")
                jobs += [dict(j, single=True) for j in T.jobs_for(sid, c["over"], c["d"], seeds=seeds, flights=flights,
                                                                    k0=k0)]
    return cands, jobs


def table(name):
    tab = T.table(name, OUT)
    df, _, _ = T.read(name, OUT)
    if "sigma_mean" in df:
        g = df.groupby("cand")
        tab["sigma"] = tab.cand.map(g.sigma_mean.mean())
        tab["narrow"] = tab.cand.map(g.narrow_frac.mean())
    return tab


COLS = ["label", "d", "n", "score", "v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "hull",
        "cross_v_coh", "cross_v_prog", "cross_wrapped", "cross_coverage", "cross_bld_pass", "cross_bld_frag",
        "pass_v_coh", "pass_bld_pass", "pass_wrapped", "street_v_coh", "street_one", "street_lost_km", "street_contact_km",
        "sigma", "narrow"]


def show(tab, n=60):
    import pandas as pd
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 80)
    print(tab[[c for c in COLS if c in tab.columns]].head(n).round(3).to_string(index=False))


def flights_arg(a, default):
    return dict(cross=a.cross, pass_=a.pass_, street=a.street) if a.cross is not None else default


def cmd_screen(a):
    spec = load_cands(a.cand)
    cands, jobs = jobs_for(spec, (T.TUNE_SEED,), flights_arg(a, T.FLIGHTS), a.k0, a.single)
    T.run(a.name, cands, jobs, a.jobs, f"screen ({len(cands)} candidates)", root=OUT, fn=run_job)
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab)


def cmd_confirm(a):
    spec = load_cands(a.cand)
    seeds = (T.TUNE_SEED,) + T.HELD_SEEDS
    cands, jobs = jobs_for(spec, seeds, dict(cross=a.cross, pass_=a.pass_, street=a.street), a.k0, a.single)
    T.run(a.name, cands, jobs, a.jobs, "confirm", root=OUT, fn=run_job)
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab)


def cmd_table(a):
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab, a.n)


# ============================================================================= the rule (rule.json, fixed before the confirmation)

RULE = dict(lost_margin=0.05, open_margin=0.02, hull_min=0.5, gain=0.05, street_floor=0.98, wrap_margin=0.10)


def judge(tab):
    """Each swarm configuration of a confirmation table against T3 by rule.json: dict(label: verdict), and the pick."""
    sw = tab[~tab.cand.astype(str).str.startswith("single")]
    t3 = sw[sw.label == "T3"].iloc[0]
    out = {}
    for _, r in sw.iterrows():
        safe = bool(r.lost_km <= t3.lost_km + RULE["lost_margin"] and r.contact_km <= t3.contact_km
                    and r.open <= t3.open + RULE["open_margin"] and r.hull >= RULE["hull_min"])
        helps = bool(r.v_coh >= (1 + RULE["gain"]) * t3.v_coh
                     and r.street_v_coh >= RULE["street_floor"] * t3.street_v_coh
                     and r.cross_wrapped >= t3.cross_wrapped - RULE["wrap_margin"])
        out[r.label] = dict(safe=safe, helps=helps, score=float(r.score), v_coh_gain=float(r.v_coh / t3.v_coh - 1),
                            street_gain=float(r.street_v_coh / t3.street_v_coh - 1),
                            wrap_change=float(r.cross_wrapped - t3.cross_wrapped),
                            lost_change=float(r.lost_km - t3.lost_km), contact_change=float(r.contact_km - t3.contact_km),
                            open_change=float(r.open - t3.open))
    ok = [(v["score"], k) for k, v in out.items() if v["safe"] and v["helps"] and k != "T3"]
    pick = max(ok)[1] if ok else None
    return out, pick


def cmd_choose(a):
    tab = table(a.name)
    verdicts, pick = judge(tab)
    for k, v in sorted(verdicts.items(), key=lambda kv: -kv[1]["score"]):
        print(f"{k:28s} safe {str(v['safe']):5s} helps {str(v['helps']):5s} score {v['score']:.3f}  v_coh "
              f"{v['v_coh_gain']:+.1%}  street {v['street_gain']:+.1%}  wrapped {v['wrap_change']:+.3f}  lost "
              f"{v['lost_change']:+.3f}  contacts {v['contact_change']:+.3f}  open {v['open_change']:+.3f}")
    print(f"\nrule: {pick or 'none: recommend none on by default'}")
    (OUT / a.name / "verdicts.json").write_text(json.dumps(dict(rule=RULE, verdicts=verdicts, pick=pick), indent=1),
                                               encoding="utf-8")


def cmd_steps(a):
    import pandas as pd
    from multiprocessing import Pool
    spec = load_cands(a.cand)
    jobs = [(c.get("label", ""), c["over"], c["d"], s) for c in spec for s in range(a.seeds)]
    with Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(T.run_steps, jobs)
    df = pd.DataFrame(res).groupby("label").mean(numeric_only=True).drop(columns="seed").round(2)
    pd.set_option("display.width", 250)
    print(df.to_string())
    if a.out:
        df.to_csv(a.out)


def cmd_bench_plan(a):
    """A bench config for BlockShapedPatchCityWorld as baked: one parameter set per candidate (its flags are
    SwarmManager fields, so the bench sets them like any gain), the flights as scenarios (routes from K_BENCH)."""
    from collections import OrderedDict
    from citysim import BENCH_DIR, camera
    spec = load_cands(a.cand)
    fields = T.sp.swarm_manager_fields()
    sets = []
    for c in spec:
        if any(k.startswith("x:") for k in c["over"]):
            raise SystemExit(f"[swarmflow] {c['label']}: replica-only extras cannot fly in Unity")
        over = {k: (int(v) if isinstance(v, bool) else v) for k, v in dict(c["over"], d_ref=c["d"]).items()}
        assigns = ",".join(f"{k}={v}" for k, v in over.items())
        sets.append(T.bench.parse_set(f"{c['label']}:{assigns}", fields, True))
    scen = T.bench_scenarios(dict(cross=a.cross, pass_=a.pass_, street=a.street), k0=a.k0)
    cfg = OrderedDict(scene=T.SCENE, timeScale=20.0, settleTime=T.BENCH_SETTLE_S, altitude=T.BENCH_ALTITUDE,
                      capture=T.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                      goalReplaySession="", layout="", droneCount=a.drones, walkers=False, paramSets=sets,
                      scenarios=scen)
    out = OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    cfile = out / "config.json"
    cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "cands.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"[swarmflow] {len(sets)} sets x {len(scen)} flights -> {cfile}\n"
          f"  & '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag swarmflow_{a.name} -TimeoutMin {a.timeout}")


def cmd_bench_gate(a):
    """Unity's flights scored with S6's measures beside replica twins flown from Unity's recorded starts, with every
    SwarmManager value the bench recorded for the set (the S7 flags included)."""
    import pandas as pd
    from pathlib import Path
    from citysim import parallel
    jobs = []
    for run in a.runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        _, _, params, flights, _, _ = T.bench.read_results(run)
        fl = {(f["set"], f["scenario"]): f for f in flights}
        for ps in cfg["paramSets"]:
            eff = (params.get(ps["label"]) or {}).get("effective")
            over = ({k: float(f"{v:.7g}") if isinstance(v, float) else v for k, v in eff.items()} if eff
                    else dict(zip(ps["names"], ps["values"])))
            d = float(over.pop("d_ref"))
            for idx, sc in enumerate(cfg["scenarios"]):
                if (ps["label"], idx) in fl and (run / f"traj_{ps['label']}_{idx}.csv").exists():
                    jobs.append((str(run), ps["label"], idx, sc, over, d, fl[(ps["label"], idx)]))
    res = parallel.run_jobs(T.bench_job, jobs, "bench gate", processes=a.jobs)
    errs = [r for r in res if "error" in r]
    for e in errs[:3]:
        print(e["error"], e.get("trace", "")[-600:])
    df = pd.DataFrame([r for r in res if "error" not in r])
    out = OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "gate.jsonl", orient="records", lines=True)
    rows = []
    for label, g in df.groupby("set", sort=False):
        for src in ("u", "r"):
            gg = g[[c for c in g.columns if c.startswith(f"{src}_")]].rename(columns=lambda c: c[2:])
            gg["kind"] = g["kind"].values
            rows.append(dict(set=label, src="unity" if src == "u" else "replica", **T.score_candidate(gg)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    cols = ["set", "src", "n", "km", "v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "hull",
            "cross_wrapped", "cross_coverage", "cross_bld_pass", "pass_bld_pass", "street_v_coh", "street_one",
            "street_lost_km", "street_contact_km"]
    print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    if errs:
        print(f"[swarmflow] {len(errs)} flights failed to score")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    for k in ("cross", "pass", "street"):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int)
    p.add_argument("--k0", type=int, default=K_SCREEN)
    p.add_argument("--single", action="store_true")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("confirm")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--cross", type=int, default=40)
    p.add_argument("--pass", dest="pass_", type=int, default=30)
    p.add_argument("--street", type=int, default=20)
    p.add_argument("--k0", type=int, default=K_CONFIRM, help="first route index (a tuning run after the confirmation "
                                                               "flies its own: K_TUNE2)")
    p.add_argument("--single", action="store_true")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("table")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=60)
    p = sub.add_parser("choose")
    p.add_argument("--name", required=True)
    p = sub.add_parser("steps")
    p.add_argument("--cand", required=True)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--out")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("bench-plan")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--cross", type=int, default=30)
    p.add_argument("--pass", dest="pass_", type=int, default=20)
    p.add_argument("--street", type=int, default=14)
    p.add_argument("--k0", type=int, default=K_BENCH)
    p.add_argument("--drones", type=int, default=T.N_DRONES)
    p.add_argument("--timeout", type=int, default=120)
    p = sub.add_parser("bench-gate")
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    dict(screen=cmd_screen, confirm=cmd_confirm, table=cmd_table, steps=cmd_steps, choose=cmd_choose,
         **{"bench-plan": cmd_bench_plan, "bench-gate": cmd_bench_gate})[a.cmd](a)


if __name__ == "__main__":
    main()
