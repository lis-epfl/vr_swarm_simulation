"""S12 Distributed Views in Unity: the spot check of the main study (README 7 step 5).

The replica flies every session of the study; this re-flies a few of them in the headless Unity bench, and their
replica twins from Unity's own start, to check that the flights the hazards are integrated over are flights Unity
would fly (lesson 48). A session's route is the pilot's executed plan (views.fly with `log_tasks`): each `go` a
waypoint taken within the capture radius or at its timeout, each `hold` a centring hold for its duration -- exactly
SwarmBenchRunner's waypoint rule, so Unity and the twin fly the same route. The swarm is flown wrapped (the main
study's pattern) with ten drones, the single drone with one; the 360 drone flies the single drone's searches, so its
airframe and route are the single drone's and need no flight of their own.

    python -B -m citysim.views_bench plan --name spot [--n 12]
    python -B -m citysim.views_bench gate --name spot --runs DIR [DIR ...]
    -> experiment_results/views_study/<name>/bench/

Measured on both: the time to fly the whole route, the centroid's mean speed on the `go` legs, the share of the
swarm's goal holds spent wrapped round the goal building (no bearing gap of 180 deg or more among the alive drones
round its centre), drones lost (events: a colliding pair is one) and building contacts, and how far Unity's centroid
is from the twin's at the same moment. The recognition hazards are not re-scored on Unity's tracks: they are a
function of where the cameras are, which is what the comparison checks.
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
import traceback  # noqa: E402
from collections import OrderedDict  # noqa: E402

import numpy as np  # noqa: E402

from citysim import views as V  # noqa: E402
from citysim import swarmtune as T  # noqa: E402

rep = T.rep
K_BENCH = 1400                      # goal draws of the spot check (none used elsewhere in the study)
CONDS = ("swarm", "single")
SLACK_S = 20.0                      # a go waypoint's timeout: the replica's own time for it plus this


def route(cond, k, seed=V.CITY_SEED):
    """(session result, wps, holds) of goal draw k flown in the main study's way, with its plan logged. wps are
    SwarmBenchRunner's (x, z, stick, timeout); a hold is a negative stick for its duration. holds: [(wp index, goal
    centre or None)]."""
    opts = dict(V.MAIN[cond], rec_model=V.REC_MAIN, log_tasks=True)
    r = V.fly(cond, seed, k, opts)
    wps, holds = [], []
    centres = [np.array(g["centre"]) for g in r["goal_geo"]]
    for kind, p, stick, dur, t0, t1 in r["tasks"]:
        if t1 is None or t1 - t0 < 0.2:
            continue
        if kind == "go":
            wps.append([p[0], p[1], max(stick, 0.05), round((t1 - t0) + SLACK_S, 1)])
        else:
            gi = int(np.argmin([np.linalg.norm(c - p) for c in centres])) if centres else -1
            near = gi >= 0 and np.linalg.norm(centres[gi] - p) < 5.0
            holds.append((len(wps), gi if near else None))
            wps.append([p[0], p[1], -max(stick, 0.05), round(t1 - t0, 1)])
    return r, wps, holds


def _plan_job(job):
    cond, k = job
    try:
        r, wps, holds = route(cond, k)
        return dict(cond=cond, k=k, wps=wps, holds=holds, task_s=r["task_s"], reason=r["reason"],
                    goal_geo=r["goal_geo"], line=r["line"], lost=r["lost"], contacts=r["contacts"])
    except Exception as e:                          # noqa: BLE001
        return dict(cond=cond, k=k, error=repr(e), trace=traceback.format_exc()[-1500:])


def cmd_plan(a):
    from citysim import BENCH_DIR, camera, parallel
    jobs = [(c, k) for c in CONDS for k in range(a.k0, a.k0 + a.n)]
    res = parallel.run_jobs(_plan_job, jobs, "views spot plan", processes=a.procs)
    errs = [r for r in res if "error" in r]
    for e in errs[:3]:
        print(e["error"], e["trace"][-800:])
    out = V.OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    spawn = np.array(T.city(V.CITY_SEED)["L"].layout["spawn"], float)[[0, 2]]
    fields = T.sp.swarm_manager_fields()
    routes = {}
    for cond in CONDS:
        rs = sorted([r for r in res if "error" not in r and r["cond"] == cond], key=lambda r: r["k"])
        scen = []
        for r in rs:
            flat = [round(float(v), 3) for w in r["wps"] for v in w]
            scen.append(dict(kind=f"views{cond}_{r['k']}", sx=round(float(spawn[0]), 3), sz=round(float(spawn[1]), 3),
                             T=round(sum(w[3] for w in r["wps"]) + 30.0, 1), wps=flat, holdGain=V.T.S.HOLD_GAIN,
                             arrive=V.CAPTURE_M, record=True, endOnDone=True))
            routes[f"{cond}|{r['k']}"] = r
        band = V.BAND[cond]["goal"]
        cfg = OrderedDict(scene=T.SCENE, timeScale=20.0, settleTime=T.BENCH_SETTLE_S, altitude=band[0],
                          capture=V.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                          goalReplaySession="", layout="", droneCount=V.N_SWARM if cond == "swarm" else 1,
                          walkers=False, paramSets=[T.bench.parse_set(f"scene:d_ref={V.D_REF}", fields, True)],
                          scenarios=scen)
        cfile = out / f"config_{cond}.json"
        cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"[views_bench] {cond}: {len(scen)} flights -> {cfile}\n"
              f"  & '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag views_{a.name}_{cond} -TimeoutMin {a.timeout}")
    (out / "routes.json").write_text(json.dumps(routes), encoding="utf-8")


# ============================================================================= the twin and the measures

def fly_twin(m, C, P0, wps, d, seed, capture=V.CAPTURE_M):
    """SwarmBenchRunner's waypoint pilot over ContactSim from Unity's start: (sim, samples [(t, centroid, P)], done)."""
    sim = rep.ContactSim(m, C["rcity"], np.array(P0, float), np.zeros((len(P0), 2)), seed)
    sim.contacts = sim.contact_ticks = 0
    sim.kills = []
    n0 = int(sim.alive.sum())
    gain = V.T.S.HOLD_GAIN / m.max_speed
    every = max(1, int(round(0.1 / m.dt)))
    done, samples = [math.nan] * len(wps), []
    wp, t0 = 0, 0.0
    total = sum(w[3] for w in wps) + 30.0
    while sim.t < total and sim.alive.any() and wp < len(wps):
        t = sim.t
        c = sim.P[sim.alive].mean(0)
        x, z, stick, timeout = wps[wp]
        dv = np.array([x, z]) - c
        dd = float(np.linalg.norm(dv))
        hold = stick < 0
        if (not hold and dd < capture) or (timeout > 0 and t - t0 > timeout):
            done[wp] = t
            wp, t0 = wp + 1, t
            continue
        if hold:
            cmd = dv * gain
            s = float(np.linalg.norm(cmd))
            if s > -stick:
                cmd *= -stick / s
        else:
            cmd = dv / max(dd, 1e-9) * stick
        if sim.tick % every == 0:
            samples.append((t, c.copy(), np.where(sim.alive[:, None], sim.P, np.nan), wp))
        sim.step(cmd, 0.0, d)
    sim.n0 = n0
    return sim, samples, done


def measures(samples, done, wps, holds, goal_geo, lost, contacts):
    t = np.array([s[0] for s in samples])
    c = np.array([s[1] for s in samples])
    wpi = np.array([s[3] for s in samples])
    v = np.r_[np.nan, np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)]
    go = np.array([wps[min(i, len(wps) - 1)][2] > 0 for i in wpi])
    wrapped, held = 0.0, 0.0
    for wi, gi in holds:
        if gi is None:
            continue
        g = np.array(goal_geo[gi]["centre"])
        for s in (s for s in samples if s[3] == wi):
            P = s[2][np.isfinite(s[2][:, 0])]
            held += 1
            if len(P) >= 3:
                ang = np.sort(np.arctan2(P[:, 1] - g[1], P[:, 0] - g[0]))
                gap = np.diff(np.r_[ang, ang[0] + 2 * math.pi]).max()
                wrapped += gap < math.pi
    last = [x for x in done if math.isfinite(x)]
    return dict(route_s=round(last[-1], 1) if len(last) == len(done) else None, wps_done=len(last),
                speed=round(float(np.nanmean(v[go])), 2) if go.any() else None,
                wrapped=round(wrapped / held, 3) if held else None, lost=int(lost), contacts=int(contacts))


def bench_job(job):
    run_dir, label, idx, sc, flight, rt = job
    from pathlib import Path
    from citysim import tierc
    try:
        cond = "swarm" if sc["kind"].startswith("viewsswarm") else "single"
        C = T.city(V.CITY_SEED)
        tr = tierc.read_traj(Path(run_dir) / f"traj_{label}_{idx}.csv")
        wps = rt["wps"]
        holds = [tuple(h) for h in rt["holds"]]
        samples = []
        for r in range(len(tr["t"])):
            al = tr["alive"][r]
            P = np.where(al[:, None], np.c_[tr["X"][r], tr["Z"][r]], np.nan)
            samples.append((float(tr["t"][r]), np.nanmean(P, axis=0) if al.any() else np.zeros(2), P,
                            int(tr["wp"][r])))
        done = [math.nan] * len(wps)
        for r in range(len(tr["t"])):
            for i in range(min(int(tr["wp"][r]), len(wps))):
                if not math.isfinite(done[i]):
                    done[i] = float(tr["t"][r])
        if len(wps) > 1 and not math.isfinite(done[-1]) and all(math.isfinite(x) for x in done[:-1]) and \
                float(tr["t"][-1]) < sum(w[3] for w in wps) + 25.0:
            done[-1] = float(tr["t"][-1])           # endOnDone: the flight stops on taking its last waypoint, unlogged
        deaths = flight.get("deaths") or {}
        u_lost = sum(v for kk, v in deaths.items() if kk != "ExternallyKilled")
        u = measures(samples, done, wps, holds, rt["goal_geo"], u_lost, flight.get("contacts") or 0)
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        m = T.model({})
        sim, rs, rdone = fly_twin(m, C, P0, wps, V.D_REF, 7000 + int(rt["k"]))
        r = measures(rs, rdone, wps, holds, rt["goal_geo"], len(sim.kills), sim.contacts)   # a kill record is an event
        tu = np.array([s[0] for s in samples])
        cu = np.array([s[1] for s in samples])
        tt = np.array([s[0] for s in rs])
        ct = np.array([s[1] for s in rs])
        n = min(len(tu), len(tt))
        gapm = np.linalg.norm(cu[:n] - np.c_[np.interp(tu[:n], tt, ct[:, 0]), np.interp(tu[:n], tt, ct[:, 1])],
                              axis=1) if n else np.zeros(0)
        return dict(cond=cond, k=rt["k"], **{f"u_{x}": v for x, v in u.items()}, **{f"r_{x}": v for x, v in r.items()},
                    gap_med=round(float(np.median(gapm)), 1) if len(gapm) else None,
                    gap_p90=round(float(np.percentile(gapm, 90)), 1) if len(gapm) else None,
                    session_task_s=rt["task_s"], session_lost=rt["lost"])
    except Exception as e:                          # noqa: BLE001
        return dict(set=label, idx=idx, error=repr(e), trace=traceback.format_exc()[-1500:])


def cmd_gate(a):
    import pandas as pd
    from pathlib import Path
    from citysim import parallel
    out = V.OUT / a.name / "bench"
    routes = json.loads((out / "routes.json").read_text(encoding="utf-8"))
    jobs = []
    for run in a.runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        _, _, _, flights, _, _ = T.bench.read_results(run)
        fl = {(f["set"], f["scenario"]): f for f in flights}
        for ps in cfg["paramSets"]:
            for idx, sc in enumerate(cfg["scenarios"]):
                kind, k = sc["kind"].rsplit("_", 1)
                rt = routes.get(f"{kind[5:]}|{k}")
                if rt and (ps["label"], idx) in fl and (run / f"traj_{ps['label']}_{idx}.csv").exists():
                    jobs.append((str(run), ps["label"], idx, sc, fl[(ps["label"], idx)], rt))
    rp = out / "runs.json"                         # which bench run flew each condition (views_replays reads it)
    rmap = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    for run in a.runs:
        cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
        rmap["swarm" if int(cfg.get("droneCount", 1)) > 1 else "single"] = str(run)
    rp.write_text(json.dumps(rmap, indent=1), encoding="utf-8")
    res = parallel.run_jobs(bench_job, jobs, "views spot gate", processes=a.jobs)
    errs = [r for r in res if "error" in r]
    for e in errs[:3]:
        print(e["error"], e.get("trace", "")[-800:])
    df = pd.DataFrame([r for r in res if "error" not in r]).sort_values(["cond", "k"])
    df.to_json(out / "gate.jsonl", orient="records", lines=True)
    pd.set_option("display.width", 300)
    print(df.to_string(index=False))
    rows = []
    for cond, g in df.groupby("cond"):
        for src in ("u", "r"):
            rows.append(dict(cond=cond, src="unity" if src == "u" else "replica", n=len(g),
                             route_s=g[f"{src}_route_s"].median(), complete=float(g[f"{src}_route_s"].notna().mean()),
                             speed=g[f"{src}_speed"].median(), wrapped=g[f"{src}_wrapped"].median(),
                             lost=int(g[f"{src}_lost"].sum()), contacts=int(g[f"{src}_contacts"].sum())))
        rows.append(dict(cond=cond, src="gap (m)", n=len(g), route_s=g.gap_med.median(), complete=g.gap_p90.median()))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate.csv", index=False)
    print(tab.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--k0", type=int, default=K_BENCH)
    p.add_argument("--procs", type=int, default=24)
    p.add_argument("--timeout", type=int, default=240)
    p = sub.add_parser("gate")
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    dict(plan=cmd_plan, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
