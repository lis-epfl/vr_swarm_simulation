"""S10 Ellipse Avoidance in Unity: the bench's plan and gate for the ellipse study, on S8's (swarmshape) with the two
flights S10 added -- the angled run-in (`angle`) and the one-drone along probe (`along`) -- scored on Unity's own
trajectory and on its replica twin, flown from Unity's recorded start, by the same measures (swarmellipse).

    python -B -m citysim.swarmellipse_bench plan --name N --cand FILE [--cross 30 --pass 20 --street 50 --face 36
                                                 --angle 36 --along 0] [--drones 1] [--d 1.2]
    python -B -m citysim.swarmellipse_bench gate --name N --runs DIR [DIR ...]
    -> experiment_results/swarm_ellipse/<name>/bench/

A separate module from swarmellipse.py so it can be written and run while a replica run is importing that one
(lesson 18). Route indices from swarmellipse.K_BENCH; every flight is in the scene's city (seed 1), as baked.
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

import numpy as np  # noqa: E402

from citysim import swarmellipse as E  # noqa: E402
from citysim import swarmflow as F  # noqa: E402
from citysim import swarmshape as SH  # noqa: E402
from citysim import swarmtune as T  # noqa: E402


def scenarios(flights, k0):
    """S8's scenarios (cross, pass, street, face) and the angled and along ones, each the replica job's own route
    (the same random stream) ending in END_HOLD_S at zero stick."""
    out = SH.bench_scenarios({k: flights.get(k, 0) for k in ("cross", "pass_", "street", "face")}, k0)
    C = T.city(T.TUNE_SEED)
    for kind in ("angle", "along"):
        for k in range(k0, k0 + flights.get(kind, 0)):
            rng = np.random.default_rng(T.route_seed(kind, T.TUNE_SEED, k))
            rng.integers(1 << 30)
            route = E.angle_route if kind == "angle" else E.along_route
            start, wps, L, poly, info = route(C, T.TUNE_SEED, k, rng)
            wl = [list(w) for w in wps] + [[wps[-1][0], wps[-1][1], -1.0, T.END_HOLD_S]]
            out.append(dict(kind=f"{kind}_{k}", sx=round(float(start[0]), 3), sz=round(float(start[1]), 3),
                            T=round(sum(w[3] for w in wps) + 10.0 + T.END_HOLD_S, 1),
                            wps=[round(float(v), 3) for w in wl for v in w], holdGain=T.ZERO_GAIN, record=True,
                            endOnDone=True))
    return out


def cmd_plan(a):
    from collections import OrderedDict
    from citysim import BENCH_DIR, camera
    spec = F.load_cands(a.cand)
    if a.d is not None:
        spec = [dict(c, d=a.d) for c in spec]
    fields = T.sp.swarm_manager_fields()
    sets = []
    for c in spec:
        if any(k.startswith("x:") for k in c["over"]):
            raise SystemExit(f"[swarmellipse_bench] {c['label']}: replica-only extras cannot fly in Unity")
        over = {k: (int(v) if isinstance(v, bool) else v) for k, v in dict(c["over"], d_ref=c["d"]).items()}
        sets.append(T.bench.parse_set(f"{c['label'].split()[0]}:{','.join(f'{k}={v}' for k, v in over.items())}",
                                      fields, True))
    scen = scenarios(dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face, angle=a.angle, along=a.along),
                     a.k0)
    cfg = OrderedDict(scene=T.SCENE, timeScale=20.0, settleTime=T.BENCH_SETTLE_S, altitude=T.BENCH_ALTITUDE,
                      capture=T.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                      goalReplaySession="", layout="", droneCount=a.drones, walkers=False, paramSets=sets,
                      scenarios=scen)
    out = E.OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    cfile = out / "config.json"
    cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "cands.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"[swarmellipse_bench] {len(sets)} sets x {len(scen)} flights -> {cfile}\n"
          f"  & '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag swarmellipse_{a.name} -TimeoutMin {a.timeout}")


def _track(tr):
    """A TrackLog of a Unity trajectory (it records every 0.1 s, as TrackLog does)."""
    from types import SimpleNamespace
    P = [np.where(tr["alive"][r][:, None], np.c_[tr["X"][r], tr["Z"][r]], np.nan) for r in range(len(tr["t"]))]
    return SimpleNamespace(t=[float(t) for t in tr["t"]], P=P)


def bench_job(job):
    """S8's bench_job (swarmshape: swarmtune's for cross, pass and street, its own for face); for an angled or along
    flight the same -- Unity's trajectory scored, its twin flown from Unity's start -- with S10's measures on both."""
    run_dir, label, idx, sc, over, d, flight = job
    kind, k = sc["kind"].rsplit("_", 1)
    if kind not in ("angle", "along"):
        return SH.bench_job(job)
    from pathlib import Path
    from types import SimpleNamespace
    from citysim import tierc
    try:
        k = int(k)
        C = T.city(T.TUNE_SEED)
        rng = np.random.default_rng(T.route_seed(kind, T.TUNE_SEED, k))
        fseed = int(rng.integers(1 << 30))
        route = E.angle_route if kind == "angle" else E.along_route
        start, wps, L, poly, info = route(C, T.TUNE_SEED, k, rng)
        tr = tierc.read_traj(Path(run_dir) / f"traj_{label}_{idx}.csv")
        samples = []
        for r in range(0, len(tr["t"]), max(1, int(round(T.SAMPLE_S / 0.1)))):
            al = tr["alive"][r]
            P = np.where(al[:, None], np.c_[tr["X"][r], tr["Z"][r]], np.nan)
            samples.append((float(tr["t"][r]), np.nanmean(P, axis=0) if al.any() else np.zeros(2), P))
        done = [math.nan] * len(wps)
        for r in range(len(tr["t"])):
            for i in range(min(int(tr["wp"][r]), len(wps))):
                if not math.isfinite(done[i]):
                    done[i] = float(tr["t"][r])
        n0 = int(tr["alive"][0].sum())
        deaths = flight.get("deaths") or {}
        ghost = SimpleNamespace(alive=tr["alive"][-1], kills=[(0.0, kk) for kk, v in deaths.items() for _ in range(v)],
                                contacts=int(flight.get("contacts") or 0), contact_ticks=0)
        m = T.model(over)
        shape = E.model_shape(m, C["rcity"])

        def measures(sim, smp, dn, log, nn):
            out = T.score_flight(C, start, wps, poly, L, sim, smp, dn, nn)
            if kind == "angle":
                out.update(SH.face_measures(C, info, smp, dn), angle=round(info["angle"], 1))
                if nn == 1:
                    out.update(E.approach_measures(C, info, log))
            else:
                out.update(E.along_measures(C, info, log, shape))
            return out

        b = measures(ghost, samples, done, _track(tr), n0)
        b["deaths"] = deaths
        b["other_contacts"] = int(flight.get("other_contacts") or 0)
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        log = E.TrackLog()
        sim, rs, rdone = T.fly(m, C, P0, wps, d, fseed, settle=0.0, hook=log)
        r = measures(sim, rs, rdone, log, sim.n0)
        return dict(set=label, idx=idx, kind=kind, k=k, d=d, **{f"u_{x}": v for x, v in b.items()},
                    **{f"r_{x}": v for x, v in r.items()})
    except Exception as e:
        return dict(set=label, idx=idx, error=repr(e), trace=traceback.format_exc()[-1500:])


def cmd_gate(a):
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
    res = parallel.run_jobs(bench_job, jobs, "bench gate", processes=a.jobs)
    errs = [r for r in res if "error" in r]
    for e in errs[:3]:
        print(e["error"], e.get("trace", "")[-600:])
    df = pd.DataFrame([r for r in res if "error" not in r])
    out = E.OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "gate.jsonl", orient="records", lines=True)
    rows = []
    for label, g in df.groupby("set", sort=False):
        for src in ("u", "r"):
            gg = g[[c for c in g.columns if c.startswith(f"{src}_")]].rename(columns=lambda c: c[2:])
            gg["kind"] = g["kind"].values
            rows.append(dict(set=label, src="unity" if src == "u" else "replica", **E.score_rows(gg)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate.csv", index=False)
    pd.set_option("display.width", 460)
    pd.set_option("display.max_columns", 80)
    cols = ["set", "src", "n", "km", "v_coh", "one", "lost_km", "contact_km", "open", "hull", "cross_wrapped",
            "street_v_coh", "street_one", "face_t_split", "face_t_split_share", "ang_t_through", "ang_onset_spread",
            "along_spread", "along_o6_spread", "along_o9_spread"]
    print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    if errs:
        print(f"[swarmellipse_bench] {len(errs)} flights failed to score")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    for k, v in (("cross", 30), ("pass", 20), ("street", 50), ("face", 36), ("angle", 36), ("along", 0)):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int, default=v)
    p.add_argument("--k0", type=int, default=E.K_BENCH)
    p.add_argument("--drones", type=int, default=T.N_DRONES)
    p.add_argument("--d", type=float, help="fly every candidate at this d_ref instead of its own")
    p.add_argument("--timeout", type=int, default=180)
    p = sub.add_parser("gate")
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    dict(plan=cmd_plan, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
