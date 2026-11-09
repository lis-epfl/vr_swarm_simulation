"""S11 Wrap Hold in Unity: the bench's plan and gate for the wrap study, on S8's (swarmshape) with the search flight
S11 added, scored on Unity's own trajectory and on its replica twin (flown from Unity's recorded start) by the same
measures (swarmwrap): the wrap measures from the trajectory's `spread` column (SwarmBenchRunner, since S11) and the
alive drones' positions, the loss instants from its alive mask.

    python -B -m citysim.swarmwrap_bench plan --name N --cand FILE [--cross 20 --pass 20 --street 50 --face 12
                                              --search 40]
    python -B -m citysim.swarmwrap_bench gate --name N --runs DIR [DIR ...]
    python -B -m citysim.swarmwrap_bench losses --name N --run DIR     # every Unity loss placed against the spread
    -> experiment_results/swarm_wrap/<name>/bench/

A separate module from swarmwrap.py so it can be written and run while a replica run is importing that one
(lesson 18). Route indices from swarmwrap.K_BENCH; every flight is in the scene's city (seed 1), as baked. Unity's
contacts come as a count per flight, not timed, so its contacts are not placed by phase.
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

from citysim import swarmflow as F  # noqa: E402
from citysim import swarmshape as SH  # noqa: E402
from citysim import swarmtune as T  # noqa: E402
from citysim import swarmwrap as W  # noqa: E402

rep = T.rep


def route_of(C, kind, k, rng):
    """(start, wps, L, poly, info) of route k of a kind: the replica job's own (swarmwrap.run_job's streams)."""
    if kind == "face":
        return SH.face_route(C, T.TUNE_SEED, k, rng)
    if kind == "search":
        return W.search_route(C, T.TUNE_SEED, k, rng)
    return (*T.route(C, kind, k, rng), None)


def scenarios(flights, k0):
    """Every flight as the replica flies it, ending in END_HOLD_S at zero stick: S8's kinds with the zero-gain final
    hold they always had, the search with the bench's centring holds at HOLD_GAIN and a final hold capped at END_STICK
    (zero stick, as swarmwrap.fly ends)."""
    out = SH.bench_scenarios({k: flights.get(k, 0) for k in ("cross", "pass_", "street", "face")}, k0)
    C = T.city(T.TUNE_SEED)
    for k in range(k0, k0 + flights.get("search", 0)):
        rng = np.random.default_rng(T.route_seed("search", T.TUNE_SEED, k))
        rng.integers(1 << 30)
        start, wps, L, poly, info = W.search_route(C, T.TUNE_SEED, k, rng)
        wl = [list(w) for w in wps] + [[wps[-1][0], wps[-1][1], -W.END_STICK, T.END_HOLD_S]]
        out.append(dict(kind=f"search_{k}", sx=round(float(start[0]), 3), sz=round(float(start[1]), 3),
                        T=round(sum(w[3] for w in wps) + 10.0 + T.END_HOLD_S, 1),
                        wps=[round(float(v), 4) for w in wl for v in w], holdGain=W.HOLD_GAIN, record=True,
                        endOnDone=True))
    return out


def cmd_plan(a):
    from collections import OrderedDict
    from citysim import BENCH_DIR, camera
    spec = F.load_cands(a.cand)
    fields = T.sp.swarm_manager_fields()
    sets = []
    for c in spec:
        over = {k: (int(v) if isinstance(v, bool) else v) for k, v in dict(c["over"], d_ref=c["d"]).items()}
        sets.append(T.bench.parse_set(f"{c['label'].split()[0]}:{','.join(f'{k}={v}' for k, v in over.items())}",
                                      fields, True))
    scen = scenarios(dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face, search=a.search), a.k0)
    cfg = OrderedDict(scene=T.SCENE, timeScale=20.0, settleTime=T.BENCH_SETTLE_S, altitude=T.BENCH_ALTITUDE,
                      capture=T.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                      goalReplaySession="", layout="", droneCount=a.drones, walkers=False, paramSets=sets,
                      scenarios=scen)
    out = W.OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    cfile = out / "config.json"
    cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "cands.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"[swarmwrap_bench] {len(sets)} sets x {len(scen)} flights -> {cfile}\n"
          f"  & '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag swarmwrap_{a.name} -TimeoutMin {a.timeout}")


def traj_measures(tr, m, rc, wps, info, kind):
    """The wrap measures (and, for a search, its own) of a Unity trajectory: the rule's wrap test on the alive
    drones' positions every row (0.1 s), the scale from its spread column, the loss instants where the alive mask
    drops."""
    cent = W.building_centres(m, rc)
    t = np.asarray(tr["t"], float)
    wrap, ring = [], []
    for r in range(len(t)):
        al = tr["alive"][r]
        P = np.c_[tr["X"][r], tr["Z"][r]][al]
        wrap.append(W.wrapped_at(P, cent))
        ring.append(float(np.linalg.norm(P - P.mean(0), axis=1).mean()) if len(P) else math.nan)
    alive_n = tr["alive"].sum(1)
    loss_t = [float(t[r]) for r in range(1, len(t)) if alive_n[r] < alive_n[r - 1]]
    sig = np.asarray(tr["spread"], float)
    out = W.wrap_measures(t, sig, wrap, ring, loss_t, [])
    if kind == "search":
        done = [math.nan] * len(wps)
        for r in range(len(t)):
            for i in range(min(int(tr["wp"][r]), len(wps))):
                if not math.isfinite(done[i]):
                    done[i] = float(t[r])
        out.update(W.search_measures(info, done, t, sig, wrap, np.asarray(tr["wp"]), None))
    return out


def bench_job(job):
    """One bench flight: S8's scores on Unity's trajectory and on its twin (swarmshape.bench_job's method), with the
    wrap measures on both; the search flight's twin flown with swarmwrap.fly's holds."""
    run_dir, label, idx, sc, over, d, flight = job
    from pathlib import Path
    from types import SimpleNamespace
    from citysim import tierc
    try:
        kind, k = sc["kind"].rsplit("_", 1)
        k = int(k)
        C = T.city(T.TUNE_SEED)
        rng = np.random.default_rng(T.route_seed(kind, T.TUNE_SEED, k))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly, info = route_of(C, kind, k, rng)
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
        b = T.score_flight(C, start, wps, poly, L, ghost, samples, done, n0)
        b.update(traj_measures(tr, m, C["rcity"], wps, info, kind))
        b["contact_events"] = int(flight.get("contacts") or 0)
        if kind == "face":
            b.update(SH.face_measures(C, info, samples, done))
        b["deaths"] = deaths
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        log = W.WrapLog(m, C["rcity"])
        fly = W.fly if kind == "search" else T.fly
        sim, rs, rdone = fly(m, C, P0, wps, d, fseed, settle=0.0, hook=log)       # Unity's start is post-settle
        r = T.score_flight(C, start, wps, poly, L, sim, rs, rdone, sim.n0)
        r.update(W.log_measures(log))
        if kind == "face":
            r.update(SH.face_measures(C, info, rs, rdone))
        elif kind == "search":
            r.update(W.search_measures(info, rdone, log.t, log.sig, log.wrap, log.wp, rs))
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
    out = W.OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "gate.jsonl", orient="records", lines=True)
    rows = []
    for label, g in df.groupby("set", sort=False):
        for src in ("u", "r"):
            gg = g[[c for c in g.columns if c.startswith(f"{src}_")]].rename(columns=lambda c: c[2:])
            gg["kind"] = g["kind"].values
            rows.append(dict(set=label, src="unity" if src == "u" else "replica", **W.score_rows(gg)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate.csv", index=False)
    pd.set_option("display.width", 460)
    pd.set_option("display.max_columns", 80)
    cols = ["set", "src", "n", "km", "wrap_full", "wrap_sigma", "exit_sigma", "search_hold_full", "search_leave_v",
            "street_v_coh", "v_coh", "one", "loss_events", "lost_km", "contact_km", "open", "hull", "cross_wrapped",
            "face_t_split", "loss_wrap", "loss_exit", "loss_other"]
    print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    if errs:
        print(f"[swarmwrap_bench] {len(errs)} flights failed to score")


def cmd_losses(a):
    """Every Unity loss instant of the run, placed (lesson 32): its time and phase, whether any building had been
    wrapped before it in that flight, the scale then and 5 s before, and the base set's scale on the same route at the
    same moment. A loss where the two scales agree, before any wrap, cannot be the wrap rule's doing: the flights
    differ only by Unity's run-to-run divergence there. -> <name>/bench/losses.json"""
    import pandas as pd
    from pathlib import Path
    from citysim import tierc
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    idx = {sc["kind"]: i for i, sc in enumerate(cfg["scenarios"])}
    g = pd.read_json(W.OUT / a.name / "bench" / "gate.jsonl", lines=True)
    m = T.model(json.loads((W.OUT / a.name / "bench" / "cands.json").read_text(encoding="utf-8"))[0]["over"])
    cent = W.building_centres(m, T.city(T.TUNE_SEED)["rcity"])

    def traj(label, kind, k):
        return tierc.read_traj(run / f"traj_{label}_{idx[f'{kind}_{k}']}.csv")

    rows = []
    for _, r in g[g.u_loss_events > 0].iterrows():
        tr = traj(r.set, r.kind, r.k)
        n = tr["alive"].sum(1)
        wrap = np.array([W.wrapped_at(np.c_[tr["X"][i], tr["Z"][i]][tr["alive"][i]], cent) for i in range(len(tr["t"]))])
        ph = W.phases(tr["t"], wrap)
        base = traj(a.base, r.kind, r.k)
        for i in range(1, len(n)):
            if n[i] >= n[i - 1]:
                continue
            t = float(tr["t"][i])
            j = int(np.searchsorted(tr["t"], t - 5))
            b = int(min(np.searchsorted(base["t"], t), len(base["t"]) - 1))
            pre = np.flatnonzero(wrap[:i])
            rows.append(dict(set=r.set, kind=r.kind, k=int(r.k), t=round(t, 1), phase=["wrap", "exit", "other"][ph[i]],
                             drones=int(n[i - 1] - n[i]), wrapped_before=bool(len(pre)),
                             since_wrap=round(t - float(tr["t"][pre[-1]]), 1) if len(pre) else None,
                             scale=round(float(tr["spread"][i]), 3), scale_5s=round(float(tr["spread"][j]), 3),
                             base_scale=round(float(base["spread"][b]), 3)))
    out = W.OUT / a.name / "bench" / "losses.json"
    out.write_text(json.dumps(dict(base=a.base, losses=rows), indent=1), encoding="utf-8")
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(d.sort_values(["set", "kind", "k", "t"]).to_string(index=False))
    for s, gg in d.groupby("set"):
        same = int(((gg.scale - gg.base_scale).abs() <= 0.01).sum())
        print(f"{s}: {len(gg)} loss instants; the scale equal to {a.base}'s at the moment in {same}; before any wrap in "
              f"{int((~gg.wrapped_before).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("losses")
    p.add_argument("--name", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--base", default="E15_35a")
    p = sub.add_parser("plan")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    for k, v in (("cross", 20), ("pass", 20), ("street", 50), ("face", 12), ("search", 40)):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int, default=v)
    p.add_argument("--k0", type=int, default=W.K_BENCH)
    p.add_argument("--drones", type=int, default=T.N_DRONES)
    p.add_argument("--timeout", type=int, default=240)
    p = sub.add_parser("gate")
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    dict(plan=cmd_plan, gate=cmd_gate, losses=cmd_losses)[a.cmd](a)


if __name__ == "__main__":
    main()
