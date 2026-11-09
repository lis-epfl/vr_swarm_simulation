"""S4 Phase 1 on the headless Unity bench: the replica's manoeuvres as bench scenarios, and the gate.

    python -B -m citysim.split_bench plan --name N --variant R --setting ... [--designs 50:12 ...] [--trials 30]
        -> experiment_results/split_study/phase1/<N>/bench/: per layout a scenario file and a config, and run.ps1
    python -B -m citysim.split_bench gate --name N --runs <bench run dirs...>
        -> gate.csv: the S4 quantities scored on Unity's own trajectories (its shown feeds included) beside the replica
           re-flown from Unity's recorded starts, per design; and the validate verdict (swarm_replica.py validate)

The scenarios are the replica's own trials (split_probe's waypoint builders, the same random streams), so every
bench flight has a replica twin. The bench flies them under its pilot, which split_probe mirrors, with the per-waypoint
spread (wpSpread, SwarmBenchRunner) and the setting's hold gain; altitude 12.4 m (the search band preview is scored
at) and a pilot heading (the look-gap fill acts as in the scenes).
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
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import BENCH_DIR, blockcity, camera, split_probe as S, use_bench_modules  # noqa: E402

use_bench_modules()
import bench  # noqa: E402
import swarm_params as sp  # noqa: E402

BENCH_ALTITUDE = 12.4
SETTLE_S = 15.0
TIME_SCALE = 20.0
GATE_TOL = 0.15


def scenarios_for(variant, p, s, seed, trials, st, kinds=("M3", "M3b", "M4")):
    """[scenario dict] of one layout: the trials of split_probe's run that fall on this city seed."""
    C = S.city(variant, p, s, seed)
    out = []
    for t in range(trials):
        if S.CITY_SEEDS[t % len(S.CITY_SEEDS)] != seed:
            continue
        for kind in kinds:
            j = dict(kind=kind, variant=variant, p=float(p), s=float(s), city_seed=seed, trial=t, setting=st)
            rng = np.random.default_rng(S.hash_seed(j))
            rng.integers(1 << 30)                                   # the flight seed run_job draws first
            tr = S.pick(C["net"], rng)
            if tr is None:
                continue
            stx = dict(S.DEFAULT_SETTING, **st)
            if kind == "M3":
                built = S.m3_wps(C, tr, stx)
            elif kind == "M3b":
                built = S.m3b_wps(C, tr, stx, rng)
            else:
                built = S.m4_wps(C, tr, stx)
            if built is None:
                continue
            flights = [(kind, built["wps"])]
            if kind == "M3b" and built.get("street_wps"):
                flights.append(("M3bstreet", built["street_wps"]))
            start = C["net"].jpos[tr["approach"][0]]
            for k, wps in flights:
                out.append(dict(kind=f"{k}_{t}", sx=round(float(start[0]), 3), sz=round(float(start[1]), 3),
                                T=round(sum(w[3] for w in wps) + 10.0, 1),
                                wps=[round(float(v), 3) for w in wps for v in w[:4]],
                                wpSpread=[round(float(w[4]), 3) for w in wps], holdGain=float(stx["hold_gain"]),
                                record=True, endOnDone=True))
    return out


def plan(name, variant, designs, trials, st, profile_sets, coupling=None):
    """coupling: a split_probe coupling run whose d*(p, s) sets each design's block spread, as split_probe's
    --coupling does."""
    out = S.OUT_ROOT / name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    fields = sp.swarm_manager_fields()
    dmap = S.coupling_map(S.read(coupling)) if coupling else {}
    lines = []
    for p, s in designs:
        st_d = dict(st, d_rule="fixed", d_block=dmap[(variant, float(p), float(s))]) if dmap else st
        for seed in S.CITY_SEEDS:
            P = blockcity.Params(variant, p, s, seed)
            path = blockcity.ensure(P)
            scen = scenarios_for(variant, p, s, seed, trials, st_d)
            sfile = out / f"{P.ident()}_scenarios.json"
            sfile.write_text(json.dumps(dict(city=P.ident(), scenarios=scen), indent=1), encoding="utf-8")
            layout_name, _ = bench.layout_file_and_id(str(path))
            sets = [bench.parse_set(profile_sets, fields, True)]
            cfg = OrderedDict(scene=S.SCENE, timeScale=TIME_SCALE, settleTime=SETTLE_S, altitude=BENCH_ALTITUDE,
                              capture=S.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                              goalReplaySession="", layout=layout_name, droneCount=S.N_DRONES, walkers=False,
                              paramSets=sets, scenarios=scen)
            cfile = out / f"{P.ident()}_config.json"
            cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
            tag = f"s4_{name}_{P.ident()}"
            lines.append(f"& '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag '{tag}' -TimeoutMin 40")
    (out / "run.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[split_bench] {len(lines)} configs -> {out}; run {out / 'run.ps1'}")
    return out


# ============================================================================= the gate

def bench_samples(tr, single=False):
    """split_probe-shaped samples from a bench trajectory (tierc.read_traj): shown cameras at their drones' recorded
    yaws, 0.4 m ahead, every row (0.1 s). single: a lone drone's feed is always shown (the bench's shown mask is the
    hull rule's, which needs three drones)."""
    out = []
    fwd = camera.CAMERA_FORWARD_M
    for k in range(len(tr["t"])):
        alive = tr["alive"][k]
        P = np.where(alive[:, None], np.c_[tr["X"][k], tr["Z"][k]], np.nan)
        sh = alive.copy() if single else tr["shown"][k] & alive
        yaws = tr["yaw"][k][sh]
        y = np.radians(yaws)
        cams = np.c_[tr["X"][k][sh] + fwd * np.sin(y), tr["Z"][k][sh] + fwd * np.cos(y)]
        c = np.nanmean(P, axis=0)
        out.append(dict(t=float(tr["t"][k]), c=c, P=P, wp=int(tr["wp"][k]), hold=False, cams=cams, yaws=yaws,
                        body=float(tr["body"][k])))
    return out[::5]                       # 0.5 s, as the replica's


def wp_times(samples, n):
    """The time each waypoint was done: the first sample whose waypoint index passed it."""
    out = [math.nan] * n
    for smp in samples:
        for i in range(min(smp["wp"], n)):
            if not math.isfinite(out[i]):
                out[i] = smp["t"]
    return out


def score_flight(C, kind, samples, done, tr, sc_wps):
    """The S4 quantities of one flight (bench or replica): preview at the commit, reform, losses, split share."""
    net, fp = C["net"], C["fp"]
    marks = dict(M3=(1, 2, 3), M3b=(1, 2, 4), M4=(1, 2, 2))[kind]
    t_J, commit, t_next = (done[i] if i < len(done) else math.nan for i in marks)
    row = dict(t_commit=commit)
    if kind in ("M3", "M3b") and math.isfinite(commit):
        row.update({k: v for k, v in S.preview_scores(net, fp, samples, S.candidates(net, tr), commit,
                                                      S.bands()).items() if not k.startswith("preview_times")})
        ser = S.cluster_series(net, samples, t0=t_J if math.isfinite(t_J) else 0.0)
        row["split_share"] = float(np.mean([x[2] for x in ser])) if ser else math.nan
        row["reform_s"], row["reform_ok"] = S.reform(ser, commit, t_next if math.isfinite(t_next) else
                                                     samples[-1]["t"])
    if kind == "M4" and math.isfinite(t_J):
        row.update(S.envelop_series(C, samples, tr, t_J, commit if math.isfinite(commit) else samples[-1]["t"]))
    return row


def gate(name, runs, variant, st):
    import pandas as pd
    from citysim import tierc
    rows = []
    for run in runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        lid = Path(cfg["layout"]).stem
        _, p, s, seed = lid.split("_")[0], float(lid.split("_p")[1].split("_")[0]), \
            float(lid.split("_s")[1].split("_")[0]), int(lid.split("seed")[1])
        C = S.city(variant, p, s, seed)
        res = {f["scenario"]: f for f in bench.read_results(run)[3]}      # (jsonl, start, params, flights, ...)
        for idx, sc in enumerate(cfg["scenarios"]):
            kind, trial = sc["kind"].rsplit("_", 1)
            if kind not in ("M3", "M3b", "M4"):
                continue
            j = dict(kind=kind, variant=variant, p=p, s=s, city_seed=seed, trial=int(trial), setting=st)
            rng = np.random.default_rng(S.hash_seed(j))
            fseed = int(rng.integers(1 << 30))
            trl = S.pick(C["net"], rng)
            traj = run / f"traj_{cfg['paramSets'][0]['label']}_{idx}.csv"
            if trl is None or not traj.exists():
                continue
            tr = tierc.read_traj(traj)
            bs = bench_samples(tr)
            nwp = len(sc["wps"]) // 4
            b = score_flight(C, kind, bs, wp_times(bs, nwp), trl, sc["wps"])
            fl = res.get(idx, {})
            b.update(lost=int(tr["alive"][0].sum() - tr["alive"][-1].sum()), contacts=fl.get("contacts"))
            # the replica twin from Unity's recorded start
            P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
            wps = [(sc["wps"][4 * i], sc["wps"][4 * i + 1], sc["wps"][4 * i + 2], sc["wps"][4 * i + 3],
                    sc["wpSpread"][i]) for i in range(nwp)]
            m = S.model(dict(S.DEFAULT_SETTING, **st), variant)
            sim, rs, done = S.fly(m, C["rcity"], P0, wps, float(dict(S.DEFAULT_SETTING, **st)["d_street"]), sc["T"],
                                  fseed, S.bands()["search"][1], hold_gain=float(sc.get("holdGain") or S.HOLD_GAIN))
            r = score_flight(C, kind, rs, done, trl, sc["wps"])
            r.update(lost=int(len(P0) - sim.alive.sum()), contacts=int(sim.contacts))
            rows.append(dict(layout=lid, p=p, s=s, seed=seed, kind=kind, trial=int(trial),
                             **{f"bench_{k}": v for k, v in b.items()}, **{f"replica_{k}": v for k, v in r.items()}))
    df = pd.DataFrame(rows)
    out = S.OUT_ROOT / name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "gate_flights.csv", index=False)
    summary = gate_summary(df)
    summary.to_csv(out / "gate.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    return summary


def gate_summary(df):
    """Per design x kind x quantity: bench and replica means, the relative difference, the bench's standard error as a
    share of its mean, and the verdict: agree within GATE_TOL (and the bench's SE within GATE_TOL / 2, or more
    flights are needed)."""
    import pandas as pd
    out = []
    for (p, s, kind), g in df.groupby(["p", "s", "kind"]):
        for q in ("preview_search", "reform_ok", "split_share", "lost", "contacts", "enveloped", "t_envelop"):
            bc, rc = f"bench_{q}", f"replica_{q}"
            if bc not in g or rc not in g:
                continue
            b = pd.to_numeric(g[bc], errors="coerce").astype(float).dropna()
            r = pd.to_numeric(g[rc], errors="coerce").astype(float).dropna()
            if not len(b) or not len(r):
                continue
            bm, rm = float(b.mean()), float(r.mean())
            se = float(b.std(ddof=1) / math.sqrt(len(b))) if len(b) > 1 else math.nan
            scale = max(abs(bm), 1e-9)
            rel = abs(rm - bm) / scale if scale > 1e-6 else (0.0 if abs(rm) < 1e-6 else math.inf)
            se_rel = se / scale if scale > 1e-6 else 0.0
            verdict = "agree" if rel <= GATE_TOL else ("more flights" if se_rel > GATE_TOL / 2 else "differ")
            out.append(dict(p=p, s=s, kind=kind, quantity=q, n=len(b), bench=round(bm, 3), replica=round(rm, 3),
                            rel_diff=round(rel, 3), bench_se_rel=round(se_rel, 3), verdict=verdict))
    return pd.DataFrame(out)


# ============================================================================= Phase 3: Phase 2's chain walks on the bench

# single_look is replica-only: on the bench a lone drone yaws only to its stick, so it cannot look at zero stick
WALK_CONDITIONS = ("single", "swarm_street", "swarm_hold", "swarm_pass", "swarm_hop")


def walk_scenarios(spec, variant, p, s, city_seed, walks):
    """{drone count: [scenario]} for Phase 2's first `walks` chain walks of one city, exactly as split.walk_job lays
    them (the same random stream, the profile's spread), every bench-flyable condition."""
    from citysim import split
    st = dict(S.DEFAULT_SETTING, **spec["setting"])
    C = split.context(variant, p, s, city_seed)
    out = {1: [], S.N_DRONES: []}
    for k in range(walks):
        rng = np.random.default_rng(S.hash_seed(dict(kind="walk", variant="chain", p=p, s=s, city_seed=city_seed,
                                                     trial=k)))
        rng.integers(1 << 30)                                       # the flight seed walk_job draws first
        chain = split.block_chain(C["net"], rng)
        if chain is None:
            continue
        dfn = split.block_spread_fn(spec, st, C["net"], C["L"], variant, p, s, "profile")
        start = C["net"].jpos[chain["approach"][0]]
        for cond in WALK_CONDITIONS:
            wps, _ = split.condition_wps(cond, C, chain, st, dfn)
            n = 1 if cond == "single" else S.N_DRONES
            out[n].append(dict(kind=f"{cond}_{k}", sx=round(float(start[0]), 3), sz=round(float(start[1]), 3),
                               T=float(spec["T_s"]), wps=[round(float(v), 3) for w in wps for v in w[:4]],
                               wpSpread=[round(float(w[4]), 3) for w in wps], holdGain=float(st["hold_gain"]),
                               record=True, endOnDone=False))
    return out


def plan_walks(name, spec_path, designs, cities, walks, profile_sets):
    """Bench configs for Phase 2's chain walks: per design x city a swarm config and a one-drone config."""
    from citysim import split
    spec = split.load_spec(spec_path)
    out = S.OUT_ROOT.parent / name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    fields = sp.swarm_manager_fields()
    seed0 = int(spec.get("city_seed0", 0))
    lines = []
    for p, s in designs:
        for c in range(seed0, seed0 + cities):
            P = blockcity.Params(spec["variant"], p, s, c)
            path = blockcity.ensure(P)
            layout_name, _ = bench.layout_file_and_id(str(path))
            for n, scen in walk_scenarios(spec, spec["variant"], float(p), float(s), c, walks).items():
                if not scen:
                    continue
                cfg = OrderedDict(scene=S.SCENE, timeScale=TIME_SCALE, settleTime=SETTLE_S, altitude=BENCH_ALTITUDE,
                                  capture=S.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                                  goalReplaySession="", layout=layout_name, droneCount=n, walkers=False,
                                  paramSets=[bench.parse_set(profile_sets, fields, True)], scenarios=scen)
                tag = f"s4_{name}_{P.ident()}_n{n}"
                cfile = out / f"{P.ident()}_n{n}_config.json"
                cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
                lines.append(f"& '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag '{tag}' -TimeoutMin 90")
    (out / "run.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[split_bench] {len(lines)} walk configs -> {out}; run {out / 'run.ps1'}")
    return out


def gate_walks(name, runs, spec_path, replica_run):
    """Phase 2's walk measures scored on the bench's own trajectories, beside the replica's flights of the same walks
    (replica_run: Phase 2's walks run): per condition, and the contrasts the hypotheses rest on, per walk."""
    import pandas as pd
    from citysim import split, tierc
    spec = split.load_spec(spec_path)
    st = dict(S.DEFAULT_SETTING, **spec["setting"])
    rep_rows = {}
    for line in (S.OUT_ROOT.parent / replica_run / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for x in r.get("rows", []):
            if x["spread"] == "profile":
                rep_rows[(r["p"], r["s"], r["city_seed"], r["walk"], x["condition"])] = x
    rows = []
    for run in runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        lid = Path(cfg["layout"]).stem
        p, s, seed = float(lid.split("_p")[1].split("_")[0]), float(lid.split("_s")[1].split("_")[0]), \
            int(lid.split("seed")[1])
        C = split.context(spec["variant"], p, s, seed)
        res = {f["scenario"]: f for f in bench.read_results(run)[3]}
        for idx, sc in enumerate(cfg["scenarios"]):
            cond, k = sc["kind"].rsplit("_", 1)
            traj = run / f"traj_{cfg['paramSets'][0]['label']}_{idx}.csv"
            if not traj.exists():
                continue
            rng = np.random.default_rng(S.hash_seed(dict(kind="walk", variant="chain", p=p, s=s, city_seed=seed,
                                                         trial=int(k))))
            rng.integers(1 << 30)
            chain = split.block_chain(C["net"], rng)
            dfn = split.block_spread_fn(spec, st, C["net"], C["L"], spec["variant"], p, s, "profile")
            wps, marks = split.condition_wps(cond, C, chain, st, dfn)
            tr = tierc.read_traj(traj)
            smp = bench_samples(tr, single=cond == "single")
            done = wp_times(smp, len(wps))

            class _Sim:                         # what score_flight reads off a replica sim
                alive = tr["alive"][-1]
                kills = []
                contacts = int(res.get(idx, {}).get("contacts", 0))
            b = split.score_flight(C, cond, smp, done, marks, chain, _Sim, int(tr["alive"][0].sum()),
                                   float(spec["T_s"]), C["find_grid"], C["in_arena"], S.bands(), C["corr_fp"])
            hh = [h for h in b["hops"] if not h.get("plaza")]
            r = rep_rows.get((p, s, seed, int(k), cond), {})
            rh = [h for h in r.get("hops", []) if not h.get("plaza")]
            rows.append(dict(p=p, s=s, seed=seed, walk=int(k), condition=cond, bench_hops=b["hops_done"],
                             replica_hops=r.get("hops_done"),
                             bench_preview=float(np.mean([h["preview_search"] for h in hh])) if hh else np.nan,
                             replica_preview=float(np.mean([h["preview_search"] for h in rh])) if rh else np.nan,
                             bench_street_km=b["street_seen_km"], replica_street_km=r.get("street_seen_km"),
                             bench_find80=b["find_rmst_any_r80"], replica_find80=r.get("find_rmst_any_r80"),
                             bench_lost=b["lost"], replica_lost=r.get("lost"), bench_contacts=b["contacts"],
                             replica_contacts=r.get("contacts")))
    df = pd.DataFrame(rows)
    out = S.OUT_ROOT.parent / name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "walk_gate_flights.csv", index=False)
    g = df.groupby("condition")[[c for c in df.columns if c.startswith(("bench_", "replica_"))]].mean().round(3)
    g.to_csv(out / "walk_gate.csv")
    pd.set_option("display.width", 250)
    print(g.to_string())
    return df


# ============================================================================= Phase 3: informed routes open-loop

def _route_job(args):
    """Re-fly one informed search target in the replica, every bench-flyable condition, recording each flight's
    waypoints (the route its decisions laid): the flights are deterministic in their seeds, so the route is the one
    Phase 2 flew. Runs in a worker; patches split_probe.Flight there only."""
    spec, variant, p, s, city_seed, target_i = args
    from citysim import split
    made = []

    class Recording(S.Flight):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)
    S.Flight = Recording
    st = dict(S.DEFAULT_SETTING, **spec["setting"])
    C = split.context(variant, p, s, city_seed)
    rng = np.random.default_rng(S.hash_seed(dict(kind="search", variant="target", p=p, s=s, city_seed=city_seed,
                                                 trial=target_i)))
    seed = int(rng.integers(1 << 30))
    start = S.pick(C["net"], rng)
    target = split.place_target(C, start["approach"][0], rng) if start else None
    if target is None:
        return None
    dfn = split.block_spread_fn(spec, st, C["net"], C["L"], variant, p, s, "profile")
    out = []
    for cond in WALK_CONDITIONS:
        made.clear()
        row = split.informed_flight(cond, C, variant, start, target, st, dfn, seed, float(spec.get("search_T_s", 1200.0)),
                                    S.bands(), "any")
        f = made[-1]
        out.append(dict(condition=cond, wps=[tuple(map(float, w)) for w in f.wps], t_end=row["t_end"],
                        found_r80=row["found_r80"], found_r200=row["found_r200"], lost=row["lost"]))
    return dict(p=p, s=s, city_seed=city_seed, target_i=target_i, target=[float(x) for x in target],
                start=[float(x) for x in C["net"].jpos[start["approach"][0]]], seed=seed, routes=out)


def plan_routes(name, spec_path, designs, cities, targets, profile_sets, procs=8):
    """Bench configs re-flying informed-search routes open-loop: per design x city a swarm config and a one-drone
    config, every bench-flyable condition on each target; routes.json keeps the replica's own results beside them."""
    from multiprocessing import Pool
    from citysim import split
    spec = split.load_spec(spec_path)
    seed0 = int(spec.get("city_seed0", 0))
    jobs = [(spec, spec["variant"], float(p), float(s), c, k) for p, s in designs
            for c in range(seed0, seed0 + cities) for k in range(targets)]
    with Pool(min(procs, len(jobs))) as pool:
        res = [r for r in pool.map(_route_job, jobs) if r is not None]
    out = S.OUT_ROOT.parent / name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "routes.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    fields = sp.swarm_manager_fields()
    lines = []
    for (p, s, c) in sorted({(r["p"], r["s"], r["city_seed"]) for r in res}):
        P = blockcity.Params(spec["variant"], p, s, c)
        layout_name, _ = bench.layout_file_and_id(str(blockcity.ensure(P)))
        scen = {1: [], S.N_DRONES: []}
        for r in [x for x in res if (x["p"], x["s"], x["city_seed"]) == (p, s, c)]:
            for rt in r["routes"]:
                n = 1 if rt["condition"] == "single" else S.N_DRONES
                scen[n].append(dict(kind=f"{rt['condition']}_{r['target_i']}", sx=round(r["start"][0], 3),
                                    sz=round(r["start"][1], 3), T=round(rt["t_end"] + 5.0, 1),
                                    wps=[round(v, 3) for w in rt["wps"] for v in w[:4]],
                                    wpSpread=[round(w[4], 3) for w in rt["wps"]],
                                    holdGain=float(dict(S.DEFAULT_SETTING, **spec["setting"])["hold_gain"]),
                                    record=True, endOnDone=True))
        for n, sc in scen.items():
            if not sc:
                continue
            cfg = OrderedDict(scene=S.SCENE, timeScale=TIME_SCALE, settleTime=SETTLE_S, altitude=BENCH_ALTITUDE,
                              capture=S.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                              goalReplaySession="", layout=layout_name, droneCount=n, walkers=False,
                              paramSets=[bench.parse_set(profile_sets, fields, True)], scenarios=sc)
            cfile = out / f"{P.ident()}_n{n}_config.json"
            cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
            lines.append(f"& '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag 's4_{name}_{P.ident()}_n{n}' "
                         "-TimeoutMin 90")
    (out / "run.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[split_bench] {sum(len(r['routes']) for r in res)} routes, {len(lines)} configs -> {out}")
    return out


def gate_routes(name, runs, spec_path):
    """The open-loop routes scored on Unity's trajectories (the target's first sight at 80 / 200 m on the shown feeds,
    split.Knowledge's rule) beside the replica's own flights of the same routes."""
    import pandas as pd
    from citysim import split, tierc
    spec = split.load_spec(spec_path)
    routes = json.loads((S.OUT_ROOT.parent / name / "bench" / "routes.json").read_text(encoding="utf-8"))
    rows = []
    for run in runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        lid = Path(cfg["layout"]).stem
        p, s, seed = float(lid.split("_p")[1].split("_")[0]), float(lid.split("_s")[1].split("_")[0]), \
            int(lid.split("seed")[1])
        C = split.context(spec["variant"], p, s, seed)
        h, pitch = S.bands()["search"]
        res = {f["scenario"]: f for f in bench.read_results(run)[3]}
        for idx, sc in enumerate(cfg["scenarios"]):
            cond, k = sc["kind"].rsplit("_", 1)
            r = next((x for x in routes if (x["p"], x["s"], x["city_seed"], x["target_i"]) == (p, s, seed, int(k))),
                     None)
            traj = run / f"traj_{cfg['paramSets'][0]['label']}_{idx}.csv"
            if r is None or not traj.exists():
                continue
            rt = next(x for x in r["routes"] if x["condition"] == cond)
            tr = tierc.read_traj(traj)
            know = split.Knowledge(C["net"], C["fp"], h, pitch, np.array(r["target"]), "any", cond == "single")
            know.update(bench_samples(tr, single=cond == "single"))
            f80, f200 = know.found[80.0], know.found[200.0]
            rows.append(dict(p=p, s=s, seed=seed, target=int(k), condition=cond,
                             bench_found_r80=f80 if math.isfinite(f80) else None, replica_found_r80=rt["found_r80"],
                             bench_found_r200=f200 if math.isfinite(f200) else None,
                             replica_found_r200=rt["found_r200"],
                             bench_lost=int(tr["alive"][0].sum() - tr["alive"][-1].sum()), replica_lost=rt["lost"],
                             bench_contacts=int(res.get(idx, {}).get("contacts", 0))))
    df = pd.DataFrame(rows)
    out = S.OUT_ROOT.parent / name / "bench"
    df.to_csv(out / "route_gate.csv", index=False)
    both = df.dropna(subset=["bench_found_r80", "replica_found_r80"])
    agree = (both.bench_found_r80 - both.replica_found_r80).abs() <= 10.0
    print(df.to_string(index=False))
    print(f"\n[split_bench] {len(df)} routes; found on both {len(both)}, within 10 s {int(agree.sum())}; found on "
          f"one only {int(df.bench_found_r80.isna().ne(df.replica_found_r80.isna()).sum())}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan-routes")
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--designs", nargs="+", required=True)
    p.add_argument("--cities", type=int, default=2)
    p.add_argument("--targets", type=int, default=2)
    p.add_argument("--sets", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p = sub.add_parser("gate-routes")
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p = sub.add_parser("plan-walks")
    p.add_argument("--name", required=True, help="the output folder under experiment_results/split_study/")
    p.add_argument("--spec", required=True)
    p.add_argument("--designs", nargs="+", required=True)
    p.add_argument("--cities", type=int, default=4)
    p.add_argument("--walks", type=int, default=4)
    p.add_argument("--sets", required=True, help="the bench parameter set of the spec's profile (bench.py --set)")
    p = sub.add_parser("gate-walks")
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--replica", required=True, help="Phase 2's walks run (experiment_results/split_study/<run>)")
    p.add_argument("--runs", nargs="+", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--name", required=True)
    p.add_argument("--variant", choices=blockcity.VARIANTS, required=True)
    p.add_argument("--designs", nargs="+", default=["50:12", "50:24", "75:18", "40:18"])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--setting", action="append")
    p.add_argument("--sets", default="s4:r0_coh=40", help="the bench parameter set (bench.py --set syntax)")
    p.add_argument("--coupling", help="a split_probe coupling run whose d*(p, s) sets each design's block spread")
    p = sub.add_parser("gate")
    p.add_argument("--name", required=True)
    p.add_argument("--variant", choices=blockcity.VARIANTS, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--setting", action="append")
    a = ap.parse_args()
    if a.cmd == "plan-routes":
        plan_routes(a.name, a.spec, [tuple(float(x) for x in d.split(":")) for d in a.designs], a.cities, a.targets,
                    a.sets, a.jobs)
        return
    if a.cmd == "gate-routes":
        gate_routes(a.name, a.runs, a.spec)
        return
    if a.cmd == "plan-walks":
        plan_walks(a.name, a.spec, [tuple(float(x) for x in d.split(":")) for d in a.designs], a.cities, a.walks,
                   a.sets)
        return
    if a.cmd == "gate-walks":
        gate_walks(a.name, a.runs, a.spec, a.replica)
        return
    st = S.parse_setting(a.setting)
    if a.cmd == "plan":
        designs = [tuple(float(x) for x in d.split(":")) for d in a.designs]
        plan(a.name, a.variant, designs, a.trials, st, a.sets, a.coupling)
    else:
        gate(a.name, a.runs, a.variant, st)


if __name__ == "__main__":
    main()
