"""The density study explorer: one self-contained HTML page with the study's data in it -- the level codes decoded,
the generated cities, a goal's envelopment animated, Unity bench flights played back, every tier's results and the
checks. Rebuild it after the tiers re-run:

    python -B -m citysim.explorer             # -> experiment_results/density_study/explorer/density_study.html

The page is citysim/explorer.html, a template whose DATA placeholder this fills; it draws with canvas and d3 (from
cdnjs) and opens in any browser. What it shows comes from the files the tiers wrote, plus three goal scenes and the
walkers of seven bench flights recomputed here with the same engine (exposure.recognised_3d, walkers.GoalWalkers
seeded as Tier B and Tier C seed them), and one replica replay of a bench flight (swarm_replica.replay_flight).
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
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import (LAYOUT_DIR, PACKAGE_DIR, REPO, STUDY_DIR, camera, catalogue, display, exposure,  # noqa: E402
                     formation, layout as lay, operator_band, single, theory, tierc, vissim, walkers)
from citysim.los import Boxes, Footprints  # noqa: E402

TEMPLATE = PACKAGE_DIR / "explorer.html"
OUT = STUDY_DIR / "explorer" / "density_study.html"
DEFAULT_ROOT_RESULTS = REPO / "experiment_results" / "internal_3" / "results"     # the committed internal_3 snapshot
ALT, PITCH = 12.4, camera.DIAL_CENTRE_PITCH_DEG      # the scenes' camera: internal_3's median, the dial centre
SCENE_S, SCENE_DT = 60.0, 0.5                         # Tier B's exposure window and clock
FLIGHT_DT, WALKER_DT = 0.5, 1.0                       # flight frames, walker frames during a hold
ROLE = {"city": 0, "goal_centre": 1, "goal_ring": 2}
SEEN = np.array(list(".0123456789abcdefghijklmnopqrstuvwxyz"))
REPLICA_FLIGHT = ("L3", "n10", "goals_0.4")           # the flight drawn with the replica beside it


# ============================================================================= helpers

def clean(v):
    """JSON-safe: NaN and inf to None, numpy to Python, recursively."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    return v


def rnd(a, nd=1):
    return np.round(np.asarray(a, float), nd)


def records(df, cols=None, nd=4):
    d = df[cols] if cols else df
    out = []
    for r in d.to_dict("records"):
        out.append({k: (round(v, nd) if isinstance(v, float) and math.isfinite(v) else v) for k, v in r.items()})
    return out


def seen_codes(vis):
    """(W, T, K) recognition -> per frame a string of W characters: the first feed that sees the walker, '.' none."""
    first = np.where(vis.any(2), vis.argmax(2), -1)           # (W, T)
    return ["".join(SEEN[np.clip(first[:, j] + 1, 0, len(SEEN) - 1)]) for j in range(first.shape[1])]


# ============================================================================= cities

def layout_geo(path, label, level=None, seed=None):
    L = lay.load(path)
    boxes = []
    for b in L.boxes:
        cx, cz, hx, hz = lay.box_rect(b)
        boxes.append([round(cx - hx, 1), round(cz - hz, 1), round(cx + hx, 1), round(cz + hz, 1),
                      ROLE.get(b.get("role", "city"), 0)])
    streets = [[round(float(s["width"]), 1), [round(float(v), 1) for p in s["polyline"] for v in p]]
               for s in L.streets]
    goals = [dict(i=int(g["index"]), c=rnd(g["centre"]), r_g=g["r_g"], c_g=round(g["c_g"], 1),
                  clear=round(g["clear_radius"], 1), w_c=g["w_c"], n_b=g["n_b"], R_c=round(g["R_c"], 1))
             for g in L.goals()]
    sp = L.layout["spawn"]
    r = L.layout
    return dict(id=L.id, label=label, level=level, seed=seed, arena=[round(float(v), 1) for v in r["arena"]],
                spawn=[round(sp[0], 1), round(sp[2], 1)], boxes=boxes, streets=streets, goals=goals,
                stats=dict(alpha=r.get("alpha_achieved"), ell=r.get("ell_mean"), n=r.get("n_buildings"),
                           minstreet=r.get("min_street_width"), max50=r.get("max_within_50m"), p=r.get("p"),
                           s=r.get("s"), size=r.get("size")))


def trial_path(name, level, seed=101):
    hits = sorted(LAYOUT_DIR.glob(f"{name}_{level}__*_seed{seed}.json"))
    return hits[0] if hits else None


# ============================================================================= a goal, up close (Tier B's machinery)

def goal_scene(path, name, gi=0):
    """One goal of a trial layout as Tier B sees it: the swarm settled on it at three spreads, the single drone's
    policies, and every walker over Tier B's 60 s window with which camera (if any) recognises it."""
    L = lay.load(path)
    boxes = L.boxes_as_tuples()
    B3, fp = Boxes(boxes), Footprints(boxes)
    g = L.goals()[gi]
    info = vissim.goal_info(L, g)
    c = info["centre"]
    seed = int(L.layout["seed"])
    near_i = B3.near(c, 100.0)
    B3n = B3.subset(near_i)
    fpn = fp.subset(fp.near(c, vissim.NEAR_FP_M))
    replica_boxes = [boxes[i] for i in B3.near(c, vissim.NEAR_REPLICA_M)]
    rng = np.random.default_rng(10_000 * seed + 17 * info["index"] + 3)
    W = walkers.GoalWalkers(info["ring_rects"], rng)
    t = np.arange(0.0, SCENE_S, SCENE_DT)
    pts = W.positions(t)
    band = operator_band.load()
    hy, hw = operator_band.head_weights(band)
    body = (info["approach_deg"] + 180.0) % 360.0               # facing the goal on arrival
    near = []
    for i in near_i:
        b = L.boxes[i]
        cx, cz, hx, hz = lay.box_rect(b)
        near.append([round(cx - hx, 1), round(cz - hz, 1), round(cx + hx, 1), round(cz + hz, 1),
                     ROLE.get(b.get("role", "city"), 0)])
    loops = []
    for b in range(len(info["ring_rects"])):
        k = int(np.flatnonzero(W.building == b)[0])
        lp = W.loops[k]
        loops.append(rnd(lp.local_xz(np.linspace(0.0, lp.perimeter, 44, endpoint=False)) + [lp.cx, lp.cz], 2))
    options = []
    for d in (0.4, 1.0, 1.6):
        env = formation.envelop(replica_boxes, c, d, n=10, seed=seed, approach_deg=info["approach_deg"],
                                start="arrive", inner_radius=info["c_g"])
        P = env["P"]
        feeds = display.swarm_feeds(P, body)
        tr = exposure.static_track(feeds, len(t), SCENE_DT, ALT, PITCH, body)
        vis, off = exposure.recognised_3d(B3n, tr, pts)
        m = exposure.summarise(vis, off, tr, hy, hw)
        options.append(dict(kind="swarm", key=f"swarm{d:.1f}", label=f"Swarm at spread {d:.1f}", d_ref=d,
                            drones=rnd(P, 2), feeds=dict(drone=feeds.drones, pos=rnd(feeds.pos, 2), yaw=rnd(feeds.yaw)),
                            enveloped=env["enveloped"], surrounds=env["surrounds"], max_gap=round(env["max_gap_deg"], 1),
                            inside=round(env["inside_frac"], 2), kills=env["kills"], E_any=m["E_any"],
                            E_head=m["E_head"], target=float(vis[W.target].any(1).mean()), seen=seen_codes(vis)))
    pols = single.policies(fpn, info, W, t, ALT, PITCH, info["approach_deg"], band=True)
    names = dict(orbit_in="Single drone, orbit inside the ring", orbit_out="Single drone, orbit outside the ring",
                 scan="Single drone, scan from one spot", static="Single drone, best fixed vantage")
    for pol in ("orbit_in", "orbit_out", "scan", "static"):
        if pol not in pols:
            continue
        tr = pols[pol][0]
        vis, off = exposure.recognised_3d(B3n, tr, pts)
        m = exposure.summarise(vis, off, tr)
        options.append(dict(kind="single", key=pol, label=names[pol], policy=pol,
                            pos=rnd(tr.pos[:, 0], 2), yaw=rnd(tr.yaw[:, 0]), E_any=m["E_any"], E_head=m["E_any"],
                            target=float(vis[W.target].any(1).mean()), seen=seen_codes(vis)))
    return dict(name=name, layout=L.id, goal=info["index"], centre=rnd(c, 2), w_c=info["w_c"], r_g=info["r_g"],
                c_g=round(info["c_g"], 1), n_b=info["n_b"], R_c=round(info["R_c"], 1), clear=round(info["clear_radius"], 1),
                approach=round(info["approach_deg"], 1), body=round(body, 1), boxes=near, loops=loops,
                building=W.building, target=int(W.target), dt=SCENE_DT, walkers=rnd(pts.transpose(1, 0, 2), 1),
                options=options)


# ============================================================================= Unity bench flights (Tier C)

def flight(run_dir, fi, title, replica=False):
    cfg, start, flights, L = tierc.read_run(run_dir)
    f = flights[fi]
    sc = cfg["scenarios"][int(f["scenario"])]
    tr = tierc.read_traj(Path(run_dir) / f"traj_{f['set']}_{int(f['scenario'])}.csv")
    step = max(1, int(round(FLIGHT_DT / max(float(np.median(np.diff(tr["t"]))), 1e-6))))
    rows = np.arange(0, len(tr["t"]), step)
    n = tr["n"]
    bits = 1 << np.arange(n)
    mask = lambda m: (m[rows].astype(np.int64) * bits).sum(1)    # noqa: E731
    holds, hold_walkers = [], []
    boxes = L.boxes_as_tuples()
    B3 = Boxes(boxes)
    single_drone = n == 1
    for order, (k, g) in enumerate(tierc.hold_goals(sc, L)):
        hr = np.flatnonzero(tr["wp"] == k)
        if len(hr) < 4:
            continue
        t0, t1 = float(tr["t"][hr[0]]), float(tr["t"][hr[-1]])
        holds.append(dict(goal=int(g["index"]), visit=order, t0=round(t0, 1), t1=round(t1, 1)))
        info = vissim.goal_info(L, g)
        rng = np.random.default_rng(10_000 * int(L.layout["seed"]) + 17 * info["index"] + 3)
        Wk = walkers.GoalWalkers(info["ring_rects"], rng)
        ts = np.arange(0.0, t1 - t0, WALKER_DT)
        rs = np.clip(np.searchsorted(tr["t"], t0 + ts), hr[0], hr[-1])
        pts = Wk.positions(ts)
        y = np.radians(tr["yaw"][rs])
        pos = np.stack([tr["X"][rs], tr["Z"][rs]], -1) + camera.CAMERA_FORWARD_M * np.stack([np.sin(y), np.cos(y)], -1)
        valid = tr["alive"][rs] & (tr["shown"][rs] | single_drone)
        track = exposure.Track(pos=pos, yaw=tr["yaw"][rs], h=tr["Y"][rs], valid=valid, pitch=np.full(len(rs), PITCH),
                               body_yaw=tr["body"][rs], dt=WALKER_DT, single=single_drone)
        vis, _ = exposure.recognised_3d(B3.subset(B3.near(info["centre"], camera.R_ID_M + 120.0)), track, pts)
        hold_walkers.append(dict(goal=int(g["index"]), t0=round(t0, 1), dt=WALKER_DT, target=int(Wk.target),
                                 walkers=rnd(pts.transpose(1, 0, 2), 1), seen=seen_codes(vis),
                                 target_seen=float(vis[Wk.target].any(1).mean()),
                                 E_any=float(vis.any(2).mean())))
    alive = tr["alive"]
    deaths = []
    for i in range(n):
        gone = np.flatnonzero(alive[:-1, i] & ~alive[1:, i])
        if len(gone):
            j = int(gone[0])
            deaths.append(dict(drone=i, t=round(float(tr["t"][j + 1]), 1), x=round(float(tr["X"][j, i]), 1),
                               z=round(float(tr["Z"][j, i]), 1)))
    out = dict(title=title, run=Path(run_dir).name, layout=L.id, kind=f["kind"], order=sc.get("order", ""), n=n,
               dt=FLIGHT_DT, t=rnd(tr["t"][rows]), x=rnd(tr["X"][rows]), z=rnd(tr["Z"][rows]),
               yaw=np.round(tr["yaw"][rows]).astype(int), alive=mask(tr["alive"]), hull=mask(tr["hull"] & tr["alive"]),
               shown=mask(tr["shown"] & tr["alive"]), wp=tr["wp"][rows], body=np.round(tr["body"][rows]).astype(int),
               holds=holds, hold_walkers=hold_walkers, deaths=deaths,
               summary=dict(lost=int(f["alive_start"]) - int(f["alive_end"]), contacts=f.get("contacts", 0),
                            minutes=round(float(f.get("sim_s", 0.0)) / 60.0, 1), path_km=round(f.get("path_len", 0) / 1000, 2),
                            deaths=f.get("deaths") or {}, nn=f.get("nn_mean"), min_pair=f.get("min_pair"),
                            hull_frac=f.get("hull_frac")))
    if replica:
        out["replica"] = replica_track(run_dir, f, sc, rows)
    return out


def replica_track(run_dir, f, sc, rows):
    """The same flight flown again by the 2-D replica (swarm_replica.replay_flight, ContactSim, as validate does)."""
    from citysim import use_bench_modules
    use_bench_modules()
    import swarm_params as sp
    import swarm_replica as rep
    cfg, start, params, flights = rep.read_bench_run(run_dir)
    job = (str(run_dir), f["set"], int(f["scenario"]), sc, params[f["set"]]["effective"], dict(sp.resolve_airframe()),
           float(cfg.get("capture", 10.0)), rep.bench_city(run_dir, cfg), 0, "contact", rep.PHYSX_FRICTION)
    res = rep.replay_flight(job, keep_positions=True)
    ts = np.array([p[0] for p in res["positions"]])
    P = np.array([p[1] for p in res["positions"]])                   # (T, n, 2)
    keep = np.clip(np.searchsorted(ts, np.arange(0.0, ts[-1] + 1e-9, FLIGHT_DT)), 0, len(ts) - 1)
    P = P[keep]
    return dict(t=rnd(ts[keep]), x=rnd(P[..., 0]), z=rnd(P[..., 1]), lost=res["lost_replica"],
                kills=res["replica_kills"], track_rmse=res["track_rmse_m"], path_m=res["path_m"])


def pick_flights(plan, pilot_plan=None):
    """The bench flights the page plays: each pick on the pilot paths (ten drones at spread 1.0, and one drone) when
    that plan has flown, then on the street route the study first flew (ten drones and one, spread 1.0), goals in
    layout order; and L3 at the tight spread on the street route, with the replica beside it."""
    out = []
    if pilot_plan is not None:
        want = []
        for name in ("L1", "L2", "L3"):
            lv = pilot_plan["pick"][name]
            want.append((name, lv, 10, "goals_pilot_1.0", f"{name} {lv} · ten drones · spread 1.0 · pilot path"))
            want.append((name, lv, 1, "goals_single", f"{name} {lv} · one drone · pilot path"))
        out += _flights(pilot_plan, want, "pilot")
    picks = plan["pick"]
    want = []
    for name in ("L1", "L2", "L3"):
        lv = picks[name]
        want.append((name, lv, 10, "goals_1.0", f"{name} {lv} · ten drones · spread 1.0 · street route"))
    want.append(("L3", picks["L3"], 10, "goals_0.4", f"L3 {picks['L3']} · ten drones · spread 0.4 · street route · "
                                                     "with the replica"))
    for name in ("L1", "L2", "L3"):
        lv = picks[name]
        want.append((name, lv, 1, "goals_1.0", f"{name} {lv} · one drone · street route"))
    return out + _flights(plan, want, "street")


def _flights(plan, want, routing):
    runs_root = Path(plan["bench_root"]) / "runs"
    out = []
    for name, lv, n, kind, title in want:
        run = next((r for r in plan["runs"] if r["level"] == lv and r["drones"] == n and r["seed"] == 101), None)
        if run is None or not (runs_root / run["tag"] / "results.jsonl").exists():
            print(f"[explorer] no run for {title}; skipped")
            continue
        cfg, start, flights, L = tierc.read_run(runs_root / run["tag"])
        fi = next((i for i, fl in enumerate(flights) if fl["kind"] == kind and
                   cfg["scenarios"][int(fl["scenario"])].get("order") == "forward"), None)
        if fi is None:
            continue
        replica = (name, f"n{n}", kind) == REPLICA_FLIGHT
        print(f"[explorer] flight: {title}", flush=True)
        out.append(dict(flight(runs_root / run["tag"], fi, title, replica), pick=name, level=lv, routing=routing))
    return out


# ============================================================================= what each interface sees, and how it flies

CAPABILITY_RUN = "capability"
STATIC_COLS = ["A_any_r40", "A_head_r40", "A_single_r40", "A_pot_r40", "A_single_pot_r40", "V_swarm_1_r40",
               "V_single_1_r40", "E_par_r40", "useful_feeds", "A_any_r80", "A_single_r80"]
MOVING_COLS = ["sweep_any_r40_m", "sweep_head_r40_m", "sweep_any_r80_m", "sweep_head_r80_m", "area_any_r40_per_m",
               "area_head_r40_per_m", "behind_corner_r40_frac", "behind_reach_r40_frac", "lead_vs_omni_r40_s",
               "persist_any_r40_s", "path_ratio", "adj_per_100m", "cmd_straightness", "morph_per_km", "street_frac",
               "drones_lost", "progress_ms", "stuck"]


def _seed_stats(df, keys, cols):
    """Mean over the layout seeds of each key group, with the SD across seeds (columns <c> and <c>_sd)."""
    cols = [c for c in cols if c in df]
    per = df.groupby(keys + ["seed"])[cols].mean()
    m = per.groupby(level=keys).mean()
    s = per.groupby(level=keys).std().add_suffix("_sd")
    return pd.concat([m, s], axis=1).reset_index()


def capability_data():
    """The capability study's tables, by clutter level: standing (d_ref 1.0), flying (every pilot), the cities."""
    cap = STUDY_DIR / "capability" / CAPABILITY_RUN
    if not (cap / "moving.csv").exists():
        return None
    st, mv, ly = (pd.read_csv(cap / f"{n}.csv") for n in ("static", "moving", "layouts"))
    head = pd.read_csv(cap / "headline.csv") if (cap / "headline.csv").exists() else pd.DataFrame()
    stat = _seed_stats(st[np.isclose(st.d_ref, 1.0)], ["clutter"], STATIC_COLS)
    stat_spread = _seed_stats(st, ["clutter", "d_ref"], ["A_any_r40", "A_head_r40", "E_par_r40", "useful_feeds"])
    m = mv[mv.band == "matched"].copy()
    m["d_ref"] = np.where(m.interface == "swarm", m.d_ref, np.nan)
    m["pilot"] = m.interface + "|" + m.routing + "|" + m.d_ref.map(lambda v: "" if not np.isfinite(v) else f"{v:g}")
    mov = _seed_stats(m, ["clutter", "pilot"], MOVING_COLS)
    own = mv[(mv.band == "own")]
    mov_own = _seed_stats(own, ["clutter", "interface", "routing"], ["sweep_any_r40_m", "area_any_r40_per_m"])
    cities = ly.groupby("clutter").agg(alpha=("alpha", "mean"), ell_m=("ell_m", "mean"),
                                        encounters_per_km=("encounters_per_km", "mean"),
                                        wall_share=("wall_share", "mean"), n_buildings=("n_buildings", "mean"),
                                        core=("core", "first"), g_shell=("g_shell", "first")).reset_index()
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    return dict(static=records(stat), static_spread=records(stat_spread), moving=records(mov),
                moving_own=records(mov_own), cities=records(cities), headline=records(head),
                band=man.get("band"), points=man.get("static_points"), pairs=man.get("pairs"),
                w_pass=(man.get("w_pass") or {}).get("w_pass_m"))


def transit_data(levels_pick):
    """The transit at the trial levels (Tier B' on the pilot paths and the street route, every pilot), the goal
    pass-by curves, the discovery predictions, and Tier C on the pilot paths when it has flown."""
    out = {}
    tr_path = STUDY_DIR / "tierBprime" / "tierB2" / "transit.csv"
    if tr_path.exists():
        tr = pd.read_csv(tr_path)
        names = {v: k for k, v in levels_pick.items() if v}
        tr = tr[tr.level.isin(names)].copy()
        disc = [c for c in tr.columns if c.startswith("disc_g") and c.endswith("_any_r80_s")]
        disc200 = [c for c in tr.columns if c.startswith("disc_g") and c.endswith("_any_r200_s")]
        tr["discovery_r80_s"] = tr[disc].mean(axis=1) if disc else np.nan
        tr["discovery_r200_s"] = tr[disc200].mean(axis=1) if disc200 else np.nan
        tr["pick"] = tr.level.map(names)
        tr["d_ref"] = np.where(tr.interface == "swarm", tr.transit_dref, np.nan)
        tr["pilot"] = tr.interface + "|" + tr.routing + "|" + tr.d_ref.map(lambda v: "" if not np.isfinite(v)
                                                                          else f"{v:g}")
        out["levels"] = records(_seed_stats(tr, ["pick", "pilot"], MOVING_COLS + ["discovery_r80_s",
                                                                                  "discovery_r200_s"]))
    pb = STUDY_DIR / "transit" / "passby.csv"
    if pb.exists():
        p = pd.read_csv(pb)
        cols = [c for c in p.columns if c.startswith("det_") and not c.endswith("_dist_m")]
        out["passby"] = records(p.groupby(["label", "interface", "offset_m"])[cols].mean().reset_index())
        W = STUDY_DIR / "transit" / "passby_W.csv"
        if W.exists():
            out["passby_W"] = records(pd.read_csv(W))
    dp = STUDY_DIR / "transit" / "discovery_predictions.csv"
    if dp.exists():
        out["discovery"] = records(pd.read_csv(dp))
    tcp = STUDY_DIR / "tierC" / "tierC_pilot" / "tierC_levels.csv"
    if tcp.exists():
        out["tierC_pilot"] = records(pd.read_csv(tcp))
    gp = STUDY_DIR / "transit" / "gates.csv"
    if gp.exists():
        g = pd.read_csv(gp)
        out["gates"] = records(g.groupby(["kind", "n", "d_ref", "gap_m", "offset_m"]).agg(
            passed=("passed", "mean"), lost=("lost", "mean"), stall=("longest_stall_s", "max"),
            t_cross=("t_cross_s", "mean")).reset_index())
        wp = PACKAGE_DIR / "data" / "w_pass.json"
        if wp.exists():
            out["w_pass"] = json.loads(wp.read_text(encoding="utf-8"))
    return out


# ============================================================================= the tiers' tables

def results():
    gate = pd.read_csv(STUDY_DIR / "gate1" / "levels_tierB2.csv")
    exp = pd.read_csv(STUDY_DIR / "tierB" / "tierB2" / "exposure.csv")
    sw = exp[exp.interface == "swarm"].groupby(["level", "d_ref"])[["E_head", "E_any"]].mean().reset_index()
    sg = exp[exp.interface == "single"].groupby(["level", "policy"])[["E_head", "E_any"]].mean().reset_index()
    geo = pd.read_csv(STUDY_DIR / "tierB" / "tierB2" / "geometry.csv")
    vcols = [c for c in geo.columns if c.startswith("V_swarm_") or c.startswith("V_single_")]
    reveal = geo.groupby(["level", "d_ref"])[vcols].mean().reset_index()
    envB = pd.read_csv(STUDY_DIR / "tierB" / "tierB2" / "envelop.csv").groupby(["level", "d_ref"]).agg(
        enveloped=("enveloped", "mean"), kills=("kills", "mean")).reset_index()
    dyn = pd.read_csv(STUDY_DIR / "tierBprime" / "tierB2" / "dynamic.csv")
    if "routing" in dyn:            # since 2026-10-13: the swarm on its pilot paths (the street route beside it)
        dyn = dyn[dyn.interface == "swarm"]
    else:
        dyn = dyn.assign(routing="street")
    dyn = dyn.groupby(["level", "routing", "transit_dref"]).agg(
        lost=("drones_lost", "mean"), nn=("nn_transit_m", "mean"), progress=("progress_ms", "mean"),
        contacts_km=("contacts_per_km", "mean"), enveloped=("enveloped_frac", "mean"),
        useful=("useful_transit", "mean")).reset_index()
    env = pd.read_csv(STUDY_DIR / "tierBprime" / "envmap.csv")
    truth = lambda c: c.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)   # noqa: E731
    env = env.assign(feasible_geometry=truth(env.feasible_geometry) > 0.5, enveloped=truth(env.enveloped))
    envm = env[env.feasible_geometry].groupby(["n", "d_ref", "w_c", "r_g"]).enveloped.mean().reset_index()
    tc = pd.read_csv(STUDY_DIR / "tierC" / "tierC" / "tierC_levels.csv")
    alive = pd.read_csv(STUDY_DIR / "tierC" / "tierC" / "tierB_at_alive.csv")
    tcf = pd.read_csv(STUDY_DIR / "tierC" / "tierC" / "tierC_flights.csv")
    d = catalogue.study_design()
    alphas = np.round(np.geomspace(0.005, 0.35, 60), 4)
    curves = {f"x{s:g}": [round(theory.ell(a, d.w * s, d.d * s), 2) for a in alphas] for s in (1.0, 2.0, 4.0)}
    scatter = pd.read_csv(STUDY_DIR / "tierA" / "tierA_vs_scatter.csv")
    return dict(
        gate=records(gate, ["level", "clutter", "goal_level", "delta", "sd", "swarm", "single", "ratio_to_L1",
                            "enveloped_1.0", "kills", "feasible", "alpha", "ell"]),
        swarm_by_spread=records(sw), single_by_policy=records(sg), reveal=records(reveal),
        envelop_B=records(envB), dynamic=records(dyn), envmap=records(envm),
        tierC=records(tc), tierC_alive=records(alive),
        tierC_flights=records(tcf, ["level", "seed", "drones", "kind", "order", "lost", "contacts", "sim_s"]),
        tierA=dict(alphas=alphas, curves=curves, design=dict(name=d.design, w=d.w, d=d.d), R=camera.R_ID_M,
                   cities=[dict(name="DiamondCityWorld", alpha=0.0235, ell=244), dict(name="ScaledCityWorld",
                                                                                     alpha=0.0249, ell=222)],
                   scatter=records(scatter)))


def validation():
    rv = STUDY_DIR / "replica_validation"
    models = []
    for key, folder, label in (("kill", rv / "kill", "a footprint entry kills the drone"),
                               ("contact", rv / "contact_nofriction", "contact, no friction"),
                               ("friction", rv, "contact with friction 0.6 (adopted)")):
        f = sorted(folder.glob("validate_verdicts_*.csv"))
        if f:
            v = pd.read_csv(f[-1])
            if "lost_bench" not in v.columns:          # verdicts written before the totals were: sum the flights
                per = pd.read_csv(sorted(folder.glob("validate_2*.csv"))[-1])
                v = v.merge(per.groupby("run")[["lost_bench", "lost_replica"]].sum().reset_index(), on="run", how="left")
            models.append(dict(key=key, label=label, rows=records(v)))
    pooled = rv / "pilot_paths2" / "gate_pooled.csv"
    pilot_gate = records(pd.read_csv(pooled)) if pooled.exists() else []
    tg, tgc = rv / "pilot_paths2" / "transit_gate.csv", rv / "pilot_paths2" / "transit_gate_contrast.csv"
    v3 = STUDY_DIR / "validate" / "internal_3"
    comp = records(pd.read_csv(v3 / "comparison.csv"))
    anchor = json.loads((STUDY_DIR / "anchor" / "anchor_internal_3.json").read_text(encoding="utf-8"))
    summ = json.loads((v3 / "summary.json").read_text(encoding="utf-8"))
    disc = DEFAULT_ROOT_RESULTS / "discovery_range.csv"
    return dict(replica=models, limits=dict(track_frac=0.1, nn_rel=0.15, lost_per10=2.0, t90_s=1.5),
                internal3=dict(comparison=comp, replay_max_diff=summ["replay_max_abs_diff_s"]),
                anchor=anchor["by_condition"], pilot_gate=pilot_gate,
                transit_gate=records(pd.read_csv(tg)) if tg.exists() else [],
                transit_gate_contrast=records(pd.read_csv(tgc)) if tgc.exists() else [],
                discovery_range=records(pd.read_csv(disc, sep=";")) if disc.exists() else [])


def levels_and_params():
    spec = vissim.load_spec(vissim.SPEC_DIR / "tierB2.json")
    lt = pd.read_csv(STUDY_DIR / "levels_table.csv")
    lj = json.loads((STUDY_DIR / "levels.json").read_text(encoding="utf-8"))
    envB = pd.read_csv(STUDY_DIR / "tierB" / "tierB2" / "envelop.csv")
    goalm = envB.groupby("goal_level").agg(c_g=("c_g", "mean"), R_c=("R_c", "mean")).reset_index()
    lays = pd.read_csv(STUDY_DIR / "tierB" / "tierB2" / "layouts.csv")
    clm = lays.groupby("clutter").agg(alpha=("alpha_achieved", "mean"), ell=("ell_mean", "mean"),
                                      n=("n_buildings", "mean"), minstreet=("min_street_width", "mean")).reset_index()
    clutter = {c: dict(spec["clutter"][c], **{k: v for k, v in clm[clm.clutter == c].iloc[0].to_dict().items()
                                               if k != "clutter"}) for c in spec["clutter"]}
    goals = {g: dict(spec["goal"][g], **{k: v for k, v in goalm[goalm.goal_level == g].iloc[0].to_dict().items()
                                          if k != "goal_level"}) for g in spec["goal"]}
    first = sorted(spec["clutter"])[0]
    for g in goals:          # whether the formation can surround this goal's occluder: Tier B at spread 1.0
        row = lt[lt.level == f"{first}-{g}"]
        goals[g]["family"] = "open" if len(row) and float(row["enveloped_B_1.0"].iloc[0]) >= 0.5 else "cramped"
    cols = [c for c in lt.columns if c not in ("clutter", "goal_level")]
    return dict(clutter=clutter, goals=goals, common=spec.get("common", {}), levels=records(lt, ["level", "clutter",
                "goal_level"] + [c for c in cols if c != "level"]), pick=lj["pick"], info=lj["info"],
                levels_json={k: dict(level=v["level"], params=v["params"]) for k, v in lj["levels"].items()})


def runs_summary():
    """What each tier ran, counted from its own outputs."""
    rd = STUDY_DIR
    exp = pd.read_csv(rd / "tierB" / "tierB2" / "exposure.csv", usecols=["layout", "goal"])
    env = pd.read_csv(rd / "tierB" / "tierB2" / "envelop.csv", usecols=["layout"])
    dyn = pd.read_csv(rd / "tierBprime" / "tierB2" / "dynamic.csv", usecols=["layout"])
    envm = pd.read_csv(rd / "tierBprime" / "envmap.csv", usecols=["w_c"])
    tcf = pd.read_csv(rd / "tierC" / "tierC" / "tierC_flights.csv", usecols=["run", "drones"])
    rv = sorted((rd / "replica_validation").glob("validate_2*.csv"))
    rvn = len(pd.read_csv(rv[-1])) if rv else 0
    rp = pd.read_csv(rd / "validate" / "internal_3" / "replay.csv", usecols=["run"])
    lays = pd.read_csv(rd / "tierB" / "tierB2" / "layouts.csv", usecols=["level"])
    tests = unit_tests()
    return [
        dict(tier="A", what="Boolean model: free-sight length against coverage, and the model against random scatter "
                            "cities", count=f"{len(pd.read_csv(rd / 'tierA' / 'tierA_vs_scatter.csv'))} checks",
             where="tierA/"),
        dict(tier="B", what=f"{len(lays)} generated cities ({lays.level.nunique()} levels x 3 seeds), each goal "
                            "settled at 3 spreads and scored against 64 walkers for 60 s over the operator band",
             count=f"{exp.drop_duplicates().shape[0]} goals, {len(env)} formation settles", where="tierB/tierB2/"),
        dict(tier="B'", what="The 2-D replica flies each city spawn to every goal, on the pilot paths and the street "
                             "route (the swarm at transit spread 0.4 and 1.0; the single drone planned, direct and "
                             "looking round), and the envelopment map over occluder x ring x spread x drones",
             count=f"{len(dyn)} flights, {len(envm)} settles", where="tierBprime/"),
        dict(tier="C", what="Headless Unity flies the same goal flights in the three picks' trial cities and their "
                            "neighbours, ten drones (and one drone on the picks), on the street route",
             count=f"{tcf.run.nunique()} runs, {len(tcf)} flights", where="tierC/tierC/"),
        *extra_runs(rd),
        dict(tier="Gate", what="The replica flies 17 recorded Unity flights per city again (DiamondCityWorld, an L1 "
                               "city, a dense C2 city) and is compared with Unity",
             count=f"{rvn} flights", where="replica_validation/"),
        dict(tier="internal_3", what="The recorded human runs replayed through the study's camera model, and "
                                     "predicted with Tier B's machinery", count=f"{len(rp)} goals",
             where="validate/internal_3/"),
        dict(tier="Unity", what="Headless Play-mode check of ProceduralCityWorld: goals, hats, walkers, the altitude "
                                "ceiling, the recorder's new fields, two auto-scored RC presses",
             count="7 checks", where="Assets/Scripts/Experiment/ProceduralCityProbe.cs"),
        dict(tier="Unit", what="Generator, line of sight, display rule, walkers, formations, exposure, contact model, "
                               "Tier C bookkeeping", count=f"{len(tests)} tests", where="citysim/tests/"),
    ]


def extra_runs(rd):
    """The rows for what the 2026-10-13 revision ran (pilot paths, capability, discovery), each where its output is."""
    rows = []
    tcp = rd / "tierC" / "tierC_pilot" / "tierC_flights.csv"
    if tcp.exists():
        f = pd.read_csv(tcp, usecols=["run"])
        rows.append(dict(tier="C, pilot paths", what="Tier C flown again with every transit on the pilots' paths",
                         count=f"{f.run.nunique()} runs, {len(f)} flights", where="tierC/tierC_pilot/"))
    cap = rd / "capability" / "capability"
    if (cap / "moving.csv").exists():
        pts = pd.read_csv(cap / "static.csv", usecols=["layout", "point"]).drop_duplicates().shape[0]
        mv = pd.read_csv(cap / "moving.csv", usecols=["band"])
        rows.append(dict(tier="Capability", what="No goals: standing at random points (three spreads, four body "
                                                 "yaws) and crossing each city (every pilot, both routings), in open, "
                                                 "porous and packed cities, three seeds each",
                         count=f"{pts} points, {int((mv.band == 'matched').sum())} crossings",
                         where="capability/capability/"))
    pb = rd / "transit" / "passby.csv"
    if pb.exists():
        rows.append(dict(tier="Pass-by", what="Straight passes past every trial goal and internal_3's, each pilot on "
                                              "its own path, for the goals' sweep widths and the discovery model",
                         count=f"{len(pd.read_csv(pb, usecols=['goal']))} passes", where="transit/"))
    p2 = rd / "replica_validation" / "pilot_paths2"
    v2 = sorted(p2.glob("validate_2*.csv"))
    tg = p2 / "transit_gate_flights.csv"
    if v2:
        n = len(pd.read_csv(v2[-1], usecols=["run"]))
        rep = int((pd.read_csv(tg, usecols=["source"]).source == "replica").sum()) if tg.exists() else 0
        rows.append(dict(tier="Gate, pilot paths", what="Unity flies the pilot paths in L1, L3 and the porous cities "
                                                        "with every flight recorded; the replica flies them again, "
                                                        "checked on spacing and losses and on the transit measures",
                         count=f"{n} flights" + (f", {rep} re-flights" if rep else ""),
                         where="replica_validation/pilot_paths2/"))
    return rows


def unit_tests():
    out = []
    for f in sorted((PACKAGE_DIR / "tests").glob("test_*.py")):
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r'^def (test_\w+)\([^)]*\):\n(?:\s+"""(.*?)""")?', src, re.M | re.S):
            name = m.group(1)
            doc = " ".join((m.group(2) or "").split())
            out.append(dict(file=f.stem, name=name, title=name[5:].replace("_", " ").capitalize(), doc=doc))
    return out


def operator_summary():
    b = operator_band.load()
    out = {}
    for cond, phases in b["conditions"].items():
        for ph, v in phases.items():
            out[f"{cond}/{ph}"] = {k: v[k] for k in ("altitude", "pitch", "spread", "speed", "head") if v.get(k)}
            if v.get("spread_share"):
                out[f"{cond}/{ph}"]["spread_share"] = v["spread_share"]
    return out


# ============================================================================= assemble

def build(out_path=OUT, bare=False):
    t0 = time.time()
    plan = json.loads((STUDY_DIR / "tierC" / "tierC" / "plan.json").read_text(encoding="utf-8"))
    lp = levels_and_params()
    picks = lp["pick"]
    layouts = []
    for name in ("L1", "L2", "L3"):
        path = trial_path(name, picks[name])
        if path:
            layouts.append(layout_geo(path, f"{name}: {picks[name]}, trial city seed 101", picks[name], 101))
    sweep = pd.read_csv(STUDY_DIR / "tierB" / "tierB2" / "layouts.csv")
    for r in sweep[sweep.seed == 0].sort_values("level").itertuples():
        layouts.append(layout_geo(r.path, f"{r.level}, sweep city seed 0", r.level, 0))
    print(f"[explorer] {len(layouts)} cities ({time.time() - t0:.0f} s)", flush=True)
    scenes = []
    for name in ("L1", "L2", "L3"):
        path = trial_path(name, picks[name])
        if path:
            print(f"[explorer] goal scene {name}", flush=True)
            scenes.append(dict(goal_scene(path, name), level=picks[name]))
    pp = STUDY_DIR / "tierC" / "tierC_pilot" / "plan.json"
    pilot_plan = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else None
    flights = pick_flights(plan, pilot_plan)
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        # inputs changed since that commit (the page itself aside) mean the page is not that commit's build
        local = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "Assets/Scripts/Experiment",
                                "Assets/Scripts/swarm/Bench", "experiment_results/density_study",
                                ":(exclude)experiment_results/density_study/explorer"],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        commit, local = "", ""
    data = dict(meta=dict(generated=time.strftime("%Y-%m-%d %H:%M"), commit=commit, local_changes=bool(local),
                          alt=ALT, pitch=PITCH,
                          R=camera.R_ID_M, hfov=round(camera.HFOV_DEG, 1), vfov=round(camera.VFOV_DEG, 1),
                          cam_fwd=camera.CAMERA_FORWARD_M, hmd_half=camera.HMD_HALF_FOV_DEG,
                          loss_max=2.5, transit_weights=json.loads((STUDY_DIR / "tierC" / "tierC" / "manifest.json")
                                                                   .read_text(encoding="utf-8")).get("transit_weights")),
                **lp, layouts=layouts, scenes=scenes, flights=flights, results=results(), validation=validation(),
                operator=operator_summary(), runs=runs_summary(), tests=unit_tests(),
                capability=capability_data(), transit=transit_data(picks))
    text = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "__DATA__" in html, "the template has no __DATA__ placeholder"
    page = html.replace("__DATA__", text)
    if not bare:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
                + page + "\n</body>\n</html>\n")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8", newline="\n")
    print(f"[explorer] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB data+page, {time.time() - t0:.0f} s)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--bare", action="store_true", help="the page without its html/head/body skeleton (what an "
                                                         "Artifact publish wraps itself)")
    a = ap.parse_args()
    build(a.out, a.bare)


if __name__ == "__main__":
    main()
