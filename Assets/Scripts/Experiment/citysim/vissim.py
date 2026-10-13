"""The visibility-scaling study's tiers, from one command line.

    python -B -m citysim.vissim tierA                                  # Tier A table + Tier B scatter check
    python -B -m citysim.vissim sweep --spec citysim/specs/tierB.json  # Tier B: layouts x goals x operator grid
    python -B -m citysim.vissim envmap                                 # Tier B': the envelopment map
    python -B -m citysim.vissim dynamic --sweep tierB                  # Tier B': replica transits through layouts
    python -B -m citysim.vissim gate1 --sweep tierB                    # G1 and the candidate levels
    python -B -m citysim.vissim validate                               # replay + predict internal_3
    python -B -m citysim.vissim anchor                                 # internal_3's empirical occlusion escape (M0)
    python -B -m citysim.vissim tierC-plan                             # Tier C: bench configs for the levels
    python -B -m citysim.vissim tierC                                  # Tier C: score the flown runs
    python -B -m citysim.vissim tierC-alive                            # Tier B at Tier C's alive count

Outputs go to experiment_results/density_study/<tier>/<name>/ (tidy CSVs + the spec and a manifest); generated
sweep layouts to Assets/Scripts/Experiment/layouts/sweep/<name>/ (git-ignored). docs/density_study/README.md says
what each tier assumes and what it cannot tell.

Tier B, per goal of every layout:
  swarm   the replica flies the formation onto the goal from the spawn side for each spread of the operator grid
          (formation.envelop) and the settled positions are its template; the display rule (display.swarm_feeds)
          gives the feeds at 8 body yaws; every walker is scored over a WINDOW_S window (exposure) at each altitude
          x pitch of the grid; E_head weights the recorded head-yaw distribution. Viewshed metrics (E, V(t), U)
          at 4 body yaws.
  single  the policies of citysim.single at the same altitudes and pitches (three orbit / scan phases each).
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import (DEFAULT_ROOT, LAYOUT_DIR, PACKAGE_DIR, STUDY_DIR, camera, citygen, display, exposure,  # noqa: E402
                     layout as lay, operator_band, single, viewshed, walkers)
from citysim.los import Boxes, Footprints  # noqa: E402

WINDOW_S, DT = 60.0, 0.5
BODY_YAWS = np.arange(0.0, 360.0, 45.0)
GEO_BODY_YAWS = np.arange(0.0, 360.0, 90.0)
NEAR_FP_M = 130.0              # footprints a goal's cameras can see past (R + formation + margin)
NEAR_REPLICA_M = 250.0         # buildings the replica flies the arrival among
SPEC_DIR = PACKAGE_DIR / "specs"
STALL_S = 900.0                # no result from a pool for this long: a worker has died and taken its job with it


def jobs_default():
    return max(1, (os.cpu_count() or 2) - 2)


def imap_guarded(pool, fn, jobs, label):
    """pool.imap_unordered with a stall guard. A pool worker that dies loses its job without a trace, and a plain
    imap then waits for it forever with every other worker idle (it has happened, rarely, with no error anywhere).
    Here STALL_S with no result ends the run, saying how many jobs were outstanding, so it can be rerun -- a
    sweep reuses its layouts -- instead of hanging silently."""
    import multiprocessing
    it = pool.imap_unordered(fn, jobs)
    for k in range(len(jobs)):
        try:
            yield it.next(timeout=STALL_S)
        except multiprocessing.TimeoutError:
            raise SystemExit(f"[vissim] {label}: no result for {STALL_S:.0f} s with {len(jobs) - k} of {len(jobs)} "
                             "job(s) outstanding; a pool worker has probably died. Rerun it.") from None


def stamp():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ============================================================================= goals of a layout

def goal_info(L, g):
    """The geometry the tiers need about one goal of a layout."""
    own = [b for b in L.boxes if b.get("goal", -1) == g["index"]]
    occ = [lay.box_rect(b) for b in own if b["role"] == "goal_centre"][0]
    ring = [lay.box_rect(b) for b in own if b["role"] == "goal_ring"]
    sp = L.layout["spawn"]
    c = g["centre"]
    return dict(index=g["index"], centre=np.array(c, float), occluder=occ, ring_rects=ring, R_c=g["R_c"],
                c_g=g["c_g"], r_g=g["r_g"], n_b=g["n_b"], w_c=g["w_c"], clear_radius=g["clear_radius"],
                R_ring=float(np.mean([math.hypot(r[2], r[3]) for r in ring])), ring_size=g.get("ring_size"),
                approach_deg=float(np.degrees(np.arctan2(sp[0] - c[0], sp[2] - c[1]))))


# ============================================================================= Tier B: one layout

def score_layout(job):
    """Every Tier B row of one layout: (exposure rows, geometry rows, envelopment rows, layout row)."""
    path, meta, opts = job
    from citysim import formation
    t0 = time.time()
    L = lay.load(path)
    boxes = L.boxes_as_tuples()
    fp = Footprints(boxes)
    B3 = Boxes(boxes)
    band = operator_band.load(opts["band_test"])
    head_yaws, head_w = operator_band.head_weights(band)
    alts, pitches, spreads = opts["altitude"], opts["pitch"], opts["spread"]
    t = np.arange(0.0, WINDOW_S, DT)
    seed = int(L.layout["seed"])
    exp_rows, geo_rows, env_rows = [], [], []
    base = dict(layout=L.id, **meta)
    failures = []
    for g in L.goals():
        try:
            score_goal(L, g, fp, B3, boxes, band, head_yaws, head_w, alts, pitches, spreads, t, seed, base,
                       exp_rows, geo_rows, env_rows, singles=opts.get("singles", True))
        except Exception as e:   # noqa: BLE001 -- one goal's failure is recorded, not fatal to the sweep
            import traceback
            failures.append(f"goal {g['index']}: {type(e).__name__}: {e}")
            print(f"[vissim] {L.id} goal {g['index']} failed:\n{traceback.format_exc()}", flush=True)
    lrow = dict(base, path=str(path), seconds=round(time.time() - t0, 1), failures="; ".join(failures),
                **{k: L.layout.get(k) for k in ("alpha_achieved", "alpha_target", "jammed", "ell_mean",
                                                "ell_anisotropy", "max_within_50m", "min_street_width",
                                                "facade_gap_min", "n_buildings", "p", "s", "g", "size")},
                **{f"narrow_{k}": v for k, v in (L.layout.get("narrow_street_frac_by_spread") or {}).items()})
    return exp_rows, geo_rows, env_rows, lrow


def score_goal(L, g, fp, B3, boxes, band, head_yaws, head_w, alts, pitches, spreads, t, seed, base,
               exp_rows, geo_rows, env_rows, singles=True):
    """Tier B rows of one goal, appended to the three lists. The formation has base["n_drones"] drones (10); the
    single-drone policies, which do not depend on it, are scored only when `singles`."""
    from citysim import formation
    info = goal_info(L, g)
    gkey = dict(base, goal=info["index"], w_c=info["w_c"], R_c=info["R_c"], r_g=info["r_g"], c_g=info["c_g"],
                n_b=info["n_b"], ring_size=info["ring_size"], spawn_dist=g.get("spawn_dist"))
    rng = np.random.default_rng(10_000 * seed + 17 * info["index"] + 3)
    W = walkers.GoalWalkers(info["ring_rects"], rng)
    pts = W.positions(t)
    fpn = fp.subset(fp.near(info["centre"], NEAR_FP_M))
    occ = Footprints.from_rects([info["occluder"]])
    near = [boxes[i] for i in B3.near(info["centre"], NEAR_REPLICA_M)]

    # ---- single drone
    pols = single.policies(fpn, info, W, t, alts[len(alts) // 2], camera.DIAL_CENTRE_PITCH_DEG,
                           info["approach_deg"], band=False) if singles else {}
    for name, tracks in pols.items():
        pre = [exposure.recognised_2d(fpn, tr, pts) for tr in tracks]
        for alt, pit in itertools.product(alts, pitches):
            ms = []
            for tr, (v2, off, hd) in zip(tracks, pre):
                vis = exposure.with_band(v2, hd, alt, pit)
                ms.append(exposure.summarise(vis, off, tr))
            exp_rows.append(dict(gkey, interface="single", policy=name, d_ref=math.nan, altitude=alt, pitch=pit,
                                 **mean_dicts(ms)))
    # single drone's viewshed at the goal (from the annulus on the approach side, facing the goal)
    # is part of the swarm geometry rows below (V_single_*), so it shares the swarm's reference point.

    # ---- swarm
    for d_ref in spreads:
        env = formation.envelop(near, info["centre"], d_ref, n=int(base.get("n_drones", 10)), seed=seed,
                                approach_deg=info["approach_deg"],
                                start="arrive", inner_radius=info["c_g"])
        env_rows.append(dict(gkey, d_ref=d_ref, enveloped=env["enveloped"], surrounds=env["surrounds"],
                             max_gap_deg=env["max_gap_deg"], kills=env["kills"], contacts=env["contacts"],
                             n_alive=env["n_alive"], inside_frac=env["inside_frac"],
                             mean_radius_m=env["mean_radius_m"],
                             diameter_m=env["diameter_m"], min_building_m=env["min_building_m"],
                             centroid_offset_m=env["centroid_offset_m"]))
        P = env["P"]
        acc = {}
        geo = []
        for by in BODY_YAWS:
            feeds = display.swarm_feeds(P, by, pitch_deg=camera.DIAL_CENTRE_PITCH_DEG)
            tr = exposure.static_track(feeds, len(t), DT, alts[0], pitches[0], by)
            v2, off, hd = exposure.recognised_2d(fpn, tr, pts)
            for alt, pit in itertools.product(alts, pitches):
                vis = exposure.with_band(v2, hd, alt, pit)
                acc.setdefault((alt, pit), []).append(exposure.summarise(vis, off, tr, head_yaws, head_w))
            if by in GEO_BODY_YAWS and len(P):
                o = viewshed.free_point(fpn, P.mean(0))
                gp = viewshed.ground_points(fpn, o)
                m = viewshed.occlusion_escape(fpn, feeds, o, gp, occ, by, head_yaws, head_w)
                m.update(viewshed.reveal(fpn, feeds, o, gp, by))
                u = display.feed_informativeness(fpn, feeds)
                m.update(useful_feeds=u["useful"], feeds=len(feeds),
                         wall_frac=float(np.mean(u["wall_frac"])) if len(feeds) else math.nan)
                geo.append(m)
        for (alt, pit), ms in acc.items():
            exp_rows.append(dict(gkey, interface="swarm", policy="hover", d_ref=d_ref, altitude=alt, pitch=pit,
                                 enveloped=env["enveloped"], **mean_dicts(ms)))
        if geo:
            geo_rows.append(dict(gkey, d_ref=d_ref, enveloped=env["enveloped"], **mean_dicts(geo)))


def mean_dicts(ms):
    """Field-wise mean of metric dicts (NaN-aware), dropping arrays."""
    keys = [k for k, v in ms[0].items() if not isinstance(v, np.ndarray)]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in ms], float)
        out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
    return out


# ============================================================================= Tier B: the sweep

def load_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    spec.setdefault("name", Path(path).stem)
    return spec


def level_params(spec):
    """[(level name, clutter name, goal name, citygen.Params)] the spec asks for."""
    common = spec.get("common", {})
    out = []
    pairs = spec.get("levels")
    if pairs in (None, "product"):
        pairs = [[c, g] for c in spec["clutter"] for g in spec["goal"]]
    for c, g in pairs:
        kw = dict(common, **spec["clutter"][c], **spec["goal"][g])
        P = replace(citygen.Params(), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})
        out.append((f"{c}-{g}", c, g, P))
    return out


def generate_layouts(spec, out_dir, jobs, levels=None):
    """Generate (or reuse) every layout of the spec (or of `levels` among them). [(path, meta)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo, done = [], []
    for level, c, g, P in level_params(spec):
        if levels and level not in levels:
            continue
        for seed in spec["seeds"]:
            path = out_dir / f"{P.ident(seed)}.json"
            meta = dict(level=level, clutter=c, goal_level=g, seed=seed)
            if path.exists():
                done.append((path, meta))
            else:
                todo.append((P, seed, out_dir, meta))
    if todo:
        with Pool(min(jobs, len(todo))) as pool:
            res = pool.map(_gen_job, todo)
        for r in res:
            if r[0] is None:
                print(f"[vissim] layout {r[1]['level']} seed {r[1]['seed']} not generated: {r[2]}")
            else:
                done.append(r[:2])
    return sorted(done, key=lambda x: str(x[0]))


def _gen_job(job):
    P, seed, out_dir, meta = job
    try:
        L = citygen.generate(P, seed)
    except ValueError as e:
        return None, meta, str(e)
    problems = L.verify()
    if problems:
        return None, meta, "; ".join(problems)
    return L.save(Path(out_dir) / f"{L.id}.json"), meta, None


def cmd_sweep(a):
    spec = load_spec(a.spec)
    name = a.name or spec["name"]
    lay_dir = LAYOUT_DIR / "sweep~" / spec["name"]       # a renamed run (--name) reuses the spec's layouts
    out = STUDY_DIR / "tierB" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    layouts = generate_layouts(spec, lay_dir, a.jobs, a.levels)
    print(f"[vissim] {len(layouts)} layouts in {lay_dir} ({time.time() - t0:.0f} s)", flush=True)
    band = operator_band.load(spec.get("band", "internal_3"))
    grid = operator_band.grid(band)
    opts = dict(band_test=spec.get("band", "internal_3"), altitude=spec.get("altitude", grid["altitude"]),
                pitch=spec.get("pitch", grid["pitch"]), spread=spec.get("spread", grid["spread"]))
    if a.limit:
        layouts = layouts[:a.limit]
    drones = a.drones or [10]
    jobs = [(p, dict(m, n_drones=n), dict(opts, singles=(k == 0))) for p, m in layouts
            for k, n in enumerate(drones)]
    t1 = time.time()
    res = []
    with Pool(min(a.jobs, len(jobs))) as pool:
        for k, r in enumerate(imap_guarded(pool, score_layout, jobs, "sweep"), start=1):
            res.append(r)
            print(f"[vissim] {k}/{len(jobs)} {r[3]['layout']} ({r[3]['seconds']} s"
                  f"{', FAILED ' + r[3]['failures'] if r[3]['failures'] else ''}) at {time.time() - t1:.0f} s",
                  flush=True)
    exp = pd.DataFrame([r for x in res for r in x[0]])
    geo = pd.DataFrame([r for x in res for r in x[1]])
    env = pd.DataFrame([r for x in res for r in x[2]])
    lrows = pd.DataFrame([x[3] for x in res])
    for nm, df in (("exposure", exp), ("geometry", geo), ("envelop", env), ("layouts", lrows)):
        df.to_csv(out / f"{nm}.csv", index=False)
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(dict(generated=stamp(), spec=str(a.spec), layouts=len(layouts),
                                                       levels=a.levels, drones=drones, opts=opts, window_s=WINDOW_S,
                                                       dt=DT,
                                                       body_yaws=BODY_YAWS.tolist(), seconds=round(time.time() - t1)),
                                                  indent=1) + "\n", encoding="utf-8")
    print(f"[vissim] scored {len(jobs)} layouts in {time.time() - t1:.0f} s -> {out}")
    print(quick_table(exp, env))


def quick_table(exp, env):
    """Swarm (hover, d_ref 1.0) vs best realistic single policy, E_head, by level, mean over seeds/goals/grid."""
    if not len(exp):
        return "(no rows)"
    sw = exp[(exp.interface == "swarm") & np.isclose(exp.d_ref, 1.0)]
    sg = exp[(exp.interface == "single") & exp.policy.isin(["orbit_in", "orbit_out", "scan"])]
    a = sw.groupby("level").E_head.mean()
    b = sg.groupby(["level", "policy"]).E_head.mean().unstack()
    t = pd.concat([a.rename("swarm_1.0"), b], axis=1)
    t["single_best"] = b.max(axis=1)
    t["delta"] = t["swarm_1.0"] - t["single_best"]
    if len(env):
        t["enveloped_1.0"] = env[np.isclose(env.d_ref, 1.0)].groupby("level").enveloped.mean()
        t["enveloped_1.6"] = env[np.isclose(env.d_ref, 1.6)].groupby("level").enveloped.mean()
    return t.round(3).to_string()


# ============================================================================= Tier B': dynamic flights

def _dyn_job(job):
    from citysim import dynamic
    path, meta, tdref, gdref, seed = job
    try:
        summ, goals, _ = dynamic.fly_layout(path, tdref, gdref, seed=seed)
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[vissim] dynamic {path} failed:\n{traceback.format_exc()}", flush=True)
        return dict(meta, layout=Path(path).stem, transit_dref=tdref, goal_dref=gdref, error=str(e)), []
    return dict(meta, **summ), [dict(meta, layout=summ["layout"], transit_dref=tdref, goal_dref=gdref, **g)
                                for g in goals]


def cmd_dynamic(a):
    spec = load_spec(SPEC_DIR / f"{a.sweep}.json") if (SPEC_DIR / f"{a.sweep}.json").exists() else None
    src = STUDY_DIR / "tierB" / a.sweep / "layouts.csv"
    lays = pd.read_csv(src)
    if a.levels and a.levels != ["all"]:
        lays = lays[lays.level.isin(a.levels)]
    jobs = [(r.path, dict(level=r.level, clutter=r.clutter, goal_level=r.goal_level, seed=int(r.seed)), td,
             a.goal_spread, int(r.seed)) for r in lays.itertuples() for td in a.transit]
    t0 = time.time()
    rows, grows = [], []
    with Pool(min(a.jobs, len(jobs))) as pool:
        for k, (r, g) in enumerate(imap_guarded(pool, _dyn_job, jobs, "dynamic"), start=1):
            rows.append(r)
            grows += g
            print(f"[vissim] dynamic {k}/{len(jobs)} {r.get('layout')} transit {r.get('transit_dref')}: "
                  f"lost {r.get('drones_lost')}, contacts {r.get('contacts')}, nn {r.get('nn_transit_m')} m "
                  f"({time.time() - t0:.0f} s)", flush=True)
    out = STUDY_DIR / "tierBprime" / a.sweep
    out.mkdir(parents=True, exist_ok=True)
    df, gd = pd.DataFrame(rows), pd.DataFrame(grows)
    df.to_csv(out / "dynamic.csv", index=False)
    gd.to_csv(out / "dynamic_goals.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(dict(generated=stamp(), sweep=a.sweep, levels=a.levels,
                                                       transit=a.transit, goal_spread=a.goal_spread,
                                                       flights=len(jobs), seconds=round(time.time() - t0)),
                                                  indent=1) + "\n", encoding="utf-8")
    cols = ["drones_lost", "contacts_per_km", "progress_ms", "nn_transit_m", "min_pair_p05_m", "close_frac",
            "useful_transit", "useful_goal", "enveloped_frac", "legs_stuck"]
    print(df.groupby(["level", "transit_dref"])[cols].mean().round(3).to_string())
    print(f"written {out}")


# ============================================================================= command line

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep", help="Tier B over a spec's levels")
    p.add_argument("--spec", required=True)
    p.add_argument("--name", help="output name (default: the spec's)")
    p.add_argument("--jobs", type=int, default=jobs_default())
    p.add_argument("--limit", type=int, default=0, help="score only the first N layouts (a smoke test)")
    p.add_argument("--levels", nargs="+", help="only these levels of the spec (e.g. the picks)")
    p.add_argument("--drones", type=int, nargs="+", help="formation sizes to fly (default 10); single-drone "
                                                         "policies are scored once")
    p.set_defaults(func=cmd_sweep)

    from citysim import checks
    p = sub.add_parser("validate", help="replay internal_3 on every basis and predict it with Tier B's machinery")
    p.add_argument("--test", default="internal_3")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--jobs", type=int, default=jobs_default())
    p.set_defaults(func=checks.cmd_validate)
    p = sub.add_parser("anchor", help="internal_3's empirical occlusion escape (M0)")
    p.add_argument("--test", default="internal_3")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.set_defaults(func=checks.cmd_anchor)
    p = sub.add_parser("gate1", help="G1 over a Tier B sweep, and the candidate levels")
    p.add_argument("--sweep", default="tierB")
    p.add_argument("--l1", default="C0-G0", help="the level that stands for today's regime")
    p.add_argument("--spread", default="pilot", help="'pilot' (best spread of the grid) or a d_ref")
    p.set_defaults(func=checks.cmd_gate1)
    p = sub.add_parser("tierA", help="the analytic table, and the Boolean model against scatter layouts")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--seeds", type=int, default=3)
    p.set_defaults(func=checks.cmd_tierA)
    p = sub.add_parser("envmap", help="Tier B': the envelopment map over (w_c, r_g, n_b, d_ref, n alive)")
    p.add_argument("--wc", type=float, nargs="+", default=[7.0, 10.0, 13.06, 16.0, 20.0])
    p.add_argument("--rg", type=float, nargs="+", default=[17.0, 20.0, 24.0, 28.0, 32.0, 36.0])
    p.add_argument("--nb", type=int, nargs="+", default=[8])
    p.add_argument("--dref", type=float, nargs="+", default=[0.4, 0.7, 1.0, 1.3, 1.6])
    p.add_argument("--n", type=int, nargs="+", default=[10, 8, 6])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--ring-size", type=float, nargs="+", default=[0.9])
    p.add_argument("--jobs", type=int, default=jobs_default())
    p.set_defaults(func=checks.cmd_envmap)
    p = sub.add_parser("roundtrip", help="compare a built procedural city's exported colliders with its layout")
    p.add_argument("--export", required=True, help="city_obstacles_ProceduralCityWorld.json")
    p.add_argument("--layout", required=True)
    p.set_defaults(func=checks.cmd_roundtrip)
    from citysim import tierc
    p = sub.add_parser("tierC-plan", help="Tier C: goal flights, configs and a runner for the candidate levels")
    p.add_argument("--name", default="tierC")
    p.add_argument("--sweep", default="tierB2", help="the Tier B sweep whose layouts stand for unpicked levels")
    p.add_argument("--levels", nargs="+", help="levels to fly (default: picks, candidates, clutter neighbours)")
    p.add_argument("--bench-root", default=str(tierc.DEFAULT_BENCH_ROOT))
    p.add_argument("--altitude", type=float, default=tierc.ALTITUDE)
    p.add_argument("--time-scale", type=float, default=20.0)
    p.set_defaults(func=tierc.cmd_plan)
    p = sub.add_parser("tierC", help="Tier C: score the flown bench runs of a plan")
    p.add_argument("--name", default="tierC")
    p.add_argument("--jobs", type=int, default=jobs_default())
    p.set_defaults(func=tierc.cmd_score)
    p = sub.add_parser("tierC-alive", help="Tier B's advantage at Tier C's alive-at-goal drone count")
    p.add_argument("--name", default="tierC")
    p.add_argument("--sweep", default="tierB2_alive", help="a sweep flown with --drones over the picks")
    p.set_defaults(func=tierc.cmd_alive)
    p = sub.add_parser("dynamic", help="Tier B': the replica flies each layout of a Tier B sweep, goal to goal")
    p.add_argument("--sweep", default="tierB2")
    p.add_argument("--levels", nargs="+", default=["all"], help="level names (C0-G0 ...) or all")
    p.add_argument("--transit", type=float, nargs="+", default=[0.4, 1.0], help="transit spreads (d_ref)")
    p.add_argument("--goal-spread", type=float, default=1.6, help="the spread flown at each goal")
    p.add_argument("--jobs", type=int, default=jobs_default())
    p.set_defaults(func=cmd_dynamic)
    a = ap.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
