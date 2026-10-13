"""vissim's checks against data and its gate: validate (replay + predict internal_3), anchor (internal_3's empirical
occlusion escape), gate1 (the Tier B gate and candidate levels), tierA (the analytic table and its scatter check)
and envmap (Tier B' envelopment map). Run through `python -B -m citysim.vissim <command>`.
"""
import sys

sys.dont_write_bytecode = True

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import (DEFAULT_ROOT, STUDY_DIR, camera, catalogue, citygen, display, exposure,  # noqa: E402
                     operator_band, replay, single, theory, use_bench_modules, viewshed, walkers)
from citysim.los import Boxes, Footprints, free_sight_probability  # noqa: E402

use_bench_modules()
import analyse  # noqa: E402


# ============================================================================= recorded runs

def load_runs(test, root=DEFAULT_ROOT):
    folder, manifest, runs = analyse.load_test(str(root), test)
    return folder, manifest, [r for r in runs if not r.practice]


def replay_test(test, root=DEFAULT_ROOT, zone=15.0):
    """Every goal of every run on every basis (citysim.replay.summary), plus the zone entry time."""
    _, _, runs = load_runs(test, root)
    rows = []
    for r in runs:
        legs = analyse.goal_legs(r, zone)
        boxes = analyse.scene_boxes(r.goals, r.city, r.goal_width)
        for row in replay.summary(r, boxes, legs):
            L = next((L for L in legs if r.goals[L["gi"]].get("goalIndex", L["gi"]) == row["goalIndex"]), None)
            row["enterSec"] = L["enter"] if L else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def cmd_validate(a):
    out = STUDY_DIR / "validate" / a.test
    out.mkdir(parents=True, exist_ok=True)
    rp = replay_test(a.test, a.root)
    rp.to_csv(out / "replay.csv", index=False)
    # (i) the all-camera basis must reproduce analyse.py's goals.csv
    goals = pd.read_csv(Path(a.root) / a.test / "results" / "goals.csv", sep=";")
    m = rp.merge(goals[["run", "goalIndex", "inViewSec", "occludedSec", "firstSightSec"]], on=["run", "goalIndex"])
    dv = (m.inView_all_s - m.inViewSec).abs()
    print(f"(i) replay vs analyse.py goals.csv inViewSec over {len(m)} goals: max |diff| {dv.max():.2f} s "
          f"({'REPRODUCED' if dv.max() <= 0.11 else 'MISMATCH'})")
    keys = [c for c in rp.columns if c.startswith("inViewSearch_")]
    rp["shown_ratio"] = rp["inViewSearch_shown_s"] / rp["search_s"]
    rp["head_ratio"] = rp["inViewSearch_head_s"] / rp["search_s"]
    rp["all_ratio"] = rp["inViewSearch_all_s"] / rp["search_s"]
    print("\nmeasured per goal, search phase (median [min, max]):")
    for cond, g in rp.groupby("condition"):
        print(f"  {cond:11s} " + "  ".join(
            f"{k} {g[k].median():.2f} [{g[k].min():.2f}, {g[k].max():.2f}]"
            for k in ("all_ratio", "shown_ratio", "head_ratio", "occluded_shown_s")))
    # (ii) predict internal_3 with Tier B's machinery on its own city and goals
    pred = predict_test(a.test, a.root, a.jobs)
    pred.to_csv(out / "predicted.csv", index=False)
    cmp = compare_prediction(rp, pred)
    cmp.to_csv(out / "comparison.csv", index=False)
    print("\n(ii) predicted vs measured (E on the shown / head basis vs the search-phase in-view share):")
    print(cmp.round(3).to_string(index=False))
    (out / "summary.json").write_text(json.dumps(dict(
        test=a.test, replay_max_abs_diff_s=float(dv.max()), reproduced=bool(dv.max() <= 0.11),
        within_band=cmp.to_dict("records")), indent=1, default=float) + "\n", encoding="utf-8")
    print(f"\nwritten {out}")


def goal_geometry(run, gi):
    """A recorded goal's occluder, ring and clearance, from analyse.goal_boxes at the run's goal width."""
    g = run.goals[gi]
    bx = analyse.goal_boxes(g["goalX"], g["goalZ"], run.goal_width)
    rects = [(b[1][0], b[1][2], b[2][0, 0], b[2][2, 2]) for b in bx]
    ring, occ = rects[:8], rects[8]
    centre = np.array(occ[:2])
    others = Footprints.from_rects(ring)
    c_g = float(others.distance([centre])[0])
    return dict(index=gi, centre=centre, occluder=occ, ring_rects=ring, R_c=float(math.hypot(occ[2], occ[3])),
                c_g=c_g, r_g=float(np.mean([math.hypot(r[0] - centre[0], r[1] - centre[1]) for r in ring])),
                R_ring=float(np.mean([math.hypot(r[2], r[3]) for r in ring])), w_c=2 * max(occ[2], occ[3]),
                n_b=8, clear_radius=math.inf)


def _predict_job(job):
    stem, gi, test, root, opts = job
    from citysim import formation
    _, _, runs = load_runs(test, root)
    run = next(r for r in runs if r.stem == stem)
    info = goal_geometry(run, gi)
    boxes = analyse.scene_boxes(run.goals, run.city, run.goal_width)
    fp = Footprints(boxes)
    fpn = fp.subset(fp.near(info["centre"], 130.0))
    near = [boxes[i] for i in Boxes(boxes).near(info["centre"], 250.0)]
    legs = analyse.goal_legs(run, 15.0)
    L = next((L for L in legs if L["gi"] == gi), None)
    if L is not None:      # arrive from where the run arrived
        x, z = analyse.pos_at(run.centroid, L["enter"])
        approach = float(np.degrees(np.arctan2(x - info["centre"][0], z - info["centre"][1])))
    else:
        approach = 270.0
    info["approach_deg"] = approach
    band = operator_band.load(opts["band_test"])
    hy, hw = operator_band.head_weights(band)
    t = np.arange(0.0, 60.0, 0.5)
    rows = []
    rng = np.random.default_rng(7 * gi + 1)
    W = walkers.GoalWalkers(info["ring_rects"], rng)
    pts = W.positions(t)
    base = dict(run=stem, goalIndex=run.goals[gi].get("goalIndex", gi), condition=run.condition, R_c=info["R_c"],
                c_g=info["c_g"])
    for name, tracks in single.policies(fpn, info, W, t, 12.0, -15.0, approach, band=False).items():
        pre = [exposure.recognised_2d(fpn, tr, pts) for tr in tracks]
        for alt, pit in itertools.product(opts["altitude"], opts["pitch"]):
            ms = [exposure.summarise(exposure.with_band(v2, hd, alt, pit), off, tr) for tr, (v2, off, hd) in
                  zip(tracks, pre)]
            rows.append(dict(base, interface="single", policy=name, d_ref=math.nan, altitude=alt, pitch=pit,
                             **{k: float(np.mean([m[k] for m in ms])) for k in ("E_any", "E_head", "first_sight_s")}))
    for d_ref in opts["spread"]:
        env = formation.envelop(near, info["centre"], d_ref, 10, 0, approach, start="arrive", inner_radius=info["c_g"])
        acc = {}
        for by in np.arange(0.0, 360.0, 45.0):
            feeds = display.swarm_feeds(env["P"], by)
            tr = exposure.static_track(feeds, len(t), 0.5, 10.0, -15.0, by)
            v2, off, hd = exposure.recognised_2d(fpn, tr, pts)
            for alt, pit in itertools.product(opts["altitude"], opts["pitch"]):
                acc.setdefault((alt, pit), []).append(
                    exposure.summarise(exposure.with_band(v2, hd, alt, pit), off, tr, hy, hw))
        for (alt, pit), ms in acc.items():
            rows.append(dict(base, interface="swarm", policy="hover", d_ref=d_ref, altitude=alt, pitch=pit,
                             enveloped=env["enveloped"],
                             **{k: float(np.mean([m[k] for m in ms])) for k in ("E_any", "E_head", "first_sight_s")}))
    return rows


def predict_test(test, root, jobs):
    _, _, runs = load_runs(test, root)
    band = operator_band.load(test)
    grid = operator_band.grid(band)
    opts = dict(band_test=test, **grid)
    jl = [(r.stem, gi, test, str(root), opts) for r in runs for gi in range(len(r.goals))]
    with Pool(min(jobs, len(jl))) as pool:
        res = pool.map(_predict_job, jl)
    return pd.DataFrame([x for r in res for x in r])


def compare_prediction(rp, pred):
    """Per condition: the measured search-phase shares against the predicted band (the operator grid's spread of
    per-goal means; the swarm at every spread, the single drone's realistic policies)."""
    rows = []
    for cond, iface in (("Swarm", "swarm"), ("SingleDrone", "single")):
        p = pred[pred.interface == iface]
        if iface == "single":
            p = p[p.policy.isin(["orbit_in", "orbit_out", "scan"])]
        meas = rp[rp.condition == cond]
        for basis, col in (("shown", "shown_ratio"), ("head", "head_ratio")):
            key = "E_any" if basis == "shown" else "E_head"
            per_setting = p.groupby(["altitude", "pitch", "d_ref", "policy"], dropna=False)[key].mean()
            lo, hi, mid = per_setting.min(), per_setting.max(), per_setting.median()
            m = meas[col].dropna()
            rows.append(dict(condition=cond, basis=basis, predicted_lo=lo, predicted_median=mid, predicted_hi=hi,
                             measured_median=float(m.median()) if len(m) else math.nan,
                             measured_mean=float(m.mean()) if len(m) else math.nan,
                             measured_within=float(((m >= lo) & (m <= hi)).mean()) if len(m) else math.nan))
        fs_pred = p.groupby(["altitude", "pitch", "d_ref", "policy"], dropna=False)["first_sight_s"].mean()
        fs_meas = (meas["firstSight_shown_s"] - meas["enterSec"]).dropna()
        rows.append(dict(condition=cond, basis="first_sight_s", predicted_lo=fs_pred.min(),
                         predicted_median=fs_pred.median(), predicted_hi=fs_pred.max(),
                         measured_median=float(fs_meas.median()) if len(fs_meas) else math.nan,
                         measured_mean=float(fs_meas.mean()) if len(fs_meas) else math.nan,
                         measured_within=float(((fs_meas >= fs_pred.min()) & (fs_meas <= fs_pred.max())).mean())
                         if len(fs_meas) else math.nan))
    return pd.DataFrame(rows)


# ============================================================================= M0.4: the empirical anchor

def cmd_anchor(a):
    """Among samples where some camera had the hat in range and heading but blocked, how often another camera
    (any / shown / head-visible) saw it -- per interface, with the candidate counts."""
    _, _, runs = load_runs(a.test, a.root)
    rows = []
    for r in runs:
        boxes = analyse.scene_boxes(r.goals, r.city, r.goal_width)
        for b in replay.hat_bases(r, boxes):
            cand = b["blocked_all"]
            dt = b["dt"]
            rows.append(dict(run=r.stem, condition=r.condition, goalIndex=b["goalIndex"],
                             candidate_s=float(dt[cand].sum()), candidates=int(cand.sum()),
                             escape_any_s=float(dt[cand & b["all"]].sum()),
                             escape_shown_s=float(dt[cand & b["shown"]].sum()),
                             escape_head_s=float(dt[cand & b["head"]].sum()),
                             inview_all_s=float(dt[b["all"]].sum()), inview_shown_s=float(dt[b["shown"]].sum())))
    df = pd.DataFrame(rows)
    out = STUDY_DIR / "anchor"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"anchor_{a.test}.csv", index=False)
    g = df.groupby("condition")[["candidate_s", "escape_any_s", "escape_shown_s", "escape_head_s", "candidates",
                                 "inview_all_s", "inview_shown_s"]].sum()
    for k in ("any", "shown", "head"):
        g[f"E_{k}"] = g[f"escape_{k}_s"] / g["candidate_s"]
    g["occluded_share"] = g["candidate_s"] / (g["candidate_s"] + g["inview_all_s"] - g["escape_any_s"])
    print(g.round(3).to_string())
    summary = {c: {k: float(v) for k, v in row.items()} for c, row in g.iterrows()}
    (out / f"anchor_{a.test}.json").write_text(json.dumps(dict(test=a.test, by_condition=summary), indent=1) + "\n",
                                               encoding="utf-8")
    print(f"written {out}")


# ============================================================================= G1

GATE_RATIO, GATE_SD, VANISH_SD = 2.0, 3.0, 1.0
REALISTIC_SINGLE = ("orbit_in", "orbit_out", "scan")


def level_deltas(exp, swarm_spread="pilot"):
    """Per (level, seed): swarm E_head - the best realistic single policy's, averaged over goals and the operator
    grid. swarm_spread "pilot" = the best spread of the grid per goal and setting (the pilot picks it), or a
    number."""
    keys = ["level", "clutter", "goal_level", "seed", "goal", "altitude", "pitch"]
    sw = exp[exp.interface == "swarm"]
    if swarm_spread == "pilot":
        s = sw.groupby(keys).E_head.max()
    else:
        s = sw[np.isclose(sw.d_ref, float(swarm_spread))].groupby(keys).E_head.mean()
    sg = exp[(exp.interface == "single") & exp.policy.isin(REALISTIC_SINGLE)].groupby(keys + ["policy"]).E_head.mean()
    best = sg.groupby(level=keys).max()
    d = (s - best).rename("delta").reset_index()
    d["swarm"], d["single"] = s.values, best.reindex(s.index).values
    per_seed = d.groupby(["level", "clutter", "goal_level", "seed"])[["delta", "swarm", "single"]].mean().reset_index()
    return per_seed


def cmd_gate1(a):
    src = STUDY_DIR / "tierB" / a.sweep
    exp = pd.read_csv(src / "exposure.csv")
    env = pd.read_csv(src / "envelop.csv")
    lays = pd.read_csv(src / "layouts.csv")
    per_seed = level_deltas(exp, a.spread)
    lev = per_seed.groupby(["level", "clutter", "goal_level"]).agg(
        delta=("delta", "mean"), sd=("delta", "std"), swarm=("swarm", "mean"), single=("single", "mean"),
        seeds=("seed", "nunique")).reset_index()
    envl = env.groupby("level").agg(enveloped_mean=("enveloped", "mean"), kills=("kills", "sum")).reset_index()
    envu = env[np.isclose(env.d_ref, 1.0)].groupby("level").enveloped.mean().rename("enveloped_1.0").reset_index()
    feas = lays.groupby("level").agg(jammed=("jammed", "any"), max50=("max_within_50m", "max"),
                                     alpha=("alpha_achieved", "mean"), ell=("ell_mean", "mean")).reset_index()
    lev = lev.merge(envl, on="level", how="left").merge(envu, on="level", how="left").merge(feas, on="level", how="left")
    lev["feasible"] = ~lev.jammed & (lev.max50 <= 56)
    L1 = lev[lev.level == a.l1]
    if not len(L1):
        raise SystemExit(f"[gate1] no level {a.l1} in {src}")
    d1 = float(L1.delta.iloc[0])
    lev["ratio_to_L1"] = lev.delta / d1 if d1 > 0 else np.inf
    lev["clears_sd"] = lev.delta > GATE_SD * lev.sd
    cand2 = lev[(lev.level != a.l1) & (lev.ratio_to_L1 >= GATE_RATIO) & lev.clears_sd & lev.feasible]
    best2 = cand2.sort_values("delta", ascending=False).head(1)
    cand3 = lev[(lev.level != a.l1) & lev.feasible & (lev.delta <= VANISH_SD * lev.sd.clip(lower=1e-6))]
    res = dict(sweep=a.sweep, spread=a.spread, L1=a.l1, delta_L1=d1, rule=dict(
        L2=f"delta >= {GATE_RATIO} x delta(L1), > {GATE_SD} x its across-seed SD, layout feasible",
        L3=f"feasible, delta <= {VANISH_SD} x its SD (vanished or reversed)"))
    res["L2_candidates"] = cand2.level.tolist()
    res["L3_candidates"] = cand3.level.tolist()
    res["pass"] = bool(len(best2) and len(cand3))
    res["L2"] = best2.level.iloc[0] if len(best2) else None
    print(lev.sort_values("delta", ascending=False).round(3).to_string(index=False))
    print(f"\nL1 = {a.l1} (delta {d1:.3f}); L2 candidates {res['L2_candidates']}; L3 candidates {res['L3_candidates']}")
    print(f"G1: {'PASS' if res['pass'] else 'FAIL'}" + (f" (L2 = {res['L2']})" if res["L2"] else ""))
    out = STUDY_DIR / "gate1"
    out.mkdir(parents=True, exist_ok=True)
    lev.to_csv(out / f"levels_{a.sweep}.csv", index=False)
    per_seed.to_csv(out / f"deltas_{a.sweep}.csv", index=False)
    (out / f"gate1_{a.sweep}.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"written {out}")


# ============================================================================= Tier A

def cmd_tierA(a):
    d = catalogue.study_design()
    rows = theory.table(d.w, d.d)
    df = pd.DataFrame(rows)
    out = STUDY_DIR / "tierA"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tierA.csv", index=False)
    print(df.to_string(index=False))
    # Tier B on scatter layouts: measured free-sight probability against the Boolean model
    chk = []
    for alpha in a.alphas:
        for seed in range(a.seeds):
            P = replace(citygen.Params(), mode="scatter", alpha=alpha, size=2.0, w_c=7.0, r_g=26.0, ring_size=0.9)
            L = citygen.generate(P, seed)
            fp = Footprints(L.boxes_as_tuples())
            rng = np.random.default_rng(seed)
            o = np.c_[rng.uniform(-300, 300, 600), rng.uniform(-300, 300, 600)]
            o = o[fp.distance(o) > 1.0][:250]
            ell = theory.ell(L.layout["alpha_achieved"], d.w * 2, d.d * 2)
            for r in (20.0, 40.0, 80.0, 160.0):
                chk.append(dict(alpha_target=alpha, alpha=L.layout["alpha_achieved"], seed=seed, r=r,
                                measured=free_sight_probability(fp, o, r), boolean=theory.p_free(r, ell)))
    c = pd.DataFrame(chk).groupby(["alpha_target", "r"])[["alpha", "measured", "boolean"]].mean().reset_index()
    c["diff"] = c.measured - c.boolean
    c.to_csv(out / "tierA_vs_scatter.csv", index=False)
    print("\nTier B (scatter) vs Tier A (Boolean): P(free r)")
    print(c.round(3).to_string(index=False))
    print(f"written {out}")


# ============================================================================= Tier B': envelopment map

def cmd_envmap(a):
    from citysim import formation
    t0 = time.time()
    frames = []
    for ring in a.ring_size:
        df = envmap_grid(a.wc, a.rg, a.nb, a.dref, a.n, a.seeds, ring, a.jobs)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = STUDY_DIR / "tierBprime"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "envmap.csv", index=False)
    ok = df[df.feasible_geometry]
    print(ok.groupby(["ring_size", "w_c", "r_g", "n", "d_ref"]).enveloped.mean().unstack("d_ref").round(2).to_string())
    print(f"[envmap] {len(df)} settles in {time.time() - t0:.0f} s -> {out / 'envmap.csv'}")


def _envmap_job2(job):
    from citysim import formation
    w_c, r_g, n_b, d_ref, n, seed, ring = job
    P = citygen.Params(w_c=w_c, r_g=r_g, n_b=n_b, size=2.0, ring_size=ring)
    rng = np.random.default_rng(seed)
    try:
        bld = citygen.goal_buildings(P, catalogue.study_design(), (0.0, 0.0), rng)
    except ValueError as e:
        return dict(w_c=w_c, r_g=r_g, n_b=n_b, d_ref=d_ref, n=n, seed=seed, ring_size=ring, feasible_geometry=False,
                    error=str(e))
    boxes = formation.goal_city((0.0, 0.0), bld)
    rects = np.array([b[0] for b in bld])
    c_g = float(Footprints.from_rects(rects[1:]).distance([[0.0, 0.0]])[0])
    r = formation.envelop(boxes, (0.0, 0.0), d_ref, n, seed, start="arrive", inner_radius=c_g)
    return dict(w_c=w_c, r_g=r_g, n_b=n_b, d_ref=d_ref, n=n, seed=seed, ring_size=ring, feasible_geometry=True,
                R_c=round(float(np.hypot(rects[0, 2], rects[0, 3])), 3), c_g=round(c_g, 3),
                enveloped=r["enveloped"], surrounds=r["surrounds"], max_gap_deg=round(r["max_gap_deg"], 1),
                inside_frac=round(r["inside_frac"], 3), mean_radius_m=round(r["mean_radius_m"], 2),
                kills=r["kills"], contacts=r["contacts"], n_alive=r["n_alive"], diameter_m=round(r["diameter_m"], 2),
                min_building_m=round(r["min_building_m"], 2), centroid_offset_m=round(r["centroid_offset_m"], 2))


def envmap_grid(w_cs, r_gs, n_bs, drefs, ns, seeds, ring, jobs):
    grid = list(itertools.product(w_cs, r_gs, n_bs, drefs, ns, range(seeds), [ring]))
    with Pool(min(jobs, len(grid))) as pool:
        rows = pool.map(_envmap_job2, grid)
    return pd.DataFrame(rows)


# ============================================================================= M3: the Unity build round trip

def roundtrip(export_path, layout_path, tol=1e-3):
    """Compare CityObstacleExport's boxes of a built procedural city with the layout it was built from: every
    layout box present by name, its centre within tol, its half-axes equal up to sign; and nothing else on the
    Obstacle layer. Returns (problems, stats)."""
    from citysim import layout as lay
    exp = json.loads(Path(export_path).read_text(encoding="utf-8"))
    L = lay.load(layout_path)
    got = {b["name"]: b for b in exp["boxes"]}
    problems, dc, dax = [], [], []
    for b in L.boxes:
        e = got.pop(b["name"], None)
        if e is None:
            problems.append(f"{b['name']}: not in the export")
            continue
        dc.append(float(np.abs(np.subtract(e["c"], b["c"])).max()))
        ea, la = np.array(e["ax"], float), np.array(b["ax"], float)
        dax.append(float(max(min(np.abs(ea[k] - la[k]).max(), np.abs(ea[k] + la[k]).max()) for k in range(3))))
        if dc[-1] > tol:
            problems.append(f"{b['name']}: centre off by {dc[-1]:.4f} m")
        if dax[-1] > tol:
            problems.append(f"{b['name']}: half-axes differ by {dax[-1]:.4f} m")
    for n in got:
        problems.append(f"{n}: on the Obstacle layer in the scene but not in the layout")
    return problems, dict(boxes=len(L.boxes), exported=len(exp["boxes"]), max_dc=max(dc) if dc else math.nan,
                          max_dax=max(dax) if dax else math.nan)


def cmd_roundtrip(a):
    problems, st = roundtrip(a.export, a.layout)
    print(f"round trip {Path(a.layout).name}: {st['boxes']} layout boxes, {st['exported']} exported; max centre "
          f"error {st['max_dc']:.2e} m, max half-axis error {st['max_dax']:.2e} m")
    for p in problems[:20]:
        print("  PROBLEM:", p)
    print("PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return 1 if problems else 0
