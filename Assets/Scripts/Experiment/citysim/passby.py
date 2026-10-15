"""Goal detection while passing, and the discovery predictions built on it.

    python -B -m citysim.vissim passby                 # -> experiment_results/density_study/transit/passby*.csv
    python -B -m citysim.vissim discovery-predict      # -> .../transit/discovery_predictions.csv

PASS-BY. For every goal of the trial layouts (TRIAL_SEEDS of L1-L3) and of internal_3's runs (DiamondCityWorld, each
run's own goal patches), each pilot flies straight-line passes past it (dynamic.fly_od on its pilot path, the swarm at
spread 1.0): PASS_HEADINGS directions, at lateral offsets PASS_OFFSETS from the goal centre, from PASS_HALF_M before the
abeam point to PASS_HALF_M beyond it. A pass DETECTS the goal when walkers are first expected on a shown feed
(discovery.track_expected, the trials' definition) at each discovery range. P(x) -- the share of passes at offset x
that detected it -- is the goal's lateral range curve, and W_goal = 2 * integral P(x) dx its sweep width: the width of
city a pilot flying past clears of that kind of goal.

PREDICTION. With the goals unannounced, a pilot is searching. Random search (Koopman) finds the next of k goals left in
a search area A at the rate k W v / A, so a leg's discovery time is A / (k W v) on average -- the trial's three legs
average 0.611 A / (W v). The swarm / single ratio needs neither A nor the pilots' search skill: (W v)_single /
(W v)_swarm, with v the pilot path's flown speed (Tier B', or Unity's from a scored Tier C plan with --speeds-from:
the replica flies packed cities too fast). The absolute scale is internal_3's: its pilots' measured
discovery times (analyse.py's discovery family) against the same model in their own city give a search efficiency eta,
which the trial predictions take over. Pilots who know where goals can be beat random search; eta says by how much.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import DEFAULT_ROOT, LAYOUT_DIR, STUDY_DIR, discovery, dynamic, layout as lay, routing  # noqa: E402
from citysim.los import Footprints  # noqa: E402

PASS_OFFSETS = (0.0, 40.0, 80.0, 120.0, 160.0, 220.0, 300.0)
PASS_HEADINGS = tuple(range(0, 360, 60))
PASS_HALF_M = 300.0
TRIAL_SEEDS = (101, 102, 103)
LEG_WEIGHT = float(np.mean([1 / 3, 1 / 2, 1.0]))      # three goals: k = 3, 2, 1 left
PILOTS = (("swarm", 1.0), ("single", 1.0))
RANGES = tuple(sorted((discovery.R_GOAL_M,) + discovery.R_GOAL_SENSITIVITY))


def _free(fp, p, margin=6.0):
    from citysim import viewshed
    return viewshed.free_point(fp, p, margin=margin, max_r=80.0)


def pass_jobs(label, boxes_key, goals, height, pitch):
    """[(label, boxes_key, goal index, centre, ring rects, heading, offset, interface, d_ref, height, pitch)]"""
    jobs = []
    for gi, (centre, rects) in enumerate(goals):
        for h in PASS_HEADINGS:
            for x in PASS_OFFSETS:
                for iface, d in PILOTS:
                    jobs.append((label, boxes_key, gi, centre, rects, float(h), float(x), iface, d, height, pitch))
    return jobs


_BOXES = {}


def _boxes(key):
    """Every pass's city, by key: a layout path, or ("run", test, stem) for one of a test's recorded runs."""
    if key not in _BOXES:
        if isinstance(key, tuple):
            import citysim
            citysim.use_bench_modules()
            import analyse
            _, test, stem, city, width = key
            goals = json.loads((DEFAULT_ROOT / test / f"{stem}_session.json").read_text(encoding="utf-8-sig"))["goals"]
            _BOXES[key] = analyse.scene_boxes(goals, city, width)
        else:
            _BOXES[key] = lay.load(key).boxes_as_tuples()
    return _BOXES[key]


def _pass_job(job):
    label, key, gi, centre, rects, heading, offset, iface, d_ref, height, pitch = job
    boxes = _boxes(key)
    centre = np.asarray(centre, float)
    near = Footprints(boxes)
    sub = near.near(centre, PASS_HALF_M + offset + 150.0)
    b2 = [boxes[i] for i in sub]
    fp = Footprints(b2)
    cyl = routing.Cylinders(b2)
    th = math.radians(heading)
    u = np.array([math.sin(th), math.cos(th)])
    n = np.array([math.cos(th), -math.sin(th)])
    a = _free(fp, centre + offset * n - PASS_HALF_M * u)
    b = _free(fp, centre + offset * n + PASS_HALF_M * u)
    summ, rec = dynamic.fly_od(b2, fp, cyl, a, b, interface=iface, routing_="pilot", d_ref=d_ref,
                               seed=int(1000 * heading + offset) % 9973, feeds=False)
    pts, w = discovery.loop_points(rects)
    rows = rec.transit_rows()
    d_min = float(np.min(np.linalg.norm(rec.centroid[rows] - centre, axis=1))) if len(rows) else math.nan
    out = dict(label=label, goal=gi, heading=heading, offset_m=offset, interface=iface, d_ref=d_ref,
               closest_m=round(d_min, 1), stuck=summ["stuck"], lost=summ["drones_lost"])
    for r in RANGES:
        e_any, e_head = discovery.track_expected(fp, rec.cam_pos[rows], rec.cam_yaw[rows], rec.cam_valid[rows],
                                                 rec.body_yaw[rows], height, pitch, pts, w, r_goal=r,
                                                 single=rec.single)
        for basis, e in (("any", e_any), ("head", e_head)):
            hit = np.flatnonzero(e >= discovery.SEEN_WALKERS - 1e-9)
            out[f"det_{basis}_r{int(r)}"] = bool(len(hit))
            out[f"det_{basis}_r{int(r)}_dist_m"] = round(float(np.linalg.norm(rec.centroid[rows][hit[0]] - centre)),
                                                         1) if len(hit) else math.nan
    return out


def sweep_widths(df):
    """W_goal per (label, interface, range, basis): 2 * trapezoid of P(x) over PASS_OFFSETS."""
    rows = []
    for (label, iface), g in df.groupby(["label", "interface"]):
        row = dict(label=label, interface=iface, passes=len(g))
        for r in RANGES:
            for basis in ("any", "head"):
                col = f"det_{basis}_r{int(r)}"
                p = g.groupby("offset_m")[col].mean().reindex(PASS_OFFSETS)
                xs = np.array(PASS_OFFSETS)
                row[f"W_{basis}_r{int(r)}_m"] = round(float(2 * np.trapz(p.fillna(0).values, xs)), 1)
                row[f"P0_{basis}_r{int(r)}"] = round(float(p.iloc[0]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def trial_goals():
    """[(label, layout path, [(centre, ring rects)])] for the trial layouts."""
    out = []
    for level in ("L1", "L2", "L3"):
        for seed in TRIAL_SEEDS:
            hits = sorted(LAYOUT_DIR.glob(f"{level}_*_seed{seed}.json"))
            if not hits:
                continue
            L = lay.load(hits[0])
            goals = []
            for g in L.goals():
                rects = [lay.box_rect(b) for b in L.boxes if b.get("goal", -1) == g["index"] and
                         b.get("role") == "goal_ring"]
                goals.append((np.array(g["centre"], float), rects))
            out.append((level, str(hits[0]), goals))
    return out


def internal_goals(test="internal_3"):
    """[(label, ("run", test, stem, city, width), [(centre, ring rects)])] for a test's recorded runs."""
    import citysim
    citysim.use_bench_modules()
    import analyse
    folder, manifest, runs = analyse.load_test(str(DEFAULT_ROOT), test)
    out = []
    for run in runs:
        if run.practice:
            continue
        goals = [(np.array([g["goalX"], g["goalZ"]], float), analyse.goal_ring_rects(run, g)) for g in run.goals]
        out.append((f"{test}:{run.condition}", ("run", test, run.stem, run.city, run.goal_width), goals))
    return out


def cmd_passby(a):
    from citysim import capability
    from citysim.parallel import run_jobs
    band = capability.band_settings()
    h, p = band["moving_h"], band["moving_p"]
    jobs = []
    for label, key, goals in trial_goals():
        jobs += pass_jobs(label, key, goals, h, p)
    if not a.no_internal:
        for label, key, goals in internal_goals(a.test):
            jobs += pass_jobs(label.split(":")[0], key, goals, h, p)
    t0 = time.time()

    def tick(done, total, _):
        if done == total or done % max(1, total // 20) == 0:
            print(f"[passby] {done}/{total} at {time.time() - t0:.0f} s", flush=True)

    rows = run_jobs(_pass_job, jobs, "passby", a.jobs, progress=tick)
    df = pd.DataFrame(rows)
    out = STUDY_DIR / "transit"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "passby.csv", index=False)
    W = sweep_widths(df)
    W.to_csv(out / "passby_W.csv", index=False)
    pd.set_option("display.width", 220)
    print(W.round(2).to_string(index=False))
    print(f"[passby] {len(df)} passes in {time.time() - t0:.0f} s -> {out}")


def cmd_predict(a):
    """The discovery predictions: per trial level and interface, the random-search time scaled by internal_3's
    efficiency, and the swarm / single ratio, at every range."""
    import citysim
    citysim.use_bench_modules()
    out = STUDY_DIR / "transit"
    W = pd.read_csv(out / "passby_W.csv")
    lv = json.loads((STUDY_DIR / "levels.json").read_text(encoding="utf-8"))
    names = {v["level"]: k for k, v in lv["levels"].items()}
    if a.speeds_from:       # Unity's flown speeds (a scored Tier C plan): the replica flies packed cities too fast
        tc = pd.read_csv(STUDY_DIR / "tierC" / a.speeds_from / "tierC_flights.csv")
        tc = tc.assign(interface=np.where(tc.drones > 1, "swarm", "single"))
        tc = tc[(tc.interface == "single") | (tc.spread_key.astype(str) == "1.0")]
        tc = tc.assign(trial=tc.level.map(names)).dropna(subset=["trial", "flown_m", "transit_s"])
        speed = (tc.flown_m / tc.transit_s).groupby([tc.trial, tc.interface]).mean()
    else:
        tr = pd.read_csv(STUDY_DIR / "tierBprime" / a.sweep / "transit.csv")
        tr = tr[(tr.routing == "pilot") & (tr.interface.isin(["swarm", "single"]))]
        tr = tr[(tr.interface == "single") | np.isclose(tr.transit_dref, 1.0)]
        tr = tr.assign(trial=tr.level.map(names)).dropna(subset=["trial"])
        speed = (tr.flown_m / tr.transit_s).groupby([tr.trial, tr.interface]).mean()
    # internal_3: measured discovery and flown speed per interface
    legs = pd.read_csv(DEFAULT_ROOT / a.test / "results" / "legs.csv", sep=";")
    runs = pd.read_csv(DEFAULT_ROOT / a.test / "results" / "runs.csv", sep=";")
    legs = legs[(legs.phase == "transit") & ~legs.practice.astype(bool)]
    runs = runs[~runs.practice.astype(bool)]
    cond = {"Swarm": "swarm", "SingleDrone": "single"}
    import analyse
    boxes, _ = analyse.load_city("DiamondCityWorld")
    c = np.array([b[1] for b in boxes])[:, [0, 2]]
    A_int = float((c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min()))
    arena = lv["levels"]["L1"]["params"].get("arena") or [-365.0, 365.0, -365.0, 365.0]
    A_trial = float((arena[1] - arena[0]) * (arena[3] - arena[2]))
    rows = []
    for r in RANGES:
        key = "discoverySec" if r == discovery.R_GOAL_M else f"discovery{int(r)}Sec"
        for basis in ("any", "head"):
            wcol = f"W_{basis}_r{int(r)}_m"
            eta = {}
            for c_name, iface in cond.items():
                w = W[(W.label == a.test) & (W.interface == iface)][wcol]
                lg = legs[legs.condition == c_name]
                rn = runs[runs.condition == c_name]
                v = float((rn.act_path_ratio * rn.straight_m).sum() / rn.transit_s.sum()) if len(rn) else math.nan
                if not len(w) or key not in lg or not np.isfinite(v):
                    continue
                pred = LEG_WEIGHT * A_int / (float(w.iloc[0]) * v)
                eta[iface] = float(lg[key].mean()) / pred
                rows.append(dict(range_m=r, basis=basis, level=a.test, interface=iface, W_m=float(w.iloc[0]),
                                 v_ms=round(v, 2), random_search_s=round(pred, 1), measured_s=round(float(lg[key].mean()), 1),
                                 eta=round(eta[iface], 3)))
            eta_pooled = float(np.mean(list(eta.values()))) if eta else math.nan
            for trial in ("L1", "L2", "L3"):
                pred_i = {}
                for iface in ("swarm", "single"):
                    w = W[(W.label == trial) & (W.interface == iface)][wcol]
                    v = speed.get((trial, iface), math.nan)
                    if not len(w) or not np.isfinite(v):
                        continue
                    rs = LEG_WEIGHT * A_trial / (float(w.iloc[0]) * v)
                    pred_i[iface] = rs
                    rows.append(dict(range_m=r, basis=basis, level=trial, interface=iface, W_m=float(w.iloc[0]),
                                     v_ms=round(float(v), 2), random_search_s=round(rs, 1),
                                     predicted_s=round(rs * eta_pooled, 1) if np.isfinite(eta_pooled) else math.nan,
                                     eta=round(eta_pooled, 3) if np.isfinite(eta_pooled) else math.nan))
                if len(pred_i) == 2:
                    rows.append(dict(range_m=r, basis=basis, level=trial, interface="swarm/single",
                                     ratio=round(pred_i["swarm"] / pred_i["single"], 3)))
    df = pd.DataFrame(rows)
    df["trial_speeds"] = f"tierC/{a.speeds_from} (Unity)" if a.speeds_from else f"tierBprime/{a.sweep} (replica)"
    df.to_csv(out / "discovery_predictions.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print(f"[passby] written {out / 'discovery_predictions.csv'}")
