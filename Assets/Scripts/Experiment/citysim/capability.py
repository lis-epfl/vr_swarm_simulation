"""The task-independent capability study: how much more the swarm sees than one drone -- standing anywhere in a city,
and flying through it on the paths its pilot would fly -- from today's sparse city to dense ones.

    python -B -m citysim.vissim capability [--spec citysim/specs/capability.json] [--quick]   # -> capability/<name>/

Cities: the spec's clutter levels -- capability.json: C0, C1, C2 as the study generated them (packed blocks, closed to
the avoidance law at C1/C2), and porous C1p / C2p (citygen core="shell": every two avoidance cylinders at least
w_pass(1.0) apart, so the formation can flow between any two buildings) -- three seeds each, today's goal patch (G0).
Nothing here is about the goals: points and flights keep away from them.

STATIC (static.csv). STATIC_POINTS free points per layout. The replica flies the formation in from the point's most open
side and lets the centring pilot settle it there (SETTLE_S) at each spread; the single drone stands on the settled
centroid. At GEO_BODY_YAWS, on transit.Cells' grid of free ground within the larger range, at the operator band's
search-phase median altitude and pitch (the same for both interfaces), for each range (40 m: the hat; 80 m: a person):
  A_any / A_head    ground recognised on the shown feeds / on the feeds in the HMD's view (recorded head yaws), m2
  A_single          the single drone's one feed, facing the body yaw
  A_pot             what could be seen at all: the hull feeds' union / the single drone's 360-degree isovist
  V_t               within t s: the swarm's feeds brought into view by a head turn and body yaw, the single drone's
                    view turned at 75 deg/s (display.swarm_reveal_deg / single_reveal_deg), m2
  E_par             occlusion escape by rim parallax: of the ground inside the feeds' wedges that is hidden from the
                    formation's centre, the share some feed sees
  U                 useful feeds, wall-filled share (display.feed_informativeness)
MOVING (moving.csv, pairs.csv). MOVING_PAIRS origin-destination pairs per layout, at least MIN_OD_M apart, each flown by
every pilot (dynamic.fly_od): the swarm at three spreads and the three single-drone pilots on the pilot paths, and the
swarm (1.0) and the single drone on the street route; scored by citysim.transit on the pair's shared cells at the
transit-phase band. pairs.csv sets each swarm flight beside each single-drone one (transit.compare).
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import (LAYOUT_DIR, STUDY_DIR, camera, display, dynamic, formation, layout as lay,  # noqa: E402
                     operator_band, routing, transit)
from citysim.los import Footprints, vertical_band, wrap_deg  # noqa: E402

STATIC_POINTS = 12
MOVING_PAIRS = 6
MIN_OD_M = 400.0
POINT_CLEAR_M = 3.0          # a point is free ground at least this far from every footprint
GOAL_KEEPOUT_M = 20.0        # beyond each goal patch's clear radius
SPREADS = (0.4, 1.0, 1.6)
GEO_BODY_YAWS = (0.0, 90.0, 180.0, 270.0)
REVEAL_T = (0.0, 1.0, 3.0, 6.0)
SETTLE_S = 60.0
APPROACH_M = 60.0
CAP_RANGES = transit.RANGES
MOVING_PILOTS = (("swarm", "pilot", 0.4), ("swarm", "pilot", 1.0), ("swarm", "pilot", 1.6),
                 ("single", "pilot", 1.0), ("single_direct", "pilot", 1.0), ("single_scan", "pilot", 1.0),
                 ("swarm", "street", 1.0), ("single", "street", 1.0))
QUICK = dict(points=3, pairs=1, spreads=(1.0,), pilots=(("swarm", "pilot", 1.0), ("single", "pilot", 1.0),
                                                         ("swarm", "street", 1.0)))


# ============================================================================= points

def free_points(L, fp, rng, k, min_sep=0.0, tries=20000):
    """k random free points of the arena: at least POINT_CLEAR_M from every footprint, GOAL_KEEPOUT_M beyond every goal
    patch, and (min_sep) that far apart."""
    x0, x1, z0, z1 = L.layout["arena"]
    goals = [(np.array(g["centre"], float), g["clear_radius"] + GOAL_KEEPOUT_M) for g in L.goals()]
    out = []
    for _ in range(tries):
        if len(out) >= k:
            break
        p = np.array([rng.uniform(x0 + 30, x1 - 30), rng.uniform(z0 + 30, z1 - 30)])
        if any(np.linalg.norm(p - c) < r for c, r in goals):
            continue
        if fp.n and fp.distance([p])[0] < POINT_CLEAR_M:
            continue
        if min_sep and any(np.linalg.norm(p - q) < min_sep for q in out):
            continue
        out.append(p)
    return out


def od_pairs(L, fp, rng, k):
    """k origin-destination pairs at least MIN_OD_M apart, from distinct free points."""
    pts = free_points(L, fp, rng, 6 * k)
    pairs, used = [], set()
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if len(pairs) >= k:
                return pairs
            if i in used or j in used or np.linalg.norm(pts[i] - pts[j]) < MIN_OD_M:
                continue
            pairs.append((pts[i], pts[j]))
            used |= {i, j}
    return pairs


def settle_at(boxes, fp, point, d_ref, seed, n=10):
    """The formation flown in from the point's most open side and held there by the centring pilot: (positions of the
    last MEASURE_S averaged (k, 2), drones lost, contacts)."""
    import swarm_replica as rep
    m = formation.model()
    ang, dist = fp.isovist(point, APPROACH_M + 10.0, step_deg=5.0) if fp.n else (np.array([0.0]), np.array([1e9]))
    b = float(ang[int(np.argmax(dist))])
    r = float(min(APPROACH_M, max(np.max(dist) - 8.0, 0.0)))
    start = point + r * np.array([math.sin(math.radians(b)), math.cos(math.radians(b))])
    P0, _ = formation.hover_template(d_ref, n, seed)
    sim = formation.ContactSim(m, rep.City(boxes), P0 + start, np.zeros((n, 2)), seed)
    tail = []
    while sim.t < SETTLE_S and sim.alive.sum() > 1:
        c = sim.P[sim.alive].mean(0)
        v = formation.PILOT_GAIN * (point - c) / m.max_speed
        s = np.linalg.norm(v)
        sim.step(v if s <= formation.PILOT_MAX else v * formation.PILOT_MAX / s, 0.0, d_ref)
        if sim.t >= SETTLE_S - formation.MEASURE_S and sim.tick % 5 == 0:
            tail.append(np.where(sim.alive[:, None], sim.P, np.nan))
    alive = sim.alive.copy()
    P = np.nanmean(np.array(tail)[:, alive], axis=0) if tail else sim.P[alive]
    return P, int(n - alive.sum()), int(sim.contacts)


# ============================================================================= static metrics

def _feed_cells(fp, cells, feeds, height, pitch, r_max):
    return [transit.seen_from(fp, cells, feeds.pos[k], feeds.yaw[k], height, pitch, r_max) for k in range(len(feeds))]


def static_metrics(fp, P, body_yaw, height, pitch, head_yaws, head_w):
    """The STATIC measures of one settled formation (P) and a single drone on its centroid, for every range."""
    r_max = max(CAP_RANGES)
    o = P.mean(0)
    if fp.n and fp.distance([o])[0] < 0.5:
        from citysim import viewshed
        o = viewshed.free_point(fp, o)
    cells = transit.Cells(fp, (o[0] - r_max - 2, o[0] + r_max + 2, o[1] - r_max - 2, o[1] + r_max + 2))
    feeds = display.swarm_feeds(P, body_yaw)
    per = _feed_cells(fp, cells, feeds, height, pitch, r_max)
    single = transit.seen_from(fp, cells, o + camera.CAMERA_FORWARD_M * np.array(
        [math.sin(math.radians(body_yaw)), math.cos(math.radians(body_yaw))]), body_yaw, height, pitch, r_max)
    iso = transit.seen_from(fp, cells, o, 0.0, height, pitch, r_max, hfov=360.0)
    a = cells.area
    out = dict(feeds=len(feeds))
    hw = np.asarray(head_w, float) / max(float(np.sum(head_w)), 1e-12)
    for r in CAP_RANGES:
        tag = f"_r{int(r)}"
        any_ = np.zeros(cells.n, bool)
        az_cells = []
        for (idx, slant, off), k in zip(per, range(len(feeds))):
            ok = slant <= r
            any_[idx[ok]] = True
            az_cells.append((idx[ok], wrap_deg(feeds.yaw[k] - body_yaw + off[ok] * camera.DISPLAY_SCALE)))
        head = 0.0
        for h, w in zip(head_yaws, hw):
            if w < 1e-4:
                continue
            m = np.zeros(cells.n, bool)
            for idx, az in az_cells:
                m[idx[np.abs(wrap_deg(az - h)) <= camera.HMD_HALF_FOV_DEG]] = True
            head += w * m.sum()
        s_idx, s_slant, _ = single
        sgl = np.zeros(cells.n, bool)
        sgl[s_idx[s_slant <= r]] = True
        i_idx, i_slant, i_off = iso
        ok = i_slant <= r
        iso_m = np.zeros(cells.n, bool)
        iso_m[i_idx[ok]] = True
        out.update({f"A_any{tag}": any_.sum() * a, f"A_head{tag}": head * a, f"A_single{tag}": sgl.sum() * a,
                    f"A_single_pot{tag}": iso_m.sum() * a, f"A_pot{tag}": any_.sum() * a})
        brg = wrap_deg(i_off[ok] - body_yaw)               # i_off is the bearing (the isovist camera's yaw is 0)
        for t in REVEAL_T:
            reach_s = float(display.swarm_reveal_deg(t))
            m = np.zeros(cells.n, bool)
            for idx, az in az_cells:
                m[idx[np.abs(az) <= reach_s]] = True
            out[f"V_swarm_{t:g}{tag}"] = m.sum() * a
            reach_1 = float(display.single_reveal_deg(t))
            out[f"V_single_{t:g}{tag}"] = float((np.abs(brg) <= reach_1).sum()) * a
        # occlusion escape by parallax: inside the union of the feeds' wedges at o, in range and band, hidden from o
        d = cells.pts - o
        hd = np.hypot(d[:, 0], d[:, 1])
        brg_all = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
        hmax = math.sqrt(max(r ** 2 - (height - camera.HAT_HEIGHT_M) ** 2, 0.0))
        wedge = np.zeros(cells.n, bool)
        for k in range(len(feeds)):
            wedge |= np.abs(wrap_deg(brg_all - feeds.yaw[k])) <= camera.HFOV_DEG / 2
        cand = wedge & (hd <= hmax) & vertical_band(height, pitch, hd) & ~iso_m
        out[f"E_par{tag}"] = float((cand & any_).sum() / cand.sum()) if cand.any() else math.nan
        out[f"occluded_share{tag}"] = float(cand.sum() / max((wedge & (hd <= hmax) &
                                                              vertical_band(height, pitch, hd)).sum(), 1))
    if len(feeds):
        u = display.feed_informativeness(fp.subset(fp.near(o, 90.0)), feeds)
        out.update(useful_feeds=u["useful"], wall_frac=float(np.mean(u["wall_frac"])))
    return out


def _static_job(job):
    path, meta, k, point, d_ref, seed, band = job
    L = lay.load(path)
    boxes = L.boxes_as_tuples()
    fp = Footprints(boxes)
    near = fp.near(point, 250.0)
    sub_boxes = [boxes[i] for i in near]
    P, lost, contacts = settle_at(sub_boxes, fp.subset(near), np.asarray(point, float), d_ref, seed)
    rows = []
    if len(P) < 3:
        return [dict(meta, point=k, d_ref=d_ref, lost=lost, contacts=contacts, failed=True)]
    for by in GEO_BODY_YAWS:
        m = static_metrics(fp.subset(fp.near(P.mean(0), 200.0)), P, by, band["static_h"], band["static_p"],
                           band["head_yaws"], band["head_w"])
        rows.append(dict(meta, point=k, x=round(float(point[0]), 1), z=round(float(point[1]), 1), d_ref=d_ref,
                         body_yaw=by, lost=lost, contacts=contacts, n_alive=len(P),
                         offset_m=round(float(np.linalg.norm(P.mean(0) - point)), 2), **m))
    return rows


# ============================================================================= moving

def _moving_job(job):
    path, meta, k, a, b, pilots, band = job
    L = lay.load(path)
    boxes = L.boxes_as_tuples()
    fp = Footprints(boxes)
    cyl = routing.Cylinders(boxes)
    graph = routing.street_graph(L)
    arena = tuple(L.layout["arena"])
    seed = int(meta["seed"]) * 1000 + k
    flights = []
    for iface, rt, d_ref in pilots:
        summ, rec = dynamic.fly_od(boxes, fp, cyl, a, b, interface=iface, routing_=rt, d_ref=d_ref, seed=seed,
                                   graph=graph, city_bounds=arena)
        flights.append((iface, rt, d_ref, summ, rec))
    tracks = [r.centroid[r.transit_rows()] for *_, r in flights] + [np.array([a, b])]
    cells = transit.Cells.around(fp, tracks, max(CAP_RANGES) + 4.0)
    rows, scored = [], []
    for iface, rt, d_ref, summ, rec in flights:
        for which, (h, p) in (("matched", (band["moving_h"], band["moving_p"])),
                              ("own", (band["own_h"], band["own_p"]))):
            if which == "own" and iface == "swarm":
                continue
            s = transit.score(rec, fp, cells, h, p, layout=L.id)
            firsts = s.pop("firsts", {})
            row = dict(meta, pair=k, ax=round(float(a[0]), 1), az=round(float(a[1]), 1), bx=round(float(b[0]), 1),
                       bz=round(float(b[1]), 1), od_m=round(float(np.linalg.norm(b - a)), 1), interface=iface,
                       routing=rt, d_ref=d_ref, band=which, stuck=summ["stuck"], drones_lost=summ["drones_lost"],
                       nn_transit_m=summ["nn_transit_m"], min_pair_m=summ["min_pair_m"],
                       useful_transit=summ["useful_transit"], **{kk: v for kk, v in s.items()
                                                                  if kk not in ("layout", "routing", "interface",
                                                                                "d_ref", "seed")})
            rows.append(row)
            if which == "matched":
                scored.append((iface, rt, d_ref, dict(firsts=firsts)))
    pairs = []
    for iface, rt, d_ref, s in scored:
        if iface != "swarm":
            continue
        for iface2, rt2, d2, s2 in scored:
            if iface2 == "swarm":
                continue
            pairs.append(dict(meta, pair=k, swarm_routing=rt, swarm_d_ref=d_ref, single=iface2, single_routing=rt2,
                              **transit.compare(s, s2, "sw")))
    return rows, pairs


# ============================================================================= layouts and the command

def porosity(L, fp, rng, w_pass_m):
    """How open a layout is to each pilot: wall members (routing.walls at w_pass), the median gap from each
    building's cylinder to its nearest neighbour's, and avoidance-cylinder encounters per km of random straight line."""
    cyl = routing.Cylinders(L.boxes_as_tuples())
    member, _ = routing.walls(cyl, w_pass_m)
    from scipy.spatial import cKDTree
    gaps = []
    if cyl.n > 1:
        d, j = cKDTree(cyl.c).query(cyl.c, k=min(8, cyl.n))
        g = d[:, 1:] - cyl.R[:, None] - cyl.R[j[:, 1:]]
        gaps = g.min(1)
    x0, x1, z0, z1 = L.layout["arena"]
    hits, total = 0, 0.0
    for _ in range(200):
        p = np.array([rng.uniform(x0, x1), rng.uniform(z0, z1)])
        u = rng.uniform(0, 2 * np.pi)
        d_ = np.array([np.sin(u), np.cos(u)])
        rel = cyl.c - p
        t = rel @ d_
        perp = np.abs(rel @ np.array([d_[1], -d_[0]]))
        hits += int(((perp < cyl.R + routing.STANDOFF_M) & (t > 0) & (t < 500.0)).sum())
        total += 500.0
    return dict(wall_share=round(float(member.mean()), 4) if cyl.n else 0.0,
                nn_cyl_gap_median_m=round(float(np.median(gaps)), 2) if len(gaps) else math.nan,
                encounters_per_km=round(hits / total * 1000.0, 2))


def band_settings(test="internal_3"):
    band = operator_band.load(test)
    c = band["conditions"]
    hy, hw = operator_band.head_weights(band, "Swarm", "search")
    return dict(static_h=c["Swarm"]["search"]["altitude"]["p50"], static_p=c["Swarm"]["search"]["pitch"]["p50"],
                moving_h=c["Swarm"]["transit"]["altitude"]["p50"], moving_p=c["Swarm"]["transit"]["pitch"]["p50"],
                own_h=c["SingleDrone"]["transit"]["altitude"]["p50"], own_p=c["SingleDrone"]["transit"]["pitch"]["p50"],
                head_yaws=[float(v) for v in hy], head_w=[float(v) for v in hw])


def cmd_capability(a):
    from citysim import vissim
    from citysim.parallel import run_jobs
    spec = vissim.load_spec(a.spec)
    name = a.name or spec["name"] + ("_quick" if a.quick else "")
    out = STUDY_DIR / "capability" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    layouts = vissim.generate_layouts(spec, LAYOUT_DIR / "sweep~" / spec["name"], a.jobs, a.levels)
    if a.quick:
        layouts = [lp for lp in layouts if lp[1]["seed"] == spec["seeds"][0]]
    band = band_settings(spec.get("band", "internal_3"))
    n_pts = QUICK["points"] if a.quick else STATIC_POINTS
    n_pairs = QUICK["pairs"] if a.quick else MOVING_PAIRS
    spreads = QUICK["spreads"] if a.quick else SPREADS
    pilots = QUICK["pilots"] if a.quick else MOVING_PILOTS
    static_jobs, moving_jobs, lrows = [], [], []
    for path, meta in layouts:
        L = lay.load(path)
        fp = Footprints(L.boxes_as_tuples())
        rng = np.random.default_rng(91_000 + int(meta["seed"]) * 97 + sum(map(ord, meta["clutter"])))
        lrows.append(dict(meta, layout=L.id, alpha=L.layout["alpha_achieved"], ell_m=L.layout["ell_mean"],
                          n_buildings=L.layout["n_buildings"], jammed=L.layout["jammed"],
                          core=L.layout.get("core", "facade"), g_shell=L.layout.get("g_shell", 0.0),
                          **porosity(L, fp, rng, routing.w_pass(1.0))))
        meta = dict(meta, layout=L.id)
        for k, p in enumerate(free_points(L, fp, rng, n_pts, min_sep=60.0)):
            for d in spreads:
                static_jobs.append((str(path), meta, k, p, d, int(meta["seed"]) * 100 + k, band))
        for k, (pa, pb) in enumerate(od_pairs(L, fp, rng, n_pairs)):
            moving_jobs.append((str(path), meta, k, pa, pb, pilots, band))
    print(f"[capability] {len(layouts)} layouts: {len(static_jobs)} static settles, {len(moving_jobs)} OD pairs x "
          f"{len(pilots)} pilots ({time.time() - t0:.0f} s)", flush=True)

    def tick(label):
        def f(done, total, _):
            if done == total or done % max(1, total // 10) == 0:
                print(f"[capability] {label} {done}/{total} at {time.time() - t0:.0f} s", flush=True)
        return f

    srows = [r for rs in run_jobs(_static_job, static_jobs, "static", a.jobs, progress=tick("static")) for r in rs]
    mres = run_jobs(_moving_job, moving_jobs, "moving", a.jobs, progress=tick("moving"))
    mrows = [r for rs, _ in mres for r in rs]
    prows = [p for _, ps in mres for p in ps]
    st, mv, pr, ly = pd.DataFrame(srows), pd.DataFrame(mrows), pd.DataFrame(prows), pd.DataFrame(lrows)
    for nm, df in (("static", st), ("moving", mv), ("pairs", pr), ("layouts", ly)):
        df.to_csv(out / f"{nm}.csv", index=False)
    head = headline(st, mv, ly)
    head.to_csv(out / "headline.csv", index=False)
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"), spec=str(a.spec), quick=bool(a.quick), layouts=len(layouts),
        static_points=n_pts, pairs=n_pairs, spreads=list(spreads), pilots=[list(p) for p in pilots], band=band,
        w_pass=json.loads(routing.W_PASS_FILE.read_text(encoding="utf-8")) if routing.W_PASS_FILE.exists() else None,
        seconds=round(time.time() - t0)), indent=1) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(head.round(3).to_string(index=False))
    print(f"[capability] written {out} ({time.time() - t0:.0f} s)")


def headline(st, mv, ly):
    """Swarm / single per clutter level, spread 1.0: standing (static) and flying (moving, the pilot paths, the
    single drone forward-facing on its own path), at the hat's range unless named."""
    rows = []
    for clutter in sorted(set(ly.clutter), key=lambda c: (c.rstrip("p"), c)):
        r = dict(clutter=clutter, alpha=ly[ly.clutter == clutter].alpha.mean(),
                 ell_m=ly[ly.clutter == clutter].ell_m.mean(),
                 encounters_per_km=ly[ly.clutter == clutter].encounters_per_km.mean(),
                 wall_share=ly[ly.clutter == clutter].wall_share.mean())
        s = st[(st.clutter == clutter) & np.isclose(st.d_ref, 1.0)] if len(st) else st
        if len(s):
            with np.errstate(all="ignore"):
                r.update(static_any_x=s.A_any_r40.mean() / s.A_single_r40.mean(),
                         static_head_x=s.A_head_r40.mean() / s.A_single_r40.mean(),
                         static_1s_x=s.V_swarm_1_r40.mean() / s["V_single_1_r40"].mean(),
                         static_pot_x=s.A_pot_r40.mean() / s.A_single_pot_r40.mean(),
                         static_E_par=s.E_par_r40.mean(), useful_feeds=s.useful_feeds.mean())
        m = mv[(mv.clutter == clutter) & (mv.band == "matched")] if len(mv) else mv
        if len(m):
            sw = m[(m.interface == "swarm") & (m.routing == "pilot") & np.isclose(m.d_ref, 1.0)]
            sg = m[(m.interface == "single") & (m.routing == "pilot")]
            if len(sw) and len(sg):
                with np.errstate(all="ignore"):
                    r.update(sweep_any_x=sw.sweep_any_r40_m.mean() / sg.sweep_any_r40_m.mean(),
                             sweep_head_x=sw.sweep_head_r40_m.mean() / sg.sweep_head_r40_m.mean(),
                             sweep80_any_x=sw.sweep_any_r80_m.mean() / sg.sweep_any_r80_m.mean(),
                             area_per_m_x=sw.area_any_r40_per_m.mean() / sg.area_any_r40_per_m.mean(),
                             behind_corner_frac=sw.behind_corner_r40_frac.mean(),
                             lead_vs_omni_s=sw.lead_vs_omni_r40_s.median(),
                             swarm_path_ratio=sw.path_ratio.mean(), single_path_ratio=sg.path_ratio.mean(),
                             swarm_adj_100m=sw.adj_per_100m.mean(), single_adj_100m=sg.adj_per_100m.mean(),
                             swarm_morph_per_km=sw.morph_per_km.mean(),
                             swarm_street_frac=sw.street_frac.mean(), single_street_frac=sg.street_frac.mean(),
                             swarm_stuck=sw.stuck.mean(), swarm_lost=sw.drones_lost.mean())
        rows.append(r)
    return pd.DataFrame(rows)
