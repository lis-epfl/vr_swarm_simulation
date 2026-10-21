"""The clutter study's pieces: goal-free cities, the ring of feeds' overlap, a short walk of every condition, and the
statistics (an effect found, a null not, the q values)."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import catalogue, citygen, clutter, clutter_stats, layout as lay, search, transit  # noqa: E402
from citysim.los import Footprints  # noqa: E402

ARENA = (-200.0, 200.0, -200.0, 200.0)


def _boxes(centres, size=2.0, yaw=0):
    d = catalogue.study_design()
    out = []
    for k, (x, z) in enumerate(centres):
        b = lay.make_box(f"t{k}", "t", x, z, d, size, yaw)
        out.append((b["tile"], np.array(b["c"], float), np.array(b["ax"], float)))
    return out


def test_goal_free_city_is_deterministic_and_has_no_goals():
    P = replace(citygen.Params(), p=110.0, alpha=0.04, goals=0)
    a, b = citygen.generate(P, 3), citygen.generate(P, 3)
    assert a.goals() == [] and not any(bx.get("goal", -1) >= 0 for bx in a.boxes)
    assert a.boxes == b.boxes and "nogoals" in a.layout["id"]
    assert abs(a.layout["alpha_achieved"] - 0.04) < 0.002 and not a.verify()
    # a goal city's id is what it always was
    assert "nogoals" not in replace(P, goals=3).ident(3)


def test_ring_overlap_sees_the_wall_the_frustums_do_not():
    """Two feeds splayed 40 deg apart share the ground ahead of them; a wall between them takes it away from the
    clear-sight count (los) but not from the frustums' (geo)."""
    pos = np.array([[-3.0, 0.0], [3.0, 0.0]])
    yaw = np.array([-20.0, 20.0])
    h, pitch = 12.4, -15.0
    for wall, conn in ((False, 1.0), (True, 0.0)):
        boxes = []
        if wall:   # a thin slab along z from 1 m to 80 m on x = 0
            b = lay.make_box("w", "t", 0.0, 40.5, catalogue.Design("slab", "test", 0.4, 79.0, 50.0, 1), 1.0, 0)
            boxes = [(b["tile"], np.array(b["c"], float), np.array(b["ax"], float))]
        fp = Footprints(boxes)
        cells = transit.Cells(fp, (-50, 50, -10, 60))
        ring = clutter.RingTally(fp, cells, h, pitch)
        ring.add(pos, yaw, 0.0, clutter._cams_with_off(fp, cells, pos, yaw, h, pitch, clutter.RANGE_M))
        s = ring.summary()
        assert s["ring_pairs"] == 1 and s["ring_geo_conn_frac"] == 1.0
        assert s["ring_conn_frac"] == conn, (wall, s)
        assert s["inst_any_r40_m2"] > 0


def test_a_short_walk_flies_every_condition_on_one_chain():
    boxes = _boxes([(60.0, 0.0), (-70.0, 40.0)])
    C = search.city_context(boxes, ARENA)
    C["streets"] = None
    from citysim import capability
    band = dict(capability.band_settings("internal_3"))
    opts = dict(band=band, T_s=40.0, run_m=60.0, spreads=[1.0],
                conditions=[("single", 1.0), ("swarm", 1.0)])
    sc = dict(start=[0.0, -60.0], heading0=30.0, seed=5)
    flights = clutter.fly_walk(C, sc, opts)
    assert [(a, d) for a, d, *_ in flights] == [("single", 1.0), ("swarm", 1.0)]
    for arm, d, fl, wall_s, attempts in flights:
        assert abs(fl.sim.t - 40.0) < 0.05 and fl.reason == "cap"
        rec = fl.record(arm=arm)
        assert len(rec.legs) >= 1 and rec.cam_valid.any()
        ring = clutter.RingTally(C["fp"], transit.Cells.around(C["fp"], [rec.centroid], 44.0), band["moving_h"],
                                 band["moving_p"])
        s = transit.score(rec, C["fp"], ring.cells, band["moving_h"], band["moving_p"], ranges=(40.0,),
                          on_sample=ring.hook(rec))
        assert s["sweep_any_r40_m"] > 10 and 0.95 < s["path_ratio"] < 1.6
        r = ring.summary()
        assert (arm == "single") == (r["ring_pairs"] == 0)
    # the same chain: both conditions set off for the same first target
    t0 = [fl.segments[0]["goal"] for _, _, fl, _, _ in flights]
    assert np.allclose(t0[0], t0[1])


def test_arena_cells_are_transit_cells():
    """The city-wide grid, built footprint by footprint, is transit.Cells' grid: the same cells in the same order."""
    fp = Footprints(_boxes([(10.0, 5.0), (-30.0, 40.0), (55.0, -20.0)]))
    b = (-60.0, 70.0, -40.0, 66.0)
    a, c = clutter.arena_cells(fp, b), transit.Cells(fp, b)
    assert a.n == c.n and np.array_equal(a.pts, c.pts) and np.array_equal(a.index, c.index) and a.area == c.area


def test_find_measures_are_the_seen_share_over_time():
    first = np.array([5.0, 20.0, 40.0, np.inf, np.inf, 100.0, 299.0, 301.0])
    inside = np.array([True] * 7 + [False])
    m = clutter.find_measures(first, inside, 300.0, "find_any_r40")
    curve = m["find_any_r40_curve"]
    assert len(curve) == 21 and curve[0] == 0.0 and all(b >= a for a, b in zip(curve, curve[1:]))
    assert abs(m["find_any_r40"] - 5 / 7) < 1e-5 and abs(curve[-1] - 5 / 7) < 1e-5      # rounded to 5 decimals
    assert abs(m["find_any_r40_rmst_s"] - (5 + 20 + 40 + 300 + 300 + 100 + 299) / 7) < 0.01
    lam = -math.log(1 - 5 / 7) / 300.0
    assert abs(m["find_any_r40_time_min"] - 1 / lam / 60) < 0.01


def test_discovery_sees_far_but_not_through_a_wall():
    """One camera at the origin facing north: ground 100 m ahead is discovered at 200 m, not behind a wall across its
    view, and never behind the camera."""
    import types
    rec = types.SimpleNamespace(t=np.array([0.0, 0.5]), cam_valid=np.ones((2, 1), bool),
                                cam_pos=np.zeros((2, 1, 2)), cam_yaw=np.zeros((2, 1)), body_yaw=np.zeros(2),
                                single=True)
    for wall in (False, True):
        boxes = []
        if wall:   # a slab across the view, 60 m north, 60 m wide
            b = lay.make_box("w", "t", 0.0, 60.0, catalogue.Design("slab", "test", 60.0, 1.0, 50.0, 1), 1.0, 0)
            boxes = [(b["tile"], np.array(b["c"], float), np.array(b["ax"], float))]
        fp = Footprints(boxes)
        grid = clutter.arena_cells(fp, (-150.0, 150.0, -150.0, 150.0), step=4.0)
        fa, fh = clutter.discovery_first(rec, fp, grid, 19.5, -15.0)
        ahead = grid.within(np.array([0.0, 100.0]), 3.0)
        behind = grid.within(np.array([0.0, -100.0]), 3.0)
        assert np.isinf(fa[behind]).all() and np.array_equal(fa, fh)          # the single drone's feed is in view
        assert np.isinf(fa[ahead]).all() if wall else (fa[ahead] == 0.0).all()


def test_near_only_offsets_match_the_full_measure():
    track = np.array([[0.0, 0.0], [30.0, 10.0], [60.0, -5.0], [61.0, 40.0]])
    rng = np.random.default_rng(1)
    pts = rng.uniform(-80, 140, size=(4000, 2))
    o1, a1, L1 = transit.lateral_offsets(track, pts)
    o2, a2, L2 = transit.lateral_offsets(track, pts, near=42.0)
    near = np.isfinite(o2)
    assert L1 == L2 and (np.abs(o1[~near]) > 42.0).all()
    assert np.allclose(o1[near], o2[near], rtol=0, atol=1e-9) and np.allclose(a1[near], a2[near], rtol=0, atol=1e-9)


def _units(effect, n_cities=8, n_units=5, seed=0, cell="c"):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cities):
        mu = effect + rng.normal(0, 0.1)
        for u in range(n_units):
            lr = mu + rng.normal(0, 0.2)
            rows.append(dict(metric="sweep", cell=cell, p=90.0, alpha=0.06, layout=f"{cell}{c}", unit=u, d_ref=1.0,
                             sw=math.exp(lr), sg=1.0, lr=lr))
    return pd.DataFrame(rows)


def test_statistics_find_an_effect_and_not_a_null():
    eff = clutter_stats.ratio_cells(_units(0.3)).iloc[0]
    assert eff.p < 0.01 and 1.2 < eff.ratio < 1.5 and eff.lo < eff.ratio < eff.hi and eff.n_cities == 8
    assert eff.p_wilcoxon_units < eff.p          # walks in one city are not independent: the unit test overstates
    null = clutter_stats.ratio_cells(_units(0.0, seed=4)).iloc[0]
    assert null.p > 0.05 and null.lo < 1.0 < null.hi


def test_benjamini_hochberg():
    q = clutter_stats.bh([0.01, 0.04, 0.03, np.nan, 0.5])       # ranks 1, 3, 2, -, 4 of four finite p values
    assert np.isnan(q[3]) and np.allclose(q[[0, 1, 2, 4]], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_trend_recovers_a_slope():
    rows = []
    rng = np.random.default_rng(2)
    for p in (50.0, 100.0, 200.0):
        for a in (0.02, 0.06, 0.10):
            for c in range(4):
                lr = 0.5 - 3.0 * (a - 0.06) + rng.normal(0, 0.02)
                rows.append(dict(metric="sweep", cell=f"{p}_{a}", p=p, alpha=a, layout=f"{p}_{a}_{c}", unit=0,
                                 d_ref=1.0, sw=math.exp(lr), sg=1.0, lr=lr))
    tr = clutter_stats.trends(pd.DataFrame(rows)).set_index("term")
    assert abs(tr.loc["alpha_per_0.01", "coef"] + 3.0) < 0.3 and tr.loc["alpha_per_0.01", "p"] < 1e-6
    assert tr.loc["log2_p", "p"] > 0.01
