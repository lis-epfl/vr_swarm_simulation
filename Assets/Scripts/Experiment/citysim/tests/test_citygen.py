"""The layout generator's invariants (M1): geometry primitives, determinism, no overlaps, a connected street
network at its full width, coverage on target, the 50 m overlap bound, the goal rules, and the file read
unchanged by analyse.load_city, swarm_replica.City and bench.Footprints."""
import conftest  # noqa: F401  (paths, bytecode)

import math
from dataclasses import replace

import numpy as np

from citysim import citygen, geometry as geo, layout as lay

BASE = citygen.Params(p=70.0, s=12.0, alpha=0.12, size=2.0, w_c=13.06, r_g=30.0, n_b=8)


# ----------------------------------------------------------------------------- geometry

def test_voronoi_tiles_the_box():
    rng = np.random.default_rng(3)
    box = (-100.0, 120.0, -80.0, 90.0)
    seeds = geo.rsa_points(rng, box, 40, 15.0)
    cells, edges = geo.voronoi_cells(seeds, box)
    total = sum(geo.area(c) for c in cells)
    assert abs(total - 220.0 * 170.0) < 1e-6 * 220 * 170
    for i, c in enumerate(cells):
        assert geo.area(c) > 0
        assert geo.contains(c, [seeds[i]])[0]
    # every interior edge is equidistant from its two seeds
    for i, j, a, b in edges:
        if j >= 0:
            m = 0.5 * (a + b)
            assert abs(np.linalg.norm(m - seeds[i]) - np.linalg.norm(m - seeds[j])) < 1e-6


def test_offset_and_clip():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    inner = geo.offset_convex(sq, 2.0)
    assert abs(geo.area(inner) - 36.0) < 1e-9
    assert len(geo.offset_convex(sq, 5.5)) == 0
    tri = np.array([[-5, 5], [15, 5], [5, 20]], float)
    c = geo.clip_convex(tri, sq)
    assert abs(geo.area(c) - geo.area(geo.clip_convex(sq, tri))) < 1e-9
    assert geo.contains(sq, [[5, 5]], margin=4.9)[0] and not geo.contains(sq, [[5, 5]], margin=5.1)[0]


def test_rect_gap_is_chebyshev():
    r = (0.0, 0.0, 1.0, 2.0)
    o = np.array([[5.0, 0.0, 1.0, 1.0], [0.0, 10.0, 1.0, 1.0], [1.5, 0.0, 1.0, 1.0]])
    assert np.allclose(geo.rect_gap(r, o), [3.0, 7.0, -0.5])
    d = geo.rect_point_distance(np.array([r]), [[4.0, 6.0], [0.5, 0.5]])
    assert np.allclose(d[0], [5.0, 0.0])


# ----------------------------------------------------------------------------- generation

def test_deterministic_in_seed():
    a = citygen.generate(BASE, 11)
    b = citygen.generate(BASE, 11)
    c = citygen.generate(BASE, 12)
    assert a.boxes == b.boxes and a.streets == b.streets and a.layout == b.layout
    assert a.boxes != c.boxes


def test_layouts_verify_across_the_range():
    cases = [BASE, replace(BASE, alpha=0.30, p=55.0, s=10.0), replace(BASE, alpha=0.024, p=90.83, s=15.0, size=1.0,
                                                                     w_c=7.0, r_g=24.0),
             replace(BASE, mode="scatter", alpha=0.08)]
    for P in cases:
        L = citygen.generate(P, 5)
        assert L.verify() == [], (P, L.verify())


def test_coverage_on_target_unless_jammed():
    for alpha in (0.05, 0.15, 0.25):
        L = citygen.generate(replace(BASE, alpha=alpha), 2)
        r = L.layout
        if not r["jammed"]:
            assert abs(r["alpha_achieved"] - alpha) <= 0.005, r["alpha_achieved"]


def test_jamming_is_reported():
    L = citygen.generate(replace(BASE, alpha=0.9), 1)
    assert L.layout["jammed"] and L.layout["alpha_achieved"] < 0.9


def test_street_network_connected_and_full_width():
    L = citygen.generate(BASE, 4)
    # union-find over street end points (Voronoi vertices are shared exactly)
    parent = {}

    def find(k):
        while parent.setdefault(k, k) != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for s in L.streets:
        a, b = (tuple(np.round(p, 2)) for p in s["polyline"])
        parent[find(a)] = find(b)
    roots = {find(k) for k in list(parent)}
    assert len(roots) == 1
    r = L.layout
    assert r["min_street_width"] >= BASE.s - 0.05
    # every block touches a street: its seed's cell edges are all streets, so each block id appears in one
    blocks_on_streets = {i for s in L.streets for i in s["blocks"] if i >= 0}
    assert blocks_on_streets == set(range(r["n_blocks"])) or len(blocks_on_streets) >= r["n_blocks"]


def test_overlap_bound():
    L = citygen.generate(replace(BASE, alpha=0.30, p=55.0, s=10.0), 3)
    assert L.layout["max_within_50m"] <= lay.OVERLAP_LIMIT
    # the bound is real: at x1 the same coverage packs far more buildings into 50 m
    small = citygen.generate(replace(BASE, alpha=0.30, p=55.0, s=10.0, size=1.0, w_c=7.0, r_g=24.0), 3)
    assert small.layout["max_within_50m"] > L.layout["max_within_50m"]


def test_goal_rules():
    L = citygen.generate(BASE, 7)
    goals = L.goals()
    assert len(goals) == 3
    c = np.array([g["centre"] for g in goals])
    sp = L.layout["spawn"]
    for i in range(3):
        assert math.hypot(c[i, 0] - sp[0], c[i, 1] - sp[2]) >= lay.GOAL_MIN_SPAWN_DIST_M
        for j in range(i + 1, 3):
            assert np.linalg.norm(c[i] - c[j]) >= lay.GOAL_MIN_SEPARATION_M
    for g in goals:
        assert abs(g["w_c"] - 13.06) < 1e-6 and g["c_g"] > g["R_c"]
        ring = [b for b in L.boxes if b["goal"] == g["index"] and b["role"] == "goal_ring"]
        assert len(ring) == 8 and all(b["patrol"] for b in ring)
    # stratified near / mid / far
    d = [g["spawn_dist"] for g in goals]
    assert d == sorted(d)


def test_mix_city_draws_every_building_from_the_tile_city():
    """A building mix (VoronoiCityWorld's layouts): every building a design the tile city has, occluders near
    their own proportions at a long side of exactly w_c, each goal's clear radius from its own ring, and the
    layout passes every rule."""
    from citysim import catalogue

    P = replace(BASE, mix="DiamondCityWorld", size=1.0, alpha=0.0227, p=90.83, s=15.0)
    L = citygen.generate(P, 3)
    assert L.verify() == [], L.verify()
    assert L.boxes == citygen.generate(P, 3).boxes
    mix = dict((d.design, c) for d, c in catalogue.city_mix("DiamondCityWorld"))
    assert set(b["design"] for b in L.boxes) <= set(mix)
    assert len(set(b["design"] for b in L.boxes if b["role"] == "city")) > 20       # a mix, not one design
    r = L.layout
    assert r["design"] == "mix:DiamondCityWorld" and r["mix_counts"] == mix and r["params"]["mix"] == "DiamondCityWorld"
    assert r["id"].endswith("_mixDiamondCityWorld_seed3")
    designs = {d.design: d for d in catalogue.load()}
    for g in L.goals():
        own = [b for b in L.boxes if b["goal"] == g["index"]]
        occ = designs[own[0]["design"]]
        assert own[0]["role"] == "goal_centre" and abs(occ.long * own[0]["size"] - g["w_c"]) < 1e-3
        assert 1 / citygen.OCCLUDER_RATIO <= own[0]["size"] <= citygen.OCCLUDER_RATIO
        half = max(0.5 * math.hypot(designs[b["design"]].w, designs[b["design"]].d) for b in own[1:])
        assert abs(g["clear_radius"] - (g["r_g"] + half * g["ring_size"] + P.walk_clear)) < 1e-3


def test_even_placement_fills_every_block_alike():
    """place="even": every block to about the same built fraction (the RSA's round robin gives each the same count,
    whatever its area), under every rule the RSA keeps, at the same coverage, and deterministic."""
    P = replace(BASE, mix="DiamondCityWorld", size=1.0, alpha=0.0227, p=90.83, s=15.0)

    def fill_spread(L):
        cells, _ = geo.voronoi_cells(geo.rsa_points(np.random.default_rng(L.layout["seed"]), P.arena,
                                                    max(int(round(730.0 ** 2 / P.p ** 2)), 4),
                                                    citygen.SEED_SPACING_RATIO * P.p), P.arena)
        rects = L.rects()
        fills = []
        for k, c in enumerate(cells):
            b = geo.offset_convex(c, P.s / 2)
            own = [i for i, x in enumerate(L.boxes) if x["tile"] == f"B{k:03d}"]
            if len(b) >= 3 and own and not any(x["role"] != "city" and geo.contains(b, [x["c"][0::2]])[0]
                                               for x in L.boxes):
                fills.append(geo.rect_area(rects[own]).sum() / abs(geo.area(b)))
        return np.percentile(fills, 90) / np.percentile(fills, 10)

    rsa, even = citygen.generate(P, 4), citygen.generate(replace(P, place="even"), 4)
    assert even.verify() == [], even.verify()
    assert even.boxes == citygen.generate(replace(P, place="even"), 4).boxes
    assert abs(even.layout["alpha_achieved"] - rsa.layout["alpha_achieved"]) < 0.002
    assert even.layout["id"].endswith("_even_seed4") and even.layout["params"]["place"] == "even"
    assert fill_spread(even) < 0.7 * fill_spread(rsa), (fill_spread(even), fill_spread(rsa))


def test_one_design_layouts_are_unchanged_by_the_mix():
    """The mix costs the one-design path no random number: a committed trial layout regenerates byte for byte."""
    import json
    import tempfile
    from pathlib import Path

    from citysim import LAYOUT_DIR

    f = sorted(LAYOUT_DIR.glob("L1_*_seed101.json"))[0]
    rec = json.loads(f.read_text(encoding="utf-8"))["layout"]
    P = replace(citygen.Params(), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in rec["params"].items()})
    L = citygen.generate(P, rec["seed"])
    for k in ("trial_level", "trial_level_name"):
        L.layout[k] = rec[k]
    with tempfile.TemporaryDirectory() as d:
        assert L.save(Path(d) / f.name).read_bytes() == f.read_bytes()


def test_infeasible_goal_geometry_raises():
    try:
        citygen.generate(replace(BASE, r_g=12.0, n_b=8), 1)
    except ValueError as e:
        assert "infeasible" in str(e)
    else:
        raise AssertionError("an 8-building ring at 12 m around a 13 m occluder should not fit")


def test_round_trip_through_the_consumers(tmp_path):
    import analyse
    import bench
    import swarm_replica

    L = citygen.generate(BASE, 9)
    path = L.save(tmp_path / f"{L.id}.json")
    L2 = lay.load(path)
    assert L2.boxes == L.boxes and L2.layout == L.layout and L2.streets == L.streets
    boxes, kerbs = analyse.load_city(str(path))
    assert kerbs == {} and len(boxes) == len(L.boxes)
    assert all(np.allclose(b[1], lb["c"]) and np.allclose(b[2], lb["ax"]) for b, lb in zip(boxes, L.boxes))
    assert analyse.is_layout(str(path))
    assert len(analyse.scene_boxes([{"goalX": 0.0, "goalZ": 0.0}], str(path))) == len(L.boxes)
    city = swarm_replica.City(boxes)
    rects = L.rects()
    assert np.allclose(city.R, np.hypot(rects[:, 2], rects[:, 3]))
    fp = bench.Footprints(str(path))
    g = L.goals()[0]
    assert not fp.clear(g["centre"], 1.0)                     # the occluder stands there
    assert fp.clear([L.layout["spawn"][0], L.layout["spawn"][2]], 5.0)


def test_goal_flights_and_tierC_bookkeeping(tmp_path):
    """bench.py's goal flights visit every goal once, holding on it; Tier C's scorer finds each hold's goal in either
    visit order and decodes the trajectory masks per drone."""
    import bench
    import pandas as pd
    from citysim import tierc

    L = citygen.generate(BASE, 9)
    path = L.save(tmp_path / f"{L.id}.json")
    flights = bench.goal_flights(str(path))
    assert sorted(f["kind"] for f in flights) == ["goals_0.4", "goals_0.4", "goals_1.0", "goals_1.0"]
    centres = [g["index"] for g in L.goals()]
    for f in flights:
        holds = tierc.hold_goals(f, L)
        assert len(holds) == len(centres)
        order = [g["index"] for _, g in holds]
        assert order == (centres if f["order"] == "forward" else centres[::-1])
        wps = f["wps"]
        for k, g in holds:
            assert wps[4 * k + 2] < 0 and np.allclose(wps[4 * k:4 * k + 2], g["centre"])
            assert wps[4 * k - 4:4 * k - 2] == wps[4 * k:4 * k + 2]       # arrival on the goal, then the hold
        assert f["endOnDone"] and f["arrive"] > 0 and f["holdSpread"] > 0
    # a two-drone trajectory as SwarmBenchRunner writes it
    df = pd.DataFrame(dict(t=[0.0, 0.1], alive_mask=[3, 1], hull_mask=[3, 0], shown_mask=[2, 1],
                           x0=[0.0, 0.1], z0=[0.0, 0.0], x1=[5.0, 5.0], z1=[0.0, 0.0], yaw0=[0.0, 1.0],
                           yaw1=[90.0, 90.0], y0=[13.7, 13.7], y1=[13.6, 13.6], wp=[0, 1], bodyYaw=[10.0, 10.0]))
    df.to_csv(tmp_path / "traj.csv", index=False)
    tr = tierc.read_traj(tmp_path / "traj.csv")
    assert tr["n"] == 2
    assert tr["alive"].tolist() == [[True, True], [True, False]]
    assert tr["shown"].tolist() == [[False, True], [True, False]]
    assert tr["Y"][0].tolist() == [13.7, 13.6] and tr["wp"].tolist() == [0, 1]

