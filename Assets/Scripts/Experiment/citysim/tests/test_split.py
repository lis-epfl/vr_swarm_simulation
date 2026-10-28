"""The split study's pieces (S4): polygon footprints against the rectangle ones, block cities (the two variants share
their streets; R's cylinder fits its block; B's inset), the street network and the free-space cluster metric, and the
replica's prism city and closest-point obstacle model."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import blockcity, streetnet  # noqa: E402
from citysim import geometry as geo  # noqa: E402
from citysim.los import Footprints  # noqa: E402
from citysim.polyfoot import ConvexFootprints  # noqa: E402


def _rects(seed=3, n=40):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-100, 100, (n, 2)), rng.uniform(2, 15, (n, 2))], rng


def test_convex_footprints_match_rectangles():
    rects, rng = _rects()
    fa, fb = Footprints.from_rects(rects), ConvexFootprints.from_rects(rects)
    p0, p1 = rng.uniform(-120, 120, (2000, 2)), rng.uniform(-120, 120, (2000, 2))
    assert (fa.blocked(p0, p1) == fb.blocked(p0, p1)).all()
    pts = rng.uniform(-120, 120, (3000, 2))
    assert (fa.inside(pts) == fb.inside(pts)).all() and (fa.inside(pts, 1.5) == fb.inside(pts, 1.5)).all()
    assert np.abs(fa.distance(pts) - fb.distance(pts)).max() < 1e-9
    assert np.abs(fa.R - fb.R).max() < 1e-9
    for _ in range(5):
        o, ang = rng.uniform(-120, 120, 2), np.arange(0.0, 360.0, 0.5)
        a, b = fa.first_hit(o, ang, 80.0), fb.first_hit(o, ang, 80.0)
        assert (np.isfinite(a) == np.isfinite(b)).all()
        assert np.abs(a[np.isfinite(a)] - b[np.isfinite(b)]).max(initial=0.0) < 1e-9


def test_first_hit_reports_the_face_it_entered():
    sq = np.array([[10.0, -5.0], [20.0, -5.0], [20.0, 5.0], [10.0, 5.0]])     # CCW; edge 3 runs (10,5) -> (10,-5)
    f = ConvexFootprints([sq])
    d, b, e = f.first_hit(np.zeros(2), [90.0], 50.0, return_face=True)        # heading 90 = +x
    assert abs(d[0] - 10.0) < 1e-9 and b[0] == 0 and e[0] == 3
    d, b, e = f.first_hit(np.array([15.0, 0.0]), [90.0], 50.0, return_face=True)   # from inside
    assert d[0] == 0.0 and e[0] == -1


def test_block_variants_share_streets_and_keep_their_rules():
    R, _ = blockcity.generate(blockcity.Params("R", 50, 12, 4))
    B, blocks = blockcity.generate(blockcity.Params("B", 50, 12, 4))
    assert R.streets == B.streets
    kerbs = {b["id"]: np.asarray(b["kerb_xz"], float).reshape(-1, 2) for b in blocks if b["kerb_xz"]}
    # R: every avoidance cylinder inside its block's inscribed circle less the margin
    for bx in R.boxes:
        k = int(bx["tile"][1:])
        centre, rho = blockcity.chebyshev(kerbs[k])
        cx, cz = bx["c"][0], bx["c"][2]
        assert math.hypot(cx - centre[0], cz - centre[1]) < 1e-3                # make_box rounds to 1e-4
        # 2 mm: the stored kerb is rounded to 1 mm a vertex, which moves its inscribed circle by ~0.3 mm
        assert blockcity.cylinder_radius(bx) <= rho - blockcity.CYL_MARGIN_M + 2e-3
    # B: every prism is its kerb inset by INSET_M, so the facades across a street stand s + 2 inset apart
    assert B.layout["facade_gap_min"] >= 12.0 + 2 * blockcity.INSET_M - 0.05
    assert R.layout["cyl_gap_frac_ge6"] == 1.0
    # a goal-free citygen city of the same p, s and seed has the same streets
    from dataclasses import replace
    from citysim import citygen
    G = citygen.generate(replace(citygen.Params(), p=50.0, s=12.0, alpha=0.02, goals=0, mix="DiamondCityWorld",
                                 size=1.0, place="even"), 4)
    assert [s["xz"] for s in G.streets] == [s["xz"] for s in B.streets]


def test_block_city_options_leave_s4_alone_and_do_what_they_say():
    """BlockShapedCityWorld's knobs: the defaults are S4's city exactly, a wider footpath moves only the facades, and
    seeds on a regular lattice keep every side of a block a street (S4's RSA seeds keep about four of six)."""
    P = blockcity.Params("B", 40, 18, 1)
    base, base_blocks = blockcity.generate(P)
    same, _ = blockcity.generate(blockcity.Params("B", 40, 18, 1, blockcity.INSET_M, "rsa", 0.3, 0))
    assert P.ident() == "bbB_p40_s18_seed1" and same.layout == base.layout
    assert same.boxes == base.boxes and same.streets == base.streets
    wide, _ = blockcity.generate(blockcity.Params("B", 40, 18, 1, inset=5.0))
    assert blockcity.Params("B", 40, 18, 1, inset=5.0).ident() == "bbB_p40_s18_in5_seed1"
    assert wide.streets == base.streets and wide.layout["facade_gap_min"] >= 18.0 + 2 * 5.0 - 0.05
    hexed, hex_blocks = blockcity.generate(blockcity.Params("B", 40, 18, 1, seeds="hex", jitter=0.0))
    assert blockcity.Params("B", 40, 18, 1, seeds="hex", jitter=0.0).ident() == "bbB_p40_s18_hex0_seed1"
    assert hexed.layout["params"]["seeds"] == "hex" and "seeds" not in base.layout["params"]
    regular, s4 = blockcity.street_stats(hexed, hex_blocks), blockcity.street_stats(base, base_blocks)
    assert regular["sides_mean"] > 5.8 and s4["sides_mean"] < 5.0 < s4["exits_mean"] - 3.0


def test_street_network_blocks_and_the_block_ahead():
    L, blocks = blockcity.generate(blockcity.Params("B", 50, 12, 0))
    net = streetnet.Network(L, blocks, blockcity.footprints(L))
    inner = net.interior_blocks()
    assert len(inner) > 50
    for B in inner[:20]:
        assert len(B["corners"]) >= 3 and len(B["radiating"]) >= 3
        for sid in B["radiating"]:
            st = net.streets[sid]
            corner = st["u"] if st["u"] in B["corners"] else st["v"]
            # arriving along a radiating street at the block's corner faces this block
            if net.streets[sid]["interior"] and len(net.j_streets[corner]) == 3:
                assert net.block_ahead(corner, sid) == B["id"]
    # every interior junction can reach every other
    js = [j for j in range(len(net.jpos)) if net.j_interior[j]]
    route, _ = net.route(js[0], js[-1])
    assert route is not None and route[0] == js[0] and route[-1] == js[-1]


def test_cluster_metric_splits_across_a_building_and_not_round_a_corner():
    sq = np.array([[-15.0, -15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0]])
    L = type("L", (), {})()
    L.streets, L.boxes = [], []
    net = streetnet.Network.__new__(streetnet.Network)
    net.fp = ConvexFootprints([sq])
    across = np.array([[-20.0, 0.0], [-20.0, 4.0], [20.0, 0.0], [20.0, 4.0]])   # 40 m apart, the building between
    assert streetnet.split_state(net.clusters(across))[2]
    corner = np.array([[-20.0, 10.0], [-10.0, 20.0]])                          # round a corner, ~14 m
    assert len(set(net.clusters(corner))) == 1


def test_poly_city_closest_point_and_push_out():
    from citysim import use_bench_modules
    use_bench_modules()
    import swarm_replica as rep
    sq = np.array([[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]])
    city = rep.PolyCity([sq])
    q, d = city.closest(np.array([[0.0, 9.0], [14.0, 9.0], [0.0, 0.0]]), np.array([0]))
    assert np.allclose(q[0, 0], [0.0, 5.0]) and abs(d[0, 0] - 4.0) < 1e-9
    assert np.allclose(q[1, 0], [10.0, 5.0]) and abs(d[1, 0] - math.hypot(4.0, 4.0)) < 1e-9
    assert d[2, 0] == 0.0
    assert abs(city.R[0] - math.hypot(10.0, 5.0)) < 1e-9          # the cylinder model's bounds circumradius
    p, v = city.push_out(np.array([0]), np.array([0.0, 4.0]), np.array([0.0, -2.0]), 0.3, 0.0)
    assert abs(p[1] - 5.301) < 1e-6 and abs(v[1]) < 1e-12           # out through the top face, inward velocity gone
    assert city.footprint_hit(np.array([[0.0, 5.2]]), np.array([0]), 0.3)[0]


def test_closest_model_repels_from_the_true_facade():
    """A drone 3 m from a long facade feels the closest-point beta-agent; under the cylinder model the same drone is
    inside the bounds circle (distance 0), which is what closes the streets."""
    from citysim import use_bench_modules
    use_bench_modules()
    import swarm_params as sp
    import swarm_replica as rep
    wall = np.array([[-30.0, -10.0], [30.0, -10.0], [30.0, 10.0], [-30.0, 10.0]])
    city = rep.PolyCity([wall])
    swarm, air = dict(sp.resolve_swarm("ProceduralCityWorld")), dict(sp.resolve_airframe())
    Pm = np.array([[0.0, 13.0]])
    Vm = np.zeros((1, 2))
    args = (Pm, Vm, Pm.mean(0), np.zeros(2), 0.0, 1.0, np.zeros(2), np.array([[0.0, 1.0]]))
    for name, extras in (("cylinder", {}), ("closest", dict(obstacle_model="closest"))):
        m = rep.Model(swarm, air, extras)
        T = rep.swarm_terms(m, city, np.array([0]), *args)
        assert T["obs"][0, 1] > 0                                    # pushed away from the wall (+z)
    m = rep.Model(swarm, air, dict(obstacle_model="closest", face_bend=1.0))
    off = rep.swarm_terms(m, city, np.array([0]), np.array([[20.0, 13.0]]), Vm, np.array([20.0, 13.0]), np.zeros(2),
                          0.0, 1.0, np.zeros(2), np.array([[0.0, 1.0]]))["obs"]
    assert off[0, 0] > 0                                             # the bend slides it toward the nearer end (+x)


def test_correspondence_links_a_street_and_fragments_a_surround():
    """Screens looking out of one street share their ground; screens looking out from round a building share
    nothing (four islands, every screen an orphan), though their order round the ring stays true."""
    from citysim import clutter, correspondence as corr
    bld = np.array([[-15.0, -15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0]])
    fp = ConvexFootprints([bld])
    cells = clutter.arena_cells(fp, (-100, 100, -100, 100))
    street = corr.Ring(fp, cells, 12.4, -20.8)
    cams = np.array([[-6.0, 30.0], [-2.0, 30.0], [2.0, 30.0], [6.0, 30.0]])
    street.add(0.0, cams, np.array([-20.0, -7.0, 7.0, 20.0]), 0.0, cams.mean(0))
    s = street.summary()
    assert s["corr_a1_linked"] == 1.0 and s["corr_islands"] == 1.0 and s["corr_orphan_share"] == 0.0
    ring = corr.Ring(fp, cells, 12.4, -20.8)
    ring.add(0.0, np.array([[0.0, 25.0], [25.0, 0.0], [0.0, -25.0], [-25.0, 0.0]]),
             np.array([0.0, 90.0, 180.0, -90.0]), 0.0, np.zeros(2))
    r = ring.summary()
    assert r["corr_a1_linked"] == 0.0 and r["corr_islands"] == 4.0 and r["corr_orphan_share"] == 1.0
    assert r["corr_b2_order_rev"] < 0.05
