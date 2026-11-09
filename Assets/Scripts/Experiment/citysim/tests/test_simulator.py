"""The simulator's pieces (M2): line of sight against analyse.py's, the hull / heading port, the walker loops, the
replica templates, the envelopment map's monotonicity, the exposure engine, and Tier A against Tier B."""
import conftest  # noqa: F401

import math
from dataclasses import replace

import numpy as np

from citysim import camera, citygen, display, exposure, geometry as geo, layout as lay, theory, walkers
from citysim.los import Boxes, Footprints, wrap_deg

BASE = citygen.Params(p=70.0, s=12.0, alpha=0.12, size=2.0, w_c=13.06, r_g=30.0, n_b=8)


# ----------------------------------------------------------------------------- line of sight

def test_slab_test_equals_analyse_on_random_segments():
    import analyse
    rng = np.random.default_rng(0)
    boxes, _ = analyse.load_city("DiamondCityWorld")
    sel = [b for b in boxes if abs(b[1][0]) < 120 and abs(b[1][2]) < 120]
    p0 = np.c_[rng.uniform(-130, 130, 10_000), rng.uniform(0, 60, 10_000), rng.uniform(-130, 130, 10_000)]
    p1 = p0 + np.c_[rng.normal(0, 40, 10_000), rng.normal(0, 15, 10_000), rng.normal(0, 40, 10_000)]
    ours = Boxes(sel).blocked(p0, p1)
    theirs = analyse.segment_blocked(p0, p1, sel)
    assert (ours == theirs).all() and 0.05 < ours.mean() < 0.95
    L = citygen.generate(BASE, 1)
    lb = L.boxes_as_tuples()
    assert (Boxes(lb).blocked(p0, p1) == analyse.segment_blocked(p0, p1, lb)).all()


def test_2d_equals_3d_below_the_roofs():
    rng = np.random.default_rng(1)
    L = citygen.generate(BASE, 2)
    b = L.boxes_as_tuples()
    n = 5000
    xz0 = np.c_[rng.uniform(-300, 300, n), rng.uniform(-300, 300, n)]
    xz1 = xz0 + rng.normal(0, 40, (n, 2))
    h0, h1 = rng.uniform(0.5, 49.5, n), rng.uniform(0.5, 49.5, n)
    three = Boxes(b).blocked(np.c_[xz0[:, 0], h0, xz0[:, 1]], np.c_[xz1[:, 0], h1, xz1[:, 1]])
    two = Footprints(b).blocked(xz0, xz1)
    assert (three == two).all()


def test_first_hit_against_marching():
    L = citygen.generate(BASE, 3)
    fp = Footprints(L.boxes_as_tuples())
    o = viewpoint = np.array([0.0, 0.0])
    if fp.inside([o])[0]:
        viewpoint = o = np.array([fp.c[0, 0] + fp.h[0, 0] + 3.0, fp.c[0, 1]])
    ang = np.arange(0.0, 360.0, 7.0)
    d = fp.first_hit(viewpoint, ang, 80.0)
    for a, dd in zip(ang, d):
        u = np.array([math.sin(math.radians(a)), math.cos(math.radians(a))])
        steps = np.arange(0.0, 80.0, 0.05)
        inside = fp.inside(viewpoint + steps[:, None] * u)
        march = steps[np.argmax(inside)] if inside.any() else math.inf
        assert march >= dd - 1e-9                                  # never misses what marching finds
        if math.isfinite(dd):                                      # a hit is on a footprint's boundary ...
            assert fp.distance([viewpoint + dd * u])[0] < 1e-6
            assert not inside[steps < dd - 0.05].any()             # ... with nothing in the way before it
            assert math.isinf(march) or march - dd <= 0.06         # (or a corner graze marching steps over)


# ----------------------------------------------------------------------------- the display rule

def test_hull_matches_scipy():
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(4)
    for _ in range(50):
        P = rng.normal(0, 10, (rng.integers(3, 15), 2))
        assert set(display.convex_hull(P)) == set(ConvexHull(P).vertices)


def test_bisectors_point_inward_and_headings_outward():
    ang = np.radians(np.arange(0, 360, 36))
    P = 10 * np.stack([np.sin(ang), np.cos(ang)], -1)               # a regular decagon, headings 0, 36, ...
    h = display.convex_hull(P)
    yaw = display.heading_deg(-display.inward_bisectors(P[h]))
    expect = np.degrees(ang[h])
    assert np.abs(wrap_deg(yaw - expect)).max() < 1e-9


def test_look_gap_shift_matches_the_csharp_rule():
    shift, raw, post = display.look_gap_shift(np.array([0.0, 90.0, 180.0, 270.0]), 45.0, 20.0, 15.0)
    assert np.allclose(shift, [15.0, -15.0, 0.0, 0.0]) and raw == 45.0 and post == 30.0
    shift, raw, post = display.look_gap_shift(np.array([0.0, 90.0, 180.0, 270.0]), 10.0, 20.0, 15.0)
    assert np.allclose(shift, 0.0) and raw == 10.0            # within the coverage angle: no turn
    shift, raw, _ = display.look_gap_shift(np.array([-170.0, -10.0, 100.0]), 140.0, 20.0, 15.0)
    assert raw == 40.0 and shift[0] == -15.0 and shift[2] == 15.0   # the pair straddles the +-180 seam


def test_display_azimuth_and_head():
    az = display.display_azimuth(120.0, 100.0, 10.0)
    assert abs(az - (20.0 + 10.0 * camera.DISPLAY_SCALE)) < 1e-9
    assert display.head_visible(az, 0.0) and not display.head_visible(az, -40.0)
    assert display.swarm_reveal_deg(0) == camera.HMD_HALF_FOV_DEG
    assert display.single_reveal_deg(0) == camera.HFOV_DEG / 2
    assert display.swarm_reveal_deg(10) == 180.0 and display.single_reveal_deg(10) == 180.0


# ----------------------------------------------------------------------------- walkers

def test_walker_loop_matches_the_patrol_rule():
    lp = walkers.Loop((10.0, -5.0, 3.0, 2.0), 1.0, 0.0, clockwise=False)
    r = lp.r
    assert abs(lp.perimeter - (2 * 2 * (3.0 - r) + 2 * 2 * (4.0 - r) + 2 * math.pi * r)) < 1e-9   # hd 3, hw 4
    t = np.arange(0.0, 60.0, 0.01)
    p = lp.position(t)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert abs(step.mean() / 0.01 - walkers.SPEED) < 0.01 and step.max() < 0.05
    # the path stays 1 m off the AABB (corners rounded inwards)
    o = np.abs(p - [10.0, -5.0]) - [3.0, 2.0]
    gap = np.where((o > 0).all(1), np.hypot(*np.maximum(o, 0).T), np.max(o, axis=1))
    assert gap.min() > 1.0 - 1e-6 and gap.max() < 1.0 + 1e-6
    cw = walkers.Loop((10.0, -5.0, 3.0, 2.0), 1.0, 0.0, clockwise=True).position(np.array([0.0, 0.5]))
    ccw = lp.position(np.array([0.0, 0.5]))
    assert cw[1, 1] < cw[0, 1] < ccw[1, 1]                    # s = 0 is on the right edge, moving -z / +z


def test_goal_walkers_count():
    W = walkers.GoalWalkers([(0.0, 0.0, 3.0, 2.0)] * 8, np.random.default_rng(0))
    assert len(W.loops) == 64 and 0 <= W.target < 64
    assert W.positions(np.arange(0, 2, 0.5)).shape == (64, 4, 2)


# ----------------------------------------------------------------------------- replica templates

def test_hover_template_is_the_replicas_settle():
    from citysim import formation
    import swarm_replica as rep
    P, alive = formation.hover_template(1.0, n=10, seed=3)
    m = formation.model()
    rng = np.random.default_rng(1003)
    sim = rep.Sim(m, rep.City([]), rng.normal(0, 6.0, (10, 2)), np.zeros((10, 2)), 3)
    while sim.t < formation.SETTLE_S:
        sim.step(np.zeros(2), 0.0, 1.0)
    Q = sim.P[sim.alive]
    assert alive.all() and np.allclose(P, Q - Q.mean(0))


def test_envelopment_needs_spread():
    from citysim import formation
    rows = [formation._envmap_job((13.06, 30.0, 8, d, 10, 0, 2.0)) for d in (0.4, 1.6)]
    assert not rows[0]["enveloped"] and rows[1]["enveloped"]
    assert rows[1]["diameter_m"] > rows[0]["diameter_m"]


# ----------------------------------------------------------------------------- exposure

def test_exposure_sees_and_is_blocked():
    W = walkers.GoalWalkers([(0.0, 20.0, 3.0, 3.0)], np.random.default_rng(1), n=8)
    t = np.arange(0.0, 20.0, 0.5)
    feeds = display.single_feed(np.array([0.0, 0.0]), 0.0)
    tr = exposure.static_track(feeds, len(t), 0.5, 5.0, -10.0, 0.0, single=True)
    open_ = exposure.exposure(Footprints([]), tr, W, t)
    assert open_["E_any"] > 0.2
    wall = Footprints.from_rects([(0.0, 10.0, 15.0, 0.5)])
    assert exposure.exposure(wall, tr, W, t)["E_any"] == 0.0
    high = exposure.static_track(feeds, len(t), 0.5, 35.0, -5.0, 0.0, single=True)
    assert exposure.exposure(Footprints([]), high, W, t)["E_any"] == 0.0    # above the band and out of range


# ----------------------------------------------------------------------------- Tier A vs Tier B

def test_boolean_model_as_alpha_goes_to_zero():
    from citysim.los import free_sight_probability
    P = replace(BASE, mode="scatter", alpha=0.02, size=2.0)
    L = citygen.generate(P, 4)
    fp = Footprints(L.boxes_as_tuples())
    rng = np.random.default_rng(5)
    o = np.c_[rng.uniform(-300, 300, 400), rng.uniform(-300, 300, 400)]
    o = o[fp.distance(o) > 1.0][:200]
    from citysim import catalogue
    d = catalogue.study_design()
    ell = theory.ell(L.layout["alpha_achieved"], d.w * 2, d.d * 2)
    for r in (40.0, 120.0):
        measured = free_sight_probability(fp, o, r)
        assert abs(measured - theory.p_free(r, ell)) < 0.05, (r, measured, theory.p_free(r, ell))


# ----------------------------------------------------------------------------- the replica's contact model

def _wall_city():
    import swarm_replica as rep
    # one 10 x 50 x 4 m wall centred on the origin, long side along x: (tile, centre, half-axis vectors)
    return rep.City([("t", np.array([0.0, 25.0, 0.0]), np.diag([5.0, 25.0, 2.0]))])


def test_push_out_removes_the_inward_velocity_and_applies_friction():
    import swarm_replica as rep
    city = _wall_city()
    cand = np.arange(city.n)
    p, v = np.array([1.0, -1.9]), np.array([3.0, 1.0])          # inside, near the -z face, moving +x and into it
    q0, v0 = rep.push_out(city, cand, p, v, friction=0.0)
    assert q0[1] < -2.0 - rep.DRONE_HALF_M and abs(q0[0] - 1.0) < 1e-9
    assert np.allclose(v0, [3.0, 0.0])
    q1, v1 = rep.push_out(city, cand, p, v, friction=0.6)
    assert np.allclose(q1, q0) and np.allclose(v1, [3.0 - 0.6 * 1.0, 0.0])
    _, v2 = rep.push_out(city, cand, p, np.array([0.2, 1.0]), friction=0.6)
    assert np.allclose(v2, [0.0, 0.0])                         # friction stops, never reverses, the slide


def test_contact_events_and_isolation():
    import swarm_replica as rep
    from citysim import formation
    m = formation.model()
    city = _wall_city()
    P0 = np.array([[0.0, -10.0], [3.0, -10.0], [-3.0, -10.0]])
    sim = rep.ContactSim(m, city, P0, np.zeros((3, 2)), 0)
    sim.cand = np.arange(city.n)
    idx = np.arange(3)

    def touch(z):
        P = P0.copy()
        P[0, 1] = z
        sim.P[:] = P
        sim._losses(idx, P, 1.0)

    touch(-1.9)                      # inside the footprint: one contact, pushed out
    assert sim.contacts == 1 and sim.alive.all() and sim.P[0, 1] < -2.0
    touch(-2.1)                      # pressed against it again (inside the 0.25 m drone margin): same contact
    assert sim.contacts == 1
    touch(-10.0)                     # away: the contact ends
    touch(-1.9)                      # and a new one begins
    assert sim.contacts == 2 and sim.contact_ticks == 3
    far = np.array([[0.0, -10.0], [3.0, -10.0], [400.0, -10.0]])
    sim.P[:] = far
    sim._losses(idx, far, 1.0)       # r0_coh 20 x S 10: a drone 397 m from its nearest neighbour is lost
    assert list(sim.alive) == [True, True, False] and sim.kills[-1][1] == "isolated"
