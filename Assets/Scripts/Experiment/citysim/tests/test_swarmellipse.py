"""S10 Ellipse Avoidance: the replica's ellipse (swarm_replica.mvee_axis, ellipse_fit, PolyCity.ellipses, ell_frame and
the S7 rules carried into its unit space). The fit encloses and is the minimum one within the cap; cap 1 is the
enclosing circle exactly; the cache is per cap; the closest point is right; and an ellipse with a = b, flown through
the ellipse's own code, is the circle to rounding -- the regression that keeps the new path honest where the shipped
one short-circuits to the circle."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)


def _rep():
    from citysim import use_bench_modules
    use_bench_modules()
    import swarm_replica as rep
    return rep


def _rect(w, h, theta, centre=(0.0, 0.0)):
    c, s = math.cos(theta), math.sin(theta)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + np.asarray(centre)


def _convex(rng, n=6, r=12.0, stretch=1.6):
    """A random convex footprint: a hull of points on a stretched, rotated circle."""
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    th = rng.uniform(0, np.pi)
    p = np.c_[r * stretch * np.cos(ang), r * np.sin(ang)]
    p = p @ np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    return p + rng.uniform(-200, 200, 2)


def test_mvee_axis_of_a_rectangle():
    """A w x h rectangle's minimum-area ellipse has the rectangle's aspect and its long side as the axis."""
    rep = _rep()
    for th in (0.0, 0.4, 1.2, 2.9):
        ax, k = rep.mvee_axis(_rect(24.0, 10.0, th, (130.0, -40.0)))
        want = np.array([math.cos(th), math.sin(th)])
        want = want if want[0] > 0 else -want
        assert abs(k - 2.4) < 1e-4, k
        assert np.allclose(ax, want, atol=1e-5), (ax, want)
    ax, k = rep.mvee_axis(_rect(10.0, 10.0, 0.3))
    assert abs(k - 1.0) < 1e-4


def test_fit_encloses_and_is_the_minimum_within_the_cap():
    rep = _rep()
    p = _rect(24.0, 10.0, 0.7, (55.0, 12.0))
    ax, k = rep.mvee_axis(p)
    c0, r0 = rep.enclosing_circle(p)
    c, e, a, b, circ = rep.ellipse_fit(p, 3.0, ax, k, c0, r0)
    assert not circ
    assert abs(a - 12.0 * math.sqrt(2)) < 1e-3 and abs(b - 5.0 * math.sqrt(2)) < 1e-3   # the MVEE of a rectangle
    assert np.allclose(c, [55.0, 12.0], atol=1e-6)
    for cap in (1.25, 1.5, 2.0):
        c, e, a, b, circ = rep.ellipse_fit(p, cap, ax, k, c0, r0)
        assert abs(a / b - cap) < 1e-9
        d = p - c
        x, y = d @ e, d @ np.array([-e[1], e[0]])
        assert ((x / a) ** 2 + (y / b) ** 2).max() <= 1.0 + 1e-6
        assert math.pi * a * b < math.pi * r0 * r0                        # tighter than the circle


def test_cap_one_is_the_enclosing_circle_and_caps_are_cached_apart():
    rep = _rep()
    rng = np.random.default_rng(4)
    city = rep.PolyCity([_convex(rng) for _ in range(12)])
    E1 = city.ellipses(1.0)
    assert E1.circ.all() and (E1.c == city.mec_c).all() and (E1.a == city.mec_R).all() and (E1.b == city.mec_R).all()
    E15, E2 = city.ellipses(1.5), city.ellipses(2.0)
    assert city.ellipses(1.5) is E15                                      # memoised per cap
    assert not np.allclose(E15.a / E15.b, E2.a / E2.b)                   # two caps, two shapes
    assert (E15.a / E15.b <= 1.5 + 1e-9).all() and (E2.a / E2.b <= 2.0 + 1e-9).all()
    for E in (E15, E2):
        for i, p in enumerate(city.polys):
            d = p - E.c[i]
            assert (((d @ E.e[i]) / E.a[i]) ** 2 + ((d @ E.ep[i]) / E.b[i]) ** 2).max() <= 1.0 + 1e-6


def test_closest_point_matches_dense_sampling():
    rep = _rep()
    rng = np.random.default_rng(0)
    for k in (1.1, 1.5, 2.0, 3.6):
        a, b = 10.0 * k, 10.0
        th = np.linspace(0, np.pi / 2, 40001)
        bx, by = a * np.cos(th), b * np.sin(th)
        P = rng.uniform(0, 1, (300, 2)) * [a + 60, b + 60]
        P = P[(P[:, 0] / a) ** 2 + (P[:, 1] / b) ** 2 >= 1]
        tx, ty = rep.ell_closest(P[:, 0], P[:, 1], a, b)
        d = np.hypot(P[:, 0] - a * tx, P[:, 1] - b * ty)
        dense = np.min(np.hypot(P[:, None, 0] - bx[None], P[:, None, 1] - by[None]), axis=1)
        assert np.abs(d - dense).max() < 1e-4, k


def test_frame_is_continuous_through_the_surface():
    """Just outside, the normal at the closest point; just inside, the gradient at the point: the same direction at
    the surface, with mu 1 and distance 0 on both sides of it."""
    rep = _rep()
    E = rep.Ellipses(np.array([[3.0, -2.0]]), np.array([[math.cos(0.5), math.sin(0.5)]]), np.array([18.0]),
                     np.array([9.0]), np.array([False]))
    t = np.linspace(0.05, 2 * np.pi, 37)
    loc = np.c_[18.0 * np.cos(t), 9.0 * np.sin(t)]
    nrm = np.c_[np.cos(t) / 18.0, np.sin(t) / 9.0]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    world = lambda q: E.c[0] + q[:, :1] * E.e[0] + q[:, 1:] * E.ep[0]     # noqa: E731
    for eps in (1e-3, 1e-6):
        o_out, d_out, mu_out, _, _, _ = rep.ell_frame(world(loc + eps * nrm), E, np.array([0]))
        o_in, d_in, mu_in, _, _, _ = rep.ell_frame(world(loc - eps * nrm), E, np.array([0]))
        assert np.abs(o_out - o_in).max() < 50 * eps
        assert np.abs(d_out).max() < 2 * eps and (d_in == 0).all() and (mu_in == 1).all()
        assert np.abs(mu_out - 1).max() < 1e-3


def test_ellipse_path_at_a_equals_b_is_the_circle_path():
    """Every term the law sums, the shielded command (slide), the straggler routing and the adaptive spread's two
    tests, flown through the ellipse's own code with a = b = the enclosing circle's radius, equal EnclosingCircle's
    to rounding. The shipped code never takes this path at a = b (it short-circuits to the circle), so this is what
    tests the ellipse rules' algebra."""
    rep = _rep()
    import swarm_params as sp
    rng = np.random.default_rng(7)
    city = rep.PolyCity([_convex(rng, stretch=1.0 + 0.1 * i) for i in range(10)])
    over = dict(obstacleShape=1, d_obs=0.35, slideRoundBuildings=True, routeStragglers=True, adaptiveSpread=True)
    mc = rep.Model(dict(sp.resolve_swarm("BlockShapedPatchCityWorld", overrides=over)), dict(sp.resolve_airframe()))
    me = rep.Model(dict(sp.resolve_swarm("BlockShapedPatchCityWorld",
                                         overrides=dict(over, obstacleShape=3, ellipseMaxAspect=1.37))),
                   dict(sp.resolve_airframe()))
    city._ell[1.37] = rep.Ellipses(city.mec_c.copy(), np.tile([1.0, 0.0], (city.n, 1)), city.mec_R.copy(),
                                   city.mec_R.copy(), np.zeros(city.n, bool))
    for trial in range(6):
        b0 = trial % city.n
        P = rng.normal(0, 12 + 4 * trial, (10, 2)) + city.mec_c[b0]
        V = rng.normal(0, 2, (10, 2))
        cand = city.near(P, 110.0)
        cen = P.mean(0)
        cmd = rng.normal(0, 3, 2)
        res = []
        for m in (mc, me):
            miss = rep.aim_miss(m, city, cand, cen, cmd)
            st = rep.spread_target(m, city, cand, cen, miss)
            T = rep.swarm_terms(m, city, cand, P, V, cen, V.mean(0), 6.0, 2.0, cmd, np.tile([0.0, 1.0], (10, 1)),
                                vbar=V.mean(0), miss=miss)
            res.append((miss, st, T))
        assert np.abs(res[0][0] - res[1][0]).max() < 1e-9
        assert res[0][1] == res[1][1]
        for k in ("obs", "cmd", "coh", "damp", "core", "prox"):
            assert np.abs(res[0][2][k] - res[1][2][k]).max() < 1e-9, k


def test_shipped_ellipse_at_cap_one_flies_as_enclosing_circle_bit_for_bit():
    rep = _rep()
    import swarm_params as sp
    rng = np.random.default_rng(11)
    city = rep.PolyCity([_convex(rng) for _ in range(8)])
    over = dict(obstacleShape=1, d_obs=0.35, slideRoundBuildings=True, adaptiveSpread=True)
    models = [rep.Model(dict(sp.resolve_swarm("BlockShapedPatchCityWorld", overrides=o)), dict(sp.resolve_airframe()))
              for o in (over, dict(over, obstacleShape=3, ellipseMaxAspect=1.0))]
    P0 = rng.normal(0, 10, (10, 2)) + city.mec_c[0] + [40.0, 0.0]
    out = []
    for m in models:
        sim = rep.ContactSim(m, city, P0, np.zeros((10, 2)), 3)
        for _ in range(300):
            sim.step(np.array([-1.0, 0.2]), 0.0, 2.0)
        out.append(sim.P.copy())
    assert (out[0] == out[1]).all()


def test_box_cities_refuse_the_ellipse():
    rep = _rep()
    import swarm_params as sp
    m = rep.Model(dict(sp.resolve_swarm("BlockShapedPatchCityWorld", overrides=dict(obstacleShape=3))),
                  dict(sp.resolve_airframe()))
    box = ("b", [0.0, 25.0, 0.0], [[5.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 3.0]])
    city = rep.City([box])
    try:
        rep.swarm_terms(m, city, np.array([0]), np.array([[20.0, 0.0]]), np.zeros((1, 2)), np.zeros(2), np.zeros(2),
                        0.0, 2.0, np.zeros(2), np.zeros((1, 2)))
    except ValueError:
        return
    raise AssertionError("a box city flew the ellipse")
