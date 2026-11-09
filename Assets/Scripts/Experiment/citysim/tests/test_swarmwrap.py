"""S11 Wrap Hold: the wrap test, the two hold rules in the replica's spread target, and the phases its costs are placed
in (citysim.swarmwrap)."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)


def _rep():
    import os
    os.environ.setdefault("SWARM_REPLICA_ALLOW_STALE", "1")
    from citysim import use_bench_modules
    use_bench_modules()
    import swarm_replica as rep
    return rep


def _hull_contains(P, q):
    """An independent containment test: q is inside the convex hull of P when it is a convex combination of P's
    points, checked by the hull's edges (monotone chain)."""
    pts = sorted(map(tuple, P))

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and np.cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out
    h = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    for a, b in zip(h, np.roll(h, -1, axis=0)):
        if np.cross(b - a, np.asarray(q) - a) < 0:
            return False
    return True


def test_inside_formation_is_convex_hull_containment():
    """No 180 degree gap in the drones' bearings round a point exactly when the point is inside their hull: random
    formations, random points, against an independent hull test; fewer than three drones enclose nothing."""
    rep = _rep()
    rng = np.random.default_rng(11)
    agree = 0
    for _ in range(300):
        P = rng.normal(0, 10, (int(rng.integers(3, 12)), 2))
        q = rng.normal(0, 12, 2)
        got = bool(rep.inside_formation(P, q[None])[0])
        assert got == _hull_contains(P, q)
        agree += 1
    assert agree == 300
    assert not rep.inside_formation(np.zeros((2, 2)), np.zeros((1, 2)))[0]


class _City:
    """A city of one round building, for the replica's spread rule: what aim_circle and ell_rules read."""

    def __init__(self, c, R):
        self.c = self.mec_c = np.array([c], float)
        self.R = self.mec_R = np.array([R], float)
        self.n = 1


def _model(rep, rule):
    from types import SimpleNamespace
    return SimpleNamespace(wrap_rule=rule, max_speed=9.31, obstacle_shape=rep.ENCLOSING, aim_fraction=0.8,
                           look_ahead=4.0, S=10.0, street_factor=0.85)


def test_hold_rules_in_the_spread_target():
    """A ring of drones round a building whose centre sits 0.9 of its radius off the centroid, the stick held away
    from it: S7's rule (Off) narrows (the centroid is beyond 0.8 R and the stick's line misses), Hold keeps the full
    spread, HoldUntilLeaving narrows while steering away and holds when the stick points back at it."""
    rep = _rep()
    R = 10.0
    ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    P = 16.0 * np.c_[np.cos(ang), np.sin(ang)]
    city = _City([0.9 * R, 0.0], R)
    cen = P.mean(0)
    cand = np.array([0])
    away, toward = np.array([-9.31, 0.0]), np.array([9.31, 0.0])
    off, hold, leave = (_model(rep, r) for r in (rep.WRAP_OFF, rep.WRAP_HOLD, rep.WRAP_UNTIL_LEAVING))
    miss = rep.aim_miss(off, city, cand, cen, away)
    assert miss[0] > 0.8
    assert rep.spread_target(off, city, cand, cen, miss, P, away) == 0.85
    assert rep.spread_target(hold, city, cand, cen, miss, P, away) == 1.0
    assert rep.spread_target(leave, city, cand, cen, miss, P, away) == 0.85
    assert rep.spread_target(leave, city, cand, cen, miss, P, np.zeros(2)) == 1.0
    assert rep.wrap_holds(leave, city, cand, P, cen, toward)[0]
    try:
        rep.spread_target(hold, city, cand, cen, miss)
    except ValueError:
        pass
    else:
        raise AssertionError("a wrap rule without the drones' positions must refuse")


def test_phases_and_measures():
    """Wrapped, then the EXIT_S after it, then elsewhere; events are placed in the phase they fall in."""
    from citysim import swarmwrap as W
    t = np.arange(0, 30, 0.1)
    wrap = (t >= 5) & (t < 10)
    ph = W.phases(t, wrap)
    assert (ph[(t >= 5) & (t < 10)] == 0).all()
    assert (ph[(t >= 10) & (t <= 19.85)] == 1).all()       # the last wrapped sample is 9.9 s; 19.9 s is the edge
    assert (ph[t >= 20.05] == 2).all() and (ph[t < 5] == 2).all()
    sig = np.where(wrap, 1.0, 0.85)
    m = W.wrap_measures(t, sig, wrap, np.full(len(t), 15.0), [7.0, 12.0, 25.0], [12.5])
    assert math.isclose(m["wrap_s"], 5.0, abs_tol=0.11) and m["wrap_full"] == 1.0
    assert (m["loss_wrap"], m["loss_exit"], m["loss_other"]) == (1, 1, 1)
    assert m["contact_exit"] == 1 and m["loss_events"] == 3
