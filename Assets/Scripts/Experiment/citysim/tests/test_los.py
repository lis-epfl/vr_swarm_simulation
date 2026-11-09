"""citysim.los's 3-D sight test on prisms (a BlockShaped city's buildings) beside boxes, and the polygon walker loops."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import discovery  # noqa: E402
from citysim.los import Boxes, Prism  # noqa: E402


def _rect(c, hx, hz, ang, y0=0.0, y1=50.0):
    """The same building as an oriented box tuple and as a prism tuple."""
    u = np.array([math.cos(ang), 0.0, math.sin(ang)])
    v = np.array([-math.sin(ang), 0.0, math.cos(ang)])
    ctr = np.array([c[0], (y0 + y1) / 2, c[1]])
    ax = np.array([hx * u, [0.0, (y1 - y0) / 2, 0.0], hz * v])
    corners = [ctr + sx * hx * u + sz * hz * v for sx, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    poly = np.array([[p[0], p[2]] for p in corners])
    return ("b", ctr, ax), ("p", ctr, Prism(poly, y0, y1))


def _segments(rng, n):
    p0 = np.c_[rng.uniform(-60, 60, n), rng.uniform(0, 60, n), rng.uniform(-60, 60, n)]
    p1 = np.c_[rng.uniform(-60, 60, n), rng.uniform(0, 60, n), rng.uniform(-60, 60, n)]
    return p0, p1


def test_a_rectangular_prism_blocks_exactly_what_its_box_does():
    """A prism with a rotated rectangle's footprint gives the slab test's answer on random 3-D segments, from below,
    beside and above its roof, mixed with boxes in one Boxes or on its own."""
    rng = np.random.default_rng(3)
    p0, p1 = _segments(rng, 4000)
    boxes, prisms = zip(*[_rect(rng.uniform(-40, 40, 2), rng.uniform(3, 15), rng.uniform(3, 15),
                                rng.uniform(0, math.pi), 0.0, rng.uniform(20, 50)) for _ in range(6)])
    ref = Boxes(boxes).blocked(p0, p1)
    assert 0.2 < ref.mean() < 0.9
    assert (Boxes(prisms).blocked(p0, p1) == ref).all()
    mixed = [b if k % 2 else p for k, (b, p) in enumerate(zip(boxes, prisms))]
    assert (Boxes(mixed).blocked(p0, p1) == ref).all()
    B = Boxes(mixed)
    idx = B.near((0.0, 0.0), 30.0)
    assert (B.subset(idx).blocked(p0, p1) == Boxes([mixed[i] for i in idx]).blocked(p0, p1)).all()


def test_a_hexagonal_prism_against_dense_sampling():
    """An irregular convex hexagon, either winding: blocked exactly where points along the segment fall inside it."""
    ang = np.sort(np.random.default_rng(5).uniform(0, 2 * math.pi, 6))
    poly = np.c_[10 * np.cos(ang), 7 * np.sin(ang)] + [4.0, -2.0]
    rng = np.random.default_rng(7)
    p0, p1 = _segments(rng, 600)
    s = np.linspace(0.0, 1.0, 4001)
    pts = p0[:, None, :] + s[None, :, None] * (p1 - p0)[:, None, :]
    pr = Prism(poly, 0.0, 30.0)
    n, d = pr.planes()
    inside = ((pts[..., [0, 2]] @ n.T) <= d + 1e-9).all(-1) & (pts[..., 1] >= 0.0) & (pts[..., 1] <= 30.0)
    ref = inside.any(1)
    for P in (pr, Prism(poly[::-1], 0.0, 30.0)):
        got = Boxes([("h", np.array([4.0, 15.0, -2.0]), P)]).blocked(p0, p1)
        assert (got | ~ref).all()                     # every sampled hit is found
        assert (got & ~ref).sum() <= 3                # a hit the 4001 samples can miss grazes a corner


def test_polygon_loops_go_round_the_footprint_at_the_gap():
    """A square's loop: every point `expand` off the footprint, the perimeter that of the square plus the corner arcs,
    and its walkers shared out over it."""
    sq = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])
    pts, w = discovery.polygon_loop_points([sq], n_walkers=16, step=1.0, expand=1.3)
    dx = np.maximum(np.maximum(-pts[:, 0], pts[:, 0] - 20.0), 0.0)
    dz = np.maximum(np.maximum(-pts[:, 1], pts[:, 1] - 10.0), 0.0)
    assert np.allclose(np.hypot(dx, dz), 1.3, atol=0.02)
    assert len(pts) == math.ceil(60 + 2 * math.pi * 1.3) and math.isclose(w.sum(), 16.0)
