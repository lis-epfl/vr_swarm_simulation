"""Tier A: what can be said with a pencil. The Boolean (random-shape) blockage model for the free-sight length, the
share of a recognition disc a single free camera sees, and the walker loop's lap and hidden arc by building size.

For buildings with intensity lam and mean projected width <w_perp> = perimeter / pi, a sight line of length r is
free with probability exp(-r / ell), ell = 1 / (lam <w_perp>) = pi <A> / (alpha <P>). Averaged over the recognition
disc of radius R (area-weighted), a 360-degree camera at a free point sees

    V(R) = (2 / R^2) int_0^R r exp(-r / ell) dr = 2 (ell / R)^2 [1 - exp(-R / ell) (1 + R / ell)]

of it: ~1 when ell >> R (range-limited, today's cities, ell / R ~ 6), ~2 (ell / R)^2 when ell << R (occlusion-limited).
Exact for independently placed buildings as alpha -> 0; a hard-core or block layout departs from it as alpha grows,
which Tier B measures directly (citysim.los.free_sight_probability on scatter-mode layouts).
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera, walkers  # noqa: E402


def ell(alpha, w, d):
    """Free-sight length (m) for coverage alpha of w x d footprints (random or 0/90-degree orientations)."""
    if alpha <= 0:
        return math.inf
    return math.pi * w * d / (alpha * 2 * (w + d))


def p_free(r, ell_m):
    return math.exp(-r / ell_m) if math.isfinite(ell_m) else 1.0


def disc_visible(R, ell_m):
    """Area-weighted share of a disc of radius R a free 360-degree camera at its centre sees."""
    if not math.isfinite(ell_m):
        return 1.0
    x = R / ell_m
    return 2.0 / x ** 2 * (1.0 - math.exp(-x) * (1.0 + x))


def regime(ell_m, R=camera.R_ID_M):
    q = ell_m / R
    return "range-limited" if q > 3 else ("occlusion-limited" if q < 1 else "transitional")


def lap_s(w, d, expand=1.25, speed=walkers.SPEED):
    """A walker's lap (s) round a w x d footprint at its mean offset (0.5-2.0 m, mean 1.25)."""
    lp = walkers.Loop((0.0, 0.0, w / 2, d / 2), expand, 0.0, False, speed)
    return lp.lap_s


def hidden_arc(w, d, dist, expand=1.25, n=720, views=36):
    """Share of a walker's loop round a w x d footprint hidden BY THE BUILDING ITSELF from a viewer `dist` metres
    from its centre, averaged over the viewer's direction. Tends to ~1/2 far away; more up close."""
    from citysim.los import Footprints
    lp = walkers.Loop((0.0, 0.0, w / 2, d / 2), expand, 0.0, False)
    s = np.linspace(0, lp.perimeter, n, endpoint=False)
    pts = lp.local_xz(s)
    fp = Footprints.from_rects([(0.0, 0.0, w / 2, d / 2)])
    hid = []
    for a in np.linspace(0, 2 * math.pi, views, endpoint=False):
        v = dist * np.array([math.sin(a), math.cos(a)])
        hid.append(fp.blocked(np.broadcast_to(v, pts.shape), pts).mean())
    return float(np.mean(hid))


def table(design_w, design_d, sizes=(1.0, 2.0, 4.0), alphas=(0.024, 0.05, 0.1, 0.15, 0.2, 0.3), R=camera.R_ID_M):
    """Rows of the Tier A overview: ell, P(free R), V(R) and the regime per (size, alpha); lap and hidden arc per
    size (viewer at 20 m)."""
    rows = []
    for s in sizes:
        w, d = design_w * s, design_d * s
        for a in alphas:
            e = ell(a, w, d)
            rows.append(dict(size=s, w=round(w, 2), d=round(d, 2), alpha=a, ell_m=round(e, 1),
                             ell_over_R=round(e / R, 2), p_free_R=round(p_free(R, e), 3),
                             disc_visible=round(disc_visible(R, e), 3), regime=regime(e, R),
                             lap_s=round(lap_s(w, d), 1), hidden_arc_20m=round(hidden_arc(w, d, 20.0), 3)))
    return rows
