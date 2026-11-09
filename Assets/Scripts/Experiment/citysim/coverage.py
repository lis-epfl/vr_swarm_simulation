"""How built-up a city is, and how far one can see through it: coverage, intensity and the free-sight length.

    python -B -m citysim.coverage --city DiamondCityWorld ScaledCityWorld      # the tile cities
    python -B -m citysim.coverage --layout layouts/<id>.json                    # a generated layout

For buildings scattered as a Boolean model with intensity lam (per m^2), the probability that a sight line of
length r misses all of them is exp(-r / ell), with ell = 1 / (lam * <w_perp>) and <w_perp> the buildings' mean
width projected across the line (Bai, Vaze & Heath's random-shape blockage model). For a convex footprint in a
random orientation <w_perp> = perimeter / pi (Cauchy), so with coverage alpha = lam <A>,
ell = pi <A> / (alpha <P>): at a fixed coverage, bigger buildings mean longer free sight lines. For boxes at
fixed orientations the projected width depends on the line's direction theta, and ell(theta) says whether a
city is isotropic -- a grid of streets is not.

The area is part of the definition, so both are reported for the tile cities: the tile-grid area (every slot of
the city's tile lattice, analyse.fit_lattice, x CityTiles.Pitch^2) and the extent of the buildings' bounding box.
A generated layout reports its arena (the square it was generated in), which is what it was built to.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))   # run as a script: import citysim, not siblings

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import EXPERIMENT_DIR  # noqa: E402
from citysim.camera import R_ID_M  # noqa: E402

PITCH = 90.8304          # CityTiles.Pitch
THETAS_DEG = np.arange(0.0, 180.0, 5.0)


def footprints(boxes):
    """(centres (n, 2), half-axis vectors (n, 2, 2)) in (x, z) of the boxes' horizontal axes.

    `boxes` is [(tile, centre (3,), axes (3, 3))] as analyse.load_city returns them: each box's three
    half-axis vectors; the two closest to horizontal are its footprint."""
    if not boxes:
        return np.zeros((0, 2)), np.zeros((0, 2, 2))
    c = np.array([b[1] for b in boxes], float)[:, [0, 2]]
    out = []
    for _, _, ax in boxes:
        ax = np.asarray(ax, float)
        vert = np.abs(ax[:, 1]) / np.maximum(np.linalg.norm(ax, axis=1), 1e-12)
        h = ax[np.argsort(vert)[:2]][:, [0, 2]]
        out.append(h)
    return c, np.array(out)


def heights(boxes):
    """(base, top) of each box in metres."""
    if not boxes:
        return np.zeros(0), np.zeros(0)
    c = np.array([b[1] for b in boxes], float)
    ext = np.abs(np.array([b[2] for b in boxes], float)).sum(axis=1)
    return c[:, 1] - ext[:, 1], c[:, 1] + ext[:, 1]


def projected_width(axes, theta_deg):
    """(n, len(theta)) width of each footprint across a line heading theta (deg from +z, clockwise).

    The width perpendicular to the line is |a . n| + |b . n| over the footprint's two half-axes, times two."""
    th = np.radians(np.atleast_1d(theta_deg))
    n = np.stack([np.cos(th), -np.sin(th)], -1)            # unit normal to the direction (sin th, cos th)
    return 2 * np.abs(np.einsum("bkj,tj->bkt", axes, n)).sum(axis=1)


def stats(boxes, area_m2, r_m=R_ID_M):
    """Coverage and free-sight statistics of `boxes` over `area_m2`."""
    c, ax = footprints(boxes)
    n = len(c)
    if n == 0:
        return dict(n=0, area_ha=area_m2 / 1e4, alpha=0.0, per_ha=0.0, ell_m=math.inf, p_free=1.0)
    half = np.linalg.norm(ax, axis=2)                       # (n, 2) half side lengths
    A = 4 * half[:, 0] * half[:, 1]
    P = 4 * (half[:, 0] + half[:, 1])
    lam = n / area_m2
    alpha = A.sum() / area_m2
    w_iso = P.mean() / math.pi
    ell = 1.0 / (lam * w_iso)
    w_theta = projected_width(ax, THETAS_DEG).mean(axis=0)
    ell_theta = 1.0 / (lam * w_theta)
    long_, short = 2 * half.max(1), 2 * half.min(1)
    base, top = heights(boxes)
    return dict(
        n=n, area_ha=round(area_m2 / 1e4, 2), alpha=round(alpha, 4), per_ha=round(lam * 1e4, 2),
        mean_area_m2=round(float(A.mean()), 1), mean_perimeter_m=round(float(P.mean()), 1),
        median_long_m=round(float(np.median(long_)), 2), median_short_m=round(float(np.median(short)), 2),
        w_perp_m=round(w_iso, 2), ell_m=round(ell, 1), p_free=round(math.exp(-r_m / ell), 3), r_m=r_m,
        ell_theta_min_m=round(float(ell_theta.min()), 1), ell_theta_max_m=round(float(ell_theta.max()), 1),
        anisotropy=round(float(ell_theta.max() / ell_theta.min()), 3),
        height_median_m=round(float(np.median(top - base)), 2),
        ell_theta_m=[round(float(v), 1) for v in ell_theta])


def tile_grid_area(kerbs):
    """Every slot of the city's tile lattice (analyse.fit_lattice), x Pitch^2, m^2."""
    from citysim import use_bench_modules
    use_bench_modules()
    import analyse
    _, slots = analyse.fit_lattice(kerbs)
    return len(slots) * PITCH ** 2, len(slots)


def box_extent_area(boxes):
    c, ax = footprints(boxes)
    ext = np.abs(ax).sum(axis=1)
    lo, hi = (c - ext).min(0), (c + ext).max(0)
    return float(np.prod(hi - lo)), (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def city_report(city):
    """Statistics of a tile city's city_obstacles_<city>.json on both area definitions."""
    from citysim import use_bench_modules
    use_bench_modules()
    import analyse
    boxes, kerbs = analyse.load_city(city)
    grid_area, slots = tile_grid_area(kerbs)
    ext_area, extent = box_extent_area(boxes)
    return dict(city=city, tile_grid=dict(slots=slots, **stats(boxes, grid_area)),
                box_extent=dict(extent=[round(v, 1) for v in extent], **stats(boxes, ext_area)))


def layout_report(path):
    from citysim import layout as lay
    L = lay.load(path)
    x0, x1, z0, z1 = L.layout["arena"]
    return dict(layout=L.layout.get("id", Path(path).stem),
                arena=dict(**stats(L.boxes_as_tuples(), (x1 - x0) * (z1 - z0))))


def describe(name, d):
    keys = ["n", "area_ha", "alpha", "per_ha", "median_long_m", "median_short_m", "w_perp_m", "ell_m",
            "p_free", "ell_theta_min_m", "ell_theta_max_m", "anisotropy", "height_median_m"]
    return f"  {name:11s} " + "  ".join(f"{k}={d[k]}" for k in keys if k in d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--city", nargs="*", default=[], help="tile cities with a city_obstacles_<city>.json")
    ap.add_argument("--layout", nargs="*", default=[], help="generated layout files")
    ap.add_argument("--json", help="also write the reports here")
    a = ap.parse_args()
    if not a.city and not a.layout:
        a.city = ["DiamondCityWorld", "ScaledCityWorld"]
    out = []
    for city in a.city:
        r = city_report(city)
        out.append(r)
        print(f"{city}:")
        print(describe("tile grid", r["tile_grid"]))
        print(describe("box extent", r["box_extent"]))
    for path in a.layout:
        r = layout_report(path)
        out.append(r)
        print(f"{r['layout']}:")
        print(describe("arena", r["arena"]))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"written {a.json}")


if __name__ == "__main__":
    main()
