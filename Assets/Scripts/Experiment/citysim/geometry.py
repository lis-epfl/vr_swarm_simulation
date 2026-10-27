"""Plane geometry for the layout generator: convex polygons, axis-aligned rectangles, blue-noise points and a
bounded Voronoi diagram. numpy + scipy.spatial only (no shapely in the stitching env).

Conventions: points are (x, z) in metres, Unity world axes; polygons are (n, 2) arrays, counter-clockwise in
the (x, z) plane (positive shoelace area), not closed (the last vertex is not repeated). A rectangle is
(cx, cz, hx, hz): centre and half-extents along x and z.
"""
import math

import numpy as np

EPS = 1e-9


# ============================================================================= convex polygons

def area(poly):
    """Signed shoelace area; positive for counter-clockwise."""
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    x, z = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def ccw(poly):
    p = np.asarray(poly, float)
    return p if area(p) >= 0 else p[::-1].copy()


def centroid(poly):
    p = np.asarray(poly, float)
    a = area(p)
    if abs(a) < EPS:
        return p.mean(0)
    x, z = p[:, 0], p[:, 1]
    xn, zn = np.roll(x, -1), np.roll(z, -1)
    cr = x * zn - xn * z
    return np.array([((x + xn) * cr).sum(), ((z + zn) * cr).sum()]) / (6 * a)


def perimeter(poly):
    p = np.asarray(poly, float)
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def clip_halfplane(poly, n, c):
    """The part of convex `poly` where n . x <= c (Sutherland-Hodgman against one edge)."""
    p = np.asarray(poly, float)
    if len(p) == 0:
        return p
    d = p @ np.asarray(n, float) - c
    out = []
    for i in range(len(p)):
        j = (i + 1) % len(p)
        a, b, da, db = p[i], p[j], d[i], d[j]
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out).reshape(-1, 2)


def edges_halfplanes(poly):
    """(normals (n, 2), offsets (n,)) of a CCW convex polygon's edges: inside is n . x <= c."""
    p = ccw(poly)
    e = np.roll(p, -1, axis=0) - p
    nrm = np.stack([e[:, 1], -e[:, 0]], 1)                 # outward for CCW
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), EPS)
    return nrm, (nrm * p).sum(1)


def clip_convex(poly, clipper):
    """`poly` intersected with the convex polygon `clipper`."""
    out = np.asarray(poly, float)
    for n, c in zip(*edges_halfplanes(clipper)):
        out = clip_halfplane(out, n, c)
        if len(out) == 0:
            break
    return out


def offset_convex(poly, d):
    """Convex polygon shrunk inward by d (every edge moved in by d); empty (0, 2) when nothing is left."""
    p = ccw(poly)
    nrm, c = edges_halfplanes(p)
    out = p.copy()
    for n, ci in zip(nrm, c - d):
        out = clip_halfplane(out, n, ci)
        if len(out) < 3:
            return np.zeros((0, 2))
    return dedupe(out)


def dedupe(poly, tol=1e-7):
    p = np.asarray(poly, float)
    if len(p) == 0:
        return p
    keep = [0]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - p[keep[-1]]) > tol:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(p[keep[-1]] - p[keep[0]]) <= tol:
        keep.pop()
    return p[keep]


def contains(poly, pts, margin=0.0):
    """(m,) whether each point is inside the convex polygon, at least `margin` from every edge."""
    nrm, c = edges_halfplanes(poly)
    pts = np.atleast_2d(np.asarray(pts, float))
    return ((pts @ nrm.T) <= (c - margin)[None, :] + 1e-9).all(axis=1)


def inradius_at(poly, pt):
    """Distance from an interior point to the nearest edge of a convex polygon (negative outside)."""
    nrm, c = edges_halfplanes(poly)
    return float((c - nrm @ np.asarray(pt, float)).min())


def point_segment_distance(pts, a, b):
    pts = np.atleast_2d(np.asarray(pts, float))
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), EPS), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


# ============================================================================= rectangles

def rect_corners(r):
    cx, cz, hx, hz = r
    return np.array([[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]])


def rect_gap(r, others):
    """(m,) separation between rectangle r and each of `others` (m, 4): the larger of the x and z gaps,
    negative when they overlap. Two rectangles are at least g apart (Chebyshev) iff rect_gap >= g."""
    o = np.atleast_2d(np.asarray(others, float))
    if len(o) == 0:
        return np.zeros(0)
    gx = np.abs(o[:, 0] - r[0]) - (o[:, 2] + r[2])
    gz = np.abs(o[:, 1] - r[1]) - (o[:, 3] + r[3])
    return np.maximum(gx, gz)


def rect_point_distance(rects, pts):
    """(m, k) Euclidean distance from each of k points to each of m rectangles (0 inside)."""
    r = np.atleast_2d(np.asarray(rects, float))
    p = np.atleast_2d(np.asarray(pts, float))
    dx = np.maximum(np.abs(p[None, :, 0] - r[:, None, 0]) - r[:, None, 2], 0.0)
    dz = np.maximum(np.abs(p[None, :, 1] - r[:, None, 1]) - r[:, None, 3], 0.0)
    return np.hypot(dx, dz)


def rect_area(rects):
    r = np.atleast_2d(np.asarray(rects, float))
    return 4 * r[:, 2] * r[:, 3]


# ============================================================================= sampling

def rsa_points(rng, bounds, n, rmin, max_tries=200000):
    """Random sequential adsorption: up to n points uniform in bounds (x0, x1, z0, z1), no two closer than
    rmin. A Poisson-disc (blue-noise) set stopped at a fixed count; returns (k, 2), k <= n."""
    x0, x1, z0, z1 = bounds
    cell = rmin / math.sqrt(2)
    nx, nz = int(math.ceil((x1 - x0) / cell)) + 1, int(math.ceil((z1 - z0) / cell)) + 1
    grid = -np.ones((nx, nz), int)
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = np.array([rng.uniform(x0, x1), rng.uniform(z0, z1)])
        gi, gj = int((p[0] - x0) / cell), int((p[1] - z0) / cell)
        ok = True
        for i in range(max(gi - 2, 0), min(gi + 3, nx)):
            for j in range(max(gj - 2, 0), min(gj + 3, nz)):
                k = grid[i, j]
                if k >= 0 and np.hypot(*(pts[k] - p)) < rmin:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[gi, gj] = len(pts)
            pts.append(p)
    return np.array(pts).reshape(-1, 2)


def rsa_extend(rng, bounds, pts, n, rmin, keep_out=(), max_tries=20000):
    """rsa_points carried on from `pts` (kept, in order) up to n points in all: new points uniform in bounds, none
    within rmin of another and none within d of a keep-out centre c, for every (c, d) in keep_out. Returns (k, 2)."""
    x0, x1, z0, z1 = bounds
    have = np.asarray(pts, float).reshape(-1, 2)
    out = [p for p in have]
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        p = np.array([rng.uniform(x0, x1), rng.uniform(z0, z1)])
        if any(math.hypot(p[0] - c[0], p[1] - c[1]) < d for c, d in keep_out):
            continue
        if len(have) and np.hypot(have[:, 0] - p[0], have[:, 1] - p[1]).min() < rmin:
            continue
        out.append(p)
        have = np.vstack([have, p])
    return np.array(out).reshape(-1, 2)


# ============================================================================= Voronoi

def voronoi_cells(seeds, bounds):
    """The Voronoi diagram of `seeds` clipped to the box `bounds` (x0, x1, z0, z1).

    Returns (cells, edges): cells[i] is seed i's CCW polygon; edges is [(i, j, a, b)] for every edge, j = -1
    on the box boundary, a and b its end points. Bounded by mirroring the seeds across the four sides, which
    makes every cell of an original seed finite and puts its outer edges exactly on the box."""
    from scipy.spatial import Voronoi

    s = np.asarray(seeds, float)
    x0, x1, z0, z1 = bounds
    n = len(s)
    mirrored = [s,
                np.c_[2 * x0 - s[:, 0], s[:, 1]], np.c_[2 * x1 - s[:, 0], s[:, 1]],
                np.c_[s[:, 0], 2 * z0 - s[:, 1]], np.c_[s[:, 0], 2 * z1 - s[:, 1]]]
    vor = Voronoi(np.vstack(mirrored))
    box = np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]])
    cells = []
    for i in range(n):
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or len(reg) < 3:
            raise ValueError(f"unbounded Voronoi cell for seed {i}")
        v = vor.vertices[reg]
        ang = np.arctan2(v[:, 1] - s[i, 1], v[:, 0] - s[i, 0])
        cells.append(dedupe(clip_convex(ccw(v[np.argsort(ang)]), box)))
    edges = []
    for (a, b), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv:
            continue
        ia, ib = (a, b) if a < b else (b, a)
        if ia >= n:
            continue
        pa, pb = vor.vertices[rv[0]], vor.vertices[rv[1]]
        if ib < n:
            seg = clip_segment_box(pa, pb, bounds)
            if seg is not None:
                edges.append((ia, ib, seg[0], seg[1]))
        else:  # between a seed and its mirror: a boundary edge of the box
            seg = clip_segment_box(pa, pb, bounds)
            if seg is not None and np.linalg.norm(seg[1] - seg[0]) > 1e-6:
                edges.append((ia, -1, seg[0], seg[1]))
    return cells, edges


def clip_segment_box(a, b, bounds):
    """Liang-Barsky: the part of segment ab inside the box, or None."""
    x0, x1, z0, z1 = bounds
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - z0), (d[1], z1 - a[1])):
        if abs(p) < EPS:
            if q < -1e-9:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1 + 1e-12:
            return None
    return a + t0 * d, a + t1 * d
