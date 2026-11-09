"""Convex polygon footprints: los.Footprints' interface for buildings that are not rectangles (the split study's
block-shaped buildings, citysim.blockcity variant B).

Every consumer of los.Footprints -- transit.Cells / seen_from, clutter.arena_cells / discovery_first / RingTally,
transit.morph_events -- touches only n, c, R, near, subset, blocked, first_hit, inside and distance, so a
ConvexFootprints stands in wherever a city of polygons is scored. R is the circumradius about c (the centroid), the
bounding disc arena_cells and near() use.

A polygon is held as its edges' half-planes (geo.edges_halfplanes: unit outward normals, inside where n . x <= c),
padded to K edges with a null half-plane (n = 0, c = +BIG) that constrains nothing; segments and rays are clipped by
Cyrus-Beck. los.Footprints is left untouched, so S1-S3 score bit for bit as before; on rectangles the two agree to
1e-9 (tests/test_split.py).
"""
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from citysim import geometry as geo  # noqa: E402

CHUNK = 2048
BIG = 1e18


class ConvexFootprints:
    """Building footprints in (x, z) as convex polygons (each (k, 2), any orientation; stored counter-clockwise)."""

    def __init__(self, polys):
        polys = [geo.ccw(np.asarray(p, float)) for p in polys]
        self.n = len(polys)
        self.K = max((len(p) for p in polys), default=3)
        n, K = self.n, self.K
        self.poly = polys
        self.N = np.zeros((n, K, 2))                 # unit outward normals
        self.C = np.full((n, K), BIG)                # offsets: inside where N . x <= C
        self.V = np.zeros((n, K, 2))                 # edge starts
        self.W = np.zeros((n, K, 2))                 # edge ends (padding edges are zero-length, at the last vertex)
        self.c = np.zeros((n, 2))
        self.R = np.zeros(n)
        for i, p in enumerate(polys):
            k = len(p)
            nrm, off = geo.edges_halfplanes(p)
            self.N[i, :k], self.C[i, :k] = nrm, off
            self.V[i, :k], self.W[i, :k] = p, np.roll(p, -1, axis=0)
            self.V[i, k:] = self.W[i, k:] = p[-1]
            self.c[i] = geo.centroid(p)
            self.R[i] = float(np.linalg.norm(p - self.c[i], axis=1).max())

    @classmethod
    def from_rects(cls, rects):
        """From axis-aligned (cx, cz, hx, hz) rectangles (as los.Footprints.from_rects)."""
        r = np.atleast_2d(np.asarray(rects, float)).reshape(-1, 4)
        return cls([np.array([[x - hx, z - hz], [x + hx, z - hz], [x + hx, z + hz], [x - hx, z + hz]])
                    for x, z, hx, hz in r])

    def subset(self, idx):
        idx = np.asarray(idx, int).reshape(-1)
        f = ConvexFootprints([])
        f.n, f.K = len(idx), self.K
        f.poly = [self.poly[i] for i in idx]
        f.N, f.C, f.V, f.W = self.N[idx], self.C[idx], self.V[idx], self.W[idx]
        f.c, f.R = self.c[idx], self.R[idx]
        return f

    def near(self, xz, radius):
        """Indices of footprints whose bounding disc comes within `radius` of the point."""
        if self.n == 0:
            return np.arange(0)
        return np.flatnonzero(np.hypot(*(self.c - np.asarray(xz, float)).T) - self.R < radius)

    # ------------------------------------------------------------------ clipping

    def _clip(self, o, v):
        """Entry / exit parameters of lines o + t v against every polygon: o, v (..., n, 2) -> t0, t1 (..., n), with
        t0 > t1 where the line misses (Cyrus-Beck over the padded half-planes)."""
        num = self.C - np.einsum("...bk,bjk->...bj", o, self.N)         # (..., n, K): C - N . o
        den = np.einsum("...bk,bjk->...bj", v, self.N)                  # (..., n, K): N . v
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        t0 = np.where(den < 0, t, -np.inf).max(axis=-1)
        t1 = np.where(den > 0, t, np.inf).min(axis=-1)
        outside = ((den == 0) & (num < 0)).any(axis=-1)                 # parallel to an edge, beyond it
        t0 = np.where(outside, np.inf, t0)
        return t0, t1

    def blocked(self, p0, p1):
        """(N,) True where the 2-D segment p0 -> p1 crosses any footprint."""
        p0, p1 = np.atleast_2d(np.asarray(p0, float)), np.atleast_2d(np.asarray(p1, float))
        out = np.zeros(len(p0), bool)
        if self.n == 0:
            return out
        d = p1 - p0
        for k in range(0, len(p0), CHUNK):
            o = np.broadcast_to(p0[k:k + CHUNK, None, :], (len(p0[k:k + CHUNK]), self.n, 2))
            v = np.broadcast_to(d[k:k + CHUNK, None, :], o.shape)
            t0, t1 = self._clip(o, v)
            out[k:k + CHUNK] = (np.maximum(t0, 0.0) <= np.minimum(t1, 1.0)).any(axis=1)
        return out

    def first_hit(self, origin, angles_deg, rmax, return_face=False):
        """(A,) distance along each heading (deg, clockwise from +z) to the first footprint, inf beyond rmax. With
        return_face, also (A,) the footprint's index (-1 for none) and (A,) the edge entered (-1 when the origin is
        inside the footprint): edge e runs from poly[e] to poly[e + 1]."""
        th = np.radians(np.asarray(angles_deg, float))
        dirs = np.stack([np.sin(th), np.cos(th)], -1)                   # (A, 2)
        out = np.full(len(th), np.inf)
        bld = np.full(len(th), -1)
        edge = np.full(len(th), -1)
        idx = self.near(origin, rmax) if self.n else np.arange(0)
        if len(idx):
            f = self.subset(idx)
            o = np.broadcast_to(np.asarray(origin, float), (len(th), f.n, 2))
            v = np.broadcast_to(dirs[:, None, :], o.shape)
            num = f.C - np.einsum("abk,bjk->abj", o, f.N)
            den = np.einsum("abk,bjk->abj", v, f.N)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num / den
            entry = np.where(den < 0, t, -np.inf)
            t0 = entry.max(axis=-1)
            t1 = np.where(den > 0, t, np.inf).min(axis=-1)
            t0 = np.where(((den == 0) & (num < 0)).any(axis=-1), np.inf, t0)
            hit = (t1 >= 0) & (t0 <= t1)
            tt = np.where(hit, np.maximum(t0, 0.0), np.inf)              # (A, n)
            j = tt.argmin(axis=1)
            best = tt[np.arange(len(th)), j]
            out = np.where(best <= rmax, best, np.inf)
            if return_face:
                ok = np.isfinite(out)
                bld = np.where(ok, idx[j], -1)
                e = entry[np.arange(len(th)), j].argmax(axis=-1)
                inside = t0[np.arange(len(th)), j] < 0
                edge = np.where(ok & ~inside, e, -1)
        if return_face:
            return out, bld, edge
        return out

    def isovist(self, origin, rmax, step_deg=0.5):
        """(angles, distances): the free distance round a point, capped at rmax."""
        ang = np.arange(0.0, 360.0, step_deg)
        d = self.first_hit(origin, ang, rmax)
        return ang, np.minimum(d, rmax)

    # ------------------------------------------------------------------ points

    def inside(self, pts, margin=0.0):
        """(N,) whether each point lies inside any footprint grown by margin (each edge moved out by margin)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self.n == 0:
            return np.zeros(len(pts), bool)
        out = np.zeros(len(pts), bool)
        for k in range(0, len(pts), CHUNK):
            s = np.einsum("nk,bjk->nbj", pts[k:k + CHUNK], self.N) - self.C[None]
            out[k:k + CHUNK] = (s <= margin).all(axis=2).any(axis=1)
        return out

    def distance(self, pts):
        """(N,) distance from each point to the nearest footprint (0 inside)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self.n == 0:
            return np.full(len(pts), np.inf)
        out = np.empty(len(pts))
        e = self.W - self.V                                              # (n, K, 2)
        ee = np.maximum((e * e).sum(-1), 1e-18)
        for k in range(0, len(pts), CHUNK):
            p = pts[k:k + CHUNK]
            rel = p[:, None, None, :] - self.V[None]                     # (m, n, K, 2)
            t = np.clip((rel * e[None]).sum(-1) / ee[None], 0.0, 1.0)
            d = np.linalg.norm(rel - t[..., None] * e[None], axis=-1).min(axis=2)   # (m, n)
            ins = ((np.einsum("nk,bjk->nbj", p, self.N) - self.C[None]) <= 0).all(axis=2)
            out[k:k + CHUNK] = np.where(ins, 0.0, d).min(axis=1)
        return out

    def closest_points(self, pts, idx=None):
        """(N, n', 2) the closest point of each footprint (idx, default all) to each point, and (N, n') the distance
        (0 when the point is inside, where the closest point is the point itself)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        f = self if idx is None else self.subset(idx)
        e = f.W - f.V
        ee = np.maximum((e * e).sum(-1), 1e-18)
        rel = pts[:, None, None, :] - f.V[None]
        t = np.clip((rel * e[None]).sum(-1) / ee[None], 0.0, 1.0)
        q = f.V[None] + t[..., None] * e[None]                           # (N, n', K, 2)
        dk = np.linalg.norm(pts[:, None, None, :] - q, axis=-1)
        j = dk.argmin(axis=2)
        qq = np.take_along_axis(q, j[..., None, None].repeat(2, -1), axis=2)[:, :, 0]
        d = np.take_along_axis(dk, j[..., None], axis=2)[..., 0]
        ins = ((np.einsum("nk,bjk->nbj", pts, f.N) - f.C[None]) <= 0).all(axis=2)
        qq = np.where(ins[..., None], pts[:, None, :], qq)
        return qq, np.where(ins, 0.0, d)
