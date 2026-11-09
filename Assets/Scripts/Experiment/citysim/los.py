"""Line of sight in a city of boxes.

Two tests:
  * 3-D: a segment against oriented boxes, the slab test analyse.segment_blocked uses (`Boxes.blocked`), batched
    over the boxes, and against upright convex prisms (`Prism`, a BlockShaped city's buildings) by the same test on
    their edges' half-planes (tests/test_los.py). Used for Tier C and wherever real altitudes matter -- a camera
    above a roof sees over it.
  * 2-D: footprints in (x, z) (`Footprints`): segment blocked / ray first-hit distance / isovist. The study's
    cities are uniform 50 m slabs and the swarm's ceiling (flown ~51.3 m) is the skyline, so a camera never sees
    over a roof to a hat at 1.8 m (the sight line clears a roof only within ~2.6 % of its length of the camera):
    blockage is a footprint problem, and altitude and gimbal pitch only decide which ground band a camera sees
    (`vertical_band`), applied as a mask.
Recognition is a range (camera.R_ID_M, an angular-size criterion on the shared feed quad).
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera  # noqa: E402

CHUNK = 4096
PRISM_CELLS = 2_000_000     # segments x prisms x edges per batch of the prism test


class Prism:
    """An upright convex prism: blockcity's ConvexPrism, a BlockShaped city's building. It stands in the third slot of
    a building tuple where a box has its half-axis vectors -- (tile, centre (3,), Prism) -- which is how
    analyse.load_city returns one. `poly` is the footprint in (x, z), either winding; the prism spans y0..y1."""

    def __init__(self, poly, y0, y1):
        p = np.asarray(poly, float).reshape(-1, 2)
        if np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) < 0:
            p = p[::-1]                                     # counter-clockwise in (x, z)
        self.poly, self.y0, self.y1 = p, float(y0), float(y1)

    def planes(self):
        """(normals (k, 2), offsets (k,)): the footprint is n . p <= d for every edge."""
        e = np.roll(self.poly, -1, axis=0) - self.poly
        n = np.c_[e[:, 1], -e[:, 0]]                       # outward, the polygon being counter-clockwise
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        return n, np.einsum("kj,kj->k", n, self.poly)


class Boxes:
    """Buildings as analyse.load_city returns them: [(tile, centre (3,), half-axis vectors (3, 3) or a Prism)].
    Boxes take the slab test; prisms the same test against their edges' half-planes and their base and top. The
    box arrays (c, u, L) hold the boxes alone, in order; c2 / R2 hold every building, in list order."""

    def __init__(self, boxes):
        boxes = list(boxes)
        self.n = len(boxes)
        self.prism = np.array([isinstance(b[2], Prism) for b in boxes], bool)
        self.pos = np.zeros(self.n, int)                    # each building's row in its own kind's arrays
        self.pos[~self.prism] = np.arange(int((~self.prism).sum()))
        self.pos[self.prism] = np.arange(int(self.prism.sum()))
        bx = [b for b, p in zip(boxes, self.prism) if not p]
        if bx:
            self.c = np.array([b[1] for b in bx], float)
            ax = np.array([b[2] for b in bx], float)
            self.L = np.linalg.norm(ax, axis=2)
            self.u = ax / np.maximum(self.L, 1e-12)[..., None]
            ext = np.abs(ax).sum(axis=1)
            box_r2 = np.hypot(ext[:, 0], ext[:, 2])        # horizontal circumradius of the AABB
        else:
            self.c, self.u, self.L = np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
            box_r2 = np.zeros(0)
        px = [b[2] for b, p in zip(boxes, self.prism) if p]
        k = max([len(p.poly) for p in px], default=0)
        self.pn = np.zeros((len(px), k, 2))                 # padding edges (n = 0, d = 0) constrain nothing
        self.pd = np.zeros((len(px), k))
        for i, p in enumerate(px):
            n, d = p.planes()
            self.pn[i, :len(d)], self.pd[i, :len(d)] = n, d
        self.py0 = np.array([p.y0 for p in px], float)
        self.py1 = np.array([p.y1 for p in px], float)
        self.c2 = np.array([np.asarray(b[1], float)[[0, 2]] for b in boxes]).reshape(-1, 2)
        self.R2 = np.zeros(self.n)
        self.R2[~self.prism] = box_r2
        self.R2[self.prism] = [float(np.max(np.linalg.norm(p.poly - c2, axis=1)))
                               for p, c2 in zip(px, self.c2[self.prism])]

    def subset(self, idx):
        idx = np.asarray(idx, int)
        b = Boxes([])
        b.n = len(idx)
        b.prism = self.prism[idx]
        bi, pi = self.pos[idx[~b.prism]], self.pos[idx[b.prism]]
        b.pos = np.zeros(b.n, int)
        b.pos[~b.prism], b.pos[b.prism] = np.arange(len(bi)), np.arange(len(pi))
        b.c, b.u, b.L = self.c[bi], self.u[bi], self.L[bi]
        b.pn, b.pd, b.py0, b.py1 = self.pn[pi], self.pd[pi], self.py0[pi], self.py1[pi]
        b.c2, b.R2 = self.c2[idx], self.R2[idx]
        return b

    def near(self, xz, radius):
        """Indices of buildings whose horizontal circumcircle comes within `radius` of the point."""
        if self.n == 0:
            return np.arange(0)
        return np.flatnonzero(np.hypot(*(self.c2 - np.asarray(xz, float)).T) - self.R2 < radius)

    def blocked(self, p0, p1):
        """(N,) True where the segment p0 -> p1 passes through any building: analyse.segment_blocked's slab test,
        batched over the boxes, and its half-plane form over the prisms."""
        p0, p1 = np.atleast_2d(np.asarray(p0, float)), np.atleast_2d(np.asarray(p1, float))
        out = np.zeros(len(p0), bool)
        if len(self.pd):
            out |= self._prisms_blocked(p0, p1)
        if not len(self.c):
            return out
        d = p1 - p0
        for k in range(0, len(p0), CHUNK):
            o = np.einsum("nbk,bjk->nbj", p0[k:k + CHUNK, None, :] - self.c[None], self.u)   # (n, B, 3)
            v = np.einsum("nk,bjk->nbj", d[k:k + CHUNK], self.u)
            with np.errstate(divide="ignore", invalid="ignore"):
                a = (-self.L[None] - o) / v
                b = (self.L[None] - o) / v
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            par = v == 0
            inside = np.abs(o) <= self.L[None]
            lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
            hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
            t0 = np.maximum(lo.max(axis=2), 0.0)
            t1 = np.minimum(hi.min(axis=2), 1.0)
            out[k:k + CHUNK] |= (t0 <= t1).any(axis=1)
        return out

    def _prisms_blocked(self, p0, p1):
        """The prisms' part of `blocked`: p0 + t (p1 - p0) is inside a prism where every edge's n . x <= d holds and
        y0 <= y <= y1, and the segment is blocked where that interval of t meets [0, 1]."""
        out = np.zeros(len(p0), bool)
        d = p1 - p0
        step = max(1, PRISM_CELLS // max(self.pd.size, 1))
        for k in range(0, len(p0), step):
            a, v = p0[k:k + step], d[k:k + step]
            num = self.pd[None] - np.einsum("nk,pjk->npj", a[:, [0, 2]], self.pn)     # (n, P, K)
            den = np.einsum("nk,pjk->npj", v[:, [0, 2]], self.pn)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num / den
                ta, tb = (self.py0[None] - a[:, 1:2]) / v[:, 1:2], (self.py1[None] - a[:, 1:2]) / v[:, 1:2]
            lo = np.where(den < 0, t, -np.inf)
            lo = np.where((den == 0) & (num < 0), np.inf, lo)                         # parallel, outside the edge
            hi = np.where(den > 0, t, np.inf)
            ylo, yhi = np.minimum(ta, tb), np.maximum(ta, tb)
            par = v[:, 1:2] == 0
            inside = (a[:, 1:2] >= self.py0[None]) & (a[:, 1:2] <= self.py1[None])
            ylo = np.where(par, np.where(inside, -np.inf, np.inf), ylo)
            yhi = np.where(par, np.where(inside, np.inf, -np.inf), yhi)
            t0 = np.maximum(np.maximum(lo.max(axis=2), ylo), 0.0)
            t1 = np.minimum(np.minimum(hi.min(axis=2), yhi), 1.0)
            out[k:k + step] = (t0 <= t1).any(axis=1)
        return out


class Footprints:
    """Building footprints in (x, z): centre, two unit axes, two half lengths (oriented rectangles)."""

    def __init__(self, boxes):
        self.n = len(boxes)
        if self.n == 0:
            self.c = np.zeros((0, 2))
            self.u = np.zeros((0, 2, 2))
            self.h = np.zeros((0, 2))
            self.R = np.zeros(0)
            return
        c, ax = [], []
        for b in boxes:
            a = np.asarray(b[2], float)
            vert = np.abs(a[:, 1]) / np.maximum(np.linalg.norm(a, axis=1), 1e-12)
            flat = a[np.argsort(vert)[:2]][:, [0, 2]]
            c.append(np.asarray(b[1], float)[[0, 2]])
            ax.append(flat)
        self.c = np.array(c)
        ax = np.array(ax)                                   # (n, 2, 2) half-axis vectors
        self.h = np.linalg.norm(ax, axis=2)
        self.u = ax / np.maximum(self.h, 1e-12)[..., None]
        self.R = np.linalg.norm(self.h, axis=1)             # circumradius

    @classmethod
    def from_rects(cls, rects):
        """From axis-aligned (cx, cz, hx, hz) rectangles."""
        f = cls([])
        r = np.atleast_2d(np.asarray(rects, float)).reshape(-1, 4)
        f.n = len(r)
        f.c = r[:, :2].copy()
        f.h = r[:, 2:].copy()
        f.u = np.broadcast_to(np.eye(2), (f.n, 2, 2)).copy()
        f.R = np.linalg.norm(f.h, axis=1)
        return f

    def subset(self, idx):
        f = Footprints([])
        f.n = len(idx)
        f.c, f.u, f.h, f.R = self.c[idx], self.u[idx], self.h[idx], self.R[idx]
        return f

    def near(self, xz, radius):
        if self.n == 0:
            return np.arange(0)
        return np.flatnonzero(np.hypot(*(self.c - np.asarray(xz, float)).T) - self.R < radius)

    def _slab(self, o, v):
        """Entry/exit parameters of rays o + t v (o, v: (..., n, 2) in each rectangle's frame)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (-self.h - o) / v
            b = (self.h - o) / v
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        par = v == 0
        inside = np.abs(o) <= self.h
        lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
        hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
        return lo.max(axis=-1), hi.min(axis=-1)

    def blocked(self, p0, p1):
        """(N,) True where the 2-D segment p0 -> p1 crosses any footprint."""
        p0, p1 = np.atleast_2d(np.asarray(p0, float)), np.atleast_2d(np.asarray(p1, float))
        out = np.zeros(len(p0), bool)
        if self.n == 0:
            return out
        d = p1 - p0
        for k in range(0, len(p0), CHUNK):
            o = np.einsum("nbk,bjk->nbj", p0[k:k + CHUNK, None, :] - self.c[None], self.u)
            v = np.einsum("nk,bjk->nbj", d[k:k + CHUNK], self.u)
            t0, t1 = self._slab(o, v)
            out[k:k + CHUNK] = (np.maximum(t0, 0.0) <= np.minimum(t1, 1.0)).any(axis=1)
        return out

    def first_hit(self, origin, angles_deg, rmax):
        """(A,) distance along each heading (deg, clockwise from +z) to the first footprint, inf beyond rmax."""
        th = np.radians(np.asarray(angles_deg, float))
        dirs = np.stack([np.sin(th), np.cos(th)], -1)       # (A, 2)
        out = np.full(len(th), np.inf)
        if self.n == 0:
            return out
        idx = self.near(origin, rmax)
        if len(idx) == 0:
            return out
        f = self.subset(idx)
        o = np.einsum("bk,bjk->bj", np.asarray(origin, float)[None] - f.c, f.u)        # (B, 2)
        v = np.einsum("ak,bjk->abj", dirs, f.u)                                         # (A, B, 2)
        t0, t1 = f._slab(np.broadcast_to(o, v.shape), v)
        hit = (t1 >= 0) & (t0 <= t1)
        t = np.where(hit, np.maximum(t0, 0.0), np.inf).min(axis=1)
        out[:] = np.where(t <= rmax, t, np.inf)
        return out

    def isovist(self, origin, rmax, step_deg=0.5):
        """(angles, distances): the free distance round a point, capped at rmax."""
        ang = np.arange(0.0, 360.0, step_deg)
        d = self.first_hit(origin, ang, rmax)
        return ang, np.minimum(d, rmax)

    def inside(self, pts, margin=0.0):
        """(N,) whether each point lies inside any footprint grown by margin."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self.n == 0:
            return np.zeros(len(pts), bool)
        o = np.einsum("nbk,bjk->nbj", pts[:, None, :] - self.c[None], self.u)
        return (np.abs(o) <= self.h[None] + margin).all(axis=2).any(axis=1)

    def distance(self, pts):
        """(N,) distance from each point to the nearest footprint (0 inside)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self.n == 0:
            return np.full(len(pts), np.inf)
        out = np.empty(len(pts))
        for k in range(0, len(pts), CHUNK):
            o = np.einsum("nbk,bjk->nbj", pts[k:k + CHUNK, None, :] - self.c[None], self.u)
            ex = np.maximum(np.abs(o) - self.h[None], 0.0)
            out[k:k + CHUNK] = np.linalg.norm(ex, axis=2).min(axis=1)
        return out


# ============================================================================= camera masks

def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def bearing_deg(cam_xz, pts_xz):
    """Heading (deg, clockwise from +z) from each camera to each point; broadcasting."""
    d = np.asarray(pts_xz, float) - np.asarray(cam_xz, float)
    return np.degrees(np.arctan2(d[..., 0], d[..., 1]))


def in_hfov(cam_xz, yaw_deg, pts_xz, hfov_deg=camera.HFOV_DEG):
    """Inside the camera's horizontal field of view, and the signed bearing off its axis (deg, + = right)."""
    off = wrap_deg(bearing_deg(cam_xz, pts_xz) - np.asarray(yaw_deg, float))
    return np.abs(off) <= hfov_deg / 2, off


def vertical_band(cam_h, pitch_deg, horiz_dist, target_h=camera.HAT_HEIGHT_M, vfov_deg=camera.VFOV_DEG):
    """Inside the camera's vertical field of view: the elevation to the target within pitch +- vfov/2."""
    elev = np.degrees(np.arctan2(target_h - np.asarray(cam_h, float), np.maximum(horiz_dist, 1e-6)))
    return np.abs(elev - np.asarray(pitch_deg, float)) <= vfov_deg / 2


def slant_range(cam_h, horiz_dist, target_h=camera.HAT_HEIGHT_M):
    return np.hypot(horiz_dist, np.asarray(cam_h, float) - target_h)


def sees(fp, cam_xz, cam_h, yaw_deg, pitch_deg, pts_xz, r_id=camera.R_ID_M, band=True):
    """(N,) whether one camera recognises each target point: range, horizontal FOV, vertical band (optional)
    and a clear 2-D sight line. Returns (visible, bearing off axis, slant range)."""
    pts = np.atleast_2d(np.asarray(pts_xz, float))
    hd = np.hypot(*(pts - np.asarray(cam_xz, float)).T)
    rng = slant_range(cam_h, hd)
    ok, off = in_hfov(cam_xz, yaw_deg, pts)
    ok &= rng <= r_id
    if band:
        ok &= vertical_band(cam_h, pitch_deg, hd)
    if ok.any():
        idx = np.flatnonzero(ok)
        near = fp.near(cam_xz, r_id + 1.0)
        if len(near):
            sub = fp.subset(near)
            ok[idx] = ~sub.blocked(np.broadcast_to(np.asarray(cam_xz, float), (len(idx), 2)), pts[idx])
    return ok, off, rng


def free_sight_probability(fp, origins, r, angles_deg=None):
    """Share of rays from the origins that run r metres without hitting a footprint (Tier B's P(r))."""
    angles = np.arange(0.0, 360.0, 2.0) if angles_deg is None else angles_deg
    hits = np.array([fp.first_hit(o, angles, r) for o in np.atleast_2d(origins)])
    return float(np.isinf(hits).mean())
