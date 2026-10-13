"""Line of sight in a city of boxes.

Two tests:
  * 3-D: a segment against oriented boxes, the slab test analyse.segment_blocked uses (`Boxes.blocked`), batched
    over the boxes. Asserted identical on random segments (tests/test_los.py). Used for Tier C and wherever real
    altitudes matter -- a camera above a roof sees over it.
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


class Boxes:
    """Oriented boxes as analyse.load_city returns them: [(tile, centre (3,), half-axis vectors (3, 3))]."""

    def __init__(self, boxes):
        self.n = len(boxes)
        if self.n == 0:
            self.c, self.u, self.L = np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
            return
        self.c = np.array([b[1] for b in boxes], float)
        ax = np.array([b[2] for b in boxes], float)
        self.L = np.linalg.norm(ax, axis=2)
        self.u = ax / np.maximum(self.L, 1e-12)[..., None]
        ext = np.abs(ax).sum(axis=1)
        self.c2 = self.c[:, [0, 2]]
        self.R2 = np.hypot(ext[:, 0], ext[:, 2])        # horizontal circumradius of the AABB

    def subset(self, idx):
        b = Boxes([])
        b.n = len(idx)
        b.c, b.u, b.L = self.c[idx], self.u[idx], self.L[idx]
        b.c2, b.R2 = self.c2[idx], self.R2[idx]
        return b

    def near(self, xz, radius):
        """Indices of boxes whose horizontal circumcircle comes within `radius` of the point."""
        if self.n == 0:
            return np.arange(0)
        return np.flatnonzero(np.hypot(*(self.c2 - np.asarray(xz, float)).T) - self.R2 < radius)

    def blocked(self, p0, p1):
        """(N,) True where the segment p0 -> p1 passes through any box: analyse.segment_blocked's slab test,
        batched over the boxes."""
        p0, p1 = np.atleast_2d(np.asarray(p0, float)), np.atleast_2d(np.asarray(p1, float))
        out = np.zeros(len(p0), bool)
        if self.n == 0:
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
            out[k:k + CHUNK] = (t0 <= t1).any(axis=1)
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
