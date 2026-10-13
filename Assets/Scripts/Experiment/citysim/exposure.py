"""Exposure of goal pedestrians to an interface's cameras: the engine every tier scores with.

Inputs are camera tracks -- per sample t, the shown feeds' positions, yaws and heights, the gimbal pitch and the
pilot's body yaw -- and the goal's walkers (walkers.GoalWalkers). A walker is RECOGNISED by a feed at a sample when
it is within camera.R_ID_M (slant range), inside the horizontal field of view and (band=True) the vertical band of
the gimbal pitch, with a clear 2-D sight line (los). From that:

  E_any        share of walker-samples recognised on some displayed feed (the target is a random walker, so the
               mean over all of them is the expected exposure of the hat)
  E_head       the same on a feed the pilot's head is turned to: the walker's display azimuth (feed yaw - body yaw
               + bearing x DISPLAY_SCALE) within the HMD's +-50 deg of the head yaw, averaged over the recorded
               head-yaw distribution (operator_band.head_weights). The single drone's one feed is in front: E_head
               = E_any.
  first_sight  mean time to a walker's first recognised sample (censored at the window), and the share ever seen
  episode_s    mean length of a recognised run (gaps of one sample joined)
  distractors  mean walkers recognised per sample (the 63 others crowd the search) and per displayed feed
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera  # noqa: E402
from citysim.los import vertical_band, wrap_deg  # noqa: E402


@dataclass
class Track:
    """Camera samples: pos (T, K, 2), yaw (T, K) deg, h (T, K) m, valid (T, K) (a feed shown at t), pitch (T,) deg,
    body_yaw (T,) deg, dt (s). K is the most feeds shown at any sample; unused slots are invalid."""
    pos: np.ndarray
    yaw: np.ndarray
    h: np.ndarray
    valid: np.ndarray
    pitch: np.ndarray
    body_yaw: np.ndarray
    dt: float
    single: bool = False

    @property
    def T(self):
        return self.pos.shape[0]


def static_track(feeds, T, dt, height, pitch, body_yaw, single=False):
    """A track holding one set of feeds (display.Feeds) for T samples."""
    K = max(len(feeds), 1)
    pos = np.zeros((T, K, 2))
    yaw = np.zeros((T, K))
    valid = np.zeros((T, K), bool)
    if len(feeds):
        pos[:] = feeds.pos[None]
        yaw[:] = feeds.yaw[None]
        valid[:] = True
    return Track(pos, yaw, np.full((T, K), float(height)), valid, np.full(T, float(pitch)),
                 np.full(T, float(body_yaw)), dt, single)


def recognised_2d(fp, track, pts, r_id=camera.R_ID_M):
    """The altitude-free part of recognition: (vis (W, T, K), off (W, T, K) deg, horizontal distance (W, T, K))
    with vis = shown & in the horizontal FOV & horizontally within r_id & a clear sight line. Slant range and the
    vertical band (with_band) only ever remove from it, so one sight-line pass serves every altitude and pitch."""
    d = pts[:, :, None, :] - track.pos[None]                                     # (W, T, K, 2)
    hd = np.hypot(d[..., 0], d[..., 1])
    off = wrap_deg(np.degrees(np.arctan2(d[..., 0], d[..., 1])) - track.yaw[None])
    ok = track.valid[None] & (hd <= r_id) & (np.abs(off) <= camera.HFOV_DEG / 2)
    if ok.any():
        w, t, k = np.nonzero(ok)
        ok[w, t, k] = ~fp.blocked(track.pos[t, k], pts[w, t])
    return ok, off, hd


def with_band(vis2d, hd, height, pitch, r_id=camera.R_ID_M, band=True):
    """Recognition at camera height `height` and gimbal `pitch` (scalars or broadcastable to (W, T, K))."""
    ok = vis2d & (np.hypot(hd, np.asarray(height, float) - camera.HAT_HEIGHT_M) <= r_id)
    if band:
        ok &= vertical_band(height, pitch, hd)
    return ok


def recognised_3d(boxes3, track, pts, r_id=camera.R_ID_M, band=True, target_h=camera.HAT_HEIGHT_M):
    """Tier C's recognition from real camera heights (track.h, per feed and sample): (vis (W, T, K), off (W, T, K)).
    Range is slant range, the vertical band is the gimbal's, and the sight line is the 3-D slab test against
    los.Boxes -- so a camera above a roof sees over it, which the footprint test cannot express."""
    d = pts[:, :, None, :] - track.pos[None]                                     # (W, T, K, 2)
    hd = np.hypot(d[..., 0], d[..., 1])
    off = wrap_deg(np.degrees(np.arctan2(d[..., 0], d[..., 1])) - track.yaw[None])
    h = np.broadcast_to(track.h[None], hd.shape)
    ok = track.valid[None] & (np.hypot(hd, h - target_h) <= r_id) & (np.abs(off) <= camera.HFOV_DEG / 2)
    if band:
        ok &= vertical_band(h, track.pitch[None, :, None], hd, target_h=target_h)
    if ok.any():
        w, t, k = np.nonzero(ok)
        cam = np.stack([track.pos[t, k, 0], track.h[t, k], track.pos[t, k, 1]], -1)
        tgt = np.stack([pts[w, t, 0], np.full(len(w), target_h), pts[w, t, 1]], -1)
        ok[w, t, k] = ~boxes3.blocked(cam, tgt)
    return ok, off


def recognised(fp, track, pts, r_id=camera.R_ID_M, band=True):
    """(vis (W, T, K) bool, off (W, T, K) deg) for walker positions pts (W, T, 2)."""
    vis2d, off, hd = recognised_2d(fp, track, pts, r_id)
    return with_band(vis2d, hd, track.h[None], track.pitch[None, :, None], r_id, band), off


def summarise(vis, off, track, head_yaws=None, head_w=None):
    """The metrics of the module docstring from recognised() output."""
    W, T, K = vis.shape
    any_t = vis.any(axis=2)                                                       # (W, T)
    out = dict(E_any=float(any_t.mean()))
    if track.single or head_yaws is None:
        out["E_head"] = out["E_any"]
    else:
        az = wrap_deg(track.yaw[None] - track.body_yaw[None, :, None] + off * camera.DISPLAY_SCALE)   # (W, T, K)
        e = 0.0
        for h, w in zip(head_yaws, head_w):
            if w <= 0:
                continue
            e += w * float((vis & (np.abs(wrap_deg(az - h)) <= camera.HMD_HALF_FOV_DEG)).any(axis=2).mean())
        out["E_head"] = e / float(np.sum(head_w))
    seen = any_t.any(axis=1)
    first = np.where(seen, np.argmax(any_t, axis=1), T) * track.dt
    out["first_sight_s"] = float(first.mean())
    out["seen_frac"] = float(seen.mean())
    runs = []
    for w in range(W):
        x = any_t[w].astype(np.int8)
        if not x.any():
            continue
        # join single-sample gaps
        y = x.copy()
        y[1:-1] |= (x[:-2] & x[2:]).astype(np.int8)
        dd = np.diff(np.r_[0, y, 0])
        runs += list((np.flatnonzero(dd == -1) - np.flatnonzero(dd == 1)) * track.dt)
    out["episode_s"] = float(np.mean(runs)) if runs else 0.0
    shown = track.valid.sum(axis=1)                                               # (T,)
    out["walkers_visible"] = float(any_t.sum(axis=0).mean())
    out["walkers_per_feed"] = float(vis.sum(axis=0).sum(axis=1).mean() / max(shown.mean(), 1e-9))
    out["feeds_shown"] = float(shown.mean())
    return out


def exposure(fp, track, walkers, t, head_yaws=None, head_w=None, band=True, r_id=camera.R_ID_M):
    """Recognition of every walker of a goal over sample times t, summarised."""
    pts = walkers.positions(t)                                                    # (W, T, 2)
    vis, off = recognised(fp, track, pts, r_id, band)
    out = summarise(vis, off, track, head_yaws, head_w)
    out["vis"] = vis
    return out
