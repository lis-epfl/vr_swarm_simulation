"""Task-independent sensing metrics (the plan's section 5.1) on ground points round a pose, in 2-D.

Reference point o: the formation's centroid, moved to the nearest point at least FREE_MARGIN from every footprint
(at a goal the centroid can sit inside the occluder the formation surrounds). Points: a 2 m grid of free ground
within R of o.

Occlusion escape (E). For each displayed feed i, a reference camera at o with feed i's yaw covers a wedge W_i(o).
Candidates C = the points of the union of those wedges whose sight line from o is blocked ("hidden from the
formation centre at the same yaw"). A candidate ESCAPES when some displayed feed j recognises it from its own
position (its wedge, range R, clear sight line). E_any = |escaped| / |C| (with |C| reported);
E_abs = |escaped| / |U W_i(o)| = E_any x occluded share. E_env is the part of E_any whose blocked line from o
crosses the goal's central occluder (the formation seeing behind what it surrounds), E_par the rest (rim parallax).
E_head counts only escapes on a feed the head is turned to (display azimuth within the HMD's half field of view
of the head yaw), averaged over the head-yaw weights.

Reveal (V(t)). The share of the points a pilot can have on screen within t seconds. Swarm: the points recognised
on some feed whose display azimuth is within display.swarm_reveal_deg(t) of the head (straight ahead at t = 0);
t = inf: every feed. Single drone at o facing the body yaw: the points it sees whose bearing is within
display.single_reveal_deg(t) of its heading; t = inf: its full 360-degree isovist.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera, display  # noqa: E402
from citysim.los import wrap_deg  # noqa: E402

FREE_MARGIN = 2.0
GRID_STEP = 2.0
REVEAL_T = (0.0, 1.0, 3.0, 6.0)


def free_point(fp, p, margin=FREE_MARGIN, max_r=60.0):
    """p if it is at least `margin` from every footprint, else the nearest such point on rings round it."""
    p = np.asarray(p, float)
    if fp.distance([p])[0] >= margin:
        return p
    for r in np.arange(1.0, max_r, 1.0):
        ang = np.radians(np.arange(0.0, 360.0, 360.0 / max(12, int(2 * math.pi * r))))
        cand = p + r * np.stack([np.sin(ang), np.cos(ang)], -1)
        ok = fp.distance(cand) >= margin
        if ok.any():
            return cand[np.argmax(ok)]
    return p


def ground_points(fp, centre, R=camera.R_ID_M, step=GRID_STEP):
    g = np.arange(-R, R + 1e-9, step)
    X, Z = np.meshgrid(g, g)
    pts = np.c_[X.ravel(), Z.ravel()] + np.asarray(centre, float)
    pts = pts[np.hypot(*(pts - centre).T) <= R]
    return pts[~fp.inside(pts)]


def visible_from(fp, cam, pts, R=camera.R_ID_M, yaw=None, hfov=camera.HFOV_DEG):
    """(N,) recognised from cam: within R, inside the wedge about yaw (None = any direction), clear line."""
    d = pts - np.asarray(cam, float)
    ok = np.hypot(d[:, 0], d[:, 1]) <= R
    off = np.zeros(len(pts))
    if yaw is not None:
        off = wrap_deg(np.degrees(np.arctan2(d[:, 0], d[:, 1])) - yaw)
        ok &= np.abs(off) <= hfov / 2
    if ok.any():
        idx = np.flatnonzero(ok)
        ok[idx] = ~fp.blocked(np.broadcast_to(np.asarray(cam, float), (len(idx), 2)), pts[idx])
    return ok, off


def occlusion_escape(fp, feeds, o, pts, occluder=None, body_yaw=0.0, head_yaws=(0.0,), head_w=(1.0,)):
    """E_any, E_head, E_abs, E_env, E_par, n_candidates, occluded_share for one set of feeds about point o."""
    k = len(feeds)
    if k == 0 or len(pts) == 0:
        return dict(E_any=math.nan, E_head=math.nan, E_abs=math.nan, E_env=math.nan, E_par=math.nan,
                    n_candidates=0, occluded_share=math.nan)
    d = pts - o
    dist = np.hypot(d[:, 0], d[:, 1])
    brg = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    in_wedge = np.zeros(len(pts), bool)
    for i in range(k):
        in_wedge |= (np.abs(wrap_deg(brg - feeds.yaw[i])) <= camera.HFOV_DEG / 2) & (dist <= camera.R_ID_M)
    blocked_o = np.zeros(len(pts), bool)
    idx = np.flatnonzero(in_wedge)
    if len(idx):
        blocked_o[idx] = fp.blocked(np.broadcast_to(o, (len(idx), 2)), pts[idx])
    C = in_wedge & blocked_o
    vis = np.zeros((k, len(pts)), bool)
    off = np.zeros((k, len(pts)))
    for j in range(k):
        vis[j], off[j] = visible_from(fp, feeds.pos[j], pts, yaw=feeds.yaw[j])
    escaped = C & vis.any(0)
    nC = int(C.sum())
    out = dict(n_candidates=nC, occluded_share=float(C.sum() / max(in_wedge.sum(), 1)))
    out["E_any"] = float(escaped.sum() / nC) if nC else math.nan
    out["E_abs"] = float(escaped.sum() / max(in_wedge.sum(), 1))
    if occluder is not None and nC:
        behind = np.zeros(len(pts), bool)
        ci = np.flatnonzero(escaped)
        if len(ci):
            behind[ci] = occluder.blocked(np.broadcast_to(o, (len(ci), 2)), pts[ci])
        out["E_env"] = float(behind.sum() / nC)
    else:
        out["E_env"] = 0.0 if nC else math.nan
    out["E_par"] = out["E_any"] - out["E_env"] if nC else math.nan
    if nC:
        az = wrap_deg(feeds.yaw[:, None] - body_yaw + off * camera.DISPLAY_SCALE)
        e = 0.0
        for h, w in zip(head_yaws, head_w):
            hv = (vis & (np.abs(wrap_deg(az - h)) <= camera.HMD_HALF_FOV_DEG)).any(0)
            e += w * float((C & hv).sum() / nC)
        out["E_head"] = e / float(np.sum(head_w))
    else:
        out["E_head"] = math.nan
    return out


def reveal(fp, feeds, o, pts, body_yaw=0.0, ts=REVEAL_T):
    """V_swarm(t), V_single(t) for t in ts and t = inf, as shares of the free points within R of o."""
    n = max(len(pts), 1)
    out = {}
    k = len(feeds)
    vis = np.zeros((k, len(pts)), bool)
    off = np.zeros((k, len(pts)))
    for j in range(k):
        vis[j], off[j] = visible_from(fp, feeds.pos[j], pts, yaw=feeds.yaw[j])
    az = wrap_deg(feeds.yaw[:, None] - body_yaw + off * camera.DISPLAY_SCALE) if k else np.zeros((0, len(pts)))
    for t in ts:
        reach = float(display.swarm_reveal_deg(t))
        out[f"V_swarm_{t:g}"] = float((vis & (np.abs(az) <= reach)).any(0).sum() / n) if k else 0.0
    out["V_swarm_inf"] = float(vis.any(0).sum() / n) if k else 0.0
    single, soff = visible_from(fp, o, pts)
    brg = wrap_deg(np.degrees(np.arctan2(pts[:, 0] - o[0], pts[:, 1] - o[1])) - body_yaw)
    for t in ts:
        out[f"V_single_{t:g}"] = float((single & (np.abs(brg) <= float(display.single_reveal_deg(t)))).sum() / n)
    out["V_single_inf"] = float(single.sum() / n)
    return out
