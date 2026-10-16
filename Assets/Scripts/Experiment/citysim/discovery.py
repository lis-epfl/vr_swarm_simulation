"""When a goal is first SEEN -- the definition behind the trials' discovery co-primary, used by the simulator (citysim
transit scoring) and by the analysis of recorded runs (analyse.py's discovery family), so that prediction and
measurement mean one thing.

A goal is seen at an instant when at least SEEN_WALKERS of its walkers are EXPECTED on a shown feed. Its walkers circle
its patrolled buildings on WalkerPatrol's loops (walkers.patrol: rounded rectangles 0.5-2.0 m off each facade, eight
per building, spread round the loop), so the walkers expected on the feeds are, per building, the share of its loop
the feeds recognise times the walkers on it -- the loop is sampled every LOOP_STEP_M at the mean offset, each point
carrying its share. A loop point is recognised as the hat is (camera.R_ID_M's test), with a person's range: within
R_GOAL_M slant range, inside a shown camera's horizontal field of view and its gimbal's vertical band, with a clear
sight line. R_GOAL_M is 80 m a priori: a 1.8 m walker then spans ~18 headset pixels on the feed quad, a figure the eye
picks out in a crowd, where the hat at its 40 m spans ~4. 40, 120 and 200 m are the sensitivities. 200 m is what
internal_3 suggests: on the 11 legs where its pilots visibly turned onto a goal for the final approach, first sight
matched that turn best at ~200 m (median lag 0.4 s; at 80 m the turn came 19 s before sight) -- with two pilots who had
flown the same city before, so knowing where goals could be is mixed into it (analyse.py's discovery range check).

    loop_points(ring_rects)            the goal's walker loops as weighted points
    sample_expected(...)               walkers expected at ONE instant of a simulated track, for several ranges and
                                       head yaws at once (the search study checks every sample as it flies)
    track_expected(...)                walkers expected per sample of a simulated camera track (2-D sight lines)
    run_expected(run, boxes, ...)      the same for a recorded run (3-D sight lines, recorded altitude and pitch)
    first_sight(t, expected)           the first instant at or above SEEN_WALKERS
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera, walkers  # noqa: E402
from citysim.los import vertical_band, wrap_deg  # noqa: E402

R_GOAL_M = 80.0
R_GOAL_SENSITIVITY = (40.0, 120.0, 200.0)
SEEN_WALKERS = 1.0
LOOP_STEP_M = 1.0
LOOP_EXPAND_M = walkers.BASE_OFFSET + 0.25 * walkers.MARGIN    # mean of 0.5 + U(0, 3) x 0.5 = 1.25 m off the facade
TARGET_H = camera.HAT_HEIGHT_M


def loop_points(ring_rects, n_walkers=walkers.NUM_WALKERS, step=LOOP_STEP_M, expand=LOOP_EXPAND_M):
    """(points (M, 2), walkers per point (M,)): each patrolled building's loop sampled every `step` m."""
    pts, w = [], []
    for rect in ring_rects:
        lp = walkers.Loop(rect, expand, 0.0, False)
        m = max(int(math.ceil(lp.perimeter / step)), 8)
        s = (np.arange(m) + 0.5) * lp.perimeter / m
        pts.append(lp.local_xz(s) + [lp.cx, lp.cz])
        w.append(np.full(m, n_walkers / m))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(pts), np.concatenate(w)


def first_sight(t, expected, threshold=SEEN_WALKERS):
    """(time, index) of the first sample at or above the threshold, or (nan, -1)."""
    hit = np.flatnonzero(np.asarray(expected) >= threshold - 1e-9)
    return (float(t[hit[0]]), int(hit[0])) if len(hit) else (math.nan, -1)


def horizontal_reach(r_goal, height):
    """How far (m, horizontally) a camera at `height` can be from a walker and still be within r_goal of slant range."""
    return math.sqrt(max(r_goal ** 2 - (height - TARGET_H) ** 2, 0.0))


def goal_reach(pts, hmax):
    """(centre, reach): no camera farther than `reach` from the loops' centre can see any loop point within hmax."""
    centre = pts.mean(0)
    return centre, float(np.max(np.linalg.norm(pts - centre, axis=1))) + hmax


def sample_expected(fp, cams, yaws, body_yaw, height, pitch, pts, w, ranges=(R_GOAL_M,), head_yaws=(0.0,),
                    single=False, hfov=camera.HFOV_DEG, los=True, geom=None):
    """Walkers expected at ONE instant on the shown feeds -- cameras `cams` (K, 2) at yaws (K,) deg -- for every range
    in `ranges` and every head yaw (relative to `body_yaw`) in `head_yaws`: (any (R,), head (R, H)).

    The per-sample core of track_expected, and what the search study calls as it flies. hfov >= 360 makes the cameras
    ideal all-round ones (a camera on the formation's centre); los=False drops the sight-line test (what a camera
    would see with no buildings in the way). geom = goal_reach(pts, max hmax), precomputed by callers that loop."""
    ranges = tuple(ranges)
    head_yaws = np.atleast_1d(np.asarray(head_yaws, float))
    out_any = np.zeros(len(ranges))
    out_head = np.zeros((len(ranges), len(head_yaws)))
    cams = np.atleast_2d(np.asarray(cams, float)).reshape(-1, 2)
    yaws = np.atleast_1d(np.asarray(yaws, float))
    if len(pts) == 0 or len(cams) == 0:
        return out_any, out_head
    hmax = np.array([horizontal_reach(r, height) for r in ranges])
    hmax_max = float(hmax.max())
    centre, reach = goal_reach(pts, hmax_max) if geom is None else geom
    if np.min(np.linalg.norm(cams - centre, axis=1)) > reach:
        return out_any, out_head
    seen_any = np.zeros((len(ranges), len(pts)), bool)
    seen_head = np.zeros((len(ranges), len(head_yaws), len(pts)), bool)
    for cam, cyaw in zip(cams, yaws):
        d = pts - cam
        hd = np.hypot(d[:, 0], d[:, 1])
        off = wrap_deg(np.degrees(np.arctan2(d[:, 0], d[:, 1])) - cyaw)
        ok = (hd <= hmax_max) & vertical_band(height, pitch, hd)
        if hfov < 360.0:
            ok &= np.abs(off) <= hfov / 2
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        vis = np.zeros(len(pts), bool)
        if los and fp.n:
            near = fp.subset(fp.near(cam, hmax_max + 1.0))
            clear = ~near.blocked(np.broadcast_to(cam, (len(idx), 2)), pts[idx]) if near.n else \
                np.ones(len(idx), bool)
            vis[idx[clear]] = True
        else:
            vis[idx] = True
        hv = None
        if not single:
            az = wrap_deg(cyaw - body_yaw + off * camera.DISPLAY_SCALE)
            hv = np.abs(wrap_deg(az[None, :] - head_yaws[:, None])) <= camera.HMD_HALF_FOV_DEG     # (H, M)
        for ri in range(len(ranges)):
            vr = vis & (hd <= hmax[ri])
            seen_any[ri] |= vr
            seen_head[ri] |= vr[None, :] if single else (vr[None, :] & hv)
    return seen_any @ w, seen_head @ w


def track_expected(fp, pos, yaw, valid, body_yaw, height, pitch, pts, w, r_goal=R_GOAL_M, head_yaw=0.0,
                   single=False):
    """Walkers expected per sample of a simulated track, on every shown feed ("any") and on the feeds inside the HMD's
    field of view about `head_yaw` (relative to the body yaw; the single drone's one feed is in front).

    pos (T, K, 2) camera positions, yaw (T, K) deg, valid (T, K) shown, body_yaw (T,) deg; 2-D sight lines (the
    study's cities are 50 m slabs), slant range from `height`, the vertical band of `pitch`. Returns (any (T,),
    head (T,))."""
    T, K = valid.shape
    out_any, out_head = np.zeros(T), np.zeros(T)
    if len(pts) == 0:
        return out_any, out_head
    geom = goal_reach(pts, horizontal_reach(r_goal, height))
    for ti in range(T):
        live = np.flatnonzero(valid[ti])
        if not len(live):
            continue
        a, h = sample_expected(fp, pos[ti, live], yaw[ti, live], body_yaw[ti], height, pitch, pts, w, (r_goal,),
                               (head_yaw,), single=single, geom=geom)
        out_any[ti], out_head[ti] = float(a[0]), float(h[0, 0])
    return out_any, out_head


def run_expected(run, boxes, pts, w, r_goal=R_GOAL_M, every_s=0.5, band=True):
    """Walkers expected per sample of a recorded run on its shown feeds, and on those inside the HMD's field of view
    about the recorded head yaw: (t (S,), any (S,), head (S,), centroid (S, 2)), on the drone-log clock thinned to
    `every_s`. 3-D sight lines (citysim.los.Boxes) from each camera's recorded height, its recorded yaw, and -- when
    `band` and the run logged it -- the vertical band of the recorded gimbal pitch (replay.hat_bases' cameras)."""
    from citysim import replay
    from citysim.los import Boxes
    dr = run.drones
    T = run.ct
    ids = sorted(dr.droneId.unique())
    piv = {k: dr.pivot_table(index="t", columns="droneId", values=k).reindex(T)[ids].values
           for k in ("gtX", "gtY", "gtZ", "yawDeg", "alive")}
    A = np.nan_to_num(piv["alive"]) > 0.5
    shown = replay.shown_mask(run) & A
    h = run.head
    hi = np.clip(np.searchsorted(h.t.values, T), 0, len(h) - 1)
    body = h.bodyYaw.values[hi]
    head_rel = wrap_deg(h.headYaw.values[hi] - body - 90.0)
    pitch = h.gimbalPitch.values[hi] if (band and "gimbalPitch" in h) else np.full(len(T), np.nan)
    single = run.condition != "Swarm" or len(ids) == 1
    keep = np.r_[0, np.flatnonzero(np.diff(np.floor(T / every_s)) > 0) + 1]
    B = Boxes(boxes)
    centre = pts.mean(0) if len(pts) else np.zeros(2)
    reach = (float(np.max(np.linalg.norm(pts - centre, axis=1))) if len(pts) else 0.0) + r_goal
    near = B.subset(B.near(centre, reach + 10.0))
    out_any, out_head = np.zeros(len(keep)), np.zeros(len(keep))
    cen = np.zeros((len(keep), 2))
    for s, k in enumerate(keep):
        live = np.flatnonzero(shown[k])
        a = np.flatnonzero(A[k])
        cen[s] = (np.nanmean(piv["gtX"][k, a]), np.nanmean(piv["gtZ"][k, a])) if len(a) else (np.nan, np.nan)
        if not len(live) or not len(pts):
            continue
        cams = np.stack([piv["gtX"][k, live], piv["gtY"][k, live], piv["gtZ"][k, live]], -1)
        if np.min(np.hypot(cams[:, 0] - centre[0], cams[:, 2] - centre[1])) > reach:
            continue
        seen_any = np.zeros(len(pts), bool)
        seen_head = np.zeros(len(pts), bool)
        for c, d_i in zip(cams, live):
            rel = np.c_[pts[:, 0] - c[0], np.full(len(pts), TARGET_H - c[1]), pts[:, 1] - c[2]]
            dist = np.linalg.norm(rel, axis=1)
            hd = np.hypot(rel[:, 0], rel[:, 2])
            off = wrap_deg(np.degrees(np.arctan2(rel[:, 0], rel[:, 2])) - piv["yawDeg"][k, d_i])
            ok = (dist <= r_goal) & (np.abs(off) <= camera.HFOV_DEG / 2)
            if np.isfinite(pitch[k]):
                ok &= vertical_band(c[1], pitch[k], hd)
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            tgt = np.c_[pts[idx, 0], np.full(len(idx), TARGET_H), pts[idx, 1]]
            clear = ~near.blocked(np.broadcast_to(c, (len(idx), 3)), tgt) if near.n else np.ones(len(idx), bool)
            vis = np.zeros(len(pts), bool)
            vis[idx[clear]] = True
            seen_any |= vis
            if single:
                seen_head |= vis
            else:
                az = wrap_deg(piv["yawDeg"][k, d_i] - body[k] + off * camera.DISPLAY_SCALE)
                seen_head |= vis & (np.abs(wrap_deg(az - head_rel[k])) <= camera.HMD_HALF_FOV_DEG)
        out_any[s] = float(w[seen_any].sum())
        out_head[s] = float(w[seen_head].sum())
    return T[keep], out_any, out_head, cen
