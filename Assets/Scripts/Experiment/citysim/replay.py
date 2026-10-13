"""Hat visibility from a RECORDED run, on the bases the study compares (all cameras, shown feeds, the head, the
gimbal band, gaze), from the run's own files. analyse.hat_family's all-camera numbers are reproduced exactly (same
test); the other bases are what the plan's E_any / E_head mean when measured instead of predicted.

  all     every alive drone's camera: range <= R (3-D), heading inside the 74.4 deg FOV, clear 3-D sight line
          (analyse.hat_family; the recorded yawDeg is the camera's heading)
  shown   ... and its feed was on the ring: a swarm drone is shown while it is a hull vertex of the alive drones
          (debounced like AttitudeAlgorithm: BOUNDARY_HYST_S on / off) and once it has turned within
          FEED_TOLERANCE_DEG of its outward heading (the BoundaryFeedReady latch); a lone drone always. Runs that
          logged `shown` (_drones.csv) use it as recorded.
  head    ... and the hat's display azimuth (feed yaw - body yaw + bearing x DISPLAY_SCALE) is inside the HMD's
          +-50 deg of the recorded head yaw (headYaw - bodyYaw - 90). The single drone's feed is in front.
  band    ... with the vertical field of view at the recorded gimbal pitch (analyse ignores pitch)
  gaze    ... and the recorded gaze ray hit that drone's feed quad (_head.csv gazeFeed; runs from 2026-10-12 on)
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import camera, display  # noqa: E402
from citysim.los import Boxes, vertical_band, wrap_deg  # noqa: E402

BOUNDARY_HYST_S = 0.5           # AttitudeAlgorithm.BoundaryHysteresisTime
FEED_TOLERANCE_DEG = 5.0        # SwarmManager.feedHeadingToleranceDeg
HAT_ABOVE_LOG = 1.6             # analyse.HAT_HEIGHT: the walker's origin is logged 0.2 m up


def shown_mask(run):
    """(T, n) bool on the drone-log clock: which drones' feeds were on the ring (see the module docstring)."""
    dr = run.drones
    T = run.ct
    ids = sorted(dr.droneId.unique())
    if "shown" in dr:
        piv = dr.pivot_table(index="t", columns="droneId", values="shown").reindex(T).fillna(0)
        return piv[ids].values > 0.5
    piv = {k: dr.pivot_table(index="t", columns="droneId", values=k).reindex(T) for k in ("gtX", "gtZ", "yawDeg", "alive")}
    X, Z, Y, A = (piv[k][ids].values for k in ("gtX", "gtZ", "yawDeg", "alive"))
    A = np.nan_to_num(A) > 0.5
    n = len(ids)
    out = np.zeros((len(T), n), bool)
    if run.condition != "Swarm" or n == 1:
        out[:] = A
        return out
    dt = np.diff(np.r_[T, T[-1] + 0.1])
    timer = np.zeros(n)
    boundary = np.zeros(n, bool)
    ready = np.zeros(n, bool)
    for k in range(len(T)):
        alive = np.flatnonzero(A[k])
        on = np.zeros(n, bool)
        target = np.full(n, np.nan)
        if len(alive) >= 3:
            P = np.c_[X[k, alive], Z[k, alive]]
            h = display.convex_hull(P)
            yaw = display.heading_deg(-display.inward_bisectors(P[h]))
            on[alive[h]] = True
            target[alive[h]] = yaw
        flip = on != boundary
        timer = np.where(flip, timer + dt[k], 0.0)
        change = flip & (timer >= BOUNDARY_HYST_S)
        boundary = np.where(change, on, boundary)
        timer = np.where(change, 0.0, timer)
        ready &= boundary
        close = np.abs(wrap_deg(Y[k] - target)) <= FEED_TOLERANCE_DEG + 15.0   # the gap fill can turn a target 15 deg
        ready |= boundary & np.nan_to_num(close, nan=False).astype(bool)
        out[k] = boundary & ready & A[k]
    return out


def hat_bases(run, boxes, r_id=camera.R_ID_M, head=True):
    """Per goal of a run: the hat's visibility per sample on every basis. Returns [dict(goalIndex, t, dt,
    all, shown, head, band, gaze, blocked_all, n_all)] with boolean (T,) arrays on the drone-log clock."""
    dr = run.drones
    T, dtc = run.ct, run.cdt
    ids = sorted(dr.droneId.unique())
    piv = {k: dr.pivot_table(index="t", columns="droneId", values=k).reindex(T)[ids].values
           for k in ("gtX", "gtY", "gtZ", "yawDeg", "alive")}
    A = np.nan_to_num(piv["alive"]) > 0.5
    shown = shown_mask(run) & A
    B = Boxes(boxes)
    h = run.head
    hi = np.clip(np.searchsorted(h.t.values, T), 0, len(h) - 1)
    body = h.bodyYaw.values[hi]
    head_rel = wrap_deg(h.headYaw.values[hi] - body - 90.0)
    pitch = h.gimbalPitch.values[hi] if "gimbalPitch" in h else np.full(len(T), np.nan)
    gaze = h.gazeFeed.values[hi] if "gazeFeed" in h else None
    out = []
    for gi, g in enumerate(run.goals):
        gidx = g.get("goalIndex", gi)
        w = run.walkers[run.walkers.goalIndex == gidx].set_index("t")[["specialX", "specialY", "specialZ"]]
        w = w[~w.index.duplicated()].reindex(T)
        hat = w.values + [0.0, HAT_ABOVE_LOG, 0.0]
        have = np.isfinite(hat).all(1)
        cam = np.stack([piv["gtX"], piv["gtY"], piv["gtZ"]], -1)               # (T, n, 3)
        rel = hat[:, None, :] - cam
        dist = np.linalg.norm(rel, axis=2)
        hd = np.hypot(rel[..., 0], rel[..., 2])
        off = wrap_deg(np.degrees(np.arctan2(rel[..., 0], rel[..., 2])) - piv["yawDeg"])
        cand = A & have[:, None] & (dist <= r_id) & (np.abs(off) <= camera.HFOV_DEG / 2)
        vis = cand.copy()
        if cand.any():
            ti, di = np.nonzero(cand)
            near = B.near(np.nanmean(hat[have][:, [0, 2]], axis=0), r_id + 80) if have.any() else np.arange(0)
            sub = B.subset(near)
            vis[ti, di] = ~sub.blocked(cam[ti, di], hat[ti])
        az = wrap_deg(piv["yawDeg"] - body[:, None] + off * camera.DISPLAY_SCALE)
        on_head = np.abs(wrap_deg(az - head_rel[:, None])) <= camera.HMD_HALF_FOV_DEG
        if run.condition != "Swarm" or len(ids) == 1:
            on_head[:] = True
        band = vertical_band(piv["gtY"], pitch[:, None], hd, target_h=camera.HAT_HEIGHT_M) if np.isfinite(pitch).any() \
            else np.ones_like(vis)
        row = dict(goalIndex=gidx, t=T, dt=dtc, all=vis.any(1), shown=(vis & shown).any(1),
                   head=(vis & shown & on_head).any(1), band=(vis & band).any(1),
                   shown_band=(vis & shown & band).any(1), head_band=(vis & shown & on_head & band).any(1),
                   blocked_all=(cand & ~vis).any(1), blocked_shown=(cand & shown & ~vis).any(1),
                   n_all=vis.sum(1), n_shown=(vis & shown).sum(1))
        if gaze is not None:
            gz = np.asarray(gaze, float)
            hit = np.array([[gz[k] == ids[d] for d in range(len(ids))] for k in range(len(T))])
            row["gaze"] = (vis & shown & hit).any(1)
        out.append(row)
    return out


def episodes_first(t, dt, flag, gap=5.0):
    """(first time, in-view seconds, episode count) joining gaps <= `gap` (analyse.episodes)."""
    idx = np.flatnonzero(flag)
    if not len(idx):
        return math.nan, 0.0, 0
    n, prev = 1, idx[0]
    for i in idx[1:]:
        if t[i] - t[prev] - dt[prev] > gap:
            n += 1
        prev = i
    return float(t[idx[0]]), float(dt[flag].sum()), n


def summary(run, boxes, legs=None, r_id=camera.R_ID_M):
    """One row per goal: seconds in view on every basis (whole run, and inside the goal's search phase when legs
    are given), first sight per basis, blocked-without-escape seconds."""
    rows = []
    by_gi = {L["gi"]: L for L in legs or []}
    for gi, b in enumerate(hat_bases(run, boxes, r_id)):
        t, dt = b["t"], b["dt"]
        row = dict(run=run.stem, condition=run.condition, goalIndex=b["goalIndex"])
        for key in ("all", "shown", "head", "band", "shown_band", "head_band", "gaze"):
            if key not in b:
                continue
            first, sec, n = episodes_first(t, dt, b[key])
            row[f"inView_{key}_s"] = round(sec, 2)
            row[f"firstSight_{key}_s"] = round(first, 2) if math.isfinite(first) else math.nan
            L = by_gi.get(gi)
            if L is not None:
                m = (t >= L["enter"]) & (t < L["exit"])
                row[f"search_s"] = round(float(dt[m].sum()), 2)
                row[f"inViewSearch_{key}_s"] = round(float(dt[m & b[key]].sum()), 2)
        row["occluded_all_s"] = round(float(dt[b["blocked_all"] & ~b["all"]].sum()), 2)
        row["occluded_shown_s"] = round(float(dt[b["blocked_shown"] & ~b["shown"]].sum()), 2)
        rows.append(row)
    return rows
