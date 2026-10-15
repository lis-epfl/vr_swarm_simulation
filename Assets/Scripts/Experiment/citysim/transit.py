"""What each interface SEES while it flies, and how its pilot flies: the moving half of the task-independent
capability study (citysim.capability) and the transit metrics of Tier B' / Tier C at the trial levels.

Ground: a CELL_M grid of free cells (outside every footprint) over the corridor round a set of flights -- one OD pair's
flights share a grid, so their cells can be compared. A cell is SEEN at a sample when a shown feed recognises a point
at hat height there: within the range (RANGES: 40 m, camera.R_ID_M, the hat; 80 m, discovery.R_GOAL_M, a person) of
slant distance, inside the camera's horizontal field of view and the gimbal's vertical band, with a clear sight line
(2-D: the study's cities are 50 m slabs). Bases: "any" (every shown feed) and "head" (the feeds inside the HMD's
+-50 deg of straight ahead -- in transit internal_3's swarm pilots looked within 25 deg of straight ahead 87 % of the
time; the single drone's one feed is in front).

Per flight, over its transit samples:
  coverage      cells seen (m2), per second of transit and per metre of centroid path
  sweep width   the lateral range curve P(x) -- the share of free cells at signed offset x from the centroid's track
                that were ever seen, over cells whose foot falls inside the track away from its ends -- and
                W = integral P(x) dx, search theory's width of ground a sensor clears per metre flown
  persistence   mean length of a cell's runs on screen (s)
  look-behind   (several cameras) cells the feeds saw FIRST while an ideal 360-degree camera on the centroid could not
                see them: "corner" when the centroid's sight line was blocked by a building, "reach" when the cell was
                out of the centroid camera's range or band; and, for cells both saw, how much earlier the feeds did
  traversal     straight and flown distance, path ratio, street share (analyse.LayoutStreetGrid), the pilot's commands
                (analyse.detect_adjustments / window_metrics on the scripted stick: adjustments per 100 m, course turn
                per minute, command straightness), morph events (buildings that stood inside the alive drones' convex
                hull, per km), stalls
  discovery     (goal legs) the first instant each goal is seen (discovery.track_expected) -- distance from it, time
                from the leg's start, and whether it was seen before the leg began
compare() sets two flights of one OD pair side by side on their shared cells: who saw a cell first, by how long, and
what only one of them saw.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera, discovery, display  # noqa: E402
from citysim.los import vertical_band, wrap_deg  # noqa: E402

CELL_M = 2.0
RANGES = (camera.R_ID_M, discovery.R_GOAL_M)
DISCOVERY_RANGES = (discovery.R_GOAL_M,) + discovery.R_GOAL_SENSITIVITY
SAMPLE_S = 0.5
LATERAL_BIN_M = 2.0
END_MARGIN_M = None          # cells whose foot is within this of a track end are left out of P(x) (None = range)
TRANSIT, GOAL = 0, 1


@dataclass
class FlightRecord:
    """One flight's tracks on the SAMPLE_S clock (transit and goal phases) and its commands on the CMD_S clock."""
    t: np.ndarray            # (T,)
    phase: np.ndarray        # (T,) TRANSIT | GOAL
    leg: np.ndarray          # (T,) the goal (or OD leg) index
    centroid: np.ndarray     # (T, 2)
    P: np.ndarray            # (T, n, 2), NaN once lost
    body_yaw: np.ndarray     # (T,) the pilot's body yaw (swarm) or the drone's heading (single), deg
    cam_pos: np.ndarray      # (T, K, 2) the shown feeds' cameras
    cam_yaw: np.ndarray      # (T, K)
    cam_valid: np.ndarray    # (T, K)
    cmd: np.ndarray          # (C, 5): t, stick x, stick z (world, |.| <= 1), heading deg, leg (< 0 = holding)
    legs: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def single(self):
        return self.meta.get("interface", "swarm") != "swarm"

    @classmethod
    def from_flight(cls, fl, **meta):
        S = fl.series
        T = len(S)
        K = max([len(s["cam_yaw"]) for s in S] + [1])
        n = len(fl.sim.P)
        pos, yaw, valid = np.zeros((T, K, 2)), np.zeros((T, K)), np.zeros((T, K), bool)
        for i, s in enumerate(S):
            k = len(s["cam_yaw"])
            if k:
                pos[i, :k], yaw[i, :k], valid[i, :k] = s["cam_pos"], s["cam_yaw"], True
        P = np.array([s["P"] for s in S], float).reshape(T, n, 2)
        meta = dict(meta, stall_events=fl.stall_events, replans=fl.replans, n=n,
                    contacts=int(getattr(fl.sim, "contacts", 0)), lost=int(n - fl.sim.alive.sum()))
        return cls(t=np.array([s["t"] for s in S], float),
                   phase=np.array([TRANSIT if s["phase"] == "transit" else GOAL for s in S], int),
                   leg=np.array([s["goal"] for s in S], int),
                   centroid=np.array([[s["cx"], s["cz"]] for s in S], float).reshape(T, 2),
                   P=P, body_yaw=np.array([s["body_yaw"] for s in S], float), cam_pos=pos, cam_yaw=yaw,
                   cam_valid=valid, cmd=np.array(fl.cmds, float).reshape(-1, 5),
                   legs=[dict(lg) for lg in fl.legs], meta=meta)

    @classmethod
    def from_bench(cls, tr, sc_meta, single, wp_transit, legs):
        """From tierc.read_traj output: the bench's recorded drones (cameras 0.4 m ahead along each drone's yaw,
        shown = BoundaryFeedReady, a lone drone always), the waypoint pilot's body yaw. wp_transit (T,) marks the
        rows flown between waypoints rather than holding."""
        T, n = tr["X"].shape
        y = np.radians(tr["yaw"])
        pos = np.stack([tr["X"], tr["Z"]], -1) + camera.CAMERA_FORWARD_M * np.stack([np.sin(y), np.cos(y)], -1)
        valid = tr["alive"] & (tr["shown"] | single)
        P = np.where(tr["alive"][..., None], np.stack([tr["X"], tr["Z"]], -1), np.nan)
        with np.errstate(all="ignore"):
            cen = np.nanmean(P, axis=1)
        body = tr["body"] if np.isfinite(tr["body"]).all() else np.nanmean(np.where(tr["alive"], tr["yaw"], np.nan), 1)
        return cls(t=tr["t"], phase=np.where(wp_transit, TRANSIT, GOAL), leg=tr["leg"], centroid=cen, P=P,
                   body_yaw=body, cam_pos=pos, cam_yaw=tr["yaw"], cam_valid=valid, cmd=np.zeros((0, 5)), legs=legs,
                   meta=dict(sc_meta))

    def transit_rows(self):
        return np.flatnonzero(self.phase == TRANSIT)


# ============================================================================= the ground

class Cells:
    """The free ground cells of a box: centres (N, 2), and a lookup from grid (i, j) to cell index."""

    def __init__(self, fp, bounds, step=CELL_M):
        x0, x1, z0, z1 = bounds
        self.step, self.x0, self.z0 = step, x0, z0
        self.nx, self.nz = int(math.ceil((x1 - x0) / step)), int(math.ceil((z1 - z0) / step))
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing="ij")
        pts = np.stack([x0 + (ii + 0.5) * step, z0 + (jj + 0.5) * step], -1).reshape(-1, 2)
        free = np.ones(len(pts), bool)
        if fp.n:
            near = fp.subset(np.flatnonzero((fp.c[:, 0] + fp.R >= x0) & (fp.c[:, 0] - fp.R <= x1) &
                                            (fp.c[:, 1] + fp.R >= z0) & (fp.c[:, 1] - fp.R <= z1)))
            for k in range(0, len(pts), 20000):
                if near.n:
                    free[k:k + 20000] = ~near.inside(pts[k:k + 20000])
        self.index = -np.ones(self.nx * self.nz, np.int64)
        self.index[np.flatnonzero(free)] = np.arange(int(free.sum()))
        self.index = self.index.reshape(self.nx, self.nz)
        self.pts = pts[free]
        self.n = len(self.pts)
        self.area = step * step

    @classmethod
    def around(cls, fp, tracks, margin):
        """Cells over the box round every point of `tracks` (a list of (k, 2) arrays), grown by margin."""
        allp = np.vstack([np.asarray(t, float).reshape(-1, 2) for t in tracks if len(t)])
        allp = allp[np.isfinite(allp).all(1)]
        lo, hi = allp.min(0) - margin, allp.max(0) + margin
        return cls(fp, (lo[0], hi[0], lo[1], hi[1]))

    def within(self, p, r):
        """Indices of cells within horizontal distance r of p."""
        i0 = max(int((p[0] - r - self.x0) // self.step), 0)
        i1 = min(int((p[0] + r - self.x0) // self.step) + 1, self.nx)
        j0 = max(int((p[1] - r - self.z0) // self.step), 0)
        j1 = min(int((p[1] + r - self.z0) // self.step) + 1, self.nz)
        if i0 >= i1 or j0 >= j1:
            return np.zeros(0, np.int64)
        idx = self.index[i0:i1, j0:j1].ravel()
        idx = idx[idx >= 0]
        d = self.pts[idx] - p
        return idx[np.hypot(d[:, 0], d[:, 1]) <= r]


def seen_from(fp, cells, cam, yaw, height, pitch, r_max, hfov=camera.HFOV_DEG):
    """Cells one camera recognises: (indices, slant ranges, bearing off its axis), within r_max slant, the horizontal
    field of view (hfov >= 360 = every direction), the vertical band and a clear 2-D sight line."""
    hmax = math.sqrt(max(r_max ** 2 - (height - camera.HAT_HEIGHT_M) ** 2, 0.0))
    idx = cells.within(cam, hmax)
    if not len(idx):
        return idx, np.zeros(0), np.zeros(0)
    d = cells.pts[idx] - cam
    hd = np.hypot(d[:, 0], d[:, 1])
    off = wrap_deg(np.degrees(np.arctan2(d[:, 0], d[:, 1])) - yaw)
    ok = vertical_band(height, pitch, hd)
    if hfov < 360.0:
        ok &= np.abs(off) <= hfov / 2
    idx, hd, off = idx[ok], hd[ok], off[ok]
    if len(idx) and fp.n:
        near = fp.subset(fp.near(cam, hmax + 1.0))
        if near.n:
            clear = ~near.blocked(np.broadcast_to(np.asarray(cam, float), (len(idx), 2)), cells.pts[idx])
            idx, hd, off = idx[clear], hd[clear], off[clear]
    return idx, np.hypot(hd, height - camera.HAT_HEIGHT_M), off


class Seen:
    """Per-cell first sight, samples on screen and runs, for each range, accumulated sample by sample."""

    def __init__(self, n, ranges=RANGES):
        self.ranges = ranges
        self.first = {r: np.full(n, np.inf) for r in ranges}
        self.count = {r: np.zeros(n, np.int32) for r in ranges}
        self.runs = {r: np.zeros(n, np.int32) for r in ranges}
        self.prev = {r: np.zeros(n, bool) for r in ranges}

    def add(self, t, slant):
        """slant (n,): the nearest slant range at which a camera saw each cell this sample (inf = unseen)."""
        new = {}
        for r in self.ranges:
            s = slant <= r
            fresh = s & np.isinf(self.first[r])
            self.first[r][fresh] = t
            self.count[r] += s
            self.runs[r] += s & ~self.prev[r]
            self.prev[r] = s
            new[r] = fresh
        return new


def visibility(fp, cells, rec, height, pitch, rows=None, omni=False, ranges=RANGES, head_yaw=0.0):
    """Every sample of `rec` (or `rows`) through its shown feeds: Seen accumulators for the "any" and "head" bases,
    and -- omni -- for an ideal 360-degree camera on the centroid, with the look-behind tally of cells the feeds saw
    first while that camera could not ({range: dict(corner=..., reach=...)} counts of cells)."""
    rows = rec.transit_rows() if rows is None else rows
    r_max = max(ranges)
    acc_any, acc_head = Seen(cells.n, ranges), Seen(cells.n, ranges)
    acc_omni = Seen(cells.n, ranges) if omni else None
    behind = {r: dict(corner=np.zeros(cells.n, bool), reach=np.zeros(cells.n, bool),
                      corner_head=np.zeros(cells.n, bool), reach_head=np.zeros(cells.n, bool)) for r in ranges}
    s_any = np.full(cells.n, np.inf)
    s_head = np.full(cells.n, np.inf)
    s_omni = np.full(cells.n, np.inf)
    single = rec.single
    for ti in rows:
        t = float(rec.t[ti])
        touched = []
        for k in np.flatnonzero(rec.cam_valid[ti]):
            idx, slant, off = seen_from(fp, cells, rec.cam_pos[ti, k], rec.cam_yaw[ti, k], height, pitch, r_max)
            if not len(idx):
                continue
            np.minimum.at(s_any, idx, slant)
            if single:
                np.minimum.at(s_head, idx, slant)
            else:
                az = wrap_deg(rec.cam_yaw[ti, k] - rec.body_yaw[ti] + off * camera.DISPLAY_SCALE)
                hv = np.abs(wrap_deg(az - head_yaw)) <= camera.HMD_HALF_FOV_DEG
                np.minimum.at(s_head, idx[hv], slant[hv])
            touched.append(idx)
        new_any = acc_any.add(t, s_any)
        new_head = acc_head.add(t, s_head)
        if omni and np.isfinite(rec.centroid[ti]).all():
            o_idx, o_slant, _ = seen_from(fp, cells, rec.centroid[ti], 0.0, height, pitch, r_max, hfov=360.0)
            s_omni[o_idx] = o_slant
            acc_omni.add(t, s_omni)
            # cells the centroid camera could not see now, but the feeds saw for the first time
            hmax = {r: math.sqrt(max(r ** 2 - (height - camera.HAT_HEIGHT_M) ** 2, 0.0)) for r in ranges}
            for r in ranges:
                for basis, fresh in (("", new_any[r]), ("_head", new_head[r])):
                    cand = np.flatnonzero(fresh & ~(s_omni <= r))
                    if not len(cand):
                        continue
                    d = cells.pts[cand] - rec.centroid[ti]
                    hd = np.hypot(d[:, 0], d[:, 1])
                    covered = (hd <= hmax[r]) & vertical_band(height, pitch, hd)   # in reach: so the line was blocked
                    behind[r]["corner" + basis][cand[covered]] = True
                    behind[r]["reach" + basis][cand[~covered]] = True
            s_omni[o_idx] = np.inf
        for idx in touched:
            s_any[idx] = np.inf
            s_head[idx] = np.inf
    return acc_any, acc_head, acc_omni, behind


# ============================================================================= geometry of a track

def lateral_offsets(track, pts):
    """(signed offset (N,), foot arclength (N,), track length): each point's distance from the polyline `track`
    (positive to the right of travel), and where along it its foot falls."""
    tr = np.asarray(track, float)
    tr = tr[np.isfinite(tr).all(1)]
    if len(tr) < 2:
        return np.full(len(pts), np.nan), np.full(len(pts), np.nan), 0.0
    keep = np.r_[True, np.linalg.norm(np.diff(tr, axis=0), axis=1) > 1e-6]
    tr = tr[keep]
    a, b = tr[:-1], tr[1:]
    ab = b - a
    L = np.linalg.norm(ab, axis=1)
    cum = np.r_[0.0, np.cumsum(L)]
    best = np.full(len(pts), np.inf)
    off = np.full(len(pts), np.nan)
    arc = np.full(len(pts), np.nan)
    for k in range(len(a)):
        u = ab[k] / max(L[k], 1e-12)
        rel = pts - a[k]
        s = np.clip(rel @ u, 0.0, L[k])
        foot = a[k] + s[:, None] * u
        d = np.linalg.norm(pts - foot, axis=1)
        better = d < best
        best[better] = d[better]
        side = np.sign(rel[:, 0] * u[1] - rel[:, 1] * u[0])          # + = right of travel ((x, z), clockwise yaw)
        off[better] = (side * d)[better]
        arc[better] = (cum[k] + s)[better]
    return off, arc, float(cum[-1])


def sweep_width(seen_first, offsets, arc, length, r_max, bin_m=LATERAL_BIN_M, end_margin=END_MARGIN_M):
    """(W m, bins, P(x)): the lateral range curve over cells whose foot lies END_MARGIN inside the track's ends."""
    margin = r_max if end_margin is None else end_margin
    sel = np.isfinite(offsets) & (arc >= margin) & (arc <= length - margin)
    edges = np.arange(-r_max - bin_m, r_max + bin_m + 1e-9, bin_m)
    if not sel.any():
        return math.nan, edges, np.full(len(edges) - 1, np.nan)
    seen = np.isfinite(seen_first[sel])
    k = np.digitize(offsets[sel], edges) - 1
    ok = (k >= 0) & (k < len(edges) - 1)
    tot = np.bincount(k[ok], minlength=len(edges) - 1).astype(float)
    hit = np.bincount(k[ok], weights=seen[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(tot > 0, hit / tot, np.nan)
    return float(np.nansum(np.nan_to_num(p) * bin_m)), edges, p


def path_length(track):
    tr = np.asarray(track, float)
    tr = tr[np.isfinite(tr).all(1)]
    return float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()) if len(tr) > 1 else 0.0


SPLIT_FACTOR = 2.5          # a formation wider than this many times its settled hover diameter has split


def morph_events(rec, fp, rows=None):
    """(distinct buildings the formation morphed round, share of samples with one inside it, share of samples the
    formation was split). A building is morphed round when its centre stands inside the alive drones' convex hull
    while the formation is whole: no wider than SPLIT_FACTOR times its settled diameter at the flight's spread, the
    building within one diameter of its centre. A formation strung out across a block (survivors on either side of
    it) would otherwise count the whole block, which is a split, not a morph."""
    rows = rec.transit_rows() if rows is None else rows
    if rec.single or not fp.n:
        return 0, 0.0, 0.0
    from citysim import routing
    d_ref = rec.meta.get("d_ref", 1.0)
    D = 2.0 * routing.formation_radius(d_ref if np.isfinite(d_ref) else 1.0)
    inside_any, hit, split, n = set(), 0, 0, 0
    for ti in rows:
        P = rec.P[ti]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 3:
            continue
        n += 1
        c0 = P.mean(0)
        if 2.0 * float(np.linalg.norm(P - c0, axis=1).max()) > SPLIT_FACTOR * D:
            split += 1
            continue
        h = display.convex_hull(P)
        if len(h) < 3:
            continue
        H = P[h]
        cand = fp.near(c0, D)
        if not len(cand):
            continue
        c = fp.c[cand]
        e = np.roll(H, -1, axis=0) - H
        cross = e[None, :, 0] * (c[:, None, 1] - H[None, :, 1]) - e[None, :, 1] * (c[:, None, 0] - H[None, :, 0])
        ins = (cross > 0).all(1) | (cross < 0).all(1)
        if ins.any():
            hit += 1
            inside_any.update(int(b) for b in np.asarray(cand)[ins])
    return len(inside_any), hit / max(n, 1), split / max(n, 1)


# ============================================================================= the pilot's commands

def command_metrics(rec, legs_spans, straight_m):
    """analyse.py's pilot-command measures on the scripted stick, over the transit spans [(t0, t1)]: the stick is
    held forward (pitch axis) and turned by yaw; the yaw axis is the heading's rate over the rig's rate (the swarm's
    body yaw, 90 deg/s) or the drone's (75 deg/s)."""
    import citysim
    citysim.use_bench_modules()
    import analyse
    c = rec.cmd
    if len(c) < 3:
        return {}
    t, sx, sz, hd = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    rate = camera.DRONE_MAX_YAW_RATE_DEG if rec.single else camera.BODY_YAW_RATE_DEG
    dh = np.r_[0.0, wrap_deg(np.diff(hd))] / np.maximum(np.r_[1.0, np.diff(t)], 1e-6)
    U = np.c_[np.hypot(sx, sz), np.zeros(len(t)), np.clip(dh / rate, -1, 1), np.zeros(len(t))]
    U = analyse.deadband(U, analyse.DEFAULTS["deadband"])
    vx, vz = sx * analyse.MAX_SPEED, sz * analyse.MAX_SPEED
    starts, axes = analyse.detect_adjustments(t, U, analyse.DEFAULTS["tol"], analyse.DEFAULTS["settle"])
    rows = [analyse.window_metrics(t, U, vx, vz, starts, axes, t0, t1) for t0, t1 in legs_spans]
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    dur = sum(r["dur_s"] for r in rows)
    adj = sum(r["adjustments"] for r in rows)
    return dict(adjustments=adj, adj_per_min=adj / dur * 60.0 if dur else math.nan,
                adj_per_100m=adj / straight_m * 100.0 if straight_m > 5 else math.nan,
                course_turn_per_min=sum(r["course_turn_deg"] for r in rows) / dur * 60.0 if dur else math.nan,
                cmd_straightness=sum(r["cmd_disp_m"] for r in rows) / max(sum(r["cmd_len_m"] for r in rows), 1e-9),
                yaw_duty=sum(r["yaw_duty"] * r["dur_s"] for r in rows) / dur if dur else math.nan)


def street_share(layout_id, track):
    """Share of the centroid's path over the city that ran in the streets (analyse.LayoutStreetGrid)."""
    import citysim
    citysim.use_bench_modules()
    import analyse
    import contextlib
    import io
    tr = np.asarray(track, float)
    tr = tr[np.isfinite(tr).all(1)]
    if len(tr) < 2 or not layout_id:
        return math.nan
    with contextlib.redirect_stdout(io.StringIO()):
        grid = analyse.street_grid(layout_id, analyse.DEFAULTS["long_street"])
    k = grid.kind(tr[:-1, 0], tr[:-1, 1], np.full(len(tr) - 1, 10.0))
    step = np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1]))
    m = np.bincount(k, weights=step, minlength=len(analyse.KINDS))
    city = m[1:].sum()
    return float((m[1] + m[2]) / city) if city > 0 else math.nan


# ============================================================================= one flight

def score(rec, fp, cells, height, pitch, omni=None, layout=None, goals=None, r_goals=(discovery.R_GOAL_M,)):
    """Every per-flight measure of the module docstring, as one flat dict (keys suffixed _r40 / _r80 by range)."""
    rows = rec.transit_rows()
    omni = (not rec.single) if omni is None else omni
    out = dict(rec.meta)
    out.pop("legs", None)
    if not len(rows):
        return out
    acc_any, acc_head, acc_omni, behind = visibility(fp, cells, rec, height, pitch, rows, omni=omni)
    track = rec.centroid[rows]
    flown = path_length(track)
    secs = len(rows) * SAMPLE_S
    off, arc, length = lateral_offsets(track, cells.pts)
    out.update(transit_s=secs, flown_m=round(flown, 1))
    for r in RANGES:
        tag = f"_r{int(r)}"
        for basis, acc in (("any", acc_any), ("head", acc_head)):
            seen = np.isfinite(acc.first[r])
            area = float(seen.sum() * cells.area)
            W, _, _ = sweep_width(acc.first[r], off, arc, length, r)
            with np.errstate(invalid="ignore", divide="ignore"):
                runs = acc.count[r][seen] / np.maximum(acc.runs[r][seen], 1)
            out.update({f"area_{basis}{tag}_m2": round(area, 0),
                        f"area_{basis}{tag}_per_s": round(area / secs, 2) if secs else math.nan,
                        f"area_{basis}{tag}_per_m": round(area / flown, 2) if flown > 1 else math.nan,
                        f"sweep_{basis}{tag}_m": round(W, 2) if math.isfinite(W) else math.nan,
                        f"persist_{basis}{tag}_s": round(float(runs.mean() * SAMPLE_S), 2) if seen.any() else math.nan})
        if omni:
            seen = np.isfinite(acc_any.first[r])
            both = seen & np.isfinite(acc_omni.first[r])
            lead = acc_omni.first[r][both] - acc_any.first[r][both]
            n_seen = max(int(seen.sum()), 1)
            for basis in ("", "_head"):
                sb = seen if basis == "" else np.isfinite(acc_head.first[r])
                nb = max(int(sb.sum()), 1)
                out[f"behind_corner{basis}{tag}_frac"] = round(float(behind[r]["corner" + basis].sum() / nb), 4)
                out[f"behind_reach{basis}{tag}_frac"] = round(float(behind[r]["reach" + basis].sum() / nb), 4)
            out[f"behind_corner{tag}_m2_per_km"] = round(float(behind[r]["corner"].sum() * cells.area) /
                                                         max(flown / 1000.0, 1e-9), 0) if flown > 1 else math.nan
            out[f"omni_area{tag}_m2"] = round(float(np.isfinite(acc_omni.first[r]).sum() * cells.area), 0)
            out[f"only_feeds_vs_omni{tag}_frac"] = round(float((seen & ~np.isfinite(acc_omni.first[r])).sum() /
                                                               n_seen), 4)
            out[f"lead_vs_omni{tag}_s"] = round(float(np.median(lead)), 2) if len(lead) else math.nan
            out[f"earlier_than_omni{tag}_frac"] = round(float((lead > 0).mean()), 4) if len(lead) else math.nan
    out["firsts"] = {f"{b}{int(r)}": acc.first[r] for r in RANGES for b, acc in (("any", acc_any), ("head", acc_head))}
    # ---- traversal
    legs = rec.legs or [dict(leg=0, t0=float(rec.t[rows[0]]), t1=float(rec.t[rows[-1]]), start=track[0],
                             goal=track[-1])]
    straight = 0.0
    spans = []
    for lg in legs:
        st = float(np.linalg.norm(np.asarray(lg["goal"]) - np.asarray(lg["start"])))
        arrive = float(lg.get("arrive_m", 0.0))
        straight += max(st - arrive, 0.0)
        spans.append((lg["t0"], lg["t1"]))
    out.update(straight_m=round(straight, 1), path_ratio=round(flown / straight, 4) if straight > 5 else math.nan,
               planned_ratio=round(sum(lg.get("planned_m", math.nan) for lg in legs) /
                                   max(sum(float(np.linalg.norm(np.asarray(lg["goal"]) - np.asarray(lg["start"])))
                                           for lg in legs), 1e-9), 4) if all("planned_m" in lg for lg in legs)
               else math.nan,
               corners=sum(lg.get("planned_corners", 0) for lg in legs),
               progress_ms=round(straight / secs, 3) if secs else math.nan)
    if layout is not None:
        out["street_frac"] = street_share(layout, track)
    if len(rec.cmd):
        out.update({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in command_metrics(rec, spans, straight).items()})
    nb, frac, split = morph_events(rec, fp, rows)
    out.update(morph_buildings=nb, morph_per_km=round(nb / max(flown / 1000.0, 1e-9), 2) if flown > 1 else math.nan,
               morph_time_frac=round(frac, 4), split_frac=round(split, 4))
    # ---- discovery of the goals
    if goals:
        for gi, (pts, w, centre) in goals.items():
            leg_rows = [lg for lg in legs if lg["leg"] == gi]
            if not leg_rows:
                continue
            lg = leg_rows[0]
            upto = np.flatnonzero(rec.t <= lg["t1"] + 1e-9)
            for r_goal in r_goals:
                e_any, e_head = discovery.track_expected(
                    fp, rec.cam_pos[upto], rec.cam_yaw[upto], rec.cam_valid[upto], rec.body_yaw[upto], height, pitch,
                    pts, w, r_goal=r_goal, single=rec.single)
                for basis, e in (("any", e_any), ("head", e_head)):
                    t1, i1 = discovery.first_sight(rec.t[upto], e)
                    tag = f"g{gi}_{basis}_r{int(r_goal)}"
                    if i1 < 0:
                        out[f"disc_{tag}_s"] = math.nan
                        out[f"disc_{tag}_dist_m"] = math.nan
                        continue
                    out[f"disc_{tag}_s"] = round(max(t1 - lg["t0"], 0.0), 2)
                    out[f"disc_{tag}_pre"] = bool(t1 < lg["t0"])
                    out[f"disc_{tag}_dist_m"] = round(float(np.linalg.norm(rec.centroid[upto][i1] - centre)), 1)
    return out


def compare(a, b, prefix):
    """Two flights of one OD pair on their shared cells (a and b are score() outputs carrying "firsts"): the share of
    cells both saw that `a` saw first, the median lead (s) of a over b, and the area only one of them saw."""
    out = {}
    for key in a.get("firsts", {}):
        if key not in b.get("firsts", {}):
            continue
        fa, fb = a["firsts"][key], b["firsts"][key]
        sa, sb = np.isfinite(fa), np.isfinite(fb)
        both = sa & sb
        lead = fb[both] - fa[both]
        out[f"{prefix}_first_frac_{key}"] = round(float((lead > 0).mean()), 4) if both.any() else math.nan
        out[f"{prefix}_lead_{key}_s"] = round(float(np.median(lead)), 2) if both.any() else math.nan
        out[f"{prefix}_only_a_{key}_m2"] = round(float((sa & ~sb).sum() * CELL_M ** 2), 0)
        out[f"{prefix}_only_b_{key}_m2"] = round(float((sb & ~sa).sum() * CELL_M ** 2), 0)
    return out
