"""What the pilot is shown: the swarm's hull / heading rule and the ring of feeds, the single drone's one feed, head
visibility, the reveal model V(t) and feed informativeness U.

Ported from the C# (numpy, 2-D):
  ConvexHull.ComputeConvexHull     strict hull (collinear points dropped), as a set of vertices
  ConvexHull.ComputeBisector       inward bisector = normalised sum of the UNIT edges to the two neighbours
  AttitudeAlgorithm                a hull drone's target heading is the outward bisector (heading = atan2(x, z),
                                   clockwise from +z), plus the look-direction gap fill (UpdateLookGapShift):
                                   the two vertices bracketing the body yaw turn towards it by
                                   clamp(d_near - coverage, 0, max_shift); on in the experiment scenes
                                   (fillLookDirectionGap 1, 20 / 15 deg), only above FPVCameraScript.NadirPitch
  ScreenSpawn OUTER_CIRCLE         each hull drone whose feed is ready (BoundaryFeedReady) is shown on a 2 m ring
                                   at its own yaw; a screen at display azimuth a = droneYaw - bodyYaw (clockwise
                                   from straight ahead), overlaps allowed; the 80 deg view cone only throttles
                                   refresh. A hat at bearing b off a camera's axis appears at a + b * DISPLAY_SCALE.
  single drone                     one feed, always in front; the pilot's view is slaved to the drone's heading
Static templates take every hull drone as settled onto its target heading (feed ready); the dynamic tier
recomputes the rule per sample from the replica's positions (no yaw lag: an upper bound on readiness). FeedLatch is the
rule with its timing, for the search study: the hull membership debounced, the target heading low-passed, each drone
turning onto it at the airframe's yaw rate, and a feed shown only once its drone has (BoundaryFeedReady).
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera  # noqa: E402
from citysim.los import wrap_deg  # noqa: E402

NADIR_PITCH_DEG = -60.0          # FPVCameraScript.NadirPitch: the gap fill stops below it
LOOK_GAP_COVERAGE_DEG = 20.0     # SwarmManager.lookGapCoverageDeg (scenes)
LOOK_GAP_MAX_SHIFT_DEG = 15.0    # SwarmManager.lookGapMaxShiftDeg (scenes)
WALL_HIT_M = 5.0                 # a feed ray that hits a facade nearer than this shows wall, not scene
STITCHED_FEEDS = 3               # PyUniSharingFast: the left / centre / right triplet the panorama hides


# ============================================================================= the hull rule

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Indices of the strict convex hull of (n, 2) points, counter-clockwise (monotone chain; collinear
    points are not vertices, as in ConvexHull.ComputeConvexHull)."""
    P = np.asarray(points, float)
    n = len(P)
    if n < 3:
        return list(range(n))
    order = sorted(range(n), key=lambda i: (P[i, 0], P[i, 1]))
    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and _cross(P[lower[-2]], P[lower[-1]], P[i]) <= 1e-12:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and _cross(P[upper[-2]], P[upper[-1]], P[i]) <= 1e-12:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def inward_bisectors(hull_pts):
    """(k, 2) ComputeBisector(hull, v, pointInwards=false) for every vertex: the unit sum of the unit edges
    to the next and previous vertex (inward for a convex vertex); centroid direction when they cancel."""
    H = np.asarray(hull_pts, float)
    k = len(H)
    if k < 2:
        return np.zeros((k, 2))
    e1 = np.roll(H, -1, axis=0) - H
    e2 = np.roll(H, 1, axis=0) - H
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12)
    b = e1 + e2
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    cen = H.mean(0)
    to_c = cen - H
    to_c /= np.maximum(np.linalg.norm(to_c, axis=1, keepdims=True), 1e-12)
    return np.where(nb > 1e-9, b / np.maximum(nb, 1e-12), to_c)


def heading_deg(v):
    """Heading of (x, z) direction(s), clockwise from +z: Atan2(x, z)."""
    v = np.asarray(v, float)
    return np.degrees(np.arctan2(v[..., 0], v[..., 1]))


def look_gap_shift(headings_deg, body_yaw_deg, coverage=LOOK_GAP_COVERAGE_DEG, max_shift=LOOK_GAP_MAX_SHIFT_DEG):
    """AttitudeAlgorithm.UpdateLookGapShift: (per-vertex shift deg, raw gap deg, gap after the shift)."""
    h = np.asarray(headings_deg, float)
    shift = np.zeros(len(h))
    if len(h) < 3 or np.isnan(h).any():
        return shift, math.nan, math.nan
    from_below = np.mod(body_yaw_deg - h, 360.0)
    from_above = np.mod(h - body_yaw_deg, 360.0)
    below, above = int(np.argmin(from_below)), int(np.argmin(from_above))
    near = float(min(from_below[below], from_above[above]))
    s = min(max(near - coverage, 0.0), max(max_shift, 0.0))
    if s <= 0 or below == above:
        return shift, near, near
    shift[below], shift[above] = s, -s
    return shift, near, near - s


@dataclass
class Feeds:
    """The feeds on the ring at one instant: drone indices, camera positions, yaws (deg)."""
    drones: np.ndarray       # (k,) indices into the formation
    pos: np.ndarray          # (k, 2) camera positions (0.4 m ahead of the drone)
    yaw: np.ndarray          # (k,) camera yaws, deg
    raw_gap: float = math.nan
    look_gap: float = math.nan

    def __len__(self):
        return len(self.drones)


def swarm_feeds(positions, body_yaw_deg, alive=None, pitch_deg=camera.DIAL_CENTRE_PITCH_DEG, look_gap=True,
                coverage=LOOK_GAP_COVERAGE_DEG, max_shift=LOOK_GAP_MAX_SHIFT_DEG, stitched_hidden=False):
    """The swarm's displayed feeds: hull vertices at their outward (+ gap fill) headings.

    stitched_hidden removes the STITCHED_FEEDS hull drones whose yaw is closest to the body yaw (what
    hideStitchedDroneScreens does while the panorama is up) -- a sensitivity option; the panorama itself is
    not modelled."""
    P = np.asarray(positions, float)
    idx = np.arange(len(P)) if alive is None else np.flatnonzero(alive)
    if len(idx) == 0:
        return Feeds(np.zeros(0, int), np.zeros((0, 2)), np.zeros(0))
    if len(idx) == 1:                            # a lone drone: always shown, heading = its own (caller's)
        yaw = np.array([body_yaw_deg], float)
        return Feeds(idx, P[idx] + camera.CAMERA_FORWARD_M * _fwd(yaw), yaw)
    hull = convex_hull(P[idx])
    if len(hull) < 3:
        return Feeds(np.zeros(0, int), np.zeros((0, 2)), np.zeros(0))
    H = P[idx][hull]
    yaw = heading_deg(-inward_bisectors(H))
    raw = gap = math.nan
    if look_gap and pitch_deg > NADIR_PITCH_DEG:
        shift, raw, gap = look_gap_shift(yaw, body_yaw_deg, coverage, max_shift)
        yaw = wrap_deg(yaw + shift)
    drones = idx[hull]
    if stitched_hidden and len(drones) > STITCHED_FEEDS:
        near = np.argsort(np.abs(wrap_deg(yaw - body_yaw_deg)))[:STITCHED_FEEDS]
        keep = np.setdiff1d(np.arange(len(drones)), near)
        drones, yaw = drones[keep], yaw[keep]
    return Feeds(drones, P[drones] + camera.CAMERA_FORWARD_M * _fwd(yaw), np.asarray(yaw, float), raw, gap)


BOUNDARY_HYST_S = 0.5            # AttitudeAlgorithm.BoundaryHysteresisTime (its depth speed-up is not modelled)
TARGET_FILTER_S = 0.3            # AttitudeAlgorithm.TargetHeadingFilterTime
FEED_TOLERANCE_DEG = 5.0         # SwarmManager.feedHeadingToleranceDeg: the BoundaryFeedReady latch


class FeedLatch:
    """The swarm's feed ring with AttitudeAlgorithm's timing. step() advances it by dt and returns the shown Feeds:

      boundary   a drone's hull membership flips only after the disagreement has lasted BOUNDARY_HYST_S
      target     while on the hull, the outward bisector (+ look-gap fill) low-passed at TARGET_FILTER_S; kept while
                 the debounced flag still holds, released when it drops
      yaw        each drone turns onto its target at the airframe's DRONE_MAX_YAW_RATE (the heading hold's limit)
      ready      latched once a boundary drone is within FEED_TOLERANCE_DEG of its target, cleared with the flag
    The camera is the drone's ACTUAL yaw, so a feed shows where the drone looks, not where it is going to. A lone
    drone is always shown and turns toward the body yaw (it is handed the yaw stick). The first step seeds a settled
    ring. instant=True returns swarm_feeds() instead: no lag, the old upper bound."""

    def __init__(self, n, look_gap=True, pitch_deg=camera.DIAL_CENTRE_PITCH_DEG, instant=False,
                 yaw_rate=camera.DRONE_MAX_YAW_RATE_DEG):
        self.n, self.look_gap, self.pitch, self.instant, self.yaw_rate = n, look_gap, pitch_deg, instant, yaw_rate
        self.boundary = np.zeros(n, bool)
        self.timer = np.zeros(n)
        self.has_target = np.zeros(n, bool)
        self.target = np.zeros(n)
        self.yaw = np.zeros(n)
        self.ready = np.zeros(n, bool)
        self.seeded = False

    def _raw(self, P, idx, body_yaw):
        """(on hull now (n,), raw target heading (n,) deg)."""
        on = np.zeros(self.n, bool)
        raw = np.full(self.n, np.nan)
        if len(idx) >= 3:
            hull = convex_hull(P[idx])
            if len(hull) >= 3:
                yaw = heading_deg(-inward_bisectors(P[idx][hull]))
                if self.look_gap and self.pitch > NADIR_PITCH_DEG:
                    shift, _, _ = look_gap_shift(yaw, body_yaw)
                    yaw = wrap_deg(yaw + shift)
                on[idx[hull]] = True
                raw[idx[hull]] = yaw
        return on, raw

    def step(self, positions, alive, body_yaw, dt):
        P = np.asarray(positions, float)
        alive = np.asarray(alive, bool)
        idx = np.flatnonzero(alive)
        if self.instant:
            return swarm_feeds(P, body_yaw, alive, self.pitch, self.look_gap)
        if len(idx) == 0:
            return Feeds(np.zeros(0, int), np.zeros((0, 2)), np.zeros(0))
        if len(idx) == 1:
            k = idx[0]
            d = float(wrap_deg(body_yaw - self.yaw[k]))
            self.yaw[k] = float(wrap_deg(self.yaw[k] + np.clip(d, -self.yaw_rate * dt, self.yaw_rate * dt)))
            return Feeds(idx, P[idx] + camera.CAMERA_FORWARD_M * _fwd(self.yaw[idx]), self.yaw[idx].copy())
        on, raw = self._raw(P, idx, body_yaw)
        if not self.seeded:
            self.boundary[:] = on
            self.has_target[:] = on
            self.target[on] = raw[on]
            self.yaw[on] = raw[on]
            self.ready[:] = on
            self.seeded = True
        flip = (on != self.boundary) & alive
        self.timer = np.where(flip, self.timer + dt, 0.0)
        change = flip & (self.timer >= BOUNDARY_HYST_S)
        self.boundary = np.where(change, on, self.boundary) & alive
        self.timer[change] = 0.0
        a = 1.0 - math.exp(-dt / TARGET_FILTER_S)
        fresh = on & ~self.has_target
        self.target[fresh] = raw[fresh]
        upd = on & self.has_target & ~fresh
        self.target[upd] = wrap_deg(self.target[upd] + a * wrap_deg(raw[upd] - self.target[upd]))
        self.has_target = (self.has_target | on) & (on | self.boundary)
        turn = self.has_target & alive
        d = wrap_deg(self.target[turn] - self.yaw[turn])
        self.yaw[turn] = wrap_deg(self.yaw[turn] + np.clip(d, -self.yaw_rate * dt, self.yaw_rate * dt))
        self.ready &= self.boundary
        close = np.abs(wrap_deg(self.target - self.yaw)) <= FEED_TOLERANCE_DEG
        self.ready |= self.boundary & self.has_target & close
        shown = np.flatnonzero(self.boundary & self.ready & alive)
        return Feeds(shown, P[shown] + camera.CAMERA_FORWARD_M * _fwd(self.yaw[shown]), self.yaw[shown].copy())


def single_feed(position, yaw_deg):
    yaw = np.array([yaw_deg], float)
    return Feeds(np.zeros(1, int), np.asarray(position, float)[None] + camera.CAMERA_FORWARD_M * _fwd(yaw), yaw)


def _fwd(yaw_deg):
    y = np.radians(np.asarray(yaw_deg, float))
    return np.stack([np.sin(y), np.cos(y)], -1)


# ============================================================================= the pilot's eye

def display_azimuth(feed_yaw_deg, body_yaw_deg, bearing_off_axis_deg=0.0):
    """Where on the ring (deg clockwise from straight ahead) a point at that bearing in that feed appears."""
    return wrap_deg(np.asarray(feed_yaw_deg) - body_yaw_deg + np.asarray(bearing_off_axis_deg) * camera.DISPLAY_SCALE)


def head_visible(display_az_deg, head_rel_deg, half_fov=camera.HMD_HALF_FOV_DEG):
    """Within the HMD's horizontal field of view about the head's yaw (relative to the body yaw)."""
    return np.abs(wrap_deg(np.asarray(display_az_deg) - head_rel_deg)) <= half_fov


# ============================================================================= revealing directions

def swarm_reveal_deg(t):
    """How far off the current head direction (deg, either side) a swarm pilot can bring a feed into the HMD's
    view within t seconds: a head turn at HEAD_TURN_RATE up to HEAD_TURN_MAX, then body yaw rotating the rig at
    BODY_YAW_RATE."""
    t = np.asarray(t, float)
    head_t = camera.HEAD_TURN_MAX_DEG / camera.HEAD_TURN_RATE_DEG
    turned = np.where(t <= head_t, camera.HEAD_TURN_RATE_DEG * t,
                      camera.HEAD_TURN_MAX_DEG + camera.BODY_YAW_RATE_DEG * (t - head_t))
    return np.minimum(camera.HMD_HALF_FOV_DEG + turned, 180.0)


def single_reveal_deg(t):
    """How far off the drone's current heading (deg, either side) a single-drone pilot can bring the view within
    t seconds: the camera's half FOV plus the drone's yaw at maxYawRate (the whole view is slaved to it)."""
    t = np.asarray(t, float)
    return np.minimum(camera.HFOV_DEG / 2 + camera.DRONE_MAX_YAW_RATE_DEG * t, 180.0)


# ============================================================================= feed informativeness

def feed_informativeness(fp, feeds, rmax=camera.R_ID_M, step_deg=1.0, wall_m=WALL_HIT_M):
    """Per feed: the share of its horizontal FOV whose first hit is nearer than wall_m (a wall-filled feed), and
    the ground area (m^2, within rmax) it reveals that no other feed does; plus the count of useful feeds (less
    than half wall, adding at least 10 % of its own visible area)."""
    k = len(feeds)
    if k == 0:
        return dict(wall_frac=np.zeros(0), unique_area=np.zeros(0), area=np.zeros(0), useful=0)
    rel = np.arange(-camera.HFOV_DEG / 2, camera.HFOV_DEG / 2 + 1e-9, step_deg)
    dist = np.array([fp.first_hit(feeds.pos[i], feeds.yaw[i] + rel, rmax) for i in range(k)])   # (k, A)
    vis = np.minimum(dist, rmax)
    wall = (dist < wall_m).mean(axis=1)
    # visible ground of each feed on a polar raster: cells within its wedge and before its first hit
    g = np.arange(-rmax, rmax + 1e-9, 2.0)
    X, Z = np.meshgrid(g, g)
    cen = feeds.pos.mean(0)
    cells = np.c_[X.ravel(), Z.ravel()] + cen
    seen = np.zeros((k, len(cells)), bool)
    for i in range(k):
        d = cells - feeds.pos[i]
        r = np.hypot(d[:, 0], d[:, 1])
        off = wrap_deg(np.degrees(np.arctan2(d[:, 0], d[:, 1])) - feeds.yaw[i])
        inw = (np.abs(off) <= camera.HFOV_DEG / 2) & (r <= rmax)
        j = np.clip(np.round((off + camera.HFOV_DEG / 2) / step_deg).astype(int), 0, len(rel) - 1)
        seen[i] = inw & (r <= vis[i, j])
    cell_area = 4.0
    area = seen.sum(1) * cell_area
    others = np.array([np.delete(seen, i, axis=0).any(0) if k > 1 else np.zeros(len(cells), bool) for i in range(k)])
    unique = (seen & ~others).sum(1) * cell_area
    useful = int(((wall < 0.5) & (unique >= 0.1 * np.maximum(area, 1e-9))).sum())
    return dict(wall_frac=wall, unique_area=unique, area=area, useful=useful)
