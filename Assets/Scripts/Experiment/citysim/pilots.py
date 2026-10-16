"""The search study's pilots (citysim.search): where a pilot who does not know where the goal is means to go, how the
single drone's heading and the swarm's body yaw follow the stick, and where the swarm pilot's head looks.

INTENT. A pilot searching an unknown city flies straight for a while, turns, and flies on: a chain of intent targets,
each a run of length l = MIN_STEP_M + Exp(mean - MIN_STEP_M) from the last one at a heading turned by an angle theta from the last heading.
  rw    the random walker: mean RW_RUN_M (sensitivities RW_RUN_SENS), theta uniform on (-180, 180] (run and tumble),
        full stick, no pauses
  fit   internal_3's pilots while still searching (citysim.pilotfit, per interface): their mean run, their turn angles
        (resampled, sign at random), their pauses at a turn (probability per turn, durations resampled), their stick
The chain is laid target to target, from the start, independent of the flight -- every arm of a scenario draws the
same chain from the same seed, and a chain can be flown again on the Unity bench. A target that leaves the arena is
reflected back in (the heading mirrored with it), then snapped to TargetMask: free ground the single drone's planner
can stand on, in the arena's largest connected open region. How the pilot gets from one target to the next is the
interface's path rule (citysim.search's planners); the chain itself knows nothing about buildings.

HEADING. Turned toward a desired bearing at no more than a rate (deg/s):
  single drone   the bearing it is flown along (the view is slaved to it), at the airframe's 75 deg/s (rw) or the
                 fitted p90 turn rate (fit); `scan` adds dynamic.scan_offset's look-round sweeps to the VIEW only (the
                 stick stays on the path) -- internal_3's single-drone pilots barely looked off their course (3 % of the
                 time more than 30 deg), so scanning is a sensitivity arm, not the fitted behaviour
  swarm          the rig's body yaw to the bearing flown, at BODY_YAW_RATE (rw) or the fitted rate (fit)

HEAD (swarm, scoring only: it changes what counts as seen, not the flight). Relative to the body yaw:
  ahead   straight ahead throughout
  fit     internal_3's dwells resampled: (angle, duration) pairs drawn i.i.d. from the pilots' own, joined by moves at
          their median head speed
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera, routing  # noqa: E402
from citysim.los import wrap_deg  # noqa: E402

RW_RUN_M = 120.0
RW_RUN_SENS = (60.0, 240.0)
MIN_STEP_M = 20.0                # the shortest run, and the least distance between consecutive targets (more than
#                                  twice search.TARGET_CAPTURE_M, so no target is reached the moment it is set)
TARGET_MARGIN_M = 25.0           # intent targets stay this far inside the arena
MASK_CELL_M = 2.0
INTENTS = ("rw", "fit")
HEAD_MODELS = ("ahead", "fit")


# ============================================================================= where targets may land

class TargetMask:
    """Cells (MASK_CELL_M) of the arena where an intent target may land: open to the single drone's planner (every
    avoidance cylinder plus routing.SINGLE_CLEARANCE_M closed), in the largest 8-connected open region, at least
    TARGET_MARGIN_M inside the arena."""

    def __init__(self, cyl, arena, h=MASK_CELL_M, margin=TARGET_MARGIN_M):
        from scipy import ndimage
        x0, x1, z0, z1 = (float(v) for v in arena)
        self.arena = (x0, x1, z0, z1)
        self.margin = margin
        g = routing.occupancy(cyl, "single", (x0, x1, z0, z1), h=h)
        free = ~g.occ
        lab, n = ndimage.label(free, structure=np.ones((3, 3), int))
        if n:
            sizes = ndimage.sum(free, lab, index=np.arange(1, n + 1))
            free &= lab == (1 + int(np.argmax(sizes)))
        ii, jj = np.meshgrid(np.arange(g.nx), np.arange(g.nz), indexing="ij")
        c = g.centre(ii, jj)
        free &= (c[..., 0] >= x0 + margin) & (c[..., 0] <= x1 - margin) & \
                (c[..., 1] >= z0 + margin) & (c[..., 1] <= z1 - margin)
        self.g, self.ok = g, free
        self.share = float(free.mean())
        _, (self.ni, self.nj) = ndimage.distance_transform_edt(~free, return_indices=True)

    def snap(self, p):
        """The centre of the allowed cell nearest p (p's own cell when allowed)."""
        i, j = self.g.cell(np.asarray(p, float))
        i = int(np.clip(i[0], 0, self.g.nx - 1))
        j = int(np.clip(j[0], 0, self.g.nz - 1))
        return self.g.centre(self.ni[i, j], self.nj[i, j]).astype(float)

    def contains(self, p):
        i, j = self.g.cell(np.asarray(p, float))
        ok = self.g.inside(i, j)
        return bool(ok[0] and self.ok[int(i[0]), int(j[0])])

    def random_near(self, p, r, rng):
        """(an allowed cell at MIN_STEP_M .. r from p, at random -- any allowed cell when none is --, its bearing)."""
        if not hasattr(self, "_cells"):
            ii, jj = np.nonzero(self.ok)
            self._cells = self.g.centre(ii, jj)
        d = np.linalg.norm(self._cells - p, axis=1)
        pool = np.flatnonzero((d >= MIN_STEP_M) & (d <= r))
        if not len(pool):
            pool = np.flatnonzero(d >= MIN_STEP_M) if (d >= MIN_STEP_M).any() else np.arange(len(d))
        q = self._cells[int(rng.choice(pool))].astype(float)
        v = q - p
        return q, float(np.degrees(np.arctan2(v[0], v[1])))


# ============================================================================= intent

def _dir(theta_deg):
    t = math.radians(theta_deg)
    return np.array([math.sin(t), math.cos(t)])


def reflect(p, theta, arena, margin=TARGET_MARGIN_M):
    """p folded back inside the arena (shrunk by margin) like a ray off a mirror, and the heading mirrored with it."""
    x0, x1, z0, z1 = arena
    lo = np.array([x0 + margin, z0 + margin])
    hi = np.array([x1 - margin, z1 - margin])
    p = np.asarray(p, float).copy()
    for _ in range(8):
        moved = False
        if p[0] < lo[0] or p[0] > hi[0]:
            p[0] = 2 * lo[0] - p[0] if p[0] < lo[0] else 2 * hi[0] - p[0]
            theta = -theta
            moved = True
        if p[1] < lo[1] or p[1] > hi[1]:
            p[1] = 2 * lo[1] - p[1] if p[1] < lo[1] else 2 * hi[1] - p[1]
            theta = 180.0 - theta
            moved = True
        if not moved:
            break
    return np.clip(p, lo, hi), float(wrap_deg(theta))


class Intent:
    """An endless chain of intent targets from `start`; target(k) is the k-th (target(0) = start), pause_s(k) the pause
    the pilot makes on reaching it. Deterministic in its seed and independent of the flight."""

    def __init__(self, kind, start, heading0, mask, seed, run_mean_m=RW_RUN_M, turns=None, pauses=None, pause_p=0.0,
                 stick=1.0):
        if kind not in INTENTS:
            raise ValueError(f"unknown intent {kind!r} (one of {INTENTS})")
        self.kind, self.mask, self.run_mean = kind, mask, float(run_mean_m)
        self.turns = np.abs(np.asarray(turns if turns is not None else [], float))
        self.pauses = np.asarray(pauses if pauses is not None else [], float)
        self.pause_p, self.stick = float(pause_p), float(stick)
        self.rng = np.random.default_rng(seed)
        self.targets = [mask.snap(start)]
        self.headings = [float(heading0)]
        self.pause = [0.0]

    def _turn(self):
        if self.kind == "rw" or not len(self.turns):
            return float(self.rng.uniform(-180.0, 180.0))
        return float(self.rng.choice(self.turns) * self.rng.choice([-1.0, 1.0]))

    def _extend(self):
        cur, th0 = self.targets[-1], self.headings[-1]
        th0 = float(wrap_deg(th0 + self._turn())) if len(self.targets) > 1 else th0
        # MIN_STEP_M plus an exponential: the run's mean stays run_mean, and no run ends inside the capture radius
        run = MIN_STEP_M + float(self.rng.exponential(max(self.run_mean - MIN_STEP_M, 1.0)))
        # A target the snap puts back within MIN_STEP_M of the last one (a pocket of the mask, a corner of the arena)
        # would be "reached" on arrival and the pilot would re-plan every tick: try the other three ways, then any
        # allowed cell at about the run's length.
        q = None
        for turn in (0.0, 90.0, -90.0, 180.0):
            p, th = reflect(cur + run * _dir(th0 + turn), th0 + turn, self.mask.arena, self.mask.margin)
            q = self.mask.snap(p)
            if np.linalg.norm(q - cur) >= MIN_STEP_M:
                break
        else:
            q, th = self.mask.random_near(cur, max(run, 2 * MIN_STEP_M), self.rng)
        pause = float(self.rng.choice(self.pauses)) if (len(self.pauses) and self.rng.random() < self.pause_p) else 0.0
        self.targets.append(q)
        self.headings.append(th)
        self.pause.append(pause)

    def target(self, k):
        while len(self.targets) <= k:
            self._extend()
        return self.targets[k]

    def pause_s(self, k):
        self.target(k)
        return self.pause[k]


def intent_params(kind, interface, model=None, run_mean_m=None):
    """Intent keyword arguments for an interface ("swarm" / "single") under a pilot kind, from the fitted model."""
    if kind == "rw":
        return dict(run_mean_m=run_mean_m or RW_RUN_M)
    e = model["conditions"]["Swarm" if interface == "swarm" else "SingleDrone"]
    return dict(run_mean_m=run_mean_m or e["run"]["mean_m"], turns=e["turn"]["angles_deg"],
                pauses=e["pause"]["durations_s"], pause_p=e["pause"]["per_turn"],
                stick=float(e["stick"]["p50"]) if e.get("stick") else 1.0)


def heading_rate(kind, interface, model=None):
    """deg/s the single drone's heading / the swarm's body yaw turns at."""
    cap = camera.DRONE_MAX_YAW_RATE_DEG if interface == "single" else camera.BODY_YAW_RATE_DEG
    if kind == "rw" or model is None:
        return cap
    e = model["conditions"]["Swarm" if interface == "swarm" else "SingleDrone"]
    r = e["turn"].get("rate_p90")
    return float(min(cap, r)) if r and math.isfinite(r) else cap


def turn_toward(h, target, rate, dt):
    """Heading h (deg) turned toward target at no more than rate deg/s."""
    if h is None:
        return float(wrap_deg(target))
    d = float(wrap_deg(target - h))
    return float(wrap_deg(h + max(-rate * dt, min(rate * dt, d))))


# ============================================================================= the swarm pilot's head

class HeadProcess:
    """The head's yaw relative to the body yaw over time: `ahead` (0) or `fit` (resampled dwells joined by moves)."""

    def __init__(self, kind, seed, model=None, horizon_s=4000.0):
        if kind not in HEAD_MODELS:
            raise ValueError(f"unknown head model {kind!r} (one of {HEAD_MODELS})")
        self.kind = kind
        if kind == "ahead":
            return
        h = model["conditions"]["Swarm"]["head"]
        dw = np.asarray(h["dwells"], float).reshape(-1, 2)
        speed = float(h["move_speed_deg_s"]) if math.isfinite(h.get("move_speed_deg_s", math.nan)) else 60.0
        rng = np.random.default_rng(seed)
        t, a = 0.0, None
        ts, ys = [], []
        # start part-way into a dwell, so t = 0 is not always a dwell's start
        while t < horizon_s:
            ang, dur = dw[rng.integers(len(dw))]
            if a is not None:
                step = float(wrap_deg(ang - a))         # the short way round, unwrapped so interp follows it
                ts.append(t)
                ys.append(a)
                t += abs(step) / max(speed, 1e-6)
                ang = a + step
            else:
                dur *= rng.random()
            ts.append(t)
            ys.append(ang)
            t += dur
            a = ang
        ts.append(t)
        ys.append(a)
        self.t = np.array(ts)
        self.y = np.array(ys)

    def __call__(self, t):
        if self.kind == "ahead":
            return 0.0
        return float(wrap_deg(np.interp(t, self.t, self.y)))
