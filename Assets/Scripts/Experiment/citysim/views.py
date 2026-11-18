"""S12 Distributed Views (the views study): the trials' whole task, flown three ways in the trial city, to measure what
the swarm's distributed viewpoints add over one drone with a 360 camera, and over one drone with the ordinary camera.

    python -B -m citysim.views pilot --name N [--sessions 40] [--seeds 1]   # the initial tests -> views_study/<N>/
    python -B -m citysim.views calib --name N                               # the simulated pilots against internal_4
    python -B -m citysim.views tables --name N                              # the tables again from runs.jsonl

THE TASK is internal_4's (docs/trials/internal_4.md), not a point: three goals drawn from the city's own buildings as
BlockGoals draws them at Play (a building and the ring of buildings across the streets from it, at least 100 m from the
spawn, 150 m apart, one from each third of the spawn distances), each ring building walked round by 16 pedestrians at
2 m/s 1.1-1.5 m off its facade, one of them in a hat. The pilot does not know where the goals are. The clock starts
when the first drone crosses the start line 30 m east of the spawn and stops at the press on the third hat; a goal's
time splits at its zone (the patch disc grown by 15 m, analyse.py's phase zone) into BETWEEN (the previous press to
the zone) and AT the goal (the zone to the press).

WHAT THE PILOT SEES is decided per sample from the geometry and two hazards fitted to internal_4
(citysim.views_human), each on the HEAD basis: what was on a shown screen inside the HMD's view.
  noticing   a goal's crowd is noticed at mu(d) per second, d the distance within which at least one of its walkers
             is expected on the head's screens (discovery's expected walkers, a clear 2-D line, the image's full frame
             and the gimbal's band)
  recognition  the hat is recognised at lambda(d) per second, d its distance from the nearest camera that has it in
             the central 80% of a shown image (both axes, at the gimbal pitch; the sphere has no frame) with a clear
             line and inside the head's view; the press follows LATENCY_S later.
Both are drawn as exponential thresholds per goal (common random numbers: every condition of a session faces the same
thresholds, goals, hats and walkers).

THE CONDITIONS, flown on the replica (ContactSim, the scene's own SwarmManager: BlockShapedPatchCityWorld's law since
S11, d_ref 2.0, the ellipse, the slide, the adaptive spread with Hold):
  swarm    ten drones, the hull drones' feeds on the ring (display.FeedLatch), the body yaw the rig
  single   one drone, its one forward camera on the screen in front
  sphere   one drone, a 360 camera shown on a sphere round the pilot (the head looks wherever it likes in the city)
Each flies its own pilot, built from what internal_4's pilots did in that condition (views_human behaviour, head):
  between goals  a persistent walk on the street network until a crowd is noticed, then the street route to it
  at a goal      swarm: to the centre building and a hold on it (the formation wraps it), then the circuit of its block's
                 corners with pauses beside the building, a hold again, and so on (S11's search, calibrated on ERKS);
                 single: round the ring buildings one by one, each orbited part of the way with the camera turned in on
                 it, a stop to look round after some; sphere: a walk over the goal's own junctions
  the head       a recorded head trace of that condition and phase, played from a random point (block bootstrap)
  altitude, gimbal  each condition's internal_4 medians per phase (BAND)

COUNTERFACTUAL BASES on the same flights, which decompose the swarm's lead without flying anything new: at every
recognition check the hazard is also integrated over the at-goal window on
  swarm:  `centre`   the same feeds (yaws, head, screens) moved to the formation's centroid: the swarm without its
                     distributed viewpoints
          `sphere_c` a 360 camera at the centroid with the sphere's head: one drone with a 360 camera on the swarm's
                     flight
          `screen`   every shown feed, wherever the head was (perfect attention)
  sphere: `fwd`      a forward camera on the sphere drone's heading: the sphere's flight without the 360
          `screen`   every bearing (perfect attention)
  single: `sphere_s` a 360 camera on the single drone's flight with the sphere's head
Reported as recognitions per minute at the goal (the integrated hazard over the window, divided by its length).
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402

import numpy as np  # noqa: E402

from citysim import REPO, blockcity, camera, display, practicecity  # noqa: E402
from citysim import swarmtune as T  # noqa: E402
from citysim.los import Prism, wrap_deg  # noqa: E402

rep = T.rep
OUT = REPO / "experiment_results" / "views_study"
HUMAN = OUT / "human"
CONDITIONS = ("swarm", "single", "sphere")
HUMAN_COND = {"swarm": "Swarm", "single": "SingleDrone", "sphere": "SingleSphere"}
CITY_SEED = 1                     # the trial city (BlockShapedPatchCityWorld's layout)
HELD_SEEDS = (2, 3, 4)
N_SWARM = 10
D_REF = 2.0
# (altitude m, gimbal pitch deg) per phase: internal_4's medians (views_human, the centroid's gtY and _head's gimbalPitch)
BAND = {"swarm": dict(between=(12.2, -2.0), goal=(10.4, -4.8)),
        "single": dict(between=(7.9, -4.3), goal=(7.0, -13.5)),
        "sphere": dict(between=(19.8, 0.0), goal=(14.3, 0.0))}
CAP_S = 900.0                     # task seconds; a session not done by then is censored (internal_4's longest: 694 s)
SETTLE_S = 15.0
LINE_M = 30.0                     # StartLine: 30 m ahead of the spawn
ZONE_EXTRA_M = 15.0
LATENCY_S = 1.0                   # recognition -> press (the hazard fit's latency)
REC_EVERY_S = 0.2
NOTICE_EVERY_S = 1.0
LATCH_S = 0.1
MIN_ALIVE = 4
R_HAT_MAX = 130.0
R_CROWD_MAX = 350.0               # WalkerPatrol.SleepDistance: beyond it the walkers are not drawn
CENTRAL = 0.8
SCREEN_SCALE = math.degrees(2 * math.atan(1.3 * camera.FEED_ASPECT / 2 / 2.0)) / camera.HFOV_DEG   # 60 deg quads
HMD_HALF = 50.0
YAW_RATE = 40.0                   # bodyYawRate: the stick turns the pilot at 40 deg/s in every condition
N_WALKERS = 16
WALK_GAP = (1.1, 1.5)             # ProceduralCityBuilder.prismWalkOffset
WALK_SPEED = 2.0
CROWD_STEP_M = 3.0
CAPTURE_M = 7.0
# the pilots' at-goal and between-goal parameters, set from internal_4 (views_human behaviour; calib checks them)
PILOT = {
    "swarm": dict(hold_s=(12.0, 22.0), hold_stick=0.3, pause_p=0.5, pause_s=(8.0, 16.0), pause_f=(0.4, 1.2),
                  pause_stick=0.5, pan_every_s=12.0, pan_deg=(45.0, 135.0), stick=1.0),
    "single": dict(orbit_p=0.5, orbit_gap=9.0, orbit_frac=(0.5, 1.0), face_p=0.6, pass_sides=(1, 2), stop_p=0.5,
                   stop_s=(6.0, 14.0), stick=0.85, sweep_rate=30.0),
    "sphere": dict(stop_p=0.15, stop_s=(3.0, 8.0), stick=0.9),
}
WALK_KAPPA = 2.0                  # the between-goal walk's persistence: P(next street) ~ exp(kappa cos(turn))
WALK_EXPLORE = 3.0                # ... and its pull towards ground not yet flown near (exploration)
SEEN_M = 80.0                     # a junction counts as explored once the centroid has passed within this
EXPLORE_M = 150.0                 # unexplored junctions within this of a candidate count for it
# Recognition models (views_human's fits): `pooled` one hazard on the head basis for every condition (the same eyes
# on every display); `ph` a hazard per condition with one shape (proportional hazards); `screen` the pooled fit on the
# screens' basis with no head test (perfect attention).
REC_MODELS = ("pooled", "ph", "screen", "start")


# ============================================================================= the human fits

_FITS = {}


def fits():
    """The recognition and noticing hazards (views_human's JSON) and the head traces."""
    if not _FITS:
        with open(HUMAN / "recognition_fits.json", encoding="utf-8") as f:
            rec = json.load(f)
        with open(HUMAN / "discovery_fits.json", encoding="utf-8") as f:
            dis = json.load(f)
        with open(HUMAN / "head_series.json", encoding="utf-8") as f:
            heads = json.load(f)
        _FITS.update(rec=rec, dis=dis, heads=heads)
    return _FITS


def hill(d, a0, d50, k):
    with np.errstate(over="ignore"):
        return a0 / (1.0 + (np.asarray(d, float) / d50) ** k)


def rec_params(model="pooled", cond="swarm", d50_scale=1.0):
    """(lam0, d50, k) of the recognition hazard, and whether it is on the head basis."""
    R = fits()["rec"]
    if model == "ph":
        f = R["ph|d_head|lat1.0"]["fit"]
        return (f["lam0"][HUMAN_COND[cond]], f["d50"] * d50_scale, f["k"]), True
    if model == "start":            # recognised at the start of the final sighting (pilots who closed in to verify)
        f = R["start|d_head|all"]["fit"]
        return (f["lam0"], f["d50"] * d50_scale, f["k"]), True
    f = R["d_screen|lat1.0|all" if model == "screen" else "d_head|lat1.0|all"]["fit"]
    return (f["lam0"], f["d50"] * d50_scale, f["k"]), model != "screen"


def notice_params(key="d_head|lat1.0", d50_scale=1.0):
    f = fits()["dis"][key]["fit"]
    return (f["mu0"], f["d50"] * d50_scale, f["k"])


class HeadPlayer:
    """A recorded head trace (offset from the view, deg) of one condition and phase, played from a random point; at
    a trace's end another starts at a random point."""

    def __init__(self, cond, rng):
        self.series = fits()["heads"][HUMAN_COND[cond]]
        self.rng = rng
        self.cur, self.phase, self.i = None, None, 0

    def __call__(self, phase):
        ph = "search" if phase == "goal" else "transit"
        if self.cur is None or ph != self.phase or self.i >= len(self.cur):
            pool = self.series[ph]
            self.cur = pool[int(self.rng.integers(len(pool)))]
            self.i = int(self.rng.integers(max(len(self.cur) - 1, 1)))
            self.phase = ph
        v = self.cur[self.i]
        return float(v)

    def advance(self, dt):
        self.i += max(int(round(dt / 0.1)), 1)


# ============================================================================= goals and walkers

def offset_path(poly, gap):
    """(points (M, 2), cumulative length (M,)) of WalkerPatrol's Polygon path: the footprint grown by `gap`, its
    corners rounded at that gap, counter-clockwise, closed."""
    pr = Prism(poly, 0.0, 1.0)
    P, (n, _) = pr.poly, pr.planes()
    path = []
    for i in range(len(P)):
        a0 = math.atan2(n[i - 1, 1], n[i - 1, 0])
        turn = (math.atan2(n[i, 1], n[i, 0]) - a0) % (2 * math.pi)
        th = a0 + turn * np.linspace(0.0, 1.0, 9)
        path.append(P[i] + gap * np.c_[np.cos(th), np.sin(th)])
    path = np.vstack(path + [path[0][:1]])
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path, cum


def path_at(path, cum, s):
    s = np.mod(s, cum[-1])
    return np.c_[np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])]


def draw_goals(C, rng, count=3):
    """BlockGoals.Draw on the layout, from numpy's generator: [goal dict] nearest the spawn first."""
    L, blocks = C["L"], C["blocks"]
    by_id = {b["id"]: b for b in blocks}
    ring = practicecity.neighbours(L.streets)
    spawn = np.array(L.layout["spawn"], float)[[0, 2]]
    cand = practicecity.candidates(L, blocks)
    centre = {k: practicecity.block_centre(L, by_id[k]) for k in cand}
    cand.sort(key=lambda k: (float(np.sum((centre[k] - spawn) ** 2)), k))

    def apart(pick):
        for i in range(len(pick)):
            for j in range(i + 1, len(pick)):
                if np.linalg.norm(centre[pick[i]] - centre[pick[j]]) < 150.0:
                    return False
                a = set(ring[pick[i]]) | {pick[i]}
                if pick[j] in a or a & set(ring[pick[j]]):
                    return False
        return True

    strata = np.array_split(np.array(cand), count)
    chosen = None
    for _ in range(2000):
        pick = [int(s[rng.integers(len(s))]) for s in strata]
        if apart(pick):
            chosen = pick
            break
    if chosen is None:
        for _ in range(5000):
            pick = [int(x) for x in rng.choice(cand, count, replace=False)]
            if apart(pick):
                chosen = sorted(pick, key=cand.index)
                break
    if chosen is None:
        raise RuntimeError("no goals")
    goals = []
    for gi, k in enumerate(chosen):
        c = centre[k]
        rb = sorted(ring[k])
        polys = [blockcity.building_poly(L.boxes[by_id[j]["building"]]) for j in rb]
        patch = max(float(np.max(np.linalg.norm(p - c, axis=1))) for p in polys)
        r_g = float(np.mean([np.linalg.norm(p.mean(0) - c) for p in polys]))
        own = blockcity.building_poly(L.boxes[by_id[k]["building"]])
        # the hat: a random patrol of the ring, a random walker of its 16 (GoalSpecialWalker; WalkerPatrol)
        h = int(rng.integers(len(polys)))
        wi = int(rng.integers(N_WALKERS))
        gap = WALK_GAP[0] + rng.uniform(0.0, WALK_GAP[1] - WALK_GAP[0])
        path, cum = offset_path(polys[h], gap)
        from citysim import discovery
        pts, w = discovery.polygon_loop_points(polys, n_walkers=N_WALKERS, step=CROWD_STEP_M)
        goals.append(dict(index=gi, block=k, centre=c, ring_blocks=rb, polys=polys, own=own, patch=patch, r_g=r_g,
                          zone=patch + ZONE_EXTRA_M, hat_b=h, hat_path=path, hat_cum=cum,
                          hat_s0=float(rng.uniform(0.0, cum[-1])), hat_sign=-1.0 if wi < N_WALKERS // 2 else 1.0,
                          crowd=(pts, w)))
    return goals


def hat_pos(g, t):
    return path_at(g["hat_path"], g["hat_cum"], g["hat_s0"] + g["hat_sign"] * WALK_SPEED * t)[0]


# ============================================================================= what is seen

def _cam_frame(yaws, pitch):
    psi = np.radians(yaws)
    th = math.radians(pitch)
    fwd = np.stack([np.sin(psi) * math.cos(th), np.full_like(psi, math.sin(th)), np.cos(psi) * math.cos(th)], -1)
    right = np.stack([np.cos(psi), np.zeros_like(psi), -np.sin(psi)], -1)
    up = np.stack([-np.sin(psi) * math.sin(th), np.full_like(psi, math.cos(th)), -np.cos(psi) * math.sin(th)], -1)
    return fwd, right, up


TX = math.tan(math.radians(camera.HFOV_DEG / 2))
TY = math.tan(math.radians(camera.VFOV_DEG / 2))


def head_ok(kind, cams, yaws, body, head, target):
    """(K,) whether the target is inside the HMD's view: screens (`swarm`) by the display azimuth against the head's
    offset from the body yaw; the single drone's one screen always; the sphere (`sphere`) by the bearing against the
    head's world yaw."""
    rel = target - cams
    bearing = np.degrees(np.arctan2(rel[:, 0], rel[:, 1]))
    if kind == "single":
        return np.ones(len(cams), bool)
    if kind == "sphere":
        return np.abs(wrap_deg(bearing - head)) <= HMD_HALF
    off = wrap_deg(bearing - yaws)
    az = wrap_deg(yaws - body + off * SCREEN_SCALE)
    return np.abs(wrap_deg(az - head)) <= HMD_HALF


def hat_distance(fp, cams, height, pitch, yaws, target, kind, body, head, use_head=True, allround=False):
    """The nearest camera's slant distance to the hat when it is in the central part of the image (or anywhere, for
    a 360 camera), with a clear 2-D line, and (use_head) inside the HMD's view; NaN otherwise."""
    if not len(cams):
        return math.nan
    rel = target - cams
    hd = np.hypot(rel[:, 0], rel[:, 1])
    dz = camera.HAT_HEIGHT_M - height
    dist = np.hypot(hd, dz)
    ok = dist <= R_HAT_MAX
    if not ok.any():
        return math.nan
    if not allround:
        fwd, right, up = _cam_frame(yaws, pitch)
        r3 = np.c_[rel[:, 0], np.full(len(rel), dz), rel[:, 1]]
        z = (r3 * fwd).sum(1)
        x = (r3 * right).sum(1) / np.maximum(z, 1e-6)
        y = (r3 * up).sum(1) / np.maximum(z, 1e-6)
        ok &= (z > 0) & (np.abs(x) <= CENTRAL * TX) & (np.abs(y) <= CENTRAL * TY)
    if use_head:
        ok &= head_ok(kind, cams, yaws, body, head, target)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return math.nan
    idx = idx[np.argsort(dist[idx])]
    for i in idx:                      # nearest first: the first clear line is the answer
        if not fp.blocked(cams[i][None], target[None])[0]:
            return float(dist[i])
    return math.nan


def crowd_distance(fp, cams, height, pitch, yaws, pts, w, kind, body, head, allround=False):
    """The distance within which at least one of the crowd's walkers is expected on the head's screens (a full image,
    the gimbal's band, a clear line); NaN when none within R_CROWD_MAX."""
    if not len(cams) or not len(pts):
        return math.nan
    best = np.full(len(pts), np.inf)
    centre = pts.mean(0)
    for k in range(len(cams)):
        cam = cams[k]
        rel = pts - cam
        hd = np.hypot(rel[:, 0], rel[:, 1])
        dist = np.hypot(hd, camera.HAT_HEIGHT_M - height)
        ok = (dist <= R_CROWD_MAX) & (dist < best)
        if not ok.any():
            continue
        bearing = np.degrees(np.arctan2(rel[:, 0], rel[:, 1]))
        if not allround:
            off = wrap_deg(bearing - yaws[k])
            elev = np.degrees(np.arctan2(camera.HAT_HEIGHT_M - height, np.maximum(hd, 1e-6)))
            ok &= (np.abs(off) <= camera.HFOV_DEG / 2) & (np.abs(elev - pitch) <= camera.VFOV_DEG / 2)
        if kind == "sphere":
            ok &= np.abs(wrap_deg(bearing - head)) <= HMD_HALF
        elif kind == "swarm":
            off = wrap_deg(bearing - yaws[k])
            ok &= np.abs(wrap_deg(wrap_deg(yaws[k] - body + off * SCREEN_SCALE) - head)) <= HMD_HALF
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        mid = 0.5 * (cam + centre)
        near = fp.near(mid, 0.5 * float(np.linalg.norm(centre - cam)) + 80.0)
        sub = fp.subset(near) if len(near) else None
        clear = ~sub.blocked(np.broadcast_to(cam, (len(idx), 2)), pts[idx]) if sub is not None and sub.n else \
            np.ones(len(idx), bool)
        v = idx[clear]
        best[v] = np.minimum(best[v], dist[v])
    o = np.argsort(best)
    cw = np.cumsum(w[o])
    k = int(np.searchsorted(cw, 1.0 - 1e-9))
    return float(best[o][k]) if k < len(o) and np.isfinite(best[o][k]) else math.nan


# ============================================================================= the pilot

class Task:
    """One step of a pilot's plan: go to a point (`go`), or hold on one for a while (`hold`), with a view policy:
    `course` (the view turns to the direction of travel), `face` a point, or `pan` (the pilot pans round)."""

    def __init__(self, kind, p, stick=1.0, dur=0.0, view="course", face=None, capture=CAPTURE_M, timeout=None):
        self.kind, self.p, self.stick, self.dur, self.view = kind, np.asarray(p, float), stick, dur, view
        self.face, self.capture, self.timeout, self.t0 = face, capture, timeout, None


class Pilot:
    """The searching pilot of one condition (see the module docstring), as a queue of Tasks re-planned by mode."""

    def __init__(self, cond, C, goals, rng, start_heading, explore=True, pattern="circuit", fly_as=None):
        self.cond, self.C, self.goals, self.rng = cond, C, goals, rng
        self.style = fly_as or cond          # whose way of searching a goal it flies (the sphere can fly as `single`)
        self.pattern = pattern              # the swarm at a goal: `circuit` (S11's search) or `wrap` (hold on it)
        self.net = C["net"]
        self.P = PILOT[self.style]
        self.q = deque()
        self.mode = "search"
        self.goal = None                       # the goal being approached or searched
        self.view = float(start_heading)
        self.view_target = float(start_heading)
        self.last_j, self.prev_j = None, None
        self.pan_next = 0.0
        self.visits = {}
        self.ring_order = None
        self.seen = np.zeros(len(self.net.jpos), bool)
        self.explore = explore

    def mark_seen(self, c):
        self.seen |= np.hypot(*(self.net.jpos - c).T) <= SEEN_M

    # ---------------------------------------------------------------- plans
    def _walk(self, c):
        """The next junction of the street walk: persistent (WALK_KAPPA on the turn) and drawn towards streets not yet
        flown near (WALK_EXPLORE on the share of the junctions within EXPLORE_M of a candidate that are unexplored),
        never straight back unless it is a dead end."""
        net = self.net
        j = self.last_j if self.last_j is not None else net.nearest_junction(c)
        if self.last_j is None:
            self.last_j = j
            self.q.append(Task("go", net.jpos[j], 1.0, timeout=60.0))
            return
        nb = [v for v, _, _ in net.adj[j]]
        if len(nb) > 1 and self.prev_j in nb:
            nb.remove(self.prev_j)
        here = net.jpos[j]
        came = here - net.jpos[self.prev_j] if self.prev_j is not None else np.array(
            [math.sin(math.radians(self.view)), math.cos(math.radians(self.view))])
        came = came / max(float(np.linalg.norm(came)), 1e-9)
        w = []
        for v in nb:
            d = net.jpos[v] - here
            d = d / max(float(np.linalg.norm(d)), 1e-9)
            s = WALK_KAPPA * float(d @ came)
            if self.explore:
                near = np.hypot(*(net.jpos - (here + EXPLORE_M * d)).T) <= EXPLORE_M
                s += WALK_EXPLORE * float((~self.seen[near]).mean()) if near.any() else 0.0
            w.append(math.exp(s) * (0.25 if v in self.visits else 1.0))
        w = np.array(w) / np.sum(w)
        nxt = int(nb[int(self.rng.choice(len(nb), p=w))])
        self.visits[nxt] = self.visits.get(nxt, 0) + 1
        self.prev_j, self.last_j = j, nxt
        self.q.append(Task("go", net.jpos[nxt], 1.0, timeout=60.0))

    def approach(self, g, c):
        """The street route from here to the corner of g's block nearest to here."""
        net = self.net
        self.q.clear()
        self.mode, self.goal = "approach", g
        j0 = net.nearest_junction(c)
        corners = net.blocks[g["block"]]["corners"]
        j1 = int(min(corners, key=lambda j: np.linalg.norm(net.jpos[j] - c)))
        js, _ = net.route(j0, j1)
        for j in (js[1:] if js and len(js) > 1 else [j1]):
            self.q.append(Task("go", net.jpos[j], 1.0, timeout=60.0))
        self.q.append(Task("go", g["centre"], 1.0, timeout=60.0, capture=12.0))

    def at_goal(self, g, c):
        self.q.clear()
        self.mode, self.goal = "goal", g
        self.ring_order = None
        self._goal_plan(c)

    def _goal_plan(self, c):
        g, P, rng, net = self.goal, self.P, self.rng, self.net
        if self.style == "swarm" and self.pattern == "wrap":
            # wrapped round the centre building for the whole search: to its centre, then a hold there, the rig panned
            self.q.append(Task("go", g["centre"], P["stick"], timeout=60.0, capture=10.0))
            self.q.append(Task("hold", g["centre"], P["hold_stick"], 60.0, view="pan"))
        elif self.style == "swarm":
            self.q.append(Task("go", g["centre"], P["stick"], timeout=60.0, capture=10.0))
            self.q.append(Task("hold", g["centre"], P["hold_stick"], float(rng.uniform(*P["hold_s"])), view="pan"))
            corners = np.array(net.blocks[g["block"]]["corners"])
            pos = net.jpos[corners]
            ang = np.arctan2(*(pos - g["centre"]).T[::-1])
            ring = [int(x) for x in corners[np.argsort(ang)]]
            k = int(rng.integers(len(ring)))
            ring = ring[k:] + ring[:k]
            if rng.random() < 0.5:
                ring = ring[::-1]
            Rb = float(np.max(np.linalg.norm(g["own"] - g["centre"], axis=1)))
            for j in ring:
                q = net.jpos[j]
                self.q.append(Task("go", q, 1.0, timeout=40.0))
                if rng.random() < P["pause_p"]:
                    u = (q - g["centre"]) / max(float(np.linalg.norm(q - g["centre"])), 1e-9)
                    p = g["centre"] + min(float(rng.uniform(*P["pause_f"])) * Rb, 16.0) * u
                    self.q.append(Task("hold", p, P["pause_stick"], float(rng.uniform(*P["pause_s"])), view="pan"))
        elif self.style == "single":
            if not self.ring_order:
                here = c if c is not None else g["centre"]
                cen = np.array([p.mean(0) for p in g["polys"]])
                ang = np.arctan2(*(cen - g["centre"]).T[::-1])
                order = list(np.argsort(ang))
                k = int(np.argmin(np.linalg.norm(cen[order] - here, axis=1)))
                order = order[k:] + order[:k]
                if rng.random() < 0.5:
                    order = [order[0]] + order[1:][::-1]
                self.ring_order = [int(i) for i in order]
            b = self.ring_order.pop(0)
            poly = g["polys"][b]
            here = c if c is not None else g["centre"]
            if rng.random() < P["orbit_p"]:
                # ALEX's way: round the building part of the way, often with the camera turned in on it
                path, cum = offset_path(poly, P["orbit_gap"])
                i0 = int(np.argmin(np.linalg.norm(path - here, axis=1)))
                frac = float(rng.uniform(*P["orbit_frac"]))
                sign = 1.0 if rng.random() < 0.5 else -1.0
                n = max(int(frac * cum[-1] / 8.0), 2)
                ss = cum[i0] + sign * np.linspace(0.0, frac * cum[-1], n + 1)
                face = poly.mean(0) if rng.random() < P["face_p"] else None
                for p in path_at(path, cum, ss):
                    self.q.append(Task("go", p, P["stick"], view="face" if face is not None else "course", face=face,
                                       timeout=20.0, capture=5.0))
                end = path_at(path, cum, ss[-1:])[0]
            else:
                # ERKS's way: along one or two of its block's streets, looking ahead
                blk = g["ring_blocks"][b]
                corners = list(net.blocks[blk]["corners"]) if blk in net.blocks else []
                if not corners:
                    corners = [net.nearest_junction(poly.mean(0))]
                pos = net.jpos[corners]
                ang = np.arctan2(*(pos - poly.mean(0)).T[::-1])
                ring = [corners[i] for i in np.argsort(ang)]
                k = int(np.argmin(np.linalg.norm(net.jpos[ring] - here, axis=1)))
                ring = ring[k:] + ring[:k]
                if rng.random() < 0.5:
                    ring = [ring[0]] + ring[1:][::-1]
                m = 1 + int(rng.integers(P["pass_sides"][0], P["pass_sides"][1] + 1))
                for jj in ring[:m]:
                    self.q.append(Task("go", net.jpos[jj], P["stick"], timeout=30.0))
                end = net.jpos[ring[min(m, len(ring)) - 1]]
            if rng.random() < P["stop_p"]:
                self.q.append(Task("hold", end, 0.3, float(rng.uniform(*P["stop_s"])), view="sweep"))
        else:
            area = [j for j in range(len(net.jpos)) if np.linalg.norm(net.jpos[j] - g["centre"]) <= g["patch"]]
            j = self.last_j if self.last_j in area else (min(area, key=lambda j: np.linalg.norm(
                net.jpos[j] - (c if c is not None else g["centre"]))) if area else None)
            if j is None:
                self.q.append(Task("go", g["centre"], P["stick"], timeout=30.0))
                return
            nb = [v for v, _, _ in net.adj[j] if v in area] or area
            cnt = np.array([self.visits.get(("g", v), 0) for v in nb], float)
            best = [v for v, k in zip(nb, cnt) if k == cnt.min()]
            nxt = int(best[int(rng.integers(len(best)))])
            self.visits[("g", nxt)] = self.visits.get(("g", nxt), 0) + 1
            self.last_j = nxt
            self.q.append(Task("go", net.jpos[nxt], P["stick"], timeout=40.0))
            if rng.random() < P["stop_p"]:
                self.q.append(Task("hold", net.jpos[nxt], 0.3, float(rng.uniform(*P["stop_s"]))))

    def resolved(self, c):
        """The goal being searched has been found: back to the walk."""
        self.q.clear()
        self.mode, self.goal = "search", None
        self.last_j, self.prev_j = self.net.nearest_junction(c), None

    # ---------------------------------------------------------------- the command
    def command(self, c, t, gain):
        """(stick (2,), view target yaw) for the centroid c at time t."""
        while True:
            if not self.q:
                if self.mode == "goal":
                    self._goal_plan(c)
                elif self.mode == "approach":
                    self.q.append(Task("go", self.goal["centre"], 1.0, timeout=60.0, capture=12.0))
                else:
                    self._walk(c)
            task = self.q[0]
            if task.t0 is None:
                task.t0 = t
            d = task.p - c
            dd = float(np.linalg.norm(d))
            if task.kind == "go":
                if dd < task.capture or (task.timeout and t - task.t0 > task.timeout):
                    self.q.popleft()
                    continue
                cmd = d / max(dd, 1e-9) * task.stick
            else:
                if t - task.t0 > task.dur:
                    self.q.popleft()
                    continue
                cmd = d * gain
                s = float(np.linalg.norm(cmd))
                if s > task.stick:
                    cmd *= task.stick / s
            break
        if task.view == "face" and task.face is not None:
            f = task.face - c
            self.view_target = math.degrees(math.atan2(f[0], f[1]))
        elif task.view in ("pan", "sweep"):
            if t >= self.pan_next:
                if task.view == "pan":
                    self.view_target = self.view + float(self.rng.choice([-1, 1]) * self.rng.uniform(
                        *self.P.get("pan_deg", (45.0, 135.0))))
                    self.pan_next = t + float(self.rng.exponential(self.P.get("pan_every_s", 12.0)))
                else:
                    self.view_target = self.view + 120.0 * float(self.rng.choice([-1, 1]))
                    self.pan_next = t + 4.0
        elif task.kind == "go" and dd > 1.0:
            self.view_target = math.degrees(math.atan2(d[0], d[1]))
        return cmd, self.view_target

    def turn_view(self, dt, rate=YAW_RATE):
        e = float(wrap_deg(self.view_target - self.view))
        self.view = float(wrap_deg(self.view + np.clip(e, -rate * dt, rate * dt)))


# ============================================================================= one session, one condition

def session_seed(city_seed, k):
    return int.from_bytes(hashlib.sha256(f"views|{city_seed}|{k}".encode()).digest()[:8], "little")


def fly(cond, city_seed, k, opts=None):
    """One session (the city, goal draw k) flown in one condition: a result dict."""
    opts = opts or {}
    t_wall = time.time()
    C = T.city(city_seed)
    seed = session_seed(city_seed, k)
    rng_g = np.random.default_rng(seed)
    goals = draw_goals(C, rng_g)
    E_rec = rng_g.exponential(1.0, len(goals))          # the thresholds, common to every condition of the session
    E_not = rng_g.exponential(1.0, len(goals))
    rng = np.random.default_rng(seed ^ int.from_bytes(cond.encode(), "little"))
    rec, on_head = rec_params(opts.get("rec_model", "pooled"), cond, opts.get("rec_d50_scale", 1.0))
    notice = notice_params(opts.get("notice_key", "d_head|lat1.0"), opts.get("notice_d50_scale", 1.0))
    # the calibrated variant: one efficiency factor per condition on each rate (cal_factors), set so the condition's
    # geometric-mean time at the goal and between goals match internal_4's
    rec = (rec[0] * opts.get("rec_scale", 1.0), rec[1], rec[2])
    probe = float(opts.get("probe_s", 0.0) or 0.0)
    notice = (notice[0] * opts.get("notice_scale", 1.0), notice[1], notice[2])
    fp = C["fp"]
    m = T.model({})
    L = C["L"]
    spawn = np.array(L.layout["spawn"], float)[[0, 2]]
    heading0 = float(L.layout.get("spawn_yaw", 90.0))
    east = np.array([math.sin(math.radians(heading0)), math.cos(math.radians(heading0))])
    swarm = cond == "swarm"
    P0 = T.start_positions(C, D_REF, int(seed % 100000), {}, spawn) if swarm else spawn[None].copy()
    sim = rep.ContactSim(m, C["rcity"], np.array(P0, float), np.zeros((len(P0), 2)), int(seed % 100000))
    while swarm and sim.t < SETTLE_S - 1e-9:
        sim.step(np.zeros(2), 0.0, D_REF)
    sim.contacts = sim.contact_ticks = 0
    sim.kills = []
    n0 = int(sim.alive.sum())
    t_off = sim.t
    pilot = Pilot(cond, C, goals, rng, heading0, explore=opts.get("explore", True),
                  pattern=opts.get("swarm_pattern", "circuit"), fly_as=opts.get("fly_as"))
    band = BAND[opts.get("band_as", cond)]
    pilot.q.append(Task("go", spawn + 120.0 * east, 1.0, timeout=60.0))
    head = HeadPlayer(cond, rng)
    latch = display.FeedLatch(len(sim.P), pitch_deg=BAND[cond]["between"][1]) if swarm else None
    feeds = None
    gain = T.S.HOLD_GAIN / m.max_speed
    line = None
    noticed = [None] * len(goals)
    zone_t = [None] * len(goals)
    rec_t = [None] * len(goals)
    rec_d = [math.nan] * len(goals)
    Lam = np.zeros(len(goals))
    Mu = np.zeros(len(goals))
    cf = {}                                              # counterfactual integrals per goal: {basis: [s, Lambda]}
    at_goal = [0.0] * len(goals)
    log = []                                             # (t, cx, cz, mode, goal, n alive, view, head)
    rec_every = max(1, int(round(REC_EVERY_S / m.dt)))
    not_every = max(1, int(round(NOTICE_EVERY_S / m.dt)))
    latch_every = max(1, int(round(LATCH_S / m.dt)))
    reason = "cap"
    stats = dict(moving=0.0, stopped=0.0, yawing=0.0, inside_ring=0.0, wrapped=0.0, goal_dist=[], speed=[])
    prev_view = pilot.view
    while True:
        t = sim.t - t_off
        if line is not None and t - line >= CAP_S:
            break
        if sim.alive.sum() < (MIN_ALIVE if swarm else 1):
            reason = "lost"
            break
        if all(r is not None for r in rec_t):
            reason = "done"
            break
        c = sim.P[sim.alive].mean(0)
        cmd, _ = pilot.command(c, t, gain)
        pilot.turn_view(m.dt)
        sim.step(cmd, 0.0, D_REF)
        if line is None and float(((sim.P[sim.alive] - spawn) @ east).max()) >= LINE_M:
            line = t
        if swarm and sim.tick % latch_every == 0:
            feeds = latch.step(sim.P, sim.alive, pilot.view, LATCH_S)
        if line is None:
            continue
        phase = "goal" if pilot.mode == "goal" else "between"
        height, pitch = band[phase]
        if latch is not None:
            latch.pitch = pitch
        # ---- zones
        for gi, g in enumerate(goals):
            if rec_t[gi] is None and np.linalg.norm(c - g["centre"]) <= g["zone"]:
                if zone_t[gi] is None:
                    zone_t[gi] = t
                if pilot.mode != "goal":
                    pilot.at_goal(g, c)
        if pilot.mode == "goal":
            gi = pilot.goal["index"]
            at_goal[gi] += m.dt
        if sim.tick % rec_every:
            continue
        dt_r = REC_EVERY_S
        head.advance(dt_r)
        hrel = head(phase)
        # ---- the cameras
        if swarm:
            if feeds is None:
                continue
            cams, yaws = feeds.pos, feeds.yaw
            body = pilot.view
            hd = hrel
            kind = "swarm"
        else:
            cams = sim.P[sim.alive][:1] + camera.CAMERA_FORWARD_M * np.array(
                [[math.sin(math.radians(pilot.view)), math.cos(math.radians(pilot.view))]])
            yaws = np.array([pilot.view])
            body = pilot.view
            kind = "single" if cond == "single" else "sphere"
            hd = wrap_deg(pilot.view + hrel) if kind == "sphere" else hrel
        allround = cond == "sphere"
        # ---- behaviour stats (the calibration's measures)
        v = float(np.linalg.norm(sim.V[sim.alive].mean(0)))
        if pilot.mode == "goal":
            g = pilot.goal
            stats["speed"].append(v)
            stats["goal_dist"].append(float(np.linalg.norm(c - g["centre"])))
            stats["moving" if v >= 1.0 else "stopped"] += dt_r
            yr = abs(float(wrap_deg(pilot.view - prev_view))) / dt_r
            stats["yawing"] += dt_r if yr > 10.0 else 0.0
            stats["inside_ring"] += dt_r if np.linalg.norm(c - g["centre"]) < g["r_g"] else 0.0
            if swarm:
                Pa = sim.P[sim.alive]
                ang = np.sort(np.arctan2(Pa[:, 1] - g["centre"][1], Pa[:, 0] - g["centre"][0]))
                gap = np.diff(np.r_[ang, ang[0] + 2 * math.pi]).max() if len(ang) >= 3 else 2 * math.pi
                stats["wrapped"] += dt_r if gap < math.pi else 0.0
        prev_view = pilot.view
        pilot.mark_seen(c)
        # ---- recognition
        for gi, g in enumerate(goals):
            if rec_t[gi] is not None:
                continue
            hp = hat_pos(g, t)
            here = pilot.mode == "goal" and pilot.goal["index"] == gi
            far = np.min(np.linalg.norm(cams - hp, axis=1)) > R_HAT_MAX
            if far and not here:
                continue
            d = math.nan if far else hat_distance(fp, cams, height, pitch, yaws, hp, kind, body, hd,
                                                  use_head=on_head, allround=allround)
            if math.isfinite(d):
                Lam[gi] += float(hill(d, *rec)) * dt_r
            if here:
                _counterfactuals(cf.setdefault(gi, {}), cond, fp, sim, cams, yaws, height, pitch, hp, body, hrel,
                                 pilot, rec, rng, dt_r, d, opts)
            if probe:
                # the probe (opts probe_s): every goal is searched for exactly probe_s at the goal, whatever is
                # recognised, so every view's integrated rate is over the same window (no stopping bias)
                if here and at_goal[gi] >= probe:
                    rec_t[gi], rec_d[gi] = t, d
                    pilot.resolved(c)
                continue
            if Lam[gi] >= E_rec[gi]:
                rec_t[gi] = t
                rec_d[gi] = d
                if pilot.mode in ("goal", "approach") and pilot.goal is not None and pilot.goal["index"] == gi:
                    pilot.resolved(c)
        # ---- noticing (each goal not yet noticed or found)
        if sim.tick % not_every == 0:
            for gi, g in enumerate(goals):
                if noticed[gi] is not None or rec_t[gi] is not None:
                    continue
                pts, w = g["crowd"]
                if np.min(np.linalg.norm(cams[:, None, :] - pts[None, ::7], axis=2)) > R_CROWD_MAX:
                    continue
                d = crowd_distance(fp, cams, height, pitch, yaws, pts, w, kind, body, hd, allround=allround)
                if math.isfinite(d):
                    Mu[gi] += float(hill(d, *notice)) * NOTICE_EVERY_S
                if Mu[gi] >= E_not[gi]:
                    noticed[gi] = (t, d)
        if pilot.mode == "search":
            cand = [g for gi, g in enumerate(goals) if noticed[gi] is not None and rec_t[gi] is None]
            if cand:
                pilot.approach(min(cand, key=lambda g: np.linalg.norm(g["centre"] - c)), c)
        if int(round(t / REC_EVERY_S)) % 5 == 0:
            entry = [round(t, 1), round(float(c[0]), 1), round(float(c[1]), 1), pilot.mode,
                     pilot.goal["index"] if pilot.goal is not None else -1, int(sim.alive.sum()),
                     round(pilot.view, 1), round(float(hrel), 1)]
            if opts.get("keep_log"):
                entry.append([[round(float(x), 1) for x in p] for p in sim.P[sim.alive]])
                entry.append([[round(float(x), 1) for x in hat_pos(g, t)] for g in goals])
            log.append(entry)
    t_end = sim.t - t_off
    task = None
    if reason == "done":
        task = max(rec_t) + LATENCY_S - line
    out = dict(cond=cond, city_seed=city_seed, k=k, seed=seed, reason=reason,
               line=line, task_s=task, t_end=round(t_end - (line or 0.0), 1), lost=int(n0 - sim.alive.sum()),
               contacts=int(sim.contacts), wall_s=round(time.time() - t_wall, 1), goals=[])
    order = sorted(range(len(goals)), key=lambda i: rec_t[i] if rec_t[i] is not None else math.inf)
    prev = line
    for gi in order:
        g = goals[gi]
        press = rec_t[gi] + LATENCY_S if rec_t[gi] is not None else None
        z = zone_t[gi]
        row = dict(goal=gi, block=g["block"], patch=round(g["patch"], 1), r_g=round(g["r_g"], 1),
                   press=None if press is None else round(press - line, 2),
                   zone=None if z is None else round(z - line, 2),
                   noticed=None if noticed[gi] is None else round(noticed[gi][0] - line, 2),
                   notice_d=None if noticed[gi] is None else round(noticed[gi][1], 1),
                   rec_d=None if not math.isfinite(rec_d[gi]) else round(rec_d[gi], 1),
                   at_goal_s=round(at_goal[gi], 2),
                   between_s=None if (z is None or prev is None) else round(max(z - prev, 0.0), 2),
                   at_s=None if (z is None or press is None) else round(max(press - z, 0.0), 2),
                   cf={b: [round(v[0], 2), round(v[1], 4)] for b, v in cf.get(gi, {}).items()})
        if press is not None:
            prev = press
        out["goals"].append(row)
    gd = np.array(stats["goal_dist"])
    sp = np.array(stats["speed"])
    tot = stats["moving"] + stats["stopped"]
    out["beh"] = dict(at_goal_s=round(tot, 1),
                      speed=round(float(sp.mean()), 2) if len(sp) else None,
                      stopped=round(stats["stopped"] / tot, 3) if tot else None,
                      yawing=round(stats["yawing"] / tot, 3) if tot else None,
                      inside_ring=round(stats["inside_ring"] / tot, 3) if tot else None,
                      wrapped=round(stats["wrapped"] / tot, 3) if (tot and swarm) else None,
                      goal_dist_med=round(float(np.median(gd)), 1) if len(gd) else None)
    if opts.get("keep_log"):
        out["log"] = log
        out["goal_geo"] = [dict(centre=[float(x) for x in g["centre"]], block=g["block"], hat_b=g["hat_b"])
                           for g in goals]
    return out


def _counterfactuals(acc, cond, fp, sim, cams, yaws, height, pitch, hp, body, hrel, pilot, rec, rng, dt, d_real,
                     opts):
    """Integrate the hazard over this at-goal sample on each counterfactual basis (see the module docstring)."""
    def add(name, d):
        a = acc.setdefault(name, [0.0, 0.0])
        a[0] += dt
        if math.isfinite(d):
            a[1] += float(hill(d, *rec)) * dt

    add("real", d_real)
    if cond == "swarm":
        cen = sim.P[sim.alive].mean(0)
        moved = np.repeat(cen[None], len(cams), axis=0)
        add("centre", hat_distance(fp, moved, height, pitch, yaws, hp, "swarm", body, hrel))
        add("screen", hat_distance(fp, cams, height, pitch, yaws, hp, "swarm", body, hrel, use_head=False))
        sph_head = pilot.view + _sphere_head(pilot, rng)
        add("sphere_c", hat_distance(fp, cen[None], BAND["sphere"]["goal"][0], 0.0, np.array([0.0]), hp, "sphere",
                                     body, sph_head, allround=True))
        # the same 360 camera on the drone nearest the centroid: always outside a building, where the centroid of a
        # swarm wrapped round one is inside it (and a camera there sees nothing)
        Pa = sim.P[sim.alive]
        near = Pa[int(np.argmin(np.linalg.norm(Pa - cen, axis=1)))]
        add("sphere_d", hat_distance(fp, near[None], BAND["sphere"]["goal"][0], 0.0, np.array([0.0]), hp, "sphere",
                                     body, sph_head, allround=True))
    elif cond == "sphere":
        add("fwd", hat_distance(fp, cams, BAND["single"]["goal"][0], BAND["single"]["goal"][1], yaws, hp, "single",
                                body, 0.0))
        add("screen", hat_distance(fp, cams, height, pitch, yaws, hp, "sphere", body, 0.0, use_head=False,
                                   allround=True))
    else:
        sph_head = pilot.view + _sphere_head(pilot, rng)
        add("sphere_s", hat_distance(fp, cams, BAND["sphere"]["goal"][0], 0.0, yaws, hp, "sphere", body, sph_head,
                                     allround=True))


def _sphere_head(pilot, rng):
    """A sphere pilot's head offset for a counterfactual basis: its own player, kept on the pilot."""
    hp = getattr(pilot, "_sph_head", None)
    if hp is None:
        hp = pilot._sph_head = HeadPlayer("sphere", np.random.default_rng(int(rng.integers(1 << 30))))
    hp.advance(REC_EVERY_S)
    return hp("goal")


# ============================================================================= runs

def job_id(j):
    return f"{j['cond']}|{j['city_seed']}|{j['k']}|{j.get('variant', '')}"


def run_job(j):
    try:
        r = fly(j["cond"], j["city_seed"], j["k"], j.get("opts"))
        r["variant"] = j.get("variant", "")
        r["job_id"] = job_id(j)
        return r
    except Exception as e:                       # noqa: BLE001 -- one session's failure costs that row
        return dict(job_id=job_id(j), cond=j["cond"], city_seed=j["city_seed"], k=j["k"], variant=j.get("variant", ""),
                    error=repr(e), trace=traceback.format_exc()[-2000:])


def _done(path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "error" not in r:
            out.add(r["job_id"])
    return out


def run(name, jobs, procs=None):
    from citysim.parallel import run_jobs
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "runs.jsonl"
    done = _done(path)
    todo = [j for j in jobs if job_id(j) not in done]
    print(f"[views] {name}: {len(todo)} of {len(jobs)} sessions to fly", flush=True)
    t0 = time.time()

    def progress(n, total, r):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        if n % 10 == 0 or n == total:
            print(f"[views] {n}/{total} ({time.time() - t0:.0f} s)", flush=True)

    run_jobs(run_job, todo, label=name, processes=procs, stall_s=1800.0, progress=progress)


def read(name):
    path = OUT / name / "runs.jsonl"
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows[r["job_id"]] = r
    return list(rows.values())


def frames(rows):
    """(sessions, goals, behaviour) DataFrames of a run's rows."""
    import pandas as pd
    ok = [r for r in rows if "error" not in r]
    S = pd.DataFrame([{k: v for k, v in r.items() if k not in ("goals", "beh", "log", "goal_geo")} for r in ok])
    G = []
    for r in ok:
        for i, g in enumerate(r["goals"]):
            cfr = {f"cf_{b}": (v[1] / v[0] * 60.0 if v[0] > 0 else np.nan) for b, v in g["cf"].items()}
            G.append(dict(variant=r.get("variant", ""), cond=r["cond"], city_seed=r["city_seed"], k=r["k"],
                          order=i, **{x: g[x] for x in ("goal", "between_s", "at_s", "rec_d", "notice_d", "press",
                                                         "zone", "noticed", "at_goal_s")}, **cfr))
    B = pd.DataFrame([dict(variant=r.get("variant", ""), cond=r["cond"], k=r["k"], **r["beh"]) for r in ok])
    return S, pd.DataFrame(G), B


FLOOR_S = 10.0                    # a per-goal time under this counts as this (a press from outside the zone is 0 s at it)


def _gmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.exp(np.mean(np.log(np.maximum(x, FLOOR_S))))) if len(x) else math.nan


def cmd_calib(a):
    """The simulated pilots against internal_4: per condition, the per-goal BETWEEN and AT times (median, mean,
    geometric mean of max(t, 1 s)), the share of sessions finished, and the at-goal behaviour, beside the people's."""
    import pandas as pd
    human = pd.read_csv(HUMAN / "goal_times.csv")
    hb = pd.read_csv(HUMAN / "behaviour_legs.csv")
    S, G, B = frames(read(a.name))
    out = []
    for cond in CONDITIONS:
        h = human[human.condition == HUMAN_COND[cond]]
        hs = hb[(hb.condition == HUMAN_COND[cond]) & (hb.phase == "search")]
        w = hs.dur_s / hs.dur_s.sum()
        out.append(dict(source="people", variant="", cond=cond, n=len(h), done=1.0,
                        between_med=h.between_s.median(), between_gm=_gmean(h.between_s),
                        at_med=h.at_s.median(), at_gm=_gmean(h.at_s), at_mean=h.at_s.mean(),
                        speed=float((hs.speed * w).sum()), stopped=float((hs.stopped * w).sum()),
                        yawing=float((hs.yawing * w).sum()), inside_ring=float((hs.inside_ring * w).sum()),
                        goal_dist=float(hs.goal_dist_med.median())))
        for v in sorted(G.variant.unique()):
            g = G[(G.cond == cond) & (G.variant == v)]
            s = S[(S.cond == cond) & (S.variant == v)]
            b = B[(B.cond == cond) & (B.variant == v)]
            out.append(dict(source="model", variant=v, cond=cond, n=len(g), done=float((s.reason == "done").mean()),
                            between_med=g.between_s.median(), between_gm=_gmean(g.between_s),
                            at_med=g.at_s.median(), at_gm=_gmean(g.at_s), at_mean=g.at_s.mean(),
                            speed=b.speed.median(), stopped=b.stopped.median(), yawing=b.yawing.median(),
                            inside_ring=b.inside_ring.median(), goal_dist=b.goal_dist_med.median()))
    df = pd.DataFrame(out)
    df.to_csv(OUT / a.name / "calib.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(2).to_string(index=False))
    # the ratios the study is about, per variant
    rr = []
    for v in [""] + sorted(G.variant.unique()):
        src = df[(df.source == ("people" if v == "" else "model")) & (df.variant == v)].set_index("cond")
        if len(src) < 3:
            continue
        rr.append(dict(variant=v or "people", at_single_swarm=src.at_gm["single"] / src.at_gm["swarm"],
                       at_sphere_swarm=src.at_gm["sphere"] / src.at_gm["swarm"],
                       between_single_swarm=src.between_gm["single"] / src.between_gm["swarm"],
                       between_sphere_swarm=src.between_gm["sphere"] / src.between_gm["swarm"]))
    print(pd.DataFrame(rr).round(2).to_string(index=False))


def cal_factors(calib, base="ph"):
    """{condition: (rec_scale, notice_scale)}: the calibration run's geometric-mean time at the goal and between goals
    under the `base` model over internal_4's, per condition (a time scales about as one over its rate)."""
    import pandas as pd
    human = pd.read_csv(HUMAN / "goal_times.csv")
    _, G, _ = frames(read(calib))
    g = G[G.variant == base]
    out = {}
    for cond in CONDITIONS:
        h = human[human.condition == HUMAN_COND[cond]]
        m = g[g.cond == cond]
        out[cond] = (_gmean(m.at_s) / _gmean(h.at_s), _gmean(m.between_s) / _gmean(h.between_s))
    return out


def cmd_pilot(a):
    jobs = []
    variants = [(v, dict(rec_model=v)) for v in a.rec_models]
    cal = cal_factors(a.cal_from, a.cal_base) if a.cal_from else None
    if cal:
        print("[views] calibration factors (recognition, noticing):",
              {c: (round(x, 2), round(y, 2)) for c, (x, y) in cal.items()}, flush=True)
    if a.sensitivity:
        variants += [("rec_short", dict(rec_d50_scale=0.75)), ("rec_long", dict(rec_d50_scale=1.33)),
                     ("no_explore", dict(explore=False))]
    if a.probe:
        variants = [(f"probe{int(a.probe)}", dict(rec_model="pooled", probe_s=a.probe))]
    for vname, vopts in variants:
        for k in range(a.k0, a.k0 + a.sessions):
            for seed in a.seeds:
                for cond in a.conds:
                    jobs.append(dict(cond=cond, city_seed=seed, k=k, variant=vname, opts=vopts))
    if cal:
        for k in range(a.k0, a.k0 + a.sessions):
            for seed in a.seeds:
                for cond in a.conds:
                    jobs.append(dict(cond=cond, city_seed=seed, k=k, variant="cal",
                                     opts=dict(rec_model=a.cal_base, rec_scale=cal[cond][0],
                                               notice_scale=cal[cond][1])))
    run(a.name, jobs, a.procs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pilot")
    s.add_argument("--name", default="pilot")
    s.add_argument("--sessions", type=int, default=40)
    s.add_argument("--k0", type=int, default=0)
    s.add_argument("--seeds", type=int, nargs="+", default=[CITY_SEED])
    s.add_argument("--conds", nargs="+", default=list(CONDITIONS))
    s.add_argument("--sensitivity", action="store_true")
    s.add_argument("--rec-models", nargs="*", default=["pooled"], choices=REC_MODELS)
    s.add_argument("--cal-from", default=None, help="a calibration run: add the calibrated variant `cal`")
    s.add_argument("--cal-base", default="ph", choices=REC_MODELS)
    s.add_argument("--probe", type=float, default=0.0, help="the view probe: search every goal for exactly this long")
    s.add_argument("--procs", type=int, default=None)
    s.set_defaults(fn=cmd_pilot)
    s = sub.add_parser("calib")
    s.add_argument("--name", default="pilot")
    s.set_defaults(fn=cmd_calib)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
