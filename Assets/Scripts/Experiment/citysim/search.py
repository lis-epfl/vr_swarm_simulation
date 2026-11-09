"""The search study: a pilot who does not know where the goal is searches a generated city for it, with the swarm or
with one drone, and the clock stops when the goal is first seen. Does the swarm's morphing round buildings find it
sooner? docs/search_study/README.md is the manual.

    python -B -m citysim.vissim pilot-fit                            # internal_3's searching pilots (citysim.pilotfit)
    python -B -m citysim.vissim search [--quick] [--pilots rw fit]   # -> experiment_results/search_study/<name>/
    python -B -m citysim.vissim search-check                         # the fitted pilot in internal_3's own city
    python -B -m citysim.vissim search-bench-plan / search-bench     # the Unity spot check

SCENARIO. One of a layout's goal patches is the hidden goal (only its walkers walk); the others are just buildings.
The pilot starts at a free point at least MIN_START_M from it, facing a random way, and flies the intent chain of
citysim.pilots until the goal is first seen, the CAP_S cap, or (swarm) fewer than MIN_ALIVE_SWARM drones are left --
the protocol's rule for a run going on. Every arm of a scenario flies the same chain from the same start.

ARMS. Two path rules (how the pilot gets from one intent target to the next) crossed with what is flown:
  single        one drone; plans round every building (routing "single": avoidance cylinder + 1 m)
  single_scan   the same flight, its view sweeping +-60 deg at internal_3's yaw duty (dynamic.scan_offset)
  swarm         ten drones; plans round WALLS only (routing "swarm": gaps under w_pass closed, a formation radius kept
                off them) and holds the stick through the rest, so the formation morphs round lone buildings
  swarm_steer   ten drones on the single drone's path rule: the pilot steers round every building
so single -> swarm_steer is more cameras on the same paths, and swarm_steer -> swarm is the morphing.
The swarm's feeds come from display.FeedLatch (hull debounce, heading low-pass, 75 deg/s yaw, the ready latch).

DISCOVERY. citysim.discovery's definition: at least one of the goal's walkers expected on a shown feed, within range,
in the field of view and the gimbal's band, with a clear sight line. Bases: "any" shown feed (perfect attention), and
"head_<model>" -- the feeds inside the HMD's +-50 deg about the head yaw of citysim.pilots' head models (the single
drone's one feed is always in front). Ranges RANGES (80 m the definition, 200 m what internal_3's turns suggest).
Altitude and pitch: each interface's internal_3 transit band (capability.band_settings).

AT DISCOVERY (any feed, 80 m): would an ideal all-round camera on the formation's centre have seen it too ("centre"),
was its line blocked by a building ("corner"), or out of its range or band ("reach"); was a building inside the alive
drones' hull in the last MORPH_RECENT_S (the formation morphing); was the drone whose feed saw it behind a building
from the centre.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import (LAYOUT_DIR, PACKAGE_DIR, REPO, camera, discovery, display, dynamic, formation,  # noqa: E402
                     layout as lay, pilotfit, pilots, routing)
from citysim.los import Footprints, wrap_deg  # noqa: E402

SEARCH_DIR = REPO / "experiment_results" / "search_study"
SPEC = PACKAGE_DIR / "specs" / "search.json"
CAP_S = 1500.0
SAMPLE_S = 0.5
LATCH_S = 0.1
CMD_S = 0.1
MIN_ALIVE_SWARM = 4
STARTS_PER_GOAL = 8
MIN_START_M = 260.0
START_CLEAR_M = (8.0, 5.0, 3.0)      # a start's clearance from every footprint, relaxed in this order
CAPTURE_M = 6.0                      # a path corner is taken within this (dynamic.PILOT_CAPTURE_M)
TARGET_CAPTURE_M = 8.0               # an intent target within this
STALL_S, STALL_MS = 5.0, 1.0         # this long under this progress speed: the pilot gives up on the target
CORRIDOR_M = 40.0
RANGES = (discovery.R_GOAL_M, 200.0)
PRIMARY_R = discovery.R_GOAL_M
MORPH_RECENT_S = 5.0
WALL_LIMIT_S = 900.0                 # wall-clock seconds one flight may take before it is ended as "wall"
ARM_RETRIES = 2
ARMS = {"single": dict(interface="single", plan="single", view="travel"),
        "single_scan": dict(interface="single", plan="single", view="scan"),
        "swarm": dict(interface="swarm", plan="swarm"),
        "swarm_steer": dict(interface="swarm", plan="single")}
CONTRASTS = (("swarm", "single", "swarm / single"), ("swarm", "swarm_steer", "morphing: swarm / swarm_steer"),
             ("swarm_steer", "single", "cameras: swarm_steer / single"), ("single_scan", "single", "scan / single"))


# ============================================================================= paths between intent targets

class Planner:
    """Paths between consecutive intent targets for one path rule, cached across the arms of a job."""

    def __init__(self, cyl, kind, d_ref, arena, cache):
        self.cyl, self.kind, self.d_ref, self.arena, self.cache = cyl, kind, float(d_ref), tuple(arena), cache

    def path(self, a, b):
        key = (self.kind, self.d_ref if self.kind == "swarm" else 0.0, tuple(np.round(a, 2)), tuple(np.round(b, 2)))
        if key not in self.cache:
            if self.kind == "single":
                wps = routing.plan(self.cyl, "single", a, b, corridor=CORRIDOR_M, city_bounds=self.arena)
            elif not routing.ram_ok(self.d_ref):
                wps = routing.plan(self.cyl, "all", a, b, corridor=CORRIDOR_M, city_bounds=self.arena)
            else:
                wps = routing.plan(self.cyl, "swarm", a, b, w_pass=routing.w_pass(self.d_ref), corridor=CORRIDOR_M,
                                   city_bounds=self.arena, radius=routing.formation_radius(self.d_ref))
            self.cache[key] = [np.asarray(w, float) for w in wps]
        return [w.copy() for w in self.cache[key]]


# ============================================================================= one search

class SearchFlight(dynamic.Flight):
    """One arm of one scenario: the replica flown along an intent chain, checked for the goal every SAMPLE_S."""

    def __init__(self, sim, m, fp, arm, d_ref, intent, planner, goal, band, heads, rate, cap_s=CAP_S, ranges=RANGES,
                 instant_feeds=False, scan_phase=0.0):
        spec = ARMS[arm]
        super().__init__(sim, m, fp, None, None, spec["interface"], "pilot", d_ref, feeds=False,
                         city_bounds=planner.arena)
        self.arm, self.spec, self.intent, self.planner = arm, spec, intent, planner
        self.pts, self.w, self.centre = goal
        self.height, self.pitch = band
        self.heads = heads                                   # {name: HeadProcess}
        self.rate, self.cap, self.ranges = rate, cap_s, tuple(ranges)
        self.min_alive = 1 if self.single else MIN_ALIVE_SWARM
        self.latch = None if self.single else display.FeedLatch(len(sim.P), pitch_deg=self.pitch,
                                                                instant=instant_feeds)
        self.scan_phase = scan_phase
        self.geom = discovery.goal_reach(self.pts, max(discovery.horizontal_reach(r, self.height) for r in ranges))
        self.bases = [("any", r) for r in self.ranges]
        if not self.single:
            self.bases += [(f"head_{h}", r) for h in heads for r in self.ranges]
        self.found = {}
        self.mech = {}
        self.samples = []      # (t, cx, cz, P (n, 2) NaN when lost, n shown, body yaw, shown drones, their yaws)
        self.seg = dict(reached=0, timeout=0, stall=0, paused=0, pause_s=0.0)

    def _feeds(self, body, heading):
        sim = self.sim
        if self.single:
            if not sim.alive.any():
                return display.Feeds(np.zeros(0, int), np.zeros((0, 2)), np.zeros(0))
            return display.single_feed(sim.P[0], heading)
        return self.latch.step(sim.P, sim.alive, body, LATCH_S)

    def _check(self, feeds, body):
        sim = self.sim
        t = sim.t
        if not len(feeds):
            return
        heads = list(self.heads.items())
        hy = [hp(t) for _, hp in heads] or [0.0]
        A, H = discovery.sample_expected(self.fp, feeds.pos, feeds.yaw, body, self.height, self.pitch, self.pts, self.w,
                                         self.ranges, hy, single=self.single, geom=self.geom)
        for ri, r in enumerate(self.ranges):
            if A[ri] >= discovery.SEEN_WALKERS - 1e-9 and ("any", r) not in self.found:
                self.found[("any", r)] = t
                if r == PRIMARY_R:
                    self._mechanism(feeds, body)
            if not self.single:
                for hi, (name, _) in enumerate(heads):
                    if H[ri, hi] >= discovery.SEEN_WALKERS - 1e-9 and (f"head_{name}", r) not in self.found:
                        self.found[(f"head_{name}", r)] = t

    def _mechanism(self, feeds, body):
        sim = self.sim
        cen = self.centroid()
        one = dict(single=True, geom=None)
        o_los = discovery.sample_expected(self.fp, cen, [0.0], 0.0, self.height, self.pitch, self.pts, self.w,
                                          (PRIMARY_R,), hfov=360.0, **one)[0][0]
        o_free = discovery.sample_expected(self.fp, cen, [0.0], 0.0, self.height, self.pitch, self.pts, self.w,
                                           (PRIMARY_R,), hfov=360.0, los=False, **one)[0][0]
        per = [discovery.sample_expected(self.fp, feeds.pos[k], [feeds.yaw[k]], body, self.height, self.pitch,
                                         self.pts, self.w, (PRIMARY_R,), **one)[0][0] for k in range(len(feeds))]
        k = int(np.argmax(per))
        drone = int(feeds.drones[k])
        behind = bool(self.fp.blocked(cen[None], sim.P[drone][None])[0]) if (not self.single and self.fp.n) else False
        self.mech = dict(disc_class="centre" if o_los >= 1 else ("corner" if o_free >= 1 else "reach"),
                         disc_behind=behind, disc_dist_m=round(float(np.linalg.norm(cen - self.centre)), 1),
                         disc_cam_dist_m=round(float(np.linalg.norm(feeds.pos[k] - self.centre)), 1),
                         disc_feeds=len(feeds), disc_seen_by=int(np.sum(np.asarray(per) > 0)))

    def fly(self):
        """Search along the intent chain until found, the cap, or too few drones."""
        return self._loop(IntentPilot(self))

    def fly_waypoints(self, wps, T, capture=CAPTURE_M):
        """Fly a fixed waypoint list [(x, z, stick, timeout)] the way the Unity bench's scripted pilot does (each taken
        within `capture` or after its timeout; afterwards the stick is released) for T s, checking for the goal --
        the replica's half of the Unity spot check. Stops at T, not when the goal is found."""
        self.cap = min(self.cap, float(T))
        return self._loop(WaypointPilot(self, wps, capture), stop_when_found=False)

    def _loop(self, pilot, stop_when_found=True):
        sim, dt = self.sim, self.m.dt
        latch_every = max(1, int(round(LATCH_S / dt)))
        heading = body = float(pilot.heading0)
        feeds = self._feeds(body, heading)
        reason = "cap"
        wall0 = time.time()
        while True:
            if sim.t >= self.cap - 1e-9:
                reason = "cap"
                break
            if sim.alive.sum() < self.min_alive:
                reason = "lost"
                break
            c = self.centroid()
            cmd, bearing, holding = pilot.command(c, sim.t)
            if self.single:
                want = bearing
                if self.spec.get("view") == "scan":
                    want = bearing + dynamic.scan_offset(sim.t, self.scan_phase)
                heading = pilots.turn_toward(heading, want, self.rate, dt)
                body = heading
            elif not holding:
                body = pilots.turn_toward(body, bearing, self.rate, dt)
            sim.step(cmd, 0.0, self.d_ref)
            if sim.tick % self.cmd_every == 0:
                self.cmds.append((sim.t, float(cmd[0]), float(cmd[1]), heading if self.single else body,
                                  pilot.index))
            if sim.tick % latch_every == 0:
                feeds = self._feeds(body, heading)
            if sim.tick % self.every == 0:
                c2 = self.centroid() if sim.alive.any() else c
                self.path_len += float(np.linalg.norm(c2 - self.prev_c))
                self.prev_c = c2
                v = sim.V[sim.alive].mean(0) if sim.alive.any() else np.zeros(2)
                pilot.progress(v, holding)
                self.samples.append((sim.t, float(c2[0]), float(c2[1]), np.where(sim.alive[:, None], sim.P, np.nan),
                                     len(feeds), body, feeds.drones.copy(), feeds.yaw.copy()))
                self._check(feeds, body)
                if stop_when_found and all(bk in self.found for bk in self.bases):
                    reason = "found"
                    break
                if time.time() - wall0 > WALL_LIMIT_S:      # a flight stuck in a loop costs its own row, flagged
                    reason = "wall"
                    break
        self.reason = reason
        self.targets_used = pilot.index
        return self

    # ---------------------------------------------------------------- summary
    def row(self):
        sim = self.sim
        n = len(sim.P)
        t_end = float(sim.t)
        out = dict(arm=self.arm, interface=self.spec["interface"], plan=self.spec["plan"], d_ref=self.d_ref,
                   reason=self.reason, t_end=round(t_end, 2), n=n, alive_end=int(sim.alive.sum()),
                   lost=int(n - sim.alive.sum()), contacts=int(getattr(sim, "contacts", 0)),
                   path_m=round(self.path_len, 1), speed_ms=round(self.path_len / max(t_end, 1e-9), 3),
                   targets=self.targets_used, **{f"seg_{k}": (round(v, 1) if isinstance(v, float) else v)
                                                 for k, v in self.seg.items()})
        for basis, r in self.bases:
            t = self.found.get((basis, r), math.nan)
            out[f"t_{basis}_r{int(r)}"] = round(t, 2) if math.isfinite(t) else math.nan
        out.update(self.mech)
        out.update(self._morph())
        if getattr(self, "streets", None) is not None:
            out.update(street_measures(self.samples, self.streets, self.height))
        return out

    def _morph(self):
        if self.single or not self.samples:
            return dict(morph_time_frac=0.0, split_frac=0.0, morph_buildings=0, morph_per_km=0.0, morph_recent=False,
                        shown_mean=float(np.mean([s[4] for s in self.samples])) if self.samples else math.nan)
        D = 2.0 * routing.formation_radius(self.d_ref)
        ins, spl, names = morph_flags([s[3] for s in self.samples], self.fp, D)
        t = np.array([s[0] for s in self.samples])
        td = self.found.get(("any", PRIMARY_R), math.nan)
        recent = bool(ins[(t <= td + 1e-9) & (t >= td - MORPH_RECENT_S)].any()) if math.isfinite(td) else False
        km = max(self.path_len / 1000.0, 1e-9)
        return dict(morph_time_frac=round(float(ins.mean()), 4), split_frac=round(float(spl.mean()), 4),
                    morph_buildings=len(names), morph_per_km=round(len(names) / km, 2), morph_recent=recent,
                    shown_mean=round(float(np.mean([s[4] for s in self.samples])), 2))


class IntentPilot:
    """The searching pilot: the planner's path to each intent target in turn; a target is given up after a timeout or
    a stall; a pause at a target is a centring hold where the pilot stopped."""

    def __init__(self, fl):
        self.fl, self.it, self.sim, self.m = fl, fl.intent, fl.sim, fl.m
        self.index = 1
        a = self.it.target(0)
        self._segment(a, self.it.target(1), self.sim.t)
        self.heading0 = float(self.it.headings[0])
        self.pause_until, self.pause_at = None, None
        self.slow = 0.0
        self.u = np.zeros(2)
        self.bearing = self.heading0

    def _segment(self, a, b, t):
        self.wps = self.fl.planner.path(a, b)
        self.seg_t0 = t
        self.timeout = 20.0 + 2.5 * routing.path_length(a, self.wps) / camera.CRUISE_SPEED_MS

    def command(self, c, t):
        if self.pause_until is not None:
            if t < self.pause_until:
                v = formation.PILOT_GAIN * (self.pause_at - c) / self.m.max_speed
                s = np.linalg.norm(v)
                return (v if s <= formation.PILOT_MAX else v * formation.PILOT_MAX / s), self.bearing, True
            self.pause_until = None
            self.seg_t0 = t
        wps = self.wps
        while len(wps) > 1 and np.linalg.norm(wps[0] - c) <= CAPTURE_M:
            wps.pop(0)
        advance = None
        if len(wps) == 1 and np.linalg.norm(wps[0] - c) <= TARGET_CAPTURE_M:
            advance = "reached"
        elif t - self.seg_t0 > self.timeout:
            advance = "timeout"
        elif self.slow >= STALL_S:
            advance = "stall"
        if advance:
            fl, it = self.fl, self.it
            fl.seg[advance] += 1
            self.slow = 0.0
            p = it.pause_s(self.index) if advance == "reached" else 0.0
            a = it.target(self.index)
            self.index += 1
            b = it.target(self.index)
            self._segment(a if advance == "reached" else it.mask.snap(c), b, t)
            if p > 0:
                fl.seg["paused"] += 1
                fl.seg["pause_s"] += p
                self.pause_until, self.pause_at = t + p, c.copy()
                return np.zeros(2), self.bearing, True
        d = self.wps[0] - c
        self.u = d / max(float(np.linalg.norm(d)), 1e-9)
        self.bearing = math.degrees(math.atan2(self.u[0], self.u[1]))
        return self.it.stick * self.u, self.bearing, False

    def progress(self, v, holding):
        p = float(v @ self.u) if not holding else STALL_MS
        self.slow = self.slow + SAMPLE_S if p < STALL_MS else 0.0


class WaypointPilot:
    """The Unity bench's scripted pilot (SwarmBenchRunner.RunScenario): toward each waypoint at its stick, taken
    within the capture radius or after its timeout; the stick released once the list is done."""

    def __init__(self, fl, wps, capture=CAPTURE_M):
        self.fl, self.sim = fl, fl.sim
        w = np.asarray(wps, float).reshape(-1, 4)
        self.wps, self.capture = w, capture
        self.index, self.t_wp = 0, fl.sim.t
        d = w[0, :2] - fl.centroid() if len(w) else np.array([0.0, 1.0])
        self.heading0 = math.degrees(math.atan2(d[0], d[1]))
        self.bearing = self.heading0

    def command(self, c, t):
        w = self.wps
        while self.index < len(w) and (np.linalg.norm(w[self.index, :2] - c) <= self.capture or
                                       t - self.t_wp > w[self.index, 3]):
            self.index += 1
            self.t_wp = t
        if self.index >= len(w):
            return np.zeros(2), self.bearing, True
        d = w[self.index, :2] - c
        u = d / max(float(np.linalg.norm(d)), 1e-9)
        self.bearing = math.degrees(math.atan2(u[0], u[1]))
        return float(w[self.index, 2]) * u, self.bearing, False

    def progress(self, v, holding):
        pass


def street_grid(layout_id):
    """analyse.py's street classification of a generated layout (LayoutStreetGrid): street within half a street's
    width of one of its centrelines, block anywhere else in the arena."""
    import contextlib
    import io
    import citysim
    citysim.use_bench_modules()
    import analyse
    with contextlib.redirect_stdout(io.StringIO()):
        return analyse.street_grid(layout_id, analyse.DEFAULTS["long_street"])


def street_measures(samples, grid, height):
    """Where a flight flew: the share of drone-time (every alive drone, every sample) over a street and over a block,
    and the share of the formation centre's path that ran along a street."""
    import analyse
    street = {analyse.KINDS.index("long"), analyse.KINDS.index("short")}
    block = {analyse.KINDS.index("block_low"), analyse.KINDS.index("block_roof")}
    P = [s[3][np.isfinite(s[3]).all(1)] for s in samples]
    P = np.vstack([p for p in P if len(p)]) if any(len(p) for p in P) else np.zeros((0, 2))
    out = dict(street_time_frac=math.nan, block_time_frac=math.nan, street_path_frac=math.nan)
    if not len(P):
        return out
    k = grid.kind(P[:, 0], P[:, 1], np.full(len(P), float(height)))
    out["street_time_frac"] = round(float(np.isin(k, list(street)).mean()), 4)
    out["block_time_frac"] = round(float(np.isin(k, list(block)).mean()), 4)
    c = np.array([[s[1], s[2]] for s in samples], float)
    if len(c) > 1:
        kc = grid.kind(c[:-1, 0], c[:-1, 1], np.full(len(c) - 1, float(height)))
        step = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))
        if step.sum() > 0:
            out["street_path_frac"] = round(float(step[np.isin(kc, list(street))].sum() / step.sum()), 4)
    return out


def morph_flags(Ps, fp, D, split_factor=None):
    """Per sample: (a building's centre inside the alive drones' convex hull while the formation is whole, the
    formation split), and the set of buildings ever inside -- transit.morph_events, sample by sample."""
    from citysim import transit
    sf = transit.SPLIT_FACTOR if split_factor is None else split_factor
    ins = np.zeros(len(Ps), bool)
    spl = np.zeros(len(Ps), bool)
    names = set()
    for i, P in enumerate(Ps):
        P = P[np.isfinite(P).all(1)]
        if len(P) < 3:
            continue
        c0 = P.mean(0)
        if 2.0 * float(np.linalg.norm(P - c0, axis=1).max()) > sf * D:
            spl[i] = True
            continue
        h = display.convex_hull(P)
        if len(h) < 3 or not fp.n:
            continue
        H = P[h]
        cand = fp.near(c0, D)
        if not len(cand):
            continue
        c = fp.c[cand]
        e = np.roll(H, -1, axis=0) - H
        cross = e[None, :, 0] * (c[:, None, 1] - H[None, :, 1]) - e[None, :, 1] * (c[:, None, 0] - H[None, :, 0])
        inside = (cross > 0).all(1) | (cross < 0).all(1)
        if inside.any():
            ins[i] = True
            names.update(int(b) for b in np.asarray(cand)[inside])
    return ins, spl, names


# ============================================================================= scenarios and jobs

def starts(L, fp, mask, goal_centre, rng, k):
    """k start points: free, inside the target mask, at least MIN_START_M from the goal centre, with as much clearance
    from the buildings as the city allows (START_CLEAR_M, relaxed in turn)."""
    x0, x1, z0, z1 = L.layout["arena"]
    m = pilots.TARGET_MARGIN_M
    for clear in START_CLEAR_M:
        out = []
        for _ in range(20000):
            if len(out) >= k:
                return out
            p = np.array([rng.uniform(x0 + m, x1 - m), rng.uniform(z0 + m, z1 - m)])
            if np.linalg.norm(p - goal_centre) < MIN_START_M or not mask.contains(p):
                continue
            if fp.n and fp.distance([p])[0] < clear:
                continue
            out.append(p)
    return out


def scenarios(layouts, n_starts=STARTS_PER_GOAL):
    """[dict]: every (layout, hidden goal, start) with its start heading and seed.

    A goal's starts, headings and seeds all come off one random stream, the n_starts points first, so the same layouts
    drawn with a different n_starts give the same first points but different headings and seeds: different searches.
    Take a subset by start_i from the full list instead (cmd_search does)."""
    from citysim import vissim
    out = []
    for path, meta in layouts:
        L = lay.load(path)
        boxes = L.boxes_as_tuples()
        fp = Footprints(boxes)
        mask = pilots.TargetMask(routing.Cylinders(boxes), L.layout["arena"])
        for g in L.goals():
            info = vissim.goal_info(L, g)
            rng = np.random.default_rng(7_100_000 + 1000 * int(meta["seed"]) + 31 * g["index"] +
                                        sum(map(ord, meta["clutter"])))
            for si, p in enumerate(starts(L, fp, mask, info["centre"], rng, n_starts)):
                out.append(dict(path=str(path), layout=L.id, clutter=meta["clutter"], level=meta["level"],
                                city_seed=int(meta["seed"]), goal=int(g["index"]), start=[float(p[0]), float(p[1])],
                                heading0=float(rng.uniform(-180.0, 180.0)), start_i=si,
                                start_dist_m=round(float(np.linalg.norm(p - info["centre"])), 1),
                                seed=int(rng.integers(1 << 30))))
    return out


_CACHE = {}


def city_context(boxes, arena):
    """What a search needs of a city: its boxes, footprints, cylinders, arena and target mask."""
    cyl = routing.Cylinders(boxes)
    arena = tuple(float(v) for v in arena)
    return dict(boxes=boxes, fp=Footprints(boxes), cyl=cyl, arena=arena, mask=pilots.TargetMask(cyl, arena))


def _city(path):
    """Per-worker cache of one layout's city context and goals."""
    if path not in _CACHE:
        from citysim import vissim
        L = lay.load(path)
        _CACHE.clear()
        _CACHE[path] = dict(city_context(L.boxes_as_tuples(), L.layout["arena"]), L=L, loops=vissim.goal_loops(L),
                            streets=street_grid(L.id) if L.streets else None)
    return _CACHE[path]


def fly_scenario(sc, pilot, model, band, cap_s=CAP_S, ranges=RANGES, head_models=pilots.HEAD_MODELS,
                 instant_feeds=False, arms=None, d_refs=None):
    """Every arm x spread of one pilot in one scenario of a layout: [row]."""
    C = _city(sc["path"])
    return fly_arms(C, C["loops"][sc["goal"]], sc, pilot, model, band, cap_s, ranges, head_models, instant_feeds,
                    arms, d_refs)


def fly_arms(C, goal, sc, pilot, model, band, cap_s=CAP_S, ranges=RANGES, head_models=pilots.HEAD_MODELS,
             instant_feeds=False, arms=None, d_refs=None, keep=False):
    """Every arm x spread of one pilot from one start in a city context C, for the goal (pts, w, centre): [row]
    (with keep, [(row, SearchFlight)])."""
    pts, w, centre = goal
    heads = {h: pilots.HeadProcess(h, sc["seed"] + 17, model) for h in head_models}
    cache = {}
    rows = []
    arms = arms or pilot["arms"]
    d_refs = d_refs or pilot.get("d_ref", [1.0])
    for arm in arms:
        spec = ARMS[arm]
        iface = spec["interface"]
        for d_ref in (d_refs if iface == "swarm" else [1.0]):
            ip = pilots.intent_params(pilot["intent"], iface, model, pilot.get("run_m"))
            h, p = (band["moving_h"], band["moving_p"]) if iface == "swarm" else (band["own_h"], band["own_p"])
            # This machine's CPU throws the odd impossible numpy error under load (memory: an i9-13900K on old
            # microcode); a flight is deterministic in its seed, so a retry flies the same flight again.
            for attempt in range(ARM_RETRIES + 1):
                t0 = time.time()
                try:
                    intent = pilots.Intent(pilot["intent"], sc["start"], sc["heading0"], C["mask"], sc["seed"], **ip)
                    planner = Planner(C["cyl"], spec["plan"], d_ref, C["arena"], cache)
                    sim, m = dynamic.make_sim(C["boxes"], intent.target(0), iface, d_ref, 10, sc["seed"] % 100000)
                    fl = SearchFlight(sim, m, C["fp"], arm, d_ref, intent, planner, (pts, w, centre), (h, p), heads,
                                      pilots.heading_rate(pilot["intent"], iface, model), cap_s, ranges,
                                      instant_feeds, scan_phase=float(np.random.default_rng(sc["seed"]).uniform(
                                          0, dynamic.SCAN_PERIOD_S)))
                    fl.streets = C.get("streets")
                    fl.fly()
                    break
                except Exception:                      # noqa: BLE001 -- retried, then raised
                    if attempt == ARM_RETRIES:
                        raise
            row = {k: v for k, v in sc.items() if k not in ("path", "start")}
            row.update(pilot=pilot["name"], intent=pilot["intent"], run_mean_m=ip["run_mean_m"], height=h, pitch=p,
                       wall_s=round(time.time() - t0, 1), attempts=attempt + 1, **fl.row())
            rows.append((row, fl) if keep else row)
    return rows


def _job(job):
    sc, pilot, opts = job
    model = pilotfit.load(opts["fit"]) if (pilot["intent"] == "fit" or "fit" in opts["head_models"]) else None
    try:
        return fly_scenario(sc, pilot, model, opts["band"], opts["cap_s"], opts["ranges"], opts["head_models"],
                            opts.get("instant_feeds", False))
    except Exception as e:                       # noqa: BLE001 -- one scenario's failure costs that scenario
        import traceback
        return [dict(layout=sc["layout"], goal=sc["goal"], start_i=sc["start_i"], pilot=pilot["name"],
                     error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:])]


# ============================================================================= analysis

def times(df, basis="any", r=PRIMARY_R, lost="cap"):
    """(time (n,), observed (n,)) of a basis: a goal not found by the cap is censored there; a swarm that fell under
    MIN_ALIVE_SWARM is censored at the cap too (lost="cap": it did not find the goal, the conservative reading) or at
    the moment it ended (lost="end")."""
    col = f"t_{basis}_r{int(r)}"
    t = df[col].to_numpy(float) if col in df else np.full(len(df), np.nan)
    cap = df["cap_s"].to_numpy(float) if "cap_s" in df else np.full(len(df), CAP_S)
    obs = np.isfinite(t)
    end = np.where(df["reason"].to_numpy() == "lost", df["t_end"].to_numpy(float), cap) if lost == "end" else cap
    return np.where(obs, t, end), obs


def km(t, obs):
    """Kaplan-Meier: (event times, survival just after each)."""
    order = np.argsort(t, kind="stable")
    t, obs = t[order], obs[order]
    uniq = np.unique(t[obs])
    s, surv = 1.0, []
    for u in uniq:
        at_risk = (t >= u).sum()
        d = ((t == u) & obs).sum()
        s *= 1.0 - d / at_risk
        surv.append(s)
    return uniq, np.array(surv)


def rmst(t, obs, cap):
    """Restricted mean time to discovery up to cap (the area under the KM curve)."""
    if not len(t):
        return math.nan
    u, s = km(np.minimum(t, cap), obs & (t <= cap))
    edges = np.r_[0.0, u[u < cap], cap]
    vals = np.r_[1.0, s[u < cap]]
    return float(np.sum(np.diff(edges) * vals))


def km_median(t, obs):
    u, s = km(t, obs)
    below = np.flatnonzero(s <= 0.5)
    return float(u[below[0]]) if len(below) else math.nan


def cell_table(df, cap, bases):
    """Per pilot x clutter x arm x spread: found share, KM median, RMST, for every basis."""
    rows = []
    keys = ["pilot", "clutter", "arm", "d_ref"]
    for k, g in df.groupby(keys, dropna=False):
        r = dict(zip(keys, k), n=len(g), lost_runs=int((g.reason == "lost").sum()), lost_mean=g.lost.mean(),
                 speed_ms=g.speed_ms.mean(), morph_per_km=g.morph_per_km.mean(),
                 morph_time_frac=g.morph_time_frac.mean(), split_frac=g.split_frac.mean())
        single = (g.interface == "single").all()
        for basis, rr in bases:
            t, obs = times(g, "any" if single else basis, rr)      # the single drone's feed is always in front
            tag = f"{basis}_r{int(rr)}"
            r[f"found_{tag}"] = float(obs.mean())
            r[f"median_{tag}_s"] = km_median(t, obs)
            r[f"rmst_{tag}_s"] = rmst(t, obs, cap)
        d = g[np.isfinite(g.get(f"t_any_r{int(PRIMARY_R)}", pd.Series(dtype=float)))]
        if "disc_class" in d and len(d):
            for c in ("centre", "corner", "reach"):
                r[f"disc_{c}_frac"] = float((d.disc_class == c).mean())
            r["disc_morph_recent_frac"] = float(d.morph_recent.astype(bool).mean())
            r["disc_behind_frac"] = float(d.disc_behind.astype(bool).mean())
            r["disc_dist_m"] = float(d.disc_dist_m.median())
        rows.append(r)
    return pd.DataFrame(rows)


def paired(df, a_arm, b_arm, basis, r, cap, d_ref_a=1.0, d_ref_b=1.0, n_boot=2000, seed=0, lost="cap"):
    """The RMST ratio a / b over the scenarios both flew (< 1: a finds the goal sooner), a cluster bootstrap over
    (layout, goal) for its 95 % interval, and the share of scenarios a found first (ties half)."""
    key = ["layout", "goal", "start_i"]
    A = df[(df.arm == a_arm) & np.isclose(df.d_ref, d_ref_a if ARMS[a_arm]["interface"] == "swarm" else 1.0)]
    B = df[(df.arm == b_arm) & np.isclose(df.d_ref, d_ref_b if ARMS[b_arm]["interface"] == "swarm" else 1.0)]
    m = A.merge(B, on=key, suffixes=("_a", "_b"))
    if not len(m):
        return None
    ta, oa = times(m.rename(columns=lambda c: c[:-2] if c.endswith("_a") else c), basis, r, lost)
    tb, ob = times(m.rename(columns=lambda c: c[:-2] if c.endswith("_b") else c), basis, r, lost)
    clusters = m.groupby(["layout", "goal"]).ngroup().to_numpy()
    ncl = clusters.max() + 1

    def ratio(idx):
        return rmst(ta[idx], oa[idx], cap) / max(rmst(tb[idx], ob[idx], cap), 1e-9)

    est = ratio(np.arange(len(m)))
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(clusters == c) for c in range(ncl)]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(ncl, size=ncl)
        idx = np.concatenate([members[c] for c in pick])
        boots.append(ratio(idx))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ea, eb = np.minimum(ta, cap), np.minimum(tb, cap)
    win = np.where(ea < eb, 1.0, np.where(ea > eb, 0.0, 0.5))
    return dict(n=len(m), clusters=int(ncl), ratio=est, lo=float(lo), hi=float(hi), a_first=float(win.mean()),
                rmst_a=rmst(ta, oa, cap), rmst_b=rmst(tb, ob, cap))


def contrast_table(df, cap, bases, n_boot=2000):
    rows = []
    for (pilot, clutter), g in df.groupby(["pilot", "clutter"]):
        for a_arm, b_arm, label in CONTRASTS:
            if a_arm not in set(g.arm) or b_arm not in set(g.arm):
                continue
            for d_ref in sorted(set(g[g.interface == "swarm"].d_ref)) or [1.0]:
                if ARMS[a_arm]["interface"] != "swarm" and ARMS[b_arm]["interface"] != "swarm" and d_ref != 1.0:
                    continue
                for basis, rr in bases:
                    if basis.startswith("head_") and "single" in (ARMS[a_arm]["interface"], ARMS[b_arm]["interface"]) \
                            and ARMS[a_arm]["interface"] == ARMS[b_arm]["interface"]:
                        continue
                    ba = basis if ARMS[a_arm]["interface"] == "swarm" else "any"
                    bb = basis if ARMS[b_arm]["interface"] == "swarm" else "any"
                    if ba != bb:
                        res = _paired_mixed(g, a_arm, b_arm, ba, bb, rr, cap, d_ref, n_boot)
                    else:
                        res = paired(g, a_arm, b_arm, basis, rr, cap, d_ref, d_ref, n_boot)
                    if res:
                        rows.append(dict(pilot=pilot, clutter=clutter, contrast=label, a=a_arm, b=b_arm, d_ref=d_ref,
                                         basis=basis, r=rr, **res))
    return pd.DataFrame(rows)


def _paired_mixed(g, a_arm, b_arm, ba, bb, r, cap, d_ref, n_boot):
    """paired() when the two arms are read on different bases (a swarm's head basis against the single drone's one
    feed, which is always in front: its head basis IS its any basis)."""
    g = g.copy()
    for basis, arm in ((ba, a_arm), (bb, b_arm)):
        col = f"t_{basis}_r{int(r)}"
        sel = g.arm == arm
        g.loc[sel, "_t"] = g.loc[sel, col] if col in g else np.nan
    g[f"t__mix_r{int(r)}"] = g["_t"]
    return paired(g, a_arm, b_arm, "_mix", r, cap, d_ref, d_ref, n_boot)


def bases_of(df):
    out = []
    for c in df.columns:
        if c.startswith("t_") and "_r" in c and c not in ("t_end",):
            basis, r = c[2:].rsplit("_r", 1)
            if basis != "_mix":
                out.append((basis, float(r)))
    return sorted(set(out), key=lambda x: (x[1], x[0]))


# ============================================================================= command line

def load_spec(path=SPEC):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    spec.setdefault("name", Path(path).stem)
    return spec


def city_layouts(spec, jobs):
    from citysim import vissim
    cities = vissim.load_spec(PACKAGE_DIR / "specs" / spec["cities"])
    return vissim.generate_layouts(cities, LAYOUT_DIR / "sweep~" / cities["name"], jobs)


def cmd_search(a):
    from citysim import capability
    from citysim.parallel import run_jobs
    spec = load_spec(a.spec)
    name = a.name or spec["name"] + ("_quick" if a.quick else "")
    out = SEARCH_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    layouts = city_layouts(spec, a.jobs)
    if a.clutter:
        layouts = [lp for lp in layouts if lp[1]["clutter"] in a.clutter]
    # Scenarios are always drawn at the spec's count and a subset is taken by start index: a start's heading and seed
    # come off the same random stream as the start points, so drawing fewer starts changes every search (see scenarios)
    n_starts = spec.get("starts_per_goal", STARTS_PER_GOAL)
    keep = getattr(a, "starts", None) or n_starts
    if a.quick:
        layouts = [lp for lp in layouts if lp[1]["seed"] == 0]
        keep = 2
    pilots_ = [p for p in spec["pilots"] if not a.pilots or p["name"] in a.pilots]
    cap = float(spec.get("cap_s", CAP_S))
    opts = dict(band=capability.band_settings(spec.get("band", "internal_3")), cap_s=cap,
                ranges=tuple(spec.get("ranges", RANGES)), head_models=tuple(spec.get("head_models", pilots.HEAD_MODELS)),
                fit=spec.get("fit", "internal_3"), instant_feeds=bool(a.instant_feeds))
    if any(p["intent"] == "fit" for p in pilots_) or "fit" in opts["head_models"]:
        pilotfit.load(opts["fit"])                       # fail now, not in every worker
    scen = [s for s in scenarios(layouts, n_starts) if s["start_i"] < keep]
    jobs = [(sc, p, opts) for p in pilots_ for sc in scen if sc["start_i"] < min(p.get("starts_per_goal", n_starts),
                                                                                 keep)]
    print(f"[search] {len(layouts)} layouts, {len(scen)} scenarios x {len(pilots_)} pilots = {len(jobs)} jobs "
          f"({time.time() - t0:.0f} s)", flush=True)
    part = out / "runs.partial.jsonl"

    def tick(done, total, res):
        if res:
            with open(part, "a", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, default=float) + "\n")
        if done == total or done % max(1, total // 20) == 0:
            print(f"[search] {done}/{total} at {time.time() - t0:.0f} s", flush=True)

    if part.exists():
        part.unlink()
    res = run_jobs(_job, jobs, "search", a.jobs, stall_s=1800, progress=tick)
    df = pd.DataFrame([r for rs in res if rs for r in rs])
    df["cap_s"] = cap
    errs = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    if len(errs):
        print(f"[search] {len(errs)} scenarios failed; first: {errs.iloc[0]['error']}\n{errs.iloc[0].get('trace', '')}")
        df = df[df["error"].isna()].drop(columns=["error", "trace"], errors="ignore")
    df.to_csv(out / "runs.csv", index=False)
    if part.exists():
        part.unlink()
    _, con = write_tables(df, out, cap, n_boot=500 if a.quick else 2000)
    figures(df, con, out / "figures")
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"), spec=str(a.spec), quick=bool(a.quick), layouts=len(layouts),
        scenarios=len(scen), pilots=[p["name"] for p in pilots_], opts={k: v for k, v in opts.items()},
        failed=int(len(errs)), seconds=round(time.time() - t0)), indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[search] written {out} ({time.time() - t0:.0f} s)")


def write_tables(df, out, cap, n_boot=2000):
    bases = bases_of(df)
    cells = cell_table(df, cap, bases)
    cells.to_csv(out / "cells.csv", index=False)
    con = contrast_table(df, cap, bases, n_boot)
    con.to_csv(out / "contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    tag = f"any_r{int(PRIMARY_R)}"
    show = cells[["pilot", "clutter", "arm", "d_ref", "n", f"found_{tag}", f"median_{tag}_s", f"rmst_{tag}_s",
                  "lost_mean", "speed_ms", "morph_per_km"]]
    print(show.round(3).to_string(index=False))
    if len(con):
        c = con[(con.basis == "any") & (con.r == PRIMARY_R)]
        print(c[["pilot", "clutter", "contrast", "d_ref", "n", "ratio", "lo", "hi", "a_first"]].round(3)
              .to_string(index=False))
    return cells, con


def cmd_tables(a):
    out = SEARCH_DIR / a.name
    df = pd.read_csv(out / "runs.csv")
    write_tables(df, out, float(df.cap_s.iloc[0]) if "cap_s" in df else CAP_S, a.boot)
    if not a.no_figures:
        figures(df, pd.read_csv(out / "contrasts.csv"), out / "figures")


# ============================================================================= figures

AQUA, YELLOW = "#1baf7a", "#eda100"         # categorical slots 3 and 4 (validated with 1 and 2; below 3:1 on the
#                                             light surface, so their lines carry a dash pattern and a direct label)
CLUTTER_ORDER = ("C0", "C1p", "C2p", "C1", "C2")
CLUTTER_NAME = {"C0": "C0 open", "C1p": "C1p porous", "C2p": "C2p porous", "C1": "C1 packed", "C2": "C2 packed"}
ARM_ORDER = ("swarm", "swarm_steer", "single", "single_scan")


def _arm_style():
    from citysim import report
    return {"swarm": (report.SWARM, "-", "swarm (lets the formation morph)"),
            "swarm_steer": (AQUA, (0, (5, 2)), "swarm on the single drone's paths"),
            "single": (report.SINGLE, "-", "single drone"),
            "single_scan": (YELLOW, (0, (1, 1.5)), "single drone, looking round")}


def figures(df, con, out):
    out = Path(out)
    for pilot in sorted(set(df.pilot)):
        for basis in ("any", "head_fit"):
            fig_km(df, out, pilot, basis)
    fig_contrasts(con, out)
    fig_mechanism(df, out)


def fig_km(df, out, pilot, basis="any", r=PRIMARY_R, d_ref=1.0):
    """Share of searches still without a sight of the goal against time, one panel per city, one line per arm."""
    from citysim import report
    p = report.plt()
    style = _arm_style()
    cl = [c for c in CLUTTER_ORDER if c in set(df.clutter)]
    fig, axes = p.subplots(1, len(cl), figsize=(2.5 * len(cl) + 0.6, 2.9), sharey=True, squeeze=False)
    cap = float(df.cap_s.iloc[0]) if "cap_s" in df else CAP_S
    for j, c in enumerate(cl):
        ax = axes[0, j]
        for arm in ARM_ORDER:
            g = df[(df.pilot == pilot) & (df.clutter == c) & (df.arm == arm) &
                   (np.isclose(df.d_ref, d_ref) | (df.interface == "single"))]
            if not len(g):
                continue
            b = basis if ARMS[arm]["interface"] == "swarm" else "any"
            t, obs = times(g, b, r)
            u, s = km(t, obs)
            xs, ys = [0.0], [1.0]
            for ui, si in zip(u, s):
                xs += [ui, ui]
                ys += [ys[-1], si]
            xs.append(cap)
            ys.append(ys[-1])
            x, y = np.array(xs) / 60.0, np.array(ys)
            colour, ls, label = style[arm]
            ax.plot(x, y, color=colour, ls=ls, lw=2.0, label=label)
        ax.set_title(CLUTTER_NAME.get(c, c), loc="left", fontsize=9)
        ax.set_xlim(0, cap / 60.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("minutes searching")
    axes[0, 0].set_ylabel("share not yet found")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 1.08), fontsize=8)
    what = "on any shown feed" if basis == "any" else f"on a feed in the head's view ({basis[5:]} head model)"
    fig.text(0.0, -0.06, f"Pilot {pilot}; first sight of the goal's walkers {what} within {r:g} m; swarm at spread "
             f"{d_ref:g}; censored at {cap / 60:g} min (a swarm under {MIN_ALIVE_SWARM} drones counts as not found).",
             fontsize=7, color=report.INK2)
    report.save(fig, Path(out) / f"km_{pilot}_{basis}.png")


def fig_contrasts(con, out, r=PRIMARY_R, d_ref=1.0):
    """RMST ratios with their 95 % intervals: per city, the three contrasts, each pilot; < 1 = the first arm finds the
    goal sooner."""
    from citysim import report
    p = report.plt()
    labels = [c for _, _, c in CONTRASTS[:3]]
    pilots_ = [x for x in ("rw", "fit", "rw60", "rw240") if x in set(con.pilot)]
    marks = {"rw": "o", "fit": "s", "rw60": "^", "rw240": "v"}
    names = {"rw": "random walker (120 m runs)", "fit": "internal_3's pilots (fitted)", "rw60": "random walker, 60 m",
             "rw240": "random walker, 240 m"}
    cl = [c for c in CLUTTER_ORDER if c in set(con.clutter)]
    fig, axes = p.subplots(1, 2, figsize=(10.5, 0.42 * len(cl) * len(labels) + 1.4), sharey=True)
    for ax, basis in zip(axes, ("any", "head_fit")):
        ylab, yt = [], []
        y = 0
        for c in cl:
            for lab in labels:
                for k, pl in enumerate(pilots_):
                    sel = con[(con.clutter == c) & (con.contrast == lab) & (con.pilot == pl) & (con.basis == basis) &
                              np.isclose(con.r, r) & np.isclose(con.d_ref, d_ref)]
                    if not len(sel):
                        continue
                    s = sel.iloc[0]
                    yy = y + (k - (len(pilots_) - 1) / 2) * 0.17
                    ax.plot([s.lo, s.hi], [yy, yy], color=report.INK2, lw=1.2)
                    ax.plot([s.ratio], [yy], marker=marks[pl], color=report.INK, ms=5, ls="none",
                            label=names[pl] if (c == cl[0] and lab == labels[0]) else None)
                yt.append(y)
                ylab.append(f"{CLUTTER_NAME.get(c, c)}: {lab.split(':')[0]}")
                y -= 1
            y -= 0.5
        from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
        ax.axvline(1.0, color=report.MUTED, lw=1.0)
        ax.set_xscale("log")
        ax.set_xlim(0.45, 2.2)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]))
        ax.set_xticklabels(["0.5", "0.67", "0.8", "1", "1.25", "1.5", "2"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(yt)
        ax.set_yticklabels(ylab, fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ratio of mean time to first sight (< 1: the first arm finds it sooner)")
        ax.set_title("on any shown feed" if basis == "any" else "in the head's view (fitted head)", loc="left",
                     fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.text(0.0, -0.02, f"Restricted mean time to first sight within {r:g} m, cap {CAP_S / 60:g} min, paired "
             f"scenarios; 95 % interval from a bootstrap over (layout, goal); swarm at spread {d_ref:g}. 'morphing' "
             "= swarm / swarm on the single drone's paths; 'cameras' = swarm on those paths / single drone.",
             fontsize=7, color=report.INK2)
    report.save(fig, Path(out) / "contrasts.png")


def fig_mechanism(df, out, d_ref=1.0):
    """How the swarm first saw the goal: the share of first sights that an all-round camera on the formation's centre
    would also have had, that were round a corner from it, or out of its reach; per city, both swarm arms."""
    from citysim import report
    p = report.plt()
    cl = [c for c in CLUTTER_ORDER if c in set(df.clutter)]
    fills = {"centre": "#c3c2b7", "reach": report.MUTED, "corner": report.SWARM}
    names = {"centre": "the centre would have seen it too", "reach": "out of the centre's range or band",
             "corner": "round a corner from the centre"}
    rows = []
    for pilot in [x for x in ("rw", "fit") if x in set(df.pilot)]:
        for arm in ("swarm", "swarm_steer"):
            for c in cl:
                g = df[(df.pilot == pilot) & (df.arm == arm) & (df.clutter == c) & np.isclose(df.d_ref, d_ref)]
                g = g[g.disc_class.notna()] if "disc_class" in g else g.iloc[:0]
                if len(g):
                    rows.append((f"{pilot} · {'morphs' if arm == 'swarm' else 'steered'} · {c}",
                                 {k: float((g.disc_class == k).mean()) for k in fills}, len(g)))
    if not rows:
        return
    fig, ax = p.subplots(figsize=(7.2, 0.26 * len(rows) + 1.0))
    for i, (lab, sh, n) in enumerate(rows):
        left = 0.0
        for k in ("corner", "reach", "centre"):
            ax.barh(-i, sh[k], left=left, color=fills[k], height=0.72, edgecolor=report.SURFACE, linewidth=1.5,
                    label=names[k] if i == 0 else None)
            left += sh[k]
        ax.text(1.01, -i, f"{sh['corner']:.0%} round a corner (n {n})", va="center", fontsize=7, color=report.INK2)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the swarm's first sights (80 m, any shown feed)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 1.6 / max(len(rows), 1)), ncol=3, fontsize=7.5)
    report.save(fig, Path(out) / "mechanism.png")


# ============================================================================= the fitted pilot in internal_3's city

CHECK_SEEDS = 8
CHECK_CAP_S = 600.0
ARMS_CHECK = ("single", "swarm", "swarm_steer")


def track_stats(t, xy, every=SAMPLE_S, window_s=30.0):
    """How a centroid track flew, the same way for a recorded run and a simulated one: path speed (m/s), the share of
    time under 1 m/s, course turns of at least pilotfit.THETA_MIN per km (the course of the 1 s displacement, turning
    faster than pilotfit.TURN_RATE_MIN, episodes closer than pilotfit.TURN_BRIDGE_S joined) and the straightness of
    30 s windows (net displacement over path)."""
    t = np.asarray(t, float)
    xy = np.asarray(xy, float).reshape(-1, 2)
    ok = np.isfinite(xy).all(1)
    t, xy = t[ok], xy[ok]
    if len(t) < 4 or t[-1] - t[0] < 2.0:
        return dict(seconds=0.0, metres=0.0, speed_ms=math.nan, still_share=math.nan, turns_per_km=math.nan,
                    straight30=math.nan)
    g = np.arange(t[0], t[-1], every)
    P = np.c_[np.interp(g, t, xy[:, 0]), np.interp(g, t, xy[:, 1])]
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    metres = float(step.sum())
    secs = float(g[-1] - g[0])
    k = max(int(round(1.0 / every)), 1)
    d = P[k:] - P[:-k]
    sp = np.linalg.norm(d, axis=1) / (k * every)
    course = np.degrees(np.unwrap(np.arctan2(d[:, 0], d[:, 1])))
    rate = np.r_[0.0, np.diff(course) / every]
    moving = sp > 2.0
    turning = moving & (np.abs(rate) > pilotfit.TURN_RATE_MIN)
    turns = 0
    for i, j in pilotfit._episodes(turning, g[k:], bridge=pilotfit.TURN_BRIDGE_S):
        a, b = max(i - 1, 0), min(j, len(course) - 1)
        if abs(course[b] - course[a]) >= pilotfit.THETA_MIN:
            turns += 1
    n = max(int(round(window_s / every)), 1)
    st = []
    for s in range(0, len(P) - n, n):
        path = step[s:s + n].sum()
        if path > 20.0:
            st.append(np.linalg.norm(P[s + n] - P[s]) / path)
    return dict(seconds=round(secs, 1), metres=round(metres, 1), speed_ms=metres / max(secs, 1e-9),
                still_share=float((sp < 1.0).mean()), turns_per_km=turns / max(metres / 1000.0, 1e-9),
                straight30=float(np.mean(st)) if st else math.nan)


def check_scenarios(test="internal_3", root=None, seeds=CHECK_SEEDS):
    """Every leg of the recorded test with a searching window: where it started, its goal, and how the pilot flew it
    until first sight (track_stats over the window)."""
    from citysim import DEFAULT_ROOT
    import analyse
    out = []
    for r, t0, t1, leg in pilotfit.windows(test, root or DEFAULT_ROOT):
        L = next(L for L in analyse.goal_legs(r, 15.0) if L["leg"] == leg)
        g = L["goal"]
        sel = (r.ct >= t0) & (r.ct < t1)
        st = track_stats(r.ct[sel], np.c_[r.cx[sel], r.cz[sel]])
        i0 = min(int(np.searchsorted(r.ct, t0)), len(r.ct) - 1)
        hi = int(np.clip(np.searchsorted(r.head.t.values, t0), 0, len(r.head) - 1))
        start = np.array([r.cx[i0], r.cz[i0]])
        out.append(dict(run=r.stem, pid=r.pid, condition=r.condition, leg=leg, city=r.city, goals=r.goals,
                        goal_width=r.goal_width, goal_xz=[float(g["goalX"]), float(g["goalZ"])],
                        rects=analyse.goal_ring_rects(r, g), start=[float(start[0]), float(start[1])],
                        heading0=float(r.head.bodyYaw.values[hi]), disc_s=float(t1 - t0),
                        straight_m=float(np.hypot(g["goalX"] - start[0], g["goalZ"] - start[1])),
                        seeds=[int(1_000_003 * (leg + 7) + 7919 * s + sum(map(ord, r.stem))) % (1 << 30)
                               for s in range(seeds)],
                        **{f"human_{k}": v for k, v in st.items()}))
    return out


_CHECK_CITY = {}


def _check_job(job):
    sc, opts = job
    import citysim
    citysim.use_bench_modules()
    import analyse
    key = (sc["city"], tuple((g["goalX"], g["goalZ"]) for g in sc["goals"]), sc["goal_width"])
    if key not in _CHECK_CITY:
        _CHECK_CITY.clear()
        boxes = analyse.scene_boxes(sc["goals"], sc["city"], sc["goal_width"])
        _CHECK_CITY[key] = city_context(boxes, routing.Cylinders(boxes).bounds())
    C = _CHECK_CITY[key]
    pts, w = discovery.loop_points(sc["rects"])
    goal = (pts, w, np.array(sc["goal_xz"]))
    model = pilotfit.load(opts["fit"])
    pilot = dict(name="fit", intent="fit", arms=list(ARMS_CHECK))
    rows = []
    for seed in sc["seeds"]:
        s = dict(start=sc["start"], heading0=sc["heading0"], seed=seed)
        try:
            for row, fl in fly_arms(C, goal, s, pilot, model, opts["band"], opts["cap_s"], RANGES, ("ahead", "fit"),
                                    keep=True):
                st = track_stats([x[0] for x in fl.samples], [(x[1], x[2]) for x in fl.samples])
                row.update(run=sc["run"], pid=sc["pid"], condition=sc["condition"], leg=sc["leg"],
                           human_disc_s=sc["disc_s"], straight_m=sc["straight_m"],
                           **{k: v for k, v in sc.items() if k.startswith("human_")},
                           **{f"sim_{k}": v for k, v in st.items()})
                rows.append(row)
        except Exception as e:                      # noqa: BLE001
            import traceback
            rows.append(dict(run=sc["run"], leg=sc["leg"], seed=seed, error=f"{type(e).__name__}: {e}",
                             trace=traceback.format_exc()[-800:]))
    return rows


def cmd_check(a):
    """The fitted pilot flown in the recorded test's own city from every searching leg's start, against how the
    pilots flew it: track statistics per interface, and discovery time against the recorded one."""
    from citysim import capability
    from citysim.parallel import run_jobs
    import citysim
    citysim.use_bench_modules()
    out = SEARCH_DIR / "check" / a.test
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scen = check_scenarios(a.test, a.root, a.seeds)
    opts = dict(band=capability.band_settings(a.test), cap_s=a.cap, fit=a.test)
    print(f"[search-check] {len(scen)} legs x {a.seeds} seeds x {len(ARMS_CHECK)} arms", flush=True)
    res = run_jobs(_check_job, [(sc, opts) for sc in scen], "check", a.jobs, stall_s=1800)
    df = pd.DataFrame([r for rs in res if rs for r in rs])
    if "error" in df and df["error"].notna().any():
        e = df[df["error"].notna()].iloc[0]
        print(f"[search-check] {int(df['error'].notna().sum())} failed; first: {e['error']}\n{e.get('trace', '')}")
        df = df[df["error"].isna()].drop(columns=["error", "trace"], errors="ignore")
    df["cap_s"] = a.cap
    df.to_csv(out / "check_runs.csv", index=False)
    summ = check_summary(df, scen, a.cap)
    summ.to_csv(out / "check_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(f"[search-check] written {out} ({time.time() - t0:.0f} s)")


def check_summary(df, scen, cap):
    """Per recorded condition: the pilots' own track statistics over their searching windows (weighted by time) beside
    the fitted pilot's over its searches from the same starts (its own interface, d_ref 1.0); and discovery: the
    recorded median and the simulated RMST / median at 80 m (any feed)."""
    hum = pd.DataFrame(scen)
    rows = []
    for cond, arm in (("SingleDrone", "single"), ("Swarm", "swarm")):
        h = hum[hum.condition == cond]
        s = df[(df.condition == cond) & (df.arm == arm) & np.isclose(df.d_ref, 1.0)]
        if not len(h) or not len(s):
            continue
        w_h, w_s = h.human_seconds.clip(lower=1e-9).to_numpy(), s.sim_seconds.clip(lower=1e-9).to_numpy()
        r = dict(condition=cond, arm=arm, legs=len(h), sims=len(s))
        for k in ("speed_ms", "still_share", "turns_per_km", "straight30"):
            hv, sv = h[f"human_{k}"].to_numpy(float), s[f"sim_{k}"].to_numpy(float)
            mh, ms = np.isfinite(hv), np.isfinite(sv)
            r[f"human_{k}"] = float(np.average(hv[mh], weights=w_h[mh])) if mh.any() else math.nan
            r[f"sim_{k}"] = float(np.average(sv[ms], weights=w_s[ms])) if ms.any() else math.nan
        t, obs = times(s, "any", PRIMARY_R)
        r["human_disc_median_s"] = float(h.disc_s.median())
        r["human_disc_per_100m_s"] = float((h.disc_s / h.straight_m * 100).median())
        r["sim_disc_rmst_s"] = rmst(t, obs, cap)
        r["sim_disc_median_s"] = km_median(t, obs)
        r["sim_found_share"] = float(obs.mean())
        r["eta"] = r["human_disc_median_s"] / r["sim_disc_median_s"] if math.isfinite(r["sim_disc_median_s"]) \
            else math.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================= the Unity spot check

BENCH_T_S = 480.0                    # each bench flight: this long, whatever it has found
BENCH_CLUTTER = ("C0", "C1p", "C2")
BENCH_PER_LAYOUT = 2                 # scenarios per layout (distinct goals)
BENCH_ARMS = ("single", "swarm", "swarm_steer")
BENCH_REPLICA_SEEDS = 2


def chain_waypoints(C, sc, arm, d_ref, pilot, model, length_m):
    """[x, z, stick, timeout]* of a scenario's intent chain for an arm: the arm's paths target to target, laid until
    they are length_m long, every waypoint timed out as bench.goal_flights times them."""
    import bench
    spec = ARMS[arm]
    iface = spec["interface"]
    ip = pilots.intent_params(pilot["intent"], iface, model, pilot.get("run_m"))
    it = pilots.Intent(pilot["intent"], sc["start"], sc["heading0"], C["mask"], sc["seed"], **ip)
    planner = Planner(C["cyl"], spec["plan"], d_ref, C["arena"], {})
    wps, total, k = [], 0.0, 1
    prev = it.target(0)
    while total < length_m:
        for q in planner.path(it.target(k - 1), it.target(k)):
            seg = float(np.linalg.norm(q - prev))
            wps += [float(q[0]), float(q[1]), float(it.stick), bench.GOAL_WP_SLACK_S + seg / bench.GOAL_MIN_PROGRESS]
            total += seg
            prev = q
        k += 1
    return wps


def cmd_bench_plan(a):
    """Bench scenarios, configs and a runner for the Unity spot check: per layout (seed `a.seed` of each clutter level),
    BENCH_PER_LAYOUT scenarios, each arm's chain of the rw pilot flown for BENCH_T_S."""
    import subprocess
    import citysim
    from citysim import BENCH_DIR, capability, tierc
    citysim.use_bench_modules()
    import bench
    spec = load_spec(a.spec)
    pilot = next(p for p in spec["pilots"] if p["name"] == a.pilot)
    model = pilotfit.load(spec.get("fit", "internal_3")) if pilot["intent"] == "fit" else None
    band = capability.band_settings(spec.get("band", "internal_3"))
    bench_root = Path(a.bench_root)
    cfg_dir = bench_root / "cfg" / "search" / a.name
    scen_dir = bench_root / "scenarios_search"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    scen_dir.mkdir(parents=True, exist_ok=True)
    layouts = [lp for lp in city_layouts(spec, a.jobs) if lp[1]["clutter"] in a.clutter and lp[1]["seed"] == a.seed]
    runs = []
    for path, meta in layouts:
        C = _city(str(path))
        L = C["L"]
        scen = scenarios([(path, meta)], 1)[:a.per_layout]
        for drones, arms in ((10, [x for x in a.arms if ARMS[x]["interface"] == "swarm"]),
                             (1, [x for x in a.arms if ARMS[x]["interface"] == "single"])):
            if not arms:
                continue
            bsc, flights = [], []
            for sc in scen:
                for arm in arms:
                    wps = chain_waypoints(C, sc, arm, a.d_ref, pilot, model, a.T * camera.CRUISE_SPEED_MS * 1.2)
                    bsc.append(dict(kind=f"search_{arm}", order=f"g{sc['goal']}s{sc['start_i']}", sx=sc["start"][0],
                                    sz=sc["start"][1], T=a.T, wps=wps, record=True, spread=[0.0, a.d_ref]))
                    flights.append(dict(sc, arm=arm, d_ref=a.d_ref, pilot=pilot["name"], wps=wps))
            sfile = scen_dir / f"{L.id}_search_{a.name}_n{drones}.json"
            bench.write_scenarios(sfile, L.id, "citysim.search", a.seed, bsc)
            tag = f"search_{a.name}_{meta['clutter']}_s{a.seed}_n{drones}"
            cfg = cfg_dir / f"{tag}.json"
            h = band["moving_h"] if drones > 1 else band["own_h"]
            subprocess.run([sys.executable, "-B", str(BENCH_DIR / "bench.py"), "config", "--scene", "ProceduralCityWorld",
                            "--layout", str(path), "--drones", str(drones), "--pilot-heading", "--altitude", f"{h:.2f}",
                            "--time-scale", str(a.time_scale), "--scenarios", str(sfile), "--capture", str(CAPTURE_M),
                            "--record-all", "--out", str(cfg)], check=True, cwd=str(BENCH_DIR),
                           stdout=subprocess.DEVNULL)
            runs.append(dict(tag=tag, config=str(cfg), layout=L.id, layout_path=str(path), clutter=meta["clutter"],
                             drones=drones, flights=flights))
    out = SEARCH_DIR / "bench" / a.name
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), name=a.name, bench_root=str(bench_root), T=a.T,
                pilot=pilot["name"], d_ref=a.d_ref, band=band, time_scale=a.time_scale, runs=runs)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    tierc.write_runner(cfg_dir, bench_root, runs, script="run_search")
    n = sum(len(r["flights"]) for r in runs)
    print(f"[search-bench] {len(runs)} runs, {n} flights of {a.T:.0f} s -> {out / 'plan.json'}\n"
          f"  fly them: powershell -NoProfile -ExecutionPolicy Bypass -File {cfg_dir / 'run_search.ps1'}")


def _first_sights(fp, t, cam_pos, cam_yaw, cam_valid, body, height, pitch, goal, heads, single, every=SAMPLE_S):
    """{(basis, r): first time} over a camera track sampled every `every` s, on the bases a search records."""
    pts, w, _ = goal
    geom = discovery.goal_reach(pts, max(discovery.horizontal_reach(r, height) for r in RANGES))
    keep = np.r_[0, np.flatnonzero(np.diff(np.floor(np.asarray(t) / every)) > 0) + 1]
    found = {}
    names = list(heads)
    for i in keep:
        live = np.flatnonzero(cam_valid[i])
        if not len(live):
            continue
        hy = [heads[h](t[i]) for h in names] or [0.0]
        A, H = discovery.sample_expected(fp, cam_pos[i, live], cam_yaw[i, live], body[i], height, pitch, pts, w,
                                         RANGES, hy, single=single, geom=geom)
        for ri, r in enumerate(RANGES):
            if A[ri] >= discovery.SEEN_WALKERS - 1e-9:
                found.setdefault(("any", r), float(t[i]))
            if not single:
                for hi, h in enumerate(names):
                    if H[ri, hi] >= discovery.SEEN_WALKERS - 1e-9:
                        found.setdefault((f"head_{h}", r), float(t[i]))
    return found


def _bench_job(job):
    run_dir, run, fi, flight, band, T, n_seeds, fit = job
    import citysim
    citysim.use_bench_modules()
    from citysim import tierc, transit
    meta = run["flights"][int(flight["scenario"])]
    C = _city(run["layout_path"])
    goal = C["loops"][meta["goal"]]
    model = pilotfit.load(fit)
    heads = {h: pilots.HeadProcess(h, meta["seed"] + 17, model) for h in pilots.HEAD_MODELS}
    single = run["drones"] == 1
    h, p = (band["own_h"], band["own_p"]) if single else (band["moving_h"], band["moving_p"])
    keys = dict(layout=run["layout"], clutter=run["clutter"], goal=meta["goal"], start_i=meta["start_i"],
                arm=meta["arm"], d_ref=meta["d_ref"], T=T)
    rows = []
    traj = Path(run_dir) / f"traj_{flight['set']}_{int(flight['scenario'])}.csv"
    if traj.exists():
        tr = tierc.read_traj(traj)
        tr.setdefault("leg", np.zeros(len(tr["t"]), int))         # a search is one leg (Tier C adds its goal legs)
        rec = transit.FlightRecord.from_bench(tr, dict(interface="single" if single else "swarm"), single,
                                              np.ones(len(tr["t"]), bool), [])
        t = tr["t"] - tr["t"][0]
        body = rec.body_yaw
        found = _first_sights(C["fp"], t, rec.cam_pos, rec.cam_yaw, rec.cam_valid, body, h, p, goal,
                              {} if single else heads, single)
        alive_end = int(tr["alive"][-1].sum())
        P = np.where(tr["alive"][..., None], np.stack([tr["X"], tr["Z"]], -1), np.nan)
        cen = np.nanmean(P, axis=1)
        path_m = float(np.nansum(np.linalg.norm(np.diff(cen, axis=0), axis=1)))
        row = dict(keys, side="unity", seed=-1, t_end=round(float(t[-1]), 1), alive_end=alive_end,
                   lost=int(tr["n"] - alive_end), path_m=round(path_m, 1),
                   shown_mean=round(float((tr["shown"] & tr["alive"]).sum(1).mean()) if not single else 1.0, 2),
                   alt_m=round(float(np.nanmedian(np.where(tr["alive"], tr["Y"], np.nan))), 2))
        row.update({f"t_{b}_r{int(r)}": round(v, 2) for (b, r), v in found.items()})
        rows.append(row)
    pilot = dict(name=meta["pilot"], intent="rw")
    for s in range(n_seeds):
        sim, m = dynamic.make_sim(C["boxes"], meta["start"], ARMS[meta["arm"]]["interface"], meta["d_ref"], 10,
                                  (meta["seed"] + 101 * s) % 100000)
        ip = pilots.intent_params("rw", ARMS[meta["arm"]]["interface"], None)
        it = pilots.Intent("rw", meta["start"], meta["heading0"], C["mask"], meta["seed"], **ip)
        planner = Planner(C["cyl"], ARMS[meta["arm"]]["plan"], meta["d_ref"], C["arena"], {})
        fl = SearchFlight(sim, m, C["fp"], meta["arm"], meta["d_ref"], it, planner, goal, (h, p),
                          {} if single else heads, pilots.heading_rate("rw", ARMS[meta["arm"]]["interface"]), T)
        fl.fly_waypoints(meta["wps"], T)
        r = fl.row()
        row = dict(keys, side="replica", seed=s, t_end=r["t_end"], alive_end=r["alive_end"], lost=r["lost"],
                   path_m=r["path_m"], shown_mean=r["shown_mean"], alt_m=h)
        row.update({k: v for k, v in r.items() if k.startswith("t_") and k != "t_end"})
        rows.append(row)
    return rows


def cmd_bench_score(a):
    """Discovery on the flown bench runs of a plan, from Unity's recorded trajectories (its feeds as BoundaryFeedReady
    showed them), beside the replica flying the same waypoint chains; and each side's contrasts."""
    from citysim import tierc
    from citysim.parallel import run_jobs
    out = SEARCH_DIR / "bench" / a.name
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    spec = load_spec(a.spec)
    runs_root = Path(plan["bench_root"]) / "runs"
    jobs, missing = [], []
    for run in plan["runs"]:
        run_dir = runs_root / run["tag"]
        if not (run_dir / "results.jsonl").exists():
            missing.append(run["tag"])
            continue
        _, _, flights, _ = tierc.read_run(run_dir)
        for fi, f in enumerate(flights):
            jobs.append((str(run_dir), run, fi, f, plan["band"], plan["T"], a.seeds, spec.get("fit", "internal_3")))
    if missing:
        print(f"[search-bench] {len(missing)} run(s) not flown yet: {', '.join(missing)}")
    if not jobs:
        raise SystemExit("[search-bench] nothing to score")
    res = run_jobs(_bench_job, jobs, "bench", a.jobs)
    df = pd.DataFrame([r for rs in res if rs for r in rs])
    df["cap_s"] = plan["T"]
    df["reason"] = "cap"
    df.to_csv(out / "bench_flights.csv", index=False)
    summ = bench_summary(df, plan["T"])
    summ.to_csv(out / "bench_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(f"[search-bench] written {out}")


def bench_summary(df, T):
    """Per arm and basis: Unity's and the replica's RMST within T over the same flights, their found shares, and the
    flight-by-flight agreement; then each side's contrasts (swarm / single, swarm / swarm_steer)."""
    rows = []
    bases = [(b, r) for b, r in bases_of(df)]
    for (arm, side), g in df.groupby(["arm", "side"]):
        for b, r in bases:
            t, obs = times(g, b, r)
            rows.append(dict(kind="arm", arm=arm, side=side, basis=b, r=r, n=len(g), found=float(obs.mean()),
                             rmst_s=rmst(t, obs, T), median_s=km_median(t, obs)))
    for b, r in bases:
        for side, g in df.groupby("side"):
            per = g.groupby(["layout", "goal", "start_i", "arm"]).apply(
                lambda x: rmst(*times(x, b if (ARMS[x.arm.iloc[0]]["interface"] == "swarm" or b == "any") else "any",
                                      r), T)).unstack("arm")
            for a_arm, b_arm in (("swarm", "single"), ("swarm", "swarm_steer"), ("swarm_steer", "single")):
                if a_arm in per and b_arm in per:
                    rows.append(dict(kind="contrast", arm=f"{a_arm} / {b_arm}", side=side, basis=b, r=r,
                                     n=int(per[[a_arm, b_arm]].dropna().shape[0]),
                                     ratio=float(per[a_arm].mean() / per[b_arm].mean())))
    return pd.DataFrame(rows)
