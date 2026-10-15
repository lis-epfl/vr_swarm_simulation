"""Tier B': the formation (or the single drone) flown through a layout by the 2-D replica (formation.ContactSim), the
dynamic counterpart of Tier B's settled templates. What it can tell: the path each interface's pilot flies, how the
formation deforms, compresses, morphs round buildings and loses spacing crossing the city, whether it envelops each
goal's occluder when the pilot arrives, and how many of its feeds stay useful on the way; with citysim.transit, what
each interface SEES while it flies. What it cannot: facade contacts as PhysX resolves them, heading timing, altitude,
street furniture -- Tier C (the headless Unity bench) is the check on those.

One flight per (layout, routing, interface, transit spread, goal spread): from the spawn, every goal in the layout's
order. Transit, at full stick:
  routing "pilot"   the pilot paths of citysim.routing (the default since 2026-10-13): the swarm's pilot steers round
                    the walls the formation cannot flow through and holds the stick through the rest; the single
                    drone's pilot steers round every building. Corners taken within PILOT_CAPTURE_M.
  routing "street"  the shortest street route (the study's original transit), waypoints taken within WAYPOINT_M.
until the centroid is ARRIVE_M from the goal centre or LEG_TIMEOUT_S passes (stuck). A layout without streets
(scatter mode) is flown straight at the goal under "street". Interfaces:
  swarm          the formation; the pilot's body yaw faces the commanded direction (feeds by display.swarm_feeds)
  single         one drone on the single pilot path, its heading (and camera) turned to the commanded direction at
                 the drone's 75 deg/s
  single_direct  one drone on the SWARM's path, relying on its own avoidance; after STALL_S under STALL_MS it re-plans
                 to the single path from where it stands (a pilot stepping in)
  single_scan    one drone on the single path that looks round: every SCAN_PERIOD_S its heading sweeps
                 +-SCAN_HALF_DEG at 75 deg/s, and since its stick is in its own heading's frame, the path bends with it
At the goal (swarm): the spread set to the goal spread and the centring pilot (formation.PILOT_GAIN) holds the
centroid on the goal for HOVER_S, then the envelopment test on the last MEASURE_S. A single drone flies on to the next
goal from where it arrived. Sampled every SAMPLE_S: alive drones, minimum pair and mean nearest-neighbour spacing, the
shown feeds and the useful ones (display.feed_informativeness), progress speed along the leg; the command every CMD_S.
The flight's tracks come back as a transit.FlightRecord for scoring.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera, display, formation, layout as lay, routing  # noqa: E402
from citysim.los import Footprints, wrap_deg  # noqa: E402
from citysim.routing import route, street_graph  # noqa: E402,F401  (bench.py reads them from here)

ARRIVE_M = 25.0
WAYPOINT_M = 12.0
PILOT_CAPTURE_M = 6.0
LEG_TIMEOUT_S = 150.0
HOVER_S = 30.0
SAMPLE_S = 0.5
CMD_S = 0.1
CLOSE_M = 3.0              # internal_3's drone-drone losses came at 2.6-3.1 m flown spacing
STALL_S, STALL_MS = 3.0, 1.0
SCAN_HALF_DEG = 60.0
# internal_3's single-drone pilots had the yaw stick out of its deadband 8-18 % of transit (runs.csv yaw_duty; median
# 0.155). A +-60 deg look takes 4 x 60 / 75 = 3.2 s, so one every 3.2 / 0.155 = 20.6 s.
SCAN_DUTY = 0.155
SCAN_S = 4 * SCAN_HALF_DEG / camera.DRONE_MAX_YAW_RATE_DEG
SCAN_PERIOD_S = SCAN_S / SCAN_DUTY
INTERFACES = ("swarm", "single", "single_direct", "single_scan")
ROUTINGS = ("pilot", "street")


def _bearing(u):
    return math.degrees(math.atan2(u[0], u[1]))


def _turn(h, target, rate, dt):
    """Heading h (deg) turned toward target at no more than rate deg/s."""
    if h is None:
        return target
    d = float(wrap_deg(target - h))
    return float(wrap_deg(h + max(-rate * dt, min(rate * dt, d))))


def scan_offset(t, phase0):
    """The single_scan pilot's look offset (deg) at time t: a triangle 0 -> +SCAN_HALF_DEG -> -SCAN_HALF_DEG -> 0 over
    SCAN_S, once every SCAN_PERIOD_S."""
    u = (t + phase0) % SCAN_PERIOD_S
    if u >= SCAN_S:
        return 0.0
    q = SCAN_S / 4
    if u < q:
        return SCAN_HALF_DEG * u / q
    if u < 3 * q:
        return SCAN_HALF_DEG * (1 - 2 * (u - q) / (2 * q))
    return -SCAN_HALF_DEG * (1 - (u - 3 * q) / q)


class Flight:
    """A replica Sim with a scripted pilot and a recorder."""

    def __init__(self, sim, m, fp, cyl, graph, interface, routing_, d_ref, feeds=True, city_bounds=None, seed=0):
        self.sim, self.m, self.fp, self.cyl, self.graph = sim, m, fp, cyl, graph
        self.interface, self.routing, self.d_ref, self.feeds = interface, routing_, d_ref, feeds
        self.city_bounds = city_bounds
        self.single = interface != "swarm"
        self.min_alive = 0 if self.single else 1
        self.every = int(round(SAMPLE_S / m.dt))
        self.cmd_every = max(1, int(round(CMD_S / m.dt)))
        self.heading = None
        self.series, self.cmds, self.legs = [], [], []
        self.path_len, self.prev_c = 0.0, sim.P[sim.alive].mean(0).copy()
        self.scan_phase = float(np.random.default_rng(7919 * seed + 17).uniform(0, SCAN_PERIOD_S))
        self.replans = self.stall_events = 0

    # ---------------------------------------------------------------- pilot
    def plan(self, start, goal, kind=None):
        if self.routing == "street" or self.cyl.n == 0:
            return (route(self.graph, start, goal) if self.graph is not None else [np.asarray(goal, float)]), WAYPOINT_M
        if kind is None:
            kind = "swarm" if self.interface in ("swarm", "single_direct") else "single"
        return routing.interface_plan(self.cyl, kind, start, goal, self.d_ref, self.city_bounds), PILOT_CAPTURE_M

    def centroid(self):
        return self.sim.P[self.sim.alive].mean(0)

    def leg(self, gi, centre, arrive_m=ARRIVE_M, timeout_s=LEG_TIMEOUT_S):
        """Transit to `centre`; returns (seconds, stuck)."""
        sim = self.sim
        t_leg = sim.t
        start = self.centroid().copy() if sim.alive.any() else self.prev_c.copy()
        wps, capture = self.plan(start, centre)
        rec = dict(leg=gi, t0=sim.t, start=start, goal=np.asarray(centre, float), arrive_m=arrive_m,
                   planned_m=routing.path_length(start, wps), planned_corners=len(wps) - 1, replans=0)
        slow, stuck = 0.0, False
        while sim.alive.sum() > self.min_alive:
            c = self.centroid()
            if float(np.linalg.norm(centre - c)) <= arrive_m:
                break
            if sim.t - t_leg > timeout_s:
                stuck = True
                break
            while len(wps) > 1 and np.linalg.norm(wps[0] - c) <= capture:
                wps.pop(0)
            d = wps[0] - c
            u = d / max(float(np.linalg.norm(d)), 1e-9)
            if self.interface == "swarm":
                cmd, heading = u, _bearing(u)
            elif self.interface == "single_scan":
                heading = _turn(self.heading, _bearing(u) + scan_offset(sim.t, self.scan_phase),
                                camera.DRONE_MAX_YAW_RATE_DEG, self.m.dt)
                h = math.radians(heading)
                cmd = np.array([math.sin(h), math.cos(h)])
            else:
                cmd, heading = u, _turn(self.heading, _bearing(u), camera.DRONE_MAX_YAW_RATE_DEG, self.m.dt)
            self.heading = heading
            sim.step(cmd, 0.0, self.d_ref)
            if sim.tick % self.cmd_every == 0:
                self.cmds.append((sim.t, float(cmd[0]), float(cmd[1]), heading, gi))
            if sim.tick % self.every == 0:
                c2 = self.centroid() if sim.alive.any() else c
                self.path_len += float(np.linalg.norm(c2 - self.prev_c))
                v = sim.V[sim.alive].mean(0) if sim.alive.any() else np.zeros(2)
                progress = float(v @ u)
                self.series.append(self.sample(c2, progress, heading, "transit", gi))
                self.prev_c = c2
                slow = slow + SAMPLE_S if progress < STALL_MS else 0.0
                if slow >= STALL_S:
                    self.stall_events += 1
                    slow = 0.0
                    if self.interface == "single_direct":
                        wps, capture = self.plan(c2, centre, kind="single")
                        self.replans += 1
                        rec["replans"] += 1
        rec.update(t1=sim.t, transit_s=sim.t - t_leg, stuck=stuck, end=self.centroid().copy() if sim.alive.any()
                   else start)
        self.legs.append(rec)
        return sim.t - t_leg, stuck

    def hold(self, gi, centre, d_ref, hover_s):
        """The centring pilot over the goal for hover_s; the positions of the last MEASURE_S."""
        sim = self.sim
        t_h = sim.t
        tail = []
        while sim.t - t_h < hover_s and sim.alive.sum() > self.min_alive:
            c = self.centroid()
            v = formation.PILOT_GAIN * (centre - c) / self.m.max_speed
            s = np.linalg.norm(v)
            sim.step(v if s <= formation.PILOT_MAX else v * formation.PILOT_MAX / s, 0.0, d_ref)
            if sim.tick % self.cmd_every == 0:
                self.cmds.append((sim.t, float(v[0]), float(v[1]), self.heading if self.heading is not None else 0.0,
                                  -1 - gi))
            if sim.tick % self.every == 0:
                c2 = self.centroid()
                self.path_len += float(np.linalg.norm(c2 - self.prev_c))
                self.prev_c = c2
                self.series.append(self.sample(c2, 0.0, 0.0, "goal", gi))
                if sim.t - t_h >= hover_s - formation.MEASURE_S:
                    tail.append(sim.P[sim.alive].copy())
        return tail

    # ---------------------------------------------------------------- recorder
    def sample(self, centroid, progress, body_yaw, phase, gi):
        sim = self.sim
        P = sim.P[sim.alive]
        n = len(P)
        if n >= 2:
            D = np.linalg.norm(P[None] - P[:, None], axis=-1)
            np.fill_diagonal(D, np.inf)
            min_pair, nn = float(D.min()), float(D.min(1).mean())
        else:
            min_pair = nn = math.nan
        useful = nfeeds = wall = math.nan
        if self.single:
            f = display.single_feed(P[0], body_yaw) if n else display.Feeds(np.zeros(0, int), np.zeros((0, 2)),
                                                                             np.zeros(0))
        else:
            f = display.swarm_feeds(P, body_yaw) if n >= 3 else display.Feeds(np.zeros(0, int), np.zeros((0, 2)),
                                                                              np.zeros(0))
        if self.feeds and ((self.single and n >= 1) or n >= 3):
            sub = self.fp.subset(self.fp.near(centroid, 90.0))
            u = display.feed_informativeness(sub, f)
            useful, nfeeds = u["useful"], len(f)
            wall = float(np.mean(u["wall_frac"])) if len(f) else math.nan
        Pall = np.where(sim.alive[:, None], sim.P, np.nan)
        return dict(t=round(sim.t, 2), phase=phase, goal=gi, alive=n, min_pair=min_pair, nn=nn, useful=useful,
                    feeds=nfeeds, wall=wall, progress=progress, cx=float(centroid[0]), cz=float(centroid[1]),
                    body_yaw=body_yaw, cam_pos=f.pos.copy(), cam_yaw=f.yaw.copy(), P=Pall)

    def record(self, **meta):
        from citysim.transit import FlightRecord
        return FlightRecord.from_flight(self, **meta)


def make_sim(boxes, start, interface, transit_dref, n=10, seed=0):
    """A ContactSim of the formation's hover template (or one drone) at `start`."""
    import swarm_replica as rep
    m = formation.model()
    if interface == "swarm":
        P0, _ = formation.hover_template(transit_dref, n, seed)
    else:
        P0 = np.zeros((1, 2))
    sim = formation.ContactSim(m, rep.City(boxes), P0 + np.asarray(start, float), np.zeros((len(P0), 2)), seed)
    return sim, m


def fly_layout(path, transit_dref=1.0, goal_dref=1.6, n=10, seed=0, start_spread_m=None, feeds=True,
               routing_="street", interface="swarm", goal_order=None):
    """One flight through a layout: its goals in the layout's order, or in `goal_order` (goal indices; the bench's
    goal flights fly the reverse too). Returns (summary dict, per-goal rows, transit.FlightRecord)."""
    if interface not in INTERFACES:
        raise ValueError(f"unknown interface {interface!r} (one of {INTERFACES})")
    L = lay.load(path)
    boxes = L.boxes_as_tuples()
    fp = Footprints(boxes)
    cyl = routing.Cylinders(boxes)
    sp = np.array([L.layout["spawn"][0], L.layout["spawn"][2]])
    sim, m = make_sim(boxes, sp, interface, transit_dref, n, seed)
    arena = L.layout.get("arena")
    fl = Flight(sim, m, fp, cyl, street_graph(L), interface, routing_, transit_dref, feeds,
                tuple(arena) if arena else None, seed)
    goals_out = []
    legs_stuck = 0
    goals = L.goals()
    if goal_order is not None:
        by_index = {g["index"]: g for g in goals}
        goals = [by_index[k] for k in goal_order]
    for g in goals:
        gi = g["index"]
        centre = np.array(g["centre"], float)
        transit_s, stuck = fl.leg(gi, centre)
        legs_stuck += int(stuck)
        if fl.single:
            goals_out.append(dict(goal=gi, transit_s=round(transit_s, 1), stuck=stuck, n_alive=int(sim.alive.sum())))
            continue
        tail = fl.hold(gi, centre, goal_dref, HOVER_S)
        P = np.mean(tail, axis=0) if tail and all(len(x) == len(tail[0]) for x in tail) else sim.P[sim.alive]
        env = formation.envelopment(P, centre)
        r = np.linalg.norm(P - centre, axis=1) if len(P) else np.zeros(0)
        inside = float((r <= g["c_g"]).mean()) if len(r) else math.nan
        goals_out.append(dict(goal=gi, transit_s=round(transit_s, 1), stuck=transit_s > LEG_TIMEOUT_S,
                              n_alive=int(sim.alive.sum()), surrounds=env["surrounds"],
                              max_gap_deg=round(env["max_gap_deg"], 1), inside_frac=round(inside, 3),
                              enveloped=bool(env["surrounds"] and env["max_gap_deg"] <= formation.MAX_GAP_DEG and
                                             inside >= formation.INSIDE_FRAC)))
    summary = summarise(fl, L.id, transit_dref, goal_dref, seed, n if interface == "swarm" else 1, legs_stuck,
                        goals_out)
    summary.update(routing=routing_, interface=interface)
    return summary, goals_out, fl.record(layout=L.id, routing=routing_, interface=interface, d_ref=transit_dref,
                                         seed=seed)


def summarise(fl, layout_id, transit_dref, goal_dref, seed, n, legs_stuck, goals_out):
    sim, series = fl.sim, fl.series
    S = np.array([[s["min_pair"], s["nn"], s["useful"], s["feeds"], s["wall"], s["progress"], s["alive"]]
                  for s in series if s["phase"] == "transit"], float).reshape(-1, 7)
    G = np.array([[s["min_pair"], s["nn"], s["useful"], s["feeds"]] for s in series if s["phase"] == "goal"],
                 float).reshape(-1, 4)
    path_len = fl.path_len
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return dict(
            layout=layout_id, transit_dref=transit_dref, goal_dref=goal_dref, seed=seed, sim_s=round(sim.t, 1),
            path_km=round(path_len / 1000.0, 3), contacts=int(sim.contacts), kills=len(sim.kills),
            drones_lost=int(n - sim.alive.sum()), legs_stuck=legs_stuck,
            contacts_per_km=round(sim.contacts / max(path_len / 1000.0, 1e-9), 2),
            progress_ms=round(float(np.nanmean(S[:, 5])), 3) if len(S) else math.nan,
            min_pair_m=round(float(np.nanmin(np.r_[S[:, 0], G[:, 0]])), 3) if len(S) or len(G) else math.nan,
            min_pair_p05_m=round(float(np.nanpercentile(S[:, 0], 5)), 3) if len(S) else math.nan,
            close_frac=round(float((S[:, 0] < CLOSE_M).mean()), 4) if len(S) else math.nan,
            nn_transit_m=round(float(np.nanmean(S[:, 1])), 3) if len(S) else math.nan,
            useful_transit=round(float(np.nanmean(S[:, 2])), 3) if len(S) else math.nan,
            feeds_transit=round(float(np.nanmean(S[:, 3])), 3) if len(S) else math.nan,
            wall_frac_transit=round(float(np.nanmean(S[:, 4])), 4) if len(S) else math.nan,
            useful_goal=round(float(np.nanmean(G[:, 2])), 3) if len(G) else math.nan,
            enveloped_frac=round(float(np.mean([g["enveloped"] for g in goals_out])), 3)
            if goals_out and "enveloped" in goals_out[0] else math.nan,
            replans=fl.replans, stall_events=fl.stall_events)


def fly_od(boxes, fp, cyl, start, goal, interface="swarm", routing_="pilot", d_ref=1.0, n=10, seed=0, graph=None,
           city_bounds=None, arrive_m=15.0, timeout_s=None, feeds=True):
    """One capability transit: from `start` to `goal` (no goals, no holds). Returns (summary, FlightRecord)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    sim, m = make_sim(boxes, start, interface, d_ref, n, seed)
    fl = Flight(sim, m, fp, cyl, graph, interface, routing_, d_ref, feeds, city_bounds, seed)
    straight = float(np.linalg.norm(goal - start))
    timeout = timeout_s if timeout_s is not None else 60.0 + 3.0 * straight / camera.CRUISE_SPEED_MS
    transit_s, stuck = fl.leg(0, goal, arrive_m=arrive_m, timeout_s=timeout)
    summary = summarise(fl, "", d_ref, math.nan, seed, n if interface == "swarm" else 1, int(stuck), [])
    summary.update(routing=routing_, interface=interface, transit_s=round(transit_s, 1), stuck=stuck,
                   straight_m=round(straight, 1))
    return summary, fl.record(routing=routing_, interface=interface, d_ref=d_ref, seed=seed)
