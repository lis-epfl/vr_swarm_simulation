"""Tier B': the formation flown through a layout by the 2-D replica (formation.ContactSim), the dynamic counterpart
of Tier B's settled templates. What it can tell: how the formation deforms, compresses and loses spacing crossing
the city, whether it envelops each goal's occluder when the pilot arrives, and how many of its feeds stay useful
(not wall-filled) on the way. What it cannot: facade contacts as PhysX resolves them, heading timing, altitude,
street furniture -- Tier C (the headless Unity bench) is the check on those.

One flight per (layout, transit spread, goal spread): from the spawn, every goal in the layout's order. Transit:
full stick along the shortest street route to the next goal (the layout's street centrelines as a graph; waypoints
taken within WAYPOINT_M, the bench's capture rule), the swarm left to steer round what is in the way, until the
centroid is ARRIVE_M from the goal centre or LEG_TIMEOUT_S passes (stuck). A layout without streets (scatter mode)
is flown straight at the goal. At the goal: the spread set to the goal
spread and the centring pilot (formation.PILOT_GAIN) holds the centroid on the goal for HOVER_S, then the
envelopment test on the last MEASURE_S. Sampled every SAMPLE_S: alive drones, minimum pair and mean nearest-
neighbour spacing, the hull feeds (display.swarm_feeds, body yaw along the leg) and the useful ones
(display.feed_informativeness), progress speed along the leg.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import display, formation, layout as lay  # noqa: E402
from citysim.los import Footprints  # noqa: E402

ARRIVE_M = 25.0
WAYPOINT_M = 12.0
LEG_TIMEOUT_S = 150.0
HOVER_S = 30.0
SAMPLE_S = 0.5
CLOSE_M = 3.0              # internal_3's drone-drone losses came at 2.6-3.1 m flown spacing


def fly_layout(path, transit_dref=1.0, goal_dref=1.6, n=10, seed=0, start_spread_m=None, feeds=True):
    """One flight through a layout. Returns (summary dict, per-goal rows, sampled series)."""
    import swarm_replica as rep
    L = lay.load(path)
    boxes = L.boxes_as_tuples()
    fp = Footprints(boxes)
    m = formation.model()
    P0, _ = formation.hover_template(transit_dref, n, seed)
    sp = np.array([L.layout["spawn"][0], L.layout["spawn"][2]])
    sim = formation.ContactSim(m, rep.City(boxes), P0 + sp, np.zeros((n, 2)), seed)
    every = int(round(SAMPLE_S / m.dt))
    graph = street_graph(L)
    series, goals_out = [], []
    path_len, prev_c = 0.0, sp.copy()
    legs_stuck = 0
    for gi, g in enumerate(L.goals()):
        centre = np.array(g["centre"], float)
        t_leg = sim.t
        wps = route(graph, sim.P[sim.alive].mean(0), centre) if graph is not None else [centre]
        # ---- transit
        while sim.alive.sum() > 1:
            c = sim.P[sim.alive].mean(0)
            dist = float(np.linalg.norm(centre - c))
            if dist <= ARRIVE_M:
                break
            if sim.t - t_leg > LEG_TIMEOUT_S:
                legs_stuck += 1
                break
            while len(wps) > 1 and np.linalg.norm(wps[0] - c) <= WAYPOINT_M:
                wps.pop(0)
            d = wps[0] - c
            u = d / max(float(np.linalg.norm(d)), 1e-9)
            sim.step(u, 0.0, transit_dref)
            if sim.tick % every == 0:
                c2 = sim.P[sim.alive].mean(0)
                path_len += float(np.linalg.norm(c2 - prev_c))
                v = sim.V[sim.alive].mean(0)
                series.append(sample(sim, fp, c2, float(v @ u), math.degrees(math.atan2(u[0], u[1])), "transit", gi,
                                     feeds))
                prev_c = c2
        transit_s = sim.t - t_leg
        # ---- hover at the goal
        t_h = sim.t
        tail = []
        while sim.t - t_h < HOVER_S and sim.alive.sum() > 1:
            c = sim.P[sim.alive].mean(0)
            v = formation.PILOT_GAIN * (centre - c) / m.max_speed
            s = np.linalg.norm(v)
            sim.step(v if s <= formation.PILOT_MAX else v * formation.PILOT_MAX / s, 0.0, goal_dref)
            if sim.tick % every == 0:
                c2 = sim.P[sim.alive].mean(0)
                path_len += float(np.linalg.norm(c2 - prev_c))
                prev_c = c2
                series.append(sample(sim, fp, c2, 0.0, 0.0, "goal", gi, feeds))
                if sim.t - t_h >= HOVER_S - formation.MEASURE_S:
                    tail.append(sim.P[sim.alive].copy())
        P = np.mean(tail, axis=0) if tail and all(len(x) == len(tail[0]) for x in tail) else sim.P[sim.alive]
        env = formation.envelopment(P, centre)
        r = np.linalg.norm(P - centre, axis=1) if len(P) else np.zeros(0)
        inside = float((r <= g["c_g"]).mean()) if len(r) else math.nan
        goals_out.append(dict(goal=gi, transit_s=round(transit_s, 1), stuck=transit_s > LEG_TIMEOUT_S,
                              n_alive=int(sim.alive.sum()), surrounds=env["surrounds"],
                              max_gap_deg=round(env["max_gap_deg"], 1), inside_frac=round(inside, 3),
                              enveloped=bool(env["surrounds"] and env["max_gap_deg"] <= formation.MAX_GAP_DEG and
                                             inside >= formation.INSIDE_FRAC)))
    S = np.array([[s["min_pair"], s["nn"], s["useful"], s["feeds"], s["wall"], s["progress"], s["alive"]]
                  for s in series if s["phase"] == "transit"], float).reshape(-1, 7)
    G = np.array([[s["min_pair"], s["nn"], s["useful"], s["feeds"]] for s in series if s["phase"] == "goal"],
                 float).reshape(-1, 4)
    kills = [k for k in sim.kills]
    summary = dict(
        layout=L.id, transit_dref=transit_dref, goal_dref=goal_dref, seed=seed, sim_s=round(sim.t, 1),
        path_km=round(path_len / 1000.0, 3), contacts=int(sim.contacts), kills=len(kills),
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
        enveloped_frac=round(float(np.mean([g["enveloped"] for g in goals_out])), 3) if goals_out else math.nan)
    return summary, goals_out, series


def sample(sim, fp, centroid, progress, body_yaw, phase, gi, feeds=True):
    P = sim.P[sim.alive]
    n = len(P)
    if n >= 2:
        D = np.linalg.norm(P[None] - P[:, None], axis=-1)
        np.fill_diagonal(D, np.inf)
        min_pair, nn = float(D.min()), float(D.min(1).mean())
    else:
        min_pair = nn = math.nan
    useful = nfeeds = wall = math.nan
    if feeds and n >= 3:
        f = display.swarm_feeds(P, body_yaw)
        sub = fp.subset(fp.near(centroid, 90.0))
        u = display.feed_informativeness(sub, f)
        useful, nfeeds = u["useful"], len(f)
        wall = float(np.mean(u["wall_frac"])) if len(f) else math.nan
    return dict(t=round(sim.t, 2), phase=phase, goal=gi, alive=n, min_pair=min_pair, nn=nn, useful=useful,
                feeds=nfeeds, wall=wall, progress=progress, cx=float(centroid[0]), cz=float(centroid[1]))


# ============================================================================= routing

def street_graph(L):
    """(nodes (k, 2), adjacency {i: [(j, length)]}) of the layout's street centrelines, or None without streets."""
    if not L.streets:
        return None
    key, nodes, adj = {}, [], {}

    def node(p):
        k = (round(p[0], 1), round(p[1], 1))
        if k not in key:
            key[k] = len(nodes)
            nodes.append(np.array(p, float))
        return key[k]

    for st in L.streets:
        pts = st["polyline"]
        for a, b in zip(pts[:-1], pts[1:]):
            i, j = node(a), node(b)
            w = float(np.linalg.norm(np.subtract(a, b)))
            adj.setdefault(i, []).append((j, w))
            adj.setdefault(j, []).append((i, w))
    return np.array(nodes), adj


def route(graph, start, goal):
    """Waypoints from start to goal along the street graph: the node nearest the start, the shortest path to the
    node nearest the goal, then the goal itself."""
    import heapq
    nodes, adj = graph
    s = int(np.argmin(np.linalg.norm(nodes - start, axis=1)))
    t = int(np.argmin(np.linalg.norm(nodes - goal, axis=1)))
    dist, prev = {s: 0.0}, {}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    path = [t]
    while path[-1] != s and path[-1] in prev:
        path.append(prev[path[-1]])
    return [nodes[i].copy() for i in reversed(path)] + [np.asarray(goal, float)]
