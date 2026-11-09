"""S4 Split and Surround, Phase 2: the comparison (the split study).

    python -B -m citysim.split walks  --name N [--quick] [--hours 12.5] [--jobs 28] [--resume]   # the paired chains
    python -B -m citysim.split search --name N [--quick] ...                                      # the informed search
    python -B -m citysim.split tables --name N                                                    # statistics
    -> experiment_results/split_study/<N>/

The spec (citysim/specs/split.json) carries Phase 1's choices: the variant, the designs, the S4 profile (a
split_probe setting, with the design's d* or the per-block rule as the block spread) and the swarm's two spreads
("profile" and the dial centre, 1.0).

Conditions (docs/split_study/README.md section 5), every one flying the same block-hop chain from the same start:
  single        one drone along the streets, round each block by the shorter side to its exit
  single_look   the same, holding at each entrance corner to yaw down each street there (75 deg/s, 0.5 s dwell)
  swarm_street  the formation on the single drone's path
  swarm_hold    swarm_street plus a centring hold of t_hop at each entrance corner (H1's time-matched comparator)
  swarm_pass    the formation aimed through each block at its exit corner (M3 when the exit is opposite)
  swarm_hop     the S4 surround (Phase 1's mode: a hold on the block centre, or aim-and-turn) for t_hop, then the exit
A block-hop chain: from the start block, each exit is drawn uniformly among the block's interior radiating streets
(the entrance and those at the entrance corner excluded); the next block is the one ahead where that street ends.
It never reads the flight, so the conditions are paired and the bench can fly it.

Per flight: preview at each commit (split_probe's definition, both bands), reform before the next decision, losses,
contacts, the shown feeds, street clearance, S3's time to find (clutter.discovery_first over the arena's open ground
at 40 / 80 / 200 m, any feed and head ahead; cells the start already shows excluded), and the view-correspondence
measures (citysim.correspondence, every CORR_EVERY_S).
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import REPO, blockcity, camera, clutter, correspondence, formation, los, split_probe as S, streetnet  # noqa: E402,E501
from citysim import geometry as geo  # noqa: E402
from citysim.polyfoot import ConvexFootprints  # noqa: E402

OUT_ROOT = REPO / "experiment_results" / "split_study"
SPEC = Path(__file__).resolve().parent / "specs" / "split.json"
CONDITIONS = ("single", "single_look", "swarm_street", "swarm_hold", "swarm_pass", "swarm_hop")
SWARM = ("swarm_street", "swarm_hold", "swarm_pass", "swarm_hop")
CHAIN_M = 6000.0
LOOK_DWELL_S = 0.5
LOOK_AHEAD_M = 30.0
CORR_EVERY_S = 1.0
FIND_RANGES = (40.0, 80.0, 200.0)
STREET_SEEN_RANGE_M = 80.0
QUICK = dict(designs=3, cities=2, walks=1, T_s=120.0)
# the informed search (H2)
SEARCH_T_S = 1200.0               # the cap; a target not found by then counts as found at the cap (RMST)
FIND_STOP_M = 80.0                # the search stops at the first sight at this range (the primary's)
MIN_SEARCH_DRONES = 4             # a swarm that falls under four drones has not found it (S2's conservative reading)
DECISION_STEP_S = 2.0             # the flight's knowledge and the target are brought up to date this often
FRONTIER_MIN_M = 20.0             # with nothing unseen next to the block, head for a street with this much unseen
TARGET_STREET_M = 300.0           # a target at least this far by street from the start ...
TARGET_HIDDEN_M = 200.0           # ... and out of clear sight of it within this
TARGET_EDGE_M = 60.0              # and this far inside the arena's edge
# also flown on head-ahead knowledge, on every other target: H2's two conditions (a single drone's one feed is always in
# view, so its head-ahead flight would be its any-feed flight again)
HEAD_BASIS = ("swarm_street", "swarm_hop")
QUICK_SEARCH = dict(designs=2, cities=1, targets=2, T_s=240.0)
_SIGHT0 = dict(FIND_RANGES=FIND_RANGES, FIND_STOP_M=FIND_STOP_M, STREET_SEEN_RANGE_M=STREET_SEEN_RANGE_M,
               TARGET_HIDDEN_M=TARGET_HIDDEN_M)


def set_sight(spec):
    """The informed search's sight range (spec "sight_m", added 2026-10-28 at the user's request: 80 m was too short
    to mean anything; absent, split_search's 80 m): what the pilot's map counts as seen, the first sight that stops
    the search (also recorded beside FIND_RANGES), and the radius within which the target starts out of the start's
    sight (at least TARGET_HIDDEN_M). Module state set per job: a run's jobs share it. The walks do not read it."""
    g = globals()
    g.update(_SIGHT0)
    m = spec.get("sight_m")
    if m:
        m = float(m)
        g.update(STREET_SEEN_RANGE_M=m, FIND_STOP_M=m, FIND_RANGES=tuple(sorted(set(_SIGHT0["FIND_RANGES"]) | {m})),
                 TARGET_HIDDEN_M=max(_SIGHT0["TARGET_HIDDEN_M"], m))


# ============================================================================= the chain

def block_chain(net, rng, length_m=CHAIN_M):
    """dict(approach [S, J0, J], hops [dict(block, J, entrance, exit, V, F, plaza)]) or None: a seeded block-hop walk.
    A chain may step onto a plaza (a block too small for a building): there is nothing there to surround or pass, so
    every condition crosses it by the streets (`plaza`), and only the blocks with a building are scored per block."""
    for _ in range(50):
        tr = S.pick(net, rng)
        if tr is None:
            return None
        hops, total = [], 0.0
        B, J, e = tr["block"], tr["J"], tr["entrance"]
        while total < length_m:
            ok = exit_options(net, B, e, J)
            if not ok:
                break
            sid, V, F, nb = ok[int(rng.integers(len(ok)))]
            js, _ = net.route(J, V)
            hops.append(dict(block=int(B), J=int(J), entrance=int(e), exit=int(sid), V=int(V), F=int(F),
                             plaza=bool(net.blocks[B]["building"] < 0)))
            total += (net.route_length(js) if js else 0.0) + net.streets[sid]["length"]
            B, J, e = nb, F, sid
        if len(hops) >= 4:
            return dict(approach=tr["approach"], hops=hops, length_m=round(total, 1))
    return None


# ============================================================================= waypoints per condition

def pt(net, j):
    return float(net.jpos[j][0]), float(net.jpos[j][1])


def approach(net, approach_js, d_street):
    """The waypoints from the start junction to the first block's entrance corner (its last one)."""
    S0, J0, J = approach_js
    out, prev = [], net.jpos[S0]
    for j in (J0, J):
        q = net.jpos[j]
        out.append((*pt(net, j), 1.0, S.timeout_for(float(np.linalg.norm(q - prev))), d_street))
        prev = q
    return out


def hop_pre(cond, C, h, st, db):
    """A block's waypoints between arriving at its entrance corner J and the decision (none for the conditions that
    decide on arrival, and for swarm_pass, which decides at J and is aimed by the decision). h: dict(block, J,
    entrance)."""
    net = C["net"]
    if cond == "single_look":
        out = []
        for sid in net.j_streets[h["J"]]:
            if sid == h["entrance"]:
                continue
            d = net.jpos[net.other(sid, h["J"])] - net.jpos[h["J"]]
            look = net.jpos[h["J"]] + d / max(float(np.linalg.norm(d)), 1e-9) * LOOK_AHEAD_M
            out.append((float(look[0]), float(look[1]), 0.0, 180.0 / camera.DRONE_MAX_YAW_RATE_DEG + LOOK_DWELL_S,
                        st["d_street"]))
        return out
    if cond == "swarm_hold":
        return [(*pt(net, h["J"]), -st["hold_stick"], st["t_hop"], db)]
    if cond == "swarm_hop":
        B = net.blocks[h["block"]]
        if st.get("m3b_mode", "hold") == "aim":
            op = net.opposite_exit(h["block"], h["entrance"], h["J"])
            # aimed through the block at its opposite exit, or (no opposite exit) at its centre: never at the exit
            # the decision picks, so the surround is the same whatever comes after it
            aim = S.aim_point(net, *op)[0] if op is not None else B["centre"]
            return [(float(aim[0]), float(aim[1]), 1.0, st["t_hop"], db)]
        return [(float(B["centre"][0]), float(B["centre"][1]), -st["hold_stick"], st["t_hop"], db)]
    return []


def hop_post(cond, C, h, st, db):
    """A block's waypoints after the decision, to the next decision point F (the exit street's far junction).
    h: dict(block, J, entrance, exit, V, F)."""
    net = C["net"]
    B = net.blocks[h["block"]]
    d_street = st["d_street"]
    if cond == "swarm_pass":
        aim, _ = S.aim_point(net, h["exit"], h["V"])
        out = [(float(aim[0]), float(aim[1]), 1.0, S.timeout_for(0.5 * geo.perimeter(B["kerb"]) + S.AIM_PAST_M), db)]
        prev = aim
    elif cond == "swarm_hop":
        out = [(*pt(net, h["V"]), st["exit_stick"], S.timeout_for(0.5 * geo.perimeter(B["kerb"])), d_street)]
        prev = net.jpos[h["V"]]
    else:
        js, _ = net.route(h["J"], h["V"])
        out, prev = [], net.jpos[h["J"]]
        for j in (js or [h["J"], h["V"]])[1:]:
            q = net.jpos[j]
            out.append((*pt(net, j), 1.0, S.timeout_for(float(np.linalg.norm(q - prev))), d_street))
            prev = q
    q = net.jpos[h["F"]]
    out.append((*pt(net, h["F"]), 1.0 if cond in ("single", "single_look") else st["exit_stick"],
                S.timeout_for(float(np.linalg.norm(q - prev))), d_street))
    return out


def commit_offset(cond, n_pre):
    """Where a block's decision falls, as an offset from the index its pre waypoints start at: the last pre waypoint,
    the arrival at J (-1) when there are none, swarm_pass's aim point (its first post waypoint)."""
    return n_pre if cond == "swarm_pass" else n_pre - 1


def condition_wps(cond, C, chain, st, d_blk_fn):
    """(wps, marks): the condition's waypoints over the whole chain, and per hop dict(commit, next) waypoint indices
    (commit: the decision; next: the next decision point, F)."""
    wps = approach(C["net"], chain["approach"], st["d_street"])
    marks = []
    for h in chain["hops"]:
        c = plaza_cond(cond, C["net"], h["block"])
        db = d_blk_fn(h["block"])
        pre, post = hop_pre(c, C, h, st, db), hop_post(c, C, h, st, db)
        commit = len(wps) + commit_offset(c, len(pre))
        wps += pre + post
        marks.append(dict(commit=commit, next=len(wps) - 1, hop=len(marks),
                          plaza=bool(C["net"].blocks[h["block"]]["building"] < 0)))
    return wps, marks


def block_spread_fn(spec, st, net, L, variant, p, s, spread):
    """The block-phase spread for one condition: the dial centre (a constant) or the S4 profile's rule."""
    if spread != "profile":
        return lambda block: float(spread)
    if st["d_rule"] == "perimeter":
        return lambda block: S.d_block(st, net, block, L)
    d = float(spec["d_star"].get(f"{variant}|{p:g}|{s:g}", st["d_block"]))
    return lambda block: d


# ============================================================================= scoring

def first_seen(fp, samples, pts, h, pitch, r=S.PREVIEW_RANGE_M):
    """(len(pts),) the first time each point was seen on a shown feed: split_probe.preview_first's rule (los.sees at
    r, the vertical band), with each camera's candidate points found through a k-d tree, since here the points are a
    whole city's street samples."""
    from scipy.spatial import cKDTree
    first = np.full(len(pts), np.inf)
    if not len(pts):
        return first
    tree = cKDTree(pts)
    for smp in samples:
        for k in range(len(smp["yaws"])):
            near = np.asarray(tree.query_ball_point(smp["cams"][k], r), int)
            if not len(near):
                continue
            todo = near[~np.isfinite(first[near])]
            if not len(todo):
                continue
            vis = los.sees(fp, smp["cams"][k], h, smp["yaws"][k], pitch, pts[todo], r_id=r, band=True)[0]
            first[todo[vis]] = smp["t"]
    return first


def hop_preview(net, first_by_band, cands, commit_t):
    """split_probe.preview_scores from first-seen times already computed over every street sample."""
    out = {}
    for name, first in first_by_band.items():
        times = []
        for sid, j in cands:
            f = np.sort(first[net.street_samples(sid, j, S.PREVIEW_REACH_M)])
            need = max(1, int(math.ceil(S.PREVIEW_FRAC * len(f))))
            times.append(float(f[need - 1]) if len(f) else math.inf)
        times = np.array(times)
        out[f"preview_{name}"] = float((times <= commit_t).mean()) if len(times) else math.nan
    return out


class Rec:
    """The FlightRecord fields clutter.discovery_first reads, from split_probe samples."""

    def __init__(self, samples, single):
        self.single = single
        self.t = np.array([s["t"] for s in samples])
        K = max((len(s["yaws"]) for s in samples), default=0)
        n = len(samples)
        self.cam_valid = np.zeros((n, max(K, 1)), bool)
        self.cam_pos = np.zeros((n, max(K, 1), 2))
        self.cam_yaw = np.zeros((n, max(K, 1)))
        for i, s in enumerate(samples):
            k = len(s["yaws"])
            self.cam_valid[i, :k] = True
            self.cam_pos[i, :k] = s["cams"]
            self.cam_yaw[i, :k] = s["yaws"]
        self.body_yaw = np.array([s["body"] for s in samples])


def score_flight(C, cond, samples, done, marks, chain, sim, n0, T, grid, in_arena, bands, corr_fp):
    net, fp = C["net"], C["fp"]
    row = dict(condition=cond, lost=int(n0 - sim.alive.sum()), kills=[k[1] for k in sim.kills],
               contacts=int(sim.contacts), t_end=round(float(samples[-1]["t"]) if samples else 0.0, 2))
    row["feeds_mean"] = round(float(np.mean([len(s["yaws"]) for s in samples])), 3) if samples else 0.0
    first = {name: first_seen(fp, samples, net.sample_xz, hh, pp) for name, (hh, pp) in bands.items()}
    # per hop: preview at the commit, reform before the next decision
    hops = []
    for mk, h in zip(marks, chain["hops"]):
        c_t = done[mk["commit"]] if mk["commit"] < len(done) else math.nan
        n_t = done[mk["next"]] if mk["next"] < len(done) else math.nan
        if not math.isfinite(c_t):
            break
        tr = dict(block=h["block"], entrance=h["entrance"], J=h["J"])
        pv = hop_preview(net, first, S.candidates(net, tr), c_t)
        r = dict(hop=mk["hop"], plaza=bool(mk.get("plaza", False)), commit_t=round(c_t, 2),
                 next_t=round(n_t, 2) if math.isfinite(n_t) else None,
                 preview_search=pv.get("preview_search"), preview_transit=pv.get("preview_transit"))
        if cond in SWARM:
            ser = S.cluster_series(net, samples, t0=c_t, t1=n_t if math.isfinite(n_t) else math.inf)
            r["reform_s"], r["reform_ok"] = S.reform(ser, c_t, n_t if math.isfinite(n_t) else samples[-1]["t"])
        hops.append(r)
    row["hops"] = hops
    row["hops_done"] = len(hops)
    # street clearance: interior street samples seen within 80 m on the shown feeds (search band)
    inter = np.array([net.streets[s]["interior"] for s in net.sample_street], bool)
    seen = np.isfinite(first["search"]) & inter
    row["street_seen_share"] = round(float(seen.sum() / max(inter.sum(), 1)), 4)
    row["street_seen_km"] = round(float(seen.sum() * net.sample_m / 1000.0), 3)
    row["street_km_per_min"] = round(row["street_seen_km"] / max(T / 60.0, 1e-9), 4)
    # time to find a hidden point (S3's measure), cells the start shows excluded
    rec = Rec(samples, cond in ("single", "single_look"))
    h, p = bands["search"]
    for r in FIND_RANGES:
        f_any, f_head = clutter.discovery_first(rec, fp, grid, h, p, r=r)
        start_vis = f_any <= (samples[0]["t"] + 0.01 if samples else 0.0)
        mask = in_arena & ~start_vis
        for basis, f in (("any", f_any), ("head", f_head)):
            ff = np.minimum(f[mask], T)
            row[f"find_rmst_{basis}_r{r:g}"] = round(float(ff.mean()), 2) if len(ff) else None
            row[f"find_p_{basis}_r{r:g}"] = round(float((f[mask] <= T).mean()), 4) if len(ff) else None
    # view correspondence (swarm only)
    if cond in SWARM and corr_fp is not None and len(samples):
        ring = correspondence.Ring(corr_fp, C["cells"], h, p)
        step = max(1, int(round(CORR_EVERY_S / S.SAMPLE_S)))
        for smp in samples[::step]:
            if len(smp["yaws"]) >= 2:
                ring.add(smp["t"], smp["cams"], smp["yaws"], smp["body"], smp["c"])
        row.update(ring.summary())
    # formation: split share over the flight
    if cond in SWARM:
        ser = S.cluster_series(net, samples[::2])
        row["split_share"] = round(float(np.mean([x[2] for x in ser])), 3) if ser else math.nan
    return row


# ============================================================================= the informed (frontier) pilot

class Knowledge:
    """What one condition's pilot has seen: each street sample's first sight on the shown feeds (within
    STREET_SEEN_RANGE_M, the search band), and the hidden target's first sight at each FIND_RANGES range, brought up
    to date from the flight's samples as they arrive. basis "head": only what falls inside the headset's view with the
    head facing travel (the single drone's one feed always does)."""

    def __init__(self, net, fp, h, pitch, target=None, basis="any", single=False):
        from scipy.spatial import cKDTree
        self.net, self.fp, self.h, self.pitch, self.basis, self.single = net, fp, h, pitch, basis, single
        self.first = np.full(len(net.sample_xz), np.inf)
        self.tree = cKDTree(net.sample_xz)
        self.target = None if target is None else np.asarray(target, float)
        self.found = {r: math.inf for r in FIND_RANGES}
        self.found_head = {r: math.inf for r in FIND_RANGES}
        self.k = 0

    def _head(self, yaw, body, off):
        if self.single:
            return np.ones(len(off), bool)
        return np.abs(los.wrap_deg(yaw - body + off * camera.DISPLAY_SCALE)) <= camera.HMD_HALF_FOV_DEG

    def update(self, samples):
        for smp in samples[self.k:]:
            for k in range(len(smp["yaws"])):
                cam, yaw = smp["cams"][k], float(smp["yaws"][k])
                near = np.asarray(self.tree.query_ball_point(cam, STREET_SEEN_RANGE_M), int)
                todo = near[~np.isfinite(self.first[near])] if len(near) else near
                if len(todo):
                    vis, off, _ = los.sees(self.fp, cam, self.h, yaw, self.pitch, self.net.sample_xz[todo],
                                           r_id=STREET_SEEN_RANGE_M, band=True)
                    if self.basis == "head":
                        vis &= self._head(yaw, smp["body"], off)
                    self.first[todo[vis]] = smp["t"]
                if self.target is not None:
                    for r in FIND_RANGES:
                        if math.isfinite(self.found_head[r]):
                            continue
                        vis, off, _ = los.sees(self.fp, cam, self.h, yaw, self.pitch, self.target[None], r_id=r,
                                               band=True)
                        if vis[0]:
                            if not math.isfinite(self.found[r]):
                                self.found[r] = smp["t"]
                            if self._head(yaw, smp["body"], off)[0]:
                                self.found_head[r] = smp["t"]
        self.k = len(samples)

    def unseen_m(self, sids):
        """Unseen street length (m) on the interior streets sids."""
        tot = 0.0
        for sid in sids:
            if not self.net.streets[sid]["interior"]:
                continue
            idx = np.flatnonzero(self.net.sample_street == sid)
            tot += float((~np.isfinite(self.first[idx])).sum()) * self.net.sample_m
        return tot


def junction_dist(net, sources):
    """{junction: street distance to the nearest of sources} (multi-source Dijkstra)."""
    import heapq
    dist = {j: 0.0 for j in sources}
    pq = [(0.0, j) for j in sources]
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        for v, w, _ in net.adj.get(u, ()):
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return dist


def exit_options(net, B, entrance, J):
    """[(sid, V, F, next block)] the chain's exit rule: an interior radiating street of block B other than the
    entrance and those at the entrance corner, leading to an interior block (a plaza included)."""
    out = []
    for sid, V in net.exits(B, entrance):
        if not net.streets[sid]["interior"] or V == J:
            continue
        F = net.other(sid, V)
        nb = net.block_ahead(F, sid) if net.j_interior[F] else None
        if nb is not None and nb in net.blocks and net.blocks[nb]["interior"]:
            out.append((sid, V, F, nb))
    return out


def fallback_exits(net, B, entrance, J):
    """The informed pilot's exits when the chain's rule leaves none (every other exit leads off the interior, which at
    p 40 with 24 m streets stopped about a third of the searches early): a street at the entrance corner, and failing
    that back down the entrance street. The chain walks never use it, so they lay exactly as before.
    [(sid, V, F, next block)]"""
    out = []
    for sid, V in net.exits(B, entrance):
        if not net.streets[sid]["interior"]:
            continue
        F = net.other(sid, V)
        nb = net.block_ahead(F, sid) if net.j_interior[F] else None
        if nb is not None and nb in net.blocks and net.blocks[nb]["interior"]:
            out.append((sid, V, F, nb))
    if out:
        return out
    F = net.other(entrance, J)
    nb = net.block_ahead(F, entrance) if net.j_interior[F] else None
    if nb is not None and nb in net.blocks and net.blocks[nb]["interior"]:
        return [(entrance, J, F, nb)]
    return []


def plaza_cond(cond, net, block):
    """The condition a block is flown under: a plaza is crossed by the streets, as one drone or the street-bound swarm
    would cross it."""
    if net.blocks[block]["building"] >= 0:
        return cond
    return "single" if cond in ("single", "single_look") else "swarm_street"


def choose_exit(net, know, B, entrance, J, rng, min_unseen_m=FRONTIER_MIN_M):
    """The informed pilot's exit at block B: the most unseen street length within two edges of the exit (the exit
    street and every street at its far junction) per metre of travel to that far junction; when no exit has any, the
    one whose far junction is nearest by street to a junction with an interior street carrying at least min_unseen_m
    unseen. Ties broken by rng. (sid, V, F, next block, scored) or None."""
    opts = exit_options(net, B, entrance, J) or fallback_exits(net, B, entrance, J)
    if not opts:
        return None
    scores = []
    for sid, V, F, nb in opts:
        js, _ = net.route(J, V)
        travel = (net.route_length(js) if js else 0.0) + net.streets[sid]["length"]
        sids = {sid} | set(net.j_streets[F])
        scores.append(know.unseen_m(sids) / max(travel, 1.0))
    scores = np.array(scores)
    if scores.max() > 0:
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return (*opts[int(rng.choice(best))], True)
    src = {j for st in net.streets if st["interior"] and know.unseen_m([st["id"]]) >= min_unseen_m
           for j in (st["u"], st["v"])}
    if not src:
        return (*opts[int(rng.integers(len(opts)))], False)
    dist = junction_dist(net, src)
    d = np.array([dist.get(F, math.inf) for _, _, F, _ in opts])
    best = np.flatnonzero(d <= d.min() + 1e-9)
    return (*opts[int(rng.choice(best))], False)


def place_target(C, start_j, rng, tries=400):
    """A hidden point on the open ground of an interior block's lawn or an interior street, at least TARGET_STREET_M
    by street from the start and out of clear sight of it within TARGET_HIDDEN_M: (xz) or None."""
    net, fp = C["net"], C["fp"]
    pts = C["find_grid"].pts
    inner = pts[C["target_ok"]]
    if not len(inner):
        return None
    dist = junction_dist(net, [start_j])
    s0 = net.jpos[start_j]
    for _ in range(tries):
        q = inner[int(rng.integers(len(inner)))]
        j = net.nearest_junction(q)
        if dist.get(j, 0.0) < TARGET_STREET_M:
            continue
        if np.linalg.norm(q - s0) <= TARGET_HIDDEN_M and not fp.blocked(s0[None], q[None])[0]:
            continue
        return q
    return None


def informed_flight(cond, C, variant, start, target, st, d_blk_fn, seed, T_cap, bands, basis="any"):
    """One condition's search for the hidden target under the informed pilot: block by block, the exit chosen at the
    decision from what this flight has shown so far. Stops at the first sight at FIND_STOP_M, at T_cap, or (swarm)
    under MIN_SEARCH_DRONES. -> dict."""
    net = C["net"]
    h, pitch = bands["search"]
    single = cond in ("single", "single_look")
    m = S.model(st, variant)
    P0 = net.jpos[start["approach"][0]][None] if single else S.start_positions(C, start, st, seed)
    f = S.Flight(m, C["rcity"], P0, st["d_street"], seed, pitch, hold_gain=float(st["hold_gain"]))
    know = Knowledge(net, C["fp"], h, pitch, target, basis, single)
    rng = np.random.default_rng(seed + 7)
    f.add(approach(net, start["approach"], st["d_street"]))
    B, J, e = start["block"], start["J"], start["entrance"]
    decisions = blind = 0
    min_alive = 1 if single else MIN_SEARCH_DRONES

    def go(until):
        while True:
            t_end = min(T_cap, f.sim.t + DECISION_STEP_S)
            f.run(t_end, until_wp=until)
            know.update(f.samples)
            if (math.isfinite(know.found[FIND_STOP_M]) or f.sim.t >= T_cap - 1e-9 or
                    f.sim.alive.sum() < min_alive or f.wp > until or f.wp >= len(f.wps)):
                return

    go(len(f.wps) - 1)                                   # to the first entrance corner
    while (not math.isfinite(know.found[FIND_STOP_M]) and f.sim.t < T_cap - 1e-9 and f.sim.alive.sum() >= min_alive
           and f.wp >= len(f.wps)):
        db = d_blk_fn(B)
        hh = dict(block=B, J=J, entrance=e)
        c = plaza_cond(cond, net, B)
        pre = hop_pre(c, C, hh, st, db)
        if pre:
            f.add(pre)
            go(len(f.wps) - 1)
            if math.isfinite(know.found[FIND_STOP_M]) or f.wp < len(f.wps):
                break
        ch = choose_exit(net, know, B, e, J, rng)
        if ch is None:
            break
        sid, V, F, nb, scored = ch
        decisions += 1
        blind += int(not scored)
        f.add(hop_post(c, C, dict(hh, exit=sid, V=V, F=F), st, db))
        go(len(f.wps) - 1)
        B, J, e = nb, F, sid
    alive = int(f.sim.alive.sum())
    found = know.found                  # the flight stops once a swarm is under four drones: nothing later is seen
    path = sum(float(np.linalg.norm(b["c"] - a["c"])) for a, b in zip(f.samples[:-1], f.samples[1:]))
    inter = np.array([net.streets[s]["interior"] for s in net.sample_street], bool)
    return dict(condition=cond, basis=basis, t_end=round(float(f.sim.t), 2), alive=alive, lost=int(len(P0) - alive),
                contacts=int(f.sim.contacts), decisions=decisions, blind_decisions=blind, path_m=round(path, 1),
                street_seen_share=round(float((np.isfinite(know.first) & inter).sum() / max(inter.sum(), 1)), 4),
                **{f"found_r{r:g}": (round(v, 2) if math.isfinite(v) else None) for r, v in found.items()},
                **{f"found_head_r{r:g}": (round(v, 2) if math.isfinite(v) else None)
                   for r, v in know.found_head.items()})


def search_job(job):
    """One city's k-th target: every condition (and the head-basis variants) searches for it from the same start."""
    t0 = time.time()
    base = dict(job_id=job["job_id"], variant=job["variant"], p=job["p"], s=job["s"], city_seed=job["city_seed"],
                target_i=job["target_i"], T_s=float(job["T_s"]))
    try:
        set_sight(job["spec"])
        spec, st = job["spec"], dict(S.DEFAULT_SETTING, **job["spec"]["setting"])
        C = context(job["variant"], job["p"], job["s"], job["city_seed"])
        rng = np.random.default_rng(S.hash_seed(dict(kind="search", variant="target", p=job["p"], s=job["s"],
                                                     city_seed=job["city_seed"], trial=job["target_i"])))
        seed = int(rng.integers(1 << 30))
        start = S.pick(C["net"], rng)
        if start is None:
            return dict(base, skipped=True)
        target = place_target(C, start["approach"][0], rng)
        if target is None:
            return dict(base, skipped=True, why="no target")
        dfn = block_spread_fn(spec, st, C["net"], C["L"], job["variant"], job["p"], job["s"], "profile")
        rows = []
        flights = [(c, "any") for c in CONDITIONS]
        if job["target_i"] % 2 == 0:
            flights += [(c, "head") for c in HEAD_BASIS]
        for cond, basis in flights:
            rows.append(informed_flight(cond, C, job["variant"], start, target, st, dfn, seed, float(job["T_s"]),
                                        S.bands(), basis))
        return dict(base, rows=rows, target=[round(float(x), 2) for x in target], wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


# ============================================================================= jobs

_CTX = {}


def context(variant, p, s, seed):
    key = (variant, p, s, seed)
    if key not in _CTX:
        _CTX.clear()
        C = S.city(variant, p, s, seed)
        L = C["L"]
        x0, x1, z0, z1 = blockcity.ARENA
        m = clutter.RANGE_M + 4.0
        C = dict(C)
        C["cells"] = clutter.arena_cells(C["fp"], (x0 - m, x1 + m, z0 - m, z1 + m))
        C["find_grid"] = clutter.arena_cells(C["fp"], (x0, x1, z0, z1), step=clutter.FIND_CELL_M)
        q = C["find_grid"].pts
        C["in_arena"] = np.ones(len(q), bool)
        e = TARGET_EDGE_M
        C["target_ok"] = (q[:, 0] > x0 + e) & (q[:, 0] < x1 - e) & (q[:, 1] > z0 + e) & (q[:, 1] < z1 - e)
        C["corr_fp"] = C["fp"] if isinstance(C["fp"], ConvexFootprints) else \
            ConvexFootprints([blockcity.building_poly(b) for b in L.boxes])
        _CTX[key] = C
    return _CTX[key]


def walk_job(job):
    """One city's k-th walk: the chain, then every condition at every spread it flies."""
    t0 = time.time()
    base = dict(job_id=job["job_id"], variant=job["variant"], p=job["p"], s=job["s"], city_seed=job["city_seed"],
                walk=job["walk"])
    try:
        spec, st = job["spec"], dict(S.DEFAULT_SETTING, **job["spec"]["setting"])
        C = context(job["variant"], job["p"], job["s"], job["city_seed"])
        rng = np.random.default_rng(S.hash_seed(dict(kind="walk", variant="chain", p=job["p"], s=job["s"],
                                                     city_seed=job["city_seed"], trial=job["walk"])))
        seed = int(rng.integers(1 << 30))
        chain = block_chain(C["net"], rng)
        if chain is None:
            return dict(base, skipped=True)
        bands = S.bands()
        T = float(job["T_s"])
        rows = []
        flown = {}                   # a flight whose waypoints and spreads match one already flown is that flight
        for cond in CONDITIONS:
            spreads = spec["spreads"] if cond in SWARM else ["profile"]
            for spread in spreads:
                dfn = block_spread_fn(spec, st, C["net"], C["L"], job["variant"], job["p"], job["s"], spread)
                stc = dict(st) if spread == "profile" else dict(st, d_street=float(spread))
                wps, marks = condition_wps(cond, C, chain, stc, dfn)
                key = (cond, stc["d_street"], tuple(wps))
                if key in flown:
                    rows.append(dict(flown[key], spread=spread, same_as=flown[key]["spread"]))
                    continue
                m = S.model(stc, job["variant"])
                if cond in SWARM:
                    P0 = S.start_positions(C, dict(approach=chain["approach"]), stc, seed)
                else:
                    P0 = C["net"].jpos[chain["approach"][0]][None]
                f = S.Flight(m, C["rcity"], P0, stc["d_street"], seed, bands["search"][1],
                             hold_gain=float(stc["hold_gain"]))
                f.add(wps)
                f.run(T)
                row = score_flight(C, cond, f.samples, f.done_t, marks, chain, f.sim, len(P0), T, C["find_grid"],
                                   C["in_arena"], bands, C["corr_fp"])
                row["spread"] = spread
                rows.append(row)
                flown[key] = row
        return dict(base, rows=rows, chain_m=chain["length_m"], hops_planned=len(chain["hops"]),
                    wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


def load_spec(path=SPEC):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_walks(spec, quick=False):
    designs = [tuple(d) for d in spec["designs"]]
    cities = int(spec["cities_per_design"])
    walks = int(spec["walks_per_city"])
    T = float(spec["T_s"])
    if quick:
        designs, cities, walks, T = designs[:QUICK["designs"]], QUICK["cities"], QUICK["walks"], QUICK["T_s"]
    seed0 = int(spec.get("city_seed0", 0))           # Phase 2's cities apart from the ones Phase 1 tuned on
    rounds = []
    for k in range(walks):
        jobs = []
        for p, s in designs:
            for c in range(seed0, seed0 + cities):
                jid = f"walk|{spec['variant']}|p{p:g}_s{s:g}|seed{c}|{k}"
                jobs.append(dict(job_id=jid, variant=spec["variant"], p=float(p), s=float(s), city_seed=c, walk=k,
                                 T_s=T, spec=spec))
        np.random.default_rng(11 + k).shuffle(jobs)
        rounds.append(jobs)
    return rounds


def plan_search(spec, quick=False):
    """Rounds of informed-search jobs: round k is every city's k-th target."""
    designs = [tuple(d) for d in spec["designs"]]
    cities = int(spec["cities_per_design"])
    targets = int(spec.get("targets_per_city", 6))
    T = float(spec.get("search_T_s", SEARCH_T_S))
    if quick:
        designs, cities, targets, T = (designs[:QUICK_SEARCH["designs"]], QUICK_SEARCH["cities"],
                                       QUICK_SEARCH["targets"], QUICK_SEARCH["T_s"])
    seed0 = int(spec.get("city_seed0", 0))
    rounds = []
    for k in range(targets):
        jobs = []
        for p, s in designs:
            for c in range(seed0, seed0 + cities):
                jid = f"search|{spec['variant']}|p{p:g}_s{s:g}|seed{c}|{k}"
                jobs.append(dict(job_id=jid, variant=spec["variant"], p=float(p), s=float(s), city_seed=c,
                                 target_i=k, T_s=T, spec=spec))
        np.random.default_rng(31 + k).shuffle(jobs)
        rounds.append(jobs)
    return rounds


def cmd_walks(a):
    run_rounds(a, plan_walks, walk_job, "walks")


def cmd_search(a):
    run_rounds(a, plan_search, search_job, "search")


def run_rounds(a, planner, job_fn, kind):
    """S3's run pattern: rounds as replicates, results appended as they arrive, --resume, a time budget."""
    from citysim import parallel
    spec = load_spec(a.spec)
    if getattr(a, "sight", None):
        spec["sight_m"] = float(a.sight)
    name = a.name + ("_quick" if a.quick else "")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    if res.exists() and not a.resume:
        res.rename(out / f"results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    done = clutter._done_ids(res)
    (out / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    clutter.keep_awake()
    t0 = time.time()
    hours = float(a.hours if a.hours is not None else spec.get("hours", 12.5))
    rounds = planner(spec, a.quick)
    # every city's layout written here, before the pool: workers generating a missing one would race on its file
    for key in sorted({(j["variant"], j["p"], j["s"], j["city_seed"]) for jobs in rounds for j in jobs}):
        blockcity.ensure(blockcity.Params(*key))
    times = []
    for k, jobs in enumerate(rounds):
        todo = [j for j in jobs if j["job_id"] not in done]
        if not todo:
            continue
        mean = float(np.mean(times)) if times else 0.0
        if (time.time() - t0) + mean > hours * 3600 - 1800:
            _log(out, f"stopping before round {k}: the budget would be overrun")
            break
        r0 = time.time()
        _log(out, f"{kind} round {k}: {len(todo)} jobs")
        parallel.run_jobs(job_fn, todo, f"{kind} round {k}", processes=a.jobs, stall_s=900.0,
                          progress=lambda i, n, r: clutter._append(res, r))
        times.append(time.time() - r0)
        _log(out, f"{kind} round {k} done in {times[-1] / 60:.1f} min")
    recheck(out, rounds, job_fn, a.jobs, kind)
    _log(out, f"finished in {(time.time() - t0) / 3600:.2f} h")


def recheck(out, rounds, job_fn, procs, kind):
    """RECHECK_SHARE of the finished jobs flown again: a flight is deterministic in its seed, so anything but an exact
    match is this machine's CPU (lesson 16) -> recheck.csv."""
    from citysim import parallel
    import pandas as pd
    res = out / "results.jsonl"
    by = {}
    for line in res.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in r:
            by[r["job_id"]] = r
    jobs = {j["job_id"]: j for jobs in rounds for j in jobs if j["job_id"] in by}
    if not jobs:
        return
    rng = np.random.default_rng(97)
    ids = sorted(jobs)
    pick = [ids[i] for i in rng.choice(len(ids), max(1, int(round(clutter.RECHECK_SHARE * len(ids)))), replace=False)]
    again = {}
    parallel.run_jobs(job_fn, [jobs[i] for i in pick], f"{kind} recheck", processes=procs, stall_s=900.0,
                      progress=lambda i, n, r: again.__setitem__(r["job_id"], r))

    def strip(r):
        return json.dumps({k: v for k, v in r.items() if k != "wall_s"}, sort_keys=True, default=clutter._jsonable)

    rows = [dict(job_id=i, identical=(i in again and "error" not in again[i] and strip(again[i]) == strip(by[i])))
            for i in pick]
    pd.DataFrame(rows).to_csv(out / "recheck.csv", index=False)
    _log(out, f"recheck: {sum(r['identical'] for r in rows)} of {len(rows)} identical")


def _log(out, msg):
    line = f"[split {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("walks", "search"):
        p = sub.add_parser(cmd)
        p.add_argument("--name", required=True)
        p.add_argument("--spec", default=str(SPEC))
        p.add_argument("--quick", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--hours", type=float)
        p.add_argument("--jobs", type=int, default=28)
        if cmd == "search":
            p.add_argument("--sight", type=float, help="the search's sight range, m (spec sight_m; default 80)")
    p = sub.add_parser("tables")
    p.add_argument("--name", required=True)
    p = sub.add_parser("primaries")
    p.add_argument("--walks", required=True)
    p.add_argument("--search", required=True)
    a = ap.parse_args()
    if a.cmd == "walks":
        cmd_walks(a)
    elif a.cmd == "search":
        cmd_search(a)
    elif a.cmd == "primaries":
        from citysim import split_stats
        split_stats.primaries(OUT_ROOT / a.walks, OUT_ROOT / a.search)
    else:
        from citysim import split_stats
        split_stats.tables(OUT_ROOT / a.name)


if __name__ == "__main__":
    main()
