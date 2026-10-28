"""S4 Split and Surround, Phase 1 (the initial test): the manoeuvre suite in the replica.

    python -B -m citysim.split_probe coupling   --name N [--trials 30]           # M4: the surround map d*(p, s)
    python -B -m citysim.split_probe manoeuvres --name N --setting KEY=V,... [--trials 50]   # M1, M3, M3b, M6
    python -B -m citysim.split_probe steps      --name N --setting ...          # M5: spread steps with those gains
    python -B -m citysim.split_probe tables     --name N                        # criteria, coupling map, decisions
    -> experiment_results/split_study/phase1/<N>/

Every manoeuvre is flown on block cities (citysim.blockcity, variants R and B, p x s designs) by the replica under
the bench's own pilot (SwarmBenchRunner: a waypoint is taken when the centroid comes within CAPTURE_M of it or its
timeout runs out, a negative stick is a centring hold at HOLD_GAIN capped at |stick|), with the spread set per
waypoint, so the bench can fly the same scenarios. The S4 profile starts from ProceduralCityWorld's SwarmManager
(the scene the bench flies generated layouts in) with r0_coh 40 (cohesion attractive to ~140 m instead of ~75 m).

  M1  street run: ~200 m of interior streets at full stick (and one drone on the same waypoints): speed, contacts,
      wall-filled feeds.
  M3  pass round: approach a block's entrance junction J along a street, aim through the block at a point on the
      opposite exit street (beyond its corner, so it cannot be taken while the formation is round the block), then on
      to that street's far junction. Commit = the aim point taken.
  M3b surround and exit: the same approach, a centring hold on the block's centre for t_hop, then out by a side exit
      (a radiating street that is neither the entrance nor the opposite one) to its far junction. Commit = the end of
      the hold. A matched flight of the same formation round the block by the streets to the same exit is flown too:
      the time price.
  M4  surround hold: the approach, then a 45 s centring hold on the block's centre at a given spread: S1's
      envelopment test (no losses, centre inside the hull, no bearing gap over 150 deg, >= 70 % of the drones
      inside the block's clearance), the time to envelop.
  M6  head-on face (variant B): one block-shaped building alone, flown at from 60 m at full stick through its centre.

Measures. Preview: a candidate street (a radiating street of the block other than the entrance) is previewed when
PREVIEW_FRAC of its 2 m centreline samples within PREVIEW_REACH_M of its corner have been seen on the shown feeds
(display.FeedLatch: hull, debounce, 75 deg/s yaw, ready latch), within PREVIEW_RANGE_M slant through a clear sight
line, at the search band (internal_3's 12.4 m, -20.8 deg: ground from ~11 m) and the transit band (19.5 m, -15 deg).
Clusters: citysim.streetnet's free-space link (30 m). Reform: after the commit, the first sample with one cluster;
it succeeds if that comes before the next decision point (the exit street's far junction), censored there.
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

from citysim import REPO, blockcity, camera, display, formation, los, pilots, streetnet, use_bench_modules  # noqa: E402
from citysim import geometry as geo  # noqa: E402

use_bench_modules()
import bench  # noqa: E402
import swarm_params as sp  # noqa: E402
import swarm_replica as rep  # noqa: E402

SCENE = "ProceduralCityWorld"
N_DRONES = 10
CAPTURE_M = 10.0                 # SwarmBenchRunner's default capture
HOLD_GAIN = 0.25                 # 1/s (formation.PILOT_GAIN, the bench's holdGain)
HOLD_STICK = 0.5                 # |stick| cap of a hold (formation.PILOT_MAX)
SAMPLE_S = 0.5
LATCH_S = 0.1
BODY_RATE = camera.BODY_YAW_RATE_DEG
PREVIEW_FRAC = 0.6
PREVIEW_REACH_M = 80.0
PREVIEW_RANGE_M = 80.0
STALL_SPEED = 1.0
STALL_S = 3.0
AIM_PAST_M = 15.0                # M3's aim point: this far down the opposite exit street from its corner
HOLD_M4_S = 45.0
ENVELOP_MEASURE_S = 10.0
DESIGNS = [(p, s) for p in blockcity.P_LEVELS for s in blockcity.S_LEVELS]
CITY_SEEDS = (0, 1, 2)
OUT_ROOT = REPO / "experiment_results" / "split_study" / "phase1"

PROFILE = dict(r0_coh=40.0)       # the S4 profile's SwarmManager overrides on ProceduralCityWorld
DEFAULT_SETTING = dict(d_street=1.0, d_rule="fixed", d_block=1.5, k=1.0, d_lo=1.0, d_hi=2.8, t_hop=10.0,
                       exit_stick=1.0, face_bend=0.5, r0_coh=40.0, hollow=None, extra_sets="",
                       hold_stick=HOLD_STICK, hold_gain=HOLD_GAIN, m3b_mode="hold")
# hold_stick / hold_gain: M3b's surround hold (S1's is 0.5 / 0.25: gentle); m3b_mode "hold" (a centring hold on the
# block's centre for t_hop) or "aim" (aim through the block at the opposite exit at full stick for t_hop, then turn off
# down the side exit): two ways to make the surround brief.


# ============================================================================= per-worker caches

_CITY, _MODEL, _TEMPLATE = {}, {}, {}


def city(variant, p, s, seed):
    key = (variant, p, s, seed)
    if key not in _CITY:
        if len(_CITY) > 8:
            _CITY.clear()
        P = blockcity.Params(variant, p, s, seed)
        L, blocks = blockcity.load(blockcity.ensure(P))
        fp = blockcity.footprints(L)
        _CITY[key] = dict(L=L, blocks=blocks, fp=fp, net=streetnet.Network(L, blocks, fp),
                          rcity=blockcity.replica_city(L))
    return _CITY[key]


def overrides(st):
    over = dict(r0_coh=float(st["r0_coh"]))
    if st.get("hollow") is not None:
        over["hollowSwarmCore"] = bool(st["hollow"])
    if st.get("extra_sets"):
        over.update(sp.parse_assignments([st["extra_sets"]]))
    return over


def model(st, variant):
    over = overrides(st)
    extras = dict(obstacle_model="closest", face_bend=float(st["face_bend"])) if variant == "B" else {}
    key = (json.dumps(over, sort_keys=True), json.dumps(extras, sort_keys=True))
    if key not in _MODEL:
        formation.check()
        _MODEL[key] = rep.Model(dict(sp.resolve_swarm(SCENE, overrides=over)), dict(sp.resolve_airframe()), extras)
    return _MODEL[key]


def template(d_ref, seed, st):
    key = (round(d_ref, 3), seed % 5, json.dumps(overrides(st), sort_keys=True))
    if key not in _TEMPLATE:
        P, _ = formation.hover_template(d_ref, N_DRONES, seed % 5, SCENE, overrides=overrides(st))
        _TEMPLATE[key] = P
    return _TEMPLATE[key]


# ============================================================================= the flight: the bench's pilot

class Flight:
    """The bench's pilot over the replica, with waypoints that may be added between runs (Phase 2's informed pilot).
    wps: (x, z, stick, timeout, spread); spread < 0 takes d_default. samples every SAMPLE_S: dict(t, c, P (NaN once
    lost), wp, hold, cams (k, 2), yaws (k,), body); done_t[i] the time waypoint i was done (NaN until then). One loop
    for every flight here, so the replica, the bench export and both phases fly the same rule."""

    def __init__(self, m, rcity, P0, d_default, seed, pitch, hold_gain=HOLD_GAIN):
        self.m, self.d_default = m, d_default
        self.sim = rep.ContactSim(m, rcity, np.array(P0, float), np.zeros((len(P0), 2)), seed)
        self.latch = display.FeedLatch(len(P0), look_gap=True, pitch_deg=pitch)
        self.every = max(1, int(round(SAMPLE_S / m.dt)))
        self.latch_every = max(1, int(round(LATCH_S / m.dt)))
        self.hold_gain = hold_gain / m.max_speed
        self.wps, self.done_t, self.samples = [], [], []
        self.wp, self.wp_t0 = 0, 0.0
        self.body, self.feeds = None, None

    def add(self, wps):
        self.wps += [tuple(w) for w in wps]
        self.done_t += [math.nan] * len(wps)
        return self

    def run(self, T, until_wp=None):
        """Fly until time T, every waypoint done, every drone lost, or (until_wp) waypoint until_wp done."""
        sim, dt = self.sim, self.m.dt
        if self.body is None:
            c0 = sim.P.mean(0)
            self.body = _bearing(np.array(self.wps[0][:2]) - c0) if self.wps else 0.0
            self.feeds = self.latch.step(sim.P, sim.alive, self.body, LATCH_S)
        while sim.t < T - 1e-9 and sim.alive.any() and self.wp < len(self.wps):
            c = sim.P[sim.alive].mean(0)
            x, z, stick, timeout, spread = self.wps[self.wp]
            d = np.array([x, z]) - c
            dd = float(np.linalg.norm(d))
            if (stick >= 0 and dd < CAPTURE_M) or (timeout > 0 and sim.t - self.wp_t0 > timeout):
                self.done_t[self.wp] = sim.t
                self.wp, self.wp_t0 = self.wp + 1, sim.t
                if self.wp >= len(self.wps) or (until_wp is not None and self.wp > until_wp):
                    break
                x, z, stick, timeout, spread = self.wps[self.wp]
                d = np.array([x, z]) - c
                dd = float(np.linalg.norm(d))
            hold = stick < 0
            if hold:
                cmd = d * self.hold_gain
                sn = float(np.linalg.norm(cmd))
                if sn > -stick:
                    cmd = cmd * (-stick / sn)
            else:
                cmd = d / max(dd, 1e-9) * min(max(stick, 0.0), 1.0)
                self.body = pilots.turn_toward(self.body, _bearing(d), BODY_RATE, dt)
            if sim.tick % self.latch_every == 0:
                self.feeds = self.latch.step(sim.P, sim.alive, self.body, LATCH_S)
            if sim.tick % self.every == 0:
                self.samples.append(dict(t=sim.t, c=c.copy(), P=np.where(sim.alive[:, None], sim.P, np.nan),
                                         wp=self.wp, hold=hold, cams=np.array(self.feeds.pos, float).reshape(-1, 2),
                                         yaws=np.array(self.feeds.yaw, float), body=self.body))
            sim.step(cmd, 0.0, spread if spread >= 0 else self.d_default)
        return self


def fly(m, rcity, P0, wps, d_default, T, seed, pitch, hold_gain=HOLD_GAIN):
    """One flight of a fixed waypoint list: (sim, samples, done_t) (see Flight)."""
    f = Flight(m, rcity, P0, d_default, seed, pitch, hold_gain).add(wps).run(T)
    return f.sim, f.samples, f.done_t


def _bearing(v):
    return float(math.degrees(math.atan2(v[0], v[1])))


def timeout_for(seg_m):
    return bench.GOAL_WP_SLACK_S + seg_m / bench.GOAL_MIN_PROGRESS


# ============================================================================= trial geometry

def pick(net, rng):
    """One interior block with an approach: dict(block, J (entrance corner), entrance (street), approach junctions
    [S, J0, J]) or None."""
    blocks = net.interior_blocks()
    for _ in range(400):
        B = blocks[int(rng.integers(len(blocks)))]
        rad = [sid for sid in B["radiating"] if net.streets[sid]["interior"]]
        if not rad:
            continue
        e = rad[int(rng.integers(len(rad)))]
        st = net.streets[e]
        J = st["u"] if st["u"] in B["corners"] else st["v"]
        J0 = net.other(e, J)
        if net.block_ahead(J, e) != B["id"]:
            continue
        back = [sid for sid in net.j_streets[J0] if sid != e and net.streets[sid]["interior"]]
        if not back:
            continue
        b = back[int(rng.integers(len(back)))]
        S = net.other(b, J0)
        if not net.j_interior[S]:
            continue
        return dict(block=B["id"], J=J, entrance=e, approach=[S, J0, J])
    return None


def d_block(st, net, B, L):
    """The spread the formation takes at block B."""
    if st["d_rule"] == "perimeter":
        b = L.boxes[net.blocks[B]["building"]]
        if blockcity.is_prism(b):
            per = geo.perimeter(blockcity.building_poly(b)) + 2 * math.pi * blockcity.STANDOFF_M
        else:
            per = 2 * math.pi * (blockcity.cylinder_radius(b) + blockcity.STANDOFF_M)
        return float(np.clip(st["k"] * per / 100.0, st["d_lo"], st["d_hi"]))
    return float(st["d_block"])


def approach_wps(net, tr, st):
    S, J0, J = tr["approach"]
    pts = [net.jpos[J0], net.jpos[J]]
    out, prev = [], net.jpos[S]
    for q in pts:
        out.append((float(q[0]), float(q[1]), 1.0, timeout_for(float(np.linalg.norm(q - prev))), st["d_street"]))
        prev = q
    return out


def start_positions(C, tr, st, seed):
    S = tr["approach"][0]
    P = template(st["d_street"], seed, st) + C["net"].jpos[S]
    return formation.clear_of_buildings(C["rcity"], P, C["net"].jpos[S])


def candidates(net, tr):
    """[(street id, corner junction)] the block's exits (every radiating street but the entrance), interior ones."""
    return [(sid, j) for sid, j in net.exits(tr["block"], tr["entrance"]) if net.streets[sid]["interior"]]


# ============================================================================= scoring

def preview_first(fp, samples, pts, h, pitch, until_t=math.inf):
    """(len(pts),) the first time each point was seen on a shown feed (inf if never, up to until_t)."""
    first = np.full(len(pts), np.inf)
    if not len(pts):
        return first
    for smp in samples:
        if smp["t"] > until_t:
            break
        for k in range(len(smp["yaws"])):
            todo = np.flatnonzero(~np.isfinite(first))
            if not len(todo):
                return first
            vis = los.sees(fp, smp["cams"][k], h, smp["yaws"][k], pitch, pts[todo], r_id=PREVIEW_RANGE_M, band=True)[0]
            first[todo[vis]] = smp["t"]
    return first


def preview_scores(net, fp, samples, cands, commit_t, bands):
    """Per band: share of candidate streets previewed by commit_t, and each street's time to preview."""
    idx = [net.street_samples(sid, j, PREVIEW_REACH_M) for sid, j in cands]
    allidx = np.concatenate(idx) if idx else np.zeros(0, int)
    pts = net.sample_xz[allidx]
    out = {}
    for name, (h, pitch) in bands.items():
        first = preview_first(fp, samples, pts, h, pitch)
        k, times = 0, []
        for ix in idx:
            f = np.sort(first[k:k + len(ix)])
            k += len(ix)
            need = max(1, int(math.ceil(PREVIEW_FRAC * len(f))))
            times.append(float(f[need - 1]) if len(f) else math.inf)
        times = np.array(times)
        out[f"preview_{name}"] = float((times <= commit_t).mean()) if len(times) else math.nan
        out[f"preview_times_{name}"] = [round(t, 2) if math.isfinite(t) else None for t in times]
    return out


def cluster_series(net, samples, t0=-math.inf, t1=math.inf):
    """[(t, n clusters incl. singletons, split flag)] for samples in [t0, t1]."""
    out = []
    for smp in samples:
        if smp["t"] < t0 or smp["t"] > t1:
            continue
        P = smp["P"]
        Pa = P[np.isfinite(P).all(1)]
        if len(Pa) < 2:
            out.append((smp["t"], 1, False))
            continue
        lab = net.clusters(Pa)
        big, strag, split = streetnet.split_state(lab)
        out.append((smp["t"], big + strag, split))
    return out


def reform(series, commit_t, end_t):
    """(reform seconds after commit or None, success flag): the first sample after commit_t with one cluster, before
    end_t (the next decision point)."""
    for t, nclu, _ in series:
        if t >= commit_t and nclu == 1:
            return round(t - commit_t, 2), bool(t <= end_t)
    return None, False


def stall(samples, t0, t1):
    """Longest stretch (s) with the centroid under STALL_SPEED while not holding, within [t0, t1]."""
    best = cur = 0.0
    prev = None
    for smp in samples:
        if smp["t"] < t0 or smp["t"] > t1:
            continue
        if prev is not None and not smp["hold"]:
            v = float(np.linalg.norm(smp["c"] - prev["c"])) / max(smp["t"] - prev["t"], 1e-9)
            cur = cur + (smp["t"] - prev["t"]) if v < STALL_SPEED else 0.0
            best = max(best, cur)
        prev = smp
    return round(best, 2)


def facade_min(fp, samples, every=2):
    best = math.inf
    for smp in samples[::every]:
        P = smp["P"]
        Pa = P[np.isfinite(P).all(1)]
        if len(Pa):
            best = min(best, float(fp.distance(Pa).min()))
    return round(best, 2)


def losses(sim, n0):
    return dict(lost=int(n0 - sim.alive.sum()), kills=[k[1] for k in sim.kills], contacts=int(sim.contacts))


_BANDS = {}


def bands():
    """internal_3's search band (the swarm's median altitude and pitch while searching) and transit band."""
    if not _BANDS:
        from citysim import capability
        b = capability.band_settings("internal_3")
        _BANDS.update(search=(b["static_h"], b["static_p"]), transit=(b["moving_h"], b["moving_p"]))
    return _BANDS


# ============================================================================= the manoeuvres

def aim_point(net, sid, V):
    """A point AIM_PAST_M down street sid from its corner V: beyond the block, so it is not taken while the formation
    is round the block."""
    F = net.other(sid, V)
    d = net.jpos[F] - net.jpos[V]
    return net.jpos[V] + d / max(float(np.linalg.norm(d)), 1e-9) * AIM_PAST_M, F


def m3_wps(C, tr, st):
    """M3's waypoints: approach, aim through the block at the opposite exit, on to its far junction. None if the block
    has no interior opposite exit. dict(wps, d_block, exit (sid, V, F)); waypoints 1 / 2 / 3 = entrance / commit /
    next decision."""
    net, L = C["net"], C["L"]
    op = net.opposite_exit(tr["block"], tr["entrance"], tr["J"])
    if op is None or not net.streets[op[0]]["interior"]:
        return None
    sid, V = op
    aim, F = aim_point(net, sid, V)
    db = d_block(st, net, tr["block"], L)
    wps = approach_wps(net, tr, st)
    around = 0.5 * geo.perimeter(net.blocks[tr["block"]]["kerb"]) + AIM_PAST_M
    wps.append((float(aim[0]), float(aim[1]), 1.0, timeout_for(around), db))
    wps.append((float(net.jpos[F][0]), float(net.jpos[F][1]), st["exit_stick"],
                timeout_for(float(np.linalg.norm(net.jpos[F] - aim))), st["d_street"]))
    return dict(wps=wps, d_block=db, exit=(sid, V, F))


def m3b_wps(C, tr, st, rng):
    """M3b's waypoints: approach; the surround for t_hop -- a centring hold on the block's centre (m3b_mode "hold",
    at hold_stick) or an aim through the block at the opposite exit at full stick ("aim"); then out by a side exit
    (neither the entrance nor the opposite street, nor at the entrance corner) to its far junction. Also the matched
    street flight (round the block by the streets to the same exit). None when the block has no side exit.
    Waypoints 1 / 2 / 4 = entrance / commit (end of the surround) / next decision."""
    net, L = C["net"], C["L"]
    op = net.opposite_exit(tr["block"], tr["entrance"], tr["J"])
    side = [e for e in candidates(net, tr) if (op is None or e[0] != op[0]) and e[1] != tr["J"]]
    if not side:
        return None
    sid, V = side[int(rng.integers(len(side)))]
    F = net.other(sid, V)
    B = net.blocks[tr["block"]]
    db = d_block(st, net, tr["block"], L)
    wps = approach_wps(net, tr, st)
    if st["m3b_mode"] == "aim":
        if op is None:
            return None
        aim, _ = aim_point(net, *op)
        wps.append((float(aim[0]), float(aim[1]), 1.0, float(st["t_hop"]), db))
    else:
        wps.append((float(B["centre"][0]), float(B["centre"][1]), -float(st["hold_stick"]), float(st["t_hop"]), db))
    wps.append((float(net.jpos[V][0]), float(net.jpos[V][1]), st["exit_stick"],
                timeout_for(0.5 * geo.perimeter(B["kerb"])), st["d_street"]))
    wps.append((float(net.jpos[F][0]), float(net.jpos[F][1]), st["exit_stick"],
                timeout_for(float(np.linalg.norm(net.jpos[F] - net.jpos[V]))), st["d_street"]))
    street = None
    js, _ = net.route(tr["J"], V)
    if js is not None:
        street = approach_wps(net, tr, st)
        prev = net.jpos[tr["J"]]
        for j in js[1:] + [F]:
            q = net.jpos[j]
            street.append((float(q[0]), float(q[1]), 1.0, timeout_for(float(np.linalg.norm(q - prev))),
                           st["d_street"]))
            prev = q
    return dict(wps=wps, street_wps=street, d_block=db, exit=(sid, V, F))


def m4_wps(C, tr, st):
    """M4's waypoints: approach, then a centring hold on the block's centre for HOLD_M4_S at d_block."""
    B = C["net"].blocks[tr["block"]]
    wps = approach_wps(C["net"], tr, st)
    wps.append((float(B["centre"][0]), float(B["centre"][1]), -float(st["hold_stick"]), HOLD_M4_S,
                float(st["d_block"])))
    return dict(wps=wps, d_block=float(st["d_block"]))


def run_m3(C, tr, st, seed, variant):
    net, fp = C["net"], C["fp"]
    built = m3_wps(C, tr, st)
    if built is None:
        return None
    wps, db = built["wps"], built["d_block"]
    m = model(st, variant)
    P0 = start_positions(C, tr, st, seed)
    T = sum(w[3] for w in wps) + 10.0
    sim, samples, done = fly(m, C["rcity"], P0, wps, st["d_street"], T, seed, bands()["search"][1],
                             hold_gain=float(st["hold_gain"]))
    t_J, commit, t_F = done[1], done[2], done[3]
    row = dict(kind="M3", d_block=round(db, 3), reached_exit=bool(math.isfinite(commit)),
               reached_next=bool(math.isfinite(t_F)), t_entrance=_r(t_J), t_commit=_r(commit), t_next=_r(t_F),
               pass_s=_r(commit - t_J), **losses(sim, len(P0)))
    cands = candidates(net, tr)
    row["n_candidates"] = len(cands)
    if math.isfinite(commit):
        row.update(preview_scores(net, fp, samples, cands, commit, bands()))
    ser = cluster_series(net, samples, t0=t_J if math.isfinite(t_J) else 0.0)
    row["split_share"] = round(float(np.mean([s[2] for s in ser])), 3) if ser else math.nan
    if math.isfinite(commit):
        row["reform_s"], row["reform_ok"] = reform(ser, commit, t_F if math.isfinite(t_F) else samples[-1]["t"])
    row["stall_s"] = stall(samples, t_J if math.isfinite(t_J) else 0.0, commit if math.isfinite(commit) else math.inf)
    row["facade_min_m"] = facade_min(fp, samples)
    return row


def run_m3b(C, tr, st, seed, variant, rng):
    net, fp = C["net"], C["fp"]
    built = m3b_wps(C, tr, st, rng)
    if built is None:
        return None
    wps, db = built["wps"], built["d_block"]
    m = model(st, variant)
    P0 = start_positions(C, tr, st, seed)
    T = sum(w[3] for w in wps) + 10.0
    sim, samples, done = fly(m, C["rcity"], P0, wps, st["d_street"], T, seed, bands()["search"][1],
                             hold_gain=float(st["hold_gain"]))
    t_J, commit, t_F = done[1], done[2], done[4]
    row = dict(kind="M3b", mode=st["m3b_mode"], d_block=round(db, 3), t_hop=st["t_hop"],
               reached_next=bool(math.isfinite(t_F)),
               t_entrance=_r(t_J), t_commit=_r(commit), t_next=_r(t_F), hop_s=_r(t_F - t_J), **losses(sim, len(P0)))
    cands = candidates(net, tr)
    row["n_candidates"] = len(cands)
    if math.isfinite(commit):
        row.update(preview_scores(net, fp, samples, cands, commit, bands()))
        # S1's envelopment during the hold (the hold starts when the entrance waypoint is taken)
        row.update(envelop_series(C, samples, tr, t_J, commit, prefix="hold_"))
    ser = cluster_series(net, samples, t0=t_J if math.isfinite(t_J) else 0.0)
    row["split_share"] = round(float(np.mean([s[2] for s in ser])), 3) if ser else math.nan
    if math.isfinite(commit):
        row["reform_s"], row["reform_ok"] = reform(ser, commit, t_F if math.isfinite(t_F) else samples[-1]["t"])
    row["stall_s"] = stall(samples, commit if math.isfinite(commit) else 0.0, math.inf)
    row["facade_min_m"] = facade_min(fp, samples)
    # the time price: the same formation round the block by the streets to the same exit, then on
    w2 = built["street_wps"]
    if w2 is not None:
        sim2, samples2, done2 = fly(m, C["rcity"], P0, w2, st["d_street"], sum(w[3] for w in w2) + 10.0, seed,
                                    bands()["search"][1], hold_gain=float(st["hold_gain"]))
        row["street_s"] = _r(done2[-1] - done2[1])
        row["street_lost"] = int(len(P0) - sim2.alive.sum())
        row["street_contacts"] = int(sim2.contacts)
        row["time_price"] = _r((t_F - t_J) / (done2[-1] - done2[1])) if (math.isfinite(t_F) and
                                                                          math.isfinite(done2[-1])) else None
    return row


def envelop_series(C, samples, tr, t0, t1, prefix=""):
    """S1's envelopment over the hold [t0, t1] round the block's centre: the time to the first enveloped sample and
    the verdict on the mean positions over the hold's last ENVELOP_MEASURE_S."""
    net, fp, L = C["net"], C["fp"], C["L"]
    B = net.blocks[tr["block"]]
    centre = B["centre"]
    own = B["building"]
    others = [i for i in range(fp.n) if i != own]
    c_g = float(fp.subset(others).distance(centre[None])[0]) if others else math.inf
    t_env = None
    tail = []
    n0 = None
    for smp in samples:
        if smp["t"] < t0 or smp["t"] > t1:
            continue
        P = smp["P"]
        alive = np.isfinite(P).all(1)
        if n0 is None:
            n0 = int(alive.sum())
        Pa = P[alive]
        if len(Pa) >= 3:
            e = formation.envelopment(Pa, centre)
            inside = float((np.linalg.norm(Pa - centre, axis=1) <= c_g).mean())
            ok = e["surrounds"] and e["max_gap_deg"] <= formation.MAX_GAP_DEG and inside >= formation.INSIDE_FRAC \
                and int(alive.sum()) == n0
            if ok and t_env is None:
                t_env = round(smp["t"] - t0, 2)
        if smp["t"] >= t1 - ENVELOP_MEASURE_S:
            tail.append(P)
    out = {f"{prefix}t_envelop": t_env, f"{prefix}c_g": round(c_g, 2)}
    if tail:
        Pm = np.nanmean(np.array(tail), axis=0)
        Pm = Pm[np.isfinite(Pm).all(1)]
        if len(Pm) >= 3:
            e = formation.envelopment(Pm, centre)
            inside = float((np.linalg.norm(Pm - centre, axis=1) <= c_g).mean())
            out[f"{prefix}enveloped"] = bool(e["surrounds"] and e["max_gap_deg"] <= formation.MAX_GAP_DEG and
                                             inside >= formation.INSIDE_FRAC)
            out[f"{prefix}max_gap_deg"] = round(e["max_gap_deg"], 1)
            out[f"{prefix}inside_frac"] = round(inside, 3)
    return out


def run_m4(C, tr, st, seed, variant):
    net, fp = C["net"], C["fp"]
    built = m4_wps(C, tr, st)
    wps, db = built["wps"], built["d_block"]
    m = model(st, variant)
    P0 = start_positions(C, tr, st, seed)
    T = sum(w[3] for w in wps) + 5.0
    sim, samples, done = fly(m, C["rcity"], P0, wps, st["d_street"], T, seed, bands()["search"][1],
                             hold_gain=float(st["hold_gain"]))
    t_J = done[1]
    t_end = done[2] if math.isfinite(done[2]) else samples[-1]["t"]
    row = dict(kind="M4", d_block=db, reached_block=bool(math.isfinite(t_J)), **losses(sim, len(P0)))
    if math.isfinite(t_J):
        row.update(envelop_series(C, samples, tr, t_J, t_end))
        row["split_share"] = round(float(np.mean([s[2] for s in cluster_series(net, samples, t0=t_J)])), 3)
        row["facade_min_m"] = facade_min(fp, [s for s in samples if s["t"] >= t_J])
    return row


def run_m1(C, rng, st, seed, variant):
    """~200 m of interior streets (a walk on the junction graph, no U-turns), the formation and then one drone."""
    net, fp = C["net"], C["fp"]
    js = [j for j in range(len(net.jpos)) if net.j_interior[j]]
    for _ in range(200):
        j = js[int(rng.integers(len(js)))]
        route, prev_s, total = [j], None, 0.0
        while total < 200.0:
            opts = [sid for sid in net.j_streets[route[-1]] if sid != prev_s and net.streets[sid]["interior"]]
            if not opts:
                break
            sid = opts[int(rng.integers(len(opts)))]
            route.append(net.other(sid, route[-1]))
            total += net.streets[sid]["length"]
            prev_s = sid
        if total >= 200.0:
            break
    else:
        return None
    wps, prev = [], net.jpos[route[0]]
    for j in route[1:]:
        q = net.jpos[j]
        wps.append((float(q[0]), float(q[1]), 1.0, timeout_for(float(np.linalg.norm(q - prev))), st["d_street"]))
        prev = q
    m = model(st, variant)
    out = dict(kind="M1", route_m=round(total, 1))
    P0 = formation.clear_of_buildings(C["rcity"], template(st["d_street"], seed, st) + net.jpos[route[0]],
                                      net.jpos[route[0]])
    for who, P in (("swarm", P0), ("single", net.jpos[route[0]][None])):
        sim, samples, done = fly(m, C["rcity"], P, wps, st["d_street"], sum(w[3] for w in wps) + 10.0, seed,
                                 bands()["search"][1])
        t_end = done[-1] if math.isfinite(done[-1]) else math.nan
        path = sum(float(np.linalg.norm(b["c"] - a["c"])) for a, b in zip(samples[:-1], samples[1:]))
        out[f"{who}_done"] = bool(math.isfinite(t_end))
        out[f"{who}_speed"] = _r(path / t_end) if math.isfinite(t_end) and t_end > 0 else None
        out[f"{who}_contacts"] = int(sim.contacts)
        out[f"{who}_lost"] = int(len(P) - sim.alive.sum())
        if who == "swarm":
            from citysim import clutter
            walls = [clutter._wall_share(fp, smp["cams"][k], smp["yaws"][k]) > 0.5
                     for smp in samples for k in range(len(smp["yaws"]))]
            out["wall_feed_frac"] = round(float(np.mean(walls)), 3) if walls else math.nan
    return out


def run_m6(C, rng, st, seed):
    """Variant B: the median interior building of the city alone, flown at from 60 m through its centre."""
    net = C["net"]
    bl = net.interior_blocks()
    polys = [blockcity.building_poly(C["L"].boxes[b["building"]]) for b in bl]
    areas = np.array([abs(geo.area(p)) for p in polys])
    poly = polys[int(np.argsort(areas)[len(areas) // 2])]
    c = geo.centroid(poly)
    poly = poly - c
    rcity = rep.PolyCity([poly])
    ang = rng.uniform(0, 2 * math.pi)
    u = np.array([math.sin(ang), math.cos(ang)])
    r_out = float(np.linalg.norm(poly, axis=1).max())
    start = -u * (r_out + 60.0)
    aim = u * (r_out + 60.0)
    m = model(st, "B")
    P0 = template(st["d_street"], seed, st) + start
    wps = [(float(aim[0]), float(aim[1]), 1.0, 75.0, st["d_street"])]
    sim, samples, done = fly(m, rcity, P0, wps, st["d_street"], 80.0, seed, bands()["search"][1])
    passed = 0
    for smp in samples:
        P = smp["P"][np.isfinite(smp["P"]).all(1)]
        if len(P) and (P @ u > r_out + 5.0).all():
            passed = smp["t"]
            break
    return dict(kind="M6", passed=bool(passed), pass_s=_r(passed) if passed else None,
                stall_s=stall(samples, 0.0, math.inf), **losses(sim, len(P0)))


def _r(x, nd=2):
    return round(float(x), nd) if x is not None and math.isfinite(x) else None


# ============================================================================= jobs

def job_id(j):
    st = j["setting"]
    skey = ",".join(f"{k}={st[k]}" for k in sorted(st) if st[k] != DEFAULT_SETTING.get(k))
    return f"{j['kind']}|{j['variant']}|p{j['p']:g}_s{j['s']:g}|seed{j['city_seed']}|{j['trial']}|{skey}"


def run_job(j):
    t0 = time.time()
    base = dict(job_id=job_id(j), kind=j["kind"], variant=j["variant"], p=j["p"], s=j["s"], city_seed=j["city_seed"],
                trial=j["trial"], setting=j["setting"])
    try:
        st = dict(DEFAULT_SETTING, **j["setting"])
        C = city(j["variant"], j["p"], j["s"], j["city_seed"])
        rng = np.random.default_rng(hash_seed(j))
        seed = int(rng.integers(1 << 30))
        if j["kind"] == "M1":
            row = run_m1(C, rng, st, seed, j["variant"])
        elif j["kind"] == "M6":
            row = run_m6(C, rng, st, seed)
        else:
            tr = pick(C["net"], rng)
            if tr is None:
                row = None
            elif j["kind"] == "M3":
                row = run_m3(C, tr, st, seed, j["variant"])
            elif j["kind"] == "M3b":
                row = run_m3b(C, tr, st, seed, j["variant"], rng)
            else:
                row = run_m4(C, tr, st, seed, j["variant"])
            if row is not None:
                row.update(block=tr["block"], entrance=tr["entrance"])
        if row is None:
            return dict(base, skipped=True, wall_s=round(time.time() - t0, 2))
        return dict(base, **row, wall_s=round(time.time() - t0, 2))
    except Exception as e:                                  # one job's failure costs its own row
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 2))


def hash_seed(j):
    s = f"{j['kind']}|{j['variant']}|{j['p']}|{j['s']}|{j['city_seed']}|{j['trial']}"
    return int.from_bytes(__import__("hashlib").sha256(s.encode()).digest()[:8], "little")


def build_jobs(kinds, variants, designs, trials, setting, per_design_setting=None, trial0=0, city_seeds=CITY_SEEDS):
    """trial0 / city_seeds: a confirmation's fresh trials (their own random streams) on fresh cities; the defaults
    are every earlier run's jobs exactly."""
    jobs = []
    for kind in kinds:
        for v in variants:
            if kind == "M6" and v != "B":
                continue
            for p, s in designs:
                st = dict(setting)
                if per_design_setting is not None:
                    st.update(per_design_setting(v, p, s))
                for t in range(trial0, trial0 + trials):
                    jobs.append(dict(kind=kind, variant=v, p=float(p), s=float(s),
                                     city_seed=int(city_seeds[t % len(city_seeds)]), trial=t, setting=st))
    return jobs


def run(name, jobs, procs, label):
    from citysim import clutter, parallel
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = clutter._done_ids(res_path)
    todo = [j for j in jobs if job_id(j) not in done]
    print(f"[split_probe] {label}: {len(todo)} of {len(jobs)} jobs to fly -> {out}", flush=True)
    # every city's layout written here, before the pool: workers generating a missing one would race on its file
    for key in sorted({(j["variant"], j["p"], j["s"], j["city_seed"]) for j in todo}):
        blockcity.ensure(blockcity.Params(*key))
    clutter.keep_awake()
    t0 = time.time()

    def progress(k, n, r):
        clutter._append(res_path, r)
        if k % max(1, n // 20) == 0 or k == n:
            print(f"[split_probe] {label}: {k}/{n} ({time.time() - t0:.0f} s)", flush=True)

    # longest-first is not knowable; shuffle so a stop leaves every design equally covered
    order = np.random.default_rng(7).permutation(len(todo))
    parallel.run_jobs(run_job, [todo[i] for i in order], label, processes=procs, stall_s=600.0, progress=progress)
    return out


def read(name):
    import pandas as pd
    path = OUT_ROOT / name / "results.jsonl"
    by = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["job_id"] in by and "error" not in by[r["job_id"]] and "error" in r:
            continue
        by[r["job_id"]] = r
    return pd.DataFrame(list(by.values()))


def parse_setting(items):
    st = {}
    for it in items or []:
        for part in it.split(","):
            if not part.strip():
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in DEFAULT_SETTING:
                raise SystemExit(f"unknown setting {k!r}; settings: {sorted(DEFAULT_SETTING)}")
            dv = DEFAULT_SETTING[k]
            if k == "extra_sets":
                st[k] = v.strip().replace(";", ",")
            elif isinstance(dv, str):
                st[k] = v.strip()
            elif k == "hollow":
                st[k] = None if v.strip().lower() in ("", "none", "scene") else v.strip().lower() in ("1", "true", "on")
            else:
                st[k] = float(v)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("coupling", "manoeuvres", "steps", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.add_argument("--jobs", type=int, default=20)
        p.add_argument("--trials", type=int, default=30 if name == "coupling" else 50)
        p.add_argument("--variants", nargs="+", default=list(blockcity.VARIANTS))
        p.add_argument("--designs", nargs="+", default=None, help="p:s pairs, e.g. 50:12 75:18 (default all 12)")
        p.add_argument("--setting", action="append", help="KEY=VALUE[,...] over the S4 defaults")
        p.add_argument("--kinds", nargs="+", default=["M1", "M3", "M3b", "M6"])
        p.add_argument("--d-grid", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8])
        p.add_argument("--coupling", help="a coupling run whose d*(p, s) sets d_block per design (d_rule=surround)")
        p.add_argument("--trial0", type=int, default=0, help="number the trials from here (fresh random streams)")
        p.add_argument("--city-seeds", type=int, nargs="+", default=list(CITY_SEEDS), help="the cities flown")
    a = ap.parse_args()
    designs = DESIGNS if not a.designs else [tuple(float(x) for x in d.split(":")) for d in a.designs]
    st = dict(DEFAULT_SETTING, **parse_setting(a.setting))
    fresh = dict(trial0=a.trial0, city_seeds=tuple(a.city_seeds))
    if a.cmd == "coupling":
        jobs = []
        for d in a.d_grid:
            jobs += build_jobs(["M4"], a.variants, designs, a.trials, dict(st, d_block=d), **fresh)
        run(a.name, jobs, a.jobs, "coupling")
    elif a.cmd == "manoeuvres":
        per = None
        if a.coupling:
            dmap = coupling_map(read(a.coupling))
            per = lambda v, p, s: dict(d_rule="fixed", d_block=float(dmap.get((v, p, s), st["d_block"])))  # noqa: E731
        run(a.name, build_jobs(a.kinds, a.variants, designs, a.trials, st, per, **fresh), a.jobs, "manoeuvres")
    elif a.cmd == "steps":
        steps(a.name, st, a.jobs)
    else:
        from citysim import split_tables
        split_tables.tables(a.name)


def coupling_map(df):
    """{(variant, p, s): d*}: split_tables' safe rule (the smallest spread enveloping in >= 80 % of the M4 holds with
    no drone lost and within the contact bar; failing that the best-enveloping spread within the bar)."""
    from citysim import split_tables
    _, cmap = split_tables.coupling(df)
    return {(r.variant, float(r.p), float(r.s)): float(r.d_star) for r in cmap.itertuples()}


def steps(name, st, procs):
    """M5: swarm_replica's spread steps (d_ref 1.0 -> 1.6 -> 0.4 -> 1.0, then cruise / stop / reversal at 0.4) with
    the setting's SwarmManager overrides beside the scene's own."""
    from multiprocessing import Pool
    air = dict(sp.resolve_airframe())
    variants = [("scene", {}), ("S4", overrides(st))]
    jobs = [(label, dict(sp.resolve_swarm(SCENE, overrides=over)), air, seed, N_DRONES)
            for label, over in variants for seed in range(6)]
    with Pool(min(procs, len(jobs))) as pool:
        res = pool.map(rep.run_step, jobs)
    rows = []
    for label, _ in variants:
        rs = [r for r in res if r[0] == label]
        mm = np.array([[*rep.step_metrics(r[2], 40, 70), *rep.step_metrics(r[2], 70, 100), *rep.step_metrics(r[2], 100,
                                                                                                             130)]
                       for r in rs]).mean(0)
        rows.append(dict(variant=label, expand_t50=round(mm[1], 2), expand_t90=round(mm[2], 2),
                         expand_over_pct=round(mm[3], 1), contract_t50=round(mm[5], 2), contract_t90=round(mm[6], 2),
                         contract_over_pct=round(mm[7], 1), reexpand_t90=round(mm[10], 2),
                         kills=int(sum(r[3] for r in rs))))
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps.json").write_text(json.dumps(dict(setting=st, rows=rows), indent=1), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
