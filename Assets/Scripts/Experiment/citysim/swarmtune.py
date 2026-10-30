"""Tuning the Olfati-Saber swarm for BlockShapedPatchCityWorld: how quickly it moves through the city by surrounding
buildings and re-forming beyond them, without fragmenting.

    python -B -m citysim.swarmtune baseline [--single]             # the scene's gains at every d_ref in D_GRID
    python -B -m citysim.swarmtune search --name N --d 2.2 [--n 120]   # random search of the gains at one d_ref
    python -B -m citysim.swarmtune sweep --name N --d 1.8          # one gain at a time round the scene's
    python -B -m citysim.swarmtune grid --name N --d 1.6 2.0 --vary d_shield=1.5,2.5 c_damp=2,4   # combinations
    python -B -m citysim.swarmtune refine --name N                 # local search round the best of a search
    python -B -m citysim.swarmtune confirm --name N --cand FILE    # many flights, the scene's city and held-out ones
    python -B -m citysim.swarmtune steps --cand FILE               # spread steps about each candidate's d_ref
    python -B -m citysim.swarmtune table --name N                  # a run's candidates, scored and ranked
    python -B -m citysim.swarmtune apply --cand FILE --label L    # write a candidate into the scene's SwarmManager
    python -B -m citysim.swarmtune bench-plan --name N --cand FILE # a bench config of the same flights in Unity
    python -B -m citysim.swarmtune bench-gate --name N --runs DIR  # Unity's flights scored, beside replica twins
    -> experiment_results/swarm_tune/<name>/

The city is the scene's own (`BlockShaped__bbB_p40_s18_patch150_j0.15_sight487_seed1`, CITY + seed 1), with the same
design at seeds 2-4 held out. Every flight is the replica's ContactSim on a PolyCity under the default obstacle model,
which is what Unity flies there: each prism avoided as the circle round its axis-aligned bounds, contacts on the true
prism, PhysX's friction, DroneHealthMonitor's isolation loss (S5's model; swarm_replica's `validate` is its gate).
The gains are BlockShapedPatchCityWorld's SwarmManager (swarm_params) plus a candidate's overrides; d_ref is held
constant through every flight (the pilot leaves the dial alone). Since S6 the scene carries its pick (T3: d_ref 2.0,
d_shield 1.6, d_obs 0.16), so a candidate with no overrides is T3; PRE_S6 overrides give the scene as it was.

The pilot is the bench's (SwarmBenchRunner): full stick at the current waypoint, taken when the live drones' centroid
comes within CAPTURE_M or its timeout runs out; after the last, END_HOLD_S at zero stick. Every flight starts with the
hover formation shrunk to fit the start junction's streets and SETTLE_S at zero stick there, as the bench settles;
times, contacts and losses count from the end of the settle. Three manoeuvres, all at full stick:
  cross   a straight line between two interior junctions CROSS_M apart: the formation meets building after building
          head on or glancing and has to flow round each and close up again. The primary flight.
  pass    S4's pass round (split_probe.m3_wps): along a street to a block's entrance corner, aimed through the block at
          a point beyond its opposite corner, on to that street's far junction. One building, met on purpose.
  street  the shortest street route between a cross flight's two ends, junction by junction: the flight a pilot keeps
          to the streets in, which a retune must not make worse.

The measures (score_flight), per flight:
  v_prog      progress speed: the route's length over the time to fly it (m/s); an unfinished flight counts the route
              it covered by the cap
  one_frac    the share of the flight the live drones were one group (streetnet's cluster metric: linked when the
              free-space path between two drones is at most 30 m, round the buildings)
  v_coh       cohesive speed, v_prog x one_frac: route covered per second with the swarm together. The headline.
  episodes    fragment episodes, each a run of samples that are not one group: how many per km, their median, p90 and
              longest durations (the reform time)
  open_end    still not one group END_HOLD_S after the last waypoint, hovering there (reform_end: how long it took)
  met / wrapped / coverage   buildings the pilot aimed at (the route passes within AIM_FRAC of the cylinder radius
              of their centre); the share the formation surrounded (the building's centre inside the drones' convex
              hull at some sample while the centroid was within R + ENC_M of it) and the bearing coverage round it
              (360 minus the widest gap between drones' bearings from its centre, best sample)
  per building    bld_pass_s: the time the centroid spent within R + ENC_M of it (getting past it); bld_frag_s: the
              time not one group from arriving there to BLD_AFTER_S after leaving; bld_reform_s: from leaving to one
              group again (0 if it never split, None if not within BLD_AFTER_S)
  lost, contacts, min_pair    drones lost (drone-drone pairs, isolation; a building is a contact, not a loss) and
              building-contact events, per km; the closest pair
  stall_s     the longest stretch the centroid moved under 1 m/s
  hull_frac   the mean share of live drones on the convex hull (AttitudeAlgorithm's boundary drones: the feeds
              OUTER_CIRCLE shows), which the hollow core exists to keep high

The selection rule (choose; experiment_results/swarm_tune/rule.json), fixed before the sweep's results were read: a
candidate is SAFE when, over its flights
(cross, pass and street together), it loses at most SAFE_LOST_KM drones and touches at most SAFE_CONTACT_KM buildings
per km, ends at most OPEN_MAX of its flights fragmented, and keeps at least HULL_MIN of its drones on the hull (the
hollow core's ring has not collapsed). Among safe candidates, the highest mean v_coh over the cross and pass flights.
With none safe, the highest `score` (score_candidate): v_coh less PEN_LOST m/s per drone lost per km, PEN_CONTACT per
contact per km and PEN_OPEN per share of flights ended fragmented, which is also what the search ranks by. The rule is
applied to the confirmation run (fresh routes, the scene's city and three held-out ones), not to the screens.
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

import numpy as np  # noqa: E402

from citysim import REPO, blockcity, formation, use_bench_modules  # noqa: E402
from citysim import split_probe as S  # noqa: E402
from citysim.display import convex_hull  # noqa: E402

use_bench_modules()
import bench  # noqa: E402
import swarm_params as sp  # noqa: E402
import swarm_replica as rep  # noqa: E402

SCENE = "BlockShapedPatchCityWorld"
CITY = dict(variant="B", p=40.0, s=18.0, seeds="patch", jitter=0.15, sight_tries=487)
TUNE_SEED = 1                                   # the scene's city
HELD_SEEDS = (2, 3, 4)
OUT = REPO / "experiment_results" / "swarm_tune"
N_DRONES = 10
CAPTURE_M = S.CAPTURE_M
SAMPLE_S = 0.5
CROSS_M = (250.0, 420.0)
AIM_FRAC = 0.6
ENC_M = 15.0
STALL_SPEED = 1.0
END_HOLD_S = 10.0
SETTLE_S = 15.0                                 # the bench's settleTime: zero stick at the start before the flight
START_MARGIN_M = 2.0
BLD_AFTER_S = 15.0
D_GRID = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0)
KINDS = ("cross", "pass", "street")
FLIGHTS = dict(cross=24, pass_=16, street=12)  # per candidate in a search
SAFE_LOST_KM, SAFE_CONTACT_KM, OPEN_MAX, HULL_MIN = 0.05, 0.1, 0.05, 0.5
PEN_LOST, PEN_CONTACT, PEN_OPEN = 2.0, 0.5, 4.0   # score: m/s per drone lost per km, per contact per km, per open end
PRE_S6 = dict(d_ref=1.08, d_shield=2.5, d_obs=0.4)  # the scene's SwarmManager before S6 wrote T3 into it

# The search space: (field, low, high, log scale). Every name is a SwarmManager field (swarm_params checks them).
SPACE = (
    ("a", 0.5, 1.6, True), ("b", 1.5, 4.5, True), ("delta", 0.04, 0.2, True), ("r0_coh", 12.0, 40.0, True),
    ("c_obs", 4.0, 24.0, True), ("d_obs", 0.1, 0.9, True), ("c2_beta", 0.6, 3.0, True),
    ("d_shield", 1.2, 4.0, True), ("c_core", 0.5, 3.0, True), ("c2_core", 0.3, 2.0, True),
    ("coreRadiusFraction", 0.3, 0.8, False), ("coreStandoffRatio", 0.3, 0.8, False),
    ("c_damp", 0.5, 4.0, True), ("d_damp", 0.2, 0.6, False), ("c_vm", 0.0, 0.15, False),
)


# ============================================================================= cities, models, formations

_CITY, _MODEL, _TEMPLATE = {}, {}, {}


def params(seed):
    return blockcity.Params(**dict(CITY, seed=seed))


def city(seed):
    P = params(seed)
    if P.ident() not in _CITY:
        C = S.city_from(P)
        _CITY[P.ident()] = C
    return _CITY[P.ident()]


def model(over):
    key = json.dumps(over, sort_keys=True)
    if key not in _MODEL:
        formation.check()
        _MODEL[key] = rep.Model(dict(sp.resolve_swarm(SCENE, overrides=over)), dict(sp.resolve_airframe()))
    return _MODEL[key]


def template(d, seed, over):
    key = (round(d, 3), seed % 5, json.dumps(over, sort_keys=True))
    if key not in _TEMPLATE:
        _TEMPLATE[key] = formation.hover_template(d, N_DRONES, seed % 5, SCENE, overrides=over)[0]
    return _TEMPLATE[key]


def cand_id(over, d):
    return hashlib.sha1(json.dumps(dict(over, d_ref=round(d, 4)), sort_keys=True).encode()).hexdigest()[:10]


# ============================================================================= routes

def _interior_junctions(net):
    return np.flatnonzero(net.j_interior)


def route(C, kind, k, rng):
    """(start (2,), [(x, z, stick, timeout)], route length m, aimed polyline (m, 2)) of route k of a kind."""
    net = C["net"]
    js = _interior_junctions(net)
    if kind == "pass":
        for _ in range(50):
            tr = S.pick(net, rng)
            if tr is None:
                continue
            built = S.m3_wps(C, tr, dict(S.DEFAULT_SETTING, d_street=1.0, d_block=1.0, d_rule="fixed",
                                         exit_stick=1.0))
            if built is None:
                continue
            start = net.jpos[tr["approach"][0]]
            wps = [tuple(float(v) for v in w[:4]) for w in built["wps"]]
            return start, wps, _length(start, wps), np.vstack([start] + [np.array(w[:2]) for w in wps])
        raise RuntimeError("no pass-round trial")
    for _ in range(500):
        a, b = (int(x) for x in rng.choice(js, 2, replace=False))
        dist = float(np.linalg.norm(net.jpos[b] - net.jpos[a]))
        if CROSS_M[0] <= dist <= CROSS_M[1]:
            break
    start = net.jpos[a]
    if kind == "cross":
        wps = [(float(net.jpos[b][0]), float(net.jpos[b][1]), 1.0, S.timeout_for(dist))]
    else:
        jl, _ = net.route(a, b)
        wps, prev = [], start
        for j in jl[1:]:
            q = net.jpos[j]
            wps.append((float(q[0]), float(q[1]), 1.0, S.timeout_for(float(np.linalg.norm(q - prev)))))
            prev = q
    return start, wps, _length(start, wps), np.vstack([start] + [np.array(w[:2]) for w in wps])


def _length(start, wps):
    pts = np.vstack([start] + [np.array(w[:2]) for w in wps])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def route_seed(kind, seed, k):
    return int.from_bytes(hashlib.sha256(f"swarmtune|{kind}|{seed}|{k}".encode()).digest()[:8], "little")


# ============================================================================= the flight

def start_positions(C, d, seed, over, start):
    """The hover formation at the start junction, shrunk about it until no drone is within START_MARGIN_M of a
    footprint (the settle then lets it open out along the streets). Placed at full size, the drones overlapping a
    building were pushed out to its far side and dragged back through the facade by cohesion in the first seconds."""
    P = template(d, seed, over)
    rc = C["rcity"]
    allb = np.arange(rc.n)
    for s in np.arange(1.0, 0.15, -0.1):
        Q = start + s * P
        if not rc.footprint_hit(Q, allb, START_MARGIN_M).any():
            return Q
    return formation.clear_of_buildings(rc, start + 0.2 * P, start)


def fly(m, C, P0, wps, d, seed, settle=SETTLE_S, hook=None):
    """`settle` s at zero stick (the bench's settle), then the bench's waypoint pilot over ContactSim at constant
    d_ref, then END_HOLD_S at zero stick once the last waypoint is taken: (sim, samples, done_t). Times, contacts and
    losses count from the end of the settle (sim.n0: the drones alive then). samples every SAMPLE_S: (t, centroid,
    positions with NaN once lost). hook (the replays' tape and feed latch) only reads: hook.start(sim) once the
    simulation exists, hook.step(sim, t, cmd, wp) before each step of the flight."""
    sim = rep.ContactSim(m, C["rcity"], np.array(P0, float), np.zeros((len(P0), 2)), seed)
    if hook is not None:
        hook.start(sim)
    while sim.t < settle - 1e-9 and sim.alive.any():
        sim.step(np.zeros(2), 0.0, d)
    t_off = sim.t
    sim.contacts = sim.contact_ticks = 0
    sim.kills = []
    sim.n0 = int(sim.alive.sum())
    every = max(1, int(round(SAMPLE_S / m.dt)))
    T = sum(w[3] for w in wps) + 10.0
    done, samples = [math.nan] * len(wps), []
    wp, t0, t_end = 0, 0.0, math.inf
    while sim.t - t_off < min(T, t_end) - 1e-9 and sim.alive.any():
        t = sim.t - t_off
        c = sim.P[sim.alive].mean(0)
        cmd = np.zeros(2)
        if wp < len(wps):
            x, z, stick, timeout = wps[wp]
            dv = np.array([x, z]) - c
            dd = float(np.linalg.norm(dv))
            if dd < CAPTURE_M or (timeout > 0 and t - t0 > timeout):
                done[wp] = t
                wp, t0 = wp + 1, t
                if wp >= len(wps):
                    t_end = t + END_HOLD_S
                else:
                    x, z, stick, timeout = wps[wp]
                    dv = np.array([x, z]) - c
                    dd = float(np.linalg.norm(dv))
            if wp < len(wps):
                cmd = dv / max(dd, 1e-9) * stick
        if sim.tick % every == 0:
            samples.append((t, c.copy(), np.where(sim.alive[:, None], sim.P, np.nan)))
        if hook is not None:
            hook.step(sim, t, cmd, wp)
        sim.step(cmd, 0.0, d)
    if sim.alive.any():
        samples.append((sim.t - t_off, sim.P[sim.alive].mean(0), np.where(sim.alive[:, None], sim.P, np.nan)))
    return sim, samples, done


# ============================================================================= scoring

def _progress(start, wps, done, c_end):
    """Route covered (m): every waypoint taken, plus the last leg's progress towards the next."""
    pts = np.vstack([start] + [np.array(w[:2]) for w in wps])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    k = sum(1 for t in done if math.isfinite(t))
    if k >= len(wps):
        return float(seg.sum())
    a, b = pts[k], pts[k + 1]
    u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
    return float(seg[:k].sum() + np.clip(np.dot(c_end - a, u), 0.0, seg[k]))


def _segdist(p, poly):
    best = math.inf
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def score_flight(C, start, wps, poly, L, sim, samples, done, n0):
    net, rc = C["net"], C["rcity"]
    finished = all(math.isfinite(t) for t in done)
    T = float(done[-1]) if finished else float(samples[-1][0])
    prog = L if finished else _progress(start, wps, done, samples[-1][1])
    v_prog = prog / max(T, 1e-6)
    hold = [smp for smp in samples if smp[0] > T + 1e-9]
    samples = [smp for smp in samples if smp[0] <= T + 1e-9]
    ts = np.array([s[0] for s in samples])
    one, mind, hull = [], math.inf, []
    for t, c, P in samples:
        Pa = P[np.isfinite(P).all(1)]
        if len(Pa) < 2:
            one.append(True)
            continue
        if len(Pa) >= 3:
            hull.append(len(convex_hull(Pa)) / len(Pa))
        D = np.linalg.norm(Pa[None] - Pa[:, None], axis=-1)
        np.fill_diagonal(D, np.inf)
        mind = min(mind, float(D.min()))
        one.append(len(np.unique(net.clusters(Pa))) == 1)
    one = np.array(one)
    eps, k = [], 0
    while k < len(one):
        if one[k]:
            k += 1
            continue
        k1 = k
        while k1 < len(one) and not one[k1]:
            k1 += 1
        eps.append(float(ts[min(k1, len(ts) - 1)] - ts[k]) if k1 < len(one) else float(ts[-1] - ts[k] + SAMPLE_S))
        k = k1
    t_reform = 0.0 if (len(one) and one[-1]) else None
    open_end = t_reform is None
    for t, c, P in hold:
        if t_reform is not None:
            break
        Pa = P[np.isfinite(P).all(1)]
        if len(Pa) < 2 or len(np.unique(net.clusters(Pa))) == 1:
            t_reform, open_end = round(t - T, 2), False
    # buildings aimed at: the route passes within AIM_FRAC of the cylinder radius of the centre
    met, wrapped, cover, b_pass, b_frag, b_reform = 0, 0, [], [], [], []
    for bi in range(rc.n):
        cb, Rb = rc.c[bi], rc.R[bi]
        if _segdist(cb, poly) > AIM_FRAC * Rb:
            continue
        near = [i for i, (t, c, P) in enumerate(samples) if np.linalg.norm(c - cb) <= Rb + ENC_M]
        if not near:
            continue
        met += 1
        best_cov, env = 0.0, False
        for i in near:
            P = samples[i][2]
            Pa = P[np.isfinite(P).all(1)]
            if len(Pa) < 3:
                continue
            e = formation.envelopment(Pa, cb)
            env = env or e["surrounds"]
            best_cov = max(best_cov, 360.0 - e["max_gap_deg"])
        wrapped += int(env)
        cover.append(best_cov)
        # per building: the time the centroid spent near it, the time fragmented from arriving to BLD_AFTER_S after
        # leaving, and the time from leaving to one group again (None: not within BLD_AFTER_S or the flight)
        i0, i1 = near[0], near[-1]
        b_pass.append(float(ts[i1] - ts[i0] + SAMPLE_S))
        i2 = int(np.searchsorted(ts, ts[i1] + BLD_AFTER_S, side="right"))
        b_frag.append(float((~one[i0:i2]).sum() * SAMPLE_S))
        after = [ts[i] - ts[i1] for i in range(i1, i2) if one[i]]
        b_reform.append(round(float(after[0]), 2) if after else None)
    # stall: the longest stretch under STALL_SPEED
    stall = cur = 0.0
    for (t0, c0, _), (t1, c1, _) in zip(samples[:-1], samples[1:]):
        v = float(np.linalg.norm(c1 - c0)) / max(t1 - t0, 1e-9)
        cur = cur + (t1 - t0) if v < STALL_SPEED else 0.0
        stall = max(stall, cur)
    kills = [k_[1] for k_ in sim.kills]
    eps_a = np.array(eps) if eps else np.zeros(0)
    one_frac = float(one.mean()) if len(one) else 1.0
    return dict(finished=finished, T=round(T, 2), L=round(L, 1), prog=round(prog, 1), v_prog=round(v_prog, 3),
                one_frac=round(one_frac, 3), v_coh=round(v_prog * one_frac, 3), n_eps=len(eps),
                ep_med=round(float(np.median(eps_a)), 2) if len(eps_a) else 0.0,
                ep_p90=round(float(np.percentile(eps_a, 90)), 2) if len(eps_a) else 0.0,
                ep_max=round(float(eps_a.max()), 2) if len(eps_a) else 0.0, open_end=open_end,
                reform_end=t_reform,
                met=met, wrapped=wrapped, coverage=round(float(np.mean(cover)), 1) if cover else None,
                lost=int(n0 - sim.alive.sum()), lost_drone=kills.count("drone"), lost_iso=kills.count("isolated"),
                contacts=int(sim.contacts), contact_ticks=int(sim.contact_ticks),
                min_pair=round(mind, 2) if math.isfinite(mind) else None, stall_s=round(stall, 2),
                hull_frac=round(float(np.mean(hull)), 3) if hull else None,
                bld_pass_s=[round(x, 2) for x in b_pass], bld_frag_s=[round(x, 2) for x in b_frag],
                bld_reform_s=b_reform)


# ============================================================================= jobs

def job_id(j):
    return f"{j['cand']}|{j['kind']}|seed{j['seed']}|{j['k']}"


def run_job(j):
    """One flight of one candidate: dict(job fields, measures) or dict(error)."""
    t0 = time.time()
    base = dict(job_id=job_id(j), cand=j["cand"], kind=j["kind"], seed=j["seed"], k=j["k"], d=j["d"])
    try:
        C = city(j["seed"])
        rng = np.random.default_rng(route_seed(j["kind"], j["seed"], j["k"]))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly = route(C, j["kind"], j["k"], rng)
        over = dict(j["over"])
        m = model(over)
        P0 = start[None].copy() if j.get("single") else start_positions(C, j["d"], fseed, over, start)
        sim, samples, done = fly(m, C, P0, wps, j["d"], fseed)
        return dict(base, **score_flight(C, start, wps, poly, L, sim, samples, done, sim.n0),
                    wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


def jobs_for(cand, over, d, seeds=(TUNE_SEED,), flights=FLIGHTS, k0=0):
    out = []
    for seed in seeds:
        for kind, n in flights.items():
            kind = kind.rstrip("_")
            out += [dict(cand=cand, over=over, d=float(d), kind=kind, seed=seed, k=k0 + k) for k in range(n)]
    return out


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _done_ids(path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out.add(r["job_id"])
    return out


def run(name, cands, jobs, procs, label):
    """Append every job not yet in <name>/results.jsonl (resumable); cands {id: dict(over, d, ...)} to cands.json."""
    from citysim import parallel
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cpath = out / "cands.json"
    known = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    known.update(cands)
    cpath.write_text(json.dumps(known, indent=1), encoding="utf-8")
    res = out / "results.jsonl"
    done = _done_ids(res)
    todo = [j for j in jobs if job_id(j) not in done]
    t0 = time.time()
    if todo:
        parallel.run_jobs(run_job, todo, label, processes=procs, stall_s=600.0,
                          progress=lambda i, n, r: _append(res, r))
    _log(name, f"{label}: {len(todo)} flights in {(time.time() - t0) / 60:.1f} min")


def _log(name, msg):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    line = f"[swarmtune {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================================= tables and the rule

def read(name):
    import pandas as pd
    out = OUT / name
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    errs = [r for r in rows if "error" in r]
    df = pd.DataFrame([r for r in rows if "error" not in r]).drop_duplicates("job_id", keep="last")
    cands = json.loads((out / "cands.json").read_text(encoding="utf-8"))
    return df, cands, errs


def score_candidate(g):
    """The measures of one candidate's flights (a DataFrame), and the search's score."""
    km = g.L.sum() / 1000.0
    fwd = g[g.kind.isin(["cross", "pass"])]
    r = dict(n=len(g), km=round(km, 2), lost_km=g.lost.sum() / km, contact_km=g.contacts.sum() / km,
             open=g.open_end.mean(), v_coh=fwd.v_coh.mean(), v_prog=fwd.v_prog.mean(), one=fwd.one_frac.mean(),
             finished=g.finished.mean(), hull=g.hull_frac.mean() if "hull_frac" in g else math.nan)
    for kind in KINDS:
        k = g[g.kind == kind]
        if not len(k):
            continue
        kkm = k.L.sum() / 1000.0
        mt = k.met.sum()
        r.update({f"{kind}_v_coh": k.v_coh.mean(), f"{kind}_v_prog": k.v_prog.mean(), f"{kind}_one": k.one_frac.mean(),
                  f"{kind}_lost_km": k.lost.sum() / kkm, f"{kind}_contact_km": k.contacts.sum() / kkm,
                  f"{kind}_eps_km": k.n_eps.sum() / kkm, f"{kind}_ep_p90": k.ep_p90.mean(),
                  f"{kind}_ep_max": k.ep_max.max(), f"{kind}_open": k.open_end.mean(),
                  f"{kind}_wrapped": k.wrapped.sum() / mt if mt else math.nan,
                  f"{kind}_coverage": k.coverage.mean(), f"{kind}_stall": k.stall_s.mean(),
                  f"{kind}_finished": k.finished.mean(), f"{kind}_min_pair": k.min_pair.min()})
        if "bld_pass_s" in k and mt:
            bp = [x for v in k.bld_pass_s.dropna() for x in v]
            bf = [x for v in k.bld_frag_s.dropna() for x in v]
            br = [x for v in k.bld_reform_s.dropna() for x in v]
            r.update({f"{kind}_bld_pass": float(np.mean(bp)) if bp else math.nan,
                      f"{kind}_bld_frag": float(np.mean(bf)) if bf else math.nan,
                      f"{kind}_bld_reformed": float(np.mean([x is not None for x in br])) if br else math.nan,
                      f"{kind}_bld_reform": float(np.median([x for x in br if x is not None]))
                      if any(x is not None for x in br) else math.nan})
    r["safe"] = bool(r["lost_km"] <= SAFE_LOST_KM and r["contact_km"] <= SAFE_CONTACT_KM and r["open"] <= OPEN_MAX
                     and not (r["hull"] < HULL_MIN))
    r["score"] = r["v_coh"] - PEN_LOST * r["lost_km"] - PEN_CONTACT * r["contact_km"] - PEN_OPEN * r["open"]
    return r


def table(name):
    import pandas as pd
    df, cands, _ = read(name)
    rows = []
    for cid, g in df.groupby("cand"):
        c = cands.get(cid, {})
        rows.append(dict(cand=cid, label=c.get("label", ""), d=float(g.d.iloc[0]), **score_candidate(g),
                         **{f"o:{k}": v for k, v in c.get("over", {}).items()}))
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def choose(tab):
    """The rule (see the docstring): (row, why). The one-drone reference rows are not candidates."""
    tab = tab[~tab.cand.astype(str).str.startswith("single")]
    safe = tab[tab.safe]
    if len(safe):
        return safe.sort_values("v_coh", ascending=False).iloc[0], "safe: the highest cohesive speed"
    return tab.sort_values("score", ascending=False).iloc[0], "none safe: the highest penalised score"


COLS = ["cand", "label", "d", "n", "score", "safe", "v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "hull",
        "cross_v_coh", "cross_wrapped", "cross_coverage", "cross_ep_p90", "pass_v_coh", "pass_wrapped",
        "street_v_coh", "street_lost_km", "street_contact_km"]


def show(tab, n=25):
    import pandas as pd
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    cols = [c for c in COLS if c in tab.columns]
    print(tab[cols].head(n).round(3).to_string(index=False))


# ============================================================================= commands

def cmd_baseline(a):
    cands, jobs = {}, []
    for d in a.d or D_GRID:
        cid = cand_id({}, d)
        cands[cid] = dict(over={}, d=d, label=f"scene gains @ {d:g}")
        jobs += jobs_for(cid, {}, d)
    if a.single:                                     # one drone on the same routes: the reference speed
        cands["single"] = dict(over={}, d=1.0, label="one drone")
        jobs += [dict(j, single=True) for j in jobs_for("single", {}, 1.0)]
    run(a.name, cands, jobs, a.jobs, "baseline")
    show(table(a.name))


def sample_space(rng, n, base=None, scale=None):
    """n override dicts: uniform (log-uniform where marked) over SPACE, or Gaussian steps of `scale` (a fraction of
    each range, in the log where marked) round `base`, clipped to the range."""
    out = []
    for _ in range(n):
        o = {}
        for f, lo, hi, lg in SPACE:
            tl, th = (math.log(lo), math.log(hi)) if lg else (lo, hi)
            if base is None:
                v = rng.uniform(tl, th)
            else:
                b0 = base.get(f, scene_value(f))
                b0 = math.log(b0) if lg else b0
                v = float(np.clip(b0 + rng.normal(0, scale * (th - tl)), tl, th))
            o[f] = round(float(math.exp(v) if lg else v), 4)
        out.append(o)
    return out


_SCENE = {}


def scene_value(f):
    if not _SCENE:
        _SCENE.update(dict(sp.resolve_swarm(SCENE)))
    return float(_SCENE[f])


def flights_arg(a):
    return dict(cross=a.cross, pass_=a.pass_, street=a.street) if a.cross is not None else FLIGHTS


def cmd_search(a):
    """Random candidates: at each --d, or with --d-range lo hi a log-uniform d_ref per candidate."""
    rng = np.random.default_rng(a.seed)
    cands, jobs = {}, []
    fl = flights_arg(a)
    if a.d_range:
        lo, hi = a.d_range
        draws = [(round(float(math.exp(rng.uniform(math.log(lo), math.log(hi)))), 2), o)
                 for o in sample_space(rng, a.n)]
    else:
        draws = [(d, o) for d in a.d for o in [{}] + sample_space(rng, a.n)]
    for d, o in draws:
        cid = cand_id(o, d)
        cands[cid] = dict(over=o, d=d, label="scene gains" if not o else "random")
        jobs += jobs_for(cid, o, d, flights=fl)
    run(a.name, cands, jobs, a.jobs, f"search ({len(draws)} candidates)")
    show(table(a.name))


SWEEP_FACTORS = (0.5, 0.7, 1.4, 2.0)


def cmd_sweep(a):
    """One gain at a time round the scene's: every SPACE field at the scene value x SWEEP_FACTORS (clipped to its
    range), plus the hollow core off, at each --d. The scene's own gains at each d are the reference."""
    cands, jobs = {}, []
    fl = flights_arg(a)
    for d in a.d:
        variants = [("scene gains", {})]
        for f, lo, hi, lg in SPACE:
            v0 = scene_value(f)
            for k in SWEEP_FACTORS:
                v = v0 * k if v0 > 0 else (lo + (hi - lo) * (k / 2.0 - 0.25))
                v = round(float(np.clip(v, lo * 0.5, hi * 1.5)), 4)
                variants.append((f"{f} x{k:g}" if v0 > 0 else f"{f} = {v:g}", {f: v}))
        variants.append(("core off", {"hollowSwarmCore": False}))
        for label, o in variants:
            cid = cand_id(o, d)
            cands[cid] = dict(over=o, d=d, label=label)
            jobs += jobs_for(cid, o, d, flights=fl)
    run(a.name, cands, jobs, a.jobs, f"sweep d {a.d}")
    show(table(a.name), 200)


def cmd_grid(a):
    """Every combination of the --vary axes (field=v1,v2,...) on top of --base (field=v,...), at each --d."""
    import itertools
    base = {k: float(v) for k, v in sp.parse_assignments([a.base]).items()} if a.base else {}
    axes = []
    for spec in a.vary or []:
        f, _, vals = spec.partition("=")
        axes.append((f.strip(), [float(v) for v in vals.split(",")]))
    cands, jobs = {}, []
    fl = flights_arg(a)
    for d in a.d:
        for combo in itertools.product(*[v for _, v in axes]):
            o = dict(base)
            o.update({f: v for (f, _), v in zip(axes, combo)})
            label = ", ".join(f"{f} {v:g}" for (f, _), v in zip(axes, combo)) or "base"
            cid = cand_id(o, d)
            cands[cid] = dict(over=o, d=d, label=label)
            jobs += jobs_for(cid, o, d, flights=fl)
    run(a.name, cands, jobs, a.jobs, f"grid ({len(cands)} candidates)")
    show(table(a.name), 300)


def cmd_refine(a):
    """Rounds of local search: the top `keep` candidates of the run (complete ones only), each perturbed `per` times
    at step `scale` (a fraction of each SPACE range), and d_ref too with --vary-d."""
    rng = np.random.default_rng(a.seed)
    fl = flights_arg(a)
    for rnd in range(a.rounds):
        tab = table(a.name)
        tab = tab[tab.n >= tab.n.max() * 0.9]
        _, cands_all, _ = read(a.name)
        top = tab.head(a.keep)
        cands, jobs = {}, []
        for _, row in top.iterrows():
            base = cands_all[row.cand]["over"]
            for o in sample_space(rng, a.per, base=base, scale=a.scale):
                d = float(row.d)
                if a.vary_d:
                    d = round(float(np.clip(d * math.exp(rng.normal(0, 0.08)), 1.0, 3.2)), 2)
                cid = cand_id(o, d)
                cands[cid] = dict(over=o, d=d, label=f"refine r{rnd} of {row.cand}")
                jobs += jobs_for(cid, o, d, flights=fl)
        run(a.name, cands, jobs, a.jobs, f"refine round {rnd}")
        a.scale *= 0.7
    show(table(a.name))


def cmd_confirm(a):
    """Fresh flights (route indices from 1000) of the named candidates in the scene's city and the held-out ones.
    With --single, one drone flies the same routes under each candidate's gains too (the single-drone condition
    shares the scene's SwarmManager, so its obstacle gains are retuned along with the swarm's)."""
    spec = json.loads(open(a.cand, encoding="utf-8").read())
    cands, jobs = {}, []
    flights = dict(cross=a.cross, pass_=a.pass_, street=a.street)
    seeds = (TUNE_SEED,) + HELD_SEEDS
    for c in spec:
        cid = cand_id(c["over"], c["d"])
        cands[cid] = dict(over=c["over"], d=c["d"], label=c.get("label", ""))
        jobs += jobs_for(cid, c["over"], c["d"], seeds=seeds, flights=flights, k0=1000)
        if a.single:
            sid = "single-" + cand_id(c["over"], 0.0)
            if sid not in cands:
                cands[sid] = dict(over=c["over"], d=c["d"], label=f"one drone, {c.get('label', '')} gains")
                jobs += [dict(j, single=True) for j in jobs_for(sid, c["over"], c["d"], seeds=seeds, flights=flights,
                                                               k0=1000)]
    run(a.name, cands, jobs, a.jobs, "confirm")
    tab = table(a.name)
    show(tab)
    df, _, _ = read(a.name)
    df["held"] = df.seed != TUNE_SEED
    import pandas as pd
    rows = []
    for (cid, held), g in df.groupby(["cand", "held"]):
        rows.append(dict(cand=cid, label=cands[cid]["label"], held=held, **score_candidate(g)))
    show(pd.DataFrame(rows).sort_values(["label", "held"]), 60)
    row, why = choose(tab)
    _log(a.name, f"rule: {row.cand} ({row.label}) at d_ref {row.d:g}: {why}; v_coh {row.v_coh:.3f}, lost/km "
                 f"{row.lost_km:.3f}, contacts/km {row.contact_km:.3f}, open {row.open:.3f}, hull {row.hull:.3f}")


def cmd_table(a):
    """A run's candidates, scored and ranked, to the console and to <run>/table.csv (the record kept in git; the raw
    results.jsonl is not)."""
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab, a.n)
    row, why = choose(tab)
    print(f"\nrule: {row.cand} ({row.label}) at d_ref {row.d:g}: {why}")


# ============================================================================= spread steps about a candidate's d_ref


SWARM_MANAGER_GUID = "0ad6d96a942db456999d589ce17f8408"      # SwarmManager.cs.meta


def cmd_apply(a):
    """Write one candidate (d_ref and its overrides) into the scene's SwarmManager, editing only those fields of that
    one component in the scene file. The editor must reload the scene (it asks) for the values to take effect."""
    import re
    spec = {c["label"]: c for c in json.loads(open(a.cand, encoding="utf-8").read())}
    c = spec[a.label]
    vals = dict(c["over"], d_ref=c["d"])
    fields = sp.swarm_manager_fields()
    path = sp.scene_path(SCENE)
    text = path.read_text(encoding="utf-8")
    docs = text.split("\n--- ")
    hits = [i for i, dd in enumerate(docs) if f"guid: {SWARM_MANAGER_GUID}" in dd]
    if len(hits) != 1:
        raise SystemExit(f"[swarmtune] {path.name}: {len(hits)} SwarmManager components, expected 1")
    doc = docs[hits[0]]
    for k, v in vals.items():
        if k not in fields:
            raise SystemExit(f"[swarmtune] SwarmManager has no field {k!r}")
        if isinstance(v, bool) or k == "hollowSwarmCore" or k == "coreRelativeVelocity":
            s = str(int(bool(v)))
        else:
            s = f"{float(v):g}"
        pat = r"(\n  " + re.escape(k) + r": )([^\n]*)"
        old = re.search(pat, doc)
        new, n = re.subn(pat, lambda mm: mm.group(1) + s, doc)
        if n != 1:
            raise SystemExit(f"[swarmtune] {k}: {n} matches in the SwarmManager block, expected 1")
        print(f"  {k}: {old.group(2)} -> {s}")
        doc = new
    docs[hits[0]] = doc
    if not a.dry_run:
        path.write_text("\n--- ".join(docs), encoding="utf-8", newline="\n")
    print(f"[swarmtune] {'would write' if a.dry_run else 'wrote'} {a.label} into {path}")


def run_steps(job):
    label, over, d, seed = job
    m = model(over)
    rng = np.random.default_rng(1000 + seed)
    sim = rep.Sim(m, rep.City([]), rng.normal(0, 6.0, (N_DRONES, 2)), np.zeros((N_DRONES, 2)), seed)
    sched = [(0.0, d), (40.0, d * 1.3), (70.0, d), (100.0, d * 0.75)]
    rec = []
    while sim.t < 130.0:
        dd = [v for t, v in sched if sim.t >= t][-1]
        sim.step(np.zeros(2), 0.0, dd)
        if sim.tick % 5 == 0:
            Pa = sim.P[sim.alive]
            rec.append((sim.t, float(np.linalg.norm(Pa - Pa.mean(0), axis=1).mean())))
    rec = np.array(rec)
    out = dict(label=label, seed=seed, kills=len(sim.kills))
    for name, (ts, te) in dict(expand=(40, 70), contract=(70, 100), shrink=(100, 130)).items():
        r1, t50, t90, over_pc = rep.step_metrics(rec, ts, te)
        out.update({f"{name}_t50": t50, f"{name}_t90": t90, f"{name}_over": over_pc})
    return out


def cmd_steps(a):
    import pandas as pd
    from multiprocessing import Pool
    spec = json.loads(open(a.cand, encoding="utf-8").read())
    jobs = [(c.get("label", ""), c["over"], c["d"], s) for c in spec for s in range(a.seeds)]
    with Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(run_steps, jobs)
    df = pd.DataFrame(res).groupby("label").mean(numeric_only=True).drop(columns="seed").round(2)
    pd.set_option("display.width", 250)
    print("spread steps at hover: d -> 1.3 d at 40 s (expand), -> d at 70 s (contract), -> 0.75 d at 100 s")
    print(df.to_string())
    if a.out:
        df.to_csv(a.out)


# ============================================================================= the Unity bench: plan and gate

BENCH_ALTITUDE = 12.4                           # S4's bench altitude (the search band), under the 45 m ceiling
BENCH_SETTLE_S = 15.0
ZERO_GAIN = 1e-6                                # a hold at this gain is zero stick (SwarmBenchRunner reads 0 as 0.25)


def bench_scenarios(flights, k0=0):
    """[scenario dict] of the scene's city: the replica's own routes (route(), the same streams), each ending in an
    END_HOLD_S hold at zero stick on its last waypoint, as fly() ends."""
    C = city(TUNE_SEED)
    out = []
    for kind, n in flights.items():
        kind = kind.rstrip("_")
        for k in range(k0, k0 + n):
            rng = np.random.default_rng(route_seed(kind, TUNE_SEED, k))
            rng.integers(1 << 30)
            start, wps, L, poly = route(C, kind, k, rng)
            wl = [list(w) for w in wps] + [[wps[-1][0], wps[-1][1], -1.0, END_HOLD_S]]
            out.append(dict(kind=f"{kind}_{k}", sx=round(float(start[0]), 3), sz=round(float(start[1]), 3),
                            T=round(sum(w[3] for w in wps) + 10.0 + END_HOLD_S, 1),
                            wps=[round(float(v), 3) for w in wl for v in w], holdGain=ZERO_GAIN, record=True,
                            endOnDone=True))
    return out


def cmd_bench_plan(a):
    """A bench config for BlockShapedPatchCityWorld as baked: the candidates in --cand as parameter sets (d_ref one
    of the overrides), the flights as scenarios."""
    from collections import OrderedDict
    from citysim import BENCH_DIR, camera
    spec = json.loads(open(a.cand, encoding="utf-8").read())
    fields = sp.swarm_manager_fields()
    sets = []
    for c in spec:
        assigns = ",".join(f"{k}={v}" for k, v in dict(c["over"], d_ref=c["d"]).items())
        sets.append(bench.parse_set(f"{c['label']}:{assigns}", fields, True))
    scen = bench_scenarios(dict(cross=a.cross, pass_=a.pass_, street=a.street), k0=a.k0)
    cfg = OrderedDict(scene=SCENE, timeScale=20.0, settleTime=BENCH_SETTLE_S, altitude=BENCH_ALTITUDE,
                      capture=CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                      goalReplaySession="", layout="", droneCount=a.drones, walkers=False, paramSets=sets,
                      scenarios=scen)
    out = OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    cfile = out / "config.json"
    cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "cands.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"[swarmtune] {len(sets)} sets x {len(scen)} flights -> {cfile}\n"
          f"  & '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag swarmtune_{a.name} -TimeoutMin {a.timeout}")


def bench_job(job):
    """Score one bench flight on Unity's own trajectory, and its replica twin flown from Unity's recorded start."""
    run_dir, label, idx, sc, over, d, flight = job
    from pathlib import Path
    from types import SimpleNamespace
    from citysim import tierc
    try:
        kind, k = sc["kind"].rsplit("_", 1)
        k = int(k)
        C = city(TUNE_SEED)
        rng = np.random.default_rng(route_seed(kind, TUNE_SEED, k))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly = route(C, kind, k, rng)
        tr = tierc.read_traj(Path(run_dir) / f"traj_{label}_{idx}.csv")
        samples = []
        for r in range(0, len(tr["t"]), max(1, int(round(SAMPLE_S / 0.1)))):
            al = tr["alive"][r]
            P = np.where(al[:, None], np.c_[tr["X"][r], tr["Z"][r]], np.nan)
            samples.append((float(tr["t"][r]), np.nanmean(P, axis=0) if al.any() else np.zeros(2), P))
        done = [math.nan] * len(wps)
        for r in range(len(tr["t"])):
            for i in range(min(int(tr["wp"][r]), len(wps))):
                if not math.isfinite(done[i]):
                    done[i] = float(tr["t"][r])
        n0 = int(tr["alive"][0].sum())
        deaths = flight.get("deaths") or {}
        ghost = SimpleNamespace(alive=tr["alive"][-1], kills=[(0.0, kk) for kk, v in deaths.items() for _ in range(v)],
                                contacts=int(flight.get("contacts") or 0), contact_ticks=0)
        b = score_flight(C, start, wps, poly, L, ghost, samples, done, n0)
        b["deaths"] = deaths
        b["other_contacts"] = int(flight.get("other_contacts") or 0)
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        sim, rs, rdone = fly(model(over), C, P0, wps, d, fseed, settle=0.0)       # Unity's start is post-settle
        r = score_flight(C, start, wps, poly, L, sim, rs, rdone, sim.n0)
        return dict(set=label, idx=idx, kind=kind, k=k, d=d, **{f"u_{x}": v for x, v in b.items()},
                    **{f"r_{x}": v for x, v in r.items()})
    except Exception as e:
        return dict(set=label, idx=idx, error=repr(e), trace=traceback.format_exc()[-1500:])


def cmd_bench_gate(a):
    import pandas as pd
    from pathlib import Path
    from citysim import parallel
    jobs = []
    for run in a.runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        _, _, params, flights, _, _ = bench.read_results(run)
        fl = {(f["set"], f["scenario"]): f for f in flights}
        for ps in cfg["paramSets"]:
            # The twin flies every value the bench recorded as in force for this set, not the live scene's plus the
            # set's overrides: the scene may have changed since the bench copied it (S6 wrote its pick into it).
            eff = (params.get(ps["label"]) or {}).get("effective")
            over = ({k: float(f"{v:.7g}") if isinstance(v, float) else v for k, v in eff.items()} if eff
                    else dict(zip(ps["names"], ps["values"])))
            d = float(over.pop("d_ref"))
            for idx, sc in enumerate(cfg["scenarios"]):
                if (ps["label"], idx) in fl and (run / f"traj_{ps['label']}_{idx}.csv").exists():
                    jobs.append((str(run), ps["label"], idx, sc, over, d, fl[(ps["label"], idx)]))
    res = parallel.run_jobs(bench_job, jobs, "bench gate", processes=a.jobs)
    errs = [r for r in res if "error" in r]
    for e in errs[:3]:
        print(e["error"], e.get("trace", "")[-600:])
    df = pd.DataFrame([r for r in res if "error" not in r])
    out = OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "gate.jsonl", orient="records", lines=True)
    rows = []
    for label, g in df.groupby("set", sort=False):
        for src in ("u", "r"):
            gg = g[[c for c in g.columns if c.startswith(f"{src}_")]].rename(columns=lambda c: c[2:])
            gg["kind"] = g["kind"].values
            rows.append(dict(set=label, src="unity" if src == "u" else "replica", **score_candidate(gg)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    cols = ["set", "src", "n", "km", "v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "hull",
            "cross_wrapped", "cross_coverage", "cross_ep_p90", "street_v_coh", "street_lost_km", "street_contact_km"]
    print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    if errs:
        print(f"[swarmtune] {len(errs)} flights failed to score")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("baseline")
    p.add_argument("--name", default="baseline")
    p.add_argument("--d", type=float, nargs="*")
    p.add_argument("--single", action="store_true", help="also one drone on the same routes")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("search")
    p.add_argument("--name", required=True)
    p.add_argument("--d", type=float, nargs="+")
    p.add_argument("--d-range", type=float, nargs=2)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    for k in ("cross", "pass", "street"):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int)
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("sweep")
    p.add_argument("--name", required=True)
    p.add_argument("--d", type=float, nargs="+", required=True)
    for k in ("cross", "pass", "street"):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int)
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("grid")
    p.add_argument("--name", required=True)
    p.add_argument("--d", type=float, nargs="+", required=True)
    p.add_argument("--base", default="", help="field=value,... under every combination")
    p.add_argument("--vary", nargs="*", help="field=v1,v2,... (one axis each)")
    for k in ("cross", "pass", "street"):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int)
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("refine")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--per", type=int, default=8)
    p.add_argument("--scale", type=float, default=0.12)
    p.add_argument("--vary-d", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    for k in ("cross", "pass", "street"):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int)
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("confirm")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True, help="JSON list of dict(label, d, over)")
    p.add_argument("--cross", type=int, default=40)
    p.add_argument("--pass", dest="pass_", type=int, default=30)
    p.add_argument("--street", type=int, default=20)
    p.add_argument("--single", action="store_true", help="also one drone under each candidate's gains")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("steps")
    p.add_argument("--cand", required=True)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--out")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("table")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=30)
    p = sub.add_parser("apply")
    p.add_argument("--cand", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("bench-plan")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--cross", type=int, default=12)
    p.add_argument("--pass", dest="pass_", type=int, default=8)
    p.add_argument("--street", type=int, default=6)
    p.add_argument("--k0", type=int, default=2000)
    p.add_argument("--drones", type=int, default=N_DRONES, help="1 = the single-drone condition")
    p.add_argument("--timeout", type=int, default=90)
    p = sub.add_parser("bench-gate")
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    dict(baseline=cmd_baseline, search=cmd_search, refine=cmd_refine, confirm=cmd_confirm, steps=cmd_steps,
         table=cmd_table, sweep=cmd_sweep, grid=cmd_grid, apply=cmd_apply, **{"bench-plan": cmd_bench_plan, "bench-gate": cmd_bench_gate})[a.cmd](a)


if __name__ == "__main__":
    main()
