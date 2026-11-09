"""S11 Wrap Hold (the wrap study): the adaptive spread kept at full spread while a building lies inside the formation
(SwarmManager.spreadWrapRule Hold / HoldUntilLeaving) against the rule as BlockShapedPatchCityWorld flies it (E15_35a,
Off), on S8's harness (citysim.swarmshape, swarmflow, swarmtune), with one flight added -- `search`, a goal search round
one building as ERKS flew them in internal_4 -- and the wrap measures logged on every flight.

    python -B -m citysim.swarmwrap human                     # ERKS's three swarm flights, every rule replayed open loop
    python -B -m citysim.swarmwrap screen --name N --cand FILE [--cross 40 --pass 30 --street 40 --face 24 --search 60]
    python -B -m citysim.swarmwrap confirm --name N --cand FILE [...]   # fresh routes, the scene's city + held out
    python -B -m citysim.swarmwrap table --name N
    python -B -m citysim.swarmwrap rule [--force]            # rule.json, written before the confirmation
    python -B -m citysim.swarmwrap choose --name confirm     # verdicts.json
    -> experiment_results/swarm_wrap/<name>/

The question is the user's (2026-11-06, from internal_4's maps): the swarm widened as it first surrounded a goal's
building and then narrowed while still wrapped round it. Replayed on ERKS's flights, the rule as flown held a narrowed
spread for 39% of the time a building lay inside the formation: it keeps the full spread only while the stick's line
ahead (or, with no stick, the centroid itself) passes near a building's centre, so a centroid drifted off the middle or
a pilot panning round it narrows the formation. Hold keeps the full spread while any building's centre lies inside the
alive drones' hull; HoldUntilLeaving releases it while the pilot steers away from that building. The cost the replay
predicts is the street entered next: the formation leaves a building wide and narrows ~3 s later.

Measures logged on every flight (WrapLog, every TRACK_EVERY ticks = 0.1 s), with "wrapped" the rule's own test (a
building's centre -- the ellipse's, else the circle the S7 rules aim by -- inside the alive drones' hull), so it means
the same in every row:
  wrap_s        time wrapped round a building
  wrap_full     the share of it at full spread (scale >= FULL): the question
  wrap_sigma    the mean scale while wrapped; wrap_ring the formation's mean radius then
  exit_s        time in the EXIT_S after the last building left the formation (none wrapped since); exit_sigma
  loss_*, contact_*   loss events (instants at which drones were lost: a colliding pair is one) and contacts, placed
                wrapped / exit / other, with each phase's exposure: the cost, where it is predicted to fall
plus S8's measures (swarmtune.score_flight; swarmshape.face_measures for `face`) and, for `search`, the share of its
hold window wrapped and at full spread, and the speed it leaves on. Route indices: the screen from K_SCREEN, the
confirmation from K_CONFIRM, the bench from K_BENCH (S6-S10 used 0-18999).
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
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from citysim import REPO  # noqa: E402
from citysim import split_probe as S  # noqa: E402
from citysim import swarmflow as F  # noqa: E402
from citysim import swarmshape as SH  # noqa: E402
from citysim import swarmspread as SP  # noqa: E402
from citysim import swarmtune as T  # noqa: E402

rep = T.rep
OUT = REPO / "experiment_results" / "swarm_wrap"
# The scene's wrap rule before S11's pick was adopted (2026-11-09): an empty override now means Hold, so merge this
# under the `over` of any candidate flown against the scene as it was (S7-S10's rows name no wrap rule; S11's all do).
# The PRE_S8 / PRE_S10 trap, again.
PRE_S11 = dict(spreadWrapRule=0)
K_SCREEN, K_CONFIRM, K_BENCH = 19000, 20000, 21000
TRACK_EVERY = 5                 # ticks between WrapLog samples: 0.1 s, as the bench records
FULL = 0.99                     # a scale at or above this is the full spread
EXIT_S = 10.0                   # the window after the last building left the formation
STD_KINDS = ("cross", "pass", "street")
# the search flight, calibrated on ERKS's wrapped moments in internal_4 (centroid a median 0.78 of the building's
# radius off its centre, beyond the rule's 0.8 half the time; the stick held half the time, at full deflection): the
# street circuit of the target building's block at full stick, pausing beside the building, a hold on it, then away
SEARCH_FROM_M = (120.0, 260.0)  # the start junction's straight-line distance from the building
SEARCH_LEAVE_M = (150.0, 300.0)
SEARCH_PAUSE_P = 0.5            # the chance of a pause after each corner of the circuit
SEARCH_PAUSE_F = (0.4, 1.2)     # a pause's point: this share of the building's enclosing radius off its centre, ...
SEARCH_PAUSE_MAX_M = 16.0       # ... at most this far (beyond it the building leaves the ~16 m formation)
SEARCH_PAUSE_S = (12.0, 22.0)   # long enough to arrive and settle there (ERKS's stick was centred half the time)
SEARCH_PAUSE_STICK = 0.5        # a hold's stick cap (the bench's centring pilot, HOLD_GAIN)
SEARCH_HOLD_STICK = 0.30
SEARCH_HOLD_S = (6.0, 12.0)
HOLD_GAIN = S.HOLD_GAIN         # 0.25 1/s: SwarmBenchRunner's DefaultHoldGain
END_STICK = 1e-6                # a hold this small is zero stick (the bench clips the centring pilot to it)
WRAP_RULES = ("Off", "Hold", "HoldUntilLeaving")    # SwarmManager.SpreadWrapRule
INTERNAL_4 = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\UAVS@BERKELEY\DroneSim\experiment\internal_4")


# ============================================================================= the log and its measures

def building_centres(m, rc):
    """Each building's centre as the S7 rules and the wrap test see it: the ellipse's, else the aim circle's."""
    E = rep.ell_rules(m, rc)
    return E.c if E is not None else rep.aim_circle(m, rc)[0]


def wrapped_at(P, cent):
    """Whether any building centre lies inside the formation P (alive drones, (n, 2))."""
    if len(P) < 3:
        return False
    cen = P.mean(0)
    rmax = float(np.linalg.norm(P - cen, axis=1).max())
    near = np.flatnonzero(np.linalg.norm(cent - cen, axis=1) <= rmax)
    return bool(len(near) and rep.inside_formation(P, cent[near]).any())


class WrapLog:
    """A read-only fly() hook: every TRACK_EVERY ticks the scale, whether a building is wrapped, the formation's mean
    radius, the stick and the waypoint; every step the loss instants and the contacts, timed."""

    def __init__(self, m, rc):
        self.cent = building_centres(m, rc)
        self.t, self.sig, self.wrap, self.ring, self.stick, self.wp = [], [], [], [], [], []
        self.loss_t, self.contact_t = [], []
        self._alive, self._contacts = None, 0

    def start(self, sim):
        pass

    def step(self, sim, t, cmd, wp):
        if self._alive is not None and (self._alive & ~sim.alive).any():
            self.loss_t.append(float(t))
        if sim.contacts > self._contacts:
            self.contact_t += [float(t)] * int(sim.contacts - self._contacts)
        self._alive, self._contacts = sim.alive.copy(), sim.contacts
        if sim.tick % TRACK_EVERY:
            return
        P = sim.P[sim.alive]
        self.t.append(float(t))
        self.sig.append(1.0 if sim.sigma is None else float(sim.sigma))
        self.wrap.append(wrapped_at(P, self.cent))
        self.ring.append(float(np.linalg.norm(P - P.mean(0), axis=1).mean()) if len(P) else math.nan)
        self.stick.append(float(np.linalg.norm(cmd)))
        self.wp.append(int(wp))


def phases(t, wrap):
    """Per sample: 0 wrapped, 1 exit (within EXIT_S after the last wrapped sample, none since), 2 other."""
    t, wrap = np.asarray(t, float), np.asarray(wrap, bool)
    last = np.where(wrap, t, -np.inf)
    last = np.maximum.accumulate(last) if len(last) else last
    return np.where(wrap, 0, np.where(t - last <= EXIT_S, 1, 2))


def wrap_measures(t, sig, wrap, ring, loss_t, contact_t, dt=0.1):
    t, sig, wrap, ring = (np.asarray(x, float) for x in (t, sig, wrap, ring))
    wrap = wrap.astype(bool)
    ph = phases(t, wrap)

    def at(times):
        if not len(t):
            return np.zeros(0, int)
        i = np.clip(np.searchsorted(t, np.asarray(times, float), side="right") - 1, 0, len(t) - 1)
        return ph[i]

    lp, cp = at(loss_t), at(contact_t)
    out = dict(flight_s=round(len(t) * dt, 1), sigma_mean=round(float(sig.mean()), 4) if len(sig) else 1.0,
               narrow_frac=round(float((sig < 0.95).mean()), 4) if len(sig) else 0.0,
               wrap_s=round(float(wrap.sum() * dt), 1),
               wrap_full=round(float((sig[wrap] >= FULL).mean()), 4) if wrap.any() else None,
               wrap_narrow=round(float((sig[wrap] < 0.95).mean()), 4) if wrap.any() else None,
               wrap_sigma=round(float(sig[wrap].mean()), 4) if wrap.any() else None,
               wrap_ring=round(float(np.nanmean(ring[wrap])), 2) if wrap.any() else None,
               ring_mean=round(float(np.nanmean(ring)), 2) if len(ring) else None,
               exit_s=round(float((ph == 1).sum() * dt), 1),
               exit_sigma=round(float(sig[ph == 1].mean()), 4) if (ph == 1).any() else None,
               other_s=round(float((ph == 2).sum() * dt), 1),
               loss_events=len(loss_t), contact_events=len(contact_t))
    for k, name in enumerate(("wrap", "exit", "other")):
        out[f"loss_{name}"] = int((lp == k).sum())
        out[f"contact_{name}"] = int((cp == k).sum())
    return out


def log_measures(log):
    return wrap_measures(log.t, log.sig, log.wrap, log.ring, log.loss_t, log.contact_t)


# ============================================================================= the search flight

def search_route(C, seed, k, rng):
    """(start, wps, L, poly, info) of a goal search round one building, as ERKS searched internal_4's goals: the street
    route from a junction SEARCH_FROM_M away to the nearest corner of the building's block; the circuit of the block's
    corners at full stick, either way round, with a pause after each corner (chance SEARCH_PAUSE_P) beside the
    building on that corner's side, SEARCH_PAUSE_F of its enclosing radius off its centre; a hold on its centre; then
    the street route from the corner it started the circuit at to a junction SEARCH_LEAVE_M away. A pause or hold is a
    negative stick (the bench's centring pilot, capped at |stick|). info's hold0..hold1 is the search window: the
    circuit's first waypoint to the end of the hold."""
    net, rc = C["net"], C["rcity"]
    blocks = net.interior_blocks()
    js = np.flatnonzero(net.j_interior)
    for _ in range(200):
        b = blocks[int(rng.integers(len(blocks)))]
        bi = int(b["building"])
        cb, Rb = rc.mec_c[bi], float(rc.mec_R[bi])
        if len(b["corners"]) < 3:
            continue
        d0 = np.linalg.norm(net.jpos[js] - cb, axis=1)
        far = js[(d0 >= SEARCH_FROM_M[0]) & (d0 <= SEARCH_FROM_M[1])]
        if not len(far):
            continue
        j0 = int(far[int(rng.integers(len(far)))])
        corners = np.array(b["corners"])
        ang = np.arctan2(*(net.jpos[corners] - b["centre"]).T[::-1])
        ring = [int(c) for c in corners[np.argsort(ang)]]
        jc = ring.index(int(corners[int(np.argmin(np.linalg.norm(net.jpos[corners] - net.jpos[j0], axis=1)))]))
        ring = ring[jc:] + ring[:jc]
        if rng.random() < 0.5:
            ring = [ring[0]] + ring[1:][::-1]
        jl, _ = net.route(j0, ring[0])
        if jl is None:
            continue
        d1 = np.linalg.norm(net.jpos[js] - net.jpos[ring[0]], axis=1)
        far1 = js[(d1 >= SEARCH_LEAVE_M[0]) & (d1 <= SEARCH_LEAVE_M[1])]
        if not len(far1):
            continue
        j1 = int(far1[int(rng.integers(len(far1)))])
        jl1, _ = net.route(ring[0], j1)
        if jl1 is None:
            continue
        wps, prev = [], net.jpos[j0]
        for j in jl[1:]:
            q = net.jpos[j]
            wps.append((float(q[0]), float(q[1]), 1.0, S.timeout_for(float(np.linalg.norm(q - prev)))))
            prev = q
        hold0, n_pause = len(wps), 0
        for j in ring[1:] + [ring[0]]:
            q = net.jpos[j]
            wps.append((float(q[0]), float(q[1]), 1.0, S.timeout_for(float(np.linalg.norm(q - prev)))))
            prev = q
            if rng.random() < SEARCH_PAUSE_P:
                u = (q - cb) / max(float(np.linalg.norm(q - cb)), 1e-9)
                p = cb + min(float(rng.uniform(*SEARCH_PAUSE_F)) * Rb, SEARCH_PAUSE_MAX_M) * u
                wps.append((float(p[0]), float(p[1]), -SEARCH_PAUSE_STICK, float(rng.uniform(*SEARCH_PAUSE_S))))
                n_pause += 1
        wps.append((float(cb[0]), float(cb[1]), -SEARCH_HOLD_STICK, float(rng.uniform(*SEARCH_HOLD_S))))
        hold1 = len(wps) - 1
        prev = net.jpos[ring[0]]
        for j in jl1[1:]:
            q = net.jpos[j]
            wps.append((float(q[0]), float(q[1]), 1.0, S.timeout_for(float(np.linalg.norm(q - prev)))))
            prev = q
        start = net.jpos[j0]
        poly = np.vstack([start] + [np.array(w[:2]) for w in wps if w[2] >= 0])
        L = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
        leave_L = float(np.linalg.norm(np.diff(np.vstack([cb] + [net.jpos[j] for j in jl1[1:]]), axis=0),
                                       axis=1).sum())
        return start, wps, L, poly, dict(building=bi, centre=[float(cb[0]), float(cb[1])], hold0=hold0, hold1=hold1,
                                         n_pan=n_pause, leave_L=leave_L)
    raise RuntimeError("no search route")


def fly(m, C, P0, wps, d, seed, settle=T.SETTLE_S, hook=None):
    """swarmtune.fly with the bench's holds: a waypoint with a negative stick is a centring pilot on its point
    (stick = HOLD_GAIN x error / maxSpeed, capped at |stick|) for its timeout, never captured (SwarmBenchRunner;
    split_probe.Flight). Every other waypoint is flown exactly as swarmtune.fly flies it."""
    sim = rep.ContactSim(m, C["rcity"], np.array(P0, float), np.zeros((len(P0), 2)), seed)
    if hook is not None:
        hook.start(sim)
    while sim.t < settle - 1e-9 and sim.alive.any():
        sim.step(np.zeros(2), 0.0, d)
    t_off = sim.t
    sim.contacts = sim.contact_ticks = 0
    sim.kills = []
    sim.n0 = int(sim.alive.sum())
    every = max(1, int(round(T.SAMPLE_S / m.dt)))
    gain = HOLD_GAIN / m.max_speed
    Tmax = sum(w[3] for w in wps) + 10.0
    done, samples = [math.nan] * len(wps), []
    wp, t0, t_end = 0, 0.0, math.inf
    while sim.t - t_off < min(Tmax, t_end) - 1e-9 and sim.alive.any():
        t = sim.t - t_off
        c = sim.P[sim.alive].mean(0)
        cmd = np.zeros(2)
        if wp < len(wps):
            x, z, stick, timeout = wps[wp]
            dv = np.array([x, z]) - c
            dd = float(np.linalg.norm(dv))
            if (stick >= 0 and dd < T.CAPTURE_M) or (timeout > 0 and t - t0 > timeout):
                done[wp] = t
                wp, t0 = wp + 1, t
                if wp >= len(wps):
                    t_end = t + T.END_HOLD_S
                else:
                    x, z, stick, timeout = wps[wp]
                    dv = np.array([x, z]) - c
                    dd = float(np.linalg.norm(dv))
            if wp < len(wps):
                if stick < 0:
                    cmd = dv * gain
                    sn = float(np.linalg.norm(cmd))
                    if sn > -stick:
                        cmd = cmd * (-stick / sn)
                else:
                    cmd = dv / max(dd, 1e-9) * stick
        if sim.tick % every == 0:
            samples.append((t, c.copy(), np.where(sim.alive[:, None], sim.P, np.nan)))
        if hook is not None:
            hook.step(sim, t, cmd, wp)
        sim.step(cmd, 0.0, d)
    if sim.alive.any():
        samples.append((sim.t - t_off, sim.P[sim.alive].mean(0), np.where(sim.alive[:, None], sim.P, np.nan)))
    return sim, samples, done


def search_measures(info, done, log_t, sig, wrap, wp, samples):
    """The hold window (from the first hold to the end of the last): the share wrapped and the share wrapped at full
    spread; the leave: the speed from the last hold's end to the last waypoint, and the scale over its first EXIT_S."""
    t, sig, wrap, wp = (np.asarray(x) for x in (log_t, sig, wrap, wp))
    win = (wp >= info["hold0"]) & (wp <= info["hold1"])
    out = dict(search_b=info["building"], n_pan=info["n_pan"],
               hold_s=round(float(win.sum() * 0.1), 1),
               hold_wrapped=round(float(wrap[win].mean()), 4) if win.any() else None,
               hold_full=round(float((wrap[win] & (sig[win] >= FULL)).mean()), 4) if win.any() else None,
               hold_sigma=round(float(sig[win].mean()), 4) if win.any() else None)
    t_leave = done[info["hold1"]]
    t_last = done[-1]
    if math.isfinite(t_leave) and math.isfinite(t_last) and t_last > t_leave:
        out["leave_v"] = round(info["leave_L"] / (t_last - t_leave), 3)
        sel = (t >= t_leave) & (t < t_leave + EXIT_S)
        out["leave_sigma"] = round(float(sig[sel].mean()), 4) if sel.any() else None
    else:
        out["leave_v"], out["leave_sigma"] = None, None
    return out


# ============================================================================= jobs

def run_job(j):
    """One flight with the WrapLog on it: S8's routes for cross, pass, street and face (swarmtune.route,
    swarmshape.face_route, the same random streams), the search route here."""
    t0 = time.time()
    base = dict(job_id=T.job_id(j), cand=j["cand"], kind=j["kind"], seed=j["seed"], k=j["k"], d=j["d"])
    try:
        C = T.city(j["seed"])
        rng = np.random.default_rng(T.route_seed(j["kind"], j["seed"], j["k"]))
        fseed = int(rng.integers(1 << 30))
        info = None
        if j["kind"] == "face":
            start, wps, L, poly, info = SH.face_route(C, j["seed"], j["k"], rng)
        elif j["kind"] == "search":
            start, wps, L, poly, info = search_route(C, j["seed"], j["k"], rng)
        else:
            start, wps, L, poly = T.route(C, j["kind"], j["k"], rng)
        over = dict(j["over"])
        m = T.model(over)
        P0 = T.start_positions(C, j["d"], fseed, over, start)
        log = WrapLog(m, C["rcity"])
        if j["kind"] == "search":
            sim, samples, done = fly(m, C, P0, wps, j["d"], fseed, hook=log)
        else:
            sim, samples, done = T.fly(m, C, P0, wps, j["d"], fseed, hook=log)
        r = dict(base, **T.score_flight(C, start, wps, poly, L, sim, samples, done, sim.n0), **log_measures(log))
        if j["kind"] == "face":
            r.update(SH.face_measures(C, info, samples, done))
        elif j["kind"] == "search":
            r.update(search_measures(info, done, log.t, log.sig, log.wrap, log.wp, samples))
        return dict(r, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


def jobs_for(spec, seeds, flights, k0):
    """Every candidate on every flight kind; ordered by route index then kind, so a run stopped early has flown every
    candidate equally (lesson 17)."""
    cands, jobs = {}, []
    for c in spec:
        cid = T.cand_id(c["over"], c["d"])
        cands[cid] = dict(over=c["over"], d=c["d"], label=c.get("label", ""))
        jobs += T.jobs_for(cid, c["over"], c["d"], seeds=seeds, flights=flights, k0=k0)
    return cands, sorted(jobs, key=lambda j: (j["k"], j["kind"], j["seed"]))


# ============================================================================= tables

def _tw(g, val, wt):
    """A time-weighted mean of a per-flight share."""
    ok = g[val].notna() & (g[wt] > 0)
    return float((g[val][ok] * g[wt][ok]).sum() / g[wt][ok].sum()) if ok.any() else math.nan


def wrap_scores(g):
    """The wrap measures over a set of flights: time-weighted shares, and loss events and contacts per hour of each
    phase's exposure."""
    r = dict(wrap_full=_tw(g, "wrap_full", "wrap_s"), wrap_sigma=_tw(g, "wrap_sigma", "wrap_s"),
             wrap_narrow=_tw(g, "wrap_narrow", "wrap_s") if "wrap_narrow" in g else math.nan,
             wrap_ring=_tw(g, "wrap_ring", "wrap_s"), exit_sigma=_tw(g, "exit_sigma", "exit_s"),
             wrap_share=float(g.wrap_s.sum() / max(g.flight_s.sum(), 1e-9)), sigma=_tw(g, "sigma_mean", "flight_s"),
             all_h=float(g.flight_s.sum() / 3600.0), all_loss_events=int(g.loss_events.sum()),
             all_contacts=int(g.contacts.sum()))
    r["all_loss_ph"] = r["all_loss_events"] / r["all_h"] if r["all_h"] else math.nan
    r["all_contact_ph"] = r["all_contacts"] / r["all_h"] if r["all_h"] else math.nan
    for ph in ("wrap", "exit", "other"):
        hours = g[f"{ph}_s"].sum() / 3600.0
        r[f"{ph}_h"] = hours
        r[f"loss_{ph}"] = int(g[f"loss_{ph}"].sum())
        r[f"loss_{ph}_ph"] = g[f"loss_{ph}"].sum() / hours if hours else math.nan
        r[f"contact_{ph}_ph"] = g[f"contact_{ph}"].sum() / hours if hours else math.nan
    return r


def search_scores(g):
    if not len(g):
        return {}
    return dict(search_n=len(g), search_hold_wrapped=float(g.hold_wrapped.mean()),
                search_hold_full=float(g.hold_full.mean()), search_hold_sigma=float(g.hold_sigma.mean()),
                search_leave_v=float(g.leave_v.astype(float).mean()),
                search_leave_sigma=float(g.leave_sigma.astype(float).mean()),
                search_lost=int(g.lost.sum()), search_loss_events=int(g.loss_events.sum()),
                search_contacts=int(g.contacts.sum()), search_T=float(g["T"].mean()),
                search_open=float(g.open_end.mean()), search_one=float(g.one_frac.mean()))


def score_rows(g):
    std = g[g.kind.isin(STD_KINDS)]
    r = T.score_candidate(std) if len(std) else {}
    if len(std):
        km = std.L.sum() / 1000.0
        r["loss_events"] = int(std.loss_events.sum())
        r["loss_events_km"] = std.loss_events.sum() / km
    r.update(SH.face_scores(g[g.kind == "face"]))
    r.update(search_scores(g[g.kind == "search"]))
    r.update(wrap_scores(g))
    r.update({f"search_{k}": v for k, v in wrap_scores(g[g.kind == "search"]).items()
              if k in ("wrap_full", "exit_sigma")} if (g.kind == "search").any() else {})
    return r


def table(name, root=None):
    import pandas as pd
    df, cands, _ = T.read(name, root or OUT)
    rows = []
    for cid, g in df.groupby("cand"):
        c = cands.get(cid, {})
        rows.append(dict(cand=cid, label=c.get("label", ""), d=float(g.d.iloc[0]), **score_rows(g),
                         **{f"o:{k}": v for k, v in c.get("over", {}).items()}))
    tab = pd.DataFrame(rows)
    order = {c.get("label", ""): i for i, c in enumerate(cands.values())}
    return tab.assign(_o=tab.label.map(order)).sort_values("_o").drop(columns="_o")


COLS = ["label", "n", "km", "wrap_full", "wrap_sigma", "wrap_ring", "exit_sigma", "search_hold_full",
        "search_leave_v", "search_leave_sigma", "street_v_coh", "v_coh", "one", "loss_events", "loss_events_km",
        "contact_km", "open", "hull", "cross_wrapped", "face_t_split", "loss_wrap", "loss_exit", "loss_other",
        "loss_exit_ph", "loss_other_ph", "contact_exit_ph", "contact_other_ph", "sigma"]


def show(tab, n=80):
    import pandas as pd
    pd.set_option("display.width", 460)
    pd.set_option("display.max_columns", 80)
    print(tab[[c for c in COLS if c in tab.columns]].head(n).round(3).to_string(index=False))


# ============================================================================= runs

def _run(a, seeds, k0, label):
    fl = dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face, search=a.search)
    cands, jobs = jobs_for(F.load_cands(a.cand), seeds, {k: v for k, v in fl.items() if v}, k0)
    T.run(a.name, cands, jobs, a.jobs, label, root=OUT, fn=run_job)
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab)


def cmd_screen(a):
    _run(a, (T.TUNE_SEED,), a.k0 if a.k0 is not None else K_SCREEN, "screen")


def cmd_confirm(a):
    _run(a, (T.TUNE_SEED,) + T.HELD_SEEDS, a.k0 if a.k0 is not None else K_CONFIRM, "confirm")


def cmd_table(a):
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab)


# ============================================================================= the rule

RULE = dict(base="E15_35a", references=["F20"], deficit_ratio=0.5, full_gain=0.25, safe_p=0.10, contact_ratio=1.25,
            open_margin=0.02, street_floor=0.97, leave_floor=0.97, split_ratio=1.1, hull_min=0.5,
            surround_margin=0.05, tie=0.005)


def rows_for_rule(name):
    """{code: row} of a run's table, by the label's first word."""
    tab = table(name)
    return {str(r["label"]).split()[0]: r for _, r in tab.iterrows()}


def judge(rows, rule):
    """Verdicts against the base (the scene): safe, delivers, and the pick (see cmd_rule's note)."""
    base = rows[rule["base"]]
    out = {}
    for code, r in rows.items():
        if code == rule["base"]:
            out[code] = dict(role="base")
            continue
        why = []
        p = SP.worse_p(int(r["all_loss_events"]), int(base["all_loss_events"]), float(r["all_h"]), float(base["all_h"]))
        if p < rule["safe_p"]:
            why.append(f"loss events {int(r['all_loss_events'])} in {r['all_h']:.1f} h against "
                       f"{int(base['all_loss_events'])} in {base['all_h']:.1f} h (p {p:.3f})")
        if r["all_contact_ph"] > rule["contact_ratio"] * base["all_contact_ph"]:
            why.append(f"contacts {r['all_contact_ph']:.1f}/h against {base['all_contact_ph']:.1f}")
        if r["open"] > base["open"] + rule["open_margin"]:
            why.append(f"in pieces {r['open']:.3f} against {base['open']:.3f}")
        if r["street_v_coh"] < rule["street_floor"] * base["street_v_coh"]:
            why.append(f"street speed {r['street_v_coh']:.3f} against {base['street_v_coh']:.3f}")
        if r["search_leave_v"] < rule["leave_floor"] * base["search_leave_v"]:
            why.append(f"speed leaving a search {r['search_leave_v']:.3f} against {base['search_leave_v']:.3f}")
        if r.get("face_t_split", 0) > rule["split_ratio"] * base.get("face_t_split", 0):
            why.append(f"square-on split {r['face_t_split']:.2f} s against {base['face_t_split']:.2f}")
        if r["hull"] < rule["hull_min"]:
            why.append(f"hull share {r['hull']:.3f}")
        if r["cross_wrapped"] < base["cross_wrapped"] - rule["surround_margin"]:
            why.append(f"buildings surrounded {r['cross_wrapped']:.3f} against {base['cross_wrapped']:.3f}")
        deliver = []
        if not (1.0 - r["wrap_sigma"]) <= rule["deficit_ratio"] * (1.0 - base["wrap_sigma"]):
            deliver.append(f"shortfall from full spread while wrapped {1 - r['wrap_sigma']:.4f} against "
                           f"{1 - base['wrap_sigma']:.4f}")
        if not r["wrap_full"] >= base["wrap_full"] + rule["full_gain"]:
            deliver.append(f"wrapped time at full spread {r['wrap_full']:.3f} against {base['wrap_full']:.3f}")
        role = "reference" if code in rule.get("references", []) else "candidate"
        out[code] = dict(role=role, safe=not why, delivers=not deliver, unsafe=why, short=deliver, loss_p=round(p, 4))
    ok = [c for c, v in out.items() if v["role"] == "candidate" and v.get("safe") and v.get("delivers")]
    pick = None
    if ok:
        best = max(rows[c]["wrap_sigma"] for c in ok)
        near = [c for c in ok if rows[c]["wrap_sigma"] >= best - rule["tie"]]
        pick = max(near, key=lambda c: rows[c]["street_v_coh"])
    return out, pick


def cmd_rule(a):
    path = OUT / "rule.json"
    if path.exists() and not a.force:
        raise SystemExit(f"{path} exists; --force to rewrite it (only before the confirmation flies)")
    rule = dict(RULE, written=time.strftime("%Y-%m-%d %H:%M:%S"),
                note="Written after the screen (screen1) and before the confirmation. Base: the scene (E15_35a, "
                     "spreadWrapRule Off); F20 (the spread held) is a reference, not a candidate. Safe against the "
                     "base: loss events over every flight (instants with a drone lost: a colliding pair is one) not "
                     "more frequent per hour of flight at one-sided exact p >= safe_p; contacts per hour at most "
                     "contact_ratio x; ending in pieces at most open_margin more; street cohesive speed and the speed "
                     "leaving a search at least street_floor x and leave_floor x (the cost ERKS's open-loop replay "
                     "predicts); the square-on split at most split_ratio x; hull share at least hull_min; buildings met "
                     "on a straight line surrounded at most surround_margin less. Delivers (the question): the "
                     "shortfall from full spread while wrapped, 1 - the mean scale then, at most deficit_ratio x the "
                     "base's, and the share of the wrapped time at full spread (scale >= 0.99) at least full_gain "
                     "above it. The full-spread share alone measures mostly the 3 s filter's lag on short wraps, which "
                     "is why the mean scale carries the bar. Pick: among the safe and delivering candidates, the one "
                     "nearest full spread while wrapped; within `tie` of it, the fastest in the streets. None: the "
                     "scene stays as it is.")
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rule, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rule, indent=1))


def cmd_choose(a):
    rule = json.loads((OUT / "rule.json").read_text(encoding="utf-8"))
    rows = rows_for_rule(a.name)
    verdicts, pick = judge(rows, rule)
    (OUT / a.name / "verdicts.json").write_text(json.dumps(dict(rule=rule, verdicts=verdicts, pick=pick), indent=1),
                                                encoding="utf-8")
    for code, v in verdicts.items():
        print(code, json.dumps(v))
    print("pick:", pick)


# ============================================================================= ERKS's flights, open loop

def cmd_human(a):
    """ERKS's three swarm flights (internal_4) with the spread rule replayed on the drones as they flew, under each wrap
    rule, the 3 s filter included: the rule as flown is checked against the logged scale, and every rule's share of
    the time wrapped at full spread, and of the street time narrowed, is reported (open loop: the drones fly as
    recorded; only the spread decision changes)."""
    import pandas as pd
    sp = T.sp
    runs = pd.read_csv(os.path.join(INTERNAL_4, "results", "runs.csv"), sep=";")
    goals = pd.read_csv(os.path.join(INTERNAL_4, "results", "goals.csv"), sep=";")
    air = dict(sp.resolve_airframe())
    rows, series = [], {}
    for stem in sorted(runs[runs.condition == "Swarm"].run):
        ses = json.loads(open(os.path.join(INTERNAL_4, stem + "_session.json"), encoding="utf-8").read())
        L = json.loads(open(ses["layout"]["path"], encoding="utf-8-sig").read())
        rc = rep.PolyCity([np.reshape(b["poly_xz"], (-1, 2)) for b in L["boxes"]])
        dr = pd.read_csv(os.path.join(INTERNAL_4, stem + "_drones.csv"), sep=";")
        hd = pd.read_csv(os.path.join(INTERNAL_4, stem + "_head.csv"), sep=";")
        sh = pd.read_csv(os.path.join(INTERNAL_4, stem + "_shape.csv"), sep=";")
        t0 = float(runs[runs.run == stem].startSec.iloc[0])
        t1 = float(goals[goals.run == stem].decisionTimeSec.max())
        models = [rep.Model(dict(sp.resolve_swarm(T.SCENE, INTERNAL_4, stem, dict(spreadWrapRule=w))), air)
                  for w in range(len(WRAP_RULES))]
        cent = building_centres(models[0], rc)
        look = models[0].look_ahead * models[0].S
        sig = [None] * len(models)
        tp = None
        for t, g in dr[dr.alive == 1].groupby("t"):
            P = g[["gtX", "gtZ"]].to_numpy()
            cen = P.mean(0)
            h = hd.iloc[int(np.abs(hd.t.to_numpy() - t).argmin())]
            cmd = models[0].stick_to_world((h.inRoll, h.inPitch), h.bodyYaw)
            cand = rc.near(cen[None], look + 40.0)
            row = dict(run=stem, t=float(t), timed=bool(t0 <= t <= t1), wrapped=wrapped_at(P, cent),
                       logged=float(np.interp(t, sh.t.to_numpy(), sh.spreadScale.to_numpy())))
            for w, m in enumerate(models):
                miss = rep.aim_miss(m, rc, cand, cen, cmd)
                target = rep.spread_target(m, rc, cand, cen, miss, P, cmd)
                sig[w] = target if sig[w] is None else sig[w] + (1 - math.exp(-(t - tp) / m.spread_filter)) * (
                    target - sig[w])
                row[WRAP_RULES[w]] = sig[w]
            tp = t
            rows.append(row)
    d = pd.DataFrame(rows)
    out = OUT / "human"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "samples.csv", index=False)
    x = d[d.timed]
    summary = dict(check_mean_abs=float((x.Off - x.logged).abs().mean()), wrapped_share=float(x.wrapped.mean()))
    for w in WRAP_RULES:
        summary[w] = dict(wrapped_full=float((x[x.wrapped][w] >= FULL).mean()),
                          wrapped_narrowed=float((x[x.wrapped][w] < 0.95).mean()),
                          street_narrowed=float((x[~x.wrapped][w] < 0.95).mean()),
                          mean=float(x[w].mean()))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("screen", "confirm"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.add_argument("--cand", required=True)
        for k, v in (("cross", 40), ("pass", 30), ("street", 40), ("face", 24), ("search", 60)):
            p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int, default=v)
        p.add_argument("--k0", type=int)
        p.add_argument("--jobs", type=int, default=24)
    p = sub.add_parser("table")
    p.add_argument("--name", required=True)
    p = sub.add_parser("rule")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("choose")
    p.add_argument("--name", required=True)
    sub.add_parser("human")
    a = ap.parse_args()
    dict(screen=cmd_screen, confirm=cmd_confirm, table=cmd_table, rule=cmd_rule, choose=cmd_choose,
         human=cmd_human)[a.cmd](a)


if __name__ == "__main__":
    main()
