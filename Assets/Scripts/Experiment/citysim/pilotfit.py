"""How internal_3's pilots flew while they were still searching: the fitted pilot of the search study (citysim.search).

    python -B -m citysim.vissim pilot-fit [--test internal_3]    # -> experiment_results/<test>/results/pilot_model_<test>.json

The searching window of a leg runs from its start (leaving the previous goal's zone, or the run's start) to the first
instant the next goal is on a shown feed at 80 m (legs.csv discoverySec, citysim.discovery's definition): before it the
pilot had nothing to steer for, after it they flew at the goal. Two pilots who had flown the city before may still have
known where goals could be; nothing here can remove that, and the window is where it matters least.

Per interface, from _head.csv (~10 Hz) and the centroid track:
  pauses   the stick centred (|roll, pitch| < DEADBAND after analyse's dead band) for at least PAUSE_MIN_S: their
           durations, and how many there were per turn (they sit at the turns: a pilot stops, looks, turns)
  turns    while the stick is out (|u| >= MOVING), the commanded course (analyse.command_velocity) is turning where it
           changes faster than TURN_RATE_MIN; a turn episode's angle is the course change across it. Turns smaller than
           THETA_MIN are not intent -- steering round a building or along a street, which the planner does -- so the runs
           either side of one merge
  runs     the straight stretches between turns and pauses, in metres of the centroid's path. The first and last run of
           a window are cut by its edges, so the mean is the censored exponential's: every metre flown over the turns
           (each ends a run; the random walker's form, with this mean)
  stick    the stick's magnitude while it is out (internal_3: full stick almost always)
  heading  how fast the heading turned in a turn (single drone: the drone's heading = the body yaw; swarm: the rig's
           body yaw), and how far the commanded course sat off it (a sideways stick)
  head     (swarm) the head's yaw relative to the body yaw, headYaw - bodyYaw - 90, cut into DWELLS (still: slower than
           HEAD_MOVE_DEG_S for at least HEAD_DWELL_MIN_S) and moves between them: the (angle, duration) of every dwell,
           which the search's head model resamples, and the moves' speed
  spread   (swarm) the share of the window nearest each spread mark
Everything is kept as empirical samples beside the summaries, so the search draws from what was flown.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import DEFAULT_ROOT, REPO, use_bench_modules  # noqa: E402
from citysim.operator_band import SPREAD_MARKS, results_dir  # noqa: E402

DEADBAND = 0.1               # analyse.DEFAULTS["deadband"]
MOVING = 0.3                 # stick magnitude above which the course is defined (analyse: 0.3 maxSpeed)
PAUSE_MIN_S = 1.0
TURN_RATE_MIN = 8.0          # deg/s of commanded course
TURN_BRIDGE_S = 0.4          # turning samples closer than this are one turn
THETA_MIN = 20.0             # deg: smaller course changes are steering, not intent
HEAD_MOVE_DEG_S = 30.0
HEAD_DWELL_MIN_S = 0.2
RECENTRE_DEG = 30.0          # a body-yaw jump this big in one sample is a recentre, not a turn
CONDITIONS = ("SingleDrone", "Swarm")


def model_path(test="internal_3"):
    return results_dir(test) / f"pilot_model_{test}.json"


def load(test="internal_3"):
    p = model_path(test)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist: run `python -B -m citysim.vissim pilot-fit --test {test}`")
    return json.loads(p.read_text(encoding="utf-8"))


def _wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _episodes(mask, t, bridge=0.0, min_s=0.0):
    """[(i0, i1)] index spans (inclusive start, exclusive end) of True runs, gaps shorter than bridge closed, runs
    shorter than min_s dropped."""
    m = np.asarray(mask, bool).copy()
    if bridge > 0 and m.any():
        idx = np.flatnonzero(m)
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1 and t[b] - t[a] <= bridge:
                m[a:b] = True
    out = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if t[min(j, n - 1)] - t[i] >= min_s:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def windows(test="internal_3", root=DEFAULT_ROOT, zone=15.0):
    """[(run, t0, t1, leg)]: every leg's searching window that lasted at least 2 s."""
    use_bench_modules()
    import analyse
    _, _, runs = analyse.load_test(str(root), test)
    legs_csv = pd.read_csv(results_dir(test) / "legs.csv", sep=None, engine="python")
    out = []
    for r in runs:
        for L in analyse.goal_legs(r, zone):
            row = legs_csv[(legs_csv.run == r.stem) & (legs_csv.leg == L["leg"]) & (legs_csv.phase == "transit")]
            if not len(row) or not np.isfinite(row.discoverySec.iloc[0]):
                continue
            t1 = L["t0"] + float(row.discoverySec.iloc[0])
            if t1 - L["t0"] >= 2.0:
                out.append((r, L["t0"], t1, L["leg"]))
    return out


def signals(r):
    """The run's head-log signals: t, stick (n, 4) dead-banded, |u|, commanded course, body yaw, head yaw rel."""
    import analyse
    h = r.head
    t = h.t.values.astype(float)
    U = analyse.deadband(h[["inPitch", "inRoll", "inYaw", "inThrottle"]].values.astype(float), DEADBAND)
    mag = np.minimum(np.hypot(U[:, 0], U[:, 1]), 1.0)
    vx, vz = analyse.command_velocity(U[:, 1], U[:, 0], h.bodyYaw.values)
    course = np.degrees(np.arctan2(vx, vz))
    body = h.bodyYaw.values.astype(float)
    head = _wrap(h.headYaw.values - body - 90.0)
    tracking = ~((h.headYaw.values == 0) & (h.headPitch.values == 0)) if "headPitch" in h else np.ones(len(t), bool)
    jump = np.r_[False, np.abs(_wrap(np.diff(body))) > RECENTRE_DEG]
    dist = np.interp(t, r.ct, np.r_[0.0, np.cumsum(np.hypot(np.diff(r.cx), np.diff(r.cz)))])
    return dict(t=t, U=U, mag=mag, course=course, body=body, head=head, ok_head=tracking & ~jump, dist=dist)


def segment(sig, t0, t1):
    """One searching window cut into runs, turns and pauses: dict(runs=[(metres, complete)], turns=[deg],
    pauses=[s], turn_rates=[deg/s], offsets=[deg], stick=[|u|], seconds, ends_in_pause=[bool per complete run])."""
    t = sig["t"]
    sel = np.flatnonzero((t >= t0) & (t < t1))
    out = dict(runs=[], turns=[], small_turns=[], pauses=[], turn_rates=[], offsets=[], stick=[], seconds=0.0,
               run_end_pause=[])
    if len(sel) < 5:
        return out
    tt, mag, course, body, dist = (sig[k][sel] for k in ("t", "mag", "course", "body", "dist"))
    out["seconds"] = float(tt[-1] - tt[0])
    moving = mag >= MOVING
    still = mag < DEADBAND
    out["stick"] = mag[moving].tolist()
    pauses = _episodes(still, tt, min_s=PAUSE_MIN_S)
    out["pauses"] = [float(tt[min(j, len(tt) - 1)] - tt[i]) for i, j in pauses]
    # course rate while moving (unwrapped across moving samples only)
    c = np.full(len(tt), np.nan)
    c[moving] = np.degrees(np.unwrap(np.radians(course[moving])))
    dt = np.diff(tt)
    rate = np.zeros(len(tt))
    both = moving[1:] & moving[:-1]
    rate[1:][both] = np.diff(c)[both] / np.maximum(dt[both], 1e-3)
    turning = moving & (np.abs(rate) > TURN_RATE_MIN)
    body_rate = np.r_[0.0, _wrap(np.diff(body)) / np.maximum(dt, 1e-3)]
    body_rate[np.abs(body_rate) > 300] = 0.0                  # recentres
    events = []                                               # (index, kind, angle)
    for i, j in _episodes(turning, tt, bridge=TURN_BRIDGE_S):
        a = i - 1 if i > 0 and moving[i - 1] else i
        b = j if j < len(tt) and moving[j] else j - 1
        ang = float(c[b] - c[a]) if np.isfinite(c[a]) and np.isfinite(c[b]) else 0.0
        if abs(ang) >= THETA_MIN:
            events.append((i, j, "turn", ang))
            r_ = np.abs(body_rate[i:j])
            out["turn_rates"].append(float(np.percentile(r_, 90)) if len(r_) else math.nan)
        else:
            out["small_turns"].append(ang)
    for i, j in pauses:
        events.append((i, j, "pause", 0.0))
    events.sort()
    # runs between events, in metres of the centroid's path
    prev_end, prev_kind = 0, "edge"
    for i, j, kind, ang in events + [(len(tt), len(tt), "edge", 0.0)]:
        if i > prev_end:
            m = float(dist[min(i, len(tt) - 1)] - dist[prev_end])
            if m > 1.0:
                complete = prev_kind != "edge" and kind != "edge"
                out["runs"].append((m, complete, kind))
        if kind == "turn":
            out["turns"].append(ang)
        prev_end, prev_kind = max(prev_end, j), kind
    off = _wrap(course[moving] - body[moving])
    out["offsets"] = off.tolist()
    return out


def head_dwells(sig, t0, t1):
    """[(angle deg, duration s)] dwells of the head's relative yaw in the window, and the moves' speeds (deg/s)."""
    t = sig["t"]
    sel = np.flatnonzero((t >= t0) & (t < t1) & sig["ok_head"])
    if len(sel) < 5:
        return [], []
    tt, hd = t[sel], sig["head"][sel]
    rate = np.r_[0.0, np.abs(_wrap(np.diff(hd))) / np.maximum(np.diff(tt), 1e-3)]
    still = rate < HEAD_MOVE_DEG_S
    dwells = [(float(np.median(hd[i:j])), float(tt[min(j, len(tt) - 1)] - tt[i]))
              for i, j in _episodes(still, tt, min_s=HEAD_DWELL_MIN_S)]
    moves = [float(np.percentile(rate[i:j], 90)) for i, j in _episodes(~still, tt) if j - i >= 1]
    return dwells, moves


def fit(test="internal_3", root=DEFAULT_ROOT):
    use_bench_modules()
    import analyse
    wins = windows(test, root)
    pools = {c: dict(runs=[], turns=[], small_turns=[], pauses=[], turn_rates=[], offsets=[], stick=[], seconds=0.0,
                     legs=0, dwells=[], moves=[], spread=[], by_pilot={}) for c in CONDITIONS}
    for r, t0, t1, leg in wins:
        sig = signals(r)
        seg = segment(sig, t0, t1)
        p = pools[r.condition]
        p["legs"] += 1
        p["seconds"] += seg["seconds"]
        for k in ("runs", "turns", "small_turns", "pauses", "turn_rates", "offsets", "stick"):
            p[k].extend(seg[k])
        bp = p["by_pilot"].setdefault(r.pid, dict(seconds=0.0, metres=0.0, turns=0, pauses=0))
        bp["seconds"] += seg["seconds"]
        bp["metres"] += sum(m for m, _, _ in seg["runs"])
        bp["turns"] += len(seg["turns"])
        bp["pauses"] += len(seg["pauses"])
        if r.condition == "Swarm":
            d, mv = head_dwells(sig, t0, t1)
            p["dwells"].extend(d)
            p["moves"].extend(mv)
            sp = analyse.spread_signal(r.head, r.shape)
            p["spread"].extend(np.asarray(sp)[(sig["t"] >= t0) & (sig["t"] < t1)].tolist())
    out = dict(test=test, generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
               window="leg start -> first sight at 80 m (legs.csv discoverySec)",
               params=dict(deadband=DEADBAND, moving=MOVING, pause_min_s=PAUSE_MIN_S, turn_rate_min=TURN_RATE_MIN,
                           turn_bridge_s=TURN_BRIDGE_S, theta_min=THETA_MIN, head_move_deg_s=HEAD_MOVE_DEG_S,
                           head_dwell_min_s=HEAD_DWELL_MIN_S), conditions={})
    for cond, p in pools.items():
        runs = p["runs"]
        metres = float(sum(m for m, _, _ in runs))
        n_turn_ends = sum(1 for _, _, k in runs if k == "turn")
        n_ends = sum(1 for _, _, k in runs if k in ("turn", "pause"))
        complete = [m for m, c, _ in runs if c]
        stick = np.array(p["stick"], float)
        off = np.abs(np.array(p["offsets"], float))
        e = dict(seconds=round(p["seconds"], 1), legs=p["legs"], metres=round(metres, 1),
                 run=dict(mean_m=round(metres / max(n_turn_ends, 1), 1), n_ends=n_ends, n_turn_ends=n_turn_ends,
                          complete_m=[round(m, 1) for m in complete],
                          complete_median_m=round(float(np.median(complete)), 1) if complete else math.nan),
                 turn=dict(angles_deg=[round(a, 1) for a in p["turns"]],
                           abs_median_deg=round(float(np.median(np.abs(p["turns"]))), 1) if p["turns"] else math.nan,
                           small_angles_deg=[round(a, 1) for a in p["small_turns"]],
                           rate_p50=round(float(np.nanmedian(p["turn_rates"])), 1) if p["turn_rates"] else math.nan,
                           rate_p90=round(float(np.nanpercentile(p["turn_rates"], 90)), 1) if p["turn_rates"]
                           else math.nan,
                           per_km=round(len(p["turns"]) / max(metres / 1000.0, 1e-9), 2)),
                 pause=dict(durations_s=[round(d, 2) for d in p["pauses"]],
                            per_turn=round(len(p["pauses"]) / max(n_turn_ends, 1), 3),
                            time_share=round(sum(p["pauses"]) / max(p["seconds"], 1e-9), 3)),
                 stick=dict(p10=round(float(np.percentile(stick, 10)), 3), p50=round(float(np.median(stick)), 3),
                            full_share=round(float((stick >= 0.95).mean()), 3)) if len(stick) else None,
                 offset=dict(p50=round(float(np.median(off)), 1), p90=round(float(np.percentile(off, 90)), 1),
                             over30_share=round(float((off > 30).mean()), 3)) if len(off) else None,
                 by_pilot={k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in p["by_pilot"].items()})
        if cond == "Swarm":
            dw = np.array(p["dwells"], float).reshape(-1, 2)
            sp = np.array(p["spread"], float)
            sp = sp[np.isfinite(sp)]
            near = np.argmin(np.abs(sp[:, None] - np.array(SPREAD_MARKS)[None]), axis=1) if len(sp) else []
            e["head"] = dict(dwells=[[round(a, 1), round(d, 2)] for a, d in dw],
                             dwell_median_s=round(float(np.median(dw[:, 1])), 2) if len(dw) else math.nan,
                             dwell_angle_p10_p50_p90=[round(float(v), 1) for v in np.percentile(dw[:, 0], [10, 50, 90])]
                             if len(dw) else None,
                             move_speed_deg_s=round(float(np.median(p["moves"])), 1) if p["moves"] else math.nan,
                             still_share=round(float(dw[:, 1].sum() / max(p["seconds"], 1e-9)), 3) if len(dw)
                             else math.nan)
            e["spread_share"] = {f"{m:.1f}": round(float((np.asarray(near) == i).mean()), 4)
                                 for i, m in enumerate(SPREAD_MARKS)} if len(sp) else {}
        out["conditions"][cond] = e
    return out


def summary_lines(model):
    lines = []
    for cond, e in model["conditions"].items():
        lines.append(f"{cond}: {e['seconds']:.0f} s searching over {e['legs']} legs, {e['metres']:.0f} m; "
                     f"run mean {e['run']['mean_m']} m ({e['run']['n_ends']} ends, {e['run']['n_turn_ends']} turns; "
                     f"complete median {e['run']['complete_median_m']} m of {len(e['run']['complete_m'])}); "
                     f"turns |median| {e['turn']['abs_median_deg']} deg, {e['turn']['per_km']}/km, "
                     f"rate p50/p90 {e['turn']['rate_p50']}/{e['turn']['rate_p90']} deg/s; "
                     f"pauses {len(e['pause']['durations_s'])} ({e['pause']['per_turn']} per turn, "
                     f"{e['pause']['time_share']} of the time); stick {e['stick']}; offset {e['offset']}")
        if "head" in e:
            h = e["head"]
            lines.append(f"  head: {len(h['dwells'])} dwells, median {h['dwell_median_s']} s, angles p10/50/90 "
                         f"{h['dwell_angle_p10_p50_p90']}, moves {h['move_speed_deg_s']} deg/s, still "
                         f"{h['still_share']}; spread share {e['spread_share']}")
        lines.append(f"  by pilot: {e['by_pilot']}")
    return lines


def cmd_fit(a):
    model = fit(a.test, a.root)
    path = model_path(a.test)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8", newline="\n")
    for ln in summary_lines(model):
        print(ln)
    print(f"[pilot-fit] written {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test", default="internal_3")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    cmd_fit(ap.parse_args())
