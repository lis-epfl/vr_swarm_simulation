"""Analysis of ExperimentRecorder runs (city search task): one script, one set of definitions.

ExperimentRecorder writes every run into the experiment folder (Unity's persistentDataPath, see --root).
Runs are analysed in *tests*, one folder per test series, so the numbers and figures of one series
never mix with another's:

    experiment/
      <stem>_*.csv, <stem>_session.json   new runs, ungrouped until a test is made from them
      internal_2/                         one test: its runs, test.json, answers.csv, editor logs
        results/                          derived tables: runs.csv, legs.csv, goals.csv, crashes.csv
      archive/                            runs that belong to no test (bench runs, false starts)
      plots/internal_2/                   the test's figures

A run stem is PID_tN_Condition_YYYYMMDD_HHMMSS.

Workflow for a new test series:
    python analyse.py status                            # what is ungrouped at the root
    python analyse.py archive AABD_t2_Swarm_20261001_101500   # a false start, if any
    python analyse.py group internal_3 AABD AABE        # move the runs in, write test.json
    ... check internal_3/test.json: its practice list is pre-filled ...
    python analyse.py answers template internal_3       # only if the answers were noted by hand
    python analyse.py answers apply internal_3
    python analyse.py run internal_3

test.json
    {"name": "internal_3", "created": "2026-10-01", "description": "", "practice": ["AABD_t1", "AABD_t4"]}
Each practice entry names runs by prefix, case-insensitively and in whole '_'-separated fields: "AABD_t1"
matches AABD_t1_Swarm_20261001_101500 but not AABD_t10_... . `group` pre-fills the list with each
participant's first trial in each condition. Practice runs stay in every table (the `practice` column) and
are left out of every figure and summary; --include-practice puts them back and writes the figures to
plots/<test>/with_practice/ instead.

Phases -- one definition, used by every metric (--zone, default 15 m). A goal's zone is its tile (a square
of half-width TILE_HALF about the goal patch centre: the block plus half of each surrounding street) grown
by `zone` metres, so 15 m takes in the whole street around the patch, up to the opposite kerb. Position is
the centroid of the alive drones (the drone itself in SingleDrone runs). Goals are ordered by first entry
into their zone. *Transit* leg k runs from the last exit of zone k-1 before that entry (the session start
for k = 1, so it includes take-off) to the first entry of zone k; *search* k runs from that entry to the
last exit of zone k before zone k+1 is entered (for the last goal, its last exit). Leaving a zone and coming
back is still searching it. A goal whose zone is never entered has no phases and is reported as unreached.

Each metric family is documented in its own section below. Subcommands: status, group, archive, answers,
run, maps, quicklook -- `python analyse.py <command> -h` for each.
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import shutil

import numpy as np
import pandas as pd

# ============================================================================= constants

DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "AppData", "LocalLow", "UAVS@BERKELEY", "DroneSim",
                            "experiment")
HERE = os.path.dirname(os.path.abspath(__file__))
OBSTACLES = os.path.join(HERE, "city_obstacles_ScaledCityWorld.json")
MANIFEST = "test.json"
ARCHIVE, PLOTS, RESULTS = "archive", "plots", "results"

STEM_RE = re.compile(r"^([A-Za-z0-9]+)_t(\d+)_([A-Za-z]+)_(\d{8})_(\d{6})$")
RUN_FILE_RE = re.compile(r"^([A-Za-z0-9]+_t\d+_[A-Za-z]+_\d{8}_\d{6})_[A-Za-z]+\.(csv|json)$")

MAX_SPEED = 9.31          # VelocityControl.maxSpeed (DroneReduced prefab, both conditions)
TILE_HALF = 90.83 / 2     # goal patch = one city tile (the 90.83 m grid pitch)
AXES = ["inPitch", "inRoll", "inYaw", "inThrottle"]
AXIS_NAMES = ["pitch", "roll", "yaw", "throttle"]
CONDS = ["SingleDrone", "Swarm"]
COLOURS = {"SingleDrone": "#d1603d", "Swarm": "#3d7dd1"}
SPREAD_RANGE = (0.4, 1.6)       # joystick dial -> d_ref
PITCH_RANGE = (-90.0, 60.0)     # FPVCameraScript.MinPitch / MaxPitch

DEFAULTS = dict(zone=15.0, deadband=0.1, tol=0.2, settle=0.5, spread_tol=0.1, pitch_tol=10.0, down=-30.0,
                range=40.0, dists=[0.0, 10.0, 15.0, 20.0, 30.0, 40.0], arrow_s=4.0)

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print(f"[warn] {msg}")


def print_warnings():
    if WARNINGS:
        print(f"\n{len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"  - {w}")


def attempt(family, stem, fn, *args):
    """fn(*args), or None with a warning: one run's broken input costs that run's metric, not the analysis."""
    try:
        return fn(*args)
    except Exception as e:     # noqa: BLE001 -- reported, and listed again at the end
        warn(f"{stem}: {family} skipped ({type(e).__name__}: {e})")
        return None


# ============================================================================= tests and runs on disk

def parse_stem(stem):
    m = STEM_RE.match(stem)
    if not m:
        return None
    return dict(pid=m.group(1).upper(), trial=int(m.group(2)), condition=m.group(3), date=m.group(4),
                timestamp=f"{m.group(4)}_{m.group(5)}")


def run_files(folder):
    """{stem: [file names]} for the run files directly in folder (subfolders are not searched)."""
    out = {}
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            m = RUN_FILE_RE.match(f)
            if m and os.path.isfile(os.path.join(folder, f)):
                out.setdefault(m.group(1), []).append(f)
    return out


def matches(stem, selector):
    """selector names stem: equal to it, or a prefix of it ending at a '_' (case-insensitive)."""
    s, x = stem.upper(), selector.strip().upper()
    return s == x or s.startswith(x + "_")


def read_manifest(folder):
    p = os.path.join(folder, MANIFEST)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def write_manifest(folder, m):
    with open(os.path.join(folder, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(m, indent=2) + "\n")


def practice_prefill(stems):
    """Each participant's first trial in each condition, as 'PID_tN'."""
    first = {}
    for s in stems:
        p = parse_stem(s)
        key = (p["pid"], p["condition"])
        first[key] = min(first.get(key, p["trial"]), p["trial"])
    return [f"{pid}_t{t}" for pid, t in sorted({(pid, t) for (pid, _), t in first.items()})]


def test_folder(root, test):
    folder = os.path.join(root, test)
    if read_manifest(folder) is None:
        raise SystemExit(f"{folder} is not a test (no {MANIFEST}); `analyse.py status` lists the tests")
    return folder


def hat_name(g):
    return g.get("hat", "").replace("Walker", "")


def read_events(path):
    """[(t, eventType, note)]; the note may itself contain ';'."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(";", 6)
            if len(parts) >= 3:
                rows.append((float(parts[0]), parts[2], parts[6] if len(parts) > 6 else ""))
    return rows


class Run:
    """One recorded run with every file read once. Optional files that are missing are None."""

    def __init__(self, folder, stem, practice=False):
        self.folder, self.stem, self.practice = folder, stem, practice
        p = parse_stem(stem)
        self.pid, self.trial, self.date, self.timestamp = p["pid"], p["trial"], p["date"], p["timestamp"]
        with open(self.path("session.json"), encoding="utf-8") as f:
            self.session = json.load(f)
        self.condition = self.session["condition"]
        self.duration = float(self.session["durationSec"])
        self.goals = self.session.get("goals", [])
        self.head = self._csv("head")
        self.drones = self._csv("drones")
        self.shape = self._csv("shape")
        self.walkers = self._csv("walkers")
        self.events = read_events(self.path("events.csv"))
        if self.drones is None:
            raise ValueError("no _drones.csv")
        self.alive = self.drones[self.drones.alive == 1]
        # Centroid of the alive drones on the drone-log clock; a sample stands for the time until the next.
        self.centroid = self.alive.groupby("t")[["gtX", "gtZ"]].mean()
        self.ct = self.centroid.index.to_numpy()
        self.cx, self.cz = self.centroid.gtX.to_numpy(), self.centroid.gtZ.to_numpy()
        self.cdt = np.minimum(np.diff(np.append(self.ct, self.duration)), 0.5)

    def path(self, suffix):
        return os.path.join(self.folder, f"{self.stem}_{suffix}")

    def _csv(self, name):
        p = self.path(name + ".csv")
        return pd.read_csv(p, sep=";") if os.path.exists(p) else None

    def ident(self):
        return dict(run=self.stem, pid=self.pid, trial=self.trial, condition=self.condition, date=self.date,
                    practice=self.practice)


def load_test(root, test):
    """(folder, manifest, [Run]) for every finalized run in the test, practice flags applied."""
    folder = test_folder(root, test)
    manifest = read_manifest(folder)
    entries = manifest.get("practice", [])
    files = run_files(folder)
    stems = [s for s in files if f"{s}_session.json" in files[s]]
    for s in sorted(set(files) - set(stems)):
        warn(f"{s}: no _session.json (aborted before finalize); not analysed")
    for e in entries:
        if not any(matches(s, e) for s in stems):
            warn(f"{MANIFEST}: practice entry {e!r} matches no run")
    runs = []
    for s in stems:
        r = attempt("loading", s, Run, folder, s, any(matches(s, e) for e in entries))
        if r is not None:
            runs.append(r)
    runs.sort(key=lambda r: (r.pid, r.trial, r.timestamp))
    seen = {}
    for r in runs:
        seen.setdefault((r.pid, r.trial), []).append(r.stem)
    for (pid, trial), group in seen.items():
        if len(group) > 1:
            warn(f"{pid} t{trial} has {len(group)} runs ({', '.join(group)}); all are analysed")
    return folder, manifest, runs


# ============================================================================= phases

def tile_distance(x, z, g):
    """Horizontal distance from (x, z) to goal g's tile, 0 anywhere over it."""
    ex = np.maximum(np.abs(x - g["goalX"]) - TILE_HALF, 0)
    ez = np.maximum(np.abs(z - g["goalZ"]) - TILE_HALF, 0)
    return np.hypot(ex, ez)


def goal_legs(run, zone):
    """Legs in zone order, dict(leg, gi, goal, t0, enter, exit): transit is [t0, enter), search [enter, exit)."""
    inside = [tile_distance(run.cx, run.cz, g) <= zone for g in run.goals]
    entry = [int(np.argmax(zn)) if zn.any() else None for zn in inside]
    order = sorted((k for k, e in enumerate(entry) if e is not None), key=lambda k: entry[k])
    legs, start = [], float(run.ct[0])
    for pos, k in enumerate(order):
        stop = entry[order[pos + 1]] if pos + 1 < len(order) else len(run.ct)
        j = np.flatnonzero(inside[k][:stop])[-1]
        exit_ = float(run.ct[j] + run.cdt[j])
        legs.append(dict(leg=pos + 1, gi=k, goal=run.goals[k], t0=start, enter=float(run.ct[entry[k]]),
                         exit=exit_))
        start = exit_
    return legs


def pos_at(c, t):
    i = min(np.searchsorted(c.index.values, t), len(c) - 1)
    return c.gtX.values[i], c.gtZ.values[i]


# ============================================================================= task time
#
# totalTaskTime is set by ExperimentRecorder at Finalize as start -> last identify keypress, so it only
# means something when the experimenter pressed the identify keys live. A run whose answers were
# back-filled by `answers apply` never gets one (it stays 0, as does an aborted run), so each run uses
# totalTaskTime when it is > 0 and falls back to durationSec (start -> Finalize); time_source says which.

def task_family(run):
    total = float(run.session.get("totalTaskTime", 0.0) or 0.0)
    time_s, source = (total, "identify") if total > 0 else (run.duration, "duration")
    return dict(time_s=time_s, time_source=source, totalTaskTime=total)


# ============================================================================= pilot commands
#
# How *the pilot* flew, not how the drones moved: did they hold one command and let the vehicles sort out
# the obstacles, or keep re-steering around them? Computed from the stick inputs in _head.csv (inPitch /
# inRoll / inYaw / inThrottle, ~10 Hz), with the trajectory used only for the phases and the achieved path.
#
# Command frame: pitch/roll are a normalised horizontal velocity, magnitude-limited to 1, rotated about
# world-up by the pilot body yaw and scaled by VelocityControl.maxSpeed:
#     v_cmd = maxSpeed * R_y(bodyYaw) (roll, 0, pitch)
# which matches the achieved centroid velocity at r = 0.92-0.97 with ~1 s lag on every AABB/AABC run.
#
# Per transit leg, then summed/averaged per run ("_all" = the whole run):
#   adjustments       discrete manual adjustments. The 4 flight axes are dead-banded (|u| < --deadband -> 0)
#                     and an adjustment starts whenever any axis leaves the last *held* position by more
#                     than --tol; the hold re-latches once the stick has stayed within tol/2 for --settle s.
#                     Push forward and hold = 1; release = 1; constant re-steering = many. Also by axis.
#   adj_per_min       adjustments per minute of transit.
#   adj_per_100m      adjustments per 100 m of straight-line progress (leg start -> the goal's zone).
#   mean_hold_s       mean time between adjustments (how long one command was held).
#   stick_tv_per_min  total variation of the stick vector, sum |du|, per minute: control activity,
#                     insensitive to the thresholds above.
#   yaw_duty / lat_duty  fraction of transit with yaw / roll stick outside the deadband.
#   course_turn_per_min  total |change| of the commanded course (direction of v_cmd, only while
#                     |v_cmd| > 0.3 maxSpeed), degrees per minute: how much the pilot steered, by yaw or roll.
#   cmd_straightness  |integral of v_cmd| / integral of |v_cmd|: 1 = one heading held for the leg.
#   cmd_path_ratio    commanded path length / straight-line leg distance.
#   act_path_ratio    achieved centroid path length / straight-line leg distance (the swarm can bend its
#                     own path around buildings without the pilot commanding it).

def deadband(u, db):
    return np.where(np.abs(u) < db, 0.0, u)


def detect_adjustments(t, U, tol, settle_s):
    """Indices where a manual adjustment starts, plus which axes moved in each.

    U is (n, k) dead-banded stick. Hysteresis on the held position: an adjustment starts when any axis
    departs the latched hold by more than tol; during the move the hold is unlatched, and it is re-latched
    once the stick has stayed within tol/2 of one position for settle_s.
    """
    starts, axes_moved = [], []
    hold = U[0].copy()
    moving, move_axes = False, np.zeros(U.shape[1], bool)
    anchor, anchor_t = U[0].copy(), t[0]
    for i in range(1, len(t)):
        u = U[i]
        if not moving:
            dev = np.abs(u - hold)
            if dev.max() > tol:
                moving, move_axes = True, dev > tol
                starts.append(i)
                anchor, anchor_t = u.copy(), t[i]
        else:
            if np.abs(u - anchor).max() > tol / 2:
                move_axes |= np.abs(u - hold) > tol
                anchor, anchor_t = u.copy(), t[i]
            elif t[i] - anchor_t >= settle_s:
                hold, moving = anchor.copy(), False
                axes_moved.append(move_axes.copy())
    if moving:
        axes_moved.append(move_axes.copy())
    return np.array(starts, int), np.array(axes_moved, bool).reshape(-1, U.shape[1])


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def command_velocity(roll, pitch, body_yaw_deg):
    u = np.c_[roll, pitch]
    m = np.linalg.norm(u, axis=1)
    u[m > 1] /= m[m > 1, None]
    psi = np.radians(body_yaw_deg)
    # Unity Quaternion.Euler(0, psi, 0) * (x, 0, z)
    vx = MAX_SPEED * (u[:, 0] * np.cos(psi) + u[:, 1] * np.sin(psi))
    vz = MAX_SPEED * (-u[:, 0] * np.sin(psi) + u[:, 1] * np.cos(psi))
    return vx, vz


def stick_command(run, db):
    """(t, dead-banded stick (n, 4), vx, vz) on the head-log clock."""
    h = run.head
    U = deadband(h[AXES].values.astype(float), db)
    vx, vz = command_velocity(U[:, 1], U[:, 0], h.bodyYaw.values)
    return h.t.values, U, vx, vz


def window_metrics(t, U, vx, vz, adj_starts, adj_axes, t0, t1):
    """Metrics for samples t0 <= t < t1."""
    sel = (t >= t0) & (t < t1)
    if sel.sum() < 3:
        return None
    ts, Us, vxs, vzs = t[sel], U[sel], vx[sel], vz[sel]
    dt = np.diff(ts, append=ts[-1] + np.median(np.diff(ts)))
    dur = float(dt.sum())
    in_win = (t[adj_starts] >= t0) & (t[adj_starts] < t1) if len(adj_starts) else np.zeros(0, bool)
    n_adj = int(in_win.sum())
    ax_counts = adj_axes[in_win[: len(adj_axes)]].sum(axis=0) if len(adj_axes) else np.zeros(4)

    spd = np.hypot(vxs, vzs)
    cmd_len = float((spd * dt).sum())
    disp = float(np.hypot((vxs * dt).sum(), (vzs * dt).sum()))
    course = np.degrees(np.arctan2(vxs, vzs))
    fast = spd > 0.3 * MAX_SPEED
    both = fast[1:] & fast[:-1]
    turn = float(np.abs(wrap_deg(np.diff(course)))[both].sum())

    return dict(
        dur_s=dur,
        adjustments=n_adj,
        **{f"adj_{a}": int(n) for a, n in zip(AXIS_NAMES, ax_counts)},
        adj_per_min=n_adj / dur * 60.0,
        mean_hold_s=dur / max(n_adj, 1),
        stick_tv_per_min=float(np.abs(np.diff(Us, axis=0)).sum()) / dur * 60.0,
        yaw_duty=float((dt * (Us[:, 2] != 0)).sum() / dur),
        lat_duty=float((dt * (Us[:, 1] != 0)).sum() / dur),
        cmd_len_m=cmd_len,
        cmd_disp_m=disp,
        cmd_straightness=disp / cmd_len if cmd_len > 0 else np.nan,
        course_turn_deg=turn,
        course_turn_per_min=turn / dur * 60.0,
    )


def achieved_len(c, t0, t1):
    w = c[(c.index >= t0) & (c.index <= t1)]
    return float(np.hypot(np.diff(w.gtX), np.diff(w.gtZ)).sum()) if len(w) > 1 else 0.0


def command_family(run, legs, cfg):
    """(run-level dict, [leg rows])."""
    c = run.centroid
    t, U, vx, vz = stick_command(run, cfg.deadband)
    starts, axes = detect_adjustments(t, U, cfg.tol, cfg.settle)
    rows = []
    for L in legs:
        g, key = L["goal"], dict(run=run.stem, leg=L["leg"], hat=hat_name(L["goal"]))
        m = window_metrics(t, U, vx, vz, starts, axes, L["t0"], L["enter"])
        if m is not None:
            x0, z0 = pos_at(c, L["t0"])
            straight = max(float(tile_distance(x0, z0, g)) - cfg.zone, 0.0)
            act = achieved_len(c, L["t0"], L["enter"])
            m.update(key, phase="transit", straight_m=straight, act_len_m=act,
                     adj_per_100m=m["adjustments"] / straight * 100.0 if straight > 5 else np.nan,
                     cmd_path_ratio=m["cmd_len_m"] / straight if straight > 5 else np.nan,
                     act_path_ratio=act / straight if straight > 5 else np.nan)
            rows.append(m)
        m = window_metrics(t, U, vx, vz, starts, axes, L["enter"], max(L["exit"], L["enter"] + 1e-3))
        if m is not None:
            m.update(key, phase="search")
            rows.append(m)

    ph = pd.DataFrame(rows)
    tr = ph[ph.phase == "transit"] if len(ph) else ph
    se = ph[ph.phase == "search"] if len(ph) else ph
    out = dict(adj_per_min_all=len(starts) / (t[-1] - t[0]) * 60.0)
    if not len(tr):
        return out, rows
    dur = tr.dur_s.sum()
    out.update(
        transit_s=dur,
        search_s=se.dur_s.sum() if len(se) else 0.0,
        adjustments=int(tr.adjustments.sum()),
        **{f"adj_{a}": int(tr[f"adj_{a}"].sum()) for a in AXIS_NAMES},
        adj_per_min=tr.adjustments.sum() / dur * 60.0,
        adj_per_100m=tr.adjustments.sum() / tr.straight_m.sum() * 100.0,
        mean_hold_s=dur / max(tr.adjustments.sum(), 1),
        stick_tv_per_min=(tr.stick_tv_per_min * tr.dur_s).sum() / dur,
        yaw_duty=(tr.yaw_duty * tr.dur_s).sum() / dur,
        lat_duty=(tr.lat_duty * tr.dur_s).sum() / dur,
        course_turn_per_min=tr.course_turn_deg.sum() / dur * 60.0,
        cmd_straightness=tr.cmd_disp_m.sum() / tr.cmd_len_m.sum(),   # length-weighted mean over the legs
        straight_m=tr.straight_m.sum(),
        cmd_path_ratio=tr.cmd_len_m.sum() / tr.straight_m.sum(),
        act_path_ratio=tr.act_len_m.sum() / tr.straight_m.sum(),
    )
    return out, rows


# ============================================================================= spread and pitch dials
#
# The two *setting* dials on the joystick: spread (how far apart the drones fly) and gimbal (every drone's
# FPV camera tilt). Both are position controls, not springs -- the value stays where the pilot left it --
# so what matters is when and how far a dial was turned and what it was left at. The question is whether
# pilots re-configure for the phase: widen the swarm and tilt the camera down to search, undo it to travel.
#
# Signals (~10 Hz):
#   spread   _head.csv inSpread: the dial value, written straight into the Olfati-Saber d_ref, 0.4 tightest
#            .. 1.6 widest. -1 means "no override": it appears before the first joystick packet and as
#            single-sample dropouts, where the swarm keeps the last value, so it is forward-filled and the
#            leading gap back-filled with the first transmitted value.
#   spacing  _shape.csv meanNNm: the achieved mean nearest-neighbour distance (m), the formation's answer to
#            the spread command, lagging it by seconds. Needs >= 2 alive drones: NaN for SingleDrone. (The
#            dial is still scored there; it does nothing to a lone drone, so any use of it is the pilot's.)
#   pitch    _head.csv gimbalPitch: FPVCameraScript.SharedPitch in degrees, 0 level, -90 straight down,
#            +60 up. Only logged from 2026-09-24 on: earlier runs have NaN for every pitch metric.
#
# Per run, for x in {spread, pitch} (d_ref for spread, degrees for pitch):
#   x_adj                  dial adjustments over the whole run, with the pilot-command hold/settle detector
#                          on the one axis (--spread-tol / --pitch-tol, --settle). Tight to wide = 1.
#   x_adj_per_min[_transit|_search]   ... per minute of the run / of each phase.
#   x_tv_per_min           total variation (sum |dx|) per minute: how far the dial was turned in all.
#   x_transit / x_search   time-weighted mean setting over all transit / all search time.
#   x_search_delta         x_search - x_transit: how much the pilot re-set the dial for searching.
#   spacing_transit_m / spacing_search_m / spacing_search_delta_m   the same, for the achieved spacing.
#   pitch_down_frac        fraction of the run with the camera below --down degrees.

DIALS = ["spread", "pitch"]


def spread_signal(h, sh):
    """Commanded d_ref on the head-log clock, -1 'no override' samples filled."""
    if "inSpread" in h:
        sp = h.inSpread.where(h.inSpread > 0)
        if sp.notna().any():
            return sp.ffill().bfill().values.astype(float)
    if sh is not None:           # dial never transmitted: the value in force is the scene's d_ref
        return np.interp(h.t.values, sh.t.values, sh.dRef.values)
    return np.full(len(h), np.nan)


def spacing_signal(h, sh):
    """Achieved mean nearest-neighbour distance (m), NaN wherever fewer than 2 drones are alive."""
    if sh is None:
        return np.full(len(h), np.nan)
    nn = np.where(sh.nAlive.values >= 2, sh.meanNNm.values, np.nan)
    i = np.clip(np.searchsorted(sh.t.values, h.t.values), 0, len(sh) - 1)
    return nn[i]


def pitch_signal(h):
    if "gimbalPitch" in h:
        return h.gimbalPitch.values.astype(float)
    return np.full(len(h), np.nan)


def dial_window(t, dt, sig, starts, t0, t1, down):
    """Stats for samples t0 <= t < t1; sig maps name -> signal, starts maps dial -> adjustment idx."""
    sel = (t >= t0) & (t < t1)
    if sel.sum() < 3:
        return None
    w = dt[sel]
    m = dict(dur_s=float(w.sum()))
    for name, x in sig.items():
        xs = x[sel]
        ok = ~np.isnan(xs)
        m[name] = float((xs[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        if name in starts:
            idx = starts[name]
            m[f"{name}_adj"] = int(sel[idx].sum()) if ok.any() else np.nan
            m[f"{name}_tv"] = float(np.abs(np.diff(xs[ok])).sum()) if ok.sum() > 1 else np.nan
    p = sig["pitch"][sel]
    m["pitch_down_s"] = float(w[p < down].sum()) if not np.isnan(p).all() else np.nan
    return m


def dial_family(run, legs, cfg):
    """(run-level dict, [leg rows], timeline series)."""
    h, sh = run.head, run.shape
    t = h.t.values
    dt = np.diff(t, append=t[-1] + np.median(np.diff(t)))
    sig = {"spread": spread_signal(h, sh), "spacing": spacing_signal(h, sh), "pitch": pitch_signal(h)}
    tols = {"spread": cfg.spread_tol, "pitch": cfg.pitch_tol}
    starts = {}
    for name in DIALS:
        x = sig[name]
        starts[name] = (detect_adjustments(t, x[:, None], tols[name], cfg.settle)[0]
                        if not np.isnan(x).any() else np.zeros(0, int))

    rows = []
    for L in legs:
        for phase, t0, t1 in (("transit", L["t0"], L["enter"]),
                              ("search", L["enter"], max(L["exit"], L["enter"] + 1e-3))):
            m = dial_window(t, dt, sig, starts, t0, t1, cfg.down)
            if m is not None:
                m.update(run=run.stem, leg=L["leg"], hat=hat_name(L["goal"]), phase=phase)
                rows.append(m)
    ph_all = pd.DataFrame(rows)

    dur = float(dt.sum())
    out = dict(pitch_logged="gimbalPitch" in h)
    for phase in ("transit", "search"):
        ph = ph_all[ph_all.phase == phase] if len(ph_all) else ph_all
        phase_s = ph.dur_s.sum() if len(ph) else 0.0
        for name in ("spread", "spacing", "pitch"):
            ok = ph[name].notna() if len(ph) else []
            out[f"{name}_{phase}"] = ((ph[name][ok] * ph.dur_s[ok]).sum() / ph.dur_s[ok].sum()
                                      if len(ph) and ok.any() else np.nan)
        for name in DIALS:
            n = ph[f"{name}_adj"].sum(min_count=1) if len(ph) else np.nan
            out[f"{name}_adj_per_min_{phase}"] = n / phase_s * 60.0 if phase_s > 0 else np.nan
    for name in DIALS:
        x = sig[name]
        logged = not np.isnan(x).all()
        out[f"{name}_adj"] = len(starts[name]) if logged else np.nan
        out[f"{name}_adj_per_min"] = len(starts[name]) / dur * 60.0 if logged else np.nan
        out[f"{name}_tv_per_min"] = float(np.abs(np.diff(x)).sum()) / dur * 60.0 if logged else np.nan
        out[f"{name}_search_delta"] = out[f"{name}_search"] - out[f"{name}_transit"]
    out["spacing_search_delta_m"] = out["spacing_search"] - out["spacing_transit"]
    out["spacing_transit_m"] = out.pop("spacing_transit")
    out["spacing_search_m"] = out.pop("spacing_search")
    p = sig["pitch"]
    out["pitch_down_frac"] = float(dt[p < cfg.down].sum() / dur) if not np.isnan(p).all() else np.nan
    for r in rows:
        r.pop("dur_s")        # the pilot-command family's dur_s is the one in legs.csv
    series = dict(t=t, **sig, visits=[(L["enter"], L["exit"]) for L in legs])
    return out, rows, series


# ============================================================================= goal proximity
#
# Time near each goal patch for a sweep of "near" distances (--dist): near{D}Sec is the total time the
# centroid was within D m of the goal's tile, so D = 0 is time over the tile and D = zone is time in the
# goal's zone. Independent of the phases. transitSec / searchSec are that goal's phases (above).

def proximity_family(run, legs, cfg):
    by_gi = {L["gi"]: L for L in legs}
    rows = []
    for gi, g in enumerate(run.goals):
        d = tile_distance(run.cx, run.cz, g)
        row = dict(run=run.stem, goalIndex=g.get("goalIndex", gi), minDistToTileM=round(float(d.min()), 1))
        for D in cfg.dists:
            row[f"near{D:g}Sec"] = round(float(run.cdt[d <= D].sum()), 1)
        L = by_gi.get(gi)
        row.update(reached=L is not None,
                   zoneOrder=L["leg"] if L else np.nan,
                   transitSec=round(L["enter"] - L["t0"], 1) if L else np.nan,
                   searchSec=round(L["exit"] - L["enter"], 1) if L else np.nan)
        rows.append(row)
    return rows


# ============================================================================= hat visibility
#
# How long each goal's hat walker was observable in the drone feeds. Being over the tile is the wrong test
# (hats were often identified on the approach, from outside it), so a goal is *in view* at a sample when at
# least one alive drone passes all three tests:
#   1. range   camera-to-hat distance <= --range (40 m). Calibrated on the data: five hats were identified
#              although they never came closer than 27-34 m, so the true limit is >= ~35 m; at 40 m the
#              walker is ~34 px tall in the 1152x648 feed and the hat ~6 px across.
#   2. heading the hat is inside the camera's horizontal field of view (74.4 deg: the DJI Mini 3 Pro's
#              82.1 deg diagonal at 16:9, as ScreenSpawn sets it).
#   3. sight   the line from the camera to the hat misses every building: the city's colliders
#              (city_obstacles_ScaledCityWorld.json, from Tools/Swarm/Export city obstacles) minus the
#              tiles the goals replaced, plus each goal patch's own nine buildings.
# The gimbal pitch is not tested (the camera is assumed pitched to where the walker is); levelCamPct is the
# share of the in-view time a level gimbal would also have covered. Head direction is not used: this is time
# the hat was on a feed, not on the pilot's eye.
# Per goal: inViewSec (summed in-view time), firstSightSec, observation episodes (in-view samples joined
# across gaps <= GAP_S) and the longest one's span, hatMinDistM (closest camera while in view),
# meanDronesInView, occludedSec (in range and heading but blocked, and no other drone saw it).

ASPECT = 16 / 9
VFOV_HALF = math.atan(math.tan(math.radians(82.1 / 2)) / math.sqrt(ASPECT ** 2 + 1))
HFOV_HALF = math.atan(math.tan(VFOV_HALF) * ASPECT)
GAP_S = 5.0          # s, gaps up to this long do not split an episode
HAT_HEIGHT = 1.6     # m above the walker's logged position (0.2 m up) -> hat centre ~1.8 m above ground
# Goal patch buildings (goal_patch_Scaled.prefab, goal-root frame): x0, x1, z0, z1; all 0.2..50.2 tall.
# StreetWidthTuner (blockScale 0.8) moves them about the kerb at runtime; sizes are unchanged.
GOAL_PILLARS = [
    (2.0, 4.7, 27.2, 30.2), (-31.2, -29.2, 26.8, 29.8), (2.0, 4.7, -31.4, -28.4), (29.8, 31.8, 26.8, 29.8),
    (-31.3, -28.3, 0.2, 2.9), (-31.2, -29.2, -31.8, -28.8), (29.8, 31.8, -31.8, -28.8), (28.9, 31.9, 0.2, 2.9),
    (-1.4, 2.1, -1.5, 1.3)]
GOAL_KERB_X = 0.36
BLOCK_SCALE = 0.8
BUILDING_TOP = 50.2

_CITY = None


def load_city():
    global _CITY
    if _CITY is None:
        with open(OBSTACLES, encoding="utf-8") as f:
            d = json.load(f)
        boxes = [(b["tile"], np.array(b["c"]), np.array(b["ax"], dtype=float)) for b in d["boxes"]]
        kerbs = {t["name"]: np.array(t["kerb"]) for t in d["tiles"] if t["kerb"]}
        _CITY = boxes, kerbs
    return _CITY


def pillar_boxes(gx, gz):
    out = []
    for x0, x1, z0, z1 in GOAL_PILLARS:
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        cx = GOAL_KERB_X + BLOCK_SCALE * (cx - GOAL_KERB_X)
        cz = BLOCK_SCALE * cz
        c = np.array([gx + cx, (0.2 + BUILDING_TOP) / 2, gz + cz])
        ax = np.diag([(x1 - x0) / 2, (BUILDING_TOP - 0.2) / 2, (z1 - z0) / 2])
        out.append(("goal", c, ax))
    return out


def scene_boxes(goals):
    """City buildings minus the tiles the goals replaced, plus each goal patch's own buildings."""
    city, kerbs = load_city()
    replaced = set()
    for g in goals:
        name, dist = min(((n, math.hypot(k[0] - GOAL_KERB_X - g["goalX"], k[2] - g["goalZ"]))
                          for n, k in kerbs.items()), key=lambda x: x[1])
        if dist > 2.0:
            raise ValueError(f"goal at ({g['goalX']:.1f}, {g['goalZ']:.1f}) matches no tile "
                             f"(nearest {name} {dist:.1f} m)")
        replaced.add(name)
    boxes = [b for b in city if b[0] not in replaced]
    for g in goals:
        boxes += pillar_boxes(g["goalX"], g["goalZ"])
    return boxes


def segment_blocked(p0, p1, boxes):
    """p0, p1: (N, 3). True where the segment p0->p1 passes through any oriented box."""
    blocked = np.zeros(len(p0), dtype=bool)
    d = p1 - p0
    for _, c, ax in boxes:
        lens = np.linalg.norm(ax, axis=1)
        u = ax / lens[:, None]                      # box axes (unit), rows
        o = (p0 - c) @ u.T                          # segment start in box frame
        v = d @ u.T
        t0 = np.zeros(len(p0))
        t1 = np.ones(len(p0))
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in range(3):
                a = (-lens[k] - o[:, k]) / v[:, k]
                b = (lens[k] - o[:, k]) / v[:, k]
                lo = np.minimum(a, b)
                hi = np.maximum(a, b)
                par = v[:, k] == 0
                inside = np.abs(o[:, k]) <= lens[k]
                lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
                hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
                t0 = np.maximum(t0, lo)
                t1 = np.minimum(t1, hi)
        blocked |= t0 <= t1
    return blocked


def episodes(t, flag, dt):
    """[(start, end, inViewSec)] joining in-view samples across gaps <= GAP_S."""
    out = []
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return out
    s = prev = idx[0]
    acc = dt[idx[0]]
    for i in idx[1:]:
        if t[i] - t[prev] - dt[prev] > GAP_S:
            out.append((t[s], t[prev] + dt[prev], acc))
            s = i
            acc = 0.0
        acc += dt[i]
        prev = i
    out.append((t[s], t[prev] + dt[prev], acc))
    return out


def hat_family(run, cfg):
    if run.walkers is None:
        raise ValueError("no _walkers.csv")
    dr, wk = run.alive, run.walkers
    ts, dt = run.ct, run.cdt
    boxes = scene_boxes(run.goals)
    r_id = cfg.range
    rows = []
    for gi, g in enumerate(run.goals):
        w = wk[wk.goalIndex == g["goalIndex"]][["t", "specialX", "specialY", "specialZ"]]
        m = dr.merge(w, on="t")
        cam = m[["gtX", "gtY", "gtZ"]].to_numpy()
        hat = m[["specialX", "specialY", "specialZ"]].to_numpy() + [0, HAT_HEIGHT, 0]
        rel = hat - cam
        dist = np.linalg.norm(rel, axis=1)
        hd = np.hypot(rel[:, 0], rel[:, 2])
        off = (np.degrees(np.arctan2(rel[:, 0], rel[:, 2])) - m.yawDeg.to_numpy() + 180) % 360 - 180
        dep = np.degrees(np.arctan2(-rel[:, 1], hd))
        cand = (dist <= r_id) & (np.abs(off) <= math.degrees(HFOV_HALF))
        vis = cand.copy()
        if cand.any():
            near = [b for b in boxes if np.linalg.norm(b[1][[0, 2]] - hat[cand][:, [0, 2]].mean(0)) < r_id + 80]
            vis[cand] = ~segment_blocked(cam[cand], hat[cand], near)
        m = m.assign(vis=vis, level=vis & (np.abs(dep) <= math.degrees(VFOV_HALF)),
                     dmin=np.where(vis, dist, np.nan), blocked=cand & ~vis)
        per_t = m.groupby("t").agg(vis=("vis", "any"), level=("level", "any"), n=("vis", "sum"),
                                   blocked=("blocked", "any"), dmin=("dmin", "min"))
        per_t = per_t.reindex(ts).fillna({"vis": False, "level": False, "n": 0, "blocked": False})
        flag = per_t.vis.to_numpy(bool)
        eps = episodes(ts, flag, dt)
        in_view = float(dt[flag].sum())
        blocked_only = float(dt[per_t.blocked.to_numpy(bool) & ~flag].sum())
        level = float(dt[per_t.level.to_numpy(bool)].sum())
        longest = max(eps, key=lambda e: e[1] - e[0]) if eps else None
        rows.append(dict(
            run=run.stem, goalIndex=g.get("goalIndex", gi),
            firstSightSec=round(eps[0][0], 1) if eps else None,
            inViewSec=round(in_view, 1),
            nEpisodes=len(eps),
            episodeSec=round(sum(e[1] - e[0] for e in eps), 1),
            longestStartSec=round(longest[0], 1) if longest else None,
            longestEndSec=round(longest[1], 1) if longest else None,
            longestSec=round(longest[1] - longest[0], 1) if longest else None,
            hatMinDistM=round(float(np.nanmin(per_t.dmin.to_numpy(float))), 1) if flag.any() else None,
            meanDronesInView=round(float(per_t.n[flag].mean()), 1) if flag.any() else 0,
            levelCamPct=round(100 * level / in_view) if in_view else None,
            occludedSec=round(blocked_only, 1),
            episodes=" | ".join(f"{a:.0f}-{b:.0f} ({v:.0f}s)" for a, b, v in eps)))
    return rows


# ============================================================================= editor logs
#
# Unity's Editor.log is the only record of two things for runs recorded before ExperimentRecorder logged
# them: panorama transitions (PyUniSharingFast's untimed "[Panorama] hidden/restored" lines) and why each
# drone died (DroneHealthMonitor's "Parking Drone N ... Reason: R ... t=XXs"). Unity overwrites
# Editor-prev.log on every editor start, so copy the log of a session day into the test folder
# (e.g. editor_log_YYYYMMDD.log) before restarting the editor; every *.log in the test folder is read.

LOG_START_RE = re.compile(r"^ExperimentRecorder: logging to .*\((\S+)_\*\.csv\)")
LOG_STEM_RE = re.compile(r"([A-Za-z0-9]+)_t(\d+)_([A-Za-z]+)_(\d{8}_\d{6})")
LOG_DEATH_RE = re.compile(r"^\[DroneHealthMonitor\] Parking .* t=([0-9.]+)s")


def hidden_reason(line):
    if "toggled off by pilot" in line:
        return "pilot"
    if "unspecified" in line:
        return "no_panorama"
    if "no overlap" in line:
        return "no_overlap"
    if "photometric" in line:
        return "photometric"
    return "other"


def parse_editor_logs(paths):
    """{(pid, 'YYYYMMDD_HHMMSS'): dict(items, deaths)}; items are ('T', on, reason) | ('A', t) | ('S', name),
    deaths are (droneId, reason, t)."""
    runs = {}
    for path in paths:
        key, rec = None, None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = LOG_START_RE.match(line)
                if m:
                    s = LOG_STEM_RE.search(m.group(1))
                    key, rec = (s.group(1).upper(), s.group(4)), dict(items=[], deaths=[])
                    continue
                if key is None:
                    continue
                if line.startswith("ExperimentRecorder: session finalized"):
                    runs.setdefault(key, rec)  # first copy wins if a log was archived twice
                    key = None
                elif line.startswith("[Panorama] hidden"):
                    rec["items"].append(("T", False, hidden_reason(line)))
                elif line.startswith("[Panorama] restored"):
                    rec["items"].append(("T", True, ""))
                elif line.startswith("PyUniSharingFast: stitcher switched to"):
                    rec["items"].append(("S", line.rsplit(" ", 1)[-1].strip(" .\n")))
                else:
                    d = LOG_DEATH_RE.match(line)
                    if d:
                        rec["items"].append(("A", float(d.group(1))))
                        i = re.search(r"Parking Drone (\d+)", line)
                        r = re.search(r"Reason: (\w+)", line)
                        rec["deaths"].append((int(i.group(1)) if i else -1, r.group(1) if r else "",
                                              float(d.group(1))))
    return runs


# ============================================================================= stitch visibility
#
# How long the stitched panorama was functional and visible in each swarm flight: the curved screen up,
# i.e. Python's quality verdict good and the pilot not having toggled it off. Two sources, best first:
# 1. ExperimentRecorder's stitch_on / stitch_off events (runs recorded since it gained them). Frame-timed,
#    and they also catch a *frozen* panorama (Python gone, screen still up), which is logged nowhere else.
# 2. The editor log (above). Its panorama lines have no timestamps, so a transition can only be placed
#    between the nearest *timed* lines around it -- the drone deaths, plus the run's start and end. Where
#    two timed lines have no transition between them the state is known exactly; where they have one or
#    more only bounds survive: visible time lies between known-on and known-on + unknown. The first
#    "hidden ... unspecified" of every run is its first frame (Python has not written a quality word yet),
#    so it is taken at t = 0.

def reconstruct(items, duration):
    """Segments (t0, t1, 'on' | 'off' | 'unknown', nTransitions) from an untimed transition list."""
    items = list(items)
    trans = [it for it in items if it[0] == "T"]
    state = True  # PyUniSharingFast starts with the panorama displayed ...
    if trans and not trans[0][1] and trans[0][2] == "no_panorama":
        state = False  # ... and hides it on the first frame
        items.remove(trans[0])
    segs, t_prev, pending = [], 0.0, 0
    for it in items + [("A", duration)]:
        if it[0] == "T":
            pending += 1
            state = it[1]
        elif it[0] == "A":
            t = min(max(it[1], t_prev), duration)
            if pending == 0:
                segs.append((t_prev, t, "on" if state else "off", 0))
            else:
                segs.append((t_prev, t, "unknown", pending))
            t_prev, pending = t, 0
    return segs


def segments_from_events(events, duration):
    segs, t_prev, on = [], 0.0, False
    for t, kind, _ in events:
        if kind not in ("stitch_on", "stitch_off"):
            continue
        segs.append((t_prev, t, "on" if on else "off", 0))
        t_prev, on = t, kind == "stitch_on"
    segs.append((t_prev, duration, "on" if on else "off", 0))
    return [s for s in segs if s[1] > s[0]]


def stitch_family(run, log_runs):
    """(row, segments, timed anchors) for a swarm run, None if neither source covers it."""
    n_ch = {}
    if any(k in ("stitch_on", "stitch_off") for _, k, _ in run.events):
        source, segs, anchors = "recorder", segments_from_events(run.events, run.duration), []
        for _, k, note in run.events:
            if k == "stitch_off":
                r = "pilot" if note == "pilot_off" else "stale" if note.startswith("stale") else \
                    hidden_reason(note)
                n_ch[r] = n_ch.get(r, 0) + 1
        n_on = sum(1 for _, k, _ in run.events if k == "stitch_on")
        switches = 0
    elif (run.pid, run.timestamp) in log_runs:
        source, items = "editor_log", log_runs[(run.pid, run.timestamp)]["items"]
        segs = reconstruct(items, run.duration)
        anchors = [it[1] for it in items if it[0] == "A" and 0 < it[1] < run.duration]
        for it in items:
            if it[0] == "T" and not it[1]:
                n_ch[it[2]] = n_ch.get(it[2], 0) + 1
        n_on = sum(1 for it in items if it[0] == "T" and it[1])
        switches = sum(1 for it in items if it[0] == "S")
    else:
        warn(f"{run.stem}: no stitch events and not in any editor log in the test folder; "
             f"stitch visibility skipped")
        return None
    on = sum(b - a for a, b, k, _ in segs if k == "on")
    off = sum(b - a for a, b, k, _ in segs if k == "off")
    unknown = sum(b - a for a, b, k, _ in segs if k == "unknown")
    row = dict(stitch_source=source, stitch_knownOnSec=round(on, 1), stitch_knownOffSec=round(off, 1),
               stitch_unknownSec=round(unknown, 1),
               stitch_visibleMinPct=round(100 * on / run.duration, 1),
               stitch_visibleMaxPct=round(100 * (on + unknown) / run.duration, 1),
               stitch_nOn=n_on, stitch_switches=switches)
    for r in ("photometric", "no_overlap", "pilot", "no_panorama", "stale", "other"):
        row[f"stitch_hidden_{r}"] = n_ch.get(r, 0)
    return row, segs, anchors


# ============================================================================= crashes
#
# A death is the last sample a drone was alive in _drones.csv; its reason (CollidingWithAnotherDrone,
# ExternallyKilled, ...) comes from the editor log when there is one. A drone-drone collision kills both
# drones on one tick, so deaths within 0.5 s of each other are one *crash* (one red line on the timelines).

def crash_family(run, log_runs, merge_s=0.5):
    """([death rows], [crash times])."""
    dr = run.drones[["t", "droneId", "alive", "gtX", "gtY", "gtZ"]].sort_values(["droneId", "t"])
    a, ids = dr.alive.values, dr.droneId.values
    died = np.flatnonzero((ids[1:] == ids[:-1]) & (a[:-1] == 1) & (a[1:] == 0))
    logged = log_runs.get((run.pid, run.timestamp), {}).get("deaths", [])
    rows = []
    for i in died:
        t, did = float(dr.t.values[i]), int(ids[i])
        cand = [(abs(lt - t), r) for lid, r, lt in logged if lid == did and abs(lt - t) <= 1.0]
        rows.append(dict(run=run.stem, droneId=did, t=t, x=float(dr.gtX.values[i]), y=float(dr.gtY.values[i]),
                         z=float(dr.gtZ.values[i]), reason=min(cand)[1] if cand else ""))
    crashes = []
    for x in np.sort(dr.t.values[died]):
        if not crashes or x - crashes[-1] > merge_s:
            crashes.append(float(x))
    return rows, crashes


# ============================================================================= figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class Figures:
    """Where a test's figures go, and the footer every one of them carries."""

    def __init__(self, root, test, cfg, sub=None):
        self.test = test
        self.dir = os.path.join(root, PLOTS, test, *(["with_practice"] if cfg.include_practice else []),
                                *([sub] if sub else []))
        self.footer = (f"{test} · zone {cfg.zone:g} m · "
                       f"{'practice included' if cfg.include_practice else 'practice excluded'}")
        self.written = []

    def save(self, fig, name, dpi=120, facecolor=None):
        plt = _plt()
        from matplotlib.transforms import offset_copy
        fig.text(1.0, 0.0, self.footer, ha="right", va="top", fontsize=7, color="0.45",
                 transform=offset_copy(fig.transFigure, fig=fig, x=-2, y=-4, units="points"))
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08,
                    **({"facecolor": facecolor} if facecolor else {}))
        plt.close(fig)
        self.written.append(name)
        print(f"[plot] {path}")


def condition_box(ax, df, col, rng, unit=None):
    """Box + jittered dots of df[col] per condition, every condition keeping its slot. Returns the data."""
    data = [df[df.condition == c][col].dropna().values for c in CONDS]
    pos = [x for x, v in enumerate(data) if len(v)]
    if pos:
        bp = ax.boxplot([data[x] for x in pos], positions=pos, widths=0.6, patch_artist=True,
                        showfliers=False)
        for patch, x in zip(bp["boxes"], pos):
            patch.set(facecolor=COLOURS[CONDS[x]], alpha=0.35)
        for med in bp["medians"]:
            med.set(color="black")
        for x in pos:
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(data[x])), data[x], color=COLOURS[CONDS[x]],
                       zorder=3, s=25)
    labels = [f"{c}\n(n = {len(v)} {unit})" for c, v in zip(CONDS, data)] if unit else CONDS
    ax.set_xticks(range(len(CONDS)), labels)     # after boxplot, which relabels its positions
    ax.set_xlim(-0.5, len(CONDS) - 0.5)
    ax.grid(axis="y", alpha=0.3)
    return data


def cond_handles(plt, conds=CONDS):
    return [plt.Rectangle((0, 0), 1, 1, color=COLOURS[c], alpha=0.6) for c in conds]


def missing(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="0.45", fontsize=9)


def pid_list(df):
    return ", ".join(sorted(df.pid.unique()))


def fig_task_time(runs, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    condition_box(ax, runs, "time_s", np.random.default_rng(0), unit="runs")
    ax.set_ylabel("Task time (s)")
    ax.set_ylim(0, runs.time_s.max() * 1.08)
    ax.legend(cond_handles(plt), CONDS, loc="lower left")
    fig.tight_layout()
    out.save(fig, "task_time_summary.png")


CMD_PANELS = {
    "adj_per_min": "Adjustments / min",
    "adj_per_100m": "Adjustments / 100 m progress",
    "mean_hold_s": "Mean command hold (s)",
    "course_turn_per_min": "Commanded turning (deg / min)",
    "cmd_straightness": "Command straightness (1 = one heading)",
    "act_path_ratio": "Path Curvature",   # achieved path length / straight-line distance
}


def fig_commands(runs, legs, out):
    plt = _plt()
    for name, cols, shape in (("command_metrics.png", list(CMD_PANELS), (2, 3)),
                              ("command_metrics_summary.png",
                               ["adj_per_min", "course_turn_per_min", "act_path_ratio"], (1, 3))):
        fig, axs = plt.subplots(*shape, figsize=(4 * shape[1], 4 * shape[0]), squeeze=False)
        rng = np.random.default_rng(0)
        for ax, col in zip(axs.flat, cols):
            condition_box(ax, runs, col, rng)
            ax.set_ylabel(CMD_PANELS[col])
            ax.set_ylim(bottom=0)
        axs.flat[0].legend(cond_handles(plt), CONDS, loc="lower left")
        fig.tight_layout()
        out.save(fig, name)

    tr = legs[legs.phase == "transit"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, (col, title) in zip(axs, [("adj_per_min", "Adjustments per minute of transit"),
                                      ("adj_per_100m", "Adjustments per 100 m of straight-line progress")]):
        condition_box(ax, tr, col, np.random.default_rng(0), unit="legs")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"Pilot adjustments per transit leg ({pid_list(tr)}; dots = legs)")
    fig.tight_layout()
    out.save(fig, "command_metrics_box.png")


# (column, label, signed): signed panels get a zero line instead of a zero floor
DIAL_PANELS = [
    ("spread_adj_per_min", "Spread adjustments / min", False),
    ("spread_search_delta", "Spread set for search\n(d_ref, search − transit)", True),
    ("spacing_search_delta_m", "Achieved spacing change\n(m, search − transit)", True),
    ("pitch_adj_per_min", "Pitch adjustments / min", False),
    ("pitch_search_delta", "Pitch set for search\n(deg, search − transit)", True),
    ("pitch_down_frac", "Fraction of run camera pitched down", False),
]
TIMELINE_COLS = {"spread": "Spread (d_ref)", "spacing": "Spacing (m)", "pitch": "Pitch (deg)"}
TIMELINE_YLIM = {"spread": (SPREAD_RANGE[0] - 0.1, SPREAD_RANGE[1] + 0.1),
                 "pitch": (PITCH_RANGE[0] - 5, PITCH_RANGE[1] + 5),
                 "spacing": (0, None)}


def fig_dials(runs, legs, series, out):
    plt = _plt()
    # One observation per run, by condition.
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (col, label, signed) in zip(axs.flat, DIAL_PANELS):
        data = condition_box(ax, runs, col, rng)
        ax.set_ylabel(label)
        if not any(len(v) for v in data):
            missing(ax, "gimbal pitch not logged\nin these runs" if col.startswith("pitch") else "no data")
            continue
        if signed:
            ax.axhline(0, color="0.3", lw=0.8, zorder=1)
        else:
            ax.set_ylim(bottom=0)
        if col == "pitch_down_frac":
            ax.set_ylim(0, 1.02)
    axs.flat[0].legend(cond_handles(plt), CONDS, loc="upper left")
    fig.suptitle(f"Pilot-set spread and camera pitch ({pid_list(runs)}; dots = runs)")
    fig.tight_layout()
    out.save(fig, "spread_pitch_metrics.png")

    # Transit -> search setting per goal leg. The thick line is the mean over legs, not the median: a dial is
    # either re-set on arrival or left alone, so with few participants the median is whichever habit has more
    # legs and reads as "no change" while one participant re-sets it on every goal.
    vals = ["spread", "spacing", "pitch"]
    tr = legs[legs.phase == "transit"].set_index(["run", "leg"])
    se = legs[legs.phase == "search"].set_index(["run", "leg"])
    wide = tr[["condition"] + vals].join(se[vals], rsuffix="_search", how="inner")
    panels = [("spread", "Commanded spread (d_ref)", ["Swarm"], SPREAD_RANGE),
              ("spacing", "Achieved spacing (m)", ["Swarm"], None),
              ("pitch", "Camera pitch (deg)", CONDS, PITCH_RANGE)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, (col, label, conds, yr) in zip(axs, panels):
        ax.set_xticks([0, 1], ["Transit", "Search"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        drawn, n_legs = False, 0
        for cond in conds:
            w = wide[wide.condition == cond][[col, f"{col}_search"]].dropna()
            if w.empty:
                continue
            drawn, n_legs = True, n_legs + len(w)
            for a, b in w.values:
                ax.plot([0, 1], [a, b], color=COLOURS[cond], lw=1, alpha=0.45, marker="o", ms=4)
            ax.plot([0, 1], w.mean().values, color=COLOURS[cond], lw=2.5, marker="o", ms=8, label=f"{cond} mean")
        if not drawn:
            missing(ax, "gimbal pitch not logged\nin these runs" if col == "pitch" else "no data")
            continue
        if yr:
            ax.set_ylim(yr[0] - 0.05 * (yr[1] - yr[0]), yr[1] + 0.05 * (yr[1] - yr[0]))
        ax.set_title(f"{'swarm' if conds == ['Swarm'] else 'all'} legs, n = {n_legs} (thin = one goal)",
                     fontsize=9)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Setting before (transit) vs. inside (search) each goal zone, time-weighted means")
    fig.tight_layout()
    out.save(fig, "spread_pitch_legs.png")

    fig_timeline(runs, series, out, "spread_pitch_timeline.png")
    swarm = runs[runs.condition == "Swarm"]
    if len(swarm):
        fig_timeline(swarm, series, out, "spread_swarm_timeline.png", keys=("spread", "spacing"),
                     title="Pilot-set spread (left) and the drones' achieved spacing (right), swarm trials")


def fig_timeline(runs, series, out, name, keys=("spread", "spacing", "pitch"), title=None):
    """Per run, one column per signal in keys, over time; search phases shaded, crashes in red.

    Each row shares its time axis and each column its value axis, set explicitly rather than with
    sharex/sharey: matplotlib 3.8's shared-axes registry is global and, after enough figures have been
    made and closed in one process, fails with "type object 'frame' has no attribute '_stale_viewlims'".
    """
    plt = _plt()
    rows = runs.sort_values(["pid", "trial"])
    n = len(rows)
    fig, axs = plt.subplots(n, len(keys), figsize=(5 * len(keys), 1.9 * n + 0.6), squeeze=False)
    for r, run in enumerate(rows.itertuples()):
        sr = series[run.run]
        colour = COLOURS.get(run.condition, "0.3")
        t0, t1 = sr["t"][0], sr["t"][-1]
        for j, key in enumerate(keys):
            ax = axs[r, j]
            ax.set_xlim(t0 - 0.05 * (t1 - t0), t1 + 0.05 * (t1 - t0))
            for enter, exit_ in sr["visits"]:
                ax.axvspan(enter, max(exit_, enter + 1.0), color="0.5", alpha=0.15, lw=0)
            for tc in sr["crashes"]:
                ax.axvline(tc, color="red", lw=1.2, alpha=0.85, zorder=3)
            y = sr[key]
            if np.isnan(y).all():
                missing(ax, "not logged" if key == "pitch" else
                        "one drone" if run.condition == "SingleDrone" else "no data")
            else:
                ax.plot(sr["t"], y, color=colour, lw=1.5, drawstyle="steps-post" if key != "spacing" else None)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(TIMELINE_COLS[key])
            if r == n - 1:
                ax.set_xlabel("time (s)")
        axs[r, 0].set_ylabel(f"{run.pid} t{run.trial}\n{run.condition}", fontsize=9)
    for j, key in enumerate(keys):
        lo, hi = TIMELINE_YLIM[key]
        if hi is None:
            tops = [np.nanmax(series[s][key]) for s in rows.run if not np.isnan(series[s][key]).all()]
            hi = 1.05 * max(tops) if tops else 1.0
        for ax in axs[:, j]:
            ax.set_ylim(lo, hi)
    conds = [c for c in CONDS if (rows.condition == c).any()]
    conds = conds if len(conds) > 1 else []     # one condition: the title names it
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.3)] + \
              [plt.Line2D([], [], color=COLOURS[c], lw=2) for c in conds]
    labels = ["search (inside goal zone)"] + conds
    if any(series[r].get("crashes") for r in rows.run):
        handles.append(plt.Line2D([], [], color="red", lw=1.2))
        labels.append("crash (drones lost)")
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=9)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985 if n > 4 else 0.965))
    out.save(fig, name, dpi=110)


def fig_proximity(goals, cfg, out):
    plt = _plt()
    dists = cfg.dists
    ncol = 3
    nrow = -(-len(dists) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, D in zip(axs.flat, dists):
        condition_box(ax, goals, f"near{D:g}Sec", rng)
        ax.set_title("Over the goal patch" if D == 0 else f"Within {D:g} m of the goal patch")
    top = max(goals[f"near{D:g}Sec"].max() for D in dists)
    for ax in axs.flat:                                      # one scale so the panels compare (no sharey:
        ax.set_ylim(0, top * 1.05)                           # see fig_timeline)
    for ax in axs[:, 1:].flat:
        ax.tick_params(labelleft=False)
    for ax in axs[:, 0]:
        ax.set_ylabel("Time near goal patch (s)")
    for ax in list(axs.flat)[len(dists):]:
        ax.set_visible(False)
    axs.flat[0].legend(cond_handles(plt), CONDS, loc="upper right")
    fig.tight_layout()
    out.save(fig, "goal_proximity.png")

    for col, ylabel, name in ((f"near{cfg.zone:g}Sec", f"Time within {cfg.zone:g} m of goal patch (s)",
                               f"goal_proximity_{cfg.zone:g}m.png"),
                              ("transitSec", "Transit time between goals (s)", f"transit_{cfg.zone:g}m.png")):
        fig, ax = plt.subplots(figsize=(4, 4))
        condition_box(ax, goals, col, np.random.default_rng(0))
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 300)                                  # fixed, so the dwell and transit figures compare
        ax.legend(cond_handles(plt), CONDS, loc="upper right")
        fig.tight_layout()
        out.save(fig, name)

    # One goal's hat at a time: its zone dwell and the transit leg ending at it, side by side.
    for hat in sorted(h for h in goals.hat.unique() if h):
        df = goals[goals.hat == hat]
        panels = [(f"near{cfg.zone:g}Sec", f"Time within {cfg.zone:g} m of the {hat} patch"),
                  ("transitSec", f"Transit to the {hat} goal")]
        fig, axs = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8), squeeze=False)
        rng = np.random.default_rng(0)
        for ax, (col, title) in zip(axs.flat, panels):
            condition_box(ax, df, col, rng, unit="goals")
            ax.set_title(title)
            ax.set_ylabel("Time (s)")
            ax.set_ylim(bottom=0)
        axs.flat[0].legend(cond_handles(plt), CONDS, loc="upper right")
        fig.suptitle(f"{hat} goal only ({pid_list(df)}; dots = goals)")
        fig.tight_layout()
        out.save(fig, f"goal_proximity_{hat.lower()}.png")


# Palette for the panorama timeline: on / hidden are states, not conditions; not recoverable = neutral.
C_ON, C_OFF = "#2a78d6", "#eb6834"
C_UNKNOWN, C_HATCH = "#f0efec", "#a8a79f"
C_SURFACE, C_TEXT, C_TEXT2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def fig_stitch(rows, out):
    """rows: [(runs.csv row, segments, anchors)], one timeline per swarm run."""
    plt = _plt()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    with plt.rc_context({"font.size": 12, "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT2,
                         "xtick.color": C_TEXT2, "ytick.color": C_TEXT, "hatch.linewidth": 0.8}):
        n = len(rows)
        fig, ax = plt.subplots(figsize=(13, 1.65 + 0.52 * n))  # 1.65 in = the fixed margins below
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_SURFACE)
        xmax = max(r["durationSec"] for r, _, _ in rows)
        h = 0.56
        for i, (row, segs, _) in enumerate(rows):
            y = n - 1 - i
            for a, b, kind, k in segs:
                if b <= a:
                    continue
                if kind == "unknown":
                    ax.barh(y, b - a, left=a, height=h, color=C_UNKNOWN, edgecolor=C_SURFACE, linewidth=1.5)
                    ax.barh(y, b - a, left=a, height=h, color="none", edgecolor=C_HATCH, hatch="////", linewidth=0)
                    if b - a > 0.07 * xmax:
                        ax.text((a + b) / 2, y, f"{k} change{'s' if k != 1 else ''}", ha="center", va="center",
                                fontsize=10, color=C_TEXT2, bbox=dict(boxstyle="round,pad=0.2", fc=C_UNKNOWN, ec="none"))
                else:
                    ax.barh(y, b - a, left=a, height=h, color=C_ON if kind == "on" else C_OFF,
                            edgecolor=C_SURFACE, linewidth=1.5)
            lo, hi = row["stitch_visibleMinPct"], row["stitch_visibleMaxPct"]
            label = f"{lo:.0f}%" if abs(hi - lo) < 0.5 else f"{lo:.0f}–{hi:.0f}%"
            ax.text(xmax * 1.015, y, label, va="center", ha="left", fontsize=12, color=C_TEXT)
        for i, (_, _, anchors) in enumerate(rows):  # timed log lines, drawn last so they sit on top
            y = n - 1 - i
            for t in anchors:
                ax.plot([t, t], [y - h / 2 - 0.06, y + h / 2 + 0.06], color=C_TEXT, linewidth=1.2, zorder=5)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{r['pid']} t{r['trial']}" for r, _, _ in reversed(rows)])
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel("time since session start (s)")
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(xmax * 1.015, n - 0.35, "visible", ha="left", va="bottom", fontsize=11, color=C_TEXT2)
        fig.suptitle(f"Stitched panorama on screen — swarm flights, {out.test}", x=0.01, ha="left", fontsize=15,
                     color=C_TEXT)
        handles = [Patch(fc=C_ON, label="panorama on screen"), Patch(fc=C_OFF, label="hidden (feeds shown)")]
        if any(r["stitch_source"] == "editor_log" for r, _, _ in rows):
            handles += [Patch(fc=C_UNKNOWN, ec=C_HATCH, hatch="////", label="changes not placeable (untimed log)"),
                        Line2D([], [], color=C_TEXT, linewidth=1.2, label="timed log line (drone lost)")]
        fig_h = fig.get_size_inches()[1]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.45 / fig_h), ncol=4,
                   frameon=False, fontsize=11, labelcolor=C_TEXT)
        # Explicit margins: tight_layout counts the out-of-axes percentage labels and leaves a gap.
        fig.subplots_adjust(left=0.07, right=0.89, top=1 - 1.05 / fig_h, bottom=0.6 / fig_h)
        out.save(fig, "stitch_visibility.png", dpi=140, facecolor=C_SURFACE)


# ============================================================================= trajectory maps
#
# Turning maps: the building footprints actually in the scene for that run (hat visibility's set), the goal
# tiles and zones numbered in visit order, the flown path (single drone: the drone; swarm: each drone faint
# and the centroid bold), and arrows every --arrow-s seconds of transit giving the direction the pilot was
# *commanding* (only while |v_cmd| > 0.3 maxSpeed, the gate course_turn_per_min uses; none while
# searching, which the metric does not count). The arrows are what the metric measures; the path is what
# the vehicles did with it. `run` draws the single-drone run with the MOST commanded turning and the swarm
# run with the LEAST; `maps` draws others into plots/<test>/maps/. All maps from one invocation share one scale.

MAP_MARGIN_M = 40.0


def footprint(c, ax):
    """XZ outline of an oriented box: the two half-axes that are closest to horizontal."""
    vert = np.abs(ax[:, 1]) / np.linalg.norm(ax, axis=1)
    a1, a2 = ax[np.argsort(vert)[:2]]
    return np.array([[p[0], p[2]] for p in (c + a1 + a2, c + a1 - a2, c - a1 - a2, c - a1 + a2)])


def command_arrows(run, every_s, windows, db):
    """(x, z, ux, uz) at the centroid every every_s seconds of the windows, along the commanded course."""
    t, _, vx, vz = stick_command(run, db)
    out = []
    for tk in np.arange(t[0], t[-1], every_s):
        if not any(t0 <= tk < t1 for t0, t1 in windows):
            continue
        i = min(np.searchsorted(t, tk), len(t) - 1)
        spd = np.hypot(vx[i], vz[i])
        if spd > 0.3 * MAX_SPEED:
            x, z = pos_at(run.centroid, t[i])
            out.append((x, z, vx[i] / spd, vz[i] / spd))
    return np.array(out).reshape(-1, 4)


def map_extent(runs):
    xs, zs = [], []
    for r in runs:
        xs += list(r.cx) + [g["goalX"] + s * TILE_HALF for g in r.goals for s in (-1, 1)]
        zs += list(r.cz) + [g["goalZ"] + s * TILE_HALF for g in r.goals for s in (-1, 1)]
    return (min(xs) - MAP_MARGIN_M, max(xs) + MAP_MARGIN_M), (min(zs) - MAP_MARGIN_M, max(zs) + MAP_MARGIN_M)


def fig_turning_map(run, legs, metrics, lims, cfg, out, headline):
    plt = _plt()
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import FancyBboxPatch

    colour = COLOURS[run.condition]
    (x0, x1), (z0, z1) = lims
    fig, ax = plt.subplots(figsize=(9, 9 * (z1 - z0) / (x1 - x0) + 0.9))
    polys = [footprint(ctr, a) for _, ctr, a in scene_boxes(run.goals)]
    ax.add_collection(PolyCollection(polys, facecolor="0.82", edgecolor="0.62", lw=0.4, zorder=1))

    order = {L["gi"]: L["leg"] for L in legs}
    for gi, g in enumerate(run.goals):
        gx, gz, r = g["goalX"], g["goalZ"], TILE_HALF
        ax.add_patch(plt.Rectangle((gx - r, gz - r), 2 * r, 2 * r, facecolor="#e8c547", alpha=0.18,
                                   edgecolor="#a8871a", lw=1.2, zorder=2))
        if cfg.zone > 0:   # the zone is the tile grown by `zone` m: a rectangle with round corners
            ax.add_patch(FancyBboxPatch((gx - r, gz - r), 2 * r, 2 * r, boxstyle=f"round,pad={cfg.zone}",
                                        facecolor="none", edgecolor="#a8871a", lw=0.9, ls="--", zorder=2))
        ax.text(gx - r, gz + r + cfg.zone + 3, f"goal {order[gi]}" if gi in order else "goal (not reached)",
                ha="left", va="bottom", fontsize=11, fontweight="bold", color="0.2", zorder=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))

    if run.condition == "Swarm":
        for _, g in run.alive.groupby("droneId"):
            ax.plot(g.gtX, g.gtZ, color=colour, lw=0.6, alpha=0.3, zorder=3)
        ax.plot(run.cx, run.cz, color=colour, lw=2.2, zorder=4, label="swarm centroid")
        ax.plot([], [], color=colour, lw=0.6, alpha=0.5, label="individual drones")
    else:
        ax.plot(run.cx, run.cz, color=colour, lw=2.2, zorder=4, label="drone")

    arr = command_arrows(run, cfg.arrow_s, [(L["t0"], L["enter"]) for L in legs], cfg.deadband)
    if len(arr):
        ax.quiver(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], angles="xy", scale_units="xy", scale=1 / 12.0,
                  width=0.0028, color="0.15", zorder=5,
                  label=f"commanded direction in transit (every {cfg.arrow_s:g} s)")
    ax.scatter(run.cx[0], run.cz[0], s=110, marker="o", color="white", edgecolor="black", lw=1.5, zorder=7,
               label="start")
    ax.scatter(run.cx[-1], run.cz[-1], s=110, marker="s", color="black", zorder=7, label="end")
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_title(f"{headline}\n{run.stem}\n"
                 f"commanded turning {metrics['course_turn_per_min']:.0f} deg/min · command straightness "
                 f"{metrics['cmd_straightness']:.2f} · path / straight-line {metrics['act_path_ratio']:.2f}",
                 fontsize=10, loc="left")
    fig.tight_layout()
    out.save(fig, f"{run.stem}_turning_map.png", dpi=130)


def draw_maps(picks, legs, cmd, cfg, out):
    """picks: [(Run, headline)]; one shared scale."""
    lims = map_extent([r for r, _ in picks])
    for run, headline in picks:
        print(f"[map] {headline}: {run.stem} ({cmd[run.stem]['course_turn_per_min']:.0f} deg/min)")
        attempt("turning map", run.stem, fig_turning_map, run, legs[run.stem], cmd[run.stem], lims, cfg, out,
                headline)


def default_picks(runs, cmd):
    """The single-drone run with the most commanded turning and the swarm run with the least."""
    picks = []
    for cond, pick, label in (("SingleDrone", max, "Single drone, highest commanded turning"),
                              ("Swarm", min, "Swarm, lowest commanded turning")):
        cands = [r for r in runs if r.condition == cond and not np.isnan(
            cmd.get(r.stem, {}).get("course_turn_per_min", np.nan))]
        if cands:
            picks.append((pick(cands, key=lambda r: cmd[r.stem]["course_turn_per_min"]), label))
    return picks


# ============================================================================= run: the full analysis

def add_tuning_args(p):
    d = DEFAULTS
    p.add_argument("--zone", type=float, default=d["zone"],
                   help="metres around the goal tile that count as being at the goal (default %(default)g)")
    p.add_argument("--deadband", type=float, default=d["deadband"], help="stick |u| below this counts as neutral")
    p.add_argument("--tol", type=float, default=d["tol"], help="stick move that counts as an adjustment")
    p.add_argument("--settle", type=float, default=d["settle"], help="seconds steady before a new hold latches")
    p.add_argument("--spread-tol", type=float, default=d["spread_tol"], help="d_ref change that counts as an adjustment")
    p.add_argument("--pitch-tol", type=float, default=d["pitch_tol"], help="pitch change (deg) that counts as an adjustment")
    p.add_argument("--down", type=float, default=d["down"], help="pitch (deg) below which the camera counts as down")
    p.add_argument("--range", type=float, default=d["range"], help="hat identification range (m)")
    p.add_argument("--dist", dest="dists", nargs="+", type=float, default=d["dists"],
                   help="near-goal distances (m) for the dwell sweep")
    p.add_argument("--arrow-s", type=float, default=d["arrow_s"], help="seconds between arrows on the turning maps")
    p.add_argument("--include-practice", action="store_true",
                   help="put the practice runs back into the figures and summaries (figures go to with_practice/)")


def analyse_runs(runs, cfg, log_runs, families):
    """Per-run results of the requested families: dict of {stem: ...} maps."""
    res = {k: {} for k in ("legs", "task", "cmd", "cmd_legs", "dial", "dial_legs", "series", "prox", "hat",
                           "stitch", "deaths", "crashes")}
    for run in runs:
        legs = goal_legs(run, cfg.zone)
        res["legs"][run.stem] = legs
        if len(legs) < len(run.goals):
            unreached = [hat_name(run.goals[gi]) or f"goal {gi}"
                         for gi in range(len(run.goals)) if gi not in {L["gi"] for L in legs}]
            warn(f"{run.stem}: zone never entered for {', '.join(unreached)}; no phases for it")
        res["task"][run.stem] = task_family(run)
        if "cmd" in families:
            r = attempt("pilot commands", run.stem, command_family, run, legs, cfg)
            if r:
                res["cmd"][run.stem], res["cmd_legs"][run.stem] = r
        if "dial" in families:
            r = attempt("spread/pitch", run.stem, dial_family, run, legs, cfg)
            if r:
                res["dial"][run.stem], res["dial_legs"][run.stem], res["series"][run.stem] = r
        if "prox" in families:
            r = attempt("goal proximity", run.stem, proximity_family, run, legs, cfg)
            if r:
                res["prox"][run.stem] = r
        if "hat" in families:
            r = attempt("hat visibility", run.stem, hat_family, run, cfg)
            if r:
                res["hat"][run.stem] = r
        if "stitch" in families and run.condition == "Swarm":
            r = attempt("stitch visibility", run.stem, stitch_family, run, log_runs)
            if r:
                res["stitch"][run.stem] = r
        if "crash" in families:
            r = attempt("crashes", run.stem, crash_family, run, log_runs)
            if r:
                res["deaths"][run.stem], res["crashes"][run.stem] = r
                if run.stem in res["series"]:
                    res["series"][run.stem]["crashes"] = r[1]
    for s in res["series"].values():
        s.setdefault("crashes", [])
    return res


def build_tables(runs, res):
    by_stem = {r.stem: r for r in runs}

    def ident(stem):
        return by_stem[stem].ident()

    run_rows = []
    for r in runs:
        row = dict(r.ident(), durationSec=r.duration, nGoals=r.session.get("nGoals"),
                   nCorrect=r.session.get("nCorrect"), goalsReached=len(res["legs"][r.stem]))
        row.update(res["task"][r.stem])
        row.update(res["cmd"].get(r.stem, {}))
        row.update(res["dial"].get(r.stem, {}))
        if r.stem in res["stitch"]:
            row.update(res["stitch"][r.stem][0])
        if r.stem in res["deaths"]:
            row.update(drones_lost=len(res["deaths"][r.stem]), crashes=len(res["crashes"][r.stem]))
        run_rows.append(row)
    runs_df = pd.DataFrame(run_rows)

    leg_frames = []
    for stem in res["legs"]:
        c = pd.DataFrame(res["cmd_legs"].get(stem, []))
        d = pd.DataFrame(res["dial_legs"].get(stem, []))
        if len(c) and len(d):
            m = c.merge(d, on=["run", "leg", "hat", "phase"], how="outer")
        else:
            m = c if len(c) else d
        if len(m):
            leg_frames.append(pd.concat([pd.DataFrame([ident(stem)] * len(m)), m.drop(columns="run")], axis=1))
    legs_df = pd.concat(leg_frames, ignore_index=True) if leg_frames else pd.DataFrame()
    if len(legs_df):
        first = ["run", "pid", "trial", "condition", "date", "practice", "leg", "phase", "hat"]
        legs_df = legs_df[first + [c for c in legs_df.columns if c not in first]]
        legs_df = legs_df.sort_values(["pid", "trial", "run", "leg", "phase"], ascending=[True, True, True, True, False],
                                      ignore_index=True)

    goal_rows = []
    for r in runs:
        prox = {p["goalIndex"]: p for p in res["prox"].get(r.stem, [])}
        hat = {h["goalIndex"]: h for h in res["hat"].get(r.stem, [])}
        for gi, g in enumerate(r.goals):
            idx = g.get("goalIndex", gi)
            row = dict(r.ident(), goalIndex=idx, hat=hat_name(g), visitOrder=g.get("visitOrder", -1),
                       answeredHat=g.get("answeredHat", ""), outcome=g.get("outcome", ""))
            row.update({k: v for k, v in prox.get(idx, {}).items() if k not in ("run", "goalIndex")})
            row.update({k: v for k, v in hat.get(idx, {}).items() if k not in ("run", "goalIndex")})
            goal_rows.append(row)
    goals_df = pd.DataFrame(goal_rows)

    death_rows = [dict(ident(stem), **{k: v for k, v in d.items() if k != "run"})
                  for stem, ds in res["deaths"].items() for d in ds]
    deaths_df = pd.DataFrame(death_rows, columns=["run", "pid", "trial", "condition", "date", "practice",
                                                  "droneId", "t", "x", "y", "z", "reason"])
    return runs_df, legs_df, goals_df, deaths_df


CMD_SHOW = ["adj_per_min", "adj_per_100m", "mean_hold_s", "stick_tv_per_min", "yaw_duty", "lat_duty",
            "course_turn_per_min", "cmd_straightness", "cmd_path_ratio", "act_path_ratio"]
DIAL_SHOW = ["spread_adj", "spread_adj_per_min", "spread_adj_per_min_transit", "spread_adj_per_min_search",
             "spread_tv_per_min", "spread_transit", "spread_search", "spread_search_delta",
             "spacing_transit_m", "spacing_search_m", "spacing_search_delta_m",
             "pitch_adj_per_min", "pitch_transit", "pitch_search", "pitch_search_delta", "pitch_down_frac"]
STITCH_SHOW = ["stitch_source", "durationSec", "stitch_knownOnSec", "stitch_knownOffSec", "stitch_unknownSec",
               "stitch_visibleMinPct", "stitch_visibleMaxPct", "stitch_nOn", "stitch_hidden_photometric",
               "stitch_hidden_no_overlap", "stitch_hidden_pilot"]


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def show(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].round(2).to_string(index=False)


def means(df, cols):
    cols = [c for c in cols if c in df.columns]
    out = ""
    for keys, label in ((["pid", "condition"], "participant x condition"), ("condition", "condition")):
        out += f"\nMean per {label}:\n" + df.groupby(keys)[cols].mean(numeric_only=True).round(2).to_string() + "\n"
    return out


def report(runs_df, legs_df, goals_df, deaths_df, cfg):
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 80)
    ids = ["pid", "trial", "condition"]

    section("Task time (s)")
    print(show(runs_df, ids + ["time_s", "time_source", "nCorrect", "nGoals", "goalsReached"]))
    g = runs_df.groupby("condition").time_s
    summ = g.agg(n="count", total="sum", mean="mean", sd="std", min="min", max="max")
    summ["se"] = summ["sd"] / np.sqrt(summ["n"])
    print("\n" + summ.reindex([c for c in CONDS if c in summ.index])[
        ["n", "total", "mean", "sd", "se", "min", "max"]].round(2).to_string())

    if "adj_per_min" in runs_df:
        section(f"Pilot commands (transit only; deadband={cfg.deadband} tol={cfg.tol} settle={cfg.settle}s)")
        print(show(runs_df, ids + ["transit_s", "adjustments"] + CMD_SHOW))
        print("\nAdjustments by axis (transit):")
        print(show(runs_df, ids + [f"adj_{x}" for x in AXIS_NAMES]))
        print(means(runs_df, CMD_SHOW))

    if "spread_adj" in runs_df:
        section(f"Spread and pitch dials (spread-tol={cfg.spread_tol} pitch-tol={cfg.pitch_tol}deg "
                f"settle={cfg.settle}s down<{cfg.down}deg)")
        if not runs_df.pitch_logged.any():
            print("[info] no run has a gimbalPitch column (logged from 2026-09-24 on): pitch metrics are NaN")
        print(show(runs_df, ids + ["durationSec", "search_s"] + DIAL_SHOW))
        print(means(runs_df, DIAL_SHOW))

    if len(goals_df):
        section("Goals: proximity and hat visibility")
        near = [c for c in goals_df.columns if c.startswith("near")]
        print(show(goals_df, ids + ["hat", "visitOrder", "outcome", "zoneOrder", "minDistToTileM"] + near +
                   ["transitSec", "searchSec"]))
        med = [c for c in near + ["transitSec", "searchSec"] if c in goals_df.columns]
        print("\nMedian seconds per goal / leg:")
        print(goals_df.groupby("condition")[med].median().to_string())
        if "inViewSec" in goals_df:
            print()
            print(show(goals_df, ids + ["hat", "firstSightSec", "inViewSec", "nEpisodes", "longestSec", "hatMinDistM",
                                        "meanDronesInView", "levelCamPct", "occludedSec", "episodes"]))

    if "stitch_source" in runs_df:
        section("Stitched panorama visibility (swarm runs)")
        print(show(runs_df[runs_df.stitch_source.notna()], ids + STITCH_SHOW))

    if "drones_lost" in runs_df:
        section("Drones lost")
        print(show(runs_df, ids + ["drones_lost", "crashes"]))
        if len(deaths_df):
            print("\n" + deaths_df.groupby(["condition", "reason"]).size().rename("deaths").to_string())


def write_tables(folder, tables, cfg, runs, manifest):
    out = os.path.join(folder, RESULTS)
    os.makedirs(out, exist_ok=True)
    for name, df in tables.items():
        path = os.path.join(out, f"{name}.csv")
        df.to_csv(path, sep=";", index=False)
        print(f"[table] {path}")
    config = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), script=os.path.basename(__file__),
                  test=manifest.get("name"), practice=manifest.get("practice", []),
                  runs=[r.stem for r in runs],
                  settings={k: getattr(cfg, k) for k in DEFAULTS})
    with open(os.path.join(out, "analysis_config.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(config, indent=2) + "\n")


def cmd_run(a):
    a.dists = sorted(set(a.dists) | {a.zone})
    folder, manifest, runs = load_test(a.root, a.test)
    if not runs:
        raise SystemExit(f"no finalized runs in {folder}")
    log_runs = parse_editor_logs(sorted(glob.glob(os.path.join(folder, "*.log"))))
    print(f"[info] {a.test}: {len(runs)} runs, {sum(r.practice for r in runs)} practice; "
          f"zone {a.zone:g} m; {len(log_runs)} run(s) in the editor log(s)")
    res = analyse_runs(runs, a, log_runs, {"cmd", "dial", "prox", "hat", "stitch", "crash"})
    runs_df, legs_df, goals_df, deaths_df = build_tables(runs, res)
    write_tables(folder, dict(runs=runs_df, legs=legs_df, goals=goals_df, crashes=deaths_df), a, runs, manifest)

    keep = (lambda df: df) if a.include_practice else (lambda df: df[~df.practice.astype(bool)] if len(df) else df)
    shown = [r for r in runs if a.include_practice or not r.practice]
    print(f"\n[info] summaries and figures: {len(shown)} runs "
          f"({'practice included' if a.include_practice else 'practice excluded: ' + ', '.join(r.stem for r in runs if r.practice)})")
    report(keep(runs_df), keep(legs_df), keep(goals_df), keep(deaths_df), a)
    if a.no_plot:
        print_warnings()
        return

    section("Figures")
    out = Figures(a.root, a.test, a)
    fr, fl, fg = keep(runs_df), keep(legs_df), keep(goals_df)
    attempt("figure", "task time", fig_task_time, fr, out)
    if "adj_per_min" in fr and len(fl):
        attempt("figure", "pilot commands", fig_commands, fr, fl, out)
    if "spread_adj" in fr and len(fl):
        attempt("figure", "spread/pitch", fig_dials, fr, fl, {s: res["series"][s] for s in fr.run if s in res["series"]},
                out)
    if len(fg) and "transitSec" in fg:
        attempt("figure", "goal proximity", fig_proximity, fg, a, out)
    stitch_rows = [(dict(pid=r.pid, trial=r.trial, durationSec=r.duration, **res["stitch"][r.stem][0]),
                    res["stitch"][r.stem][1], res["stitch"][r.stem][2]) for r in shown if r.stem in res["stitch"]]
    if stitch_rows:
        attempt("figure", "stitch visibility", fig_stitch, stitch_rows, out)
    picks = default_picks(shown, res["cmd"])
    if picks:
        draw_maps(picks, res["legs"], res["cmd"], a, out)
    print_warnings()


# ============================================================================= maps / quicklook

def cmd_maps(a):
    folder, manifest, runs = load_test(a.root, a.test)
    if not a.include_practice and not a.stem:
        runs = [r for r in runs if not r.practice]
    if a.stem:
        runs = [r for r in runs if any(matches(r.stem, s) for s in a.stem)]
        if not runs:
            raise SystemExit(f"no runs match {a.stem} in {folder}")
    res = analyse_runs(runs, a, {}, {"cmd"})
    runs = [r for r in runs if r.stem in res["cmd"] and "course_turn_per_min" in res["cmd"][r.stem]]
    out = Figures(a.root, a.test, a, sub="maps")      # never overwrites the two maps `run` draws
    if a.all or a.stem:
        picks = []
        for cond, label in (("SingleDrone", "Single drone"), ("Swarm", "Swarm")):
            sub = sorted((r for r in runs if r.condition == cond),
                         key=lambda r: -res["cmd"][r.stem]["course_turn_per_min"])
            picks += [(r, f"{label}, commanded turning rank {k} of {len(sub)} (1 = most)")
                      for k, r in enumerate(sub, start=1)]
    else:
        picks = default_picks(runs, res["cmd"])
    draw_maps(picks, res["legs"], res["cmd"], a, out)
    print_warnings()


def cmd_quicklook(a):
    """Top-down trajectories and altitude of each matching run (the old plot_experiment.py)."""
    folder = os.path.join(a.root, a.test)
    stems = [s for s, files in run_files(folder).items()
             if matches(s, a.stem) and f"{s}_drones.csv" in files]
    if not stems:
        raise SystemExit(f"no runs match {a.stem!r} in {folder}")
    plt = _plt()
    out_dir = os.path.join(a.root, PLOTS, a.test)
    os.makedirs(out_dir, exist_ok=True)
    for stem in stems:
        base = os.path.join(folder, stem)
        drones = pd.read_csv(base + "_drones.csv", sep=";")
        head = pd.read_csv(base + "_head.csv", sep=";") if os.path.exists(base + "_head.csv") else None
        walkers = pd.read_csv(base + "_walkers.csv", sep=";") if os.path.exists(base + "_walkers.csv") else None
        events = pd.read_csv(base + "_events.csv", sep=";") if os.path.exists(base + "_events.csv") else None
        session = None
        if os.path.exists(base + "_session.json"):
            with open(base + "_session.json", encoding="utf-8") as f:
                session = json.load(f)
        with plt.rc_context({"axes.titlesize": 20, "axes.labelsize": 18, "xtick.labelsize": 14,
                             "ytick.labelsize": 14, "figure.titlesize": 20}):
            fig_traj, ax = plt.subplots(figsize=(10, 7))
            fig_alt, ax2 = plt.subplots(figsize=(8, 6))
            # Shift XZ so the swarm start sits at (0, 0): origin = mean of each drone's earliest sample.
            first = drones.sort_values("t").groupby("droneId").first()
            ox, oz = first["gtX"].mean(), first["gtZ"].mean()
            for did, g in drones.groupby("droneId"):
                ax.plot(g["gtX"] - ox, g["gtZ"] - oz, lw=1, alpha=0.8, label=f"drone {did}")
                ax.scatter(g["gtX"].iloc[0] - ox, g["gtZ"].iloc[0] - oz, s=20, marker="o", color="green", zorder=5)
            ax.scatter(0, 0, s=90, marker="o", color="green", edgecolor="black", zorder=7)
            ax.annotate("start", (0, 0), textcoords="offset points", xytext=(8, 8), fontsize=11,
                        fontweight="bold", color="green")
            if head is not None and not head.empty:
                ax.plot(head["headX"] - ox, head["headZ"] - oz, "k--", lw=1.2, alpha=0.7, label="head")
            if walkers is not None and not walkers.empty:
                for _, w in walkers.groupby("goalIndex"):
                    ax.plot(w["specialX"] - ox, w["specialZ"] - oz, lw=1, alpha=0.5)
            for g in (session or {}).get("goals", []):
                ax.scatter(g["goalX"] - ox, g["goalZ"] - oz, s=220, marker="*", color="gold", edgecolor="black",
                           zorder=6)
                ax.annotate(f"goal {g['goalIndex']}", (g["goalX"] - ox, g["goalZ"] - oz),
                            textcoords="offset points", xytext=(6, 6))
                ax.scatter(g["specialSpawnX"] - ox, g["specialSpawnZ"] - oz, s=60, marker="X", color="red", zorder=6)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title("Top-down trajectories (XZ)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(loc="upper right", fontsize=12, ncol=2)
            ax.grid(True, alpha=0.3)
            for _, g in drones.groupby("droneId"):
                ax2.plot(g["t"], g["gtY"], lw=1, alpha=0.8)
            if events is not None and not events.empty:
                for _, e in events[events["eventType"] == "identify"].iterrows():
                    ax2.axvline(e["t"], color="blue", ls=":", alpha=0.6)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("altitude Y (m)")
            ax2.set_title("Altitude vs time")
            ax2.grid(True, alpha=0.3)
            for fig, suffix in ((fig_traj, "_trajectory.png"), (fig_alt, "_altitude.png")):
                fig.suptitle(stem)
                fig.tight_layout()
                path = os.path.join(out_dir, stem + suffix)
                fig.savefig(path, dpi=130)
                plt.close(fig)
                print(f"[plot] {path}")


# ============================================================================= answers
#
# Back-fill identify answers when the experimenter's identify keys were not pressed and the participant's
# answers were noted by hand instead, as the hat named at each goal patch in the order the patches were
# visited (e.g. "Cap, Cowboy, Bucket").
#   answers template <test>   writes <test>/answers.csv: run;answer1;answer2;answer3;visitOrder (auto)
#   answers apply <test>      applies it to each run's _session.json
# `answerK` is the hat named at the K-th patch visited: Cap / Cowboy / Bucket (case-insensitive, "Walker"
# optional), `skip` if no answer was given there, blank if the participant never got that far. Visit order
# is the zone order (see Phases); a goal whose zone was never entered is placed by its closest approach and
# marked `~` in visitOrder -- check those by hand. `apply` sets, per goal: answered / outcome (correct iff
# the named hat is the hat on that patch), visitOrder, visitEnterSec, visitExitSec, visitApprox, visitZoneM,
# and recomputes nCorrect. decisionTimeSec, swarmToWalkerDist and the centroid stay unset (-1 / 0) and no
# identify events are written: the moment of each answer was not recorded. Safe to re-run.

N_ANSWERS = 3


def norm_hat(h):
    h = h.strip().lower()
    return h[:-len("walker")] if h.endswith("walker") else h


def answer_visits(run, zone):
    """[(goal, enterSec, exitSec, approx)] in visit order."""
    legs = goal_legs(run, zone)
    out = [(L["goal"], L["enter"], L["exit"], False) for L in legs]
    reached = {L["gi"] for L in legs}
    for gi, g in enumerate(run.goals):
        if gi not in reached:
            k = int(np.argmin((run.cx - g["goalX"]) ** 2 + (run.cz - g["goalZ"]) ** 2))
            out.append((g, float(run.ct[k]), float(run.ct[k]), True))
    return sorted(out, key=lambda v: v[1])


def order_string(vs):
    return " > ".join(("~" if a else "") + hat_name(g) for g, _, _, a in vs)


def read_rows(path):
    # utf-8-sig: Excel (and PowerShell) save answers.csv with a BOM.
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter=";"))


def cmd_answers(a):
    folder, _, runs = load_test(a.root, a.test)
    path = os.path.join(folder, "answers.csv")
    if a.mode == "template":
        if os.path.exists(path):
            raise SystemExit(f"{path} already exists; delete it first if you want a fresh template")
        rows = [["run"] + [f"answer{k + 1}" for k in range(N_ANSWERS)] + ["visitOrder (auto, do not edit)"]]
        rows += [[r.stem] + [""] * N_ANSWERS + [order_string(answer_visits(r, a.zone))] for r in runs]
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter=";").writerows(rows)
        print(f"wrote {path} ({len(rows) - 1} runs)")
        return

    by_stem = {r.stem: r for r in runs}
    for row in read_rows(path)[1:]:
        if not row or not row[0].strip():
            continue
        stem = row[0].strip()
        answers = [x.strip() for x in row[1:1 + N_ANSWERS]]
        answers += [""] * (N_ANSWERS - len(answers))
        if not any(answers):
            continue
        run = by_stem.get(stem)
        if run is None:
            raise SystemExit(f"{stem}: not a finalized run in {folder} (renamed or archived since the template?)")
        j = run.session
        hats = {norm_hat(g["hat"]) for g in j["goals"]}
        for x in answers:
            if x and x.lower() != "skip" and norm_hat(x) not in hats:
                raise SystemExit(f"{stem}: answer '{x}' is not one of {sorted(hats)} or 'skip'")
        vs = answer_visits(run, a.zone)
        for k, (g, enter, exit_, approx) in enumerate(vs):
            x = answers[k] if k < len(answers) else ""
            outcome = "" if not x else "skip" if x.lower() == "skip" else \
                "correct" if norm_hat(x) == norm_hat(g["hat"]) else "incorrect"
            g.update(answered=bool(outcome), outcome=outcome, answeredHat=x,
                     decisionTimeSec=-1.0, swarmToWalkerDist=-1.0,
                     centroidAtAnswerX=0.0, centroidAtAnswerY=0.0, centroidAtAnswerZ=0.0,
                     visitOrder=k + 1, visitEnterSec=round(enter, 4), visitExitSec=round(exit_, 4),
                     visitApprox=approx, visitZoneM=a.zone)
        j["nCorrect"] = sum(g["outcome"] == "correct" for g in j["goals"])
        j["answersSource"] = "manual (analyse.py answers, matched by visit order)"
        with open(run.path("session.json"), "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps(j, indent=4))
        detail = ", ".join(f"{hat_name(g)}<-{g['answeredHat'] or '-'}:{g['outcome'] or '-'}" for g, *_ in vs)
        print(f"{stem}: nCorrect={j['nCorrect']}  [{detail}]")


# ============================================================================= status / group / archive

def describe_runs(folder, stems, practice=()):
    rows = []
    for s in stems:
        p = parse_stem(s)
        sj = os.path.join(folder, f"{s}_session.json")
        dur = ""
        if os.path.exists(sj):
            with open(sj, encoding="utf-8") as f:
                dur = f"{float(json.load(f)['durationSec']):.0f}"
        rows.append(dict(run=s, pid=p["pid"], trial=p["trial"], condition=p["condition"], date=p["date"],
                         durationSec=dur or "-", finalized="yes" if dur else "NO",
                         practice="practice" if any(matches(s, e) for e in practice) else ""))
    return pd.DataFrame(rows)


def cmd_status(a):
    root = a.root
    print(f"experiment folder: {root}")
    tests = sorted(d for d in os.listdir(root) if read_manifest(os.path.join(root, d)) is not None)
    for t in tests:
        folder = os.path.join(root, t)
        m = read_manifest(folder)
        stems = sorted(run_files(folder))
        df = describe_runs(folder, stems, m.get("practice", []))
        print(f"\n{t}: {len(stems)} runs, {(df.practice != '').sum() if len(df) else 0} practice"
              + (f" -- {m['description']}" if m.get("description") else ""))
        if len(df):
            counts = df.groupby(["pid", "condition"]).size().unstack(fill_value=0)
            print("  " + counts.to_string().replace("\n", "\n  "))
            print(f"  practice: {', '.join(m.get('practice', [])) or '(none)'}")
            bad = df[df.finalized == "NO"]
            if len(bad):
                print(f"  not finalized: {', '.join(bad.run)}")
    print(f"\narchive: {len(run_files(os.path.join(root, ARCHIVE)))} runs")
    stems = sorted(run_files(root))
    print(f"\nungrouped runs at the root: {len(stems)}")
    if stems:
        pd.set_option("display.width", 200)
        print(describe_runs(root, stems).drop(columns="practice").to_string(index=False))


def select_root_runs(root, selectors, date):
    files = run_files(root)
    chosen = {}
    for x in selectors:
        hit = [s for s in files if matches(s, x) and (not date or parse_stem(s)["date"] == date)]
        if not hit:
            raise SystemExit(f"{x!r} matches no ungrouped run at the root" + (f" on {date}" if date else "")
                             + "; `analyse.py status` lists them")
        for s in hit:
            chosen[s] = files[s]
    return chosen


def move_runs(root, dest, chosen, dry_run):
    clash = [s for s in chosen if s in run_files(dest)]
    if clash:
        raise SystemExit(f"already in {dest}: {', '.join(clash)}")
    for s in sorted(chosen):
        print(f"  {s}  ({len(chosen[s])} files) -> {os.path.relpath(dest, root)}/")
    if dry_run:
        print("(dry run: nothing moved)")
        return False
    os.makedirs(dest, exist_ok=True)
    for files in chosen.values():
        for f in files:
            shutil.move(os.path.join(root, f), os.path.join(dest, f))
    return True


def cmd_group(a):
    if a.test in (ARCHIVE, PLOTS) or not re.match(r"^[A-Za-z0-9_\-]+$", a.test):
        raise SystemExit(f"{a.test!r} can't be a test name" + (" (use `archive`)" if a.test == ARCHIVE else ""))
    dest = os.path.join(a.root, a.test)
    chosen = select_root_runs(a.root, a.selectors, a.date)
    if not move_runs(a.root, dest, chosen, a.dry_run):
        return
    m = read_manifest(dest) or dict(name=a.test, created=datetime.date.today().isoformat(), description="",
                                    practice=[])
    all_stems = sorted(run_files(dest))
    new_pids = {parse_stem(s)["pid"] for s in chosen}
    old_pids = {parse_stem(s)["pid"] for s in all_stems if s not in chosen}
    added = [e for e in practice_prefill([s for s in all_stems if parse_stem(s)["pid"] in new_pids - old_pids])
             if e not in m["practice"]]
    m["practice"] = m["practice"] + added
    write_manifest(dest, m)
    print(f"{os.path.join(dest, MANIFEST)}: practice = {m['practice']}")
    if added:
        print(f"  pre-filled {', '.join(added)} (first trial per participant and condition) -- check it")
    if new_pids & old_pids:
        print(f"  {', '.join(sorted(new_pids & old_pids))} already had runs here: their practice entries are unchanged")


def cmd_archive(a):
    chosen = select_root_runs(a.root, a.selectors, a.date)
    move_runs(a.root, os.path.join(a.root, ARCHIVE), chosen, a.dry_run)


# ============================================================================= CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="experiment folder (default: Unity's persistentDataPath)")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="list the tests, the archive and the ungrouped runs")

    for name, text in (("group", "move ungrouped runs into a test folder and pre-fill its test.json"),
                       ("archive", "move ungrouped runs into archive/ (analysed by nothing)")):
        p = sub.add_parser(name, help=text)
        if name == "group":
            p.add_argument("test", help="test folder name, e.g. internal_3")
        p.add_argument("selectors", nargs="+", help="participant id (AABD), PID_tN, or a full run stem")
        p.add_argument("--date", help="only runs recorded on this day (YYYYMMDD)")
        p.add_argument("--dry-run", action="store_true", help="show what would move, move nothing")

    p = sub.add_parser("answers", help="write or apply a test's answers.csv (hand-noted identify answers)")
    p.add_argument("mode", choices=["template", "apply"])
    p.add_argument("test")
    p.add_argument("--zone", type=float, default=DEFAULTS["zone"], help="goal zone (m) that defines visit order")

    p = sub.add_parser("run", help="the full analysis of one test: tables, report, figures")
    p.add_argument("test")
    add_tuning_args(p)
    p.add_argument("--no-plot", action="store_true")

    p = sub.add_parser("maps", help="turning maps for more runs than `run` draws (into plots/<test>/maps/)")
    p.add_argument("test")
    p.add_argument("--all", action="store_true", help="every run, ranked within its condition")
    p.add_argument("--stem", nargs="+", help="these runs (stem prefixes; practice runs allowed)")
    add_tuning_args(p)

    p = sub.add_parser("quicklook", help="top-down trajectory + altitude plots of one run (or several)")
    p.add_argument("test", help="folder the run is in (a test, or archive)")
    p.add_argument("stem", help="run stem or prefix, e.g. ERIC_t7")

    a = ap.parse_args()
    {"status": cmd_status, "group": cmd_group, "archive": cmd_archive, "answers": cmd_answers, "run": cmd_run,
     "maps": cmd_maps, "quicklook": cmd_quicklook}[a.command](a)


if __name__ == "__main__":
    main()
