"""Pilot-command directness metrics for the city search task (ExperimentRecorder output).

The question is how *the pilot* flew, not how the drones moved: did they hold one command and let
the vehicles sort out the obstacles, or keep re-steering around them? Everything here is computed
from the stick inputs logged in _head.csv (inPitch / inRoll / inYaw / inThrottle, ~10 Hz), with the
drone trajectory used only to split the run into phases and to give the achieved path for contrast.

Command frame. Pitch/roll are a normalised horizontal velocity, magnitude-limited to 1, rotated about
world-up by the pilot body yaw (VR command frame, forced in the hull attitude modes and equal to the
drone's own heading for a lone drone) and scaled by VelocityControl.maxSpeed:
    v_cmd = maxSpeed * R_y(bodyYaw) (roll, 0, pitch)
This matches the achieved centroid velocity at r = 0.92-0.97 with ~1 s lag on every AABB/AABC run.

Phases. Visits come from the session JSON (visitEnterSec / visitExitSec, written by apply_answers.py
or reconstructed here the same way): *transit* leg k runs from leaving the previous goal tile (the
start, for k = 1) to entering goal tile k; *search* is the time inside a tile. Searching for a walker
is hovering and peering, and would otherwise swamp the navigation numbers.

Metrics (per transit leg, then summed/averaged per run; "_all" variants cover the whole run):
  adjustments       Discrete manual adjustments. The 4 flight axes are dead-banded (|u| < DEADBAND
                    -> 0) and an adjustment is registered whenever any axis leaves the last *held*
                    position by more than ADJ_TOL; the held position is re-latched once the stick has
                    stayed within ADJ_TOL/2 for SETTLE_S. Pushing forward and holding = 1; releasing
                    = 1; constant re-steering = many. Also broken down by axis.
  adj_per_min       adjustments per minute of transit.
  adj_per_100m      adjustments per 100 m of straight-line progress (leg start -> goal tile edge).
  mean_hold_s       mean time between adjustments (how long one command was held).
  stick_tv_per_min  total variation of the stick vector, sum |du|, per minute: continuous control
                    activity, insensitive to the thresholds above.
  yaw_duty / lat_duty  fraction of transit time with yaw / roll (lateral) stick outside the deadband.
  course_turn_per_min  total |change| of the commanded course (direction of v_cmd, only while
                    |v_cmd| > 0.3 maxSpeed) in degrees per minute of transit: how much the pilot
                    steered, whether by yaw or by roll.
  cmd_straightness  |integral of v_cmd| / integral of |v_cmd|: 1 = one heading held for the leg.
  cmd_path_ratio    commanded path length / straight-line leg distance.
  act_path_ratio    achieved centroid path length / straight-line leg distance (context: the swarm
                    can bend its own path around buildings without the pilot commanding it).

Each participant's first trial in each condition is practice and is dropped (--include-practice
keeps it); the remaining runs are compared by condition only.

Usage:
    python command_metrics.py                       # AABB + AABC, default data folder
    python command_metrics.py --pid AABB AABC ERIC --dir D:/experiment
    python command_metrics.py --deadband 0.15 --tol 0.3    # sensitivity check

Writes command_metrics_runs.csv, command_metrics_legs.csv and plots/command_metrics*.png into the
data folder, and prints a per-run table and a participant x condition summary.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "AppData", "LocalLow",
                           "UAVS@BERKELEY", "DroneSim", "experiment")
MAX_SPEED = 9.31          # VelocityControl.maxSpeed (DroneReduced prefab, both conditions)
TILE_HALF = 90.83 / 2     # goal patch = one city tile; same test as apply_answers.py
AXES = ["inPitch", "inRoll", "inYaw", "inThrottle"]
AXIS_NAMES = ["pitch", "roll", "yaw", "throttle"]


# ----------------------------------------------------------------------------- loading

def load_run(d, stem):
    h = pd.read_csv(os.path.join(d, stem + "_head.csv"), sep=";")
    dr = pd.read_csv(os.path.join(d, stem + "_drones.csv"), sep=";")
    s = json.load(open(os.path.join(d, stem + "_session.json"), encoding="utf-8"))
    c = dr[dr.alive == 1].groupby("t")[["gtX", "gtZ"]].mean()
    return h, c, s


def visits(s, c):
    """[(goal, enter, exit)] in visit order; reuses apply_answers.py's reconstruction if present."""
    out = []
    for g in s["goals"]:
        if "visitEnterSec" in g:
            out.append((g, g["visitEnterSec"], g["visitExitSec"]))
            continue
        x, z, t = c.gtX.values, c.gtZ.values, c.index.values
        inside = (np.abs(x - g["goalX"]) <= TILE_HALF) & (np.abs(z - g["goalZ"]) <= TILE_HALF)
        if inside.any():
            i = int(np.argmax(inside))
            j = i + int(np.argmax(~inside[i:])) if (~inside[i:]).any() else len(t) - 1
            out.append((g, t[i], t[j]))
        else:
            k = int(np.argmin((x - g["goalX"]) ** 2 + (z - g["goalZ"]) ** 2))
            out.append((g, t[k], t[k]))
    return sorted(out, key=lambda v: v[1])


def dist_to_tile(x, z, g):
    """Straight-line distance from (x, z) to the edge of goal g's tile (0 inside)."""
    dx = max(abs(x - g["goalX"]) - TILE_HALF, 0.0)
    dz = max(abs(z - g["goalZ"]) - TILE_HALF, 0.0)
    return float(np.hypot(dx, dz))


# ----------------------------------------------------------------------------- signal metrics

def deadband(u, db):
    return np.where(np.abs(u) < db, 0.0, u)


def detect_adjustments(t, U, tol, settle_s):
    """Indices where a manual adjustment starts, plus which axes moved in each.

    U is (n, 4) dead-banded stick. Hysteresis on the held position: an adjustment starts when any
    axis departs the latched hold by more than tol; during the move the hold is unlatched, and it is
    re-latched once the stick has stayed within tol/2 of one position for settle_s.
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


def command_velocity(h):
    u = np.c_[h.inRoll.values, h.inPitch.values]
    m = np.linalg.norm(u, axis=1)
    u[m > 1] /= m[m > 1, None]
    psi = np.radians(h.bodyYaw.values)
    # Unity Quaternion.Euler(0, psi, 0) * (x, 0, z)
    vx = MAX_SPEED * (u[:, 0] * np.cos(psi) + u[:, 1] * np.sin(psi))
    vz = MAX_SPEED * (-u[:, 0] * np.sin(psi) + u[:, 1] * np.cos(psi))
    return vx, vz


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


def pos_at(c, t):
    i = min(np.searchsorted(c.index.values, t), len(c) - 1)
    return c.gtX.values[i], c.gtZ.values[i]


# ----------------------------------------------------------------------------- per run

def analyse_run(d, stem, db, tol, settle_s):
    h, c, s = load_run(d, stem)
    t = h.t.values
    U = deadband(h[AXES].values.astype(float), db)
    vx, vz = command_velocity(h.assign(inRoll=U[:, 1], inPitch=U[:, 0]))
    starts, axes = detect_adjustments(t, U, tol, settle_s)

    pid, trial = stem.split("_")[0], int(stem.split("_")[1][1:])
    base = dict(run=stem, pid=pid, trial=trial, condition=s["condition"], nCorrect=s.get("nCorrect"))

    legs, prev_exit = [], t[0]
    for k, (g, enter, exit_) in enumerate(visits(s, c), start=1):
        m = window_metrics(t, U, vx, vz, starts, axes, prev_exit, enter)
        if m is not None:
            x0, z0 = pos_at(c, prev_exit)
            straight = dist_to_tile(x0, z0, g)
            act = achieved_len(c, prev_exit, enter)
            m.update(base, phase="transit", leg=k, hat=g["hat"], straight_m=straight,
                     act_len_m=act,
                     adj_per_100m=m["adjustments"] / straight * 100.0 if straight > 5 else np.nan,
                     cmd_path_ratio=m["cmd_len_m"] / straight if straight > 5 else np.nan,
                     act_path_ratio=act / straight if straight > 5 else np.nan)
            legs.append(m)
        m = window_metrics(t, U, vx, vz, starts, axes, enter, max(exit_, enter + 1e-3))
        if m is not None:
            m.update(base, phase="search", leg=k, hat=g["hat"])
            legs.append(m)
        prev_exit = exit_
    legs = pd.DataFrame(legs)

    tr = legs[legs.phase == "transit"]
    run = dict(base)
    dur = tr.dur_s.sum()
    run.update(
        transit_s=dur,
        search_s=legs[legs.phase == "search"].dur_s.sum(),
        adjustments=int(tr.adjustments.sum()),
        **{f"adj_{a}": int(tr[f"adj_{a}"].sum()) for a in AXIS_NAMES},
        adj_per_min=tr.adjustments.sum() / dur * 60.0,
        adj_per_100m=tr.adjustments.sum() / tr.straight_m.sum() * 100.0,
        mean_hold_s=dur / max(tr.adjustments.sum(), 1),
        stick_tv_per_min=(tr.stick_tv_per_min * tr.dur_s).sum() / dur,
        yaw_duty=(tr.yaw_duty * tr.dur_s).sum() / dur,
        lat_duty=(tr.lat_duty * tr.dur_s).sum() / dur,
        course_turn_per_min=tr.course_turn_deg.sum() / dur * 60.0,
        # length-weighted mean of the per-leg straightness
        cmd_straightness=tr.cmd_disp_m.sum() / tr.cmd_len_m.sum(),
        straight_m=tr.straight_m.sum(),
        cmd_path_ratio=tr.cmd_len_m.sum() / tr.straight_m.sum(),
        act_path_ratio=tr.act_len_m.sum() / tr.straight_m.sum(),
        adj_per_min_all=len(starts) / (t[-1] - t[0]) * 60.0,
    )
    return run, legs


# ----------------------------------------------------------------------------- output

SHOW = ["adj_per_min", "adj_per_100m", "mean_hold_s", "stick_tv_per_min", "yaw_duty", "lat_duty",
        "course_turn_per_min", "cmd_straightness", "cmd_path_ratio", "act_path_ratio"]


PANELS = {
    "adj_per_min": "Adjustments / min",
    "adj_per_100m": "Adjustments / 100 m progress",
    "mean_hold_s": "Mean command hold (s)",
    "course_turn_per_min": "Commanded turning (deg / min)",
    "cmd_straightness": "Command straightness (1 = one heading)",
    "act_path_ratio": "Path Curvature",   # achieved path length / straight-line distance
}


def plot(runs, out, cols, shape):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = ["SingleDrone", "Swarm"]
    colours = {"SingleDrone": "#d1603d", "Swarm": "#3d7dd1"}
    fig, axs = plt.subplots(*shape, figsize=(4 * shape[1], 4 * shape[0]), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, col in zip(axs.flat, cols):
        title = PANELS[col]
        data = [runs[runs.condition == cond][col].dropna().values for cond in conds]
        bp = ax.boxplot(data, positions=range(len(conds)), widths=0.6, patch_artist=True, showfliers=False)
        for patch, cond in zip(bp["boxes"], conds):
            patch.set(facecolor=colours[cond], alpha=0.35)
        for med in bp["medians"]:
            med.set(color="black")
        for x, (cond, v) in enumerate(zip(conds, data)):
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, color=colours[cond], zorder=3, s=25)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c], alpha=0.6) for c in conds]
    axs.flat[0].legend(handles, conds, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[info] saved {out}")


def boxplot(legs, out):
    """Adjustments per minute and per 100 m, one observation per transit leg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = legs[legs.phase == "transit"]
    conds = ["SingleDrone", "Swarm"]
    colours = {"SingleDrone": "#d1603d", "Swarm": "#3d7dd1"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, (col, title) in zip(axs, [("adj_per_min", "Adjustments per minute of transit"),
                                      ("adj_per_100m", "Adjustments per 100 m of straight-line progress")]):
        data = [tr[tr.condition == c][col].dropna().values for c in conds]
        bp = ax.boxplot(data, positions=range(len(conds)), widths=0.6, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], conds):
            patch.set(facecolor=colours[c], alpha=0.35)
        for med in bp["medians"]:
            med.set(color="black")
        rng = np.random.default_rng(0)
        for p, v, c in zip(range(len(conds)), data, conds):
            ax.scatter(p + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=colours[c], zorder=3)
        ax.set_xticks(range(len(conds)), [f"{c}\n(n = {len(v)} legs)" for c, v in zip(conds, data)])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Pilot adjustments per transit leg ({', '.join(sorted(tr.pid.unique()))}; dots = legs)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[info] saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--pid", nargs="*", default=["AABB", "AABC"])
    ap.add_argument("--deadband", type=float, default=0.1, help="stick |u| below this counts as neutral")
    ap.add_argument("--tol", type=float, default=0.2, help="stick move that counts as an adjustment")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds steady before a new hold latches")
    ap.add_argument("--include-practice", action="store_true",
                    help="keep each participant's first trial of each condition (dropped as practice by default)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    stems = []
    for sj in sorted(glob.glob(os.path.join(a.dir, "*_session.json"))):
        stem = os.path.basename(sj)[: -len("_session.json")]
        if stem.split("_")[0].upper() in {p.upper() for p in a.pid} and \
                os.path.exists(os.path.join(a.dir, stem + "_head.csv")):
            stems.append(stem)
    if not stems:
        raise SystemExit(f"no runs for {a.pid} in {a.dir}")

    runs, legs = [], []
    for stem in stems:
        r, l = analyse_run(a.dir, stem, a.deadband, a.tol, a.settle)
        runs.append(r)
        legs.append(l)
    runs = pd.DataFrame(runs).sort_values(["pid", "trial"])
    legs = pd.concat(legs, ignore_index=True)

    # The first trial a participant flies in each condition is practice.
    first = runs.groupby(["pid", "condition"]).trial.transform("min")
    practice = set(runs.run[runs.trial == first])
    if not a.include_practice:
        print("[info] dropping practice runs: " + ", ".join(sorted(practice)))
        runs = runs[~runs.run.isin(practice)]
        legs = legs[~legs.run.isin(practice)]

    runs.to_csv(os.path.join(a.dir, "command_metrics_runs.csv"), sep=";", index=False)
    legs.to_csv(os.path.join(a.dir, "command_metrics_legs.csv"), sep=";", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"\ndeadband={a.deadband} tol={a.tol} settle={a.settle}s  (transit phases only)\n")
    print(runs[["pid", "trial", "condition", "transit_s", "adjustments"] + SHOW].round(2).to_string(index=False))
    print("\nAdjustments by axis (transit):")
    print(runs[["pid", "trial", "condition"] + [f"adj_{x}" for x in AXIS_NAMES]].to_string(index=False))
    print("\nMean per participant x condition:")
    print(runs.groupby(["pid", "condition"])[SHOW].mean().round(2).to_string())
    print("\nMean per condition:")
    print(runs.groupby("condition")[SHOW].mean().round(2).to_string())

    if not a.no_plot:
        os.makedirs(os.path.join(a.dir, "plots"), exist_ok=True)
        plot(runs, os.path.join(a.dir, "plots", "command_metrics.png"), list(PANELS), (2, 3))
        plot(runs, os.path.join(a.dir, "plots", "command_metrics_summary.png"),
             ["adj_per_min", "course_turn_per_min", "act_path_ratio"], (1, 3))
        boxplot(legs, os.path.join(a.dir, "plots", "command_metrics_box.png"))


if __name__ == "__main__":
    main()
