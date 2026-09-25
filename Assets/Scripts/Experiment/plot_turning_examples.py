"""Trajectory maps of two contrasting runs: the single-drone flight with the MOST commanded turning and
the swarm flight with the LEAST, by command_metrics.py's course_turn_per_min (total |change| of the
commanded course per minute of transit). One figure per run, drawn at the same scale over the city.

What is drawn: the building footprints actually in the scene for that run (hat_visibility.py's set --
the city colliders minus the three tiles the goals replaced, plus each goal patch's own buildings), the
goal tiles numbered in visit order, the flown path (single drone: the drone; swarm: each drone faint and
the alive-drone centroid bold), and small arrows every --arrow-s seconds of transit giving the direction
the pilot was *commanding* at that moment (only while |v_cmd| > 0.3 maxSpeed, the same gate the turning
metric uses; none while searching, which the metric does not count either). The arrows are what the
metric measures; the path is what the vehicles did with it.

Runs are chosen after command_metrics' practice drop (first trial per participant and condition), with
its default thresholds. --single / --swarm name a run stem instead.

Usage:
    python plot_turning_examples.py                       # AABB + AABC, default data folder
    python plot_turning_examples.py --pid AABB AABC ERIC
    python plot_turning_examples.py --single AABB_t5_SingleDrone_20260923_175457
    python plot_turning_examples.py --all                 # every run, ranked within its condition

Writes plots/<run>_turning_map.png for each run drawn. All maps from one invocation share one scale.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import command_metrics as cm
from hat_visibility import load_city, scene_boxes

COLOURS = {"SingleDrone": "#d1603d", "Swarm": "#3d7dd1"}   # as command_metrics.py
MARGIN_M = 40.0


def run_table(d, pids, include_practice):
    stems = []
    for sj in sorted(glob.glob(os.path.join(d, "*_session.json"))):
        stem = os.path.basename(sj)[: -len("_session.json")]
        if stem.split("_")[0].upper() in {p.upper() for p in pids} and \
                os.path.exists(os.path.join(d, stem + "_head.csv")):
            stems.append(stem)
    if not stems:
        raise SystemExit(f"no runs for {pids} in {d}")
    runs = pd.DataFrame([cm.analyse_run(d, s, 0.1, 0.2, 0.5)[0] for s in stems])
    if not include_practice:
        first = runs.groupby(["pid", "condition"]).trial.transform("min")
        runs = runs[runs.trial != first]
    return runs.set_index("run")


def footprint(c, ax):
    """XZ outline of an oriented box: the two half-axes that are closest to horizontal."""
    vert = np.abs(ax[:, 1]) / np.linalg.norm(ax, axis=1)
    a1, a2 = ax[np.argsort(vert)[:2]]
    return np.array([[p[0], p[2]] for p in
                     (c + a1 + a2, c + a1 - a2, c - a1 - a2, c - a1 + a2)])


def load_track(d, stem):
    h, c, s = cm.load_run(d, stem)
    dr = pd.read_csv(os.path.join(d, stem + "_drones.csv"), sep=";")
    return h, c, s, dr[dr.alive == 1]


def command_arrows(h, c, every_s, windows):
    """(x, z, ux, uz) at the centroid every every_s seconds, pointing along the commanded course.

    Only inside windows [(t0, t1)] -- the transit legs, because that is all course_turn_per_min
    measures; search is hovering and peering, and its arrows would bury the tile.
    """
    U = cm.deadband(h[cm.AXES].values.astype(float), 0.1)
    vx, vz = cm.command_velocity(h.assign(inRoll=U[:, 1], inPitch=U[:, 0]))
    t = h.t.values
    out = []
    for tk in np.arange(t[0], t[-1], every_s):
        if not any(t0 <= tk < t1 for t0, t1 in windows):
            continue
        i = min(np.searchsorted(t, tk), len(t) - 1)
        spd = np.hypot(vx[i], vz[i])
        if spd > 0.3 * cm.MAX_SPEED:
            x, z = cm.pos_at(c, t[i])
            out.append((x, z, vx[i] / spd, vz[i] / spd))
    return np.array(out).reshape(-1, 4)


def extent(tracks):
    xs, zs = [], []
    for _, c, s, _ in tracks:
        xs += list(c.gtX) + [g["goalX"] + sgn * cm.TILE_HALF for g in s["goals"] for sgn in (-1, 1)]
        zs += list(c.gtZ) + [g["goalZ"] + sgn * cm.TILE_HALF for g in s["goals"] for sgn in (-1, 1)]
    return (min(xs) - MARGIN_M, max(xs) + MARGIN_M), (min(zs) - MARGIN_M, max(zs) + MARGIN_M)


def plot_map(stem, track, row, city, kerbs, lims, arrow_s, out, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    h, c, s, dr = track
    colour = COLOURS[row.condition]
    (x0, x1), (z0, z1) = lims
    fig, ax = plt.subplots(figsize=(9, 9 * (z1 - z0) / (x1 - x0) + 0.9))

    polys = [footprint(ctr, a) for _, ctr, a in scene_boxes(s["goals"], city, kerbs)]
    ax.add_collection(PolyCollection(polys, facecolor="0.82", edgecolor="0.62", lw=0.4, zorder=1))

    transit, prev_exit = [], h.t.values[0]
    for k, (g, enter, exit_) in enumerate(cm.visits(s, c), start=1):
        transit.append((prev_exit, enter))
        prev_exit = exit_
        gx, gz, r = g["goalX"], g["goalZ"], cm.TILE_HALF
        ax.add_patch(plt.Rectangle((gx - r, gz - r), 2 * r, 2 * r, facecolor="#e8c547", alpha=0.18,
                                   edgecolor="#a8871a", lw=1.2, zorder=2))
        ax.text(gx - r, gz + r + 3, f"goal {k}", ha="left", va="bottom", fontsize=11,
                fontweight="bold", color="0.2", zorder=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))

    if row.condition == "Swarm":
        for _, g in dr.groupby("droneId"):
            ax.plot(g.gtX, g.gtZ, color=colour, lw=0.6, alpha=0.3, zorder=3)
        ax.plot(c.gtX, c.gtZ, color=colour, lw=2.2, zorder=4, label="swarm centroid")
        ax.plot([], [], color=colour, lw=0.6, alpha=0.5, label="individual drones")
    else:
        ax.plot(c.gtX, c.gtZ, color=colour, lw=2.2, zorder=4, label="drone")

    arr = command_arrows(h, c, arrow_s, transit)
    if len(arr):
        ax.quiver(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], angles="xy", scale_units="xy",
                  scale=1 / 12.0, width=0.0028, color="0.15", zorder=5,
                  label=f"commanded direction in transit (every {arrow_s:g} s)")

    ax.scatter(c.gtX.iloc[0], c.gtZ.iloc[0], s=110, marker="o", color="white", edgecolor="black",
               lw=1.5, zorder=7, label="start")
    ax.scatter(c.gtX.iloc[-1], c.gtZ.iloc[-1], s=110, marker="s", color="black", zorder=7, label="end")

    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_title(f"{headline}\n{stem}\n"
                 f"commanded turning {row.course_turn_per_min:.0f} deg/min · command straightness "
                 f"{row.cmd_straightness:.2f} · path / straight-line {row.act_path_ratio:.2f}",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[info] saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=cm.DEFAULT_DIR)
    ap.add_argument("--pid", nargs="*", default=["AABB", "AABC"])
    ap.add_argument("--single", help="single-drone run stem to plot instead of the most-turning one")
    ap.add_argument("--swarm", help="swarm run stem to plot instead of the least-turning one")
    ap.add_argument("--arrow-s", type=float, default=4.0, help="seconds between command arrows")
    ap.add_argument("--all", action="store_true",
                    help="map every analysed run (one figure each, all at one scale) instead of the two picks")
    ap.add_argument("--include-practice", action="store_true")
    a = ap.parse_args()

    runs = run_table(a.dir, a.pid, a.include_practice)
    print(runs[["condition", "course_turn_per_min", "cmd_straightness", "act_path_ratio"]]
          .sort_values(["condition", "course_turn_per_min"]).round(2).to_string())

    single = a.single or runs[runs.condition == "SingleDrone"].course_turn_per_min.idxmax()
    swarm = a.swarm or runs[runs.condition == "Swarm"].course_turn_per_min.idxmin()
    picks = [(single, "Single drone, highest commanded turning"),
             (swarm, "Swarm, lowest commanded turning")]
    if a.all:
        picks = []
        for cond, label in (("SingleDrone", "Single drone"), ("Swarm", "Swarm")):
            sub = runs[runs.condition == cond].sort_values("course_turn_per_min", ascending=False)
            picks += [(stem, f"{label}, commanded turning rank {k} of {len(sub)} (1 = most)")
                      for k, stem in enumerate(sub.index, start=1)]
    elif a.single or a.swarm:
        picks = [(st, f"{runs.loc[st].condition} run") if st in (a.single, a.swarm) else (st, hl)
                 for st, hl in picks]
    for stem, _ in picks:
        if stem not in runs.index:
            raise SystemExit(f"{stem} is not among the analysed runs (practice runs need --include-practice)")

    city, kerbs = load_city()
    tracks = {stem: load_track(a.dir, stem) for stem, _ in picks}
    lims = extent(tracks.values())
    os.makedirs(os.path.join(a.dir, "plots"), exist_ok=True)
    for stem, headline in picks:
        print(f"[pick] {headline}: {stem} ({runs.loc[stem].course_turn_per_min:.0f} deg/min)")
        plot_map(stem, tracks[stem], runs.loc[stem], city, kerbs, lims, a.arrow_s,
                 os.path.join(a.dir, "plots", f"{stem}_turning_map.png"), headline)


if __name__ == "__main__":
    main()
