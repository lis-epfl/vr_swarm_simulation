"""Time spent near each goal patch, for a sweep of "near" distances.

    python goal_proximity.py [--dir DIR] [--pid AABB AABC] [--dist 0 10 15 20 30 40] [--out FILE]
                             [--include-practice] [--hat Cap] [--no-plot]

Position is the centroid of the alive drones (the single drone itself in SingleDrone runs). Its distance
from a goal patch is the horizontal distance to the nearest point of the goal's tile -- a square of
half-width TILE_HALF about the goal centre, i.e. the block plus half of each surrounding street -- and is
0 anywhere over the tile. `near{D}Sec` is the total time that distance was <= D metres, so D = 0 is the
time over the tile (the visit test apply_answers.py uses) and D = 90.83 would take in the whole ring of
neighbouring tiles.

Transit uses one of those distances, FOCUS_D (15 m: the patch plus the whole street around it, up to
the opposite kerb), as the goal's zone. Goals are ordered by first entry into their zone, and
`transitSec` is the leg ending at that goal: from last leaving the previous goal's zone before this
entry (the session start for the first leg, so it includes take-off) to first entering this one --
command_metrics.py's transit phase, with 15 m zones instead of the bare tile.

Each participant's first trial in each condition is practice and is dropped, as in command_metrics.py
(--include-practice keeps it). Writes into the data folder's plots/, in command_metrics.py's style:
goal_proximity.png (one panel per distance, one dot per goal), goal_proximity_15m.png (FOCUS_D only)
and transit_15m.png (one dot per transit leg). --hat keeps one hat's goal only and writes just
goal_proximity_<hat>.png (its FOCUS_D time and the transit leg ending at it).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DEFAULT_DIR = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\UAVS@BERKELEY\DroneSim\experiment")
TILE_HALF = 90.83 / 2
DISTS = [0, 10, 15, 20, 30, 40]   # > 45.4 would let two goals (>= 90.8 m apart) share a sample
FOCUS_D = 15
CONDS = ["SingleDrone", "Swarm"]
COLOURS = {"SingleDrone": "#d1603d", "Swarm": "#3d7dd1"}


def centroid_track(d, stem, duration):
    dr = pd.read_csv(os.path.join(d, stem + "_drones.csv"), sep=";")
    c = dr[dr.alive == 1].groupby("t")[["gtX", "gtZ"]].mean()
    t = c.index.to_numpy()
    dt = np.minimum(np.diff(np.append(t, duration)), 0.5)   # a sample stands for the time until the next
    return t, c.gtX.to_numpy(), c.gtZ.to_numpy(), dt


def analyse(d, stem, dists):
    j = json.load(open(os.path.join(d, stem + "_session.json"), encoding="utf-8"))
    t, x, z, dt = centroid_track(d, stem, j["durationSec"])
    rows, zones = [], []
    for g in j["goals"]:
        ex = np.maximum(np.abs(x - g["goalX"]) - TILE_HALF, 0)
        ez = np.maximum(np.abs(z - g["goalZ"]) - TILE_HALF, 0)
        dist = np.hypot(ex, ez)
        zones.append(dist <= FOCUS_D)
        row = dict(run=stem, pid=stem.split("_")[0], trial=int(stem.split("_")[1][1:]), condition=j["condition"],
                   hat=g["hat"].replace("Walker", ""), visitOrder=g.get("visitOrder", -1),
                   minDistM=round(float(dist.min()), 1))
        for D in dists:
            row[f"near{D:g}Sec"] = round(float(dt[dist <= D].sum()), 1)
        rows.append(row)

    # Transit legs between the FOCUS_D zones, in order of first entry.
    entry = [int(np.argmax(zn)) if zn.any() else None for zn in zones]
    order = sorted((k for k in range(len(zones)) if entry[k] is not None), key=lambda k: entry[k])
    prev = None
    for leg, k in enumerate(order, start=1):
        i = entry[k]
        if prev is None:
            start = t[0]
        else:
            inside = np.flatnonzero(zones[prev][:i])
            start = t[inside[-1]] + dt[inside[-1]]
        rows[k].update(zoneOrder=leg, transitSec=round(float(t[i] - start), 1))
        prev = k
    return rows


def box_panel(ax, df, col, rng):
    data = [df[df.condition == cond][col].dropna().values for cond in CONDS]
    bp = ax.boxplot(data, positions=range(len(CONDS)), widths=0.6, patch_artist=True, showfliers=False)
    for patch, cond in zip(bp["boxes"], CONDS):
        patch.set(facecolor=COLOURS[cond], alpha=0.35)
    for med in bp["medians"]:
        med.set(color="black")
    for x, (cond, v) in enumerate(zip(CONDS, data)):
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, color=COLOURS[cond], zorder=3, s=25)
    ax.set_xticks(range(len(CONDS)), CONDS)
    ax.grid(axis="y", alpha=0.3)


def plot_single(df, col, ylabel, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    box_panel(ax, df, col, np.random.default_rng(0))
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 300)                                      # fixed, so the dwell and transit figures compare
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[c], alpha=0.6) for c in CONDS]
    ax.legend(handles, CONDS, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def plot_hat(df, hat, out):
    """One goal's hat only: the FOCUS_D dwell and the transit leg ending at it, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(f"near{FOCUS_D:g}Sec", f"Time within {FOCUS_D:g} m of the {hat} patch"),
              ("transitSec", f"Transit to the {hat} goal")]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axs.flat, panels):
        box_panel(ax, df, col, rng)
        ax.set_xticks(range(len(CONDS)),
                      [f"{c}\n(n = {df[df.condition == c][col].notna().sum()})" for c in CONDS])
        ax.set_title(title)
        ax.set_ylabel("Time (s)")
        ax.set_ylim(bottom=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[c], alpha=0.6) for c in CONDS]
    axs.flat[0].legend(handles, CONDS, loc="upper right")
    fig.suptitle(f"{hat} goal only ({', '.join(sorted(df.pid.unique()))}; dots = goals)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def plot(df, dists, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 3
    nrow = -(-len(dists) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for ax, D in zip(axs.flat, dists):
        box_panel(ax, df, f"near{D:g}Sec", rng)
        ax.set_title("Over the goal patch" if D == 0 else f"Within {D:g} m of the goal patch")
    top = max(df[f"near{D:g}Sec"].max() for D in dists)
    axs.flat[0].set_ylim(0, top * 1.05)                      # shared: one scale so the panels compare
    for ax in axs[:, 0]:
        ax.set_ylabel("Time near goal patch (s)")
    for ax in list(axs.flat)[len(dists):]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[c], alpha=0.6) for c in CONDS]
    axs.flat[0].legend(handles, CONDS, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--pid", nargs="*", default=[])
    p.add_argument("--dist", nargs="*", type=float, default=DISTS)
    p.add_argument("--out", default="")
    p.add_argument("--include-practice", action="store_true",
                   help="keep each participant's first trial of each condition (dropped as practice by default)")
    p.add_argument("--hat", default="",
                   help="keep only the goal with this hat (Cap, Cowboy, Bucket) and plot it alone, as "
                        "goal_proximity_<hat>.png in place of the three all-goal plots")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    rows = []
    for s in sorted(glob.glob(os.path.join(a.dir, "*_session.json"))):
        stem = os.path.basename(s)[: -len("_session.json")]
        if a.pid and stem.split("_")[0].upper() not in {x.upper() for x in a.pid}:
            continue
        rows += analyse(a.dir, stem, a.dist)
    df = pd.DataFrame(rows).sort_values(["pid", "trial", "visitOrder"])
    if not a.include_practice:
        first = df.groupby(["pid", "condition"]).trial.transform("min")
        print("dropping practice runs: " + ", ".join(sorted(df.run[df.trial == first].unique())))
        df = df[df.trial != first]
    if a.hat:
        df = df[df.hat.str.lower() == a.hat.lower()]
        if df.empty:
            raise SystemExit(f"no goals with hat {a.hat!r}")
    pd.set_option("display.width", 250)
    print(df.drop(columns=["run"]).to_string(index=False))
    cols = [c for c in df.columns if c.startswith("near")] + ["transitSec"]
    print("\nmedian seconds per goal / leg")
    print(df.groupby("condition")[cols].median().to_string())
    if a.out:
        df.to_csv(a.out, sep=";", index=False)
        print("wrote", a.out)
    if not a.no_plot and a.hat:
        hat = df.hat.iloc[0]
        os.makedirs(os.path.join(a.dir, "plots"), exist_ok=True)
        plot_hat(df, hat, os.path.join(a.dir, "plots", f"goal_proximity_{hat.lower()}.png"))
    elif not a.no_plot:
        os.makedirs(os.path.join(a.dir, "plots"), exist_ok=True)
        plot(df, a.dist, os.path.join(a.dir, "plots", "goal_proximity.png"))
        plot_single(df, f"near{FOCUS_D:g}Sec", f"Time within {FOCUS_D:g} m of goal patch (s)",
                    os.path.join(a.dir, "plots", f"goal_proximity_{FOCUS_D:g}m.png"))
        plot_single(df, "transitSec", "Transit time between goals (s)",
                    os.path.join(a.dir, "plots", f"transit_{FOCUS_D:g}m.png"))
