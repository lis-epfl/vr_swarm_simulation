"""
Quick-look plotter for ExperimentRecorder output (city search task).

Reads the ';'-delimited CSVs + session JSON written by ExperimentRecorder.cs and draws a
top-down (XZ) trajectory map plus an altitude-vs-time panel, so you can eyeball a run.

Because it's easy to forget to re-label between runs, the same participant+trial often has
several recorded sessions (distinguished only by their timestamp). This script never silently
guesses: if more than one run matches, it prints a numbered table and lets you pick one
(--run N), plot them all (--all), or target an exact file (--stem).

Usage:
    python plot_experiment.py                          # AAAA trial 1 (lists if ambiguous)
    python plot_experiment.py --pid ERIC --trial 7     # pick a participant/trial
    python plot_experiment.py --pid ERIC --trial 7 --list     # just list the matching runs
    python plot_experiment.py --pid ERIC --trial 7 --run 2    # plot the 3rd run in the table
    python plot_experiment.py --pid ERIC --trial 7 --all      # a PNG for every match
    python plot_experiment.py --stem <path_without_drones.csv_suffix>   # exact run
    python plot_experiment.py --dir "D:/some/experiment"

Two PNGs (<run>_trajectory.png and <run>_altitude.png) are always written into a 'plots'
subfolder of the data folder; windows also open for a single selected run unless --no-show.

Default data folder is Unity's persistentDataPath for this project:
    %USERPROFILE%/AppData/LocalLow/UAVS@BERKELEY/DroneSim/experiment
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def default_dir():
    return os.path.join(os.path.expanduser("~"), "AppData", "LocalLow",
                        "UAVS@BERKELEY", "DroneSim", "experiment")


def list_runs(exp_dir, pid, trial):
    """All run stems for a participant+trial, oldest→newest (by file mtime)."""
    pattern = os.path.join(exp_dir, f"{pid}_t{trial}_*_drones.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return [m[: -len("_drones.csv")] for m in matches]


def run_summary(stem):
    """Short human-readable descriptor of a run, pulling from its session JSON when present."""
    ts = os.path.basename(stem).split("_")[-1]  # HHmmss part of the timestamp
    session_path = stem + "_session.json"
    if not os.path.exists(session_path):
        return f"timestamp {ts}  (no session.json - aborted before finalize)"
    try:
        with open(session_path) as f:
            s = json.load(f)
    except (OSError, json.JSONDecodeError):
        return f"timestamp {ts}  (unreadable session.json)"
    goals = s.get("goals", [])
    answered = sum(1 for g in goals if g.get("answered"))
    return (f"timestamp {ts}  cond={s.get('condition'):<11} "
            f"drones={s.get('droneCount'):<3} dur={s.get('durationSec', 0):6.1f}s "
            f"answered={answered}/{s.get('nGoals', len(goals))} "
            f"correct={s.get('nCorrect', 0)}")


def print_run_table(stems):
    print(f"[info] {len(stems)} run(s) match:")
    for i, stem in enumerate(stems):
        print(f"  [{i}] {run_summary(stem)}")


def resolve_stem(args):
    """Turn CLI args into a list of run stems to plot (may be one or many)."""
    if args.stem:
        s = args.stem
        return [s[: -len("_drones.csv")] if s.endswith("_drones.csv") else s]

    stems = list_runs(args.dir, args.pid, args.trial)
    if not stems:
        raise SystemExit(f"No runs match {args.pid} trial {args.trial} in {args.dir}")

    if args.list:
        print_run_table(stems)
        raise SystemExit(0)

    if args.all:
        return stems

    if len(stems) == 1:
        return stems

    # Ambiguous: show the table and require an explicit pick — never silently guess a run.
    print_run_table(stems)
    if args.run is not None:
        if not 0 <= args.run < len(stems):
            raise SystemExit(f"--run {args.run} out of range 0..{len(stems) - 1}")
        return [stems[args.run]]

    # Offer an interactive prompt, but fall back cleanly (no crash) when there's no console.
    try:
        raw = input(f"Select run [0-{len(stems) - 1}, blank = newest, q = quit]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = None
    if raw is None:
        raise SystemExit("Multiple runs match - re-run with --run N, --all, or --stem to choose.")
    if raw.lower() in ("q", "quit"):
        raise SystemExit(0)
    if raw == "":
        return [stems[-1]]
    try:
        idx = int(raw)
        if not 0 <= idx < len(stems):
            raise ValueError
    except ValueError:
        raise SystemExit(f"Invalid selection '{raw}' — expected 0..{len(stems) - 1}.")
    return [stems[idx]]


def load_csv(path):
    return pd.read_csv(path, sep=";") if os.path.exists(path) else None


def plot_run(stem, show):
    print(f"[info] loading {os.path.basename(stem)}_*")
    drones = load_csv(stem + "_drones.csv")
    head = load_csv(stem + "_head.csv")
    walkers = load_csv(stem + "_walkers.csv")
    events = load_csv(stem + "_events.csv")
    session = None
    if os.path.exists(stem + "_session.json"):
        with open(stem + "_session.json") as f:
            session = json.load(f)

    if drones is None or drones.empty:
        print(f"[warn] {os.path.basename(stem)}_drones.csv missing or empty — skipping.")
        return

    if session:
        print(f"[session] pid={session.get('participantId')} trial={session.get('trialNumber')} "
              f"condition={session.get('condition')} drones={session.get('droneCount')} "
              f"nCorrect={session.get('nCorrect')}/{session.get('nGoals')} "
              f"duration={session.get('durationSec', 0):.1f}s")
        for g in session.get("goals", []):
            print(f"  goal {g['goalIndex']}: outcome={g.get('outcome') or 'n/a':<9} "
                  f"decisionT={g.get('decisionTimeSec', -1):.1f}s "
                  f"swarm->walker={g.get('swarmToWalkerDist', -1):.1f}m")

    # Larger axis text (titles, labels, ticks). Set before the figures are created.
    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 20,
    })

    # Two independent figures so each graph is written to its own file.
    fig_traj, ax = plt.subplots(figsize=(10, 7))
    fig_alt, ax2 = plt.subplots(figsize=(8, 6))

    # Shift all XZ so the swarm start sits at (0, 0): origin = mean of each drone's earliest sample.
    first = drones.sort_values("t").groupby("droneId").first()
    ox, oz = first["gtX"].mean(), first["gtZ"].mean()

    for did, g in drones.groupby("droneId"):
        ax.plot(g["gtX"] - ox, g["gtZ"] - oz, lw=1, alpha=0.8, label=f"drone {did}")
        ax.scatter(g["gtX"].iloc[0] - ox, g["gtZ"].iloc[0] - oz, s=20, marker="o",
                   color="green", zorder=5)

    # Mark the (0, 0) start position.
    ax.scatter(0, 0, s=90, marker="o", color="green", edgecolor="black", zorder=7)
    ax.annotate("start", (0, 0), textcoords="offset points", xytext=(8, 8),
                fontsize=11, fontweight="bold", color="green")

    if head is not None and not head.empty:
        ax.plot(head["headX"] - ox, head["headZ"] - oz, "k--", lw=1.2, alpha=0.7, label="head")

    if walkers is not None and not walkers.empty:
        for gi, w in walkers.groupby("goalIndex"):
            ax.plot(w["specialX"] - ox, w["specialZ"] - oz, lw=1, alpha=0.5)

    if session:
        for g in session.get("goals", []):
            ax.scatter(g["goalX"] - ox, g["goalZ"] - oz, s=220, marker="*", color="gold",
                       edgecolor="black", zorder=6)
            ax.annotate(f"goal {g['goalIndex']}", (g["goalX"] - ox, g["goalZ"] - oz),
                        textcoords="offset points", xytext=(6, 6))
            ax.scatter(g["specialSpawnX"] - ox, g["specialSpawnZ"] - oz, s=60, marker="X",
                       color="red", zorder=6)
            if g.get("answered"):
                ax.scatter(g["centroidAtAnswerX"] - ox, g["centroidAtAnswerZ"] - oz, s=90,
                           marker="P", color="blue", zorder=6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Top-down trajectories (XZ)")
    ax.set_xlim(-100, 900)
    ax.set_ylim(-300, 300)
    ax.set_aspect("equal", adjustable="box")  # equal data scale, honouring the fixed limits
    ax.legend(loc="upper right", fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3)

    for did, g in drones.groupby("droneId"):
        ax2.plot(g["t"], g["gtY"], lw=1, alpha=0.8)
    if events is not None and not events.empty:
        for _, e in events[events["eventType"] == "identify"].iterrows():
            ax2.axvline(e["t"], color="blue", ls=":", alpha=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("altitude Y (m)")
    ax2.set_title("Altitude vs time")
    ax2.set_xlim(0, 700)
    ax2.set_ylim(top=150)
    ax2.grid(True, alpha=0.3)

    run_name = os.path.basename(stem)
    fig_traj.suptitle(run_name)
    fig_alt.suptitle(run_name)
    fig_traj.tight_layout()
    fig_alt.tight_layout()

    # Write each graph to its own PNG in a 'plots' subfolder, kept separate from the data files.
    plots_dir = os.path.join(os.path.dirname(stem), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for fig, suffix in ((fig_traj, "_trajectory.png"), (fig_alt, "_altitude.png")):
        out = os.path.join(plots_dir, run_name + suffix)
        fig.savefig(out, dpi=130)
        print(f"[info] saved {out}")
        if not show:
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", default="AAAA", help="participant id (default AAAA)")
    ap.add_argument("--trial", default="1", help="trial number (default 1)")
    ap.add_argument("--dir", default=default_dir(), help="experiment output folder")
    ap.add_argument("--stem", default=None,
                    help="exact run stem or a *_drones.csv path, overriding --pid/--trial")
    ap.add_argument("--list", action="store_true",
                    help="list the runs matching --pid/--trial and exit")
    ap.add_argument("--run", type=int, default=None,
                    help="when several runs match, pick this index from the table")
    ap.add_argument("--all", action="store_true",
                    help="plot every run matching --pid/--trial")
    ap.add_argument("--no-show", action="store_true",
                    help="don't open a window; just write the PNG(s)")
    args = ap.parse_args()

    stems = resolve_stem(args)
    # A PNG is always written. Only pop up a window for a single, explicitly-selected run
    # (never for --all, which would open many blocking windows).
    show = not args.no_show and not args.all and len(stems) == 1
    for stem in stems:
        plot_run(stem, show)
    if show:
        plt.show()


if __name__ == "__main__":
    main()
