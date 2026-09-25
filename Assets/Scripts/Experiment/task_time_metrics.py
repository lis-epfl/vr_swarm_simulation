"""Task completion time metrics for the city search task (ExperimentRecorder output).

Companion to command_metrics.py: where that script asks how the pilot flew, this asks how long the
whole run took. Timing source, per run:
  totalTaskTime   Set by ExperimentRecorder itself at Finalize, as start -> last identify keypress.
                  Only meaningful when the experimenter pressed the identify keys live.
  durationSec     Session length (start -> Finalize), always recorded.
A run backfilled by apply_answers.py (identify keys not pressed, participant's spoken answers noted
by hand instead) never gets a totalTaskTime -- it stays 0, same as an aborted run with no answers at
all. So each run uses totalTaskTime when it is > 0 and falls back to durationSec otherwise; time_source
in the output says which one was used, so a mix of live and backfilled runs stays honest.

Each participant's first trial in each condition is practice, exactly as in command_metrics.py. Rather
than dropping it, this script reports both cuts side by side -- practice trials plausibly run long
(or erratically) simply from unfamiliarity, not from the condition -- so the "excluding practice"
panel is the one to trust for a condition comparison, and "all trials" shows what that first exposure
costs.

Usage:
    python task_time_metrics.py                       # AABB + AABC, default data folder
    python task_time_metrics.py --pid AABB AABC ERIC --dir D:/experiment

Writes task_time_runs.csv into the data folder and plots/task_time_summary.png, and prints a per-run
table plus per-condition summaries (n, total, mean, sd, se, min, max) for both trial sets.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "AppData", "LocalLow",
                           "UAVS@BERKELEY", "DroneSim", "experiment")
CONDS = ["SingleDrone", "Swarm"]


# ----------------------------------------------------------------------------- loading

def load_run(d, stem):
    s = json.load(open(os.path.join(d, stem + "_session.json"), encoding="utf-8"))
    pid, trial = stem.split("_")[0].upper(), int(stem.split("_")[1][1:])

    duration = float(s["durationSec"])
    total_task_time = float(s.get("totalTaskTime", 0.0) or 0.0)
    time_s, source = (total_task_time, "identify") if total_task_time > 0 else (duration, "duration")

    return dict(run=stem, pid=pid, trial=trial, condition=s["condition"],
                time_s=time_s, time_source=source, durationSec=duration, totalTaskTime=total_task_time,
                nCorrect=s.get("nCorrect"), nGoals=s.get("nGoals"))


# ----------------------------------------------------------------------------- summary

def summarize(df):
    g = df.groupby("condition").time_s
    out = g.agg(n="count", total="sum", mean="mean", sd="std", min="min", max="max")
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out.reindex([c for c in CONDS if c in out.index])[["n", "total", "mean", "sd", "se", "min", "max"]]


# ----------------------------------------------------------------------------- plot

def plot(runs_all, runs_trim, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"SingleDrone": "#d1603d", "Swarm": "#3d7dd1"}
    y_max = runs_all.time_s.max() * 1.08
    fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 4.5), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, df in zip(axs.flat, [runs_all, runs_trim]):
        data = [df[df.condition == cond].time_s.dropna().values for cond in CONDS]
        bp = ax.boxplot(data, positions=range(len(CONDS)), widths=0.6, patch_artist=True, showfliers=False)
        for patch, cond in zip(bp["boxes"], CONDS):
            patch.set(facecolor=colours[cond], alpha=0.35)
        for med in bp["medians"]:
            med.set(color="black")
        for x, (cond, v) in enumerate(zip(CONDS, data)):
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, color=colours[cond], zorder=3, s=25)
        ax.set_xticks(range(len(CONDS)), [f"{c}\n(n = {len(v)})" for c, v in zip(CONDS, data)])
        ax.set_ylabel("Task time (s)")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c], alpha=0.6) for c in CONDS]
    axs.flat[0].legend(handles, CONDS, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[info] saved {out}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--pid", nargs="*", default=["AABB", "AABC"])
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    stems = []
    for sj in sorted(glob.glob(os.path.join(a.dir, "*_session.json"))):
        stem = os.path.basename(sj)[: -len("_session.json")]
        if stem.split("_")[0].upper() in {p.upper() for p in a.pid}:
            stems.append(stem)
    if not stems:
        raise SystemExit(f"no runs for {a.pid} in {a.dir}")

    runs = pd.DataFrame([load_run(a.dir, stem) for stem in stems]).sort_values(["pid", "trial"])
    runs.to_csv(os.path.join(a.dir, "task_time_runs.csv"), sep=";", index=False)

    # The first trial a participant flies in each condition is practice.
    first = runs.groupby(["pid", "condition"]).trial.transform("min")
    practice = sorted(runs.run[runs.trial == first])
    print("[info] practice runs (first per participant x condition): " + ", ".join(practice))
    runs_trim = runs[~runs.run.isin(practice)]

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\nPer-run task time (s):\n")
    print(runs[["pid", "trial", "condition", "time_s", "time_source", "nCorrect", "nGoals"]]
          .round(2).to_string(index=False))

    print("\n--- All trials ---")
    print(summarize(runs).round(2).to_string())
    print("\n--- Excluding each participant's 1st trial per condition ---")
    print(summarize(runs_trim).round(2).to_string())

    if not a.no_plot:
        os.makedirs(os.path.join(a.dir, "plots"), exist_ok=True)
        plot(runs, runs_trim, os.path.join(a.dir, "plots", "task_time_summary.png"))


if __name__ == "__main__":
    main()
