"""Pilot-set swarm spread and camera pitch for the city search task (ExperimentRecorder output).

Companion to command_metrics.py: that script scores the flight sticks, this one scores the two
*setting* dials on the joystick -- the spread dial, which sets how far apart the drones fly, and the
gimbal dial, which tilts every drone's FPV camera. Both are position controls, not springs: the value
stays wherever the pilot left it, so what matters is when and how far the dial was turned and what it
was left at. The question is whether pilots re-configure for the task phase -- widen the swarm and
tilt the camera down to search a goal tile, and undo it to travel.

Signals (all ~10 Hz):
  spread   _head.csv inSpread: the dial value, written straight into the Olfati-Saber d_ref
           (SwarmAlgorithm.SetSwarmSpread), 0.4 = tightest .. 1.6 = widest. -1 means "no override":
           it appears before the first joystick packet and as single-sample dropouts mid-run, where
           the swarm keeps the last value, so it is forward-filled; the leading gap is back-filled
           with the first transmitted value (a position control was already at that position).
  spacing  _shape.csv meanNNm: the achieved mean nearest-neighbour distance in metres -- the
           formation's answer to the spread command, lagging it by several seconds. Needs >= 2 alive
           drones, so it is NaN in the SingleDrone condition. (The dial is still scored there: it
           does nothing to a lone drone, so any use of it is the pilot's, not the task's.)
  pitch    _head.csv gimbalPitch: FPVCameraScript.SharedPitch in degrees, 0 = level horizon,
           -90 = straight down, +60 = up; the dial maps linearly onto that range. Only logged from
           2026-09-24 on: earlier runs have no such column and every pitch metric is NaN for them.
  crashes  _drones.csv alive: the times drones died, drawn as red lines on the timeline plots. A
           drone-drone collision kills both drones on one tick, so it is one line, not two.

Phases are command_metrics.py's, imported from it so the two cannot disagree: transit leg k runs
from leaving the previous goal tile to entering goal tile k, search is the time inside it.

Metrics per run, for x in {spread, pitch} (units: d_ref for spread, degrees for pitch):
  x_adj             Discrete dial adjustments over the whole run, with command_metrics' hold/settle
                    detector on the one axis: an adjustment starts when the dial leaves its held
                    position by more than --spread-tol / --pitch-tol, and the hold re-latches once it
                    has stayed within half that for --settle s. One twist from tight to wide = 1.
  x_adj_per_min     ... per minute of the whole run.
  x_adj_per_min_transit / x_adj_per_min_search   ... per minute of each phase.
  x_tv_per_min      Total variation (sum |delta x|) per minute: how far the dial was turned in all,
                    insensitive to the adjustment thresholds.
  x_transit / x_search   Time-weighted mean setting over all transit / all search time.
  x_search_delta    x_search - x_transit: how much the pilot re-set the dial for searching.
  spacing_transit_m / spacing_search_m / spacing_search_delta_m   Same, for the achieved spacing.
  pitch_down_frac   Fraction of the run with the camera below --down degrees.

Each participant's first trial in each condition is practice and is dropped (--include-practice
keeps it), as in command_metrics.py.

Usage:
    python spread_pitch_metrics.py                       # AABB + AABC, default data folder
    python spread_pitch_metrics.py --pid AABB AABC ERIC --dir D:/experiment
    python spread_pitch_metrics.py --spread-tol 0.05 --pitch-tol 5    # sensitivity check

Writes spread_pitch_runs.csv, spread_pitch_legs.csv and plots/spread_pitch_*.png into the data folder,
and prints a per-run table and participant x condition / condition summaries.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from command_metrics import DEFAULT_DIR, detect_adjustments, load_run, visits

CONDS = ["SingleDrone", "Swarm"]
COLOURS = {"SingleDrone": "#d1603d", "Swarm": "#3d7dd1"}   # as command_metrics.py
DIALS = ["spread", "pitch"]
SPREAD_RANGE = (0.4, 1.6)       # joystick dial -> d_ref
PITCH_RANGE = (-90.0, 60.0)     # FPVCameraScript.MinPitch / MaxPitch


# ----------------------------------------------------------------------------- signals

def load_shape(d, stem):
    p = os.path.join(d, stem + "_shape.csv")
    return pd.read_csv(p, sep=";") if os.path.exists(p) else None


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


def crash_times(d, stem, merge_s=0.5):
    """Time of each crash: the last sample a dying drone was alive. Deaths within merge_s of each
    other are one crash -- a drone-drone collision parks both drones on the same tick."""
    p = os.path.join(d, stem + "_drones.csv")
    if not os.path.exists(p):
        return []
    dr = pd.read_csv(p, sep=";", usecols=["t", "droneId", "alive"]).sort_values(["droneId", "t"])
    a, ids = dr.alive.values, dr.droneId.values
    died = np.flatnonzero((ids[1:] == ids[:-1]) & (a[:-1] == 1) & (a[1:] == 0))
    crashes = []
    for x in np.sort(dr.t.values[died]):
        if not crashes or x - crashes[-1] > merge_s:
            crashes.append(float(x))
    return crashes


# ----------------------------------------------------------------------------- per window

def window(t, dt, sig, starts, t0, t1, down):
    """Stats for samples t0 <= t < t1; sig maps name -> signal, starts maps dial -> adjustment idx."""
    sel = (t >= t0) & (t < t1)
    if sel.sum() < 3:
        return None
    w = dt[sel]
    dur = float(w.sum())
    m = dict(dur_s=dur)
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


# ----------------------------------------------------------------------------- per run

def analyse_run(d, stem, tols, settle_s, down):
    h, c, s = load_run(d, stem)
    sh = load_shape(d, stem)
    t = h.t.values
    dt = np.diff(t, append=t[-1] + np.median(np.diff(t)))
    sig = {"spread": spread_signal(h, sh), "spacing": spacing_signal(h, sh), "pitch": pitch_signal(h)}
    starts = {}
    for name in DIALS:
        x = sig[name]
        starts[name] = (detect_adjustments(t, x[:, None], tols[name], settle_s)[0]
                        if not np.isnan(x).any() else np.zeros(0, int))

    pid, trial = stem.split("_")[0], int(stem.split("_")[1][1:])
    base = dict(run=stem, pid=pid, trial=trial, condition=s["condition"], nCorrect=s.get("nCorrect"))

    vis = visits(s, c)
    legs, prev_exit = [], t[0]
    for k, (g, enter, exit_) in enumerate(vis, start=1):
        for phase, t0, t1 in (("transit", prev_exit, enter), ("search", enter, max(exit_, enter + 1e-3))):
            m = window(t, dt, sig, starts, t0, t1, down)
            if m is not None:
                m.update(base, phase=phase, leg=k, hat=g["hat"])
                legs.append(m)
        prev_exit = exit_
    legs = pd.DataFrame(legs)

    run = dict(base)
    dur = float(dt.sum())
    run.update(dur_s=dur, pitch_logged="gimbalPitch" in h)
    phase_s = {}
    for phase in ("transit", "search"):
        ph = legs[legs.phase == phase] if len(legs) else legs
        phase_s[phase] = ph.dur_s.sum() if len(ph) else 0.0
        run[f"{phase}_s"] = phase_s[phase]
        for name in ("spread", "spacing", "pitch"):
            ok = ph[name].notna() if len(ph) else []
            run[f"{name}_{phase}"] = ((ph[name][ok] * ph.dur_s[ok]).sum() / ph.dur_s[ok].sum()
                                      if len(ph) and ok.any() else np.nan)
        for name in DIALS:
            n = ph[f"{name}_adj"].sum(min_count=1) if len(ph) else np.nan
            run[f"{name}_adj_per_min_{phase}"] = n / phase_s[phase] * 60.0 if phase_s[phase] > 0 else np.nan
    for name in DIALS:
        x = sig[name]
        logged = not np.isnan(x).all()
        run[f"{name}_adj"] = len(starts[name]) if logged else np.nan
        run[f"{name}_adj_per_min"] = len(starts[name]) / dur * 60.0 if logged else np.nan
        run[f"{name}_tv_per_min"] = float(np.abs(np.diff(x)).sum()) / dur * 60.0 if logged else np.nan
        run[f"{name}_search_delta"] = run[f"{name}_search"] - run[f"{name}_transit"]
    run["spacing_search_delta_m"] = run["spacing_search"] - run["spacing_transit"]
    run["spacing_transit_m"] = run.pop("spacing_transit")
    run["spacing_search_m"] = run.pop("spacing_search")
    p = sig["pitch"]
    run["pitch_down_frac"] = float(dt[p < down].sum() / dur) if not np.isnan(p).all() else np.nan
    return run, legs, dict(t=t, **sig, visits=[(e, x) for _, e, x in vis], crashes=crash_times(d, stem))


# ----------------------------------------------------------------------------- output

SHOW = ["spread_adj", "spread_adj_per_min", "spread_adj_per_min_transit", "spread_adj_per_min_search",
        "spread_tv_per_min", "spread_transit", "spread_search", "spread_search_delta",
        "spacing_transit_m", "spacing_search_m", "spacing_search_delta_m",
        "pitch_adj_per_min", "pitch_transit", "pitch_search", "pitch_search_delta", "pitch_down_frac"]

# (column, label, signed): signed panels get a zero line instead of a zero floor
PANELS = [
    ("spread_adj_per_min", "Spread adjustments / min", False),
    ("spread_search_delta", "Spread set for search\n(d_ref, search − transit)", True),
    ("spacing_search_delta_m", "Achieved spacing change\n(m, search − transit)", True),
    ("pitch_adj_per_min", "Pitch adjustments / min", False),
    ("pitch_search_delta", "Pitch set for search\n(deg, search − transit)", True),
    ("pitch_down_frac", "Fraction of run camera pitched down", False),
]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _missing(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="0.45", fontsize=9)


def plot_summary(runs, out):
    """One observation per run, by condition -- the same form as command_metrics.png."""
    plt = _mpl()
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (col, label, signed) in zip(axs.flat, PANELS):
        data = [runs[runs.condition == cond][col].dropna().values for cond in CONDS]
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        if not any(len(v) for v in data):
            ax.set_xticks(range(len(CONDS)), CONDS)
            ax.set_xlim(-0.5, len(CONDS) - 0.5)
            _missing(ax, "gimbal pitch not logged\nin these runs" if col.startswith("pitch")
                     else "no data")
            continue
        pos = [x for x, v in enumerate(data) if len(v)]
        bp = ax.boxplot([data[x] for x in pos], positions=pos, widths=0.6, patch_artist=True,
                        showfliers=False)
        for patch, x in zip(bp["boxes"], pos):
            patch.set(facecolor=COLOURS[CONDS[x]], alpha=0.35)
        for med in bp["medians"]:
            med.set(color="black")
        for x, (cond, v) in enumerate(zip(CONDS, data)):
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, color=COLOURS[cond], zorder=3, s=25)
        ax.set_xticks(range(len(CONDS)), CONDS)     # after boxplot, which relabels its positions
        ax.set_xlim(-0.5, len(CONDS) - 0.5)
        if signed:
            ax.axhline(0, color="0.3", lw=0.8, zorder=1)
        else:
            ax.set_ylim(bottom=0)
        if col == "pitch_down_frac":
            ax.set_ylim(0, 1.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[c], alpha=0.6) for c in CONDS]
    axs.flat[0].legend(handles, CONDS, loc="upper left")
    fig.suptitle(f"Pilot-set spread and camera pitch ({', '.join(sorted(runs.pid.unique()))}; dots = runs)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[info] saved {out}")


def plot_legs(legs, out):
    """Transit -> search setting per goal leg: does the pilot re-set the dial on arrival?

    The thick line is the mean over legs, not the median: a dial is either re-set on arrival or
    left alone, so with few participants the median is whichever habit has more legs and reads as
    "no change" while one participant re-sets it on every goal.
    """
    plt = _mpl()
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
            ax.plot([0, 1], w.mean().values, color=COLOURS[cond], lw=2.5, marker="o", ms=8,
                    label=f"{cond} mean")
        if not drawn:
            _missing(ax, "gimbal pitch not logged\nin these runs" if col == "pitch" else "no data")
            continue
        if yr:
            ax.set_ylim(yr[0] - 0.05 * (yr[1] - yr[0]), yr[1] + 0.05 * (yr[1] - yr[0]))
        ax.set_title(f"{'swarm' if conds == ['Swarm'] else 'all'} legs, n = {n_legs} (thin = one goal)",
                     fontsize=9)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Setting before (transit) vs. inside (search) each goal tile, time-weighted means")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[info] saved {out}")


TIMELINE_COLS = {"spread": "Spread (d_ref)", "spacing": "Spacing (m)", "pitch": "Pitch (deg)"}
TIMELINE_YLIM = {"spread": (SPREAD_RANGE[0] - 0.1, SPREAD_RANGE[1] + 0.1),
                 "pitch": (PITCH_RANGE[0] - 5, PITCH_RANGE[1] + 5),
                 "spacing": (0, None)}


def plot_timeline(runs, series, out, keys=("spread", "spacing", "pitch"), title=None):
    """Per run, one column per signal in keys, over time; search phases shaded, crashes in red."""
    plt = _mpl()
    rows = runs.sort_values(["pid", "trial"])
    n = len(rows)
    fig, axs = plt.subplots(n, len(keys), figsize=(5 * len(keys), 1.9 * n + 0.6), sharex="row",
                            sharey="col", squeeze=False)
    cols = [(k, TIMELINE_COLS[k]) for k in keys]
    for r, run in enumerate(rows.itertuples()):
        sr = series[run.run]
        colour = COLOURS.get(run.condition, "0.3")
        for j, (key, label) in enumerate(cols):
            ax = axs[r, j]
            for enter, exit_ in sr["visits"]:
                ax.axvspan(enter, max(exit_, enter + 1.0), color="0.5", alpha=0.15, lw=0)
            for tc in sr["crashes"]:
                ax.axvline(tc, color="red", lw=1.2, alpha=0.85, zorder=3)
            y = sr[key]
            if np.isnan(y).all():
                _missing(ax, "not logged" if key == "pitch" else
                         "one drone" if run.condition == "SingleDrone" else "no data")
            else:
                ax.plot(sr["t"], y, color=colour, lw=1.5, drawstyle="steps-post" if key != "spacing" else None)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(label)
            if r == n - 1:
                ax.set_xlabel("time (s)")
        axs[r, 0].set_ylabel(f"{run.pid} t{run.trial}\n{run.condition}", fontsize=9)
    for j, key in enumerate(keys):
        axs[0, j].set_ylim(*TIMELINE_YLIM[key])
    conds = [c for c in CONDS if (rows.condition == c).any()]
    conds = conds if len(conds) > 1 else []     # one condition: the title names it
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.3)] + \
              [plt.Line2D([], [], color=COLOURS[c], lw=2) for c in conds]
    labels = ["search (inside goal tile)"] + conds
    if any(series[r].get("crashes") for r in rows.run):
        handles.append(plt.Line2D([], [], color="red", lw=1.2))
        labels.append("crash (drones lost)")
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=9)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985 if n > 4 else 0.965))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[info] saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--pid", nargs="*", default=["AABB", "AABC"])
    ap.add_argument("--spread-tol", type=float, default=0.1, help="d_ref change that counts as an adjustment")
    ap.add_argument("--pitch-tol", type=float, default=10.0, help="pitch change (deg) that counts as an adjustment")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds steady before a new hold latches")
    ap.add_argument("--down", type=float, default=-30.0, help="pitch (deg) below which the camera counts as down")
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

    tols = {"spread": a.spread_tol, "pitch": a.pitch_tol}
    runs, legs, series = [], [], {}
    for stem in stems:
        r, l, sr = analyse_run(a.dir, stem, tols, a.settle, a.down)
        runs.append(r)
        legs.append(l)
        series[stem] = sr
    runs = pd.DataFrame(runs).sort_values(["pid", "trial"])
    legs = pd.concat(legs, ignore_index=True)

    # The first trial a participant flies in each condition is practice.
    first = runs.groupby(["pid", "condition"]).trial.transform("min")
    practice = set(runs.run[runs.trial == first])
    if not a.include_practice:
        print("[info] dropping practice runs: " + ", ".join(sorted(practice)))
        runs = runs[~runs.run.isin(practice)]
        legs = legs[~legs.run.isin(practice)]
    if not runs.pitch_logged.any():
        print("[info] no run has a gimbalPitch column (logged from 2026-09-24 on): pitch metrics are NaN")

    runs.to_csv(os.path.join(a.dir, "spread_pitch_runs.csv"), sep=";", index=False)
    legs.to_csv(os.path.join(a.dir, "spread_pitch_legs.csv"), sep=";", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"\nspread-tol={a.spread_tol} pitch-tol={a.pitch_tol}deg settle={a.settle}s down<{a.down}deg\n")
    print(runs[["pid", "trial", "condition", "dur_s", "search_s"] + SHOW].round(2).to_string(index=False))
    print("\nMean per participant x condition:")
    print(runs.groupby(["pid", "condition"])[SHOW].mean().round(2).to_string())
    print("\nMean per condition:")
    print(runs.groupby("condition")[SHOW].mean().round(2).to_string())

    if not a.no_plot:
        os.makedirs(os.path.join(a.dir, "plots"), exist_ok=True)
        plot_summary(runs, os.path.join(a.dir, "plots", "spread_pitch_metrics.png"))
        plot_legs(legs, os.path.join(a.dir, "plots", "spread_pitch_legs.png"))
        plot_timeline(runs, series, os.path.join(a.dir, "plots", "spread_pitch_timeline.png"))
        swarm = runs[runs.condition == "Swarm"]
        if len(swarm):
            plot_timeline(swarm, series, os.path.join(a.dir, "plots", "spread_swarm_timeline.png"),
                          keys=("spread", "spacing"),
                          title="Pilot-set spread (left) and the drones' achieved spacing (right), swarm trials")


if __name__ == "__main__":
    main()
