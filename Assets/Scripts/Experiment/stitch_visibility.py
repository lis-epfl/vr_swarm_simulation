"""How long the stitched panorama was functional and visible in each swarm flight.

    python stitch_visibility.py [--dir DIR] [--pid AABB AABC] [--date 20260923] [--log PATH ...] [--show]

"Functional and visible" means the curved panorama screen was up: Python's quality verdict good and the pilot
not having toggled it off (otherwise PyUniSharingFast hides it and shows the individual feeds). Two sources,
best first:

1. ExperimentRecorder's `stitch_on` / `stitch_off` events, in runs recorded since it gained them. Frame-timed,
   so the timeline is exact, and they also catch a *frozen* panorama (Python gone, screen still up), which
   PyUniSharingFast never logs.
2. Unity's Editor.log, for older runs. PyUniSharingFast logs every panorama transition (`[Panorama] hidden
   ...` / `[Panorama] restored ...`) but Editor.log has no timestamps, so a transition can only be placed
   between the nearest *timed* lines around it. Inside a run the only timed lines are DroneHealthMonitor's
   `Parking Drone N ... t=XXs` (its clock agrees with the recorder's to one 10 Hz sample), plus the run's
   start and end. Where two timed lines have no transition between them the state is known exactly; where
   they have one or more it is not, and only bounds survive: visible time lies between the known-on time and
   known-on + unknown. The first `hidden ... unspecified` of every run is its first frame (Python has not
   written a quality word yet; it is logged before the first Update's own lines), so it is taken at t = 0.

Unity overwrites Editor-prev.log on every editor start, so copy the log of a session day next to the data
(e.g. editor_log_YYYYMMDD.log) before restarting the editor; `--log` defaults to every *.log in --dir plus
both live editor logs.

Writes stitch_visibility_<pids>.csv (per run) and stitch_visibility_<pids>.png to --dir.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

DEFAULT_DIR = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\UAVS@BERKELEY\DroneSim\experiment")
EDITOR_LOG_DIR = os.path.expandvars(r"%LOCALAPPDATA%\Unity\Editor")

STEM_RE = re.compile(r"([A-Za-z0-9]+)_t(\d+)_([A-Za-z]+)_(\d{8}_\d{6})")
START_RE = re.compile(r"^ExperimentRecorder: logging to .*\((\S+)_\*\.csv\)")
DEATH_RE = re.compile(r"^\[DroneHealthMonitor\] Parking .* t=([0-9.]+)s")

# Palette (dataviz reference, light mode): on = series 1, hidden = series 2, not recoverable = neutral.
C_ON, C_OFF = "#2a78d6", "#eb6834"
C_UNKNOWN, C_HATCH = "#f0efec", "#a8a79f"
C_SURFACE, C_TEXT, C_TEXT2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


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
    """{(pid, 'YYYYMMDD_HHMMSS'): [item, ...]} with items ('T', on, reason) | ('A', t) | ('S', stitcher)."""
    runs = {}
    for path in paths:
        key, items = None, None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = START_RE.match(line)
                if m:
                    s = STEM_RE.search(m.group(1))
                    key, items = (s.group(1).upper(), s.group(4)), []
                    continue
                if key is None:
                    continue
                if line.startswith("ExperimentRecorder: session finalized"):
                    runs.setdefault(key, items)  # first copy wins if a log was archived twice
                    key = None
                elif line.startswith("[Panorama] hidden"):
                    items.append(("T", False, hidden_reason(line)))
                elif line.startswith("[Panorama] restored"):
                    items.append(("T", True, ""))
                elif line.startswith("PyUniSharingFast: stitcher switched to"):
                    items.append(("S", line.rsplit(" ", 1)[-1].strip(" .\n")))
                else:
                    d = DEATH_RE.match(line)
                    if d:
                        items.append(("A", float(d.group(1))))
    return runs


def reconstruct(items, duration):
    """Segments (t0, t1, 'on' | 'off' | 'unknown', nTransitions) from an untimed transition list."""
    items = list(items)
    trans = [it for it in items if it[0] == "T"]
    state = True  # PyUniSharingFast starts with the panorama displayed ...
    if trans and not trans[0][1] and trans[0][2] == "no_panorama":
        state = False  # ... and hides it on the first frame; see the module docstring
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


def read_events(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(";", 6)  # the note may itself contain ';'
            if len(parts) >= 3:
                rows.append((float(parts[0]), parts[2], parts[6] if len(parts) > 6 else ""))
    return rows


def segments_from_events(events, duration):
    segs, t_prev, on = [], 0.0, False
    for t, kind, _ in events:
        if kind not in ("stitch_on", "stitch_off"):
            continue
        segs.append((t_prev, t, "on" if on else "off", 0))
        t_prev, on = t, kind == "stitch_on"
    segs.append((t_prev, duration, "on" if on else "off", 0))
    return [s for s in segs if s[1] > s[0]]


def summarise(segs, duration):
    known_on = sum(b - a for a, b, k, _ in segs if k == "on")
    known_off = sum(b - a for a, b, k, _ in segs if k == "off")
    unknown = sum(b - a for a, b, k, _ in segs if k == "unknown")
    return {
        "durationSec": round(duration, 1),
        "knownOnSec": round(known_on, 1),
        "knownOffSec": round(known_off, 1),
        "unknownSec": round(unknown, 1),
        "visibleMinPct": round(100 * known_on / duration, 1),
        "visibleMaxPct": round(100 * (known_on + unknown) / duration, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--pid", nargs="+", default=["AABB", "AABC"])
    ap.add_argument("--date", default=None, help="YYYYMMDD; default: every date")
    ap.add_argument("--log", nargs="*", default=None, help="Editor logs to reconstruct older runs from")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    pids = [p.upper() for p in args.pid]
    logs = args.log if args.log is not None else (
        sorted(glob.glob(os.path.join(args.dir, "*.log"))) +
        [p for p in (os.path.join(EDITOR_LOG_DIR, n) for n in ("Editor-prev.log", "Editor.log")) if os.path.exists(p)])
    log_runs = parse_editor_logs(logs)

    runs, anchor_times = [], {}
    for sj in sorted(glob.glob(os.path.join(args.dir, "*_session.json"))):
        stem = sj[: -len("_session.json")]
        m = STEM_RE.search(os.path.basename(stem))
        if not m or m.group(3) != "Swarm" or m.group(1).upper() not in pids:
            continue
        if args.date and not m.group(4).startswith(args.date):
            continue
        with open(sj) as f:
            duration = float(json.load(f)["durationSec"])
        pid, trial, key = m.group(1).upper(), int(m.group(2)), (m.group(1).upper(), m.group(4))
        events = read_events(stem + "_events.csv") if os.path.exists(stem + "_events.csv") else []
        n_ch = {}
        if any(k in ("stitch_on", "stitch_off") for _, k, _ in events):
            source, segs, anchors = "recorder", segments_from_events(events, duration), []
            for _, k, note in events:
                if k == "stitch_off":
                    r = "pilot" if note == "pilot_off" else "stale" if note.startswith("stale") else \
                        hidden_reason(note)
                    n_ch[r] = n_ch.get(r, 0) + 1
            n_on = sum(1 for _, k, _ in events if k == "stitch_on")
            switches = 0
        elif key in log_runs:
            source, items = "editor_log", log_runs[key]
            segs = reconstruct(items, duration)
            anchors = [it[1] for it in items if it[0] == "A" and 0 < it[1] < duration]
            for it in items:
                if it[0] == "T" and not it[1]:
                    n_ch[it[2]] = n_ch.get(it[2], 0) + 1
            n_on = sum(1 for it in items if it[0] == "T" and it[1])
            switches = sum(1 for it in items if it[0] == "S")
        else:
            print(f"[warn] {os.path.basename(stem)}: no stitch events and not found in {len(logs)} log(s); skipped")
            continue
        row = {"stem": os.path.basename(stem), "pid": pid, "trial": trial, "source": source, **summarise(segs, duration),
               "nOn": n_on, "stitcherSwitches": switches}
        for r in ("photometric", "no_overlap", "pilot", "no_panorama", "stale", "other"):
            row[f"hidden_{r}"] = n_ch.get(r, 0)
        runs.append((row, segs))
        anchor_times[row["stem"]] = anchors

    if not runs:
        raise SystemExit("no matching swarm runs")
    runs.sort(key=lambda r: (r[0]["pid"], r[0]["trial"]))
    tag = "_".join(pids)

    import csv
    out_csv = os.path.join(args.dir, f"stitch_visibility_{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0][0].keys()))
        w.writeheader()
        for row, _ in runs:
            w.writerow(row)

    cols = ["pid", "trial", "source", "durationSec", "knownOnSec", "knownOffSec", "unknownSec", "visibleMinPct",
            "visibleMaxPct", "nOn", "hidden_photometric", "hidden_no_overlap", "hidden_pilot"]
    print(";".join(cols))
    for row, _ in runs:
        print(";".join(str(row[c]) for c in cols))

    # ---- figure: one timeline per run ----
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT2,
                         "xtick.color": C_TEXT2, "ytick.color": C_TEXT, "hatch.linewidth": 0.8})
    n = len(runs)
    fig, ax = plt.subplots(figsize=(13, 1.65 + 0.52 * n))  # 1.65 in = the fixed margins below
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    xmax = max(r["durationSec"] for r, _ in runs)
    h = 0.56
    for i, (row, segs) in enumerate(runs):
        y = n - 1 - i
        for a, b, kind, k in segs:
            if b <= a:
                continue
            if kind == "unknown":
                ax.barh(y, b - a, left=a, height=h, color=C_UNKNOWN, edgecolor=C_SURFACE, linewidth=1.5)
                ax.barh(y, b - a, left=a, height=h, color="none", edgecolor=C_HATCH, hatch="////", linewidth=0)
                if b - a > 0.07 * xmax:
                    ax.text((a + b) / 2, y, f"{k} change{'s' if k != 1 else ''}", ha="center", va="center",
                            fontsize=10, color=C_TEXT2,
                            bbox=dict(boxstyle="round,pad=0.2", fc=C_UNKNOWN, ec="none"))
            else:
                ax.barh(y, b - a, left=a, height=h, color=C_ON if kind == "on" else C_OFF,
                        edgecolor=C_SURFACE, linewidth=1.5)
        lo, hi = row["visibleMinPct"], row["visibleMaxPct"]
        label = f"{lo:.0f}%" if abs(hi - lo) < 0.5 else f"{lo:.0f}–{hi:.0f}%"
        ax.text(xmax * 1.015, y, label, va="center", ha="left", fontsize=12, color=C_TEXT)
    for i, (row, segs) in enumerate(runs):  # timed log lines, drawn last so they sit on top
        y = n - 1 - i
        for t in anchor_times[row["stem"]]:
            ax.plot([t, t], [y - h / 2 - 0.06, y + h / 2 + 0.06], color=C_TEXT, linewidth=1.2, zorder=5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r['pid']} t{r['trial']}" for r, _ in reversed(runs)])
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("time since session start (s)")
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(xmax * 1.015, n - 0.35, "visible", ha="left", va="bottom", fontsize=11, color=C_TEXT2)
    title = "Stitched panorama on screen — swarm flights"
    if args.date:
        title += f", {args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, color=C_TEXT)
    handles = [Patch(fc=C_ON, label="panorama on screen"), Patch(fc=C_OFF, label="hidden (feeds shown)")]
    if any(r["source"] == "editor_log" for r, _ in runs):
        handles += [Patch(fc=C_UNKNOWN, ec=C_HATCH, hatch="////", label="changes not placeable (untimed log)"),
                    Line2D([], [], color=C_TEXT, linewidth=1.2, label="timed log line (drone lost)")]
    fig_h = fig.get_size_inches()[1]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.45 / fig_h), ncol=4,
               frameon=False, fontsize=11, labelcolor=C_TEXT)
    # Explicit margins: tight_layout counts the out-of-axes percentage labels and leaves a gap.
    fig.subplots_adjust(left=0.07, right=0.89, top=1 - 1.05 / fig_h, bottom=0.6 / fig_h)
    out_png = os.path.join(args.dir, f"stitch_visibility_{tag}.png")
    fig.savefig(out_png, dpi=140, facecolor=C_SURFACE)
    print(f"[out] {out_csv}\n[out] {out_png}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
