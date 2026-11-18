"""User trials: one results page per test series (internal_1, internal_2, ...) and an index of them, in the study pages'
family (citysim/swarmwrap_report's: S4's styles read at build, the replays played by citysim/replay_view.js).

    python -B trials_report.py all                 # every series with a docs/trials/<test>.md, then the index
    python -B trials_report.py page internal_4     # one series' page (and its data.json, which the index reads)
    python -B trials_report.py index               # the index, from the series pages' data.json
        -> experiment_results/trials/<test>/<test>.html (the replays inlined, to open as a file), <test>_bare.html (what
           the artifact is published from), <test>_files.json (the replay files to publish beside it), data.json and
           tables/*.csv; experiment_results/trials/trials_index.html (+ trials_index_bare.html)

Every number is analyse.py's, run in memory on the series' runs (its definitions of the phases, the pilot-command
metrics, the losses), with one change to its defaults: every run is timed from the start line (analyse's --start line:
the crossing the recorder logged, from 2026-10-26, or the same crossing derived from the drone logs before it), so
every series is on one clock. The tables beside each page are what the page shows. Practice runs (test.json) are drawn
but counted in no summary. The prose -- what changed since the series before, the setup, the captions and the reading --
is docs/trials/<test>.md, and the page URLs are docs/trials/pages.json; nothing in either is a number the runs give.
A series whose runs carry no answer times (internal_1) can be given hand-timed ones in <test folder>/manual_times.csv
(run;timeSec;source).
"""
import sys

sys.dont_write_bytecode = True   # nothing under Assets/ may grow a __pycache__

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import html  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse as A  # noqa: E402
from citysim import playback as PB  # noqa: E402
from citysim import replay as RP  # noqa: E402
from citysim.swarmtune_report import _clean, s4_style, VIEW_JS  # noqa: E402

REPO = HERE.parents[2]
DOCS = REPO / "docs" / "trials"
OUT = REPO / "experiment_results" / "trials"
TEMPLATE = HERE / "trials_report.html"
INDEX_TEMPLATE = HERE / "trials_index.html"
HUB_URL = "https://claude.ai/artifact/8FbWjQ3bF7C65khZYJ3aN5"
FAMILIES = {"cmd", "dial", "prox", "hat", "street", "disc", "stitch", "crash"}
COND = {"Swarm": "swarm", "SingleDrone": "single", "SingleSphere": "sphere"}
SINGLE, SPHERE = ("Swarm", "SingleDrone"), ("Swarm", "SingleSphere")   # the two comparisons: swarm ÷ the other
DT = 0.2                        # s between replay frames
SMOOTH = ["stick_tv_per_min", "adj_per_min", "mean_hold_s", "course_turn_per_min", "cmd_straightness", "act_path_ratio"]
RUN_COLS = ["run", "pid", "order", "trial", "condition", "practice", "date", "durationSec", "startSec", "startSource",
            "time_s", "time_source", "transit_s", "search_s", "goalsReached", "nGoals", "nCorrect", "falseAlarms",
            "identifyOverrides", "drones_lost", "crashes", *SMOOTH, "stitch_visibleMinPct", "stitch_visibleMaxPct"]
WHY = {"CollidingWithAnotherDrone": "drone", "ExternallyKilled": "drone", "TooFarFromSwarm": "isolated"}


# ============================================================================= the notes (docs/trials/<test>.md)

def inline(s):
    """Markdown inline to HTML: **bold**, *italic*, `code`, [text](url); everything else escaped."""
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r'<span class="code">\1</span>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)


def blocks(lines):
    """Lines to [("p", text) | ("ul", [items])], a bullet continued by indented lines."""
    out, para = [], []

    def flush():
        if para:
            out.append(("p", " ".join(para)))
            para.clear()
    for ln in lines:
        if not ln.strip():
            flush()
        elif re.match(r"^\s*[-*] ", ln) and not ln.startswith("  "):
            flush()
            item = re.sub(r"^\s*[-*] ", "", ln).strip()
            if out and out[-1][0] == "ul":
                out[-1][1].append(item)
            else:
                out.append(("ul", [item]))
        elif ln.startswith("  ") and out and out[-1][0] == "ul" and not para:
            out[-1][1][-1] += " " + ln.strip()
        else:
            para.append(ln.strip())
    flush()
    return out


def html_of(lines):
    return "".join(f"<p>{inline(x)}</p>" if k == "p" else "<ul>" + "".join(f"<li>{inline(i)}</li>" for i in x) + "</ul>"
                   for k, x in blocks(lines))


def sections(lines, level):
    """{heading: lines} for the headings of one level, in order, and the lines before the first."""
    mark = "#" * level + " "
    head, out, cur = [], {}, None
    for ln in lines:
        if ln.startswith(mark):
            cur = ln[len(mark):].strip()
            out[cur] = []
        elif cur is None:
            head.append(ln)
        else:
            out[cur].append(ln)
    return head, out


def table_of(lines):
    rows = [ln.strip().strip("|").split("|") for ln in lines if ln.strip().startswith("|")]
    rows = [[c.strip() for c in r] for r in rows if not all(re.fullmatch(r":?-+:?", c.strip()) for c in r)]
    return dict(head=[inline(c) for c in rows[0]], rows=[[inline(c) for c in r] for r in rows[1:]]) if rows else None


def read_notes(test):
    """The series' prose: title, its meta lines (- **Key:** value), and the fixed sections."""
    text = (DOCS / f"{test}.md").read_text(encoding="utf-8").splitlines()
    title = text[0].lstrip("# ").strip()
    head, sec = sections(text[1:], 2)
    meta = {}
    for ln in head:
        m = re.match(r"^- \*\*(.+?):\*\*\s*(.*)$", ln.strip())
        if m:
            meta[m.group(1).strip().lower()] = m.group(2).strip()
    notes = dict(title=title, meta=meta, summary="", changes_title="", changes=[], within=[], setup=None, notable={},
                 reading="", data_notes=[])
    for name, body in sec.items():
        low = name.lower()
        if low == "summary":
            notes["summary"] = html_of(body)
            notes["summary_text"] = " ".join(x for k, x in blocks(body) if k == "p")
        elif low.startswith("changes since") or low.startswith("the starting setup"):
            notes["changes_title"] = name
            _, areas = sections(body, 3)
            for area, lines in areas.items():
                items = [i for k, x in blocks(lines) if k == "ul" for i in x]
                notes["changes"].append(dict(area=area, items=[inline(i) for i in items],
                                             leads=[inline(m.group(1).rstrip(":,;. ")) for m in
                                                    (re.match(r"^\*\*(.+?)\*\*", i) for i in items) if m]))
        elif low.startswith("within the series"):
            notes["within"] = [inline(i) for k, x in blocks(body) if k == "ul" for i in x]
        elif low == "setup":
            notes["setup"] = table_of(body)
        elif low == "notable":
            _, cards = sections(body, 3)
            for h, lines in cards.items():
                cid, _, ttl = h.partition(":")
                notes["notable"][cid.strip()] = dict(title=ttl.strip(), html=html_of(lines))
        elif low == "reading":
            notes["reading"] = html_of(body)
        elif low == "data notes":
            notes["data_notes"] = [inline(i) for k, x in blocks(body) if k == "ul" for i in x]
    return notes


def pages():
    p = DOCS / "pages.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def series_list():
    return sorted((p.stem for p in DOCS.glob("internal_*.md")), key=lambda s: int(re.sub(r"\D", "", s) or 0))


# ============================================================================= the analysis

def analyse_test(test):
    """analyse.py on the series' runs, in memory, every run timed from the start line."""
    folder, manifest, runs = A.load_test(A.DEFAULT_ROOT, test)
    cfg = argparse.Namespace(**A.DEFAULTS, include_practice=False)
    cfg.range = float(manifest.get("range", A.DEFAULTS["range"]))
    cfg.dists = sorted(set(cfg.dists) | {cfg.zone})
    A.apply_start(runs, "line")
    logs = A.parse_editor_logs(sorted(glob.glob(os.path.join(folder, "*.log"))))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        res = A.analyse_runs(runs, cfg, logs, FAMILIES)
        rd, ld, gd, dd = A.build_tables(runs, res)
    order = {}
    for pid in sorted({r.pid for r in runs}):
        for k, r in enumerate(sorted((r for r in runs if r.pid == pid), key=lambda r: r.timestamp)):
            order[r.stem] = k + 1
    for df in (rd, ld, gd, dd):
        if len(df):
            df.insert(1, "order", df.run.map(order))
    manual = os.path.join(folder, "manual_times.csv")
    if os.path.exists(manual):
        m = pd.read_csv(manual, sep=";")
        for _, x in m.iterrows():
            sel = rd.run == x["run"]
            if sel.any():
                rd.loc[sel, "time_s"] = float(x["timeSec"])
                rd.loc[sel, "time_source"] = "by hand" + (f" ({x['source']})" if "source" in m and x["source"] == x["source"] else "")
    return dict(test=test, folder=folder, manifest=manifest, runs=runs, cfg=cfg, res=res, rd=rd, ld=ld, gd=gd, dd=dd,
                order=order, manual=os.path.exists(manual))


def timed(T):
    """Whether the runs say when the task ended: an answer press (or a hand-timed list), not only when the recording
    stopped. internal_2's answers were entered afterwards, but its sessions ended at the last answer's word."""
    rd = T["rd"]
    return bool(T["manual"] or (rd.time_source == "identify").any() or
                any(g.get("answered") for r in T["runs"] for g in r.goals))


def compare(df, col, conds=SINGLE):
    """Swarm against the single drone (or, with conds=SPHERE, against one drone's 360 camera) on one per-run measure:
    per pilot the geometric means and their ratio, the ratio pooled over the pilots who flew both (the geometric mean
    of theirs), and the exact two-sided permutation p of that pooled log ratio, relabelling the conditions within each
    pilot (the runs per condition kept). `floor` is the smallest p the design can give. None when no pilot flew both."""
    d = df[df[col].notna() & (df[col] > 0) & df.condition.isin(conds)]
    per, groups = [], []
    for pid, g in d.groupby("pid"):
        s, o = np.log(g[g.condition == conds[0]][col].to_numpy(float)), np.log(g[g.condition == conds[1]][col].to_numpy(float))
        if not len(s) or not len(o):
            continue
        per.append(dict(pid=pid, n_s=len(s), n_d=len(o), gm_s=float(np.exp(s.mean())), gm_d=float(np.exp(o.mean())),
                        mean_s=float(np.exp(s).mean()), mean_d=float(np.exp(o).mean()), ratio=float(np.exp(s.mean() - o.mean()))))
        groups.append((np.r_[s, o], len(s)))
    if not per:
        return None
    obs = float(np.mean([math.log(p["ratio"]) for p in per]))
    stats = []
    for combo in itertools.product(*[list(itertools.combinations(range(len(v)), k)) for v, k in groups]):
        tot = 0.0
        for (v, k), sel in zip(groups, combo):
            m = np.zeros(len(v), bool)
            m[list(sel)] = True
            tot += v[m].mean() - v[~m].mean()
        stats.append(abs(tot / len(groups)))
    stats = np.array(stats)
    p = float((stats >= abs(obs) - 1e-12).mean())
    floor = float((stats >= stats.max() - 1e-12).mean())
    # the typical run of each condition, pooled as the ratio is (the geometric mean over the pilots of theirs), so that
    # typ_s / typ_d is the pooled ratio exactly; the arithmetic means are kept beside them for the tables
    typ_s = float(np.exp(np.mean([math.log(x["gm_s"]) for x in per])))
    typ_d = float(np.exp(np.mean([math.log(x["gm_d"]) for x in per])))
    return dict(per=per, ratio=float(math.exp(obs)), p=p, floor=floor, n_perm=len(stats), typ_s=typ_s, typ_d=typ_d,
                n_lower=sum(x["ratio"] < 1 for x in per), n_pilots=len(per),
                mean_s=float(d[d.condition == conds[0]][col].mean()), mean_d=float(d[d.condition == conds[1]][col].mean()))


def run_rows(T):
    rd = T["rd"]
    keep = [c for c in RUN_COLS if c in rd]
    out = rd[keep].copy()
    out["cond"] = out.condition.map(COND)
    return out


def press_events(run):
    """[(t, goalIndex, outcome, note)]: the run's identify presses as the recorder logged them."""
    out = []
    with open(run.path("events.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(";", 6)
            if len(p) == 7 and p[2] == "identify":
                out.append((float(p[0]), int(p[3]), p[4], p[6]))
    return out


def identify(T):
    """Per run: how answers were given and scored."""
    rows = []
    for r in T["runs"]:
        pr = press_events(r)
        mode = r.session.get("identifyMode") or ("presses, outcomes not keyed" if pr else
                                                  "entered afterwards" if any(g.get("answered") for g in r.goals) else "none recorded")
        g = T["gd"][T["gd"].run == r.stem]
        s2p = g.sightToPressSec[g.sightToPressSec >= 0] if "sightToPressSec" in g else pd.Series([], dtype=float)
        rows.append(dict(run=r.stem, pid=r.pid, order=T["order"][r.stem], condition=r.condition, practice=r.practice,
                         mode=mode, presses=len(pr), correct=int(r.session.get("nCorrect", 0) or 0),
                         goals=len(r.goals), false_alarms=int(r.session.get("falseAlarms", 0) or 0),
                         overrides=int(r.session.get("identifyOverrides", 0) or 0),
                         out_of_zone=sum(1 for x in pr if x[2] == "out_of_zone"),
                         sight_to_press=float(s2p.median()) if len(s2p) else None))
    return rows


def losses(T):
    """Every drone lost, with the pilot's spread setting and the run's task clock at that moment."""
    out = []
    for _, d in T["dd"].iterrows():
        run = next(r for r in T["runs"] if r.stem == d.run)
        s = T["res"]["series"].get(d.run)
        spread = float(np.interp(d.t, s["t"], s["spread"])) if s is not None and np.isfinite(s["spread"]).any() else None
        start = run.start if run.start is not None else 0.0
        out.append(dict(run=d.run, pid=d.pid, order=int(d.order), condition=d.condition, practice=bool(d.practice),
                        drone=int(d.droneId), t=float(d.t), t_task=float(d.t - start), x=float(d.x), z=float(d.z),
                        reason=d.reason if isinstance(d.reason, str) else "",
                        partner=None if pd.isna(d.get("partner", np.nan)) else int(d.partner),
                        speed=None if pd.isna(d.get("speed", np.nan)) else float(d.speed), spread=spread))
    return out


def restarts(T):
    """Recordings per (pilot, trial, condition): a trial restarted shows as several."""
    rd = T["rd"]
    out = []
    for (pid, trial, cond), g in rd.groupby(["pid", "trial", "condition"]):
        g = g.sort_values("order")
        out.append(dict(pid=pid, trial=int(trial), condition=cond, n=len(g), order=[int(x) for x in g.order],
                        lengths=[float(x) for x in g.durationSec], practice=bool(g.practice.all())))
    return sorted(out, key=lambda x: (x["pid"], x["order"][0]))


def setups(T):
    """What each pilot's runs flew, from the files a run writes (the session and, since 2026-11-03, _swarm.json): a column
    per pilot from their first run, and another from their first run on the 360 sphere where they flew one (its display
    has its own settings); only the fields some runs carry, and the ones that differ marked."""
    fields = [("scene", "scene"), ("display", "display"), ("identifyMode", "scoring"), ("identifyRangeM", "identify range (m)"),
              ("identifyLookbackSec", "look-back (s)"), ("identifyZoneM", "zone (m)"), ("screenStyle", "screen layout"),
              ("screenScale", "screen scale"), ("screenRadius", "screen radius (m)"), ("feed", "feed size"),
              ("sphere", "360 sphere"), ("stitchingEnabled", "stitching"), ("cockpitRoomStyle", "backdrop"),
              ("displayRefreshHz", "headset refresh (Hz)"), ("gitCommit", "commit")]
    swarm = [("spreadWrapRule", "wrap rule"), ("obstacleShape", "avoidance shape"), ("adaptiveSpread", "adaptive spread"),
             ("d_ref", "d_ref")]
    per = {}
    for r in sorted(T["runs"], key=lambda r: r.timestamp):
        s = r.session
        row = {k: s.get(k) for k, _ in fields if k in s}
        lay = s.get("layout") if isinstance(s.get("layout"), dict) else {}
        if lay.get("scene"):
            row["scene"] = lay["scene"]
        sphere = s.get("displayMode") == "SPHERE"
        row["display"] = "360 sphere" if sphere else "feed screens"     # every run before the field (2026-11-06) had screens
        if sphere:
            sp = s.get("sphere360") or {}
            stream = (f"shown as a {sp['streamWidth']}×{sp['streamHeight']} live stream, "
                      if sp.get("streamWidth") else "shown as rendered, ")      # absent before 2026-11-18: the cube
            row["sphere"] = (f"cube {sp.get('cubeSize')} px, {stream}{sp.get('refreshHz', 0):g} Hz"
                             f"{', faces split' if sp.get('splitFaces') else ''}, {sp.get('heading', '')}"
                             f"{', band ±%g°' % sp['bandHalfDeg'] if sp.get('bandHalfDeg') else ''}")
            for k in ("screenStyle", "screenScale", "screenRadius", "feed"):    # no feed screens on the sphere
                row.pop(k, None)
        if "feedWidth" in s and not sphere:
            row["feed"] = f"{s['feedWidth']}×{s['feedHeight']}"
        if "gitCommit" in s:
            row["gitCommit"] = s["gitCommit"][:8] + (" (dirty)" if s.get("gitDirty") else "")
        sj = r.path("swarm.json")
        if os.path.exists(sj):
            with open(sj, encoding="utf-8") as f:
                sm = json.load(f).get("swarmManager", {})
            row.update({k: sm.get(k) for k, _ in swarm if k in sm})
        disp = next((note for _, k, note in r.events if k == "display"), "")
        m = re.search(r"offset=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)", disp)
        if m and not sphere:
            row["screenHeight"] = float(m.group(2))
        row["_sphere"] = sphere
        per.setdefault(f"{r.pid} · sphere" if sphere else r.pid, row)
    names = dict(fields + swarm + [("screenHeight", "screens below eye level (m)")])
    keys = [k for k in names if any(k in v for v in per.values())]
    if not any(v["_sphere"] for v in per.values()):
        keys = [k for k in keys if k != "display"]          # a series that only had feed screens has nothing to say here
    if len(keys) < 3:
        return None
    enums = dict(spreadWrapRule=["Off", "Hold", "HoldUntilLeaving"], obstacleShape=["Cylinder", "EnclosingCircle", "Footprint", "Ellipse"])
    # what one display has and the other has not: shown as such, and left out of the test for a difference
    screen_only, sphere_only, NA = {"screenStyle", "screenScale", "screenRadius", "feed", "screenHeight"}, {"sphere"}, "n/a"

    def show(k, v, on_sphere=False):
        if (on_sphere and k in screen_only) or (not on_sphere and k in sphere_only):
            return NA
        if v is None:
            return "Off (before the field)" if k == "spreadWrapRule" else "none" if k == "identifyLookbackSec" else "–"
        if k in enums and isinstance(v, (int, float)) and 0 <= int(v) < len(enums[k]):
            return enums[k][int(v)]
        if isinstance(v, bool):
            return "on" if v else "off"
        if isinstance(v, float):
            v = (0.0 if v >= 0 else -v) if k == "screenHeight" else v
            return f"{round(v, 2):g}"
        return str(v)
    rows = []
    for k in keys:
        val = [show(k, per[p].get(k), per[p]["_sphere"]) for p in per]
        rows.append(dict(key=names[k], vals=val, differs=len(set(v for v in val if v != NA)) > 1))
    return dict(pilots=list(per), rows=rows)


def long_legs(T, n=6):
    g = T["gd"]
    g = g[g.reached.astype(bool) & ~g.practice.astype(bool)] if len(g) and "reached" in g else g.iloc[0:0]
    out = []
    for col, kind in (("searchSec", "at a goal"), ("transitSec", "between goals")):
        for _, x in g.sort_values(col, ascending=False).head(n).iterrows():
            out.append(dict(kind=kind, run=x.run, pid=x.pid, order=int(x.order), condition=x.condition, hat=x.hat,
                            leg=int(x.zoneOrder), s=float(x[col])))
    return out


def panorama(T):
    rd = T["rd"]
    if "stitch_visibleMinPct" not in rd:
        return None
    s = rd[rd.stitch_visibleMinPct.notna()]
    return [dict(run=x.run, pid=x.pid, order=int(x.order), practice=bool(x.practice), lo=float(x.stitch_visibleMinPct),
                 hi=float(x.stitch_visibleMaxPct), source=x.stitch_source) for _, x in s.iterrows()] if len(s) else None


# ============================================================================= the replays

def box_poly(ax_or_prism, c):
    """A building's footprint (k, 2): a prism's own, or the hull of an oriented box's corners in (x, z)."""
    if isinstance(ax_or_prism, A.Prism):
        return np.asarray(ax_or_prism.poly, float)
    ax = np.asarray(ax_or_prism, float)
    pts = np.array([c + a * ax[0] + b * ax[1] + d * ax[2] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])[:, [0, 2]]
    return hull(pts)


def hull(p):
    p = np.unique(np.round(p, 3), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, up = half(p), half(p[::-1])
    return np.array(lo[:-1] + up[:-1])


def square(x, z, h):
    return np.array([[x - h, z - h], [x + h, z - h], [x + h, z + h], [x - h, z + h]])


def tile_city(run, kerb_city):
    """(arena, blocks, flags, buildings, note) for a tile city: each tile's block, a diamond's lawns as plazas, the
    buildings with the run's goal patches in place. Where the goals match no tile of the city file (internal_1, flown in
    CityWorld, which has no export, and whose goals stood off the tiles), the tiles of `kerb_city` and no buildings."""
    note = ""
    try:
        boxes = A.scene_boxes(run.goals, run.city, run.goal_width)
        city = run.city
    except ValueError:
        boxes, city = [], kerb_city
        note = "Only the tile grid is drawn: the city these runs flew has no export of its buildings."
    _, kerbs = A.load_city(city)
    tiles, slots = A.fit_lattice(kerbs)
    diamonds, _ = A.find_diamonds(slots)
    half = A.BLOCK_HALF if boxes else 38.1            # CityWorld's blocks were not shrunk
    blocks = [square(c[0], c[1], half) for c in tiles.values()]
    flags = [1] * len(blocks)
    for (cx, cz), _ in diamonds:
        for w, h in ((A.PITCH + A.LAWN_HALF, A.LAWN_HALF), (A.LAWN_HALF, A.PITCH / 2 + A.LAWN_HALF)):
            blocks.append(np.array([[cx - w, cz - h], [cx + w, cz - h], [cx + w, cz + h], [cx - w, cz + h]]))
            flags.append(3)
    xs, zs = [s[1] for s in slots], [s[2] for s in slots]
    arena = [min(xs) - A.TILE_HALF, max(xs) + A.TILE_HALF, min(zs) - A.TILE_HALF, max(zs) + A.TILE_HALF]
    blds = [box_poly(ax, np.asarray(c, float)) for _, c, ax in boxes]
    return arena, blocks, flags, blds, note


def layout_city(layout_id):
    from citysim import layout as lay
    with open(lay.resolve(layout_id), encoding="utf-8-sig") as f:
        d = json.load(f)
    blocks = [np.reshape(b["kerb_xz"], (-1, 2)) if len(b["kerb_xz"]) >= 6 else np.zeros((0, 2)) for b in d["blocks"]]
    flags = [(1 if b.get("interior") else 0) | (2 if b.get("building", 0) < 0 else 0) for b in d["blocks"]]
    blds = [np.reshape(b["poly_xz"], (-1, 2)) for b in d["boxes"] if b.get("type") == "ConvexPrism"]
    return list(map(float, d["layout"]["arena"])), blocks, flags, blds, ""


def city_record(T, run):
    """(city id, city JSON) for a run: one per layout, one per set of goal tiles in a tile city."""
    if run.layout_id:
        cid = re.sub(r"[^\w.-]", "_", run.layout_id)[:80]
        arena, blocks, flags, blds, note = layout_city(run.layout_id)
    else:
        key = "_".join(f"{g['goalX']:.0f}_{g['goalZ']:.0f}" for g in sorted(run.goals, key=lambda g: (g["goalX"], g["goalZ"])))
        cid = f"{run.city}__{hashlib.md5(key.encode()).hexdigest()[:8]}"
        arena, blocks, flags, blds, note = tile_city(run, A.DEFAULT_CITY)
    pad = 30.0
    c = run.centroid
    arena = [min(arena[0], float(c.gtX.min()) - pad), max(arena[1], float(c.gtX.max()) + pad),
             min(arena[2], float(c.gtZ.min()) - pad), max(arena[3], float(c.gtZ.max()) + pad)]
    return cid, PB.city_json(cid, arena, blocks, blds, flags, city=run.city, note=note)


def flight_record(T, run, loss_rows):
    """The run as replay_view.js plays it: every drone every DT s on the session clock (t = 0 at Play), the feeds shown,
    the pilot's heading, the losses, the goals with their zones, the presses and the start line."""
    dr = run.drones
    ids = sorted(dr.droneId.unique())
    t = np.sort(dr.t.unique())
    piv = {k: dr.pivot_table(index="t", columns="droneId", values=k).reindex(t)[ids].ffill().bfill().to_numpy(float)
           for k in ("gtX", "gtZ", "yawDeg", "alive")}
    grid = np.arange(t[0], t[-1] + 1e-9, DT)
    k = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    alive = piv["alive"][k] > 0.5
    for i in range(len(ids)):                     # a drone once lost stays lost
        gone = np.flatnonzero(~alive[:, i])
        if len(gone):
            alive[gone[0]:, i] = False
    shown_ct = RP.shown_mask(run)
    kc = np.clip(np.searchsorted(run.ct, grid, side="right") - 1, 0, len(run.ct) - 1)
    shown = shown_ct[kc] & alive
    h = run.head
    kh = np.clip(np.searchsorted(h.t.to_numpy(), grid, side="right") - 1, 0, len(h) - 1)
    body = h.bodyYaw.to_numpy(float)[kh]
    rec = dict(dt=DT, t=grid, P=np.stack([piv["gtX"][k], piv["gtZ"][k]], -1), alive=alive, yaw=piv["yawDeg"][k],
               shown=shown, body=body, wp=np.zeros(len(grid)), contacts=[],
               losses=[dict(t=round(x["t"], 2), i=ids.index(x["drone"]) if x["drone"] in ids else 0, x=round(x["x"], 1),
                            z=round(x["z"], 1), why=WHY.get(x["reason"], "lost")) for x in loss_rows])
    legs = T["res"]["legs"][run.stem]
    found = {}
    for g in run.goals:
        if g.get("answered") and float(g.get("decisionTimeSec", -1)) >= 0:
            found[g.get("goalIndex")] = float(g["decisionTimeSec"])
    goals = []
    for gi, g in enumerate(run.goals):
        disc = g.get("zoneShape") == "disc"
        goals.append(dict(x=round(g["goalX"], 1), z=round(g["goalZ"], 1), shape="disc" if disc else "square",
                          half=round(A.patch_half(g), 1), zone=T["cfg"].zone, label=A.hat_name(g) or f"goal {gi + 1}",
                          found=found.get(g.get("goalIndex", gi))))
    presses = []
    for tp, gidx, outcome, note in press_events(run):
        m = re.search(r"centroid=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)", note)
        if m:
            x, z = float(m.group(1)), float(m.group(3))
        else:
            x, z = A.pos_at(run.centroid, tp)
        g = next((g for g in run.goals if g.get("goalIndex") == gidx), None)
        label = (A.hat_name(g) if g else "") or ""
        judged = next((g for g in run.goals if g.get("overrideWhy") is not None
                       and abs(float(g.get("decisionTimeSec", -1)) - tp) < 0.05), None)   # test.json identifyOverrides
        if judged is not None:
            scored = {"false_alarm": "a false alarm", "out_of_zone": "outside every zone"}.get(outcome, outcome or "unscored")
            outcome, label = "correct", f"{A.hat_name(judged)} (scored {scored} at the time, judged correct afterwards)"
        presses.append(dict(t=round(tp, 2), x=round(float(x), 1), z=round(float(z), 1), outcome=outcome or "", label=label))
    s = run.session
    start = None
    if run.start is not None:
        if float(s.get("startLineSec", -1)) >= 0:
            start = dict(t=run.start, x=s["startLineX"], z=s["startLineZ"], heading=s.get("startLineHeadingDeg", 90.0))
        else:
            hd = math.radians(A.START_LINE_HEADING_DEG)
            start = dict(t=run.start, x=float(run.cx[0] + math.sin(hd) * A.START_LINE_DISTANCE_M),
                         z=float(run.cz[0] + math.cos(hd) * A.START_LINE_DISTANCE_M), heading=A.START_LINE_HEADING_DEG,
                         derived=True)
    bands = [dict(label="at a goal (in its zone)", tone="good", spans=[[round(L["enter"], 1), round(L["exit"], 1)] for L in legs])]
    st = T["res"]["stitch"].get(run.stem)
    if st:
        on = [[round(a, 1), round(b, 1)] for a, b, kind, _ in st[1] if kind == "on"]
        if on:
            bands.append(dict(label="panorama on screen", tone="muted", spans=on))
    return rec, dict(goals=goals, presses=presses, start=start), bands


def run_title(r, order):
    return f"{order}. {'Swarm' if r.condition == 'Swarm' else 'Single drone' if r.condition == 'SingleDrone' else r.condition} t{r.trial}"


def replays(T, dest):
    """Writes replay/cities and replay/flights under dest; returns the player's groups (one per pilot) and the files."""
    files, groups, done = {}, {}, set()
    rdx = T["rd"].set_index("run")
    loss = losses(T)
    for r in sorted(T["runs"], key=lambda r: (r.pid, r.timestamp)):
        cid, city = city_record(T, r)
        if cid not in done:
            p = dest / "replay" / "cities" / f"{cid}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(city, separators=(",", ":")), encoding="utf-8")
            files[f"replay/cities/{cid}.json"] = p
            done.add(cid)
        rec, marks, bands = flight_record(T, r, [x for x in loss if x["run"] == r.stem])
        row = rdx.loc[r.stem]
        order = T["order"][r.stem]
        summ = dict(time_s=row.get("time_s"), time_source=row.get("time_source"), duration=r.duration,
                    start=r.start, start_source=r.start_source, search_s=row.get("search_s"), transit_s=row.get("transit_s"),
                    reached=int(row.get("goalsReached", 0)), goals=len(r.goals), correct=int(r.session.get("nCorrect", 0) or 0),
                    false_alarms=int(r.session.get("falseAlarms", 0) or 0), lost=len([x for x in loss if x["run"] == r.stem]),
                    drones=int(r.session.get("droneCount", 0) or 0), practice=bool(r.practice))
        f = PB.flight_json(rec, id=r.stem, title=run_title(r, order), city=cid, cond=COND.get(r.condition, "single"),
                           source="trial", source_label="the recorded run", summary=_clean(summ), marks=marks, bands=bands,
                           span=300 if r.condition == "Swarm" else 220, pid=r.pid)
        p = dest / "replay" / "flights" / f"{r.stem}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_clean(f), separators=(",", ":")), encoding="utf-8")
        files[f"replay/flights/{r.stem}.json"] = p
        g = groups.setdefault(r.pid, dict(id=r.pid, title=r.pid, short=r.pid, kind="runs", flights=[],
                                          about=f"{r.pid}'s runs, in the order flown. The clock is the recording's, from Play; "
                                                f"the task starts at the start line."))
        g["flights"].append(dict(id=r.stem, title=run_title(r, order) + (" (practice)" if r.practice else ""),
                                 cond=COND.get(r.condition, "single"), city=cid, source="trial",
                                 about=f"{r.stem}: {r.session.get('droneCount', '?')} drone(s), recorded "
                                       f"{fmt_len(r.duration)}." + (" A practice run: shown, counted in no summary." if r.practice else "")))
    return dict(groups=list(groups.values())), files


def fmt_len(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


# ============================================================================= the pages

def fill(template, data, files):
    text = json.dumps(_clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = template.read_text(encoding="utf-8")
    for ph in ("__DATA__", "/*__REPLAY_VIEW__*/", "<!--__RV_INLINE__-->", "/*__S4_STYLE__*/"):
        assert ph in page, f"{template.name} has no {ph} placeholder"
    page = page.replace("/*__S4_STYLE__*/", s4_style()).replace("/*__REPLAY_VIEW__*/", VIEW_JS.read_text(encoding="utf-8"))
    page = page.replace("__DATA__", text)
    bare = page.replace("<!--__RV_INLINE__-->", "")
    inline_ = {k: json.loads(Path(v).read_text(encoding="utf-8")) for k, v in files.items()}
    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1,viewport-fit=cover\"></head><body>" +
            page.replace("<!--__RV_INLINE__-->", '<script id="rv-inline" type="application/json">' +
                         json.dumps(inline_, separators=(",", ":")).replace("</", "<\\/") + "</script>") +
            "</body></html>")
    return full, bare


def build_page(test):
    notes = read_notes(test)
    T = analyse_test(test)
    dest = OUT / test
    (dest / "tables").mkdir(parents=True, exist_ok=True)
    for name in ("runs", "legs", "goals", "crashes"):
        df = T[{"runs": "rd", "legs": "ld", "goals": "gd", "crashes": "dd"}[name]]
        df.to_csv(dest / "tables" / f"{name}.csv", sep=";", index=False)
    rows = run_rows(T)
    counted = rows[~rows.practice.astype(bool)]
    is_timed = timed(T)
    gd = T["gd"]
    goals = []
    if len(gd) and "reached" in gd:
        for _, x in gd[gd.reached.astype(bool)].iterrows():
            goals.append(dict(run=x.run, pid=x.pid, order=int(x.order), condition=x.condition, cond=COND.get(x.condition),
                              practice=bool(x.practice), hat=x.hat, leg=int(x.zoneOrder), search=float(x.searchSec),
                              transit=float(x.transitSec), outcome=x.outcome if isinstance(x.outcome, str) else "",
                              decision=float(x.decisionTimeSec) if x.decisionTimeSec >= 0 else None,
                              sight_to_press=float(x.sightToPressSec) if "sightToPressSec" in x and pd.notna(x.sightToPressSec)
                              and x.sightToPressSec >= 0 else None))
    total_col = "time_s" if is_timed else "durationSec"
    # the phases only add up over a run that reached every goal; a series with no task times has no total to compare
    complete = counted[counted.goalsReached >= counted.nGoals]
    stats = dict(total=compare(counted, total_col) if is_timed else None, search=compare(complete, "search_s"),
                 transit=compare(complete, "transit_s"), smooth={m: compare(counted, m) for m in SMOOTH if m in counted})
    # the swarm against one drone's 360 camera (SingleSphere, from internal_4's SIMI), over the pilots who flew both
    if (counted.condition == "SingleSphere").any():
        stats["sphere"] = dict(total=compare(counted, total_col, SPHERE) if is_timed else None,
                               search=compare(complete, "search_s", SPHERE), transit=compare(complete, "transit_s", SPHERE),
                               smooth={m: compare(counted, m, SPHERE) for m in SMOOTH if m in counted})
    rp, files = replays(T, dest)
    pg = pages()
    names = series_list()
    i = names.index(test) if test in names else -1
    nav = dict(prev=names[i - 1] if i > 0 else None, next=names[i + 1] if 0 <= i < len(names) - 1 else None)
    runs = T["runs"]
    data = dict(
        built=time.strftime("%Y-%m-%d %H:%M"), test=test, notes=notes, timed=is_timed, total_col=total_col,
        pages=pg, hub=HUB_URL, nav=nav, zone=T["cfg"].zone, range=T["cfg"].range,
        pilots=sorted({r.pid for r in runs}), n_runs=len(runs), n_practice=int(rows.practice.astype(bool).sum()),
        dates=sorted({r.date for r in runs}),
        city=[notes["meta"]["city"]] if notes["meta"].get("city") else sorted({r.city for r in runs}),
        start_sources=rows.startSource.value_counts().to_dict() if "startSource" in rows else {},
        time_sources=rows.time_source.value_counts().to_dict() if "time_source" in rows else {},
        runs=rows.to_dict("records"), goals=goals, stats=stats, identify=identify(T), losses=losses(T),
        restarts=restarts(T), setups=setups(T), long_legs=long_legs(T), panorama=panorama(T),
        replay=rp, manual=T["manual"], warnings=list(A.WARNINGS))
    A.WARNINGS.clear()
    (dest / "data.json").write_text(json.dumps(_clean(data), indent=1), encoding="utf-8")
    (dest / f"{test}_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1), encoding="utf-8")
    return render_page(test)


def render_page(test):
    """The page again from the series' data.json and replay files, with the notes and page addresses re-read: for a
    change to the template or the prose, without running the analysis again."""
    dest = OUT / test
    data = json.loads((dest / "data.json").read_text(encoding="utf-8"))
    data.update(notes=read_notes(test), pages=pages(), built=data["built"])
    files = {k: Path(v) for k, v in json.loads((dest / f"{test}_files.json").read_text(encoding="utf-8")).items()}
    full, bare = fill(TEMPLATE, data, files)
    # the page's name in the gallery and the tab: the series' own (the template's is a placeholder)
    name = f"<title>{html.escape(test)} Trial Results</title>"
    full, bare = (x.replace("<title>User trial results</title>", name, 1) for x in (full, bare))
    (dest / f"{test}.html").write_text(full, encoding="utf-8")
    (dest / f"{test}_bare.html").write_text(bare, encoding="utf-8")
    size = len(bare) + sum(Path(v).stat().st_size for v in files.values())
    print(f"[trials_report] {test}: {dest / (test + '.html')} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + "
          f"{len(files)} replay files, {size / 1e6:.1f} MB in all")
    return dest


def build_index():
    out = []
    for test in series_list():
        p = OUT / test / "data.json"
        if not p.exists():
            print(f"[trials_report] {test}: no data.json yet (build its page first); left off the index")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        st = d["stats"]
        sph = st.get("sphere")
        out.append(dict(test=test, notes=d["notes"], timed=d["timed"], pilots=d["pilots"], n_runs=d["n_runs"],
                        n_practice=d["n_practice"], dates=d["dates"], city=d["city"], built=d["built"],
                        stats=dict(total=st["total"], search=st["search"], transit=st["transit"]),
                        sphere=dict(total=sph["total"], search=sph["search"], transit=sph["transit"]) if sph else None,
                        smooth={k: st["smooth"].get(k) for k in ("stick_tv_per_min", "adj_per_min")},
                        lost=dict(swarm=sum(1 for x in d["losses"] if x["condition"] == "Swarm" and not x["practice"]),
                                  runs=sum(1 for x in d["runs"] if x["condition"] == "Swarm" and not x["practice"]))))
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), series=out, pages=pages(), hub=HUB_URL)
    full, bare = fill(INDEX_TEMPLATE, data, {})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trials_index.html").write_text(full, encoding="utf-8")
    (OUT / "trials_index_bare.html").write_text(bare, encoding="utf-8")
    print(f"[trials_report] index: {OUT / 'trials_index.html'} ({len(out)} series)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("page", help="one series' page")
    p.add_argument("test")
    p = sub.add_parser("render", help="one series' page again from its data.json (the template or the prose changed)")
    p.add_argument("test")
    sub.add_parser("index", help="the index of the series built so far")
    sub.add_parser("all", help="every series in docs/trials/, then the index")
    a = ap.parse_args()
    if a.cmd == "page":
        build_page(a.test)
    elif a.cmd == "render":
        render_page(a.test)
    elif a.cmd == "index":
        build_index()
    else:
        for t in series_list():
            build_page(t)
        build_index()


if __name__ == "__main__":
    main()
