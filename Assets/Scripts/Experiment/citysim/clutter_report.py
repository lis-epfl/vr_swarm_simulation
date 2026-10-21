"""The clutter study's figures and its report page, from a run's tables (citysim.clutter_stats writes them).

    python -B -m citysim.clutter_report [--name clutter]   # -> experiment_results/clutter_study/<name>/

figures(): one PNG per measure in <run>/figures/, swarm / single (or the measure itself) over the grid of block
spacing x coverage, one panel per spread. build(): <run>/clutter_study.html, a self-contained page (the template
citysim/clutter_report.html with the run's data in it; d3 from cdnjs) with the heatmaps, the limits of spreading, the
spreads against each other, the trends, the cities and the method. A conclusions.html beside the tables, if there is
one, is set into the page's findings as written.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import PACKAGE_DIR, REPO, layout as lay  # noqa: E402

TEMPLATE = PACKAGE_DIR / "clutter_report.html"
CLUTTER_DIR = REPO / "experiment_results" / "clutter_study"
SIG = 0.05


# ============================================================================= helpers

def clean(v):
    """JSON-safe: NaN / inf to None, numpy to Python, recursively."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def records(df, nd=4):
    if df is None or not len(df):
        return []
    out = []
    for r in df.to_dict("records"):
        out.append({k: (round(v, nd) if isinstance(v, float) and math.isfinite(v) else v) for k, v in r.items()})
    return out


def _read(out, name):
    p = out / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()


def stars(q):
    if q is None or not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


# ============================================================================= figures

def figures(out):
    """PNG heatmaps: every ratio measure (swarm / single, log colour, a star per significance level of q) and every
    absolute swarm measure, three spreads side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from citysim import clutter_stats as cs
    rc, lv = _read(out, "ratio_cells.csv"), _read(out, "level_cells.csv")
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    if not len(rc):
        return
    ps = sorted(set(rc.p_m))
    al = sorted(set(rc.alpha))
    div = LinearSegmentedColormap.from_list("ss", ["#c44f1c", "#eb6834", "#f3c3a8", "#e8e7e3", "#a9c6ee", "#2a78d6",
                                                   "#14467f"])
    seq = LinearSegmentedColormap.from_list("sq", ["#e3f1ea", "#7cc7a3", "#1baf7a", "#0b5e41"])
    labels = {k: lab for k, _, lab, _ in cs.WALK_RATIOS}
    labels.update({k: lab for k, _, _, lab, _ in cs.STATIC_RATIOS})
    spreads = sorted(set(rc.d_ref))

    def grid(df, col):
        G = np.full((len(ps), len(al)), np.nan)
        Q = np.full((len(ps), len(al)), np.nan)
        for r in df.itertuples():
            G[ps.index(r.p_m), al.index(r.alpha)] = getattr(r, col)
            if hasattr(r, "q"):
                Q[ps.index(r.p_m), al.index(r.alpha)] = r.q
        return G, Q

    for metric in sorted(set(rc.metric)):
        fig, axes = plt.subplots(1, len(spreads), figsize=(4.2 * len(spreads) + 1, 4.4), squeeze=False)
        for ax, d in zip(axes[0], spreads):
            G, Q = grid(rc[(rc.metric == metric) & (rc.d_ref == d)], "ratio")
            L = np.log2(G)
            lim = max(0.5, float(np.nanmax(np.abs(L))) if np.isfinite(L).any() else 1.0)
            im = ax.imshow(L, origin="lower", cmap=div, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
            for i in range(len(ps)):
                for j in range(len(al)):
                    if np.isfinite(G[i, j]):
                        ax.text(j, i, f"{G[i, j]:.2f}{stars(Q[i, j])}", ha="center", va="center", fontsize=6.5,
                                color="white" if abs(L[i, j]) > 0.6 * lim else "#111c25")
                    else:
                        ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#7c8893")
            ax.set_xticks(range(len(al)), [f"{a:g}" for a in al], fontsize=7)
            ax.set_yticks(range(len(ps)), [f"{p:g}" for p in ps], fontsize=7)
            ax.set_xlabel("coverage α")
            ax.set_ylabel("block spacing p (m)")
            ax.set_title(f"spread {d:g}", fontsize=9)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="log2 swarm / single")
        fig.suptitle(f"{labels.get(metric, metric)}: swarm ÷ single drone (* q<0.05, ** <0.01, *** <0.001)", fontsize=10)
        fig.savefig(fig_dir / f"ratio_{metric}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    if not len(lv):
        return
    lab2 = {k: lab for k, _, lab, _ in cs.WALK_LEVELS + cs.STATIC_LEVELS}
    for metric in sorted(set(lv.metric)):
        sub = lv[lv.metric == metric]
        arms = [("swarm", d) for d in spreads] + ([("single", 1.0)] if (sub.arm == "single").any() else [])
        fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms) + 1, 4.4), squeeze=False)
        vmin, vmax = np.nanmin(sub["mean"]), np.nanmax(sub["mean"])
        for ax, (arm, d) in zip(axes[0], arms):
            G, _ = grid(sub[(sub.arm == arm) & np.isclose(sub.d_ref, d)], "mean")
            im = ax.imshow(G, origin="lower", cmap=seq, vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(ps)):
                for j in range(len(al)):
                    if np.isfinite(G[i, j]):
                        ax.text(j, i, f"{G[i, j]:.2g}", ha="center", va="center", fontsize=6.5,
                                color="white" if (G[i, j] - vmin) > 0.6 * (vmax - vmin + 1e-12) else "#111c25")
            ax.set_xticks(range(len(al)), [f"{a:g}" for a in al], fontsize=7)
            ax.set_yticks(range(len(ps)), [f"{p:g}" for p in ps], fontsize=7)
            ax.set_xlabel("coverage α")
            ax.set_ylabel("block spacing p (m)")
            ax.set_title("single drone" if arm == "single" else f"swarm, spread {d:g}", fontsize=9)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        fig.suptitle(lab2.get(metric, metric), fontsize=10)
        fig.savefig(fig_dir / f"level_{metric}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


# ============================================================================= what each measure is

# key: (what it is, how it is measured, how to read it) -- the page's "About this measure" box and its glossary.
# Numbers quoted here are the run's settings: 40 m recognition range; walks at internal_3's swarm transit median
# (19.5 m, -15 deg: ground from about 23 to 36 m out of each camera), standing at its search median (12.4 m, -21 deg).
DESCRIPTIONS = {
    "sweep": (
        "How wide a strip of city one pass clears.",
        "The ground beside the track is grouped by its distance from the track, in 2 m strips either side. Each strip scores the share of its ground that ever appeared on a shown feed within 40 m; the "
        "shares times 2 m are added up. A sensor that saw everything within 20 m either side, and nothing beyond, "
        "would score 40 m.",
        "Swarm ÷ single drone: how many times wider a strip the swarm clears per pass. Speed does not enter it."),
    "rate": (
        "New ground seen per second.",
        "Every 2 m patch of ground that first appeared on a shown feed within 40 m during the walk, divided by the "
        "walk's duration.",
        "Roughly sweep width times speed, less ground seen twice. An interface that has to fly slower pays for it "
        "here."),
    "per_m": (
        "New ground seen per metre flown.",
        "The same ground divided by the distance the formation's centre (or the drone) flew.",
        "Between the two above: it counts the walk crossing its own path, but not its speed."),
    "sweep_head": (
        "Sweep width counting only the screens in the headset's view.",
        "As sweep width, but a swarm feed counts only while its screen sits within ±50° of straight ahead. Flying, "
        "the head faces the direction of travel. The single drone's one feed is always in view.",
        "What the swarm gives a pilot who does not look round."),
    "inst": (
        "Ground on screen at any one moment.",
        "The ground within 40 m visible on all the shown feeds together at an instant, averaged over every half "
        "second of the walk.",
        "How much there is to watch at once: the swarm's ring of about eight feeds against one."),
    "inst_head": (
        "Ground on the screens in the headset's view at any one moment.",
        "As above, counting only screens within ±50° of straight ahead, with the head facing the direction of travel.",
        "Closer to what a pilot can take in at once without turning."),
    "find": (
        "Chance that a target hidden at a random spot is found during the walk.",
        "The target is a point at hat height anywhere on the city's open ground (no goal patch, no walkers). It is "
        "found the first time a shown feed has it within 200 m with a clear sight line (the distance internal_3's "
        "pilots turned onto goals from; recognising a hat still takes 40 m), so the chance it has been found by any "
        "time is the share of the city's open ground seen by then. A walk lasts 5 minutes; the city is 730 m across.",
        "Swarm ÷ single drone: how much likelier the swarm is to have found it. The Finding tab shows the chance "
        "growing through the walk."),
    "find_speed": (
        "How much sooner a hidden target is found, on average.",
        "The random-search model: the chance of having found it by time t is 1 − e^(−λt). λ is fitted to the share "
        "found by the end of the walk; 1/λ is the mean time to find it.",
        "Swarm ÷ single drone of λ: how many times sooner the swarm finds a target hidden anywhere in the city."),
    "st_any": (
        "Ground on screen while hovering.",
        "The formation is flown to a random open point and held there for 60 s by the centring pilot; the single "
        "drone hovers at its centre, facing the same way. Ground within 40 m on the shown feeds, averaged over four "
        "facing directions, at internal_3's search altitude and pitch (12.4 m, −21°).",
        "Swarm ÷ single drone, standing still."),
    "st_head": (
        "Ground on the screens in the headset's view while hovering.",
        "As above, counting only screens in the headset's view. The head direction is weighted by how often "
        "internal_3's swarm pilots looked each way while searching: mostly within ±45° of straight ahead.",
        "Swarm ÷ single drone for a pilot who looks round as internal_3's pilots did."),
    "st_1s": (
        "Ground the pilot can bring into view within one second.",
        "Swarm: the head turns at 180°/s for up to 90°, then the body yaw turns the whole ring at 90°/s; within 1 s "
        "that reaches every screen. Single drone: the drone yaws at 75°/s, so its view reaches ±112° of where it "
        "pointed.",
        "Swarm ÷ single drone: what each can show the pilot after a short look round."),
    "ring_conn": (
        "How often neighbouring feeds show some of the same ground.",
        "Feeds next to each other on the ring of screens are paired. A pair is connected when at least 20 m² of "
        "ground within 40 m appears on both, through clear sight lines. The share of pairs connected, over the walk.",
        "Low values mean the screens show separate pieces of the city, so a pilot cannot follow something from one "
        "screen to the next."),
    "ring_geo_conn": (
        "The same, ignoring the buildings.",
        "As above with every sight line clear: what the camera geometry alone allows. A wider formation sets its "
        "cameras further apart, so their views meet further out.",
        "The gap between this and the measure with buildings is what the buildings take."),
    "ring_whole": (
        "Time the whole ring of screens forms one unbroken view.",
        "The share of the walk during which every neighbouring pair of feeds is connected.",
        "Higher is better."),
    "ring_overlap": (
        "Ground a neighbouring pair of feeds shares.",
        "The mean area (m²) within 40 m that both feeds of a neighbouring pair show through clear sight lines.",
        "Higher means more of each screen continues onto the next."),
    "wall_feeds": (
        "Feeds filled by a building.",
        "The share of shown feeds where more than half the view meets a facade within 5 m: a drone pressed against "
        "a building.",
        "Lower is better. It rises when the formation is wider than the street."),
    "uncovered": (
        "Horizon no feed covers.",
        "Degrees round the formation that lie outside every shown feed's 74° field of view.",
        "Lower is better. It rises when drones leave the ring (inside the outline, or turning)."),
    "facade": (
        "Distance from a camera to the nearest building.",
        "The median, over the shown cameras and the walk, of the distance to the nearest building face.",
        "Lower means the cameras fly closer to walls."),
    "shown": (
        "Feeds on the ring.",
        "Drones on the formation's outline whose camera has turned to its outward heading, averaged over the walk.",
        "Out of ten drones."),
    "behind": (
        "Ground the swarm saw round corners.",
        "Of the ground the swarm saw, the share its feeds saw first while a camera at the formation's centre could "
        "not, because a building blocked the line.",
        "What the spread of cameras adds over one viewpoint at the centre."),
    "find_time": (
        "Mean time to find a hidden target (estimate).",
        "1/λ of the random-search model fitted to the share of the city seen within 200 m in the 5-minute walk, in "
        "minutes.",
        "Lower is better."),
    "find_p": (
        "Chance of finding a hidden target within the walk.",
        "The share of the city's open ground seen within 200 m during the 5-minute walk.",
        "Higher is better."),
    "lost_km": (
        "Drones lost per km.",
        "Drones lost (collisions, or too far from the swarm) per km the formation's centre flew.",
        "Lower is better. The replica's collision rate at spread 0.4 is known to be high."),
    "lost": ("Drones lost per walk.", "Out of ten, over the 5-minute walk.", "Lower is better."),
    "split": (
        "Time the formation is split.",
        "The share of the walk the formation was more than 2.5 times its settled width across.",
        "Lower is better."),
    "speed": ("Speed.", "Mean speed of the formation's centre (or the drone).", "Higher is better."),
    "path_ratio": (
        "Distance flown per metre of progress.",
        "Distance flown divided by the net progress made toward the walk's targets.",
        "Closer to 1 is better: more detours mean more steering round buildings."),
    "adj": (
        "Stick adjustments per 100 m.",
        "Changes of the scripted stick per 100 m of progress, counted the way the human runs are scored.",
        "Lower means less steering work."),
    "morph": (
        "Buildings the formation flowed round, per km.",
        "Distinct buildings that stood inside the formation's outline while it stayed whole, per km.",
        "How often the swarm morphs round a building instead of steering round it."),
    "street": (
        "Path along the streets.",
        "The share of the path flown over a street rather than over a block.",
        "The swarm may cross blocks between buildings; the single drone plans round every building."),
    "first": (
        "Ground the swarm saw first.",
        "Of the ground both saw on the same walk, the share the swarm saw before the single drone did.",
        "Above 0.5: the swarm usually sees things first."),
    "st_ring_conn": (
        "Neighbouring feeds that share ground, hovering.",
        "As for walks, for the settled formation at a random open point.", "Higher is better."),
    "st_ring_geo_conn": (
        "The same, ignoring the buildings, hovering.", "As for walks, settled.",
        "The gap to the measure with buildings is what the buildings take."),
    "st_wall_feeds": ("Feeds filled by a building, hovering.", "As for walks, settled.", "Lower is better."),
    "st_useful": (
        "Useful feeds, hovering.",
        "Feeds that are less than half wall and show at least 10% of ground no other feed shows.", "Out of ten."),
    "st_epar": (
        "Hidden ground seen by parallax, hovering.",
        "Of the ground in the feeds' directions that a camera at the formation's centre cannot see, the share some "
        "feed sees.", "What spreading the cameras uncovers."),
}

HEAD_MODEL = (
    "Screens sit on a ring 2 m round the pilot, each at its drone's heading relative to the body yaw; a 74° feed "
    "fills 48° of the ring. The headset sees ±50°. Flying, the head faces the direction of travel, because in transit "
    "internal_3's swarm pilots looked within 25° of straight ahead 87% of the time, and the body yaw turns to the "
    "direction flown at 90°/s. Hovering, the head direction is weighted by internal_3's recorded head directions while "
    "searching. The single drone's one feed is always in front.")


# ============================================================================= the page's data

def _tracks(out):
    """{layout: {condition: [[x, z], ...]}} of the first walk of each city that kept its tracks."""
    tr = {}
    p = out / "results.jsonl"
    if not p.exists():
        return tr
    for line in p.read_text(encoding="utf-8").splitlines():
        if '"tracks"' not in line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "tracks" in r:
            tr[r["job_id"].split("|")[0]] = r["tracks"]
    return tr


def city_maps(cities, tracks):
    """One city per flown cell (the first whose first walk kept its tracks): its building rectangles and tracks."""
    maps = {}
    if not len(cities):
        return maps
    flyable = cities.buildable.astype(bool) & (cities.fly.astype(bool) if "fly" in cities else True)
    for cell, g in cities[flyable].groupby("cell"):
        g = g.sort_values("seed")
        pick = next((r for r in g.itertuples() if r.layout in tracks), g.iloc[0])
        path = getattr(pick, "path", None) if not isinstance(pick, pd.Series) else pick["path"]
        layout = getattr(pick, "layout", None) if not isinstance(pick, pd.Series) else pick["layout"]
        seed = getattr(pick, "seed", None) if not isinstance(pick, pd.Series) else pick["seed"]
        try:
            L = lay.load(path if Path(str(path)).exists() else lay.resolve(layout))   # by id, if the folder moved
        except Exception:                        # noqa: BLE001 -- a missing layout costs its map
            continue
        R = L.rects()
        maps[cell] = dict(layout=layout, seed=int(seed), arena=L.layout["arena"],
                          rects=np.round(R[:, :4], 1).tolist() if len(R) else [],
                          streets=[[round(float(v), 1) for v in st["xz"]] for st in (L.streets or [])],
                          street_w=float(L.layout.get("s", 15.0)), tracks=tracks.get(layout, {}))
    return maps


def findings(rc, lv, tr, sc):
    """Plain sentences with the numbers behind them, written from the tables (the conclusions are written by hand)."""
    from citysim import clutter_stats as cs
    out = []
    if not len(rc):
        return out
    lab = {k: lab for k, _, lab, _ in cs.WALK_RATIOS}
    lab.update({k: lab for k, _, _, lab, _ in cs.STATIC_RATIOS})
    for metric in (cs.PRIMARY, "rate", "st_any"):
        for d in sorted(set(rc.d_ref)):
            g = rc[(rc.metric == metric) & (rc.d_ref == d)].dropna(subset=["ratio"])
            if not len(g):
                continue
            hi, lo = g.loc[g.ratio.idxmax()], g.loc[g.ratio.idxmin()]
            sig_sw = int(((g.q < SIG) & (g.ratio > 1)).sum())
            sig_sg = int(((g.q < SIG) & (g.ratio < 1)).sum())
            out.append(dict(metric=metric, d_ref=d, text=(
                f"{lab.get(metric, metric)}, spread {d:g}: the swarm ÷ single drone ratio runs from "
                f"{lo.ratio:.2f} (p = {lo.p_m:g} m, α = {lo.alpha:g}) to {hi.ratio:.2f} (p = {hi.p_m:g} m, "
                f"α = {hi.alpha:g}); the swarm is ahead with q < 0.05 in {sig_sw} of {len(g)} cells and the single "
                f"drone in {sig_sg}.")))
    if len(tr):
        for d in sorted(set(tr.d_ref)):
            t = tr[(tr.metric == cs.PRIMARY) & (tr.d_ref == d)].set_index("term")
            if "alpha_per_0.01" in t.index:
                a, p_ = t.loc["alpha_per_0.01"], t.loc["log2_p"]
                out.append(dict(metric="trend", d_ref=d, text=(
                    f"Trend of the sweep-width ratio at spread {d:g} over {int(a.n_cities)} cities: "
                    f"{a.pct:+.1f} % per +0.01 coverage (p = {a.p:.2g}) and {p_.pct:+.1f} % per doubling of the block "
                    f"spacing (p = {p_.p:.2g}); R² {a.r2:.2f}.")))
    if len(lv):
        for key, what in (("ring_conn", "neighbouring feeds that share ground at 40 m"),
                          ("lost_km", "drones lost per km")):
            g = lv[(lv.metric == key) & (lv.arm == "swarm")]
            if len(g):
                parts = [f"{r['mean']:.2f} at spread {d:g}" for d, r in
                         g.groupby("d_ref")["mean"].agg(["mean"]).iterrows()]
                out.append(dict(metric=key, d_ref=None, text=f"Mean over the grid, {what}: " + ", ".join(parts) + "."))
    return out


def find_curve_data(fc):
    """{cell: {condition: dict(t, m, s, n)}} -- the time-to-find curves, compact."""
    out = {}
    if not len(fc):
        return out
    for (cell, arm, d), g in fc.groupby(["cell", "arm", "d_ref"]):
        g = g.sort_values("t")
        out.setdefault(cell, {})[f"{arm}_{d:g}"] = dict(
            t=[round(float(v), 1) for v in g.t], m=[round(float(v), 5) for v in g["mean"]],
            s=[(round(float(v), 5) if np.isfinite(v) else None) for v in g.sd_city], n=int(g.n_cities.iloc[0]))
    return out


def sweep_window_note(spec):
    """What the sweep width's strips reached in this run. Runs before spec.sweep_window_m existed stopped them at the
    40 m range plus one strip, which the swarm nearly fills (S3's ceiling, docs/studies/lessons.md 1)."""
    window = spec.get("sweep_window_m")
    if window:
        edge = window + 2.0
        return (f" The strips run {edge:g} m out (at most {2 * edge:g} m), wider than the swarm's outer cameras reach "
                "at every spread.")
    return (" In this run the strips stop 42 m out, so the swarm's score saturates near the 84 m maximum and what its "
            "outer cameras see beyond 42 m is not counted: the swarm's ratio is understated, by about 6–7% at spread "
            "1.0 and more at 1.6. Re-measured over ±82 m, spread 1.6 sweeps 6–13% more than 1.0, so compare spreads on "
            "ground per metre or per second here. The code now measures to 80 m either side (from 2026-10-21).")


def page_data(out):
    from citysim import clutter_stats as cs
    rc, sc, lv, tr = (_read(out, n) for n in ("ratio_cells.csv", "spread_contrasts.csv", "level_cells.csv",
                                               "trends.csv"))
    csum, lk = _read(out, "city_summary.csv"), _read(out, "links.csv")
    keep = ["cell", "p", "alpha"] + [k for k, _ in cs.LINK_X + cs.LINK_Y if k not in ("alpha",)]
    csum = csum[[c for c in dict.fromkeys(keep) if c in csum]] if len(csum) else csum
    cities = _read(out, "cities.csv")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8")) if (out / "spec.json").exists() else {}
    rchk = _read(out, "recheck.csv")
    errs = _read(out, "errors.csv")
    walks = _read(out, "walks.csv")
    cells = []
    if len(cities):
        for cell, g in cities.groupby("cell"):
            b = g[g.buildable.astype(bool)]
            row = dict(cell=cell, p=float(g.p.iloc[0]), alpha=float(g.alpha.iloc[0]), asked=len(g), buildable=len(b),
                       flown=bool(g.cell_flown.iloc[0]), why=";".join(sorted(set(g.why.dropna()) - {""}))[:120])
            for col in ("alpha_achieved", "block_fill", "ell_m", "n_buildings", "n_blocks", "wall_share",
                        "encounters_per_km", "nn_cyl_gap_median_m", "max_within_50m"):
                if col in b and len(b):
                    row[col] = float(b[col].mean())
            cells.append(row)
    flight_h = float(walks.transit_s.sum() / 3600.0) if len(walks) and "transit_s" in walks else None
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except Exception:                            # noqa: BLE001
        pass
    concl = (out / "conclusions.html").read_text(encoding="utf-8") if (out / "conclusions.html").exists() else ""
    descriptions = {k: dict(what=w, how=h, read=r) for k, (w, h, r) in DESCRIPTIONS.items()}
    descriptions["sweep"]["read"] += sweep_window_note(spec)
    return dict(
        meta=dict(generated=time.strftime("%Y-%m-%d %H:%M"), commit=commit, run=man, spec=spec,
                  name=out.name, flight_hours=flight_h,
                  recheck=dict(n=int(len(rchk)), identical=int(rchk.identical.sum()) if len(rchk) else 0),
                  failed=int(len(errs)), walks=int(walks.job_id.nunique()) if len(walks) else 0,
                  cities_flown=int(walks.layout.nunique()) if len(walks) else 0,
                  overlap_min_m2=20.0, range_m=40.0),
        grid=dict(p=sorted(set(c["p"] for c in cells)), alpha=sorted(set(c["alpha"] for c in cells)), cells=cells),
        metrics=dict(
            ratio=[dict(key=k, label=lab, unit=u, kind="walk") for k, _, lab, u in cs.WALK_RATIOS] +
                  [dict(key=k, label=lab, unit=u, kind="static") for k, _, _, lab, u in cs.STATIC_RATIOS],
            level=[dict(key=k, label=lab, unit=u, kind="walk") for k, _, lab, u in cs.WALK_LEVELS] +
                  [dict(key=k, label=lab, unit=u, kind="static") for k, _, lab, u in cs.STATIC_LEVELS],
            primary=cs.PRIMARY),
        descriptions=descriptions, head_model=HEAD_MODEL,
        find_curves=find_curve_data(_read(out, "find_curves.csv")),
        ratio_cells=records(rc), contrasts=records(sc), levels=records(lv), trends=records(tr),
        city_summary=records(csum), links=records(lk),
        link_axes=dict(x=[dict(key=k, label=lab) for k, lab in cs.LINK_X],
                       y=[dict(key=k, label=lab) for k, lab in cs.LINK_Y]),
        maps=city_maps(cities, _tracks(out)), findings=findings(rc, lv, tr, sc), conclusions=concl)


def build(out, path=None):
    """The page twice: <run>/clutter_study.html to open as a file, and <run>/clutter_study_bare.html without the
    html/head/body skeleton (what an artifact is published from)."""
    out = Path(out)
    data = page_data(out)
    text = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "__DATA__" in html, "the template has no __DATA__ placeholder"
    bare = html.replace("__DATA__", text)
    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1,viewport-fit=cover\"></head><body>" + bare + "</body></html>")
    dest = Path(path) if path else out / "clutter_study.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name(dest.stem + "_bare.html").write_text(bare, encoding="utf-8")
    print(f"[clutter] page {dest} ({len(full) / 1e6:.2f} MB)")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="clutter")
    ap.add_argument("--out", help="the page's path (default: <run>/clutter_study.html)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    out = CLUTTER_DIR / a.name
    if not a.no_figures:
        figures(out)
    build(out, path=a.out)


if __name__ == "__main__":
    main()
