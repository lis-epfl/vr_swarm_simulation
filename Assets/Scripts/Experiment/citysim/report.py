"""Figures for the visibility-scaling study (matplotlib, PNG), written beside the tables they draw.

    python -B -m citysim.report --sweep tierB2 [--levels L1 L2 L3]

Colours follow the project's data-viz palette: swarm = categorical slot 1 (blue), single drone = slot 2 (orange) --
the two slots that validate against each other in both modes -- text in ink, never in a series colour; a single
blue ramp for the envelopment map's magnitude; one axis per chart.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import LAYOUT_DIR, STUDY_DIR, camera, catalogue, geometry as geo, layout as lay, theory  # noqa: E402

SWARM, SINGLE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
BUILDING, GOAL_C, GOAL_R, STREET = "#9a9994", "#52514e", "#2a78d6", "#e4e3df"


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    p.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                       "axes.titlecolor": INK, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                       "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                       "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9,
                       "legend.frameon": False})
    return p


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.08)
    import matplotlib.pyplot as p
    p.close(fig)
    print(f"[plot] {path}")


# ============================================================================= Tier A

def fig_tierA(out):
    p = plt()
    d = catalogue.study_design()
    alphas = np.linspace(0.005, 0.35, 200)
    fig, ax = p.subplots(figsize=(6.2, 3.8))
    for s, ls in ((1.0, ":"), (2.0, "-"), (4.0, "--")):
        ell = [theory.ell(a, d.w * s, d.d * s) for a in alphas]
        ax.plot(alphas, ell, color=SWARM if s == 2 else INK2, lw=1.6, ls=ls, label=f"x{s:g} ({d.w * s:.0f} x {d.d * s:.0f} m)")
    ax.axhline(camera.R_ID_M, color=SINGLE, lw=1.2)
    ax.text(0.345, camera.R_ID_M * 1.08, f"recognition range R = {camera.R_ID_M:g} m", ha="right", color=INK2, fontsize=8)
    chk = STUDY_DIR / "tierA" / "tierA_vs_scatter.csv"
    for x, y, lab in ((0.0235, 244, "DiamondCityWorld"), (0.0249, 222, "ScaledCityWorld")):
        ax.scatter([x], [y], s=36, color=INK, zorder=4)
        ax.annotate(lab, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("coverage alpha (built share of the ground)")
    ax.set_ylabel("free-sight length ell (m)")
    ax.set_title(f"Tier A: the Boolean model, study design {d.design}", loc="left", fontsize=10)
    ax.legend(title="building size", fontsize=8, title_fontsize=8)
    save(fig, out / "tierA_ell.png")
    if chk.exists():
        c = pd.read_csv(chk)
        fig, ax = p.subplots(figsize=(6.2, 3.6))
        for r, m in zip(sorted(c.r.unique()), ["o", "s", "^", "D"]):
            g = c[c.r == r]
            ax.plot(g.alpha, g.boolean, color=INK2, lw=1.2)
            ax.scatter(g.alpha, g.measured, color=SWARM, marker=m, s=40, zorder=3, label=f"r = {r:g} m")
        ax.set_xlabel("coverage alpha (scatter layouts)")
        ax.set_ylabel("P(sight line of length r is free)")
        ax.set_title("Tier B (points, scatter layouts) against Tier A (lines, exp(-r / ell))", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        save(fig, out / "tierA_vs_scatter.png")


# ============================================================================= Tier B

def best_swarm_single(exp):
    """Per (level, seed, goal, altitude, pitch): the swarm at its best spread, the best realistic single policy."""
    keys = ["level", "clutter", "goal_level", "seed", "goal", "altitude", "pitch"]
    sw = exp[exp.interface == "swarm"].groupby(keys)[["E_head", "E_any"]].max()
    sg = exp[(exp.interface == "single") & exp.policy.isin(["orbit_in", "orbit_out", "scan"])]
    sg = sg.groupby(keys + ["policy"])[["E_head", "E_any"]].mean().groupby(level=keys).max()
    return sw, sg


def fig_exposure(exp, levels_order, out, name="tierB_exposure.png", metric="E_head"):
    p = plt()
    sw, sg = best_swarm_single(exp)
    a = sw[metric].groupby(["level", "seed"]).mean().groupby("level").agg(["mean", "std"])
    b = sg[metric].groupby(["level", "seed"]).mean().groupby("level").agg(["mean", "std"])
    lv = [l for l in levels_order if l in a.index]
    x = np.arange(len(lv))
    fig, ax = p.subplots(figsize=(max(6.0, 0.42 * len(lv) + 2), 3.8))
    ax.errorbar(x - 0.12, a.loc[lv, "mean"], a.loc[lv, "std"], fmt="o", ms=6, color=SWARM, lw=1.4,
                label="swarm (feed ring, best spread)")
    ax.errorbar(x + 0.12, b.loc[lv, "mean"], b.loc[lv, "std"], fmt="s", ms=6, color=SINGLE, lw=1.4,
                label="single drone (best of orbit / scan)")
    ax.set_xticks(x, lv, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel({"E_head": "E_head: share of walker-time on a\nhead-visible feed",
                   "E_any": "E_any: share of walker-time on\nany displayed feed"}[metric])
    ax.set_ylim(0, None)
    ax.set_title(f"Tier B: target exposure by level (mean over seeds, goals and the operator grid; bars = SD over seeds)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, out / name)


def fig_delta(levels, l1, out, name="tierB_delta.png", picks=None):
    p = plt()
    lv = levels.sort_values("delta", ascending=False)
    names = {v: k for k, v in (picks or {}).items() if v}
    fig, ax = p.subplots(figsize=(max(6.0, 0.42 * len(lv) + 2), 3.6))
    ax.set_axisbelow(True)
    x = np.arange(len(lv))
    colours = [SWARM if d > 0 else SINGLE for d in lv.delta]
    ax.bar(x, lv.delta, color=colours, width=0.62)
    ax.errorbar(x, lv.delta, lv.sd, fmt="none", ecolor=INK2, lw=1.0)
    d1 = float(levels.loc[levels.level == l1, "delta"].iloc[0])
    ax.axhline(d1, color=INK2, lw=1.0, ls=":")
    ax.axhline(2 * d1, color=INK2, lw=1.0, ls="--")
    pad = 0.012 * (lv.delta.max() - min(lv.delta.min(), 0.0))
    ax.text(len(lv) - 0.5, d1 + pad, f" {l1} (L1)", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.text(len(lv) - 0.5, 2 * d1 + pad, " 2 x L1 (G1's bar)", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"{names[l]}: {l}" if l in names else l for l in lv.level], rotation=60, ha="right",
                  fontsize=8)
    for t in ax.get_xticklabels():
        if t.get_text().split(":")[0] in ("L1", "L2", "L3"):
            t.set_fontweight("bold")
    ax.set_ylabel("swarm - single, E_head")
    ax.set_title("G1: the swarm's exposure advantage by level (blue = swarm ahead, orange = single drone ahead)",
                 loc="left", fontsize=9)
    save(fig, out / name)


def fig_reveal(geo_df, levels_pick, out, name="tierB_reveal.png"):
    p = plt()
    ts = [0, 1, 3, 6]
    fig, axs = p.subplots(1, len(levels_pick), figsize=(3.2 * len(levels_pick), 3.2), squeeze=False)
    for ax, lv in zip(axs[0], levels_pick):
        g = geo_df[geo_df.level == lv]
        if not len(g):
            continue
        best = g.groupby("d_ref").V_swarm_0.mean().idxmax()
        g = g[g.d_ref == best]
        vs = [g[f"V_swarm_{t}"].mean() for t in ts] + [g["V_swarm_inf"].mean()]
        vg = [g[f"V_single_{t}"].mean() for t in ts] + [g["V_single_inf"].mean()]
        xs = ts + [9]
        ax.plot(xs, vs, "o-", color=SWARM, lw=1.6, ms=5, label="swarm")
        ax.plot(xs, vg, "s-", color=SINGLE, lw=1.6, ms=5, label="single drone")
        ax.set_xticks(xs, ["0", "1", "3", "6", "inf"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("seconds to bring into view")
        ax.set_title(f"{lv} (d_ref {best:g})", fontsize=9, loc="left")
    axs[0][0].set_ylabel("share of ground within R\nthe pilot can have on screen")
    axs[0][-1].legend(fontsize=8, loc="lower right")
    save(fig, out / name)


# ============================================================================= Tier B': envelopment map

def fig_envmap(env, out, name="envmap.png", n=10):
    p = plt()
    from matplotlib.colors import ListedColormap
    def truth(col):     # a CSV column of True / False with blanks for infeasible rows reads back as objects
        return col.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)

    env = env.assign(feasible_geometry=truth(env.feasible_geometry) > 0.5, enveloped=truth(env.enveloped))
    e = env[(env.feasible_geometry) & (env.n == n)]
    drefs = sorted(e.d_ref.unique())
    fig, axs = p.subplots(1, len(drefs), figsize=(2.6 * len(drefs) + 0.8, 3.3), squeeze=False, sharey=True)
    cmap = ListedColormap(BLUES)
    for ax, d in zip(axs[0], drefs):
        t = e[e.d_ref == d].groupby(["w_c", "r_g"]).enveloped.mean().unstack()
        im = ax.imshow(t.values, origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(t.columns)), [f"{v:g}" for v in t.columns], fontsize=7)
        ax.set_yticks(range(len(t.index)), [f"{v:g}" for v in t.index], fontsize=7)
        ax.tick_params(axis="y", labelleft=ax is axs[0][0])
        ax.set_xlabel("ring radius r_g (m)")
        ax.set_title(f"d_ref {d:.1f}", fontsize=9, loc="left")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(t.values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color=INK if v < 0.6 else SURFACE)
    axs[0][0].set_ylabel("occluder w_c (m)")
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.85, label="share of settles enveloped")
    fig.suptitle(f"Tier B': where {n} drones envelop the goal's occluder (arriving from 60 m, centring pilot); "
                 "blank: no room for the ring round that occluder", x=0.01, ha="left", fontsize=9, y=1.04)
    save(fig, out / name)


# ============================================================================= Tier C

def fig_tierC(lev, picks, out, loss_max, name="tierC_losses.png"):
    """Drones lost per run by level (tierc.per_run_losses: internal_3's transit-spread use), ten drones and one
    where it flew, against G3's bar."""
    p = plt()
    names = {v: k for k, v in picks.items() if v}
    order = [picks[k] for k in ("L1", "L2", "L3") if picks.get(k) in set(lev.level)]
    order += sorted(l for l in lev.level if l not in order)
    lv = lev.set_index("level").loc[order]
    x = np.arange(len(order))
    fig, ax = p.subplots(figsize=(max(6.0, 0.6 * len(order) + 2.5), 3.6))
    ax.errorbar(x - 0.1, lv.loss_rate, lv.loss_rate_sd.fillna(0), fmt="o", ms=6, color=SWARM, lw=1.4,
                label="swarm, 10 drones (bars: SD over layout seeds)")
    if "single_loss_rate" in lv and lv.single_loss_rate.notna().any():
        ax.plot(x + 0.1, lv.single_loss_rate, "s", ms=6, color=SINGLE, label="single drone", clip_on=False,
                zorder=4)
    ax.axhline(loss_max, color=INK2, lw=1.0, ls="--")
    ax.text(len(order) - 1.6, loss_max * 1.02, f"G3's bar: {loss_max:g} per run", va="bottom", ha="right",
            fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{names[l]}\n{l}" if l in names else l for l in order], fontsize=8)
    ax.set_ylim(0, max(loss_max, float(np.nanmax(lv.loss_rate + lv.loss_rate_sd.fillna(0)))) * 1.12)
    ax.set_ylabel("drones lost per run\n(spawn -> 3 goals, internal_3's spread use)")
    ax.set_title("Tier C: losses in the headless Unity bench, the picks and their neighbours", loc="left", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, out / name)


# ============================================================================= maps

def fig_layout(path, ax, title=None):
    from matplotlib.collections import LineCollection, PolyCollection
    L = lay.load(path)
    ax.add_collection(LineCollection([s["polyline"] for s in L.streets], colors=STREET,
                                     linewidths=[max(s["width"] * 0.18, 0.5) for s in L.streets]))
    polys, cols = [], []
    for b in L.boxes:
        polys.append(geo.rect_corners(lay.box_rect(b)))
        cols.append({"city": BUILDING, "goal_centre": GOAL_C, "goal_ring": GOAL_R}[b.get("role", "city")])
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="none"))
    for g in L.goals():
        ax.add_patch(__import__("matplotlib.pyplot", fromlist=["Circle"]).Circle(
            g["centre"], g["clear_radius"], fill=False, color=INK2, lw=0.6, ls="--"))
    sp = L.layout["spawn"]
    ax.plot(sp[0], sp[2], marker="^", color=INK, ms=7)
    x0, x1, z0, z1 = L.layout["arena"]
    ax.set_xlim(x0 - 110, x1 + 10)
    ax.set_ylim(z0 - 10, z1 + 10)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    r = L.layout
    ax.set_title(title or L.id, fontsize=8, loc="left")
    ax.text(0.0, -0.02, f"alpha {r['alpha_achieved']:.3f}   ell {r['ell_mean']:.0f} m   w_c {r['goals'][0]['w_c']:g} m   "
            f"r_g {r['goals'][0]['r_g']:g} m", transform=ax.transAxes, fontsize=7, color=INK2, va="top")


def fig_maps(paths_titles, out, name):
    p = plt()
    fig, axs = p.subplots(1, len(paths_titles), figsize=(4.2 * len(paths_titles), 4.2), squeeze=False)
    for ax, (path, title) in zip(axs[0], paths_titles):
        fig_layout(path, ax, title)
    fig.text(0.01, -0.04, "triangle: the swarm's spawn; dashed circles: the goals' clearings; dark: the goals' "
             "occluders; blue: their patrolled ring buildings", fontsize=7, color=INK2)
    save(fig, out / name)


# ============================================================================= command line

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweep", default="tierB2")
    ap.add_argument("--l1", default="C0-G0")
    ap.add_argument("--tierC", default="tierC", help="the Tier C plan whose tierC_levels.csv to draw")
    ap.add_argument("--loss-max", type=float, default=2.5, help="G3's bar (select_levels --loss-max)")
    a = ap.parse_args()
    out = STUDY_DIR / "figures"
    lj = STUDY_DIR / "levels.json"
    picks = json.loads(lj.read_text(encoding="utf-8"))["pick"] if lj.exists() else {"L1": a.l1}
    fig_tierA(out)
    src = STUDY_DIR / "tierB" / a.sweep
    if (src / "exposure.csv").exists():
        exp = pd.read_csv(src / "exposure.csv")
        order = sorted(exp.level.unique(), key=lambda s: (s.split("-")[1][0], int(s.split("-")[1][1:]), s))
        fig_exposure(exp, order, out, f"{a.sweep}_exposure.png")
        fig_exposure(exp, order, out, f"{a.sweep}_exposure_any.png", metric="E_any")
        lv = STUDY_DIR / "gate1" / f"levels_{a.sweep}.csv"
        if lv.exists():
            fig_delta(pd.read_csv(lv), a.l1, out, f"{a.sweep}_delta.png", picks)
        g = pd.read_csv(src / "geometry.csv")
        pick = [picks[k] for k in ("L1", "L2", "L3") if picks.get(k) in set(g.level)]
        fig_reveal(g, pick, out, f"{a.sweep}_reveal.png")
    env = STUDY_DIR / "tierBprime" / "envmap.csv"
    if env.exists():
        e = pd.read_csv(env)
        for n in sorted(e.n.unique()):
            fig_envmap(e, out, f"envmap_n{n}.png", n=n)
    trial = []
    for k in ("L1", "L2", "L3"):
        hits = sorted(LAYOUT_DIR.glob(f"{k}_{picks.get(k)}__*_seed101.json")) if picks.get(k) else []
        if hits:
            trial.append((hits[0], f"{k}: {picks[k]} (seed 101)"))
    if trial:
        fig_maps(trial, out, "trial_layouts.png")
    tc = STUDY_DIR / "tierC" / a.tierC / "tierC_levels.csv"
    if tc.exists():
        fig_tierC(pd.read_csv(tc), picks, out, a.loss_max)


if __name__ == "__main__":
    main()
