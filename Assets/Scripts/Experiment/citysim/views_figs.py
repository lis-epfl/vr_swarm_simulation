"""S12 Distributed Views: the proposal's figures (citysim.views, citysim.views_human).

    python -B -m citysim.views_figs all --name N     # -> experiment_results/views_study/<N>/figures/

  recognition.png   the hat's recognition hazard by distance (internal_4, head basis): the binned rates with their
                    exact intervals and the fitted curve, pooled and per condition
  noticing.png      the crowd-noticing hazard by distance, the same way
  tracks.png        at-goal tracks, the people's (internal_4) beside the simulated pilots', per condition
  times.png         per-goal times between goals and at the goal, people against the model, per condition
  views.png         recognitions per minute at the goal on the counterfactual bases (the decomposition)
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import views as V  # noqa: E402

COL = {"swarm": "#2f6fdf", "single": "#d0562f", "sphere": "#2c9a5b", "people": "#444444"}
LABEL = {"swarm": "Swarm", "single": "One drone", "sphere": "360 sphere"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_hazard(out, which="recognition"):
    plt = _plt()
    if which == "recognition":
        R = json.load(open(V.HUMAN / "recognition_fits.json", encoding="utf-8"))
        keys = [("d_head|lat1.0|all", "all conditions", "#444444"), ("d_head|lat1.0|Swarm", "swarm", COL["swarm"]),
                ("d_head|lat1.0|SingleDrone", "one drone", COL["single"]),
                ("d_head|lat1.0|SingleSphere", "360 sphere", COL["sphere"])]
        xmax, a0 = 130, "lam0"
    else:
        R = json.load(open(V.HUMAN / "discovery_fits.json", encoding="utf-8"))
        keys = [("d_head|lat1.0", "all conditions", "#444444")]
        xmax, a0 = 350, "mu0"
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.linspace(1, xmax, 300)
    for k, lab, c in keys:
        if k not in R:
            continue
        f = R[k]["fit"]
        ax.plot(x, V.hill(x, f[a0], f["d50"], f["k"]), color=c, lw=2 if lab == "all conditions" else 1.2,
                label=f"{lab} (d50 {f['d50']:.0f} m)")
        if lab == "all conditions":
            tab = pd.DataFrame(R[k]["table"])
            mid = 0.5 * (tab.d_lo + tab.d_hi) if "d_lo" in tab else np.arange(len(tab)) * 50 + 25
            rate = tab.hazard if "hazard" in tab else tab.rate
            ax.errorbar(mid, rate, yerr=[rate - tab.lo, tab.hi - rate] if "lo" in tab else None, fmt="o",
                        color=c, ms=4, capsize=2, alpha=0.8)
    ax.set_xlabel("distance from the camera (m)")
    ax.set_ylabel("per second on screen" if which == "recognition" else "per second on screen")
    ax.set_title("Hat recognised" if which == "recognition" else "Crowd noticed", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_times(name, out):
    plt = _plt()
    human = pd.read_csv(V.HUMAN / "goal_times.csv")
    _, G, _ = V.frames(V.read(name))
    variants = sorted(G.variant.unique())
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in zip(axs, ("between_s", "at_s"), ("Between goals", "At the goal")):
        pos = 0
        ticks, labs = [], []
        for cond in V.CONDITIONS:
            h = human[human.condition == V.HUMAN_COND[cond]][col].dropna()
            ax.scatter(np.full(len(h), pos) + np.random.default_rng(0).uniform(-0.15, 0.15, len(h)), h, s=10,
                       color=COL["people"], alpha=0.7)
            ax.plot([pos - 0.25, pos + 0.25], [h.median()] * 2, color=COL["people"], lw=2)
            ticks.append(pos)
            labs.append(f"{LABEL[cond]}\npeople")
            pos += 1
            for v in variants:
                g = G[(G.cond == cond) & (G.variant == v)][col].dropna()
                ax.boxplot([g], positions=[pos], widths=0.5, showfliers=False,
                           boxprops=dict(color=COL[cond]), medianprops=dict(color=COL[cond], lw=2),
                           whiskerprops=dict(color=COL[cond]), capprops=dict(color=COL[cond]))
                ticks.append(pos)
                labs.append(f"model\n{v}")
                pos += 1
            pos += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("seconds")
        ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_tracks(log_rows, out, n_people=3):
    """At-goal tracks: each condition's people (first n) and simulated sessions (log_rows: fly(..., keep_log))."""
    plt = _plt()
    from citysim import views_human as VH, use_bench_modules, blockcity
    use_bench_modules()
    import analyse
    C = V.T.city(V.CITY_SEED)
    polys = [blockcity.building_poly(b) for b in C["L"].boxes if blockcity.is_prism(b)]
    _, rr = VH.runs()
    fig, axs = plt.subplots(3, 2 * n_people, figsize=(4.0 * n_people, 6.4))
    for row, cond in enumerate(V.CONDITIONS):
        people = [r for r in rr if r.condition == V.HUMAN_COND[cond]]
        k = 0
        for r in people:
            for L in analyse.goal_legs(r, VH.ZONE_M, rule="zone"):
                if k >= n_people:
                    break
                g = L["goal"]
                ax = axs[row, k]
                _draw_city(ax, polys, (g["goalX"], g["goalZ"]))
                m = (r.ct >= L["enter"]) & (r.ct <= L["exit"])
                if cond == "swarm":
                    sub = r.drones[(r.drones.t >= L["enter"]) & (r.drones.t <= L["exit"]) & (r.drones.alive == 1)]
                    ax.plot(sub.gtX, sub.gtZ, ",", color=COL[cond], alpha=0.25)
                ax.plot(r.cx[m], r.cz[m], "-", color=COL["people"], lw=1)
                ax.set_title(f"{LABEL[cond]}: {r.pid} ({L['exit'] - L['enter']:.0f} s)", fontsize=8)
                k += 1
        sims = [s for s in log_rows if s["cond"] == cond]
        k2 = 0
        for s in sims:
            for gi, geo in enumerate(s["goal_geo"]):
                if k2 >= n_people:
                    break
                lg = [e for e in s["log"] if e[3] == "goal" and e[4] == gi]
                if len(lg) < 3:
                    continue
                ax = axs[row, n_people + k2]
                _draw_city(ax, polys, geo["centre"])
                if cond == "swarm":
                    P = np.array([p for e in lg for p in e[8]])
                    ax.plot(P[:, 0], P[:, 1], ",", color=COL[cond], alpha=0.25)
                xy = np.array([[e[1], e[2]] for e in lg])
                ax.plot(xy[:, 0], xy[:, 1], "-", color=COL[cond], lw=1)
                hat = np.array([e[9][gi] for e in lg])
                ax.plot(hat[:, 0], hat[:, 1], ".", color="crimson", ms=1)
                ax.set_title(f"{LABEL[cond]}: model ({lg[-1][0] - lg[0][0]:.0f} s)", fontsize=8)
                k2 += 1
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _draw_city(ax, polys, c, half=95.0):
    for p in polys:
        q = p.mean(0)
        if abs(q[0] - c[0]) < half + 30 and abs(q[1] - c[1]) < half + 30:
            ax.fill(p[:, 0], p[:, 1], color="0.85", lw=0)
    ax.plot(c[0], c[1], "k+", ms=6)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")


def fig_views(name, out, variant="pooled"):
    plt = _plt()
    _, G, _ = V.frames(V.read(name))
    G = G[G.variant == variant]
    bases = {"swarm": [("cf_real", "as flown"), ("cf_centre", "feeds moved to the centroid"),
                       ("cf_sphere_c", "a 360 camera at the centroid"), ("cf_screen", "every feed, any head")],
             "sphere": [("cf_real", "as flown"), ("cf_fwd", "forward camera only"),
                        ("cf_screen", "every bearing, any head")],
             "single": [("cf_real", "as flown"), ("cf_sphere_s", "a 360 camera instead")]}
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y, ticks, labs = 0, [], []
    for cond in V.CONDITIONS:
        g = G[G.cond == cond]
        for col, lab in bases[cond]:
            if col not in g:
                continue
            v = g[col].dropna()
            m = v.mean()
            se = v.std() / math.sqrt(max(len(v), 1))
            ax.barh(y, m, xerr=1.96 * se, color=COL[cond], alpha=1.0 if col == "cf_real" else 0.45)
            ticks.append(y)
            labs.append(f"{LABEL[cond]}: {lab}")
            y += 1
        y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("hat recognitions per minute at the goal (the integrated hazard over the at-goal time)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "hazards", "times", "views", "tracks"])
    p.add_argument("--name", default="pilot")
    p.add_argument("--tracks", default=None, help="a JSON list of fly(..., keep_log=True) results")
    a = p.parse_args()
    out = V.OUT / a.name / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("all", "hazards"):
        fig_hazard(out / "recognition.png", "recognition")
        fig_hazard(out / "noticing.png", "noticing")
    if a.cmd in ("all", "times"):
        fig_times(a.name, out / "times.png")
    if a.cmd in ("all", "views"):
        fig_views(a.name, out / "views.png")
    if a.cmd in ("all", "tracks") and a.tracks:
        fig_tracks(json.load(open(a.tracks, encoding="utf-8")), out / "tracks.png")


if __name__ == "__main__":
    main()
