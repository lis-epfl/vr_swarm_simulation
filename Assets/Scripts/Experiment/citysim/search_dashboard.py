"""The search study's (S2 Hidden Goal Search's) results dashboard: the data files it reads, cut from the study's tables.

    python -B -m citysim.search_dashboard   # -> experiment_results/search_study/search/dashboard/

The dashboard is a private claude.ai artifact of the Dashboard type (its link is in docs/studies/README.md), filled
through its own store: the four CSVs written here are uploaded as its sources and citysim/search_dashboard.html is
its page (the store's files/index.html). It is the light rebuild of 2026-10-21, after the first dashboard was lost:
the search-time ratios with their intervals, every condition's searches, and the two checks; not the first one's
city maps, search replays or pilots' target chains. Aggregates only, no run names. Labels say "condition", never
"arm", as the user asked of anything they read.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from citysim import REPO  # noqa: E402

SEARCH_DIR = REPO / "experiment_results" / "search_study"
OUT = SEARCH_DIR / "search" / "dashboard"

CITIES = {"C0": "C0 open", "C1p": "C1p porous", "C2p": "C2p porous", "C1": "C1 packed", "C2": "C2 packed"}
PILOTS = {"rw": "Random walker", "fit": "Fitted to internal_3's pilots", "rw60": "Random walker, 60 m runs",
          "rw240": "Random walker, 240 m runs"}
CONDITIONS = {"single": "Single drone", "single_scan": "Single drone, looking round",
              "swarm": "Swarm", "swarm_steer": "Swarm, steered like one drone"}
COMPARISONS = {"swarm / single": ("swarm_single", "Swarm ÷ single drone"),
               "cameras: swarm_steer / single": ("cameras", "Cameras: steered swarm ÷ single drone"),
               "morphing: swarm / swarm_steer": ("morphing", "Morphing: swarm ÷ steered swarm"),
               "scan / single": ("scan", "Looking round ÷ single drone"),
               "swarm_steer / single": ("cameras", "Cameras: steered swarm ÷ single drone"),
               "swarm / swarm_steer": ("morphing", "Morphing: swarm ÷ steered swarm")}
BASES = {"any": "Any screen", "head_ahead": "Screens straight ahead", "head_fit": "Where internal_3's pilots looked"}


def _key(*parts):
    return "|".join(str(p) for p in parts)


def contrasts():
    c = pd.read_csv(SEARCH_DIR / "search" / "contrasts.csv")
    c["comparison"] = c.contrast.map(lambda s: COMPARISONS[s][0])
    c["comparison_label"] = c.contrast.map(lambda s: COMPARISONS[s][1])
    out = pd.DataFrame(dict(
        key=[_key(*r) for r in zip(c.pilot, c.clutter, c.comparison, c.d_ref, c.basis, c.r.astype(int))],
        pilot=c.pilot, city=c.clutter, city_label=c.clutter.map(CITIES), city_order=c.clutter.map(list(CITIES).index),
        comparison=c.comparison, comparison_label=c.comparison_label, spread=c.d_ref, basis=c.basis,
        range_m=c.r.astype(int), searches=c.n, goals=c.clusters, ratio=c.ratio.round(4), lo=c.lo.round(4),
        hi=c.hi.round(4)))
    return out.sort_values(["pilot", "comparison", "basis", "range_m", "city_order", "spread"])


def conditions():
    c = pd.read_csv(SEARCH_DIR / "search" / "cells.csv")
    rows = []
    for _, r in c.iterrows():
        for basis in BASES:
            for rng in (80, 200):
                median = r[f"median_{basis}_r{rng}_s"]
                # Rounded as the page shows them, so a value on the page is the value in the file. A single drone
                # has no spread; a median is missing where fewer than half the searches found the goal.
                rows.append(dict(
                    key=_key(r.pilot, r.clutter, r.arm, r.d_ref, basis, rng), pilot=r.pilot, city=r.clutter,
                    city_label=CITIES[r.clutter], city_order=list(CITIES).index(r.clutter), condition=r.arm,
                    condition_label=CONDITIONS[r.arm], condition_order=list(CONDITIONS).index(r.arm),
                    spread=None if r.arm.startswith("single") else r.d_ref, basis=basis, range_m=rng,
                    searches=int(r.n), found_pct=round(100 * r[f"found_{basis}_r{rng}"], 1),
                    median_min=round(median / 60, 1) if pd.notna(median) else None,
                    mean_min=round(r[f"rmst_{basis}_r{rng}_s"] / 60, 1),
                    lost_per_search=round(r.lost_mean, 2), split_pct=round(100 * r.split_frac, 1),
                    speed_ms=round(r.speed_ms, 2)))
    return pd.DataFrame(rows).sort_values(["pilot", "basis", "range_m", "city_order", "condition_order", "spread"])


def unity_check():
    b = pd.read_csv(SEARCH_DIR / "bench" / "spot" / "bench_summary.csv")
    b = b[b.kind == "contrast"]
    w = b.pivot_table(index=["arm", "basis", "r"], columns="side", values="ratio").reset_index()
    return pd.DataFrame(dict(
        key=[_key(COMPARISONS[a][0], s, int(r)) for a, s, r in zip(w.arm, w.basis, w.r)],
        comparison=w.arm.map(lambda s: COMPARISONS[s][0]), comparison_label=w.arm.map(lambda s: COMPARISONS[s][1]),
        basis=w.basis, range_m=w.r.astype(int), replica=w.replica.round(3), unity=w.unity.round(3)))


def pilot_check():
    k = pd.read_csv(SEARCH_DIR / "check" / "internal_3" / "check_summary.csv")
    return pd.DataFrame(dict(
        key=k.arm, condition_label=k.arm.map(CONDITIONS), legs=k.legs, sims=k.sims,
        human_speed_ms=k.human_speed_ms.round(2), model_speed_ms=k.sim_speed_ms.round(2),
        human_turns_per_km=k.human_turns_per_km.round(1), model_turns_per_km=k.sim_turns_per_km.round(1),
        human_first_sight_s=k.human_disc_median_s.round(1), model_first_sight_s=k.sim_disc_median_s.round(1)))


def build(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("contrasts", contrasts()), ("conditions", conditions()), ("unity_check", unity_check()),
                     ("pilot_check", pilot_check())):
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"[search] {out / name}.csv ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
