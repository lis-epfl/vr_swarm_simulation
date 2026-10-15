"""The trial levels, by the plan's pre-specified rule (section 8), from the tiers' tables.

    python -B -m citysim.select_levels --sweep tierB2 [--tierC <tag>]       # -> experiment_results/density_study/levels.json

  L1  today's regime: the level named by --l1 (C0-G0: today's coverage, today's goal patch). A control that the
      effect is geometry-driven.
  L2  the level with the largest swarm - single advantage (Tier B, E_head) among those that pass G1 against L1
      (>= 2x L1's advantage, > 3 SD), are envelopment-feasible at a spread pilots use (Tier B at d_ref 1.0 and the
      Tier B' flights' arrivals, each >= ENVELOP_MIN), keep U (useful feeds at the goal) >= 0.7 of L1's, and whose
      layouts are feasible (not jammed, <= 56 buildings within 50 m).
  L3  among the densest clutter level's goal levels: those whose advantage vanished or reversed (G1's L3 test),
      whose envelopment is predicted to fail (enveloped <= ENVELOP_FAIL in Tier B and B') or whose useful feeds fall
      to <= 0.6 of L1's, that pass G2 (Tier B' transit at d_ref 1.0: mean nearest-neighbour spacing >= 3 m, the
      flown spacing internal_3's losses came at) and G3 (Tier C's loss rate at or below --loss-max, when there is a
      Tier C table; otherwise G3 is reported as pending). Of those, the one with the fewest predicted losses, then
      the smallest advantage.
Every level's geometry and its predictions are written out with the margins, so the choice can be checked.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import STUDY_DIR  # noqa: E402
from citysim.checks import GATE_RATIO, GATE_SD, VANISH_SD, level_deltas  # noqa: E402

ENVELOP_MIN, ENVELOP_FAIL = 0.9, 0.1
U_KEEP, U_DROP = 0.7, 0.6
NN_MIN = 3.0
TRANSIT_COLS = ["path_ratio", "street_frac", "adj_per_100m", "cmd_straightness", "morph_per_km", "sweep_any_r40_m",
                "sweep_head_r40_m", "area_any_r40_per_m", "behind_corner_r40_frac", "progress_ms"]


def collect(sweep, tierc=None, routing="pilot"):
    src = STUDY_DIR / "tierB" / sweep
    exp = pd.read_csv(src / "exposure.csv")
    geo = pd.read_csv(src / "geometry.csv")
    env = pd.read_csv(src / "envelop.csv")
    lays = pd.read_csv(src / "layouts.csv")
    spec = json.loads((src / "spec.json").read_text(encoding="utf-8"))
    per_seed = level_deltas(exp, "pilot")
    lev = per_seed.groupby(["level", "clutter", "goal_level"]).agg(
        delta=("delta", "mean"), sd=("delta", "std"), swarm_E_head=("swarm", "mean"),
        single_E_head=("single", "mean")).reset_index()
    sw_any = exp[exp.interface == "swarm"].groupby(["level", "seed", "goal", "altitude", "pitch"]).E_any.max()
    lev = lev.merge(sw_any.groupby("level").mean().rename("swarm_E_any").reset_index(), on="level", how="left")
    sg = exp[(exp.interface == "single") & exp.policy.isin(["orbit_in", "orbit_out", "scan"])]
    sg_any = sg.groupby(["level", "seed", "goal", "altitude", "pitch", "policy"]).E_any.mean()
    sg_any = sg_any.groupby(level=["level", "seed", "goal", "altitude", "pitch"]).max()
    lev = lev.merge(sg_any.groupby("level").mean().rename("single_E_any").reset_index(), on="level", how="left")
    fs = exp[(exp.interface == "swarm") & np.isclose(exp.d_ref, 1.0)].groupby("level").first_sight_s.mean()
    lev = lev.merge(fs.rename("swarm_first_sight_s").reset_index(), on="level", how="left")
    u = geo[np.isclose(geo.d_ref, 1.0)].groupby("level")[["useful_feeds", "E_any", "E_env", "E_par", "V_swarm_0",
                                                             "V_single_0"]].mean()
    u = u.rename(columns={"E_any": "escape_E_any", "E_env": "escape_E_env", "E_par": "escape_E_par"})
    lev = lev.merge(u.reset_index(), on="level", how="left")
    e10 = env[np.isclose(env.d_ref, 1.0)].groupby("level").enveloped.mean().rename("enveloped_B_1.0")
    lev = lev.merge(e10.reset_index(), on="level", how="left")
    feas = lays.groupby("level").agg(jammed=("jammed", "any"), max_within_50m=("max_within_50m", "max"),
                                     alpha=("alpha_achieved", "mean"), ell_m=("ell_mean", "mean")).reset_index()
    lev = lev.merge(feas, on="level", how="left")
    dyn_path = STUDY_DIR / "tierBprime" / sweep / "dynamic.csv"
    if dyn_path.exists():
        dyn = pd.read_csv(dyn_path)
        if "routing" in dyn:              # since 2026-10-13: the swarm on its pilot paths (the street route is the comparison)
            dyn = dyn[(dyn.routing == routing) & (dyn.interface == "swarm")]
        d1 = dyn[np.isclose(dyn.transit_dref, 1.0)].groupby("level").agg(
            lost_per_flight_B1=("drones_lost", "mean"), nn_transit_m=("nn_transit_m", "mean"),
            progress_ms=("progress_ms", "mean"), useful_transit=("useful_transit", "mean"),
            useful_goal_dyn=("useful_goal", "mean"), enveloped_dyn=("enveloped_frac", "mean"),
            contacts_per_km=("contacts_per_km", "mean"))
        d4 = dyn[np.isclose(dyn.transit_dref, 0.4)].groupby("level").agg(
            lost_per_flight_B04=("drones_lost", "mean"), nn_transit_04_m=("nn_transit_m", "mean"))
        lev = lev.merge(d1.reset_index(), on="level", how="left").merge(d4.reset_index(), on="level", how="left")
    tr_path = STUDY_DIR / "tierBprime" / sweep / "transit.csv"
    if tr_path.exists():                 # the transit predictions (citysim.transit), reported beside the rule's inputs
        tr = pd.read_csv(tr_path)
        tr = tr[(tr.routing == routing) & ((tr.interface != "swarm") | np.isclose(tr.transit_dref, 1.0))]
        cols = [c for c in TRANSIT_COLS if c in tr]
        disc = [c for c in tr.columns if c.startswith("disc_g") and c.endswith("_any_r80_s")]
        if disc:
            tr = tr.assign(discovery_r80_s=tr[disc].mean(axis=1))
            cols.append("discovery_r80_s")
        piv = tr.groupby(["level", "interface"])[cols].mean().unstack("interface")
        piv.columns = [f"{iface}_{c}" for c, iface in piv.columns]
        lev = lev.merge(piv.reset_index(), on="level", how="left")
    if tierc:
        tc = pd.read_csv(STUDY_DIR / "tierC" / tierc / "tierC_levels.csv")
        lev = lev.merge(tc, on="level", how="left", suffixes=("", "_C"))     # a clash keeps B''s name, C's gets _C
        alive = STUDY_DIR / "tierC" / tierc / "tierB_at_alive.csv"
        if alive.exists():         # Tier B's advantage with Tier C's surviving drones (reported, not a gate)
            al = pd.read_csv(alive)[["level", "swarm_E_head_alive", "delta_alive", "delta_alive_sd"]]
            lev = lev.merge(al, on="level", how="left")
    return lev, spec


def select(lev, spec, l1, loss_max):
    L1 = lev[lev.level == l1]
    if not len(L1):
        raise SystemExit(f"[levels] no level {l1}")
    L1 = L1.iloc[0]
    d1, u1 = float(L1.delta), float(L1.useful_feeds)
    lev = lev.copy()
    lev["ratio_to_L1"] = lev.delta / d1
    lev["feasible"] = ~lev.jammed.astype(bool) & (lev.max_within_50m <= 56)
    env_dyn = lev["enveloped_dyn"] if "enveloped_dyn" in lev else pd.Series(np.nan, index=lev.index)
    lev["envelop_ok"] = (lev["enveloped_B_1.0"] >= ENVELOP_MIN) & (env_dyn.isna() | (env_dyn >= ENVELOP_MIN))
    lev["envelop_fails"] = (lev["enveloped_B_1.0"] <= ENVELOP_FAIL) & (env_dyn.isna() | (env_dyn <= ENVELOP_FAIL))
    lev["U_ratio"] = lev.useful_feeds / u1
    lev["G1_L2"] = (lev.ratio_to_L1 >= GATE_RATIO) & (lev.delta > GATE_SD * lev.sd)
    lev["G1_L3"] = lev.delta <= VANISH_SD * lev.sd.clip(lower=1e-6)
    lev["G2"] = (lev["nn_transit_m"] >= NN_MIN) if "nn_transit_m" in lev else np.nan
    if "loss_rate" in lev:
        lev["G3"] = lev.loss_rate <= loss_max
    else:
        lev["G3"] = np.nan
    c2 = lev[lev.level != l1]
    l2 = c2[c2.G1_L2 & c2.envelop_ok & (c2.U_ratio >= U_KEEP) & c2.feasible].sort_values("delta", ascending=False)
    densest = lev.loc[lev.alpha.idxmax(), "clutter"]
    c3 = lev[(lev.clutter == densest) & lev.feasible & lev.G1_L3 & (lev.envelop_fails | (lev.U_ratio <= U_DROP))]
    c3 = c3[c3.G2.fillna(False).astype(bool)]
    g3_pending = lev.G3.isna().all()
    if not g3_pending:
        c3 = c3[c3.G3.fillna(False).astype(bool)]
    sort3 = [c for c in ("loss_rate", "lost_per_flight_B1") if c in c3]
    c3 = c3.sort_values(sort3 + ["delta"])
    pick = dict(L1=l1, L2=l2.level.iloc[0] if len(l2) else None, L3=c3.level.iloc[0] if len(c3) else None)
    return lev, pick, dict(L2_candidates=l2.level.tolist(), L3_candidates=c3.level.tolist(), densest=densest,
                           G3_pending=bool(g3_pending))


def level_params(spec, level):
    c, g = level.split("-")
    return dict(spec.get("common", {}), **spec["clutter"][c], **spec["goal"][g])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweep", default="tierB2")
    ap.add_argument("--l1", default="C0-G0")
    ap.add_argument("--tierC", help="a Tier C tag (experiment_results/density_study/tierC/<tag>/tierC_levels.csv)")
    ap.add_argument("--loss-max", type=float, default=2.5,
                    help="G3: drones lost per run allowed at L3 (internal_3's rate was ~2.5 per run)")
    ap.add_argument("--routing", choices=["pilot", "street"], default="pilot",
                    help="whose Tier B' transits G2 reads: the pilot paths (since 2026-10-13) or the street route")
    ap.add_argument("--out", default="levels", help="output stem under experiment_results/density_study")
    a = ap.parse_args()
    lev, spec = collect(a.sweep, a.tierC, a.routing)
    lev, pick, info = select(lev, spec, a.l1, a.loss_max)
    pd.set_option("display.width", 260)
    cols = ["level", "alpha", "ell_m", "delta", "sd", "ratio_to_L1", "swarm_E_head", "single_E_head",
            "swarm_E_any", "single_E_any", "useful_feeds", "U_ratio", "enveloped_B_1.0", "enveloped_dyn",
            "nn_transit_m", "lost_per_flight_B1", "progress_ms", "G1_L2", "G1_L3", "G2", "G3", "feasible"]
    print(lev[[c for c in cols if c in lev]].sort_values("delta", ascending=False).round(3).to_string(index=False))
    print(f"\nL1 {pick['L1']}   L2 {pick['L2']}   L3 {pick['L3']}   (L2 candidates {info['L2_candidates']}; "
          f"L3 candidates {info['L3_candidates']}; densest clutter {info['densest']}"
          f"{'; G3 pending Tier C' if info['G3_pending'] else ''})")
    out = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), sweep=a.sweep, routing=a.routing,
               tierC=a.tierC, rule=__doc__,
               pick=pick, info=info, levels={})
    for k in ("L1", "L2", "L3"):
        lvl = pick[k]
        if lvl is None:
            continue
        row = lev[lev.level == lvl].iloc[0]
        out["levels"][k] = dict(level=lvl, params=level_params(spec, lvl),
                                predicted={c: (None if (isinstance(v, float) and math.isnan(v)) else
                                               (bool(v) if isinstance(v, (bool, np.bool_)) else
                                                (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                                                 else str(v))))
                                           for c, v in row.items() if c not in ("level", "clutter", "goal_level")})
    path = STUDY_DIR / f"{a.out}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    lev.to_csv(STUDY_DIR / f"{a.out}_table.csv", index=False)
    print(f"written {path}")


if __name__ == "__main__":
    main()
