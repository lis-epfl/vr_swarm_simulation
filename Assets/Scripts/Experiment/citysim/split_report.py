"""S4 Split and Surround: the results page, citysim/split_report.html filled with the runs' data.

    python -B -m citysim.split_report phase1 [--out PATH]
        -> experiment_results/split_study/phase1/split_phase1.html (to open as a file, the replays inlined) and
           split_phase1_bare.html (without the html/head/body skeleton: what the artifact is published from), with
           split_phase1_files.json, the replay files to publish beside it ({published path: source})

The page: the Cities tab (every design's city and the flights the page replays: citysim.split_replays' data, played
by citysim/replay_view.js, which is inlined), the surround map d*(p, s) against its prediction, the surround-and-exit
(M3b) by mode and length, the pass round (M3), safety by variant, the criteria and the variant decision, Phase 2's
walks, search and screens, the method and the terms. Runs that do not exist yet are left out, so the page can be
rebuilt as they arrive. conclusions.html beside the runs (phase1/), if present, is the masthead's findings box.
Imported only by the parent: safe to revise while a run goes.
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, split_probe as S, split_tables as T  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "split_report.html"
VIEW_JS = Path(__file__).resolve().parent / "replay_view.js"
REPLAY = S.OUT_ROOT.parent / "replay"                    # citysim.split_replays' output
RINDEX = REPLAY / "index.json"
COUPLING = "p1_coupling"
STEPS = "p1_steps"
# (key, run, label, what changed from the soft screen)
SCREENS = [
    ("soft", "p1_scr_soft", "S1's hold", "a centring hold on the block's centre at stick 0.5, gain 0.25 (S1's), at d*"),
    ("firm", "p1_scr_firm", "firm hold", "the hold at stick 1.0, gain 0.5"),
    ("aim", "p1_scr_aim", "aim and turn", "full stick at the opposite exit for the surround, then off down the side "
                                         "exit"),
    ("perim", "p1_scr_perim", "per-block spread", "each block's spread its own ring perimeter / 100 (k 1)"),
    ("perim125", "p1_scr_perim125", "per-block x 1.25", "the per-block spread x 1.25"),
    ("d10", "p1_scr_d10", "spread 1.0", "spread 1.0 at every block (no coupling)"),
]
FOLLOWUPS = [
    ("r20", "p1_fu_r20", "r0_coh 20", "cohesion's cutoff back at the scene's 20 (repulsive beyond ~75 m)"),
    ("nocore", "p1_fu_nocore", "core off", "the hollow core off"),
    ("k0", "p1_fu_k0", "B: kappa 0", "variant B's face bend off (a plain closest point)"),
    ("k1", "p1_fu_k1", "B: kappa 1", "variant B's face bend doubled"),
]
STEP4_T = (6, 8, 10, 12)
STEP4_MODES = (("firm", "firm hold"), ("aim", "aim and turn"))
EXIT = [("exit05", "p1_t4_firm_t8_exit05", 0.5), ("exit075", "p1_t4_firm_t8_exit075", 0.75)]
CONFIRM = "p1_confirm"
# The sight range the page's Phase 3 and its answer use: how far a feed counts as seeing, for looking down a street and
# for finding a target. Phases 1-3 were flown at 80 m; the user judged that too short (2026-10-28) and Phase 3 and the
# informed search were re-run at 250 m (runs suffixed _r250). Phase 2's own tabs stay its pre-registered 80 m record.
SIGHT_M = 250.0


def sight_run(base, sight=SIGHT_M):
    """A run's name at a sight range: the 80 m runs carry none, the others a _r<metres> suffix."""
    return base if sight == 80.0 else f"{base}_r{sight:g}"


LOOKDOWN_A, LOOKDOWN_B = sight_run("lookdown_A"), sight_run("lookdown_B")    # Phase 3, the look-down pilot
LOOKDOWN_A1 = sight_run("lookdown_A1")                   # the lone drone over Part A's trials
P2_SEARCH_SIGHT = sight_run("split_search")              # Phase 2's informed search at the page's sight range
ANSWER_D = 1.8                                           # the widest block spread safe at p 40-60 (Part A)
ANSWER_SEARCH = ("swarm_street", "swarm_pass", "swarm_hop", "swarm_hold")
CURVE_T = [float(x) for x in np.arange(0.0, 40.01, 1.0)]


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _read(name):
    path = S.OUT_ROOT / name / "results.jsonl"
    if not path.exists():
        return None
    return T._ok(S.read(name))


def summarise(df):
    """Per variant x design: what the page shows of one run (every manoeuvre it flew)."""
    out = []
    for (v, p, s), g in df.groupby(["variant", "p", "s"]):
        r = dict(v=v, p=float(p), s=float(s))
        m3, m3b, m1, m6 = (g[g.kind == k] for k in ("M3", "M3b", "M1", "M6"))
        if len(m3):
            com = m3[m3.t_commit.notna()]
            r.update(m3_n=len(m3), m3_cont=_f(m3.contacts.mean()), m3_lost=_f(m3.lost.mean()),
                     m3_anyloss_ub=_f(T.wilson_upper(int((m3.lost > 0).sum()), len(m3))),
                     m3_ref=_f(com.reform_ok.fillna(False).astype(bool).mean()) if len(com) else None,
                     m3_pass=_f(m3.pass_s.median(), 1), m3_prev=_f(com.preview_search.mean()) if len(com) else None,
                     m3_pass_ok=_f((m3.reached_exit.fillna(False).astype(bool) & (m3.stall_s.fillna(0) <= 3.0)).mean()),
                     m3_d=_f(m3.d_block.median(), 2))
        if len(m3b):
            com = m3b[m3b.t_commit.notna()]
            t80 = T.t80_per_trial(m3b)
            tp = m3b.time_price.dropna() if "time_price" in m3b else []
            r.update(m3b_n=len(m3b), m3b_cont=_f(m3b.contacts.mean()), m3b_lost=_f(m3b.lost.mean()),
                     m3b_anyloss_ub=_f(T.wilson_upper(int((m3b.lost > 0).sum()), len(m3b))),
                     m3b_ref=_f(com.reform_ok.fillna(False).astype(bool).mean()) if len(com) else None,
                     m3b_prev_ok=_f((com.preview_search >= T.BARS["preview_streets"]).mean()) if len(com) else None,
                     m3b_prev=_f(com.preview_search.mean()) if len(com) else None,
                     m3b_price=_f(np.median(tp)) if len(tp) else None, m3b_hop=_f(m3b.hop_s.median(), 1),
                     m3b_street=_f(m3b.street_s.median(), 1) if "street_s" in m3b else None,
                     m3b_t_hop=_f(m3b.t_hop.median(), 1), m3b_d=_f(m3b.d_block.median(), 2),
                     curve=[_f((t80 <= t).mean()) for t in CURVE_T] if len(t80) else None,
                     t_prev=_f(np.sort(t80)[max(1, int(math.ceil(0.8 * len(t80)))) - 1], 1) if len(t80) else None,
                     t80_med=_f(np.median(t80), 1) if len(t80) else None)
        if len(m1):
            ratio = (m1.swarm_speed / m1.single_speed).dropna()
            r.update(m1_ratio=_f(ratio.median()) if len(ratio) else None,
                     m1_cont100=_f((m1.swarm_contacts / m1.route_m * 100).mean(), 4),
                     m1_wall=_f(m1.wall_feed_frac.mean()))
        if len(m6):
            r.update(m6_passed=_f(m6.passed.astype(bool).mean()), m6_stall=_f(m6.stall_s.median(), 1),
                     m6_cont=_f(m6.contacts.mean()))
        out.append(r)
    return out


def coupling_data():
    df = _read(COUPLING)
    if df is None:
        return None
    cells, cmap = T.coupling(df)
    out = []
    for (v, p, s), c in cells.groupby(["variant", "p", "s"]):
        m = cmap[(cmap.variant == v) & (cmap.p == p) & (cmap.s == s)].iloc[0]
        c = c.sort_values("d_block")
        out.append(dict(v=v, p=float(p), s=float(s), d=[_f(x, 2) for x in c.d_block],
                        env=[_f(x) for x in c.enveloped], cont=[_f(x) for x in c.contacts_per],
                        lost=[_f(x) for x in c.lost_per], t_env=[_f(x, 1) for x in c.t_envelop_median],
                        d_star=_f(m.d_star, 2), rule=m.d_star_rule, pred=_f(m.d_star_pred, 2),
                        t_env_star=_f(m.t_envelop_at_d_star, 1)))
    return dict(cells=out, trials=int(cells.trials.median()), holds=int(cells.trials.sum()))


def run_block(entries):
    out = []
    for key, name, label, what in entries:
        df = _read(name)
        if df is None:
            continue
        out.append(dict(key=key, run=name, label=label, what=what, n_jobs=int(len(df)), rows=summarise(df)))
    return out


def step4_data():
    out = []
    for mode, label in STEP4_MODES:
        for t in STEP4_T:
            df = _read(f"p1_t4_{mode}_t{t}")
            if df is None:
                continue
            out.append(dict(mode=mode, label=label, t_hop=t, rows=summarise(df)))
    return out


def exit_data():
    out = []
    for key, name, stick in EXIT:
        df = _read(name)
        if df is None:
            continue
        out.append(dict(key=key, run=name, exit_stick=stick, rows=summarise(df)))
    return out


def confirm_data():
    df = _read(CONFIRM)
    if df is None:
        return None
    crit = T.criteria(df)
    dec = T.decide(crit)
    rows = []
    for _, c in crit.iterrows():
        rows.append({k: (bool(v) if isinstance(v, (bool, np.bool_)) else (_f(v, 4) if isinstance(v, (int, float, np.floating,
                                                                                                    np.integer)) else v))
                     for k, v in c.items()})
    settings = {}
    for v, g in df.groupby("variant"):                  # each variant flies its own profile
        st = g.iloc[0].setting
        st = json.loads(st) if isinstance(st, str) else st
        settings[v] = {k: x for k, x in dict(S.DEFAULT_SETTING, **st).items()
                       if k != "d_block" and x != S.DEFAULT_SETTING.get(k)}
    return dict(run=CONFIRM, rows=rows, decision=dec, n_jobs=int(len(df)), summary=summarise(df), settings=settings,
                trials=int(df.groupby(["variant", "p", "s", "kind"]).size().median()),
                city_seeds=sorted(int(x) for x in df.city_seed.unique()))


def gate_data():
    """The Unity bench check (split_bench gate: the S4 quantities on Unity's trajectories beside replica twins flown
    from Unity's recorded starts; swarm_replica.py validate: track, spacing, losses, contacts)."""
    import pandas as pd
    base = S.OUT_ROOT / "p1_bench"
    out = {}
    g = base / "bench" / "gate.csv"
    if g.exists():
        df = pd.read_csv(g)
        out["gate"] = [{k: (_f(v, 3) if isinstance(v, (int, float, np.floating, np.integer)) and k not in ("n",) else
                            (int(v) if k == "n" else v)) for k, v in r.items()} for r in df.to_dict("records")]
    v = sorted((base / "validate").glob("validate_verdicts_*.csv")) if (base / "validate").exists() else []
    if v:
        dv = pd.read_csv(v[-1])
        out["validate"] = [dict(run=r["run"].replace("s4_p1_bench_", ""), flights=int(r["flights"]),
                                lost_bench=int(r["lost_bench"]), lost_replica=int(r["lost_replica"]),
                                contacts_bench=int(r["contacts_bench"]), contacts_replica=int(r["contacts_replica"]),
                                track=_f(r["track_frac_median"], 4), nn=_f(r["nn_rel_median"], 4), verdict=r["verdict"])
                           for _, r in dv.iterrows()]
    return out or None


P2_WALKS, P2_SEARCH, P2_BENCH, P2_ROUTES = "split", "split_search", "p2_bench", "p2_routes"
P2_COND_COLS = ("hops_done", "preview_common", "preview_common_transit", "street_km_per_min", "find_rmst_any_r40",
                "find_rmst_any_r80", "find_rmst_head_r80", "find_rmst_any_r200", "lost", "contacts",
                "reform_ok_share", "split_share", "feeds_mean", "corr_a1_linked", "corr_a2_linked", "corr_islands",
                "corr_orphan_share", "corr_largest_island", "corr_a4_reach", "corr_b1_deg", "corr_b2_err_deg",
                "corr_b2_order_rev", "corr_c1_linked_share")
P2_SWARM = ("swarm_street", "swarm_hold", "swarm_pass", "swarm_hop")
# the chain walks' headline ratios, for the full-length sensitivity (walks every condition flew for the whole 600 s)
P2_KEY_CONTRASTS = [("swarm_hop", "swarm_street"), ("swarm_hop", "swarm_hold"), ("swarm_street", "single"),
                    ("swarm_pass", "swarm_street"), ("swarm_hop", "single_look"), ("single_look", "single")]


def _city_means(df, by, cols):
    """Means over cities of the per-city means (cities are the units)."""
    cols = [c for c in cols if c in df]
    keys = list(dict.fromkeys(by + ["p", "s", "city_seed"]))
    per = df.groupby(keys)[cols].mean().reset_index()
    return per.groupby(by)[cols].mean().reset_index(), int(per[["p", "s", "city_seed"]].drop_duplicates().shape[0])


def _pooled(cm, log=True):
    """A city-mean paired quantity pooled over every city: (estimate, lo, hi, n), a ratio when log."""
    x = cm.d.values[np.isfinite(cm.d.values)] if len(cm) else np.array([])
    if len(x) < 2:
        return None
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    f = math.exp if log else (lambda v: v)
    return dict(est=f(m), lo=f(m - 1.96 * se), hi=f(m + 1.96 * se), n_cities=int(len(x)))


def _spread_check(fl):
    """d* against spread 1.0 on the design where they differ (p 40 s 24, d* 0.8), every swarm condition that sets its
    spread at the block: the preview difference and the time-to-find ratio, 1.0 against d*, city means."""
    from citysim import split_stats as SS
    out = []
    for cond in P2_SWARM:
        for meas, log in (("preview_common", False), ("find_rmst_any_r80", True), ("lost", False),
                          ("contacts", False)):
            cm = SS.paired(fl, cond, cond, meas, spread_a="1.0", spread_b=SS.PROFILE, log=log)
            for (p, s), g in cm.groupby(["p", "s"]):
                r = _pooled(g, log)
                if r is not None and not (abs(r["est"] - (1.0 if log else 0.0)) < 1e-12 and r["lo"] == r["hi"]):
                    out.append(dict(condition=cond, measure=meas, p=float(p), s=float(s), **r))
    return out


def _full_length(fl):
    """The chain walks' headline time-to-find ratios on every walk, and on the walks every condition flew for the
    whole T_s only (a chain stopped by the arena's edge ends early, and an ended flight sees nothing more)."""
    from citysim import split_stats as SS
    prof = fl[fl.spread.astype(str) == SS.PROFILE]
    keys = ["p", "s", "city_seed", "walk"]
    t_min = prof.groupby(keys).t_end.min()
    T = float(prof.t_end.max())
    full = t_min[t_min >= T - 1.0].index
    sub = prof.set_index(keys).loc[full].reset_index()
    rows = []
    for a, b in P2_KEY_CONTRASTS:
        r_all, r_full = (_pooled(SS.paired(d, a, b, "find_rmst_any_r80")) for d in (prof, sub))
        if r_all and r_full:
            rows.append(dict(a=a, b=b, all=r_all["est"], full=r_full["est"]))
    return dict(share_full=float(len(full) / max(len(t_min), 1)), walks=int(len(t_min)), rows=rows)


def _bench_walk_contrasts(path):
    """The hypotheses' chain-walk contrasts on the Unity bench's own walks, replica twins beside them (per walk, then
    per city, as the replica's tables)."""
    import pandas as pd
    df = pd.read_csv(path)
    keys = ["p", "s", "seed", "walk"]
    out = []
    for a, b, meas, log in (("swarm_hop", "swarm_hold", "preview", False), ("swarm_hop", "swarm_street", "find80", True),
                            ("swarm_hop", "swarm_hold", "find80", True), ("swarm_street", "single", "find80", True),
                            ("swarm_pass", "swarm_street", "preview", False)):
        row = dict(a=a, b=b, measure=meas)
        for side in ("bench", "replica"):
            A = df[df.condition == a].set_index(keys)[f"{side}_{meas}"]
            B = df[df.condition == b].set_index(keys)[f"{side}_{meas}"]
            d = (np.log(A / B) if log else A - B).dropna()
            cm = d.groupby(["p", "s", "seed"]).mean().rename("d").reset_index()
            r = _pooled(cm, log)
            row[side] = r
        out.append(row)
    return out


def phase2_data():
    """Phase 2's results: the primaries, every condition's measures, the contrasts, the informed search, the bench."""
    import pandas as pd
    from citysim import split_stats as SS
    root = S.OUT_ROOT.parent
    w = root / P2_WALKS
    if not (w / "results.jsonl").exists():
        return None
    out = {}
    fl, _ = SS.read(w)
    if len(fl):
        spec = json.loads((w / "spec.json").read_text(encoding="utf-8"))
        prof = fl[fl.spread.astype(str) == "profile"]
        h1, h1_per = SS.primary_h1(fl)
        cond, n_c = _city_means(prof, ["condition"], P2_COND_COLS)
        by_design, _ = _city_means(prof, ["condition", "p", "s"], P2_COND_COLS)
        ct = SS.contrast_table(fl)
        rc = w / "recheck.csv"
        rech = pd.read_csv(rc) if rc.exists() else None
        out["walks"] = dict(spec={k: spec[k] for k in ("variant", "designs", "cities_per_design", "walks_per_city",
                                                       "T_s", "setting", "d_star") if k in spec},
                            n_walks=int(fl.groupby(["p", "s", "city_seed", "walk"]).ngroups), n_cities=int(n_c),
                            n_flights=int(len(fl)), h1=_clean(h1), h1_per=_records(h1_per), conditions=_records(cond),
                            by_design=_records(by_design), contrasts=_records(ct),
                            trends=_records(SS.trends(fl)), spread=_clean(_spread_check(fl)),
                            full_length=_clean(_full_length(fl)),
                            recheck=None if rech is None else [int(rech.identical.sum()), int(len(rech))])
    s = root / P2_SEARCH
    if (s / "results.jsonl").exists():
        df, caps = SS.read_search(s)
        if len(df):
            h2 = SS.primary_h2(df)
            g = df.groupby(["condition", "basis"])
            summ = g.agg(found_r80=("hit_r80", "mean"), rmst_r80=("t_r80", "mean"), rmst_r40=("t_r40", "mean"),
                         rmst_r200=("t_r200", "mean"), found_r200=("hit_r200", "mean"), lost=("lost", "mean"),
                         contacts=("contacts", "mean"), decisions=("decisions", "mean"),
                         blind=("blind_decisions", "mean"), path_m=("path_m", "mean"),
                         streets=("street_seen_share", "mean")).reset_index()
            rows = []
            for a, b in SS.CONTRASTS:
                for basis in ("any", "head"):
                    cm = SS.search_ratio(df, a, b, basis=basis)
                    if len(cm) > 1:
                        m, se = float(cm.d.mean()), float(cm.d.std(ddof=1) / math.sqrt(len(cm)))
                        rows.append(dict(a=a, b=b, basis=basis, design="pooled", n_cities=int(len(cm)),
                                         ratio=math.exp(m), lo=math.exp(m - 1.96 * se), hi=math.exp(m + 1.96 * se)))
                        for (p, s_), gg in cm.groupby(["p", "s"]):
                            if len(gg) > 1:
                                m, se = float(gg.d.mean()), float(gg.d.std(ddof=1) / math.sqrt(len(gg)))
                                rows.append(dict(a=a, b=b, basis=basis, design=f"p {p:g} · s {s_:g}",
                                                 n_cities=int(len(gg)), ratio=math.exp(m), lo=math.exp(m - 1.96 * se),
                                                 hi=math.exp(m + 1.96 * se)))
            # the share of targets found by t, any feed: every flight runs to the find or the cap, and a swarm that
            # falls under four drones counts as never finding it (the conservative reading), so with nothing censored
            # before the cap the empirical share is the Kaplan-Meier estimate
            ts = [float(t) for t in np.arange(0.0, max(caps) + 1e-6, 30.0)]
            curves = {}
            for c, gg in df[df.basis == "any"].groupby("condition"):
                hit = gg.t_r80.values[gg.hit_r80.values.astype(bool)]
                curves[c] = [_f(float((hit <= t).sum()) / len(gg), 4) for t in ts]
            rcs = s / "recheck.csv"
            rech = pd.read_csv(rcs) if rcs.exists() else None
            out["search"] = dict(h2=_clean(h2), summary=_records(summ), contrasts=_records(pd.DataFrame(rows)),
                                 caps=caps, n_targets=int(df.groupby(["p", "s", "city_seed", "target_i"]).ngroups),
                                 curve_t=ts, curves=curves,
                                 recheck=None if rech is None else [int(rech.identical.sum()), int(len(rech))])
    pj = s / "primaries.json"
    if pj.exists():
        out["primaries"] = _clean(json.loads(pj.read_text(encoding="utf-8")))
    bw = root / P2_BENCH / "bench" / "walk_gate.csv"
    if bw.exists():
        out["bench"] = _records(pd.read_csv(bw))
        bf = bw.with_name("walk_gate_flights.csv")
        if bf.exists():
            fl_b = pd.read_csv(bf)
            out["bench_n"] = dict(walks=int(fl_b.groupby(["p", "s", "seed", "walk"]).ngroups),
                                  cities=int(fl_b.groupby(["p", "s", "seed"]).ngroups), flights=int(len(fl_b)))
            out["bench_contrasts"] = _clean(_bench_walk_contrasts(bf))
    rg = root / P2_ROUTES / "bench" / "route_gate.csv"
    if rg.exists():
        d = pd.read_csv(rg)
        both = d.dropna(subset=["bench_found_r80", "replica_found_r80"])
        out["routes"] = dict(n=int(len(d)), both=int(len(both)),
                             within10=int(((both.bench_found_r80 - both.replica_found_r80).abs() <= 10.0).sum()),
                             one_only=int(d.bench_found_r80.isna().ne(d.replica_found_r80.isna()).sum()),
                             rows=_records(d))
    return out or None


def _tci(x):
    """(mean, lo, hi) with a t interval (95%) over the values given."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else None), None, None
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    q = float(stats.t.ppf(0.975, len(x) - 1))
    return m, m - q * se, m + q * se


def lookdown_data():
    """Phase 3, the look-down pilot: Part A's cells (per design, spread and condition: every level of sight, the time
    to the next junction, its price over the street-bound swarm's and safety), the surround the rule picks per design,
    and Part B's search, when they exist."""
    import pandas as pd
    from citysim import lookdown as LD, lookdown_search as LS
    root = S.OUT_ROOT.parent
    if not (root / LOOKDOWN_A / "results.jsonl").exists():
        return None
    df, by = LD.read(LOOKDOWN_A)
    if not len(df):
        return None
    # looked down on arrival at the corner, where the street-bound swarm and the pass decide (the pass is aimed by
    # its decision, so what it sees going round the block counts for the next one): from the per-exit times
    look_j = []
    for r in by.values():
        for x in r["rows"]:
            v = [e["look"] is not None and e["look"] <= 1e-9 for e in x["exits"] if e["kind"] is not None]
            look_j.append(float(np.mean(v)) if v else math.nan)
    df["look_J"] = look_j
    df["reform_ok"] = df["reform_ok"].astype(float)
    if "enveloped" in df:
        df["enveloped"] = df["enveloped"].astype(float)
    key = ["p", "s", "city_seed", "trial", "d"]
    street = df[df.cond == "street"].set_index(key)
    df = df.join(street[["look_c", "look_F", "hop_s", "ent_c"]].rename(columns=lambda c: c + "_st"), on=key)
    df["d_look"] = df.look_c - df.look_c_st
    df["d_lookF"] = df.look_F - df.look_F_st
    df["price"] = df.hop_s / df.hop_s_st
    cells = []
    for (p, s, d, lab), g in df.groupby(["p", "s", "d", "label"]):
        r0 = g.iloc[0]
        m, lo, hi = _tci(g.look_c)
        dm, dlo, dhi = _tci(g.d_look)
        fm, flo, fhi = _tci(g.d_lookF)
        cells.append(dict(p=p, s=s, d=d, label=lab, cond=r0.cond, mode=r0["mode"], T=r0["T"], n=int(len(g)),
                          look_c=m, look_lo=lo, look_hi=hi, d_look=dm, d_look_lo=dlo, d_look_hi=dhi,
                          d_lookF=fm, d_lookF_lo=flo, d_lookF_hi=fhi,
                          look_J=g.look_J.mean(), look_F=g.look_F.mean(), ent_c=g.ent_c.mean(), end_c=g.end_c.mean(),
                          look_c_far=g.look_c_far.mean(), hop_s=g.hop_s.median(), price=g.price.median(),
                          t80_look=g.t80_look.astype(float).median(), lost=g.lost.mean(), contacts=g.contacts.mean(),
                          kills=g.kills.apply(len).mean() if "kills" in g else None, reform_ok=g.reform_ok.mean(),
                          enveloped=g.enveloped.mean() if "enveloped" in g and r0.cond == "hop" else None,
                          t_envelop=g.t_envelop.astype(float).median() if "t_envelop" in g and r0.cond == "hop" else None))
    spec = json.loads((root / LOOKDOWN_A / "spec.json").read_text(encoding="utf-8"))
    chosen = LS.choose(LOOKDOWN_A, designs=[tuple(x) for x in spec["designs"]])
    located = None                                           # citysim.lookdown_contacts: which building each touched
    contacts_csv = root / "lookdown_A" / "contacts.csv"       # located on the 80 m run: the same flights
    if contacts_csv.exists():
        ct = pd.read_csv(contacts_csv)
        if len(ct):
            dec = (ct.t >= ct.t_J) & (ct.t <= ct.t_commit)
            located = dict(n=int(len(ct)), own=int((ct.kind == "own").sum()), across=int((ct.kind == "across").sum()),
                           other=int((ct.kind == "other").sum()), own_before_decision=int(((ct.kind == "own") & dec).sum()),
                           d=sorted(float(x) for x in ct.d.unique()), p=sorted(float(x) for x in ct.p.unique()))
    out = dict(a=dict(spec=spec, sight_m=float(spec.get("look_range_m", 80.0)),
                      n_trials=int(df.groupby(["p", "s", "city_seed", "trial"]).ngroups),
                      n_flights=int(len(df)), cells=_clean(cells), chosen=_clean(chosen), contacts=located,
                      p1_dstar={f"{p:g}|{s:g}": d for (p, s), d in LS.P2_DSTAR.items()},
                      rule=dict(lost=LS.SAFE_LOST, contacts=LS.SAFE_CONTACTS, time_price=LS.TIME_PRICE)))
    b = root / LOOKDOWN_B
    if (b / "results.jsonl").exists():
        dfb, _ = LS.read(LOOKDOWN_B)
        if len(dfb):
            specb = json.loads((b / "spec.json").read_text(encoding="utf-8"))
            k = f"r{float(specb.get('sight_m', 80.0)):g}"
            g = dfb.groupby(["p", "s", "condition"])
            summ = g.agg(n=("t_r80", "size"), found=(f"hit_{k}", "mean"), rmst=(f"t_{k}", "mean"),
                         found_r80=("hit_r80", "mean"), rmst_r80=("t_r80", "mean"),
                         found_r40=("hit_r40", "mean"), rmst_r40=("t_r40", "mean"), found_r200=("hit_r200", "mean"),
                         rmst_r200=("t_r200", "mean"), lost=("lost", "mean"), contacts=("contacts", "mean"),
                         decisions=("decisions", "mean"), blind=("blind_decisions", "mean"), path_m=("path_m", "mean"),
                         footpath=("footpath_seen_share", "mean"), d_block=("d_block", "first"),
                         d_street=("d_street", "first"), t_hop=("t_hop", "first"), mode=("mode", "first")).reset_index()
            rat = LS.ratios(dfb, measure=f"t_{k}")
            out["b"] = dict(spec={x: specb[x] for x in ("designs", "cities", "targets", "T_s", "conditions", "chosen",
                                                        "p2_dstar", "rule", "city_seed0") if x in specb},
                            sight_m=float(specb.get("sight_m", 80.0)),
                            n_targets=int(dfb.groupby(["p", "s", "city_seed", "target_i"]).ngroups),
                            n_flights=int(len(dfb)), summary=_records(summ), ratios=_records(rat))
    notes = root / (sight_run("lookdown_notes") + ".html")   # the tab's closing box, written from that range's results
    if notes.exists():
        out["notes"] = notes.read_text(encoding="utf-8")
    return out


def _log_ratio_ci(d):
    """(ratio, lo, hi, n) from per-city log ratios: a t-free normal interval, as the tables beside it use."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return None
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    return dict(ratio=math.exp(m), lo=math.exp(m - 1.96 * se), hi=math.exp(m + 1.96 * se), n_cities=int(len(d)))


def _decompose(df, a, b, keys, found="hit_r80", t="t_r80"):
    """Condition a against b over the searches both found: the geometric-mean ratios of the decisions taken, of the
    time per decision and of the time to find, and each condition's median seconds per decision."""
    A, B = (df[df.condition == c].set_index(keys) for c in (a, b))
    B = B.reindex(A.index)
    both = A[found].astype(bool) & B[found].fillna(False).astype(bool)
    both &= (A.decisions > 0) & (B.decisions.fillna(0) > 0)     # a target seen before the first decision has no ratio
    A, B = A[both], B[both]
    if not len(A):
        return None
    dec, tt = np.log(A.decisions / B.decisions), np.log(A[t] / B[t])
    return dict(n=int(len(A)), decisions=math.exp(float(dec.mean())), per_decision=math.exp(float((tt - dec).mean())),
                time=math.exp(float(tt.mean())), s_per_a=float((A[t] / A.decisions).median()),
                s_per_b=float((B[t] / B.decisions).median()))


def answer_data():
    """The Answer tab: only what the study's purpose needs. Per block size, what one drone, the swarm in the streets
    and the swarm surrounding look down before choosing (Part A, with the lone drone's run beside it) and what it takes
    to surround the block at all; time to find against one drone (Part B and Phase 2's informed search); and why the
    surround does not turn what it sees into speed (decisions against time per decision)."""
    import pandas as pd
    from citysim import lookdown as LD, lookdown_search as LS, split_stats as SS
    root = S.OUT_ROOT.parent
    if not (root / LOOKDOWN_A / "results.jsonl").exists():
        return None
    out = dict(d=ANSWER_D, dial_max=1.6, sight_m=SIGHT_M)
    hit, tk = f"hit_r{SIGHT_M:g}", f"t_r{SIGHT_M:g}"           # the find at the page's sight range
    a, _ = LD.read(LOOKDOWN_A)
    if (root / LOOKDOWN_A1 / "results.jsonl").exists():
        a1, _ = LD.read(LOOKDOWN_A1)
        out["single_trials"] = int(len(a1) // 2)
        a = pd.concat([a, a1.dropna(axis=1, how="all")], ignore_index=True)
    rule = LS.SAFE_LOST, LS.SAFE_CONTACTS
    safe = lambda c: c["lost"] <= rule[0] and c["contacts"] <= rule[1]      # noqa: E731

    def cell(g):
        m, lo, hi = _tci(g.look_c)
        env = g.enveloped.dropna().astype(float) if "enveloped" in g else []      # S1's test, surrounds only
        return dict(look=m, lo=lo, hi=hi, look_F=float(g.look_F.mean()), hop_s=float(g.hop_s.median()),
                    lost=float(g.lost.mean()), contacts=float(g.contacts.mean()), n=int(len(g)),
                    env=float(np.mean(env)) if len(env) else None)

    designs = []
    for (p, s), g in a.groupby(["p", "s"]):
        st = g[g.cond == "street"].drop_duplicates(["city_seed", "trial"])        # flown at d 1.0 in every job
        row = dict(p=float(p), s=float(s), street=cell(st))
        for c in ("single", "single_look"):
            if (g.cond == c).any():
                row[c] = cell(g[g.cond == c])
        for T in (6.0, 12.0, 20.0):
            h = g[(g.cond == "hop") & (g["mode"] == "aim") & (g["T"] == T) & (g.d == ANSWER_D)]
            if len(h):
                row[f"hop{T:g}"] = cell(h)
        cands = []                                       # every surround setting: the most a safe one looks down
        for (d, mode, T), h in g[g.cond == "hop"].groupby(["d", "mode", "T"]):
            c = cell(h)
            cands.append(dict(c, d=float(d), mode=mode, T=float(T)))
        ok = [c for c in cands if safe(c)]
        row["widest_safe"] = max((c["d"] for c in ok), default=None)
        best = sorted(ok, key=lambda c: (-c["look"], c["T"], c["d"]))
        row["best"] = best[0] if best else None
        # whether the formation gets round the block at all (S1's enclosure test, which no sight range changes): the
        # aim-and-turn surround at the widest safe spread, by its length
        row["enclose"] = {f"{c['T']:g}": c for c in cands
                          if c["mode"] == "aim" and row["widest_safe"] is not None and c["d"] == row["widest_safe"]}
        designs.append(row)
    if RINDEX.exists():                                  # block and building sizes, from the gallery's cities
        gal = json.loads(RINDEX.read_text(encoding="utf-8")).get("gallery", [])
        for row in designs:
            g = next((x for x in gal if x["v"] == "R" and x["p"] == row["p"] and x["s"] == row["s"]), None)
            if g and g.get("stats"):
                row.update(block_m=g["stats"]["block_m"], building_m=g["stats"]["building_m"],
                           neighbour_m=g["stats"]["neighbour_m"])
    out["designs"] = designs
    if (root / LOOKDOWN_B / "results.jsonl").exists():   # Part B: a goal at a building
        b, _ = LS.read(LOOKDOWN_B)
        keys = ["p", "s", "city_seed", "target_i"]
        rat = LS.ratios(b, pairs=[(c, "single") for c in ANSWER_SEARCH], measure=tk)
        found = b.groupby("condition")[hit].mean()
        out["b"] = dict(n_targets=int(b.groupby(keys).ngroups), n_cities=int(b.groupby(["p", "s", "city_seed"]).ngroups),
                        designs=sorted({float(p) for p in b.p}),
                        ratios=_records(rat[rat.design == "pooled"]),
                        per_design=_records(rat[rat.design != "pooled"]),
                        found={k: float(v) for k, v in found.items()},
                        decomp={f"{p:g}": {c: _decompose(b[b.p == p], c, "swarm_street", keys, hit, tk)
                                           for c in ("single", "swarm_hop")} for p in sorted(set(b.p))})
    s2 = root / P2_SEARCH_SIGHT
    if (s2 / "results.jsonl").exists():                  # Phase 2: a point hidden at random, 48 cities
        df, _ = SS.read_search(s2)
        df = df[df.basis == "any"]
        rows = []
        for c in ANSWER_SEARCH:
            r = _log_ratio_ci(SS.search_ratio(df, c, "single", measure=tk).d.values)
            if r:
                rows.append(dict(a=c, b="single", **r))
        keys = ["variant", "p", "s", "city_seed", "target_i"]
        out["p2"] = dict(n_targets=int(df.groupby(keys).ngroups), n_cities=int(df.groupby(["p", "s", "city_seed"]).ngroups),
                         ratios=rows, found={k: float(v) for k, v in df.groupby("condition")[hit].mean().items()},
                         decomp={c: _decompose(df, c, "swarm_street", keys, hit, tk) for c in ("single", "swarm_hop")})
    return _clean(out)


def _clean(x):
    """JSON-safe: non-finite floats to None, numpy scalars to Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _records(df):
    return _clean(df.to_dict("records")) if df is not None and len(df) else []


def steps_data():
    """M5's rows: the scene (cohesion cutoff 20), the screens' profile (40) and R's confirmed one (30)."""
    rows = []
    for name, label in ((STEPS, None), (STEPS + "_r30", "R's profile (r0_coh 30)")):
        path = S.OUT_ROOT / name / "steps.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        for r in d["rows"]:
            if label and r["variant"] == "scene":
                continue                        # the scene row is the same flight in both runs
            rows.append(dict(r, label=label or ("the scene (r0_coh 20)" if r["variant"] == "scene"
                                                else "the screens (r0_coh 40)")))
    return dict(rows=rows) if rows else None


def feasibility():
    path = blockcity.LAYOUT_DIR / "sweep~" / blockcity.OUT_SET / "feasibility.csv"
    if not path.exists():
        return None
    import pandas as pd
    f = pd.read_csv(path)
    out = []
    for (v, p, s), g in f.groupby(["variant", "p", "s"]):
        row = dict(v=v, p=float(p), s=float(s), cities=int(len(g)))
        for col in ("alpha_achieved", "block_share", "ell_mean", "n_interior_blocks", "n_plazas", "n_buildings",
                    "cyl_gap_frac_ge6", "facade_gap_median", "d_star_pred_median", "d_star_pred_p90"):
            if col in g:
                row[col] = _f(g[col].mean(), 3)
        out.append(row)
    return out


def totals():
    """(manoeuvres flown, runs): every Phase 1 run's jobs that flew, whether or not the page shows them."""
    n = runs = 0
    for d in sorted(S.OUT_ROOT.iterdir()):
        if (d / "results.jsonl").exists():
            df = _read(d.name)
            if df is not None:
                n += int(len(df))
                runs += 1
    return n, runs


def replay_files():
    """{published path: source file}: the replays' cities and flights, which the Cities tab fetches beside the page
    (index.json is embedded in the page itself; picks/ is split_replays' own bookkeeping)."""
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def build(out_path=None):
    out = S.OUT_ROOT
    concl = (out / "conclusions.html")
    n_total, n_runs = totals()
    rindex = REPLAY / "index.json"
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), bars=T.BARS, n_total=n_total, n_runs=n_runs, sight=SIGHT_M,
                coupling=coupling_data(), screens=run_block(SCREENS),
                followups=run_block(FOLLOWUPS), step4=step4_data(), exits=exit_data(), confirm=confirm_data(),
                steps=steps_data(), feasibility=feasibility(), curve_t=CURVE_T, unity=gate_data(), phase2=phase2_data(),
                lookdown=lookdown_data(), answer=answer_data(),
                replay=json.loads(rindex.read_text(encoding="utf-8")) if rindex.exists() else None,
                conclusions=concl.read_text(encoding="utf-8") if concl.exists() else "")
    text = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    for ph in ("__DATA__", "/*__REPLAY_VIEW__*/", "<!--__RV_INLINE__-->"):
        assert ph in html, f"the template has no {ph} placeholder"
    view = VIEW_JS.read_text(encoding="utf-8")
    assert "</script" not in view.lower(), "replay_view.js must not contain a closing script tag"
    html = html.replace("/*__REPLAY_VIEW__*/", view).replace("__DATA__", text)
    files = replay_files()
    bare = html.replace("<!--__RV_INLINE__-->", "")
    inline = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in files.items()}
    full_body = html.replace("<!--__RV_INLINE__-->", '<script id="rv-inline" type="application/json">' +
                             json.dumps(inline, separators=(",", ":")).replace("</", "<\\/") + "</script>")
    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1,viewport-fit=cover\"></head><body>" + full_body +
            "</body></html>")
    dest = Path(out_path) if out_path else out / "split_phase1.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name(dest.stem + "_bare.html").write_text(bare, encoding="utf-8")
    manifest = {k: str(v) for k, v in files.items()}
    dest.with_name(dest.stem + "_files.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size = sum(v.stat().st_size for v in files.values())
    print(f"[split_report] {dest} ({len(full) / 1024:.0f} KB with the replays inlined); bare {len(bare) / 1024:.0f} KB "
          f"+ {len(files)} replay files ({size / 1e6:.1f} MB) to publish beside it")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("phase1")
    p.add_argument("--out")
    a = ap.parse_args()
    build(a.out)


if __name__ == "__main__":
    main()
