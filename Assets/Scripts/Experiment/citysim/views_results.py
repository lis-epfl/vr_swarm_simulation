"""S12 Distributed Views: the results page, citysim/views_results.html filled with the main study's data.

    python -B -m citysim.views_results
    -> experiment_results/views_study/views_results.html (everything inlined), views_results_bare.html (the page alone,
       for publishing with views_results_files.json's replay files beside it)

Reads main_main, main_probe, main_sens, main_held, main_calib and spot (the bench gate) under
experiment_results/views_study, the human fits in human/, rule.json, and the replay index (views_replays). Every number
on the page is computed here from those files; the prose that interprets them is READING below, written after the runs
were read, and the verdicts follow rule.json, written before.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import views as V  # noqa: E402
from citysim import views_report as R  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "views_results.html"
S4_TEMPLATE = HERE / "split_report.html"
VIEW_JS = HERE / "replay_view.js"
OUT = V.OUT
MAIN, PROBE, SENS, HELD, CALIB, SPOT = "main_main", "main_probe", "main_sens", "main_held", "main_calib", "spot"
PRIMARY, OTHER = V.REC_MAIN, ("start" if V.REC_MAIN == "pooled" else "pooled")
CREDIT = {"pooled": "press-credited", "start": "first-sight"}
N_BOOT = 2000
_FR = {}


def frames(run):
    if run not in _FR:
        _FR[run] = V.frames(V.read(run))
    return _FR[run]


def sess(run, variant, cond, seed=V.CITY_SEED):
    S, G, B = frames(run)
    s = S[(S.variant == variant) & (S.cond == cond) & (S.city_seed == seed)]
    g = G[(G.variant == variant) & (G.cond == cond) & (G.city_seed == seed)]
    return {int(r.k): r for r in s.itertuples()}, g


def _t(r, cap=V.CAP_S):
    return r.task_s if r.reason == "done" else (r.t_end if r.reason == "lost" else cap)


def rmst(rows, cap=V.CAP_S):
    return R.km_rmst([_t(r) for r in rows], [r.reason == "done" for r in rows], cap)


def boot(ks, stat, seed=7):
    rng = np.random.default_rng(seed)
    ks = list(ks)
    out = []
    for _ in range(N_BOOT):
        v = stat([ks[i] for i in rng.integers(len(ks), size=len(ks))])
        if math.isfinite(v):
            out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (math.nan, math.nan)


def ratio(a, b):
    """Task time a / b (restricted means), paired over the goal draws both flew; a, b = (run, variant, cond[, seed]).
    Also at and between goals (geometric means over the goals found, the same draws)."""
    sa, ga = sess(*a)
    sb, gb = sess(*b)
    ks = sorted(set(sa) & set(sb))
    if not ks:
        return None
    est = rmst([sa[k] for k in ks]) / rmst([sb[k] for k in ks])
    lo, hi = boot(ks, lambda s_: rmst([sa[k] for k in s_]) / rmst([sb[k] for k in s_]))
    gai, gbi = ga[ga.k.isin(ks)], gb[gb.k.isin(ks)]

    def gm_ratio(col):
        by_a = {k: x[col].to_numpy() for k, x in gai.groupby("k")}
        by_b = {k: x[col].to_numpy() for k, x in gbi.groupby("k")}
        f = lambda s_: R.gmean(np.concatenate([by_a.get(k, []) for k in s_] or [[]])) / \
            R.gmean(np.concatenate([by_b.get(k, []) for k in s_] or [[]]))   # noqa: E731
        e = R.gmean(gai[col]) / R.gmean(gbi[col])
        l_, h_ = boot(ks, f, seed=11)
        return dict(est=e, lo=l_, hi=h_)
    return dict(est=est, lo=lo, hi=hi, n=len(ks), at=gm_ratio("at_s"), between=gm_ratio("between_s"))


def verdict(r, other):
    """rule.json's verdict on a swarm / other ratio."""
    if r is None:
        return dict(kind="und", text="not flown")
    if r["hi"] < 0.9:
        return dict(kind="swarm", text=f"the swarm is at least 10% faster than {other}")
    if r["lo"] > 1 / 0.9:
        return dict(kind="other", text=f"{other} is at least 10% faster than the swarm")
    if r["lo"] >= 0.9 and r["hi"] <= 1 / 0.9:
        return dict(kind="eq", text=f"equivalent to {other} within 10%")
    return dict(kind="und", text=f"undecided against {other} at this size")


def cond_row(run, variant, cond, seed=V.CITY_SEED):
    s, g = sess(run, variant, cond, seed)
    ks = sorted(s)
    rows = [s[k] for k in ks]
    est = rmst(rows)
    lo, hi = boot(ks, lambda s_: rmst([s[k] for k in s_]))
    by = {k: x for k, x in g.groupby("k")}

    def gm(col):
        e = R.gmean(g[col])
        l_, h_ = boot(ks, lambda s_: R.gmean(np.concatenate([by[k][col].to_numpy() for k in s_ if k in by] or [[]])),
                      seed=3)
        return dict(est=e, lo=l_, hi=h_)
    return dict(cond=cond, n=len(ks), task=dict(est=est, lo=lo, hi=hi),
                done=float(np.mean([r.reason == "done" for r in rows])),
                lost_sessions=float(np.mean([r.reason == "lost" for r in rows])),
                at=gm("at_s"), between=gm("between_s"), rec_d=float(g.rec_d.median()),
                notice_d=float(g.notice_d.median()), lost=int(sum(r.lost for r in rows)),
                contacts=int(sum(r.contacts for r in rows)))


# ============================================================================= sections

def main_section():
    rows = {v: {c: cond_row(MAIN, v, c) for c in V.CONDITIONS} for v in (PRIMARY, OTHER)}
    rat = {}
    for v in (PRIMARY, OTHER):
        rat[v] = dict(sphere=ratio((MAIN, v, "swarm"), (MAIN, v, "sphere")),
                      single=ratio((MAIN, v, "swarm"), (MAIN, v, "single")),
                      cam=ratio((MAIN, v, "sphere"), (MAIN, v, "single")))
    p = rat[PRIMARY]
    vs, vg = verdict(p["sphere"], "the 360 drone"), verdict(p["single"], "one drone")
    forest = []
    for v in (PRIMARY, OTHER):
        h = v != PRIMARY
        tag = f" ({CREDIT[v]})"
        forest += [dict(label=f"swarm ÷ 360 drone, task{tag}", c="sphere", hollow=h, head=not h, **p_(rat[v]["sphere"])),
                   dict(label=f"  at the goal{tag}", c="sphere", hollow=h, **p_(rat[v]["sphere"] and rat[v]["sphere"]["at"])),
                   dict(label=f"  between goals{tag}", c="sphere", hollow=h, **p_(rat[v]["sphere"] and rat[v]["sphere"]["between"])),
                   dict(label=f"swarm ÷ one drone, task{tag}", c="single", hollow=h, head=not h, **p_(rat[v]["single"])),
                   dict(label=f"  at the goal{tag}", c="single", hollow=h, **p_(rat[v]["single"] and rat[v]["single"]["at"])),
                   dict(label=f"  between goals{tag}", c="single", hollow=h, **p_(rat[v]["single"] and rat[v]["single"]["between"])),
                   dict(label=f"360 drone ÷ one drone (the camera){tag}", c="swarm", hollow=h, **p_(rat[v]["cam"]))]
    phase = {}
    for ph, col in (("at", "at"), ("between", "between")):
        phase[ph] = [dict(label=CREDIT[v], bars=[dict(c=c, label=NAMES[c], faint=v != PRIMARY,
                                                       **p_(rows[v][c][col])) for c in V.CONDITIONS])
                     for v in (PRIMARY, OTHER)]
        human = pd.read_csv(V.HUMAN / "goal_times.csv")
        phase[ph].append(dict(label="internal_4's pilots (95% over runs)", bars=[
            dict(c=c, label=NAMES[c], faint=True, **dict(zip(("est", "lo", "hi"),
                                                            R.people_ci(human, c, f"{col}_s"))))
            for c in V.CONDITIONS]))
    return dict(rows=rows, rat=rat, verdict=dict(sphere=vs, single=vg), forest=forest, phase=phase,
                n=rows[PRIMARY]["swarm"]["n"])


NAMES = {"swarm": "Swarm", "single": "One drone", "sphere": "360 drone"}


def p_(r):
    return dict(est=r["est"], lo=r["lo"], hi=r["hi"]) if r else dict(est=None, lo=None, hi=None)


def probe_section():
    out = {}
    cols = dict(swarm=["cf_real", "cf_centre", "cf_sphere_d", "cf_sphere_h", "cf_screen"],
                sphere=["cf_real", "cf_fwd", "cf_screen"], single=["cf_real", "cf_sphere_s", "cf_sphere_t"])
    for v in (PRIMARY, OTHER):
        var = f"probe_{v}"
        rates = {c: R.probe_rates(PROBE, var, c, cols[c]) for c in V.CONDITIONS}
        rat = dict(spread_vs_360=R.probe_ratio(PROBE, var, "swarm", "cf_real", "cf_sphere_d"),
                   spread_vs_360_low=R.probe_ratio(PROBE, var, "swarm", "cf_real", "cf_sphere_h"),
                   spread_vs_centre=R.probe_ratio(PROBE, var, "swarm", "cf_real", "cf_centre"),
                   sphere_vs_fwd=R.probe_ratio(PROBE, var, "sphere", "cf_real", "cf_fwd"),
                   single_360=R.probe_ratio(PROBE, var, "single", "cf_sphere_t", "cf_real"))
        _, G, _ = frames(PROBE)
        out[v] = dict(rates=rates, rat=rat, goals=int(((G.variant == var) & (G.at_goal_s >= 100)).sum()),
                      n=int(G[G.variant == var].k.nunique()))
    r = out[PRIMARY]["rates"]
    lab = {("swarm", "cf_real"): "the swarm as flown (wrapped)", ("swarm", "cf_sphere_h"): "a 360 camera (2K stream) on its most central drone",
           ("swarm", "cf_sphere_d"): "… the same at 9.9 m", ("swarm", "cf_centre"): "its ten feeds moved to its centre",
           ("sphere", "cf_real"): "the 360 drone as flown", ("sphere", "cf_fwd"): "its forward camera instead",
           ("single", "cf_real"): "one drone as flown", ("single", "cf_sphere_t"): "a 360 camera (2K stream) on its flight"}
    groups = []
    for c in V.CONDITIONS:
        bars = []
        for col in {"swarm": ["cf_real", "cf_sphere_h", "cf_sphere_d", "cf_centre"], "sphere": ["cf_real", "cf_fwd"],
                    "single": ["cf_real", "cf_sphere_t"]}[c]:
            x = r[c][col]
            bars.append(dict(c=c if col == "cf_real" else ("sphere" if "sphere" in col else c), label=lab[(c, col)],
                             faint=col != "cf_real", est=x["est"], lo=x["lo"], hi=x["hi"]))
        groups.append(dict(label=f"on the {NAMES[c].lower()}'s flights", bars=bars))
    return dict(by=out, groups=groups)


SENS_ALL = [("rec_short", "the hat's d50 × 0.75"), ("rec_long", "the hat's d50 × 1.33"),
            ("crowd_short", "the crowd's d50 at 200 m"), ("crowd_long", "the crowd's d50 at 400 m"),
            ("no_explore", "the walk without its exploration term")]
SENS_SPHERE = [("sph_cube", "the 360 drone shows the cube, as SIMI flew it (22 px/deg)"),
               ("sph_soft", "the stream costs half as much (√ of the scaling), crowd unscaled"),
               ("sph_crowd", "the stream scales the hat but not the crowd"),
               ("sph_simi", "the 360 drone flown SIMI's way (the goal's junctions, 9.9 / 19.7 m)"),
               ("sph_high", "the 360 drone flown the single drone's way at SIMI's altitudes")]


def sens_section():
    rows, forest = [], []
    base_s = ratio((MAIN, PRIMARY, "swarm"), (MAIN, PRIMARY, "sphere"))
    for v, lab in SENS_ALL:
        a = ratio((SENS, v, "swarm"), (SENS, v, "sphere"))
        b = ratio((SENS, v, "swarm"), (SENS, v, "single"))
        rows.append(dict(label=lab, sphere=a, single=b))
    for v, lab in SENS_SPHERE:
        a = ratio((MAIN, PRIMARY, "swarm"), (SENS, v, "sphere"))
        cam = ratio((SENS, v, "sphere"), (MAIN, PRIMARY, "single"))
        rows.append(dict(label=lab, sphere=a, single=None, cam=cam))
    a = ratio((SENS, "sw_circuit", "swarm"), (MAIN, PRIMARY, "sphere"))
    b = ratio((SENS, "sw_circuit", "swarm"), (MAIN, PRIMARY, "single"))
    rows.append(dict(label="the swarm circles the goal's block instead of wrapping its building", sphere=a, single=b))
    forest.append(dict(label="main study: swarm ÷ 360 drone", c="sphere", head=True, **p_(base_s)))
    for r in rows:
        if r["sphere"]:
            forest.append(dict(label=r["label"], c="sphere", **p_(r["sphere"])))
    return dict(rows=rows, forest=forest)


def held_section():
    out = []
    for seed in V.HELD_SEEDS:
        a = ratio((HELD, PRIMARY, "swarm", seed), (HELD, PRIMARY, "sphere", seed))
        b = ratio((HELD, PRIMARY, "swarm", seed), (HELD, PRIMARY, "single", seed))
        c = ratio((HELD, PRIMARY, "sphere", seed), (HELD, PRIMARY, "single", seed))
        out.append(dict(seed=seed, sphere=a, single=b, cam=c))
    return out


def calib_section():
    human = pd.read_csv(V.HUMAN / "goal_times.csv")
    ci = {(c, col): R.people_ci(human, c, col) for c in V.CONDITIONS for col in ("at_s", "between_s")}
    rci = {b: R.people_ratio_ci(human, "swarm", b, "at_s") for b in ("single", "sphere")}
    _, G, _ = frames(CALIB)
    rows = [["internal_4's pilots (95% over runs)"] + [f"{ci[(c, 'at_s')][0]:.0f} ({ci[(c, 'at_s')][1]:.0f}–{ci[(c, 'at_s')][2]:.0f})"
                                                       for c in V.CONDITIONS]
            + [f"{rci['single'][0]:.2f} ({rci['single'][1]:.2f}–{rci['single'][2]:.2f})",
               f"{rci['sphere'][0]:.2f} ({rci['sphere'][1]:.2f}–{rci['sphere'][2]:.2f})"]
            + [f"{ci[(c, 'between_s')][0]:.0f} ({ci[(c, 'between_s')][1]:.0f}–{ci[(c, 'between_s')][2]:.0f})"
               for c in V.CONDITIONS] + [""]]
    res = {}
    for rm in ("pooled", "start"):
        for pat in ("", "_wrap"):
            g = G[G.variant == rm]
            if pat:
                g = pd.concat([g[g.cond != "swarm"], G[G.variant == rm + pat]])
            at = {c: R.gmean(g[g.cond == c].at_s) for c in V.CONDITIONS}
            bt = {c: R.gmean(g[g.cond == c].between_s) for c in V.CONDITIONS}
            r1, r2 = at["swarm"] / at["single"], at["swarm"] / at["sphere"]
            passes = [ci[(c, "at_s")][1] <= at[c] <= ci[(c, "at_s")][2] for c in V.CONDITIONS] + \
                     [ci[(c, "between_s")][1] <= bt[c] <= ci[(c, "between_s")][2] for c in V.CONDITIONS] + \
                     [rci["single"][1] <= r1 <= rci["single"][2], rci["sphere"][1] <= r2 <= rci["sphere"][2]]
            n = int(sum(passes))
            res[rm + (pat or "_circuit")] = n
            pill = f'<span class="pill {"ok" if n == 8 else "na" if n >= 5 else "no"}">{n} of 8</span>'
            rows.append([f"{CREDIT[rm]}, swarm {'wrapped' if pat else 'circling'}"] +
                        [f"{at[c]:.0f}" for c in V.CONDITIONS] + [f"{r1:.2f}", f"{r2:.2f}"] +
                        [f"{bt[c]:.0f}" for c in V.CONDITIONS] + [pill])
    return dict(rows=rows, passes=res, n=int(frames(CALIB)[0].k.nunique()))


def spot_section():
    p = OUT / SPOT / "bench" / "gate.jsonl"
    if not p.exists():
        return dict(rows=[], agree=None)
    g = pd.read_json(p, lines=True)
    rows, agree = [], {}
    for cond, x in g.groupby("cond"):
        u, r = x.u_route_s.median(), x.r_route_s.median()
        uw, rw = x.u_wrapped.median(), x.r_wrapped.median()
        ul, rl = int(x.u_lost.sum()), int(x.r_lost.sum())
        from scipy import stats
        def pci(k):
            return (stats.chi2.ppf(0.025, 2 * k) / 2 if k else 0.0, stats.chi2.ppf(0.975, 2 * k + 2) / 2)
        lo_r, hi_r = pci(rl)
        lo_u, hi_u = pci(ul)
        ok_t = abs(u / r - 1) <= 0.10 if r else False
        ok_w = (abs(uw - rw) <= 0.10) if cond == "swarm" and pd.notna(uw) and pd.notna(rw) else None
        ok_l = (lo_r <= ul <= hi_r) and (lo_u <= rl <= hi_u)
        agree[cond] = dict(time=ok_t, wrap=ok_w, losses=ok_l)
        rows.append([NAMES[cond], str(len(x)), f"{u:.0f} / {r:.0f} s", f"{x.u_route_s.notna().mean():.0%} / {x.r_route_s.notna().mean():.0%}",
                     f"{x.u_speed.median():.2f} / {x.r_speed.median():.2f} m/s",
                     f"{uw:.2f} / {rw:.2f}" if cond == "swarm" else "–", f"{ul} / {rl}",
                     f"{int(x.u_contacts.sum())} / {int(x.r_contacts.sum())}", f"{x.gap_med.median():.1f} m",
                     " ".join(f'<span class="pill {"ok" if v else "no"}">{k}</span>' for k, v in agree[cond].items()
                              if v is not None)])
    return dict(rows=rows, agree=agree)


def power_section(pred):
    t = pd.read_csv(V.REPO / "experiment_results" / "trials" / "internal_4" / "tables" / "runs.csv", sep=";")
    t["lt"] = np.log(t.time_s)
    grp = t.groupby(["pid", "condition"])["lt"]
    res = t["lt"] - grp.transform("mean")
    df = int((grp.count() - 1).sum())
    sd_w = float(np.sqrt((res ** 2).sum() / df))
    pw = R.power(sd_w=sd_w, sd_int=0.30)
    tab = [[f"{ratio_:.2f}×", *[str(next(n for rr, n in r["pts"] if abs(rr - ratio_) < 1e-6)) for r in pw]]
           for ratio_ in (1.25, 1.4, 1.5, 1.75, 2.0)]
    n_pred = None
    if pred and pred >= 1.1:
        for r in pw:
            if r["runs"] == 2:
                n_pred = next((n for ratio_, n in r["pts"] if ratio_ >= pred - 1e-9), None)
    return dict(sd_w=sd_w, df=df, table=tab, n_pred=n_pred, pred=pred)


def human_section():
    rec = json.loads((V.HUMAN / "recognition_fits.json").read_text(encoding="utf-8"))
    dis = json.loads((V.HUMAN / "discovery_fits.json").read_text(encoding="utf-8"))
    basis = json.loads((V.HUMAN / "basis.json").read_text(encoding="utf-8")) if (V.HUMAN / "basis.json").exists() else {}
    beh = pd.read_csv(V.HUMAN / "behaviour_legs.csv")
    return rec, dis, basis, beh


def replay_files():
    rp = OUT / "replay"
    if not (rp / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(rp).as_posix()}": f for d in ("cities", "flights") for f in sorted((rp / d).glob("*.json"))}


def s4_style():
    m = re.search(r"<style>(.*?)</style>", S4_TEMPLATE.read_text(encoding="utf-8"), re.S)
    return m.group(1)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def fr(r, nd=2):
    return "–" if not r or r.get("est") is None else f"{r['est']:.{nd}f} ({r['lo']:.{nd}f}–{r['hi']:.{nd}f})"


# ============================================================================= assembly

def assemble():
    M = main_section()
    P = probe_section()
    Sx = sens_section()
    Hd = held_section()
    Cb = calib_section()
    Sp = spot_section()
    rec, dis, basis, beh = human_section()
    rule = json.loads((OUT / "rule.json").read_text(encoding="utf-8"))
    s = M["rat"][PRIMARY]
    pred = s["sphere"]["est"]
    Pw = power_section(max(pred, 1 / pred) if pred else None)
    cube = cond_row(SENS, "sph_cube", "sphere")
    return dict(M=M, P=P, S=Sx, H=Hd, C=Cb, Sp=Sp, rec=rec, dis=dis, basis=basis, beh=beh, rule=rule, Pw=Pw, cube=cube)


def page(A, reading):
    M, P, Sx, Hd, Cb, Sp, Pw = A["M"], A["P"], A["S"], A["H"], A["C"], A["Sp"], A["Pw"]
    rec, dis, basis, beh = A["rec"], A["dis"], A["basis"], A["beh"]
    rp, op = M["rows"][PRIMARY], M["rows"][OTHER]
    s, so = M["rat"][PRIMARY], M["rat"][OTHER]
    vs, vg = M["verdict"]["sphere"], M["verdict"]["single"]
    pr, prr = P["by"][PRIMARY]["rates"], P["by"][PRIMARY]["rat"]
    sens = {r["label"]: r for r in Sx["rows"]}
    cube = next(r for r in Sx["rows"] if r["label"].startswith("the 360 drone shows the cube"))
    soft = next(r for r in Sx["rows"] if r["label"].startswith("the stream costs half"))
    simi = next(r for r in Sx["rows"] if r["label"].startswith("the 360 drone flown SIMI"))
    pf = rec["d_head|lat1.0|all"]["fit"]
    sf = rec["start|d_head|all"]["fit"]
    nf = dis["d_head|lat1.0"]["fit"]
    fs = V.sphere_factor()
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"))
    data["cond_about"] = dict(swarm="ten drones wrapped round each goal building, a feed per hull drone on the ring of screens",
                              single="one drone, the trials' forward camera on one screen",
                              sphere="one drone flown the single drone's way, its 360 camera on a sphere round the pilot as a 2K stream")
    data["answer"] = reading["answer"](A)
    data["reading_short"] = reading["short"](A)
    data["reading"] = reading["long"](A)
    data["intro"] = (f"{M['n']} sessions in the trial city, each flown in all three conditions with the same three goals, hats and "
                     f"walkers and the same random thresholds, so every comparison is paired. Task time is the restricted mean "
                     f"to the {V.CAP_S:.0f} s cap (a session not done counts to the cap; a swarm under four drones to where it "
                     f"stopped), with 95% bootstrap intervals over sessions. The verdict rule was written before the run "
                     f"(<code>rule.json</code>): an interval wholly below 0.9 means the swarm is at least 10% faster, wholly "
                     f"above 1/0.9 the other condition is, wholly inside the green band equivalent within 10%. Recognition is "
                     f"credited at the press (the credit the calibration gate passed); first-sight credit is shown hollow.")
    # ---- the headline: four task times, the 360 drone at the resolution it was modelled at and at full resolution
    cb = A["cube"]
    def tile(key, name, r, ratio_, note):
        return dict(key=key, name=name, est=r["task"]["est"], lo=r["task"]["lo"], hi=r["task"]["hi"], done=r["done"],
                    at=r["at"]["est"], ratio=ratio_, note=note)
    data["headline"] = dict(
        cap=V.CAP_S, n=M["n"],
        tiles=[tile("swarm", "Swarm", rp["swarm"], None, "ten drones, a feed per hull drone"),
               tile("single", "One drone", rp["single"], s["single"], "the trials' forward camera"),
               tile("cube", "360 drone, full resolution", cb, cube["sphere"],
                    f"the cube SIMI flew, {V.resolution(True, 0):.0f} px/deg"),
               tile("sphere", "360 drone, 2K stream", rp["sphere"], s["sphere"],
                    f"the A1's live view, {V.resolution(True):.1f} px/deg")],
        res=[dict(c="cube", label="360 drone, full resolution (the cube)", est=V.resolution(True, 0)),
             dict(c="swarm", label="a feed screen (swarm and one drone)", est=V.resolution()),
             dict(c="sphere", label="360 drone, 2K stream (the main study)", est=V.resolution(True))],
        fs=fs)
    data["main"] = dict(
        n=M["n"], forest=M["forest"], phase=M["phase"],
        verdict_text=(f"<b>Swarm ÷ 360 drone {fr(s['sphere'])}</b>: {vtag(vs)}. <b>Swarm ÷ one drone {fr(s['single'])}</b>: "
                      f"{vtag(vg)}. First-sight credit: {fr(so['sphere'])} and {fr(so['single'])}. "
                      f"The 360 drone ÷ one drone, the same flights with only the camera changed: {fr(s['cam'])}."),
        forest_caption=("Ratios of task times, paired over sessions; filled the press-credited model, hollow first sight. The "
                        "green band is ±10%. The indented rows are the geometric-mean time per goal at the goal and between "
                        "goals over the same sessions."),
        phase_text=(f"At a goal the swarm takes {rp['swarm']['at']['est']:.0f} s per goal, the 360 drone "
                    f"{rp['sphere']['at']['est']:.0f} s and one drone {rp['single']['at']['est']:.0f} s; between goals "
                    f"{rp['swarm']['between']['est']:.0f}, {rp['sphere']['between']['est']:.0f} and "
                    f"{rp['single']['between']['est']:.0f} s. internal_4's pilots are beside them (their sphere is SIMI's, "
                    f"the cube, flown SIMI's way)."),
        phase_caption="Geometric mean per goal (times under 10 s count as 10 s), 95% over sessions; the people's over runs.")
    data["probe"] = dict(
        groups=P["groups"],
        text=(f"The probe searches every goal for exactly 120 s whatever is recognised, so every view of the same flight is "
              f"scored over the same window ({P['by'][PRIMARY]['n']} sessions, {P['by'][PRIMARY]['goals']} goals). On the "
              f"swarm's own flights, wrapped round the goal building, its spread-out feeds recognise "
              f"{fr(prr['spread_vs_360_low'], 0)}× the hats of a 360 camera with the 2K stream on its most central drone at the "
              f"swarm's height ({fr(prr['spread_vs_360'], 0)}× at the 360 drone's 9.9 m), and {fr(prr['spread_vs_centre'], 0)}× the "
              f"same ten feeds moved to its centre. On the 360 drone's flights the stream recognises {fr(prr['sphere_vs_fwd'])}× "
              f"what a forward camera would; on one drone's flights a 360 stream would recognise {fr(prr['single_360'])}× its "
              f"forward camera."),
        caption="Hat recognitions per minute at a goal over the probe's 120 s (goals searched at least 100 s), the "
                "press-credited rate, 95% over sessions. Faint bars: the same flight seen another way.")
    srows = [dict(label="swarm ÷ 360 drone, the 2K stream (main)", c="sphere", head=True, **p_(s["sphere"])),
             dict(label="swarm ÷ 360 drone, the cube SIMI flew", c="sphere", **p_(cube["sphere"])),
             dict(label="swarm ÷ 360 drone, the stream at half its cost", c="sphere", **p_(soft["sphere"])),
             dict(label="360 drone ÷ one drone, the 2K stream", c="swarm", head=True, **p_(s["cam"])),
             dict(label="360 drone ÷ one drone, the cube", c="swarm", **p_(cube.get("cam"))),
             dict(label="360 drone ÷ one drone, the stream at half its cost", c="swarm", **p_(soft.get("cam")))]
    data["stream"] = dict(
        forest=srows,
        text=(f"internal_4's only 360 flights (SIMI's) showed the cube itself, about {V.resolution(True, 0):.0f} pixels per "
              f"degree reaching the eye, more than a feed screen's {V.resolution():.1f}. Since 2026-11-18 the sphere shows "
              f"what a real 360 drone's live link carries, the Antigravity A1's 2K view: {V.resolution(True):.1f} pixels per "
              f"degree, so a hat at d metres covers as many pixels as one {fs:.1f} d metres away on a feed. Recognising a "
              f"21 cm hat is a matter of pixels across it, so the stream puts the 360 drone's hats {fs:.1f} times further "
              f"away, its half-rate distance at {pf['d50'] / fs:.0f} m instead of {pf['d50']:.0f} m, and a crowd's at "
              f"{nf['d50'] / fs:.0f} m instead of {nf['d50']:.0f} m. No one has flown the stream yet, so this part is a model, "
              f"not a measurement: the ratios below put it beside the cube SIMI flew and beside a stream that costs half as "
              f"much."),
        caption=("Swarm ÷ 360 drone and 360 drone ÷ one drone, task times, with the 360 drone's display as flown in the main "
                 "study (the 2K stream), as SIMI flew it (the cube) and with the stream's cost halved in log (the hat's "
                 "distance × √3.1, the crowd's not scaled): the optimistic bound."))
    data["flights_intro"] = (
        "The trial city (BlockShapedPatchCityWorld's layout, ScenicPatchCityWorld's too), with sessions of the main study "
        "flown in all three conditions on the same goal draw: each goal's building patch (☆, its zone dashed), each hat "
        "recognised (▲ where the formation was at the press) and the start line. Every replay is the study's own flight, "
        "re-flown with a tape and kept only because it re-flies exactly as stored. The Unity spot check's routes follow: "
        "Unity's own flight beside the replica twin flown from Unity's start. Space plays or pauses, ← and → step 5 s.")
    if data["flights_intro"] and not replay_files():
        data["flights_intro"] = ""
    # ---- All results
    tables = {}
    head = ["condition", "credit", "task (s, restricted mean, 95%)", "done in 900 s", "at a goal (s)", "between goals (s)",
            "hat recognised at (m, median)", "crowd noticed at (m, median)", "drones lost", "contacts"]
    rows = []
    for v in (PRIMARY, OTHER):
        for c in V.CONDITIONS:
            r = M["rows"][v][c]
            rows.append([NAMES[c], CREDIT[v], fr(r["task"], 0), f"{r['done']:.0%}", fr(r["at"], 0), fr(r["between"], 0),
                         f"{r['rec_d']:.0f}", f"{r['notice_d']:.0f}", str(r["lost"]), str(r["contacts"])])
    tables["cond"] = dict(head=head, rows=rows, note=(
        "The swarm's losses are drones (a colliding pair is two); one drone and the 360 drone lose none. Distances are "
        "metres in the world, so the 360 drone's recognitions, made through the 2K stream, are close in."))
    rr = []
    for v in (PRIMARY, OTHER):
        for key, lab in (("sphere", "swarm ÷ 360 drone"), ("single", "swarm ÷ one drone"), ("cam", "360 drone ÷ one drone")):
            x = M["rat"][v][key]
            rr.append([lab, CREDIT[v], fr(x), fr(x["at"]), fr(x["between"]), str(x["n"])])
    tables["ratios"] = dict(head=["ratio", "credit", "task", "at the goal", "between goals", "sessions"], rows=rr)
    sr = []
    for r in Sx["rows"]:
        sr.append([r["label"], fr(r["sphere"]), fr(r.get("single")) if r.get("single") else "–",
                   fr(r.get("cam")) if r.get("cam") else "–", str(r["sphere"]["n"]) if r["sphere"] else "–"])
    tables["sens"] = dict(head=["changed", "swarm ÷ 360 drone", "swarm ÷ one drone", "360 drone ÷ one drone", "sessions"],
                          rows=sr, forest=Sx["forest"],
                          note=("100 of the main sessions each (k 1000–1099), press-credited. A change to the 360 drone "
                                "alone is set against the main study's swarm and one drone on the same goal draws."))
    tables["held"] = dict(head=["held-out city", "swarm ÷ 360 drone", "swarm ÷ one drone", "360 drone ÷ one drone", "sessions"],
                          rows=[[f"seed {h['seed']}", fr(h["sphere"]), fr(h["single"]), fr(h["cam"]),
                                 str(h["sphere"]["n"]) if h["sphere"] else "–"] for h in Hd],
                          note="The same design at three other seeds (S6's held-out set), reported per city, never pooled.")
    pb = []
    labels = {"cf_real": "as flown", "cf_centre": "the same feeds at the swarm's centre", "cf_sphere_d": "a 360 stream on its most central drone, 9.9 m",
              "cf_sphere_h": "a 360 stream on its most central drone, the swarm's height", "cf_screen": "every shown screen, wherever the head was",
              "cf_fwd": "a forward camera instead", "cf_sphere_s": "a 360 stream on its flight, 9.9 m", "cf_sphere_t": "a 360 stream on its flight, its own height"}
    for v in (PRIMARY, OTHER):
        for c in V.CONDITIONS:
            for col, x in P["by"][v]["rates"][c].items():
                pb.append([NAMES[c], CREDIT[v], labels.get(col, col), fr(x)])
    tables["probe"] = dict(head=["flights", "credit", "seen as", "recognitions / min at a goal"], rows=pb,
                           note="The 360 bases are scored through the 2K stream (its pixel factor), the feed bases at the feeds'.")
    tables["calib"] = dict(head=["", "at the goal (s): swarm", "one drone", "360", "at: swarm ÷ one", "swarm ÷ 360",
                                 "between (s): swarm", "one drone", "360", "gate"], rows=Cb["rows"],
                           text=(f"Before the main run the model flew {Cb['n']} sessions per condition the way internal_4's "
                                 "pilots flew (the swarm circling the block as most did, and wrapped; the sphere SIMI's way, "
                                 "showing the cube), and was held to the people's times per goal. Press-credited recognition "
                                 "passes the most checks, so the verdict is read on it."),
                           note=("Geometric means per goal; a time under 10 s counts as 10 s. The people's sphere is SIMI's two "
                                 "runs, so its between-goal interval is narrow. Every model is slower than the people between goals."))
    tables["spot"] = dict(head=["condition", "routes", "route time, Unity / twin", "completed", "speed on go legs", "wrapped at the goal",
                                "drones lost", "contacts", "centroid gap (median)", "agrees"], rows=Sp["rows"],
                          text=("Twelve of the study's routes per condition, flown by Unity on the headless bench and by the "
                                "replica from Unity's own start (the 360 drone flies the single drone's routes). Agreement "
                                "rule (rule.json): the median route time within 10%, the swarm's wrapped share within 0.10, "
                                "the drone losses each inside the other's exact 95% Poisson interval."))
    tables["power"] = dict(head=["true ratio", "1 run per condition", "2 runs", "3 runs"], rows=Pw["table"],
                           text=(f"Pilots for 80% power (two-sided 0.05, a t test over pilots) on the swarm ÷ 360 task-time "
                                 f"ratio, from internal_4's run-to-run spread within a pilot and condition (SD {Pw['sd_w']:.2f} "
                                 f"in log time, {Pw['df']} df, YEGO included) and an assumed 0.3 spread of the pilots' own "
                                 f"ratios." + (f" At the predicted {Pw['pred']:.2f}× (either way round), two runs per condition "
                                               f"need {Pw['n_pred']} pilots." if Pw["n_pred"] else "")))
    tables["intro"] = ("Every number behind the answer. Sessions are the units; a ratio is paired over the goal draws both "
                       "conditions flew, with a 95% bootstrap interval over them.")
    data["tables"] = tables
    # ---- the people
    tab = rec["d_head|lat1.0|all"]["table"]
    pilots = [[pid, str(rec[f"pilot|{pid}"]["n"]), f"{rec[f'pilot|{pid}']['exposed_s']:.0f}",
               f"{rec[f'pilot|{pid}']['fit']['lam0']:.2f}", f"{rec[f'pilot|{pid}']['fit']['d50']:.0f}",
               f"{rec[f'pilot|{pid}']['median_event_m']:.0f}"] for pid in ("ERKS", "ALEX", "SIMI", "YEGO") if f"pilot|{pid}" in rec]
    tenth = pf["d50"] * 9 ** (1 / pf["k"])
    old = basis.get("central0.8|px0|noYEGO", {}).get("press", {})
    data["human"] = dict(
        intro=("How far a hat and a goal's crowd can be seen, and how each condition was flown, all fitted to internal_4's "
               "recorded runs (ERKS, ALEX, SIMI, YEGO): what was on the pilot's screens inside the headset's view, measured "
               "every 0.1 s, and what the pilot did."),
        changes=CHANGES(A),
        rec=dict(
            text=(f"For every goal, each 0.1 s the hat was on a shown screen inside the head's view with a clear line is "
                  f"exposure, and the press (credited 1 s before it) an event. The hazard by distance is fitted as "
                  f"λ(d) = λ0 / (1 + (d/d50)^k): <b>λ0 {pf['lam0']:.2f}/s, half that at {pf['d50']:.0f} m "
                  f"(95% {pf['d50_lo']:.0f}–{pf['d50_hi']:.0f}), a tenth at {tenth:.0f} m</b> (k {pf['k']:.1f}), over "
                  f"{int(sum(b['events'] for b in tab))} presses. Credited at the start of the final sighting instead: "
                  f"λ0 {sf['lam0']:.2f}/s, d50 {sf['d50']:.0f} m. The proposal's fit, before YEGO, on the central 80%: "
                  f"λ0 {old.get('lam0', float('nan')):.2f}, d50 {old.get('d50', float('nan')):.0f} m."),
            caption="Dots: presses per second of exposure in distance bins, with 95% intervals; lines: the fits.",
            curves=[dict(lab=f"press-credited, d50 {pf['d50']:.0f} m", c="people", a=pf["lam0"], d50=pf["d50"], k=pf["k"]),
                    dict(lab=f"first sight, d50 {sf['d50']:.0f} m", c="people", dash="5 4", a=sf["lam0"], d50=sf["d50"], k=sf["k"])],
            table=tab, bins_head=["distance (m)", "s in view", "presses", "per second (95%)"],
            bins=[[f"{b['d_lo']}–{b['d_hi']}", f"{b['exposure_s']:.0f}", str(b["events"]),
                   f"{b['hazard']:.2f} ({b['lo']:.2f}–{b['hi']:.2f})"] for b in tab],
            pilots_head=["pilot", "presses", "s in view", "λ0 (/s)", "d50 (m)", "median press distance (m)"], pilots=pilots,
            note=("Each pilot's own fit; YEGO, who flew one drone at about 2 m, recognised quickly but only up close. "
                  "Distances are pixel-equivalent at the trials' feeds (below); for the feed conditions they are metres.")),
        res=dict(
            text=(f"Distances are fitted in pixel-equivalent metres: a hat d metres away on a display that delivers r pixels per "
                  f"world degree counts as d × r_feed / r metres, r_feed being the trials' feeds on their 60° screens. What "
                  f"reaches the eye is the lesser of what the image holds and what the headset (about 22 px/deg) shows of it."),
            head=["display", "image (px/deg)", "shown at", "reaching the eye (px/deg)", "distance factor"],
            rows=[["feed, 1920 × 1080 on a 60° screen (scale 1.3)", f"{1920 / 74.4:.1f}", "0.81 of life size",
                   f"{V.resolution():.1f}", "1.00"],
                  ["360 cube, 2048 per face (SIMI, internal_4)", f"{2048 / 90:.1f}", "life size", f"{V.resolution(True, 0):.1f}",
                   f"{V.sphere_factor(dict(sphere_stream=0)):.2f}"],
                  ["360 live stream, 2048 × 1024 (since 2026-11-18)", f"{2048 / 360:.1f}", "life size",
                   f"{V.resolution(True):.1f}", f"{fs:.2f}"]],
            curves=[dict(lab="a feed", c="single", a=pf["lam0"], d50=pf["d50"], k=pf["k"]),
                    dict(lab="the cube", c="sphere", dash="5 4", a=pf["lam0"], d50=pf["d50"], k=pf["k"],
                         scale=V.sphere_factor(dict(sphere_stream=0))),
                    dict(lab="the 2K stream", c="sphere", a=pf["lam0"], d50=pf["d50"], k=pf["k"], scale=fs)],
            caption=("The press-credited recognition rate against the hat's real distance, through each display. Through the "
                     f"stream a hat is recognised at half rate at {pf['d50'] / fs:.0f} m; the pixels across a 21 cm hat at "
                     f"the fit's d50 are {0.21 / pf['d50'] * 180 / math.pi * V.resolution():.1f} on a feed.")),
        notice=dict(
            text=(f"On each leg between goals, the distance within which one of the goal's walkers was expected on the head's "
                  f"screens, and the moment the pilot turned onto the goal for good; legs that began already on the final "
                  f"approach are censored. With YEGO's legs the fit is flat in its steepness (k from 1 to 20 moves the "
                  f"likelihood by under 0.3), so k is fixed at {nf['k']:.0f}, a near-step: <b>noticed at {nf['mu0']:.2f}/s "
                  f"while within about {nf['d50']:.0f} m</b> (95% from {nf['d50_lo']:.0f} m, no upper bound). The "
                  f"sensitivities put it at 200 and 400 m. Walkers are drawn to 500 m, the scenes' sleep distance since "
                  f"2026-11-18."),
            caption="Dots: turns per second of exposure on the legs with a turn (understating the rate by the censored legs); line: the fit.",
            table=[dict(d_lo=int(b["bin"].split("-")[0]), d_hi=int(b["bin"].split("-")[1]), rate=b["rate"], lo=None, hi=None)
                   for b in dis["d_head|lat1.0"]["table"]],
            curves=[dict(lab=f"k {nf['k']:.0f}, d50 {nf['d50']:.0f} m", c="people", a=nf["mu0"], d50=nf["d50"], k=nf["k"]),
                    dict(lab=f"k free ({dis['free|d_head|lat1.0']['fit']['k']:.1f})", c="people", dash="5 4",
                         a=dis["free|d_head|lat1.0"]["fit"]["mu0"], d50=dis["free|d_head|lat1.0"]["fit"]["d50"],
                         k=dis["free|d_head|lat1.0"]["fit"]["k"])],
            head=["latency", "μ0 (/s)", "d50 (m, k 8)", "events", "censored"],
            rows=[[k_.split("lat")[1] + " s", f"{v['fit']['mu0']:.3f}", f"{v['fit']['d50']:.0f} ({v['fit']['d50_lo']:.0f}–"
                   f"{'∞' if v['fit'].get('d50_hi_open') else round(v['fit']['d50_hi'])})", str(v["fit"]["n_events"]),
                   str(v["fit"]["n_censored"])] for k_, v in dis.items() if k_.startswith("d_head|")]),
        beh=dict(text=BEH_TEXT, head=["at a goal", "", "speed (m/s)", "stopped", "yawing", "inside the ring", "from the centre (m)"],
                 rows=beh_rows(beh)))
    data["terms"] = TERMS
    data["foot"] = (f"Built {data['built']} by citysim/views_results.py from experiment_results/views_study (main_main, "
                    "main_probe, main_sens, main_held, main_calib, spot, human/, rule.json). Manual: docs/views_study/README.md.")
    data["replay"] = json.loads((OUT / "replay" / "index.json").read_text(encoding="utf-8")) if replay_files() else None
    return data


BEH_TEXT = ("At a goal the model's pilots against internal_4's (time-weighted over the at-goal time). The swarm wraps "
            "the goal building (the main study's pattern, the experimenter's call), so it holds still more and sits nearer the "
            "centre than the people, who wrapped about a third of the time; the 360 drone flies the single drone's searches.")


def beh_rows(beh):
    rows = []
    _, _, B = frames(MAIN)
    for c in V.CONDITIONS:
        hs = beh[(beh.condition == V.HUMAN_COND[c]) & (beh.phase == "search")]
        w = hs.dur_s / hs.dur_s.sum()
        rows.append([NAMES[c], "internal_4", f"{(hs.speed * w).sum():.1f}", f"{(hs.stopped * w).sum():.0%}",
                     f"{(hs.yawing * w).sum():.0%}", f"{(hs.inside_ring * w).sum():.0%}", f"{hs.goal_dist_med.median():.0f}"])
        b = B[(B.cond == c) & (B.variant == PRIMARY)]
        rows.append(["", "the model", f"{b.speed.median():.1f}", f"{b.stopped.median():.0%}", f"{b.yawing.median():.0%}",
                     f"{b.inside_ring.median():.0%}", f"{b.goal_dist_med.median():.0f}"])
    return rows


def vtag(v):
    cls = {"swarm": "swarm", "other": "other", "eq": "eq", "und": "und"}[v["kind"]]
    return f'<span class="vtag {cls}">{v["text"]}</span>'


def CHANGES(A):
    b = A["basis"]
    g = lambda k: b.get(k, {}).get("press", {})   # noqa: E731
    return [
        f"<b>YEGO's four runs joined internal_4</b> (two swarm, two one-drone, ScenicPatchCityWorld, 2026-11-18): 54 correct "
        f"presses instead of 41, 4 pilots instead of 3. On the proposal's basis the hat's half-rate distance went "
        f"{g('central0.8|px0|noYEGO').get('d50', 0):.0f} → {g('central0.8|px0|all').get('d50', 0):.0f} m: YEGO flew one drone "
        f"at about 2 m and the swarm low, and recognised up close.",
        f"<b>A hat counts anywhere in the frame</b>, as the recorder has scored since 2026-11-18, not only in its central "
        f"80%: the four presses the old basis could not explain are explained (d50 {g('central1.0|px0|all').get('d50', 0):.0f} m).",
        f"<b>The 360 drone shows the 2K live stream</b> (Sphere360Display.streamWidth 2048 since 2026-11-18), about a third of "
        f"the pixels per degree a feed screen delivers, where SIMI flew the cube. Distances are now fitted in pixel-equivalent "
        f"metres (d50 {g('central1.0|px1|all').get('d50', 0):.0f} m at the feeds), and the 360 drone's hats and crowds count "
        f"{V.sphere_factor():.1f}× further away. No human has flown the stream yet: this is a model of it, not a measurement.",
        "<b>The crowd's noticing curve has a fixed shape</b> (k 8): with YEGO's legs its steepness is not identified. "
        "Walkers are drawn to 500 m (the scenes' sleep distance since 2026-11-18, 350 before).",
        "<b>Altitudes and gimbal pitches are the median over pilots of each pilot's median</b>, so YEGO's 2 m single-drone "
        "flying counts once: one drone 6.6 m at a goal (was 7.0), the swarm 7.2 m (10.4), the sphere as SIMI flew 9.9 m (14.3).",
        "<b>The verdict is read on press-credited recognition</b>, the credit the calibration gate passes with the new fits "
        "(first sight was the better one before, and is now faster than the people at the goal).",
        "<b>The swarm wraps every goal building</b> and <b>the 360 drone flies the single drone's searches at its altitudes</b> "
        "(the experimenter's two calls in review), so the 360 drone and one drone differ only in the camera.",
    ]


TERMS = [
    ["The conditions", [
        ["Swarm", "Ten drones at d_ref 2.0 under BlockShapedPatchCityWorld's law; the hull drones' feeds on a ring of 60° screens round the pilot; at a goal it flies to the centre building and holds there so the formation wraps it."],
        ["One drone", "One drone with the trials' forward camera on one screen, searching a goal by working round the ring buildings, orbiting some with the camera turned in and passing others, stopping to look round."],
        ["360 drone", "The same drone and the same searches, its 360 camera shown on a sphere round the pilot as the Antigravity A1's 2K live stream; the pilot's head picks where to look."]]],
    ["The task", [
        ["Goal", "A building with a ring of buildings across the streets round it, 16 pedestrians walking round each ring building, one wearing the hat; three goals per session, drawn as BlockGoals draws them at Play."],
        ["At the goal / between goals", "A goal's time is split at its zone (the patch grown by 15 m): from the previous press to the zone is between goals, from the zone to the press at the goal."],
        ["Task time", "From the start line to the press on the third hat; the restricted mean to 900 s over sessions."],
        ["Session", "One goal draw (goals, hats, walkers, the random thresholds) flown in every condition: the unit of every interval."]]],
    ["The model", [
        ["Recognition hazard", "Hat recognitions per second while the hat is on a screen in the head's view with a clear line, as a function of distance: λ0 / (1 + (d/d50)^k), fitted to internal_4."],
        ["Press-credited / first sight", "A recognition credited at the press (1 s before it), or at the start of the final continuous sighting before it (pilots sometimes closed in to check). The two bound the truth."],
        ["Noticing hazard", "The same for a goal's crowd: noticed per second while its walkers are expected on the head's screens."],
        ["Pixel-equivalent distance", "A distance scaled by the trials' feeds' pixels per degree over the display's: the 2K stream counts 3.1× further, SIMI's cube 0.81×."],
        ["Probe", "Every goal searched for exactly 120 s whatever is recognised, so different views of the same flight are scored over the same window."],
        ["Twin", "The replica flown from Unity's recorded start along the same route, for the spot check."]]],
]


def write(data):
    text = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    for ph in ("__DATA__", "/*__REPLAY_VIEW__*/", "<!--__RV_INLINE__-->", "/*__S4_STYLE__*/"):
        assert ph in html, ph
    html = html.replace("/*__S4_STYLE__*/", s4_style()).replace("/*__REPLAY_VIEW__*/", VIEW_JS.read_text(encoding="utf-8"))
    html = html.replace("__DATA__", text)
    files = replay_files()
    bare = html.replace("<!--__RV_INLINE__-->", "")
    inline = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in files.items()}
    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1,viewport-fit=cover\"></head><body>" +
            html.replace("<!--__RV_INLINE__-->", '<script id="rv-inline" type="application/json">' +
                         json.dumps(inline, separators=(",", ":")).replace("</", "<\\/") + "</script>") + "</body></html>")
    dest = OUT / "views_results.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("views_results_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("views_results_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                          encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(clean({k: v for k, v in data.items() if k not in ("replay",)}), indent=1,
                                                 allow_nan=False), encoding="utf-8")
    print(f"[views_results] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} replay files")
    return dest


def main():
    from citysim import views_reading
    A = assemble()
    write(page(A, views_reading.READING))


if __name__ == "__main__":
    main()
