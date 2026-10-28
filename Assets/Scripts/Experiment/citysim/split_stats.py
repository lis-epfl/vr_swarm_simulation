"""S4 Phase 2 statistics (the split study): the hypotheses fixed in docs/split_study/README.md section 5.

    python -B -m citysim.split tables --name N      # -> experiment_results/split_study/<N>/*.csv, primary.json

    python -B -m citysim.split primaries --walks N --search M   # H1 and H2 with Holm over the two

Cities are the units (lesson 13): every paired quantity is averaged over a city's walks (or targets) first.
  H1 (primary)  the share of a block's radiating streets previewed at the commit, swarm_hop minus the time-matched
                swarm_hold, both at the profile's separation, over the hops every condition of the walk reached;
                one-sided t over the city means pooled across the designs flown
  H2 (primary)  the informed search's time to find at 80 m (any feed), capped (RMST): the city-mean log ratio
                swarm_hop / swarm_street at the profile's separation, one-sided t below 0; Holm over H1 and H2
  secondary     log ratios of the chain walks' time to find (RMST) and street clearance between conditions, per
                design (t over cities, Benjamini-Hochberg over designs) and pooled; trends of the city-mean log ratio
                on centred log2 p, s and their product; the costs (losses, contacts, reform failures) and the
                correspondence measures per condition
Imported only by the parent, so it may change while a run goes.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import clutter_stats  # noqa: E402

PROFILE = "profile"
CONTRASTS = [("swarm_hop", "swarm_street"), ("swarm_hop", "swarm_hold"), ("swarm_hop", "single"),
             ("swarm_street", "single"), ("swarm_pass", "swarm_street"), ("swarm_hop", "single_look"),
             ("single_look", "single")]
RATIO_MEASURES = ["find_rmst_any_r80", "find_rmst_head_r80", "find_rmst_any_r40", "find_rmst_any_r200",
                  "street_km_per_min"]


def read(out):
    by = {}
    for line in (Path(out) / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["job_id"] in by and "error" not in by[r["job_id"]] and "error" in r:
            continue
        by[r["job_id"]] = r
    flights, hops = [], []
    for r in by.values():
        if "rows" not in r:
            continue
        key = dict(job_id=r["job_id"], variant=r["variant"], p=r["p"], s=r["s"], city_seed=r["city_seed"],
                   walk=r["walk"])
        common = min((x["hops_done"] for x in r["rows"]), default=0)
        for x in r["rows"]:
            f = dict(key, **{k: v for k, v in x.items() if k not in ("hops", "kills")})
            f["common_hops"] = common
            # the per-block measures are taken over the blocks with a building: a plaza is crossed by the streets
            hh = [h for h in x["hops"] if h["hop"] < common and not h.get("plaza", False)]
            f["preview_common"] = float(np.mean([h["preview_search"] for h in hh])) if hh else math.nan
            f["preview_common_transit"] = float(np.mean([h["preview_transit"] for h in hh])) if hh else math.nan
            rf = [h.get("reform_ok") for h in hh if h.get("reform_ok") is not None]
            f["reform_ok_share"] = float(np.mean(rf)) if rf else math.nan
            flights.append(f)
            for h in x["hops"]:
                hops.append(dict(key, condition=x["condition"], spread=x["spread"], **h))
    return pd.DataFrame(flights), pd.DataFrame(hops)


def ttest_one_sided(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return math.nan, math.nan, n
    from scipy import stats
    t = float(x.mean() / (x.std(ddof=1) / math.sqrt(n))) if x.std(ddof=1) > 0 else math.inf
    return t, float(stats.t.sf(t, n - 1)), n


def paired(flights, a, b, measure, spread_a=PROFILE, spread_b=PROFILE, log=True):
    """Per walk: measure of condition a against b (log ratio, or difference), averaged per city."""
    fa = flights[(flights.condition == a) & (flights.spread.astype(str) == str(spread_a))]
    fb = flights[(flights.condition == b) & (flights.spread.astype(str) == str(spread_b))]
    keys = ["variant", "p", "s", "city_seed", "walk"]
    m = fa[keys + [measure]].merge(fb[keys + [measure]], on=keys, suffixes=("_a", "_b"))
    va, vb = m[f"{measure}_a"].astype(float), m[f"{measure}_b"].astype(float)
    m["d"] = np.log(va / vb) if log else va - vb
    m = m[np.isfinite(m["d"])]
    return m.groupby(["variant", "p", "s", "city_seed"]).d.mean().reset_index()


def contrast_table(flights):
    rows = []
    for a, b in CONTRASTS:
        for meas in RATIO_MEASURES:
            if meas not in flights:
                continue
            cm = paired(flights, a, b, meas)
            if not len(cm):
                continue
            for (p, s), g in cm.groupby(["p", "s"]):
                x = g.d.values
                n = len(x)
                mu = float(x.mean())
                se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
                from scipy import stats
                pval = float(2 * stats.t.sf(abs(mu / se), n - 1)) if n > 1 and se > 0 else math.nan
                rows.append(dict(a=a, b=b, measure=meas, p=p, s=s, n_cities=n, ratio=round(math.exp(mu), 4),
                                 lo=round(math.exp(mu - 1.96 * se), 4) if math.isfinite(se) else math.nan,
                                 hi=round(math.exp(mu + 1.96 * se), 4) if math.isfinite(se) else math.nan, p_value=pval))
    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = np.nan
        for (a, b, meas), g in df.groupby(["a", "b", "measure"]):
            df.loc[g.index, "q"] = clutter_stats.bh(g.p_value.fillna(1.0).values)
    return df


def trends(flights):
    rows = []
    for a, b in CONTRASTS:
        for meas in RATIO_MEASURES:
            if meas not in flights:
                continue
            cm = paired(flights, a, b, meas)
            if len(cm) < 6:
                continue
            # a factor flown at one level has no trend (Phase 2 flies one block pitch): fit only those that vary
            lp, sc = np.log2(cm.p) - np.log2(cm.p).mean(), cm.s - cm.s.mean()
            cols, names = [np.ones(len(cm))], ["c"]
            if lp.std() > 0:
                cols.append(lp.values)
                names.append("p")
            if sc.std() > 0:
                cols.append(sc.values)
                names.append("s")
            if "p" in names and "s" in names:
                cols.append(lp.values * sc.values)
                names.append("ps")
            if len(names) == 1:
                continue
            X = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(X, cm.d.values, rcond=None)
            resid = cm.d.values - X @ beta
            dof = max(len(cm) - X.shape[1], 1)
            cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            row = dict(a=a, b=b, measure=meas, n_cities=len(cm))
            if "p" in names:
                i = names.index("p")
                row.update(pct_per_doubling_p=round(100 * (math.exp(beta[i]) - 1), 2), se_log2p=round(float(se[i]), 4))
            if "s" in names:
                i = names.index("s")
                row.update(pct_per_m_street=round(100 * (math.exp(beta[i]) - 1), 3), se_s=round(float(se[i]), 4))
            rows.append(row)
    return pd.DataFrame(rows)


def primary_h1(flights):
    cm = paired(flights, "swarm_hop", "swarm_hold", "preview_common", log=False)
    t, p, n = ttest_one_sided(cm.d.values)
    per = []
    for (pp, ss), g in cm.groupby(["p", "s"]):
        tt, pv, nn = ttest_one_sided(g.d.values)
        per.append(dict(p=pp, s=ss, n_cities=nn, diff=round(float(g.d.mean()), 4), p_one_sided=pv))
    per = pd.DataFrame(per)
    if len(per):
        per["q"] = clutter_stats.bh(per.p_one_sided.fillna(1.0).values)
    return dict(hypothesis="H1", measure="share of radiating streets previewed at the commit (search band)",
                contrast="swarm_hop - swarm_hold at the profile's separation", n_cities=n,
                mean_diff=round(float(np.nanmean(cm.d)), 4) if len(cm) else None, t=t, p_one_sided=p), per


def read_search(out):
    """The informed search's flights: one row per target x condition x basis, with the time to find at each range
    capped at the search's cap (a target not found counts as found at the cap: RMST)."""
    by = {}
    for line in (Path(out) / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["job_id"] in by and "error" not in by[r["job_id"]] and "error" in r:
            continue
        by[r["job_id"]] = r
    spec = json.loads((Path(out) / "spec.json").read_text(encoding="utf-8"))
    rows = []
    for r in by.values():
        if "rows" not in r:
            continue
        cap = float(r.get("T_s") or spec.get("search_T_s", 1200.0))    # each job's own cap (a --quick run's is short)
        key = dict(variant=r["variant"], p=r["p"], s=r["s"], city_seed=r["city_seed"], target_i=r["target_i"])
        for x in r["rows"]:
            row = dict(key, **x)
            for rng_m in [k[len("found_r"):] for k in x if k.startswith("found_r")]:     # every range the run recorded
                v = x.get(f"found_r{rng_m}")
                row[f"t_r{rng_m}"] = min(float(v), cap) if v is not None else cap
                row[f"hit_r{rng_m}"] = v is not None and float(v) <= cap
            rows.append(row)
    caps = sorted({float(r.get("T_s") or spec.get("search_T_s", 1200.0)) for r in by.values() if "rows" in r})
    return pd.DataFrame(rows), caps


def search_ratio(df, a, b, measure="t_r80", basis="any"):
    """Per city: log of condition a's RMST over condition b's (both on the same targets)."""
    keys = ["variant", "p", "s", "city_seed", "target_i"]
    fa = df[(df.condition == a) & (df.basis == basis)][keys + [measure]]
    fb = df[(df.condition == b) & (df.basis == basis)][keys + [measure]]
    m = fa.merge(fb, on=keys, suffixes=("_a", "_b"))
    g = m.groupby(["variant", "p", "s", "city_seed"])
    out = g.agg(a=(f"{measure}_a", "mean"), b=(f"{measure}_b", "mean"), n=(f"{measure}_a", "size")).reset_index()
    out["d"] = np.log(out.a / out.b)
    return out


def primary_h2(df):
    cm = search_ratio(df, "swarm_hop", "swarm_street")
    t, p, n = ttest_one_sided(-cm.d.values)          # H2: the log ratio is below 0
    # the secondary equivalence test (±10%) per design: the plan's small-building case, which both of Phase 2's are
    eq = {f"p{pp:g}_s{ss:g}": tost(g.d.values) for (pp, ss), g in cm.groupby(["p", "s"])} if len(cm) else {}
    return dict(hypothesis="H2", measure="informed search: RMST of the time to find at 80 m (any feed), capped",
                contrast="log(swarm_hop / swarm_street) at the profile's separation, city means", n_cities=n,
                mean_log_ratio=round(float(cm.d.mean()), 4) if len(cm) else None,
                ratio=round(float(math.exp(cm.d.mean())), 4) if len(cm) else None, t=-t if t == t else t,
                p_one_sided=p, equivalence=eq)


def tost(x, margin=math.log(1.1)):
    """Two one-sided t tests that the mean log ratio lies within ±margin (equivalence at 10%): None under 3 cities."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return None
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))
    p = max(float(stats.t.sf((m + margin) / se, n - 1)), float(stats.t.sf((margin - m) / se, n - 1)))
    return dict(n_cities=n, ratio=round(math.exp(m), 4), margin=round(math.exp(margin), 3), p_tost=p,
                equivalent=bool(p < 0.05))


def holm(ps):
    """Holm-adjusted p values, in the order given."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def search_tables(out):
    out = Path(out)
    df, caps = read_search(out)
    if not len(df):
        print("[split_stats] no search flights yet")
        return None
    df.to_csv(out / "search_flights.csv", index=False)
    h2 = primary_h2(df)
    rows = []
    for a, b in CONTRASTS:
        for basis in ("any", "head"):
            cm = search_ratio(df, a, b, basis=basis)
            if not len(cm):
                continue
            for (p, s), g in cm.groupby(["p", "s"]):
                x = g.d.values
                n = len(x)
                se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
                rows.append(dict(a=a, b=b, basis=basis, p=p, s=s, n_cities=n, ratio=round(math.exp(x.mean()), 4),
                                 lo=round(math.exp(x.mean() - 1.96 * se), 4) if se == se else None,
                                 hi=round(math.exp(x.mean() + 1.96 * se), 4) if se == se else None))
    ct = pd.DataFrame(rows)
    ct.to_csv(out / "search_contrasts.csv", index=False)
    found = df.groupby(["condition", "basis"]).agg(found_r80=("hit_r80", "mean"), rmst_r80=("t_r80", "mean"),
                                                   lost=("lost", "mean"), contacts=("contacts", "mean"),
                                                   decisions=("decisions", "mean")).round(3).reset_index()
    found.to_csv(out / "search_summary.csv", index=False)
    (out / "primary_h2.json").write_text(json.dumps(h2, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print("== H2:", h2)
    print(found.to_string(index=False))
    if len(ct):
        print(ct[ct.basis == "any"].to_string(index=False))
    return h2


def primaries(walks_out, search_out):
    """H1 (the walks) and H2 (the search) with Holm over the two: <search_out>/primaries.json."""
    flights, _ = read(walks_out)
    h1, _ = primary_h1(flights)
    df, _ = read_search(search_out)
    h2 = primary_h2(df)
    adj = holm([h1["p_one_sided"] if h1["p_one_sided"] == h1["p_one_sided"] else 1.0,
                h2["p_one_sided"] if h2["p_one_sided"] == h2["p_one_sided"] else 1.0])
    h1["p_holm"], h2["p_holm"] = float(adj[0]), float(adj[1])
    res = dict(H1=h1, H2=h2)
    (Path(search_out) / "primaries.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    return res


def costs(flights):
    g = flights.groupby(["condition", "spread", "p", "s"])
    cols = [c for c in ("lost", "contacts", "reform_ok_share", "split_share", "feeds_mean", "preview_common",
                        "street_km_per_min", "corr_a1_linked", "corr_islands", "corr_orphan_share",
                        "corr_a4_reach", "corr_b2_err_deg", "corr_b2_order_rev", "corr_c1_linked_share")
            if c in flights]
    return g[cols].mean().round(4).reset_index()


def tables(out):
    out = Path(out)
    first = (out / "results.jsonl").read_text(encoding="utf-8").split("\n", 1)[0] if (out / "results.jsonl").exists() \
        else ""
    if '"job_id": "search|' in first:                 # a search run's folder (split search), not the walks'
        return search_tables(out)
    flights, hops = read(out)
    if not len(flights):
        print("[split_stats] no flights yet")
        return
    flights.to_csv(out / "flights.csv", index=False)
    hops.to_csv(out / "hops.csv", index=False)
    h1, h1_per = primary_h1(flights)
    (out / "primary.json").write_text(json.dumps(h1, indent=1, default=str), encoding="utf-8")
    h1_per.to_csv(out / "h1_by_design.csv", index=False)
    ct = contrast_table(flights)
    ct.to_csv(out / "contrasts.csv", index=False)
    tr = trends(flights)
    tr.to_csv(out / "trends.csv", index=False)
    co = costs(flights)
    co.to_csv(out / "costs.csv", index=False)
    pd.set_option("display.width", 250)
    print("== H1:", h1)
    print(h1_per.to_string(index=False))
    if len(ct):
        print("\n== contrasts (time to find at 80 m, any feed)")
        print(ct[ct.measure == "find_rmst_any_r80"].to_string(index=False))
    print("\n== costs and correspondence by condition")
    print(co.groupby(["condition", "spread"]).mean(numeric_only=True).drop(columns=["p", "s"]).round(3).to_string())
