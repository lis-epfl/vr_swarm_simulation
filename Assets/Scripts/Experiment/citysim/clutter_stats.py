"""The clutter study's statistics (citysim.clutter flies it; docs/clutter_study/README.md is the manual).

Per paired unit (a walk; a standing point averaged over its body yaws) the log of swarm / single; averaged per city
(the cities are the independent units); per cell and spread the geometric-mean ratio, its 95 % t interval over the
cities, the SD of the city ratios, the SD of the unit log ratios, the one-sample t-test p over the cities and the
Benjamini-Hochberg q over a metric's cells (per spread); the spreads against each other the same way; and a
city-level least-squares trend of the log ratio on alpha and log2 p. A Wilcoxon signed-rank p over the units is kept
as a sensitivity: it treats walks in one city as independent, which they are not.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim.clutter import read_results  # noqa: E402


WALK_RATIOS = (   # key, column (both conditions), label, unit
    ("sweep", "sweep_any_r40_m", "Sweep width", "m"),
    ("rate", "area_any_r40_per_s", "Ground cleared per second", "m²/s"),
    ("per_m", "area_any_r40_per_m", "Ground cleared per metre flown", "m²/m"),
    ("sweep_head", "sweep_head_r40_m", "Sweep width, headset view", "m"),
    ("inst", "inst_any_r40_m2", "On screen while flying", "m²"),
    ("inst_head", "inst_head_r40_m2", "In the headset's view while flying", "m²"),
    ("find", "find_any_r200", "Chance of finding a hidden target in 5 min", "share"),
    ("find_speed", "find_any_r200_rate_per_h", "How much sooner a hidden target is found", "finds/h"),
)
STATIC_RATIOS = (  # key, swarm column, single column, label, unit
    ("st_any", "A_any_r40", "A_single_r40", "On screen, standing", "m²"),
    ("st_head", "A_head_r40", "A_single_r40", "In the headset's view, standing", "m²"),
    ("st_1s", "V_swarm_1_r40", "V_single_1_r40", "Within 1 s, standing", "m²"),
)
WALK_LEVELS = (   # absolute measures per condition: key, column, label, unit
    ("ring_conn", "ring_conn_frac", "Neighbouring feeds that share ground", "share"),
    ("ring_geo_conn", "ring_geo_conn_frac", "... if there were no buildings", "share"),
    ("ring_whole", "ring_whole_frac", "Time every neighbouring pair shares ground", "share"),
    ("ring_overlap", "ring_overlap_m2", "Ground a neighbouring pair shares", "m²"),
    ("wall_feeds", "wall_feed_frac", "Feeds filled by a facade", "share"),
    ("uncovered", "uncovered_deg", "Azimuth no feed covers", "deg"),
    ("facade", "cam_facade_m", "Camera to the nearest facade (median)", "m"),
    ("shown", "shown_feeds", "Feeds shown", "feeds"),
    ("behind", "behind_corner_r40_frac", "Ground seen first round a corner", "share"),
    ("find_time", "find_any_r200_time_min", "Mean time to find a hidden target (estimate)", "min"),
    ("find_p", "find_any_r200", "Chance of finding a hidden target in 5 min", "share"),
    ("lost_km", "lost_per_km", "Drones lost per km", "per km"),
    ("lost", "lost", "Drones lost per walk", "drones"),
    ("split", "split_frac", "Time the formation is split", "share"),
    ("speed", "speed_ms", "Speed", "m/s"),
    ("path_ratio", "path_ratio", "Path flown per metre of progress", "ratio"),
    ("adj", "adj_per_100m", "Stick adjustments per 100 m", "per 100 m"),
    ("morph", "morph_per_km", "Buildings morphed round per km", "per km"),
    ("street", "street_path_frac", "Path along the streets", "share"),
    ("first", "sw_first_frac_any40", "Shared ground the swarm saw first", "share"),
)
STATIC_LEVELS = (
    ("st_ring_conn", "ring_conn_frac", "Neighbouring feeds that share ground, standing", "share"),
    ("st_ring_geo_conn", "ring_geo_conn_frac", "... if there were no buildings, standing", "share"),
    ("st_wall_feeds", "wall_feed_frac", "Feeds filled by a facade, standing", "share"),
    ("st_useful", "useful_feeds", "Useful feeds, standing", "feeds"),
    ("st_epar", "E_par_r40", "Hidden ground seen by parallax, standing", "share"),
)
PRIMARY = "sweep"


def _t_interval(c):
    """(mean, lo, hi, sd, t, p) of a sample c (city means), the 95 % t interval and the one-sample t-test of 0."""
    from scipy import stats as st
    c = np.asarray(c, float)
    c = c[np.isfinite(c)]
    n = len(c)
    if n < 2:
        return (float(c.mean()) if n else math.nan,) + (math.nan,) * 5
    m, sd = float(c.mean()), float(c.std(ddof=1))
    if sd == 0.0:
        return m, m, m, 0.0, math.inf if m else math.nan, 0.0 if m else 1.0
    h = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    t = m / (sd / math.sqrt(n))
    return m, m - h, m + h, sd, t, float(2 * st.t.sf(abs(t), n - 1))


def bh(p):
    """Benjamini-Hochberg q values (NaN kept)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q


def paired_units(walks, static):
    """One row per paired unit and spread: (metric, cell, p, alpha, layout, unit, d_ref, sw, sg, lr) for every ratio
    metric -- a walk's swarm against its single drone (matched band), a standing point's swarm against the single
    drone on its centroid, averaged over the body yaws first."""
    out = []
    if len(walks):
        w = walks[walks.band == "matched"]
        sg = w[w.arm == "single"].set_index(["layout", "unit"])
        for key, col, _, _ in WALK_RATIOS:
            if col not in w:
                continue
            for d_ref, g in w[w.arm == "swarm"].groupby("d_ref"):
                m = g.set_index(["layout", "unit"])[[col, "cell", "p", "alpha"]].join(
                    sg[[col]].rename(columns={col: "_sg"}), how="inner")
                out.append(pd.DataFrame(dict(metric=key, cell=m.cell, p=m.p, alpha=m.alpha,
                                             layout=m.index.get_level_values(0), unit=m.index.get_level_values(1),
                                             d_ref=d_ref, sw=m[col].to_numpy(float), sg=m["_sg"].to_numpy(float))))
    if len(static):
        s = static[~static.failed.astype(bool)] if "failed" in static else static
        g = s.groupby(["cell", "p", "alpha", "layout", "unit", "d_ref"], as_index=False).mean(numeric_only=True)
        for key, a, b, _, _ in STATIC_RATIOS:
            if a in g and b in g:
                out.append(pd.DataFrame(dict(metric=key, cell=g.cell, p=g.p, alpha=g.alpha, layout=g.layout,
                                             unit=g.unit, d_ref=g.d_ref, sw=g[a].to_numpy(float),
                                             sg=g[b].to_numpy(float))))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lr"] = np.where((df.sw > 0) & (df.sg > 0), np.log(df.sw / df.sg), np.nan)
    return df


def ratio_cells(units):
    """Per metric x spread x cell: the swarm / single statistics of the module docstring."""
    rows = []
    for (metric, d_ref, cell), g in units.groupby(["metric", "d_ref", "cell"]):
        cm = g.groupby("layout").lr.mean()
        m, lo, hi, sd, t, p = _t_interval(cm.to_numpy())
        lr = g.lr.to_numpy(float)
        lr = lr[np.isfinite(lr)]
        rows.append(dict(metric=metric, d_ref=d_ref, cell=cell, p_m=g.p.iloc[0], alpha=g.alpha.iloc[0],
                         n_cities=int(cm.notna().sum()), n_units=int(len(lr)), ratio=math.exp(m) if np.isfinite(m)
                         else math.nan, lo=math.exp(lo) if np.isfinite(lo) else math.nan,
                         hi=math.exp(hi) if np.isfinite(hi) else math.nan,
                         sd_city_ratio=float(np.exp(cm.dropna()).std(ddof=1)) if cm.notna().sum() > 1 else math.nan,
                         sd_log_city=sd, sd_log_unit=float(lr.std(ddof=1)) if len(lr) > 1 else math.nan,
                         sw_mean=float(g.sw.mean()), sw_sd=float(g.sw.std(ddof=1)), sg_mean=float(g.sg.mean()),
                         sg_sd=float(g.sg.std(ddof=1)), t=t, p=p,
                         d_z=m / sd if sd and np.isfinite(sd) and sd > 0 else math.nan,
                         p_wilcoxon_units=_wilcoxon(lr), swarm_better_units=float((lr > 0).mean()) if len(lr) else
                         math.nan))
    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = np.nan
        for _, idx in df.groupby(["metric", "d_ref"]).groups.items():
            df.loc[idx, "q"] = bh(df.loc[idx, "p"].to_numpy())
    return df


def _wilcoxon(x):
    from scipy import stats as st
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) < 6:
        return math.nan
    return float(st.wilcoxon(x).pvalue)


def spread_contrasts(walks, static):
    """The swarm's spreads against each other (1.6 / 1.0 and 0.4 / 1.0), paired by unit, for every ratio metric:
    the same statistics as ratio_cells on the log of the two swarms' values."""
    u = paired_units(walks, static)
    if not len(u):
        return pd.DataFrame()
    piv = u.pivot_table(index=["metric", "cell", "p", "alpha", "layout", "unit"], columns="d_ref", values="sw")
    rows = []
    for a_, b_ in ((1.6, 1.0), (0.4, 1.0)):
        if a_ not in piv or b_ not in piv:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where((piv[a_] > 0) & (piv[b_] > 0), np.log(piv[a_] / piv[b_]), np.nan)
        d = piv.reset_index()[["metric", "cell", "p", "alpha", "layout", "unit"]].assign(lr=lr, sw=piv[a_].values,
                                                                                        sg=piv[b_].values)
        r = ratio_cells(d.assign(d_ref=a_))
        r["contrast"] = f"{a_:g} / {b_:g}"
        rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def level_cells(walks, static, comp):
    """Per measure x condition x cell: mean, SD over units and over cities, n (absolute measures, both interfaces)."""
    rows = []
    srcs = []
    if len(walks):
        w = walks[walks.band == "matched"].copy()
        if len(comp):
            first = comp.groupby(["layout", "unit", "d_ref"])["sw_first_frac_any40"].mean()
            w = w.join(first, on=["layout", "unit", "d_ref"])
        srcs.append((w, WALK_LEVELS))
    if len(static):
        s = static[~static.failed.astype(bool)] if "failed" in static else static
        s = s.groupby(["cell", "p", "alpha", "layout", "unit", "d_ref"], as_index=False).mean(numeric_only=True)
        s["arm"] = "swarm"
        srcs.append((s, STATIC_LEVELS))
    for df, spec in srcs:
        for key, col, _, _ in spec:
            if col not in df:
                continue
            for (arm, d_ref, cell), g in df.groupby(["arm", "d_ref", "cell"]):
                v = g[col].to_numpy(float)
                if not np.isfinite(v).any():
                    continue
                cm = g.groupby("layout")[col].mean()
                m, lo, hi, sd_c, _, _ = _t_interval(cm.to_numpy())
                rows.append(dict(metric=key, arm=arm, d_ref=d_ref, cell=cell, p_m=g.p.iloc[0], alpha=g.alpha.iloc[0],
                                 mean=float(np.nanmean(v)), sd_unit=float(np.nanstd(v, ddof=1)) if
                                 np.isfinite(v).sum() > 1 else math.nan, sd_city=sd_c, lo=lo, hi=hi,
                                 n_units=int(np.isfinite(v).sum()), n_cities=int(cm.notna().sum())))
    return pd.DataFrame(rows)


def trends(units):
    """Per metric x spread: OLS of the city-mean log ratio on alpha and log2 p (centred) and their product, cities as
    the units. The coefficient as a % change of the ratio per +0.01 alpha and per doubling of p, with SE and p."""
    from scipy import stats as st
    rows = []
    for (metric, d_ref), g in units.groupby(["metric", "d_ref"]):
        c = g.groupby(["layout", "p", "alpha"], as_index=False).lr.mean().dropna()
        if len(c) < 8:
            continue
        a = c.alpha.to_numpy() - c.alpha.mean()
        lp = np.log2(c.p.to_numpy()) - np.log2(c.p.to_numpy()).mean()
        X = np.c_[np.ones(len(c)), a, lp, a * lp]
        y = c.lr.to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        dof = len(y) - X.shape[1]
        s2 = float(res @ res) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tt = beta / se
        pp = 2 * st.t.sf(np.abs(tt), dof)
        r2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
        for k, (term, scale) in enumerate((("intercept", 1.0), ("alpha_per_0.01", 0.01), ("log2_p", 1.0),
                                           ("alpha_x_log2_p", 0.01))):
            rows.append(dict(metric=metric, d_ref=d_ref, term=term, coef=beta[k], se=se[k], t=tt[k], p=pp[k],
                             pct=(math.exp(beta[k] * scale) - 1) * 100 if term != "intercept" else
                             (math.exp(beta[k]) - 1) * 100, n_cities=len(c), r2=r2))
    return pd.DataFrame(rows)


CITY_LEVELS = ("ring_conn_frac", "ring_geo_conn_frac", "ring_whole_frac", "wall_feed_frac", "uncovered_deg",
               "cam_facade_m", "lost_per_km", "split_frac", "speed_ms", "shown_feeds")
LINK_Y = (("sweep_16_10", "Sweep width, spread 1.6 ÷ 1.0"), ("rate_16_10", "Ground per second, spread 1.6 ÷ 1.0"),
          ("sweep_04_10", "Sweep width, spread 0.4 ÷ 1.0"), ("sweep_16_sg", "Sweep width, swarm 1.6 ÷ single"),
          ("sweep_10_sg", "Sweep width, swarm 1.0 ÷ single"))
LINK_X = (("alpha", "Coverage α"), ("log2_p", "log2 block spacing"), ("ring_conn_frac_1.6", "Neighbour overlap at 1.6"),
          ("ring_conn_drop", "Neighbour overlap, 1.0 minus 1.6"), ("wall_feed_frac_1.6", "Wall-filled feeds at 1.6"),
          ("lost_per_km_1.6", "Drones lost per km at 1.6"), ("ring_conn_frac_1", "Neighbour overlap at 1.0"))


def city_summary(walks, units):
    """One row per city: its design, the swarm's ring and cost measures per spread (means over its walks), and the
    city-mean log ratios -- swarm / single per spread, and the spreads against 1.0 -- for sweep width and rate."""
    if not len(walks) or not len(units):
        return pd.DataFrame()
    w = walks[(walks.band == "matched") & (walks.arm == "swarm")]
    base = walks.groupby("layout")[["cell", "p", "alpha", "city_seed"]].first()
    cols = [c for c in CITY_LEVELS if c in w]
    lv = w.groupby(["layout", "d_ref"])[cols].mean().unstack("d_ref")
    lv.columns = [f"{c}_{d:g}" for c, d in lv.columns]
    out = base.join(lv)
    u = units[units.metric.isin(("sweep", "rate"))]
    lr = u.groupby(["layout", "metric", "d_ref"]).lr.mean().unstack(["metric", "d_ref"])
    for m in ("sweep", "rate"):
        for d, tag in ((0.4, "04"), (1.0, "10"), (1.6, "16")):
            if (m, d) in lr:
                out[f"{m}_{tag}_sg"] = lr[(m, d)]
        piv = u[u.metric == m].pivot_table(index=["layout", "unit"], columns="d_ref", values="sw")
        for a, tag in ((1.6, "16"), (0.4, "04")):
            if a in piv and 1.0 in piv:
                with np.errstate(divide="ignore", invalid="ignore"):
                    x = np.log(piv[a] / piv[1.0]).replace([np.inf, -np.inf], np.nan)
                out[f"{m}_{tag}_10"] = x.groupby(level=0).mean()
    out["log2_p"] = np.log2(out.p)
    if "ring_conn_frac_1" in out and "ring_conn_frac_1.6" in out:
        out["ring_conn_drop"] = out["ring_conn_frac_1"] - out["ring_conn_frac_1.6"]
    return out.reset_index()


def links(cs):
    """Across cities: Pearson r (and p) between each LINK_Y log ratio and each LINK_X city measure, and the same
    r with coverage and log2 block spacing partialled out (the design factors that drive both)."""
    from scipy import stats as st
    rows = []
    if not len(cs):
        return pd.DataFrame()
    for yk, yl in LINK_Y:
        for xk, xl in LINK_X:
            if yk not in cs or xk not in cs:
                continue
            d = cs[list(dict.fromkeys([yk, xk, "alpha", "log2_p"]))].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 8 or d[xk].std() == 0:
                continue
            r, p = st.pearsonr(d[xk], d[yk])
            row = dict(y=yk, y_label=yl, x=xk, x_label=xl, n_cities=len(d), r=r, p=p)
            if xk not in ("alpha", "log2_p"):
                Z = np.c_[np.ones(len(d)), d.alpha, d.log2_p]
                rx = d[xk] - Z @ np.linalg.lstsq(Z, d[xk], rcond=None)[0]
                ry = d[yk] - Z @ np.linalg.lstsq(Z, d[yk], rcond=None)[0]
                if rx.std() > 0 and ry.std() > 0:
                    pr = float(np.corrcoef(rx, ry)[0, 1])
                    dof = len(d) - 4
                    t = pr * math.sqrt(dof / max(1 - pr * pr, 1e-12))
                    row.update(r_partial=pr, p_partial=float(2 * st.t.sf(abs(t), dof)))
            rows.append(row)
    return pd.DataFrame(rows)


FIND_CURVE = "find_any_r200_curve"


def find_curves(walks):
    """Per design and condition: the chance a hidden target has been found by each step of the walk -- the mean of
    each city's mean curve, with the SD across cities."""
    from citysim.clutter import FIND_STEP_S
    rows = []
    if not len(walks) or FIND_CURVE not in walks:
        return pd.DataFrame()
    w = walks[walks.band == "matched"]
    for (cell, arm, d_ref), g in w.groupby(["cell", "arm", "d_ref"]):
        per_city = []
        for _, gg in g.groupby("layout"):
            cs = [c for c in gg[FIND_CURVE] if isinstance(c, (list, tuple)) and len(c)]
            if cs:
                n = min(len(c) for c in cs)
                per_city.append(np.mean([c[:n] for c in cs], axis=0))
        if not per_city:
            continue
        n = min(len(c) for c in per_city)
        A = np.array([c[:n] for c in per_city])
        m, sd = A.mean(0), (A.std(0, ddof=1) if len(A) > 1 else np.full(n, np.nan))
        for i in range(n):
            rows.append(dict(cell=cell, p=g.p.iloc[0], alpha=g.alpha.iloc[0], arm=arm, d_ref=d_ref, t=i * FIND_STEP_S,
                             mean=float(m[i]), sd_city=float(sd[i]), n_cities=len(A)))
    return pd.DataFrame(rows)


def tables(out):
    """Every table, the figures and the report page from a run's results.jsonl."""
    walks, comp, static, errs = read_results(out)
    for nm, df in (("walks", walks), ("compare", comp), ("static", static), ("errors", errs)):
        df.to_csv(out / f"{nm}.csv", index=False)
    units = paired_units(walks, static)
    units.to_csv(out / "units.csv", index=False)
    rc = ratio_cells(units) if len(units) else pd.DataFrame()
    rc.to_csv(out / "ratio_cells.csv", index=False)
    sc = spread_contrasts(walks, static)
    sc.to_csv(out / "spread_contrasts.csv", index=False)
    lv = level_cells(walks, static, comp)
    lv.to_csv(out / "level_cells.csv", index=False)
    tr = trends(units) if len(units) else pd.DataFrame()
    tr.to_csv(out / "trends.csv", index=False)
    find_curves(walks).to_csv(out / "find_curves.csv", index=False)
    cs = city_summary(walks, units)
    cs.to_csv(out / "city_summary.csv", index=False)
    links(cs).to_csv(out / "links.csv", index=False)
    if (out / "cities.csv").exists():
        cdf = pd.read_csv(out / "cities.csv")
        cdf[cdf.buildable].groupby("cell")[["alpha_achieved", "block_fill", "ell_m", "n_buildings", "n_blocks"]] \
            .mean().to_csv(out / "cell_descriptors.csv")
    pd.set_option("display.width", 250)
    if len(rc):
        show = rc[rc.metric == PRIMARY].sort_values(["d_ref", "p_m", "alpha"])
        print(show[["d_ref", "cell", "n_cities", "n_units", "ratio", "lo", "hi", "sd_city_ratio", "p", "q"]]
              .round(4).to_string(index=False))
    try:
        from citysim import clutter_report
        clutter_report.figures(out)
        clutter_report.build(out)
    except Exception as e:                       # noqa: BLE001 -- the tables stand without the page
        print(f"[clutter] figures / page not built: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
