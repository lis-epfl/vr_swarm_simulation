"""S4 Phase 1 tables: the coupling map, the criteria fixed in the plan, T_hop, and the variant decision.

    python -B -m citysim.split_probe tables --name N        # -> experiment_results/split_study/phase1/<N>/*.csv

Imported only by the parent process, so it may be revised while a run is going (lesson 18).

Criteria (the plan's P1.5, fixed before any flight), per design, every one for both M3 and M3b:
  safety     drones lost per manoeuvre <= 0.03 and the Wilson 95 % upper bound on the share of manoeuvres losing any
             drone <= 0.06; contacts per manoeuvre <= 0.1
  reform     one cluster before the next decision point in >= 90 % of the manoeuvres that committed
  preview    M3b: >= 80 % of the block's radiating streets previewed at the commit (search band) in >= 80 % of trials;
             M3: the exit reached with no stall over 3 s in >= 95 %
  time price M3b's hop time <= 1.5 x the same formation round the block by the streets to the same exit (median)
  streets    M1: the swarm's speed >= 80 % of one drone's; <= 0.05 contacts per 100 m
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import blockcity  # noqa: E402

COUPLING_SHARE = 0.8
# the plan's grid was 2-10 s; a surround reaches a block's far side in ~20 s at p 50 (the first smoke flights), so the
# table runs on to 30 s and reports the continuous T_hop beside it
T_HOPS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
BARS = dict(lost_per=0.03, any_loss_ub=0.06, contacts_per=0.1, reform=0.9, preview_trials=0.8, preview_streets=0.8,
            m3_pass=0.95, time_price=1.5, speed_ratio=0.8, contacts_per_100m=0.05)


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return math.nan
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + half) / den


def _ok(df):
    return df[~df.get("error", pd.Series(index=df.index, dtype=object)).notna() &
              ~df.get("skipped", pd.Series(False, index=df.index)).fillna(False).astype(bool)]


def coupling(df):
    """Per variant x design x spread: S1's envelopment share (no drone lost), losses, contacts, time to envelop;
    d* = the smallest spread enveloping in >= COUPLING_SHARE of trials; beside it the predicted d* (ring perimeter /
    100) from the layouts."""
    m4 = _ok(df[df.kind == "M4"]).copy()
    if not len(m4):
        return pd.DataFrame(), pd.DataFrame()
    m4["env_ok"] = m4["enveloped"].fillna(False).astype(bool) & (m4["lost"] == 0)
    g = m4.groupby(["variant", "p", "s", "d_block"])
    cells = g.agg(trials=("env_ok", "size"), enveloped=("env_ok", "mean"), lost_per=("lost", "mean"),
                  contacts_per=("contacts", "mean"), t_envelop_median=("t_envelop", "median"),
                  split_share=("split_share", "mean")).reset_index()
    rows = []
    pred = predicted_d_star()
    for (v, p, s), c in cells.groupby(["variant", "p", "s"]):
        good = c[c.enveloped >= COUPLING_SHARE]
        d_env = float(good.d_block.min()) if len(good) else math.nan
        d_safe, rule = safe_d_star(c)
        at = c[c.d_block == d_safe]
        rows.append(dict(variant=v, p=p, s=s, d_star=d_safe, d_star_rule=rule, d_star_env=d_env,
                         d_star_pred=pred.get((v, p, s), math.nan),
                         share_at_d_star=round(float(at.enveloped.iloc[0]), 3) if len(at) else math.nan,
                         contacts_at_d_star=round(float(at.contacts_per.iloc[0]), 3) if len(at) else math.nan,
                         lost_at_d_star=float(at.lost_per.iloc[0]) if len(at) else math.nan,
                         t_envelop_at_d_star=float(at.t_envelop_median.iloc[0]) if len(at) else math.nan))
    return cells, pd.DataFrame(rows)


def safe_d_star(c):
    """(spread, rule) for one design: the smallest spread enveloping in >= COUPLING_SHARE of holds with no drone lost
    and at most BARS['contacts_per'] contacts per hold ('safe'); failing that, the best-enveloping spread within the
    contact bar ('best within contacts'); failing that, the smallest enveloping spread whatever its contacts ('envelop
    only', which the criteria will then fail on safety)."""
    ok_c = c[(c.contacts_per <= BARS["contacts_per"]) & (c.lost_per == 0)]
    safe = ok_c[ok_c.enveloped >= COUPLING_SHARE]
    if len(safe):
        return float(safe.d_block.min()), "safe"
    if len(ok_c):
        return float(ok_c.loc[ok_c.enveloped.idxmax()].d_block), "best within contacts"
    env = c[c.enveloped >= COUPLING_SHARE]
    return (float(env.d_block.min()), "envelop only") if len(env) else (float(c.d_block.max()), "none")


def predicted_d_star():
    path = blockcity.LAYOUT_DIR / "sweep~" / blockcity.OUT_SET / "feasibility.csv"
    if not path.exists():
        return {}
    f = pd.read_csv(path)
    return {(v, float(p), float(s)): round(float(g.d_star_pred_median.mean()), 3)
            for (v, p, s), g in f.groupby(["variant", "p", "s"])}


def setting_key(st):
    """The setting as a label, d_block left out: under a coupling map it differs per design, and each row reports the
    spread it flew (M3_d_block, M3b_d_block)."""
    st = json.loads(st) if isinstance(st, str) else st
    d = dict(blockcity_defaults(), **st)
    return ",".join(f"{k}={d[k]}" for k in sorted(d) if k != "d_block")


def blockcity_defaults():
    from citysim import split_probe
    return dict(split_probe.DEFAULT_SETTING)


def criteria(df):
    """Per variant x design x setting: every criterion's value and pass flag, for M1, M3, M3b (and M6)."""
    d = _ok(df[df.kind.isin(["M1", "M3", "M3b", "M6"])]).copy()
    if not len(d):
        return pd.DataFrame()
    d["skey"] = d.setting.apply(setting_key)
    out = []
    for (v, p, s, sk), g in d.groupby(["variant", "p", "s", "skey"]):
        row = dict(variant=v, p=p, s=s, setting=sk)
        for kind in ("M3", "M3b"):
            k = g[g.kind == kind]
            if not len(k):
                continue
            n = len(k)
            anyl = int((k.lost > 0).sum())
            row[f"{kind}_n"] = n
            row[f"{kind}_lost_per"] = round(float(k.lost.mean()), 4)
            row[f"{kind}_any_loss_ub"] = round(wilson_upper(anyl, n), 4)
            row[f"{kind}_contacts_per"] = round(float(k.contacts.mean()), 3)
            com = k[k.t_commit.notna()]
            row[f"{kind}_reform_ok"] = round(float(com.reform_ok.fillna(False).astype(bool).mean()), 3) if len(com) \
                else math.nan
            row[f"{kind}_reform_s_median"] = round(float(com.reform_s.median()), 2) if len(com) else math.nan
            row[f"{kind}_preview_mean"] = round(float(com.preview_search.mean()), 3) if "preview_search" in com and \
                len(com) else math.nan
            row[f"{kind}_d_block"] = round(float(k.d_block.median()), 3)
            if kind == "M3":
                row["M3_pass_ok"] = round(float((k.reached_exit.fillna(False).astype(bool) &
                                                 (k.stall_s.fillna(0) <= 3.0)).mean()), 3)
                row["M3_pass_s_median"] = round(float(k.pass_s.median()), 2)
            else:
                row["M3b_preview_ok"] = round(float((com.preview_search >= BARS["preview_streets"]).mean()), 3) \
                    if len(com) else math.nan
                tp = k.time_price.dropna() if "time_price" in k else pd.Series(dtype=float)
                row["M3b_time_price_median"] = round(float(tp.median()), 3) if len(tp) else math.nan
                row["M3b_hop_s_median"] = round(float(k.hop_s.median()), 2)
                row["M3b_street_lost_per"] = round(float(k.street_lost.mean()), 4) if "street_lost" in k else math.nan
                row["M3b_t_hop"] = float(k.t_hop.median())
        m1 = g[g.kind == "M1"]
        if len(m1):
            ratio = (m1.swarm_speed / m1.single_speed).dropna()
            row["M1_speed_ratio"] = round(float(ratio.median()), 3) if len(ratio) else math.nan
            row["M1_contacts_per_100m"] = round(float((m1.swarm_contacts / m1.route_m * 100).mean()), 4)
            row["M1_wall_feed_frac"] = round(float(m1.wall_feed_frac.mean()), 3)
            row["M1_lost_per"] = round(float(m1.swarm_lost.mean()), 4)
        m6 = g[g.kind == "M6"]
        if len(m6):
            row["M6_passed"] = round(float(m6.passed.astype(bool).mean()), 3)
            row["M6_stall_s_median"] = round(float(m6.stall_s.median()), 2)
            row["M6_lost_per"] = round(float(m6.lost.mean()), 4)
        out.append(row)
    c = pd.DataFrame(out)
    return verdicts(c)


def verdicts(c):
    b = BARS
    safe = pd.Series(True, index=c.index)
    for kind in ("M3", "M3b"):
        if f"{kind}_lost_per" in c:
            safe &= (c[f"{kind}_lost_per"] <= b["lost_per"]) & (c[f"{kind}_any_loss_ub"] <= b["any_loss_ub"]) & \
                    (c[f"{kind}_contacts_per"] <= b["contacts_per"])
    c["pass_safety"] = safe
    reform = pd.Series(True, index=c.index)
    for kind in ("M3", "M3b"):
        if f"{kind}_reform_ok" in c:
            reform &= c[f"{kind}_reform_ok"] >= b["reform"]
    c["pass_reform"] = reform
    c["pass_preview"] = (c.get("M3b_preview_ok", math.nan) >= b["preview_trials"]) & \
        (c.get("M3_pass_ok", math.nan) >= b["m3_pass"])
    c["pass_time_price"] = c.get("M3b_time_price_median", math.nan) <= b["time_price"]
    c["pass_streets"] = (c.get("M1_speed_ratio", math.nan) >= b["speed_ratio"]) & \
        (c.get("M1_contacts_per_100m", math.nan) <= b["contacts_per_100m"])
    c["pass_all"] = c[["pass_safety", "pass_reform", "pass_preview", "pass_time_price", "pass_streets"]].all(axis=1)
    return c


def t80_per_trial(g, band="search"):
    """(n,) per M3b trial: seconds after the entrance until >= 80 % of its radiating streets had been previewed (inf if
    never in the flight)."""
    col = f"preview_times_{band}"
    t80 = []
    for _, x in g.iterrows():
        times = list(x[col]) if isinstance(x[col], list) else []
        n = len(times)
        if not n or x.t_entrance is None or (isinstance(x.t_entrance, float) and math.isnan(x.t_entrance)):
            continue
        rel = np.sort([t - x.t_entrance if t is not None else math.inf for t in times])
        t80.append(max(0.0, float(rel[max(1, int(math.ceil(BARS["preview_streets"] * n))) - 1])))
    return np.array(t80)


def t_hop_table(df):
    """From M3b flights with a long hold: for each T, the share of trials with >= 80 % of the radiating streets
    previewed by T seconds into the hold (search band) -- what committing at T would have previewed."""
    d = _ok(df[df.kind == "M3b"]).copy()
    if not len(d) or "preview_times_search" not in d:
        return pd.DataFrame()
    rows = []
    keys = ["variant", "p", "s"] + (["mode"] if "mode" in d else [])
    for kv, g in d.groupby(keys, dropna=False):
        r = dict(zip(keys, kv), n=len(g))
        t80 = t80_per_trial(g)
        for T in T_HOPS:
            r[f"T{T:g}"] = round(float((t80 <= T).mean()), 3) if len(t80) else math.nan
        good = [T for T in T_HOPS if r.get(f"T{T:g}", 0) >= BARS["preview_trials"]]
        r["t_hop"] = min(good) if good else math.nan
        # the continuous form: the smallest T meeting the bar, the 80th percentile of the trials' times
        if len(t80):
            k = max(1, int(math.ceil(BARS["preview_trials"] * len(t80)))) - 1
            r["t_hop_exact"] = round(float(np.sort(t80)[k]), 2)
        else:
            r["t_hop_exact"] = math.nan
        r["t80_median"] = round(float(np.median(t80)), 2) if len(t80) else math.nan
        r["never_share"] = round(float(np.isinf(t80).mean()), 3) if len(t80) else math.nan
        rows.append(r)
    return pd.DataFrame(rows)


def decide(crit):
    """The plan's variant rule over the designs' pass_all (best setting per design)."""
    if not len(crit):
        return {}
    best = crit.sort_values(["pass_all", "pass_safety", "pass_reform"], ascending=False).groupby(
        ["variant", "p", "s"]).head(1)
    n = best.groupby("variant").pass_all.sum().to_dict()
    nsafe = best.groupby("variant").pass_safety.sum().to_dict()
    if n.get("B", 0) >= 9:
        choice = "B"
    elif n.get("R", 0) >= 6:
        choice = "R"
    elif max(nsafe.get("B", 0), nsafe.get("R", 0)) < 4:
        choice = "stop"
    elif max(n.get("B", 0), n.get("R", 0)) == 0:
        choice = "stop (no design passes)"         # Phase 2 runs on the passing designs only: there are none
    else:
        choice = "B" if n.get("B", 0) >= n.get("R", 0) else "R"
    return dict(choice=choice, designs_passing=n, designs_safe=nsafe,
                passing=best[best.pass_all][["variant", "p", "s"]].values.tolist())


def tables(name):
    from citysim import split_probe
    out = split_probe.OUT_ROOT / name
    df = split_probe.read(name)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    cells, cmap = coupling(df)
    if len(cells):
        cells.to_csv(out / "coupling_cells.csv", index=False)
        cmap.to_csv(out / "coupling_map.csv", index=False)
        print("== coupling (M4): envelopment share by spread")
        print(cells.pivot_table(index=["variant", "p", "s"], columns="d_block", values="enveloped").round(2).to_string())
        print("\n== the surround map d* (smallest spread enveloping in >= 80 % of trials, no loss) against the prediction")
        print(cmap.to_string(index=False))
    crit = criteria(df)
    if len(crit):
        crit.to_csv(out / "criteria.csv", index=False)
        print("\n== criteria per design and setting")
        cols = [c for c in ("variant", "p", "s", "M3_lost_per", "M3b_lost_per", "M3_contacts_per", "M3b_contacts_per",
                            "M3_reform_ok", "M3b_reform_ok", "M3b_preview_ok", "M3_pass_ok", "M3b_time_price_median",
                            "M1_speed_ratio", "M1_contacts_per_100m", "pass_safety", "pass_reform", "pass_preview",
                            "pass_time_price", "pass_streets", "pass_all") if c in crit]
        for sk, g in crit.groupby("setting"):
            print(f"-- setting {sk}")
            print(g[cols].to_string(index=False))
        dec = decide(crit)
        (out / "decision.json").write_text(json.dumps(dec, indent=1, default=str), encoding="utf-8")
        print("\n== decision:", dec)
    th = t_hop_table(df)
    if len(th):
        th.to_csv(out / "t_hop.csv", index=False)
        print("\n== T_hop: share of M3b trials with >= 80 % of streets previewed by T s into the hold")
        print(th.to_string(index=False))
    errs = df[df.get("error", pd.Series(index=df.index, dtype=object)).notna()] if "error" in df else df.iloc[0:0]
    print(f"\n[split_tables] {len(df)} jobs, {len(errs)} errors -> {out}")
