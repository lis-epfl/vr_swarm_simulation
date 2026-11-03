"""S9 Street Spread, a second reading: the study's flights re-analysed from the per-flight records, and its page.

    python -B -m citysim.swarmspread_review
        -> experiment_results/swarm_spread/review/review.json (every number the page shows),
           swarm_spread_review.html (the replays inlined, to open as a file) and swarm_spread_review_bare.html (what the
           artifact is published from), with swarm_spread_review_files.json, the replay files to publish beside it

Nothing is flown. The inputs are the replica runs (run1, run2: results.jsonl), the three Unity gates
(bench_spread*/bench/gate.jsonl: Unity and its twin per flight) and the Unity benches' own flight records and
trajectories on the bench copy (results.jsonl for the close-pair and view measures, traj_*.csv for when each loss
happened). The study's summaries (analysis.json, settle.json) are not read: the point is to check them. The one
exception is its phase tests (temporal.json, unity_temporal.json), quoted for an omnibus test S9 computed and did not
report. The replays are S9's own (swarmspread_replays), reused as they are.

What it adds to S9's own analysis: progress speed and the one-group share reported apart (S6's cohesive speed is their
product); the near-miss and close-pair measures Unity records on every flight; the fresh bench apart from the benches
seen before the settling rule was written; the replica and Unity loss ratios set side by side with a heterogeneity test;
losses split by the time into the flight; and the power of a further Unity run.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from citysim import REPO  # noqa: E402
from citysim.swarmtune_report import _clean, s4_style, VIEW_JS  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "swarmspread_review.html"
OUT = REPO / "experiment_results" / "swarm_spread"
DEST = OUT / "review"
REPLAY = OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")
BENCHES = ("bench_spread", "bench_spread2", "bench_spread3")
FRESH = "bench_spread3"              # flown after settle_rule.json was written; the other two were seen before it
ORIGINAL_PAGE = "https://claude.ai/artifact/M6fGKgM1WvMVw3mRSusMwK"
ROWS2 = ("F20", "A95", "A90", "A85", "F18", "F17")
ROWS1 = ("F20", "A95", "A90", "A85", "F19", "F18", "F17", "F14", "L85", "L95", "F12", "F26", "H85")
NEAR_M = 0.75                        # a near miss: the closest pair came within this (the kill distance is 0.5 m)
EARLY_S = 15.0                       # a loss this soon after the settle counts as at the start
BENCH_MIN_PER_FLIGHT = 50 / 600      # bench_spread3: 600 flights in about 50 minutes (docs/swarm_spread §9)
SEED = 20261102


# ---------------------------------------------------------------- statistics

def poisson_ci(k, conf=0.95):
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.chi2.ppf(a / 2, 2 * k) / 2
    return float(lo), float(stats.chi2.ppf(1 - a / 2, 2 * k + 2) / 2)


def rate_ratio(k, k0, e=1.0, e0=1.0, conf=0.95):
    """Exact interval on (k / e) / (k0 / e0): Clopper-Pearson on k given k + k0, and the two-sided exact p."""
    k, k0 = int(k), int(k0)
    n, a = k + k0, 1 - conf
    lo_p = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi_p = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    to_r = lambda p: None if p >= 1 else p / (1 - p) * e0 / e
    p = float(stats.binomtest(k, n, e / (e + e0)).pvalue) if n else 1.0
    return dict(k=k, k0=k0, ratio=(k / e) / (k0 / e0) if k0 else None, lo=to_r(lo_p), hi=to_r(hi_p), p=p)


def paired_pct(a, b):
    """Route-by-route difference a - b as a share of b's mean, with its 95% interval and the paired t test."""
    d = np.asarray(a, float) - np.asarray(b, float)
    base = float(np.mean(b))
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return dict(n=len(d), pct=float(d.mean() / base), lo=float(lo / base), hi=float(hi / base),
                p=float(stats.ttest_1samp(d, 0).pvalue), higher=float(np.mean(d > 0)), diff=float(d.mean()))


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    return float(stats.wilcoxon(d).pvalue) if len(d) else 1.0


# ---------------------------------------------------------------- the replica

def load_run(name):
    cands = json.loads((OUT / name / "cands.json").read_text(encoding="utf-8"))
    code = {k: v["label"].split()[0] for k, v in cands.items()}
    label = {code[k]: v["label"] for k, v in cands.items()}
    df = pd.DataFrame([json.loads(l) for l in open(OUT / name / "results.jsonl", encoding="utf-8")])
    df["row"] = df.cand.map(code)
    df["events"] = df.loss_events.map(lambda e: len(e) if isinstance(e, list) else 0)
    df["km"] = df.L / 1000
    return df, label


def replica_rows(df, label, rows):
    out = {}
    for c in rows:
        x = df[df.row == c]
        if not len(x):
            continue
        st, cr = x[x.kind == "street"], x[x.kind.isin(["cross", "pass"])]
        k = int(x.events.sum())
        out[c] = dict(label=label[c], n=len(x), km=float(x.km.sum()), events=k, ci=poisson_ci(k),
                      flights_lost=int((x.lost > 0).sum()), drones=int(x.lost.sum()),
                      contacts=int(x.contacts.sum()), contact_km=float(x.contacts.sum() / x.km.sum()),
                      st_vcoh=float(st.v_coh.mean()), st_vprog=float(st.v_prog.mean()), st_one=float(st.one_frac.mean()),
                      near=float((x.min_pair < NEAR_M).mean()), hull=float(x.hull_frac.mean()),
                      wrap=float(cr.wrapped.sum() / cr.met.sum()) if cr.met.sum() else None,
                      cov=float(cr.coverage.mean()) if len(cr) else None,
                      by_seed={str(s): int(g.events.sum()) for s, g in x.groupby("seed")})
    return out


def replica_paired(df, pairs):
    st = df[df.kind == "street"]
    key = ["seed", "k"]
    out = {}
    for a, b in pairs:
        m = st[st.row == a].merge(st[st.row == b], on=key, suffixes=("_a", "_b"))
        out[f"{a}-{b}"] = dict(vprog=paired_pct(m.v_prog_a, m.v_prog_b), vcoh=paired_pct(m.v_coh_a, m.v_coh_b),
                               one=float((m.one_frac_a - m.one_frac_b).mean()),
                               one_p=float(stats.ttest_1samp(m.one_frac_a - m.one_frac_b, 0).pvalue))
    return out


def replica_times(df, rows):
    """Loss times on the street routes (s from the end of the settle)."""
    out = {}
    for c in rows:
        t = [e["t"] for es in df[(df.row == c) & (df.kind == "street")].loss_events for e in (es or [])]
        out[c] = sorted(round(float(v), 1) for v in t)
    return out


# ---------------------------------------------------------------- Unity

def load_unity():
    """Street routes of all three benches: the gate's Unity and twin columns, joined to Unity's own flight record."""
    gs = []
    for b in BENCHES:
        g = pd.read_json(OUT / b / "bench" / "gate.jsonl", lines=True)
        g["bench"] = b
        gs.append(g)
    G = pd.concat(gs, ignore_index=True)
    G = G[G.kind == "street"].copy()
    deaths = lambda r: lambda d: int((d or {}).get(r, 0))
    G["u_events"] = [sum(v for r, v in (d or {}).items() if r != "ExternallyKilled") for d in G.u_deaths]
    G["u_coll"] = G.u_deaths.map(deaths("CollidingWithAnotherDrone"))
    G["u_crashed"] = G.u_deaths.map(deaths("Crashed"))
    G["km"] = G.u_L / 1000
    rec = []
    for b in BENCHES:
        for l in open(RUNS / f"swarmspread_{b}" / "results.jsonl", encoding="utf-8"):
            d = json.loads(l)
            if d.get("event") == "flight" and d["kind"].startswith("street_"):
                rec.append(dict(bench=b, set=d["set"], k=int(d["kind"].split("_")[1]), scenario=d["scenario"],
                                near_ticks=d["near_ticks"], hull=d["hull_frac"], split=d["split_frac"],
                                nn=d["nn_mean"], look=d.get("live_shown_mean"), blind=d.get("live_shown_blind_frac")))
    R = pd.DataFrame(rec)
    G = G.merge(R, on=["bench", "set", "k"], how="left", validate="one_to_one")
    assert G.near_ticks.notna().all(), "a gated flight has no Unity flight record"
    return G


def unity_times(G):
    """Each Unity loss's time from the trajectory's alive mask (s from the end of the settle), pairs merged."""
    want = {(r.bench, r.set, r.scenario) for r in G.itertuples()}
    out = {}
    for b in BENCHES:
        for f in glob.glob(str(RUNS / f"swarmspread_{b}" / "traj_*.csv")):
            m = re.match(r"traj_(\w+?)_(\d+)\.csv", os.path.basename(f))
            s, i = m.group(1), int(m.group(2))
            if (b, s, i) not in want:
                continue
            t = pd.read_csv(f, usecols=["t", "alive_mask"])
            alive, tt, last = t.alive_mask.to_numpy(), t.t.to_numpy(), -1e9
            for j in range(1, len(alive)):
                if int(alive[j - 1]) & ~int(alive[j]):
                    if tt[j] - last > 0.3:
                        out.setdefault(s, []).append(round(float(tt[j]), 1))
                    last = tt[j]
    return {s: sorted(v) for s, v in out.items()}


def unity_rows(G):
    out = {}
    for s, x in G.groupby("set"):
        k = int(x.u_events.sum())
        out[s] = dict(n=len(x), km=float(x.km.sum()), events=k, ci=poisson_ci(k), coll=int(x.u_coll.sum()),
                      crashed=int(x.u_crashed.sum()), flights_lost=int((x.u_lost > 0).sum()),
                      contacts=int(x.u_contacts.sum()), contact_km=float(x.u_contacts.sum() / x.km.sum()),
                      vprog=float(x.u_v_prog.mean()), vcoh=float(x.u_v_coh.mean()), one=float(x.u_one_frac.mean()),
                      near=float((x.u_min_pair < NEAR_M).mean()), min_pair=float(x.u_min_pair.median()),
                      near_ticks=float(x.near_ticks.mean()), hull=float(x.hull.mean()), split=float(x.split.mean()),
                      look=float(x.look.mean()), blind=float(x.blind.mean()),
                      twin_flights_lost=int((x.r_lost > 0).sum()), twin_contacts=int(x.r_contacts.sum()),
                      twin_near=float((x.r_min_pair < NEAR_M).mean()), both_lost=int(((x.u_lost > 0) & (x.r_lost > 0)).sum()),
                      benches=sorted(x.bench.unique().tolist()))
    return out


def unity_paired(G, a, b, benches=BENCHES):
    g = G[G.bench.isin(benches)]
    m = g[g.set == a].merge(g[g.set == b], on=["bench", "k"], suffixes=("_a", "_b"))
    rng = np.random.default_rng(SEED)
    ca, cb = m.u_contacts_a.to_numpy(), m.u_contacts_b.to_numpy()
    boot = []
    for _ in range(5000):
        i = rng.integers(0, len(m), len(m))
        boot.append(ca[i].sum() / max(1, cb[i].sum()))
    na, nb = m.u_min_pair_a < NEAR_M, m.u_min_pair_b < NEAR_M
    n10, n01 = int((na & ~nb).sum()), int((nb & ~na).sum())
    return dict(n=len(m), vprog=paired_pct(m.u_v_prog_a, m.u_v_prog_b), vcoh=paired_pct(m.u_v_coh_a, m.u_v_coh_b),
                one=float((m.u_one_frac_a - m.u_one_frac_b).mean()),
                twin_vprog=paired_pct(m.r_v_prog_a, m.r_v_prog_b),
                contacts=dict(ratio=float(ca.sum() / cb.sum()), lo=float(np.percentile(boot, 2.5)),
                              hi=float(np.percentile(boot, 97.5)), p=wilcoxon_p(ca - cb),
                              flights_a=int((ca > 0).sum()), flights_b=int((cb > 0).sum())),
                min_pair=dict(diff=float((m.u_min_pair_a - m.u_min_pair_b).median()),
                              p=wilcoxon_p(m.u_min_pair_a - m.u_min_pair_b)),
                near_ticks=dict(ratio=float(m.near_ticks_a.mean() / m.near_ticks_b.mean()),
                                p=wilcoxon_p(m.near_ticks_a - m.near_ticks_b)),
                near=dict(a_only=n10, b_only=n01, both=int((na & nb).sum()),
                          p=float(stats.binomtest(n10, n10 + n01).pvalue) if n10 + n01 else 1.0),
                hull=dict(diff=float((m.hull_a - m.hull_b).mean()), p=wilcoxon_p(m.hull_a - m.hull_b)),
                split=dict(diff=float((m.split_a - m.split_b).mean()), p=wilcoxon_p(m.split_a - m.split_b)),
                look=dict(diff=float((m.look_a - m.look_b).mean()), p=wilcoxon_p(m.look_a - m.look_b)))


def unity_ratio(G, a, b, benches=BENCHES, col="u_events"):
    g = G[G.bench.isin(benches)]
    x, y = g[g.set == a], g[g.set == b]
    if col == "flights":
        return rate_ratio((x.u_lost > 0).sum(), (y.u_lost > 0).sum(), x.km.sum(), y.km.sum())
    if col == "twin":
        return rate_ratio((x.r_lost > 0).sum(), (y.r_lost > 0).sum(), x.km.sum(), y.km.sum())
    return rate_ratio(x[col].sum(), y[col].sum(), x.km.sum(), y.km.sum())


# ---------------------------------------------------------------- power of a further run

def power(rate, km_per_route, ratios=(1.3, 1.5, 2.0), routes=(440, 1000, 1500, 2000, 3000)):
    """Exact, not simulated: per arm, events ~ Poisson(rate x km). Given N events in both arms the narrow arm's count is
    Binomial(N, R / (1 + R)). 'detect': the two-sided exact test at 0.05 rejects equal rates when the truth is R.
    'exclude': when the truth is 1, the 95% interval's upper bound falls below R."""
    out = []
    for R in ratios:
        for n in routes:
            mu = rate * km_per_route * n
            Ns = np.arange(0, int(mu * (1 + R) + 12 * np.sqrt(mu * (1 + R)) + 20))
            det = exc = 0.0
            for N in Ns:
                if N == 0:
                    continue
                ks = np.arange(N + 1)
                # the exact two-sided test at 1/2 is symmetric: p = 2 P(K <= min(k, N - k))
                rej = 2 * stats.binom.cdf(np.minimum(ks, N - ks), N, 0.5) < 0.05
                with np.errstate(divide="ignore"):
                    ph = stats.beta.ppf(0.975, ks + 1, np.maximum(N - ks, 1e-12))
                    hi = np.where(ks == N, np.inf, ph / (1 - ph))
                det += stats.poisson.pmf(N, mu * (1 + R)) * stats.binom.pmf(ks, N, R / (1 + R))[rej].sum()
                exc += stats.poisson.pmf(N, 2 * mu) * stats.binom.pmf(ks, N, 0.5)[hi < R].sum()
            out.append(dict(R=R, routes=n, km=n * km_per_route, detect=float(det), exclude=float(exc)))
    return out


# ---------------------------------------------------------------- build

def analyse():
    r1, lab1 = load_run("run1")
    r2, lab2 = load_run("run2")
    rep2 = replica_rows(r2, lab2, ROWS2)
    rep1 = replica_rows(r1, lab1, ROWS1)
    ratios2 = {c: rate_ratio(rep2[c]["events"], rep2["F20"]["events"], rep2[c]["km"], rep2["F20"]["km"])
               for c in ROWS2 if c != "F20"}
    ratios2["F17-A85"] = rate_ratio(rep2["F17"]["events"], rep2["A85"]["events"])
    pairs = [(c, "F20") for c in ROWS2 if c != "F20"] + [("F17", "A85"), ("F18", "A85")]
    rtimes = replica_times(r2, ("F20", "A85", "F17"))
    scene_city = {c: dict(n=int(((r2.row == c) & (r2.kind == "street") & (r2.seed == 1)).sum()),
                          flights_lost=int(((r2.row == c) & (r2.kind == "street") & (r2.seed == 1) & (r2.lost > 0)).sum()))
                  for c in ("F20", "A85", "F17")}

    G = load_unity()
    urows = unity_rows(G)
    utimes = unity_times(G)
    earlier = tuple(b for b in BENCHES if b != FRESH)
    uni = dict(rows=urows, paired=unity_paired(G, "A85", "F20"),
               paired_f17=unity_paired(G, "F17", "A85", earlier), paired_f17_f20=unity_paired(G, "F17", "F20", earlier),
               ratio=unity_ratio(G, "A85", "F20"), ratio_fresh=unity_ratio(G, "A85", "F20", (FRESH,)),
               ratio_earlier=unity_ratio(G, "A85", "F20", earlier), ratio_coll=unity_ratio(G, "A85", "F20", col="u_coll"),
               ratio_flights=unity_ratio(G, "A85", "F20", col="flights"), ratio_twin=unity_ratio(G, "A85", "F20", col="twin"),
               ratio_f17=unity_ratio(G, "F17", "F20", earlier), ratio_f17_a85=unity_ratio(G, "F17", "A85", earlier),
               f17_benches=list(earlier), fresh=FRESH,
               by_bench={b: {s: int(G[(G.bench == b) & (G.set == s)].u_events.sum()) for s in ("F20", "A85", "F17")
                             if ((G.bench == b) & (G.set == s)).any()} for b in BENCHES})

    def late(times, a, b, cut=EARLY_S):
        return rate_ratio(sum(t >= cut for t in times[a]), sum(t >= cut for t in times[b]))

    timing = dict(cut=EARLY_S, unity={s: dict(early=sum(t < EARLY_S for t in v), late=sum(t >= EARLY_S for t in v),
                                              first10=sum(t < 10 for t in v), n=len(v), median=float(np.median(v)))
                                      for s, v in utimes.items()},
                  replica={s: dict(early=sum(t < EARLY_S for t in v), late=sum(t >= EARLY_S for t in v),
                                   first10=sum(t < 10 for t in v), n=len(v), median=float(np.median(v)))
                           for s, v in rtimes.items()},
                  unity_late=late(utimes, "A85", "F20"), replica_late=late(rtimes, "A85", "F20"))
    assert all(timing["unity"][s]["n"] == urows[s]["events"] for s in timing["unity"]), "loss times do not match counts"

    # The replica's own flights against Unity: log rate ratios side by side, and their fixed-effect combination.
    def lr(k, k0, e, e0):
        return float(np.log((k / e) / (k0 / e0))), 1 / k + 1 / k0
    l1, v1 = lr(rep2["A85"]["events"], rep2["F20"]["events"], rep2["A85"]["km"], rep2["F20"]["km"])
    l2, v2 = lr(urows["A85"]["events"], urows["F20"]["events"], urows["A85"]["km"], urows["F20"]["km"])
    w1, w2 = 1 / v1, 1 / v2
    lc, sc = (w1 * l1 + w2 * l2) / (w1 + w2), (w1 + w2) ** -0.5
    combined = dict(ratio=float(np.exp(lc)), lo=float(np.exp(lc - 1.96 * sc)), hi=float(np.exp(lc + 1.96 * sc)),
                    q_p=float(stats.chi2.sf((l1 - l2) ** 2 / (v1 + v2), 1)),
                    fisher_p=float(stats.fisher_exact([[rep2["A85"]["events"], rep2["F20"]["events"]],
                                                       [urows["A85"]["events"], urows["F20"]["events"]]]).pvalue))
    twin_own = {c: dict(own=scene_city[c]["flights_lost"], own_n=scene_city[c]["n"],
                        twin=urows[c]["twin_flights_lost"], twin_n=urows[c]["n"],
                        test=rate_ratio(urows[c]["twin_flights_lost"], scene_city[c]["flights_lost"],
                                        urows[c]["n"], scene_city[c]["n"])) for c in ("F20", "A85")}

    f20 = urows["F20"]
    km_route = f20["km"] / f20["n"]
    pw = power(f20["events"] / f20["km"], km_route)

    # The one place S9's own analysis is read: its phase tests (swarmspread_temporal), for the omnibus test it
    # computed beside the narrowing-against-the-rest contrast it reported.
    def phase(path):
        c = json.loads(path.read_text(encoding="utf-8"))["compare"]["A85>F20"]
        return dict(omnibus_p=c["homogeneity"]["p"], contrast_p=c["narrowing_vs_rest"]["p"],
                    wide=dict(a=c["per_phase"]["wide"]["ka"], b=c["per_phase"]["wide"]["kb"]),
                    narrowing=dict(a=c["per_phase"]["narrowing"]["ka"], b=c["per_phase"]["narrowing"]["kb"]))
    phases = dict(replica=phase(OUT / "run2" / "temporal.json"), unity=phase(OUT / "unity_temporal.json"))
    return dict(built=time.strftime("%Y-%m-%d %H:%M"), original=ORIGINAL_PAGE, near_m=NEAR_M,
                replica=dict(rows=rep2, rows1=rep1, ratios=ratios2, paired=replica_paired(r2, pairs),
                             flights=dict(n=int(len(r2)), street=int((r2.kind == "street").sum()),
                                          cross=int((r2.kind == "cross").sum()), n1=int(len(r1)))),
                unity=uni, timing=timing, combined=combined, twin_own=twin_own, phases=phases,
                power=dict(rows=pw, rate=f20["events"] / f20["km"], km_route=km_route,
                           min_per_flight=BENCH_MIN_PER_FLIGHT))


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    data = analyse()
    (DEST / "review.json").write_text(json.dumps(_clean(data), indent=1), encoding="utf-8")
    rindex = REPLAY / "index.json"
    data["replay"] = json.loads(rindex.read_text(encoding="utf-8")) if rindex.exists() else None
    text = json.dumps(_clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    for ph in ("__DATA__", "/*__REPLAY_VIEW__*/", "<!--__RV_INLINE__-->", "/*__S4_STYLE__*/"):
        assert ph in html, f"the template has no {ph} placeholder"
    view = VIEW_JS.read_text(encoding="utf-8")
    html = html.replace("/*__S4_STYLE__*/", s4_style()).replace("/*__REPLAY_VIEW__*/", view).replace("__DATA__", text)
    files = replay_files()
    bare = html.replace("<!--__RV_INLINE__-->", "")
    inline = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in files.items()}
    full = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1,viewport-fit=cover\"></head><body>" +
            html.replace("<!--__RV_INLINE__-->", '<script id="rv-inline" type="application/json">' +
                         json.dumps(inline, separators=(",", ":")).replace("</", "<\\/") + "</script>") +
            "</body></html>")
    dest = DEST / "swarm_spread_review.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("swarm_spread_review_bare.html").write_text(bare, encoding="utf-8")
    # Published from root experiment_results/swarm_spread, so the page and S9's replay files keep their relative paths.
    dest.with_name("swarm_spread_review_files.json").write_text(
        json.dumps({k: str(v) for k, v in files.items()}, indent=1), encoding="utf-8")
    u, r = data["unity"], data["replica"]
    print(f"[swarmspread_review] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} "
          f"replay files; Unity A85/F20 x{u['ratio']['ratio']:.2f} ({u['ratio']['lo']:.2f}-{u['ratio']['hi']:.2f}), "
          f"replica x{r['ratios']['A85']['ratio']:.2f}, combined x{data['combined']['ratio']:.2f} "
          f"(heterogeneity p {data['combined']['q_p']:.3f})")
    return dest


if __name__ == "__main__":
    build()
