"""S9's settling run: does a narrower street spread cost drones in Unity, as the replica's doubling says?

The replica loses about twice as many drones at 0.85 as at 2.0 held; 156 Unity flights a row showed no difference and
cannot exclude one. bench_spread3 flies the two rows on 300 fresh street routes each. Fixed before it flew (rule):
judged on the rate ratio of Unity's loss events (A85 / F20, a colliding pair is one event) with its exact 95% interval,
over the street routes of all three benches.

Commands (from Assets/Scripts/Experiment, the stitching env; parent-only):
  rule       settle_rule.json, written before the run
  analyse    -> settle.json
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import swarmspread as X  # noqa: E402

FRESH, EARLIER = "bench_spread3", ("bench_spread", "bench_spread2")
NARROW, BASE = "A85", "F20"
# Replica run 2's ratio, the claim on trial.
REPLICA_RATIO = 2.0
RULE = dict(question="Does the narrower street spread cost drones in Unity, as the replica's doubling says?",
            narrow=NARROW, base=BASE, fresh=FRESH, earlier=list(EARLIER), kind="street", conf=0.95,
            measure="loss events per km: drones lost, less the partner of each collision (ExternallyKilled)",
            judged_on="all three benches' street routes; the fresh bench alone is reported beside it",
            replica_ratio=REPLICA_RATIO,
            verdicts=dict(shows_it="the interval's lower bound is above 1: narrower costs drones in Unity too",
                          rules_out_doubling="the upper bound is below replica_ratio and the lower is not above 1",
                          unsettled="the interval holds both 1 and replica_ratio"))


def gate(name):
    p = X.OUT / name / "bench" / "gate.jsonl"
    g = pd.read_json(p, lines=True)
    g = g[g.kind == RULE["kind"]].copy()
    g["u_events"] = [sum(v for r, v in (d or {}).items() if r != "ExternallyKilled") for d in g.u_deaths]
    # The twin is scored as one pair per two drones unless it died alone: only its loss flights are comparable.
    g["bench"] = name
    return g


def rate_ratio(k, k0, e, e0, conf=0.95):
    """Exact interval on the rate ratio (k over exposure e / k0 over e0): Clopper-Pearson on k given k + k0, mapped
    through p = r e / (r e + e0)."""
    from scipy.stats import beta
    n, a = k + k0, 1 - conf
    if n == 0:
        return dict(ratio=None, lo=0.0, hi=None)
    lo_p = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi_p = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    to_r = lambda p: None if p >= 1 else p / (1 - p) * e0 / e
    return dict(ratio=None if k0 == 0 else (k / e) / (k0 / e0), lo=to_r(lo_p), hi=to_r(hi_p))


def two_sided_p(k, k0, e, e0):
    from scipy.stats import binomtest
    n = k + k0
    return 1.0 if n == 0 else float(binomtest(k, n, e / (e + e0)).pvalue)


def verdict(rr):
    if rr["lo"] is not None and rr["lo"] > 1:
        return "shows_it"
    if rr["hi"] is not None and rr["hi"] < REPLICA_RATIO:
        return "rules_out_doubling"
    return "unsettled"


def compare(g):
    """A85 against F20 over the rows of g, which flew the same routes."""
    from scipy import stats
    a, b = g[g.set == NARROW], g[g.set == BASE]
    row = {}
    for s, x in ((NARROW, a), (BASE, b)):
        km = float(x.u_L.sum()) / 1000
        lo, hi = X.poisson_ci(int(x.u_events.sum()))
        row[s] = dict(flights=len(x), km=km, events=int(x.u_events.sum()), events_lo=lo, events_hi=hi,
                      flights_lost=int((x.u_lost > 0).sum()), drones=int(x.u_lost.sum()),
                      contacts=int(x.u_contacts.sum()), clean=int(((x.u_lost == 0) & (x.u_contacts == 0)).sum()),
                      v_coh=float(x.u_v_coh.mean()), one_frac=float(x.u_one_frac.mean()),
                      min_pair=float(x.u_min_pair.median()),
                      deaths={r: int(sum((d or {}).get(r, 0) for d in x.u_deaths))
                              for r in sorted({r for d in x.u_deaths for r in (d or {})})},
                      twin_flights_lost=int((x.r_lost > 0).sum()), twin_drones=int(x.r_lost.sum()),
                      both=int(((x.u_lost > 0) & (x.r_lost > 0)).sum()),
                      twin_v_coh=float(x.r_v_coh.mean()), twin_contacts=int(x.r_contacts.sum()))
    ka, kb, ea, eb = row[NARROW]["events"], row[BASE]["events"], row[NARROW]["km"], row[BASE]["km"]
    rr = rate_ratio(ka, kb, ea, eb, RULE["conf"])
    ta, tb = row[NARROW]["twin_flights_lost"], row[BASE]["twin_flights_lost"]
    ca, cb = row[NARROW]["contacts"], row[BASE]["contacts"]
    # Paired over routes: every row flies the same (bench, k).
    key = ["bench", "k"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    d = (m.u_v_coh_a - m.u_v_coh_b).to_numpy()
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    dc = (m.u_contacts_a - m.u_contacts_b).to_numpy()
    lost_a, lost_b = m.u_lost_a > 0, m.u_lost_b > 0
    return dict(rows=row, loss=dict(rr, p=two_sided_p(ka, kb, ea, eb), verdict=verdict(rr)),
                twins=dict(rate_ratio(ta, tb, ea, eb, RULE["conf"]), p=two_sided_p(ta, tb, ea, eb)),
                contacts=dict(rate_ratio(ca, cb, ea, eb, RULE["conf"]), p=two_sided_p(ca, cb, ea, eb),
                              paired_p=float(stats.wilcoxon(dc).pvalue) if np.any(dc) else 1.0),
                speed=dict(n=len(d), mean=float(d.mean()), pct=float(d.mean() / m.u_v_coh_b.mean()),
                           lo=float(ci[0]), hi=float(ci[1]), p=float(stats.ttest_1samp(d, 0).pvalue),
                           p_w=float(stats.wilcoxon(d).pvalue), faster=float((d > 0).mean()),
                           twin_pct=float((m.r_v_coh_a - m.r_v_coh_b).mean() / m.r_v_coh_b.mean())),
                discordant=dict(narrow_only=int((lost_a & ~lost_b).sum()), base_only=int((lost_b & ~lost_a).sum()),
                                both=int((lost_a & lost_b).sum())))


def analyse():
    fresh = gate(FRESH)
    pool = pd.concat([gate(n)[lambda g: g.set.isin([NARROW, BASE])] for n in (*EARLIER, FRESH)], ignore_index=True)
    out = dict(rule=RULE, fresh=compare(fresh), pooled=compare(pool),
               by_bench={n: compare(pool[pool.bench == n]) for n in (*EARLIER, FRESH)})
    # Where the replica's doubling lies: its twins of Unity's flights (Unity's starts, the scene's city) against its own
    # flights of the follow-up (its own settled starts, four cities), flights that lost drones per km, row by row.
    own = json.loads((X.OUT / "run2" / "analysis.json").read_text(encoding="utf-8"))["rows"]
    out["twin_vs_own"] = {}
    for c in (NARROW, BASE):
        t, o = out["pooled"]["rows"][c], own[c]
        out["twin_vs_own"][c] = dict(rate_ratio(t["twin_flights_lost"], o["flights_lost"], t["km"], o["km"], RULE["conf"]),
                                     p=two_sided_p(t["twin_flights_lost"], o["flights_lost"], t["km"], o["km"]),
                                     twin=t["twin_flights_lost"], twin_km=t["km"], own=o["flights_lost"], own_km=o["km"])
    (X.OUT / "settle.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name in ("fresh", "pooled"):
        c = out[name]
        print(f"[settle] {name}: "
              + "  ".join(f"{s} {r['events']} events {r['km']:.0f} km, {r['flights_lost']}/{r['flights']} flights lost, "
                          f"{r['contacts']} contacts" for s, r in c["rows"].items()))
        lo, hi = c["loss"]["lo"], c["loss"]["hi"]
        print(f"[settle]   loss ratio {c['loss']['ratio']:.2f} ({lo:.2f} to {hi:.2f}), p {c['loss']['p']:.3f}: "
              f"{c['loss']['verdict']}")
        print(f"[settle]   contact ratio {c['contacts']['ratio']:.2f} ({c['contacts']['lo']:.2f} to "
              f"{c['contacts']['hi']:.2f}), speed {c['speed']['pct']:+.1%} (twin {c['speed']['twin_pct']:+.1%}), "
              f"p {c['speed']['p']:.2g}, faster on {c['speed']['faster']:.0%}")


def cmd_rule(_):
    p = X.OUT / "settle_rule.json"
    if p.exists():
        raise SystemExit(f"[settle] {p} exists: the rule is fixed")
    p.write_text(json.dumps(dict(RULE, written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1), encoding="utf-8")
    print(f"[settle] {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rule")
    sub.add_parser("analyse")
    a = ap.parse_args()
    dict(rule=cmd_rule, analyse=lambda _: analyse())[a.cmd](a)


if __name__ == "__main__":
    main()
