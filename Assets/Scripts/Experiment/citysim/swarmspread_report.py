"""S9 Street Spread: the results page, citysim/swarmspread_report.html filled with the runs' data.

    python -B -m citysim.swarmspread_report
        -> experiment_results/swarm_spread/swarm_spread.html (the replays inlined, to open as a file) and
           swarm_spread_bare.html (what the artifact is published from), with swarm_spread_files.json, the replay files
           to publish beside it

S4's page family (lesson 36): the question for the human trials and the answer first, the cities and flights
(swarmspread_replays), every table behind All results, the terms. Every number is read from the runs here: the replica
run (swarmspread analyse -> analysis.json, and its table), the Unity gate, and S8's counts for the two settings both
studies flew (C35 = F20, C35a = A85), pooled as independent replicates on different routes.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from citysim import swarmshape as SS, swarmspread as X, swarmtune as T  # noqa: E402
from citysim.swarmflow_report import bench_counts  # noqa: E402
from citysim.swarmtune_report import _clean, _row, s4_style, VIEW_JS  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "swarmspread_report.html"
REPLAY = X.OUT / "replay"
RUN, RUN2 = "run1", "run2"
BENCH_RUN, BENCH_RUN2 = "swarmspread_bench_spread", "swarmspread_bench_spread2"
S8_PAIR = {"F20": "C35", "A85": "C35a"}       # the same settings in S8 (its confirmation and Unity bench)


def table(name):
    if not (X.OUT / name / "results.jsonl").exists():
        return None
    out = []
    for r in SS.table(name, root=X.OUT).to_dict("records"):
        o = _row(r)
        o["cand"], o["sigma"] = r.get("cand"), r.get("sigma")
        out.append(o)
    return out


def by_seed(name):
    df, cands, _ = T.read(name, X.OUT)
    out = {}
    for (cid, seed), g in df.groupby(["cand", "seed"]):
        r = SS.score_rows(g)
        out.setdefault(cands[cid]["label"].split()[0], {})[str(seed)] = {
            k: r.get(k) for k in ("v_coh", "street_v_coh", "lost_km", "contact_km", "km")} | dict(
            flights_lost=int((g.lost > 0).sum()))
    return out


def gate(name):
    import pandas as pd
    path = X.OUT / name / "bench" / "gate.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return [_row(r) | {"set": r["set"], "src": r["src"]} for r in df.to_dict("records")]


def s8_pool():
    """S8's counts for the settings S9 flies again: flights that lost drones and km in its confirmation, drones lost and
    flights in its Unity bench."""
    df, cands, _ = T.read("confirm", SS.OUT)
    out = {}
    for code, s8 in S8_PAIR.items():
        cid = next(c for c, v in cands.items() if v["label"].split()[0] == s8 and not c.startswith("single"))
        g = df[df.cand == cid]
        g = g[g.kind != "face"]
        out[code] = dict(s8=s8, flights_lost=int((g.lost > 0).sum()), km=float(g.L.sum() / 1000))
    for run in ("swarmshape_bench_shape", "swarmshape_bench_extra"):
        for s, c in (bench_counts(run) or {}).items():
            for code, s8 in S8_PAIR.items():
                if s == s8:
                    out[code]["unity_lost"], out[code]["unity_flights"] = c["lost"], c["flights"]
    return out


PAIRS = (("A85", "F20"), ("A90", "F20"), ("A95", "F20"), ("F18", "F20"), ("F17", "F20"), ("F17", "A85"),
         ("F18", "A85"), ("F18", "A90"))


def paired(name):
    """Route-by-route street-speed differences (every row flew the same routes): mean, 95% interval, the paired t and
    Wilcoxon tests over routes, the t test over the four cities' mean differences (the cities are the units, lesson
    13), and the share of routes the first row was faster on."""
    import numpy as np
    from scipy import stats
    df, cands, _ = T.read(name, X.OUT)
    df = df[df.kind == "street"]
    code = {c: v["label"].split()[0] for c, v in cands.items()}
    v = {}
    for r in df.itertuples():
        v.setdefault(code[r.cand], {})[(r.seed, r.k)] = r.v_coh
    out = {}
    for a, b in PAIRS:
        if a not in v or b not in v:
            continue
        keys = sorted(set(v[a]) & set(v[b]))
        d = np.array([v[a][k] - v[b][k] for k in keys])
        base = float(np.mean([v[b][k] for k in keys]))
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        seeds = sorted({k[0] for k in keys})
        city = [float(np.mean([v[a][k] - v[b][k] for k in keys if k[0] == s])) for s in seeds]
        out[f"{a}-{b}"] = dict(n=len(d), mean=float(d.mean()), pct=float(d.mean() / base), lo=float(ci[0]),
                               hi=float(ci[1]), p_t=float(stats.ttest_1samp(d, 0).pvalue),
                               p_w=float(stats.wilcoxon(d).pvalue), faster=float((d > 0).mean()), city=city,
                               p_city=float(stats.ttest_1samp(city, 0).pvalue) if len(city) > 2 else None)
    return out


def unity_losses(name):
    """Per set of a gated Unity run: flights, flights that lost drones and drones lost, Unity's and the twins'."""
    import pandas as pd
    p = X.OUT / name / "bench" / "gate.jsonl"
    if not p.exists():
        return None
    g = pd.read_json(p, lines=True)
    return {s: dict(flights=len(x), u_flights_lost=int((x.u_lost > 0).sum()), u_lost=int(x.u_lost.sum()),
                    r_flights_lost=int((x.r_lost > 0).sum()), r_lost=int(x.r_lost.sum()),
                    both=int(((x.u_lost > 0) & (x.r_lost > 0)).sum()))
            for s, x in g.groupby("set")}


def temporal(name):
    p = X.OUT / name / "temporal.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def analysis(name):
    p = X.OUT / name / "analysis.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def build():
    rindex = REPLAY / "index.json"
    rule = json.loads((X.OUT / "rule.json").read_text(encoding="utf-8"))
    ana = json.loads((X.OUT / RUN / "analysis.json").read_text(encoding="utf-8"))
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), scene=T.SCENE, rule=rule["rule"], rule_written=rule["written"],
                analysis=ana, run=table(RUN), seeds=by_seed(RUN), flights=dict(X.FLIGHTS, seeds=[T.TUNE_SEED,
                                                                                                 *T.HELD_SEEDS]),
                trans=dict(window=X.TRANS_S, ds=X.TRANS_DS), s8=s8_pool(),
                unity=gate("bench_spread"), counts=bench_counts(BENCH_RUN),
                replay=json.loads(rindex.read_text(encoding="utf-8")) if rindex.exists() else None,
                follow=dict(paired1=paired(RUN), paired2=paired(RUN2) if (X.OUT / RUN2 / "results.jsonl").exists()
                            else None, run2=analysis(RUN2), run2_table=table(RUN2), temporal=temporal(RUN2),
                            unity_temporal=json.loads((X.OUT / "unity_temporal.json").read_text(encoding="utf-8"))
                            if (X.OUT / "unity_temporal.json").exists() else None,
                            unity2=gate("bench_spread2"), counts2=bench_counts(BENCH_RUN2),
                            ulost=dict(b1=unity_losses("bench_spread"), b2=unity_losses("bench_spread2")),
                            flights2=dict(cross=25, street=250, k0=12000), window=dict(before=X.WIN_BEFORE,
                                                                                       after=X.WIN_AFTER,
                                                                                       dt=X.WIN_DT)))
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
    dest = X.OUT / "swarm_spread.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("swarm_spread_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("swarm_spread_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                         encoding="utf-8")
    print(f"[swarmspread_report] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} "
          f"replay files; rule pick: {ana.get('pick')}")
    return dest


if __name__ == "__main__":
    build()
