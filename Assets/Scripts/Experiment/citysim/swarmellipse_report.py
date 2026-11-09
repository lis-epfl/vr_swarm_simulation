"""S10 Ellipse Avoidance: the results page, citysim/swarmellipse_report.html filled with the runs' data.

    python -B -m citysim.swarmellipse_report
        -> experiment_results/swarm_ellipse/swarm_ellipse.html (the replays inlined, to open as a file) and
           swarm_ellipse_bare.html (what the artifact is published from), with swarm_ellipse_files.json, the replay
           files to publish beside it

S4's page family (lesson 36): the question and the answer first, the cities and flights (swarmellipse_replays), every
table behind All results, the terms. Its styles are S4's page's own, read from split_report.html at build. Every
number is read from the runs here, never typed into the template.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import swarmellipse as E, swarmspread as SS, swarmtune as T  # noqa: E402
from citysim.swarmflow_report import bench_counts  # noqa: E402
from citysim.swarmtune_report import _clean, _row, s4_style, VIEW_JS  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "swarmellipse_report.html"
REPLAY = E.OUT / "replay"
ORDER = ("C35a", "E15_35a", "E125_35a", "C45a", "E15_45a", "C35", "E15_35")


def code(label):
    return label.replace("one drone, ", "").split()[0]


def row(r):
    out = _row(r)
    out["cand"] = r.get("cand")
    out["d"] = r.get("d")
    for pre in ("face", "ang", "along"):
        out[pre] = {k[len(pre) + 1:]: v for k, v in r.items() if k.startswith(pre + "_")}
    out["sigma"] = r.get("sigma")
    return out


def table(name):
    if not (E.OUT / name / "results.jsonl").exists():
        return None
    return [row(r) for r in E.table(name).to_dict("records")]


def rule_rows(name):
    """rows_for_rule with each row's loss events' exact interval per km."""
    rows = E.rows_for_rule(name)
    out = {}
    for k, r in rows.items():
        lo, hi = SS.poisson_ci(r["events"])
        out[k] = {x: v for x, v in r.items() if x not in ("over", "key")} | dict(
            ev_km=r["events"] / r["km"], ev_lo=lo / r["km"], ev_hi=hi / r["km"], d_obs=r["key"][2], cap=r["key"][1],
            adaptive=r["key"][3])
    base = out.get("C35a")
    if base:
        for k, r in out.items():
            r["p_more_than_scene"] = SS.worse_p(r["events"], base["events"], r["km"], base["km"])
            r["p_fewer_than_scene"] = SS.worse_p(base["events"], r["events"], base["km"], r["km"])
    return out


def by_city(name):
    """{code: {seed: dict(street, events, km, along_spread)}}: the cities are the units."""
    df, cands, _ = T.read(name, E.OUT)
    out = {}
    for cid, g in df.groupby("cand"):
        c = code(cands[cid]["label"]) + ("@1.2" if float(cands[cid]["d"]) == 1.2 else "")
        single = cid.startswith("single")
        for seed, gg in g.groupby("seed"):
            d = out.setdefault(("1:" if single else "") + c, {}).setdefault(str(seed), {})
            if single:
                a = gg[(gg.kind == "along") & gg.alongside.astype(bool)]
                sp = [np.percentile(x.standoff, 90) - np.percentile(x.standoff, 10) for _, x in a.groupby("along_off")]
                d["along_spread"] = float(np.mean(sp)) if sp else None
            else:
                std = gg[gg.kind.isin(E.STD_KINDS)]
                d.update(street=float(gg[gg.kind == "street"].v_coh.mean()) if (gg.kind == "street").any() else None,
                         events=int((std.lost - std.lost_drone).sum()), km=float(std.L.sum() / 1000.0))
    return out


def csv(name):
    import pandas as pd
    p = E.OUT / name
    return pd.read_csv(p).to_dict("records") if p.exists() else None


def gate(name):
    import pandas as pd
    p = E.OUT / name / "bench" / "gate.csv"
    if not p.exists():
        return None
    return [row(r) | {"set": r["set"], "src": r["src"]} for r in pd.read_csv(p).to_dict("records")]


def example():
    """One of the scene's buildings drawn with its circle and its ellipse (cap 1.5) and each wall's distance to each:
    the building whose ellipse is nearest the scene's p90 aspect, among those with four to six walls."""
    rc = T.city(T.TUNE_SEED)["rcity"]
    sh = E.shapes(rc, (1.5,))
    Ec, Ee = sh["enclosing circle"], sh["ellipse 1.5"]
    asp = np.array([k for _, k in rc._mvee])
    target = float(np.percentile(asp, 90))
    cand = [i for i in range(rc.n) if 4 <= len(E.walls(rc, i)) <= 6]
    b = min(cand, key=lambda i: abs(asp[i] - target))
    wd_c, wd_e = E.wall_distances(rc, Ec)[b], E.wall_distances(rc, Ee)[b]
    walls = [dict(mid=[float(x) for x in w[3]], n=[float(x) for x in w[5]], L=w[2], circle=wd_c[j], ellipse=wd_e[j])
             for j, w in enumerate(E.walls(rc, b))]
    spread = {}
    for name, S in (("circle", Ec), ("ellipse", Ee)):
        wd = E.wall_distances(rc, S)
        sp = np.array([max(w) - min(w) for w in wd if w])
        spread[name] = dict(hist=np.histogram(sp, bins=np.arange(0, 15.5, 1.0))[0].tolist(), median=float(np.median(sp)),
                            p90=float(np.percentile(sp, 90)), far6=int(sum(1 for w in wd if w and max(w) > 6.0)))
    return dict(building=int(b), aspect=float(asp[b]), poly=rc.polys[b].tolist(),
                circle=dict(c=Ec.c[b].tolist(), r=float(Ec.a[b])),
                ellipse=dict(c=Ee.c[b].tolist(), e=Ee.e[b].tolist(), a=float(Ee.a[b]), b=float(Ee.b[b])),
                walls=walls, spread=spread, n=rc.n)


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def build():
    rindex = REPLAY / "index.json"
    rule = json.loads((E.OUT / "rule.json").read_text(encoding="utf-8"))
    vpath = E.OUT / "confirm" / "verdicts.json"
    verdicts = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), scene=T.SCENE, rule=rule["rule"], rule_written=rule["written"],
                verdicts=verdicts.get("verdicts", {}), pick=verdicts.get("pick"),
                corridors=verdicts.get("corridors", {}), order=list(ORDER),
                walls=csv("walls.csv"), gaps=csv("gaps.csv"), example=example(),
                screen=table("screen1"), confirm=table("confirm"), rows=rule_rows("confirm"),
                screen_rows=rule_rows("screen1"), city=by_city("confirm"),
                seeds=[T.TUNE_SEED, *T.HELD_SEEDS], cap=E.SH.FACE_CAP_S,
                flights=dict(confirm=dict(cross=40, pass_=30, street=200, face=36, angle=36, along=180, single_angle=36),
                             screen=dict(cross=48, pass_=32, street=24, face=36, angle=36, along=120, single_angle=36)),
                unity=gate("bench1"), unity12=gate("bench12"), unity1=gate("bench1d"),
                counts={k: bench_counts(f"swarmellipse_{k}") for k in ("bench1", "bench12", "bench1d")},
                replay=json.loads(rindex.read_text(encoding="utf-8")) if rindex.exists() else None)
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
    dest = E.OUT / "swarm_ellipse.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("swarm_ellipse_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("swarm_ellipse_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                          encoding="utf-8")
    print(f"[swarmellipse_report] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} "
          f"replay files; rule pick: {data['pick']}")
    return dest


if __name__ == "__main__":
    build()
