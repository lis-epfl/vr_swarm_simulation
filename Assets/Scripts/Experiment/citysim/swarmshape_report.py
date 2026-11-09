"""S8 Footprint Avoidance: the results page, citysim/swarmshape_report.html filled with the runs' data.

    python -B -m citysim.swarmshape_report
        -> experiment_results/swarm_shape/swarm_shape.html (the replays inlined, to open as a file) and
           swarm_shape_bare.html (what the artifact is published from), with swarm_shape_files.json, the replay files
           to publish beside it

S4's page family (lesson 36): the question for the human trials and the answer first, the cities and flights
(swarmshape_replays), every table behind All results, the terms. Its styles are S4's page's own, read from
split_report.html at build. Every number is read from the runs here, never typed into the template.
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

from citysim import swarmshape as X, swarmtune as T  # noqa: E402
from citysim.swarmflow_report import bench_counts  # noqa: E402
from citysim.swarmtune_report import _clean, _row, s4_style, VIEW_JS  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "swarmshape_report.html"
REPLAY = X.OUT / "replay"
RUNS = Path(r"D:\claude_obstacle_bench\runs")


def row(r):
    out = _row(r)
    out["cand"] = r.get("cand")
    out["face"] = {k[5:]: v for k, v in r.items() if k.startswith("face_")}
    out["sigma"] = r.get("sigma")
    return out


def table(name):
    if not (X.OUT / name / "results.jsonl").exists():
        return None
    return [row(r) for r in X.table(name).to_dict("records")]


def by_seed(name):
    df, cands, _ = T.read(name, X.OUT)
    out = {}
    for (cid, seed), g in df.groupby(["cand", "seed"]):
        r = X.score_rows(g)
        out.setdefault(cands[cid]["label"], {})[str(seed)] = {
            k: r.get(k) for k in ("v_coh", "lost_km", "contact_km", "street_v_coh", "face_t_split", "km")}
    return out


def gate(name):
    import pandas as pd
    path = X.OUT / name / "bench" / "gate.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return [row(r) | {"set": r["set"], "src": r["src"]} for r in df.to_dict("records")]


def counts(runs):
    """bench_counts over every run of a gate (each set flew in exactly one of them)."""
    out = {}
    for run in runs:
        out.update(bench_counts(run) or {})
    return out or None


def gaps():
    import pandas as pd
    path = X.OUT / "gaps.csv"
    return pd.read_csv(path).to_dict("records") if path.exists() else None


def events():
    import pandas as pd
    path = X.OUT / "s7_events" / "summary.csv"
    return pd.read_csv(path).to_dict("records") if path.exists() else None


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def build():
    rindex = REPLAY / "index.json"
    rule = json.loads((X.OUT / "rule.json").read_text(encoding="utf-8"))
    vpath = X.OUT / "confirm" / "verdicts.json"
    verdicts = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), scene=T.SCENE, rule=rule["rule"], rule_written=rule["written"],
                verdicts=verdicts.get("verdicts", {}), pick=verdicts.get("pick"), gaps=gaps(),
                confirm=table("confirm"), confirmc=table("confirmc"), confirm_seed=by_seed("confirm"),
                screen=table("screen1"), screenc=table("screen1c"), cap=X.FACE_CAP_S,
                flights=dict(confirm=dict(cross=40, pass_=30, street=20, face=36), seeds=[T.TUNE_SEED, *T.HELD_SEEDS],
                             screen=dict(cross=48, pass_=32, street=24, face=36), screenc=60, confirmc=36),
                unity=gate("bench_shape"), unity_single=gate("bench_single"), unity_face12=gate("bench_face12"),
                counts={k: counts(runs) for k, runs in (
                    ("shape", ("swarmshape_bench_shape", "swarmshape_bench_extra")),
                    ("single", ("swarmshape_bench_single", "swarmshape_bench_single_extra")),
                    ("face12", ("swarmshape_bench_face12", "swarmshape_bench_face12_extra")))},
                events=events(),
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
    dest = X.OUT / "swarm_shape.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("swarm_shape_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("swarm_shape_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                        encoding="utf-8")
    print(f"[swarmshape_report] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} "
          f"replay files; rule pick: {data['pick']}")
    return dest


if __name__ == "__main__":
    build()
