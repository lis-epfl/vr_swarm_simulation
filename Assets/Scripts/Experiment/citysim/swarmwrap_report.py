"""S11 Wrap Hold: the results page, citysim/swarmwrap_report.html filled with the runs' data.

    python -B -m citysim.swarmwrap_report
        -> experiment_results/swarm_wrap/swarm_wrap.html (the replays inlined, to open as a file) and
           swarm_wrap_bare.html (what the artifact is published from), with swarm_wrap_files.json, the replay files to
           publish beside it

S4's page family (lesson 36): the question and the answer first, the cities and flights (swarmwrap_replays), every
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from citysim import swarmspread as SS, swarmtune as T, swarmwrap as W  # noqa: E402
from citysim.swarmflow_report import bench_counts  # noqa: E402
from citysim.swarmtune_report import _clean, s4_style, VIEW_JS  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "swarmwrap_report.html"
REPLAY = W.OUT / "replay"
ORDER = ("E15_35a", "H15_35a", "L15_35a", "F20")


def code(label):
    return str(label).split()[0]


def rows(name):
    """{code: the run's table row}, each with its loss events' exact interval per hour of flight and the one-sided p
    that it loses more often than the scene, and its loss events per phase with their exposure."""
    if not (W.OUT / name / "results.jsonl").exists():
        return None
    tab = W.table(name)
    out = {}
    for r in tab.to_dict("records"):
        c = code(r["label"])
        lo, hi = SS.poisson_ci(int(r["all_loss_events"]))
        out[c] = {k: v for k, v in r.items() if not k.startswith("o:")} | dict(
            code=c, ev_ph=r["all_loss_ph"], ev_lo=lo / r["all_h"], ev_hi=hi / r["all_h"],
            over={k[2:]: v for k, v in r.items() if k.startswith("o:") and v == v})
    base = out.get("E15_35a")
    if base:
        for c, r in out.items():
            r["p_more_than_scene"] = SS.worse_p(int(r["all_loss_events"]), int(base["all_loss_events"]), r["all_h"],
                                                base["all_h"])
            r["p_fewer_than_scene"] = SS.worse_p(int(base["all_loss_events"]), int(r["all_loss_events"]),
                                                 base["all_h"], r["all_h"])
    return out


def by_city(name):
    """{code: {seed: dict(wrap_sigma, wrap_full, street, leave, events, hours)}}: the cities are the units."""
    df, cands, _ = T.read(name, W.OUT)
    out = {}
    for cid, g in df.groupby("cand"):
        c = code(cands[cid]["label"])
        for seed, gg in g.groupby("seed"):
            s = W.wrap_scores(gg)
            out.setdefault(c, {})[str(seed)] = dict(
                wrap_sigma=s["wrap_sigma"], wrap_full=s["wrap_full"], wrap_narrow=s["wrap_narrow"],
                events=s["all_loss_events"], hours=s["all_h"],
                street=float(gg[gg.kind == "street"].v_coh.mean()) if (gg.kind == "street").any() else None,
                leave=float(gg[gg.kind == "search"].leave_v.astype(float).mean()) if (gg.kind == "search").any()
                else None)
    return out


def by_kind(name):
    """{code: {kind: dict(wrap_ring, wrap_sigma, wrap_s)}}: the ring while wrapped by flight, where the building, not
    the spread, mostly sets the formation's size."""
    df, cands, _ = T.read(name, W.OUT)
    out = {}
    for cid, g in df.groupby("cand"):
        c = code(cands[cid]["label"])
        for kind, gg in g.groupby("kind"):
            s = W.wrap_scores(gg)
            out.setdefault(c, {})[kind] = dict(wrap_ring=s["wrap_ring"], wrap_sigma=s["wrap_sigma"],
                                               wrap_s=float(gg.wrap_s.mean()))
    return out


def gate(name):
    import pandas as pd
    p = W.OUT / name / "bench" / "gate.csv"
    if not p.exists():
        return None
    out = []
    for r in pd.read_csv(p).to_dict("records"):
        lo, hi = SS.poisson_ci(int(r["all_loss_events"]))
        out.append(dict(r, code=r["set"], ev_lo=lo / r["all_h"], ev_hi=hi / r["all_h"]))
    return out


def unity_p(rows, pick="H15_35a", base="E15_35a"):
    """One-sided exact p that Unity's pick loses more often per hour than the scene."""
    u = {r["set"]: r for r in rows or [] if r["src"] == "unity"}
    if pick not in u or base not in u:
        return None
    return SS.worse_p(int(u[pick]["all_loss_events"]), int(u[base]["all_loss_events"]), u[pick]["all_h"], u[base]["all_h"])


def human():
    p = W.OUT / "human" / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def build():
    rindex = REPLAY / "index.json"
    rule = json.loads((W.OUT / "rule.json").read_text(encoding="utf-8"))
    vpath = W.OUT / "confirm" / "verdicts.json"
    verdicts = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), scene=T.SCENE, rule={k: v for k, v in rule.items()
                                                                            if k not in ("written", "note")},
                rule_note=rule["note"], rule_written=rule["written"], verdicts=verdicts.get("verdicts", {}),
                pick=verdicts.get("pick"), order=list(ORDER), human=human(),
                screen=rows("screen1"), confirm=rows("confirm"), city=by_city("confirm"), kinds=by_kind("confirm"),
                seeds=[T.TUNE_SEED, *T.HELD_SEEDS],
                flights=dict(screen=dict(cross=40, pass_=30, street=60, face=24, search=80),
                             confirm=dict(cross=30, pass_=30, street=100, face=12, search=100),
                             bench=dict(cross=20, pass_=20, street=50, face=12, search=40)),
                search=dict(pause_p=W.SEARCH_PAUSE_P, pause_f=W.SEARCH_PAUSE_F, pause_s=W.SEARCH_PAUSE_S,
                            hold_s=W.SEARCH_HOLD_S, from_m=W.SEARCH_FROM_M, leave_m=W.SEARCH_LEAVE_M),
                calib=json.loads((W.OUT / "calibration.json").read_text(encoding="utf-8")),
                unity=gate("bench1"), unity_p=unity_p(gate("bench1")), counts=bench_counts("swarmwrap_bench1"),
                full=W.FULL, exit_s=W.EXIT_S,
                unity_losses=json.loads((W.OUT / "bench1" / "bench" / "losses.json").read_text(encoding="utf-8")),
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
    dest = W.OUT / "swarm_wrap.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("swarm_wrap_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("swarm_wrap_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                       encoding="utf-8")
    print(f"[swarmwrap_report] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} "
          f"replay files; rule pick: {data['pick']}")
    return dest


if __name__ == "__main__":
    build()
