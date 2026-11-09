"""S7 Building Flow: the results page, citysim/swarmflow_report.html filled with the runs' data.

    python -B -m citysim.swarmflow_report
        -> experiment_results/swarm_flow/swarm_flow.html (the replays inlined, to open as a file) and
           swarm_flow_bare.html (what the artifact is published from), with swarm_flow_files.json, the replay files to
           publish beside it

The page is S4's family (lesson 36), kept short at the user's request: the question for the human trials and the answer
first, the flights (swarmflow_replays), every table behind All results, the terms. Its styles are S4's page's own, read
from split_report.html at build. Every number is read from the runs here, never typed into the template.
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

from citysim import swarmflow as F, swarmtune as T  # noqa: E402
from citysim.swarmtune_report import _clean, _row, s4_style, VIEW_JS  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "swarmflow_report.html"
REPLAY = F.OUT / "replay"
S7_FIELDS = ("slideRoundBuildings", "slideHeadOnFraction", "slideSplitAimFraction", "routeStragglers",
             "stragglerPressShare", "adaptiveSpread", "streetSpreadFactor", "surroundLookAhead", "surroundAimFraction",
             "spreadFilterTime")

# What the screens tried before the confirmation, each read against its own screen's T3 (same routes): (run, label,
# what it is, kept or why not). Screens 1-3 flew 52 flights per candidate, 4-6 flew 104, all in the scene's city.
SCREENS = [
    ("screen1", "slide own 0.8", "Slide: every drone round its own side of the building", "the base rule"),
    ("screen1", "slide formation 0.5", "Slide: the formation in halves everywhere (the suggestion as written)",
     "dropped: at a street corner the trailing half goes round the far side of the corner building"),
    ("screen1", "slide group 0.5", "Slide: always round on the swarm's side", "dropped: stops surrounding"),
    ("screen2", "slide auto 0.8 aim 1.0", "Slide: halves only round a building the swarm flies at (within 1.0 R)",
     "dropped: the split fires at street bends"),
    ("screen3", "slide", "… within 0.6 R", "kept"),
    ("screen4", "slide hidden", "… and the swarm's side when the building hides the centroid",
     "kept: the shipped rule, steadier in the streets"),
    ("screen5", "slide head-on 1.0", "Slide for anything the stick line hits (head-on fraction 1.0)",
     "not taken: faster, but loses drones in the streets"),
    ("screen1", "route lean", "Route: cohesion's whole inward part turned along the building",
     "dropped: turned the drones' mutual repulsion too, so neighbours collided"),
    ("screen2", "route lean", "Route: only attraction to hidden neighbours, all of it turned",
     "dropped: nothing left holding a surround's two halves together"),
    ("screen3", "route press 0.5", "Route: keep up to 0.5 of the field's ceiling, turn the rest", "fewer contacts"),
    ("screen3", "route press 0.7", "… 0.7", "kept"),
    ("screen1", "adaptive 0.7", "Adaptive spread: 0.7 of d_ref in the streets", "faster, but loses drones in tight gaps"),
    ("screen4", "adaptive 0.7 + damper", "… with a stronger close-range damper (c_damp 4, d_damp 0.5)",
     "dropped: the damper does not stop these collisions"),
    ("screen5", "adaptive 0.8", "Adaptive spread: 0.8 of d_ref in the streets", "kept"),
    ("screen6", "adaptive 0.8 x1.2", "… and 1.2 of d_ref at a building", "dropped: no better surrounding, slower streets"),
]
SCREEN_COLS = ("v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "cross_wrapped", "street_v_coh", "street_one",
               "cross_bld_pass", "n")


def table(name):
    rows = []
    for r in F.table(name).to_dict("records"):
        out = _row(r)
        out["cand"] = r["cand"]
        out["sigma"], out["narrow"] = r.get("sigma"), r.get("narrow")
        rows.append(out)
    return rows


def by_seed(name):
    df, cands, _ = T.read(name, F.OUT)
    out = {}
    for (cid, seed), g in df.groupby(["cand", "seed"]):
        r = T.score_candidate(g)
        out.setdefault(cands[cid]["label"], {})[str(seed)] = {
            k: r.get(k) for k in ("v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "km", "cross_wrapped",
                                  "street_v_coh")}
    return out


def screens():
    tabs, out = {}, []
    for run, label, what, why in SCREENS:
        if run not in tabs:
            tabs[run] = {r["label"]: r for r in F.table(run).to_dict("records")}
        t = tabs[run]
        if label not in t:
            continue
        out.append(dict(run=run, label=label, what=what, why=why, row={k: t[label].get(k) for k in SCREEN_COLS},
                        t3={k: t["T3"].get(k) for k in SCREEN_COLS}))
    return out


def gate(name):
    import pandas as pd
    path = F.OUT / name / "bench" / "gate.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return [_row(r) | {"set": r["set"], "src": r["src"]} for r in df.to_dict("records")]


def bench_counts(run):
    """Contacts (and hard ones, over 2 m/s), drones lost and flights per set, from the bench's own flight lines."""
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    p = Path(r"D:\claude_obstacle_bench\runs") / run
    if not (p / "results.jsonl").exists():
        return None
    _, _, _, flights, _, _ = bench.read_results(p)
    out = {}
    for f in flights:
        o = out.setdefault(f["set"], dict(contacts=0, hard=0, flights=0, lost=0, deaths={}))
        o["contacts"] += int(f.get("contacts") or 0)
        o["hard"] += int(f.get("hard_contacts") or 0)
        o["lost"] += int(f["alive_start"] - f["alive_end"])
        o["flights"] += 1
        for k, v in (f.get("deaths") or {}).items():
            o["deaths"][k] = o["deaths"].get(k, 0) + int(v)
    return out


def params():
    """The S7 settings as SwarmManager.cs ships them (the scene saves none of them, so these are what Unity flies)."""
    f = T.sp.swarm_manager_fields()
    return {k: f[k].default for k in S7_FIELDS}


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def build():
    import pandas as pd
    rindex = REPLAY / "index.json"
    rule = json.loads((F.OUT / "rule.json").read_text(encoding="utf-8"))
    tab = F.table("confirm")
    verdicts, pick = F.judge(tab)
    s3 = pd.read_csv(T.OUT / "steps.csv").to_dict("records") if (T.OUT / "steps.csv").exists() else None
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), scene=T.SCENE, params=params(), rule=rule, rule_numbers=F.RULE,
                verdicts=verdicts, pick=pick, confirm=table("confirm"), confirm_seed=by_seed("confirm"),
                flights=dict(confirm=dict(cross=40, pass_=30, street=20), seeds=[T.TUNE_SEED, *T.HELD_SEEDS],
                             screen=dict(T.FLIGHTS)),
                unity=gate("bench_flow"), unity_single=gate("bench_single"), unity_tune2=gate("bench_tune2"),
                tune2=table("tune2") if (F.OUT / "tune2" / "results.jsonl").exists() else None,
                counts=dict(flow=bench_counts("swarmflow_bench_flow"), single=bench_counts("swarmflow_bench_single"),
                            tune2=bench_counts("swarmflow_bench_tune2")),
                screens=screens(), s6_steps=s3,
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
    dest = F.OUT / "swarm_flow.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("swarm_flow_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("swarm_flow_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                       encoding="utf-8")
    print(f"[swarmflow_report] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} replay "
          f"files; rule pick: {pick}")
    return dest


if __name__ == "__main__":
    build()
