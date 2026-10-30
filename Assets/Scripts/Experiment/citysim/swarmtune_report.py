"""S6 Patch City Tuning: the results page, citysim/swarmtune_report.html filled with the runs' data.

    python -B -m citysim.swarmtune_report
        -> experiment_results/swarm_tune/swarm_tune.html (the replays inlined, to open as a file) and
           swarm_tune_bare.html (what the artifact is published from), with swarm_tune_files.json, the replay files to
           publish beside it

The page is S4's family (lesson 36): the question for the human trials and the answer first, then the cities and the
flights (swarmtune_replays), every table behind All results, and the terms. Its styles are S4's page's own, read from
split_report.html at build, as S5's page does. Every number is read from the runs here, never typed into the template.
Imported only by the parent.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, swarmtune as T  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "swarmtune_report.html"
S4_TEMPLATE = HERE / "split_report.html"
VIEW_JS = HERE / "replay_view.js"
REPLAY = T.OUT / "replay"
RING_D = (1.0, 1.08, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0)
KEEP = ("label", "d", "n", "km", "score", "safe", "v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "hull")
PER_KIND = ("v_coh", "v_prog", "one", "lost_km", "contact_km", "eps_km", "ep_p90", "open", "wrapped", "coverage",
            "stall", "finished", "bld_pass", "bld_frag", "bld_reformed")


def _clean(x):
    """JSON-safe: NaN / inf to None, numpy to Python, floats rounded to 4."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4) if math.isfinite(float(x)) else None
    return x


def _row(r):
    out = {k: r.get(k) for k in KEEP if k in r}
    out["kinds"] = {kind: {m: r.get(f"{kind}_{m}") for m in PER_KIND if f"{kind}_{m}" in r} for kind in T.KINDS}
    out["over"] = {k[2:]: v for k, v in r.items() if k.startswith("o:") and v == v}
    return out


def table(name):
    return [_row(r) for r in T.table(name).to_dict("records")]


def geometry():
    """The scene's buildings as the force law sees them, the ring the formation holds at each d_ref, and the median
    building to draw (cached: the rings take a minute)."""
    path = T.OUT / "geometry.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from citysim import formation
    C = T.city(T.TUNE_SEED)
    rc, L = C["rcity"], C["L"]
    polys = [blockcity.building_poly(b) for b in L.boxes]
    R = rc.R
    i = int(np.argmin(np.abs(R - np.median(R))))
    c = rc.c[i]
    rings = {}
    for d in RING_D:
        ds = [formation.ring_diameter(formation.hover_template(d, T.N_DRONES, s, T.SCENE)[0]) for s in range(3)]
        rings[f"{d:g}"] = round(float(np.mean(ds)), 2)
    ext = np.array([np.ptp(p, 0) for p in polys])
    out = dict(n_buildings=int(rc.n), cyl_r=dict(p10=float(np.percentile(R, 10)), median=float(np.median(R)),
                                                p90=float(np.percentile(R, 90))),
               bounds=[float(v) for v in np.median(ext, 0)], street_m=18.0, footpath_m=3.5, facade_gap_m=25.0,
               ring_diameter=rings, building=dict(poly=(polys[i] - c).round(2).tolist(), R=round(float(R[i]), 2)),
               scale=float(T.model({}).S))
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def confirm_by(field):
    """Confirmation candidates scored per city (field 'seed') or per flight kind."""
    df, cands, _ = T.read("confirm")
    out = {}
    for (cid, key), g in df.groupby(["cand", field]):
        r = T.score_candidate(g)
        out.setdefault(cid, {})[str(key)] = {k: r.get(k) for k in ("v_coh", "v_prog", "one", "lost_km", "contact_km",
                                                                     "open", "km")}
    return out


def gate(name):
    import pandas as pd
    path = T.OUT / name / "bench" / "gate.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return [_row(r) | {"set": r["set"], "src": r["src"]} for r in df.to_dict("records")]


def steps():
    import pandas as pd
    path = T.OUT / "steps.csv"
    return pd.read_csv(path).to_dict("records") if path.exists() else None


def bench_hard(run):
    """Hard contacts (over 2 m/s into the facade) per set, from the bench's own flight lines."""
    from citysim import use_bench_modules
    use_bench_modules()
    import bench
    p = Path(r"D:\claude_obstacle_bench\runs") / run
    if not (p / "results.jsonl").exists():
        return None
    _, _, _, flights, _, _ = bench.read_results(p)
    out = {}
    for f in flights:
        o = out.setdefault(f["set"], dict(contacts=0, hard=0, flights=0, lost=0))
        o["contacts"] += int(f.get("contacts") or 0)
        o["hard"] += int(f.get("hard_contacts") or 0)
        o["lost"] += int(f["alive_start"] - f["alive_end"])
        o["flights"] += 1
    return out


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def s4_style():
    m = re.search(r"<style>(.*?)</style>", S4_TEMPLATE.read_text(encoding="utf-8"), re.S)
    return m.group(1)


def build():
    rindex = REPLAY / "index.json"
    rule = json.loads((T.OUT / "rule.json").read_text(encoding="utf-8"))
    chosen = json.loads((T.OUT / "chosen.json").read_text(encoding="utf-8"))[0]
    cands = json.loads((T.OUT / "confirm_cands.json").read_text(encoding="utf-8"))
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), scene=T.SCENE, pre_s6=T.PRE_S6, chosen=chosen, rule=rule,
                bars=dict(lost=T.SAFE_LOST_KM, contacts=T.SAFE_CONTACT_KM, open=T.OPEN_MAX, hull=T.HULL_MIN,
                          pen=[T.PEN_LOST, T.PEN_CONTACT, T.PEN_OPEN]),
                flights=dict(search=T.FLIGHTS, confirm=dict(cross=40, pass_=30, street=20), seeds=list((T.TUNE_SEED,) + T.HELD_SEEDS)),
                geometry=geometry(), baseline=table("baseline2"), sweep=table("sweep2"), grid2=table("grid2"),
                grid3=table("grid3"), confirm=table("confirm"), confirm_cands=cands,
                confirm_seed=confirm_by("seed"), confirm_ids={T.cand_id(c["over"], c["d"]): c["label"] for c in cands},
                bench_base=gate("bench_base"), bench_tuned=gate("bench_tuned"), bench_single=gate("bench_single"),
                hard=dict(tuned=bench_hard("swarmtune_bench_tuned"), base=bench_hard("swarmtune_bench_base"),
                          single=bench_hard("swarmtune_bench_single")),
                steps=steps(), replay=json.loads(rindex.read_text(encoding="utf-8")) if rindex.exists() else None)
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
    dest = T.OUT / "swarm_tune.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("swarm_tune_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("swarm_tune_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                       encoding="utf-8")
    print(f"[swarmtune_report] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} replay "
          f"files")
    return dest


if __name__ == "__main__":
    build()
