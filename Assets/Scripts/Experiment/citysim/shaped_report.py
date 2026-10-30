"""S5 Block-Shaped Surround: the results page, citysim/shaped_report.html filled with the runs' data.

    python -B -m citysim.shaped_report
        -> experiment_results/shaped_study/shaped_study.html (the replays inlined, to open as a file) and
           shaped_study_bare.html (what the artifact is published from), with shaped_study_files.json, the replay files
           to publish beside it

The page is S4's family (lesson 36): the question for the human trials and the answer first, then the cities and the
flights, every table behind All results, and the terms. Its styles are S4's page's own (read from split_report.html),
so the two pages look alike without a second copy to keep. Imported only by the parent.
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

from citysim import shaped as SH  # noqa: E402
from citysim.split_report import _clean, _decompose, _log_ratio_ci, _records, _tci  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "shaped_report.html"
S4_TEMPLATE = HERE / "split_report.html"
VIEW_JS = HERE / "replay_view.js"
REPLAY = SH.OUT / "replay"
LABELS = {"single": "one drone", "single_look": "one drone, stopping to look", "street": "swarm in the streets",
          "hold": "swarm holding at the corner", "pass": "swarm passing round", "hop": "swarm surrounding"}
SEARCH = ("swarm_street", "swarm_pass", "swarm_hop", "swarm_hold")


def geometry():
    import pandas as pd
    path = SH.OUT / "geometry.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    a = df[df.seed.isin(SH.CITY_SEEDS)]                        # Part A's cities
    g = a.groupby("design").mean(numeric_only=True)
    out = {}
    for name in SH.DESIGNS:
        if name not in g.index:
            continue
        r = g.loc[name]
        out[name] = dict(buildings=float(r.buildings), plazas=float(r.plazas), interior=float(r.interior),
                         plaza_share=float(r.plazas / r.interior), sides=float(r.sides_mean),
                         corners=float(r.corners_mean), exits=float(r.exits_mean),
                         closed=float(r.cyl_closed), pairs=float(r.cyl_pairs), narrow=float(r.cyl_narrow),
                         closed_share=float(r.cyl_closed / r.cyl_pairs) if r.cyl_pairs else None)
    return out


def dref():
    path = SH.OUT / "chosen.json"
    if not path.exists():
        return None
    ch = json.loads(path.read_text(encoding="utf-8"))
    s1 = ch.get("step1") or {}
    return dict(chosen={k: ch.get(k) for k in ("d", "mode", "T", "why", "n_ok", "n_safe", "meets_bar", "d_street")},
                row=ch["row"], street=ch["street"], table=ch["table"], grid=list(SH.D2_GRID),
                rule=dict(lost=SH.SAFE_LOST, contacts=SH.SAFE_CONTACTS, enclose=SH.ENCLOSE_MIN),
                step1=dict(pick=s1.get("pick"), street=s1.get("street"), grid=list(SH.D_GRID), table=s1.get("table")))


def part_a():
    from citysim import lookdown as LD
    if not (SH.OUT / "partA" / "results.jsonl").exists():
        return None
    df, _ = LD.read("partA", root=SH.OUT)
    df["reform_ok"] = df.reform_ok.astype(float)
    out = {}
    for (name, cond), g in df.groupby(["design", "cond"]):
        m, lo, hi = _tci(g.look_c)
        env = g.enveloped.dropna().astype(float) if "enveloped" in g else []
        out.setdefault(name, {})[cond] = dict(look=m, lo=lo, hi=hi, look_F=float(g.look_F.mean()),
                                              ent=float(g.ent_c.mean()), hop_s=float(g.hop_s.median()),
                                              lost=float(g.lost.mean()), contacts=float(g.contacts.mean()),
                                              reform=float(g.reform_ok.mean()), n=int(len(g)),
                                              enclosed=float(np.mean(env)) if len(env) else None)
    return _clean(out)


def part_b():
    from citysim import lookdown_search as LS
    out = {}
    tk, hk = f"t_r{SH.SIGHT_M:g}", f"hit_r{SH.SIGHT_M:g}"
    keys = ["p", "s", "city_seed", "target_i"]
    for name in SH.PART_B:
        if not (SH.OUT / f"partB_{name}" / "results.jsonl").exists():
            continue
        b, _ = LS.read(f"partB_{name}", root=SH.OUT)
        if not len(b):
            continue
        rat = LS.ratios(b, pairs=[(c, "single") for c in SEARCH] + [("swarm_hop", "swarm_street"),
                                                                    ("swarm_pass", "swarm_street")], measure=tk)
        rat = rat[rat.design == "pooled"]                       # one design per run: its own rows are the pooled ones
        g = b.groupby("condition")
        out[name] = dict(n_targets=int(b.groupby(keys).ngroups), n_cities=int(b.groupby(["p", "s", "city_seed"]).ngroups),
                         ratios=_records(rat), found={k: float(v) for k, v in g[hk].mean().items()},
                         rmst={k: float(v) for k, v in g[tk].mean().items()},
                         lost={k: float(v) for k, v in g.lost.mean().items()},
                         contacts={k: float(v) for k, v in g.contacts.mean().items()},
                         decomp={c: _decompose(b, c, "swarm_street", keys, hk, tk) for c in ("single", "swarm_hop")})
    return _clean(out) or None


def s4_ref():
    """S4's catalogue city at p 40 · s 18 on the same streets, at the same 250 m sight range (lookdown_A_r250 /
    lookdown_A1_r250): what each way looks down there, for the comparison S5 exists to make."""
    from citysim import lookdown as LD
    try:
        a, _ = LD.read("lookdown_A_r250")
        a1, _ = LD.read("lookdown_A1_r250")
    except FileNotFoundError:
        return None
    a, a1 = a[(a.p == 40) & (a.s == 18)], a1[(a1.p == 40) & (a1.s == 18)]
    st = a[a.cond == "street"].drop_duplicates(["city_seed", "trial"])
    hop = a[(a.cond == "hop") & (a["mode"] == "aim") & (a["T"] == 12.0) & (a.d == 1.8)]
    return dict(single=float(a1[a1.cond == "single"].look_c.mean()),
                single_look=float(a1[a1.cond == "single_look"].look_c.mean()),
                street=float(st.look_c.mean()), hop=float(hop.look_c.mean()), street_lost=float(st.lost.mean()),
                hop_label="aim and turn 12 s at d_ref 1.8")


def replay_files():
    if not (REPLAY / "index.json").exists():
        return {}
    return {f"replay/{f.relative_to(REPLAY).as_posix()}": f for d in ("cities", "flights")
            for f in sorted((REPLAY / d).glob("*.json"))}


def s4_style():
    """S4's page's <style> block: the study pages' family, one copy."""
    m = re.search(r"<style>(.*?)</style>", S4_TEMPLATE.read_text(encoding="utf-8"), re.S)
    return m.group(1)


def build():
    rindex = REPLAY / "index.json"
    data = dict(built=time.strftime("%Y-%m-%d %H:%M"), sight=SH.SIGHT_M,
                designs={k: dict(p=v[0], s=v[1], city=v[2], what=v[3]) for k, v in SH.DESIGNS.items()},
                part_b_designs=list(SH.PART_B), geometry=geometry(), dref=dref(), a=part_a(), b=part_b(), s4=s4_ref(),
                trials=SH.TRIALS, labels=LABELS,
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
    dest = SH.OUT / "shaped_study.html"
    dest.write_text(full, encoding="utf-8")
    dest.with_name("shaped_study_bare.html").write_text(bare, encoding="utf-8")
    dest.with_name("shaped_study_files.json").write_text(json.dumps({k: str(v) for k, v in files.items()}, indent=1),
                                                         encoding="utf-8")
    print(f"[shaped_report] {dest} ({len(full) / 1024:.0f} KB); bare {len(bare) / 1024:.0f} KB + {len(files)} replay files")
    return dest


if __name__ == "__main__":
    build()
