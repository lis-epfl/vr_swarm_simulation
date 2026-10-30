"""S5 Block-Shaped Surround (the block-shaped study): S4's question on BlockShapedCityWorld's city.

    python -B -m citysim.shaped geometry              # every design's streets, plazas and blocked streets
    python -B -m citysim.shaped dref [--jobs 24]      # step 1: every way of taking a block at every d_ref, base design
    python -B -m citysim.shaped choose                # step 1's rule -> chosen.json (d_ref, surround)
    python -B -m citysim.shaped partA [--jobs 24]     # step 2: one block per trial, every design, at the chosen d_ref
    python -B -m citysim.shaped partB [--jobs 24]     # step 3: a search for a goal at a building, Part B's designs
    -> experiment_results/shaped_study/

S4 flew block-shaped buildings (its variant B) only in the replica, under a closest-point obstacle model the C# never
had, and dropped them. BlockShapedCityWorld (2026-10-28) builds them in Unity: one convex prism per block at the back of
a 3.5 m footpath, avoided through today's force law, which wraps each prism in the circumradius of its axis-aligned
bounds (blockcity.cylinder_radius), with contacts on the true prism. swarm_replica's PolyCity does exactly that under the
default obstacle model, so every flight here uses split_probe.model(st, "R") (the default law) on a PolyCity, never the
closest-point model. The city is S4's own `bbB_p40_s18` (RSA block seeds), the scene's.

Every flight is S4 Phase 3's (lookdown.run_job, lookdown_search.run_job), scored at a 250 m sight range (the user's,
2026-10-28), with the city named per job (lookdown.job_params). Designs (DESIGNS): the base (p 40, s 18, 3.5 m footpath,
RSA), the footpath at 1.5 and 0 m, p 35 and p 30 (at 1.5 m: a 3.5 m footpath leaves under one block in ten with a
building at p 30), s 20, and three tessellations (Lloyd-relaxed RSA, a jittered and an exact hexagonal lattice).

Step 1's rule, fixed before its results were read (choose): the d_ref and surround that pass round a block most
efficiently. For every surround setting flown on the base design (aim and turn, or a firm hold; 6, 12 or 20 s; every
d_ref in D_GRID): safe (at most SAFE_LOST drones lost and SAFE_CONTACTS building contacts per block), one group again
before the next junction in at least REFORM_MIN of trials, and enclosing the block (S1's test) in at least ENCLOSE_MIN
of them. Among those, the quickest from arriving at the block to the next junction (median); ties to the smaller d_ref,
then the shorter surround. With none, the safe setting that encloses the block most often. Every later step flies that
d_ref and that surround (and holds at the corner for the same time, and passes round at the same spread).

Step 1b (dref2), added 2026-10-28 after step 1's results: no setting met the safety bar, because the swarm lost drones
in drone-drone collisions everywhere, the street-bound swarm too (0.33 per block at the streets' spread of 1.0, 5 of 30
trials, each 10-20 s after reaching the block, on the block's sides where the avoidance circles bulge into the
street). S4's city of catalogue buildings on the same streets lost almost none. Step 1 held the street spread at 1.0, so
it could not find a spread that fits these streets, and its rule's fallback ("the least unsafe") picked a surround that
never enclosed the block; that pick is not used. Step 1b flies one spread for the whole flight (streets and block, as a
pilot leaves the dial), d in D2_GRID, and its rule, fixed before it ran (choose): among the surround settings that
enclose the block in at least ENCLOSE_MIN of trials, the one that loses the fewest drones per block; ties to the
quicker, then the smaller spread. The bar (SAFE_LOST) is reported against, not required, since step 1 showed it may be
out of reach. Every later step flies that spread in the streets and at the block.
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

from citysim import blockcity, clutter, lookdown as LD, lookdown_search as LS, split_probe as S  # noqa: E402

OUT = S.OUT_ROOT.parent.parent / "shaped_study"          # experiment_results/shaped_study
SIGHT_M = 250.0
TRIALS = 30
CITY_SEEDS = (0, 1, 2)                                   # Part A's cities, S4's (seed 1 is the scene's)
D_GRID = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2)             # the ring at 3 m round p 40's cylinders predicts about 1.1
D2_GRID = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)            # step 1b: one spread for streets and block
D2_CONDS = (("street", None, None), ("pass", None, None), ("hold", None, 12.0), ("hold", None, 20.0),
            ("hop", "aim", 12.0), ("hop", "aim", 20.0), ("hop", "firm", 12.0), ("hop", "firm", 20.0))
T_GRID = (6.0, 12.0, 20.0)
SAFE_LOST, SAFE_CONTACTS, REFORM_MIN, ENCLOSE_MIN = 0.03, 0.1, 0.9, 0.7
B_CITY_SEED0, B_CITIES, B_TARGETS = 200, 8, 4            # Part B: S4's

# name: (p, s, city, what it varies)
DESIGNS = {
    "base": (40.0, 18.0, dict(variant="B"), "BlockShapedCityWorld: p 40, s 18, 3.5 m footpath, RSA seeds"),
    "foot15": (40.0, 18.0, dict(variant="B", inset=1.5), "a 1.5 m footpath"),
    "foot0": (40.0, 18.0, dict(variant="B", inset=0.0), "no footpath: the facade on the kerb"),
    "p35": (35.0, 18.0, dict(variant="B"), "blocks 35 m apart"),
    "p30": (30.0, 18.0, dict(variant="B", inset=1.5), "blocks 30 m apart, 1.5 m footpath"),
    "s20": (40.0, 20.0, dict(variant="B"), "20 m streets"),
    "lloyd2": (40.0, 18.0, dict(variant="B", relax=2), "RSA seeds relaxed twice (Lloyd)"),
    "hex15": (40.0, 18.0, dict(variant="B", seeds="hex", jitter=0.15), "a jittered hexagonal lattice"),
    "hex0": (40.0, 18.0, dict(variant="B", seeds="hex", jitter=0.0), "an exact hexagonal lattice"),
}
PART_B = ("base", "foot15", "p35", "p30", "s20", "hex15")
B_CONDITIONS = ("single", "swarm_street", "swarm_hold", "swarm_pass", "swarm_hop")


def params(name, seed):
    p, s, city, _ = DESIGNS[name]
    return blockcity.Params(**dict(city, p=p, s=s, seed=seed))


def _log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[shaped {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(jobs, fn, name, label, procs, key):
    """Append every job not yet in name/results.jsonl, resumable."""
    from citysim import parallel
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    done = clutter._done_ids(res)
    todo = [j for j in jobs if key(j) not in done]
    t0 = time.time()
    if todo:
        parallel.run_jobs(fn, todo, label, processes=procs, stall_s=900.0, progress=lambda i, n, r: clutter._append(res, r))
    _log(f"{name}: {len(todo)} jobs in {(time.time() - t0) / 60:.1f} min")


# ============================================================================= geometry

def cmd_geometry(a):
    """Per design and city seed: blocks, buildings, plazas, the block's sides drawn as streets, the streets leaving its
    corners (blockcity.street_stats), and the streets Unity's avoidance cylinders close or narrow below 6 m."""
    import pandas as pd
    rows = []
    for name in DESIGNS:
        for seed in CITY_SEEDS + tuple(range(B_CITY_SEED0, B_CITY_SEED0 + B_CITIES)):
            P = params(name, seed)
            L, blocks = blockcity.load(blockcity.ensure(P))
            st = blockcity.street_stats(L, blocks)
            inter = [b for b in blocks if b["interior"]]
            rows.append(dict(design=name, seed=seed, p=P.p, s=P.s, inset=P.inset, seeds=P.seeds, relax=P.relax,
                             interior=len(inter), buildings=sum(b["building"] >= 0 for b in inter),
                             plazas=sum(b["building"] < 0 for b in inter),
                             **{k: v for k, v in st.items() if k != "sides_hist"}))
            print(f"[shaped] {P.ident()}: {st.get('sides_mean')} sides, {st.get('exits_mean')} exits, "
                  f"{st.get('cyl_closed')} of {st.get('cyl_pairs')} closed", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "geometry.csv", index=False)
    g = df.groupby("design").mean(numeric_only=True)
    g["plaza_share"] = g.plazas / g.interior
    g["closed_share"] = g.cyl_closed / g.cyl_pairs
    g.round(3).to_csv(OUT / "geometry_by_design.csv")
    print(g[["buildings", "plazas", "plaza_share", "sides_mean", "corners_mean", "exits_mean", "closed_share",
             "cyl_narrow"]].round(2).to_string())


# ============================================================================= step 1: d_ref

def a_job(name, trial, d=None, conds=None, single=False, d_street=None):
    p, s, city, _ = DESIGNS[name]
    j = dict(p=p, s=s, city_seed=int(CITY_SEEDS[trial % len(CITY_SEEDS)]), trial=trial, city=city, design=name,
             look_range=SIGHT_M)
    if single:
        j["single"] = True
    else:
        j.update(d=float(d), T_grid=list(T_GRID))
        if conds:
            j["conds"] = [list(c) for c in conds]
        if d_street:
            j["d_street"] = float(d_street)
    return j


def cmd_dref2(a):
    jobs = [a_job("base", t, d, D2_CONDS, d_street=d) for t in range(TRIALS) for d in D2_GRID]
    (OUT / "dref2").mkdir(parents=True, exist_ok=True)
    (OUT / "dref2" / "spec.json").write_text(json.dumps(dict(
        step="d_ref, one spread for streets and block", design="base", d_grid=D2_GRID, conditions=D2_CONDS,
        trials=TRIALS, city_seeds=CITY_SEEDS, sight_m=SIGHT_M,
        rule=dict(enclose=ENCLOSE_MIN, then="fewest drones lost per block, then quickest, then smaller spread",
                  bar_lost=SAFE_LOST, bar_contacts=SAFE_CONTACTS)), indent=1), encoding="utf-8")
    run(jobs, LD.run_job, "dref2", "S5 d_ref (streets and block)", a.jobs, LD.job_id)


def cmd_dref(a):
    jobs = [a_job("base", t, d) for t in range(TRIALS) for d in D_GRID]
    (OUT / "dref").mkdir(parents=True, exist_ok=True)
    (OUT / "dref" / "spec.json").write_text(json.dumps(dict(
        step="d_ref", design="base", d_grid=D_GRID, T_grid=T_GRID, trials=TRIALS, city_seeds=CITY_SEEDS, sight_m=SIGHT_M,
        rule=dict(lost=SAFE_LOST, contacts=SAFE_CONTACTS, reform=REFORM_MIN, enclose=ENCLOSE_MIN)), indent=1),
        encoding="utf-8")
    for P in {params("base", c).ident(): params("base", c) for c in CITY_SEEDS}.values():
        blockcity.ensure(P)
    run(jobs, LD.run_job, "dref", "S5 d_ref", a.jobs, LD.job_id)


def dref_table(name="dref"):
    """Every setting flown in a step-1 run on the base design: (cond, mode, T, d) -> the measures the rule reads. In
    step 1 the streets are flown at 1.0 whatever d is; in step 1b at d, so the street-bound swarm is a row per d."""
    import pandas as pd
    df, _ = LD.read(name, root=OUT)
    df["reform_ok"] = df.reform_ok.astype(float)
    one_street = df[df.cond == "street"].d_street.nunique() == 1
    key = ["city_seed", "trial"] if one_street else ["city_seed", "trial", "d"]
    sdf = df[df.cond == "street"].drop_duplicates(key)
    st = sdf.set_index(key).hop_s
    rows = []
    for (cond, mode, T, d), g in df.groupby(["cond", "mode", "T", "d"], dropna=False):
        if cond == "street" and one_street:
            continue
        hs = g.set_index(key).hop_s
        env = g.enveloped.dropna().astype(float) if "enveloped" in g else pd.Series(dtype=float)
        rows.append(dict(cond=cond, mode=mode if isinstance(mode, str) else None, T=None if T != T else float(T),
                         d=float(d), d_street=float(g.d_street.iloc[0]), n=int(len(g)), lost=float(g.lost.mean()),
                         contacts=float(g.contacts.mean()), reform=float(g.reform_ok.mean()),
                         enclosed=float(env.mean()) if len(env) else None, hop_s=float(g.hop_s.median()),
                         price=float(np.nanmedian((hs / st.reindex(hs.index)).values)),
                         look_c=float(g.look_c.mean()), look_F=float(g.look_F.mean())))
    street = dict(cond="street", hop_s=float(st.median()), look_c=float(sdf.look_c.mean()),
                  lost=float(sdf.lost.mean()))
    return rows, street


def choose2(rows):
    """Step 1b's rule: among surrounds that enclose the block in at least ENCLOSE_MIN of trials, the fewest drones lost
    per block; ties to the quicker, then the smaller spread."""
    surr = [r for r in rows if r["cond"] == "hop"]
    ok = [r for r in surr if (r["enclosed"] or 0) >= ENCLOSE_MIN]
    if ok:
        best = sorted(ok, key=lambda r: (round(r["lost"], 3), r["hop_s"], r["d"]))[0]
        why = "encloses the block: the fewest drones lost, then the quickest"
    else:
        best = sorted(surr, key=lambda r: (-(r["enclosed"] or 0), r["lost"], r["d"]))[0]
        why = "none encloses often enough: the one that encloses most"
    safe = best["lost"] <= SAFE_LOST and best["contacts"] <= SAFE_CONTACTS
    return dict(d=best["d"], mode=best["mode"], T=best["T"], why=why, row=best, n_ok=len(ok), meets_bar=bool(safe),
                n_safe=sum(r["lost"] <= SAFE_LOST and r["contacts"] <= SAFE_CONTACTS for r in surr))


def choose(rows):
    surr = [r for r in rows if r["cond"] == "hop"]
    safe = [r for r in surr if r["lost"] <= SAFE_LOST and r["contacts"] <= SAFE_CONTACTS]
    ok = [r for r in safe if r["reform"] >= REFORM_MIN and (r["enclosed"] or 0) >= ENCLOSE_MIN]
    if ok:
        best = sorted(ok, key=lambda r: (r["hop_s"], r["d"], r["T"]))[0]
        why = "safe, reforms, encloses the block: the quickest"
    elif safe:
        best = sorted(safe, key=lambda r: (-(r["enclosed"] or 0), r["hop_s"], r["d"]))[0]
        why = "none encloses often enough: the safe setting that encloses most"
    else:
        best = sorted(surr, key=lambda r: (r["lost"] + r["contacts"], r["d"]))[0]
        why = "none safe: the least unsafe"
    return dict(d=best["d"], mode=best["mode"], T=best["T"], why=why, row=best, n_ok=len(ok), n_safe=len(safe))


def cmd_choose(a):
    """Step 1b's rule on step 1b's run (step 1's own pick is kept beside it, as step1_pick, and not used)."""
    rows1, street1 = dref_table("dref")
    first = choose(rows1)
    rows, street = dref_table("dref2")
    ch = choose2(rows)
    (OUT / "chosen.json").write_text(json.dumps(dict(ch, street=street, table=rows, d_street=ch["d"],
                                                     step1=dict(pick={k: first[k] for k in ("d", "mode", "T", "why")},
                                                                street=street1, table=rows1)), indent=1),
                                     encoding="utf-8")
    r = ch["row"]
    _log(f"chosen: {ch['mode']} {ch['T']:g} s at d_ref {ch['d']:g}, streets too ({ch['why']}; {ch['n_ok']} enclose): "
         f"encloses {r['enclosed']:.2f}, {r['hop_s']:.1f} s to the next junction, lost {r['lost']:.3f} per block, "
         f"contacts {r['contacts']:.2f}; meets the bar: {ch['meets_bar']}")


def chosen():
    return json.loads((OUT / "chosen.json").read_text(encoding="utf-8"))


# ============================================================================= step 2: Part A on every design

def cmd_partA(a):
    ch = chosen()
    conds = [("street", None, None), ("hold", None, ch["T"]), ("pass", None, None), ("hop", ch["mode"], ch["T"])]
    jobs = []
    for name in DESIGNS:
        for t in range(TRIALS):
            jobs.append(a_job(name, t, ch["d"], conds, d_street=ch["d_street"]))
            jobs.append(a_job(name, t, single=True))
    (OUT / "partA").mkdir(parents=True, exist_ok=True)
    (OUT / "partA" / "spec.json").write_text(json.dumps(dict(
        step="A", designs={k: dict(p=v[0], s=v[1], city=v[2], what=v[3]) for k, v in DESIGNS.items()}, trials=TRIALS,
        city_seeds=CITY_SEEDS, sight_m=SIGHT_M, chosen={k: ch[k] for k in ("d", "mode", "T")},
        conditions=[list(c) for c in conds] + [["single", None, None], ["single_look", None, None]]), indent=1),
        encoding="utf-8")
    for name in DESIGNS:
        for c in CITY_SEEDS:
            blockcity.ensure(params(name, c))
    run(jobs, LD.run_job, "partA", "S5 Part A", a.jobs, LD.job_id)


# ============================================================================= step 3: Part B

def b_spec(name):
    p, s, city, what = DESIGNS[name]
    ch = chosen()
    return dict(part="B", study="S5", design=name, what=what, city=city, designs=[[p, s]], city_seed0=B_CITY_SEED0,
                cities=B_CITIES, targets=B_TARGETS, T_s=LS.SEARCH_T_S, conditions=list(B_CONDITIONS),
                chosen={f"{p:g}|{s:g}": dict(mode=ch["mode"], T=ch["T"], d=ch["d"])}, sight_m=SIGHT_M,
                d_street=ch["d_street"], frontier_min_m=LS.FRONTIER_MIN_M)


def cmd_partB(a):
    for name in PART_B:
        spec = b_spec(name)
        out = OUT / f"partB_{name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        jobs = [j for js in LS.plan(spec) for j in js]
        for c in range(B_CITY_SEED0, B_CITY_SEED0 + B_CITIES):
            blockcity.ensure(params(name, c))
        run(jobs, LS.run_job, f"partB_{name}", f"S5 Part B {name}", a.jobs, LS.job_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("geometry", "dref", "dref2", "choose", "partA", "partB"):
        p = sub.add_parser(c)
        p.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    {"geometry": cmd_geometry, "dref": cmd_dref, "dref2": cmd_dref2, "choose": cmd_choose, "partA": cmd_partA,
     "partB": cmd_partB}[a.cmd](a)


if __name__ == "__main__":
    main()
