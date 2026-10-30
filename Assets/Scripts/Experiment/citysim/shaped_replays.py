"""S5 Block-Shaped Surround: the page's Cities tab data (lessons 30-31), S4's split_replays applied to S5's runs.

    python -B -m citysim.shaped_replays [--stage all|refly|assemble] [--jobs 6]
    -> experiment_results/shaped_study/replay/ (cities/, flights/, picks/, index.json)

The cities: every design at the scene's seed (1), with the avoidance circles Unity's law draws round each prism. The
flights: one Part A block on each of three designs (the scene's city, no footpath, the exact hexagonal lattice), flown
every way S5 flies it, and one Part B search on the scene's city in every condition. Each is S4's worker
(split_replays.p3_worker / p3b_worker) re-flying the stored job with a tape, kept only if its rows come out identical
to the run's. A pick is the trial (or target) whose surround-minus-street difference is nearest its design's median.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, playback, shaped as SH, split_probe as S, split_replays as SR  # noqa: E402
from citysim import geometry as geo  # noqa: E402

OUT = SH.OUT / "replay"
GALLERY_SEED = 1                                  # the scene's city
A_PICKS = [("s5a_base", "base", "Looking down a block: the scene's city (p 40 · s 18, 3.5 m footpath)"),
           ("s5a_foot0", "foot0", "Looking down a block: no footpath"),
           ("s5a_hex0", "hex0", "Looking down a block: an exact hexagonal lattice")]
B_PICKS = [("s5b_base", "base", "Search for a goal at a building: the scene's city")]
LABEL = {"street": "In the street, d_ref 1.0", "hold": "Hold at the corner", "pass": "Pass round", "hop": "Surround"}


def city_payload(P):
    """split_replays.city_payload for any design, with the avoidance circles (centre and radius of each prism's
    axis-aligned bounds: what Unity's law steers round)."""
    C = S.city_from(P)
    L, net = C["L"], C["net"]
    blocks = sorted(C["blocks"], key=lambda b: b["id"])
    polys = [blockcity.unflat(b["kerb_xz"]) if len(b["kerb_xz"]) >= 6 else np.zeros((0, 2)) for b in blocks]
    flags = [(1 if net.blocks.get(b["id"], {}).get("interior") else 0) | (2 if b["building"] < 0 else 0) for b in blocks]
    blds = [blockcity.building_poly(x) for x in L.boxes]
    cyl = [[*((q.max(0) + q.min(0)) / 2), blockcity.cylinder_radius(x)] for q, x in zip(blds, L.boxes)]
    streets = [[*net.jpos[st["u"]], *net.jpos[st["v"]]] for st in net.streets]
    inter = [b for b in net.blocks.values() if b["interior"]]
    st = blockcity.street_stats(L, blocks)
    stats = dict(blocks=int(sum(1 for q in polys if len(q))), interior=len(inter), buildings=len(L.boxes),
                 plazas=int(sum(1 for b in inter if b["building"] < 0)),
                 plaza_share_interior=round(sum(1 for b in inter if b["building"] < 0) / max(len(inter), 1), 3),
                 block_m=round(float(np.median([math.sqrt(abs(geo.area(b["kerb"]))) for b in inter])), 1),
                 street_m=round(float(np.median([x["length"] for x in net.streets if x["interior"]])), 1),
                 cyl_r=round(float(np.median([c[2] for c in cyl])), 1) if cyl else None,
                 sides=st.get("sides_mean"), exits=st.get("exits_mean"), closed=st.get("cyl_closed"),
                 pairs=st.get("cyl_pairs"), narrow=st.get("cyl_narrow"))
    block_bld = np.array([b["building"] for b in blocks], "<i2")
    return playback.city_json(Path(blockcity.path_for(P)).stem, blockcity.ARENA, polys, blds, flags, cyl,
                              net.sample_xz, streets, variant=P.variant, p=float(P.p), s=float(P.s), seed=int(P.seed),
                              stats=stats, block_building=playback.b64(block_bld, "<i2"))


def a_select():
    """[(pick, flight list, stored rows)] for A_PICKS from Part A."""
    from citysim import lookdown as LD
    df, by = LD.read("partA", root=SH.OUT)
    ch = SH.chosen()
    spec = json.loads((SH.OUT / "partA" / "spec.json").read_text(encoding="utf-8"))
    conds = [tuple(c) for c in spec["conditions"] if c[0] not in ("single", "single_look")]
    out = []
    for key, name, title in A_PICKS:
        g = df[df.design == name]
        hop = g[g.cond == "hop"].set_index(["city_seed", "trial"]).look_c
        st = g[g.cond == "street"].set_index(["city_seed", "trial"]).look_c
        diff = (hop - st.reindex(hop.index)).dropna()
        dev = (diff - diff.median()).abs().sort_values(kind="stable")
        c, t = dev.index[0]
        p, s, city, _ = SH.DESIGNS[name]
        job = dict(city=city, design=name, conds=[list(x) for x in conds], look_range=SH.SIGHT_M, d_street=ch["d_street"])
        jb = dict(job, p=p, s=s, city_seed=int(c), trial=int(t), d=float(ch["d"]), T_grid=list(SH.T_GRID))
        stored = {float(ch["d"]): by[LD.job_id(jb)]}
        j1 = dict(city=city, design=name, p=p, s=s, city_seed=int(c), trial=int(t), single=True)
        stored["single"] = by[LD.job_id(j1)]
        surr = f"{'aim and turn' if ch['mode'] == 'aim' else 'firm hold'} {ch['T']:g} s"
        fl = [(cn, mode, T, ch["d"], (f"{LABEL[cn]}, d_ref {ch['d']:g}" if cn in ("pass",) else
                                      f"Hold {T:g} s at the corner, d_ref {ch['d']:g}" if cn == "hold" else
                                      f"Surround ({surr}), d_ref {ch['d']:g}" if cn == "hop" else LABEL[cn]),
               {"street": "swarm_street", "hold": "swarm_hold", "pass": "swarm_pass", "hop": "swarm_hop"}[cn])
              for cn, mode, T in conds]
        out.append((dict(key=key, design=name, p=p, s=s, city_seed=int(c), trial=int(t), title=title, job=job,
                         chosen={k: ch[k] for k in ("d", "mode", "T")}, median_diff=float(diff.median()),
                         n_trials=int(len(diff))), fl, stored))
    return out


def b_select():
    from citysim import lookdown_search as LS
    out = []
    for key, name, title in B_PICKS:
        spec = json.loads((SH.OUT / f"partB_{name}" / "spec.json").read_text(encoding="utf-8"))
        df, by = LS.read(f"partB_{name}", root=SH.OUT)
        jobs = {LS.job_id(j): j for js in LS.plan(spec) for j in js}
        k = f"t_r{SH.SIGHT_M:g}"
        a = df[df.condition == "swarm_hop"].set_index(["city_seed", "target_i"])[k]
        b = df[df.condition == "swarm_street"].set_index(["city_seed", "target_i"])[k]
        r = np.log(a / b.reindex(a.index)).dropna()
        dev = (r - r.median()).abs().sort_values(kind="stable")
        c, i = dev.index[0]
        p, s = spec["designs"][0]
        jid = LS.job_id(dict(p=p, s=s, city_seed=int(c), target_i=int(i)))
        out.append((dict(key=key, design=name, p=p, s=s, city_seed=int(c), i=int(i), title=title,
                         ratio=float(math.exp(r.loc[(c, i)])), median_ratio=float(math.exp(r.median()))),
                    jobs[jid], by[jid]))
    return out


def _dispatch(args):
    kind, payload = args
    return (SR.p3_worker if kind == "a" else SR.p3b_worker)(payload)


def refly(jobs=6):
    from multiprocessing import Pool
    for d in ("cities", "flights", "picks"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    work = [("a", x) for x in a_select()] + [("b", x) for x in b_select()]
    with Pool(min(jobs, len(work))) as pool:
        for r in pool.imap_unordered(_dispatch, work):
            key = r["pick"]["key"]
            if "error" in r:
                print(f"[shaped_replays] {key}: ERROR {r['error']}\n{r['trace']}", flush=True)
                continue
            for fl in r["flights"]:
                (OUT / "flights" / f"{fl['id']}.json").write_text(json.dumps(fl, separators=(",", ":"), allow_nan=False),
                                                                  encoding="utf-8")
            meta = dict(pick=r["pick"], identical=bool(r["identical"]), flights=[fl["id"] for fl in r["flights"]],
                        target=playback.rounded(r.get("target"), 2), wall_s=r.get("wall_s"))
            (OUT / "picks" / f"{key}.json").write_text(json.dumps(meta, indent=1, allow_nan=False), encoding="utf-8")
            print(f"[shaped_replays] {key}: {len(r['flights'])} flights, identical to the run: {r['identical']}",
                  flush=True)


def assemble():
    index = dict(built=time.strftime("%Y-%m-%d %H:%M"), groups=[], gallery=[])
    cities = {}

    def want(P):
        cities[Path(blockcity.path_for(P)).stem] = P

    order = ("single", "swarm_street", "swarm_hold", "swarm_pass", "swarm_hop")
    for key, name, title in A_PICKS + B_PICKS:
        path = OUT / "picks" / f"{key}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if not m["identical"]:
            print(f"[shaped_replays] {key}: NOT identical to its run; left out", flush=True)
            continue
        pk = m["pick"]
        fls = [json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8")) for fid in m["flights"]]
        fls.sort(key=lambda f: order.index(f["cond"]) if f["cond"] in order else len(order))
        want(SH.params(name, pk["city_seed"]))
        if key.startswith("s5a"):
            ch = pk["chosen"]
            about = (f"One block from Part A on {SH.DESIGNS[name][3]}, flown {len(fls)} ways from the same start, with the "
                     f"same seed and the same exit (the arrow): one drone, then the swarm in the streets, holding at the "
                     f"corner, passing round and surrounding at the d_ref step 1 chose ({ch['d']:g}). The squares are the "
                     f"buildings at the far ends of the block's other streets, filled once a shown feed has looked "
                     f"down to them. This trial's surround-minus-street difference is the nearest to the design's median "
                     f"over {pk['n_trials']} trials.")
            index["groups"].append(dict(id=key, kind="block", title=title, about=about, p=pk["p"], s=pk["s"],
                                        seed=pk["city_seed"], city=fls[0]["city"], flights=[SR._lib_entry(f) for f in fls]))
        else:
            about = (f"Part B on the scene's city: the target (star) stands on the footpath in front of a building. Each "
                     f"condition's pilot heads for the exit with the most unseen footpath ahead, from what its own feeds "
                     f"have shown (the dots), and stops at the first sight within {SH.SIGHT_M:g} m. This target's "
                     f"surround-over-street time ratio ({pk['ratio']:.2f}) is the nearest to the design's median "
                     f"({pk['median_ratio']:.2f}).")
            index["groups"].append(dict(id=key, kind="search", title=title, about=about, p=pk["p"], s=pk["s"],
                                        seed=pk["city_seed"], flights=[SR._lib_entry(f) for f in fls],
                                        city=fls[0]["city"], target=m["target"]))
    for name in SH.DESIGNS:
        P = SH.params(name, GALLERY_SEED)
        want(P)
        index["gallery"].append(dict(design=name, what=SH.DESIGNS[name][3], p=P.p, s=P.s, city=Path(blockcity.path_for(P)).stem))
    (OUT / "cities").mkdir(parents=True, exist_ok=True)
    for stem, P in cities.items():
        c = city_payload(P)
        (OUT / "cities" / f"{stem}.json").write_text(json.dumps(c, separators=(",", ":"), allow_nan=False),
                                                     encoding="utf-8")
        if stem in [g["city"] for g in index["gallery"]]:
            for g in index["gallery"]:
                if g["city"] == stem:
                    g["stats"] = c["stats"]
    (OUT / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[shaped_replays] {len(index['groups'])} groups, {len(cities)} cities -> {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="all", choices=("all", "refly", "assemble"))
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.stage in ("all", "refly"):
        refly(a.jobs)
    if a.stage in ("all", "assemble"):
        assemble()


if __name__ == "__main__":
    main()
