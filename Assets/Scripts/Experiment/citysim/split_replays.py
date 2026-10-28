"""S4 Split and Surround: the data behind the page's Cities tab -- every design's city, and the flights it replays.

    python -B -m citysim.split_replays [--stage all|refly|assemble] [--jobs 6] [--only KEY ...]
        -> experiment_results/split_study/replay/: index.json (the gallery and the flight library), cities/<id>.json,
           flights/<id>.json (citysim/playback.py's format, played by citysim/replay_view.js)

Every replica flight is the study's own. It is re-flown from its job with a tape on it (citysim.playback): Phase 1's
manoeuvres through split_probe.run_job, Phase 2's walks and searches through split.walk_job / split.search_job. The
job's scored rows must come out identical to the ones its run stored; a flight is deterministic in its seed, so any
difference is this machine's CPU (lessons.md 16), and the flight is not exported. The Unity flights are the bench's
recorded trajectories of the same walks and search routes (p2_bench, p2_routes on the bench copy), scored with the
replica's rules, as split_bench scores them.

Beside the tapes each flight carries what the viewer draws: the time each street sample was first seen on a shown
feed (the search band, 80 m), the decisions (each block's arrival, commit and candidate streets with the time each was
previewed), which drones were apart from the formation's main group (the cluster metric, every 0.5 s), and the scored
row's headline numbers. Imported only by the parent and its own workers. It imports split and split_probe, so do not
run it while a pool of theirs is going (lessons.md 18).
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
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, clutter, playback, split, split_bench, split_probe as S, streetnet  # noqa: E402
from citysim.split_report import SIGHT_M, sight_run  # noqa: E402
from citysim import geometry as geo  # noqa: E402

OUT = S.OUT_ROOT.parent / "replay"                      # experiment_results/split_study/replay
RUNS = Path(r"D:\claude_obstacle_bench\runs")           # the bench copy's runs (run_bench.ps1)
DT_MANOEUVRE, DT_LONG = 0.1, 0.2
GALLERY_SEED = 0                                        # Phase 1's first city of every design
RING_M = blockcity.STANDOFF_M

# Phase 1: (key, run, kind, variant, p, s, rule, title, about)
P1_PICKS = [
    ("p1_m3_R40_18", "p1_confirm", "M3", "R", 40, 18, "clean", "Pass round, R, p 40 · s 18",
     "Aimed at full stick through the block at the street opposite, the formation flows round both sides of the "
     "building and is one group again before the next junction. This design passed every criterion."),
    ("p1_m3_R75_18", "p1_confirm", "M3", "R", 75, 18, "straggler", "Pass round, R, p 75 · s 18",
     "A block twice as wide: the formation flows round both sides, and one drone is still apart from the others "
     "(ringed) when the next junction comes, so the reform fails. The commonest failure at p 60 and 75."),
    ("p1_m3b_R40_18", "p1_confirm", "M3b", "R", 40, 18, "clean", "Surround and exit, R, p 40 · s 18",
     "The surround Phase 2 flew: 6 s aimed through the block at the opposite street, so the formation wraps the "
     "building, then off down a side street. Watch the block's streets turn from dashed to solid as a feed previews "
     "each one: 8 of its 9 by the time the pilot turns off."),
    ("p1_m3b_R75_18", "p1_confirm", "M3b", "R", 75, 18, "short", "Surround and exit, R, p 75 · s 18",
     "The same 6 s surround round a block twice as wide: when the pilot turns off, only 3 of the block's 6 streets "
     "have been previewed, and 2 are never seen. A useful surround here needs 20 s or more."),
    ("p1_m4_R50_12", "p1_coupling", "M4", "R", 50, 12, "enveloped", "Surround hold (45 s), R, p 50 · s 12",
     "S1's centring hold on the block's centre at the design's matched spread d*: the formation spreads into a ring "
     "round the building, every drone on its edge facing outward, and holds it. Phase 1's surround map is made of "
     "these."),
    ("p1_m3_B50_18", "p1_confirm", "M3", "B", 50, 18, "contact", "Pass round, B, p 50 · s 18",
     "Variant B: passing round a block-shaped building, drones are pressed into its facade (amber): the building being "
     "passed, not the one across the street, as in nearly all of B's contacts. R's buildings, steered round as "
     "cylinders, stay near the contact bar."),
    ("p1_m3b_B50_12", "p1_confirm", "M3b", "B", 50, 12, "contact", "Surround and exit, B, p 50 · s 12",
     "Variant B with 12 m streets: the longer ring round a block-shaped building is slow to go round (2 of its 5 "
     "streets previewed in 6 s), and a drone scrapes the building as the formation turns off at its corner."),
]
# Phase 2: (key, p, s, city seed, walk / target index, title, about)
WALK_PICKS = [
    ("walk_s18", 40, 18, 101, 3, "Block-hop walk, p 40 · s 18 (with losses)",
     "One of Phase 2's chains: the same blocks and exits in every condition, 10 minutes. In this walk the surround "
     "lost two drones to a collision in the replica and four in Unity, and in Unity the hold lost two as well; the "
     "other flights lost none."),
    ("walk_s24", 40, 24, 103, 2, "Block-hop walk, p 40 · s 24",
     "A typical walk at the wider streets. Half the blocks here are plazas (too small for a building), which every "
     "condition crosses by the streets."),
]
SEARCH_PICKS = [
    ("search_s18", 40, 18, 100, 0, "Informed search, p 40 · s 18",
     "A hidden target (star); each condition's pilot picks the exit with the most unseen street nearby, from what "
     "its own feeds have shown, and stops at the first sight within 80 m. The swarm in the streets and the pass find "
     "it in about 2 minutes, the surround in 2.4, one drone in 6.5."),
    ("search_s24", 40, 24, 101, 1, "Informed search, p 40 · s 24",
     "A target the usual order does not hold for: the hold finds it first (3.3 min), one drone next (5.0), and the "
     "street-bound swarm last of the swarm conditions (6.8). Whoever happens to look down the right street first wins."),
]
COND_ORDER = ("swarm_street", "swarm_hop", "swarm_hop_p2", "swarm_pass", "swarm_hold", "single", "single_look")
# Phase 3 (the look-down pilot, citysim.lookdown / lookdown_search): one block per design flown every way, and one
# building-target search per design. Each pick is the trial (or target) nearest its design's median result.
# the runs at the page's sight range (split_report.SIGHT_M: 250 m since 2026-10-28; the 80 m runs carry no suffix)
P3_PARTA, P3_PARTB = sight_run("lookdown_A"), sight_run("lookdown_B")
P3_SINGLE = sight_run("lookdown_A1")          # the lone drone over Part A's trials (lookdown single)
P3_BLOCKS = [("p3_p40", 40.0, 18.0, "Looking down a block, p 40 · s 18"),
             ("p3_p75", 75.0, 18.0, "Looking down a block, p 75 · s 18")]
P3_SPAN = {40.0: 180.0, 50.0: 200.0, 60.0: 230.0, 75.0: 270.0}


# ============================================================================= helpers

def _strip(r):
    return json.dumps({k: v for k, v in r.items() if k != "wall_s"}, sort_keys=True, default=clutter._jsonable)


def stored_rows(path, ids):
    """{job_id: the run's stored row} for ids (an error row never displaces a good one)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("job_id") in ids and not (r["job_id"] in out and "error" not in out[r["job_id"]] and "error" in r):
            out[r["job_id"]] = r
    return out


def taped(fn, dt):
    """fn() with every split_probe.Flight taped: (fn's result, [(flight, tape)] in creation order)."""
    made = []
    base = S.Flight

    class Taped(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append((self, playback.Tape.on_flight(self, dt)))

    S.Flight = Taped
    try:
        return fn(), made
    finally:
        S.Flight = base


def first_seen_all(C, samples, r=S.PREVIEW_RANGE_M):
    """Each street sample's first sight on a shown feed: split.first_seen at the search band, within r (80 m: Phases
    1 and 2; Phase 3's picks pass the page's SIGHT_M)."""
    h, pitch = S.bands()["search"]
    return split.first_seen(C["fp"], samples, C["net"].sample_xz, h, pitch, r=r)


def cand_times(net, first, cands):
    """The time each candidate street (sid, corner) was previewed: PREVIEW_FRAC of its first PREVIEW_REACH_M seen."""
    out = []
    for sid, j in cands:
        f = np.sort(first[net.street_samples(sid, j, S.PREVIEW_REACH_M)])
        need = max(1, int(math.ceil(S.PREVIEW_FRAC * len(f))))
        out.append(float(f[need - 1]) if len(f) else math.inf)
    return out


def street_seg(net, sid, j, reach=S.PREVIEW_REACH_M):
    """The first `reach` m of street sid from its end at junction j: [x0, z0, x1, z1]."""
    a, b = net.jpos[j], net.jpos[net.other(sid, j)]
    L = float(np.linalg.norm(b - a))
    e = a + (b - a) * min(1.0, reach / max(L, 1e-9))
    return [round(float(v), 1) for v in (a[0], a[1], e[0], e[1])]


def decision(net, block, J, t0, t, cands, times, chosen=None, plaza=False, scores=None, prev=None):
    opts = []
    for k, ((sid, j), tt) in enumerate(zip(cands, times)):
        o = dict(xz=street_seg(net, sid, j), t=playback._r(tt, 1) if math.isfinite(tt) else None, sid=int(sid),
                 j=int(j))
        if chosen is not None and sid == chosen:
            o["chosen"] = True
        if scores is not None:
            o["score"] = playback._r(scores[k], 3)
        opts.append(o)
    return dict(block=int(block), at=[round(float(net.jpos[J][0]), 1), round(float(net.jpos[J][1]), 1)],
                t0=playback._r(t0, 2), t=playback._r(t, 2), opts=opts, plaza=bool(plaza),
                prev=None if prev is None or not math.isfinite(prev) else round(float(prev), 3))


def corner_of(net, block, sid):
    st = net.streets[sid]
    return st["u"] if st["u"] in net.blocks[block]["corners"] else st["v"]


def apart_series(net, samples):
    """Every 0.5 s sample: the drones outside the formation's largest group under the cluster metric (free-space
    path <= 30 m), and whether the formation is split (two groups of two or more). -> (t0, dt, masks, split spans,
    apart spans)."""
    if not samples:
        return None
    masks, split_flag, apart_flag = [], [], []
    for smp in samples:
        P = smp["P"]
        alive = np.isfinite(P).all(1)
        m = np.zeros(len(P), bool)
        sp = False
        if alive.sum() >= 2:
            lab = net.clusters(P[alive])
            vals, counts = np.unique(lab, return_counts=True)
            main = vals[int(np.argmax(counts))]
            m[np.flatnonzero(alive)[lab != main]] = True
            sp = streetnet.split_state(lab)[2]
        masks.append(m)
        split_flag.append(sp)
        apart_flag.append(bool(m.any()))
    t = np.array([s["t"] for s in samples])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else S.SAMPLE_S
    return dict(t0=round(float(t[0]), 3), dt=round(dt, 4), mask=playback.b64(playback.mask16(np.array(masks)), "<u2"),
                split=spans(t, split_flag, dt), apart=spans(t, apart_flag, dt))


def spans(t, flags, dt):
    out, start = [], None
    for tt, f in zip(t, flags):
        if f and start is None:
            start = tt
        elif not f and start is not None:
            out.append([round(float(start), 2), round(float(tt), 2)])
            start = None
    if start is not None:
        out.append([round(float(start), 2), round(float(t[-1] + dt), 2)])
    return out


def seen_b64(first):
    return playback.b64(playback.times16(first), "<u2")


def fmt_s(x):
    return "never" if x is None or not math.isfinite(x) else f"{x:.0f} s"


def wps_json(wps):
    return [[round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 2), round(float(w[4]), 2)] for w in wps]


def ring_round(poly, d, step_deg=15.0):
    """A convex polygon grown by d with rounded corners (its Minkowski sum with a disc): the formation's ring at
    STANDOFF_M round a block-shaped building."""
    p = geo.ccw(np.asarray(poly, float))
    out = []
    n = len(p)
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        e0, e1 = b - a, c - b
        n0 = np.array([e0[1], -e0[0]]) / max(np.linalg.norm(e0), 1e-9)       # outward for a CCW polygon
        n1 = np.array([e1[1], -e1[0]]) / max(np.linalg.norm(e1), 1e-9)
        a0, a1 = math.atan2(n0[1], n0[0]), math.atan2(n1[1], n1[0])
        while a1 < a0:
            a1 += 2 * math.pi
        k = max(1, int(math.ceil(math.degrees(a1 - a0) / step_deg)))
        for th in np.linspace(a0, a1, k + 1):
            out.append(b + d * np.array([math.cos(th), math.sin(th)]))
    return np.array(out)


# ============================================================================= cities

def city_payload(variant, p, s, seed, anatomy=False):
    C = S.city(variant, float(p), float(s), int(seed))
    L, net = C["L"], C["net"]
    blocks = sorted(C["blocks"], key=lambda b: b["id"])
    assert [b["id"] for b in blocks] == list(range(len(blocks))), "block ids are not 0..n-1"
    polys = [blockcity.unflat(b["kerb_xz"]) if len(b["kerb_xz"]) >= 6 else np.zeros((0, 2)) for b in blocks]
    flags = [(1 if net.blocks.get(b["id"], {}).get("interior") else 0) | (2 if b["building"] < 0 else 0) for b in blocks]
    blds = [blockcity.building_poly(x) for x in L.boxes]
    cyl = None
    if variant == "R":
        cyl = [[*((q.max(0) + q.min(0)) / 2), blockcity.cylinder_radius(x)] for q, x in zip(blds, L.boxes)]
    streets = [[*net.jpos[st["u"]], *net.jpos[st["v"]]] for st in net.streets]
    inter = [b for b in net.blocks.values() if b["interior"]]
    cen = {b["id"]: b["centre"] for b in net.blocks.values()}
    nd = [float(np.linalg.norm(cen[st["blocks"][0]] - cen[st["blocks"][1]])) for st in net.streets
          if st["interior"] and len(st["blocks"]) == 2 and all(b in cen for b in st["blocks"])]
    sides = [np.sort(q.max(0) - q.min(0))[::-1] for q, b in zip(blds, L.boxes)] if variant == "R" else []
    rec = L.layout
    stats = dict(blocks=int(sum(1 for q in polys if len(q))), interior=len(inter), buildings=len(L.boxes),
                 plazas=int(rec["n_plazas"]), plaza_share_interior=round(sum(1 for b in inter if b["building"] < 0) /
                                                                         max(len(inter), 1), 3),
                 alpha=rec["alpha_achieved"], block_share=rec["block_share"],
                 block_m=round(float(np.median([math.sqrt(abs(geo.area(b["kerb"]))) for b in inter])), 1),
                 neighbour_m=round(float(np.median(nd)), 1),
                 street_m=round(float(np.median([st["length"] for st in net.streets if st["interior"]])), 1),
                 building_m=[round(float(np.median([x[0] for x in sides])), 1),
                             round(float(np.median([x[1] for x in sides])), 1)] if sides else None,
                 cyl_r=round(float(np.median([c[2] for c in cyl])), 1) if cyl else None,
                 cyl_gap_median=rec.get("cyl_gap_median"), cyl_gap_ge6=rec.get("cyl_gap_frac_ge6"),
                 facade_gap_median=rec.get("facade_gap_median"), d_star_pred=rec.get("d_star_pred_median"))
    block_bld = np.array([b["building"] for b in blocks], "<i2")
    out = playback.city_json(Path(blockcity.path_for(blockcity.Params(variant, float(p), float(s), int(seed)))).stem,
                           blockcity.ARENA, polys, blds, flags, cyl, net.sample_xz, streets,
                           variant=variant, p=float(p), s=float(s), seed=int(seed), stats=stats,
                           block_building=playback.b64(block_bld, "<i2"))
    if anatomy:
        out["anatomy"] = anatomy_of(C, variant)
    return out


def anatomy_of(C, variant):
    """One typical interior block with a building and its neighbour across a street, for the page's annotated
    figure: the block nearest the median area among those whose every neighbour also has a building."""
    net, L = C["net"], C["L"]
    cand = [b for b in net.interior_blocks()]
    areas = np.array([abs(geo.area(b["kerb"])) for b in cand])
    order = np.argsort(np.abs(areas - np.median(areas)))
    for k in order:
        B = cand[int(k)]
        for sid in B["streets"]:
            st = net.streets[sid]
            nb = [x for x in st["blocks"] if x != B["id"]]
            if not nb or nb[0] not in net.blocks or net.blocks[nb[0]]["building"] < 0 or not st["interior"]:
                continue
            N = net.blocks[nb[0]]
            b = L.boxes[B["building"]]
            poly = blockcity.building_poly(b)
            c_in, r_in = blockcity.chebyshev(B["kerb"])
            out = dict(block=int(B["id"]), neighbour=int(N["id"]),
                       street=[round(float(v), 2) for v in (*net.jpos[st["u"]], *net.jpos[st["v"]])],
                       centre=[round(float(v), 2) for v in B["centre"]], nb_centre=[round(float(v), 2) for v in N["centre"]],
                       incircle=[round(float(c_in[0]), 2), round(float(c_in[1]), 2), round(float(r_in), 2)])
            if variant == "R":
                bc = (poly.max(0) + poly.min(0)) / 2
                r = blockcity.cylinder_radius(b)
                out.update(cylinder=[round(float(bc[0]), 2), round(float(bc[1]), 2), round(r, 2)],
                           ring=[round(float(bc[0]), 2), round(float(bc[1]), 2), round(r + RING_M, 2)],
                           d_pred=round(2 * math.pi * (r + RING_M) / 100.0, 2))
            else:
                ring = ring_round(poly, RING_M)
                out.update(ring_poly=[round(float(v), 2) for v in ring.ravel()],
                           d_pred=round((geo.perimeter(poly) + 2 * math.pi * RING_M) / 100.0, 2))
            return out
    return None


# ============================================================================= Phase 1: the manoeuvres

def p1_select():
    """[(pick, job, stored row)] for P1_PICKS: each rule's trial nearest its design's median time."""
    out = []
    runs = {}
    for key, run, kind, v, p, s, rule, title, about in P1_PICKS:
        if run not in runs:
            runs[run] = S.read(run)
        df = runs[run]
        if "error" in df:
            df = df[df["error"].isna()]
        if "skipped" in df:
            df = df[df["skipped"].isna() | (df["skipped"] != True)]                        # noqa: E712
        g = df[(df.kind == kind) & (df.variant == v) & (df.p == p) & (df.s == s)]
        if kind == "M4":
            dstar = S.coupling_map(df).get((v, float(p), float(s)))
            g = g[np.isclose(g.d_block.astype(float), dstar)]
        tcol = {"M3": "pass_s", "M3b": "hop_s", "M4": "t_envelop"}[kind]
        med = float(np.nanmedian(g[tcol].astype(float)))
        okr = g.reform_ok.fillna(False).astype(bool) if "reform_ok" in g else np.zeros(len(g), bool)
        prev = g.preview_search.astype(float) if "preview_search" in g else np.full(len(g), np.nan)
        if rule == "clean":
            c = g[(g.lost == 0) & (g.contacts == 0) & okr & (prev >= (0.9 if kind == "M3" else 0.8))]
        elif rule == "straggler":
            c = g[(g.lost == 0) & (g.contacts == 0) & ~okr & g.reached_next.fillna(False).astype(bool)]
        elif rule == "short":
            c = g[(g.lost == 0) & (g.contacts == 0) & (prev <= 0.5)]
        elif rule == "contact":
            c = g[(g.lost == 0) & (g.contacts >= 1)]
        elif rule == "enveloped":
            c = g[(g.lost == 0) & (g.contacts == 0) & g.enveloped.fillna(False).astype(bool)]
        else:
            raise ValueError(rule)
        if not len(c):
            print(f"[split_replays] {key}: no trial meets {rule!r}; skipped", flush=True)
            continue
        r = c.iloc[int(np.argmin(np.abs(c[tcol].astype(float).fillna(1e9).values - med)))]
        job = dict(kind=kind, variant=v, p=float(p), s=float(s), city_seed=int(r.city_seed), trial=int(r.trial),
                   setting=r.setting if isinstance(r.setting, dict) else json.loads(r.setting))
        assert S.job_id(job) == r.job_id, (S.job_id(job), r.job_id)
        stored = stored_rows(S.OUT_ROOT / run / "results.jsonl", {r.job_id})[r.job_id]
        out.append((dict(key=key, run=run, kind=kind, v=v, p=p, s=s, rule=rule, title=title, about=about), job,
                    stored))
    return out


def p1_worker(args):
    pick, job, stored = args
    t0 = time.time()
    try:
        res, made = taped(lambda: S.run_job(job), DT_MANOEUVRE)
        same = _strip(res) == _strip(stored)
        f, tape = made[0]
        C = S.city(job["variant"], job["p"], job["s"], job["city_seed"])
        net = C["net"]
        first = first_seen_all(C, f.samples)
        block, entrance = int(stored["block"]), int(stored["entrance"])
        J = corner_of(net, block, entrance)
        tr = dict(block=block, entrance=entrance, J=J)
        cands = S.candidates(net, tr)
        times = cand_times(net, first, cands)
        kind = job["kind"]
        done = f.done_t
        chosen, t_next = None, math.nan
        if kind == "M3":
            op = net.opposite_exit(block, entrance, J)
            chosen, t_next = (op[0] if op else None), done[3] if len(done) > 3 else math.nan
        elif kind == "M3b":
            V = np.array(f.wps[3][:2])
            near = [c for c in cands if np.linalg.norm(net.jpos[c[1]] - V) < 0.5]
            chosen, t_next = (near[0][0] if near else None), done[4] if len(done) > 4 else math.nan
        t_commit = done[2] if len(done) > 2 else math.nan
        dec = decision(net, block, J, done[1], t_commit, cands, times, chosen,
                       prev=stored.get("preview_search"))
        st = dict(S.DEFAULT_SETTING, **job["setting"])
        summary = dict(lost=stored.get("lost"), contacts=stored.get("contacts"), d_block=stored.get("d_block"),
                       d_street=st["d_street"], preview=stored.get("preview_search"), reform_ok=stored.get("reform_ok"),
                       reform_s=stored.get("reform_s"), pass_s=stored.get("pass_s"), hop_s=stored.get("hop_s"),
                       t_hop=stored.get("t_hop"), t_envelop=stored.get("t_envelop"), enveloped=stored.get("enveloped"),
                       r0_coh=st["r0_coh"], mode=st["m3b_mode"] if kind == "M3b" else None,
                       time_price=stored.get("time_price"), street_s=stored.get("street_s"),
                       kills=stored.get("kills"))
        fl = playback.flight_json(tape.record(), id=pick["key"], title=pick["title"], about=pick["about"],
                                city=Path(blockcity.path_for(blockcity.Params(job["variant"], job["p"], job["s"],
                                                                             job["city_seed"]))).stem,
                                source="replica", kind=kind, cond={"M3": "swarm_pass", "M3b": "swarm_hop",
                                                                   "M4": "swarm_hop"}[kind],
                                job_id=stored["job_id"], identical=bool(same), wps=wps_json(f.wps),
                                decisions=[dec], events=dict(next=playback._r(t_next, 2)),
                                apart=apart_series(net, f.samples), seen=seen_b64(first), summary=playback.rounded(summary, 3))
        return dict(pick=pick, flights=[fl], identical=same, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, error=repr(e), trace=traceback.format_exc()[-3000:])


# ============================================================================= Phase 2: the walks

def walk_hop_decisions(cond, C, chain, st, dfn, wps, marks, done, first, hops_scored):
    """Per hop: arrival at its entrance corner, the commit, the candidate streets and when each was previewed."""
    net = C["net"]
    out = []
    start = len(split.approach(net, chain["approach"], st["d_street"]))
    for k, (mk, h) in enumerate(zip(marks, chain["hops"])):
        if mk["commit"] >= len(done) or not math.isfinite(done[mk["commit"]]):
            break
        arrive = start - 1
        t0 = done[arrive] if arrive < len(done) else math.nan
        cands = S.candidates(net, dict(block=h["block"], entrance=h["entrance"], J=h["J"]))
        times = cand_times(net, first, cands)
        sc = hops_scored[k] if k < len(hops_scored) else {}
        out.append(decision(net, h["block"], h["J"], t0, done[mk["commit"]], cands, times, h["exit"], mk["plaza"],
                            prev=sc.get("preview_search")))
        start = mk["next"] + 1
    return out


def walk_worker(args):
    pick, job, stored = args
    t0 = time.time()
    try:
        res, made = taped(lambda: split.walk_job(job), DT_LONG)
        same = _strip(res) == _strip(stored)
        spec, st = job["spec"], dict(S.DEFAULT_SETTING, **job["spec"]["setting"])
        C = split.context(job["variant"], job["p"], job["s"], job["city_seed"])
        net = C["net"]
        rng = np.random.default_rng(S.hash_seed(dict(kind="walk", variant="chain", p=job["p"], s=job["s"],
                                                     city_seed=job["city_seed"], trial=job["walk"])))
        rng.integers(1 << 30)
        chain = split.block_chain(net, rng)
        city = Path(blockcity.path_for(blockcity.Params(job["variant"], job["p"], job["s"], job["city_seed"]))).stem
        flights, it = [], iter(made)
        for row in res["rows"]:
            if "same_as" in row:
                continue
            f, tape = next(it)
            if row["spread"] != "profile":
                continue
            cond = row["condition"]
            dfn = split.block_spread_fn(spec, st, net, C["L"], job["variant"], job["p"], job["s"], "profile")
            wps, marks = split.condition_wps(cond, C, chain, st, dfn)
            assert [tuple(w) for w in wps] == list(f.wps), f"{cond}: the flight's waypoints are not the chain's"
            first = first_seen_all(C, f.samples)
            decs = walk_hop_decisions(cond, C, chain, st, dfn, wps, marks, f.done_t, first, row["hops"])
            hh = [h for h in row["hops"] if not h.get("plaza")]
            summary = dict(blocks=row["hops_done"], preview=float(np.mean([h["preview_search"] for h in hh])) if hh
                           else None, street_km=row.get("street_seen_km"), find80=row.get("find_rmst_any_r80"),
                           lost=row["lost"], contacts=row["contacts"], kills=row.get("kills"),
                           reform=row.get("reform_ok_share") if "reform_ok_share" in row else _reform_share(row),
                           feeds=row.get("feeds_mean"), a1=row.get("corr_a1_linked"))
            flights.append(playback.flight_json(
                tape.record(), id=f"{pick['key']}_{cond}_replica", group=pick["key"], city=city, source="replica",
                kind="walk", cond=cond, job_id=stored["job_id"], identical=bool(same), wps=wps_json(f.wps),
                decisions=decs, apart=apart_series(net, f.samples) if cond in split.SWARM else None,
                seen=seen_b64(first), summary=playback.rounded(summary, 3)))
        return dict(pick=pick, flights=flights, identical=same, chain=chain_json(net, chain), wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, error=repr(e), trace=traceback.format_exc()[-3000:])


def _reform_share(row):
    r = [h["reform_ok"] for h in row.get("hops", []) if "reform_ok" in h and not h.get("plaza")]
    return float(np.mean(r)) if r else None


def chain_json(net, chain):
    return [dict(block=int(h["block"]), J=[round(float(v), 1) for v in net.jpos[h["J"]]],
                 exit=street_seg(net, h["exit"], h["V"]), plaza=bool(h["plaza"])) for h in chain["hops"]]


# ============================================================================= Phase 2: the searches

def search_worker(args):
    """One search target re-flown with tapes, and every decision the informed pilot took logged on the way: the
    exits it weighed with their scores (split.choose_exit's own rule, recomputed beside it, read-only) and what it
    chose."""
    pick, job, stored = args
    t0 = time.time()
    log, made = [], []
    orig = split.choose_exit

    def recording(net, know, B, entrance, J, rng, min_unseen_m=split.FRONTIER_MIN_M):
        f = made[-1][0]
        opts = split.exit_options(net, B, entrance, J) or split.fallback_exits(net, B, entrance, J)
        scores = []
        for sid, V, F, nb in opts:
            js, _ = net.route(J, V)
            travel = (net.route_length(js) if js else 0.0) + net.streets[sid]["length"]
            scores.append(know.unseen_m({sid} | set(net.j_streets[F])) / max(travel, 1.0))
        out = orig(net, know, B, entrance, J, rng, min_unseen_m)
        log.append(dict(flight=len(made) - 1, t=float(f.sim.t), B=int(B), entrance=int(entrance), J=int(J),
                        n_wps=len(f.wps), opts=[(int(o[0]), int(o[1])) for o in opts], scores=scores,
                        choice=None if out is None else (int(out[0]), int(out[1]), bool(out[4]))))
        return out

    base = S.Flight

    class Taped(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append((self, playback.Tape.on_flight(self, DT_LONG)))

    try:
        S.Flight, split.choose_exit = Taped, recording
        try:
            res = split.search_job(job)
        finally:
            S.Flight, split.choose_exit = base, orig
        made_holder = made
        same = _strip(res) == _strip(stored)
        spec, st = job["spec"], dict(S.DEFAULT_SETTING, **job["spec"]["setting"])
        C = split.context(job["variant"], job["p"], job["s"], job["city_seed"])
        net = C["net"]
        city = Path(blockcity.path_for(blockcity.Params(job["variant"], job["p"], job["s"], job["city_seed"]))).stem
        dfn = split.block_spread_fn(spec, st, net, C["L"], job["variant"], job["p"], job["s"], "profile")
        flights = []
        for fi, (row, (f, tape)) in enumerate(zip(res["rows"], made_holder)):
            if row["basis"] != "any":
                continue
            cond = row["condition"]
            first = first_seen_all(C, f.samples)
            decs = []
            for d in [x for x in log if x["flight"] == fi]:
                c = split.plaza_cond(cond, net, d["B"])
                pre = split.hop_pre(c, C, dict(block=d["B"], J=d["J"], entrance=d["entrance"]), st, dfn(d["B"]))
                arrive = d["n_wps"] - len(pre) - 1
                t0_ = f.done_t[arrive] if 0 <= arrive < len(f.done_t) else math.nan
                cands = [(sid, V) for sid, V in d["opts"]]
                times = cand_times(net, first, cands)
                chosen = d["choice"][0] if d["choice"] else None
                dd = decision(net, d["B"], d["J"], t0_, d["t"], cands, times, chosen,
                              plaza=net.blocks[d["B"]]["building"] < 0, scores=d["scores"])
                dd["wp"], dd["pre_n"] = d["n_wps"], len(pre)       # where Unity's flight of the route decided
                dd["blind"] = bool(d["choice"] and not d["choice"][2])
                decs.append(dd)
            summary = dict(found80=row.get("found_r80"), found200=row.get("found_r200"), found40=row.get("found_r40"),
                           t_end=row["t_end"], decisions=row["decisions"], blind=row["blind_decisions"],
                           path_m=row["path_m"], lost=row["lost"], contacts=row["contacts"],
                           streets=row["street_seen_share"])
            flights.append(playback.flight_json(
                tape.record(), id=f"{pick['key']}_{cond}_replica", group=pick["key"], city=city, source="replica",
                kind="search", cond=cond, job_id=stored["job_id"], identical=bool(same), wps=wps_json(f.wps),
                decisions=decs, apart=apart_series(net, f.samples) if cond in split.SWARM else None,
                seen=seen_b64(first), target=[round(float(x), 1) for x in res["target"]],
                found=dict(r40=row.get("found_r40"), r80=row.get("found_r80"), r200=row.get("found_r200")),
                summary=playback.rounded(summary, 3)))
        return dict(pick=pick, flights=flights, identical=same, target=res["target"], log=log,
                    wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, error=repr(e), trace=traceback.format_exc()[-3000:])


# ============================================================================= Unity's flights

def _run_dir(prefix, ident, n):
    d = RUNS / f"s4_{prefix}_{ident}_n{n}"
    return d if (d / "config.json").exists() else None


def unity_walk_flights(pick, replica):
    """The bench's flights of a walk: every bench-flyable condition, scored as split_bench.gate_walks scores them."""
    import pandas as pd
    from citysim import tierc
    key, p, s, seed, walk = pick["key"], float(pick["p"]), float(pick["s"]), int(pick["seed"]), int(pick["i"])
    spec = split.load_spec(S.OUT_ROOT.parent / "split" / "spec.json")
    st = dict(S.DEFAULT_SETTING, **spec["setting"])
    C = split.context(spec["variant"], p, s, seed)
    net = C["net"]
    ident = blockcity.Params(spec["variant"], p, s, seed).ident()
    rng = np.random.default_rng(S.hash_seed(dict(kind="walk", variant="chain", p=p, s=s, city_seed=seed, trial=walk)))
    rng.integers(1 << 30)
    chain = split.block_chain(net, rng)
    dfn = split.block_spread_fn(spec, st, net, C["L"], spec["variant"], p, s, "profile")
    gate = pd.read_csv(S.OUT_ROOT.parent / "p2_bench" / "bench" / "walk_gate_flights.csv")
    out = []
    for cond in split_bench.WALK_CONDITIONS:
        n = 1 if cond == "single" else S.N_DRONES
        run = _run_dir("p2_bench", ident, n)
        if run is None:
            print(f"[split_replays] {key}: no bench run for {cond}", flush=True)
            continue
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        idx = next((i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{cond}_{walk}"), None)
        traj = run / f"traj_{cfg['paramSets'][0]['label']}_{idx}.csv" if idx is not None else None
        if traj is None or not traj.exists():
            continue
        tr = tierc.read_traj(traj)
        single = cond == "single"
        smp = split_bench.bench_samples(tr, single=single)
        wps, marks = split.condition_wps(cond, C, chain, st, dfn)
        done = split_bench.wp_times(smp, len(wps))
        first = first_seen_all(C, smp)
        decs = walk_hop_decisions(cond, C, chain, st, dfn, wps, marks, done, first, [])
        for d in decs:                    # Unity's own preview at each commit, from its own first sights
            ok = [o["t"] is not None and d["t"] is not None and o["t"] <= d["t"] + 1e-6 for o in d["opts"]]
            d["prev"] = round(float(np.mean(ok)), 3) if ok else None
        res = {f["scenario"]: f for f in split_bench.bench.read_results(run)[3]}
        g = gate[(gate.p == p) & (gate.s == s) & (gate.seed == seed) & (gate.walk == walk) & (gate.condition == cond)]
        g = g.iloc[0] if len(g) else None
        summary = dict(blocks=None if g is None else int(g.bench_hops), preview=None if g is None else g.bench_preview,
                       street_km=None if g is None else g.bench_street_km, find80=None if g is None else g.bench_find80,
                       lost=int(tr["alive"][0].sum() - tr["alive"][-1].sum()),
                       contacts=int(res.get(idx, {}).get("contacts", 0)),
                       deaths=res.get(idx, {}).get("deaths") or {})
        out.append(playback.flight_json(
            playback.unity_tape(tr, DT_LONG, single), id=f"{key}_{cond}_unity", group=key, city=ident, source="unity",
            kind="walk", cond=cond, run=run.name, scenario=int(idx), wps=wps_json(wps), decisions=decs,
            apart=apart_series(net, smp) if not single else None, seen=seen_b64(first),
            summary=playback.rounded(summary, 3)))
    return out


def unity_search_flights(pick, rep):
    """The bench's open-loop flights of a search's routes: the waypoints each replica flight's decisions laid."""
    from citysim import tierc
    key, p, s, seed, ti = pick["key"], float(pick["p"]), float(pick["s"]), int(pick["seed"]), int(pick["i"])
    spec = split.load_spec(S.OUT_ROOT.parent / "split_search" / "spec.json")
    C = split.context(spec["variant"], p, s, seed)
    net = C["net"]
    h, pitch = S.bands()["search"]
    ident = blockcity.Params(spec["variant"], p, s, seed).ident()
    routes = json.loads((S.OUT_ROOT.parent / "p2_routes" / "bench" / "routes.json").read_text(encoding="utf-8"))
    r = next((x for x in routes if (float(x["p"]), float(x["s"]), int(x["city_seed"]), int(x["target_i"])) ==
              (p, s, seed, ti)), None)
    if r is None:
        return []
    target = np.array(r["target"], float)
    by_cond = {f["cond"]: f for f in rep}
    out = []
    for rt in r["routes"]:
        cond = rt["condition"]
        n = 1 if cond == "single" else S.N_DRONES
        run = _run_dir("p2_routes", ident, n)
        if run is None:
            continue
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        idx = next((i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"{cond}_{ti}"), None)
        traj = run / f"traj_{cfg['paramSets'][0]['label']}_{idx}.csv" if idx is not None else None
        if traj is None or not traj.exists():
            continue
        tr = tierc.read_traj(traj)
        single = cond == "single"
        smp = split_bench.bench_samples(tr, single=single)
        know = split.Knowledge(net, C["fp"], h, pitch, target, "any", single)
        know.update(smp)
        wps = rt["wps"]
        done = split_bench.wp_times(smp, len(wps))
        decs = []
        for d in (by_cond.get(cond) or {}).get("decisions", []):
            k, pre_n = d.get("wp"), d.get("pre_n", 0)
            if k is None:
                continue
            # the replica decided once waypoint k-1 was done (it had arrived at the corner at k-1-pre_n); Unity flew
            # the same waypoints
            dd = dict(d, opts=[dict(o) for o in d["opts"]])
            at = lambda i: playback._r(done[i], 2) if 0 <= i < len(done) and math.isfinite(done[i]) else None  # noqa
            dd["t"], dd["t0"] = at(k - 1), at(k - 1 - pre_n)
            if dd["t"] is None:
                break
            decs.append(dd)
        first = know.first
        for dd in decs:
            for o, tt in zip(dd["opts"], cand_times(net, first, [(o["sid"], o["j"]) for o in dd["opts"]])):
                o["t"] = playback._r(tt, 1) if math.isfinite(tt) else None
        found = {f"r{rr:g}": (round(v, 2) if math.isfinite(v) else None) for rr, v in know.found.items()}
        res = {f["scenario"]: f for f in split_bench.bench.read_results(run)[3]}
        summary = dict(found80=found.get("r80"), found200=found.get("r200"), found40=found.get("r40"),
                       t_end=round(float(tr["t"][-1]), 2), lost=int(tr["alive"][0].sum() - tr["alive"][-1].sum()),
                       contacts=int(res.get(idx, {}).get("contacts", 0)),
                       deaths=res.get(idx, {}).get("deaths") or {})
        out.append(playback.flight_json(
            playback.unity_tape(tr, DT_LONG, single), id=f"{key}_{cond}_unity", group=key, city=ident, source="unity",
            kind="search", cond=cond, run=run.name, scenario=int(idx), wps=wps_json(wps), decisions=decs,
            apart=apart_series(net, smp) if not single else None, seen=seen_b64(first),
            target=[round(float(x), 1) for x in target], found=dict(r40=found.get("r40"), r80=found.get("r80"),
                                                                       r200=found.get("r200")),
            summary=playback.rounded(summary, 3)))
    return out


# ============================================================================= Phase 3: the look-down pilot

def p3_flight_list(p, s):
    """(the chosen surround, [(cond, mode, T, d, title, viewer condition)]): the block flown in the street, held at
    the corner and passed round at the chosen spread, surrounded at Phase 2's setting (aim and turn 6 s at Phase 1's
    d*, Part B's swarm_hop_p2) and at the setting lookdown_search.choose's rule picks (applied here to a design Part B
    does not fly too). Where the two surrounds coincide, the second is flown at the widest spread instead."""
    from citysim import lookdown as LD, lookdown_search as LS
    ch = LS.choose(P3_PARTA, designs=[(p, s)])[f"{p:g}|{s:g}"]
    grid = np.array(LD.D_GRID)
    d_p2 = float(grid[np.argmin(np.abs(grid - LS.P2_DSTAR[(p, s)]))])      # d* on Part A's grid
    d, mode, T = ch["d"], ch["mode"], ch["T"]
    how = "aim and turn" if mode == "aim" else "firm hold"
    fl = [("street", None, None, d, "In the street, d_ref 1.0 throughout", "swarm_street"),
          ("hold", None, T, d, f"Hold {T:g} s at the corner, d_ref {d:.1f}", "swarm_hold"),
          ("pass", None, None, d, f"Pass round, d_ref {d:.1f}", "swarm_pass"),
          ("hop", "aim", 6.0, d_p2, f"Phase 2's surround: aim and turn 6 s, d_ref {d_p2:.1f} (d*)", "swarm_hop_p2")]
    if (mode, T, d) == ("aim", 6.0, d_p2):
        d = float(grid.max())
        fl.append(("hop", mode, T, d, f"Surround ({how}) {T:g} s at the widest spread, d_ref {d:.1f}", "swarm_hop"))
    else:
        fl.append(("hop", mode, T, d, f"Surround ({how}) {T:g} s, d_ref {d:.1f}: the rule's pick", "swarm_hop"))
    return ch, fl


def p3_select():
    """[(pick, flight list, {d: stored Part A job row})]: per P3 block, the trial whose looked-down difference
    (the chosen surround minus the swarm in the street) is nearest the design's median. When the lone drone's run
    exists, its stored row for that trial is under the key "single"."""
    from citysim import lookdown as LD
    df, by = LD.read(P3_PARTA)
    by1 = LD.read(P3_SINGLE)[1] if (S.OUT_ROOT.parent / P3_SINGLE / "results.jsonl").exists() else {}
    out = []
    for key, p, s, title in P3_BLOCKS:
        ch, fl = p3_flight_list(p, s)
        g = df[(df.p == p) & (df.s == s)]
        hop = g[(g.cond == "hop") & (g["mode"] == ch["mode"]) & (g["T"] == ch["T"]) & (g.d == ch["d"])]
        st = g[(g.cond == "street") & (g.d == ch["d"])]
        diff = (hop.set_index(["city_seed", "trial"]).look_c -
                st.set_index(["city_seed", "trial"]).look_c).dropna()
        dev = (diff - diff.median()).abs().sort_values(kind="stable")
        c, t = dev.index[0]
        stored = {}
        for d in {x[3] for x in fl}:
            jid = LD.job_id(dict(p=p, s=s, city_seed=int(c), trial=int(t), d=float(d)))
            stored[float(d)] = by[jid]
        jid1 = LD.job_id(dict(p=p, s=s, city_seed=int(c), trial=int(t), single=True))
        if jid1 in by1:
            stored["single"] = by1[jid1]
        out.append((dict(key=key, p=p, s=s, city_seed=int(c), trial=int(t), title=title, chosen=ch,
                         median_diff=float(diff.median()), n_trials=int(len(diff))), fl, stored))
    return out


def p3_worker(args):
    """One Part A trial re-flown at each spread its flights need, every condition taped; the listed ones exported."""
    pick, fl, stored = args
    from citysim import lookdown as LD
    t0 = time.time()
    try:
        p, s, c, t = pick["p"], pick["s"], pick["city_seed"], pick["trial"]
        C = LD.city_targets(S.city("R", p, s, c))
        net = C["net"]
        rng = np.random.default_rng(LD.trial_seed(p, s, c, t))
        rng.integers(1 << 30)
        tr = LD.trial_geometry(C, rng)
        cands = S.candidates(net, tr)
        conds = LD.conditions(LD.T_GRID)
        city = Path(blockcity.path_for(blockcity.Params("R", p, s, c))).stem
        flights, same = [], True

        def export(f, tape, row, res, cond, mode, T, d, title, vcond):
            single = cond in ("single", "single_look")
            t_J, t_c, t_F = row["t_J"], row["t_commit"], row["t_next"]
            rel = (lambda x: None if x is None or t_J is None else round(t_J + x, 2))  # noqa: E731
            opts = []
            for (sid, V), ex in zip(cands, row["exits"]):
                assert ex["sid"] == sid
                tg, _ = LD.end_targets(C, sid, V)
                opts.append(dict(xz=street_seg(net, sid, V), t=rel(ex["ent"]), te=rel(ex["look"]), tf=rel(ex["end"]),
                                 end=tg.mean(0).round(1).tolist() if tg is not None and len(tg) else None,
                                 sid=int(sid), j=int(V), **({"chosen": True} if sid == tr["exit"] else {})))
            dec = dict(block=int(tr["block"]), at=[round(float(v), 1) for v in net.jpos[tr["J"]]], t0=t_J,
                       t=t_c, opts=opts, plaza=False, prev=playback._r(row["ent_c"], 3),
                       look=playback._r(row["look_c"], 3))
            by = (lambda key, tt: sum(o[key] is not None and tt is not None and o[key] <= tt + 1e-6  # noqa: E731
                                      for o in opts))
            n, n_end = len(opts), sum(o["end"] is not None for o in opts)
            about = (f"By its decision at {fmt_s(t_c)} it had looked down {by('te', t_c)} of the block's {n_end} "
                     f"exits with a building at the end (entered {by('t', t_c)} of {n}), and "
                     f"{by('te', t_F)} by the next junction, {fmt_s(row['hop_s'])} after arriving at the block.")
            summary = dict(d_block=row["d_block"], d_street=row["d_street"], T=T, mode=mode, n_exits=n,
                           look_c=row["look_c"], ent_c=row["ent_c"], end_c=row["end_c"], look_F=row["look_F"],
                           look_c_far=row["look_c_far"], hop_s=row["hop_s"], t80_look=row["t80_look"],
                           lost=row["lost"], contacts=row["contacts"], reform_ok=row.get("reform_ok"),
                           reform_s=row.get("reform_s"), enveloped=row.get("enveloped"),
                           t_envelop=row.get("t_envelop"), kills=row.get("kills"))
            fid = (f"{pick['key']}_{cond}" if cond == "street" or single else
                   f"{pick['key']}_{cond}{'_' + mode if mode else ''}{f'_{T:g}' if T else ''}_d{d:g}")
            flights.append(playback.flight_json(
                tape.record(), id=fid, title=title, about=about, group=pick["key"], city=city, source="replica",
                kind="A", cond=vcond, job_id=res["job_id"], identical=bool(same), wps=wps_json(f.wps),
                decisions=[dec], events=dict(next=t_F), apart=None if single else apart_series(net, f.samples),
                seen=seen_b64(first_seen_all(C, f.samples, SIGHT_M)), span=P3_SPAN.get(p, 240.0),
                summary=playback.rounded(summary, 3)))

        for d in sorted({x[3] for x in fl}):
            j = dict(p=p, s=s, city_seed=c, trial=t, d=float(d), T_grid=list(LD.T_GRID), look_range=SIGHT_M)
            res, made = taped(lambda: LD.run_job(j), DT_MANOEUVRE)
            same = same and _strip(res) == _strip(stored[float(d)])
            for cond, mode, T, _, title, vcond in [x for x in fl if x[3] == d]:
                k = conds.index((cond, mode, T))
                export(*made[k], res["rows"][k], res, cond, mode, T, d, title, vcond)
        if "single" in stored:                    # the lone drone over the same block (lookdown_A1), first in the list
            j1 = dict(p=p, s=s, city_seed=c, trial=t, single=True, look_range=SIGHT_M)
            res1, made1 = taped(lambda: LD.run_job(j1), DT_MANOEUVRE)
            same = same and _strip(res1) == _strip(stored["single"])
            k = [x[0] for x in LD.SINGLE_CONDS].index("single")
            export(*made1[k], res1["rows"][k], res1, "single", None, None, None, "One drone, in the street", "single")
        return dict(pick=pick, flights=flights, identical=same, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, error=repr(e), trace=traceback.format_exc()[-3000:])


def p3b_select():
    """[(pick, job, stored row)]: per Part B design, the target whose surround / street time ratio (80 m) is nearest
    the design's median."""
    from citysim import lookdown_search as LS
    spec = json.loads((S.OUT_ROOT.parent / P3_PARTB / "spec.json").read_text(encoding="utf-8"))
    df, by = LS.read(P3_PARTB)
    jobs = {LS.job_id(j): j for js in LS.plan(spec) for j in js}
    out = []
    for p, s in [tuple(x) for x in spec["designs"]]:
        g = df[(df.p == p) & (df.s == s)]
        a = g[g.condition == "swarm_hop"].set_index(["city_seed", "target_i"])[f"t_r{SIGHT_M:g}"]
        b = g[g.condition == "swarm_street"].set_index(["city_seed", "target_i"])[f"t_r{SIGHT_M:g}"]
        r = np.log(a / b.reindex(a.index)).dropna()
        dev = (r - r.median()).abs().sort_values(kind="stable")
        c, k = dev.index[0]
        jid = LS.job_id(dict(p=p, s=s, city_seed=int(c), target_i=int(k)))
        out.append((dict(key=f"p3b_p{p:g}", p=p, s=s, city_seed=int(c), i=int(k),
                         title=f"Search for a goal at a building, p {p:g} · s {s:g}",
                         ratio=float(math.exp(r.loc[(c, k)])), median_ratio=float(math.exp(r.median()))),
                    jobs[jid], by[jid]))
    return out


def p3b_worker(args):
    """One Part B target re-flown with tapes; every decision the pilot took logged beside the real call, and its
    footpath map kept for the flight's seen overlay."""
    pick, job, stored = args
    from citysim import lookdown as LD, lookdown_search as LS
    t0 = time.time()
    log, made, knows = [], [], []
    base, orig_choose, orig_know = S.Flight, LS.choose_exit, LS.FootKnowledge

    class Taped(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append((self, playback.Tape.on_flight(self, DT_LONG)))

    class Know(orig_know):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            knows.append(self)

    def recording(C, know, B, entrance, J, rng):
        f = made[-1][0]
        net = C["net"]
        opts = split.exit_options(net, B, entrance, J) or split.fallback_exits(net, B, entrance, J)
        scores = []
        for sid, V, F, nb in opts:
            js, _ = net.route(J, V)
            travel = (net.route_length(js) if js else 0.0) + net.streets[sid]["length"]
            scores.append(know.unseen_m(set(net.streets[sid]["blocks"]) | set(C["j_blocks"][F])) / max(travel, 1.0))
        out = orig_choose(C, know, B, entrance, J, rng)
        log.append(dict(flight=len(made) - 1, t=float(f.sim.t), B=int(B), entrance=int(entrance), J=int(J),
                        n_wps=len(f.wps), opts=[(int(o[0]), int(o[1])) for o in opts], scores=scores,
                        choice=None if out is None else (int(out[0]), int(out[1]), bool(out[4]))))
        return out

    try:
        S.Flight, LS.choose_exit, LS.FootKnowledge = Taped, recording, Know
        try:
            res = LS.run_job(job)
        finally:
            S.Flight, LS.choose_exit, LS.FootKnowledge = base, orig_choose, orig_know
        same = _strip(res) == _strip(stored)
        p, s = job["p"], job["s"]
        C = LS.context(p, s, job["city_seed"])
        net = C["net"]
        city = Path(blockcity.path_for(blockcity.Params("R", p, s, job["city_seed"]))).stem
        seen_xz = playback.b64(playback.dm16(C["fp_pts"].reshape(-1)), "<i2")
        flights = []
        for fi, (row, (f, tape), know) in enumerate(zip(res["rows"], made, knows)):
            cond = row["condition"]
            c_, st, db = LS.settings_for(cond, job["spec"], p, s)
            first = first_seen_all(C, f.samples, SIGHT_M)   # the streets, for each exit's entered time
            decs = []
            for d in [x for x in log if x["flight"] == fi]:
                cc = split.plaza_cond(c_, net, d["B"])
                pre = split.hop_pre(cc, C, dict(block=d["B"], J=d["J"], entrance=d["entrance"]), st, db)
                arrive = d["n_wps"] - len(pre) - 1
                t0_ = f.done_t[arrive] if 0 <= arrive < len(f.done_t) else math.nan
                opts = []
                entered = cand_times(net, first, d["opts"])
                for (sid, V), sc, te_ in zip(d["opts"], d["scores"], entered):
                    F = net.other(sid, V)
                    blk = net.block_ahead(F, sid)
                    te, end = None, None
                    if blk is not None and blk in net.blocks and net.blocks[blk]["building"] >= 0:
                        idx = np.flatnonzero(C["fp_bld"] == net.blocks[blk]["building"])
                        dd = np.hypot(*(C["fp_pts"][idx] - net.jpos[F]).T)
                        sel = idx[dd <= dd.min() + LD.END_REACH_M]
                        fs = np.sort(know.first[sel])
                        tt = float(fs[max(1, int(math.ceil(LD.END_FRACTION * len(fs)))) - 1])
                        te = round(tt, 2) if math.isfinite(tt) else None
                        end = C["fp_pts"][sel].mean(0).round(1).tolist()
                    o = dict(xz=street_seg(net, sid, V), t=playback._r(te_, 1), te=te, end=end, sid=int(sid),
                             j=int(V), score=playback._r(sc, 3))
                    if d["choice"] and sid == d["choice"][0]:
                        o["chosen"] = True
                    opts.append(o)
                at_t = (lambda key: sum(o[key] is not None and o[key] <= d["t"] + 1e-6 for o in opts))  # noqa: E731
                n_end = sum(o["end"] is not None for o in opts)
                decs.append(dict(block=int(d["B"]), at=[round(float(v), 1) for v in net.jpos[d["J"]]],
                                 t0=playback._r(t0_, 2), t=round(d["t"], 2), opts=opts,
                                 plaza=net.blocks[d["B"]]["building"] < 0,
                                 blind=bool(d["choice"] and not d["choice"][2]),
                                 prev=round(at_t("t") / len(opts), 3) if opts else None,
                                 look=round(at_t("te") / n_end, 3) if n_end else None))
            summary = dict(found=row.get(f"found_r{SIGHT_M:g}"), find_m=SIGHT_M, found80=row.get("found_r80"),
                           found200=row.get("found_r200"), found40=row.get("found_r40"),
                           t_end=row["t_end"], decisions=row["decisions"], blind=row["blind_decisions"],
                           path_m=row["path_m"], lost=row["lost"], contacts=row["contacts"],
                           footpath=row["footpath_seen_share"], d_block=row["d_block"], d_street=row["d_street"],
                           T=row.get("t_hop"), mode=row.get("mode"))
            flights.append(playback.flight_json(
                tape.record(), id=f"{pick['key']}_{cond}_replica", group=pick["key"], city=city, source="replica",
                kind="searchB", cond=cond, job_id=res["job_id"], identical=bool(same), wps=wps_json(f.wps),
                decisions=decs, apart=apart_series(net, f.samples) if cond != "single" else None,
                seen=seen_b64(know.first), seen_xz=seen_xz, target=[round(float(x), 1) for x in res["target"]],
                found={f"r{r:g}": row.get(f"found_r{r:g}") for r in sorted({40.0, 80.0, 200.0, SIGHT_M})},
                find_m=SIGHT_M,
                summary=playback.rounded(summary, 3)))
        return dict(pick=pick, flights=flights, identical=same, target=res["target"], wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(pick=pick, error=repr(e), trace=traceback.format_exc()[-3000:])


# ============================================================================= the whole export

def p2_jobs():
    """[(pick, job, stored row)] for the walks and the searches, rebuilt as their runs planned them."""
    out = []
    wspec = json.loads((S.OUT_ROOT.parent / "split" / "spec.json").read_text(encoding="utf-8"))
    sspec = json.loads((S.OUT_ROOT.parent / "split_search" / "spec.json").read_text(encoding="utf-8"))
    wj = {j["job_id"]: j for jobs in split.plan_walks(wspec) for j in jobs}
    sj = {j["job_id"]: j for jobs in split.plan_search(sspec) for j in jobs}
    ws = stored_rows(S.OUT_ROOT.parent / "split" / "results.jsonl",
                     {f"walk|R|p{p:g}_s{s:g}|seed{c}|{k}" for _, p, s, c, k, _, _ in WALK_PICKS})
    ss = stored_rows(S.OUT_ROOT.parent / "split_search" / "results.jsonl",
                     {f"search|R|p{p:g}_s{s:g}|seed{c}|{k}" for _, p, s, c, k, _, _ in SEARCH_PICKS})
    for key, p, s, c, k, title, about in WALK_PICKS:
        jid = f"walk|R|p{p:g}_s{s:g}|seed{c}|{k}"
        out.append(("walk", dict(key=key, p=p, s=s, seed=c, i=k, title=title, about=about), wj[jid], ws[jid]))
    for key, p, s, c, k, title, about in SEARCH_PICKS:
        jid = f"search|R|p{p:g}_s{s:g}|seed{c}|{k}"
        out.append(("search", dict(key=key, p=p, s=s, seed=c, i=k, title=title, about=about), sj[jid], ss[jid]))
    return out


def p3_jobs(want=lambda keys: True):
    """Phase 3's picks, for whichever of its parts has results and is wanted (want(keys): any of these pick keys)."""
    out = []
    if (S.OUT_ROOT.parent / P3_PARTA / "results.jsonl").exists() and want([k for k, *_ in P3_BLOCKS]):
        out += [("p3", x) for x in p3_select()]
    spec = S.OUT_ROOT.parent / P3_PARTB / "spec.json"
    if (S.OUT_ROOT.parent / P3_PARTB / "results.jsonl").exists() and spec.exists() and want(
            [f"p3b_p{p:g}" for p, _ in json.loads(spec.read_text(encoding="utf-8"))["designs"]]):
        out += [("p3b", x) for x in p3b_select()]
    return out


def _city_of(kind, payload):
    """(variant, p, s, city seed) a work item flies in."""
    pick, job = payload[0], payload[1]
    if kind == "p3":
        return "R", float(pick["p"]), float(pick["s"]), int(pick["city_seed"])
    return job.get("variant", "R"), float(job["p"]), float(job["s"]), int(job["city_seed"])


def _dispatch(args):
    kind, payload = args
    return {"p1": p1_worker, "walk": walk_worker, "search": search_worker, "p3": p3_worker,
            "p3b": p3b_worker}[kind](payload)


def refly(jobs=6, only=None):
    """Stage 1: re-fly the picks (every one, or those named) with tapes. Each pick's flights go to flights/, and its
    record (the pick, whether the job's rows came out identical, the chain or the target, its flights) to
    picks/<key>.json."""
    from multiprocessing import Pool
    for d in ("cities", "flights", "picks"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    want = (lambda keys: not only or any(k in only for k in keys))                    # noqa: E731
    work = [("p1", x) for x in (p1_select() if want([x[0] for x in P1_PICKS]) else [])]
    if want([x[0] for x in WALK_PICKS + SEARCH_PICKS]):
        work += [(k, (pick, job, st)) for k, pick, job, st in p2_jobs()]
    work += p3_jobs(want)
    if only:
        work = [w for w in work if w[1][0]["key"] in only]
    for (v, p, s, c) in sorted({_city_of(*x) for x in work}):
        blockcity.ensure(blockcity.Params(v, p, s, c))
    print(f"[split_replays] re-flying {len(work)} jobs on {jobs} workers", flush=True)
    t0 = time.time()
    with Pool(min(jobs, len(work))) as pool:
        for r in pool.imap_unordered(_dispatch, work):
            key = r["pick"]["key"]
            if "error" in r:
                print(f"[split_replays] {key}: ERROR {r['error']}\n{r['trace']}", flush=True)
                continue
            for fl in r["flights"]:
                _write_flight(fl)
            meta = dict(pick=r["pick"], identical=bool(r["identical"]), flights=[fl["id"] for fl in r["flights"]],
                        chain=r.get("chain"), target=playback.rounded(r.get("target"), 2), wall_s=r.get("wall_s"))
            (OUT / "picks" / f"{key}.json").write_text(json.dumps(meta, indent=1, allow_nan=False), encoding="utf-8")
            print(f"[split_replays] {key}: {len(r['flights'])} flights, identical to the run: {r['identical']} "
                  f"({r.get('wall_s')} s)", flush=True)
    print(f"[split_replays] re-flown in {time.time() - t0:.0f} s", flush=True)


def assemble():
    """Stage 2, from what stage 1 wrote: the Unity flights of the walks and searches, every city the flights fly in
    and the gallery's (every design and variant, seed GALLERY_SEED), and index.json."""
    def pick_meta(key):
        path = OUT / "picks" / f"{key}.json"
        if not path.exists():
            print(f"[split_replays] {key}: not re-flown yet; left out", flush=True)
            return None
        m = json.loads(path.read_text(encoding="utf-8"))
        if not m["identical"]:
            print(f"[split_replays] {key}: NOT identical to its run; left out", flush=True)
            return None
        return m

    def flight(fid):
        return json.loads((OUT / "flights" / f"{fid}.json").read_text(encoding="utf-8"))

    index = dict(built=time.strftime("%Y-%m-%d %H:%M"), groups=[], gallery=[])
    cities = set()
    p1 = []
    for key, run, kind, v, p, s, rule, title, about in P1_PICKS:
        m = pick_meta(key)
        if m is None:
            continue
        fl = flight(m["flights"][0])
        cities.add((v, float(p), float(s), int(fl["city"].split("seed")[1])))
        p1.append(dict(_lib_entry(fl), title=title, about=about))          # the captions as the table has them now
    if p1:
        index["groups"].append(dict(id="phase1", kind="manoeuvres", title="Phase 1: the manoeuvres, one block each",
                                    about="Single manoeuvres from Phase 1's runs, in the replica, each variant at its "
                                          "confirmed profile (the surround hold at the screens' profile, cohesion "
                                          "cutoff 40).", flights=p1))
    for kind, picks in (("walk", WALK_PICKS), ("search", SEARCH_PICKS)):
        for key, p, s, c, k, title, about in picks:
            m = pick_meta(key)
            if m is None:
                continue
            rep = [flight(fid) for fid in m["flights"]]
            uni = []
            try:
                uni = (unity_walk_flights if kind == "walk" else unity_search_flights)(m["pick"], rep)
            except Exception as e:                                   # Unity's flights are an extra, not the point
                print(f"[split_replays] {key}: Unity flights failed: {e!r}\n{traceback.format_exc()[-1500:]}",
                      flush=True)
            spec = json.loads((S.OUT_ROOT.parent / ("split" if kind == "walk" else "split_search") /
                               "spec.json").read_text(encoding="utf-8"))
            st = dict(S.DEFAULT_SETTING, **spec["setting"])
            for fl in rep + uni:                 # every swarm flight reports both its spreads (the profile's d*)
                if fl["cond"] not in ("single", "single_look"):
                    fl["summary"] = dict(fl.get("summary") or {}, d_street=float(st["d_street"]),
                                         d_block=float(spec["d_star"].get(f"R|{p:g}|{s:g}", st["d_block"])))
            for fl in rep + uni:
                _write_flight(fl)
            print(f"[split_replays] {key}: {len(rep)} replica and {len(uni)} Unity flights", flush=True)
            cities.add(("R", float(p), float(s), int(c)))
            entries = sorted([_lib_entry(fl) for fl in rep + uni],
                             key=lambda e: (COND_ORDER.index(e["cond"]), e["source"] != "replica"))
            g = dict(id=key, kind=kind, title=title, about=about, p=p, s=s, seed=c, flights=entries,
                     city=rep[0]["city"])
            if kind == "walk":
                g["chain"] = m["chain"]
            else:
                g["target"] = m["target"]
            index["groups"].append(g)
    for key, p, s, title in P3_BLOCKS:                       # Phase 3: one block per design, flown every way
        m = pick_meta(key)
        if m is None:
            continue
        pk, ch = m["pick"], m["pick"]["chosen"]
        fls = [flight(fid) for fid in m["flights"]]
        cities.add(("R", float(p), float(s), int(pk["city_seed"])))
        how = "an aim-and-turn surround" if ch["mode"] == "aim" else "a firm-hold surround"
        ways = {5: "five", 6: "six"}.get(len(fls), str(len(fls)))
        lone = " The lone drone takes the same exit by the shorter way round." if any(
            fl["cond"] == "single" for fl in fls) else ""
        about = (f"One block from Part A, flown {ways} ways from the same start, with the same seed and the same exit "
                 f"(the arrow). The squares are the buildings at the far ends of the block's other streets, filled once "
                 f"a shown feed has seen the footpath in front of one: looked down. Part A's rule picked {how} of "
                 f"{ch['T']:g} s at d_ref {ch['d']:.1f} for this design ({ch['rule']}). The hold and the pass fly the "
                 f"same spread; Phase 2's surround flies its own setting, aim and turn for 6 s at Phase 1's d*. This "
                 f"trial is the one whose surround-minus-street difference in exits looked down is nearest the "
                 f"design's median over {pk['n_trials']} trials.{lone}")
        order = ("single", "swarm_street", "swarm_hold", "swarm_pass", "swarm_hop_p2", "swarm_hop")
        fls.sort(key=lambda fl: order.index(fl["cond"]) if fl["cond"] in order else len(order))
        index["groups"].append(dict(id=key, kind="block", title=title, about=about, p=p, s=s, seed=pk["city_seed"],
                                    city=fls[0]["city"], flights=[_lib_entry(fl) for fl in fls]))
    if (S.OUT_ROOT.parent / P3_PARTB / "spec.json").exists():
        for p, s in json.loads((S.OUT_ROOT.parent / P3_PARTB / "spec.json").read_text(encoding="utf-8"))["designs"]:
            key = f"p3b_p{p:g}"
            m = pick_meta(key)
            if m is None:
                continue
            pk = m["pick"]
            rep = [flight(fid) for fid in m["flights"]]
            cities.add(("R", float(p), float(s), int(pk["city_seed"])))
            about = (f"Part B: the target (star) stands on the footpath in front of a building, as a goal does. Each "
                     f"condition's pilot heads for the exit with the most unseen footpath ahead, from what its own feeds "
                     f"have shown (the dots: footpath seen), and stops at the first sight within {SIGHT_M:g} m. This target's "
                     f"surround-over-street time ratio ({pk['ratio']:.2f}) is the nearest to the design's median "
                     f"({pk['median_ratio']:.2f}).")
            entries = sorted([_lib_entry(fl) for fl in rep], key=lambda e: COND_ORDER.index(e["cond"]))
            index["groups"].append(dict(id=key, kind="search", title=pk["title"], about=about, p=p, s=s,
                                        seed=pk["city_seed"], flights=entries, city=rep[0]["city"],
                                        target=m["target"]))
    for v in blockcity.VARIANTS:
        for p in blockcity.P_LEVELS:
            for s in blockcity.S_LEVELS:
                cities.add((v, float(p), float(s), GALLERY_SEED))
    for v, p, s, c in sorted(cities):
        gal = c == GALLERY_SEED
        cj = playback.rounded(city_payload(v, p, s, c, anatomy=gal), 4)
        (OUT / "cities" / f"{cj['id']}.json").write_text(json.dumps(cj, separators=(",", ":"), allow_nan=False),
                                                          encoding="utf-8")
        if gal:
            index["gallery"].append(dict(v=v, p=p, s=s, city=cj["id"], stats=cj["stats"], anatomy=cj.get("anatomy")))
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"[split_replays] {len(index['groups'])} groups, {len(cities)} cities -> {OUT} ({size / 1e6:.1f} MB)",
          flush=True)


def _write_flight(fl):
    (OUT / "flights" / f"{fl['id']}.json").write_text(json.dumps(fl, separators=(",", ":"), allow_nan=False),
                                                       encoding="utf-8")


def _lib_entry(fl):
    e = {k: fl.get(k) for k in ("id", "title", "about", "city", "source", "kind", "cond", "n", "identical")}
    e["duration"] = round(fl["t0"] + (fl["frames"] - 1) * fl["dt"], 1)
    e["summary"] = fl.get("summary")
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("all", "refly", "assemble"), default="all")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="+", help="re-fly only these pick keys (the others keep what stage 1 wrote)")
    a = ap.parse_args()
    if a.stage in ("all", "refly"):
        refly(a.jobs, a.only)
    if a.stage in ("all", "assemble"):
        assemble()


if __name__ == "__main__":
    main()
