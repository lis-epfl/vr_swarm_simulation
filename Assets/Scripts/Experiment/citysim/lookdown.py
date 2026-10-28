"""S4 Phase 3, a short pilot (the look-down pilot): can the swarm look DOWN a block's streets, seeing each one to its
far end and the building that stands there, and does surrounding the block do that better? At which spread?

    python -B -m citysim.lookdown manoeuvres --name lookdown_A [--trials 30] [--jobs 24] [--quick]   # Part A
    python -B -m citysim.lookdown tables --name lookdown_A
    -> experiment_results/split_study/<name>/

Phases 1 and 2 scored a street as previewed once 60% of its centreline within 80 m of its corner had been seen. Seeing
the near part of the street meets that, often from across it at the junction, and it says nothing about what lies down
the street. Here every exit of the block (a radiating street other than the entrance) is scored at three nested
levels. Each counts what a shown feed (FeedLatch) saw within LOOK_RANGE_M slant, through a clear sight line, at the
search band:
  entered      the old preview: 60% of the street's centreline samples within 80 m of its corner
  to its end   60% of its centreline samples within STREET_END_M of its far junction
  looked down  (the primary) END_FRACTION of its END TARGETS: the footpath in front of the building at the street's far
               end, within END_REACH_M of the building's point nearest that junction. That building belongs to the
               block ahead there (streetnet.Network.block_ahead). When that block is a plaza, the targets are its lawn
               near the junction.
The per-block measure is the share of the block's exits looked down when the pilot commits.

Part A flies one block per trial. It is paired: every condition flies the same block, entrance, start, seed and exit
(a side exit, drawn by Phase 2's chain rule `split.exit_options`). Each trial is flown at every block spread (d_ref)
in D_GRID:
  street        decides on arrival at the entrance corner (spread d_street 1.0 throughout)
  hold T        a centring hold on the entrance corner for T s at the block spread (Phase 2's swarm_hold)
  pass          aimed through the block at its exit's corner at the block spread (Phase 2's swarm_pass)
  hop aim T     Phase 2's surround: aimed at the opposite exit at full stick for T s, then off down the exit
  hop firm T    a centring hold on the block's centre at stick 1.0 for T s (Phase 1's firm hold), then off
Every flight records its block spread (d_block) and street spread (d_street): d_ref is a reported factor.
Imported only by the parent and its own workers. It imports split and split_probe, so do not edit those while a run
goes (lessons.md 18).
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from citysim import clutter, split, split_probe as S  # noqa: E402
from citysim import geometry as geo  # noqa: E402

OUT_ROOT = S.OUT_ROOT.parent                   # experiment_results/split_study
DESIGNS = ((40.0, 18.0), (50.0, 18.0), (60.0, 18.0), (75.0, 18.0))
D_GRID = (1.0, 1.4, 1.8, 2.2, 2.6)
T_GRID = (6.0, 12.0, 20.0)
CITY_SEEDS = (0, 1, 2)                         # Phase 1's cities (seed 0 is the page's gallery city)
LOOK_RANGE_M = 80.0
STREET_END_M = 10.0
END_REACH_M = 12.0
END_FRACTION = 0.5
FOOTPATH_OFF_M = 2.0                           # the target ring: this far outside a building's footprint
FOOTPATH_STEP_M = 2.0
PLAZA_STEP_M = 3.0
BASE = dict(r0_coh=30.0, d_street=1.0, exit_stick=1.0, d_rule="fixed")      # R's confirmed profile (Phase 2's)
FIRM = dict(hold_stick=1.0, hold_gain=0.5)     # Phase 1's firm hold
QUICK = dict(designs=((40.0, 18.0),), trials=2, d=(1.0, 2.2), T=(6.0,))
SPLIT_COND = dict(street="swarm_street", hold="swarm_hold", pass_="swarm_pass", hop="swarm_hop", single="single",
                  single_look="single_look")
SINGLE_CONDS = (("single", None, None), ("single_look", None, None))     # the lone drone's run (lookdown_A1)


# ============================================================================= targets

def footpath_ring(poly, off=FOOTPATH_OFF_M, step=FOOTPATH_STEP_M):
    """Points every `step` m round a convex footprint grown by `off` with rounded corners: the footpath in front of
    its facades."""
    p = geo.ccw(np.asarray(poly, float))
    n = len(p)
    pts = []
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        e0, e1 = b - a, c - b
        n0 = np.array([e0[1], -e0[0]]) / max(np.linalg.norm(e0), 1e-9)          # outward for a CCW polygon
        n1 = np.array([e1[1], -e1[0]]) / max(np.linalg.norm(e1), 1e-9)
        a0, a1 = math.atan2(n0[1], n0[0]), math.atan2(n1[1], n1[0])
        while a1 < a0:
            a1 += 2 * math.pi
        k = max(1, int(math.ceil((a1 - a0) * off / step)))
        for th in np.linspace(a0, a1, k + 1)[:-1]:
            pts.append(b + off * np.array([math.cos(th), math.sin(th)]))
        q0, q1 = b + off * n1, c + off * n1                                       # the offset edge b -> c
        m = max(1, int(math.ceil(np.linalg.norm(q1 - q0) / step)))
        for u in np.arange(m) / m:
            pts.append(q0 + u * (q1 - q0))
    return np.array(pts)


def city_targets(C):
    """Per building, its footpath ring; per plaza block, a lawn grid. Cached on the city context."""
    if "ld_rings" in C:
        return C
    from citysim import blockcity
    L = C["L"]
    C["ld_rings"] = [footpath_ring(blockcity.building_poly(b)) for b in L.boxes]
    lawns = {}
    for bid, B in C["net"].blocks.items():
        if B["building"] >= 0:
            continue
        kerb = B["kerb"]
        lo, hi = kerb.min(0), kerb.max(0)
        g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], PLAZA_STEP_M), np.arange(lo[1], hi[1], PLAZA_STEP_M)),
                     -1).reshape(-1, 2) + PLAZA_STEP_M / 2
        g = g[geo.contains(kerb, g)] if len(g) else g
        lawns[bid] = g if len(g) else geo.centroid(kerb)[None]
    C["ld_lawns"] = lawns
    return C


def end_targets(C, sid, V):
    """(points, kind) for exit street sid leaving corner V: what stands at its far end, or (None, None) when the
    street's far junction has no block ahead."""
    net = C["net"]
    F = net.other(sid, V)
    blk = net.block_ahead(F, sid)
    if blk is None or blk not in net.blocks:
        return None, None
    B = net.blocks[blk]
    pf = net.jpos[F]
    if B["building"] >= 0:
        ring, kind = C["ld_rings"][B["building"]], "building"
    else:
        ring, kind = C["ld_lawns"][blk], "plaza"
    d = np.hypot(*(ring - pf).T)
    return ring[d <= d.min() + END_REACH_M], kind


def street_points(C, cands):
    """The points every level is scored on, for the block's exits: [(sid, V, entered idx, end idx, target idx,
    target kind, end target centre)], and the stacked points."""
    net = C["net"]
    chunks, out, k = [], [], 0

    def add(p):
        nonlocal k
        p = np.asarray(p, float).reshape(-1, 2)
        chunks.append(p)
        idx = np.arange(k, k + len(p))
        k += len(p)
        return idx

    for sid, V in cands:
        ent = net.street_samples(sid, V, S.PREVIEW_REACH_M)
        F = net.other(sid, V)
        allp = net.street_samples(sid, V, 1e9)
        near_f = allp[np.hypot(*(net.sample_xz[allp] - net.jpos[F]).T) <= STREET_END_M]
        tg, kind = end_targets(C, sid, V)
        out.append(dict(sid=int(sid), V=int(V), ent=add(net.sample_xz[ent]), end=add(net.sample_xz[near_f]),
                        tgt=add(tg) if tg is not None else np.zeros(0, int), kind=kind,
                        tgt_c=tg.mean(0).round(1).tolist() if tg is not None and len(tg) else None))
    pts = np.vstack(chunks) if chunks else np.zeros((0, 2))
    return out, pts


def level_times(first, info):
    """Per exit: the time it was entered, seen to its end and looked down (inf if never)."""
    def q(idx, frac):
        if not len(idx):
            return math.inf
        f = np.sort(first[idx])
        return float(f[max(1, int(math.ceil(frac * len(f)))) - 1])
    return [dict(ent=q(x["ent"], S.PREVIEW_FRAC), end=q(x["end"], S.PREVIEW_FRAC),
                 look=q(x["tgt"], END_FRACTION) if len(x["tgt"]) else math.nan) for x in info]


# ============================================================================= Part A: one block, every condition

def trial_geometry(C, rng):
    """dict(block, J, entrance, approach, exit, V, F, nb) or None: the block, its entrance and approach (split_probe's
    pick) and a side exit by Phase 2's chain rule."""
    net = C["net"]
    for _ in range(20):
        tr = S.pick(net, rng)
        if tr is None:
            return None
        ex = split.exit_options(net, tr["block"], tr["entrance"], tr["J"])
        if not ex:
            continue
        sid, V, F, nb = ex[int(rng.integers(len(ex)))]
        return dict(tr, exit=int(sid), V=int(V), F=int(F), nb=int(nb))
    return None


def conditions(T_grid=T_GRID):
    out = [("street", None, None)]
    out += [("hold", None, T) for T in T_grid]
    out += [("pass", None, None)]
    out += [("hop", mode, T) for mode in ("aim", "firm") for T in T_grid]
    return out


def setting(cond, mode, T):
    st = dict(S.DEFAULT_SETTING, **BASE)
    if T is not None:
        st["t_hop"] = float(T)
    if cond == "hop":
        st["m3b_mode"] = "aim" if mode == "aim" else "hold"
        if mode == "firm":
            st.update(FIRM)
    return st


def flight_wps(cond, C, tr, st, db):
    """(waypoints, dict(arrive, commit, next)) of one condition over the trial's block."""
    net = C["net"]
    c = SPLIT_COND["pass_" if cond == "pass" else cond]
    wps = split.approach(net, tr["approach"], st["d_street"])
    h = dict(block=tr["block"], J=tr["J"], entrance=tr["entrance"], exit=tr["exit"], V=tr["V"], F=tr["F"])
    pre, post = split.hop_pre(c, C, h, st, db), split.hop_post(c, C, h, st, db)
    arrive = len(wps) - 1
    commit = len(wps) + split.commit_offset(c, len(pre))
    wps = wps + pre + post
    return wps, dict(arrive=arrive, commit=commit, next=len(wps) - 1)


def fly_condition(C, tr, cond, mode, T, db, seed):
    st = setting(cond, mode, T)
    wps, idx = flight_wps(cond, C, tr, st, db)
    m = S.model(st, "R")
    if cond in ("single", "single_look"):
        P0 = C["net"].jpos[tr["approach"][0]][None]
    else:
        P0 = S.start_positions(C, dict(approach=tr["approach"]), st, seed)
    f = S.Flight(m, C["rcity"], P0, st["d_street"], seed, S.bands()["search"][1], hold_gain=float(st["hold_gain"]))
    f.add(wps)
    f.run(sum(w[3] for w in wps) + 10.0)
    return f, idx, st, len(P0)


def score_condition(C, tr, f, idx, n0, info, pts, cond, look_range=LOOK_RANGE_M):
    net, fp = C["net"], C["fp"]
    h, pitch = S.bands()["search"]
    done, samples = f.done_t, f.samples
    t_J, t_c, t_F = (done[idx[k]] if idx[k] < len(done) else math.nan for k in ("arrive", "commit", "next"))
    first = split.first_seen(fp, samples, pts, h, pitch, r=look_range)
    lv = level_times(first, info)
    at_J = [x["V"] == tr["J"] for x in info]

    def share(key, t, only=None):
        vals = [(not math.isnan(v[key])) and v[key] <= t + 1e-9 for v, j in zip(lv, at_J)
                if (only is None or j == only) and not math.isnan(v[key])]
        return float(np.mean(vals)) if vals else math.nan

    row = dict(t_J=S._r(t_J), t_commit=S._r(t_c), t_next=S._r(t_F), hop_s=S._r(t_F - t_J), n_exits=len(info),
               n_exits_far=int(sum(not j for j in at_J)), n_plaza_ends=int(sum(x["kind"] == "plaza" for x in info)),
               **S.losses(f.sim, n0))
    for name, t in (("c", t_c), ("F", t_F)):
        tt = t if math.isfinite(t) else -math.inf
        row[f"ent_{name}"], row[f"end_{name}"], row[f"look_{name}"] = (share(k, tt) for k in ("ent", "end", "look"))
    row["look_c_far"] = share("look", t_c if math.isfinite(t_c) else -math.inf, only=False)
    row["look_c_near"] = share("look", t_c if math.isfinite(t_c) else -math.inf, only=True)
    looks = sorted(v["look"] - t_J for v in lv if not math.isnan(v["look"]))
    need = max(1, int(math.ceil(0.8 * len(looks)))) if looks else 0
    row["t80_look"] = S._r(looks[need - 1]) if looks and math.isfinite(looks[need - 1]) else None
    row["exits"] = [dict(sid=x["sid"], at_J=bool(j), kind=x["kind"], ent=S._r(v["ent"] - t_J), end=S._r(v["end"] - t_J),
                         look=S._r(v["look"] - t_J) if not math.isnan(v["look"]) else None)
                    for x, v, j in zip(info, lv, at_J)]
    if math.isfinite(t_c):
        ser = S.cluster_series(net, samples, t0=t_c, t1=t_F if math.isfinite(t_F) else math.inf)
        row["reform_s"], row["reform_ok"] = S.reform(ser, t_c, t_F if math.isfinite(t_F) else samples[-1]["t"])
        if cond == "hop" and math.isfinite(t_J):
            row.update(S.envelop_series(C, samples, tr, t_J, t_c))
    row["facade_min_m"] = S.facade_min(fp, samples)
    return row


def trial_seed(p, s, city_seed, trial):
    key = f"lookdown|R|{p:g}|{s:g}|{city_seed}|{trial}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def job_id(j):
    if j.get("single"):
        return f"A1|p{j['p']:g}_s{j['s']:g}|seed{j['city_seed']}|{j['trial']}"
    return f"A|p{j['p']:g}_s{j['s']:g}|seed{j['city_seed']}|{j['trial']}|d{j['d']:g}"


def run_job(j):
    """One trial at one block spread: every condition over the same block, start, seed and exit. A `single` job flies
    the lone drone's two conditions over that trial instead (no spread: d is None)."""
    t0 = time.time()
    base = dict(job_id=job_id(j), p=j["p"], s=j["s"], city_seed=j["city_seed"], trial=j["trial"], d=j.get("d"))
    try:
        C = city_targets(S.city("R", j["p"], j["s"], j["city_seed"]))
        rng = np.random.default_rng(trial_seed(j["p"], j["s"], j["city_seed"], j["trial"]))
        seed = int(rng.integers(1 << 30))
        tr = trial_geometry(C, rng)
        if tr is None:
            return dict(base, skipped=True)
        cands = S.candidates(C["net"], tr)
        info, pts = street_points(C, cands)
        rows = []
        conds = SINGLE_CONDS if j.get("single") else conditions(j.get("T_grid", T_GRID))
        for cond, mode, T in conds:
            f, idx, st, n0 = fly_condition(C, tr, cond, mode, T, j.get("d", 1.0), seed)
            r = score_condition(C, tr, f, idx, n0, info, pts, cond, float(j.get("look_range", LOOK_RANGE_M)))
            rows.append(dict(cond=cond, mode=mode, T=T,
                             d_block=float(j["d"]) if cond not in ("street", "single", "single_look") else None,
                             d_street=None if cond in ("single", "single_look") else float(st["d_street"]), **r))
        return dict(base, block=int(tr["block"]), entrance=int(tr["entrance"]), J=int(tr["J"]), exit=int(tr["exit"]),
                    rows=rows, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2500:], wall_s=round(time.time() - t0, 1))


def plan(trials, designs=DESIGNS, d_grid=D_GRID, T_grid=T_GRID, look_range=LOOK_RANGE_M):
    """Rounds of jobs: round k is every design and spread's k-th trial, so a stop leaves every cell equally covered."""
    rounds = []
    for t in range(trials):
        jobs = []
        for p, s in designs:
            for d in d_grid:
                jobs.append(dict(p=float(p), s=float(s), city_seed=int(CITY_SEEDS[t % len(CITY_SEEDS)]), trial=t,
                                 look_range=float(look_range),
                                 d=float(d), T_grid=list(T_grid)))
        np.random.default_rng(101 + t).shuffle(jobs)
        rounds.append(jobs)
    return rounds


def cmd_manoeuvres(a):
    from citysim import blockcity, parallel
    name = a.name + ("_quick" if a.quick else "")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    if a.quick:
        rounds = plan(QUICK["trials"], QUICK["designs"], QUICK["d"], QUICK["T"], a.sight)
    else:
        rounds = plan(a.trials, look_range=a.sight)
    done = clutter._done_ids(res)
    spec = dict(part="A", designs=[list(x) for x in (QUICK["designs"] if a.quick else DESIGNS)],
                d_grid=list(QUICK["d"] if a.quick else D_GRID), T_grid=list(QUICK["T"] if a.quick else T_GRID),
                trials=QUICK["trials"] if a.quick else a.trials, city_seeds=list(CITY_SEEDS), base=BASE, firm=FIRM,
                look_range_m=float(a.sight), street_end_m=STREET_END_M, end_reach_m=END_REACH_M,
                end_fraction=END_FRACTION, footpath_off_m=FOOTPATH_OFF_M)
    (out / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    for key in sorted({(j["p"], j["s"], j["city_seed"]) for jobs in rounds for j in jobs}):
        blockcity.ensure(blockcity.Params("R", *key))
    clutter.keep_awake()
    t0 = time.time()
    for k, jobs in enumerate(rounds):
        todo = [j for j in jobs if job_id(j) not in done]
        if not todo:
            continue
        parallel.run_jobs(run_job, todo, f"lookdown A round {k}", processes=a.jobs, stall_s=900.0,
                          progress=lambda i, n, r: clutter._append(res, r))
        _log(out, f"round {k} done ({time.time() - t0:.0f} s)")
    _log(out, f"finished in {(time.time() - t0) / 60:.1f} min")


def cmd_single(a):
    """The lone drone over Part A's trials (the same block, entrance, start, seed and exit; added 2026-10-28 so the
    page can set what one drone looks down beside the swarm): `single` and `single_look`, one job per trial."""
    from citysim import blockcity, parallel
    out = OUT_ROOT / a.name
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    done = clutter._done_ids(res)
    jobs = [dict(p=float(p), s=float(s), city_seed=int(CITY_SEEDS[t % len(CITY_SEEDS)]), trial=t, single=True,
                 look_range=float(a.sight))
            for t in range(a.trials) for p, s in DESIGNS]
    spec = dict(part="A1", of="lookdown_A", designs=[list(x) for x in DESIGNS], trials=a.trials,
                city_seeds=list(CITY_SEEDS), conditions=[c for c, _, _ in SINGLE_CONDS], base=BASE,
                look_range_m=float(a.sight), street_end_m=STREET_END_M, end_reach_m=END_REACH_M,
                end_fraction=END_FRACTION, footpath_off_m=FOOTPATH_OFF_M)
    (out / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    for key in sorted({(j["p"], j["s"], j["city_seed"]) for j in jobs}):
        blockcity.ensure(blockcity.Params("R", *key))
    t0 = time.time()
    todo = [j for j in jobs if job_id(j) not in done]
    if todo:
        parallel.run_jobs(run_job, todo, "lookdown A1 (one drone)", processes=a.jobs, stall_s=900.0,
                          progress=lambda i, n, r: clutter._append(res, r))
    _log(out, f"finished in {(time.time() - t0) / 60:.1f} min")


def _log(out, msg):
    line = f"[lookdown {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================================= tables

def read(name):
    """(flights DataFrame, jobs) of a run: one row per flight, the job's keys beside each."""
    import pandas as pd
    by = {}
    for line in (OUT_ROOT / name / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r or r.get("skipped"):
            continue
        by[r["job_id"]] = r
    rows = []
    for r in by.values():
        for x in r["rows"]:
            rows.append(dict({k: r[k] for k in ("p", "s", "city_seed", "trial", "d")},
                             **{k: v for k, v in x.items() if k != "exits"}))
    df = pd.DataFrame(rows)
    df["label"] = [label(c, m, t) for c, m, t in zip(df.cond, df["mode"], df["T"])]
    return df, by


def label(cond, mode, T):
    if cond in ("single", "single_look"):
        return {"single": "one drone", "single_look": "one drone, looking"}[cond]
    if cond == "street":
        return "street"
    if cond == "pass":
        return "pass"
    if cond == "hold":
        return f"hold {T:g} s"
    return f"surround ({mode}) {T:g} s"


def cmd_tables(a):
    import pandas as pd
    df, _ = read(a.name)
    out = OUT_ROOT / a.name
    cols = ["ent_c", "end_c", "look_c", "look_c_far", "look_F", "hop_s", "lost", "contacts", "reform_ok", "t80_look",
            "enveloped", "t_envelop"]
    df["reform_ok"] = df["reform_ok"].astype(float)
    if "enveloped" in df:
        df["enveloped"] = df["enveloped"].astype(float)
    g = df.groupby(["p", "s", "d", "label"])[[c for c in cols if c in df]].mean().reset_index()
    g["n"] = df.groupby(["p", "s", "d", "label"]).size().values
    g.to_csv(out / "cells.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(g.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manoeuvres")
    p.add_argument("--name", default="lookdown_A")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--sight", type=float, default=LOOK_RANGE_M,
                   help="the sight range every level is scored at, m (the 80 m runs: lookdown_A, lookdown_A1)")
    p.add_argument("--quick", action="store_true")
    p = sub.add_parser("single")
    p.add_argument("--name", default="lookdown_A1")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--sight", type=float, default=LOOK_RANGE_M,
                   help="the sight range every level is scored at, m (the 80 m runs: lookdown_A, lookdown_A1)")
    p = sub.add_parser("tables")
    p.add_argument("--name", default="lookdown_A")
    a = ap.parse_args()
    {"manoeuvres": cmd_manoeuvres, "single": cmd_single, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
