"""S4 Phase 3, Part B (the look-down pilot's search): the target stands at a building, as a goal does, so seeing what
is down a street can find it. Does surrounding the block find it sooner?

    python -B -m citysim.lookdown_search choose --partA lookdown_A --name lookdown_B   # the surround setting, by rule
    python -B -m citysim.lookdown_search search --name lookdown_B [--jobs 16] [--quick]
    python -B -m citysim.lookdown_search tables --name lookdown_B
    -> experiment_results/split_study/<name>/

Target. A point on the footpath round a random interior building (lookdown.footpath_ring: 2 m outside its footprint).
It must be at least TARGET_STREET_M by street from the start, out of the start's clear sight within TARGET_HIDDEN_M,
and TARGET_EDGE_M inside the arena (Phase 2's rules). It is found at its first sight on a shown feed within 80 m (the
primary; 40 and 200 m are recorded too). The cap is SEARCH_T_S, and a swarm under four drones has not found it.

The pilot is informed, as Phase 2's is, but its map is of where the target can be: the footpath points round every
building that a shown feed has seen within 80 m. At each decision it takes the exit with the most unseen footpath per
metre of travel. The footpath counted belongs to the buildings of the blocks along the exit street and at its far
junction. When none is unseen nearby, it heads for the nearest junction whose blocks still have FRONTIER_MIN_M
unseen (Phase 2's 20 m). The code is the same in every condition. A row's blind_decisions counts the decisions with no
exit scored, which includes those frontier moves: it is not a count of random choices. That misreading had the rule
changed to one footpath point mid-run (lookdown_B_v2, stopped). A per-decision log then showed the 20 m pilot making
no random choice in the searches it had failed, and the rule was put back.

Conditions: one drone; the swarm in the streets; holding at the corner (T, d); passing round (d); surrounding (mode,
T, d); and surrounding at Phase 2's setting (aim 6 s at the design's d*), to compare the spreads. The surround's
setting comes from Part A by the rule in `choose`, fixed before Part A's results were read. The hold flies its T and
d, the pass its d. Imported only by the parent and its own workers.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import blockcity, clutter, los, lookdown as LD, split, split_probe as S  # noqa: E402

OUT_ROOT = S.OUT_ROOT.parent
DESIGNS = ((40.0, 18.0), (60.0, 18.0))
CITY_SEED0, CITIES, TARGETS = 200, 8, 4
SEARCH_T_S = 1200.0
FIND_RANGES = (40.0, 80.0, 200.0)
FIND_STOP_M = 80.0
SEEN_RANGE_M = 80.0
MIN_SEARCH_DRONES = 4
DECISION_STEP_S = 2.0
FRONTIER_MIN_M = 20.0                            # lookdown_B_v2 (stopped) flew one footpath point: see the docstring
TARGET_STREET_M, TARGET_HIDDEN_M, TARGET_EDGE_M = 300.0, 200.0, 60.0
P2_DSTAR = {(40.0, 18.0): 1.0, (50.0, 18.0): 1.0, (60.0, 18.0): 1.2, (75.0, 18.0): 1.8}   # Phase 1's d* (coupling)
QUICK = dict(designs=((40.0, 18.0),), cities=1, targets=2, T_s=300.0)
# the rule that picks Part B's surround from Part A (fixed before Part A's results were read)
SAFE_LOST, SAFE_CONTACTS, TIME_PRICE = 0.03, 0.1, 1.5
_SIGHT0 = dict(FIND_RANGES=FIND_RANGES, FIND_STOP_M=FIND_STOP_M, SEEN_RANGE_M=SEEN_RANGE_M,
               TARGET_HIDDEN_M=TARGET_HIDDEN_M)


def set_sight(spec):
    """A run's sight range (spec "sight_m", added 2026-10-28 at the user's request: 80 m was too short to mean
    anything; absent, lookdown_B's 80 m): what the pilot's map counts as seen, the first sight that stops the search
    (also recorded beside FIND_RANGES), and the radius within which the target starts out of the start's sight (at
    least TARGET_HIDDEN_M, so no target is in view at the start). Module state set per job: a run's jobs share it."""
    g = globals()
    g.update(_SIGHT0)
    m = spec.get("sight_m")
    if m:
        m = float(m)
        g.update(SEEN_RANGE_M=m, FIND_STOP_M=m, FIND_RANGES=tuple(sorted(set(_SIGHT0["FIND_RANGES"]) | {m})),
                 TARGET_HIDDEN_M=max(_SIGHT0["TARGET_HIDDEN_M"], m))


# ============================================================================= the city: footpath, blocks, junctions

_CTX = {}


def context(p, s, seed, city=None):
    """The search's city: S4's variant R, or the blockcity design a spec names under "city" (S5)."""
    P = blockcity.Params(**dict(city, p=p, s=s, seed=seed)) if city else blockcity.Params("R", p, s, seed)
    key = P.ident()
    if key not in _CTX:
        _CTX.clear()
        C = dict(LD.city_targets(S.city_from(P)))
        net = C["net"]
        rings = C["ld_rings"]
        C["fp_pts"] = np.vstack(rings) if rings else np.zeros((0, 2))
        C["fp_bld"] = np.concatenate([np.full(len(r), i) for i, r in enumerate(rings)]) if rings else np.zeros(0, int)
        bld_block = {B["building"]: bid for bid, B in net.blocks.items() if B["building"] >= 0}
        C["fp_block"] = np.array([bld_block.get(int(b), -1) for b in C["fp_bld"]])
        C["j_blocks"] = {j: [bid for bid, B in net.blocks.items() if j in B["corners"]] for j in range(len(net.jpos))}
        x0, x1, z0, z1 = blockcity.ARENA
        q = C["fp_pts"]
        e = TARGET_EDGE_M
        inter = np.array([net.blocks[b]["interior"] if b in net.blocks else False for b in C["fp_block"]])
        C["target_ok"] = inter & (q[:, 0] > x0 + e) & (q[:, 0] < x1 - e) & (q[:, 1] > z0 + e) & (q[:, 1] < z1 - e)
        _CTX[key] = C
    return _CTX[key]


class FootKnowledge:
    """What one flight's pilot has seen of the footpath round the buildings (within SEEN_RANGE_M on a shown feed, the
    search band), and the target's first sight at each of FIND_RANGES, brought up to date as samples arrive."""

    def __init__(self, C, h, pitch, target):
        from scipy.spatial import cKDTree
        self.C, self.h, self.pitch = C, h, pitch
        self.pts = C["fp_pts"]
        self.first = np.full(len(self.pts), np.inf)
        self.tree = cKDTree(self.pts)
        self.target = np.asarray(target, float)
        self.found = {r: math.inf for r in FIND_RANGES}
        self.k = 0

    def update(self, samples):
        fp = self.C["fp"]
        for smp in samples[self.k:]:
            for k in range(len(smp["yaws"])):
                cam, yaw = smp["cams"][k], float(smp["yaws"][k])
                near = np.asarray(self.tree.query_ball_point(cam, SEEN_RANGE_M), int)
                todo = near[~np.isfinite(self.first[near])] if len(near) else near
                if len(todo):
                    vis = los.sees(fp, cam, self.h, yaw, self.pitch, self.pts[todo], r_id=SEEN_RANGE_M, band=True)[0]
                    self.first[todo[vis]] = smp["t"]
                for r in FIND_RANGES:
                    if not math.isfinite(self.found[r]) and los.sees(fp, cam, self.h, yaw, self.pitch, self.target[None],
                                                                      r_id=r, band=True)[0][0]:
                        self.found[r] = smp["t"]
        self.k = len(samples)

    def unseen_m(self, blocks):
        m = np.isin(self.C["fp_block"], list(blocks)) & ~np.isfinite(self.first)
        return float(m.sum()) * LD.FOOTPATH_STEP_M


def choose_exit(C, know, B, entrance, J, rng):
    """(sid, V, F, next block, scored) or None: the exit with the most unseen footpath of the blocks along it and at
    its far junction per metre of travel; with none, the one nearest by street to a junction whose blocks still have
    FRONTIER_MIN_M unseen."""
    net = C["net"]
    opts = split.exit_options(net, B, entrance, J) or split.fallback_exits(net, B, entrance, J)
    if not opts:
        return None
    scores = []
    for sid, V, F, nb in opts:
        js, _ = net.route(J, V)
        travel = (net.route_length(js) if js else 0.0) + net.streets[sid]["length"]
        near = set(net.streets[sid]["blocks"]) | set(C["j_blocks"][F])
        scores.append(know.unseen_m(near) / max(travel, 1.0))
    scores = np.array(scores)
    if scores.max() > 0:
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return (*opts[int(rng.choice(best))], True)
    src = {j for j, bl in C["j_blocks"].items() if net.j_interior[j] and know.unseen_m(bl) >= FRONTIER_MIN_M}
    if not src:
        return (*opts[int(rng.integers(len(opts)))], False)
    dist = split.junction_dist(net, src)
    d = np.array([dist.get(F, math.inf) for _, _, F, _ in opts])
    best = np.flatnonzero(d <= d.min() + 1e-9)
    return (*opts[int(rng.choice(best))], False)


def place_target(C, start_j, rng, tries=400):
    """A footpath point round an interior building, far enough by street from the start and hidden from it."""
    net, fp = C["net"], C["fp"]
    pts = C["fp_pts"][C["target_ok"]]
    if not len(pts):
        return None
    dist = split.junction_dist(net, [start_j])
    s0 = net.jpos[start_j]
    for _ in range(tries):
        q = pts[int(rng.integers(len(pts)))]
        if dist.get(net.nearest_junction(q), 0.0) < TARGET_STREET_M:
            continue
        if np.linalg.norm(q - s0) <= TARGET_HIDDEN_M and not fp.blocked(s0[None], q[None])[0]:
            continue
        return q
    return None


# ============================================================================= the flight

def settings_for(cond, spec, p, s):
    """(split condition, setting, block spread) of one Part B condition at one design."""
    base = dict(S.DEFAULT_SETTING, **LD.BASE)
    ds = spec.get("d_street")                     # S5: one spread for the streets and the block
    if ds:
        base["d_street"] = float(ds)
    ch = spec["chosen"][f"{p:g}|{s:g}"]
    if cond == "single":
        return "single", base, base["d_street"]
    if cond == "swarm_street":
        return "swarm_street", base, base["d_street"]
    if cond == "swarm_hold":
        return "swarm_hold", dict(base, t_hop=ch["T"]), ch["d"]
    if cond == "swarm_pass":
        return "swarm_pass", base, ch["d"]
    if cond == "swarm_hop":
        return "swarm_hop", LD.setting("hop", ch["mode"], ch["T"], ds), ch["d"]
    if cond == "swarm_hop_p2":
        return "swarm_hop", LD.setting("hop", "aim", 6.0), P2_DSTAR[(p, s)]
    raise ValueError(cond)


def search_flight(cond, C, start, target, spec, p, s, seed, T_cap):
    net = C["net"]
    h, pitch = S.bands()["search"]
    c, st, db = settings_for(cond, spec, p, s)
    single = c == "single"
    m = S.model(st, "R")
    P0 = net.jpos[start["approach"][0]][None] if single else S.start_positions(C, start, st, seed)
    f = S.Flight(m, C["rcity"], P0, st["d_street"], seed, pitch, hold_gain=float(st["hold_gain"]))
    know = FootKnowledge(C, h, pitch, target)
    rng = np.random.default_rng(seed + 7)
    f.add(split.approach(net, start["approach"], st["d_street"]))
    B, J, e = start["block"], start["J"], start["entrance"]
    decisions = blind = 0
    min_alive = 1 if single else MIN_SEARCH_DRONES

    def go(until):
        while True:
            f.run(min(T_cap, f.sim.t + DECISION_STEP_S), until_wp=until)
            know.update(f.samples)
            if (math.isfinite(know.found[FIND_STOP_M]) or f.sim.t >= T_cap - 1e-9 or f.sim.alive.sum() < min_alive
                    or f.wp > until or f.wp >= len(f.wps)):
                return

    go(len(f.wps) - 1)
    while (not math.isfinite(know.found[FIND_STOP_M]) and f.sim.t < T_cap - 1e-9 and f.sim.alive.sum() >= min_alive
           and f.wp >= len(f.wps)):
        hh = dict(block=B, J=J, entrance=e)
        cc = split.plaza_cond(c, net, B)
        pre = split.hop_pre(cc, C, hh, st, db)
        if pre:
            f.add(pre)
            go(len(f.wps) - 1)
            if math.isfinite(know.found[FIND_STOP_M]) or f.wp < len(f.wps):
                break
        ch = choose_exit(C, know, B, e, J, rng)
        if ch is None:
            break
        sid, V, F, nb, scored = ch
        decisions += 1
        blind += int(not scored)
        f.add(split.hop_post(cc, C, dict(hh, exit=sid, V=V, F=F), st, db))
        go(len(f.wps) - 1)
        B, J, e = nb, F, sid
    alive = int(f.sim.alive.sum())
    path = sum(float(np.linalg.norm(b["c"] - a["c"])) for a, b in zip(f.samples[:-1], f.samples[1:]))
    inter = C["target_ok"]
    return dict(condition=cond, d_block=float(db), d_street=float(st["d_street"]),
                t_hop=float(st["t_hop"]) if c in ("swarm_hold", "swarm_hop") else None,
                mode=st["m3b_mode"] if c == "swarm_hop" else None, t_end=round(float(f.sim.t), 2), alive=alive,
                lost=int(len(P0) - alive), contacts=int(f.sim.contacts), decisions=decisions, blind_decisions=blind,
                path_m=round(path, 1),
                footpath_seen_share=round(float((np.isfinite(know.first) & inter).sum() / max(inter.sum(), 1)), 4),
                **{f"found_r{r:g}": (round(v, 2) if math.isfinite(v) else None) for r, v in know.found.items()})


CONDITIONS = ("single", "swarm_street", "swarm_hold", "swarm_pass", "swarm_hop", "swarm_hop_p2")


def job_seed(p, s, city, k):
    key = f"lookdown_search|R|{p:g}|{s:g}|{city}|{k}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def job_id(j):
    return f"B|p{j['p']:g}_s{j['s']:g}|seed{j['city_seed']}|{j['target_i']}"


def run_job(j):
    t0 = time.time()
    base = dict(job_id=job_id(j), p=j["p"], s=j["s"], city_seed=j["city_seed"], target_i=j["target_i"],
                T_s=float(j["T_s"]))
    try:
        set_sight(j["spec"])
        C = context(j["p"], j["s"], j["city_seed"], j["spec"].get("city"))
        rng = np.random.default_rng(job_seed(j["p"], j["s"], j["city_seed"], j["target_i"]))
        seed = int(rng.integers(1 << 30))
        start = S.pick(C["net"], rng)
        if start is None:
            return dict(base, skipped=True)
        target = place_target(C, start["approach"][0], rng)
        if target is None:
            return dict(base, skipped=True, why="no target")
        rows = [search_flight(c, C, start, target, j["spec"], j["p"], j["s"], seed, float(j["T_s"]))
                for c in j["spec"].get("conditions", CONDITIONS)]
        return dict(base, rows=rows, target=[round(float(x), 2) for x in target], wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2500:], wall_s=round(time.time() - t0, 1))


# ============================================================================= choosing the surround (from Part A)

def choose(partA, designs=DESIGNS):
    """Part B's surround per design, by the rule fixed before Part A's results were read: among the surround settings
    (mode, T, d) that are safe (drones lost <= SAFE_LOST and contacts <= SAFE_CONTACTS per manoeuvre) and whose time
    price (the median of each trial's hop time over the street-bound swarm's to the same exit) is <= TIME_PRICE, the
    one with the most exits looked down at the commit; ties to the shorter T, then the smaller d. With none inside
    the time bar, the safe setting with the most looked down per second of hop time."""
    df, _ = LD.read(partA)
    out = {}
    for p, s in designs:
        g = df[(df.p == p) & (df.s == s)]
        street = g[g.cond == "street"].set_index(["city_seed", "trial", "d"]).hop_s
        rows = []
        for (mode, T, d), h in g[g.cond == "hop"].groupby(["mode", "T", "d"]):
            hs = h.set_index(["city_seed", "trial", "d"]).hop_s
            price = float(np.nanmedian((hs / street.reindex(hs.index)).values))
            rows.append(dict(mode=mode, T=float(T), d=float(d), look_c=float(h.look_c.mean()),
                             lost=float(h.lost.mean()), contacts=float(h.contacts.mean()), price=price,
                             hop_s=float(h.hop_s.median()), n=int(len(h))))
        safe = [r for r in rows if r["lost"] <= SAFE_LOST and r["contacts"] <= SAFE_CONTACTS]
        ok = [r for r in safe if r["price"] <= TIME_PRICE]
        if ok:
            best = sorted(ok, key=lambda r: (-r["look_c"], r["T"], r["d"]))[0]
            rule = "safe, inside the time price"
        elif safe:
            best = sorted(safe, key=lambda r: (-r["look_c"] / max(r["hop_s"], 1e-9), r["T"], r["d"]))[0]
            rule = "safe; none inside the time price: most looked down per second"
        else:
            best = sorted(rows, key=lambda r: (r["lost"], r["contacts"]))[0]
            rule = "none safe: the least unsafe"
        out[f"{p:g}|{s:g}"] = dict(best, rule=rule, candidates=sorted(rows, key=lambda r: (r["mode"], r["T"], r["d"])))
    return out


def cmd_choose(a):
    chosen = choose(a.partA)
    out = OUT_ROOT / a.name
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(part="B", partA=a.partA, designs=[list(x) for x in DESIGNS], city_seed0=CITY_SEED0, cities=CITIES,
                targets=TARGETS, T_s=SEARCH_T_S, conditions=list(CONDITIONS), p2_dstar={f"{p:g}|{s:g}": d for (p, s), d
                                                                                        in P2_DSTAR.items()},
                rule=dict(lost=SAFE_LOST, contacts=SAFE_CONTACTS, time_price=TIME_PRICE), chosen=chosen,
                frontier_min_m=FRONTIER_MIN_M, sight_m=float(a.sight))
    (out / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    for k, v in chosen.items():
        print(f"[lookdown_search] {k}: surround {v['mode']} {v['T']:g} s at d {v['d']:g} -- looked down {v['look_c']:.2f}, "
              f"time price {v['price']:.2f}, lost {v['lost']:.3f}, contacts {v['contacts']:.2f} ({v['rule']})")


# ============================================================================= the run

def plan(spec, quick=False):
    designs = [tuple(d) for d in spec["designs"]]
    cities, targets, T = spec["cities"], spec["targets"], spec["T_s"]
    if quick:
        designs, cities, targets, T = QUICK["designs"], QUICK["cities"], QUICK["targets"], QUICK["T_s"]
    rounds = []
    for k in range(targets):
        jobs = [dict(p=float(p), s=float(s), city_seed=CITY_SEED0 + c, target_i=k, T_s=T, spec=spec)
                for p, s in designs for c in range(cities)]
        np.random.default_rng(301 + k).shuffle(jobs)
        rounds.append(jobs)
    return rounds


def cmd_search(a):
    from citysim import parallel
    out = OUT_ROOT / a.name
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    name = a.name + ("_quick" if a.quick else "")
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.quick:
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    res = run_dir / "results.jsonl"
    done = clutter._done_ids(res)
    rounds = plan(spec, a.quick)
    for key in sorted({(j["p"], j["s"], j["city_seed"]) for jobs in rounds for j in jobs}):
        blockcity.ensure(blockcity.Params("R", *key))
    clutter.keep_awake()
    t0 = time.time()
    for k, jobs in enumerate(rounds):
        todo = [j for j in jobs if job_id(j) not in done]
        if todo:
            parallel.run_jobs(run_job, todo, f"lookdown B round {k}", processes=a.jobs, stall_s=900.0,
                              progress=lambda i, n, r: clutter._append(res, r))
            LD._log(run_dir, f"search round {k} done ({time.time() - t0:.0f} s)")
    LD._log(run_dir, f"search finished in {(time.time() - t0) / 60:.1f} min")


def read(name, root=None):
    import pandas as pd
    by = {}
    for line in (Path(root or OUT_ROOT) / name / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r or r.get("skipped"):
            continue
        by[r["job_id"]] = r
    rows = [dict({k: r[k] for k in ("p", "s", "city_seed", "target_i", "T_s")}, **x) for r in by.values() for x in r["rows"]]
    df = pd.DataFrame(rows)
    if len(df):
        for col in [c for c in df.columns if c.startswith("found_r")]:
            r = col[len("found_r"):]
            df[f"hit_r{r}"] = df[col].notna()
            df[f"t_r{r}"] = df[col].fillna(df.T_s).clip(upper=df.T_s)
    return df, by


def cmd_tables(a):
    import pandas as pd
    df, _ = read(a.name)
    out = OUT_ROOT / a.name
    g = df.groupby(["p", "s", "condition"]).agg(n=("t_r80", "size"), found=("hit_r80", "mean"), rmst=("t_r80", "mean"),
                                                rmst40=("t_r40", "mean"), lost=("lost", "mean"),
                                                contacts=("contacts", "mean"), decisions=("decisions", "mean"),
                                                d_block=("d_block", "first"), t_hop=("t_hop", "first")).reset_index()
    g.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(3).to_string(index=False))
    print(ratios(df).round(3).to_string(index=False))


def ratios(df, pairs=(("swarm_hop", "swarm_street"), ("swarm_hop", "swarm_hold"), ("swarm_hop", "swarm_hop_p2"),
                      ("swarm_street", "single"), ("swarm_pass", "swarm_street")), measure="t_r80"):
    """City-mean log ratios of the restricted mean time to find (80 m), per design and pooled: t intervals over
    cities (8 per design: a pilot's width)."""
    import pandas as pd
    rows = []
    for a, b in pairs:
        A = df[df.condition == a].set_index(["p", "s", "city_seed", "target_i"])[measure]
        B = df[df.condition == b].set_index(["p", "s", "city_seed", "target_i"])[measure]
        d = np.log(A / B.reindex(A.index)).dropna()
        cm = d.groupby(["p", "s", "city_seed"]).mean().rename("d").reset_index()
        for key, g in list(cm.groupby(["p", "s"])) + [(("pooled", ""), cm)]:
            x = g.d.values
            if len(x) < 2:
                continue
            m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
            rows.append(dict(a=a, b=b, design=f"p {key[0]:g} · s {key[1]:g}" if key[0] != "pooled" else "pooled",
                             n_cities=int(len(x)), ratio=math.exp(m), lo=math.exp(m - 1.96 * se),
                             hi=math.exp(m + 1.96 * se)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("choose")
    p.add_argument("--partA", default="lookdown_A")
    p.add_argument("--sight", type=float, default=80.0, help="the run's sight range, m (spec sight_m)")
    p.add_argument("--name", default="lookdown_B")
    p = sub.add_parser("search")
    p.add_argument("--name", default="lookdown_B")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--quick", action="store_true")
    p = sub.add_parser("tables")
    p.add_argument("--name", default="lookdown_B")
    a = ap.parse_args()
    {"choose": cmd_choose, "search": cmd_search, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
