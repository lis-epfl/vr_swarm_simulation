"""S8 Footprint Avoidance (the shape study): what the swarm avoids each building as -- the circle round its bounds (as
always), the smallest circle round its footprint, or the footprint itself (SwarmManager.obstacleShape) -- flown on S7's
harness (citysim.swarmflow, citysim.swarmtune) in BlockShapedPatchCityWorld with S7's slide on and route stragglers off.

    python -B -m citysim.swarmshape gaps [--seeds 1 2 3 4]                # what each shape leaves of the streets
    python -B -m citysim.swarmshape screen --name N --cand FILE [--cross 48 --pass 32 --street 24 --face 36] [--single]
    python -B -m citysim.swarmshape confirm --name N --cand FILE [--single]    # fresh routes, the scene's city + held out
    python -B -m citysim.swarmshape table --name N
    python -B -m citysim.swarmshape rule                                   # rule.json, written before the confirmation
    python -B -m citysim.swarmshape choose --name confirm --base LABEL     # rule.json applied: verdicts.json
    python -B -m citysim.swarmshape bench-plan --name N --cand FILE [--drones 1]
    python -B -m citysim.swarmshape bench-gate --name N --runs DIR [DIR ...]
    -> experiment_results/swarm_shape/<name>/

S6's three flights and measures (swarmtune.score_flight) plus a fourth flight, `face`, for the case the cylinder was
chosen for: the formation flown square at a flat wall. A flat face gives no sideways push, so a bounding box left the
swarm pressed against it, slow to split and surround (the user's first city); the cylinder's normal turns with the
approach. Under Footprint only the slide (S7) supplies that push.

  face    from open ground 50 m out along the outward normal of a straight face (>= 12 m) of a building on the city's
          edge, at a lateral offset of 0, 0.3 or 0.6 of its half-length, aimed through the building at the first point
          beyond it clear of every footprint by FACE_FREE_M, then the usual end hold. Measured from arrival (the
          centroid within the bounds circle's R + ENC_M, as bld_pass_s counts, so every shape is timed alike), each
          censored at FACE_CAP_S:
            t_side     the first drone alongside the building beyond either end of the face (the slide has started)
            t_split    drones alongside beyond both ends (the formation has split round it)
            t_wrap     the building's centre inside the drones' convex hull (formation.envelopment: surrounded)
            t_through  the centroid past the building's back, or the point behind it taken
          and the flight's longest stall (stall_s), contacts and losses.

Scoring stays on the bounds circles whatever the law flies (met, wrapped, bld_pass_s in score_flight; arrival here),
so the shapes are measured alike. Route indices: screens from K_SCREEN, the confirmation from K_CONFIRM, the bench from
K_BENCH (S6 and S7 used 0-6999).
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

import numpy as np  # noqa: E402

from citysim import REPO, blockcity, formation  # noqa: E402
from citysim import split_probe as S  # noqa: E402
from citysim import swarmflow as F  # noqa: E402
from citysim import swarmtune as T  # noqa: E402

OUT = REPO / "experiment_results" / "swarm_shape"
K_SCREEN, K_CONFIRM, K_BENCH = 7000, 8000, 9000
SHAPES = ("Cylinder", "EnclosingCircle", "Footprint")

FACE_RUN_M = 50.0          # the run-in: start this far out along the face's normal
FACE_MIN_LEN_M = 12.0
FACE_OFFSETS = (0.0, 0.3, 0.6)   # lateral aim offset as a share of the half-face, by route index k % 3
FACE_CLEAR_M = 2.0         # the run-in corridor: no footprint within this of any probe point
FACE_LAT_M = (-18.0, -9.0, 0.0, 9.0, 18.0)   # probe lines across the corridor (the formation is ~31 m wide)
FACE_FREE_M = 6.0          # the far waypoint: this clear of every footprint
FACE_BEYOND_M = 120.0
FACE_CAP_S = 40.0
FACE_KEYS = ("t_side", "t_split", "t_wrap", "t_through")

_FACES = {}


# ============================================================================= the head-on flight

def faces(C, seed):
    """[dict(b, e, L, m, t, n, depth)] every straight face of at least FACE_MIN_LEN_M with a clear run-in of
    FACE_RUN_M along its outward normal, at every offset: the faces a formation can be flown square at."""
    if seed in _FACES:
        return _FACES[seed]
    rc = C["rcity"]
    allb = np.arange(rc.n)
    out = []
    for i in range(rc.n):
        k = int((rc.Cn[i] < 1e17).sum())
        verts = rc.V[i, :k]
        for e in range(k):
            a, b = rc.V[i, e], rc.W[i, e]
            L = float(np.linalg.norm(b - a))
            if L < FACE_MIN_LEN_M:
                continue
            n, m, t = rc.N[i, e], 0.5 * (a + b), (b - a) / L
            probes = [m + o * L / 2 * t * sg + n * s + lat * t
                      for o in FACE_OFFSETS for sg in (-1, 1) for s in np.linspace(3.0, FACE_RUN_M + 10.0, 20)
                      for lat in FACE_LAT_M]
            if rc.distance(np.array(probes), allb).min() < FACE_CLEAR_M:
                continue
            depth = float(((m[None] - verts) @ n).max())          # how far the building reaches behind the face
            out.append(dict(b=i, e=e, L=L, m=m, t=t, n=n, depth=depth))
    _FACES[seed] = out
    return out


def face_route(C, seed, k, rng):
    """(start, [(x, z, stick, timeout)], route length, aimed polyline, face) of head-on route k: offset by k % 3,
    its sign and the face drawn from rng."""
    fs = faces(C, seed)
    if not fs:
        raise RuntimeError(f"no clear face in city seed {seed}")
    rc = C["rcity"]
    allb = np.arange(rc.n)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    off = FACE_OFFSETS[k % len(FACE_OFFSETS)] * sign
    for _ in range(50):
        f = fs[int(rng.integers(len(fs)))]
        aim = f["m"] + off * f["L"] / 2 * f["t"]
        u = -f["n"]
        start = aim + FACE_RUN_M * f["n"]
        line = aim[None] + np.arange(1.0, FACE_BEYOND_M)[:, None] * u[None]
        d_own = rc.distance(line, np.array([f["b"]]))[:, 0]
        d_all = rc.distance(line, allb).min(1)
        past = np.flatnonzero((np.cumsum(d_own == 0) > 0) & (d_own > 0) & (d_all >= FACE_FREE_M))
        if not len(past):
            continue
        far = line[past[0]]
        dist = float(np.linalg.norm(far - start))
        wps = [(float(far[0]), float(far[1]), 1.0, S.timeout_for(dist))]
        face = dict(f, off=off, aim=aim)
        return start, wps, dist, np.vstack([start, far]), face
    raise RuntimeError("no head-on face with a clear point behind it")


def face_measures(C, face, samples, done=None):
    """The head-on measures of one flight (see the module docstring). done: the waypoint times, so a formation that
    takes the point behind the building before its centroid crosses the back plane has still got through."""
    rc = C["rcity"]
    b = face["b"]
    cb, Rb = rc.c[b], float(rc.R[b])
    m, t, n, L, depth = face["m"], face["t"], face["n"], face["L"], face["depth"]
    arrive = next((s[0] for s in samples if np.linalg.norm(s[1] - cb) <= Rb + T.ENC_M), None)
    out = dict(face_b=int(b), face_off=round(float(face["off"]), 2), face_L=round(L, 1), arrived=arrive is not None)
    for key in FACE_KEYS:
        out[key] = None
    if arrive is None:
        return out
    for tt, c, P in samples:
        if tt < arrive - 1e-9:
            continue
        el = tt - arrive
        if el > FACE_CAP_S:
            break
        Pa = P[np.isfinite(P).all(1)]
        if not len(Pa):
            break
        rel = Pa - m[None]
        lat, dep = rel @ t, rel @ n
        beside = dep <= 0.0
        left, right = (beside & (lat < -L / 2)).any(), (beside & (lat > L / 2)).any()
        if out["t_side"] is None and (left or right):
            out["t_side"] = round(el, 2)
        if out["t_split"] is None and left and right:
            out["t_split"] = round(el, 2)
        if out["t_wrap"] is None and len(Pa) >= 3 and formation.envelopment(Pa, cb)["surrounds"]:
            out["t_wrap"] = round(el, 2)
        took = done is not None and math.isfinite(done[0]) and done[0] <= tt + 1e-9
        if out["t_through"] is None and (float((c - m) @ n) <= -depth or took):
            out["t_through"] = round(el, 2)
    return out


def run_job(j):
    """One flight: S7's (swarmflow.run_job, the spread logged) for cross, pass and street; the head-on flight here."""
    if j["kind"] != "face":
        return F.run_job(j)
    t0 = time.time()
    base = dict(job_id=T.job_id(j), cand=j["cand"], kind=j["kind"], seed=j["seed"], k=j["k"], d=j["d"])
    try:
        C = T.city(j["seed"])
        rng = np.random.default_rng(T.route_seed("face", j["seed"], j["k"]))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly, face = face_route(C, j["seed"], j["k"], rng)
        over = dict(j["over"])
        m = T.model(over)
        P0 = start[None].copy() if j.get("single") else T.start_positions(C, j["d"], fseed, over, start)
        log = F.SpreadLog()
        sim, samples, done = T.fly(m, C, P0, wps, j["d"], fseed, hook=log)
        sig = np.array(log.sig) if log.sig else np.ones(1)
        return dict(base, **T.score_flight(C, start, wps, poly, L, sim, samples, done, sim.n0),
                    **face_measures(C, face, samples, done), sigma_mean=round(float(sig.mean()), 3),
                    narrow_frac=round(float((sig < 0.95).mean()), 3), wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


# ============================================================================= tables

def face_scores(g):
    """The head-on measures of one candidate's face flights: each time's restricted mean (censored at FACE_CAP_S) and
    the share reached within it, by offset too."""
    r = {}
    if not len(g):
        return r
    r["face_n"] = len(g)
    r["face_arrived"] = float(g.arrived.mean())
    for key in FACE_KEYS:
        v = g[key].astype(float)
        r[f"face_{key}"] = float(np.fmin(v.fillna(FACE_CAP_S), FACE_CAP_S).mean())
        r[f"face_{key}_share"] = float(v.notna().mean())
        for off, gg in g.groupby(g.face_off.abs().round(1)):
            vv = gg[key].astype(float)
            r[f"face_{key}_o{off:g}"] = float(np.fmin(vv.fillna(FACE_CAP_S), FACE_CAP_S).mean())
    r["face_stall"] = float(g.stall_s.mean())
    r["face_contacts"] = float(g.contacts.mean())
    r["face_lost"] = float(g.lost.mean())
    r["face_one"] = float(g.one_frac.mean())
    return r


def score_rows(g):
    """S6's measures over the cross, pass and street flights, and the head-on measures over the face flights."""
    std = g[g.kind != "face"]
    r = T.score_candidate(std) if len(std) else {}
    r.update(face_scores(g[g.kind == "face"]))
    if "sigma_mean" in g:
        r["sigma"] = float(std.sigma_mean.mean()) if len(std) else math.nan
    return r


def table(name, root=None):
    import pandas as pd
    df, cands, _ = T.read(name, root or OUT)
    rows = []
    for cid, g in df.groupby("cand"):
        c = cands.get(cid, {})
        rows.append(dict(cand=cid, label=c.get("label", ""), d=float(g.d.iloc[0]), **score_rows(g),
                         **{f"o:{k}": v for k, v in c.get("over", {}).items()}))
    tab = pd.DataFrame(rows)
    order = {c.get("label", ""): i for i, c in enumerate(cands.values())}
    return tab.assign(_o=tab.label.map(order)).sort_values("_o").drop(columns="_o")


COLS = ["label", "n", "score", "v_coh", "one", "lost_km", "contact_km", "open", "hull", "cross_wrapped",
        "cross_bld_pass", "pass_v_coh", "street_v_coh", "street_one", "street_lost_km", "street_contact_km",
        "face_t_side", "face_t_split", "face_t_split_share", "face_t_wrap", "face_t_wrap_share", "face_t_through",
        "face_stall", "face_contacts", "face_lost", "sigma"]


def show(tab, n=60):
    import pandas as pd
    pd.set_option("display.width", 420)
    pd.set_option("display.max_columns", 80)
    print(tab[[c for c in COLS if c in tab.columns]].head(n).round(3).to_string(index=False))


def jobs_for(spec, seeds, flights, k0, single):
    return F.jobs_for(spec, seeds, flights, k0, single)


def flights_of(a, default):
    return default if a.cross is None else dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face)


def cmd_screen(a):
    spec = F.load_cands(a.cand)
    fl = flights_of(a, dict(cross=48, pass_=32, street=24, face=36))
    cands, jobs = jobs_for(spec, (T.TUNE_SEED,), fl, a.k0, a.single)
    T.run(a.name, cands, jobs, a.jobs, f"screen ({len(cands)} candidates)", root=OUT, fn=run_job)
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab)


def cmd_confirm(a):
    spec = F.load_cands(a.cand)
    fl = dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face)
    cands, jobs = jobs_for(spec, (T.TUNE_SEED,) + T.HELD_SEEDS, fl, a.k0, a.single)
    T.run(a.name, cands, jobs, a.jobs, "confirm", root=OUT, fn=run_job)
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab)


def cmd_table(a):
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab, a.n)


# ============================================================================= the rule (rule.json, fixed before the confirmation)

# base: S7's pick (the cylinder with the slide), what the trials would fly today. contact_base: the scene's law as it
# flies (T3, the slide off), the bound S7's rule put on contacts -- not the slide's own, lower count, which would rule
# out any shape that lets the drones nearer the facades however much else it gains.
RULE = dict(base="A1", contact_base="A0", lost_margin=0.05, open_margin=0.02, hull_min=0.5, split_ratio=1.1,
            street_gain=0.05, v_coh_floor=0.98, wrap_margin=0.10)


def judge(tab, rule):
    """Every swarm configuration of a confirmation table against the baseline by rule.json: (verdicts, pick)."""
    sw = tab[~tab.cand.astype(str).str.startswith("single")]
    code = sw.label.str.split().str[0]
    b = sw[code == rule["base"]].iloc[0]
    cb = sw[code == rule.get("contact_base", rule["base"])].iloc[0]
    out = {}
    for _, r in sw.iterrows():
        safe = bool(r.lost_km <= b.lost_km + rule["lost_margin"] and r.contact_km <= cb.contact_km
                    and r.open <= b.open + rule["open_margin"] and r.hull >= rule["hull_min"]
                    and r.face_t_split <= rule["split_ratio"] * b.face_t_split)
        helps = bool(r.street_v_coh >= (1 + rule["street_gain"]) * b.street_v_coh
                     and r.v_coh >= rule["v_coh_floor"] * b.v_coh
                     and r.cross_wrapped >= b.cross_wrapped - rule["wrap_margin"])
        out[r.label] = dict(safe=safe, helps=helps, score=float(r.score),
                            street_gain=float(r.street_v_coh / b.street_v_coh - 1),
                            v_coh_gain=float(r.v_coh / b.v_coh - 1), wrap_change=float(r.cross_wrapped - b.cross_wrapped),
                            split_ratio=float(r.face_t_split / b.face_t_split),
                            lost_change=float(r.lost_km - b.lost_km), contact_change=float(r.contact_km - b.contact_km),
                            open_change=float(r.open - b.open))
    ok = [(v["score"], k) for k, v in out.items() if v["safe"] and v["helps"] and k not in (b.label, cb.label)]
    return out, (max(ok)[1] if ok else None)


def cmd_rule(a):
    path = OUT / "rule.json"
    if path.exists() and not a.force:
        raise SystemExit(f"[swarmshape] {path} exists: the rule is fixed (--force to overwrite)")
    OUT.mkdir(parents=True, exist_ok=True)
    rule = dict(RULE, **({} if not a.set else json.loads(a.set)))
    path.write_text(json.dumps(dict(rule=rule, written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1),
                    encoding="utf-8")
    print(json.dumps(rule, indent=1))


def cmd_choose(a):
    rule = json.loads((OUT / "rule.json").read_text(encoding="utf-8"))["rule"]
    tab = table(a.name)
    verdicts, pick = judge(tab, rule)
    for k, v in sorted(verdicts.items(), key=lambda kv: -kv[1]["score"]):
        print(f"{k:34s} safe {str(v['safe']):5s} helps {str(v['helps']):5s} score {v['score']:.3f}  street "
              f"{v['street_gain']:+.1%}  v_coh {v['v_coh_gain']:+.1%}  wrapped {v['wrap_change']:+.3f}  split x"
              f"{v['split_ratio']:.2f}  lost {v['lost_change']:+.3f}  contacts {v['contact_change']:+.3f}  open "
              f"{v['open_change']:+.3f}")
    print(f"\nrule: {pick or 'none: recommend none'}")
    (OUT / a.name / "verdicts.json").write_text(json.dumps(dict(rule=rule, verdicts=verdicts, pick=pick), indent=1),
                                               encoding="utf-8")


# ============================================================================= what each shape leaves of the streets

def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def street_gaps(seed, standoffs=(0.0, 1.6, 2.5, 3.5)):
    """{shape: gaps (n,)} the narrowest clear width along every interior street with a building on each side, between
    the two buildings as each shape draws them: the bounds circles, the enclosing circles, and the footprints less a
    standoff on each side (d_obs in metres, where the field takes hold)."""
    C = T.city(seed)
    rc = C["rcity"]
    lay = json.loads(blockcity.ensure(T.params(seed)).read_text(encoding="utf-8"))
    bidx = [blk.get("building") for blk in lay["blocks"]]
    out = {"bounds circle": [], "enclosing circle": [], "facades": []}
    for s in lay["streets"]:
        bl = s.get("blocks") or []
        if len(bl) != 2 or s.get("edge"):
            continue
        ia, ib = bidx[bl[0]], bidx[bl[1]]
        if ia is None or ib is None or ia < 0 or ib < 0:
            continue
        pl = np.array(s["polyline"], float)
        pts = np.vstack([pl[i] + np.linspace(0, 1, 9)[:, None] * (pl[i + 1] - pl[i]) for i in range(len(pl) - 1)])
        d = rc.distance(pts, np.array([ia, ib]))
        out["facades"].append(float(d.sum(1).min()))
        for key, c, R in (("bounds circle", rc.c, rc.R), ("enclosing circle", rc.mec_c, rc.mec_R)):
            g = (np.linalg.norm(pts - c[ia], axis=1) - R[ia]) + (np.linalg.norm(pts - c[ib], axis=1) - R[ib])
            out[key].append(float(g.min()))
    res = {k: np.array(v) for k, v in out.items()}
    for sd in standoffs[1:]:
        res[f"facades less {sd:g} m each side"] = res["facades"] - 2 * sd
    return res


def cmd_gaps(a):
    import pandas as pd
    rows = []
    for seed in a.seeds:
        C = T.city(seed)
        rc = C["rcity"]
        for k, g in street_gaps(seed).items():
            rows.append(dict(seed=seed, shape=k, pairs=len(g), median=np.median(g), p10=np.percentile(g, 10),
                             closed=int((g < 0).sum()), under6=int((g < 6).sum()), under10=int((g < 10).sum())))
        bulge = []
        for i in range(rc.n):
            k = int((rc.Cn[i] < 1e17).sum())
            edge = min(float(_seg_dist(rc.c[i][None], rc.V[i, e], rc.W[i, e])[0]) for e in range(k))
            bulge.append(float(rc.R[i]) - edge)     # the bounds circle beyond the facade nearest its centre
        rows.append(dict(seed=seed, shape="bounds radius / enclosing radius (median m)", pairs=rc.n,
                         median=float(np.median(rc.R)), p10=float(np.median(rc.mec_R)), closed=0, under6=0, under10=0))
        print(f"seed {seed}: the bounds circle reaches a median {np.median(bulge):.1f} m past the nearest facade "
              f"(p90 {np.percentile(bulge, 90):.1f}); faces fit for the head-on flight: {len(faces(C, seed))}")
    df = pd.DataFrame(rows).round(2)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "gaps.csv", index=False)


# ============================================================================= S7's events, located

class EventLog:
    """A read-only fly() hook: every building contact and drone loss of the flight with where it happened, and every
    live drone's position each OCC_EVERY ticks (where the drones spent their time), for comparing the two."""
    OCC_EVERY = 25

    def __init__(self):
        self.events, self.occ, self.first, self.n = [], [], True, 0

    def start(self, sim):
        pass

    def step(self, sim, t, cmd, wp):
        if not self.first:
            for i in np.flatnonzero(sim.touching & ~self.touch):
                self.events.append(("contact", round(t, 2), int(i), *map(float, sim.P[i])))
            for i in np.flatnonzero(self.alive & ~sim.alive):
                self.events.append(("lost", round(t, 2), int(i), *map(float, self.P[i])))
        self.first = False
        self.touch, self.alive, self.P = sim.touching.copy(), sim.alive.copy(), sim.P.copy()
        if self.n % self.OCC_EVERY == 0:
            self.occ.append(sim.P[sim.alive].copy())
        self.n += 1


def place_class(C, pts):
    """For points (n, 2): (inside a bounds circle, within d_obs of two or more bounds circles, the narrowest gap the
    bounds circles leave there, the narrowest the facades leave there): where the cylinder law sees each point."""
    rc = C["rcity"]
    pts = np.atleast_2d(pts)
    near = rc.near(pts, 40.0)
    if not len(near):
        z = np.zeros(len(pts), bool)
        return z, z, np.full(len(pts), np.inf), np.full(len(pts), np.inf)
    dc = np.linalg.norm(pts[:, None] - rc.c[near][None], axis=-1) - rc.R[near][None]   # to each bounds circle
    df = rc.distance(pts, near)                                                     # to each facade
    oc = np.sort(dc, axis=1)
    of = np.sort(df, axis=1)
    inside = oc[:, 0] < 0
    two = (dc < 1.6).sum(1) >= 2
    gap_c = oc[:, 0] + (oc[:, 1] if oc.shape[1] > 1 else np.inf)
    gap_f = of[:, 0] + (of[:, 1] if of.shape[1] > 1 else np.inf)
    return inside, two, gap_c, gap_f


def events_job(j):
    """One stored S7 flight re-flown with the event log: its events placed, its occupancy placed, and whether the
    re-flown measures equal the stored row (the flight is only used if they do)."""
    try:
        C = T.city(j["seed"])
        rng = np.random.default_rng(T.route_seed(j["kind"], j["seed"], j["k"]))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly = T.route(C, j["kind"], j["k"], rng)
        over = dict(j["over"])
        m = T.model(over)
        P0 = T.start_positions(C, j["d"], fseed, over, start)
        log = EventLog()
        sim, samples, done = T.fly(m, C, P0, wps, j["d"], fseed, hook=log)
        r = T.score_flight(C, start, wps, poly, L, sim, samples, done, sim.n0)
        same = all(r[k] == j["row"][k] for k in r if k in j["row"])
        occ = np.vstack(log.occ) if log.occ else np.zeros((0, 2))
        oi, ot, ogc, ogf = place_class(C, occ)
        ev = []
        for kind, t, i, x, z in log.events:
            ins, two, gc, gf = place_class(C, np.array([[x, z]]))
            ev.append(dict(kind=kind, t=t, i=i, x=round(x, 2), z=round(z, 2), inside=bool(ins[0]), two=bool(two[0]),
                           gap_c=round(float(gc[0]), 2), gap_f=round(float(gf[0]), 2)))
        return dict(job_id=j["row"]["job_id"], label=j["label"], kind=j["kind"], seed=j["seed"], same=same,
                    L=r["L"], events=ev, occ_n=len(occ), occ_inside=int(oi.sum()), occ_two=int(ot.sum()),
                    occ_gap_c10=int((ogc < 10).sum()))
    except Exception as e:
        return dict(job_id=j["row"]["job_id"], error=repr(e), trace=traceback.format_exc()[-1500:])


def cmd_events(a):
    """Re-fly S7's confirmation flights of the given configurations with the event log, and compare where the
    contacts and losses happened with where the drones spent their time, classed by the bounds circles."""
    import pandas as pd
    from citysim import parallel
    src = F.OUT / a.run
    rows = [json.loads(x) for x in (src / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    cands = json.loads((src / "cands.json").read_text(encoding="utf-8"))
    jobs = []
    for r in rows:
        c = cands.get(r["cand"])
        if "error" in r or c is None or c["label"] not in a.labels or r["cand"].startswith("single"):
            continue
        over = dict(c["over"])
        if over.get("adaptiveSpread") and a.run == "confirm":
            over.update(streetSpreadFactor=0.8, spreadFilterTime=1.0)     # the defaults S7's confirmation flew
        jobs.append(dict(row=r, label=c["label"], over=over, d=r["d"], kind=r["kind"], seed=r["seed"], k=r["k"]))
    res = parallel.run_jobs(events_job, jobs, "S7 events", processes=a.jobs)
    out = OUT / "s7_events"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "flights.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in res if "error" not in r and r["same"]]
    print(f"[swarmshape] {len(ok)} of {len(res)} flights re-flown identically "
          f"({sum('error' in r for r in res)} errors)")
    summ = []
    for lab in a.labels:
        fl = [r for r in ok if r["label"] == lab]
        if not fl:
            continue
        occ = sum(r["occ_n"] for r in fl)
        row = dict(label=lab, flights=len(fl), km=round(sum(r["L"] for r in fl) / 1000, 1),
                   time_inside=sum(r["occ_inside"] for r in fl) / occ, time_two=sum(r["occ_two"] for r in fl) / occ,
                   time_gap10=sum(r["occ_gap_c10"] for r in fl) / occ)
        for kind in ("contact", "lost"):
            ev = [e for r in fl for e in r["events"] if e["kind"] == kind]
            row[f"{kind}s"] = len(ev)
            if ev:
                row[f"{kind}_inside"] = np.mean([e["inside"] for e in ev])
                row[f"{kind}_two"] = np.mean([e["two"] for e in ev])
                row[f"{kind}_gap10"] = np.mean([e["gap_c"] < 10 for e in ev])
                row[f"{kind}_gap_c_med"] = float(np.median([e["gap_c"] for e in ev]))
                row[f"{kind}_gap_f_med"] = float(np.median([e["gap_f"] for e in ev]))
        summ.append(row)
    df = pd.DataFrame(summ)
    df.round(3).to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(df.round(3).to_string(index=False))


# ============================================================================= Unity

def bench_scenarios(flights, k0):
    """S6's scenarios for cross, pass and street (swarmtune.bench_scenarios), and the head-on ones from face_route,
    each ending in END_HOLD_S at zero stick on its last waypoint."""
    std = {k: v for k, v in flights.items() if k != "face"}
    out = T.bench_scenarios(std, k0=k0)
    C = T.city(T.TUNE_SEED)
    for k in range(k0, k0 + flights.get("face", 0)):
        rng = np.random.default_rng(T.route_seed("face", T.TUNE_SEED, k))
        rng.integers(1 << 30)
        start, wps, L, poly, face = face_route(C, T.TUNE_SEED, k, rng)
        wl = [list(w) for w in wps] + [[wps[-1][0], wps[-1][1], -1.0, T.END_HOLD_S]]
        out.append(dict(kind=f"face_{k}", sx=round(float(start[0]), 3), sz=round(float(start[1]), 3),
                        T=round(sum(w[3] for w in wps) + 10.0 + T.END_HOLD_S, 1),
                        wps=[round(float(v), 3) for w in wl for v in w], holdGain=T.ZERO_GAIN, record=True,
                        endOnDone=True))
    return out


def cmd_bench_plan(a):
    from collections import OrderedDict
    from citysim import BENCH_DIR, camera
    spec = F.load_cands(a.cand)
    fields = T.sp.swarm_manager_fields()
    sets = []
    for c in spec:
        if any(k.startswith("x:") for k in c["over"]):
            raise SystemExit(f"[swarmshape] {c['label']}: replica-only extras cannot fly in Unity")
        over = {k: (int(v) if isinstance(v, bool) else v) for k, v in dict(c["over"], d_ref=c["d"]).items()}
        assigns = ",".join(f"{k}={v}" for k, v in over.items())
        sets.append(T.bench.parse_set(f"{c['label'].split()[0]}:{assigns}", fields, True))
    scen = bench_scenarios(dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face), a.k0)
    cfg = OrderedDict(scene=T.SCENE, timeScale=20.0, settleTime=T.BENCH_SETTLE_S, altitude=T.BENCH_ALTITUDE,
                      capture=T.CAPTURE_M, pilotHeading=True, pilotYawRate=camera.BODY_YAW_RATE_DEG,
                      goalReplaySession="", layout="", droneCount=a.drones, walkers=False, paramSets=sets,
                      scenarios=scen)
    out = OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    cfile = out / "config.json"
    cfile.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "cands.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"[swarmshape] {len(sets)} sets x {len(scen)} flights -> {cfile}\n"
          f"  & '{BENCH_DIR / 'run_bench.ps1'}' -Config '{cfile}' -Tag swarmshape_{a.name} -TimeoutMin {a.timeout}")


def bench_job(job):
    """swarmtune.bench_job for cross, pass and street; for a head-on flight the same (Unity's trajectory scored, and
    its replica twin flown from Unity's recorded start) with the head-on measures on both."""
    run_dir, label, idx, sc, over, d, flight = job
    kind, k = sc["kind"].rsplit("_", 1)
    if kind != "face":
        return T.bench_job(job)
    from pathlib import Path
    from types import SimpleNamespace
    from citysim import tierc
    try:
        k = int(k)
        C = T.city(T.TUNE_SEED)
        rng = np.random.default_rng(T.route_seed("face", T.TUNE_SEED, k))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly, face = face_route(C, T.TUNE_SEED, k, rng)
        tr = tierc.read_traj(Path(run_dir) / f"traj_{label}_{idx}.csv")
        samples = []
        for r in range(0, len(tr["t"]), max(1, int(round(T.SAMPLE_S / 0.1)))):
            al = tr["alive"][r]
            P = np.where(al[:, None], np.c_[tr["X"][r], tr["Z"][r]], np.nan)
            samples.append((float(tr["t"][r]), np.nanmean(P, axis=0) if al.any() else np.zeros(2), P))
        done = [math.nan] * len(wps)
        for r in range(len(tr["t"])):
            for i in range(min(int(tr["wp"][r]), len(wps))):
                if not math.isfinite(done[i]):
                    done[i] = float(tr["t"][r])
        n0 = int(tr["alive"][0].sum())
        deaths = flight.get("deaths") or {}
        ghost = SimpleNamespace(alive=tr["alive"][-1], kills=[(0.0, kk) for kk, v in deaths.items() for _ in range(v)],
                                contacts=int(flight.get("contacts") or 0), contact_ticks=0)
        b = T.score_flight(C, start, wps, poly, L, ghost, samples, done, n0)
        b.update(face_measures(C, face, samples, done))
        b["deaths"] = deaths
        b["other_contacts"] = int(flight.get("other_contacts") or 0)
        P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
        sim, rs, rdone = T.fly(T.model(over), C, P0, wps, d, fseed, settle=0.0)
        r = T.score_flight(C, start, wps, poly, L, sim, rs, rdone, sim.n0)
        r.update(face_measures(C, face, rs, rdone))
        return dict(set=label, idx=idx, kind=kind, k=k, d=d, **{f"u_{x}": v for x, v in b.items()},
                    **{f"r_{x}": v for x, v in r.items()})
    except Exception as e:
        return dict(set=label, idx=idx, error=repr(e), trace=traceback.format_exc()[-1500:])


def cmd_bench_gate(a):
    import pandas as pd
    from pathlib import Path
    from citysim import parallel
    jobs = []
    for run in a.runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        _, _, params, flights, _, _ = T.bench.read_results(run)
        fl = {(f["set"], f["scenario"]): f for f in flights}
        for ps in cfg["paramSets"]:
            eff = (params.get(ps["label"]) or {}).get("effective")
            over = ({k: float(f"{v:.7g}") if isinstance(v, float) else v for k, v in eff.items()} if eff
                    else dict(zip(ps["names"], ps["values"])))
            d = float(over.pop("d_ref"))
            for idx, sc in enumerate(cfg["scenarios"]):
                if (ps["label"], idx) in fl and (run / f"traj_{ps['label']}_{idx}.csv").exists():
                    jobs.append((str(run), ps["label"], idx, sc, over, d, fl[(ps["label"], idx)]))
    res = parallel.run_jobs(bench_job, jobs, "bench gate", processes=a.jobs)
    errs = [r for r in res if "error" in r]
    for e in errs[:3]:
        print(e["error"], e.get("trace", "")[-600:])
    df = pd.DataFrame([r for r in res if "error" not in r])
    out = OUT / a.name / "bench"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "gate.jsonl", orient="records", lines=True)
    rows = []
    for label, g in df.groupby("set", sort=False):
        for src in ("u", "r"):
            gg = g[[c for c in g.columns if c.startswith(f"{src}_")]].rename(columns=lambda c: c[2:])
            gg["kind"] = g["kind"].values
            rows.append(dict(set=label, src="unity" if src == "u" else "replica", **score_rows(gg)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate.csv", index=False)
    pd.set_option("display.width", 420)
    pd.set_option("display.max_columns", 80)
    cols = ["set", "src", "n", "km", "v_coh", "one", "lost_km", "contact_km", "open", "hull", "cross_wrapped",
            "cross_bld_pass", "street_v_coh", "street_one", "face_t_side", "face_t_split", "face_t_split_share",
            "face_t_wrap", "face_t_through", "face_stall", "face_contacts"]
    print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    if errs:
        print(f"[swarmshape] {len(errs)} flights failed to score")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gaps")
    p.add_argument("--seeds", type=int, nargs="+", default=[T.TUNE_SEED, *T.HELD_SEEDS])
    p = sub.add_parser("screen")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    for k in ("cross", "pass", "street", "face"):
        p.add_argument(f"--{k}", dest=k.replace("pass", "pass_"), type=int)
    p.add_argument("--k0", type=int, default=K_SCREEN)
    p.add_argument("--single", action="store_true")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("confirm")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--cross", type=int, default=40)
    p.add_argument("--pass", dest="pass_", type=int, default=30)
    p.add_argument("--street", type=int, default=20)
    p.add_argument("--face", type=int, default=36)
    p.add_argument("--k0", type=int, default=K_CONFIRM)
    p.add_argument("--single", action="store_true")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("table")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=60)
    p = sub.add_parser("rule")
    p.add_argument("--set", help="JSON overrides of RULE")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("choose")
    p.add_argument("--name", required=True)
    p = sub.add_parser("bench-plan")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--cross", type=int, default=30)
    p.add_argument("--pass", dest="pass_", type=int, default=20)
    p.add_argument("--street", type=int, default=14)
    p.add_argument("--face", type=int, default=36)
    p.add_argument("--k0", type=int, default=K_BENCH)
    p.add_argument("--drones", type=int, default=T.N_DRONES)
    p.add_argument("--timeout", type=int, default=180)
    p = sub.add_parser("bench-gate")
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=16)
    p = sub.add_parser("events")
    p.add_argument("--run", default="confirm", help="an S7 run folder under experiment_results/swarm_flow")
    p.add_argument("--labels", nargs="+", default=["T3", "slide", "adaptive", "slide + adaptive"])
    p.add_argument("--jobs", type=int, default=22)
    a = ap.parse_args()
    dict(gaps=cmd_gaps, screen=cmd_screen, confirm=cmd_confirm, table=cmd_table, rule=cmd_rule, choose=cmd_choose,
         events=cmd_events, **{"bench-plan": cmd_bench_plan, "bench-gate": cmd_bench_gate})[a.cmd](a)


if __name__ == "__main__":
    main()
