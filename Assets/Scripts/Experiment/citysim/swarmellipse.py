"""S10 Ellipse Avoidance (the ellipse study): each building avoided as the smallest ellipse round its footprint, its
aspect capped (SwarmManager.obstacleShape Ellipse, ellipseMaxAspect), against S8's enclosing circle, flown on S8's
harness (citysim.swarmshape, citysim.swarmflow, citysim.swarmtune) in BlockShapedPatchCityWorld.

    python -B -m citysim.swarmellipse gaps [--seeds 1 2 3 4]          # walls, streets and corridors each shape leaves
    python -B -m citysim.swarmellipse screen --name N --cand FILE [--cross 48 --pass 32 --street 24 --face 36
                                             --angle 36] [--probes FILE --along 120 --single-angle 36]
    python -B -m citysim.swarmellipse confirm --name N --cand FILE [...]      # fresh routes, the scene's city + held out
    python -B -m citysim.swarmellipse table --name N
    python -B -m citysim.swarmellipse rule [--set JSON] [--force]             # rule.json, written before the confirmation
    python -B -m citysim.swarmellipse choose --name confirm
    -> experiment_results/swarm_ellipse/<name>/

The question is the user's: the enclosing circle lies very close to some of a building's walls and far from others, so
a drone is pushed off a wall at a distance that depends on which wall it is; an ellipse fits the footprint closer, so
every wall should be avoided at a more even distance, and the narrow streets should open. Measured three ways:

  gaps    geometry, deterministic: per building, each wall's distance to the shape (at the wall's middle, along its
          outward normal; walls of WALL_MIN_M or more): the nearest, the farthest and the spread between them; per
          interior street with a building on each side, the narrowest gap the two shapes leave (S8's street_gaps), and
          the corridor left once d_obs is taken off both sides; how many streets each shape narrows and widens.
  along   one drone flown down an interior street parallel to one of its walls, ALONG_OFFSETS m off it, from the street's
          junction before the wall to the one after: how far it is pushed off the wall while alongside (`push`), the
          distance it keeps (`standoff`, the median while alongside; `closest`), its speed there, and the shape's own
          distance from that wall at its middle (`shell`). Even avoidance is a small spread of the standoff across
          walls at one offset; under the circle a drone 6 m off a wall the circle stands 7 m off is pushed out, and one
          6 m off a wall it stands 1 m off is not.
  angle   S8's head-on flight (face) at an angle: from 50 m out, at ANGLE_DEG from the face's normal, aimed through the
          building at one of S8's offsets. For the formation S8's head-on measures (split, wrap, through); for one
          drone also the facade distance where avoidance takes hold (`onset`: its speed towards the building first
          below ONSET_FRAC of its run-in speed) and the closest it comes.
plus S8's four flights (cross, pass, street, face) and measures. Every row names every law field: an empty override is
the scene, which has been C35a since S8. Route indices: screens from K_SCREEN, the confirmation from K_CONFIRM, the
probes' own from K_PROBE, the bench from K_BENCH; S6-S9 used 0-14299.
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

from citysim import REPO, blockcity  # noqa: E402
from citysim import split_probe as S  # noqa: E402
from citysim import swarmflow as F  # noqa: E402
from citysim import swarmshape as SH  # noqa: E402
from citysim import swarmtune as T  # noqa: E402

rep = T.rep
OUT = REPO / "experiment_results" / "swarm_ellipse"
# The scene's shape before S10's pick, for when it is adopted: an empty override then means the ellipse, so merge this
# under the `over` of any candidate flown against the scene as it was (S8's and S9's rows, and S10's own circle rows,
# which all name obstacleShape anyway). S8's PRE_S8 trap, again.
PRE_S10 = dict(obstacleShape=1)
K_SCREEN, K_CONFIRM, K_PROBE, K_BENCH = 15000, 16000, 17000, 18000
CAPS = (1.25, 1.5, 2.0)
WALL_MIN_M = 4.0
ALONG_OFFSETS = (6.0, 9.0, 12.5)     # the probe line's distance from the wall: 12.5 is the street's centre line
ALONG_RUN_M = (20.0, 15.0, 10.0, 5.0)  # how far past each end of the wall the line runs, longest that is clear
ALONG_CLEAR_M = 2.0
ALONG_WALL_MIN_M = 8.0
ANGLE_DEG = (0.0, 30.0, 60.0)        # the run-in's angle from the face's normal: square on, 60 and 30 deg to the face
ONSET_FRAC = 0.8
TRACK_EVERY = 5                      # ticks between track samples: 0.1 s, as the bench records
STD_KINDS = ("cross", "pass", "street")


# ============================================================================= shapes as ellipses

def shapes(rc, caps=CAPS):
    """{name: rep.Ellipses} every shape of the study written as an ellipse (a circle is one with a = b): the bounds
    circle (Cylinder), the enclosing circle (EnclosingCircle) and the ellipse at each cap (Ellipse). circ is cleared
    so every distance goes through one piece of code (rep.ell_frame), which is exact for a = b."""
    n = rc.n
    ex = np.tile([1.0, 0.0], (n, 1))
    out = {"bounds circle": rep.Ellipses(rc.c.copy(), ex, rc.R.copy(), rc.R.copy(), np.zeros(n, bool)),
           "enclosing circle": rep.Ellipses(rc.mec_c.copy(), ex, rc.mec_R.copy(), rc.mec_R.copy(), np.zeros(n, bool))}
    for cap in caps:
        E = rc.ellipses(cap)
        out[f"ellipse {cap:g}"] = rep.Ellipses(E.c.copy(), E.e.copy(), E.a.copy(), E.b.copy(), np.zeros(n, bool))
    return out


def model_shape(m, rc):
    """The rep.Ellipses a model avoids rc's buildings as (None under Footprint)."""
    if m.obstacle_shape == rep.FOOTPRINT:
        return None
    if m.obstacle_shape == rep.CYLINDER:
        return shapes(rc, ())["bounds circle"]
    if m.obstacle_shape == rep.ENCLOSING:
        return shapes(rc, ())["enclosing circle"]
    E = rc.ellipses(m.ellipse_cap)
    return rep.Ellipses(E.c, E.e, E.a, E.b, np.zeros(rc.n, bool))


def ray_exit(E, b, p, d):
    """How far along the ray p + s d (s >= 0) it leaves building b's ellipse (0 if p is outside and the ray misses)."""
    A = rep.ell_unit(p[None], E, b)[0]
    B = np.array([(d @ E.e[b]) / E.a[b], (d @ E.ep[b]) / E.b[b]])
    qa, qb, qc = B @ B, 2.0 * (A @ B), A @ A - 1.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0:
        return 0.0
    return max((-qb + math.sqrt(disc)) / (2.0 * qa), 0.0)


def walls(rc, b):
    """[(V, W, L, mid, t, n)] building b's walls of WALL_MIN_M or more: ends, length, middle, direction, outward
    normal."""
    p = rc.polys[b]
    out = []
    for e in range(len(p)):
        V, W = p[e], p[(e + 1) % len(p)]
        L = float(np.linalg.norm(W - V))
        if L < WALL_MIN_M:
            continue
        t = (W - V) / L
        out.append((V, W, L, 0.5 * (V + W), t, np.array([t[1], -t[0]])))
    return out


def wall_distances(rc, E):
    """(n,) lists: for every building, each wall's distance to the shape at the wall's middle along its normal."""
    return [[ray_exit(E, b, mid, nrm) for _, _, _, mid, _, nrm in walls(rc, b)] for b in range(rc.n)]


def street_pairs(seed):
    """[(building a, building b, sample points (k, 2), street polyline)] every interior street with a building on each
    side (S8's street_gaps' streets, sampled the same way)."""
    lay = json.loads(blockcity.ensure(T.params(seed)).read_text(encoding="utf-8"))
    bidx = [blk.get("building") for blk in lay["blocks"]]
    out = []
    for s in lay["streets"]:
        bl = s.get("blocks") or []
        if len(bl) != 2 or s.get("edge"):
            continue
        ia, ib = bidx[bl[0]], bidx[bl[1]]
        if ia is None or ib is None or ia < 0 or ib < 0:
            continue
        pl = np.array(s["polyline"], float)
        pts = np.vstack([pl[i] + np.linspace(0, 1, 9)[:, None] * (pl[i + 1] - pl[i]) for i in range(len(pl) - 1)])
        out.append((ia, ib, pts, pl))
    return out


def street_gaps(rc, pairs, E):
    """(n_streets,) the narrowest gap the two buildings' shapes leave along each street: S8's measure, through
    rep.ell_frame's signed distance (exactly S8's numbers for the two circles)."""
    g = []
    for ia, ib, pts, _ in pairs:
        sd = rep.ell_frame(pts, E, np.array([ia, ib]))[5]
        g.append(float(sd.sum(1).min()))
    return np.array(g)


def cmd_gaps(a):
    import pandas as pd
    m = T.model(dict(obstacleShape=1))
    S_ = m.S
    wrows, grows = [], []
    for seed in a.seeds:
        rc = T.city(seed)["rcity"]
        pairs = street_pairs(seed)
        fac = np.array([float(rc.distance(pts, np.array([ia, ib])).sum(1).min()) for ia, ib, pts, _ in pairs])
        sh = shapes(rc)
        g_enc = None
        for name, E in sh.items():
            wd = wall_distances(rc, E)
            near = np.array([min(w) for w in wd if w])
            far = np.array([max(w) for w in wd if w])
            allw = np.concatenate([w for w in wd if w])
            wrows.append(dict(seed=seed, shape=name, buildings=len(near), walls=len(allw),
                              wall_med=np.median(allw), wall_p90=np.percentile(allw, 90),
                              nearest_med=np.median(near), farthest_med=np.median(far),
                              farthest_p90=np.percentile(far, 90), spread_med=np.median(far - near),
                              spread_p90=np.percentile(far - near, 90), bld_wall_over6=int((far > 6.0).sum()),
                              walls_over6=float((allw > 6.0).mean())))
            g = street_gaps(rc, pairs, E)
            if name == "enclosing circle":
                g_enc = g
            row = dict(seed=seed, shape=name, streets=len(g), gap_med=np.median(g), gap_p10=np.percentile(g, 10),
                       gap_p1=np.percentile(g, 1), closed=int((g < 0).sum()), under10=int((g < 10).sum()))
            for d_obs in (0.25, 0.35, 0.45):
                h = g - 2 * d_obs * S_
                row.update({f"corr{int(d_obs * 100)}_med": np.median(h), f"corr{int(d_obs * 100)}_p10":
                            np.percentile(h, 10), f"corr{int(d_obs * 100)}_under6": int((h < 6).sum())})
            if g_enc is not None and name.startswith("ellipse"):
                dg = g - g_enc
                row.update(narrower=float((dg < -1e-6).mean()), narrower1=float((dg < -1).mean()),
                           narrower2=float((dg < -2).mean()), wider1=float((dg > 1).mean()),
                           wider4=float((dg > 4).mean()), worst=float(dg.min()), narrowed_min=float(g[dg < -1e-6].min())
                           if (dg < -1e-6).any() else math.nan)
            grows.append(row)
        grows.append(dict(seed=seed, shape="facades", streets=len(fac), gap_med=np.median(fac),
                          gap_p10=np.percentile(fac, 10), gap_p1=np.percentile(fac, 1), closed=0,
                          under10=int((fac < 10).sum())))
        print(f"seed {seed}: {rc.n} buildings, {len(pairs)} streets", flush=True)
    wdf, gdf = pd.DataFrame(wrows).round(3), pd.DataFrame(grows).round(3)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(wdf.to_string(index=False))
    print(gdf.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    wdf.to_csv(OUT / "walls.csv", index=False)
    gdf.to_csv(OUT / "gaps.csv", index=False)


def cmd_fitcheck(a):
    """Unity's ellipses (Assets/Scripts/swarm/Bench/Editor/EllipseFitProbe.cs, run in batchmode on the bench copy)
    against the replica's, building by building, matched by the enclosing circle's centre: the gate the C# port has
    to pass (the plan's 1 cm)."""
    fits = json.loads((OUT / a.file).read_text(encoding="utf-8"))
    rc = T.city(T.TUNE_SEED)["rcity"]
    worst = dict(circle=0.0)
    flags = 0
    for f in fits:
        cu = np.array(f["circle"][:2])
        i = int(np.argmin(np.linalg.norm(rc.mec_c - cu, axis=1)))
        worst["circle"] = max(worst["circle"], float(np.linalg.norm(rc.mec_c[i] - cu)),
                              abs(float(rc.mec_R[i]) - f["circle"][2]))
        for cap, (cx, cz, ex, ez, ea, eb, circ) in f["ellipses"].items():
            E = rc.ellipses(float(cap))
            flags += int(bool(circ) != bool(E.circ[i]))
            if circ or E.circ[i]:
                continue
            # the same ellipse whichever way the axis points: compare the boundary at the four axis ends
            ends_u = [np.array([cx, cz]) + s * np.array([ex, ez]) * ea for s in (-1, 1)] + \
                     [np.array([cx, cz]) + s * np.array([-ez, ex]) * eb for s in (-1, 1)]
            sd = rep.ell_frame(np.array(ends_u), E, np.array([i]))[5][:, 0]
            k = f"cap {cap}"
            worst[k] = max(worst.get(k, 0.0), float(np.abs(sd).max()), float(np.linalg.norm(E.c[i] - [cx, cz])))
    print(f"[swarmellipse] {len(fits)} Unity colliders against {rc.n} replica buildings; circle-flag disagreements "
          f"{flags}")
    for k, v in worst.items():
        print(f"  {k}: worst disagreement {v * 100:.3f} cm")
    ok = flags == 0 and all(v <= 0.01 for v in worst.values()) and len(fits) == rc.n
    print("  PASS (within 1 cm)" if ok else "  FAIL")


# ============================================================================= the probes

class TrackLog:
    """A read-only fly() hook: every live drone's position every TRACK_EVERY ticks (0.1 s, as the bench records)."""

    def __init__(self):
        self.t, self.P, self.n = [], [], 0

    def start(self, sim):
        pass

    def step(self, sim, t, cmd, wp):
        if self.n % TRACK_EVERY == 0:
            self.t.append(float(t))
            self.P.append(np.where(sim.alive[:, None], sim.P, np.nan))
        self.n += 1


def along_route(C, seed, k, rng):
    """(start, waypoints, route length, polyline, wall) of along-probe k: a wall drawn from rng among the interior
    streets' facing walls of ALONG_WALL_MIN_M or more, the offset ALONG_OFFSETS[k % 3]; the line runs parallel to the
    wall from ALONG_RUN_M past one end to as far past the other, the longest run that keeps ALONG_CLEAR_M off every
    footprint."""
    rc = C["rcity"]
    pairs = _pairs(seed)
    allb = np.arange(rc.n)
    off = ALONG_OFFSETS[k % len(ALONG_OFFSETS)]
    for _ in range(200):
        ia, ib, pts, pl = pairs[int(rng.integers(len(pairs)))]
        b = ia if rng.random() < 0.5 else ib
        mid_st = pts[len(pts) // 2]
        p = rc.polys[b]
        sd = rc.N[b, :len(p)] @ mid_st - rc.Cn[b, :len(p)]
        e = int(np.argmax(sd))
        V, W = p[e], p[(e + 1) % len(p)]
        L = float(np.linalg.norm(W - V))
        if L < ALONG_WALL_MIN_M:
            continue
        t = (W - V) / L
        n = rc.N[b, e]
        mid = 0.5 * (V + W)
        if rng.random() < 0.5:
            t = -t
        for run in ALONG_RUN_M:
            s = np.linspace(-(L / 2 + run), L / 2 + run, int(L + 2 * run) + 2)
            line = mid[None] + off * n[None] + s[:, None] * t[None]
            if rc.distance(line, allb).min() >= ALONG_CLEAR_M:
                start, end = line[0], line[-1]
                dist = float(np.linalg.norm(end - start))
                wps = [(float(end[0]), float(end[1]), 1.0, S.timeout_for(dist))]
                wall = dict(b=int(b), e=e, L=L, mid=mid, t=t, n=n, off=off, run=run)
                return start, wps, dist, np.vstack([start, end]), wall
    raise RuntimeError(f"no clear wall for an along probe in city seed {seed}")


_PAIRS = {}


def _pairs(seed):
    if seed not in _PAIRS:
        _PAIRS[seed] = street_pairs(seed)
    return _PAIRS[seed]


def along_measures(C, wall, log, E):
    """The probe's measures while the drone is alongside the wall (its projection within the wall's length)."""
    out = dict(along_b=wall["b"], along_off=wall["off"], along_L=round(wall["L"], 1), alongside=False,
               shell=round(ray_exit(E, wall["b"], wall["mid"], wall["n"]), 3) if E is not None else 0.0)
    for key in ("standoff", "closest", "push", "inward", "v_along"):
        out[key] = None
    if not log.t:
        return out
    t = np.array(log.t)
    P = np.array([p[0] for p in log.P])
    ok = np.isfinite(P).all(1)
    rel = P - wall["mid"][None]
    s, lat = rel @ wall["t"], rel @ wall["n"]
    side = ok & (np.abs(s) <= wall["L"] / 2)
    if side.sum() < 2:
        return out
    out.update(alongside=True, standoff=round(float(np.median(lat[side])), 3),
               closest=round(float(lat[side].min()), 3), push=round(float(lat[side].max() - wall["off"]), 3),
               inward=round(float(wall["off"] - lat[side].min()), 3))
    ts, ss = t[side], s[side]
    out["v_along"] = round(float((ss[-1] - ss[0]) / max(ts[-1] - ts[0], 1e-6)), 3)
    return out


def angle_route(C, seed, k, rng):
    """S8's face_route at an angle: offset FACE_OFFSETS[k % 3], angle ANGLE_DEG[(k // 3) % 3] from the face's normal
    (its sign, and the offset's, drawn from rng); the run-in must clear every other footprint by FACE_CLEAR_M on lines
    across the corridor, as faces() asks of the square-on run-in. Not the target's own: at an angle the lines beside
    the aim point run into its face, which is the approach being flown (square on they never do, so every face of
    faces() passes either test and the square-on routes are the same)."""
    fs = SH.faces(C, seed)
    rc = C["rcity"]
    allb = np.arange(rc.n)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    off = SH.FACE_OFFSETS[k % len(SH.FACE_OFFSETS)] * sign
    beta = math.radians(ANGLE_DEG[(k // len(SH.FACE_OFFSETS)) % len(ANGLE_DEG)])
    turn = 1.0 if rng.random() < 0.5 else -1.0
    for _ in range(80):
        f = fs[int(rng.integers(len(fs)))]
        aim = f["m"] + off * f["L"] / 2 * f["t"]
        back = math.cos(beta) * f["n"] + turn * math.sin(beta) * f["t"]       # from the aim point out to the start
        u = -back
        across = np.array([-u[1], u[0]])
        probes = [aim + s * back + lat * across for s in np.linspace(3.0, SH.FACE_RUN_M + 10.0, 20)
                  for lat in SH.FACE_LAT_M]
        if rc.distance(np.array(probes), allb[allb != f["b"]]).min() < SH.FACE_CLEAR_M:
            continue
        start = aim + SH.FACE_RUN_M * back
        line = aim[None] + np.arange(1.0, SH.FACE_BEYOND_M)[:, None] * u[None]
        d_own = rc.distance(line, np.array([f["b"]]))[:, 0]
        d_all = rc.distance(line, allb).min(1)
        past = np.flatnonzero((np.cumsum(d_own == 0) > 0) & (d_own > 0) & (d_all >= SH.FACE_FREE_M))
        if not len(past):
            continue
        far = line[past[0]]
        dist = float(np.linalg.norm(far - start))
        wps = [(float(far[0]), float(far[1]), 1.0, S.timeout_for(dist))]
        return start, wps, dist, np.vstack([start, far]), dict(f, off=off, aim=aim, angle=math.degrees(beta) * turn)
    raise RuntimeError(f"no clear face for a run-in at {math.degrees(beta):g} deg in city seed {seed}")


def approach_measures(C, face, log):
    """One drone's approach to the face's building: `onset`, the facade distance where its speed towards the building
    first falls below ONSET_FRAC of its run-in speed (25-40 m out), and `closest`."""
    out = dict(onset=None, closest=None)
    if len(log.t) < 3:
        return out
    t = np.array(log.t)
    P = np.array([p[0] for p in log.P])
    ok = np.isfinite(P).all(1)
    t, P = t[ok], P[ok]
    if len(t) < 3:
        return out
    d = C["rcity"].distance(P, np.array([face["b"]]))[:, 0]
    out["closest"] = round(float(d.min()), 3)
    vn = -np.gradient(d, t)
    ref = vn[(d > 25.0) & (d < 40.0)]
    if not len(ref) or np.median(ref) <= 0.5:
        return out
    v0 = float(np.median(ref))
    i0 = int(np.flatnonzero(d < 25.0)[0]) if (d < 25.0).any() else len(d)
    slow = np.flatnonzero(vn[i0:] < ONSET_FRAC * v0)
    if len(slow):
        out["onset"] = round(float(d[i0 + slow[0]]), 3)
    return out


def run_job(j):
    """One flight: S8's (swarmshape.run_job) for cross, pass, street and face; the angled run-in and the along probe
    here."""
    if j["kind"] not in ("angle", "along"):
        return SH.run_job(j)
    t0 = time.time()
    base = dict(job_id=T.job_id(j), cand=j["cand"], kind=j["kind"], seed=j["seed"], k=j["k"], d=j["d"])
    try:
        C = T.city(j["seed"])
        rng = np.random.default_rng(T.route_seed(j["kind"], j["seed"], j["k"]))
        fseed = int(rng.integers(1 << 30))
        route = angle_route if j["kind"] == "angle" else along_route
        start, wps, L, poly, info = route(C, j["seed"], j["k"], rng)
        over = dict(j["over"])
        m = T.model(over)
        single = j.get("single") or j["kind"] == "along"
        P0 = start[None].copy() if single else T.start_positions(C, j["d"], fseed, over, start)
        log = TrackLog()
        sim, samples, done = T.fly(m, C, P0, wps, j["d"], fseed, hook=log)
        r = dict(base, **T.score_flight(C, start, wps, poly, L, sim, samples, done, sim.n0))
        if j["kind"] == "angle":
            r.update(SH.face_measures(C, info, samples, done), angle=round(info["angle"], 1))
            if single:
                r.update(approach_measures(C, info, log))
        else:
            r.update(along_measures(C, info, log, model_shape(m, C["rcity"])))
        return dict(r, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


# ============================================================================= tables

def angle_scores(g, pre="ang"):
    """S8's head-on scores over angled flights, by angle too, and the one-drone approach's onset and closest."""
    r = {f"{pre}_{k[5:]}": v for k, v in SH.face_scores(g).items()}
    for ang, gg in g.groupby(g.angle.abs().round(0)):
        for k in ("t_split", "t_through"):
            v = gg[k].astype(float)
            r[f"{pre}_{k}_a{ang:g}"] = float(np.fmin(v.fillna(SH.FACE_CAP_S), SH.FACE_CAP_S).mean())
    if "onset" in g:
        on = g.onset.astype(float).dropna()
        if len(on):
            r.update({f"{pre}_onset": float(on.median()), f"{pre}_onset_spread": float(np.percentile(on, 90) -
                     np.percentile(on, 10)), f"{pre}_closest": float(g.closest.astype(float).median())})
    return r


def along_scores(g):
    """The along probes: per offset the standoff's median and spread across walls (p90 - p10), the push's, the share
    of walls a drone was pushed more than 0.5 m off its line, and the speed alongside; `along_spread` is the standoff's
    spread averaged over the offsets: the headline."""
    r = dict(along_n=len(g), along_alongside=float(g.alongside.mean()))
    a = g[g.alongside.astype(bool)]
    spreads = []
    for off, gg in a.groupby("along_off"):
        sd = gg.standoff.astype(float)
        pu = gg.push.astype(float)
        key = f"o{off:g}"
        sp = float(np.percentile(sd, 90) - np.percentile(sd, 10))
        spreads.append(sp)
        r.update({f"along_{key}_standoff": float(sd.median()), f"along_{key}_spread": sp,
                  f"along_{key}_push": float(pu.median()), f"along_{key}_push_p90": float(np.percentile(pu, 90)),
                  f"along_{key}_pushed": float((pu > 0.5).mean()), f"along_{key}_v": float(gg.v_along.median()),
                  f"along_{key}_shell_corr": float(np.corrcoef(gg.shell, pu)[0, 1]) if len(gg) > 2 else math.nan})
    r["along_spread"] = float(np.mean(spreads)) if spreads else math.nan
    r["along_contacts"] = float(g.contacts.mean())
    r["along_lost"] = float(g.lost.mean())
    return r


def score_rows(g):
    std = g[g.kind.isin(STD_KINDS)]
    r = T.score_candidate(std) if len(std) else {}
    r.update(SH.face_scores(g[g.kind == "face"]))
    if (g.kind == "angle").any():
        r.update(angle_scores(g[g.kind == "angle"]))
    if (g.kind == "along").any():
        r.update(along_scores(g[g.kind == "along"]))
    if "sigma_mean" in g and len(std):
        r["sigma"] = float(std.sigma_mean.mean())
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
        "street_v_coh", "street_lost_km", "street_contact_km", "face_t_split", "face_t_split_share", "face_stall",
        "ang_t_split", "ang_t_split_share", "ang_onset", "ang_onset_spread", "ang_closest", "along_spread",
        "along_o6_spread", "along_o6_pushed", "along_o9_spread", "along_o12.5_spread", "sigma"]


def show(tab, n=80):
    import pandas as pd
    pd.set_option("display.width", 460)
    pd.set_option("display.max_columns", 80)
    print(tab[[c for c in COLS if c in tab.columns]].head(n).round(3).to_string(index=False))


# ============================================================================= runs

def jobs_for(spec, seeds, flights, k0, single=False):
    """F.jobs_for, every candidate on every flight kind in `flights`; the jobs ordered by route index then kind, so a
    run stopped early has flown every candidate equally (lesson 17)."""
    cands, jobs = F.jobs_for(spec, seeds, flights, k0, single)
    return cands, sorted(jobs, key=lambda j: (j["k"], j["kind"], j["seed"]))


def probe_jobs(spec, seeds, along, angle, k0):
    """The one-drone probes for each candidate (their own ids, `single-` + the S8 hash, so they never mix with the
    formation's rows): `along` along-probes and `angle` angled run-ins per city."""
    cands, jobs = {}, []
    for c in spec:
        sid = "single-" + T.cand_id(c["over"], 0.0)
        cands[sid] = dict(over=c["over"], d=c["d"], label=f"one drone, {c.get('label', '')}")
        for seed in seeds:
            jobs += [dict(cand=sid, over=c["over"], d=float(c["d"]), kind="along", seed=seed, k=k0 + k, single=True)
                     for k in range(along)]
            jobs += [dict(cand=sid, over=c["over"], d=float(c["d"]), kind="angle", seed=seed, k=k0 + k, single=True)
                     for k in range(angle)]
    return cands, sorted(jobs, key=lambda j: (j["k"], j["kind"], j["seed"]))


def _run(a, seeds, fl, k0, label):
    spec = F.load_cands(a.cand)
    cands, jobs = jobs_for(spec, seeds, fl, k0)
    if a.probes:
        pc, pj = probe_jobs(F.load_cands(a.probes), seeds, a.along, a.single_angle, a.probe_k0)
        cands.update(pc)
        jobs += pj
    T.run(a.name, cands, jobs, a.jobs, label, root=OUT, fn=run_job)
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab)


def cmd_screen(a):
    fl = dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face, angle=a.angle)
    _run(a, (T.TUNE_SEED,), {k: v for k, v in fl.items() if v}, a.k0, "screen")


def cmd_confirm(a):
    fl = dict(cross=a.cross, pass_=a.pass_, street=a.street, face=a.face, angle=a.angle)
    _run(a, (T.TUNE_SEED,) + T.HELD_SEEDS, {k: v for k, v in fl.items() if v}, a.k0, "confirm")


def cmd_table(a):
    tab = table(a.name)
    tab.round(4).to_csv(OUT / a.name / "table.csv", index=False)
    show(tab, a.n)


# ============================================================================= the rule (rule.json, fixed before the confirmation)

# base: the scene (C35a), what the trials fly; each ellipse row is also judged against its partner, the circle at its
# own d_obs and spread mode. Safety is judged against both (lesson 44: the scene's bound). Loss events are drones lost
# less the partner of each collision (S9), tested one-sided and exact on their rate per km (swarmspread.worse_p);
# contacts as a ratio of rates; the split on square-on flights only (an angled pass the formation takes whole on one
# side is not a failure: manual section 5). Delivers: the user's question in flight -- the one-drone along probes'
# standoff spread across walls -- at most spread_ratio of the partner's, with nothing else worse than the scene.
# The corridor after the standoff is reported for the user's choice, not judged (their call, 2026-11-02).
# The pick is by measured fit, not by cap: the rows within spread_tie of the smallest standoff spread, then the
# roundest of those (lesson 43: a longer ellipse has flatter sides, and in the screen cap 2 fitted no closer than 1.5),
# then the best score. The plan had "the largest cap"; changed before the rule was written, for that reason.
RULE = dict(base="C35a", safe_p=0.10, contact_ratio=1.25, open_margin=0.02, hull_min=0.5, split_ratio=1.1,
            spread_ratio=0.8, street_floor=0.98, wrap_margin=0.05, spread_tie=0.05)


def _key(over, spread=True):
    """(shape, cap, d_obs[, adaptive]) of a candidate's law: what pairs a row with its partner, its probe and its
    d_ref 1.2 row."""
    shape = int(over.get("obstacleShape", 1))
    k = (shape, round(float(over.get("ellipseMaxAspect", 1.5)), 3) if shape == rep.ELLIPSE else None,
         round(float(over.get("d_obs", 0.35)), 3))
    return k + (bool(over.get("adaptiveSpread", False)),) if spread else k


def rows_for_rule(name):
    """Every d_ref 2.0 formation row of a run with what the rule reads: S6's measures, loss events and km over the
    cross, pass and street flights, its square-on split at d_ref 2.0 and 1.2, and its probe's standoff spread."""
    df, cands, _ = T.read(name, OUT)
    tab = table(name)
    std = df[df.kind.isin(STD_KINDS)]
    ev = std.groupby("cand").apply(lambda g: int((g.lost - g.lost_drone).sum()))
    km = std.groupby("cand").L.sum() / 1000.0
    probe = {_key(cands[c]["over"], False): r for c, r in tab.set_index("cand").iterrows() if c.startswith("single")}
    at12 = {_key(cands[c]["over"]): r for c, r in tab.set_index("cand").iterrows()
            if not c.startswith("single") and float(cands[c]["d"]) == 1.2}
    out = {}
    for c, r in tab.set_index("cand").iterrows():
        if c.startswith("single") or float(cands[c]["d"]) != 2.0:
            continue
        over = cands[c]["over"]
        pr = probe.get(_key(over, False))
        r12 = at12.get(_key(over))
        out[cands[c]["label"].split()[0]] = dict(
            cand=c, over=over, key=_key(over), events=int(ev.get(c, 0)), km=float(km.get(c, 0.0)),
            contact_km=float(r.contact_km), open=float(r.open), hull=float(r.hull), v_coh=float(r.v_coh),
            street_v_coh=float(r.street_v_coh), wrapped=float(r.cross_wrapped), score=float(r.score),
            split=float(r.face_t_split), split12=float(r12.face_t_split) if r12 is not None else math.nan,
            along_spread=float(pr.along_spread) if pr is not None else math.nan,
            aim_circle=over.get("x:ell_aim") == "circle")
    return out


def judge(rows, rule):
    """(verdicts, pick): every ellipse row against its partner and the scene."""
    from citysim import swarmspread as SS
    base = rows[rule["base"]]
    circles = {r["key"]: code for code, r in rows.items() if r["key"][0] == rep.ENCLOSING}
    out = {}
    for code, r in rows.items():
        if r["key"][0] != rep.ELLIPSE or r["aim_circle"]:
            continue
        part = rows[circles[(rep.ENCLOSING, None) + r["key"][2:]]]
        checks = {}
        for who, x in (("partner", part), ("scene", base)):
            checks[f"losses vs {who}"] = SS.worse_p(r["events"], x["events"], r["km"], x["km"]) >= rule["safe_p"]
            checks[f"contacts vs {who}"] = r["contact_km"] <= rule["contact_ratio"] * x["contact_km"]
            checks[f"in pieces vs {who}"] = r["open"] <= x["open"] + rule["open_margin"]
            checks[f"split vs {who}"] = r["split"] <= rule["split_ratio"] * x["split"]
            if math.isfinite(r["split12"]) and math.isfinite(x["split12"]):
                checks[f"split at d_ref 1.2 vs {who}"] = r["split12"] <= rule["split_ratio"] * x["split12"]
        checks["hull"] = r["hull"] >= rule["hull_min"]
        safe = all(checks.values())
        probe_part = part["along_spread"]
        dl = dict(spread=r["along_spread"] <= rule["spread_ratio"] * probe_part,
                  street=r["street_v_coh"] >= rule["street_floor"] * base["street_v_coh"],
                  wrapped=r["wrapped"] >= base["wrapped"] - rule["wrap_margin"])
        out[code] = dict(partner=[k for k, v in rows.items() if v is part][0], safe=safe, delivers=all(dl.values()),
                         missed=[k for k, v in {**checks, **dl}.items() if not v], cap=r["key"][1], score=r["score"],
                         spread_ratio=r["along_spread"] / probe_part, street_gain=r["street_v_coh"] /
                         base["street_v_coh"] - 1, events=r["events"], km=round(r["km"], 1),
                         loss_p_partner=SS.worse_p(r["events"], part["events"], r["km"], part["km"]),
                         loss_p_scene=SS.worse_p(r["events"], base["events"], r["km"], base["km"]),
                         contact_ratio_scene=r["contact_km"] / base["contact_km"])
    ok = [k for k, v in out.items() if v["safe"] and v["delivers"]]
    if not ok:
        return out, None
    best = min(out[k]["spread_ratio"] for k in ok)
    near = [k for k in ok if out[k]["spread_ratio"] <= (1 + rule["spread_tie"]) * best]
    return out, min(near, key=lambda k: (out[k]["cap"], -out[k]["score"]))


def corridors(seeds=(1, 2, 3, 4)):
    """{(cap or None, d_obs): (median, p10, streets under 6 m)} the corridor each law leaves, over the cities."""
    m = T.model(dict(obstacleShape=1))
    gs = {}
    for seed in seeds:
        rc = T.city(seed)["rcity"]
        pairs = _pairs(seed)
        for name, E in shapes(rc).items():
            if name == "bounds circle":
                continue
            cap = None if name == "enclosing circle" else float(name.split()[1])
            gs.setdefault(cap, []).append(street_gaps(rc, pairs, E))
    out = {}
    for cap, g in gs.items():
        g = np.concatenate(g)
        for d_obs in (0.25, 0.35, 0.45):
            h = g - 2 * d_obs * m.S
            out[(cap, d_obs)] = (float(np.median(h)), float(np.percentile(h, 10)), int((h < 6).sum()))
    return out


def cmd_rule(a):
    path = OUT / "rule.json"
    if path.exists() and not a.force:
        raise SystemExit(f"[swarmellipse] {path} exists: the rule is fixed (--force to overwrite)")
    OUT.mkdir(parents=True, exist_ok=True)
    rule = dict(RULE, **({} if not a.set else json.loads(a.set)))
    path.write_text(json.dumps(dict(rule=rule, written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1),
                    encoding="utf-8")
    print(json.dumps(rule, indent=1))


def cmd_choose(a):
    rule = json.loads((OUT / "rule.json").read_text(encoding="utf-8"))["rule"]
    rows = rows_for_rule(a.name)
    verdicts, pick = judge(rows, rule)
    cor = corridors()
    for k, v in sorted(verdicts.items(), key=lambda kv: (-kv[1]["cap"], -kv[1]["score"])):
        r = rows[k]
        c = cor[(r["key"][1], r["key"][2])]
        print(f"{k:10s} vs {v['partner']:5s} safe {str(v['safe']):5s} delivers {str(v['delivers']):5s} "
              f"spread x{v['spread_ratio']:.2f} street {v['street_gain']:+.1%} events {v['events']} in {v['km']} km "
              f"(p {v['loss_p_partner']:.2f} / {v['loss_p_scene']:.2f}) contacts x{v['contact_ratio_scene']:.2f} "
              f"corridor {c[0]:.1f} m (p10 {c[1]:.1f}, {c[2]} under 6 m)  missed {v['missed'] or '-'}")
    sc = cor[(None, 0.35)]
    print(f"the scene's corridor: {sc[0]:.1f} m (p10 {sc[1]:.1f}, {sc[2]} under 6 m)")
    print(f"\nrule: {pick or 'none qualifies: report every bar missed'}")
    (OUT / a.name / "verdicts.json").write_text(json.dumps(dict(rule=rule, verdicts=verdicts, pick=pick,
                                                                corridors={f"{k[0]}|{k[1]}": v for k, v in cor.items()}),
                                                           indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gaps")
    p.add_argument("--seeds", type=int, nargs="+", default=[T.TUNE_SEED, *T.HELD_SEEDS])
    for name, k0, pk0, d in (("screen", K_SCREEN, K_PROBE, (48, 32, 24, 36, 36, 120, 36)),
                             ("confirm", K_CONFIRM, K_PROBE + 500, (40, 30, 200, 36, 36, 180, 36))):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.add_argument("--cand", required=True)
        p.add_argument("--probes", help="candidates for the one-drone probes (along, angle)")
        for k, v in zip(("cross", "pass", "street", "face", "angle", "along", "single-angle"), d):
            p.add_argument(f"--{k}", dest=k.replace("pass", "pass_").replace("-", "_"), type=int, default=v)
        p.add_argument("--k0", type=int, default=k0)
        p.add_argument("--probe-k0", dest="probe_k0", type=int, default=pk0)
        p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("table")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=80)
    p = sub.add_parser("fitcheck")
    p.add_argument("--file", default="unity_fits.json")
    p = sub.add_parser("rule")
    p.add_argument("--set", help="JSON overrides of RULE")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("choose")
    p.add_argument("--name", required=True)
    a = ap.parse_args()
    dict(gaps=cmd_gaps, screen=cmd_screen, confirm=cmd_confirm, table=cmd_table, fitcheck=cmd_fitcheck, rule=cmd_rule,
         choose=cmd_choose)[a.cmd](a)


if __name__ == "__main__":
    main()
