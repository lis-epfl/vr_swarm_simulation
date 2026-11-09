"""Long sight lines down a block city's streets: the measure, and the pass that shortens them (blockcity --sight-tries).

From every interior street midpoint (a layout street that is not on the arena's edge, at least EDGE_SKIP_M inside the
arena) rays are cast every STEP_DEG against the buildings' footprints, capped at the arena edge, and the longest clear
one is that midpoint's view. It is 2D, which is right while the drones fly under the roofs, and it counts a view in any
direction, diagonals across junctions included: a straight line a pilot can see down.

The pass moves block seeds, never buildings: it picks a long view (weighted by its excess over FLOOR_M), takes a seed
of one of the six blocks whose centres lie nearest that line of sight, and moves it 3-8 m towards the line, so that
block's building steps into the gap. A move is kept only when the summed excess falls and nothing else gets worse:
the sides drawn as streets, the plazas, and the streets Unity's avoidance cylinders close or narrow below 6 m. A move
re-casts only the midpoints within LOCAL_M of the moved seed (a full re-measure every REMEASURE_EVERY kept moves). It
is deterministic in the layout's seed and the number of tries; a try costs ~1.5-4 s.

Measured at p 40 s 18 over seeds 1-3 (2026-10-29): S4's RSA seeding gives a median view of 268 m with 3% of views
over 400 m; hex 0.15 584 m and 98% (its streets leave straight channels in three directions); hex patches of 150 m
(blockcity --seeds patch) 278 m and 7%; patches plus ~480 tries 251 m and 1%, keeping the patches' 5.06 sides.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

STEP_DEG = 2.0            # ray fan resolution
EDGE_SKIP_M = 60.0        # midpoints at least this far inside the arena
LONG_M = 200.0            # a long view
FLOOR_M = 150.0           # the pass's cost: sum over midpoints of (view - FLOOR_M), where positive
LOCAL_M = 300.0           # a move re-casts the midpoints this close to the moved seed
REMEASURE_EVERY = 40      # kept moves between full re-measures

_ang = np.deg2rad(np.arange(0.0, 360.0, STEP_DEG))
DIRS = np.c_[np.cos(_ang), np.sin(_ang)]


def segments(polys):
    """(start points, edge vectors) of every polygon edge."""
    A = np.vstack(polys)
    B = np.vstack([np.roll(p, -1, 0) for p in polys])
    return A, B - A


def cast(o, A, E, arena, dirs=DIRS):
    """Clear distance from o along each direction: the first building edge, or the arena edge."""
    x0, x1, z0, z1 = arena
    den = dirs[:, 0, None] * E[None, :, 1] - dirs[:, 1, None] * E[None, :, 0]
    w = A - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * E[:, 1] - w[:, 1] * E[:, 0])[None, :] / den
        s = (w[:, 0][None, :] * dirs[:, 1, None] - w[:, 1][None, :] * dirs[:, 0, None]) / den
        t = np.where((np.abs(den) > 1e-9) & (t > 1e-6) & (s >= 0) & (s <= 1), t, np.inf).min(1)
        tx = np.where(dirs[:, 0] > 0, (x1 - o[0]) / dirs[:, 0],
                      np.where(dirs[:, 0] < 0, (x0 - o[0]) / dirs[:, 0], np.inf))
        tz = np.where(dirs[:, 1] > 0, (z1 - o[1]) / dirs[:, 1],
                      np.where(dirs[:, 1] < 0, (z0 - o[1]) / dirs[:, 1], np.inf))
    return np.minimum(t, np.minimum(tx, tz))


def inside_any(pts, polys):
    """Points inside any convex CCW polygon."""
    pts = np.asarray(pts, float)
    out = np.zeros(len(pts), bool)
    for p in polys:
        lo, hi = p.min(0), p.max(0)
        cand = np.all((pts >= lo) & (pts <= hi), 1) & ~out
        if not cand.any():
            continue
        q = pts[cand]
        e = np.roll(p, -1, 0) - p
        cr = e[None, :, 0] * (q[:, None, 1] - p[None, :, 1]) - e[None, :, 1] * (q[:, None, 0] - p[None, :, 0])
        out[np.flatnonzero(cand)[np.all(cr >= -1e-9, 1)]] = True
    return out


def street_midpoints(streets, polys, arena):
    """Midpoints of the interior layout streets, EDGE_SKIP_M inside the arena and outside every footprint."""
    x0, x1, z0, z1 = arena
    m = np.array([(np.array(s["xz"][:2]) + np.array(s["xz"][2:])) / 2 for s in streets if not s["edge"]])
    keep = ((m[:, 0] > x0 + EDGE_SKIP_M) & (m[:, 0] < x1 - EDGE_SKIP_M) &
            (m[:, 1] > z0 + EDGE_SKIP_M) & (m[:, 1] < z1 - EDGE_SKIP_M))
    m = m[keep]
    return m[~inside_any(m, polys)]


def measure(polys, mids, arena):
    """(longest view, its direction index into DIRS) per midpoint."""
    A, E = segments(polys)
    best, bdir = np.zeros(len(mids)), np.zeros(len(mids), int)
    for k, o in enumerate(mids):
        t = cast(o, A, E, arena)
        bdir[k] = int(np.argmax(t))
        best[k] = t[bdir[k]]
    return best, bdir


def stats(streets, polys, arena):
    """The views' summary: median and p90 (m), the shares over 200 and 400 m, and how many midpoints were cast."""
    mids = street_midpoints(streets, polys, arena)
    best, _ = measure(polys, mids, arena)
    return dict(view_median=round(float(np.median(best)), 1), view_p90=round(float(np.percentile(best, 90)), 1),
                view_gt200=round(float((best > 200).mean()), 3), view_gt400=round(float((best > 400).mean()), 3),
                n_views=int(len(mids)))


def _cost(best):
    return float(np.maximum(best - FLOOR_M, 0).sum())


def optimise(seeds, build, constraints, arena, tries, rng, log=None):
    """Seeds after `tries` tries of the pass. build(seeds) -> (streets, polys, state); constraints(state) -> dict of
    sides, plazas, closed, narrow for that layout (called only for a move that already shortens the views)."""
    seeds = np.array(seeds, float)
    streets, polys, state = build(seeds)
    mids = street_midpoints(streets, polys, arena)
    best, bdir = measure(polys, mids, arena)

    def key(m):
        return tuple(np.round(m, 3))

    cache = {key(m): (b, d) for m, b, d in zip(mids, best, bdir)}
    c0 = constraints(state)
    J = _cost(best)
    if log:
        log(f"sight pass: start cost {J:.0f}, median {np.median(best):.0f} m, {c0}")
    x0, x1, z0, z1 = arena
    tried = kept = 0
    while tried < tries:
        tried += 1
        long_ = np.flatnonzero(best > LONG_M)
        if len(long_) == 0:
            break
        i = rng.choice(long_, p=(best[long_] - FLOOR_M) / (best[long_] - FLOOR_M).sum())
        o, u, t = mids[i], DIRS[bdir[i]], best[i]
        rel = seeds - o
        along = rel @ u
        perp = rel[:, 0] * u[1] - rel[:, 1] * u[0]
        cand = np.flatnonzero((along > 30) & (along < t - 10) & (np.abs(perp) < 45))
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(np.abs(perp[cand]))[:6]]
        k = int(rng.choice(cand))
        n = np.array([-u[1], u[0]]) * (1 if perp[k] < 0 else -1)          # towards the line of sight
        trial = seeds.copy()
        trial[k] = trial[k] + n * rng.uniform(3, 8) + rng.normal(0, 1.5, 2)
        trial[k] = np.clip(trial[k], [x0 + 1, z0 + 1], [x1 - 1, z1 - 1])
        try:
            st_t, pt, state_t = build(trial)
        except Exception:                      # a degenerate Voronoi cell: not a move
            continue
        mt = street_midpoints(st_t, pt, arena)
        near = np.linalg.norm(mt - trial[k], axis=1) < LOCAL_M
        bt, dt = np.zeros(len(mt)), np.zeros(len(mt), int)
        A, E = segments(pt)
        for q, m in enumerate(mt):
            hit = None if near[q] else cache.get(key(m))
            if hit is None:
                tt = cast(m, A, E, arena)
                dt[q] = int(np.argmax(tt))
                bt[q] = tt[dt[q]]
            else:
                bt[q], dt[q] = hit
        Jt = _cost(bt)
        if Jt >= J:
            continue
        ct = constraints(state_t)
        if ct["sides"] < c0["sides"] or ct["plazas"] > c0["plazas"] or ct["closed"] > c0["closed"] or \
                ct["narrow"] > c0["narrow"]:
            continue
        seeds, polys, mids, best, bdir, J = trial, pt, mt, bt, dt, Jt
        cache = {key(m): (b, d) for m, b, d in zip(mids, best, bdir)}
        kept += 1
        if kept % REMEASURE_EVERY == 0:
            best, bdir = measure(polys, mids, arena)
            cache = {key(m): (b, d) for m, b, d in zip(mids, best, bdir)}
            J = _cost(best)
            if log:
                log(f"sight pass: {tried} tried, {kept} kept, cost {J:.0f}, median {np.median(best):.0f} m, {ct}")
    if log:
        log(f"sight pass: done, {tried} tried, {kept} kept")
    return seeds
