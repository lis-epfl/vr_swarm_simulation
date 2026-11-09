"""The paths the scripted pilots fly through a city, and what each one steers round.

    street   the shortest route along the layout's street centrelines (Dijkstra over the Voronoi edge graph) -- what
             Tier B' and Tier C flew until 2026-10-13. Kept as the comparison: internal_3's pilots did not keep to
             the streets (the swarm's paths were 1.06-1.12x the straight line in three of four runs).
    pilot    the any-angle cheapest path through the space a pilot steers round: a 1 m grid, Dijkstra over its
             16-connected cells, then string-pulled to the corners it has to turn at. What is in the way depends on
             the interface:
               swarm   the formation flows round a building that stands alone and through any gap between avoidance
                       cylinders of at least w_pass (the gate probe, formation.gate_map), so the pilot holds the
                       stick through those and the replica / bench then tests whether it really does. Buildings
                       closer together than that are WALLS: their cylinders, and the gaps narrower than G_HARD_M
                       between them, are closed, and the pilot keeps the formation off them -- a metre flown within
                       the formation's own radius r (its settled hover diameter at the spread, halved) of a wall
                       costs 1 + WALL_COST (1 - d / r)^2, d the distance to the wall. So in a packed city the pilot
                       flies the middle of the corridors (mostly the streets) and in an open one the straight line.
                       Without it the paths grazed the walls: on the bench such a path through L3 lost 2.5 drones a
                       flight at spread 1.0 (the replica 2.3) where the street route lost 0.8 (0.7) and was no
                       slower.
               single  every building's avoidance cylinder plus SINGLE_CLEARANCE_M is closed: a pilot steering one
                       drone round each building.
The avoidance cylinder is OlfatiSaber's: the circumradius of the collider's world AABB about its centre
(swarm_replica.City.R), whose repulsion reaches STANDOFF_M beyond it.

Numpy + scipy only. Grids are cut to the corridor round start and goal (CORRIDOR_M), widened to the whole city when
the corridor holds no path.
"""
import sys

sys.dont_write_bytecode = True

import heapq  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import PACKAGE_DIR  # noqa: E402

GRID_M = 1.0
STANDOFF_M = 4.0             # d_obs * scaleFactor: the obstacle repulsion's reach beyond the cylinder
SINGLE_CLEARANCE_M = 1.0     # how far off an avoidance cylinder the single-drone pilot plans to keep
G_HARD_M = 2.0               # a gap between two cylinders narrower than this is closed to the formation
WALL_COST = 4.0              # per-metre surcharge for flying the formation's centre against a wall
CORRIDOR_M = 160.0           # grid margin round the start-goal box before widening to the whole city
LOS_STEP = 0.5               # cells between line-of-sight samples when pulling the string
W_PASS_FILE = PACKAGE_DIR / "data" / "w_pass.json"
W_PASS_FALLBACK_M = 8.0      # until the gate probe has run: two standoffs between cylinders


# ============================================================================= streets

def street_graph(L):
    """(nodes (k, 2), adjacency {i: [(j, length)]}) of the layout's street centrelines, or None without streets."""
    if not L.streets:
        return None
    key, nodes, adj = {}, [], {}

    def node(p):
        k = (round(p[0], 1), round(p[1], 1))
        if k not in key:
            key[k] = len(nodes)
            nodes.append(np.array(p, float))
        return key[k]

    for st in L.streets:
        pts = st["polyline"]
        for a, b in zip(pts[:-1], pts[1:]):
            i, j = node(a), node(b)
            w = float(np.linalg.norm(np.subtract(a, b)))
            adj.setdefault(i, []).append((j, w))
            adj.setdefault(j, []).append((i, w))
    return np.array(nodes), adj


def route(graph, start, goal):
    """Waypoints from start to goal along the street graph: the node nearest the start, the shortest path to the
    node nearest the goal, then the goal itself."""
    nodes, adj = graph
    s = int(np.argmin(np.linalg.norm(nodes - start, axis=1)))
    t = int(np.argmin(np.linalg.norm(nodes - goal, axis=1)))
    dist, prev = {s: 0.0}, {}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    path = [t]
    while path[-1] != s and path[-1] in prev:
        path.append(prev[path[-1]])
    return [nodes[i].copy() for i in reversed(path)] + [np.asarray(goal, float)]


# ============================================================================= obstacles

class Cylinders:
    """Buildings as OlfatiSaber sees them: centre (n, 2) and avoidance-cylinder radius (n,), from analyse.load_city
    tuples (tile, centre (3,), half-axis vectors (3, 3))."""

    def __init__(self, boxes):
        self.n = len(boxes)
        if self.n == 0:
            self.c, self.R = np.zeros((0, 2)), np.zeros(0)
            return
        c3 = np.array([b[1] for b in boxes], float)
        ax = np.array([b[2] for b in boxes], float)
        ext = np.abs(ax).sum(axis=1)                       # bounds.extents
        self.c = c3[:, [0, 2]]
        self.R = np.hypot(ext[:, 0], ext[:, 2])

    def gaps(self, reach):
        """[(i, j, gap)] for every pair whose cylinders come within `reach` of each other (gap < reach)."""
        from scipy.spatial import cKDTree
        if self.n < 2:
            return []
        tree = cKDTree(self.c)
        out = []
        for i, j in sorted(tree.query_pairs(2 * float(self.R.max()) + reach)):
            g = float(np.linalg.norm(self.c[i] - self.c[j]) - self.R[i] - self.R[j])
            if g < reach:
                out.append((i, j, g))
        return out

    def bounds(self, margin=0.0):
        if self.n == 0:
            return None
        return (float((self.c[:, 0] - self.R).min() - margin), float((self.c[:, 0] + self.R).max() + margin),
                float((self.c[:, 1] - self.R).min() - margin), float((self.c[:, 1] + self.R).max() + margin))


def walls(cyl, w_pass):
    """(member mask (n,), [(i, j, gap)]): the buildings with a neighbour closer than w_pass -- the ones the formation
    cannot simply flow round -- and the gaps narrower than w_pass between them."""
    pairs = cyl.gaps(w_pass)
    member = np.zeros(cyl.n, bool)
    for i, j, _ in pairs:
        member[i] = member[j] = True
    return member, pairs


# ============================================================================= the grid

class Grid:
    """Occupancy over a box: `occ` closed, `cost` the per-metre cost of an open cell (1, more against a wall). Cell
    (i, j) is centred on (x0 + (i + 0.5) h, z0 + (j + 0.5) h)."""

    def __init__(self, bounds, h=GRID_M):
        x0, x1, z0, z1 = bounds
        self.h = h
        self.x0, self.z0 = x0, z0
        self.nx, self.nz = max(int(math.ceil((x1 - x0) / h)), 1), max(int(math.ceil((z1 - z0) / h)), 1)
        self.occ = np.zeros((self.nx, self.nz), bool)
        self.cost = np.ones((self.nx, self.nz))

    def cell(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        return (np.floor((p[:, 0] - self.x0) / self.h).astype(int),
                np.floor((p[:, 1] - self.z0) / self.h).astype(int))

    def centre(self, i, j):
        return np.stack([self.x0 + (np.asarray(i) + 0.5) * self.h, self.z0 + (np.asarray(j) + 0.5) * self.h], -1)

    def inside(self, i, j):
        return (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.nz)

    def _window(self, lo, hi):
        i0, j0 = self.cell(lo)
        i1, j1 = self.cell(hi)
        i0, i1 = max(int(i0[0]), 0), min(int(i1[0]), self.nx - 1)
        j0, j1 = max(int(j0[0]), 0), min(int(j1[0]), self.nz - 1)
        if i0 > i1 or j0 > j1:
            return None
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        return (slice(i0, i1 + 1), slice(j0, j1 + 1)), self.centre(ii, jj)

    def mark_disc(self, c, r, layer="occ"):
        w = self._window(np.asarray(c) - r, np.asarray(c) + r)
        if w is not None:
            sl, q = w
            getattr(self, layer)[sl] |= np.hypot(q[..., 0] - c[0], q[..., 1] - c[1]) <= r

    def mark_capsule(self, a, b, r):
        """Every cell within r of the segment ab, closed."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        w = self._window(np.minimum(a, b) - r, np.maximum(a, b) + r)
        if w is None:
            return
        sl, q = w
        ab = b - a
        t = np.clip(((q - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(q - (a + t[..., None] * ab), axis=-1)
        self.occ[sl] |= d <= r

    def sample(self, pts):
        """(open, cost) per point: outside the grid counts as open at cost 1."""
        i, j = self.cell(pts)
        ok = self.inside(i, j)
        occ, cost = np.zeros(len(i), bool), np.ones(len(i))
        occ[ok], cost[ok] = self.occ[i[ok], j[ok]], self.cost[i[ok], j[ok]]
        return ~occ, cost

    def segment(self, a, b):
        """(clear of closed cells, the squeeze cost beyond its length) of the segment ab, sampled LOS_STEP cells
        apart."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        n = max(int(math.ceil(L / (LOS_STEP * self.h))), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        free, cost = self.sample(a + t[:, None] * (b - a))
        return bool(free.all()), float((cost[1:] - 1.0).sum()) * L / n


def occupancy(cyl, kind, bounds, w_pass=None, clearance=SINGLE_CLEARANCE_M, h=GRID_M, radius=0.0):
    """The grid a pilot plans on (see the module docstring). kind "single": every cylinder grown by `clearance`
    closed; "swarm": the walls at w_pass closed, and flying within `radius` (the formation's) of them surcharged;
    "all": every cylinder closed, as is (a formation that cannot flow round a lone building)."""
    g = Grid(bounds, h)
    if cyl.n == 0:
        return g
    x0, x1, z0, z1 = bounds
    pad = (w_pass or 0.0) + clearance + 1.0
    near = ((cyl.c[:, 0] + cyl.R + pad >= x0) & (cyl.c[:, 0] - cyl.R - pad <= x1) &
            (cyl.c[:, 1] + cyl.R + pad >= z0) & (cyl.c[:, 1] - cyl.R - pad <= z1))
    if kind in ("single", "all"):
        grow = clearance if kind == "single" else 0.0
        for k in np.flatnonzero(near):
            g.mark_disc(cyl.c[k], cyl.R[k] + grow)
    elif kind == "swarm":
        if w_pass is None:
            raise ValueError("a swarm occupancy needs w_pass")
        member, pairs = walls(cyl, w_pass)
        for k in np.flatnonzero(member & near):
            g.mark_disc(cyl.c[k], cyl.R[k])
        for i, j, gap in pairs:
            if (near[i] or near[j]) and gap < G_HARD_M:
                g.mark_capsule(cyl.c[i], cyl.c[j], min(cyl.R[i], cyl.R[j]))
        if radius and g.occ.any():
            from scipy import ndimage
            d = ndimage.distance_transform_edt(~g.occ) * h
            g.cost = 1.0 + WALL_COST * np.clip(1.0 - d / radius, 0.0, 1.0) ** 2
    else:
        raise ValueError(f"unknown occupancy {kind!r}")
    return g


# ============================================================================= planning

# 16-connected moves (di >= 0) and the cells each one passes through: an 8-connected grid measures octile distance, up to
# 8 % long against the straight line, enough to pick a squeeze over a slightly longer clear way round; with the knight
# moves the error is under 3 %.
MOVES = (((1, 0), ()), ((0, 1), ()), ((1, 1), ((1, 0), (0, 1))), ((1, -1), ((1, 0), (0, -1))),
         ((1, 2), ((0, 1), (1, 1))), ((2, 1), ((1, 0), (1, 1))), ((1, -2), ((0, -1), (1, -1))),
         ((2, -1), ((1, 0), (1, -1))))


def _edges(free, di, dj, via=()):
    """Source cells (i, j) of the move (i, j) -> (i + di, j + dj) between two open cells through open cells `via`."""
    nx, nz = free.shape
    i1 = nx - di
    j0, j1 = max(0, -dj), nz - max(0, dj)
    if i1 <= 0 or j1 <= j0:
        return np.zeros(0, int), np.zeros(0, int)
    ok = free[0:i1, j0:j1] & free[di:i1 + di, j0 + dj:j1 + dj]
    for vi, vj in via:
        ok &= free[vi:i1 + vi, j0 + vj:j1 + vj]
    ai, aj = np.nonzero(ok)
    return ai, aj + j0


def _grid_path(g, s, t):
    """Cells of the cheapest 8-connected path from cell s to cell t (a squeeze costs its factor per metre); to the
    reachable cell nearest t when t cannot be reached. [] when s is closed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    free = ~g.occ
    if not free[s]:
        return []
    ids = -np.ones(g.occ.shape, np.int64)
    fi, fj = np.nonzero(free)
    ids[fi, fj] = np.arange(len(fi))
    factor = g.cost
    rows, cols, w = [], [], []
    for (di, dj), via in MOVES:
        ai, aj = _edges(free, di, dj, via)
        bi, bj = ai + di, aj + dj
        step = math.hypot(di, dj) * g.h
        rows.append(ids[ai, aj])
        cols.append(ids[bi, bj])
        w.append(step * 0.5 * (factor[ai, aj] + factor[bi, bj]))
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    n = len(fi)
    A = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    src = int(ids[s])
    dist, pred = dijkstra(A, directed=False, indices=src, return_predecessors=True)
    if free[t] and np.isfinite(dist[ids[t]]):
        tgt = int(ids[t])
    else:
        reach = np.flatnonzero(np.isfinite(dist))
        q = g.centre(fi[reach], fj[reach])
        tc = g.centre(t[0], t[1])
        tgt = int(reach[np.argmin(np.hypot(q[:, 0] - tc[0], q[:, 1] - tc[1]))])
    path = [tgt]
    while path[-1] != src:
        p = int(pred[path[-1]])
        if p < 0:
            return []
        path.append(p)
    path.reverse()
    return [(int(fi[k]), int(fj[k])) for k in path]


def _pull(g, pts):
    """String-pulling: the corners of a path of points, keeping a straightened leg only where it is clear of closed
    cells and squeezes no harder than the path it replaces (its squeeze cost against the path's, cumulated)."""
    if len(pts) <= 2:
        return list(pts)
    seg_cost = [g.segment(a, b)[1] for a, b in zip(pts[:-1], pts[1:])]
    cum = np.r_[0.0, np.cumsum(seg_cost)]
    out, ia = [pts[0]], 0
    for k in range(2, len(pts)):
        clear, extra = g.segment(pts[ia], pts[k])
        if not clear or extra > cum[k] - cum[ia] + g.h:
            ia = k - 1
            out.append(pts[ia])
    out.append(pts[-1])
    return out


def plan(cyl, kind, start, goal, w_pass=None, clearance=SINGLE_CLEARANCE_M, corridor=CORRIDOR_M, h=GRID_M,
         city_bounds=None, radius=0.0):
    """Waypoints [(2,)] from start to goal for a pilot of `kind` (occupancy()), the goal itself last. A start in a
    closed cell first flies to the nearest open one; an unreachable goal (inside a closed group: a cramped goal patch)
    is approached to the reachable point nearest it. [goal] when the straight line is clear."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    lo, hi = np.minimum(start, goal) - corridor, np.maximum(start, goal) + corridor
    tries = [(lo[0], hi[0], lo[1], hi[1])]
    cb = city_bounds if city_bounds is not None else cyl.bounds()
    if cb is not None:
        tries.append((min(lo[0], cb[0] - corridor), max(hi[0], cb[1] + corridor),
                      min(lo[1], cb[2] - corridor), max(hi[1], cb[3] + corridor)))
    best = None
    for bounds in tries:
        g = occupancy(cyl, kind, bounds, w_pass, clearance, h, radius)
        clear, extra = g.segment(start, goal)
        if clear and extra <= 0.0:
            return [goal.copy()]
        s = _nearest_open(g, start)
        t = _nearest_open(g, goal)
        if s is None:
            continue
        tc = t if t is not None else tuple(int(np.clip(v[0], 0, m - 1)) for v, m in zip(g.cell(goal), (g.nx, g.nz)))
        cells = _grid_path(g, s, tc)
        if not cells:
            continue
        pts = [g.centre(i, j) for i, j in cells]
        end_gap = float(np.linalg.norm(pts[-1] - goal))
        start_open = bool(g.sample(start[None])[0][0])
        head = [] if start_open else [pts[0]]
        corners = _pull(g, ([start] if start_open else [pts[0]]) + pts[1:])
        wps = head + [np.asarray(p, float) for p in corners[1:]]
        if best is None or end_gap < best[0] - 1e-6:
            best = (end_gap, wps)
        if end_gap <= 2 * h:
            break
    if best is None:
        return [goal.copy()]
    wps = best[1]
    if not wps or np.linalg.norm(wps[-1] - goal) > 1e-6:
        wps = wps + [goal.copy()]
    return wps


def _nearest_open(g, p):
    """The open cell nearest p (p's own when open), as (i, j), or None on a closed grid."""
    i, j = g.cell(p)
    i, j = int(np.clip(i[0], 0, g.nx - 1)), int(np.clip(j[0], 0, g.nz - 1))
    if not g.occ[i, j]:
        return i, j
    if g.occ.all():
        return None
    from scipy import ndimage
    _, (ii, jj) = ndimage.distance_transform_edt(g.occ, return_indices=True)
    return int(ii[i, j]), int(jj[i, j])


def path_length(start, wps):
    pts = [np.asarray(start, float)] + [np.asarray(w, float) for w in wps]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))


# ============================================================================= w_pass and the pilots

def _probe(path=W_PASS_FILE):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def w_pass(d_ref, path=W_PASS_FILE):
    """The narrowest cylinder gap the formation flows through at spread d_ref (formation.gate_map, in
    data/w_pass.json), interpolated in d_ref; W_PASS_FALLBACK_M before the probe has run, and the widest gap probed
    where the formation passed none."""
    d = _probe(path)
    if d is None:
        return W_PASS_FALLBACK_M
    tab = sorted((float(k), float(v)) for k, v in d["w_pass_m"].items() if v is not None)
    if not tab:
        return W_PASS_FALLBACK_M
    xs, ys = zip(*tab)
    return float(np.interp(float(d_ref), xs, ys))


def ram_ok(d_ref, path=W_PASS_FILE):
    """Whether the probe found the formation flows round a lone building at the probed spread nearest d_ref."""
    d = _probe(path)
    tab = {float(k): bool(v) for k, v in (d or {}).get("ram_ok", {}).items()}
    if not tab:
        return True
    return tab[min(tab, key=lambda k: abs(k - float(d_ref)))]


PILOTS = ("swarm", "single")


def formation_radius(d_ref):
    """Half the formation's settled hover diameter at spread d_ref (formation.py's data/formation_diameters.json)."""
    from citysim import citygen
    diam, _ = citygen.load_diameters()
    xs = sorted(diam)
    return 0.5 * float(np.interp(float(d_ref), xs, [diam[x] for x in xs]))


def interface_plan(cyl, interface, start, goal, d_ref=1.0, city_bounds=None):
    """plan() for one of the study's two pilots. "swarm": the walls at w_pass(d_ref), kept a formation's radius off --
    or every cylinder, as is, at a spread where the probe found the formation cannot flow round a lone building;
    "single": every cylinder plus SINGLE_CLEARANCE_M."""
    if interface == "swarm":
        if not ram_ok(d_ref):
            return plan(cyl, "all", start, goal, city_bounds=city_bounds)
        return plan(cyl, "swarm", start, goal, w_pass=w_pass(d_ref), city_bounds=city_bounds,
                    radius=formation_radius(d_ref))
    if interface == "single":
        return plan(cyl, "single", start, goal, city_bounds=city_bounds)
    raise ValueError(f"unknown pilot {interface!r} (one of {PILOTS})")
