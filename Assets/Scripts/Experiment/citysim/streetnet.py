"""The street network of a block city (citysim.blockcity) as the split study's pilots and measures use it: junctions,
streets between them, blocks with their corners and the streets that radiate from them, the block ahead of an
arriving formation, street samples, routes, and the one cluster metric.

    net = streetnet.Network(L, blocks, fp)          # L, blocks = blockcity.load(path); fp = blockcity.footprints(L)

Junctions. Voronoi vertices closer than MERGE_FRAC x the street width are one junction (at p 40 a tenth of the street
segments are under 9 m), placed at their mean; a street whose two ends merge is absorbed into its junction.
A junction, street or block is *interior* when it touches no street on the arena's boundary: walks, targets and
manoeuvres keep to interior ones.

Blocks. A block's corners are the junctions its streets meet at; its *radiating* streets leave its corners and do not
run along it (at a three-way corner, the street between its two neighbours). Arriving at a corner J along street e,
the block ahead is the one at J that e does not run along; at a merged junction with more than one, the one whose
centre lies nearest the direction of arrival.

The cluster metric. Two drones are linked when the free-space path between them is at most LINK_M: the straight
segment when it crosses no building, otherwise the shortest path round the buildings in the way (a visibility graph
over their corners grown by LINK_MARGIN_M). A straight-line test would call a formation split whenever it turned a
corner. A *split* is two or more clusters of at least two drones; a lone drone is a straggler.
"""
import sys

sys.dont_write_bytecode = True

import heapq  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import geometry as geo  # noqa: E402

MERGE_FRAC = 1.0          # vertices closer than this x the street width are one junction
SAMPLE_M = 2.0            # street centreline samples
LINK_M = 30.0             # the cluster metric's free-space link
LINK_MARGIN_M = 0.5       # corners grown by this for the detour


class Network:
    def __init__(self, L, blocks, fp, merge_frac=MERGE_FRAC, sample_m=SAMPLE_M):
        self.L, self.fp = L, fp
        streets = L.streets
        s_width = float(streets[0]["width"]) if streets else 12.0
        self.width = s_width
        # ---- junctions: union-find over the street end points
        ends = np.array([pt for st in streets for pt in (st["polyline"][0], st["polyline"][-1])], float)
        parent = list(range(len(ends)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        if len(ends):
            from scipy.spatial import cKDTree
            for i, j in cKDTree(ends).query_pairs(merge_frac * s_width):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        roots = sorted({find(i) for i in range(len(ends))})
        jid = {r: k for k, r in enumerate(roots)}
        end_j = np.array([jid[find(i)] for i in range(len(ends))], int).reshape(-1, 2)
        self.jpos = np.zeros((len(roots), 2))
        cnt = np.zeros(len(roots))
        for i, j in enumerate(end_j.ravel()):
            self.jpos[j] += ends[i]
            cnt[j] += 1
        self.jpos /= np.maximum(cnt, 1)[:, None]
        # ---- streets between distinct junctions
        self.streets = []                       # dict(id, u, v, length, blocks, edge, layout_id)
        self.j_streets = {j: [] for j in range(len(roots))}
        for k, st in enumerate(streets):
            u, v = end_j[k]
            if u == v:
                continue
            sid = len(self.streets)
            self.streets.append(dict(id=sid, u=int(u), v=int(v), length=float(np.linalg.norm(self.jpos[v] -
                                                                                              self.jpos[u])),
                                     blocks=[b for b in st["blocks"] if b >= 0], edge=bool(st["edge"]),
                                     layout_id=int(st["id"])))
            self.j_streets[int(u)].append(sid)
            self.j_streets[int(v)].append(sid)
        self.adj = {j: [] for j in range(len(roots))}
        for st in self.streets:
            self.adj[st["u"]].append((st["v"], st["length"], st["id"]))
            self.adj[st["v"]].append((st["u"], st["length"], st["id"]))
        edge_j = {j for st in self.streets if st["edge"] for j in (st["u"], st["v"])}
        self.j_interior = np.array([j not in edge_j for j in range(len(roots))], bool)
        for st in self.streets:
            st["interior"] = bool(not st["edge"] and self.j_interior[st["u"]] and self.j_interior[st["v"]])
        # ---- blocks
        self.blocks = {}
        for b in blocks:
            kerb = np.asarray(b["kerb_xz"], float).reshape(-1, 2)
            if len(kerb) < 3:
                continue
            own = [st["id"] for st in self.streets if b["id"] in st["blocks"]]
            corners = sorted({j for sid in own for j in (self.streets[sid]["u"], self.streets[sid]["v"])})
            radiating = sorted({sid for j in corners for sid in self.j_streets[j]
                                if b["id"] not in self.streets[sid]["blocks"]})
            self.blocks[b["id"]] = dict(id=b["id"], kerb=kerb, centre=geo.centroid(kerb), building=b["building"],
                                        streets=own, corners=corners, radiating=radiating,
                                        interior=bool(b["interior"] and all(self.j_interior[j] for j in corners) and
                                                      all(self.streets[s]["interior"] for s in radiating)))
        # ---- samples
        self.sample_m = sample_m
        pts, sid, along = [], [], []
        for st in self.streets:
            a, b = self.jpos[st["u"]], self.jpos[st["v"]]
            n = max(int(math.ceil(st["length"] / sample_m)), 1)
            t = (np.arange(n) + 0.5) / n
            pts.append(a + t[:, None] * (b - a))
            sid.append(np.full(n, st["id"]))
            along.append(t * st["length"])
        self.sample_xz = np.vstack(pts) if pts else np.zeros((0, 2))
        self.sample_street = np.concatenate(sid) if sid else np.zeros(0, int)
        self.sample_along = np.concatenate(along) if along else np.zeros(0)       # from the street's u end

    # ------------------------------------------------------------------ lookups

    def other(self, sid, j):
        st = self.streets[sid]
        return st["v"] if st["u"] == j else st["u"]

    def nearest_junction(self, xz):
        return int(np.argmin(np.hypot(*(self.jpos - np.asarray(xz, float)).T)))

    def block_ahead(self, j, arrival_sid):
        """The block a formation arriving at junction j along street arrival_sid faces, or None."""
        st = self.streets[arrival_sid]
        cand = [b for b in self.blocks.values() if j in b["corners"] and b["id"] not in st["blocks"]]
        if not cand:
            return None
        if len(cand) == 1:
            return cand[0]["id"]
        frm = self.jpos[self.other(arrival_sid, j)]
        d = self.jpos[j] - frm
        d /= max(float(np.linalg.norm(d)), 1e-9)
        score = [float(np.dot((b["centre"] - self.jpos[j]) / max(float(np.linalg.norm(b["centre"] - self.jpos[j])),
                                                                     1e-9), d)) for b in cand]
        return cand[int(np.argmax(score))]["id"]

    def exits(self, block, entrance_sid):
        """[(street id, its corner junction)] the formation may leave block `block` by, the entrance street excluded."""
        B = self.blocks[block]
        out = []
        for sid in B["radiating"]:
            if sid == entrance_sid:
                continue
            st = self.streets[sid]
            corner = st["u"] if st["u"] in B["corners"] else st["v"]
            out.append((sid, corner))
        return out

    def opposite_exit(self, block, entrance_sid, entrance_j):
        """The exit whose corner lies furthest round the block from the entrance corner (M3's), as (sid, corner)."""
        ex = self.exits(block, entrance_sid)
        if not ex:
            return None
        c = self.blocks[block]["centre"]
        a0 = math.atan2(*(self.jpos[entrance_j] - c)[::-1])
        return max(ex, key=lambda e: abs(_wrap(math.atan2(*(self.jpos[e[1]] - c)[::-1]) - a0)))

    def street_samples(self, sid, from_j, reach_m=80.0):
        """Indices into the samples of street sid, ordered from its end at junction from_j, within reach_m of it."""
        st = self.streets[sid]
        idx = np.flatnonzero(self.sample_street == sid)
        along = self.sample_along[idx] if st["u"] == from_j else st["length"] - self.sample_along[idx]
        keep = along <= reach_m
        order = np.argsort(along[keep])
        return idx[keep][order]

    # ------------------------------------------------------------------ routes

    def route(self, j0, j1):
        """(junction list, street list) of the shortest street route j0 -> j1 (Dijkstra)."""
        dist, prev = {j0: 0.0}, {}
        pq = [(0.0, j0)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == j1:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, w, sid in self.adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, (u, sid)
                    heapq.heappush(pq, (nd, v))
        if j1 not in dist:
            return None, None
        js, ss = [j1], []
        while js[-1] != j0:
            u, sid = prev[js[-1]]
            ss.append(sid)
            js.append(u)
        return js[::-1], ss[::-1]

    def route_length(self, js):
        return float(sum(np.linalg.norm(self.jpos[b] - self.jpos[a]) for a, b in zip(js[:-1], js[1:])))

    def interior_blocks(self):
        return [b for b in self.blocks.values() if b["interior"] and b["building"] >= 0]

    # ------------------------------------------------------------------ the cluster metric

    def clusters(self, P, link_m=LINK_M):
        """Cluster labels (k,) of positions P (k, 2) under the free-space link."""
        P = np.asarray(P, float)
        k = len(P)
        lab = np.arange(k)
        if k < 2:
            return lab
        ii, jj = np.triu_indices(k, 1)
        d = np.linalg.norm(P[ii] - P[jj], axis=1)
        close = d <= link_m
        ii, jj, d = ii[close], jj[close], d[close]
        if len(ii):
            blocked = self.fp.blocked(P[ii], P[jj]) if self.fp.n else np.zeros(len(ii), bool)
            for a, b, dd, bl in zip(ii, jj, d, blocked):
                if bl and self.detour(P[a], P[b], link_m) > link_m:
                    continue
                ra, rb = _root(lab, a), _root(lab, b)
                if ra != rb:
                    lab[ra] = rb
        return np.array([_root(lab, i) for i in range(k)])

    def detour(self, a, b, cap):
        """Shortest free-space path a -> b round the buildings within reach (a visibility graph over their corners,
        grown by LINK_MARGIN_M), or inf beyond cap."""
        mid, r = (a + b) / 2, cap / 2 + 1.0
        near = self.fp.near(mid, r)
        if not len(near):
            return float(np.linalg.norm(b - a))
        corners = []
        for i in near:
            poly = _poly_of(self.fp, i)
            c = poly.mean(0)
            v = poly - c
            corners.append(poly + LINK_MARGIN_M * v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9))
        nodes = np.vstack([a[None], b[None]] + corners)
        nodes = nodes[(np.linalg.norm(nodes - a, axis=1) + np.linalg.norm(nodes - b, axis=1)) <= cap + 1e-6]
        n = len(nodes)
        ii, jj = np.triu_indices(n, 1)
        clear = ~self.fp.blocked(nodes[ii], nodes[jj])
        w = np.linalg.norm(nodes[ii] - nodes[jj], axis=1)
        adj = [[] for _ in range(n)]
        for i, j, ok, ww in zip(ii, jj, clear, w):
            if ok:
                adj[i].append((j, ww))
                adj[j].append((i, ww))
        dist = np.full(n, np.inf)
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            dd, u = heapq.heappop(pq)
            if u == 1 or dd > cap:
                break
            if dd > dist[u]:
                continue
            for v, ww in adj[u]:
                if dd + ww < dist[v]:
                    dist[v] = dd + ww
                    heapq.heappush(pq, (dist[v], v))
        return float(dist[1])


def split_state(labels, alive=None):
    """(n clusters of at least 2, n stragglers, split flag) of cluster labels."""
    lab = np.asarray(labels)
    if alive is not None:
        lab = lab[np.asarray(alive, bool)]
    if not len(lab):
        return 0, 0, False
    _, counts = np.unique(lab, return_counts=True)
    big = int((counts >= 2).sum())
    return big, int((counts == 1).sum()), big >= 2


def _root(lab, i):
    while lab[i] != i:
        lab[i] = lab[lab[i]]
        i = lab[i]
    return i


def _poly_of(fp, i):
    if hasattr(fp, "poly"):
        return np.asarray(fp.poly[i], float)
    c, u, h = fp.c[i], fp.u[i], fp.h[i]
    return np.array([c + sx * u[0] * h[0] + sz * u[1] * h[1] for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
