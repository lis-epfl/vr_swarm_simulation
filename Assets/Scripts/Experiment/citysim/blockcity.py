"""S4 Split and Surround (the split study): cities of one large building per Voronoi block.

    python -B -m citysim.blockcity one --variant B --p 50 --s 12 --seed 0     # one layout -> layouts/sweep~/split/
    python -B -m citysim.blockcity one --variant B --p 40 --s 18 --seed 1 --scene   # -> layouts/BlockShaped__<id>.json
    python -B -m citysim.blockcity feasibility [--seeds 3] [--jobs N]         # every design, both variants: the table

The streets are the generator's own (citygen blocks mode, drawn the same way: RSA block seeds 0.8 p apart, Voronoi
cells, kerbs s/2 inside each street), so the two variants of one seed share every street, and both share them with
the goal-free citygen `bl_` city of the same p, s and seed. No goals: every block holds one building or none.

Two options change the city away from S4's, and only when asked (the defaults draw S4's layouts byte for byte; the id
names any option that is not the default):
  --inset    B's footpath width, kerb to facade (INSET_M, CityDressing's footpath).
  --seeds    where the block seeds go. `rsa` is S4's. `hex` puts them on a triangular lattice, each moved up to
             --jitter x the lattice spacing; --relax N then moves every seed to its cell's centroid N times (Lloyd).
             A Voronoi block always has about six sides, but a side shorter than its two junctions' setbacks is not
             drawn as a street (StreetSurface merges its junctions; streetnet does the same at one street width): at
             p 40 s 18 S4's seeds keep 4.3 of the six as streets, with 8.4 streets leaving the corners, and hex at
             jitter 0.15 keeps 5.6, with 6.4 leaving. Regular blocks have more sides, merged junctions more exits.
`one` prints both counts (street_stats) beside the layout's summary.

  R  "cylinder-fit": one catalogue design (R_DESIGN, the near-square BnP_Large_Building_B) per block at yaw 0 or 90,
     uniformly scaled to the largest size whose avoidance cylinder -- OlfatiSaber's, the circumradius of the collider's
     axis-aligned bounds -- fits inside the block's inscribed circle less CYL_MARGIN_M, centred on the block's
     Chebyshev centre. Today's force law, the replica, the bench and ProceduralCityBuilder fly it unchanged, and every
     street keeps a cylinder gap of at least its width. 28-41 % of the block.
  B  "block-shaped": the kerb polygon inset by INSET_M (CityDressing's 3.5 m footpath), one convex prism per block.
     Needs an obstacle model that sees its shape (swarm_replica's ClosestPoint): any single cylinder round it closes
     many 12 m streets at p >= 50. 30-80 % of the block.
A block whose building would be narrower than MIN_BUILDING_M is left empty (a plaza) and counted.

The layout is a citygen layout (boxes, tiles, streets, layout record) plus a top-level `blocks` list: per Voronoi
cell its kerb polygon (flat kerb_xz, counter-clockwise), whether it is interior (no street on the arena's boundary)
and the index of its building in `boxes` (-1 for a plaza). A variant B building is
{name: Block_B<k>, type: ConvexPrism, tile, poly_xz (flat, CCW), c, height, role, goal, patrol}: it has no `ax`, so
the readers of box axes (layout.Layout.boxes_as_tuples, analyse.load_city, bench.Footprints) cannot take it for a box;
read every S4 footprint through footprints() and replica_city() here.
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
import os  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import LAYOUT_DIR, catalogue, citygen  # noqa: E402
from citysim import geometry as geo  # noqa: E402
from citysim import layout as lay  # noqa: E402
from citysim.polyfoot import ConvexFootprints  # noqa: E402

GENERATOR = "blockcity"
GENERATOR_VERSION = "s4-1"
ARENA = citygen.ARENA
SPAWN = citygen.SPAWN
SPAWN_YAW = citygen.SPAWN_YAW
R_DESIGN = "BnP_Large_Building_B"     # 15.22 x 17.66 m at size 1: the most nearly square large design
CYL_MARGIN_M = 1.0                    # R: the cylinder stays this far inside the block's inscribed circle
INSET_M = 3.5                         # B: the facade stands at the back of CityDressing's 3.5 m footpath
MIN_BUILDING_M = 8.0                  # narrower than this (R: the short side; B: the inscribed width) -> a plaza
STANDOFF_M = 3.0                      # where the formation's ring wraps a building: the d* prediction
OUT_SET = "split"                     # layouts/sweep~/split/
SCENE_PREFIX = "BlockShaped__"        # layouts/BlockShaped__<id>.json: BlockShapedCityWorld's layouts
P_LEVELS = (40, 50, 60, 75)
S_LEVELS = (12, 18, 24)
VARIANTS = ("R", "B")
SEEDINGS = ("rsa", "hex")
HEX_JITTER = 0.15                     # --seeds hex: the default move, a fraction of the lattice spacing


@dataclass
class Params:
    variant: str = "B"
    p: float = 50.0
    s: float = 12.0
    seed: int = 0
    inset: float = INSET_M            # B: kerb to facade
    seeds: str = "rsa"                # block seeds: S4's RSA, or a jittered triangular lattice
    jitter: float = HEX_JITTER        # hex only
    relax: int = 0                    # Lloyd iterations after either

    def ident(self):
        out = f"bb{self.variant}_p{self.p:g}_s{self.s:g}"
        if self.variant == "B" and self.inset != INSET_M:
            out += f"_in{self.inset:g}"
        if self.seeds == "hex":
            out += f"_hex{self.jitter:g}"
        if self.relax:
            out += f"_lloyd{self.relax}"
        return out + f"_seed{self.seed}"

    @property
    def default_streets(self):
        return self.seeds == "rsa" and not self.relax


# ============================================================================= geometry

def chebyshev(poly):
    """(centre (2,), radius) of the largest circle inside a convex polygon: a small LP over its half-planes."""
    from scipy.optimize import linprog
    nrm, c = geo.edges_halfplanes(geo.ccw(poly))
    a = np.hstack([nrm, np.ones((len(nrm), 1))])          # unit normals: n . x + r <= c
    r = linprog([0.0, 0.0, -1.0], A_ub=a, b_ub=c, bounds=[(None, None), (None, None), (0.0, None)],
                method="highs")
    if not r.success:
        return geo.centroid(poly), 0.0
    return np.array(r.x[:2]), float(r.x[2])


def principal_x(poly):
    """True when the polygon's long axis lies nearer x than z (second moments of its vertices about the centroid)."""
    p = np.asarray(poly, float) - geo.centroid(poly)
    cov = p.T @ p
    w, v = np.linalg.eigh(cov)
    major = v[:, int(np.argmax(w))]
    return abs(major[0]) >= abs(major[1])


def hex_seeds(rng, p, arena, jitter):
    """One seed per p x p on a triangular lattice (rows along x, spacing a = p sqrt(2 / sqrt 3)) at a random offset,
    each moved uniformly within jitter x a, and kept half a metre inside the arena (voronoi_cells mirrors the seeds
    across its sides, which needs them inside)."""
    x0, x1, z0, z1 = arena
    a = p * math.sqrt(2.0 / math.sqrt(3.0))
    h = a * math.sqrt(3.0) / 2.0
    ox, oz = rng.uniform(0.0, a), rng.uniform(0.0, h)
    cols, rows = int((x1 - x0) / a) + 3, int((z1 - z0) / h) + 3
    pts = np.array([(x0 + ox + (i - 1 + 0.5 * (j % 2)) * a, z0 + oz + (j - 1) * h)
                    for j in range(rows) for i in range(cols)], float)
    if jitter > 0:
        r = jitter * a * np.sqrt(rng.uniform(0.0, 1.0, len(pts)))
        t = rng.uniform(0.0, 2.0 * math.pi, len(pts))
        pts = pts + np.c_[r * np.cos(t), r * np.sin(t)]
    keep = (pts[:, 0] > x0 + 0.5) & (pts[:, 0] < x1 - 0.5) & (pts[:, 1] > z0 + 0.5) & (pts[:, 1] < z1 - 0.5)
    return pts[keep]


def lloyd(seeds, arena, n):
    """n Lloyd iterations: every seed moved to the centroid of its (arena-clipped) cell."""
    for _ in range(n):
        cells, _ = geo.voronoi_cells(seeds, arena)
        seeds = np.array([geo.centroid(c) for c in cells])
    return seeds


def streets_and_blocks(p, s, seed, arena=ARENA, seeds="rsa", jitter=HEX_JITTER, relax=0):
    """(cells, kerbs, streets) drawn as citygen.generate draws a goal-free blocks city: the same random stream, so
    the same streets as its `bl_` layout of this p, s and seed. Other seedings (see the module notes) draw their own."""
    rng = np.random.default_rng(seed)
    x0, x1, z0, z1 = arena
    if seeds == "rsa":
        n_seeds = max(int(round((x1 - x0) * (z1 - z0) / p ** 2)), 4)
        pts = geo.rsa_points(rng, arena, n_seeds, citygen.SEED_SPACING_RATIO * p)
    elif seeds == "hex":
        pts = hex_seeds(rng, p, arena, jitter)
    else:
        raise ValueError(f"seeds {seeds!r}: one of {SEEDINGS}")
    if relax:
        pts = lloyd(pts, arena, relax)
    cells, edges = geo.voronoi_cells(pts, arena)
    kerbs = [geo.offset_convex(c, s / 2) for c in cells]
    streets = []
    for k, (i, j, a, b) in enumerate(edges):
        ab = [round(float(a[0]), 3), round(float(a[1]), 3), round(float(b[0]), 3), round(float(b[1]), 3)]
        streets.append(dict(id=k, polyline=[ab[:2], ab[2:]], xz=ab, width=s, edge=bool(j < 0),
                            blocks=[int(i), int(j)]))
    return cells, kerbs, streets


def flat(poly, nd=3):
    return [round(float(v), nd) for v in np.asarray(poly, float).ravel()]


def unflat(xz):
    return np.asarray(xz, float).reshape(-1, 2)


# ============================================================================= buildings

def r_building(kerb, k, design):
    """Variant R's box for block k, or None (a plaza)."""
    centre, rho = chebyshev(kerb)
    half_diag = 0.5 * math.hypot(design.w, design.d)
    size = (rho - CYL_MARGIN_M) / half_diag
    if size * min(design.w, design.d) < MIN_BUILDING_M:
        return None
    yaw = 90 if principal_x(kerb) else 0          # yaw 90 lays the design's long side (local Y, d) along x
    return lay.make_box(f"{design.design}_{k:04d}", f"B{k:03d}", centre[0], centre[1], design, size, yaw)


def b_building(kerb, k, inset=INSET_M):
    """Variant B's prism for block k, or None (a plaza)."""
    poly = geo.offset_convex(kerb, inset)
    if len(poly) < 3 or 2.0 * chebyshev(poly)[1] < MIN_BUILDING_M:
        return None
    poly = geo.ccw(poly)
    c = geo.centroid(poly)
    return dict(name=f"Block_B{k:03d}", type="ConvexPrism", tile=f"B{k:03d}", poly_xz=flat(poly),
                c=[round(float(c[0]), 4), round(lay.BUILDING_BASE_M + lay.BUILDING_HEIGHT_M / 2, 4),
                   round(float(c[1]), 4)],
                height=lay.BUILDING_HEIGHT_M, role="city", goal=-1, patrol=False)


def is_prism(b):
    return b.get("type") == "ConvexPrism"


def building_poly(b):
    """A layout building's footprint polygon (CCW): the prism's, or the box's rectangle."""
    if is_prism(b):
        return unflat(b["poly_xz"])
    cx, cz, hx, hz = lay.box_rect(b)
    return np.array([[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]])


def cylinder_radius(b):
    """OlfatiSaber's avoidance cylinder: the circumradius of the footprint's axis-aligned bounds."""
    p = building_poly(b)
    return 0.5 * float(np.hypot(*(p.max(0) - p.min(0))))


# ============================================================================= generation

def generate(P):
    """The layout of one variant / design / seed: (lay.Layout, blocks list)."""
    if P.variant not in VARIANTS:
        raise ValueError(f"variant {P.variant!r}: one of {VARIANTS}")
    if P.seeds not in SEEDINGS:
        raise ValueError(f"seeds {P.seeds!r}: one of {SEEDINGS}")
    design = catalogue.get(R_DESIGN)
    cells, kerbs, streets = streets_and_blocks(P.p, P.s, P.seed, seeds=P.seeds, jitter=P.jitter, relax=P.relax)
    edge_blocks = {i for st in streets if st["edge"] for i in st["blocks"] if i >= 0}
    boxes, blocks = [], []
    for k, (cell, kerb) in enumerate(zip(cells, kerbs)):
        rec = dict(id=k, kerb_xz=flat(geo.ccw(kerb)) if len(kerb) >= 3 else [], interior=k not in edge_blocks,
                   building=-1)
        if len(kerb) >= 3:
            b = r_building(kerb, k, design) if P.variant == "R" else b_building(kerb, k, P.inset)
            if b is not None:
                rec["building"] = len(boxes)
                boxes.append(b)
        blocks.append(rec)
    L = lay.Layout(boxes=boxes, streets=streets)
    L.layout = record(P, L, blocks, cells, kerbs, design)
    return L, blocks


def record(P, L, blocks, cells, kerbs, design):
    x0, x1, z0, z1 = ARENA
    A = (x1 - x0) * (z1 - z0)
    polys = [building_poly(b) for b in L.boxes]
    areas = np.array([abs(geo.area(p)) for p in polys]) if polys else np.zeros(0)
    perims = np.array([geo.perimeter(p) for p in polys]) if polys else np.zeros(0)
    kerb_area = sum(abs(geo.area(k)) for k in kerbs if len(k) >= 3)
    built = float(areas.sum())
    alpha = built / A
    rec = dict(
        generator=GENERATOR, generator_version=GENERATOR_VERSION, id=P.ident(), seed=int(P.seed),
        mode="block_buildings", variant=P.variant, p=float(P.p), s=float(P.s),
        design=R_DESIGN if P.variant == "R" else "block", cyl_margin_m=CYL_MARGIN_M if P.variant == "R" else None,
        inset_m=P.inset if P.variant == "B" else None, min_building_m=MIN_BUILDING_M,
        n_blocks=int(sum(1 for k in kerbs if len(k) >= 3)), n_interior_blocks=int(sum(b["interior"] for b in blocks)),
        n_buildings=len(L.boxes), n_plazas=int(sum(1 for b, k in zip(blocks, kerbs) if len(k) >= 3 and
                                                   b["building"] < 0)),
        alpha_achieved=round(alpha, 4), block_share=round(built / kerb_area, 4) if kerb_area else None,
        building_area_mean=round(float(areas.mean()), 1) if len(areas) else None,
        ell_mean=round(math.pi * float(areas.mean()) / (alpha * float(perims.mean())), 1) if len(areas) and alpha
        else None,
        arena=list(ARENA), extent=extent(polys), spawn=list(SPAWN), spawn_yaw=SPAWN_YAW, goals=[],
        params=dict(variant=P.variant, p=float(P.p), s=float(P.s)))
    if not P.default_streets:                     # S4's layouts carry no such keys: written only when asked for
        rec["params"].update(seeds=P.seeds, relax=int(P.relax))
        if P.seeds == "hex":
            rec["params"]["jitter"] = float(P.jitter)
    rec.update(gap_stats(L, blocks))
    rec.update(ring_stats(L, blocks))
    rec["max_within_50m"] = max_within(L)
    return rec


def extent(polys):
    if not polys:
        return None
    a = np.vstack(polys)
    return [round(float(a[:, 0].min()), 2), round(float(a[:, 0].max()), 2), round(float(a[:, 1].min()), 2),
            round(float(a[:, 1].max()), 2)]


def gap_stats(L, blocks):
    """Across every interior street with a building on each side: the gap between the two avoidance cylinders
    (OlfatiSaber's, about each footprint's bounds centre) and the true facade-to-facade gap."""
    by_block = {b["id"]: b["building"] for b in blocks}
    cyl, true = [], []
    for st in L.streets:
        if st["edge"]:
            continue
        i, j = st["blocks"]
        bi, bj = by_block.get(i, -1), by_block.get(j, -1)
        if bi < 0 or bj < 0:
            continue
        pi, pj = building_poly(L.boxes[bi]), building_poly(L.boxes[bj])
        ci, cj = (pi.max(0) + pi.min(0)) / 2, (pj.max(0) + pj.min(0)) / 2
        cyl.append(float(np.linalg.norm(ci - cj)) - cylinder_radius(L.boxes[bi]) - cylinder_radius(L.boxes[bj]))
        true.append(poly_gap(pi, pj))
    cyl, true = np.array(cyl), np.array(true)
    if not len(cyl):
        return dict(n_street_pairs=0)
    return dict(n_street_pairs=len(cyl),
                cyl_gap_median=round(float(np.median(cyl)), 2), cyl_gap_p10=round(float(np.percentile(cyl, 10)), 2),
                cyl_gap_frac_ge6=round(float((cyl >= 6.0).mean()), 3),
                cyl_gap_frac_ge12=round(float((cyl >= 12.0).mean()), 3),
                facade_gap_min=round(float(true.min()), 2), facade_gap_median=round(float(np.median(true)), 2))


def street_stats(L, blocks):
    """What a formation meets round a building, over the interior blocks that have one (streetnet's network: vertices
    within one street width are one junction, as StreetSurface draws them): its sides that are drawn as streets, its
    corners, and the streets leaving its corners. Also, across the interior streets with a building on each side, the
    share Unity's avoidance law closes (OlfatiSaber's cylinder is the circumradius of each footprint's axis-aligned
    bounds, so the two cylinders overlap) or narrows below 6 m. Not written to the layout, so S4's files stay as they
    were."""
    from citysim import streetnet
    net = streetnet.Network(L, blocks, footprints(L))
    rows = [(len(B["streets"]), len(B["corners"]), len(B["radiating"])) for B in net.blocks.values()
            if B["interior"] and B["building"] >= 0]
    out = dict(n_blocks_counted=len(rows))
    if rows:
        r = np.array(rows, float)
        out.update(sides_mean=round(float(r[:, 0].mean()), 2), corners_mean=round(float(r[:, 1].mean()), 2),
                   exits_mean=round(float(r[:, 2].mean()), 2),
                   sides_hist={int(k): int((r[:, 0] == k).sum()) for k in np.unique(r[:, 0])})
    by_block = {b["id"]: b["building"] for b in blocks}
    cyl = []
    for st in L.streets:
        i, j = st["blocks"]
        bi, bj = by_block.get(i, -1), by_block.get(j, -1)
        if st["edge"] or bi < 0 or bj < 0:
            continue
        pi, pj = building_poly(L.boxes[bi]), building_poly(L.boxes[bj])
        ci, cj = (pi.max(0) + pi.min(0)) / 2, (pj.max(0) + pj.min(0)) / 2
        cyl.append(float(np.linalg.norm(ci - cj)) - cylinder_radius(L.boxes[bi]) - cylinder_radius(L.boxes[bj]))
    if cyl:
        cyl = np.array(cyl)
        out.update(cyl_closed=int((cyl < 0).sum()), cyl_narrow=int((cyl < 6.0).sum()), cyl_pairs=len(cyl))
    return out


def poly_gap(a, b):
    """Distance between two convex polygons (0 if they overlap)."""
    fa, fb = ConvexFootprints([a]), ConvexFootprints([b])
    return float(min(fa.distance(b).min(), fb.distance(a).min()))


def ring_stats(L, blocks):
    """The predicted surround separation d* = ring perimeter / 100 over the interior blocks' buildings: a ring of ten
    drones at STANDOFF_M round the building, commanded spacing (10 d_ref m) one tenth of its perimeter. R wraps its
    cylinder, B its footprint."""
    d = []
    for b in blocks:
        if not b["interior"] or b["building"] < 0:
            continue
        bb = L.boxes[b["building"]]
        if is_prism(bb):
            d.append((geo.perimeter(building_poly(bb)) + 2 * math.pi * STANDOFF_M) / 100.0)
        else:
            d.append(2 * math.pi * (cylinder_radius(bb) + STANDOFF_M) / 100.0)
    if not d:
        return dict(d_star_pred_median=None, d_star_pred_p90=None)
    return dict(d_star_pred_median=round(float(np.median(d)), 3), d_star_pred_p90=round(float(np.percentile(d, 90)), 3))


def max_within(L, step=5.0):
    """Most buildings within OlfatiSaber's 50 m query of any free point of the arena (the 64-collider buffer)."""
    fp = footprints(L)
    if fp.n == 0:
        return 0
    x0, x1, z0, z1 = ARENA
    xs, zs = np.arange(x0, x1 + 1e-6, step), np.arange(z0, z1 + 1e-6, step)
    pts = np.stack(np.meshgrid(xs, zs), -1).reshape(-1, 2)
    pts = pts[~fp.inside(pts)]
    cnt = np.zeros(len(pts), int)
    for i in range(fp.n):
        near = np.hypot(*(pts - fp.c[i]).T) - fp.R[i] <= lay.QUERY_RADIUS_M
        if near.any():
            cnt[near] += fp.subset([i]).distance(pts[near]) <= lay.QUERY_RADIUS_M
    return int(cnt.max()) if len(cnt) else 0


# ============================================================================= I/O and accessors

def save(L, blocks, path):
    """citygen's layout format (one box / street per line) plus the top-level `blocks` list, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes = ",\n".join(json.dumps(b, separators=(",", ":")) for b in L.boxes)
    streets = ",\n".join(json.dumps(s, separators=(",", ":")) for s in L.streets)
    blks = ",\n".join(json.dumps(b, separators=(",", ":")) for b in blocks)
    text = ('{"boxes":[\n' + boxes + '\n],\n"tiles":' + json.dumps(L.tiles) + ',\n"streets":[\n' + streets +
            '\n],\n"blocks":[\n' + blks + '\n],\n"layout":' + json.dumps(L.layout, indent=1) + "\n}\n")
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load(path):
    """(lay.Layout, blocks) of a block-city layout."""
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    L = lay.Layout(boxes=d["boxes"], streets=d.get("streets", []), layout=d.get("layout", {}), tiles=d.get("tiles", []))
    return L, d.get("blocks", [])


def footprints(L):
    """The footprints every S4 scorer reads: ConvexFootprints for a variant B city, los.Footprints otherwise."""
    if any(is_prism(b) for b in L.boxes):
        return ConvexFootprints([building_poly(b) for b in L.boxes])
    from citysim.los import Footprints
    return Footprints(L.boxes_as_tuples())


def replica_city(L, obstacle_model=None):
    """The replica's city: rep.City of the boxes (variant R, today's law) or rep.PolyCity of the prisms (variant B,
    which only the ClosestPoint obstacle model reads by its true shape)."""
    from citysim import use_bench_modules
    use_bench_modules()
    import swarm_replica as rep
    if any(is_prism(b) for b in L.boxes):
        return rep.PolyCity([building_poly(b) for b in L.boxes])
    return rep.City(L.boxes_as_tuples())


def path_for(P):
    return LAYOUT_DIR / "sweep~" / OUT_SET / f"{P.ident()}.json"


def scene_path_for(P):
    """BlockShapedCityWorld's file for P: the same layout as path_for's, under the scene's prefix in layouts/."""
    return LAYOUT_DIR / f"{SCENE_PREFIX}{P.ident()}.json"


def ensure(P):
    """The layout file for P, generated if missing: its path."""
    path = path_for(P)
    if not path.exists():
        L, blocks = generate(P)
        save(L, blocks, path)
    return path


# ============================================================================= command line

FEAS_KEYS = ("variant", "p", "s", "seed", "n_blocks", "n_interior_blocks", "n_buildings", "n_plazas", "alpha_achieved",
             "block_share", "ell_mean", "max_within_50m", "n_street_pairs", "cyl_gap_median", "cyl_gap_p10",
             "cyl_gap_frac_ge6", "cyl_gap_frac_ge12", "facade_gap_min", "facade_gap_median", "d_star_pred_median",
             "d_star_pred_p90")


def _feas_job(args):
    variant, p, s, seed = args
    P = Params(variant, p, s, seed)
    L, blocks = generate(P)
    save(L, blocks, path_for(P))
    r = L.layout
    return {k: r.get(k) for k in FEAS_KEYS}


def cmd_feasibility(a):
    import pandas as pd
    jobs = [(v, p, s, seed) for v in VARIANTS for p in P_LEVELS for s in S_LEVELS for seed in range(a.seeds)]
    with Pool(min(a.jobs, len(jobs))) as pool:
        rows = pool.map(_feas_job, jobs)
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else LAYOUT_DIR / "sweep~" / OUT_SET / "feasibility.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    g = df.groupby(["variant", "p", "s"]).mean(numeric_only=True).drop(columns=["seed"]).round(3)
    pd.set_option("display.width", 250)
    print(g[["n_buildings", "n_plazas", "alpha_achieved", "block_share", "ell_mean", "max_within_50m",
             "cyl_gap_median", "cyl_gap_frac_ge6", "facade_gap_min", "d_star_pred_median",
             "d_star_pred_p90"]].to_string())
    print(f"[blockcity] {len(df)} layouts -> {out.parent}; table {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("one")
    p.add_argument("--variant", choices=VARIANTS, default="B")
    p.add_argument("--p", type=float, default=50.0)
    p.add_argument("--s", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inset", type=float, default=INSET_M, help="B: kerb to facade, m (S4: %(default)s)")
    p.add_argument("--seeds", choices=SEEDINGS, default="rsa", help="block seeds (S4: rsa)")
    p.add_argument("--jitter", type=float, default=HEX_JITTER, help="hex: move, x the lattice spacing")
    p.add_argument("--relax", type=int, default=0, help="Lloyd iterations after the seeding")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--out", help="file (default layouts/sweep~/split/<id>.json)")
    g.add_argument("--scene", action="store_true", help=f"write layouts/{SCENE_PREFIX}<id>.json (BlockShapedCityWorld)")
    p = sub.add_parser("feasibility")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--jobs", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "one":
        P = Params(a.variant, a.p, a.s, a.seed, a.inset, a.seeds, a.jitter, a.relax)
        L, blocks = generate(P)
        path = save(L, blocks, Path(a.out) if a.out else scene_path_for(P) if a.scene else path_for(P))
        r = L.layout
        print(f"[blockcity] {path.name}: {r['n_buildings']} buildings ({r['n_plazas']} plazas), alpha "
              f"{r['alpha_achieved']:.3f}, block share {r['block_share']:.2f}, cylinder gap median "
              f"{r.get('cyl_gap_median')} m (>= 6 m: {r.get('cyl_gap_frac_ge6')}), facade gap min "
              f"{r.get('facade_gap_min')} m, predicted d* {r['d_star_pred_median']} (p90 {r['d_star_pred_p90']}), "
              f"max within 50 m {r['max_within_50m']}")
        st = street_stats(L, blocks)
        if st.get("n_blocks_counted"):
            print(f"[blockcity] streets: round an interior building {st['sides_mean']} sides drawn as streets "
                  f"({', '.join(f'{k}: {v}' for k, v in st['sides_hist'].items())}), {st['corners_mean']} corners, "
                  f"{st['exits_mean']} streets leaving them")
        if "cyl_pairs" in st:
            print(f"[blockcity] avoidance: Unity's cylinders close {st['cyl_closed']} and narrow below 6 m "
                  f"{st['cyl_narrow']} of {st['cyl_pairs']} interior streets between two buildings")
        print(f"LAYOUT_PATH={path}")
    else:
        cmd_feasibility(a)


if __name__ == "__main__":
    main()
