"""The irregular-block city generator: isotropic, no grid, with a connected street network.

    python -B -m citysim.citygen one --p 70 --s 12 --alpha 0.15 --wc 13.06 --rg 30 --seed 1     # -> layouts/<id>.json
    python -B -m citysim.citygen sweep --spec sweep.json --seeds 0 1 2 --jobs 16                # -> layouts/sweep~/
    python -B -m citysim.citygen trial --seeds 101 102 103 104    # the trial levels (levels.json) -> layouts/
    python -B -m citysim.citygen mixcity --city DiamondCityWorld --seed 1   # VoronoiCityWorld's layout -> layouts/

Blocks mode (the study's cities):
  1. Block seeds: random sequential adsorption over the arena, round(A / p^2) points no two closer than 0.8 p --
     a blue-noise set at one block per p x p, so p is the mean block spacing (today's tile pitch is 90.8 m).
  2. Blocks are the seeds' Voronoi cells clipped to the arena; every cell edge is a street centreline, and each
     block is its cell shrunk by s/2 on every side, so every street is s wide and the street graph is connected
     by construction (it is the Voronoi diagram's edge graph). Arena-boundary edges are streets open outside.
  3. Goals first (so clearing them costs no coverage): GOALS blocks at least 150 m apart and 100 m from the
     spawn, one from each third of the eligible spawn distances (near / mid / far), each rebuilt to the goal
     geometry G: a central occluder (the study design scaled to a long side of w_c) and n_b patrolled
     buildings (the design at the ring size) centred on a circle of radius r_g, long side tangential. Nothing
     else may stand within the patch's clear radius r_g + ring half-diagonal + walk_clear of its centre.
  4. City buildings: the study design at `size`, yaw 0 / 90, dropped by random sequential adsorption into the
     blocks round-robin, wholly inside a block, at least g (the facade gap) from every other building and
     outside every patch, until the coverage reaches alpha (or block fill f) or every block has jammed.
     core="shell" (a POROUS city, for the capability study) also keeps every two avoidance cylinders -- the
     circumcircles OlfatiSaber steers round -- at least g_shell apart, so with g_shell >= the formation's pass width
     (routing.w_pass) the swarm can flow between any two buildings and the blocks are open to it. The default
     core="facade" is the study's original packing, under which dense blocks close to the avoidance law.
Scatter mode (`--mode scatter`, the Boolean-model comparison): no blocks or streets; buildings dropped over the
whole arena with a hard core between their avoidance cylinders (centres at least R_i + R_j + g apart).

A building MIX (`mix` = a tile city, catalogue.city_mix) replaces the one design: every city building is drawn from
that city's designs in proportion to how often each stands there; a goal's occluder is a mix design whose long side
is near w_c (OCCLUDER_RATIO), scaled to exactly w_c, and its ring buildings are drawn from the mix too, so a goal is
not given away by a ring of identical buildings. A goal's clear radius then follows its own ring. `mixcity` is the
preset: a tile city's own buildings at size 1 (the ScaledCity scale), its coverage over its tile grid and its tile
pitch as p -- the same city in everything but its streets (VoronoiCityWorld.unity builds it).

Coverage alpha is over the arena (730 m square, the size of today's 8 x 8 city), so the tile cities sit on the
same axis (citysim.coverage). c_g, the clear radius round the occluder, is MEASURED: the distance from the goal
centre to the nearest building other than the occluder (normally the ring's inner faces) -- the room the
surroundings leave the formation. Everything is deterministic in (parameters, seed).
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from dataclasses import asdict, dataclass, field, replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import LAYOUT_DIR, catalogue, coverage  # noqa: E402
from citysim import geometry as geo  # noqa: E402
from citysim import layout as lay  # noqa: E402

ARENA = (-365.0, 365.0, -365.0, 365.0)      # 730 m square about the origin (DiamondCityWorld spans ~730 m)
SPAWN = (-456.0, 0.0, 0.0)                    # DiamondCityWorld's swarmSpawn start: 91 m west of the arena
SPAWN_YAW = 90.0                              # facing east, into the city
SEED_SPACING_RATIO = 0.8                      # RSA exclusion distance / p (disc coverage 0.50 < RSA jamming 0.547)
JAM_TRIES = 400                               # consecutive failed drops after which a block is jammed
TRIES_PER_PASS = 40
GRID_CELL = 40.0                              # spatial hash cell for the gap tests

# Formation diameters (m) by d_ref, for the narrow-street statistic: the settled hollow-core hover template of 10
# drones (formation.py writes data/formation_diameters.json; these are the fallback until it has run).
FALLBACK_DIAMETERS = {0.4: 6.5, 0.7: 10.5, 1.0: 14.5, 1.3: 18.5, 1.6: 22.5}
STANDOFF_M = 4.0                              # d_obs * scaleFactor: the obstacle repulsion's reach
# Params fields added after generator version 2, with the value that reproduces it: recorded in a layout's params only
# when set otherwise, so a version-2 layout regenerated today is byte-identical to the committed one.
LATER_FIELDS = {"core": "facade", "g_shell": 0.0, "mix": "", "place": "rsa"}
MIXCITY_PREFIX = "VoronoiCity__"            # mixcity's file names: <prefix><id>.json (layout.resolve finds the id)


@dataclass
class Params:
    mode: str = "blocks"
    p: float = 90.83          # mean block spacing (m)
    s: float = 15.0           # street width (m)
    alpha: float = 0.10       # target coverage over the arena (ignored when f is given)
    f: float = -1.0           # target block fill (built / block area); < 0 = use alpha
    g: float = 4.0            # facade gap between buildings (m)
    size: float = 2.0         # horizontal scale of the study design for city buildings
    design: str = ""          # "" = catalogue.study_design()
    w_c: float = 13.06        # goal occluder long side (m)
    r_g: float = 30.0         # goal ring radius (m)
    n_b: int = 8              # patrolled buildings per goal
    ring_size: float = -1.0   # scale of the ring buildings; < 0 = size
    goals: int = 3
    walk_clear: float = 5.0   # kept free round each ring building (walkers loop 0.5-2 m off its facade)
    goal_gap: float = 5.0     # least gap between two goal buildings
    arena: tuple = ARENA
    spawn: tuple = SPAWN
    spawn_yaw: float = SPAWN_YAW
    core: str = "facade"      # "shell": also keep avoidance cylinders g_shell apart (a porous city)
    g_shell: float = 0.0      # least gap between two avoidance cylinders under core="shell" (m)
    mix: str = ""             # a tile city whose building mix replaces the one design ("" = one design)
    place: str = "rsa"        # city buildings in blocks: "rsa" (the study's) or "even" (fill_even)

    def ident(self, seed):
        if self.mode == "scatter":
            core = f"sc_a{self.alpha:g}_g{self.g:g}"
        else:
            fill = f"f{self.f:g}" if self.f >= 0 else f"a{self.alpha:g}"
            core = f"bl_p{self.p:g}_s{self.s:g}_{fill}_g{self.g:g}"
            if self.core == "shell":
                core += f"_gs{self.g_shell:g}"
        ring = "" if self.ring_size < 0 else f"_rs{self.ring_size:g}"
        mix = f"_mix{self.mix}" if self.mix else ""
        place = "" if self.place == "rsa" else f"_{self.place}"
        return f"{core}_x{self.size:g}_wc{self.w_c:g}_rg{self.r_g:g}_nb{self.n_b}{ring}{mix}{place}_seed{seed}"


@dataclass(frozen=True)
class Mix:
    """A building mix: a tile city's designs and the share of its buildings each one is (catalogue.city_mix)."""
    city: str
    designs: tuple
    counts: tuple

    @classmethod
    def of(cls, city):
        m = catalogue.city_mix(city)
        return cls(city, tuple(d for d, _ in m), tuple(c for _, c in m))

    @property
    def p(self):
        c = np.array(self.counts, float)
        return c / c.sum()

    @property
    def mean_area(self):
        """Mean footprint at size 1 (m^2), over the city's buildings."""
        return float(sum(d.w * d.d * q for d, q in zip(self.designs, self.p)))

    @property
    def max_half_diag(self):
        return max(0.5 * math.hypot(d.w, d.d) for d in self.designs)

    def draw(self, rng, n=None):
        """One design, or a list of n, drawn by share."""
        k = rng.choice(len(self.designs), size=n, p=self.p)
        return self.designs[int(k)] if n is None else [self.designs[int(i)] for i in k]


class RectIndex:
    """Axis-aligned rectangles with a grid hash, for 'is this new rectangle at least g from all of them'."""

    def __init__(self, cell=GRID_CELL):
        self.cell = cell
        self.rects = []
        self.grid = {}
        self.max_half = 0.0

    def _cells(self, r, pad):
        c = self.cell
        i0, i1 = int(math.floor((r[0] - r[2] - pad) / c)), int(math.floor((r[0] + r[2] + pad) / c))
        j0, j1 = int(math.floor((r[1] - r[3] - pad) / c)), int(math.floor((r[1] + r[3] + pad) / c))
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    def add(self, r):
        k = len(self.rects)
        self.rects.append(tuple(r))
        self.max_half = max(self.max_half, r[2], r[3])
        for key in self._cells(r, 0.0):
            self.grid.setdefault(key, []).append(k)
        return k

    def near(self, r, pad):
        out = set()
        for key in self._cells(r, pad + self.max_half):
            out.update(self.grid.get(key, ()))
        return sorted(out)

    def clear(self, r, gap):
        idx = self.near(r, gap)
        if not idx:
            return True
        return bool((geo.rect_gap(r, np.array([self.rects[k] for k in idx])) >= gap - 1e-9).all())

    def shell_clear(self, r, gap):
        """Whether r's circumcircle is at least `gap` from every other rectangle's circumcircle (the avoidance
        cylinders OlfatiSaber steers round)."""
        R = math.hypot(r[2], r[3])
        idx = self.near(r, gap + R + math.hypot(self.max_half, self.max_half))
        if not idx:
            return True
        o = np.array([self.rects[k] for k in idx])
        d = np.hypot(o[:, 0] - r[0], o[:, 1] - r[1]) - np.hypot(o[:, 2], o[:, 3]) - R
        return bool((d >= gap - 1e-9).all())


def load_diameters():
    from citysim import PACKAGE_DIR
    p = PACKAGE_DIR / "data" / "formation_diameters.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return {float(k): float(v) for k, v in d["diameter_m"].items()}, "formation.py"
    return dict(FALLBACK_DIAMETERS), "fallback"


# ============================================================================= goals

RING_ROTATIONS = 24       # ring phases tried, in random order, before a goal geometry is declared infeasible
RING_DRAWS = 8            # a mix's ring designs drawn afresh this often before a goal geometry is declared infeasible
OCCLUDER_RATIO = 1.4      # a mix's occluder: a design whose long side is within this factor of w_c


def goal_buildings(P, design, centre, rng):
    """[(rect, yaw, size, role)] of one goal patch at `centre`, or raises ValueError if G does not fit.

    The ring's phase is random; phases whose buildings come closer than goal_gap (axis-aligned footprints on a
    circle collide at some phases and not others) are skipped, and G is infeasible only if none fits."""
    ring_size = P.size if P.ring_size < 0 else P.ring_size
    c_size = P.w_c / design.long
    yaw_c = int(rng.integers(2)) * 90
    hx, hz = design.footprint(c_size, yaw_c)
    base = [((centre[0], centre[1], hx, hz), yaw_c, c_size, "goal_centre")]
    out, worst = fit_ring(P, base, [design] * P.n_b, centre, ring_size, rng)
    if out is None:
        raise infeasible(P, worst)
    return out


def goal_buildings_mix(P, mix, centre, rng):
    """[(rect, yaw, size, role, design)] of one goal patch drawn from a building Mix, or raises ValueError.

    The occluder is a design whose long side is within OCCLUDER_RATIO of w_c, drawn by share (else the nearest one),
    scaled to a long side of exactly w_c, so it stays close to its own proportions in the city; the n_b ring
    buildings are drawn from the mix at the ring size. A draw no ring phase fits is redrawn, RING_DRAWS times."""
    ring_size = P.size if P.ring_size < 0 else P.ring_size
    occ = occluder_design(mix, P.w_c, rng)
    c_size = P.w_c / occ.long
    yaw_c = int(rng.integers(2)) * 90
    hx, hz = occ.footprint(c_size, yaw_c)
    base = [((centre[0], centre[1], hx, hz), yaw_c, c_size, "goal_centre")]
    worst = None
    for _ in range(RING_DRAWS):
        ring = mix.draw(rng, P.n_b)
        out, clash = fit_ring(P, base, ring, centre, ring_size, rng)
        if out is not None:
            return [o + (d,) for o, d in zip(out, [occ] + ring)]
        worst = clash if worst is None or clash[0] > worst[0] else worst
    raise infeasible(P, worst)


def occluder_design(mix, w_c, rng):
    near = [k for k, d in enumerate(mix.designs) if 1 / OCCLUDER_RATIO <= d.long / w_c <= OCCLUDER_RATIO]
    if not near:
        return min(mix.designs, key=lambda d: (abs(math.log(d.long / w_c)), d.design))
    p = mix.p[near]
    return mix.designs[near[int(rng.choice(len(near), p=p / p.sum()))]]


def fit_ring(P, base, designs, centre, ring_size, rng):
    """(base + the ring of `designs` at the first random phase that keeps every two goal buildings goal_gap apart,
    None), or (None, the least-bad clash)."""
    phases = rng.uniform(0, 2 * math.pi) + 2 * math.pi / max(P.n_b, 1) * rng.permutation(RING_ROTATIONS) / RING_ROTATIONS
    worst = None
    for theta0 in phases:
        out = base + ring_buildings(P, designs, centre, theta0, ring_size)
        rects = np.array([o[0] for o in out])
        clash = None
        for i in range(len(rects)):
            gap = geo.rect_gap(rects[i], rects[i + 1:])
            if len(gap) and gap.min() < P.goal_gap - 1e-9:
                clash = (float(gap.min()), out[i][3], i, out[i + 1 + int(np.argmin(gap))][3], i + 1 + int(np.argmin(gap)))
                break
        if clash is None:
            return out, None
        worst = clash if worst is None or clash[0] > worst[0] else worst
    return None, worst


def infeasible(P, worst):
    gmin, ra, i, rb, j = worst
    return ValueError(f"goal geometry infeasible: at every ring phase two goal buildings come closer than goal_gap "
                      f"{P.goal_gap:g} m (best {gmin:.1f} m, {ra} {i} / {rb} {j}); raise r_g or lower n_b / w_c")


def ring_buildings(P, designs, centre, theta0, ring_size):
    """The ring: designs[k] (one per patrolled building) centred on the circle at phase theta0 + 2 pi k / n_b."""
    out = []
    for k, design in enumerate(designs):
        th = theta0 + 2 * math.pi * k / P.n_b
        x, z = centre[0] + P.r_g * math.cos(th), centre[1] + P.r_g * math.sin(th)
        # long side tangential: the design's long side is its local X (yaw 0 -> world x)
        long_on_x = design.w >= design.d
        tangent_on_x = abs(math.sin(th)) >= abs(math.cos(th))
        yaw = 0 if long_on_x == tangent_on_x else 90
        hx, hz = design.footprint(ring_size, yaw)
        out.append(((x, z, hx, hz), yaw, ring_size, "goal_ring"))
    return out


def clear_radius(P, design, mix=None):
    """r_g + the ring buildings' half-diagonal + walk_clear; for a mix, the largest any of its designs could need."""
    ring_size = P.size if P.ring_size < 0 else P.ring_size
    half = mix.max_half_diag if mix is not None else 0.5 * math.hypot(design.w, design.d)
    return P.r_g + half * ring_size + P.walk_clear


def choose_goal_centres(P, rng, candidates, r_clear):
    """GOALS centres from `candidates` [(block_id, (x, z))]: inside the arena by r_clear, >= 100 m from the
    spawn, >= 150 m apart, one from each third of the eligible spawn distances. [(block_id, centre)]."""
    x0, x1, z0, z1 = P.arena
    sx, sz = P.spawn[0], P.spawn[2]
    elig = [(b, c) for b, c in candidates
            if x0 + r_clear <= c[0] <= x1 - r_clear and z0 + r_clear <= c[1] <= z1 - r_clear
            and math.hypot(c[0] - sx, c[1] - sz) >= lay.GOAL_MIN_SPAWN_DIST_M]
    if len(elig) < P.goals:
        raise ValueError(f"only {len(elig)} blocks can hold a goal patch of clear radius {r_clear:.0f} m")
    dist = np.array([math.hypot(c[0] - sx, c[1] - sz) for _, c in elig])
    order = np.argsort(dist)
    strata = np.array_split(order, P.goals)
    for _ in range(2000):
        pick = [int(rng.choice(s)) for s in strata]
        cs = np.array([elig[k][1] for k in pick])
        dd = np.linalg.norm(cs[:, None] - cs[None], axis=-1)
        if (dd[np.triu_indices(len(pick), 1)] >= lay.GOAL_MIN_SEPARATION_M).all():
            return [elig[k] for k in pick]
    for _ in range(5000):               # strata too tight: any well-separated triple
        pick = list(rng.choice(len(elig), P.goals, replace=False))
        cs = np.array([elig[k][1] for k in pick])
        dd = np.linalg.norm(cs[:, None] - cs[None], axis=-1)
        if (dd[np.triu_indices(len(pick), 1)] >= lay.GOAL_MIN_SEPARATION_M).all():
            return sorted((elig[k] for k in pick), key=lambda e: math.hypot(e[1][0] - sx, e[1][1] - sz))
    raise ValueError(f"no {P.goals} goal blocks {lay.GOAL_MIN_SEPARATION_M:g} m apart")


# ============================================================================= generation

def generate(P, seed):
    """The layout for parameters P and seed: a lay.Layout (raises ValueError if G cannot be built)."""
    rng = np.random.default_rng(seed)
    if P.mix and P.design:
        raise ValueError("give a design or a mix, not both")
    if P.place not in ("rsa", "even") or (P.place == "even" and P.mode != "blocks"):
        raise ValueError(f"place {P.place!r}: 'rsa', or 'even' in blocks mode")
    design = catalogue.get(P.design) if P.design else catalogue.study_design()
    mix = Mix.of(P.mix) if P.mix else None
    ring_size = P.size if P.ring_size < 0 else P.ring_size
    x0, x1, z0, z1 = P.arena
    A = (x1 - x0) * (z1 - z0)

    # ---- blocks and streets
    blocks, streets = [], []
    if P.mode == "blocks":
        n_seeds = max(int(round(A / P.p ** 2)), 4)
        seeds = geo.rsa_points(rng, P.arena, n_seeds, SEED_SPACING_RATIO * P.p)
        cells, edges = geo.voronoi_cells(seeds, P.arena)
        blocks = [geo.offset_convex(c, P.s / 2) for c in cells]
        for k, (i, j, a, b) in enumerate(edges):
            ab = [round(float(a[0]), 3), round(float(a[1]), 3), round(float(b[0]), 3), round(float(b[1]), 3)]
            streets.append(dict(id=k, polyline=[ab[:2], ab[2:]], xz=ab, width=P.s, edge=bool(j < 0),
                                blocks=[int(i), int(j)]))
        candidates = [(k, tuple(geo.centroid(b))) for k, b in enumerate(blocks) if len(b) >= 3]
    elif P.mode == "scatter":
        g = np.linspace(0, 1, 9)[1:-1]
        candidates = [(-1, (x0 + u * (x1 - x0), z0 + v * (z1 - z0))) for u in g for v in g]
    else:
        raise ValueError(f"unknown mode {P.mode!r}")

    # ---- goals: reserved before the city is filled
    index, boxes = RectIndex(), []
    r_clear = clear_radius(P, design, mix)      # under a mix, the most any goal could need: which blocks can hold one
    goal_recs, radii = [], []
    for gi, (block_id, centre) in enumerate(choose_goal_centres(P, rng, candidates, r_clear)):
        if mix is None:
            bld = [b + (design,) for b in goal_buildings(P, design, centre, rng)]
            r_goal = r_clear
        else:
            bld = goal_buildings_mix(P, mix, centre, rng)
            half = max(0.5 * math.hypot(b[4].w, b[4].d) for b in bld if b[3] == "goal_ring")
            r_goal = P.r_g + half * ring_size + P.walk_clear
        names = []
        for rect, yaw, size, role, dsg in bld:
            k = index.add(rect)
            name = f"{dsg.design}_G{gi}_{len(names):02d}"
            boxes.append(lay.make_box(name, f"goal_{gi}", rect[0], rect[1], dsg, size, yaw, role=role, goal=gi,
                                      patrol=role == "goal_ring"))
            names.append(name)
        radii.append(r_goal)
        goal_recs.append(dict(index=gi, block_id=int(block_id), centre=[round(centre[0], 3), round(centre[1], 3)],
                              centre_box=names[0], ring_boxes=names[1:], w_c=round(P.w_c, 3),
                              r_g=round(P.r_g, 3), n_b=P.n_b, clear_radius=round(r_goal, 3),
                              ring_size=round(ring_size, 4)))
    centres = np.array([g["centre"] for g in goal_recs], float)
    radii = np.array(radii, float)

    # ---- city buildings
    target_area = (P.f * sum(abs(geo.area(b)) for b in blocks if len(b) >= 3) if P.f >= 0 and P.mode == "blocks"
                   else P.alpha * A)
    built = float(geo.rect_area(np.array(index.rects)).sum())
    one = (design.w * design.d if mix is None else mix.mean_area) * P.size ** 2
    n_city = 0

    def pick():
        """The next city building's design (a mix draws one; one design costs no random number)."""
        return design if mix is None else mix.draw(rng)

    def allowed(r):
        if len(centres) and (geo.rect_point_distance([r], centres) < radii).any():
            return False
        if geo.rect_point_distance([r], [[P.spawn[0], P.spawn[2]]])[0, 0] < lay.SPAWN_CLEAR_M:
            return False
        return True

    def drop(rect, yaw, tile, dsg):
        nonlocal built, n_city
        index.add(rect)
        boxes.append(lay.make_box(f"{dsg.design}_{n_city:04d}", tile, rect[0], rect[1], dsg, P.size, yaw))
        built += 4 * rect[2] * rect[3]
        n_city += 1

    def usable(pts):
        """Which points a building could stand on as far as the goals and the spawn go (allowed, for points)."""
        ok = np.ones(len(pts), bool)
        if len(centres):
            ok &= (np.hypot(pts[:, None, 0] - centres[None, :, 0], pts[:, None, 1] - centres[None, :, 1])
                   >= radii[None, :]).all(axis=1)
        return ok & (np.hypot(pts[:, 0] - P.spawn[0], pts[:, 1] - P.spawn[2]) >= lay.SPAWN_CLEAR_M)

    jammed_blocks = 0
    if P.mode == "blocks" and P.place == "even":
        jammed_blocks = fill_even(P, blocks, index, rng, pick, allowed, usable, drop,
                                  lambda: built < target_area - 0.5 * one)
        jammed = built < target_area - 0.5 * one
    elif P.mode == "blocks":
        live = [k for k, b in enumerate(blocks) if len(b) >= 3]
        fails = {k: 0 for k in live}
        halfplanes = {k: geo.edges_halfplanes(blocks[k]) for k in live}
        bbox = {k: (blocks[k][:, 0].min(), blocks[k][:, 0].max(), blocks[k][:, 1].min(), blocks[k][:, 1].max())
                for k in live}
        order = list(live)
        while built < target_area - 0.5 * one and live:
            rng.shuffle(order)
            for k in list(order):
                if built >= target_area - 0.5 * one:
                    break
                if fails[k] >= JAM_TRIES:
                    continue
                nrm, cc = halfplanes[k]
                bx0, bx1, bz0, bz1 = bbox[k]
                placed = False
                for _ in range(TRIES_PER_PASS):
                    dsg = pick()
                    yaw = int(rng.integers(2)) * 90
                    hx, hz = dsg.footprint(P.size, yaw)
                    if bx1 - bx0 < 2 * hx or bz1 - bz0 < 2 * hz:
                        fails[k] += 1
                        continue
                    r = (rng.uniform(bx0 + hx, bx1 - hx), rng.uniform(bz0 + hz, bz1 - hz), hx, hz)
                    corners = geo.rect_corners(r)
                    if not ((corners @ nrm.T) <= cc[None, :] + 1e-9).all():
                        fails[k] += 1
                        continue
                    if not allowed(r) or not index.clear(r, P.g) or \
                            (P.core == "shell" and not index.shell_clear(r, P.g_shell)):
                        fails[k] += 1
                        continue
                    drop(r, yaw, f"B{k:03d}", dsg)
                    fails[k] = 0
                    placed = True
                    break
                if not placed and fails[k] >= JAM_TRIES:
                    order.remove(k)
            live = order
        jammed_blocks = sum(1 for k in fails if fails[k] >= JAM_TRIES)
        jammed = built < target_area - 0.5 * one
    else:
        fails = 0
        while built < target_area - 0.5 * one and fails < 20 * JAM_TRIES:
            dsg = pick()
            yaw = int(rng.integers(2)) * 90
            hx, hz = dsg.footprint(P.size, yaw)
            r = (rng.uniform(x0 + hx, x1 - hx), rng.uniform(z0 + hz, z1 - hz), hx, hz)
            R = math.hypot(hx, hz)
            near = index.near(r, P.g + 2 * R)
            if near:
                o = np.array([index.rects[i] for i in near])
                d = np.hypot(o[:, 0] - r[0], o[:, 1] - r[1])
                if (d < R + np.hypot(o[:, 2], o[:, 3]) + P.g).any():
                    fails += 1
                    continue
            if not allowed(r):
                fails += 1
                continue
            drop(r, yaw, "scatter", dsg)
            fails = 0
        jammed = built < target_area - 0.5 * one

    L = lay.Layout(boxes=boxes, streets=streets)
    rects = L.rects()
    for g in goal_recs:
        own = [k for k, b in enumerate(boxes) if b["goal"] == g["index"]]
        occ = own[0]
        R_c = math.hypot(rects[occ, 2], rects[occ, 3])
        others = [k for k in range(len(rects)) if k != occ]
        g["R_c"] = round(R_c, 3)
        g["c_g"] = round(float(geo.rect_point_distance(rects[others], [g["centre"]]).min()), 3)
        g["spawn_dist"] = round(math.hypot(g["centre"][0] - P.spawn[0], g["centre"][1] - P.spawn[2]), 1)

    # ---- the record
    st = coverage.stats(L.boxes_as_tuples(), A)
    diam, diam_src = load_diameters()
    record = dict(
        generator_version=lay.GENERATOR_VERSION, id=P.ident(seed), seed=int(seed), mode=P.mode,
        design=design.design if mix is None else f"mix:{mix.city}",
        design_wd=[design.w, design.d] if mix is None else None, size=P.size, p=P.p, s=P.s, g=P.g,
        f=round(built / sum(abs(geo.area(b)) for b in blocks if len(b) >= 3), 4) if P.mode == "blocks" else None,
        f_target=P.f if P.f >= 0 else None, alpha_target=round(target_area / A, 4),
        alpha_achieved=round(built / A, 4), jammed=bool(jammed), blocks_jammed=int(jammed_blocks),
        n_blocks=int(sum(1 for b in blocks if len(b) >= 3)), n_buildings=len(boxes), n_city_buildings=n_city,
        ell_mean=st["ell_m"], ell_theta=st["ell_theta_m"], ell_anisotropy=st["anisotropy"], p_free_40m=st["p_free"],
        arena=list(P.arena), extent=extent(rects), spawn=list(P.spawn), spawn_yaw=P.spawn_yaw, goals=goal_recs,
        params={k: v for k, v in asdict(P).items() if k not in ("arena", "spawn") and
                not (k in LATER_FIELDS and v == LATER_FIELDS[k])})
    if P.core != "facade":
        record.update(core=P.core, g_shell=P.g_shell)
    if mix is not None:
        # the shares drawn from, as they stood when the layout was made (the city may be restyled since)
        record.update(mix=mix.city, mix_counts={d.design: c for d, c in zip(mix.designs, mix.counts)})
    record.update(neighbourhood_stats(rects, P))
    record.update(street_stats(L, rects, diam, goal_recs))
    record["formation_diameters_source"] = diam_src
    L.layout = record
    return L


EVEN_CANDIDATES = 16      # place="even": candidate spots per building; the best spread one is built
EVEN_EDGE_WEIGHT = 2.0    # ... a block's edge counts as a neighbour this many times its distance (half a gap)
EVEN_GRID_M = 2.0         # ... sampling step for a block's usable area


def fill_even(P, blocks, index, rng, pick, allowed, usable, drop, wanting):
    """place="even": the block with the least built fraction of its usable area (outside goal patches and the
    spawn) takes the next building, at the best spread of EVEN_CANDIDATES spots -- Mitchell's best candidate: the
    spot whose least distance to another building, or EVEN_EDGE_WEIGHT times its distance to the block's edge, is
    largest. So every block fills to the same fraction, and a block's buildings spread over it rather than clumping:
    the RSA's round robin gives every block the same COUNT whatever its area, and leaves big blocks mostly empty at
    low coverage. Same rules as the RSA otherwise (wholly inside, the facade gap, the shell). Returns the number of
    jammed blocks (JAM_TRIES candidates in a row that fit nowhere)."""
    live, area, fill, fails, hp, bbox = [], {}, {}, {}, {}, {}
    for k, b in enumerate(blocks):
        if len(b) < 3:
            continue
        bb = (b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max())
        X, Z = np.meshgrid(np.arange(bb[0] + EVEN_GRID_M / 2, bb[1], EVEN_GRID_M),
                           np.arange(bb[2] + EVEN_GRID_M / 2, bb[3], EVEN_GRID_M))
        pts = np.c_[X.ravel(), Z.ravel()]
        pts = pts[geo.contains(b, pts)]
        a = float(usable(pts).sum()) * EVEN_GRID_M ** 2
        if a <= 0:
            continue
        live.append(k)
        area[k], fill[k], fails[k], hp[k], bbox[k] = a, 0.0, 0, geo.edges_halfplanes(b), bb
    jam = max(JAM_TRIES // EVEN_CANDIDATES, 1)
    while wanting() and live:
        k = min(live, key=lambda q: (fill[q], q))
        nrm, cc = hp[k]
        bx0, bx1, bz0, bz1 = bbox[k]
        dsg = pick()
        best, best_yaw, best_score = None, 0, -math.inf
        for _ in range(EVEN_CANDIDATES):
            yaw = int(rng.integers(2)) * 90
            hx, hz = dsg.footprint(P.size, yaw)
            if bx1 - bx0 < 2 * hx or bz1 - bz0 < 2 * hz:
                continue
            r = (rng.uniform(bx0 + hx, bx1 - hx), rng.uniform(bz0 + hz, bz1 - hz), hx, hz)
            slack = cc[None, :] - geo.rect_corners(r) @ nrm.T          # each corner's distance inside each edge
            if (slack < -1e-9).any():
                continue
            if not allowed(r) or not index.clear(r, P.g) or \
                    (P.core == "shell" and not index.shell_clear(r, P.g_shell)):
                continue
            score = EVEN_EDGE_WEIGHT * float(slack.min())
            near = index.near(r, score)
            if near:
                score = min(score, float(geo.rect_gap(r, np.array([index.rects[i] for i in near])).min()))
            if score > best_score:
                best, best_yaw, best_score = r, yaw, score
        if best is None:
            fails[k] += 1
            if fails[k] >= jam:
                live.remove(k)
            continue
        drop(best, best_yaw, f"B{k:03d}", dsg)
        fill[k] += 4 * best[2] * best[3] / area[k]
        fails[k] = 0
    return sum(1 for k in fails if fails[k] >= jam)


def extent(rects):
    if not len(rects):
        return None
    return [round(float((rects[:, 0] - rects[:, 2]).min()), 2), round(float((rects[:, 0] + rects[:, 2]).max()), 2),
            round(float((rects[:, 1] - rects[:, 3]).min()), 2), round(float((rects[:, 1] + rects[:, 3]).max()), 2)]


def neighbourhood_stats(rects, P, step=5.0):
    """The OverlapSphere bound (most buildings within 50 m of any free point) and the facade gaps."""
    x0, x1, z0, z1 = P.arena
    xs, zs = np.arange(x0, x1 + 1e-6, step), np.arange(z0, z1 + 1e-6, step)
    X, Z = np.meshgrid(xs, zs)
    pts = np.c_[X.ravel(), Z.ravel()]
    worst = 0
    for k in range(0, len(pts), 1500):
        d = geo.rect_point_distance(rects, pts[k:k + 1500])          # (n_rects, m)
        free = (d > 0).all(axis=0)
        cnt = (d <= lay.QUERY_RADIUS_M).sum(axis=0)
        if free.any():
            worst = max(worst, int(cnt[free].max()))
    gaps = []
    for i in range(len(rects)):
        g = geo.rect_gap(rects[i], np.delete(rects, i, axis=0))
        gaps.append(float(g.min()) if len(g) else math.inf)
    gaps = np.array(gaps)
    return dict(max_within_50m=worst, facade_gap_min=round(float(gaps.min()), 3) if len(gaps) else None,
                facade_gap_median=round(float(np.median(gaps)), 3) if len(gaps) else None)


def street_stats(L, rects, diameters, goal_recs, step=5.0):
    """Clear width along the streets: twice the distance from a centreline point (sampled every `step` m, outside
    the goal patches) to the nearest building. min_street_width and street_width_p10 count city buildings only --
    the generator's guarantee, s -- while the narrow-street shares (too narrow for the formation's diameter plus a
    standoff each side) count every building, a goal ring spilling over a street included."""
    pts = lay.street_points(L.streets, step, skip_edges=True)
    for g in goal_recs:
        pts = pts[np.hypot(pts[:, 0] - g["centre"][0], pts[:, 1] - g["centre"][1]) > g["clear_radius"]]
    city = rects[[b["role"] == "city" for b in L.boxes]]
    if len(pts) == 0 or len(city) == 0:
        return dict(min_street_width=None, street_width_p10=None,
                    narrow_street_frac_by_spread={f"{k:g}": None for k in diameters})
    w_city, w_all = np.empty(len(pts)), np.empty(len(pts))
    for k in range(0, len(pts), 1500):
        w_city[k:k + 1500] = 2 * geo.rect_point_distance(city, pts[k:k + 1500]).min(axis=0)
        w_all[k:k + 1500] = 2 * geo.rect_point_distance(rects, pts[k:k + 1500]).min(axis=0)
    return dict(min_street_width=round(float(w_city.min()), 2),
                street_width_p10=round(float(np.percentile(w_city, 10)), 2),
                narrow_street_frac_by_spread={f"{k:g}": round(float((w_all < D + 2 * STANDOFF_M).mean()), 4)
                                              for k, D in sorted(diameters.items())})


# ============================================================================= command line

def params_from(a):
    return Params(mode=a.mode, p=a.p, s=a.s, alpha=a.alpha, f=a.f, g=a.g, size=a.size, design=a.design or "",
                  w_c=a.wc, r_g=a.rg, n_b=a.nb, ring_size=a.ring_size, goals=a.goals, core=a.core,
                  g_shell=a.g_shell, mix=a.mix or "", place=a.place)


def cmd_one(a):
    P = params_from(a)
    L = generate(P, a.seed)
    return report(L, Path(a.out) if a.out else LAYOUT_DIR / f"{L.id}.json")


def report(L, out):
    """Saves L to `out`, prints its summary and any rule it breaks; 1 if it breaks one."""
    problems = L.verify()
    L.save(out)
    r = L.layout
    print(f"[citygen] {out}: {r['n_buildings']} buildings, alpha {r['alpha_achieved']:.4f} (target {r['alpha_target']:.4f}"
          f"{', JAMMED' if r['jammed'] else ''}), ell {r['ell_mean']:.0f} m, max within 50 m {r['max_within_50m']}, "
          f"min street {r['min_street_width']} m")
    for g in r["goals"]:
        print(f"  goal {g['index']}: ({g['centre'][0]:.0f}, {g['centre'][1]:.0f}) {g['spawn_dist']:.0f} m from the "
              f"spawn, w_c {g['w_c']} R_c {g['R_c']}, r_g {g['r_g']}, c_g {g['c_g']}")
    for p in problems:
        print("  PROBLEM:", p)
    return 1 if problems else 0


def cmd_mixcity(a):
    """A tile city rebuilt as a Voronoi city: its own buildings (its mix, at size 1, the ScaledCity scale) at its
    own coverage over its tile grid (coverage.city_report), with its tile pitch as the block spacing -- the same
    city in everything but its streets. Any --param overrides the preset; --alpha overrides the measured one."""
    P = replace(params_from(a), mix=a.city)
    if a.alpha is None:
        P = replace(P, alpha=round(coverage.city_report(a.city)["tile_grid"]["alpha"], 4))
    L = generate(P, a.seed)
    n = sum(1 for b in L.boxes if b["role"] == "city")
    print(f"[citygen] {a.city}'s mix: {len(set(b['design'] for b in L.boxes))} designs in {len(L.boxes)} buildings "
          f"({n} city), alpha {P.alpha:g} from its tile grid" + (" (overridden)" if a.alpha is not None else ""))
    out = Path(a.out) if a.out else LAYOUT_DIR / f"{MIXCITY_PREFIX}{L.id}.json"
    code = report(L, out)
    print(f"LAYOUT_PATH={out.resolve()}")             # for Tools/Swarm/Voronoi city (VoronoiCityWindow)
    return code


def _sweep_job(job):
    P, seed, out_dir = job
    try:
        L = generate(P, seed)
    except ValueError as e:
        return dict(id=P.ident(seed), error=str(e))
    path = Path(out_dir) / f"{L.id}.json"
    L.save(path)
    r = L.layout
    return dict(id=L.id, path=str(path), alpha=r["alpha_achieved"], jammed=r["jammed"],
                max_within_50m=r["max_within_50m"], problems=L.verify())


def cmd_sweep(a):
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    base = Params()
    keys = list(spec)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(spec[k] if isinstance(spec[k], list)
                                                                    else [spec[k]] for k in keys))]
    jobs = [(replace(base, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}), s, a.out_dir)
            for c in combos for s in a.seeds]
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    with Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(_sweep_job, jobs)
    bad = [r for r in res if r.get("error") or r.get("problems")]
    print(f"[citygen] {len(res) - len(bad)} of {len(res)} layouts written to {a.out_dir}")
    for r in bad:
        print("  ", r["id"], r.get("error") or r.get("problems"))
    (Path(a.out_dir) / "sweep_index.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def cmd_trial(a):
    """Trial layouts for the chosen levels (select_levels' levels.json): one file per (level, seed) in layouts/,
    named <L>_<level>__<id>.json: a trial sheet lists them by level, and layout.resolve finds one by its id."""
    from citysim import STUDY_DIR
    levels = json.loads(Path(a.levels or STUDY_DIR / "levels.json").read_text(encoding="utf-8"))
    bad = 0
    for name, entry in levels["levels"].items():
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in entry["params"].items()}
        P = replace(Params(), **kw)
        for seed in a.seeds:
            L = citygen_one(P, seed)
            L.layout["trial_level"] = name
            L.layout["trial_level_name"] = entry["level"]
            out = LAYOUT_DIR / f"{name}_{entry['level']}__{L.id}.json"     # layout.resolve finds it by id
            L.save(out)
            problems = L.verify()
            bad += bool(problems)
            print(f"[citygen] {out.name}: alpha {L.layout['alpha_achieved']:.3f}, ell {L.layout['ell_mean']:.0f} m, "
                  f"{L.layout['n_buildings']} buildings" + (f"  PROBLEMS {problems}" if problems else ""))
    return 1 if bad else 0


def citygen_one(P, seed):
    return generate(P, seed)


def add_params(p):
    d = Params()
    p.add_argument("--mode", choices=["blocks", "scatter"], default=d.mode)
    p.add_argument("--p", type=float, default=d.p, help="mean block spacing (m)")
    p.add_argument("--s", type=float, default=d.s, help="street width (m)")
    p.add_argument("--alpha", type=float, default=d.alpha, help="target coverage over the arena")
    p.add_argument("--f", type=float, default=d.f, help="target block fill instead of alpha (>= 0)")
    p.add_argument("--g", type=float, default=d.g, help="facade gap (m)")
    p.add_argument("--size", type=float, default=d.size, help="horizontal scale of the study design")
    p.add_argument("--design", default="", help="catalogue design (default: the study design)")
    p.add_argument("--wc", type=float, default=d.w_c, help="goal occluder long side (m)")
    p.add_argument("--rg", type=float, default=d.r_g, help="goal ring radius (m)")
    p.add_argument("--nb", type=int, default=d.n_b, help="patrolled buildings per goal")
    p.add_argument("--ring-size", type=float, default=d.ring_size, help="ring building scale (< 0 = --size)")
    p.add_argument("--goals", type=int, default=d.goals)
    p.add_argument("--core", choices=["facade", "shell"], default=d.core,
                   help="shell: also keep avoidance cylinders --g-shell apart (a porous city)")
    p.add_argument("--g-shell", type=float, default=d.g_shell, help="least gap between avoidance cylinders (m)")
    p.add_argument("--mix", default="", help="a tile city whose building mix replaces the one design "
                                             "(city_obstacles_<city>.json)")
    p.add_argument("--place", choices=["rsa", "even"], default=d.place,
                   help="city buildings: rsa (the study's round robin) or even (every block to the same fill, spread)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("one", help="generate one layout")
    add_params(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help=f"default {LAYOUT_DIR}/<id>.json")
    p.set_defaults(func=cmd_one)
    p = sub.add_parser("sweep", help="generate a grid of layouts (JSON spec of Params field -> value or list)")
    p.add_argument("--spec", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out-dir", default=str(LAYOUT_DIR / "sweep~"))
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.set_defaults(func=cmd_sweep)
    p = sub.add_parser("trial", help="the trial layouts of levels.json (select_levels) into layouts/")
    p.add_argument("--levels", help="default experiment_results/density_study/levels.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104])
    p.set_defaults(func=cmd_trial)
    p = sub.add_parser("mixcity", help="a tile city's own buildings and density on Voronoi streets "
                                       f"-> layouts/{MIXCITY_PREFIX}<id>.json")
    add_params(p)
    p.add_argument("--city", default="DiamondCityWorld", help="the tile city (its city_obstacles_<city>.json)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", help=f"default {LAYOUT_DIR}/{MIXCITY_PREFIX}<id>.json")
    p.set_defaults(func=cmd_mixcity, size=1.0, alpha=None, place="even")
    a = ap.parse_args()
    try:
        sys.exit(a.func(a) or 0)
    except ValueError as e:
        sys.exit(f"[citygen] {e}")


if __name__ == "__main__":
    main()
