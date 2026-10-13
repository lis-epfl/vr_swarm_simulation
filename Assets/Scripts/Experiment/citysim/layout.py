"""A generated city: the layout file the generator writes, Unity builds (ProceduralCityBuilder) and every analysis
tool reads.

The file IS a city_obstacles file -- `boxes` with name / type / tile / c / ax exactly as CityObstacleExport writes
them, and `tiles: []` -- so analyse.load_city, swarm_replica.City and bench.Footprints read it unchanged, plus:

    boxes[k]   design, size (horizontal scale of the design), yaw (0 | 90), role ("city" | "goal_centre" |
               "goal_ring"), goal (its goal's index, -1 for city buildings), patrol (walkers circle it): what
               the Unity builder needs to place the same building, which c / ax alone do not say
    streets    [{id, polyline: [[x, z], ...], xz: [x0, z0, x1, z1, ...], width, edge, blocks}]: the street
               centrelines (edge = on the arena's boundary, open on the outside); xz is the polyline flattened,
               because Unity's JsonUtility cannot read nested arrays
    layout     the generator's record: generator_version, id, seed, mode, design, size, p, s, f, g,
               alpha_target, alpha_achieved, ell_mean, ell_theta, arena, extent, spawn, spawn_yaw, goals[...],
               max_within_50m, min_street_width, narrow_street_frac_by_spread, ... (see citygen.generate)

A box's three half-axis vectors follow Unity: a pack building is authored +Z up, so the builder gives it the
rotation Euler(-90, yaw, 0), under which the collider's local X maps to (cos yaw, 0, -sin yaw), local Y to
(-sin yaw, 0, -cos yaw) and local Z to world up. `ax` = [local X * w/2, local Y * d/2, up * h/2] in world
terms, w and d being the design's footprint times `size`. With yaw restricted to 0 and 90 every footprint is
axis-aligned, which is what OlfatiSaber's bounds-circumradius cylinders and WalkerPatrol's AABB loops assume.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import geometry as geo  # noqa: E402

GENERATOR_VERSION = "2"   # 2: streets carry the flat xz list the Unity builder reads
TERRAIN_BOUNDS = (-1000.0, 1000.0, -1000.0, 1000.0)   # the four kept 1000 m Terrain tiles: no ceiling outside
OVERLAP_BUFFER = 64            # OlfatiSaber.overlapBuffer
OVERLAP_LIMIT = 56             # what a layout may put within one drone's 50 m query, leaving headroom
QUERY_RADIUS_M = 50.0          # r0_obs * scaleFactor in the city scenes
SPAWN_CLEAR_M = 5.0
GOAL_MIN_SEPARATION_M = 150.0
GOAL_MIN_SPAWN_DIST_M = 100.0
BUILDING_BASE_M = 0.0          # the Terrain sits at y = -0.017; buildings stand on y = 0
BUILDING_HEIGHT_M = 50.0


def box_axes(w, d, h, yaw_deg):
    """Unity half-axis vectors (3, 3) of a pack building's collider at yaw 0 / 90 (see the module docstring)."""
    y = math.radians(yaw_deg)
    c, s = round(math.cos(y), 12), round(math.sin(y), 12)
    return np.array([[c * w / 2, 0.0, -s * w / 2],
                     [-s * d / 2, 0.0, -c * d / 2],
                     [0.0, h / 2, 0.0]])


def make_box(name, tile, cx, cz, design, size, yaw, role="city", goal=-1, patrol=False,
             base=BUILDING_BASE_M, height=BUILDING_HEIGHT_M):
    """One building as a layout box. `design` is a catalogue.Design."""
    w, d = design.w * size, design.d * size
    ax = box_axes(w, d, height, yaw)
    return dict(name=name, type="BoxCollider", tile=tile,
                c=[round(float(cx), 4), round(base + height / 2, 4), round(float(cz), 4)],
                ax=[[round(float(v), 6) for v in row] for row in ax],
                design=design.design, size=round(float(size), 6), yaw=int(yaw) % 360, role=role, goal=int(goal),
                patrol=bool(patrol))


def box_rect(b):
    """(cx, cz, hx, hz) of a box's axis-aligned footprint."""
    ax = np.asarray(b["ax"], float)
    flat = ax[np.argsort(np.abs(ax[:, 1]) / np.maximum(np.linalg.norm(ax, axis=1), 1e-12))[:2]]
    return (float(b["c"][0]), float(b["c"][2]), float(np.abs(flat[:, 0]).sum()), float(np.abs(flat[:, 2]).sum()))


@dataclass
class Layout:
    boxes: list
    streets: list = field(default_factory=list)
    layout: dict = field(default_factory=dict)
    tiles: list = field(default_factory=list)

    # ------------------------------------------------------------------ I/O

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(boxes=d["boxes"], streets=d.get("streets", []), layout=d.get("layout", {}),
                   tiles=d.get("tiles", []))

    def save(self, path):
        """One box / street per line, so a diff of two layouts reads."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        boxes = ",\n".join(json.dumps(b, separators=(",", ":")) for b in self.boxes)
        streets = ",\n".join(json.dumps(s, separators=(",", ":")) for s in self.streets)
        text = ('{"boxes":[\n' + boxes + '\n],\n"tiles":' + json.dumps(self.tiles) + ',\n"streets":[\n' + streets +
                '\n],\n"layout":' + json.dumps(self.layout, indent=1) + "\n}\n")
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    # ------------------------------------------------------------------ views

    @property
    def id(self):
        return self.layout.get("id", "")

    def boxes_as_tuples(self):
        """[(tile, centre, axes)] as analyse.load_city returns a city."""
        return [(b["tile"], np.array(b["c"], float), np.array(b["ax"], float)) for b in self.boxes]

    def rects(self, roles=None):
        sel = [b for b in self.boxes if roles is None or b.get("role", "city") in roles]
        return np.array([box_rect(b) for b in sel], float).reshape(-1, 4)

    def goals(self):
        return self.layout.get("goals", [])

    # ------------------------------------------------------------------ checks

    def verify(self, tol=1e-3):
        """Every rule a layout must keep, as a list of problems (empty = fine)."""
        problems = []
        L = self.layout
        rects = self.rects()
        roles = [b.get("role", "city") for b in self.boxes]
        if len(rects):
            # overlaps
            for i in range(len(rects)):
                gap = geo.rect_gap(rects[i], rects[i + 1:])
                bad = np.flatnonzero(gap < -tol)
                if len(bad):
                    problems.append(f"{self.boxes[i]['name']} overlaps {self.boxes[i + 1 + bad[0]]['name']} "
                                    f"({-gap[bad[0]]:.2f} m)")
                    break
            # inside the arena and the terrain
            for name, bounds in (("arena", L.get("arena")), ("terrain", TERRAIN_BOUNDS)):
                if bounds is None:
                    continue
                x0, x1, z0, z1 = bounds
                out = ((rects[:, 0] - rects[:, 2] < x0 - tol) | (rects[:, 0] + rects[:, 2] > x1 + tol) |
                       (rects[:, 1] - rects[:, 3] < z0 - tol) | (rects[:, 1] + rects[:, 3] > z1 + tol))
                if out.any():
                    problems.append(f"{int(out.sum())} building(s) outside the {name}")
            # coverage as recorded
            if L.get("arena") is not None and "alpha_achieved" in L:
                x0, x1, z0, z1 = L["arena"]
                alpha = geo.rect_area(rects).sum() / ((x1 - x0) * (z1 - z0))
                if abs(alpha - L["alpha_achieved"]) > 5e-4:
                    problems.append(f"alpha {alpha:.4f} differs from the recorded {L['alpha_achieved']:.4f}")
        # the swarm's overlap buffer
        if L.get("max_within_50m", 0) > OVERLAP_LIMIT:
            problems.append(f"{L['max_within_50m']} buildings within {QUERY_RADIUS_M:g} m of one point "
                            f"(OlfatiSaber's buffer holds {OVERLAP_BUFFER}; the limit is {OVERLAP_LIMIT})")
        # spawn
        sp = L.get("spawn")
        if sp is not None and len(rects):
            dmin = geo.rect_point_distance(rects, [[sp[0], sp[2]]]).min()
            if dmin < SPAWN_CLEAR_M:
                problems.append(f"a building stands {dmin:.1f} m from the spawn (needs {SPAWN_CLEAR_M:g})")
        # streets: no city building within half a street width of a centreline
        if L.get("mode") == "blocks" and len(rects):
            city = rects[[r == "city" for r in roles]]
            worst = street_clearance(self.streets, city, step=5.0, skip_edges=True)
            if worst is not None and worst < min(s["width"] for s in self.streets) / 2 - 0.05:
                problems.append(f"a city building stands {worst:.2f} m from a street centreline "
                                f"(half the street width is {min(s['width'] for s in self.streets) / 2:.2f})")
        # goals
        goals = self.goals()
        centres = np.array([g["centre"] for g in goals], float).reshape(-1, 2)
        for i in range(len(goals)):
            for j in range(i + 1, len(goals)):
                dd = float(np.linalg.norm(centres[i] - centres[j]))
                if dd < GOAL_MIN_SEPARATION_M - tol:
                    problems.append(f"goals {i} and {j} are {dd:.0f} m apart (< {GOAL_MIN_SEPARATION_M:g})")
        for g in goals:
            if sp is not None:
                dd = math.hypot(g["centre"][0] - sp[0], g["centre"][1] - sp[2])
                if dd < GOAL_MIN_SPAWN_DIST_M - tol:
                    problems.append(f"goal {g['index']} is {dd:.0f} m from the spawn (< {GOAL_MIN_SPAWN_DIST_M:g})")
            own = [k for k, b in enumerate(self.boxes) if b.get("goal", -1) == g["index"]]
            centre = [k for k in own if self.boxes[k]["role"] == "goal_centre"]
            ring = [k for k in own if self.boxes[k]["role"] == "goal_ring"]
            if len(centre) != 1 or len(ring) != g["n_b"]:
                problems.append(f"goal {g['index']}: {len(centre)} centre and {len(ring)} ring buildings "
                                f"(expected 1 and {g['n_b']})")
                continue
            c = rects[centre[0]]
            if abs(2 * max(c[2], c[3]) - g["w_c"]) > 0.05:
                problems.append(f"goal {g['index']}: occluder is {2 * max(c[2], c[3]):.2f} m, recorded w_c {g['w_c']}")
            rr = np.hypot(rects[ring, 0] - g["centre"][0], rects[ring, 1] - g["centre"][1])
            if np.abs(rr - g["r_g"]).max() > 0.05:
                problems.append(f"goal {g['index']}: ring buildings at {rr.min():.1f}-{rr.max():.1f} m, r_g {g['r_g']}")
            others = [k for k in range(len(rects)) if k not in own]
            if others:
                d_other = geo.rect_point_distance(rects[others], [g["centre"]]).min()
                if d_other < g["clear_radius"] - tol:
                    problems.append(f"goal {g['index']}: a city building stands {d_other:.1f} m from its centre "
                                    f"(cleared to {g['clear_radius']:.1f})")
            nonocc = [k for k in range(len(rects)) if k != centre[0]]
            cg = geo.rect_point_distance(rects[nonocc], [g["centre"]]).min()
            if abs(cg - g["c_g"]) > 0.05:
                problems.append(f"goal {g['index']}: clear radius {cg:.2f} m, recorded c_g {g['c_g']}")
        return problems


def street_clearance(streets, rects, step=5.0, skip_edges=True):
    """Smallest distance from a sampled point of a street centreline to a building, or None."""
    pts = street_points(streets, step, skip_edges)
    if len(pts) == 0 or len(rects) == 0:
        return None
    best = math.inf
    for k in range(0, len(pts), 2000):
        best = min(best, float(geo.rect_point_distance(rects, pts[k:k + 2000]).min()))
    return best


def street_points(streets, step=5.0, skip_edges=True):
    out = []
    for s in streets:
        if skip_edges and s.get("edge"):
            continue
        p = np.asarray(s["polyline"], float)
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
            t = (np.arange(n) + 0.5) / n
            out.append(a + t[:, None] * (b - a))
    return np.vstack(out) if out else np.zeros((0, 2))


def load(path):
    return Layout.load(path)


def resolve(name_or_path):
    """A layout by path, or by id: Assets/Scripts/Experiment/layouts/<id>.json (the trial layouts, committed), then
    layouts/sweep~/<sweep>/<id>.json (generated by vissim / citygen sweeps; git-ignored, and a folder ending in '~'
    is one Unity's asset database skips)."""
    from citysim import LAYOUT_DIR
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    cand = [LAYOUT_DIR / f"{name_or_path}.json", LAYOUT_DIR / name_or_path]
    cand += sorted(LAYOUT_DIR.glob(f"*__{name_or_path}.json"))            # trial layouts: <L>_<level>__<id>.json
    cand += sorted((LAYOUT_DIR / "sweep~").glob(f"*/{name_or_path}.json"))
    cand += [LAYOUT_DIR / "sweep~" / f"{name_or_path}.json"]
    for c in cand:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"no layout {name_or_path!r} (looked in {LAYOUT_DIR} and its sweep~ folders)")
