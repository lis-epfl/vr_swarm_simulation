"""PracticeBlockWorld's city: a piece of a block-shaped patch city round one goal (PracticeBlockWorldBuilder).

    python -B -m citysim.practicecity                     # -> layouts/PracticeBlock__<source id>_cut_b<goal>.json
    python -B -m citysim.practicecity --goal-block 72     # round another goal
    python -B -m citysim.practicecity --source FILE --out FILE

The practice scene is BlockShapedPatchCityWorld with a small city: one goal, its ring of neighbours, and the blocks
round the ring. The goal is not written into the layout. At Play `BlockGoals.Draw` draws it from the city, as it draws
the trial scene's three, with `goalsAtPlay` 1, so its hat and target walker change from session to session. The cut
makes that draw certain:
  - K = the goal G, its ring N1 (every block across a street from G) and N1's own neighbours N2.
  - A block on a street to a block outside K is not interior in the cut, so no N2 block qualifies, and every N1 block
    has an N2 neighbour, so no N1 block does either.
  - G and its ring are interior, and built in the source, so G is the one candidate BlockGoals has.
`main` asserts it (`candidates(cut) == [G]`).

Nothing is moved or reshaped. Every building, street and block keeps its source record (only `building` is
re-indexed, `interior` recomputed, and a street's side outside K made -1, edge), so the dressing's random streams,
keyed on block and street ids, dress each kept block as they would in the source city. Every street naming a block of
K is kept, so each block keeps all its sides: CityDressing.BuildBlocks clips its kerb from them.

The default source is the held-out seed 2 of BlockShapedPatchCityWorld's design (S6's), so participants never fly any
part of the trial city (seed 1). It is git-ignored; this draws it again (~24 min):
    python -B -m citysim.blockcity one --variant B --p 40 --s 18 --seed 2 --seeds patch --jitter 0.15 --patch 150 \
        --sight-tries 487
The goal it picks there, block 28, stands 191 m from the spawn, two blocks in from the city's west edge at x -365
where ParkRoad meets it, so no translation is needed: the edge row is the first thing the swarm flies round, then the
ring, then the goal.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import LAYOUT_DIR, blockcity  # noqa: E402
from citysim import geometry as geo  # noqa: E402
from citysim import layout as lay  # noqa: E402

GENERATOR = "practicecity"
GENERATOR_VERSION = "practice-1"
SCENE_PREFIX = "PracticeBlock__"      # layouts/PracticeBlock__<id>.json: PracticeBlockWorld's layouts
DEFAULT_SOURCE = LAYOUT_DIR / "sweep~" / "split" / "bbB_p40_s18_patch150_j0.15_sight487_seed2.json"
MIN_SPAWN_DISTANCE = 100.0            # m: BlockGoals.MinSpawnDistance
MIN_RING = 3                          # buildings: what ProceduralCityProbe asks of a drawn goal's ring


def neighbours(streets):
    """Block id -> the blocks across a street from it: two blocks are neighbours when a street names both
    (BlockGoals.Draw's rule, so a block meeting another only at a merged junction is one)."""
    ring = {}
    for s in streets:
        a, b = s["blocks"]
        if a < 0 or b < 0 or a == b:
            continue
        ring.setdefault(a, set()).add(b)
        ring.setdefault(b, set()).add(a)
    return ring


def _built(L, b):
    k = b["building"]
    return 0 <= k < len(L.boxes) and blockcity.is_prism(L.boxes[k])


def block_centre(L, b):
    """A block's centre: its building's (BlockGoals' Centre), else its kerb's centroid."""
    if _built(L, b):
        c = L.boxes[b["building"]]["c"]
        return np.array([c[0], c[2]])
    return blockcity.unflat(b["kerb_xz"]).mean(axis=0)


def candidates(L, blocks):
    """The blocks BlockGoals.Draw could draw a goal at, in the blocks list's order: interior and built, a ring of
    interior built blocks, and at least MIN_SPAWN_DISTANCE from the spawn."""
    usable = {b["id"] for b in blocks if b["interior"] and _built(L, b)}
    ring = neighbours(L.streets)
    sx, _, sz = L.layout["spawn"]
    out = []
    for b in blocks:
        k = b["id"]
        if k not in usable or not ring.get(k) or not ring[k] <= usable:
            continue
        c = L.boxes[b["building"]]["c"]
        if math.hypot(c[0] - sx, c[2] - sz) >= MIN_SPAWN_DISTANCE:
            out.append(k)
    return out


def _frame(L):
    """(spawn xz, the spawn's heading as a unit xz vector): spawn_yaw is a compass yaw, 90 = +x = east."""
    sx, _, sz = L.layout["spawn"]
    yaw = math.radians(L.layout["spawn_yaw"])
    return np.array([sx, sz]), np.array([math.sin(yaw), math.cos(yaw)])


def rings(ring, g):
    """(N1, N2) round block g."""
    n1 = set(ring[g])
    n2 = set().union(*(ring.get(j, set()) for j in n1)) - n1 - {g}
    return n1, n2


def approach(L, by_id, ring, g):
    """The N2 blocks between the spawn and g's ring (nearer along the spawn's heading than every ring block): the first
    buildings the swarm flies round on its way in."""
    spawn, ahead = _frame(L)
    n1, n2 = rings(ring, g)
    front = min(float((block_centre(L, by_id[j]) - spawn) @ ahead) for j in n1)
    return sorted(j for j in n2 if float((block_centre(L, by_id[j]) - spawn) @ ahead) < front)


def pick_goal(L, blocks):
    """The default goal and the runners-up: of the candidates whose ring is met only after a row on the city's edge (an
    N2 block before the ring that is not interior in the source, so the edge row is what the swarm flies round first),
    the one nearest the spawn's line of flight (ParkRoad's axis), then by block id."""
    by_id = {b["id"]: b for b in blocks}
    ring = neighbours(L.streets)
    spawn, ahead = _frame(L)
    side = np.array([ahead[1], -ahead[0]])
    ranked = []
    for k in candidates(L, blocks):
        if not any(not by_id[j]["interior"] for j in approach(L, by_id, ring, k)):
            continue
        ranked.append((abs(float((block_centre(L, by_id[k]) - spawn) @ side)), k))
    ranked.sort()
    if not ranked:
        raise SystemExit("[practicecity] no candidate goal sits two blocks in from the spawn's edge of the city; "
                         "name one with --goal-block")
    return ranked[0][1], [k for _, k in ranked[1:5]]


def cut(L, blocks, g):
    """(Layout, blocks) of the practice city round goal block g (see the module docstring)."""
    by_id = {b["id"]: b for b in blocks}
    ring = neighbours(L.streets)
    if g not in candidates(L, blocks):
        raise SystemExit(f"[practicecity] block {g} is not a goal BlockGoals could draw in {L.layout['id']}")
    n1, n2 = rings(ring, g)
    keep = {g} | n1 | n2

    old_boxes = sorted(by_id[k]["building"] for k in keep if by_id[k]["building"] >= 0)
    index = {old: new for new, old in enumerate(old_boxes)}
    boxes = [copy.deepcopy(L.boxes[i]) for i in old_boxes]

    streets = []
    for s in L.streets:
        if not any(k in keep for k in s["blocks"]):
            continue
        s = copy.deepcopy(s)
        sides = [k if k in keep else -1 for k in s["blocks"]]
        s["edge"] = bool(s["edge"] or -1 in sides)
        s["blocks"] = sides
        streets.append(s)
    edge = {k for s in streets if -1 in s["blocks"] for k in s["blocks"] if k >= 0}

    kept = []
    for b in blocks:
        if b["id"] not in keep:
            continue
        b = copy.deepcopy(b)
        b["building"] = index.get(b["building"], -1)
        b["interior"] = b["id"] not in edge
        kept.append(b)

    src = L.layout
    polys = [blockcity.building_poly(b) for b in boxes]
    pts = np.vstack([blockcity.unflat(s["xz"]) for s in streets])
    arena = [math.floor(pts[:, 0].min()), math.ceil(pts[:, 0].max()), math.floor(pts[:, 1].min()),
             math.ceil(pts[:, 1].max())]
    built = sum(abs(geo.area(p)) for p in polys)
    rec = dict(generator=GENERATOR, generator_version=GENERATOR_VERSION, id=f"{src['id']}_cut_b{g}")
    for key in ("seed", "mode", "variant", "p", "s", "design", "cyl_margin_m", "inset_m", "min_building_m"):
        if key in src:
            rec[key] = src[key]
    rec.update(
        n_blocks=len(kept), n_interior_blocks=sum(b["interior"] for b in kept), n_buildings=len(boxes),
        n_plazas=sum(1 for b in kept if b["building"] < 0),
        alpha_achieved=round(built / ((arena[1] - arena[0]) * (arena[3] - arena[2])), 4),
        arena=[float(v) for v in arena], extent=blockcity.extent(polys), spawn=src["spawn"],
        spawn_yaw=src["spawn_yaw"], goals=[], params=src.get("params"),
        practice=dict(source=src["id"], goal_block=g, ring_blocks=sorted(n1), outer_blocks=sorted(n2),
                      approach_blocks=approach(L, by_id, ring, g)))
    return lay.Layout(boxes=boxes, streets=streets, layout=rec, tiles=[]), kept


def check(L, blocks, source_blocks=None):
    """The cut's invariants; raises AssertionError. With the source's blocks, also that no kept block lost a street."""
    pr = L.layout["practice"]
    g = pr["goal_block"]
    assert candidates(L, blocks) == [g], f"BlockGoals could draw {candidates(L, blocks)}, not only block {g}"
    assert len(pr["ring_blocks"]) >= MIN_RING, f"the goal's ring has {len(pr['ring_blocks'])} buildings"
    assert pr["approach_blocks"], "nothing stands between the spawn and the ring"
    for b in blocks:
        assert closed(L.streets, b["id"]), f"block {b['id']} is not closed by its streets"
    if source_blocks is not None:
        kept = {b["id"] for b in blocks}
        assert kept == {g} | set(pr["ring_blocks"]) | set(pr["outer_blocks"])


def closed(streets, k):
    """Whether the streets naming block k close round it: every end point is shared by exactly two of them."""
    ends = {}
    for s in streets:
        if k not in s["blocks"]:
            continue
        xz = s["xz"]
        for x, z in ((xz[0], xz[1]), (xz[-2], xz[-1])):
            key = (round(x, 2), round(z, 2))
            ends[key] = ends.get(key, 0) + 1
    return bool(ends) and all(n == 2 for n in ends.values())


def path_for(L):
    return LAYOUT_DIR / f"{SCENE_PREFIX}{L.layout['id']}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="a block-shaped layout (default: %(default)s)")
    ap.add_argument("--goal-block", type=int, help="the goal's block id (default: picked, see pick_goal)")
    ap.add_argument("--out", help=f"file (default layouts/{SCENE_PREFIX}<id>.json)")
    a = ap.parse_args()
    src = Path(a.source)
    if not src.exists():
        raise SystemExit(f"[practicecity] no source layout at {src}; the module docstring says how to draw it")
    L0, blocks0 = blockcity.load(src)
    if a.goal_block is None:
        g, others = pick_goal(L0, blocks0)
    else:
        g, others = a.goal_block, []
    L, blocks = cut(L0, blocks0, g)
    check(L, blocks, blocks0)
    path = blockcity.save(L, blocks, Path(a.out) if a.out else path_for(L))

    by_id = {b["id"]: b for b in blocks}
    spawn, _ = _frame(L)
    c = block_centre(L, by_id[g])
    pr, r = L.layout["practice"], L.layout
    print(f"[practicecity] goal: block {g} ({L.boxes[by_id[g]['building']]['name']}) at ({c[0]:.0f}, {c[1]:.0f}), "
          f"{np.linalg.norm(c - spawn):.0f} m from the spawn; ring {len(pr['ring_blocks'])} blocks, round it "
          f"{len(pr['outer_blocks'])} more, {len(pr['approach_blocks'])} of them before the ring "
          f"{pr['approach_blocks']}")
    if others:
        print(f"[practicecity] runners-up (--goal-block): {others}")
    print(f"[practicecity] {path.name}: {r['n_buildings']} buildings ({r['n_plazas']} plazas), {len(L.streets)} "
          f"streets, arena {r['arena']}")
    print(f"LAYOUT_PATH={path}")


if __name__ == "__main__":
    main()
