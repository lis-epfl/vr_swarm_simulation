"""PracticeBlockWorld's city (citysim.practicecity): BlockGoals can draw only the intended goal from it, every block is
closed by its streets, something stands between the spawn and the ring, and the committed file is what the generator
draws from its source (when the git-ignored source is present)."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import LAYOUT_DIR, blockcity, practicecity  # noqa: E402


def _committed():
    files = sorted(LAYOUT_DIR.glob(f"{practicecity.SCENE_PREFIX}*.json"))
    if not files:
        conftest.skip("no PracticeBlock layout in layouts/")
    return files


def test_the_goal_is_the_only_one_blockgoals_can_draw():
    for f in _committed():
        L, blocks = blockcity.load(f)
        practicecity.check(L, blocks)
        g = L.layout["practice"]["goal_block"]
        building = next(b["building"] for b in blocks if b["id"] == g)
        c, (sx, _, sz) = L.boxes[building]["c"], L.layout["spawn"]
        assert math.hypot(c[0] - sx, c[2] - sz) >= practicecity.MIN_SPAWN_DISTANCE


def test_a_lost_street_opens_its_blocks():
    L, blocks = blockcity.load(_committed()[0])
    g = L.layout["practice"]["goal_block"]
    side = next(i for i, s in enumerate(L.streets) if g in s["blocks"])
    others = L.streets[:side] + L.streets[side + 1:]
    assert practicecity.closed(L.streets, g) and not practicecity.closed(others, g)


def test_regenerates_byte_for_byte(tmp_path):
    for f in _committed():
        L, _ = blockcity.load(f)
        src = practicecity.DEFAULT_SOURCE.parent / f"{L.layout['practice']['source']}.json"
        if not src.exists():
            conftest.skip(f"the source {src.name} is git-ignored and not drawn here (practicecity's docstring)")
        L0, blocks0 = blockcity.load(src)
        cut, blocks = practicecity.cut(L0, blocks0, L.layout["practice"]["goal_block"])
        out = blockcity.save(cut, blocks, Path(tmp_path) / f.name)
        assert out.read_bytes() == f.read_bytes()
