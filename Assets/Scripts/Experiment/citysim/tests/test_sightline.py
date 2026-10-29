"""blockcity's patch seeding and the sight-line pass (citysim.sightline)."""
import sys

sys.dont_write_bytecode = True

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import blockcity, sightline  # noqa: E402


def _views(L):
    polys = [blockcity.building_poly(b) for b in L.boxes]
    mids = sightline.street_midpoints(L.streets, polys, blockcity.ARENA)
    best, _ = sightline.measure(polys, mids, blockcity.ARENA)
    return best


def test_hex_streets_hold_straight_channels_and_patches_break_them():
    """A hex lattice's zigzag streets leave a straight channel across the city in each of its three directions; patches
    of lattice turned against each other end every channel at a patch's edge, and keep more sides than S4's seeds."""
    hexed, _ = blockcity.generate(blockcity.Params("B", 40, 18, 1, seeds="hex", jitter=0.15))
    P = blockcity.Params("B", 40, 18, 1, seeds="patch")
    assert P.ident() == "bbB_p40_s18_patch150_j0.15_seed1"
    patched, blocks = blockcity.generate(P)
    assert patched.layout["params"]["seeds"] == "patch" and patched.layout["params"]["patch"] == 150.0
    h, p = _views(hexed), _views(patched)
    assert (h > 400).mean() > 0.9 and (p > 400).mean() < 0.15 and np.median(p) < 0.6 * np.median(h)
    assert blockcity.street_stats(patched, blocks)["sides_mean"] > 4.8


def test_sight_pass_shortens_the_views_and_holds_the_rest():
    """A few tries already shorten the long views, and no kept move costs sides, plazas or street clearance."""
    P0 = blockcity.Params("B", 40, 18, 1, seeds="patch")
    P = replace(P0, sight_tries=12)
    assert P.ident() == "bbB_p40_s18_patch150_j0.15_sight12_seed1" and not P.default_streets
    (L0, b0), (L1, b1) = blockcity.generate(P0), blockcity.generate(P)
    assert L1.layout["params"]["sight_tries"] == 12

    def excess(L):
        return np.maximum(_views(L) - sightline.FLOOR_M, 0).sum()

    assert excess(L1) < excess(L0)
    s0, s1 = blockcity.street_stats(L0, b0), blockcity.street_stats(L1, b1)
    assert s1["sides_mean"] >= s0["sides_mean"]
    assert L1.layout["n_plazas"] <= L0.layout["n_plazas"]
    assert s1["cyl_closed"] <= s0["cyl_closed"] and s1["cyl_narrow"] <= s0["cyl_narrow"]
