"""S4 Phase 3's look-down measure (citysim.lookdown): the footpath ring stands its offset outside a footprint, an exit's
end targets are the footpath of the building ahead facing the far junction (or a plaza's lawn), and each level is the
right quantile of its points' first sights."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import blockcity, lookdown as LD, split_probe as S, streetnet  # noqa: E402
from citysim import geometry as geo  # noqa: E402
from citysim.polyfoot import ConvexFootprints  # noqa: E402


def test_footpath_ring_stands_off_the_footprint():
    th = 0.4
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    poly = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]) @ rot.T + [5.0, -3.0]
    ring = LD.footpath_ring(poly)
    off, step = LD.FOOTPATH_OFF_M, LD.FOOTPATH_STEP_M
    assert np.abs(ConvexFootprints([poly]).distance(ring) - off).max() < 1e-9
    gaps = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1)
    assert gaps.max() <= step + 1e-9 and gaps.min() > 0.1
    assert abs(len(ring) - (2 * (20 + 10) + 2 * math.pi * off) / step) <= 8


def _city(p, s, seed):
    L, blocks = blockcity.generate(blockcity.Params("R", p, s, seed))
    net = streetnet.Network(L, blocks, blockcity.footprints(L))
    return LD.city_targets(dict(L=L, net=net)), net


def _exits(C, net):
    for B in net.interior_blocks():
        for sid in B["radiating"]:
            st = net.streets[sid]
            if st["interior"]:
                V = st["u"] if st["u"] in B["corners"] else st["v"]
                tg, kind = LD.end_targets(C, sid, V)
                if tg is not None:
                    F = net.other(sid, V)
                    yield tg, kind, net.blocks[net.block_ahead(F, sid)], net.jpos[F]


def test_end_targets_face_the_far_junction():
    C, net = _city(40, 18, 0)
    n = 0
    for tg, kind, ahead, pF in _exits(C, net):
        if kind != "building":
            continue
        ring = C["ld_rings"][ahead["building"]]
        d, dr = np.hypot(*(tg - pF).T), np.hypot(*(ring - pF).T)
        assert abs(d.min() - dr.min()) < 1e-9                       # the ring's point nearest F is a target
        assert d.max() <= dr.min() + LD.END_REACH_M + 1e-9           # and none is beyond the reach
        assert len(tg) >= 3
        n += 1
    assert n > 50


def test_a_plaza_ahead_is_scored_on_its_lawn():
    C, net = _city(40, 24, 0)                                         # half the interior blocks are plazas
    n = 0
    for tg, kind, ahead, pF in _exits(C, net):
        if kind == "plaza":
            assert ahead["building"] < 0 and geo.contains(ahead["kerb"], tg).all()
            n += 1
    assert n > 5


def test_level_times_take_each_levels_quantile():
    first = np.array([1.0, 2.0, 3.0, np.inf, 5.0, 6.0, np.inf, np.inf])
    info = [dict(ent=np.array([0, 1, 2, 3, 4]), end=np.array([4, 5]), tgt=np.array([5, 6, 7])),
            dict(ent=np.array([0]), end=np.array([1]), tgt=np.zeros(0, int))]

    def q(idx, frac):
        f = np.sort(first[idx])
        return f[max(1, math.ceil(frac * len(f))) - 1]

    a, b = LD.level_times(first, info)
    assert a["ent"] == q(info[0]["ent"], S.PREVIEW_FRAC) and a["end"] == q(info[0]["end"], S.PREVIEW_FRAC)
    assert a["look"] == q(info[0]["tgt"], LD.END_FRACTION) == math.inf        # one of three targets seen: not half
    assert math.isnan(b["look"])                                             # no targets: not scored
