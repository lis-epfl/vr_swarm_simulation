"""The transit study's pieces: the pilots' paths (routing), the porous generator, moving visibility (transit), goal
discovery, and the scripted pilot's command metrics."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import camera, catalogue, discovery, layout as lay, routing, transit  # noqa: E402
from citysim.los import Footprints  # noqa: E402


def _boxes(centres, size=2.0, yaw=0):
    d = catalogue.study_design()
    out = []
    for k, (x, z) in enumerate(centres):
        b = lay.make_box(f"t{k}", "t", x, z, d, size, yaw)
        out.append((b["tile"], np.array(b["c"], float), np.array(b["ax"], float)))
    return out


def _R(size=2.0):
    hx, hz = catalogue.study_design().footprint(size, 0)
    return math.hypot(hx, hz)


# ============================================================================= routing

def test_straight_in_an_empty_city():
    cyl = routing.Cylinders([])
    for kind in ("single", "swarm"):
        wps = routing.plan(cyl, kind, (0.0, 0.0), (300.0, 40.0), w_pass=6.0)
        assert len(wps) == 1 and np.allclose(wps[0], [300.0, 40.0])


def test_single_steers_round_a_lone_building_the_swarm_holds_through_it():
    cyl = routing.Cylinders(_boxes([(0.0, 0.0)]))
    start, goal = np.array([-150.0, 0.0]), np.array([150.0, 0.0])
    swarm = routing.plan(cyl, "swarm", start, goal, w_pass=6.0)
    assert len(swarm) == 1                                  # an isolated building is left to the formation
    single = routing.plan(cyl, "single", start, goal)
    assert len(single) >= 2
    pts = np.vstack([start] + single)
    s = np.linspace(0, 1, 200)
    seg = np.vstack([a + s[:, None] * (b - a) for a, b in zip(pts[:-1], pts[1:])])
    clear = np.hypot(seg[:, 0] - cyl.c[0, 0], seg[:, 1] - cyl.c[0, 1]).min()
    assert clear >= cyl.R[0] + routing.SINGLE_CLEARANCE_M - 1.5      # the grid's half cell and the pulled corners
    assert routing.path_length(start, single) < 1.1 * 300.0


def test_walls_close_narrow_gaps_and_leave_wide_ones():
    R = _R()

    def wall(g, k_side):
        """A wall across x = 0: k_side buildings either side of one gap of g m between avoidance cylinders at z = 0,
        neighbours' cylinders overlapping."""
        z = R + g / 2
        cs = []
        for k in range(k_side):
            cs += [(0.0, z + k * 1.75 * R), (0.0, -z - k * 1.75 * R)]
        return routing.Cylinders(_boxes(cs))

    start, goal = np.array([-120.0, 0.0]), np.array([120.0, 0.0])
    straight = 240.0
    r = routing.formation_radius(1.0)

    def length(g, k):
        return routing.path_length(start, routing.plan(wall(g, k), "swarm", start, goal, w_pass=6.0, radius=r))

    # a wide gap is flown straight through
    assert length(10.0, 6) < straight + 2.0
    # a narrow one presses the formation against the walls either side: round a short wall, which is cheaper ...
    assert length(4.0, 2) > straight + 5.0
    # ... and through a long one, where every way round costs more than the squeeze
    assert length(4.0, 14) < straight + 2.0
    # narrower than G_HARD_M: closed
    assert length(1.0, 6) > 1.2 * straight
    member, _ = routing.walls(wall(10.0, 6), 6.0)
    assert member.all()


def test_street_route_is_reexported():
    from citysim import dynamic
    assert dynamic.route is routing.route and dynamic.street_graph is routing.street_graph


# ============================================================================= the porous generator

def test_shell_core_keeps_cylinders_apart_and_is_deterministic():
    from dataclasses import replace
    from citysim import citygen
    P = replace(citygen.Params(), p=70.0, s=12.0, alpha=0.06, w_c=7.0, r_g=26.0, ring_size=0.9, core="shell",
                g_shell=6.0)
    L1, L2 = citygen.generate(P, 3), citygen.generate(P, 3)
    assert [b["c"] for b in L1.boxes] == [b["c"] for b in L2.boxes]
    assert "_gs6_" in L1.id and L1.layout["core"] == "shell"
    r = L1.rects([ "city"])
    c, R = r[:, :2], np.hypot(r[:, 2], r[:, 3])
    D = np.linalg.norm(c[:, None] - c[None], axis=-1) - R[:, None] - R[None]
    np.fill_diagonal(D, np.inf)
    assert D.min() >= 6.0 - 1e-6
    assert not L1.verify()


def test_facade_core_ident_and_params_unchanged():
    from citysim import citygen
    P = citygen.Params()
    assert "_gs" not in P.ident(0)
    L = citygen.generate(citygen.Params(p=90.83, s=15.0, alpha=0.02, w_c=7.0, r_g=26.0, ring_size=0.9), 1)
    assert "core" not in L.layout["params"] and "g_shell" not in L.layout["params"] and "core" not in L.layout


# ============================================================================= moving visibility

def _straight_record(n_cams=1, speed=7.0, length=600.0, single=True, spread=0.0):
    T = int(length / speed / transit.SAMPLE_S)
    t = np.arange(T) * transit.SAMPLE_S
    x = -length / 2 + speed * t
    cen = np.c_[x, np.zeros(T)]
    off = np.linspace(-spread, spread, n_cams) if n_cams > 1 else np.zeros(1)
    pos = np.stack([np.c_[x + camera.CAMERA_FORWARD_M, np.full(T, o)] for o in off], 1)
    yaw = np.full((T, n_cams), 90.0)
    rec = transit.FlightRecord(t=t, phase=np.zeros(T, int), leg=np.zeros(T, int), centroid=cen,
                               P=pos.copy(), body_yaw=np.full(T, 90.0), cam_pos=pos, cam_yaw=yaw,
                               cam_valid=np.ones((T, n_cams), bool), cmd=np.zeros((0, 5)),
                               meta=dict(interface="single" if single else "swarm"))
    return rec


def test_sweep_width_of_one_forward_camera_is_analytic():
    fp = Footprints([])
    rec = _straight_record()
    cells = transit.Cells.around(fp, [rec.centroid], 90.0)
    h, p = 19.5, -15.0
    s = transit.score(rec, fp, cells, h, p, omni=False)
    dmax = math.sqrt(camera.R_ID_M ** 2 - (h - camera.HAT_HEIGHT_M) ** 2)
    expect = 2 * dmax * math.sin(math.radians(camera.HFOV_DEG / 2))
    assert abs(s["sweep_any_r40_m"] - expect) < 4.0, (s["sweep_any_r40_m"], expect)
    assert s["sweep_head_r40_m"] == s["sweep_any_r40_m"]        # the single drone's feed is in front


def test_look_behind_counts_what_the_centroid_cannot_see():
    # a building beside the track: two cameras 30 m either side see past it where the centroid camera cannot
    fp = Footprints(_boxes([(0.0, 22.0)]))
    rec = _straight_record(n_cams=2, single=False, spread=15.0)
    rec.cam_yaw[:] = np.array([45.0, 135.0])[None]
    cells = transit.Cells.around(fp, [rec.centroid], 90.0)
    s = transit.score(rec, fp, cells, 19.5, -15.0, omni=True)
    assert s["behind_corner_r40_frac"] >= 0.0 and s["behind_reach_r40_frac"] > 0.0
    s1 = transit.score(_straight_record(), fp, cells, 19.5, -15.0)
    assert "behind_corner_r40_frac" not in s1                    # one camera: no look-behind by construction


def test_lateral_offsets_sign_and_foot():
    track = np.array([[0.0, 0.0], [100.0, 0.0]])
    off, arc, L = transit.lateral_offsets(track, np.array([[50.0, -10.0], [50.0, 10.0]]))
    assert L == 100.0 and np.allclose(arc, [50.0, 50.0])
    assert off[0] * off[1] < 0 and np.allclose(np.abs(off), 10.0)


# ============================================================================= discovery

def test_loop_points_carry_the_walkers():
    rects = [(0.0, 0.0, 5.0, 4.0), (30.0, 0.0, 5.0, 4.0)]
    pts, w = discovery.loop_points(rects)
    assert abs(w.sum() - 16.0) < 1e-9
    d = np.abs(pts - np.array([[0.0, 0.0]])) - np.array([5.0, 4.0])
    assert (np.max(d[: len(pts) // 2], axis=1) > 0).all()        # every point outside the facade


def test_track_expected_by_range():
    fp = Footprints([])
    pts, w = discovery.loop_points([(0.0, 0.0, 5.0, 4.0)])
    for dist, seen in ((60.0, True), (150.0, False)):
        pos = np.array([[[-dist, 0.0]]])
        e_any, e_head = discovery.track_expected(fp, pos, np.array([[90.0]]), np.ones((1, 1), bool), np.array([90.0]),
                                                 12.0, -10.0, pts, w, r_goal=discovery.R_GOAL_M, single=True)
        assert (e_any[0] >= discovery.SEEN_WALKERS) == seen
        assert e_head[0] == e_any[0]


# ============================================================================= commands

def test_command_metrics_count_turns():
    t = np.arange(0, 60, 0.1)
    # a straight leg, then one 90-degree turn held
    hd = np.where(t < 30, 90.0, 0.0)
    u = np.c_[np.sin(np.radians(hd)), np.cos(np.radians(hd))]
    rec = _straight_record()
    rec.cmd = np.c_[t, u, hd, np.zeros(len(t))]
    m = transit.command_metrics(rec, [(0.0, 60.0)], 300.0)
    assert m["adjustments"] in (1, 2)
    assert 0.6 < m["cmd_straightness"] < 0.8                     # two equal perpendicular legs: 1/sqrt(2)


def test_run_jobs_keeps_order():
    from citysim.parallel import run_jobs
    out = run_jobs(math.sqrt, [4.0, 9.0, 16.0, 25.0], "test", processes=2, stall_s=60.0)
    assert out == [2.0, 3.0, 4.0, 5.0]
