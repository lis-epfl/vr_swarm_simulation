"""The search study's pieces: one-instant discovery, the feed latch, the intent chain and its mask, the head process,
the survival summaries, track statistics, the fitted pilot's segmentation, and a short search flight."""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import catalogue, discovery, display, layout as lay, pilotfit, pilots, routing, search  # noqa: E402
from citysim.los import Footprints  # noqa: E402

ARENA = (-200.0, 200.0, -200.0, 200.0)


def _boxes(centres, size=2.0, yaw=0):
    d = catalogue.study_design()
    out = []
    for k, (x, z) in enumerate(centres):
        b = lay.make_box(f"t{k}", "t", x, z, d, size, yaw)
        out.append((b["tile"], np.array(b["c"], float), np.array(b["ax"], float)))
    return out


def _ring(cx=0.0, cz=0.0, r=26.0, n=8, h=5.0):
    return [(cx + r * math.sin(2 * math.pi * k / n), cz + r * math.cos(2 * math.pi * k / n), h, h) for k in range(n)]


def _model():
    """A minimal fitted model, the shape pilotfit writes."""
    return dict(conditions=dict(
        SingleDrone=dict(run=dict(mean_m=90.0), turn=dict(angles_deg=[30.0, -45.0, 90.0], rate_p90=66.0),
                         pause=dict(durations_s=[2.0, 4.0], per_turn=0.2), stick=dict(p50=1.0)),
        Swarm=dict(run=dict(mean_m=75.0), turn=dict(angles_deg=[25.0, -30.0], rate_p90=89.0),
                   pause=dict(durations_s=[3.0], per_turn=0.3), stick=dict(p50=1.0),
                   head=dict(dwells=[[0.0, 2.0], [-40.0, 1.0], [20.0, 0.5]], move_speed_deg_s=60.0))))


# ============================================================================= discovery at one instant

def test_sample_expected_is_track_expected_one_sample_at_a_time():
    fp = Footprints.from_rects(_ring() + [(0.0, 0.0, 3.5, 3.5), (60.0, -20.0, 6.0, 4.0)])
    pts, w = discovery.loop_points(_ring())
    T, K = 40, 3
    t = np.linspace(0, 1, T)[:, None]
    pos = np.stack([-150 + 300 * t + 4 * np.arange(K)[None], 40 + 0 * t + 3 * np.arange(K)[None]], -1)
    yaw = np.full((T, K), 120.0) + np.arange(K)[None] * 30
    valid = np.ones((T, K), bool)
    body = np.full(T, 90.0)
    a, h = discovery.track_expected(fp, pos, yaw, valid, body, 19.5, -15.0, pts, w, r_goal=80.0, head_yaw=-20.0)
    assert (a >= 1).any(), "the track should see the goal at some point"
    for ti in range(T):
        A, H = discovery.sample_expected(fp, pos[ti], yaw[ti], body[ti], 19.5, -15.0, pts, w, (80.0, 200.0),
                                         (-20.0, 0.0))
        assert abs(A[0] - a[ti]) < 1e-9 and abs(H[0, 0] - h[ti]) < 1e-9
        assert A[1] >= A[0] - 1e-9                          # a longer range sees at least as much
        assert (H <= A[:, None] + 1e-9).all()               # the head basis is a subset of any


def test_omni_without_sight_lines_sees_everything_in_range():
    pts, w = discovery.loop_points(_ring())
    fp = Footprints.from_rects([(-60.0, 0.0, 10.0, 30.0)])      # a wall between the camera and the goal
    cam = np.array([-100.0, 0.0])
    blocked = discovery.sample_expected(fp, cam, [0.0], 0.0, 19.5, -15.0, pts, w, (200.0,), hfov=360.0,
                                        single=True)[0][0]
    free = discovery.sample_expected(fp, cam, [0.0], 0.0, 19.5, -15.0, pts, w, (200.0,), hfov=360.0, los=False,
                                     single=True)[0][0]
    assert free > blocked and free >= 1


# ============================================================================= the feed latch

def test_feed_latch_settles_onto_the_instant_rule_and_holds_new_feeds_back():
    a = np.radians(np.arange(9) * 40.0 + 5.0)
    P = np.vstack([np.c_[7.0 * np.sin(a), 7.0 * np.cos(a)], [[0.5, 0.3]]])     # a ring of nine and one inside it
    lat = display.FeedLatch(10)
    alive = np.ones(10, bool)
    for _ in range(20):
        f = lat.step(P, alive, 30.0, 0.1)
    ref = display.swarm_feeds(P, 30.0)
    assert sorted(f.drones.tolist()) == sorted(ref.drones.tolist()) and 9 not in set(f.drones.tolist())
    assert np.allclose(np.sort(f.yaw), np.sort(ref.yaw), atol=1e-6)
    # move the inner drone far out: it becomes a hull vertex, but is shown only after the hysteresis and its turn
    k = 9
    Q = P.copy()
    Q[k] = [0.0, -40.0]
    shown_at = None
    for i in range(60):
        f = lat.step(Q, alive, 30.0, 0.1)
        if k in set(f.drones.tolist()):
            shown_at = (i + 1) * 0.1
            break
    assert shown_at is not None and shown_at >= display.BOUNDARY_HYST_S - 1e-9


def test_feed_latch_lone_drone_turns_to_the_body_yaw():
    lat = display.FeedLatch(3)
    P = np.zeros((3, 2))
    alive = np.array([True, False, False])
    f = None
    for _ in range(30):
        f = lat.step(P, alive, 90.0, 0.1)
    assert len(f) == 1 and abs(f.yaw[0] - 90.0) < 1e-6


# ============================================================================= intent

def _mask(centres=((0, 0), (40, 40), (-60, 80))):
    return pilots.TargetMask(routing.Cylinders(_boxes(centres)), ARENA)


def test_reflect_folds_back_inside_and_mirrors_the_heading():
    p, th = pilots.reflect(np.array([250.0, 0.0]), 90.0, ARENA, 25.0)
    assert -175.0 <= p[0] <= 175.0 and abs(p[0] - 100.0) < 1e-9 and abs(th + 90.0) < 1e-9
    p, th = pilots.reflect(np.array([0.0, -260.0]), 180.0, ARENA, 25.0)
    assert abs(p[1] + 90.0) < 1e-9 and abs(abs(th) - 0.0) < 1e-9


def test_intent_chain_is_deterministic_free_and_spaced():
    mask = _mask()
    a = pilots.Intent("rw", (10.0, -30.0), 45.0, mask, seed=7)
    b = pilots.Intent("rw", (10.0, -30.0), 45.0, mask, seed=7)
    c = pilots.Intent("rw", (10.0, -30.0), 45.0, mask, seed=8)
    ta = np.array([a.target(k) for k in range(60)])
    tb = np.array([b.target(k) for k in range(60)])
    tc = np.array([c.target(k) for k in range(60)])
    assert np.allclose(ta, tb) and not np.allclose(ta, tc)
    assert all(mask.contains(p) for p in ta)
    assert (np.linalg.norm(np.diff(ta, axis=0), axis=1) >= pilots.MIN_STEP_M - 1e-6).all()


def test_fitted_intent_uses_the_fitted_turns_and_pauses():
    mask = _mask()
    ip = pilots.intent_params("fit", "swarm", _model())
    it = pilots.Intent("fit", (0.0, -100.0), 0.0, mask, seed=3, **ip)
    for k in range(80):
        it.target(k)
    assert ip["run_mean_m"] == 75.0 and ip["pause_p"] == 0.3
    assert set(it.pause[1:]) <= {0.0, 3.0} and any(p > 0 for p in it.pause)
    # turns are drawn from |25|, |30| with a random sign (before any reflection), so most consecutive headings differ
    # by one of them
    d = np.abs(np.asarray([float((b - a + 180) % 360 - 180) for a, b in zip(it.headings[1:-1], it.headings[2:])]))
    assert np.mean(np.isclose(d, 25.0) | np.isclose(d, 30.0)) > 0.5


def test_mask_excludes_buildings_and_the_arena_rim():
    mask = _mask()
    assert not mask.contains((0.0, 0.0))                   # a building
    assert not mask.contains((195.0, 0.0))                 # inside the arena's margin
    q = mask.snap((0.0, 0.0))
    assert mask.contains(q) and Footprints(_boxes([(0, 0)])).distance([q])[0] > 0


# ============================================================================= head

def test_head_process_dwells_and_moves_between_them():
    hp = pilots.HeadProcess("fit", 5, _model())
    ys = np.array([hp(t) for t in np.arange(0, 120, 0.1)])
    assert ys.min() >= -40.0 - 1e-6 and ys.max() <= 20.0 + 1e-6
    assert np.mean(np.isin(np.round(ys, 6), [0.0, -40.0, 20.0])) > 0.5      # mostly dwelling
    assert pilots.HeadProcess("ahead", 5)(12.3) == 0.0


# ============================================================================= survival summaries

def test_rmst_and_km_median():
    t = np.array([10.0, 20.0, 30.0, 100.0])
    obs = np.array([True, True, True, False])
    assert abs(search.rmst(t, obs, 100.0) - np.mean(np.minimum(t, 100.0))) < 1e-9
    assert search.km_median(t, obs) == 20.0
    assert math.isnan(search.km_median(np.array([100.0, 100.0]), np.array([False, False])))


# ============================================================================= track statistics

def test_track_stats_straight_and_square():
    t = np.arange(0, 120, 0.5)
    st = search.track_stats(t, np.c_[5.0 * t, 0 * t])
    assert st["turns_per_km"] == 0 and abs(st["straight30"] - 1.0) < 1e-6 and abs(st["speed_ms"] - 5.0) < 1e-6
    side = 100.0
    s = np.mod(5.0 * t, 4 * side)
    xy = np.where((s < side)[:, None], np.c_[s, 0 * s],
                  np.where((s < 2 * side)[:, None], np.c_[0 * s + side, s - side],
                           np.where((s < 3 * side)[:, None], np.c_[3 * side - s, 0 * s + side],
                                    np.c_[0 * s, 4 * side - s])))
    st = search.track_stats(t, xy)
    assert 4 <= st["turns_per_km"] * st["metres"] / 1000.0 <= 6     # five corners in 600 m (four, then one more)


# ============================================================================= the fitted pilot's segmentation

def test_pilotfit_segment_finds_turns_runs_and_pauses():
    dt = 0.1
    t = np.arange(0, 60, dt)
    mag = np.ones_like(t)
    mag[(t >= 30) & (t < 33)] = 0.0                                     # a 3 s pause
    course = np.where(t < 20, 0.0, np.where(t < 21, 90.0 * (t - 20), 90.0))   # a 90 deg turn over 1 s at 20 s
    course = np.where(t >= 45, 90.0 - 40.0 * np.clip((t - 45) / 0.5, 0, 1), course)   # 40 deg back at 45 s
    dist = np.cumsum(mag * 7.0 * dt)
    sig = dict(t=t, mag=mag, course=course, body=course.copy(), dist=dist, head=np.zeros_like(t),
               ok_head=np.ones_like(t, bool))
    seg = pilotfit.segment(sig, 0.0, 60.0)
    assert len(seg["turns"]) == 2 and abs(abs(seg["turns"][0]) - 90.0) < 15 and abs(abs(seg["turns"][1]) - 40.0) < 15
    assert len(seg["pauses"]) == 1 and abs(seg["pauses"][0] - 3.0) < 0.3
    assert sum(m for m, _, _ in seg["runs"]) > 300


# ============================================================================= a short search

def test_a_single_drone_search_finds_a_goal_beside_its_start():
    from citysim import dynamic
    boxes = _boxes([(60.0, 0.0)])
    C = search.city_context(boxes, ARENA)
    ring = _ring(0.0, 60.0, r=20.0)
    pts, w = discovery.loop_points(ring)
    goal = (pts, w, np.array([0.0, 60.0]))
    sc = dict(start=[0.0, -60.0], heading0=0.0, seed=11)
    band = dict(moving_h=19.5, moving_p=-15.0, own_h=17.0, own_p=-14.0)
    pilot = dict(name="rw", intent="rw", arms=["single"])
    rows = search.fly_arms(C, goal, sc, pilot, None, band, cap_s=120.0, head_models=("ahead",))
    assert len(rows) == 1 and rows[0]["arm"] == "single"
    assert math.isfinite(rows[0]["t_any_r200"]) and rows[0]["t_any_r200"] < 5.0     # 120 m away, facing it
    # the waypoint pilot follows a fixed list and does not stop when it sees the goal
    sim, m = dynamic.make_sim(boxes, (0.0, -60.0), "single", 1.0)
    it = pilots.Intent("rw", (0.0, -60.0), 0.0, C["mask"], 11)
    fl = search.SearchFlight(sim, m, C["fp"], "single", 1.0, it, search.Planner(C["cyl"], "single", 1.0, ARENA, {}),
                             goal, (17.0, -14.0), {}, 75.0, 60.0)
    fl.fly_waypoints([0.0, 60.0, 1.0, 40.0, -80.0, 60.0, 1.0, 40.0], 30.0)
    assert fl.reason == "cap" and abs(fl.sim.t - 30.0) < 0.05 and ("any", 80.0) in fl.found
