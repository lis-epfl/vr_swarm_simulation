"""
Synthetic drone feed for exercising the real-drone PLANAR path without aircraft.

    cd Assets/Scripts/ImageStitching && python tools/planar_feed_bench.py

Writes real ``DroneFeedSharedMemory`` blocks in the v2 (48-byte) layout, so Unity's
``ImageSharing.cs`` and ``StitcherThreading.py`` run completely unmodified: the same
map, the same handshake, the same pose fields the DJI producer will write.  What it
replaces is only the aircraft.

Why this is worth having
------------------------
The real-drone PLANAR path crosses two repos, two languages and two shared-memory
maps, and its failure mode is silent -- a mosaic built from a mirrored or rotated
pose still renders, it just never lines up.  Discovering that on a rooftop with five
aircraft in the air is the expensive way.  Here the ground truth is known exactly,
so a wrong sign is obvious immediately.

What it does NOT cover
----------------------
The GPS/gimbal -> pose conversion, which lives in the other repo and is checked by
``DJI_Swarm/AOS server/dji_pose_selftest.py`` against an independent derivation.
This tool starts from poses that are already in Unity world.  The seam between the
two halves is the pose convention, stated in that file and in CLAUDE.md.

Rendering
---------
Each view is produced by warping a synthetic facade through the *same* geometry the
stitcher will invert (``planar_geometry.build_G``).  That is deliberate: it means a
correct pipeline must reproduce the facade, so the panorama is its own assertion.
``--pose-error`` then perturbs the poses that get published while leaving the imagery
alone, which is exactly the error the estimators exist to absorb.
"""

import argparse
import math
import mmap
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import planar_geometry as pg  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


# Must match ImageSharing.cs (MaxFeedBlocks / ImageWidth / ImageHeight / MetadataSize)
# and DJI_Swarm/AOS server/utils/imageSharingUtil.py. A mismatch here is silent: it
# reads image bytes as a header rather than failing.
MAP_NAME = "DroneFeedSharedMemory"
MAX_DRONES = 10
IMAGE_W, IMAGE_H = 800, 450
HEADER_BYTES = 48
CAM_POS_OFFSET = 12
CAM_ROT_OFFSET = 24
CAPTURE_TIME_OFFSET = 40
POSE_STATUS_OFFSET = 44
POSE_VALID = 1 << 0

IMAGE_BYTES = IMAGE_W * IMAGE_H * 3
BLOCK_BYTES = HEADER_BYTES + IMAGE_BYTES


def facade_texture(height_px=1400, width_px=2400, seed=11, metres_per_texel=0.01):
    """
    A facade: rendered brick with a grid of window bays, at realistic metric sizes.

    The sizes matter more than the appearance.  Everything the estimators see has
    been resampled onto a canvas whose pixels are tens of centimetres, so detail
    finer than that aliases into noise and carries no alignment information: a
    facade whose only structure is 2 cm mortar is, to the refiner, a blank wall.
    Real buildings have window bays every 3-4 m and metre-scale panel and staining
    variation, and it is that scale which makes them registrable at all.

    Getting this wrong the first time made the refiner look broken when it was in
    fact correctly refusing an ambiguous measurement -- so the geometry here is in
    metres, converted to texels, rather than in pixels chosen to look right.
    """
    rng = np.random.default_rng(seed)

    def px(metres):
        return max(1, int(round(metres / metres_per_texel)))

    # Fine brick, plus metre-scale panel/staining variation. The low-frequency term
    # is what survives to the estimator canvas.
    fine = rng.random((height_px, width_px)).astype(np.float32)
    coarse = rng.random((max(2, height_px // px(1.5)),
                         max(2, width_px // px(1.5)))).astype(np.float32)
    if cv2 is not None:
        fine = cv2.GaussianBlur(fine, (0, 0), 2.0)
        coarse = cv2.resize(coarse, (width_px, height_px), interpolation=cv2.INTER_CUBIC)
        coarse = cv2.GaussianBlur(coarse, (0, 0), px(0.4))
    else:
        coarse = np.zeros_like(fine)
    base = 0.45 * _norm01(fine) + 0.55 * _norm01(coarse)

    img = np.stack([(base * 90 + 130).astype(np.uint8)] * 3, axis=-1)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int32) - 30, 0, 255)   # warmer brick

    win_h, win_w = px(1.6), px(1.1)
    pitch_y, pitch_x = px(3.2), px(3.5)
    for y in range(px(1.0), height_px - win_h, pitch_y):
        for x in range(px(1.0), width_px - win_w, pitch_x):
            shade = int(rng.integers(25, 75))
            img[y:y + win_h, x:x + win_w] = (shade + 20, shade + 8, shade)
            img[y:y + px(0.12), x:x + win_w] = (205, 205, 200)           # lintel
    return img


def _norm01(a):
    return (a - a.min()) / max(1e-9, float(a.ptp()))


def build_formation(count, standoff, baseline, altitude, rows):
    """
    A wall of drones facing a facade, in Unity world (+X East, +Y Up, +Z North).

    Returns ``[(drone_id, pos, quat_xyzw)]``.  The facade is at ``z = standoff``,
    every camera looks along +Z, so the formation is parallel to it -- the
    configuration ``ScenePlaneMode.FormationRelative`` is designed around.
    """
    per_row = int(math.ceil(count / float(max(1, rows))))
    out = []
    for i in range(count):
        col, row = i % per_row, i // per_row
        x = (col - (per_row - 1) * 0.5) * baseline
        y = altitude + (row - (rows - 1) * 0.5) * baseline
        # Identity rotation looks along +Z in Unity, which is North here.
        out.append((i, (x, y, 0.0), (0.0, 0.0, 0.0, 1.0)))
    return out


def render_view(texture, K, pos, quat, frame, metres_per_texel):
    """Warp the facade into one camera, through the geometry the stitcher inverts."""
    th, tw = texture.shape[:2]
    R_cv, C = pg.unity_pose_to_cv(pos, quat)
    M_tex = pg.canvas_to_plane_matrix(metres_per_texel, tw / 2.0, th / 2.0)
    H = pg.homography_canvas_to_image(pg.build_G(K, R_cv, C, frame), M_tex)
    return cv2.warpPerspective(texture, H, (IMAGE_W, IMAGE_H), flags=cv2.INTER_LINEAR)


def perturb(pos, quat, sigma_lateral, sigma_depth, sigma_deg, rng):
    """
    Differential pose error, split by what the refiner can and cannot absorb.

    Lateral (x, y here -- along the facade) is a pure in-plane shift of that view's
    footprint, which is exactly the two degrees of freedom planarPoseRefine estimates.
    Depth (z, along the facade normal) is not: it scales that view's content about its
    own footprint, and no per-view translation can represent a scale. The plane sweep
    cannot fix it either, being a single global number rather than one per drone.

    They are separate knobs because lumping them into one "position error" makes the
    refiner look broken when it is in fact recovering everything it ever claimed to.
    """
    if sigma_lateral <= 0.0 and sigma_depth <= 0.0 and sigma_deg <= 0.0:
        return pos, quat
    p = (float(pos[0]) + rng.normal(0.0, sigma_lateral),
         float(pos[1]) + rng.normal(0.0, sigma_lateral),
         float(pos[2]) + rng.normal(0.0, sigma_depth))
    if sigma_deg > 0.0:
        # Yaw only: a compass bias is the dominant attitude error outdoors, and
        # unlike position error it does not shrink with range.
        half = math.radians(rng.normal(0.0, sigma_deg)) * 0.5
        qy = (0.0, math.sin(half), 0.0, math.cos(half))
        x1, y1, z1, w1 = qy
        x2, y2, z2, w2 = quat
        quat = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    return p, quat


def write_block(mm, drone_id, image_bgr, heading, pos, quat, capture_time,
                timeout_s=0.5):
    """
    One block, with the flag handshake ImageSharing.cs expects.

    Returns False if the consumer held the block busy for the whole timeout, which
    means Unity is not draining the map (or is not running).
    """
    offset = drone_id * BLOCK_BYTES
    deadline = time.time() + timeout_s
    while True:
        mm.seek(offset)
        if struct.unpack("<i", mm.read(4))[0] == 0:
            break
        if time.time() > deadline:
            return False
        time.sleep(0.002)

    mm.seek(offset)
    mm.write(struct.pack("<i", 1))                       # busy
    mm.seek(offset + 4)
    mm.write(struct.pack("<if", drone_id, heading))
    mm.seek(offset + CAM_POS_OFFSET)
    mm.write(struct.pack("<3f", *pos))
    mm.write(struct.pack("<4f", *quat))
    mm.write(struct.pack("<f", capture_time))
    mm.write(struct.pack("<i", POSE_VALID))
    mm.seek(offset + HEADER_BYTES)
    mm.write(image_bgr.tobytes())
    mm.seek(offset)
    mm.write(struct.pack("<i", 0))                       # ready
    return True


def run_selftest(frames_truth, K, args, fy):
    """
    Drive PlanarStitcher over the synthetic views, headless, with no Unity at all.

    Checks the half of the path that lives in this repo -- FormationRelative plane
    derivation, the geometric solve, the render -- before any of it is asked to work
    across two processes and a shared-memory map.  If this fails, nothing about the
    Unity wiring is worth debugging yet.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import PlanarStitcher as ps_mod
    from planar_selftest import make_bare_stitcher

    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    views = [{'slot': i, 'drone_id': did, 'heading': 0.0, 'image': img,
              'pos': pos, 'quat': quat, 'capture_time': 0.0,
              'pose_status': POSE_VALID, 'cached': False}
             for i, (did, img, pos, quat) in enumerate(frames_truth)]

    # metres_per_pixel sized from the formation's ACTUAL extent plus one view's
    # footprint, not from drones x baseline: with more than one row that overestimates
    # the width by the row count, and the canvas scale is what sets the estimators'
    # resolution -- the refiner quantises to whole estimator-canvas pixels, so an
    # over-wide canvas silently caps how much pose error it can recover.
    canvas_w, canvas_h = 1200, 800
    xs = [p[0] for _, _, p, _ in frames_truth]
    ys = [p[1] for _, _, p, _ in frames_truth]
    span_x = (max(xs) - min(xs)) + IMAGE_W * args.standoff / fy
    span_y = (max(ys) - min(ys)) + IMAGE_H * args.standoff / fy
    mpp = max(span_x / canvas_w, span_y / canvas_h)
    cfg = {"canvas": (canvas_w, canvas_h), "metres_per_pixel": mpp,
           "max_range": 400.0, "feather_px": 40, "aniso_max": 12.0,
           "min_coverage": 0.05, "pose_source": 0, "psnr_gate": False,
           "blend_mode": ps_mod.BLEND_NEAREST, "debug_view": ps_mod.DEBUG_OFF,
           "plane_sweep": False, "pose_refine": False,
           "sweep_range": 4.0, "sweep_steps": 9,
           "refine_rate": 0.25, "refine_max_shift": 3.0,
           "standoff": args.standoff}
    plane = {"plane_normal": (0.0, 0.0, -1.0), "plane_d": 0.0, "plane_valid": True,
             "plane_mode": ps_mod.PLANE_MODE_FORMATION_RELATIVE,
             "gimbal_pitch": 0.0, "centre_drone_id": args.drones // 2}

    failures = []

    def check(name, ok, detail=""):
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                               "  --  " + detail if detail else ""))
        if not ok:
            failures.append(name)
        return ok

    s = make_bare_stitcher(ps_mod, torch)
    pano, ok, reason = s.planar_pano(views, intr, plane, cfg)
    if not check("panorama rendered from the derived plane", ok and pano is not None,
                 "ok=%s reason=%s" % (ok, reason)):
        return 1

    st = s._last_stats
    check("every drone contributed a view", st.get("views") == args.drones,
          "%s of %d kept, %s unposed" % (st.get("views"), args.drones, st.get("unposed")))
    # Which rule should win depends on the formation, not on preference: a single row
    # of drones is collinear, so no plane fits its positions and the mean-forward rule
    # is the correct answer there. Asserting "positions" unconditionally would be
    # asserting that the fallback never runs.
    want_source = "positions" if args.rows > 1 else "forward"
    check("the plane came from the %s rule (rows=%d)" % (want_source, args.rows),
          s._plane_source == want_source, "source = %s" % s._plane_source)
    psnr = st.get("overlap_psnr", float("nan"))
    # Exact poses and an exact plane: the views must agree to resampling accuracy.
    # Anything low here is a geometry bug, not a limitation -- there is nothing
    # being estimated.
    check("views agree where they overlap", np.isfinite(psnr) and psnr > 25.0,
          "overlap PSNR %.1f dB" % psnr)
    check("coverage is plausible for the formation", st.get("coverage", 0.0) > 0.25,
          "%.0f%% of canvas" % (100.0 * st.get("coverage", 0.0)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "selftest_feed_bench.jpg")
    if cv2 is not None:
        cv2.imwrite(out, pano)
        print("       wrote %s" % out)

    # With pose error injected, the refiner must recover most of it.
    if args.pose_error > 0.0 or args.depth_error > 0.0 or args.yaw_error > 0.0:
        rng = np.random.default_rng(args.seed)
        bad = []
        for v in views:
            p, q = perturb(v["pos"], v["quat"], args.pose_error, args.depth_error,
                           args.yaw_error, rng)
            bad.append(dict(v, pos=p, quat=q))
        s_bad = make_bare_stitcher(ps_mod, torch)
        s_bad.planar_pano(bad, intr, plane, cfg)
        before = s_bad._last_stats.get("overlap_psnr", float("nan"))

        s_fix = make_bare_stitcher(ps_mod, torch)
        cfg_fix = dict(cfg, pose_refine=True, refine_rate=1.0)
        # The estimator switches come from set_live_config, not from the frame
        # snapshot: they must have independent freshness, or turning an estimator off
        # while the render path is failing leaves it running on one frozen frame.
        # Mirrored here so the bench exercises the production path.
        s_fix.set_live_config(cfg_fix)
        for _ in range(8):
            s_fix._plane_invalid_since = None
            s_fix.planar_pano(bad, intr, plane, cfg_fix)
            s_fix.compute_warps()
        s_fix.set_live_config(cfg)
        s_fix.planar_pano(bad, intr, plane, cfg)
        after = s_fix._last_stats.get("overlap_psnr", float("nan"))
        # Reported alongside the result: a refiner that rejected every measurement and
        # one that accepted them and helped nothing are different problems, and the
        # PSNR alone cannot tell them apart.
        rs = s_fix._refine_stats or {}
        check("the pose refiner recovers injected pose error", after > before + 3.0,
              "overlap PSNR %.1f -> %.1f dB (%s accepted, %s rejected, worst "
              "correction %s)" % (before, after, rs.get("accepted"), rs.get("rejected"),
                                  ("%.2f m" % rs["worst_shift"]) if "worst_shift" in rs
                                  else rs.get("skipped", "n/a")))

    print("\n" + ("FAILED (%d)" % len(failures) if failures else "All checks passed."))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drones", type=int, default=5, help="fleet size (max %d)" % MAX_DRONES)
    ap.add_argument("--rows", type=int, default=1, help="rows in the wall")
    ap.add_argument("--standoff", type=float, default=30.0, help="metres to the facade")
    ap.add_argument("--baseline", type=float, default=8.0, help="metres between drones")
    ap.add_argument("--altitude", type=float, default=15.0)
    ap.add_argument("--vfov", type=float, default=46.4,
                    help="vertical FOV; must match PyUniSharingFast.manualVerticalFovDeg")
    ap.add_argument("--pose-error", type=float, default=0.0,
                    help="differential LATERAL position error sigma, metres, along the "
                         "facade (published only). This is what planarPoseRefine absorbs.")
    ap.add_argument("--depth-error", type=float, default=0.0,
                    help="differential DEPTH error sigma, metres, along the facade normal "
                         "(published only). Deliberately separate: it is a per-view scale "
                         "error, which neither estimator can represent.")
    ap.add_argument("--yaw-error", type=float, default=0.0,
                    help="differential yaw error sigma, degrees (published only)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = run until Ctrl-C")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--selftest", action="store_true",
                    help="Render the views and mosaic them here, headless -- no Unity, "
                         "no shared memory. Checks this repo's half of the path before "
                         "the cross-process half is worth debugging.")
    args = ap.parse_args()

    if cv2 is None:
        print("opencv is required (conda env 'stitching')")
        return 1
    if not 1 <= args.drones <= MAX_DRONES:
        print("--drones must be 1..%d (the map's fixed capacity)" % MAX_DRONES)
        return 1

    fy = (IMAGE_H * 0.5) / math.tan(math.radians(args.vfov) * 0.5)
    K = pg.intrinsics_matrix(fy, fy, IMAGE_W * 0.5, IMAGE_H * 0.5)

    # Facade at z = +standoff in Unity, normal pointing back at the cameras.
    n_rh = pg.unity_dir_to_rh((0.0, 0.0, -1.0))
    n_rh = n_rh / np.linalg.norm(n_rh)
    origin_rh = pg.unity_point_to_rh((0.0, args.altitude, args.standoff))
    d = float(np.dot(n_rh, origin_rh))
    frame = pg.build_plane_frame(n_rh, d, origin_rh, np.array([1.0, 0.0, 0.0]))

    texture = facade_texture()
    # One texel per centimetre of facade: finer than the ~5.7 cm/px the cameras
    # resolve at 30 m, so resampling never limits what the mosaic can show.
    metres_per_texel = 0.01

    formation = build_formation(args.drones, args.standoff, args.baseline,
                                args.altitude, args.rows)
    rng = np.random.default_rng(args.seed)

    print("planar_feed_bench: %d drones, %.1f m standoff, %.1f m baseline, "
          "vfov %.1f deg (f = %.0f px)"
          % (args.drones, args.standoff, args.baseline, args.vfov, fy))
    print("  facade footprint per view: %.1f x %.1f m, GSD %.1f cm/px"
          % (IMAGE_W * args.standoff / fy, IMAGE_H * args.standoff / fy,
             args.standoff / fy * 100.0))
    if args.pose_error or args.depth_error or args.yaw_error:
        print("  publishing pose error: %.2f m lateral / %.2f m depth / %.2f deg yaw "
              "(imagery stays truthful)"
              % (args.pose_error, args.depth_error, args.yaw_error))
    print("  In Unity set: typeOfStitcher = PLANAR, useManualIntrinsics = true, "
          "manualVerticalFovDeg = %.1f," % args.vfov)
    print("               scenePlaneMode = FormationRelative, planarStandoffMetres "
          "= %.1f, ImageSharing.stitchSlots >= %d" % (args.standoff, args.drones))

    # Render once: the formation is static, so the only per-frame work is the copy.
    frames = []
    frames_truth = []
    for drone_id, pos, quat in formation:
        img = render_view(texture, K, pos, quat, frame, metres_per_texel)
        pub_pos, pub_quat = perturb(pos, quat, args.pose_error, args.depth_error,
                                    args.yaw_error, rng)
        frames.append((drone_id, img, pub_pos, pub_quat))
        frames_truth.append((drone_id, img, pos, quat))
        print("  drone %d at (%.1f, %.1f, %.1f)" % ((drone_id,) + tuple(pos)))

    if args.selftest:
        print("\nHeadless self-test (no Unity, no shared memory)")
        return run_selftest(frames_truth, K, args, fy)

    mm = mmap.mmap(-1, MAX_DRONES * BLOCK_BYTES, MAP_NAME)
    # Retire the capacity blocks no drone will write, or Unity reads whatever a
    # previous run left there. droneId = -1 is the "no new frame" marker on this map.
    for slot in range(args.drones, MAX_DRONES):
        mm.seek(slot * BLOCK_BYTES)
        mm.write(struct.pack("<ii", 0, -1))

    period = 1.0 / max(1e-3, args.fps)
    t0 = time.perf_counter()
    published = stalled = 0
    try:
        while args.duration <= 0.0 or (time.perf_counter() - t0) < args.duration:
            tick = time.perf_counter()
            for drone_id, img, pub_pos, pub_quat in frames:
                if write_block(mm, drone_id, img, 0.0, pub_pos, pub_quat, tick - t0):
                    published += 1
                else:
                    stalled += 1
            if published and published % (args.drones * 40) == 0:
                print("  ... %d blocks published, %d stalled" % (published, stalled))
            sleep = period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        mm.close()

    print("published %d blocks, %d stalled" % (published, stalled))
    if stalled and not published:
        print("Nothing was consumed -- is the Unity DJI scene running with "
              "ImageSharing enabled?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
