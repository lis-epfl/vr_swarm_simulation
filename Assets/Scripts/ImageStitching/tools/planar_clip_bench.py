"""
Replay a recorded real-drone clip through PLANAR, headless, and measure the seam.

No Unity, no shared memory, no clip_replay.py.  A clip carries everything the stitcher
needs -- per-frame Unity-world camera poses in ``drone*_frames.csv``, the video, the
intrinsics and a measured scene plane in ``session.json`` -- so the whole pose-driven
path can be exercised in one process, deterministically, and re-run against a change.

WHY THIS EXISTS
---------------
``planar_feed_bench.py`` renders a synthetic facade: the poses are exact by construction
and the plane is whatever it was told, so it measures the geometry, not the field.
``planar_selftest.py`` never touches a clip at all.  Everything asserted about real-drone
alignment -- how much of the error is the plane, how much is per-drone pose, whether the
refiner ever accepts a measurement -- was inference from the error budget until this ran.

WHAT IT MEASURES, AND WHY IT IS NOT OVERLAP PSNR
------------------------------------------------
PLANAR only has to align the plane.  Sky, cranes and background buildings may be
misaligned across seams; they are not part of the objective.  Overlap PSNR cannot express
that -- it averages over all overlapping pixels, so on the MED facade it is dominated by
background content sitting 217 canvas pixels out of place, and it would score a mosaic
worse for correctly ignoring it.

So the headline number is the **plane-supported seam residual**: correspondences between
warped view pairs, triangulated across the formation's real baseline, restricted to the
dominant surface, and reported as the median disagreement in metres on that surface.  The
off-plane residual is reported beside it as context, never as the score.  PSNR is still
printed because it is what the existing code logs and comparability is worth keeping.

USAGE
-----
    python tools/planar_clip_bench.py --clip MED_facade_stationary_1
    python tools/planar_clip_bench.py --clip MED_facade_stationary_1 --matrix
    python tools/planar_clip_bench.py --list

``--matrix`` sweeps standoff x refine x sweep, which is the baseline the estimator work
is measured against.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import planar_bundle as pb                                          # noqa: E402
import planar_geometry as pg                                        # noqa: E402

try:
    import cv2
except ImportError:                                                 # pragma: no cover
    cv2 = None

# Where DJI_Swarm keeps its recordings, relative to this repo. The clip bench is a tool
# in this repo that reads the other one, rather than the reverse, because the stitcher
# under test lives here and the clips are data.
DEFAULT_RECORDINGS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "..", "DJI_Swarm", "AOS server", "recordings"))

# The block resolution PLANAR actually receives. Not the 1920x1080 the clip was recorded
# at: ImageSharing publishes 800x450 on the wire, and measuring at source resolution
# would report a seam the stitcher never had the pixels to produce.
WIRE_W, WIRE_H = 800, 450


def find_clip(recordings, label):
    """Resolve a clip label (or a bare folder name) to its directory."""
    index = os.path.join(recordings, "clips.csv")
    if os.path.isfile(index):
        with open(index, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("clip") == label:
                    return os.path.join(recordings, row["folder"]), row
    direct = os.path.join(recordings, label)
    if os.path.isdir(direct):
        return direct, {}
    return None, None


def list_clips(recordings):
    index = os.path.join(recordings, "clips.csv")
    if not os.path.isfile(index):
        print("no clips.csv under %s" % recordings)
        return 1
    with open(index, newline="") as fh:
        for row in csv.DictReader(fh):
            print("  %-32s %s  %s, %s, %s drones ready=%s"
                  % (row["clip"], row["folder"], row.get("site", "?"),
                     row.get("camera", "?"), row.get("formation", "?"),
                     row.get("planar_ready", "?")))
    return 0


def load_poses(clip_dir, drone):
    path = os.path.join(clip_dir, "drone%d_frames.csv" % drone)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append({
                "frame": int(row["frame"]),
                "t": float(row["t_epoch"]),
                "pos": (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])),
                "quat": (float(row["quat_x"]), float(row["quat_y"]),
                         float(row["quat_z"]), float(row["quat_w"])),
                "status": int(row["pose_status"]),
                "heading": float(row["heading"]),
                "gimbal_yaw": float(row["gimbal_yaw"]),
            })
    return out


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _quat_yaw(deg):
    """Unity left-handed yaw about +Y."""
    h = np.deg2rad(deg) * 0.5
    return (0.0, np.sin(h), 0.0, np.cos(h))


def gimbal_yaw_bias(clip_dir, drones):
    """
    Per-drone ``gimbal_yaw - heading``, in degrees.

    ``dji_camera_pose`` documents ``gimbal_yaw`` as "compass bearing, 0 = North" and
    builds the camera rotation from it directly.  On this fleet it is that bearing plus a
    fixed per-aircraft offset -- the gimbal's yaw zero disagrees with the flight
    controller's compass by a constant.  Measured on the 2026-08-11 clips the offsets are
    about (+3.0, -15.5, -6.4) degrees and they reproduce across sites and across gimbal
    pitches, so they are a mount/encoder calibration rather than anything about a flight.

    The *differential* part is what breaks a mosaic: it points the drones' cameras up to
    18 degrees apart in the solve while they were physically parallel, which at 34 m is
    over 11 m of seam -- larger than every other term in the error budget put together,
    and not correctable by any per-view translation.  The common part merely rotates the
    whole mosaic on the plane and costs nothing.

    Returned as a dict keyed by zero-based drone id, matching ``view["drone_id"]``.
    """
    bias = {}
    for d in drones:
        poses = load_poses(clip_dir, d)
        if not poses:
            continue
        delta = np.array([(p["gimbal_yaw"] - p["heading"] + 180.0) % 360.0 - 180.0
                          for p in poses])
        bias[d - 1] = float(np.median(delta))
    return bias


def read_frame(clip_dir, drone, frame_index):
    """One decoded, wire-sized BGR frame.  ``frame`` in the CSV is 1-based."""
    if cv2 is None:
        raise RuntimeError("opencv is required to decode clip video")
    cap = cv2.VideoCapture(os.path.join(clip_dir, "drone%d.mp4" % drone))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index - 1))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.resize(img, (WIRE_W, WIRE_H), interpolation=cv2.INTER_AREA)


def build_views(clip_dir, drones, t_target, yaw_bias=None):
    """
    The block records for the instant nearest ``t_target``, one per drone.

    Each drone is sampled independently at its own nearest frame rather than by index:
    the three streams run at slightly different rates (19.6-20.0 fps here) so equal
    indices drift apart over a clip, and the pose/frame pairing is what the whole solve
    rests on.
    """
    views = []
    for slot, d in enumerate(drones):
        poses = load_poses(clip_dir, d)
        if not poses:
            continue
        rec = min(poses, key=lambda r: abs(r["t"] - t_target))
        img = read_frame(clip_dir, d, rec["frame"])
        if img is None:
            continue
        quat = rec["quat"]
        if yaw_bias:
            # Yaw is the outermost factor in quat_from_gimbal (q = qYaw*qPitch*qRoll),
            # so pre-multiplying rewrites the yaw and leaves pitch and roll alone.
            quat = _quat_mul(_quat_yaw(-yaw_bias.get(d - 1, 0.0)), quat)
        views.append({
            "slot": slot,
            "drone_id": d - 1,                 # feed indices are zero-based on the wire
            "heading": 0.0,
            "image": img,
            "pos": rec["pos"],
            "quat": quat,
            "capture_time": rec["t"] - t_target,
            "pose_status": rec["status"],
            "cached": False,
            "_skew": rec["t"] - t_target,
        })
    return views


def make_config(session, standoff, canvas, refine, sweep, mpp=None):
    import PlanarStitcher as ps_mod

    cam = session["meta"]["camera"] if "meta" in session else session["camera"]
    canvas_w, canvas_h = canvas
    if mpp is None:
        # Frame the canvas on one view's footprint at this standoff, so the mosaic fills
        # it rather than sitting in a corner. The live scene uses a fixed 0.05 m/px, but
        # that is framing, not alignment, and a canvas the formation overflows would clip
        # the very overlap being measured.
        fy_wire = cam["fy"] * (WIRE_H / float(cam["height"]))
        mpp = 2.2 * WIRE_W * standoff / fy_wire / canvas_w

    return {
        "canvas": (canvas_w, canvas_h),
        "metres_per_pixel": mpp,
        "max_range": 400.0,
        "feather_px": 40,
        "aniso_max": 12.0,
        "min_coverage": 0.05,
        "pose_source": 0,
        "psnr_gate": False,
        "blend_mode": ps_mod.BLEND_NEAREST,
        "debug_view": ps_mod.DEBUG_OFF,
        "canvas_mode": ps_mod.CANVAS_MODE_FIXED,
        "zoom": 1.0,
        "pan": (0.0, 0.0),
        "plane_sweep": bool(sweep),
        "pose_refine": bool(refine),
        "sweep_range": 4.0,
        "sweep_steps": 9,
        "refine_rate": 0.25,
        "refine_max_shift": 3.0,
        "standoff": standoff,
    }


def wire_intrinsics(session):
    """Intrinsics scaled from the recorded resolution to the 800x450 wire size."""
    cam = session["meta"]["camera"] if "meta" in session else session["camera"]
    sx, sy = WIRE_W / float(cam["width"]), WIRE_H / float(cam["height"])
    return (cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy)


def run_once(ps_mod, torch, views, intr, plane, cfg, passes):
    """One configuration: render, converge the estimators, render again, measure."""
    from planar_selftest import make_bare_stitcher

    s = make_bare_stitcher(ps_mod, torch)
    if cfg["plane_sweep"] or cfg["pose_refine"]:
        s.set_live_config(cfg)
        for _ in range(passes):
            s._plane_invalid_since = None
            s.planar_pano(views, intr, plane, cfg)
            s.compute_warps()

    s._plane_invalid_since = None
    pano, ok, reason = s.planar_pano(views, intr, plane, cfg)
    if not ok:
        return None, {"reason": reason}, s

    frame, cams, M, cw, ch, mpp = s._last_geometry
    grey, valid = s._warp_grey(cams, M, cw, ch)
    K = pg.intrinsics_matrix(*intr)
    support = pb.plane_support(cams, frame, M,
                               [grey[i] for i in range(len(cams))],
                               [valid[i] for i in range(len(cams))],
                               K, mpp)
    return pano, support, s


def _fmt_m(x):
    return "  n/a " if not np.isfinite(x) else "%6.2f" % x


def report(name, support, stats, refine_stats, sweep_stats, px_per_m_facade):
    res = support.get("residual_m", np.zeros(0))
    off = support.get("residual_off_plane_m", np.zeros(0))
    med = float(np.median(res)) if len(res) else float("nan")
    p90 = float(np.percentile(res, 90)) if len(res) else float("nan")
    med_off = float(np.median(off)) if len(off) else float("nan")

    print("  %-34s plane seam %s m (p90 %s) = %s px | support %4d/%-4d | "
          "off-plane %s m | plane offset %s m | PSNR %5.1f dB"
          % (name, _fmt_m(med), _fmt_m(p90),
             _fmt_m(med * px_per_m_facade), support.get("support", 0),
             support.get("total", 0), _fmt_m(med_off),
             _fmt_m(support.get("offset", float("nan"))),
             stats.get("overlap_psnr", float("nan"))))
    if refine_stats:
        print("      refine: %s" % _stats_line(refine_stats))
    if sweep_stats:
        print("      sweep : %s" % _stats_line(sweep_stats))


def _stats_line(d):
    return ", ".join("%s=%s" % (k, ("%.3f" % v) if isinstance(v, float) else v)
                     for k, v in d.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", help="clip label from clips.csv, or a folder name")
    ap.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS)
    ap.add_argument("--list", action="store_true", help="list available clips")
    ap.add_argument("--drones", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--at", type=float, default=0.5,
                    help="fraction through the clip to sample (0..1)")
    ap.add_argument("--canvas", type=int, nargs=2, default=[1200, 800])
    ap.add_argument("--passes", type=int, default=12,
                    help="estimator passes before the measured render")
    ap.add_argument("--matrix", action="store_true",
                    help="sweep standoff x refine x sweep")
    ap.add_argument("--standoff", type=float, default=None,
                    help="override the clip's measured standoff")
    ap.add_argument("--gimbal-yaw-cal", metavar="SPEC", default="off",
                    help="'off' (default), 'self' to calibrate from this clip, a clip "
                         "label to calibrate from another one, or comma-separated "
                         "per-drone degrees")
    ap.add_argument("--save", metavar="PATH", default=None)
    args = ap.parse_args()

    if args.list or not args.clip:
        return list_clips(args.recordings_dir)

    clip_dir, row = find_clip(args.recordings_dir, args.clip)
    if clip_dir is None:
        print("no such clip: %s (under %s)" % (args.clip, args.recordings_dir))
        return 2

    with open(os.path.join(clip_dir, "session.json")) as fh:
        session = json.load(fh)
    meta = session.get("meta", session)
    scene = meta.get("scene_plane") or {}

    import torch
    import PlanarStitcher as ps_mod

    truth_standoff = float(scene.get("standoff_m", 30.0))
    baseline = float(scene.get("baseline_m", float("nan")))
    intr = wire_intrinsics(session)
    # Pixels per metre ON the facade at this standoff -- how a seam in metres converts to
    # a seam in source pixels. Deliberately not scene_plane.px_per_m, which is seam-pixels
    # per metre of *standoff error*: a different quantity that happens to read 4.9 here.
    px_per_m_facade = intr[1] / truth_standoff

    poses = load_poses(clip_dir, args.drones[0])
    if not poses:
        print("no pose CSV for drone %d in %s" % (args.drones[0], clip_dir))
        return 2
    t0, t1 = poses[0]["t"], poses[-1]["t"]
    t_target = t0 + (t1 - t0) * max(0.0, min(1.0, args.at))

    print("clip %s  (%s)" % (args.clip, os.path.basename(clip_dir)))
    print("  site=%s camera=%s formation=%s motion=%s"
          % (row.get("site", "?"), row.get("camera", "?"),
             row.get("formation", "?"), row.get("motion", "?")))
    print("  standoff %.3f m, baseline %.2f m, f_wire %.1f px, %.2f px per metre "
          "on the facade" % (truth_standoff, baseline, intr[1], px_per_m_facade))

    # Always measure the gimbal-yaw bias, whether or not it is applied: its differential
    # part dominates every other term in the budget when it is non-zero, so a run that
    # does not mention it is a run whose numbers cannot be interpreted.
    measured = gimbal_yaw_bias(clip_dir, args.drones)
    spread = (max(measured.values()) - min(measured.values())) if measured else 0.0
    print("  gimbal_yaw - heading: %s  (spread %.1f deg -> %.1f m at this standoff)"
          % (", ".join("d%d %+.1f" % (k + 1, v) for k, v in sorted(measured.items())),
             spread, truth_standoff * np.tan(np.deg2rad(spread))))

    spec = (args.gimbal_yaw_cal or "off").strip()
    if spec in ("off", ""):
        yaw_bias = None
    elif spec == "self":
        yaw_bias = measured
    elif "," in spec:
        yaw_bias = {i: float(x) for i, x in enumerate(spec.split(","))}
    else:
        other, _ = find_clip(args.recordings_dir, spec)
        if other is None:
            print("no such clip to calibrate from: %s" % spec)
            return 2
        yaw_bias = gimbal_yaw_bias(other, args.drones)
    if yaw_bias:
        print("  applying gimbal-yaw calibration: %s"
              % ", ".join("d%d %+.1f" % (k + 1, v) for k, v in sorted(yaw_bias.items())))

    views = build_views(clip_dir, args.drones, t_target, yaw_bias)
    if len(views) < 2:
        print("  only %d views decoded -- nothing to stitch" % len(views))
        return 2
    skew = [v["_skew"] for v in views]
    print("  %d views at t+%.2fs, pose/frame skew spread %.0f ms"
          % (len(views), t_target - t0, 1000.0 * (max(skew) - min(skew))))

    plane = {"plane_normal": (0.0, 0.0, -1.0), "plane_d": 0.0, "plane_valid": True,
             "plane_mode": ps_mod.PLANE_MODE_FORMATION_RELATIVE,
             "gimbal_pitch": float(meta.get("gimbal_pitch", 0.0)),
             "centre_drone_id": views[len(views) // 2]["drone_id"]}

    if args.matrix:
        standoffs = [30.0, truth_standoff]
        combos = [(r, w) for r in (False, True) for w in (False, True)]
    else:
        standoffs = [args.standoff if args.standoff is not None else truth_standoff]
        combos = [(False, False), (True, False)]

    # One canvas scale for every configuration. The metric is in metres, but the matches
    # are found on the canvas, so letting each standoff pick its own scale would change
    # the measurement resolution between rows of the table being compared.
    mpp = make_config(session, truth_standoff, args.canvas, False, False)[
        "metres_per_pixel"]
    print("  canvas %dx%d at %.4f m/px\n" % (args.canvas[0], args.canvas[1], mpp))

    last_pano = None
    for so in standoffs:
        print("standoff %.3f m%s" % (so, "" if abs(so - truth_standoff) > 1e-6
                                     else "   <- measured"))
        for refine, sweep in combos:
            cfg = make_config(session, so, args.canvas, refine, sweep, mpp=mpp)
            pano, support, s = run_once(ps_mod, torch, views, intr, plane, cfg,
                                        args.passes)
            name = "refine=%-5s sweep=%-5s" % (refine, sweep)
            if pano is None:
                print("  %-34s no panorama (%s)" % (name, support.get("reason")))
                continue
            report(name, support, s._last_stats, s._refine_stats, s._sweep_stats,
                   px_per_m_facade)
            last_pano = pano
        print()

    if args.save and last_pano is not None and cv2 is not None:
        cv2.imwrite(args.save, last_pano)
        print("wrote %s" % args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
