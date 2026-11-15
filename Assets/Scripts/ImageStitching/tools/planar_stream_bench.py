"""
Play a recorded real-drone clip through PLANAR the way the live system would see it.

``planar_clip_bench.py`` measures one instant: it converges the estimators on a single
set of frames and reports the seam.  That is the right test of *accuracy* and the wrong
test of a real-time system, where the estimators run on a background thread, their
answer arrives late, and it is applied to frames whose poses have moved on since.

This tool steps a clip at its own frame rate on a simulated clock:

- every frame, each drone contributes the newest frame it had published by then, and
  the mosaic is rendered from that frame's pose plus whatever correction is in force;
- the estimators run at their cadence on the frame that was current when they started,
  and their result only takes effect after the time they actually took to compute.

WHAT IT REPORTS
---------------
``seam``     the plane-supported seam residual of ``planar_clip_bench`` (ORB matches
             between warped view pairs, restricted to the dominant surface), sampled
             through the clip.  ORB on purpose: the learned aligner uses SuperPoint and
             LightGlue, so the score comes from a detector it never saw.
``jitter``   how much the views move *against each other* from one frame to the next.
             Points are tracked in each drone's own video (pyramidal LK) and carried
             through that frame's warp; on a rigid scene every view should then report
             the same canvas motion, and the RMS disagreement is seam shimmer that no
             single-frame number shows.  Reported in source pixels.

USAGE
-----
    python tools/planar_stream_bench.py --clip MED_facade_stationary_1
    python tools/planar_stream_bench.py --all
    python tools/planar_stream_bench.py --clip nadir_6drone_30m --video out.mp4

``--configs`` picks among ``off`` (pose only), ``legacy`` (the photometric sweep and
phase-correlation refiner) and ``learned`` (planar_align); the default runs all three.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import planar_bundle as pb                                          # noqa: E402
import planar_geometry as pg                                        # noqa: E402
import planar_clip_bench as bench                                   # noqa: E402

try:
    import cv2
except ImportError:                                                 # pragma: no cover
    cv2 = None

# Clips worth streaming, with the drones each carries. The two the index marks unusable
# (a starved link, a crashed aircraft) are left out.
CLIPS = [
    ("MED_facade_stationary_1", [1, 2, 3]),
    ("MED_facade_stationary_2", [1, 2, 3]),
    ("MED_facade_5drone_short", [1, 2, 3, 4, 5]),
    ("grass_nadir_long", [1, 2, 3]),
    ("grass_nadir_short_contracting", [1, 2, 3]),
    ("nadir_6drone_30m", [1, 2, 3, 4, 5, 6]),
    ("nadir_6drone_40m", [1, 2, 3, 4, 5, 6]),
]


class ClipStream:
    """Sequential reader: the newest frame each drone had published by time ``t``."""

    def __init__(self, clip_dir, drones, pose_noise=None):
        self.clip_dir = clip_dir
        self.drones = list(drones)
        self.poses = {d: bench.load_poses(clip_dir, d) for d in self.drones}
        # A simulated clip is an image sequence; a real one is a video per drone.
        self.sequence = bench.is_image_sequence(clip_dir, self.drones[0])
        self.caps = {} if self.sequence else {
            d: cv2.VideoCapture(os.path.join(clip_dir, "drone%d.mp4" % d)) for d in self.drones}
        # Injected pose error: one bias per drone for the whole clip, which is what GNSS
        # and compass error look like over ten seconds (their correlation times are a
        # minute or more). ``(sigma_m, yaw_deg, seed)``; vertical error is half the
        # horizontal, barometric altitude being the better of the two between aircraft.
        self.bias = {}
        if pose_noise:
            sigma, yaw, seed = pose_noise
            rng = np.random.default_rng(int(seed))
            for d in self.drones:
                self.bias[d] = (rng.normal(0.0, [sigma, 0.5 * sigma, sigma]),
                                float(rng.normal(0.0, yaw)))
        self.cursor = {d: -1 for d in self.drones}       # index into poses of held frame
        self.decoded = {d: 0 for d in self.drones}       # frames read from the file
        self.image = {d: None for d in self.drones}
        starts = [p[0]["t"] for p in self.poses.values() if p]
        ends = [p[-1]["t"] for p in self.poses.values() if p]
        self.t0, self.t1 = max(starts), min(ends)

    def views_at(self, t):
        views = []
        for slot, d in enumerate(self.drones):
            poses = self.poses[d]
            k = self.cursor[d]
            while k + 1 < len(poses) and poses[k + 1]["t"] <= t:
                k += 1
            if k < 0:
                continue
            if k != self.cursor[d]:
                # ``frame`` is 1-based in the CSV; decode forward to it.
                want = poses[k]["frame"]
                img = None
                if self.sequence:
                    img = cv2.imread(bench.frame_image_path(self.clip_dir, d, want))
                else:
                    while self.decoded[d] < want:
                        ok, frame = self.caps[d].read()
                        if not ok:
                            break
                        self.decoded[d] += 1
                        img = frame
                if img is not None:
                    if img.shape[1] != bench.WIRE_W or img.shape[0] != bench.WIRE_H:
                        img = cv2.resize(img, (bench.WIRE_W, bench.WIRE_H),
                                         interpolation=cv2.INTER_AREA)
                    self.image[d] = img
                self.cursor[d] = k
            if self.image[d] is None:
                continue
            rec = poses[k]
            pos, quat = rec["pos"], rec["quat"]
            if d in self.bias:
                shift, yaw = self.bias[d]
                pos = tuple(float(a + b) for a, b in zip(pos, shift))
                quat = bench._quat_mul(bench._quat_yaw(yaw), quat)
            views.append({"slot": slot, "drone_id": d - 1, "heading": 0.0,
                          "image": self.image[d], "pos": pos, "quat": quat,
                          "capture_time": rec["t"] - self.t0 + 1.0,
                          "pose_status": rec["status"], "cached": False})
        return views

    def close(self):
        for cap in self.caps.values():
            cap.release()


class JitterMeter:
    """Frame-to-frame disagreement between the views' canvas motion (see module doc)."""

    def __init__(self):
        self.prev = {}                       # drone_id -> (grey, H_image_to_canvas)
        self.samples = []

    @staticmethod
    def _to_canvas(Hinv, pts):
        q = np.column_stack([pts, np.ones(len(pts))]) @ Hinv.T
        return q[:, :2] / q[:, 2:3]

    def step(self, cams, canvas_px_to_source_px, record=True):
        motion = {}
        now = {}
        for cam in cams:
            did = cam["view"]["drone_id"]
            grey = cv2.cvtColor(cam["view"]["image"], cv2.COLOR_BGR2GRAY)
            try:
                Hinv = np.linalg.inv(cam["H"])
            except np.linalg.LinAlgError:
                continue
            now[did] = (grey, Hinv)
            if did not in self.prev:
                continue
            g0, Hinv0 = self.prev[did]
            if g0 is grey or g0.shape != grey.shape:
                continue
            p0 = cv2.goodFeaturesToTrack(g0, 120, 0.01, 12)
            if p0 is None or len(p0) < 12:
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, grey, p0, None, winSize=(21, 21),
                                                 maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() < 12:
                continue
            a = self._to_canvas(Hinv0, p0.reshape(-1, 2)[ok])
            b = self._to_canvas(Hinv, p1.reshape(-1, 2)[ok])
            motion[did] = np.median(b - a, axis=0)
        self.prev = now
        if record and len(motion) >= 2:
            m = np.stack(list(motion.values()))
            dev = m - m.mean(axis=0)
            self.samples.append(float(np.sqrt((dev ** 2).sum(axis=1).mean()))
                                * canvas_px_to_source_px)

    def rms(self):
        return float(np.sqrt(np.mean(np.square(self.samples)))) if self.samples else float("nan")


def stream_clip(ps_mod, torch, label, drones, recordings, config_name, fps=20.0,
                warmup_s=2.0, eval_period_s=0.5, align_period_s=0.25, max_seconds=None,
                canvas=(1200, 800), video=None, standoff=None, device=None,
                pose_delay=None, blend=None, pose_noise=None, rate=None, pose_filter=True):
    """One clip, one estimator configuration.  Returns a result dict."""
    from planar_selftest import make_bare_stitcher

    clip_dir, row = bench.find_clip(recordings, label)
    with open(os.path.join(clip_dir, "session.json")) as fh:
        session = json.load(fh)
    bench.use_clip_wire_size(session)
    meta = session.get("meta", session)
    scene = meta.get("scene_plane") or {}
    truth = float(scene.get("standoff_m", 30.0))
    so = truth if standoff is None else float(standoff)
    intr = bench.wire_intrinsics(session)
    K = pg.intrinsics_matrix(*intr)
    px_per_m = intr[1] / truth

    on = config_name != "off"
    ps_mod.LEGACY_ESTIMATORS = config_name == "legacy"
    mpp = bench.make_config(session, truth, canvas, False, False)["metres_per_pixel"]
    cfg = bench.make_config(session, so, canvas, on, on, mpp=mpp)
    if rate is not None:
        cfg["refine_rate"] = float(rate)

    s = make_bare_stitcher(ps_mod, torch)
    if device is not None:
        s.render_device = torch.device(device)
    s.ALIGN_MIN_PERIOD_S = 0.0
    if pose_filter is False:
        s.POSE_FILTER = False                    # the poses exactly as recorded
    elif pose_delay is not None:
        import planar_pose_filter
        s._pose_filter = planar_pose_filter.PoseFilter(pose_delay)
    if blend is not None:
        cfg["blend_mode"] = {"feather": ps_mod.BLEND_FEATHER, "nearest": ps_mod.BLEND_NEAREST,
                             "multiband": ps_mod.BLEND_MULTIBAND}[blend]
    sim = {"t": 0.0}
    s._clock = lambda: sim["t"]                  # blends run on clip time, not wall time
    s.set_live_config(cfg)
    if config_name == "learned":
        # Load and warm the networks before the clock starts, as a running stitcher has
        # long since done: a clip is a few seconds, and the one-off cost of the first
        # forward pass would otherwise be charged to its first alignment.
        s._learned_alignment()

    stream = ClipStream(clip_dir, drones, pose_noise=pose_noise)
    plane = {"plane_normal": (0.0, 0.0, -1.0), "plane_d": 0.0, "plane_valid": True,
             "plane_mode": ps_mod.PLANE_MODE_FORMATION_RELATIVE,
             "gimbal_pitch": float(meta.get("gimbal_pitch", 0.0)),
             "centre_drone_id": drones[len(drones) // 2] - 1}
    if meta.get("sim") and scene.get("plane_n_unity") and standoff is None:
        # A simulated clip carries the plane Unity's raycast would publish, so it is
        # replayed the way the simulator runs: a measured plane, not one derived from
        # the formation. (n . X is the same number in Unity's frame and the right-handed
        # one: the handedness flip negates z in both.)
        plane.update(plane_normal=tuple(scene["plane_n_unity"]),
                     plane_d=float(scene["plane_d_unity"]), plane_mode=1)

    # A simulated clip's recorded poses and plane are exact, which gives a second measure
    # that needs no feature detector and cannot be fooled by a repeating wall.
    truth = n_true = d_true = None
    if meta.get("sim") and scene.get("plane_n_unity"):
        truth = stream if pose_noise is None else ClipStream(clip_dir, drones)
        n_true = np.asarray(pg.unity_dir_to_rh(scene["plane_n_unity"]), dtype=np.float64)
        d_true = float(scene["plane_d_unity"])
    true_px = []

    writer = None
    jitter = JitterMeter()
    seams, supports, psnrs, update_ms, trace = [], [], [], [], []
    pending, pending_at, next_update, next_eval = None, 0.0, 0.0, warmup_s
    t_end = stream.t1 if max_seconds is None else min(stream.t1, stream.t0 + max_seconds)
    n_frames = failed = 0

    t = stream.t0 + 1.0 / fps
    while t <= t_end:
        rel = t - stream.t0
        views = stream.views_at(t)
        sim["t"] = rel
        if pending is not None and rel >= pending_at:
            if pending.get("blend") is not None:
                # The ease-in starts when the answer arrives, not when it was asked for.
                pending = dict(pending, blend=(rel,) + tuple(pending["blend"][1:]))
            s._correction, pending = pending, None

        s._plane_invalid_since = None
        pano, ok, _reason = s.planar_pano(views, intr, plane, cfg) if len(views) >= 2 \
            else (None, False, 0)
        n_frames += 1

        if on and rel >= next_update and pending is None and s._frame_snapshot is not None:
            # The estimator sees this frame and answers later: hold the correction back
            # for as long as the pass really took.
            before = s._correction
            wall = time.perf_counter()
            s.compute_warps()
            took = time.perf_counter() - wall
            if config_name == "legacy":
                # The photometric pair runs at ~20 Hz and its sleep is part of ``took``.
                took = max(0.0, took - 0.05)
            update_ms.append(1e3 * took)
            pending, s._correction = s._correction, before
            pending_at = rel + took
            next_update = rel + max(align_period_s, took)

        if not ok:
            failed += 1
            t += 1.0 / fps
            continue

        frame, cams, M, cw, ch, mpp_view = s._last_geometry
        # Counted after the warm-up only: the first lock is a deliberate jump of many
        # pixels, and on a six-second clip it would be most of the RMS.
        jitter.step(cams, mpp_view * px_per_m, record=rel >= warmup_s)

        if rel >= next_eval:
            next_eval = rel + eval_period_s
            grey, valid = s._warp_grey(cams, M, cw, ch)
            sup = pb.plane_support(cams, frame, M, [grey[i] for i in range(len(cams))],
                                   [valid[i] for i in range(len(cams))], K, mpp_view)
            res = sup.get("residual_m", np.zeros(0))
            if len(res) >= pb.MIN_SUPPORT:
                seams.append(float(np.median(res)) * px_per_m)
                supports.append(int(sup.get("support", 0)))
                a = getattr(s, "_align_stats", {}) or {}
                trace.append((round(rel, 2), round(seams[-1], 2), a.get("mode", "-"),
                              a.get("inliers", 0), a.get("held", "")))
            psnrs.append(float(s._last_stats.get("overlap_psnr", float("nan"))))
            if truth is not None:
                tp = true_misplacement(cams, frame, truth.views_at(t), K, n_true, d_true, px_per_m)
                if tp is not None:
                    true_px.append(tp)

        if video is not None:
            if writer is None:
                writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (pano.shape[1], pano.shape[0]))
            writer.write(pano)
        t += 1.0 / fps

    stream.close()
    if truth is not None and truth is not stream:
        truth.close()
    if writer is not None:
        writer.release()

    seams = np.asarray(seams)
    return {
        "clip": label, "config": config_name, "frames": n_frames, "failed": failed,
        "evals": int(len(seams)),
        "seam_px": float(np.median(seams)) if len(seams) else float("nan"),
        "seam_px_p90": float(np.percentile(seams, 90)) if len(seams) else float("nan"),
        "seam_px_worst": float(seams.max()) if len(seams) else float("nan"),
        "support": float(np.median(supports)) if supports else 0.0,
        "psnr": float(np.nanmedian(psnrs)) if psnrs and np.isfinite(psnrs).any() else float("nan"),
        "jitter_px": jitter.rms(),
        "true_px": float(np.median(true_px)) if true_px else None,
        "true_px_p90": float(np.percentile(true_px, 90)) if true_px else None,
        "true_px_worst": float(np.max(true_px)) if true_px else None,
        "update_ms": float(np.median(update_ms)) if update_ms else 0.0,
        "align": dict(getattr(s, "_align_stats", {}) or {}),
        "trace": trace,
        "px_per_m": px_per_m,
    }


def true_misplacement(cams, frame, truth_views, K, n_true, d_true, px_per_m):
    """
    How far the views sit from each other's truth, for a simulated clip: source pixels.

    Each view's footprint is placed twice -- by the mapping that was drawn, and by its
    exact pose through the exact plane -- and the difference is its placement error on the
    plane. A common error moves the whole mosaic and costs nothing, so what is returned is
    the mean, over pairs of views, of how much their two errors differ: the seam a pixel
    on the plane would show between them. Unlike the ORB seam it cannot be fooled by a
    wall that repeats -- a view one whole window out matches its neighbours perfectly and
    is three metres wrong.
    """
    h, w = cams[0]["view"]["image"].shape[:2]
    px = np.array([[w * x, h * y, 1.0] for x in (0.25, 0.5, 0.75) for y in (0.25, 0.5, 0.75)]).T
    rays_cam = (np.linalg.inv(K) @ px).T
    errs = []
    for c in cams:
        tv = next((v for v in truth_views if v["drone_id"] == c["view"]["drone_id"]), None)
        if tv is None:
            continue
        R, C = pg.unity_pose_to_cv(tv["pos"], tv["quat"])
        rays = rays_cam @ R
        lam = (d_true - C @ n_true) / (rays @ n_true)
        ab_true = frame.to_plane_coords(C + lam[:, None] * rays)
        q = np.linalg.solve(c["G"], px)
        errs.append(((q[:2] / q[2]).T - ab_true).mean(axis=0))
    if len(errs) < 2:
        return None
    errs = np.asarray(errs)
    diff = errs[:, None, :] - errs[None, :, :]
    iu = np.triu_indices(len(errs), 1)
    return float(np.linalg.norm(diff, axis=2)[iu].mean()) * px_per_m


def print_result(r):
    print("  %-8s seam %6.2f px (p90 %6.2f, worst %6.2f) over %2d samples | support %4.0f | "
          "PSNR %5.1f dB | jitter %5.2f px | update %4.0f ms%s"
          % (r["config"], r["seam_px"], r["seam_px_p90"], r["seam_px_worst"], r["evals"],
             r["support"], r["psnr"], r["jitter_px"], r["update_ms"],
             "" if not r["failed"] else " | %d frames without a panorama" % r["failed"]))
    if r.get("true_px") is not None:
        print("           against the true poses: views misplaced by %6.2f px (p90 %6.2f, "
              "worst %6.2f)" % (r["true_px"], r["true_px_p90"], r["true_px_worst"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", help="clip label from clips.csv")
    ap.add_argument("--all", action="store_true", help="every usable clip")
    ap.add_argument("--drones", type=int, nargs="+", default=None)
    ap.add_argument("--recordings-dir", default=bench.DEFAULT_RECORDINGS)
    ap.add_argument("--configs", nargs="+", default=["off", "legacy", "learned"],
                    choices=["off", "legacy", "learned"])
    ap.add_argument("--seconds", type=float, default=None, help="stop after this much clip")
    ap.add_argument("--standoff", type=float, default=None,
                    help="override the clip's measured standoff (a typed, wrong prior)")
    ap.add_argument("--pose-delay", type=float, default=None,
                    help="seconds the picture lags its telemetry (planar_pose_filter); "
                         "negative looks ahead instead")
    ap.add_argument("--no-pose-filter", action="store_true",
                    help="use the poses exactly as recorded, held between telemetry samples")
    ap.add_argument("--blend", choices=["feather", "nearest", "multiband"], default=None)
    ap.add_argument("--pose-noise", metavar="M,DEG[,SEED]", default=None,
                    help="inject a constant pose bias per drone: position sigma in metres "
                         "and yaw sigma in degrees -- for simulated clips, whose poses "
                         "are exact")
    ap.add_argument("--video", metavar="PATH", default=None,
                    help="write the panorama stream; the config name is appended")
    ap.add_argument("--json", metavar="PATH", default=None, help="write all results")
    ap.add_argument("--trace", action="store_true",
                    help="print the seam at every sample, with the aligner's state")
    ap.add_argument("--rate", type=float, default=None,
                    help="override the estimators' low-pass rate (planarRefineRate)")
    args = ap.parse_args()

    if cv2 is None:
        print("opencv is required")
        return 2
    if args.all:
        todo = CLIPS
    elif args.clip:
        known = dict(CLIPS)
        drones = args.drones or known.get(args.clip)
        if drones is None:
            # Not in the table: count the pose files (a simulated clip, or a new one).
            clip_dir, _row = bench.find_clip(args.recordings_dir, args.clip)
            drones = [d for d in range(1, 11) if clip_dir and os.path.isfile(
                os.path.join(clip_dir, "drone%d_frames.csv" % d))] or [1, 2, 3]
        todo = [(args.clip, drones)]
    else:
        ap.print_help()
        return 2

    import torch
    import PlanarStitcher as ps_mod

    noise = None
    if args.pose_noise:
        parts = [float(x) for x in args.pose_noise.split(",")]
        noise = (parts[0], parts[1] if len(parts) > 1 else 0.0, parts[2] if len(parts) > 2 else 1)

    results = []
    for label, drones in todo:
        print("%s  (drones %s%s)" % (label, ",".join(map(str, drones)),
                                     "" if args.standoff is None
                                     else ", standoff forced to %.1f m" % args.standoff))
        for name in args.configs:
            video = None
            if args.video:
                root, ext = os.path.splitext(args.video)
                video = "%s_%s_%s%s" % (root, label, name, ext or ".mp4")
            r = stream_clip(ps_mod, torch, label, drones, args.recordings_dir, name,
                            max_seconds=args.seconds, video=video, standoff=args.standoff,
                            pose_delay=args.pose_delay, blend=args.blend,
                            pose_noise=noise, rate=args.rate,
                            pose_filter=not args.no_pose_filter,
                            device="cuda" if torch.cuda.is_available() else None)
            results.append(r)
            print_result(r)
            if args.trace:
                for row in r["trace"]:
                    print("      t %5.1f s  seam %6.2f px  [%s, %s inliers] %s" % row)
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
