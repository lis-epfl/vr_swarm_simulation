"""
Stand in for Unity so the whole PLANAR bridge can be timed without the editor.

    cd Assets/Scripts/ImageStitching
    python tools/planar_bridge_bench.py --clip nadir_6drone_30m --scale 2
    python tools/planar_bridge_bench.py --clip MED_facade_stationary_1 --align --save out.png
    python tools/planar_bridge_bench.py --scale 2 --switch-scale 3      # the ceiling moves mid-run

What it does
------------
Plays PyUniSharingFast's and ImageSharing's parts of the shared-memory contract from a
recorded clip -- metadata with the planar settings, the scene plane under its seqlock, a
heartbeat, one posed block per aircraft whenever that aircraft has a new frame, and the
panorama read -- and runs the real, unmodified ``StitcherThreading.py`` against it as a
subprocess.  ``planar_stream_bench.py`` measures how well the mosaic is aligned, on a
simulated clock; this measures how fast the live system produces it and what reaches
Unity: the Python side under its real threading (block read, tracker, render, aligner,
panorama write, one GIL), at the output size Unity would publish.

The panorama size it publishes is the planar canvas times ``--scale``, which is what
``PyUniSharingFast.ResolvePanoramaWireSize`` publishes for ``planarMaxRenderScale``: a
ceiling.  The stitcher draws each panorama with as many pixels as its feeds need under it
and writes the size in the header, and the bench reads each one at that size, as Unity
does.  ``PLANAR_AUTO_OUTPUT=0`` in the environment makes the stitcher draw at the ceiling.

It uses shared-memory sections of its own (``STITCH_SHM_SUFFIX``, default ``_pbench``), so
it can run beside a live editor -- the two then share the GPU, which is itself a useful
measurement.

What it reports
---------------
* the stitcher's stitch rate and its own ``[PLANAR]`` lines (milliseconds a frame, the
  output size, output pixels per source pixel);
* the panoramas delivered per second on the Unity side under its real upload policy --
  a new sequence number, at a size the stitcher can have drawn -- and the sizes they
  arrived at;
* with ``--switch-scale``, how many panoramas arrived at a size the new ceiling does not
  allow.  Those are the frames the size check exists to keep off the screen.

``--send-hz`` switches from the real fleet's pacing (a block when a feed has a new frame,
~20 per second per aircraft) to the simulator's (every view rewritten at that rate).
"""

import os

# Captured before anything imports torch; see stabstitch_bridge_bench.py.
_CHILD_ENV = dict(os.environ)

import argparse
import json
import mmap
import struct
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2  # noqa: E402

import StitcherThreading as st  # noqa: E402
import PlanarStitcher as ps  # noqa: E402
import planar_clip_bench as bench  # noqa: E402
import planar_stream_bench as stream_bench  # noqa: E402
from stabstitch_bridge_bench import put_string, invalidate_slots, gpu_load_thread, parse_size  # noqa: E402

BLENDS = {"feather": ps.BLEND_FEATHER, "nearest": ps.BLEND_NEAREST,
          "multiband": ps.BLEND_MULTIBAND}


def wire_size(canvas, scale):
    """The panorama size Unity publishes in PLANAR (ResolvePanoramaWireSize)."""
    scale = min(max(scale, 0.25), st.PANORAMA_MAX_WIDTH / canvas[0],
                st.PANORAMA_MAX_HEIGHT / canvas[1])
    return max(2, int(round(canvas[0] * scale))), max(2, int(round(canvas[1] * scale)))


def write_metadata(mm, args, frame, wire, n_views, intr, mpp, standoff):
    """The whole metadata block, as WriteMetadata lays it out, with PLANAR selected."""
    b = bytearray(st.METADATA_SIZE)
    o = 0
    struct.pack_into("<iiiii", b, o, frame[0], frame[1], n_views, wire[0], wire[1]); o += 20
    put_string(b, o, "PLANAR"); o += 64
    struct.pack_into("<B", b, o, 0); o += 1                      # isCylindrical
    put_string(b, o, "BF"); o += 64
    struct.pack_into("<B", b, o, 0); o += 1                      # isRANSAC
    struct.pack_into("<i", b, o, 50); o += 4                     # checks
    struct.pack_into("<ff", b, o, 0.7, 0.1); o += 8              # ratio, score thresholds
    struct.pack_into("<i", b, o, 1000); o += 4                   # focal
    o += 1                                                       # reserved byte
    put_string(b, o, "REFERENCE_BLEND"); o += 64
    struct.pack_into("<i", b, o, 41); o += 4
    struct.pack_into("<f", b, o, 15.0); o += 4
    struct.pack_into("<i", b, o, 60); o += 4
    struct.pack_into("<B", b, o, 0); o += 1                      # quality fallback off
    struct.pack_into("<f", b, o, 0.0); o += 4
    struct.pack_into("<f", b, o, 0.0); o += 4                    # head angle
    struct.pack_into("<B", b, o, 1); o += 1                      # print rate
    assert o == 253, o

    struct.pack_into("<ii", b, st.META_BLOCK_HEADER_SIZE_OFFSET, st.BLOCK_HEADER_SIZE_V2,
                     st.PLANAR_WIRE_VERSION)
    struct.pack_into("<ffff", b, st.META_FX_OFFSET, *intr)
    struct.pack_into("<ii", b, st.META_PLANAR_CANVAS_OFFSET, args.canvas[0], args.canvas[1])
    struct.pack_into("<ff", b, st.META_PLANAR_MPP_OFFSET, mpp, 400.0)
    struct.pack_into("<i", b, st.META_PLANAR_FEATHER_OFFSET, 40)
    struct.pack_into("<ff", b, st.META_PLANAR_ANISO_OFFSET, 12.0, 0.05)
    struct.pack_into("<BB", b, st.META_PLANAR_POSE_SOURCE_OFFSET, 0, 0)
    struct.pack_into("<B", b, st.META_PLANAR_BLEND_MODE_OFFSET, BLENDS[args.blend])
    struct.pack_into("<B", b, st.META_PLANAR_DEBUG_VIEW_OFFSET, 0)
    struct.pack_into("<BB", b, st.META_PLANAR_SWEEP_ENABLED_OFFSET,
                     1 if args.align else 0, 1 if args.align else 0)
    struct.pack_into("<f", b, st.META_PLANAR_SWEEP_RANGE_OFFSET, 4.0)
    struct.pack_into("<i", b, st.META_PLANAR_SWEEP_STEPS_OFFSET, 9)
    struct.pack_into("<ff", b, st.META_PLANAR_REFINE_RATE_OFFSET, 0.25, 3.0)
    struct.pack_into("<f", b, st.META_PLANAR_STANDOFF_OFFSET, standoff)
    struct.pack_into("<B", b, st.META_PLANAR_CANVAS_MODE_OFFSET,
                     ps.CANVAS_MODE_AUTOFIT if args.canvas_mode == "autofit"
                     else ps.CANVAS_MODE_FIXED)
    struct.pack_into("<ii", b, st.META_BLOCK_SLOT_CAPACITY_OFFSET, st.BLOCK_SLOT_CAPACITY,
                     st.BLOCK_SLOT_STRIDE)
    struct.pack_into("<ii", b, st.META_BLOCK_SECTION_BYTES_OFFSET, st.BLOCK_SECTION_BYTES,
                     st.PANORAMA_SECTION_BYTES)
    # Everything but the dynamic block and the heartbeat, which have writers of their own
    # and must not be stepped back by a republish.
    mm.seek(0)
    mm.write(bytes(b[:st.META_DYN_SEQ_OFFSET]))
    mm.seek(st.META_PLANAR_SWEEP_ENABLED_OFFSET)
    mm.write(bytes(b[st.META_PLANAR_SWEEP_ENABLED_OFFSET:st.META_PLANAR_ZOOM_OFFSET]))
    mm.seek(st.META_PLANAR_CANVAS_MODE_OFFSET)
    mm.write(bytes(b[st.META_PLANAR_CANVAS_MODE_OFFSET:st.META_PLANAR_CANVAS_MODE_OFFSET + 1]))
    mm.seek(st.META_BLOCK_SLOT_CAPACITY_OFFSET)
    mm.write(bytes(b[st.META_BLOCK_SLOT_CAPACITY_OFFSET:st.META_BLOCK_SLOT_CAPACITY_OFFSET + 8]))
    mm.seek(st.META_BLOCK_SECTION_BYTES_OFFSET)
    mm.write(bytes(b[st.META_BLOCK_SECTION_BYTES_OFFSET:]))


def write_dynamic(mm, seq, plane):
    """The seqlocked dynamic block: plane, gimbal pitch, centre drone, zoom and pan."""
    mm.seek(st.META_DYN_SEQ_OFFSET)
    mm.write(struct.pack("<i", seq + 1))                         # odd: in flux
    mm.seek(st.META_PLANE_N_OFFSET)
    mm.write(struct.pack("<ffff", *plane["plane_normal"], plane["plane_d"]))
    mm.write(struct.pack("<BB", 1, plane["plane_mode"]))
    mm.seek(st.META_GIMBAL_PITCH_OFFSET)
    mm.write(struct.pack("<fi", plane["gimbal_pitch"], plane["centre_drone_id"]))
    mm.seek(st.META_PLANAR_ZOOM_OFFSET)
    mm.write(struct.pack("<fff", 1.0, 0.0, 0.0))
    mm.seek(st.META_DYN_SEQ_OFFSET)
    mm.write(struct.pack("<i", seq + 2))
    return seq + 2


def write_block(mm, slot, view, capture_time):
    """One posed block under the flag handshake. False if the consumer holds the flag."""
    base = slot * st.BLOCK_SLOT_STRIDE
    mm.seek(base)
    if struct.unpack("<i", mm.read(4))[0] != 0:
        return False
    mm.seek(base)
    mm.write(struct.pack("<i", 1))
    header = struct.pack("<if3f4ffi", view["drone_id"], 0.0, *view["pos"], *view["quat"],
                         capture_time, view["pose_status"])
    assert 4 + len(header) == st.BLOCK_HEADER_SIZE_V2
    mm.seek(base + 4)
    mm.write(header)
    mm.seek(base + st.BLOCK_HEADER_SIZE_V2)
    mm.write(memoryview(view["image"]).cast("B"))
    mm.seek(base)
    mm.write(struct.pack("<i", 0))
    return True


def load_clip(args):
    """``(session, drones, timeline)``; the timeline is one view list per clip frame time."""
    clip_dir, _row = bench.find_clip(args.recordings, args.clip)
    if clip_dir is None:
        raise SystemExit(f"clip {args.clip!r} not found under {args.recordings}")
    with open(os.path.join(clip_dir, "session.json")) as fh:
        session = json.load(fh)
    bench.use_clip_wire_size(session)
    if args.frame:
        bench.WIRE_W, bench.WIRE_H = args.frame
    drones = args.drones or next((d for label, d in stream_bench.CLIPS if label == args.clip),
                                 None)
    if drones is None:
        n = 1
        while os.path.exists(os.path.join(clip_dir, "drone%d_frames.csv" % (n + 1))):
            n += 1
        drones = list(range(1, n + 1))
    stream = stream_bench.ClipStream(clip_dir, drones)
    timeline, t = [], stream.t0 + 0.05
    while t <= min(stream.t1, stream.t0 + args.clip_seconds):
        views = stream.views_at(t)
        if len(views) == len(drones):
            timeline.append([dict(v, image=np.ascontiguousarray(v["image"])) for v in views])
        t += 0.05
    stream.close()
    if len(timeline) < 4:
        raise SystemExit("too few usable frames in the clip")
    if args.extra > 0:
        # More views than the clip has, for load only: copies of its own, a few metres
        # along and up, under ids of their own. Their pixels do not belong where their
        # poses put them, so the mosaic is not one to look at.
        for views in timeline:
            for k in range(args.extra):
                src = views[k % len(drones)]
                shift = 3.0 * (1 + k // len(drones))
                pos = (src["pos"][0] + shift, src["pos"][1] + 0.5 * shift, src["pos"][2])
                views.append(dict(src, drone_id=len(drones) + k, slot=len(drones) + k,
                                  pos=pos, image=src["image"].copy()))
        drones = drones + list(range(len(drones) + 1, len(drones) + args.extra + 1))
    return session, drones, timeline


class PanoramaSide:
    """PyUniSharingFast's panorama read: a new sequence number, at an acceptable size."""

    def __init__(self, mm, read_interval, canvas, ceiling):
        self.mm = mm
        self.read_interval = read_interval
        self.canvas = canvas
        self.next_receive = time.perf_counter()
        self.last_seq = 0
        self.seqs = []
        self.sizes = {}
        self.refused = 0
        self.last_refused_seq = 0
        self.image = None
        self.shape = None
        self.set_ceiling(ceiling)

    def set_ceiling(self, ceiling):
        self.ceiling = ceiling
        self.last_seq = 0

    def acceptable(self, w, h):
        """PanoramaSizeAcceptable for PLANAR: the canvas' aspect, under the ceiling."""
        if w < 2 or h < 2 or w > self.ceiling[0] or h > self.ceiling[1]:
            return False
        return abs(w * self.canvas[1] - h * self.canvas[0]) <= max(self.canvas)

    def frame(self, now):
        mm = self.mm
        if now < self.next_receive:
            return
        mm.seek(0)
        if struct.unpack("<i", mm.read(4))[0] != 0:
            return
        mm.seek(0)
        mm.write(struct.pack("<i", 1))
        mm.seek(4)
        word, w, h = struct.unpack("<Iii", mm.read(12))
        good = bool(word & 1)
        seq = (word >> st.PANORAMA_SEQ_SHIFT) & st.PANORAMA_SEQ_MASK
        fits = self.acceptable(w, h)
        if good and not fits and seq != self.last_refused_seq:
            self.refused += 1
            self.last_refused_seq = seq
        upload = good and fits and (seq == 0 or seq != self.last_seq)
        if upload:
            # LoadRawTextureData's copy, so the flag is held about as long as Unity holds it.
            self.image = np.frombuffer(mm, dtype=np.uint8, count=w * h * 3,
                                       offset=st.PANORAMA_HEADER_BYTES).copy()
            self.shape = (h, w)
            self.last_seq = seq
            self.seqs.append((now, seq))
            self.sizes[(w, h)] = self.sizes.get((w, h), 0) + 1
        mm.seek(0)
        mm.write(struct.pack("<i", 0))
        if upload or not good:
            self.next_receive = max(self.next_receive + self.read_interval,
                                    now - self.read_interval)

    def picture(self):
        """The last uploaded panorama as a BGR, top-down image."""
        if self.image is None:
            return None
        rgb = self.image.reshape(self.shape[0], self.shape[1], 3)
        return cv2.cvtColor(cv2.flip(rgb, 0), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Time the PLANAR bridge end to end, with a "
                                             "stand-in for Unity.")
    ap.add_argument("--clip", default="nadir_6drone_30m")
    ap.add_argument("--recordings", default=bench.DEFAULT_RECORDINGS)
    ap.add_argument("--drones", type=int, nargs="+", default=None)
    ap.add_argument("--extra", type=int, default=0,
                    help="add this many copies of the clip's views, for load (ten views "
                         "from a six-drone clip is --extra 4)")
    ap.add_argument("--clip-seconds", type=float, default=6.0,
                    help="seconds of the clip to hold in memory; played forwards and back")
    ap.add_argument("--frame", type=parse_size, default=None,
                    help="resize the feeds to WxH (the simulator's blocks are 1152x648)")
    ap.add_argument("--canvas", type=parse_size, default=(1200, 800),
                    help="planarCanvasWidth x Height")
    ap.add_argument("--scale", type=float, default=2.5,
                    help="planarMaxRenderScale: the ceiling on the output size")
    ap.add_argument("--mpp", type=float, default=None,
                    help="planarMetresPerPixel (Unity's default is 0.05); default: sized to the clip")
    ap.add_argument("--canvas-mode", choices=["fixed", "autofit"], default="autofit",
                    help="planarCanvasMode (Unity's default is autofit)")
    ap.add_argument("--wire", type=parse_size, default=None,
                    help="publish this panorama size instead of canvas x scale, as a Unity "
                         "that predates planarRenderScale does (DJIScene's was 1600x600)")
    ap.add_argument("--stitcher-dir", default=ROOT,
                    help="run the StitcherThreading.py in this folder instead (another "
                         "checkout, for a before/after on one harness)")
    ap.add_argument("--blend", choices=sorted(BLENDS), default="multiband")
    ap.add_argument("--align", action="store_true",
                    help="both estimator switches on (the learned aligner and the tracker)")
    ap.add_argument("--send-hz", type=float, default=0.0,
                    help="rewrite every view at this rate, as the simulator does; 0 = write "
                         "a block when its feed has a new frame, as the real fleet does")
    ap.add_argument("--read-interval", type=float, default=0.0333)
    ap.add_argument("--unity-hz", type=float, default=72.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=6.0)
    ap.add_argument("--switch-scale", type=float, default=None,
                    help="half-way through, republish at this planarRenderScale")
    ap.add_argument("--gpu-load-ms", type=float, default=0.0)
    ap.add_argument("--save", default=None, help="write the last delivered panorama here")
    ap.add_argument("--suffix", default="_pbench")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    if not args.suffix:
        raise SystemExit("--suffix must not be empty: the real section names belong to Unity")

    session, drones, timeline = load_clip(args)
    meta_s = session.get("meta", session)
    scene = meta_s.get("scene_plane") or {}
    standoff = float(scene.get("standoff_m", 30.0))
    intr = bench.wire_intrinsics(session)
    frame = (bench.WIRE_W, bench.WIRE_H)
    mpp = args.mpp or bench.make_config(session, standoff, args.canvas, False, False)[
        "metres_per_pixel"]
    wire = args.wire or wire_size(args.canvas, args.scale)
    plane = {"plane_normal": (0.0, 0.0, -1.0), "plane_d": 0.0,
             "plane_mode": ps.PLANE_MODE_FORMATION_RELATIVE,
             "gimbal_pitch": float(meta_s.get("gimbal_pitch", 0.0)),
             "centre_drone_id": drones[len(drones) // 2] - 1}
    if meta_s.get("sim") and scene.get("plane_n_unity"):
        plane.update(plane_normal=tuple(scene["plane_n_unity"]),
                     plane_d=float(scene["plane_d_unity"]), plane_mode=1)

    print(f"bench: {args.clip}, {len(drones)} views of {frame[0]}x{frame[1]}, canvas "
          f"{args.canvas[0]}x{args.canvas[1]} at {mpp * 100:.1f} cm/px, panorama "
          f"up to {wire[0]}x{wire[1]}" + ("" if args.wire else f" (scale {args.scale:g})")
          + f", canvas {args.canvas_mode}"
          + f", blend {args.blend}, "
          f"estimators {'on' if args.align else 'off'}, "
          + (f"every view at {args.send_hz:g} Hz" if args.send_hz > 0 else "blocks on new frames")
          + f", sections *{args.suffix}")

    meta = mmap.mmap(-1, st.METADATA_SIZE, "MetadataSharedMemory" + args.suffix)
    blocks = mmap.mmap(-1, st.BLOCK_SECTION_BYTES, "BlockSharedMemory" + args.suffix)
    pano = mmap.mmap(-1, st.PANORAMA_SECTION_BYTES, "PanoramaSharedMemory" + args.suffix)
    # Sections of a previous run can outlive it for as long as anything holds them.
    pano.seek(0)
    pano.write(struct.pack("<iIii", 0, 0, 0, 0))
    write_metadata(meta, args, frame, wire, len(drones), intr, mpp, standoff)
    dyn_seq = write_dynamic(meta, 0, plane)
    invalidate_slots(blocks, 0, st.BLOCK_SLOT_CAPACITY)

    env = dict(_CHILD_ENV, STITCH_SHM_SUFFIX=args.suffix, STITCH_PRINT_RATES="1",
               PYTHONUNBUFFERED="1")
    measuring = {"on": False}
    proc = subprocess.Popen([args.python, "-u", "-B", "-W", "ignore", "StitcherThreading.py"],
                            cwd=args.stitcher_dir, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    latest = {"stitch": None}
    planar_lines, log_tail, loop_ms = [], [], []

    def pump():
        for line in proc.stdout:
            line = line.rstrip()
            log_tail.append(line)
            del log_tail[:-60]
            if line.startswith("[stitching_thread]"):
                latest["stitch"] = line
                # "... | last loop 0.012s": one whole pass of the render thread, which is
                # printed by every revision of the stitcher and so is what a before/after
                # on this harness compares.
                if "last loop" in line and measuring["on"]:
                    try:
                        loop_ms.append(1e3 * float(line.split("last loop")[1].strip().rstrip("s")))
                    except ValueError:
                        pass
            elif "[PLANAR]" in line:
                planar_lines.append(line)
                del planar_lines[:-12]

    threading.Thread(target=pump, daemon=True).start()
    stop = threading.Event()
    if args.gpu_load_ms > 0:
        threading.Thread(target=gpu_load_thread, args=(stop, args.gpu_load_ms, args.unity_hz),
                         daemon=True).start()

    side = PanoramaSide(pano, args.read_interval, args.canvas, wire)
    frame_period = 1.0 / args.unity_hz
    t_start = time.perf_counter()
    next_frame = t_start
    next_send = t_start
    heartbeat = 0
    written = skipped = 0
    last_written = [None] * len(drones)
    first_pano = measure_from = None
    base = None
    stitch_hz = []
    last_print = 0.0
    switched_at = None
    span_frames = len(timeline) - 1

    try:
        while True:
            now = time.perf_counter()
            if now < next_frame:
                time.sleep(max(0.0, next_frame - now))
                continue
            next_frame += frame_period
            if next_frame < now - frame_period:
                next_frame = now + frame_period

            heartbeat = (heartbeat + 1) & 0xFFFFFFFF
            meta.seek(st.META_HEARTBEAT_OFFSET)
            meta.write(struct.pack("<I", heartbeat))
            dyn_seq = write_dynamic(meta, dyn_seq, plane)

            # The clip, forwards then backwards, so the poses never jump at a loop point.
            step = int((now - t_start) / 0.05)
            k = step % (2 * span_frames)
            views = timeline[k if k <= span_frames else 2 * span_frames - k]
            capture = now - t_start
            send_all = False
            if args.send_hz > 0 and now >= next_send:
                next_send = max(next_send + 1.0 / args.send_hz, now - 1.0 / args.send_hz)
                send_all = True
            for slot, view in enumerate(views):
                if args.send_hz > 0:
                    due = send_all
                else:
                    due = last_written[slot] is not view["image"]
                if not due:
                    continue
                if write_block(blocks, slot, view, capture):
                    written += 1
                    last_written[slot] = view["image"]
                else:
                    skipped += 1

            side.frame(now)

            if proc.poll() is not None:
                print("\n".join(log_tail))
                raise SystemExit(f"StitcherThreading.py exited with {proc.returncode}")

            if first_pano is None and side.seqs:
                first_pano = now
                print(f"  first panorama after {now - t_start:.1f} s; warming up "
                      f"{args.warmup:g} s")
            if first_pano is not None and measure_from is None and now - first_pano >= args.warmup:
                measure_from = now
                measuring["on"] = True
                base = (len(side.seqs), written, skipped, side.refused)
                print("  measuring ...")

            if (args.switch_scale is not None and switched_at is None
                    and measure_from is not None and now - measure_from >= 0.5 * args.duration):
                # What SetStitcherType and OnValidate do: republish, then read at the new size.
                switched_at = now
                wire = wire_size(args.canvas, args.switch_scale)
                write_metadata(meta, args, frame, wire, len(drones), intr, mpp, standoff)
                side.set_ceiling(wire)
                print(f"  panorama ceiling switched to {wire[0]}x{wire[1]}")

            if measure_from is not None and now - last_print >= 1.0:
                last_print = now
                line = latest["stitch"]
                if line and " Hz" in line:
                    try:
                        stitch_hz.append(float(line.split("]")[1].split("Hz")[0]))
                    except ValueError:
                        pass

            if measure_from is not None and now - measure_from >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if measure_from is None:
        print("\n".join(log_tail))
        raise SystemExit("no panorama was measured")

    span = time.perf_counter() - measure_from
    seqs = side.seqs[base[0]:]
    delivered = len(seqs)
    gaps = np.diff([t for t, _s in seqs]) * 1e3 if delivered > 2 else np.zeros(1)
    # Sequence numbers Python wrote but Unity never uploaded (overwritten between reads).
    numbers = [s for _t, s in seqs]
    produced = (numbers[-1] - numbers[0]) % st.PANORAMA_SEQ_MASK + 1 if numbers else 0

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")

    print()
    print(f"result over {span:.0f} s")
    print(f"  stitcher      : {med(stitch_hz):.1f} Hz (median of its 5 s averages), render "
          f"thread pass {med(loop_ms):.1f} ms (median of {len(loop_ms)} printed)")
    for line in planar_lines[-3:]:
        print(f"  {line.strip()}")
    print(f"  blocks        : {(written - base[1]) / span:.1f}/s written, "
          f"{(skipped - base[2]) / span:.2f}/s skipped on a busy flag")
    print(f"  panoramas     : {produced / span:.1f}/s written by Python, "
          f"{delivered / span:.1f}/s uploaded, "
          f"{max(0, produced - delivered) / span:.1f}/s overwritten before a read")
    print(f"  upload spacing: median {np.median(gaps):.1f} ms, 95th percentile "
          f"{np.percentile(gaps, 95):.1f} ms, longest {gaps.max():.1f} ms")
    print("  sizes         : " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in
                                           sorted(side.sizes.items())))
    if args.switch_scale is not None:
        print(f"  ceiling change: {side.refused - base[3]} panorama(s) arrived at a size the new "
              f"ceiling does not allow and were refused")
    if args.save:
        picture = side.picture()
        if picture is not None:
            cv2.imwrite(args.save, picture)
            print(f"  wrote {args.save} ({picture.shape[1]}x{picture.shape[0]})")


if __name__ == "__main__":
    main()
