"""
Stand in for Unity so the whole STABSTITCH bridge can be timed without the editor.

    cd Assets/Scripts/ImageStitching
    python tools/stabstitch_bridge_bench.py --frame 1024x576 --wire 1920x720 --duration 40

What it does
------------
Plays PyUniSharingFast's part of the shared-memory contract -- metadata, three posed blocks
at ``--send-hz``, a heartbeat at ``--unity-hz``, panorama reads -- and runs the real,
unmodified ``StitcherThreading.py`` against it as a subprocess. So the numbers it prints are
the whole Python side under its real threading (block read, render, warp, panorama write,
one GIL), not the per-stage timings ``stabstitch_selftest.py`` measures in isolation.

It uses shared-memory sections of its own (``STITCH_SHM_SUFFIX``, default ``_bench``).
Named sections are machine-global: on the real names this would be a second producer inside
a live editor's session, corrupting its frames. With the suffix it can run beside Unity in
Play mode -- though the two then share the GPU, which is itself a useful measurement.

What it reports
---------------
* the stitcher's own stitch and warp rates (its once-a-second console lines);
* on the Unity side, the panoramas actually delivered per second, and how the panorama
  sequence number changes what Unity uploads: the old policy uploaded on every read timer
  tick, so it re-uploaded panoramas already on screen and missed ones overwritten between
  ticks; the sequence-gated policy uploads each new panorama once, on the next frame.

What it does not model: Unity's own frame cost. The heartbeat here ticks at a fixed rate,
so the "unity fps" the stitcher prints is the bench's, and ``--gpu-load-ms`` is only a crude
stand-in for the GPU time the headset rendering takes.
"""

import os

# The environment the stitcher subprocess gets, captured before anything imports torch:
# torch's DLL loader rewrites PATH in the importing process, and a child that inherits that
# PATH fails its own torch import on Windows (WinError 1114 loading shm.dll).
_CHILD_ENV = dict(os.environ)

import argparse
import math
import mmap
import random
import struct
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

# Constants come from the stitcher itself rather than being restated: this file writes the
# layout StitcherThreading.py reads, and a restated copy is exactly what drifts.
import StitcherThreading as st  # noqa: E402

PRINT_PERIOD = 1.0


def parse_size(text):
    w, h = (int(v) for v in text.lower().split("x"))
    return w, h


# --------------------------------------------------------------------------------------
# Unity's half of the contract
# --------------------------------------------------------------------------------------

def put_string(buf, offset, text, size=64):
    raw = text.encode("utf-8")[:size]
    buf[offset:offset + size] = raw + b"\x00" * (size - len(raw))


def write_metadata(mm, args, frame, wire):
    """
    The metadata layout, in the order readMetadataMemory reads it: a sequential v1 prefix
    (0..252) and a tail addressed by absolute offset. Values are STABSTITCH's; the planar
    fields stay zero, which is what a STABSTITCH scene publishes for everything it ignores.
    """
    b = bytearray(st.METADATA_SIZE)
    o = 0
    struct.pack_into("<iiiii", b, o, frame[0], frame[1], 3, wire[0], wire[1]); o += 20
    put_string(b, o, "STABSTITCH"); o += 64
    struct.pack_into("<B", b, o, 0); o += 1                      # isCylindrical
    put_string(b, o, "BF"); o += 64
    struct.pack_into("<B", b, o, 0); o += 1                      # isRANSAC
    struct.pack_into("<i", b, o, 50); o += 4                     # checks
    struct.pack_into("<ff", b, o, 0.7, 0.1); o += 8              # ratio, score thresholds
    struct.pack_into("<i", b, o, 1000); o += 4                   # focal
    o += 1                                                       # reserved byte
    put_string(b, o, args.fusion); o += 64
    struct.pack_into("<i", b, o, args.blur_kernel); o += 4
    struct.pack_into("<f", b, o, args.blur_sigma); o += 4
    struct.pack_into("<i", b, o, args.border); o += 4
    struct.pack_into("<B", b, o, 1 if args.quality else 0); o += 1
    struct.pack_into("<f", b, o, args.quality_threshold); o += 4
    struct.pack_into("<f", b, o, 0.0); o += 4                    # head angle: centre view
    struct.pack_into("<B", b, o, 1); o += 1                      # print rate
    assert o == 253, o

    struct.pack_into("<ii", b, st.META_BLOCK_HEADER_SIZE_OFFSET, st.BLOCK_HEADER_SIZE_V2,
                     st.PLANAR_WIRE_VERSION)
    struct.pack_into("<ii", b, st.META_BLOCK_SLOT_CAPACITY_OFFSET, st.BLOCK_SLOT_CAPACITY,
                     st.BLOCK_SLOT_STRIDE)
    struct.pack_into("<ii", b, st.META_BLOCK_SECTION_BYTES_OFFSET, st.BLOCK_SECTION_BYTES,
                     st.PANORAMA_SECTION_BYTES)
    mm.seek(0)
    mm.write(bytes(b))


def make_frames(frame, count):
    """
    ``count`` frames per view from the saved debug inputs, with a little per-frame motion
    and exposure drift so TemporalNet has real work -- the same construction the self-test
    uses. Views are left / centre / right.
    """
    w, h = frame
    views = []
    for i in range(3):
        img = cv2.imread(os.path.join(ROOT, f"debug_input_drone_{i}.jpg"))
        if img is None:
            raise SystemExit(f"missing debug_input_drone_{i}.jpg beside StitcherThreading.py")
        img = cv2.resize(img, (w, h))
        seq = []
        for k in range(count):
            # A back-and-forth sweep, so the loop point is not a jump.
            phase = math.sin(2 * math.pi * k / count)
            dx = 3.0 * phase * (1 if i != 1 else 0.5) + 0.3 * math.sin(0.7 * k + i)
            dy = 1.5 * phase + 0.2 * math.cos(0.5 * k + i)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            f = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            f = np.clip(f.astype(np.int16) + int(2 * phase), 0, 255).astype(np.uint8)
            seq.append(np.ascontiguousarray(f))
        views.append(seq)
    return views


def write_block(mm, slot, drone_id, heading, image, capture_time):
    """One block under the flag handshake. False if the consumer holds the flag (skipped,
    exactly as OnBlockReadback skips a slot Python is mid-read on)."""
    base = slot * st.BLOCK_SLOT_STRIDE
    mm.seek(base)
    if struct.unpack("<i", mm.read(4))[0] != 0:
        return False
    mm.seek(base)
    mm.write(struct.pack("<i", 1))
    # droneId | heading | camPos xyz | camRot xyzw | captureTime | poseStatus  (offsets 4..47)
    header = struct.pack("<if3f4ffi", drone_id, heading, 0.0, 10.0, 0.0,
                         0.0, 0.0, 0.0, 1.0, capture_time, st.POSE_VALID)
    assert 4 + len(header) == st.BLOCK_HEADER_SIZE_V2
    mm.seek(base + 4)
    mm.write(header)
    mm.seek(base + st.BLOCK_HEADER_SIZE_V2)
    mm.write(memoryview(image).cast("B"))
    mm.seek(base)
    mm.write(struct.pack("<i", 0))
    return True


def invalidate_slots(mm, first, capacity):
    """droneId = -1 on every slot this producer does not fill, as Unity marks them."""
    for slot in range(first, capacity):
        base = slot * st.BLOCK_SLOT_STRIDE
        mm.seek(base + 4)
        mm.write(struct.pack("<i", -1))


class PanoramaSide:
    """
    PyUniSharingFast's panorama read, plus a shadow of the policy it replaced so the two can
    be compared on the same stream.

    Sequence-gated (current): after readInterval, look every frame; upload only a panorama
    whose sequence number differs from the last one uploaded.
    Timer (previous): every readInterval, upload whatever is there.
    """

    def __init__(self, mm, read_interval, wire):
        self.mm = mm
        self.read_interval = read_interval
        self.size = wire[0] * wire[1] * 3
        self.next_receive = time.perf_counter()
        self.last_seq = 0
        self.seen = set()
        self.uploads = 0
        self.looks = 0
        self.old_next = self.next_receive
        self.old_last = 0
        self.old_uploads = 0
        self.old_redundant = 0
        self.old_seen = set()
        self.max_seq = 0
        self.min_seq = None

    def frame(self, now):
        mm = self.mm
        # Shadow of the old timer policy: it would have read (and uploaded) here.
        if now >= self.old_next:
            mm.seek(4)
            word = struct.unpack("<I", mm.read(4))[0]
            if word & 1:
                seq = (word >> st.PANORAMA_SEQ_SHIFT) & st.PANORAMA_SEQ_MASK
                self.old_uploads += 1
                if seq == self.old_last:
                    self.old_redundant += 1
                self.old_last = seq
                self.old_seen.add(seq)
            self.old_next += self.read_interval

        if now < self.next_receive:
            return
        mm.seek(0)
        if struct.unpack("<i", mm.read(4))[0] != 0:
            return
        mm.seek(0)
        mm.write(struct.pack("<i", 1))
        mm.seek(4)
        word = struct.unpack("<I", mm.read(4))[0]
        good = bool(word & 1)
        seq = (word >> st.PANORAMA_SEQ_SHIFT) & st.PANORAMA_SEQ_MASK
        upload = good and (seq == 0 or seq != self.last_seq)
        self.looks += 1
        if upload:
            # LoadRawTextureData's copy, so the bench's own flag hold is realistic.
            np.frombuffer(mm, dtype=np.uint8, count=self.size, offset=8).copy()
            self.last_seq = seq
            self.uploads += 1
            self.seen.add(seq)
            self.max_seq = max(self.max_seq, seq)
            self.min_seq = seq if self.min_seq is None else min(self.min_seq, seq)
        mm.seek(0)
        mm.write(struct.pack("<i", 0))
        if upload or not good:
            self.next_receive = max(self.next_receive + self.read_interval,
                                    now - self.read_interval)

    def produced(self):
        """Panoramas Python wrote over the window (from the sequence numbers observed)."""
        if self.min_seq is None:
            return 0
        return self.max_seq - self.min_seq + 1


# --------------------------------------------------------------------------------------
# Optional GPU contention
# --------------------------------------------------------------------------------------

def gpu_load_thread(stop, ms, hz):
    """Burn roughly ``ms`` of GPU time per ``1/hz`` s. A crude stand-in for headset rendering."""
    import torch
    import torch.nn.functional as F
    x = torch.randn(1, 16, 1024, 1024, device="cuda")
    w = torch.randn(16, 16, 3, 3, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    F.conv2d(x, w, padding=1)
    torch.cuda.synchronize()
    per = max(1e-4, time.perf_counter() - t0)
    reps = max(1, int(round(ms / 1000.0 / per)))
    period = 1.0 / hz
    nxt = time.perf_counter()
    while not stop.is_set():
        for _ in range(reps):
            F.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        nxt += period
        time.sleep(max(0.0, nxt - time.perf_counter()))


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Time the STABSTITCH bridge end to end, with a "
                                             "stand-in for Unity.")
    ap.add_argument("--frame", default="1024x576", help="block (feed) resolution WxH")
    ap.add_argument("--wire", default="1920x720", help="panorama resolution WxH")
    ap.add_argument("--send-hz", type=float, default=30.0, help="Unity block publish rate")
    ap.add_argument("--read-interval", type=float, default=0.0333,
                    help="PyUniSharingFast.readInterval, seconds")
    ap.add_argument("--unity-hz", type=float, default=72.0,
                    help="frame rate of the stand-in (heartbeat, readback completion, panorama "
                         "reads); 72 is the Quest Pro's default refresh")
    ap.add_argument("--duration", type=float, default=40.0, help="measured seconds, after warm-up")
    ap.add_argument("--warmup", type=float, default=8.0,
                    help="seconds after the first panorama before measuring")
    ap.add_argument("--fusion", default="REFERENCE_BLEND")
    ap.add_argument("--blur-kernel", type=int, default=None,
                    help="blurKernelSize; default scales 41 by the block width / 768")
    ap.add_argument("--blur-sigma", type=float, default=None, help="default scales 15 likewise")
    ap.add_argument("--border", type=int, default=None, help="default scales 60 likewise")
    ap.add_argument("--quality-threshold", type=float, default=0.0,
                    help="PSNR gate; 0 keeps the panorama on (the synthetic scene stitches at "
                         "~16 dB), so the render path is always exercised")
    ap.add_argument("--no-quality", dest="quality", action="store_false",
                    help="turn the quality estimate off entirely")
    ap.add_argument("--gpu-load-ms", type=float, default=0.0,
                    help="burn this much GPU time per stand-in frame (crude headset stand-in)")
    ap.add_argument("--suffix", default="_bench", help="STITCH_SHM_SUFFIX for the sections")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the stitcher")
    args = ap.parse_args()

    if not args.suffix:
        raise SystemExit("--suffix must not be empty: the real section names belong to Unity")

    frame, wire = parse_size(args.frame), parse_size(args.wire)
    scale = frame[0] / 768.0
    if args.blur_kernel is None:
        args.blur_kernel = int(round(41 * scale)) | 1
    if args.blur_sigma is None:
        args.blur_sigma = 15.0 * scale
    if args.border is None:
        args.border = int(round(60 * scale))

    print(f"bench: blocks {frame[0]}x{frame[1]}, panorama {wire[0]}x{wire[1]}, send "
          f"{args.send_hz:g} Hz, stand-in frame rate {args.unity_hz:g} Hz, blend "
          f"{args.blur_kernel}/{args.blur_sigma:g}/{args.border}, sections *{args.suffix}")

    meta = mmap.mmap(-1, st.METADATA_SIZE, "MetadataSharedMemory" + args.suffix)
    blocks = mmap.mmap(-1, st.BLOCK_SECTION_BYTES, "BlockSharedMemory" + args.suffix)
    pano = mmap.mmap(-1, st.PANORAMA_SECTION_BYTES, "PanoramaSharedMemory" + args.suffix)
    write_metadata(meta, args, frame, wire)
    invalidate_slots(blocks, 3, st.BLOCK_SLOT_CAPACITY)

    frames = make_frames(frame, count=30)
    headings = (-40.0, 0.0, 40.0)     # left, centre, right about head angle 0

    env = dict(_CHILD_ENV, STITCH_SHM_SUFFIX=args.suffix, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([args.python, "-u", "-B", "-W", "ignore", "StitcherThreading.py"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    latest = {"stitch": None, "warp": None, "unity": None}
    log_tail = []

    def pump():
        for line in proc.stdout:
            line = line.rstrip()
            log_tail.append(line)
            del log_tail[:-40]
            if line.startswith("[stitching_thread]"):
                latest["stitch"] = line
            elif line.startswith("[warp_thread]"):
                latest["warp"] = line

    threading.Thread(target=pump, daemon=True).start()

    stop = threading.Event()
    if args.gpu_load_ms > 0:
        threading.Thread(target=gpu_load_thread, args=(stop, args.gpu_load_ms, args.unity_hz),
                         daemon=True).start()

    side = PanoramaSide(pano, args.read_interval, wire)
    frame_period = 1.0 / args.unity_hz
    send_period = 1.0 / args.send_hz
    t_start = time.perf_counter()
    next_frame = t_start
    next_send = t_start
    pending = []                      # (due_time, slot, frame_index, capture_time)
    heartbeat = 0
    send_index = 0
    written = skipped = 0
    first_pano = None
    measure_from = None
    base = None
    stitch_hz, warp_hz = [], []
    last_print = 0.0

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

            # A send snapshots all three cameras; each readback completes 1-2 frames later,
            # and not necessarily on the same frame -- the pattern RENDER_MIN_PERIOD bounds.
            if now >= next_send:
                next_send = max(next_send + send_period, now - send_period)
                capture = now - t_start
                for slot in range(3):
                    due = now + random.randint(1, 2) * frame_period
                    pending.append((due, slot, send_index % len(frames[0]), capture))
                send_index += 1
            still = []
            for due, slot, fi, capture in pending:
                if now >= due:
                    if write_block(blocks, slot, slot, headings[slot], frames[slot][fi], capture):
                        written += 1
                    else:
                        skipped += 1
                else:
                    still.append((due, slot, fi, capture))
            pending = still

            side.frame(now)

            if proc.poll() is not None:
                print("\n".join(log_tail))
                raise SystemExit(f"StitcherThreading.py exited with {proc.returncode}")

            if first_pano is None and side.uploads > 0:
                first_pano = now
                print(f"  first panorama after {now - t_start:.1f} s; warming up "
                      f"{args.warmup:g} s")
            if first_pano is not None and measure_from is None and now - first_pano >= args.warmup:
                measure_from = now
                base = (side.uploads, len(side.seen), side.old_uploads, side.old_redundant,
                        len(side.old_seen), written, skipped, side.looks)
                side.seen.clear(); side.old_seen.clear()
                side.min_seq = None; side.max_seq = 0
                print("  measuring ...")

            if measure_from is not None and now - last_print >= PRINT_PERIOD:
                last_print = now
                for key, bucket in (("stitch", stitch_hz), ("warp", warp_hz)):
                    line = latest[key]
                    if line and " Hz" in line:
                        try:
                            bucket.append(float(line.split("]")[1].split("Hz")[0]))
                        except ValueError:
                            pass
                print(f"    {latest['stitch']}\n    {latest['warp']}")

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
    uploads = side.uploads - base[0]
    old_uploads = side.old_uploads - base[2]
    old_redundant = side.old_redundant - base[3]
    produced = side.produced()
    delivered = len(side.seen)
    old_delivered = len(side.old_seen)
    blocks_written = written - base[5]
    blocks_skipped = skipped - base[6]

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")

    print()
    print(f"result over {span:.0f} s  (blocks {frame[0]}x{frame[1]}, panorama "
          f"{wire[0]}x{wire[1]}, gpu load {args.gpu_load_ms:g} ms/frame)")
    print(f"  stitcher      : stitch {med(stitch_hz):.1f} Hz, warp {med(warp_hz):.1f} Hz "
          f"(medians of its 5 s averages)")
    print(f"  blocks        : {blocks_written / span:.1f}/s written, "
          f"{blocks_skipped / span:.2f}/s skipped on a busy flag")
    print(f"  panoramas     : {produced / span:.1f}/s written by Python")
    print(f"  seq-gated read: {uploads / span:.1f} uploads/s, {delivered / span:.1f}/s distinct, "
          f"{max(0, produced - delivered) / span:.1f}/s never shown")
    print(f"  timer read    : {old_uploads / span:.1f} uploads/s, {old_redundant / span:.1f}/s "
          f"re-uploading a panorama already shown, {max(0, produced - old_delivered) / span:.1f}/s "
          f"never shown")


if __name__ == "__main__":
    main()
