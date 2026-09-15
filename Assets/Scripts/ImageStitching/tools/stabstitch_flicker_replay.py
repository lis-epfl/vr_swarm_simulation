"""
Measure how steady the STABSTITCH panorama is, on real frames recorded from Unity.

    cd Assets/Scripts/ImageStitching
    # 1. With Unity in Play (STABSTITCH), record the three stitch slots -- read-only, it never
    #    takes a block flag, so it is safe beside a running stitcher:
    python tools/stabstitch_flicker_replay.py record D:\\stab_rec\\hover 15
    # 2. Replay the recording through StabStitcher on a simulated clock and score it:
    python tools/stabstitch_flicker_replay.py replay D:\\stab_rec\\hover --switch

Why
---
"It flickers more" is not something a per-stage timing or an equivalence test can see. This
replays exactly what the render thread does -- one render per recorded send, frames admitted to
the temporal window on NET_FRAME_PERIOD, a warp update per admission -- and records, for every
displayed frame, where the three views' source control points land in the panorama. Over a
hovering swarm those should barely move, so their frame-to-frame motion is the flicker and warp
the pilot sees, in panorama pixels. The recording is 1.8 MB per view per send (a 15 s hover at
1024x576 is 2.4 GB), so keep it off the repo.

``--switch`` replaces the middle third of the sends with a different, still stitchable triplet
(every view mirrored, order reversed): every slot's content changes at once, as when the pilot
yaws onto new drones, which a hover recording otherwise never contains.

Scores (panorama px)
  jitter      RMS control-point displacement between consecutive displayed frames
  shape       the part of it not explained by one translation + scale (the warp changing shape)
  roughness   the paper's smoothness score (test_metric_ssd.py) on the displayed path
  start_err   RMS error over each segment's first 0.5 s, against its geometry 1-1.5 s in
  concat_s    time each segment showed the side-by-side concat instead of a panorama

Measured on a 15 s hover at 1024x576 (2026-09-15): before the temporal fixes jitter 0.48, shape
0.31, roughness 0.87, a switch 16-33 px off for its first 10 frames and 0.38 s of concat at a
cold start; after, 0.38 / 0.22 / 0.50, a switch within 1.3-3.6 px, no concat. BUFFER_LEN 5, 9
and 11 all scored worse than the trained 7.
"""

import argparse
import json
import mmap
import os
import struct
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------------------

def record(out, duration):
    import StitcherThreading as stt

    stride, cap = stt.BLOCK_SLOT_STRIDE, stt.BLOCK_SLOT_CAPACITY
    blocks = mmap.mmap(-1, cap * stride, stt.shm_name("BlockSharedMemory"), access=mmap.ACCESS_READ)
    meta = mmap.mmap(-1, stt.METADATA_SIZE, stt.shm_name("MetadataSharedMemory"),
                     access=mmap.ACCESS_READ)
    meta.seek(0)
    w, h = struct.unpack("<ii", meta.read(8))
    if w <= 0 or h <= 0:
        raise SystemExit("Unity is not publishing (metadata sizes are zero): enter Play mode first")
    n = w * h * 3
    header = stt.BLOCK_HEADER_SIZE_V2
    os.makedirs(out, exist_ok=True)

    frames = {0: [], 1: [], 2: []}
    last = [None, None, None]
    torn = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        for s in range(3):
            base = s * stride
            blocks.seek(base)
            flag, did = struct.unpack("<ii", blocks.read(8))
            blocks.seek(base + stt.BLOCK_CAPTURE_TIME_OFFSET)
            ct = struct.unpack("<f", blocks.read(4))[0]
            if flag != 0 or did < 0 or ct == last[s]:
                continue
            img = np.frombuffer(blocks, dtype=np.uint8, count=n, offset=base + header).copy()
            # Unity writes the header (capture time included) before the image, so a write that
            # began during the copy shows up as a changed capture time or a held flag.
            blocks.seek(base)
            flag2 = struct.unpack("<i", blocks.read(4))[0]
            blocks.seek(base + stt.BLOCK_CAPTURE_TIME_OFFSET)
            if flag2 != 0 or struct.unpack("<f", blocks.read(4))[0] != ct:
                torn += 1
                continue
            last[s] = ct
            frames[s].append((ct, did, img.reshape(h, w, 3)))
        time.sleep(0.002)

    # One send = the three cameras captured in one Unity Update, a few ms apart.
    sends = []
    for ct, did, img in frames[1]:
        row = [None, (ct, did, img), None]
        for s in (0, 2):
            best = min(frames[s], key=lambda f: abs(f[0] - ct), default=None)
            if best is not None and abs(best[0] - ct) < 0.02:
                row[s] = best
        if row[0] is not None and row[2] is not None:
            sends.append(row)
    if not sends:
        raise SystemExit("no complete sends recorded (is the stitcher STABSTITCH?)")
    np.save(os.path.join(out, "frames.npy"), np.stack([np.stack([r[s][2] for s in range(3)])
                                                         for r in sends]))
    times = np.array([r[1][0] for r in sends], dtype=np.float64)
    ids = np.array([[r[s][1] for s in range(3)] for r in sends])
    np.savez(os.path.join(out, "meta.npz"), times=times, ids=ids)
    print(f"{w}x{h}: {len(sends)} sends over {times[-1] - times[0]:.1f} s, {torn} torn reads "
          f"discarded, triplets {np.unique(ids, axis=0).tolist()}")


# --------------------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------------------

def replay(args):
    import cv2
    import torch
    import torch.nn.functional as F
    import StabStitcher as ss

    frames = np.load(os.path.join(args.rec, "frames.npy"), mmap_mode="r")    # [T, 3, H, W, 3]
    meta = np.load(os.path.join(args.rec, "meta.npz"))
    times = meta["times"] - meta["times"][0]
    rec_ids = [tuple(int(v) for v in row) for row in meta["ids"]]
    T = len(times)
    wire = (args.wire[1], args.wire[0])

    clock = {"t": 0.0}
    ss.time = types.SimpleNamespace(perf_counter=lambda: clock["t"], sleep=time.sleep, time=time.time)
    if args.period is not None:
        ss.NET_FRAME_PERIOD = args.period
        ss.VIDEO_GAP_S = 4 * args.period

    st = ss.StabStitcher(timing=False)
    for net in (st.spatial_net, st.temporal_net, st.smooth_net):
        net.cuda()
    if args.buffer != st.BUFFER_LEN:
        st.BUFFER_LEN = args.buffer
        for name in ("_buf_img1", "_buf_img2", "_buf_img3", "_smesh12_1", "_smesh12_2",
                     "_smesh23_1", "_smesh23_2", "_tsm12_1", "_tsm12_2", "_tsm23_1", "_tsm23_2"):
            setattr(st, name, type(getattr(st, name))(maxlen=args.buffer))
    st.quality_threshold = args.threshold
    if args.antialias is not None:
        st.lr_antialias = bool(args.antialias)
    if args.no_pad:
        st.pad_new_video = False

    bounds = [0, T // 3, 2 * T // 3, T] if args.switch else [0, T]

    def segment(i):
        return max(k for k in range(len(bounds) - 1) if i >= bounds[k])

    def triplet(i):
        imgs = [np.ascontiguousarray(frames[i, s]) for s in range(3)]
        if args.scale:
            hh = int(round(imgs[0].shape[0] * args.scale / imgs[0].shape[1]))
            imgs = [cv2.resize(im, (args.scale, hh), interpolation=cv2.INTER_AREA) for im in imgs]
        ids = rec_ids[i]
        if args.switch and segment(i) == 1:
            imgs = [np.ascontiguousarray(im[:, ::-1]) for im in imgs[::-1]]
            ids = tuple(1000 + v for v in ids[::-1])
        return imgs, ids

    def displayed(wp):
        oh, ow = wp['out_size']
        P = torch.cat([wp['m1_final'], wp['m2_final'], wp['m3_final']], 0).reshape(-1, 2)
        scale = torch.tensor([wire[1] / ow, wire[0] / oh], device=P.device)
        return (P * scale).cpu().numpy().astype(np.float64)

    pts, concat, psnr = [], [], []
    last_wp = None
    with torch.no_grad():
        for i in range(T):
            clock["t"] = float(times[i])
            imgs, ids = triplet(i)
            pano, _, _ = st.stab_pano(imgs, [0, 1], [1, 2], out_size=wire,
                                      view_ids=None if args.no_view_ids else ids)
            if st._pending:
                st._update_warps()
            wp = st._cached_warp
            is_concat = pano is not None and not isinstance(pano, ss.WireReadyPanorama)
            concat.append(is_concat)
            if wp is None or is_concat:
                pts.append(None)
                continue
            pts.append(displayed(wp))
            if wp is not last_wp:
                last_wp = wp
                psnr.append(wp['quality_score'])

    dt = float(np.median(np.diff(times)))
    n0 = int(round(0.5 / dt))
    jit, shape, rough, start_err, concat_s = [], [], [], [], []
    for k in range(len(bounds) - 1):
        concat_s.append(round(sum(concat[bounds[k]:bounds[k + 1]]) * dt, 3))
        idx = [i for i in range(bounds[k], bounds[k + 1]) if pts[i] is not None]
        if len(idx) < max(20, int(1.6 / dt)):
            continue
        P = np.stack([pts[i] for i in idx])
        ref = P[int(round(1.0 / dt)):int(round(1.5 / dt))].mean(0)
        start_err.append(round(float(np.sqrt(((P[:n0] - ref) ** 2).sum(-1).mean())), 2))
        Q = P[n0:]
        d = Q[1:] - Q[:-1]
        jit.append((d ** 2).sum(-1).mean())
        for a, b in zip(Q[:-1], Q[1:]):
            A = a - a.mean(0)
            s = (A * (b - b.mean(0))).sum() / max(1e-9, (A * A).sum())
            shape.append(((b - (b.mean(0) + s * A)) ** 2).sum(-1).mean())
        if len(Q) >= 7:
            mid = Q[3:-3]
            l2 = lambda o: ((Q[3 + o:len(Q) - 3 + o] - mid) ** 2).sum(-1).mean()
            rough.append((l2(-3) + l2(3)) * 0.1 + (l2(-2) + l2(2)) * 0.3 + (l2(-1) + l2(1)) * 0.9)

    result = dict(label=args.label, jitter=round(float(np.sqrt(np.mean(jit))), 3),
                  shape=round(float(np.sqrt(np.mean(shape))), 3),
                  roughness=round(float(np.mean(rough)), 3), psnr=round(float(np.mean(psnr)), 2),
                  start_err=start_err, concat_s=concat_s)
    print(json.dumps(result))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(result) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Record Unity stitch frames / score panorama stability.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record the three STABSTITCH slots from a playing Unity")
    r.add_argument("out")
    r.add_argument("duration", type=float, nargs="?", default=15.0)
    p = sub.add_parser("replay", help="replay a recording through StabStitcher and score it")
    p.add_argument("rec")
    p.add_argument("--label", default="replay")
    p.add_argument("--wire", default="1920x720", type=lambda t: tuple(int(v) for v in t.split("x")))
    p.add_argument("--switch", action="store_true", help="simulate a triplet change mid-run")
    p.add_argument("--buffer", type=int, default=7, help="temporal window length (trained: 7)")
    p.add_argument("--period", type=float, default=None, help="NET_FRAME_PERIOD override")
    p.add_argument("--antialias", type=int, choices=(0, 1), default=None)
    p.add_argument("--no-pad", action="store_true", help="wait for real frames at a new video")
    p.add_argument("--no-view-ids", action="store_true", help="do not report triplet changes")
    p.add_argument("--scale", type=int, default=0, help="resize frames to this width (INTER_AREA)")
    p.add_argument("--threshold", type=float, default=16.0, help="PSNR gate")
    p.add_argument("--json", default=None, help="append the scores to this file")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.out, args.duration)
    else:
        os.chdir(ROOT)
        replay(args)


if __name__ == "__main__":
    main()
