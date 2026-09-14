"""
Offline self-test for the StabStitch++ pipeline optimisations.

No Unity, no shared memory. Needs the ``stitching`` conda env (torch + CUDA) and the
``debug_input_drone_{0,1,2}.jpg`` frames beside ``StitcherThreading.py``::

    cd Assets/Scripts/ImageStitching && python tools/stabstitch_selftest.py

Why this exists
---------------
The stitcher was restructured for speed without touching the StabStitch++ networks or
weights: the nets now run incrementally (per admitted frame, cached) instead of over the
whole 7-frame window on every update; preprocessing moved to the GPU; the render samples
straight into the panorama's wire layout; and the vendored StabStitch2 code had its
per-call host constants and ``torch.inverse`` error-check syncs removed. Every one of
those is meant to be a pure speed-up. This file is where that claim is checked:

  1. GPU preprocessing == cv2.resize preprocessing (to within uint8 quantisation).
  2. Vendored edits == the pristine upstream files. StabStitch2_main is gitignored, so
     the edits live only as ``tools/stabstitch2_perf.patch``; the pristine copy is
     rebuilt by reverse-applying that patch to a temp copy and run with the same weights.
     (After re-downloading StabStitch2, re-apply the patch from ``Codes/`` with
     ``git apply <repo>/Assets/Scripts/ImageStitching/tools/stabstitch2_perf.patch``.)
  3. Incremental warp == legacy full-window warp, on the same frame sequence, whether
     the frames are ingested one per update or several at once. Compared with TF32 off
     and deterministic cuDNN: under the default (TF32, autotuned) settings the two paths
     pick different conv algorithms for their different batch sizes and the meshes drift
     by a few tenths of a pixel run to run, which is float noise, not a logic difference.
  4. Wire-ready render == legacy canvas render + cv2 resize/flip/BGR->RGB.
  5. A warp update performs at most a handful of device syncs.

It also prints the timings that motivated the change, so a regression is visible here
before it is felt in the headset.
"""

import os
import sys
import time
import math
import types
import warnings
import tempfile
import subprocess
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import StabStitcher as ss_mod  # noqa: E402
from StabStitcher import StabStitcher, WireReadyPanorama  # noqa: E402

FRAME_W, FRAME_H = 768, 432        # the sim scenes' block resolution
WIRE_SIZE = (600, 1600)            # (h, w): the sim scenes' panorama resolution
SEQ_LEN = 12                       # frames fed; the window is the last 7
SYNC_BUDGET = 12                   # device syncs per incremental warp update

_failures = []


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        _failures.append(msg)


def sync():
    torch.cuda.synchronize()


def timeit(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t) / n


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10.0 * math.log10(255.0 ** 2 / mse)


# --------------------------------------------------------------------------------------
# Frames: a short "video" made from the saved debug inputs
# --------------------------------------------------------------------------------------

def load_sequence():
    base = []
    for i in range(3):
        img = cv2.imread(os.path.join(ROOT, f"debug_input_drone_{i}.jpg"))
        if img is None:
            raise SystemExit(f"missing debug_input_drone_{i}.jpg beside StitcherThreading.py")
        base.append(cv2.resize(img, (FRAME_W, FRAME_H)))
    seq = []
    for k in range(SEQ_LEN):
        # Small per-frame camera motion (sub-pixel to a few px) plus an exposure drift,
        # so consecutive frames differ the way real feeds do and TemporalNet has work.
        frames = []
        for j, img in enumerate(base):
            dx = 0.6 * k * (1 if j != 1 else 0.5) + 0.3 * math.sin(0.7 * k + j)
            dy = 0.25 * k + 0.2 * math.cos(0.5 * k + j)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            f = cv2.warpAffine(img, M, (FRAME_W, FRAME_H), borderMode=cv2.BORDER_REFLECT)
            f = np.clip(f.astype(np.int16) + (k % 5) - 2, 0, 255).astype(np.uint8)
            frames.append(np.ascontiguousarray(f))
        seq.append(frames)
    return seq


# --------------------------------------------------------------------------------------
# Pristine vendored code, for the equivalence check
# --------------------------------------------------------------------------------------

CODES = os.path.join(ROOT, "StabStitch2_main", "Full_model_inference", "Codes")
PATCH = os.path.join(HERE, "stabstitch2_perf.patch")
VENDORED_FILES = [
    "spatial_network.py", "temporal_network.py", "smooth_network.py", "grid_res.py",
    "utils/torch_DLT.py", "utils/torch_homo_transform.py",
    "utils/torch_tps_transform.py", "utils/torch_tps_transform_point.py",
]
ORIG_MODULES = ["spatial_network", "temporal_network", "smooth_network", "grid_res",
                "utils", "utils.torch_DLT", "utils.torch_homo_transform",
                "utils.torch_tps_transform", "utils.torch_tps_transform_point"]


def load_pristine_vendored():
    """
    Import the pristine vendored modules under a private namespace.

    The vendored directory is gitignored, so the baseline is rebuilt: the current files
    are copied to a temp dir and ``tools/stabstitch2_perf.patch`` is reverse-applied
    there. Returns (namespace, None) or (None, reason).
    """
    tmp = tempfile.mkdtemp(prefix="stabstitch_pristine_")
    os.makedirs(os.path.join(tmp, "utils"), exist_ok=True)
    for rel in VENDORED_FILES:
        with open(os.path.join(CODES, rel), "rb") as f:
            data = f.read()
        with open(os.path.join(tmp, rel), "wb") as f:
            f.write(data)
    try:
        r = subprocess.run(["git", "apply", "-R", os.path.abspath(PATCH)], cwd=tmp,
                           capture_output=True, text=True)
    except Exception as e:
        return None, f"git not runnable: {e}"
    if r.returncode != 0:
        return None, ("the perf patch does not reverse-apply to the vendored files -- either "
                      "the edits are not present (apply tools/stabstitch2_perf.patch) or the "
                      f"patch is out of date: {r.stderr.strip()}")
    saved = {m: sys.modules.pop(m) for m in ORIG_MODULES if m in sys.modules}
    sys.path.insert(0, tmp)
    try:
        import grid_res  # noqa: F401  (the StabStitch2 one; same values)
        import spatial_network as o_spatial
        import temporal_network as o_temporal
        import utils.torch_tps_transform_point as o_tps_point
        import utils.torch_tps_transform as o_tps
        import utils.torch_DLT as o_dlt
        import utils.torch_homo_transform as o_homo
    finally:
        sys.path.remove(tmp)
        for m in ORIG_MODULES:
            sys.modules.pop(m, None)
        sys.modules.update(saved)
    return types.SimpleNamespace(spatial=o_spatial, temporal=o_temporal, tps_point=o_tps_point,
                                 tps=o_tps, dlt=o_dlt, homo=o_homo), None


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_preprocess(st, frames):
    print("\n[1] GPU preprocessing vs cv2.resize reference")
    with torch.no_grad(), torch.cuda.stream(st._render_stream):
        u8 = st._upload_frames(frames)
        lr = st._make_lr(u8)
        sync()
    worst = 0.0
    for j in range(3):
        ref = st._preprocess(frames[j])
        worst = max(worst, (lr[j:j + 1].cpu() - ref).abs().max().item())
    lsb = 1.0 / 127.5
    print(f"  max |gpu - cv2| = {worst:.5f}  (one uint8 LSB = {lsb:.5f})")
    check(worst <= lsb + 1e-6, "GPU preprocess matches cv2.resize to within one uint8 step")


def test_vendored(st, frames):
    print("\n[2] Vendored StabStitch2 edits vs pristine upstream copy")
    if not os.path.isfile(PATCH):
        check(False, "tools/stabstitch2_perf.patch is missing")
        return
    try:
        o, why = load_pristine_vendored()
    except Exception as e:  # pragma: no cover
        o, why = None, f"could not load pristine modules: {e}"
    if o is None:
        check(False, f"pristine baseline unavailable: {why}")
        return
    print("  pristine copy rebuilt by reverse-applying tools/stabstitch2_perf.patch")

    with torch.no_grad(), torch.cuda.stream(st._render_stream):
        lr = st._make_lr(st._upload_frames(frames))
        sync()
    a = torch.cat([lr[0:1], lr[1:2]], 0)
    b = torch.cat([lr[1:2], lr[2:3]], 0)

    # Same weights into the pristine SpatialNet / TemporalNet definitions.
    o_sp = o.spatial.SpatialNet().cuda().eval()
    o_sp.load_state_dict(st.spatial_net.state_dict())
    o_tp = o.temporal.TemporalNet().cuda().eval()
    o_tp.load_state_dict(st.temporal_net.state_dict())

    import spatial_network as n_spatial
    import temporal_network as n_temporal
    import utils.torch_tps_transform_point as n_tps_point
    with torch.no_grad():
        ro = o.spatial.build_SpatialNet(o_sp, a, b)
        rn = n_spatial.build_SpatialNet(st.spatial_net, a, b)
        d_sp = max((ro["motion1"] - rn["motion1"]).abs().max().item(),
                   (ro["motion2"] - rn["motion2"]).abs().max().item())
        print(f"  SpatialNet motion max |pristine - edited| = {d_sp:.3e} px (NET res)")
        check(d_sp <= 1e-3, "SpatialNet output unchanged by the vendored edits")

        seq = [torch.cat([lr[j:j + 1]] * 3, 0) for j in range(3)]
        to = o.temporal.build_TemporalNet(o_tp, seq)["motion_list"]
        tn = n_temporal.build_TemporalNet(st.temporal_net, seq)["motion_list"]
        d_tp = max((x - y).abs().max().item() for x, y in zip(to, tn))
        print(f"  TemporalNet motion max |pristine - edited| = {d_tp:.3e} px (NET res)")
        check(d_tp <= 1e-3, "TemporalNet output unchanged by the vendored edits")

        st._ensure_net_meshes()
        rigid = st._rigid_mesh_net
        src = ss_mod._get_norm_mesh(rigid + rn["motion1"][0:1], st.NET_H, st.NET_W)
        tgt = ss_mod._get_norm_mesh(rigid + rn["motion2"][0:1], st.NET_H, st.NET_W)
        pt = st._norm_rigid_mesh_net
        po = o.tps_point.transformer(pt, src, tgt)
        pn = n_tps_point.transformer(pt, src, tgt)
        d_pt = (po - pn).abs().max().item()
        print(f"  TPS point transform max |pristine - edited| = {d_pt:.3e} (normalised)")
        check(d_pt <= 1e-6, "TPS point transformer unchanged by the vendored edits")

        # The image-warping TPS transformer is untouched, but _overlap_psnr_terms now
        # builds the same field itself; pin that equivalence too.
        img = torch.cat([lr[0:1], torch.ones_like(lr[0:1, :1])], 1)
        wo = o.tps.transformer(img, src, pt, (90, 120), mode="FAST")
        flow = ss_mod._compute_tps_flow(src, pt, 90, 120)
        wn = F.grid_sample(img, flow, align_corners=True)
        d_w = (wo - wn).abs().max().item()
        print(f"  TPS image warp (FAST) vs _compute_tps_flow+grid_sample: max diff {d_w:.3e}")
        check(d_w <= 1e-3, "quality-gate warp equals the vendored FAST transformer")


def run_incremental(st, seq, every_frame):
    """Feed the sequence; ingest after every frame or only at the end."""
    st.reset()
    st.legacy_warp = False
    for frames in seq:
        st.stab_pano(frames, [0, 1], [1, 2], out_size=WIRE_SIZE)
        if every_frame:
            st._update_warps()
    st._update_warps()
    return dict(st._cached_warp)


def run_legacy(st, seq):
    st.reset()
    st.legacy_warp = True
    for frames in seq:
        st.stab_pano(frames, [0, 1], [1, 2], out_size=WIRE_SIZE)
    st._update_warps()
    st.legacy_warp = False
    return dict(st._cached_warp)


def mesh_diff(p, q):
    return max((p[k] - q[k]).abs().max().item() for k in ("m1_final", "m2_final", "m3_final"))


def test_incremental(st, seq):
    print("\n[3] Incremental warp cache vs legacy full-window warp")
    ss_mod.NET_FRAME_PERIOD = 0.0   # admit every frame the test feeds

    # Under the default settings the two paths run the convolutions at different batch
    # sizes, cuDNN autotunes a different algorithm for each, and TF32 accumulation makes
    # the results differ in the third significant figure. That reaches the meshes as a
    # few tenths of a pixel that varies run to run. The equivalence question is answered
    # with that noise switched off; the default-mode spread is printed for reference.
    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        leg = run_legacy(st, seq)
        inc1 = run_incremental(st, seq, every_frame=True)
        incN = run_incremental(st, seq, every_frame=False)
    finally:
        (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) = saved

    print(f"  canvas: legacy {leg['out_size']}, incremental/1 {inc1['out_size']}, "
          f"incremental/N {incN['out_size']}")
    d1 = mesh_diff(leg, inc1)
    dN = mesh_diff(leg, incN)
    print(f"  final meshes max |legacy - incremental| (fp32, deterministic): per-frame "
          f"ingest {d1:.4f} px, batched ingest {dN:.4f} px (HR canvas px)")
    check(d1 <= 0.01, "per-frame ingest reproduces the legacy meshes (<= 0.01 px)")
    check(dN <= 0.01, "batched ingest reproduces the legacy meshes (<= 0.01 px)")
    check(leg['out_size'] == inc1['out_size'] == incN['out_size'], "canvas size is identical")
    for k in ("m1_final", "flow", "flow_out"):
        check(bool(torch.isfinite(inc1[k]).all().item()), f"{k} is finite")

    leg_d = run_legacy(st, seq)
    inc_d = run_incremental(st, seq, every_frame=True)
    print(f"  for reference, default settings (TF32, autotuned): {mesh_diff(leg_d, inc_d):.3f} px")
    return leg_d, inc_d


def test_render(st, seq, params):
    print("\n[4] Wire-ready render vs legacy canvas render + cv2 post-processing")
    frames = seq[-1]
    st.legacy_warp = False
    with st._warp_lock:
        st._cached_warp = params
    with torch.no_grad(), torch.cuda.stream(st._render_stream):
        u8 = st._upload_frames(frames)
        canvas = st._render_with_params(u8, params, out_size=None)
        wire = st._render_with_params(u8, params, out_size=WIRE_SIZE)
    check(isinstance(wire, WireReadyPanorama), "out_size render returns a WireReadyPanorama")
    check(tuple(wire.shape) == (WIRE_SIZE[0], WIRE_SIZE[1], 3), f"wire panorama is {WIRE_SIZE}")
    legacy = cv2.cvtColor(cv2.flip(cv2.resize(canvas, (WIRE_SIZE[1], WIRE_SIZE[0])), 0),
                          cv2.COLOR_BGR2RGB)
    p = psnr(legacy, np.asarray(wire))
    p_unflipped = psnr(legacy[::-1], np.asarray(wire))
    p_swapped = psnr(legacy[..., ::-1], np.asarray(wire))
    print(f"  PSNR(wire, legacy-post) = {p:.1f} dB   (row-flipped {p_unflipped:.1f} dB, "
          f"channel-swapped {p_swapped:.1f} dB)")
    check(p >= 30.0, "wire render matches the legacy resize/flip/convert (>= 30 dB)")
    check(p > p_unflipped + 3 and p > p_swapped + 3,
          "orientation and channel order are the wire's (flip/swap score lower)")
    cv2.imwrite(os.path.join(HERE, "selftest_stabstitch_pano.jpg"),
                cv2.cvtColor(cv2.flip(np.asarray(wire), 0), cv2.COLOR_RGB2BGR))


def test_syncs(st, seq):
    print("\n[5] Device syncs per incremental warp update")
    ss_mod.NET_FRAME_PERIOD = 0.0
    st.reset()
    for frames in seq[:-1]:
        st.stab_pano(frames, [0, 1], [1, 2], out_size=WIRE_SIZE)
        st._update_warps()
    st.stab_pano(seq[-1], [0, 1], [1, 2], out_size=WIRE_SIZE)   # one frame pending
    sync()
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st._update_warps()
    torch.cuda.set_sync_debug_mode("default")
    cnt = collections.Counter(f"{os.path.basename(x.filename)}:{x.lineno}" for x in w)
    for k, v in cnt.most_common():
        print(f"    {v:3d}  {k}")
    total = sum(cnt.values())
    print(f"  total: {total} (budget {SYNC_BUDGET})")
    check(total <= SYNC_BUDGET, "warp update stays within the sync budget")


def timings(st, seq):
    print("\n[6] Timings (Unity may be sharing the GPU)")
    ss_mod.NET_FRAME_PERIOD = 0.0
    st.reset()
    for frames in seq:
        st.stab_pano(frames, [0, 1], [1, 2], out_size=WIRE_SIZE)
        st._update_warps()
    frames = seq[-1]

    def one_update():
        st.stab_pano(frames, [0, 1], [1, 2], out_size=WIRE_SIZE)
        st._update_warps()

    t_inc = timeit(one_update, n=15)
    ss_mod.NET_FRAME_PERIOD = 1e9   # no more admissions: pure render
    t_render = timeit(lambda: st.stab_pano(frames, [0, 1], [1, 2], out_size=WIRE_SIZE), n=40)
    ss_mod.NET_FRAME_PERIOD = 0.0
    st.reset()
    st.legacy_warp = True
    for f in seq:
        st.stab_pano(f, [0, 1], [1, 2], out_size=WIRE_SIZE)
    t_leg = timeit(lambda: (st.stab_pano(frames, [0, 1], [1, 2], out_size=WIRE_SIZE),
                            st._update_warps()), n=8)
    st.legacy_warp = False
    print(f"  warp update, incremental (1 new frame + render): {t_inc*1000:.1f} ms")
    print(f"  warp update, legacy full window (+ render):      {t_leg*1000:.1f} ms")
    print(f"  render only (wire-ready {WIRE_SIZE[1]}x{WIRE_SIZE[0]}): {t_render*1000:.2f} ms")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (the stitcher runs on the GPU)")
    seq = load_sequence()
    st = StabStitcher(timing=False)
    for net in (st.spatial_net, st.temporal_net, st.smooth_net):
        net.cuda()
    st.quality_enabled = True

    test_preprocess(st, seq[0])
    test_vendored(st, seq[0])
    leg, inc = test_incremental(st, seq)
    test_render(st, seq, inc)
    test_syncs(st, seq)
    timings(st, seq)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
