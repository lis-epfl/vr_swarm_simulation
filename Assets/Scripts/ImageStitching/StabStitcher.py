import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os
import sys
import glob
import time
import math
import threading
import contextlib
from collections import deque, OrderedDict

from BaseStitcher import BaseStitcher

# ---------------------------------------------------------------------------
# Path setup – add the StabStitch2 Codes directory so that its local imports
# (spatial_network, temporal_network, smooth_network, grid_res, utils.*) all
# resolve correctly when this file is imported from the ImageStitching/ CWD.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
STABSTITCH2_CODES = os.path.join(
    _THIS_DIR, "StabStitch2_main", "Full_model_inference", "Codes"
)
STABSTITCH2_MODEL_DIR = os.path.join(
    _THIS_DIR, "StabStitch2_main", "Full_model_inference", "full_model_ssd"
)

if STABSTITCH2_CODES not in sys.path:
    sys.path.insert(0, STABSTITCH2_CODES)

# ---------------------------------------------------------------------------
# Grid-res conflict fix:
# UDIS2_main also has a grid_res.py (GRID_H=12, GRID_W=12) and its network.py
# does a plain `import grid_res`, which caches the UDIS version in
# sys.modules['grid_res'] before StabStitch2's networks are imported.
# We force-load StabStitch2's grid_res.py (GRID_H=6, GRID_W=8) into
# sys.modules so the StabStitch2 networks see the correct values at init.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util

_ss2_grid_res_spec = _importlib_util.spec_from_file_location(
    "grid_res", os.path.join(STABSTITCH2_CODES, "grid_res.py")
)
_ss2_grid_res = _importlib_util.module_from_spec(_ss2_grid_res_spec)
_ss2_grid_res_spec.loader.exec_module(_ss2_grid_res)
sys.modules['grid_res'] = _ss2_grid_res

# Clear any cached StabStitch2 network modules that may have been partially
# loaded in a previous failed import (e.g. when einops was missing).
for _mod in ['spatial_network', 'temporal_network', 'smooth_network']:
    sys.modules.pop(_mod, None)

from spatial_network import SpatialNet, build_SpatialNet
from temporal_network import TemporalNet, build_TemporalNet
from smooth_network import SmoothNet, build_SmoothNet
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_transform_point as torch_tps_transform_point
from torchvision.transforms import GaussianBlur

grid_h = _ss2_grid_res.GRID_H  # 6
grid_w = _ss2_grid_res.GRID_W  # 8

# Cadence at which frames are admitted to the nets' 7-frame temporal buffer, in seconds.
# Deliberately NOT the render rate: the paper's SmoothNet is trained on a window of
# consecutive video frames, and the sim has always fed it at ~20 Hz (Unity's old 0.05 s
# send interval). Keeping this fixed means raising the render/publish rate does not
# shorten the ~350 ms window the smoothing sees. Averaged, not exact: an admission that
# falls between renders is taken on the next one and the phase carries over, so the mean
# cadence is 1 / NET_FRAME_PERIOD whatever the render rate.
NET_FRAME_PERIOD = float(os.environ.get("STABSTITCH_NET_FRAME_PERIOD", "0.05"))


def _env_flag(name, default="0"):
    return os.environ.get(name, default).lower() not in ("0", "", "false", "no")


class WireReadyPanorama(np.ndarray):
    """
    A panorama already in the PanoramaSharedMemory layout: RGB, bottom-up rows, at the
    panorama size Unity asked for. ``stab_pano`` returns one of these when it rendered
    straight to the wire; ``first_thread`` then writes it without the resize / flip /
    BGR->RGB pass it applies to a plain BGR canvas.

    An ndarray subclass rather than a wrapper so every existing ``.shape`` / ``imwrite``
    use keeps working; the backing memory is a pinned buffer owned by the stitcher and is
    reused a few renders later, so consumers must not hold on to it.
    """


# ---------------------------------------------------------------------------
# Mesh helpers (mirrors those in test_online_tra_threeview.py)
# ---------------------------------------------------------------------------

def _get_rigid_mesh(batch_size, height, width):
    """Uniform grid of control points covering [0,W] x [0,H]."""
    ww = torch.matmul(
        torch.ones([grid_h + 1, 1]),
        torch.unsqueeze(torch.linspace(0.0, float(width), grid_w + 1), 0)
    )
    hh = torch.matmul(
        torch.unsqueeze(torch.linspace(0.0, float(height), grid_h + 1), 1),
        torch.ones([1, grid_w + 1])
    )
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()
    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)), 2)  # (H+1, W+1, 2)
    return ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)


def _get_norm_mesh(mesh, height, width):
    """
    Normalise mesh coordinates to [-1, 1], flatten grid dims.

    ``height`` / ``width`` must be Python numbers, not 0-d CUDA tensors: ``float()`` on a
    device tensor is a device->host sync, and this used to be called ~16 times per warp
    update with the canvas size still on the GPU.
    """
    mesh_w = mesh[..., 0] * 2.0 / float(width) - 1.0
    mesh_h = mesh[..., 1] * 2.0 / float(height) - 1.0
    norm_mesh = torch.stack([mesh_w, mesh_h], -1)
    return norm_mesh.reshape([mesh.size(0), -1, 2])


def _recover_mesh(norm_mesh, height, width):
    """Invert _get_norm_mesh – from normalised to pixel coordinates."""
    batch_size = norm_mesh.size(0)
    mesh_w = (norm_mesh[..., 0] + 1) * float(width) / 2.0
    mesh_h = (norm_mesh[..., 1] + 1) * float(height) / 2.0
    mesh = torch.stack([mesh_w, mesh_h], 2)
    return mesh.reshape([batch_size, grid_h + 1, grid_w + 1, 2])


# Constant tensors _compute_tps_flow needs per (batch, points, canvas, device). Built on the
# device once rather than per call: every fresh host tensor .cuda()'d is a stream sync.
# Bounded LRU: the canvas size moves a few pixels between warp updates, so an unbounded
# dict keyed on it would grow by a few MB per distinct size for the whole session.
_TPS_CONST_CACHE = OrderedDict()
_TPS_CONST_CACHE_MAX = 8


def _tps_consts(B, P, out_h, out_w, dev):
    key = (B, P, out_h, out_w, str(dev))
    c = _TPS_CONST_CACHE.get(key)
    if c is not None:
        _TPS_CONST_CACHE.move_to_end(key)
    else:
        x_t = torch.matmul(torch.ones(out_h, 1, device=dev),
                           torch.linspace(-1.0, 1.0, out_w, device=dev).unsqueeze(0))
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, out_h, device=dev).unsqueeze(1),
                           torch.ones(1, out_w, device=dev))
        c = {
            'ones_P': torch.ones(B, P, 1, device=dev),
            'zeros33': torch.zeros(B, 3, 3, device=dev),
            'zeros32': torch.zeros(B, 3, 2, device=dev),
            'x_tf': x_t.reshape(1, 1, -1),
            'y_tf': y_t.reshape(1, 1, -1),
            'ones_hw': torch.ones(B, 1, out_h * out_w, device=dev),
        }
        _TPS_CONST_CACHE[key] = c
        while len(_TPS_CONST_CACHE) > _TPS_CONST_CACHE_MAX:
            _TPS_CONST_CACHE.popitem(last=False)
    return c


def _compute_tps_flow(source, target, out_h, out_w, downscale=1):
    """
    Precompute the normalised TPS sampling field for ``F.grid_sample``.

    Mirrors the FAST path of ``torch_tps_transform.transformer`` (solve the TPS
    system, evaluate the radial-basis meshgrid over every output pixel, apply the
    coefficient matrix) but returns the resulting flow field instead of a warped
    image.  Because the field depends only on the meshes — not the pixels — it
    can be computed once per warp update and reused by every render frame, which
    removes the float64 solve + per-pixel RBF from the hot render loop.

    Parameters
    ----------
    source, target : [B, P, 2] normalised control-point meshes (in [-1, 1]).
    out_h, out_w   : output canvas size.
    downscale      : when > 1, evaluate the per-pixel RBF on a canvas reduced by
                     this factor and bilinearly upsample the resulting field.
                     The RBF term is the memory-bound part (a [B, P+3, h*w]
                     tensor), so this cuts its cost quadratically. A TPS field is
                     smooth between control points, so the upsampled field is
                     visually indistinguishable at small factors.

    Returns
    -------
    flow : [B, out_h, out_w, 2] tensor for
           ``F.grid_sample(img, flow, align_corners=True)``.
    """
    dev = source.device
    B, P, _ = source.shape

    full_h, full_w = out_h, out_w
    if downscale > 1:
        out_h = max(2, int(round(out_h / downscale)))
        out_w = max(2, int(round(out_w / downscale)))

    c = _tps_consts(B, P, out_h, out_w, dev)

    # --- solve the TPS system (matches torch_tps_transform._solve_system) ---
    p = torch.cat([c['ones_P'], source], 2)                       # [B, P, 3]
    d2 = torch.sum((p.reshape(B, -1, 1, 3) - p.reshape(B, 1, -1, 3)) ** 2, 3)
    r = d2 * torch.log(d2 + 1e-6)                                 # [B, P, P]
    W = torch.cat([torch.cat([p, r], 2),
                   torch.cat([c['zeros33'], p.permute(0, 2, 1)], 2)], 1)  # [B, P+3, P+3]
    # inv_ex: torch.inverse without the singularity check, which is a host sync per call.
    W_inv = torch.linalg.inv_ex(W.double()).inverse
    tp = torch.cat([target, c['zeros32']], 1)                     # [B, P+3, 2]
    T = torch.matmul(W_inv, tp.double()).permute(0, 2, 1).float()  # [B, 2, P+3]

    # --- radial-basis meshgrid over the output canvas (matches _meshgrid) ---
    x_tf = c['x_tf']                                              # [1, 1, h*w]
    y_tf = c['y_tf']
    px = source[:, :, 0:1]                                        # [B, P, 1]
    py = source[:, :, 1:2]
    d2g = (x_tf - px) ** 2 + (y_tf - py) ** 2                     # [B, P, h*w]
    rg = d2g * torch.log(d2g + 1e-6)
    grid = torch.cat([c['ones_hw'],
                      x_tf.expand(B, -1, -1), y_tf.expand(B, -1, -1), rg], 1)  # [B, P+3, h*w]

    Tg = torch.matmul(T, grid)                                   # [B, 2, h*w]
    xs = Tg[:, 0, :].reshape(B, 1, out_h, out_w)
    ys = Tg[:, 1, :].reshape(B, 1, out_h, out_w)
    flow = torch.cat([xs, ys], 1)                                # [B, 2, out_h, out_w]

    if downscale > 1:
        # linspace(-1, 1, n) is align_corners=True sampling of the same normalised
        # span at both resolutions, so bilinear upsampling with align_corners=True
        # lands the coarse samples exactly on their full-res counterparts.
        flow = F.interpolate(flow, size=(full_h, full_w),
                             mode='bilinear', align_corners=True)

    return flow.permute(0, 2, 3, 1)                              # [B, full_h, full_w, 2]


class SeparableGaussianBlur:
    """
    Drop-in replacement for ``torchvision.transforms.GaussianBlur`` that applies
    the kernel as two 1-D passes instead of one dense 2-D convolution.

    A 2-D Gaussian is the outer product of two 1-D Gaussians, so this is the same
    filter — but torchvision materialises the full k*k kernel and convolves with
    it, costing O(k^2) per pixel. At the sizes used here that dominated the warp
    update: on a 1690x653 mask the 41x41 blur measured 125 ms and the 21x21 blur
    38 ms, versus ~6 ms and ~4 ms separably.

    Padding is ``reflect`` to match torchvision's behaviour at the borders.
    """

    def __init__(self, kernel_size: int, sigma: float):
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self._k1d = None      # cached [1, 1, 1, k] kernel
        self._key = None      # (kernel_size, sigma, dtype, device)

    def _kernel(self, dtype, device):
        key = (self.kernel_size, self.sigma, dtype, device)
        if self._key != key:
            k = self.kernel_size
            # Matches torchvision's _get_gaussian_kernel1d.
            x = torch.linspace(-(k - 1) / 2.0, (k - 1) / 2.0, steps=k,
                               dtype=dtype, device=device)
            pdf = torch.exp(-0.5 * (x / self.sigma).pow(2))
            self._k1d = (pdf / pdf.sum()).reshape(1, 1, 1, k)
            self._key = key
        return self._k1d

    def __call__(self, img):
        k = self.kernel_size
        pad = k // 2
        c = img.shape[-3]
        k1d = self._kernel(img.dtype, img.device)

        kh = k1d.reshape(1, 1, k, 1).expand(c, 1, k, 1)
        kw = k1d.expand(c, 1, 1, k)

        out = F.pad(img, (0, 0, pad, pad), mode='reflect')
        out = F.conv2d(out, kh, groups=c)
        out = F.pad(out, (pad, pad, 0, 0), mode='reflect')
        return F.conv2d(out, kw, groups=c)


def _inter_grid_loss(mesh):
    """
    Angle-preservation (shape) distortion of a TPS mesh.

    Ported from ``inter_grid_loss`` in StabStitch2's ``test_metric_ssd.py``.
    Scale-invariant (cosine based), so it works regardless of canvas size.

    mesh : [bs, T, grid_h+1, grid_w+1, 2] pixel-coordinate mesh.
    Returns a scalar tensor — higher ⇒ more sheared/folded grid.
    """
    eps = 1e-7
    # horizontal edges + angle between successive horizontal edges
    w_edges = mesh[:, :, :, 0:grid_w, :] - mesh[:, :, :, 1:grid_w + 1, :]
    cos_w = torch.sum(w_edges[:, :, :, 0:grid_w - 1, :] * w_edges[:, :, :, 1:grid_w, :], 4) / (
        torch.sqrt(torch.sum(w_edges[:, :, :, 0:grid_w - 1, :] ** 2, 4)) *
        torch.sqrt(torch.sum(w_edges[:, :, :, 1:grid_w, :] ** 2, 4)) + eps)
    delta_w_angle = 1 - cos_w
    delta_w_angle = delta_w_angle[:, :, 0:grid_h, :] + delta_w_angle[:, :, 1:grid_h + 1, :]

    # vertical edges + angle between successive vertical edges
    h_edges = mesh[:, :, 0:grid_h, :, :] - mesh[:, :, 1:grid_h + 1, :, :]
    cos_h = torch.sum(h_edges[:, :, 0:grid_h - 1, :, :] * h_edges[:, :, 1:grid_h, :, :], 4) / (
        torch.sqrt(torch.sum(h_edges[:, :, 0:grid_h - 1, :, :] ** 2, 4)) *
        torch.sqrt(torch.sum(h_edges[:, :, 1:grid_h, :, :] ** 2, 4)) + eps)
    delta_h_angle = 1 - cos_h
    delta_h_angle = delta_h_angle[:, :, :, 0:grid_w] + delta_h_angle[:, :, :, 1:grid_w + 1]

    return torch.mean(delta_w_angle) + torch.mean(delta_h_angle)


def _compute_blend_weights(ref_m, tgt_m, blur):
    """
    Compute per-pixel linear-blend weight maps from two soft masks.

    Returns (w_ref, w_tgt) each of shape [1, 1, H, W].  These are
    geometry-only: they do not depend on image pixel values, so they can
    be precomputed once per warp update and reused every render frame.

    Written to stay entirely on the GPU.  The original formulation used
    ``torch.nonzero`` (three times per call, twice per warp update) to gather
    mask centroids and the overlap extent; each of those is a device→host sync
    that drains the whole CUDA queue, and under contention with the render
    thread they dominated the warp update.  Centroids and the overlap min/max
    are computed here as masked reductions instead, which is numerically the
    same but never leaves the device.
    """
    dev = ref_m.device
    H, W = ref_m.shape[-2:]

    m1 = (ref_m[0, 0] > 0.01).float()
    m2 = (tgt_m[0, 0] > 0.01).float()

    rows = torch.arange(H, device=dev, dtype=torch.float32).unsqueeze(1)  # [H, 1]
    cols = torch.arange(W, device=dev, dtype=torch.float32).unsqueeze(0)  # [1, W]

    # Masked centroids. clamp_min(1) keeps an empty mask from producing NaN;
    # the resulting centroid is unused because the overlap is empty too.
    n1 = m1.sum().clamp_min(1.0)
    n2 = m2.sum().clamp_min(1.0)
    c1r, c1c = (m1 * rows).sum() / n1, (m1 * cols).sum() / n1
    c2r, c2c = (m2 * rows).sum() / n2, (m2 * cols).sum() / n2
    vec_r, vec_c = c2r - c1r, c2c - c1c

    ovl    = (ref_m * tgt_m)[:, 0].unsqueeze(1)
    ref_m_ = (ref_m[:, 0].unsqueeze(1) - ovl).clamp(0, 1)

    # Ramp across the overlap along the centre-to-centre direction, normalised by
    # the overlap's own extent along that direction (masked min/max, no sync).
    ovl_b = (ovl[0, 0] > 0.01)
    proj  = (rows - c1r) * vec_r + (cols - c1c) * vec_c            # [H, W]
    pmin  = proj.masked_fill(~ovl_b, float('inf')).amin()
    pmax  = proj.masked_fill(~ovl_b, float('-inf')).amax()
    ramp  = (proj - pmin) / (pmax - pmin + 1e-3)
    # Empty overlap ⇒ pmin/pmax are ±inf and ramp is non-finite; zero it out so
    # the result matches the original's all-zero ovl_mask in that case.
    ramp  = torch.nan_to_num(ramp, nan=0.0, posinf=0.0, neginf=0.0)
    ovl_mask = (ramp * ovl_b).reshape(1, 1, H, W)

    w_ref = (
        blur(ref_m_ + (1 - ovl_mask) * ref_m[:, 0].unsqueeze(1)) * ref_m + ref_m_
    ).clamp(0, 1)
    w_tgt = (1 - w_ref) * tgt_m
    return w_ref, w_tgt


# ---------------------------------------------------------------------------
# StabStitcher
# ---------------------------------------------------------------------------

class StabStitcher(BaseStitcher):
    """
    Video-aware stitcher using StabStitch++ (Spatial + Temporal + Smooth warp).

    Two threads share it. The **render** thread (``stab_pano``, Unity-paced) uploads
    the three current frames, admits one to the temporal buffer every
    ``NET_FRAME_PERIOD`` and warps the frames with the cached TPS field. The **warp**
    thread (``compute_warps``) runs the nets and refreshes that cache.

    The nets run **incrementally**: SpatialNet's output for a frame depends only on
    that frame, TemporalNet's on that frame and its predecessor, so both are computed
    once when a frame is admitted and kept in per-frame deques alongside the frame.
    Each warp update then only runs the nets on the frames admitted since the last one
    and re-runs the (cheap) SmoothNet over the 7-frame window. This is the same
    computation the paper's online protocol performs — SpatialNet per frame, TemporalNet
    per transition, SmoothNet per window — just not repeated 7x per update. The previous
    "snapshot the window, run everything" path is kept as ``legacy_warp``
    (``STABSTITCH_LEGACY_WARP=1``) so the two can be compared.

    Pipeline per admitted frame:
      1. SpatialNet on the (left, centre) and (centre, right) pairs (one batch-2 call).
      2. TemporalNet on the transition from the previous frame, all three streams (batch 3).
      3. Temporal-spatial motion (tsmotion) for the transition.
    Per warp update:
      4. SmoothNet over the 7-frame window (both pairs, batch 2).
      5. Mesh alignment, canvas, TPS sampling field, blend weights, quality gate — all for
         the latest frame only.

    While the buffer is filling (fewer than BUFFER_LEN frames) a simple
    horizontal concatenation is returned as a fallback.
    """

    # Images are internally resized to this resolution for the networks.
    NET_H = 360
    NET_W = 480
    BUFFER_LEN = 7  # SmoothNet requires exactly 7 frames

    def __init__(self, warp_mode: str = "FAST", fusion_mode: str = "REFERENCE_BLEND", timing: bool = False,
                 save_masks: bool = False, mask_save_dir: str = None,
                 blur_kernel_size: int = 41, blur_sigma: float = 15.0,
                 border_size: int = 60,
                 quality_enabled: bool = True, quality_threshold: float = 18.0,
                 distortion_threshold: float = 1.0, canvas_ratio_max: float = 5.0,
                 quality_hysteresis: int = 2):
        # BaseStitcher sets up attributes consumed by StitcherManager's
        # hyperparameter-change detection (active_matcher_type, isRANSAC, …).
        # We pass device="cpu" so its SuperPoint model stays off-GPU; our
        # StabStitch networks are moved to GPU explicitly by StitcherManager.
        super().__init__(device="cpu")

        self.warp_mode = warp_mode
        self.fusion_mode = fusion_mode
        # Env-gated so the per-stage warp/render breakdown can be turned on for
        # profiling without touching Unity or the shared-memory metadata.
        self.timing = timing or _env_flag("STABSTITCH_TIMING")
        self.save_masks = save_masks
        self.mask_save_dir = mask_save_dir or os.path.join(_THIS_DIR, "mask_debug")
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.border_size = border_size

        # Mixed-precision toggles. Render fp16 halves the warp's compute + PCIe
        # transfer and is visually lossless for image resampling. Net fp16 (the
        # 3 Hz warp thread) is off by default — these StabStitch++ nets aren't
        # validated in fp16 and a NaN there is hard to spot; enable it if the
        # warp thread becomes the bottleneck.
        self.render_fp16 = True
        # Net fp16 does not work with these StabStitch++ nets: SpatialNet's
        # H2Mesh calls torch.inverse on the estimated homography, and
        # linalg.inv has no half implementation ("Low precision dtypes not
        # supported. Got Half"). Leave off unless that vendored code is patched
        # to force the solve back to fp32.
        self.net_fp16 = False

        # Downscale factor for the per-pixel TPS RBF evaluation (see
        # _compute_tps_flow). 1 = exact; 2 evaluates on a half-size canvas and
        # bilinearly upsamples the field. Default 2: the RBF over a ~1400x550 canvas
        # is 15 ms at full resolution and 4 ms at half, for a max field error of
        # ~0.085 px -- well under the resampling that follows anyway.
        self.flow_downscale = int(os.environ.get("STABSTITCH_FLOW_DOWNSCALE", "2"))

        # Full-window recompute every update, as before the incremental cache. Kept for
        # the self-test's equivalence check and for bisecting; see the class docstring.
        self.legacy_warp = _env_flag("STABSTITCH_LEGACY_WARP")

        # --- Panorama quality estimate / auto-fallback ---
        # When the stitched panorama is judged bad (poor overlap alignment,
        # distorted/folded mesh, or a blown-up canvas) ``stab_pano`` returns
        # quality_ok=False so the caller can fall back to the individual feeds.
        self.quality_enabled = quality_enabled
        self.quality_threshold = quality_threshold        # min overlap PSNR (dB)
        self.distortion_threshold = distortion_threshold  # max inter-grid (shape) loss
        self.canvas_ratio_max = canvas_ratio_max          # max canvas / input dim ratio
        self.quality_hysteresis = quality_hysteresis      # consecutive updates before switching
        self._fallback_active = False
        self._bad_count = 0
        self._good_count = 0

        # --- Quality diagnostics (for tuning which gate flags a bad stitch) ---
        # Enabled by `timing` or env STABSTITCH_QUALITY_DEBUG=1 so it can be
        # toggled without touching Unity / the shared-memory metadata contract.
        self.quality_debug = bool(timing) or _env_flag("STABSTITCH_QUALITY_DEBUG")
        self.quality_summary_every = int(os.environ.get(
            "STABSTITCH_QUALITY_SUMMARY_EVERY", "50"))
        self._quality_eval_count = 0
        self._quality_bad_count = 0
        # Per-gate attribution; a single BAD frame can trip more than one gate.
        self._quality_fail_counts = {'canvas': 0, 'distortion': 0, 'photometric': 0}
        # Failing-gate bitmask of the most recent eval (canvas=1, distortion=2,
        # photometric=4; 0 = good). Written by _record_quality and read back by
        # the warp update — both in the warp thread — so Unity can be told
        # *why* the panorama dropped to fallback. See REASON_* in PyUniSharingFast.
        self.last_quality_reason = 0
        # The per-eval quality line is printed on a verdict change and otherwise at most
        # once a second: the warp thread now runs at ~15-20 Hz, and that many lines/s is
        # console load, not information.
        self._quality_last_print = 0.0
        self._quality_last_verdict = None

        # --- Networks ---
        self.spatial_net = SpatialNet()
        self.temporal_net = TemporalNet()
        self.smooth_net = SmoothNet()

        self.spatial_net.eval()
        self.temporal_net.eval()
        self.smooth_net.eval()

        self._load_models()

        self._cuda = torch.cuda.is_available()
        self._dev = torch.device('cuda') if self._cuda else torch.device('cpu')

        # --- Rolling frame buffers (GPU tensors at NET_H x NET_W) ---
        self._buf_img1 = deque(maxlen=self.BUFFER_LEN)   # left camera
        self._buf_img2 = deque(maxlen=self.BUFFER_LEN)   # centre camera
        self._buf_img3 = deque(maxlen=self.BUFFER_LEN)   # right camera
        # Per-frame net outputs, index-aligned with the frame deques (incremental path).
        # smesh = rigid mesh + SpatialNet motion, one per (pair, side); tsm = the
        # temporal-spatial motion of the transition INTO that frame (zero for a frame
        # that had no predecessor when it was admitted).
        self._smesh12_1 = deque(maxlen=self.BUFFER_LEN)
        self._smesh12_2 = deque(maxlen=self.BUFFER_LEN)
        self._smesh23_1 = deque(maxlen=self.BUFFER_LEN)
        self._smesh23_2 = deque(maxlen=self.BUFFER_LEN)
        self._tsm12_1 = deque(maxlen=self.BUFFER_LEN)
        self._tsm12_2 = deque(maxlen=self.BUFFER_LEN)
        self._tsm23_1 = deque(maxlen=self.BUFFER_LEN)
        self._tsm23_2 = deque(maxlen=self.BUFFER_LEN)
        # Frames admitted by the render thread and not yet run through the nets:
        # (lr1, lr2, lr3, cuda_event) tuples, oldest first.
        self._pending = []
        self._next_admit = -1.0
        # Only the latest HR frame is ever warped (stab_pano supplies it directly),
        # so the warp pipeline only needs the most-recent input's (H, W).
        self._hr_shape = None
        # Panorama size the wire wants, (h, w); the warp update resamples its field to
        # it so the render can produce the wire layout directly. None = canvas output.
        self._wire_size = None

        # --- Warp caching for decoupled render/warp pipeline ---
        self._cached_warp = None   # dict with precomputed sampling field, weights, size
        self._warp_lock = threading.Lock()   # protects _cached_warp reads/writes
        self._buf_lock = threading.Lock()    # protects buffer deque / pending access
        self._compute_lock = threading.RLock()  # serializes neural-net inference
        self._frame_event = threading.Event()   # set by stab_pano on admission

        # Separate CUDA streams so warp and render GPU kernels can
        # interleave instead of serializing on the default stream.
        self._warp_stream = torch.cuda.Stream() if self._cuda else None
        self._render_stream = torch.cuda.Stream() if self._cuda else None

        # Pinned host staging for the input frames (two, alternated: the previous
        # upload may still be in flight when the next render arrives) and a small ring of
        # pinned output buffers the wire-ready panorama lands in.
        self._stage_in = [None, None]
        self._stage_ev = [None, None]
        self._stage_idx = 0
        self._out_ring = []
        self._out_idx = 0

        # Pre-built blur kernels – reused every frame to avoid per-call
        # kernel allocation and recompilation overhead.
        self._blur_linear = SeparableGaussianBlur(21, 20)
        self._blur_ref = SeparableGaussianBlur(self.blur_kernel_size, self.blur_sigma)

        # --- Cached rigid meshes (avoid recomputation every call) ---
        self._rigid_mesh_net = None       # [1, grid_h+1, grid_w+1, 2] at NET resolution
        self._norm_rigid_mesh_net = None   # normalised version
        self._rigid_mesh_hr = None         # at high-res (lazily initialised)
        self._norm_rigid_mesh_hr = None
        self._hr_cache_key = None          # (hr_h, hr_w) to detect resolution changes

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def reset(self):
        """Drop every buffered frame, cached net output and cached warp (for tests)."""
        with self._buf_lock:
            for d in (self._buf_img1, self._buf_img2, self._buf_img3,
                      self._smesh12_1, self._smesh12_2, self._smesh23_1, self._smesh23_2,
                      self._tsm12_1, self._tsm12_2, self._tsm23_1, self._tsm23_2):
                d.clear()
            self._pending = []
            self._next_admit = -1.0
            self._hr_shape = None
        with self._warp_lock:
            self._cached_warp = None
        self._frame_event.clear()
        self._fallback_active = False
        self._bad_count = 0
        self._good_count = 0

    # ------------------------------------------------------------------
    # Cached mesh initialisation
    # ------------------------------------------------------------------

    def _ensure_net_meshes(self):
        """Lazily create and cache rigid meshes at NET resolution."""
        if self._rigid_mesh_net is None:
            self._rigid_mesh_net = _get_rigid_mesh(1, self.NET_H, self.NET_W)
            self._norm_rigid_mesh_net = _get_norm_mesh(self._rigid_mesh_net, self.NET_H, self.NET_W)

    def _ensure_hr_meshes(self, hr_h, hr_w):
        """Lazily create and cache rigid meshes at high resolution."""
        key = (hr_h, hr_w)
        if self._hr_cache_key != key:
            self._rigid_mesh_hr = _get_rigid_mesh(1, hr_h, hr_w)
            self._norm_rigid_mesh_hr = _get_norm_mesh(self._rigid_mesh_hr, hr_h, hr_w)
            self._hr_cache_key = key

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        model_files = {
            "spatial_warp.pth": self.spatial_net,
            "temporal_warp.pth": self.temporal_net,
            "smooth_warp.pth": self.smooth_net,
        }
        all_found = all(
            os.path.isfile(os.path.join(STABSTITCH2_MODEL_DIR, f))
            for f in model_files
        )
        if not all_found:
            print(
                f"[StabStitcher] Warning: could not find all .pth files in "
                f"{STABSTITCH2_MODEL_DIR}. Networks will use random weights."
            )
            return

        for fname, net in model_files.items():
            path = os.path.join(STABSTITCH2_MODEL_DIR, fname)
            ckpt = torch.load(path, map_location="cpu")
            net.load_state_dict(ckpt["model"])
            print(f"[StabStitcher] Loaded {fname}")

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        BGR uint8 numpy → normalised float tensor at NET resolution (CPU).

        The reference (paper) preprocessing. The live path does the same on the GPU in
        ``_make_lr``; the self-test checks the two agree to within uint8 quantisation.
        """
        resized = cv2.resize(img, (self.NET_W, self.NET_H))
        arr = resized.astype(np.float32)
        arr = (arr / 127.5) - 1.0
        arr = np.transpose(arr, [2, 0, 1])
        return torch.tensor(arr).unsqueeze(0)   # [1, 3, NET_H, NET_W]

    def _to_hr_tensor(self, img: np.ndarray) -> torch.Tensor:
        """BGR uint8 numpy → float32 tensor at original resolution (CPU)."""
        arr = img.astype(np.float32)
        arr = np.transpose(arr, [2, 0, 1])
        return torch.tensor(arr).unsqueeze(0)   # [1, 3, H, W]

    def _stream(self, stream):
        return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

    def _upload_frames(self, imgs):
        """
        Three BGR uint8 HWC frames → one [3, H, W, 3] uint8 device tensor.

        Goes through a pinned staging buffer so the copy is a DMA rather than a
        pageable memcpy, and stays uint8: 3 MB per frame-triple instead of the 12 MB
        the old float32 path pushed across PCIe, with the float conversion done on
        the GPU where it is free.
        """
        h, w = imgs[0].shape[:2]
        i = self._stage_idx
        self._stage_idx ^= 1
        buf = self._stage_in[i]
        if buf is None or tuple(buf.shape[1:3]) != (h, w):
            buf = torch.empty((3, h, w, 3), dtype=torch.uint8)
            if self._cuda:
                buf = buf.pin_memory()
            self._stage_in[i] = buf
            self._stage_ev[i] = None
        # The previous upload out of this staging buffer may still be in flight.
        if self._stage_ev[i] is not None:
            self._stage_ev[i].synchronize()
        host = buf.numpy()
        for k in range(3):
            np.copyto(host[k], imgs[k])
        if not self._cuda:
            return buf.clone()
        g = buf.to(self._dev, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._stage_ev[i] = ev
        return g

    def _make_lr(self, frames_u8):
        """
        [3, H, W, 3] uint8 device frames → [3, 3, NET_H, NET_W] normalised float.

        Bilinear without antialiasing at half-pixel centres is what
        ``cv2.resize(..., INTER_LINEAR)`` computes for these ratios; the two agree to
        within one uint8 quantisation step (cv2 rounds to uint8 before the /127.5).
        """
        x = frames_u8.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(self.NET_H, self.NET_W), mode='bilinear',
                          align_corners=False, antialias=False)
        return x / 127.5 - 1.0

    # ------------------------------------------------------------------
    # Network wrappers
    # ------------------------------------------------------------------

    def _run_spatial_batched(self, img1_list_a, img2_list_a, img1_list_b, img2_list_b):
        """
        Run SpatialNet on two sets of frame pairs in a single batched call.

        Stacks all frame pairs along the batch dimension and runs one forward
        pass; with the incremental cache the lists hold only the newly admitted
        frames (usually one), so this is a batch of 2.

        Returns
        -------
        For each pair (a and b):
        smotion_list, smotion_list2 : lists of [1, H, W, 2] tensors (CUDA)
        smesh_list,   smesh_list2   : lists of [1, H, W, 2] tensors (CUDA)
        """
        rigid = self._rigid_mesh_net  # cached [1, grid_h+1, grid_w+1, 2]
        n = len(img1_list_a)

        # Stack all pairs: [n, C, H, W] for pair-a + [n, C, H, W] for pair-b → [2n, C, H, W]
        all_t1 = torch.cat([torch.cat(img1_list_a, 0), torch.cat(img1_list_b, 0)], 0).to(self._dev)
        all_t2 = torch.cat([torch.cat(img2_list_a, 0), torch.cat(img2_list_b, 0)], 0).to(self._dev)

        with torch.autocast('cuda', enabled=self.net_fp16):
            out = build_SpatialNet(self.spatial_net, all_t1, all_t2)
        # motion shapes: [2n, grid_h+1, grid_w+1, 2] — back to fp32 for mesh math
        all_s1 = out["motion1"].float()
        all_s2 = out["motion2"].float()

        # Split back into per-frame lists for each pair
        def _split_to_lists(batched, offset, count):
            return [batched[offset + i : offset + i + 1] for i in range(count)]

        smotion_a1 = _split_to_lists(all_s1, 0, n)
        smotion_a2 = _split_to_lists(all_s2, 0, n)
        smotion_b1 = _split_to_lists(all_s1, n, n)
        smotion_b2 = _split_to_lists(all_s2, n, n)

        smesh_a1 = [rigid + s for s in smotion_a1]
        smesh_a2 = [rigid + s for s in smotion_a2]
        smesh_b1 = [rigid + s for s in smotion_b1]
        smesh_b2 = [rigid + s for s in smotion_b2]

        return (smotion_a1, smotion_a2, smesh_a1, smesh_a2,
                smotion_b1, smotion_b2, smesh_b1, smesh_b2)

    def _run_temporal_batched(self, img_list1, img_list2, img_list3):
        """
        Run TemporalNet on three camera streams in a single batched call.

        Stacks the three streams along the batch dimension (batch=3) so
        the network processes all streams in one forward pass instead of three.
        Each transition's motion depends only on its own two frames, so a list of
        length 2 (previous frame + new frame) yields exactly the motion the full
        window would for that transition.

        Returns
        -------
        tmotion1, tmotion2, tmotion3 : lists of len(img_list) tensors [1, H, W, 2];
        entry 0 is zero motion (no predecessor in the list).
        """
        # Each img_list is a list of tensors of shape [1, C, H, W]
        # Stack to [3, C, H, W] per frame
        batched_list = [
            torch.cat([img_list1[i], img_list2[i], img_list3[i]], 0).to(self._dev)
            for i in range(len(img_list1))
        ]
        with torch.autocast('cuda', enabled=self.net_fp16):
            out = build_TemporalNet(self.temporal_net, batched_list)
        # list of tensors [3, grid_h+1, grid_w+1, 2] — fp32 for mesh math
        motion_list = [m.float() for m in out["motion_list"]]

        # Unbatch: split each [3, ...] tensor into three [1, ...] tensors
        tmotion1 = [m[0:1] for m in motion_list]
        tmotion2 = [m[1:2] for m in motion_list]
        tmotion3 = [m[2:3] for m in motion_list]
        return tmotion1, tmotion2, tmotion3

    def _compute_tsmotion_batched(self, specs):
        """
        Convert frame-t temporal motion into a temporal-spatial motion relative to
        the (t-1)-th frame's spatial mesh, for several streams at once.  Mirrors
        the data-preparation step in test_online_tra_threeview.py.

        Frame 0 has no predecessor and is defined as zero motion; every other
        (stream, frame) is an independent TPS point transform.  Done one at a time
        that is 4 streams x 6 frames = 24 tiny GPU launches, each carrying its own
        float64 (P+3)x(P+3) inverse, and the launch overhead dominates.  Stacking
        them into a single batch call gives identical results for ~1/10 the time.

        Parameters
        ----------
        specs : list of (smotion_list, smesh_list, tmotion_list) triples. Only
                ``smotion_list[0]`` is read (for the zero entry's shape), so the
                smesh list may be passed in its place.

        Returns
        -------
        list of tsmotion lists, one per input triple, in the same order.
        """
        rigid = self._rigid_mesh_net
        norm_rigid = self._norm_rigid_mesh_net
        T = len(specs[0][0])

        tmesh_all, smesh_prev_all = [], []
        for _smotion_list, smesh_list, tmotion_list in specs:
            for k in range(1, T):
                tmesh_all.append(rigid + tmotion_list[k])
                smesh_prev_all.append(smesh_list[k - 1])

        if not tmesh_all:
            # A single-entry list: every entry is the zero-motion frame-0 case.
            return [[spec[0][0] * 0] for spec in specs]

        tmesh_cat = torch.cat(tmesh_all, 0)
        smesh_prev_cat = torch.cat(smesh_prev_all, 0)
        n = tmesh_cat.size(0)

        norm_tmesh = _get_norm_mesh(tmesh_cat, self.NET_H, self.NET_W)
        norm_smesh_prev = _get_norm_mesh(smesh_prev_cat, self.NET_H, self.NET_W)
        norm_rigid_n = norm_rigid.expand(n, -1, -1)

        tsmesh = torch_tps_transform_point.transformer(
            norm_tmesh, norm_rigid_n, norm_smesh_prev
        )
        recovered = _recover_mesh(tsmesh, self.NET_H, self.NET_W)

        out, i = [], 0
        for smotion_list, smesh_list, _tmotion_list in specs:
            lst = [smotion_list[0] * 0]
            for k in range(1, T):
                lst.append(recovered[i:i + 1] - smesh_list[k])
                i += 1
            out.append(lst)
        return out

    def _run_smooth_batched(self, tsmotion_a1, tsmotion_a2, smesh_a1, smesh_a2,
                            tsmotion_b1, tsmotion_b2, smesh_b1, smesh_b2):
        """
        Run SmoothNet on two pairs simultaneously by batching (batch=2).

        Returns smooth_mesh for each pair, each shape [1, T, grid_h+1, grid_w+1, 2].
        """
        tsmotion_a1 = list(tsmotion_a1)
        tsmotion_a2 = list(tsmotion_a2)
        tsmotion_b1 = list(tsmotion_b1)
        tsmotion_b2 = list(tsmotion_b2)
        tsmotion_a1[0] = tsmotion_a1[0] * 0
        tsmotion_a2[0] = tsmotion_a2[0] * 0
        tsmotion_b1[0] = tsmotion_b1[0] * 0
        tsmotion_b2[0] = tsmotion_b2[0] * 0

        # Stack both pairs along batch dim: [1,...] + [1,...] → [2,...]
        # smesh_list entries are per-frame [1, grid_h+1, grid_w+1, 2]
        # SmoothNet expects lists of [batch, grid_h+1, grid_w+1, 2]
        combined_smesh1 = [torch.cat([a, b], 0) for a, b in zip(smesh_a1, smesh_b1)]
        combined_smesh2 = [torch.cat([a, b], 0) for a, b in zip(smesh_a2, smesh_b2)]
        combined_tsm1   = [torch.cat([a, b], 0) for a, b in zip(tsmotion_a1, tsmotion_b1)]
        combined_tsm2   = [torch.cat([a, b], 0) for a, b in zip(tsmotion_a2, tsmotion_b2)]

        with torch.autocast('cuda', enabled=self.net_fp16):
            out = build_SmoothNet(
                self.smooth_net,
                combined_tsm1, combined_tsm2,
                combined_smesh1, combined_smesh2,
            )
        # shape: [2, T, grid_h+1, grid_w+1, 2] — back to fp32 for mesh math
        smooth1 = out["smooth_mesh1"].float()
        smooth2 = out["smooth_mesh2"].float()

        return smooth1[0:1], smooth1[1:2], smooth2[0:1], smooth2[1:2]

    # ------------------------------------------------------------------
    # Blend-weight precomputation (runs in warp thread)
    # ------------------------------------------------------------------

    def _precompute_blend_weights(self, mask1, mask2, mask3):
        """
        Derive final per-camera weight maps from warped alpha masks.

        Because blending weights depend only on warp geometry (not pixel
        values) they can be computed once per warp update and cached.
        The render thread then reduces to a single weighted sum.

        Returns a dict with 'w1', 'w2', 'w3' each of shape [1, H, W],
        or None for modes that require per-pixel image data (AVERAGE).
        """
        if self.fusion_mode == "REFERENCE":
            # canvas = img2 + img3*(1-mask2) + img1*(1-mask3)*(1-mask2)
            w1 = ((1 - mask3) * (1 - mask2))[:, 0]
            w2 = torch.ones_like(w1)
            w3 = ((1 - mask2))[:, 0]
            return {'w1': w1, 'w2': w2, 'w3': w3}

        elif self.fusion_mode == "REFERENCE_BLEND":
            # StitcherManager writes blur_kernel_size / blur_sigma straight onto
            # the stitcher when Unity's metadata changes, so rebuild the kernel
            # when they no longer match the one we cached.
            if (self._blur_ref.kernel_size != self.blur_kernel_size
                    or self._blur_ref.sigma != self.blur_sigma):
                self._blur_ref = SeparableGaussianBlur(self.blur_kernel_size,
                                                       self.blur_sigma)

            w1_12, w2_12 = _compute_blend_weights(mask1, mask2, self._blur_linear)
            mask12 = mask1 + mask2 - mask1 * mask2
            w12_123, w3_123 = _compute_blend_weights(mask12, mask3, self._blur_linear)

            mask2_b = (mask2 > 0.5).float()
            ks = 2 * self.border_size + 1
            # Separable erosion: a square structuring element factorises into a
            # vertical then a horizontal pass, giving an identical result for
            # O(H*W*2k) work instead of O(H*W*k^2). At border_size=60 (k=121) the
            # square max_pool was ~0.2s -- by far the dominant warp-thread cost.
            mask2_eroded = -F.max_pool2d(-mask2_b, kernel_size=(ks, 1), stride=1,
                                         padding=(self.border_size, 0))
            mask2_eroded = (-F.max_pool2d(-mask2_eroded, kernel_size=(1, ks), stride=1,
                                          padding=(0, self.border_size))).clamp(0, 1)
            mask2_interior = self._blur_ref(mask2_eroded).clamp(0, 1) * mask2_b  # [1,1,H,W]

            mi = mask2_interior  # shorthand
            final_w1 = (w1_12  * w12_123 * (1 - mi))[:, 0]
            final_w2 = (mi + w2_12 * w12_123 * (1 - mi))[:, 0]
            final_w3 = (w3_123 * (1 - mi))[:, 0]
            return {'w1': final_w1, 'w2': final_w2, 'w3': final_w3}

        elif self.fusion_mode == "LINEAR":
            w1_12, w2_12 = _compute_blend_weights(mask1, mask2, self._blur_linear)
            mask12 = mask1 + mask2 - mask1 * mask2
            w12_123, w3_123 = _compute_blend_weights(mask12, mask3, self._blur_linear)
            return {
                'w1': (w1_12 * w12_123)[:, 0],
                'w2': (w2_12 * w12_123)[:, 0],
                'w3': w3_123[:, 0],
            }

        return None  # AVERAGE / EDGE_BLEND: fall back to per-frame path

    # ------------------------------------------------------------------
    # Mask visualisation
    # ------------------------------------------------------------------

    def _save_mask_viz(self, mask1, mask2, mask3, out_size):
        """
        Save a colour-coded overlap map so you can inspect seam placement before fusion.

        Colour key (BGR stored by OpenCV):
          Dark red    – left only
          Dark green  – centre only
          Dark blue   – right only
          Yellow      – left ∩ centre
          Cyan        – centre ∩ right
          Magenta     – left ∩ right  (rare)
          White       – all three overlap
        """
        H, W = out_size
        m1 = mask1.squeeze().cpu().numpy() > 0.5
        m2 = mask2.squeeze().cpu().numpy() > 0.5
        m3 = mask3.squeeze().cpu().numpy() > 0.5

        viz = np.zeros((H, W, 3), dtype=np.uint8)
        viz[m1 & ~m2 & ~m3] = (  0,   0, 200)   # red   – left only
        viz[~m1 & m2 & ~m3] = (  0, 200,   0)   # green – centre only
        viz[~m1 & ~m2 & m3] = (200,   0,   0)   # blue  – right only
        viz[m1 & m2 & ~m3]  = (  0, 220, 220)   # yellow – left+centre
        viz[~m1 & m2 & m3]  = (220, 220,   0)   # cyan   – centre+right
        viz[m1 & ~m2 & m3]  = (220,   0, 220)   # magenta – left+right
        viz[m1 & m2 & m3]   = (255, 255, 255)   # white   – all three

        os.makedirs(self.mask_save_dir, exist_ok=True)
        fname = os.path.join(self.mask_save_dir, "mask_overlap.png")
        cv2.imwrite(fname, viz)

    # ------------------------------------------------------------------
    # Panorama quality estimate (runs in warp thread)
    # ------------------------------------------------------------------

    def _overlap_psnr_terms(self, norm_m1, norm_m2, norm_m3, out_size,
                            img1_lr, img2_lr, img3_lr):
        """
        Photometric consistency of the warped feeds in their overlap regions.

        Warps the latest *low-res* frames with the same normalised meshes used
        for the final panorama (the normalised rigid mesh is resolution
        independent) onto a small canvas and accumulates the squared error and
        pixel count over the overlapping pixels of (left, centre) and
        (centre, right).  Returns the two as device tensors so the caller can
        fetch every quality number in a single device->host copy.

        The warp is ``_compute_tps_flow`` + ``grid_sample``, which is exactly the
        FAST mode of ``torch_tps_transform.transformer`` without its per-call
        host-built constants.
        """
        self._ensure_net_meshes()
        norm_rig = self._norm_rigid_mesh_net
        norm_rig3 = torch.cat([norm_rig, norm_rig, norm_rig], 0)

        # Downscaled canvas — quality is judged at low res to stay cheap.
        sh = int(max(8, min(512, out_size[0] // 4)))
        sw = int(max(8, min(512, out_size[1] // 4)))

        def _prep(t):
            rgb = (t.to(self._dev) + 1.0) * 127.5      # [-1,1] → [0,255]
            alpha = torch.ones_like(rgb[:, :1])
            return torch.cat([rgb, alpha], 1)          # [1, 4, H, W]

        stack = torch.cat([_prep(img1_lr), _prep(img2_lr), _prep(img3_lr)], 0)
        flow = _compute_tps_flow(torch.cat([norm_m1, norm_m2, norm_m3], 0), norm_rig3, sh, sw)
        warp = F.grid_sample(stack, flow, align_corners=True)
        rgb = warp[:, :3]
        m = (warp[:, 3:4] > 0.5).float()               # [3, 1, H, W]

        def _pair(a, b):
            ov = m[a] * m[b]                            # [1, H, W]
            n = ov.sum()
            se = ((rgb[a] - rgb[b]) ** 2 * ov).sum()    # [3,H,W]*[1,H,W]
            return se, n

        se12, n12 = _pair(0, 1)
        se23, n23 = _pair(1, 2)
        return se12 + se23, n12 + n23

    def _estimate_quality(self, norm_m1, norm_m2, norm_m3,
                          m1_final, m2_final, m3_final,
                          out_size, hr_h, hr_w,
                          img1_lr, img2_lr, img3_lr):
        """
        Judge whether the current panorama is good enough to display.

        Combines three signals (mirroring the StabStitch++ evaluation):
          * canvas sanity   — degenerate warps blow the canvas up/collapse it
          * mesh distortion — inter-grid (shape) loss detects folded/torn warps
          * overlap PSNR    — photometric consistency in the overlap regions

        Returns ``(quality_ok: bool, score: float)`` where ``score`` is the
        overlap PSNR (primary, user-tunable signal) for logging.

        Every device-side number is gathered with one ``.cpu()``; the previous
        per-metric ``.item()`` calls were six separate syncs per update.
        """
        with torch.no_grad():
            out_h2, out_w2 = out_size
            # --- canvas sanity (also guards the photometric warp from OOM) ---
            w_ratio = out_w2 / max(1, hr_w)
            h_ratio = out_h2 / max(1, hr_h)
            canvas_ok = (1.0 <= w_ratio <= self.canvas_ratio_max) and \
                        (h_ratio <= self.canvas_ratio_max)
            if not canvas_ok:
                # Degenerate canvas: skip the photometric warp (OOM guard) but
                # still attribute/log canvas as the failing gate.
                self._record_quality(
                    canvas_ok=False, w_ratio=w_ratio, h_ratio=h_ratio,
                    distortion=None, distortion_ok=None,
                    psnr=None, photometric_ok=None, quality_ok=False,
                )
                return False, 0.0

            # --- mesh shape distortion (scale-invariant) ---
            distortion_t = torch.stack([
                _inter_grid_loss(m1_final.unsqueeze(1)),
                _inter_grid_loss(m2_final.unsqueeze(1)),
                _inter_grid_loss(m3_final.unsqueeze(1)),
            ]).max()

            # --- overlap photometric consistency ---
            se, n = self._overlap_psnr_terms(
                norm_m1, norm_m2, norm_m3, out_size, img1_lr, img2_lr, img3_lr
            )
            distortion, se, n = torch.stack([distortion_t, se, n]).cpu().tolist()

            if n < 1:
                psnr = 99.0                                # no overlap to judge
            else:
                mse = se / (n * 3 + 1e-6)
                psnr = 99.0 if mse <= 1e-6 else 10.0 * math.log10((255.0 ** 2) / mse)

            distortion_ok = distortion <= self.distortion_threshold
            photometric_ok = psnr >= self.quality_threshold

            quality_ok = canvas_ok and distortion_ok and photometric_ok

            self._record_quality(
                canvas_ok=True, w_ratio=w_ratio, h_ratio=h_ratio,
                distortion=distortion, distortion_ok=distortion_ok,
                psnr=psnr, photometric_ok=photometric_ok, quality_ok=quality_ok,
            )
        return quality_ok, psnr

    def _record_quality(self, canvas_ok, w_ratio, h_ratio,
                        distortion, distortion_ok, psnr, photometric_ok,
                        quality_ok):
        """
        Attribute and log a quality verdict per gate, so it is clear *which*
        metric flags a stitch as bad (and how each value compares to its
        threshold).  ``*_ok=None`` means that gate was skipped (e.g. the
        photometric warp is skipped when the canvas is already degenerate).

        Maintains cumulative per-gate failure counts; a single BAD frame can
        trip more than one gate, so the counts are independent (they need not
        sum to the BAD-frame total).  When ``quality_debug`` is on, prints a
        per-frame breakdown on BAD verdicts plus a periodic cumulative summary.
        """
        self._quality_eval_count += 1
        reason_mask = 0
        if not quality_ok:
            self._quality_bad_count += 1
            if canvas_ok is False:
                reason_mask |= 1   # REASON_CANVAS
                self._quality_fail_counts['canvas'] += 1
            if distortion_ok is False:
                reason_mask |= 2   # REASON_DISTORTION
                self._quality_fail_counts['distortion'] += 1
            if photometric_ok is False:
                reason_mask |= 4   # REASON_PHOTOMETRIC
                self._quality_fail_counts['photometric'] += 1
        self.last_quality_reason = reason_mask

        # Per-eval metric print so the photometric signal can be watched live while
        # tuning `quality_threshold`: on every verdict change, otherwise at most 1 Hz.
        now = time.perf_counter()
        if quality_ok != self._quality_last_verdict or now - self._quality_last_print >= 1.0:
            self._quality_last_print = now
            self._quality_last_verdict = quality_ok
            psnr_str = "n/a(canvas)" if psnr is None else f"{psnr:.2f}dB"
            print(f"[StabStitch quality] psnr={psnr_str} "
                  f"(threshold>={self.quality_threshold:.1f}) "
                  f"{'OK' if quality_ok else 'BAD'}")

        if not self.quality_debug:
            return

        # Per-frame breakdown on BAD frames (keeps the OK stream quiet).
        if not quality_ok:
            def _gate(ok, name, value, cmp, thresh):
                if ok is None:
                    return f"{name}={value}(skipped)"
                return f"{name}={value} ({cmp}{thresh}) {'PASS' if ok else 'FAIL'}"
            parts = [
                _gate(canvas_ok, "canvas", f"{w_ratio:.2f}x{h_ratio:.2f}",
                      "<=", f"{self.canvas_ratio_max:.1f}"),
                _gate(distortion_ok, "distortion",
                      "n/a" if distortion is None else f"{distortion:.3f}",
                      "<=", f"{self.distortion_threshold:.2f}"),
                _gate(photometric_ok, "psnr",
                      "n/a" if psnr is None else f"{psnr:.2f}dB",
                      ">=", f"{self.quality_threshold:.1f}"),
            ]
            print("[StabStitch quality] BAD  " + "  ".join(parts))

        # Periodic cumulative attribution so the dominant cause is obvious.
        if self._quality_eval_count % self.quality_summary_every == 0:
            c = self._quality_fail_counts
            print(
                f"[StabStitch quality] summary over {self._quality_eval_count} "
                f"evals: bad={self._quality_bad_count} "
                f"(canvas={c['canvas']} distortion={c['distortion']} "
                f"photometric={c['photometric']})"
            )

    def _apply_hysteresis(self, raw_ok):
        """
        Debounce the raw per-update quality decision so the display does not
        flicker between panorama and fallback.  Requires ``quality_hysteresis``
        consecutive updates of the opposite verdict before switching state.

        Returns the (debounced) panorama-ok flag: True ⇒ show panorama.
        """
        if raw_ok:
            self._good_count += 1
            self._bad_count = 0
            if self._fallback_active and self._good_count >= self.quality_hysteresis:
                self._fallback_active = False
        else:
            self._bad_count += 1
            self._good_count = 0
            if not self._fallback_active and self._bad_count >= self.quality_hysteresis:
                self._fallback_active = True
        return not self._fallback_active

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stab_pano(self, images, subset1, subset2, out_size=None):
        """
        Produce a stabilised panorama from three images.

        Uses cached warp parameters when available (fast path, well under a
        millisecond of GPU work).  On the very first call with a full buffer,
        computes warps synchronously so a real panorama is returned immediately.
        Subsequent warp updates are handled by ``compute_warps()`` running on a
        dedicated thread.

        Parameters
        ----------
        images  : list of numpy BGR uint8 arrays
        subset1 : [left_idx, centre_idx]
        subset2 : [centre_idx, right_idx]
        out_size : optional (height, width) the panorama is wanted at.  When
            given and a matching sampling field is cached, the panorama is
            rendered straight into that size, vertically flipped and RGB — the
            PanoramaSharedMemory layout — and returned as a
            :class:`WireReadyPanorama`.  Otherwise a BGR canvas is returned.

        Returns
        -------
        (panorama, quality_ok, quality_reason) : (ndarray or None, bool, int)
            ``quality_ok`` is False when the panorama is judged bad; in that
            case ``panorama`` is None and the render is skipped so the caller
            can display the individual feeds instead.  ``quality_reason`` is a
            failing-gate bitmask (canvas=1, distortion=2, photometric=4; 0 when
            good) so the caller can report *why* it fell back.
        """
        img1 = images[subset1[0]]   # left
        img2 = images[subset1[1]]   # centre
        img3 = images[subset2[1]]   # right
        hr_h, hr_w = img1.shape[0], img1.shape[1]
        out_size = tuple(int(v) for v in out_size) if out_size is not None else None
        self._wire_size = out_size

        with torch.no_grad(), self._stream(self._render_stream):
            frames_u8 = self._upload_frames((img1, img2, img3))

            # Admit this frame to the temporal buffer on the fixed cadence (see
            # NET_FRAME_PERIOD). The phase is carried over so a late admission does
            # not shorten the mean period, and a stall does not cause a burst.
            now = time.perf_counter()
            if self._next_admit < 0.0:
                self._next_admit = now
            admit = now >= self._next_admit
            if admit:
                self._next_admit = max(self._next_admit + NET_FRAME_PERIOD,
                                       now - NET_FRAME_PERIOD)
                lr = self._make_lr(frames_u8)
                ev = None
                if self._cuda:
                    ev = torch.cuda.Event()
                    ev.record()
                with self._buf_lock:
                    self._hr_shape = (hr_h, hr_w)
                    if self.legacy_warp:
                        # The legacy path reads these on the warp stream with no
                        # event; make them complete first (it is a test path).
                        if ev is not None:
                            ev.synchronize()
                        self._buf_img1.append(lr[0:1])
                        self._buf_img2.append(lr[1:2])
                        self._buf_img3.append(lr[2:3])
                    else:
                        self._pending.append((lr[0:1], lr[1:2], lr[2:3], ev))
                    buf_ready = len(self._buf_img1) + len(self._pending) >= self.BUFFER_LEN
                self._frame_event.set()
            else:
                with self._buf_lock:
                    buf_ready = len(self._buf_img1) + len(self._pending) >= self.BUFFER_LEN

            if not buf_ready:
                # Buffer still filling: show the crude concat, but don't trip the
                # quality fallback (no real warp has been computed yet).
                return self._fallback_concat(img1, img2, img3), True, 0

            # First computation: synchronous so the caller gets a real panorama.
            # _update_warps takes the compute lock, so it cannot race the warp thread.
            if self._cached_warp is None:
                self._update_warps()
                if self._cached_warp is None:
                    return self._fallback_concat(img1, img2, img3), True, 0

            with self._warp_lock:
                warp_params = self._cached_warp

            # Quality gate: ``quality_ok`` is the debounced verdict computed in the
            # warp thread.  When bad, skip the (expensive) render entirely and let
            # the caller switch to the individual feeds.
            quality_ok = warp_params.get('quality_ok', True)
            if not quality_ok:
                return None, False, warp_params.get('quality_reason', 0)

            pano = self._render_with_params(frames_u8, warp_params, out_size)
            return pano, True, 0

    def compute_warps(self):
        """
        Run one warp update on the dedicated warp thread.

        Blocks until ``stab_pano`` has admitted a new frame (with a timeout so the
        caller's loop stays responsive to a stitcher switch), then ingests every
        frame admitted since the last update and refreshes the cached warp
        parameters.  The render path keeps using the previous cache meanwhile.

        Returns True when the cache was refreshed, False otherwise (timed out, or
        the window is still filling), so the caller can count real updates.
        """
        if not self._frame_event.wait(timeout=0.1):
            return False
        self._frame_event.clear()
        return self._update_warps()

    def _update_warps(self):
        """
        Ingest pending frames, then (if the window is full) refresh the cache.

        Returns True when new warp parameters were stored.
        """
        with self._compute_lock:
            with self._buf_lock:
                pending = self._pending
                self._pending = []
                hr_shape = self._hr_shape
                if self.legacy_warp:
                    img1_list = list(self._buf_img1)
                    img2_list = list(self._buf_img2)
                    img3_list = list(self._buf_img3)

            if hr_shape is None:
                return False
            hr_h, hr_w = hr_shape

            if self.legacy_warp:
                if len(img1_list) < self.BUFFER_LEN:
                    return False
                warp_params = self._compute_warp_params(img1_list, img2_list, img3_list, hr_h, hr_w)
            else:
                with torch.no_grad(), self._stream(self._warp_stream):
                    t0 = time.perf_counter() if self.timing else None
                    if pending:
                        self._ingest(pending)
                    if len(self._buf_img1) < self.BUFFER_LEN:
                        return False
                    if self.timing:
                        torch.cuda.synchronize()
                        t_nets = time.perf_counter()
                    warp_params = self._window_params(hr_h, hr_w)
                    if self.timing:
                        print(f"[StabStitch warp] nets(new={len(pending)})={t_nets-t0:.3f}  "
                              f"window={time.perf_counter()-t_nets:.3f}s")
                if self._warp_stream is not None:
                    self._warp_stream.synchronize()

            with self._warp_lock:
                self._cached_warp = warp_params
            return True

    # ------------------------------------------------------------------
    # Internal: incremental net cache
    # ------------------------------------------------------------------

    def _ingest(self, pending):
        """
        Run the per-frame nets on newly admitted frames and append them to the window.

        ``pending`` is a list of ``(lr1, lr2, lr3, event)`` oldest first.  SpatialNet
        runs once on all of them (batch 2n); TemporalNet and the tsmotion transform
        run on the transitions from the last buffered frame through each new one
        (a list of n+1 frames, or n when the buffer is empty).  Caller holds
        ``_compute_lock`` and runs on the warp stream.
        """
        self._ensure_net_meshes()
        n = len(pending)

        new1, new2, new3 = [], [], []
        for lr1, lr2, lr3, ev in pending:
            if self._cuda:
                if ev is not None:
                    self._warp_stream.wait_event(ev)
                for t in (lr1, lr2, lr3):
                    # Allocated on the render stream, read on this one: keep the
                    # allocator from recycling the block until this stream is done.
                    t.record_stream(self._warp_stream)
            new1.append(lr1)
            new2.append(lr2)
            new3.append(lr3)

        # ---------- Spatial: both pairs for every new frame (batch 2n) ----------
        (_sm12_1, _sm12_2, me12_1, me12_2,
         _sm23_1, _sm23_2, me23_1, me23_2) = \
            self._run_spatial_batched(new1, new2, new2, new3)

        # ---------- Temporal + tsmotion: transitions prev -> new_0 -> ... -> new_n-1 ----------
        with self._buf_lock:
            have_prev = len(self._buf_img1) > 0
            if have_prev:
                base = ([self._buf_img1[-1]], [self._buf_img2[-1]], [self._buf_img3[-1]])
                base_me = ([self._smesh12_1[-1]], [self._smesh12_2[-1]],
                           [self._smesh23_1[-1]], [self._smesh23_2[-1]])
            else:
                base = ([], [], [])
                base_me = ([], [], [], [])

        tm1, tm2, tm3 = self._run_temporal_batched(
            base[0] + new1, base[1] + new2, base[2] + new3)

        l12_1 = base_me[0] + me12_1
        l12_2 = base_me[1] + me12_2
        l23_1 = base_me[2] + me23_1
        l23_2 = base_me[3] + me23_2
        # smesh doubles as the smotion argument: only its [0] is read, for shape.
        ts12_1, ts12_2, ts23_1, ts23_2 = self._compute_tsmotion_batched([
            (l12_1, l12_1, tm1),
            (l12_2, l12_2, tm2),
            (l23_1, l23_1, tm2),
            (l23_2, l23_2, tm3),
        ])

        with self._buf_lock:
            for i in range(n):
                self._buf_img1.append(new1[i])
                self._buf_img2.append(new2[i])
                self._buf_img3.append(new3[i])
                self._smesh12_1.append(me12_1[i])
                self._smesh12_2.append(me12_2[i])
                self._smesh23_1.append(me23_1[i])
                self._smesh23_2.append(me23_2[i])
                # The last n entries belong to the new frames; the leading entry (if
                # any) is the previous frame's zero placeholder.
                self._tsm12_1.append(ts12_1[len(ts12_1) - n + i])
                self._tsm12_2.append(ts12_2[len(ts12_2) - n + i])
                self._tsm23_1.append(ts23_1[len(ts23_1) - n + i])
                self._tsm23_2.append(ts23_2[len(ts23_2) - n + i])

    def _window_params(self, hr_h, hr_w):
        """SmoothNet over the cached 7-frame window, then the latest-frame geometry."""
        with self._buf_lock:
            smesh12_1 = list(self._smesh12_1)
            smesh12_2 = list(self._smesh12_2)
            smesh23_1 = list(self._smesh23_1)
            smesh23_2 = list(self._smesh23_2)
            tsm12_1 = list(self._tsm12_1)
            tsm12_2 = list(self._tsm12_2)
            tsm23_1 = list(self._tsm23_1)
            tsm23_2 = list(self._tsm23_2)
            lr_latest = (self._buf_img1[-1], self._buf_img2[-1], self._buf_img3[-1])

        smooth12_1, smooth23_1, smooth12_2, smooth23_2 = self._run_smooth_batched(
            tsm12_1, tsm12_2, smesh12_1, smesh12_2,
            tsm23_1, tsm23_2, smesh23_1, smesh23_2,
        )
        return self._meshes_to_params(smooth12_1, smooth12_2, smooth23_1, smooth23_2,
                                      hr_h, hr_w, lr_latest)

    # ------------------------------------------------------------------
    # Internal: fallback and legacy full pipeline
    # ------------------------------------------------------------------

    def _fallback_concat(self, img1, img2, img3) -> np.ndarray:
        """Naive horizontal concatenation used while the buffer is filling."""
        return np.hstack([img1, img2, img3]).astype(np.uint8)

    def _compute_warp_params(self, img1_list, img2_list, img3_list, hr_h, hr_w):
        """
        Legacy path: run the full neural-net warp pipeline (Spatial → Temporal →
        Smooth) on a whole 7-frame window and derive the cached warp parameters.

        Caller holds ``_compute_lock`` (it is re-entrant).  Runs on the warp stream
        so its GPU kernels can interleave with the render stream.
        """
        with self._compute_lock:
            with torch.no_grad(), self._stream(self._warp_stream):
                result = self._compute_warp_params_unlocked(
                    img1_list, img2_list, img3_list, hr_h, hr_w
                )
            if self._warp_stream is not None:
                self._warp_stream.synchronize()
            return result

    def _compute_warp_params_unlocked(self, img1_list, img2_list, img3_list, hr_h, hr_w):
        """Inner legacy implementation — caller must hold ``_compute_lock``."""
        self._ensure_net_meshes()

        t0 = time.perf_counter() if self.timing else None

        # ---------- Spatial: both pairs batched (14 frames → 1 call) ----------
        (smotion12_1, smotion12_2, smesh12_1, smesh12_2,
         smotion23_1, smotion23_2, smesh23_1, smesh23_2) = \
            self._run_spatial_batched(img1_list, img2_list, img2_list, img3_list)
        t1 = time.perf_counter() if self.timing else None

        # ---------- Temporal: all 3 streams batched (3 calls → 1) ----------
        tmotion_stream1, tmotion_stream2, tmotion_stream3 = \
            self._run_temporal_batched(img1_list, img2_list, img3_list)
        t2 = time.perf_counter() if self.timing else None

        # ---------- TSMotion (all four streams in one batched TPS solve) ----------
        tsmotion12_1, tsmotion12_2, tsmotion23_1, tsmotion23_2 = \
            self._compute_tsmotion_batched([
                (smotion12_1, smesh12_1, tmotion_stream1),
                (smotion12_2, smesh12_2, tmotion_stream2),
                (smotion23_1, smesh23_1, tmotion_stream2),
                (smotion23_2, smesh23_2, tmotion_stream3),
            ])

        # ---------- Smooth: both pairs batched (2 calls → 1) ----------
        smooth12_1, smooth23_1, smooth12_2, smooth23_2 = \
            self._run_smooth_batched(
                tsmotion12_1, tsmotion12_2, smesh12_1, smesh12_2,
                tsmotion23_1, tsmotion23_2, smesh23_1, smesh23_2,
            )
        if self.timing:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            print(f"[StabStitch warp/legacy] spatial={t1-t0:.3f}  temporal={t2-t1:.3f}  "
                  f"smooth+tsm={t3-t2:.3f}s")

        return self._meshes_to_params(smooth12_1, smooth12_2, smooth23_1, smooth23_2,
                                      hr_h, hr_w,
                                      (img1_list[-1], img2_list[-1], img3_list[-1]))

    # ------------------------------------------------------------------
    # Internal: latest-frame geometry from the smoothed meshes
    # ------------------------------------------------------------------

    def _meshes_to_params(self, smooth12_1, smooth12_2, smooth23_1, smooth23_2,
                          hr_h, hr_w, lr_latest):
        """
        From the smoothed meshes of the 7-frame window ([1, T, gh+1, gw+1, 2] each)
        derive everything the render needs for the *latest* frame: the normalised
        meshes, the canvas size, the TPS sampling field (at canvas size and, when a
        wire size is known, resampled to it, flipped), the blend weights and the
        quality verdict.

        The canvas bounds are the only values that have to come back to the host
        (``out_size`` is a Python tuple), and they come back in two ``.tolist()``
        calls; every other quantity stays on the device.
        """
        self._ensure_hr_meshes(hr_h, hr_w)
        t3 = time.perf_counter() if self.timing else None

        # ---------- scale meshes to HR resolution ----------
        sx = hr_w / self.NET_W
        sy = hr_h / self.NET_H

        def _scale(m):
            return torch.stack([m[..., 0] * sx, m[..., 1] * sy], -1)

        warp12_mesh1 = _scale(smooth12_1)
        warp12_mesh2 = _scale(smooth12_2)
        warp23_mesh1 = _scale(smooth23_1)
        warp23_mesh2 = _scale(smooth23_2)

        # ---------- work only on the latest frame ----------
        fi = self.BUFFER_LEN - 1

        m12_2_fi = warp12_mesh2[:, fi, ...]
        m23_1_fi = warp23_mesh1[:, fi, ...]
        offset = (m12_2_fi - m23_1_fi).reshape(1, -1, 2).mean(1, keepdim=True)
        offset = offset.unsqueeze(1)

        m12_1_fi = warp12_mesh1[:, fi, ...]
        m12_2_fi_aligned = m12_2_fi
        m23_1_fi_aligned = m23_1_fi + offset.squeeze(1)
        m23_2_fi_aligned = warp23_mesh2[:, fi, ...] + offset.squeeze(1)
        middle_mesh_fi   = (m12_2_fi_aligned + m23_1_fi_aligned) / 2.0

        # ---------- first canvas: bounding box ----------
        all_x = torch.stack([
            m12_1_fi[..., 0], m12_2_fi_aligned[..., 0],
            m23_1_fi_aligned[..., 0], m23_2_fi_aligned[..., 0]
        ])
        all_y = torch.stack([
            m12_1_fi[..., 1], m12_2_fi_aligned[..., 1],
            m23_1_fi_aligned[..., 1], m23_2_fi_aligned[..., 1]
        ])
        # One host round-trip for all four bounds. The subtraction is done in float32
        # so the canvas size is bit-for-bit what the on-device subtraction gave.
        width_min, width_max, height_min, height_max = torch.stack([
            all_x.min(), all_x.max(), all_y.min(), all_y.max()]).tolist()
        out_w = float(np.float32(width_max) - np.float32(width_min))
        out_h = float(np.float32(height_max) - np.float32(height_min))

        def _shift(m, wx, hy):
            return torch.stack([m[..., 0] - wx, m[..., 1] - hy], -1)

        m12_1_s   = _shift(m12_1_fi,         width_min, height_min)
        m12_2_s   = _shift(m12_2_fi_aligned, width_min, height_min)
        m23_1_s   = _shift(m23_1_fi_aligned, width_min, height_min)
        m23_2_s   = _shift(m23_2_fi_aligned, width_min, height_min)
        mid_s     = _shift(middle_mesh_fi,   width_min, height_min)

        # ---------- TPS alignment through middle plane ----------
        norm_m12_1 = _get_norm_mesh(m12_1_s, out_h, out_w)
        norm_m12_2 = _get_norm_mesh(m12_2_s, out_h, out_w)
        norm_m23_1 = _get_norm_mesh(m23_1_s, out_h, out_w)
        norm_m23_2 = _get_norm_mesh(m23_2_s, out_h, out_w)
        norm_mid   = _get_norm_mesh(mid_s,   out_h, out_w)

        norm_m12_1_tps = torch_tps_transform_point.transformer(
            norm_m12_1, norm_m12_2, norm_mid
        )
        m12_1_tps = _recover_mesh(norm_m12_1_tps, out_h, out_w)

        norm_m23_2_tps = torch_tps_transform_point.transformer(
            norm_m23_2, norm_m23_1, norm_mid
        )
        m23_2_tps = _recover_mesh(norm_m23_2_tps, out_h, out_w)

        # ---------- second (final) canvas ----------
        all_x2 = torch.stack([
            m12_1_tps[..., 0], mid_s[..., 0], m23_2_tps[..., 0]
        ])
        all_y2 = torch.stack([
            m12_1_tps[..., 1], mid_s[..., 1], m23_2_tps[..., 1]
        ])
        w2_min, w2_max, h2_min, h2_max = torch.stack([
            all_x2.min(), all_x2.max(), all_y2.min(), all_y2.max()]).tolist()
        out_w2 = float(np.float32(w2_max) - np.float32(w2_min))
        out_h2 = float(np.float32(h2_max) - np.float32(h2_min))

        out_size = (int(out_h2), int(out_w2))

        m1_final  = _shift(m12_1_tps, w2_min, h2_min)
        m2_final  = _shift(mid_s,     w2_min, h2_min)
        m3_final  = _shift(m23_2_tps, w2_min, h2_min)

        norm_m1 = _get_norm_mesh(m1_final, out_h2, out_w2)
        norm_m2 = _get_norm_mesh(m2_final, out_h2, out_w2)
        norm_m3 = _get_norm_mesh(m3_final, out_h2, out_w2)

        if self.timing:
            torch.cuda.synchronize()
            t_mesh = time.perf_counter()

        # ---------- precompute the TPS sampling field (geometry-only, warp thread) ----------
        # The render loop only changes the input *pixels*; the warp field is fixed
        # until the next warp update. Computing it here (once per warp update)
        # turns each render into a single grid_sample — removing the float64 solve
        # and per-pixel RBF from the hot path, which is the dominant high-res cost.
        norm_rigid_hr = self._norm_rigid_mesh_hr
        norm_rig3     = torch.cat([norm_rigid_hr, norm_rigid_hr, norm_rigid_hr], 0)
        flow = _compute_tps_flow(
            torch.cat([norm_m1, norm_m2, norm_m3], 0), norm_rig3,
            out_size[0], out_size[1], downscale=self.flow_downscale,
        )

        if self.timing:
            torch.cuda.synchronize()
            t_flow = time.perf_counter()

        # ---------- precompute blending weights (geometry-only, warp thread) ----------
        # The alpha masks are the same warp as the panorama, so sample them with the
        # field we just built rather than running a second full-canvas TPS solve+RBF.
        alpha_dummy  = torch.ones(3, 1, hr_h, hr_w, device=flow.device)
        masks_warped = F.grid_sample(alpha_dummy, flow, align_corners=True)
        mask1_pre = masks_warped[0].unsqueeze(0)   # [1, 1, H, W]
        mask2_pre = masks_warped[1].unsqueeze(0)
        mask3_pre = masks_warped[2].unsqueeze(0)
        blend_weights = self._precompute_blend_weights(mask1_pre, mask2_pre, mask3_pre)

        # ---------- wire-size field: the same geometry sampled at the panorama size ----------
        # first_thread used to cv2.resize the finished canvas to the panorama size and
        # flip it; sampling the field at that size (rows reversed) lets the render
        # produce the wire layout directly, with one resampling of the sources
        # instead of two. Half-pixel-centre resampling, as cv2.resize does.
        wire_size = self._wire_size
        flow_out = weights_out = None
        if wire_size is not None and blend_weights is not None:
            H, W = wire_size
            flow_out = F.interpolate(flow.permute(0, 3, 1, 2), size=(H, W), mode='bilinear',
                                     align_corners=False).flip(2).permute(0, 2, 3, 1).contiguous()
            weights_out = {
                k: F.interpolate(v.unsqueeze(0), size=(H, W), mode='bilinear',
                                 align_corners=False)[0].flip(1).contiguous()
                for k, v in blend_weights.items()
            }

        if self.timing:
            torch.cuda.synchronize()
            t_blend = time.perf_counter()

        # ---------- panorama quality estimate + hysteresis (warp thread) ----------
        if self.quality_enabled:
            raw_ok, quality_score = self._estimate_quality(
                norm_m1, norm_m2, norm_m3,
                m1_final, m2_final, m3_final,
                out_size, hr_h, hr_w,
                lr_latest[0], lr_latest[1], lr_latest[2],
            )
            quality_ok = self._apply_hysteresis(raw_ok)
            # _estimate_quality just set self.last_quality_reason (same thread).
            # Only meaningful once the debounced verdict has actually flipped bad.
            quality_reason = 0 if quality_ok else self.last_quality_reason
        else:
            quality_score = float('nan')
            quality_ok = True
            quality_reason = 0

        if self.timing:
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            print(
                f"[StabStitch warp] canvas={out_size[1]}x{out_size[0]} "
                f"mesh={t_mesh-t3:.3f}  "
                f"tpsflow={t_flow-t_mesh:.3f}  "
                f"blend={t_blend-t_flow:.3f}  "
                f"quality={t_end-t_blend:.3f}  "
                f"geometry_total={t_end-t3:.3f}s"
            )

        return {
            'norm_m1': norm_m1, 'norm_m2': norm_m2, 'norm_m3': norm_m3,
            'm1_final': m1_final, 'm2_final': m2_final, 'm3_final': m3_final,
            'out_h2': out_h2, 'out_w2': out_w2, 'out_size': out_size,
            'flow': flow,
            'blend_weights': blend_weights,
            'wire_size': wire_size,
            'flow_out': flow_out,
            'weights_out': weights_out,
            'quality_ok': quality_ok, 'quality_score': quality_score,
            'quality_reason': quality_reason,
        }

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _out_buffer(self, shape):
        """Next pinned output buffer of the given shape from a small ring."""
        if not self._out_ring or tuple(self._out_ring[0].shape) != tuple(shape):
            self._out_ring = []
            for _ in range(3):
                t = torch.empty(shape, dtype=torch.uint8)
                if self._cuda:
                    t = t.pin_memory()
                self._out_ring.append(t)
            self._out_idx = 0
        buf = self._out_ring[self._out_idx]
        self._out_idx = (self._out_idx + 1) % len(self._out_ring)
        return buf

    def _render_with_params(self, frames_u8, warp_params, out_size=None):
        """
        Fast render path: warp the three high-res frames with the precomputed
        sampling field and apply the cached blend weights.

        ``frames_u8`` is the [3, H, W, 3] uint8 device tensor from ``_upload_frames``.
        With ``out_size`` and a matching cached wire-size field the result is a
        :class:`WireReadyPanorama` (RGB, bottom-up, at ``out_size``); otherwise a BGR
        canvas ndarray.  Runs on the render CUDA stream; the final host copy is the
        one synchronisation point.
        """
        t0 = time.perf_counter() if self.timing else None

        with torch.no_grad(), self._stream(self._render_stream):
            stack = frames_u8.permute(0, 3, 1, 2)
            stack = stack.half() if self.render_fp16 else stack.float()

            flow_out      = warp_params.get('flow_out')
            weights_out   = warp_params.get('weights_out')
            flow          = warp_params['flow']
            blend_weights = warp_params.get('blend_weights')
            out_h2, out_w2 = warp_params['out_size']

            if (out_size is not None and flow_out is not None and weights_out is not None
                    and warp_params.get('wire_size') == out_size and not self.save_masks):
                # ---------- wire path: sample straight into the panorama layout ----------
                if self._cuda:
                    for t in (flow_out, weights_out['w1'], weights_out['w2'], weights_out['w3']):
                        t.record_stream(self._render_stream)
                img_warp = F.grid_sample(stack, flow_out.to(stack.dtype), align_corners=True)
                fusion = (
                    img_warp[0] * weights_out['w1']
                    + img_warp[1] * weights_out['w2']
                    + img_warp[2] * weights_out['w3']
                )
                # BGR -> RGB, CHW -> HWC; rows are already bottom-up via the flipped field.
                rgb = fusion.clamp(0, 255).to(torch.uint8)[[2, 1, 0]].permute(1, 2, 0).contiguous()
                out = self._out_buffer(rgb.shape)
                out.copy_(rgb, non_blocking=True)
                if self._cuda:
                    torch.cuda.current_stream().synchronize()
                pano = out.numpy().view(WireReadyPanorama)
                if self.timing:
                    print(f"[StabStitch render] wire {out_size[1]}x{out_size[0]}: "
                          f"{time.perf_counter()-t0:.4f}s")
                return pano

            if self._cuda:
                flow.record_stream(self._render_stream)

            def _warp(inp):
                return F.grid_sample(inp, flow.to(inp.dtype), align_corners=True)

            if blend_weights is not None:
                # ---------- canvas path: RGB-only warp + precomputed weighted sum ----------
                if self._cuda:
                    for t in blend_weights.values():
                        t.record_stream(self._render_stream)
                img_warp = _warp(stack)
                # w* shape [1, H, W] broadcasts over [3, H, W]
                fusion = (
                    img_warp[0, :3] * blend_weights['w1']
                    + img_warp[1, :3] * blend_weights['w2']
                    + img_warp[2, :3] * blend_weights['w3']
                )

                if self.save_masks:
                    # Regenerate masks from warped alpha for visualisation only
                    alpha  = torch.ones_like(stack[:, :1])
                    _warp4 = _warp(torch.cat([stack, alpha], 1))
                    self._save_mask_viz(
                        _warp4[0, 3].unsqueeze(0).unsqueeze(0),
                        _warp4[1, 3].unsqueeze(0).unsqueeze(0),
                        _warp4[2, 3].unsqueeze(0).unsqueeze(0),
                        (out_h2, out_w2),
                    )

            else:
                # ---------- fallback path: warp RGB+alpha, compute masks per-frame ----------
                alpha  = torch.ones_like(stack[:, :1])
                img_warp = _warp(torch.cat([stack, alpha], 1)).float()

                mask1 = img_warp[0, 3].unsqueeze(0).unsqueeze(0)
                mask2 = img_warp[1, 3].unsqueeze(0).unsqueeze(0)
                mask3 = img_warp[2, 3].unsqueeze(0).unsqueeze(0)

                if self.save_masks:
                    self._save_mask_viz(mask1, mask2, mask3, (out_h2, out_w2))

                if self.fusion_mode == "AVERAGE":
                    w1, w2, w3 = img_warp[0, :3], img_warp[1, :3], img_warp[2, :3]
                    img12 = (
                        w1 * (w1 / (w1 + w2 + 1e-6))
                        + w2 * (w2 / (w1 + w2 + 1e-6))
                    )
                    fusion = (
                        img12 * (img12 / (img12 + w3 + 1e-6))
                        + w3   * (w3   / (img12 + w3 + 1e-6))
                    )

                elif self.fusion_mode == "EDGE_BLEND":
                    mask1_b = (mask1 > 0.5).float()
                    mask2_b = (mask2 > 0.5).float()
                    mask3_b = (mask3 > 0.5).float()
                    blur_ref = GaussianBlur(kernel_size=(51, 51), sigma=20)
                    mask2_soft = blur_ref(mask2_b).clamp(0, 1)
                    canvas = img_warp[0, :3].unsqueeze(0) * mask1_b
                    canvas = img_warp[2, :3].unsqueeze(0) * mask3_b + canvas * (1 - mask3_b)
                    canvas = img_warp[1, :3].unsqueeze(0) * mask2_soft + canvas * (1 - mask2_soft)
                    fusion = canvas[0]

                else:
                    fusion = img_warp[0, :3]  # should not reach here

            # Clamp + uint8 on the GPU so only 3 bytes/pixel cross PCIe (vs 12 as
            # float32), then a single host copy. Output is HWC BGR uint8.
            pano = fusion.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

        if self.timing:
            print(f"[StabStitch render] warp+blend={time.perf_counter()-t0:.4f}s")

        return pano
