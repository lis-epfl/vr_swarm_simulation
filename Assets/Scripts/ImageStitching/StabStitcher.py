import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os
import sys
import glob
import time
from collections import deque

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
    """Normalise mesh coordinates to [-1, 1], flatten grid dims."""
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


def _linear_blender(ref, tgt, ref_m, tgt_m):
    """Linear (gradient) blending in the overlap region."""
    blur = GaussianBlur(kernel_size=(21, 21), sigma=20)

    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    if r1.numel() == 0 or r2.numel() == 0:
        # Degenerate case: one mask is empty
        return ref * ref_m + tgt * tgt_m

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())
    vec = (center2[0] - center1[0], center2[1] - center1[1])

    # ---- FIX: keep overlap soft (no .round()) to preserve TPS anti-aliasing ----
    ovl = (ref_m * tgt_m)[:, 0].unsqueeze(1)          # soft overlap
    ref_m_ = (ref_m[:, 0].unsqueeze(1) - ovl).clamp(0, 1)
    r, c = torch.nonzero(ovl[0, 0] > 0.01, as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_)
    if r.numel() > 0:
        proj_val = (r.float() - center1[0]) * vec[0] + (c.float() - center1[1]) * vec[1]
        ovl_mask[0, 0, r, c] = (proj_val - proj_val.min()) / (
            proj_val.max() - proj_val.min() + 1e-3
        )

    mask1 = (
        blur(ref_m_ + (1 - ovl_mask) * ref_m[:, 0].unsqueeze(1)) * ref_m + ref_m_
    ).clamp(0, 1)
    mask2 = (1 - mask1) * tgt_m
    return ref * mask1 + tgt * mask2


# ---------------------------------------------------------------------------
# StabStitcher
# ---------------------------------------------------------------------------

class StabStitcher(BaseStitcher):
    """
    Video-aware stitcher using StabStitch++ (Spatial + Temporal + Smooth warp).

    Maintains a rolling buffer of the last BUFFER_LEN frames for temporal
    stability.  On each call to stab_pano() the pipeline:

      1. Preprocesses the three input images and adds them to per-camera buffers.
      2. Runs SpatialNet on every frame-pair in the buffer (14 calls).
      3. Runs TemporalNet independently on each of the three camera streams (3 calls).
      4. Derives temporal-spatial motion (tsmotion) for each pair.
      5. Runs SmoothNet on the full buffer for both pairs.
      6. Performs mesh alignment and TPS warping / blending for the **latest**
         frame only (the one that was just added to the buffer).

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
                 border_size: int = 60):
        # BaseStitcher sets up attributes consumed by StitcherManager's
        # hyperparameter-change detection (active_matcher_type, isRANSAC, …).
        # We pass device="cpu" so its SuperPoint model stays off-GPU; our
        # StabStitch networks are moved to GPU explicitly by StitcherManager.
        super().__init__(device="cpu")

        self.warp_mode = warp_mode
        self.fusion_mode = fusion_mode
        self.timing = timing
        self.save_masks = save_masks
        self.mask_save_dir = mask_save_dir or os.path.join(_THIS_DIR, "mask_debug")
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.border_size = border_size

        # --- Networks ---
        self.spatial_net = SpatialNet()
        self.temporal_net = TemporalNet()
        self.smooth_net = SmoothNet()

        self.spatial_net.eval()
        self.temporal_net.eval()
        self.smooth_net.eval()

        self._load_models()

        # --- Rolling frame buffers ---
        # Low-res tensors (NET_H × NET_W) for motion estimation networks
        self._buf_img1 = deque(maxlen=self.BUFFER_LEN)   # left camera
        self._buf_img2 = deque(maxlen=self.BUFFER_LEN)   # centre camera
        self._buf_img3 = deque(maxlen=self.BUFFER_LEN)   # right camera
        # High-res tensors (original resolution) for final warping
        self._buf_img1_hr = deque(maxlen=self.BUFFER_LEN)
        self._buf_img2_hr = deque(maxlen=self.BUFFER_LEN)
        self._buf_img3_hr = deque(maxlen=self.BUFFER_LEN)

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
        """BGR uint8 numpy → normalised float tensor at NET resolution (CPU)."""
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

    # ------------------------------------------------------------------
    # Network wrappers
    # ------------------------------------------------------------------

    def _run_spatial(self, img1_list, img2_list):
        """
        Run SpatialNet on each consecutive frame pair.

        Returns
        -------
        smotion_list1, smotion_list2 : lists of [1, H, W, 2] tensors (CUDA)
        smesh_list1,   smesh_list2   : lists of [1, H, W, 2] tensors (CUDA)
        """
        rigid = _get_rigid_mesh(1, self.NET_H, self.NET_W)  # CUDA if available
        smotion1_list, smotion2_list = [], []
        smesh1_list,   smesh2_list   = [], []

        for t1, t2 in zip(img1_list, img2_list):
            with torch.no_grad():
                out = build_SpatialNet(
                    self.spatial_net, t1.cuda(), t2.cuda()
                )
            s1 = out["motion1"]   # [1, grid_h+1, grid_w+1, 2]
            s2 = out["motion2"]
            smotion1_list.append(s1)
            smotion2_list.append(s2)
            smesh1_list.append(rigid + s1)
            smesh2_list.append(rigid + s2)

        return smotion1_list, smotion2_list, smesh1_list, smesh2_list

    def _run_temporal(self, img_list):
        """
        Run TemporalNet on a sequence.  Returns motion_list with len == BUFFER_LEN
        (first entry is zeros, as produced by build_TemporalNet).
        """
        with torch.no_grad():
            out = build_TemporalNet(self.temporal_net, img_list)
        return out["motion_list"]  # list of BUFFER_LEN tensors [1, H, W, 2] CUDA

    def _compute_tsmotion(self, smotion_list, smesh_list, tmotion_list):
        """
        Convert frame-t temporal motion into a temporal-spatial motion relative
        to the (t-1)-th frame's spatial mesh.  Mirrors the data-preparation step
        in test_online_tra_threeview.py.
        """
        rigid = _get_rigid_mesh(1, self.NET_H, self.NET_W)
        norm_rigid = _get_norm_mesh(rigid, self.NET_H, self.NET_W)
        tsmotion_list = []

        for k in range(len(smotion_list)):
            if k == 0:
                tsmotion = smotion_list[0].clone() * 0
            else:
                smesh_prev    = smesh_list[k - 1]
                tmotion_k     = tmotion_list[k]           # temporal motion at frame k
                tmesh         = rigid + tmotion_k

                norm_smesh_prev = _get_norm_mesh(smesh_prev, self.NET_H, self.NET_W)
                norm_tmesh      = _get_norm_mesh(tmesh,      self.NET_H, self.NET_W)

                tsmesh   = torch_tps_transform_point.transformer(
                    norm_tmesh, norm_rigid, norm_smesh_prev
                )
                tsmotion = _recover_mesh(tsmesh, self.NET_H, self.NET_W) - smesh_list[k]

            tsmotion_list.append(tsmotion)

        return tsmotion_list

    def _run_smooth(self, tsmotion_list1, tsmotion_list2, smesh_list1, smesh_list2):
        """
        Run SmoothNet on the full BUFFER_LEN-frame window.
        Zeroes the first tsmotion entry (no previous history at frame 0).

        Returns smooth_mesh1, smooth_mesh2  shape [1, T, grid_h+1, grid_w+1, 2].
        """
        tsmotion_list1 = list(tsmotion_list1)   # avoid mutating originals
        tsmotion_list2 = list(tsmotion_list2)
        tsmotion_list1[0] = tsmotion_list1[0] * 0
        tsmotion_list2[0] = tsmotion_list2[0] * 0

        with torch.no_grad():
            out = build_SmoothNet(
                self.smooth_net,
                tsmotion_list1, tsmotion_list2,
                smesh_list1,    smesh_list2,
            )
        return out["smooth_mesh1"], out["smooth_mesh2"]

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
    # Public API
    # ------------------------------------------------------------------

    def stab_pano(self, images, subset1, subset2) -> np.ndarray:
        """
        Produce a stabilised panorama from three images.

        Parameters
        ----------
        images  : list of numpy BGR uint8 arrays
        subset1 : [left_idx, centre_idx]
        subset2 : [centre_idx, right_idx]

        Returns
        -------
        panorama : numpy uint8 array
        """
        img1 = images[subset1[0]]   # left
        img2 = images[subset1[1]]   # centre
        img3 = images[subset2[1]]   # right

        # Add current frame to buffers
        self._buf_img1.append(self._preprocess(img1))
        self._buf_img2.append(self._preprocess(img2))
        self._buf_img3.append(self._preprocess(img3))
        self._buf_img1_hr.append(self._to_hr_tensor(img1))
        self._buf_img2_hr.append(self._to_hr_tensor(img2))
        self._buf_img3_hr.append(self._to_hr_tensor(img3))

        if len(self._buf_img1) < self.BUFFER_LEN:
            return self._fallback_concat(img1, img2, img3)

        return self._full_pipeline()

    # ------------------------------------------------------------------
    # Internal: fallback and full pipeline
    # ------------------------------------------------------------------

    def _fallback_concat(self, img1, img2, img3) -> np.ndarray:
        """Naive horizontal concatenation used while the buffer is filling."""
        return np.hstack([img1, img2, img3]).astype(np.uint8)

    def _full_pipeline(self) -> np.ndarray:
        """
        Run the complete StabStitch++ pipeline on the current buffer and
        return the panorama for the most recent frame.
        """
        img1_list    = list(self._buf_img1)
        img2_list    = list(self._buf_img2)
        img3_list    = list(self._buf_img3)
        img1_hr_list = list(self._buf_img1_hr)
        img2_hr_list = list(self._buf_img2_hr)
        img3_hr_list = list(self._buf_img3_hr)

        t0 = time.perf_counter() if self.timing else None

        # ---------- pair 1-2 ----------
        smotion12_1, smotion12_2, smesh12_1, smesh12_2 = self._run_spatial(img1_list, img2_list)
        t1 = time.perf_counter() if self.timing else None

        tmotion_stream1 = self._run_temporal(img1_list)
        tmotion_stream2 = self._run_temporal(img2_list)
        t2 = time.perf_counter() if self.timing else None

        tsmotion12_1 = self._compute_tsmotion(smotion12_1, smesh12_1, tmotion_stream1)
        tsmotion12_2 = self._compute_tsmotion(smotion12_2, smesh12_2, tmotion_stream2)
        smooth12_1, smooth12_2 = self._run_smooth(
            tsmotion12_1, tsmotion12_2, smesh12_1, smesh12_2
        )
        t3 = time.perf_counter() if self.timing else None

        # ---------- pair 2-3 ----------
        smotion23_1, smotion23_2, smesh23_1, smesh23_2 = self._run_spatial(img2_list, img3_list)
        t4 = time.perf_counter() if self.timing else None

        # Stream 2 temporal is shared; stream 3 is new
        tmotion_stream3 = self._run_temporal(img3_list)
        t5 = time.perf_counter() if self.timing else None

        tsmotion23_1 = self._compute_tsmotion(smotion23_1, smesh23_1, tmotion_stream2)
        tsmotion23_2 = self._compute_tsmotion(smotion23_2, smesh23_2, tmotion_stream3)
        smooth23_1, smooth23_2 = self._run_smooth(
            tsmotion23_1, tsmotion23_2, smesh23_1, smesh23_2
        )
        t6 = time.perf_counter() if self.timing else None

        # smooth*_* shape: [1, BUFFER_LEN, grid_h+1, grid_w+1, 2]
        # Rename to match the test script convention:
        #   warp12_mesh1/2 = left pair;  warp23_mesh1/2 = right pair
        warp12_mesh1 = smooth12_1
        warp12_mesh2 = smooth12_2
        warp23_mesh1 = smooth23_1
        warp23_mesh2 = smooth23_2

        # ---------- scale meshes to HR resolution ----------
        _, _, hr_h, hr_w = img1_hr_list[-1].shape
        sx = hr_w / self.NET_W
        sy = hr_h / self.NET_H

        def _scale(m):
            return torch.stack([m[..., 0] * sx, m[..., 1] * sy], -1)

        warp12_mesh1 = _scale(warp12_mesh1)
        warp12_mesh2 = _scale(warp12_mesh2)
        warp23_mesh1 = _scale(warp23_mesh1)
        warp23_mesh2 = _scale(warp23_mesh2)

        # ---------- work only on the latest frame ----------
        # All T-frame meshes are used to compute the per-frame offset, but we
        # only need the output for frame index -1 (the most recent frame).
        fi = self.BUFFER_LEN - 1  # == -1

        # Per-frame mesh alignment: warp23 meshes are offset so that the
        # overlapping centre-image meshes (warp12_mesh2 and warp23_mesh1)
        # coincide.  Offset is averaged over all grid points for this frame.
        m12_2_fi = warp12_mesh2[:, fi, ...]   # [1, grid_h+1, grid_w+1, 2]
        m23_1_fi = warp23_mesh1[:, fi, ...]
        offset = (m12_2_fi - m23_1_fi).reshape(1, -1, 2).mean(1, keepdim=True)  # [1, 1, 2]
        offset = offset.unsqueeze(1)  # [1, 1, 1, 2] — broadcast over grid dims

        m12_1_fi = warp12_mesh1[:, fi, ...]
        m12_2_fi_aligned = m12_2_fi                          # unchanged
        m23_1_fi_aligned = m23_1_fi + offset.squeeze(1)     # shifted
        m23_2_fi_aligned = warp23_mesh2[:, fi, ...] + offset.squeeze(1)
        middle_mesh_fi   = (m12_2_fi_aligned + m23_1_fi_aligned) / 2.0

        # ---------- first canvas: find bounding box of all four meshes ----------
        all_x = torch.stack([
            m12_1_fi[..., 0], m12_2_fi_aligned[..., 0],
            m23_1_fi_aligned[..., 0], m23_2_fi_aligned[..., 0]
        ])
        all_y = torch.stack([
            m12_1_fi[..., 1], m12_2_fi_aligned[..., 1],
            m23_1_fi_aligned[..., 1], m23_2_fi_aligned[..., 1]
        ])
        width_min  = all_x.min()
        width_max  = all_x.max()
        height_min = all_y.min()
        height_max = all_y.max()
        out_w = width_max  - width_min
        out_h = height_max - height_min

        def _shift(m, wx, hy):
            return torch.stack([m[..., 0] - wx, m[..., 1] - hy], -1)

        m12_1_s   = _shift(m12_1_fi,         width_min, height_min) 
        m12_2_s   = _shift(m12_2_fi_aligned, width_min, height_min)
        m23_1_s   = _shift(m23_1_fi_aligned, width_min, height_min)
        m23_2_s   = _shift(m23_2_fi_aligned, width_min, height_min)
        mid_s     = _shift(middle_mesh_fi,   width_min, height_min)

        # ---------- TPS alignment: img1 and img3 meshes through middle plane ----------
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
        w2_min = all_x2.min()
        w2_max = all_x2.max()
        h2_min = all_y2.min()
        h2_max = all_y2.max()
        out_w2 = w2_max - w2_min
        out_h2 = h2_max - h2_min

        out_size = (int(out_h2.item()), int(out_w2.item()))

        # Translate meshes to new canvas origin
        m1_final  = _shift(m12_1_tps, w2_min, h2_min)
        m2_final  = _shift(mid_s,     w2_min, h2_min)
        m3_final  = _shift(m23_2_tps, w2_min, h2_min)

        # Normalised meshes for the HR rigid grid
        rigid_hr      = _get_rigid_mesh(1, hr_h, hr_w)
        norm_rigid_hr = _get_norm_mesh(rigid_hr, hr_h, hr_w)

        norm_m1 = _get_norm_mesh(m1_final, out_h2, out_w2)
        norm_m2 = _get_norm_mesh(m2_final, out_h2, out_w2)
        norm_m3 = _get_norm_mesh(m3_final, out_h2, out_w2)

        img1_t = img1_hr_list[fi].cuda()
        img2_t = img2_hr_list[fi].cuda()
        img3_t = img3_hr_list[fi].cuda()

        norm_rig3 = torch.cat([norm_rigid_hr, norm_rigid_hr, norm_rigid_hr], 0)

        # ---------- single warp pass (always with alpha for mask extraction) ----------
        alpha = torch.ones_like(img1_t[:, 0].unsqueeze(1))
        img1_t = torch.cat([img1_t, alpha], 1)
        img2_t = torch.cat([img2_t, alpha], 1)
        img3_t = torch.cat([img3_t, alpha], 1)

        img_warp = torch_tps_transform.transformer(
            torch.cat([img1_t, img2_t, img3_t], 0),
            torch.cat([norm_m1, norm_m2, norm_m3], 0),
            norm_rig3,
            out_size,
            mode=self.warp_mode,
        )

        # Soft alpha masks — each fusion mode thresholds as needed
        mask1 = img_warp[0, 3].unsqueeze(0).unsqueeze(0)
        mask2 = img_warp[1, 3].unsqueeze(0).unsqueeze(0)
        mask3 = img_warp[2, 3].unsqueeze(0).unsqueeze(0)

        # ---------- optional mask visualisation (before fusion) ----------
        if self.save_masks:
            self._save_mask_viz(mask1, mask2, mask3, out_size)

        # ---------- fusion ----------
        if self.fusion_mode == "AVERAGE":
            w1, w2, w3 = img_warp[0, :3], img_warp[1, :3], img_warp[2, :3]
            img12 = (
                w1 * (w1 / (w1 + w2 + 1e-6))
                + w2 * (w2 / (w1 + w2 + 1e-6))
            )
            fusion = (
                img12 * (img12 / (img12 + w3 + 1e-6))
                + w3 * (w3 / (img12 + w3 + 1e-6))
            )

        elif self.fusion_mode == "REFERENCE":
            # Premultiplied-alpha "over" composite.
            # The TPS warp produces premultiplied boundary pixels (colour * alpha,
            # black outside), so the correct formula is simply:
            #   result = fg_premult + bg_premult * (1 - fg_alpha)
            # This gives sub-pixel anti-aliasing at every edge instead of the dark
            # fringe that a hard 0.5 threshold creates.
            # Layer order: img1 base → img3 over img1 → img2 (reference) on top.
            canvas = img_warp[0, :3].unsqueeze(0)
            canvas = img_warp[2, :3].unsqueeze(0) + canvas * (1 - mask3)
            canvas = img_warp[1, :3].unsqueeze(0) + canvas * (1 - mask2)
            fusion = canvas[0]

        elif self.fusion_mode == "EDGE_BLEND":
            # Hybrid: reference (img2) interior is pixel-perfect, edges feather
            # into the side images.  Blurring the reference's binary mask produces
            # a weight that is ~1 deep inside and falls to 0 at the boundary,
            # so only the border strip ever sees any mixing.
            mask1_b = (mask1 > 0.5).float()
            mask2_b = (mask2 > 0.5).float()
            mask3_b = (mask3 > 0.5).float()

            blur_ref = GaussianBlur(kernel_size=(51, 51), sigma=20)
            mask2_soft = blur_ref(mask2_b).clamp(0, 1)

            canvas = img_warp[0, :3].unsqueeze(0) * mask1_b
            canvas = img_warp[2, :3].unsqueeze(0) * mask3_b + canvas * (1 - mask3_b)
            canvas = img_warp[1, :3].unsqueeze(0) * mask2_soft + canvas * (1 - mask2_soft)
            fusion = canvas[0]

        elif self.fusion_mode == "REFERENCE_BLEND":
            # Two-step composite:
            #   1. Full LINEAR blend — gives good seam quality across all three images.
            #   2. Composite img2 (centre) on top using a weight map that is exactly
            #      1.0 in the interior of the reference (no ghosting there) and
            #      transitions to 0 over a thin border strip (LINEAR is used there).
            #      Strategy: erode the binary reference mask by border_size pixels so
            #      the interior stays at 1, then blur the hard eroded edge for a
            #      gradual ramp.  Only the border strip ever sees LINEAR blending.
            img12 = _linear_blender(
                img_warp[0, :3].unsqueeze(0), img_warp[1, :3].unsqueeze(0),
                mask1, mask2
            )
            mask12 = mask1 + mask2 - mask1 * mask2
            linear_fusion = _linear_blender(
                img12, img_warp[2, :3].unsqueeze(0), mask12, mask3
            )[0]  # [3, H, W]

            # Erode the reference mask — pixels within border_size of any edge
            # are zeroed so only the true interior survives.
            mask2_b = (mask2 > 0.5).float()
            ks = 2 * self.border_size + 1
            mask2_eroded = (-F.max_pool2d(-mask2_b, kernel_size=ks, stride=1,
                                          padding=self.border_size)).clamp(0, 1)

            # Smooth the hard eroded boundary for a gradual ramp into LINEAR.
            blur_ref = GaussianBlur(kernel_size=(self.blur_kernel_size, self.blur_kernel_size), sigma=self.blur_sigma)
            # Clamp to mask2_b so the weight never leaks outside img2's boundary
            # (outside, img_warp[1] is 0 and would otherwise darken linear_fusion).
            mask2_interior = blur_ref(mask2_eroded).clamp(0, 1) * mask2_b

            fusion = img_warp[1, :3] * mask2_interior[0] + linear_fusion * (1 - mask2_interior[0])

        else:  # LINEAR
            img12 = _linear_blender(
                img_warp[0, :3].unsqueeze(0), img_warp[1, :3].unsqueeze(0),
                mask1, mask2
            )
            mask12 = mask1 + mask2 - mask1 * mask2
            fusion = _linear_blender(img12, img_warp[2, :3].unsqueeze(0), mask12, mask3)
            fusion = fusion[0]

        pano = fusion.cpu().numpy().transpose(1, 2, 0)

        if self.timing:
            t7 = time.perf_counter()
            print(
                f"[StabStitch timing] "
                f"spatial12={t1-t0:.3f}s  "
                f"temporal12={t2-t1:.3f}s  "
                f"smooth12={t3-t2:.3f}s  "
                f"spatial23={t4-t3:.3f}s  "
                f"temporal3={t5-t4:.3f}s  "
                f"smooth23={t6-t5:.3f}s  "
                f"warp+blend={t7-t6:.3f}s  "
                f"total={t7-t0:.3f}s"
            )

        return pano.clip(0, 255).astype(np.uint8)