"""
Pose-initialized planar homography stitcher.

For the vertical-plane (facade) and nadir (cameras-down) swarm configurations, the
scene is one dominant plane and parallax is structurally absent.  There a single
homography per view is *geometrically exact* and computable in closed form from camera
intrinsics + camera pose + the plane, with no image content at all -- no feature
matching, no neural network, and therefore no failure on low-texture surfaces like
asphalt, water or a uniform facade.

This is an alternative to :class:`StabStitcher`, not a replacement.  StabStitch++'s
parallax-tolerant TPS warps are what make the radially-outward configuration work, and
it stays the correct choice there.

Because Unity supplies exact pose and exact intrinsics, the mosaic should be
pixel-perfect on a truly planar scene with pose noise off.  Any visible seam in that
configuration is a bug, not a limitation -- there is nothing being estimated.  That is
the acceptance criterion this class is written against.

Structure
---------
The geometry lives in :mod:`planar_geometry` (pure numpy, unit-tested offline by
``tools/planar_selftest.py``).  This class owns the torch render path and the gates.

``_correction`` holds delta-pose and plane corrections and is applied on top of the
published pose every frame.  It is the identity here; :meth:`compute_warps` is the slot
where a future refiner (photometric or bundle-adjustment) will optimise it on the slow
thread.  That split is deliberate: poses change every frame, so the geometric solve must
run inline at frame rate, but pose *corrections* (GNSS bias) drift slowly and belong at
the warp thread's cadence.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from BaseStitcher import BaseStitcher
import planar_geometry as pg

# Failing-gate bits. Pre-shift values, matching StitcherThreading's REASON_* -- see
# write_panorama_memory for how they are packed.
REASON_CANVAS = 1
REASON_DISTORTION = 2
REASON_PHOTOMETRIC = 4
REASON_PLANE_INVALID = 32


class PlanarStitcher(BaseStitcher):
    # How long the plane may stay raycast-invalid before the panorama is pulled. A
    # single miss is harmless -- a wrong plane distance is only a uniform scale error --
    # but a sustained miss means we are mosaicking onto a plane that isn't there.
    PLANE_INVALID_GRACE_S = 2.0

    def __init__(self, device="cpu"):
        # device="cpu" mirrors StabStitcher: it keeps BaseStitcher's SuperPoint off the
        # GPU. Subclassing BaseStitcher is required rather than stylistic --
        # checkHyperparaChanges unconditionally reads cylindricalWarp, active_matcher_type,
        # isRANSAC, checks, ratio_thresh, score_threshold and focal off the active
        # stitcher, and set_fusion_mode raises if fusion_mode is missing.
        super().__init__(device="cpu")
        self.fusion_mode = "REFERENCE_BLEND"   # accepted and ignored; see _blend_weights

        self.render_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Refiner slot. dpose is per-view (rotation, translation) corrections in the
        # camera frame; plane is a (normal, d) override. Identity for now.
        self._correction = {"dpose": None, "plane": None}

        # Cached canvas pixel grid, keyed on canvas size: the only thing that changes
        # between frames is the homography, so this is built once.
        self._grid_key = None
        self._grid = None

        self._plane_invalid_since = None
        self._last_stats = {}
        self._last_log = 0.0

    # ------------------------------------------------------------------ public API

    def planar_pano(self, views, intrinsics, plane, config):
        """
        Build the mosaic from pose alone.

        ``views``      list of block records (image + Unity pose) from read_block_memory
        ``intrinsics`` (fx, fy, cx, cy) for the block resolution
        ``plane``      dict from read_dynamic_state (Unity-world normal + offset)
        ``config``     planar_* metadata fields

        Returns ``(panorama BGR uint8 | None, quality_ok, quality_reason)`` -- the same
        contract as ``StabStitcher.stab_pano``.
        """
        canvas_w, canvas_h = config["canvas"]
        if canvas_w <= 0 or canvas_h <= 0:
            return None, False, REASON_CANVAS

        # Sustained plane invalidity is a real failure; a transient miss is not.
        if not plane.get("plane_valid", False):
            if self._plane_invalid_since is None:
                self._plane_invalid_since = time.time()
            elif time.time() - self._plane_invalid_since > self.PLANE_INVALID_GRACE_S:
                return None, False, REASON_PLANE_INVALID
        else:
            self._plane_invalid_since = None

        K = pg.intrinsics_matrix(*intrinsics)
        frame, cams = self._build_geometry(views, K, plane, config)
        if frame is None:
            return None, False, REASON_CANVAS
        if len(cams) < 2:
            return None, False, REASON_CANVAS

        # Canvas centred on the reference view's footprint, at a fixed scale.
        mpp = config["metres_per_pixel"]
        if mpp <= 0.0:
            return None, False, REASON_CANVAS
        M = pg.canvas_to_plane_matrix(mpp, canvas_w * 0.5, canvas_h * 0.5)

        # Projective sanity: a homography cannot fold the way a TPS mesh can, so mesh
        # distortion is meaningless. Anisotropic stretch is the pathology that occurs.
        aniso_max = config.get("aniso_max", 0.0) or float("inf")
        kept = []
        for cam in cams:
            H = pg.homography_canvas_to_image(cam["G"], M)
            aniso = pg.homography_anisotropy(H, canvas_w, canvas_h)
            if not np.isfinite(aniso) or aniso > aniso_max:
                # Not a hard failure: drop this view and mosaic the rest. A single
                # grazing drone should not cost the whole panorama.
                continue
            cam["H"] = H
            cam["aniso"] = aniso
            kept.append(cam)

        if len(kept) < 2:
            return None, False, REASON_DISTORTION

        pano, coverage, overlap_psnr = self._render(kept, canvas_w, canvas_h, config)
        if pano is None:
            return None, False, REASON_CANVAS

        self._last_stats = {
            "views": len(kept),
            "dropped": len(cams) - len(kept),
            "coverage": coverage,
            "mean_range": float(np.mean([c["range"] for c in kept])),
            "max_aniso": float(max(c["aniso"] for c in kept)),
            "overlap_psnr": overlap_psnr,
        }
        self._maybe_log()

        min_coverage = config.get("min_coverage", 0.0)
        if coverage < min_coverage:
            return None, False, REASON_CANVAS

        # Off by default. The measurement is always taken (it is what quantifies pose
        # error), but gating on it would hide the panorama permanently once pose noise
        # is injected, which defeats the purpose of injecting it.
        if config.get("psnr_gate", False):
            threshold = config.get("psnr_threshold", 0.0)
            if np.isfinite(overlap_psnr) and overlap_psnr < threshold:
                return None, False, REASON_PHOTOMETRIC

        return pano, True, 0

    def compute_warps(self):
        """
        Refiner slot, run on the warp thread.

        Intentionally a no-op: the planar geometric solve costs microseconds and runs
        inline in :meth:`planar_pano` every frame, because the poses it consumes change
        every frame -- deferring it here would reintroduce exactly the pose lag the
        per-block pose snapshot exists to remove.  What belongs on this thread is
        estimating ``_correction`` (photometric alignment or a 2-DoF-landmark bundle
        adjustment), since pose *corrections* are slowly varying even though poses are not.
        """
        time.sleep(0.1)

    # ------------------------------------------------------------------ geometry

    def _build_geometry(self, views, K, plane, config):
        """
        Convert Unity poses to CV convention, build the plane frame, and compute G per
        view.  Returns ``(PlaneFrame | None, [cam dicts])``.
        """
        posed = [v for v in views if v.get("pos") is not None and v.get("quat") is not None]
        if not posed:
            return None, []

        n = pg.unity_dir_to_rh(plane["plane_normal"])
        if np.linalg.norm(n) < 1e-6:
            return None, []
        n = n / np.linalg.norm(n)
        d = float(plane["plane_d"])

        # Reference view = the middle of the published selection. Unity orders slots by
        # camera index, so this is a stable choice frame to frame, which matters because
        # the canvas frame is built from it.
        ref = posed[len(posed) // 2]
        R_ref, C_ref = pg.unity_pose_to_cv(ref["pos"], ref["quat"])

        # Canvas origin: where the reference camera's principal ray meets the plane.
        # Its forward axis is the third row of the world->camera rotation.
        forward_ref = R_ref[2]
        origin, t = pg.ray_plane_intersect(C_ref, forward_ref, n, d,
                                           max_range=config.get("max_range", np.inf))
        if origin is None:
            # Reference camera isn't looking at the plane; fall back to the foot of the
            # perpendicular from it, so the canvas is still somewhere sensible.
            origin = C_ref - (np.dot(n, C_ref) - d) * n

        # e1 from the reference camera's right axis (row 0), so "right in the panorama"
        # is "right in the view the pilot is nominally looking through". e2 = n x e1 is
        # right-handed by construction.
        frame = pg.build_plane_frame(n, d, origin, R_ref[0], -R_ref[1])

        max_range = config.get("max_range", np.inf)
        cams = []
        for v in posed:
            R, C = pg.unity_pose_to_cv(v["pos"], v["quat"])
            R, C = self._apply_correction(v, R, C)

            # Reject views that cannot see the plane at all before doing any work.
            _, rng = pg.ray_plane_intersect(C, R[2], n, d, max_range=max_range)
            if not np.isfinite(rng):
                continue

            cams.append({
                "view": v,
                "G": pg.build_G(K, R, C, frame),
                "range": rng,
            })

        return frame, cams

    def _apply_correction(self, view, R, C):
        """Apply the refiner's delta-pose. Identity until compute_warps estimates one."""
        dpose = self._correction.get("dpose")
        if dpose is None:
            return R, C
        delta = dpose.get(view["drone_id"])
        if delta is None:
            return R, C
        dR, dC = delta
        return dR @ R, C + dC

    # ------------------------------------------------------------------ render

    def _canvas_grid(self, canvas_w, canvas_h):
        """Homogeneous canvas pixel grid ``[3, H*W]``, cached per canvas size."""
        key = (canvas_w, canvas_h, str(self.render_device))
        if self._grid_key == key:
            return self._grid

        ys, xs = torch.meshgrid(
            torch.arange(canvas_h, dtype=torch.float32, device=self.render_device),
            torch.arange(canvas_w, dtype=torch.float32, device=self.render_device),
            indexing="ij")
        grid = torch.stack([xs.reshape(-1), ys.reshape(-1),
                            torch.ones(canvas_h * canvas_w, dtype=torch.float32,
                                       device=self.render_device)], dim=0)
        self._grid_key, self._grid = key, grid
        return grid

    def _render(self, cams, canvas_w, canvas_h, config):
        """
        Warp every view into the canvas and blend.

        Returns ``(pano BGR uint8, coverage, worst pairwise overlap PSNR)``.
        """
        dev = self.render_device
        grid = self._canvas_grid(canvas_w, canvas_h)          # [3, HW]
        src_h, src_w = cams[0]["view"]["image"].shape[:2]
        max_range = float(config.get("max_range", 0.0)) or float("inf")
        feather_px = max(1.0, float(config.get("feather_px", 1)))

        H_stack = torch.from_numpy(
            np.stack([c["H"] for c in cams]).astype(np.float32)).to(dev)   # [N,3,3]

        # One batched matmul yields the sampling coordinate *and* every rejection test,
        # because the third component is camera depth in metres (K's last row is [0,0,1]).
        uvw = torch.matmul(H_stack, grid)                                  # [N,3,HW]
        w = uvw[:, 2]
        valid_depth = (w > 1e-6) & (w < max_range)
        w_safe = torch.where(valid_depth, w, torch.ones_like(w))
        u = uvw[:, 0] / w_safe
        v = uvw[:, 1] / w_safe

        in_bounds = (u >= 0) & (u <= src_w - 1) & (v >= 0) & (v <= src_h - 1)
        valid = valid_depth & in_bounds                                    # [N,HW]

        # Distance-to-border feather. A homography's alpha mask has a closed form, so
        # this needs no convolution at all -- unlike a TPS mesh's, which is why
        # StabStitcher blurs and erodes instead. Measured in SOURCE pixels, so the seam
        # width in canvas pixels scales with each view's local magnification.
        du = torch.minimum(u, (src_w - 1) - u)
        dv = torch.minimum(v, (src_h - 1) - v)
        feather = (torch.minimum(du, dv) / feather_px).clamp(0.0, 1.0)
        weight = torch.where(valid, feather, torch.zeros_like(feather))    # [N,HW]

        # Upload uint8 and convert on the GPU: at N=8 and 800x450 that is 8.6 MB over
        # PCIe instead of 34 MB, and moves the cast off the CPU.
        imgs = torch.from_numpy(
            np.stack([c["view"]["image"] for c in cams])).to(dev)          # [N,H,W,3] u8
        imgs = imgs.permute(0, 3, 1, 2).float()                            # [N,3,H,W]

        # grid_sample wants normalised coords with the same align_corners convention
        # StabStitcher's TPS flow uses -- match it exactly or everything shifts half a pixel.
        flow = torch.stack([2.0 * u / (src_w - 1) - 1.0,
                            2.0 * v / (src_h - 1) - 1.0], dim=-1)
        flow = flow.view(len(cams), canvas_h, canvas_w, 2)

        # fp16 halves the sampling cost on the GPU, but grid_sampler_2d has no CPU half
        # kernel — so the dtype follows the device rather than being assumed.
        sample_dtype = torch.float16 if dev.type == "cuda" else torch.float32
        warped = F.grid_sample(imgs.to(sample_dtype), flow.to(sample_dtype),
                               mode="bilinear", padding_mode="zeros",
                               align_corners=True).float()

        wgt = weight.view(len(cams), 1, canvas_h, canvas_w)
        wsum = wgt.sum(dim=0)                                              # [1,Hc,Wc]
        covered = wsum > 1e-6
        # Normalising by the per-pixel weight sum is what generalises this to any N;
        # a reference-based or pairwise scheme does not.
        pano = (warped * wgt).sum(dim=0) / wsum.clamp_min(1e-6)
        pano = torch.where(covered, pano, torch.zeros_like(pano))

        coverage = float(covered.float().mean().item())
        out = pano.clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        # Always measured: this is the number that quantifies pose error, and it costs
        # only a few ops given the already-warped stack. Whether it gates is the
        # caller's decision.
        return out, coverage, self._overlap_psnr(warped, wgt)

    @staticmethod
    def _overlap_psnr(warped, wgt):
        """Worst pairwise PSNR over the region two views both cover."""
        n = warped.shape[0]
        worst = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                both = (wgt[i] > 1e-3) & (wgt[j] > 1e-3)
                count = int(both.sum().item())
                if count < 1024:
                    continue
                diff = (warped[i] - warped[j]) * both
                mse = float((diff ** 2).sum().item()) / (count * warped.shape[1])
                if mse > 1e-9:
                    worst = min(worst, 10.0 * np.log10(255.0 ** 2 / mse))
        return worst

    def _maybe_log(self, period=5.0):
        now = time.time()
        if now - self._last_log < period:
            return
        self._last_log = now
        s = self._last_stats
        psnr = s.get("overlap_psnr", float("inf"))
        psnr_txt = "n/a" if not np.isfinite(psnr) else f"{psnr:.1f} dB"
        print(f"[PLANAR] {s.get('views', 0)} views "
              f"(+{s.get('dropped', 0)} dropped) | coverage {s.get('coverage', 0):.0%} | "
              f"mean range {s.get('mean_range', 0):.1f} m | "
              f"max anisotropy {s.get('max_aniso', 0):.2f} | overlap PSNR {psnr_txt}")
