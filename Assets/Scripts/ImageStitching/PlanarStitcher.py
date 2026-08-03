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

# How overlapping views are combined. Mirrors PlanarBlendMode in PyUniSharingFast.cs.
#
# FEATHER cross-fades every view that covers a pixel, weighted by distance to its own
# image border. That is the right thing only while the geometry is exact: any residual
# misalignment (pose error, a facade that is not quite a plane) shows up as ghosting
# across the whole overlap region, and in a wall formation the overlap *is* most of the
# canvas, so the ghosting is everywhere rather than confined to a seam.
#
# NEAREST gives each pixel to exactly one view -- the one seeing that point closest to
# the plane normal, i.e. the most face-on -- so nothing is ever averaged and residual
# error can only appear as a discontinuity along the seam, not as a doubled image.
BLEND_FEATHER = 0
BLEND_NEAREST = 1

# Diagnostic overlay: which view a patch came from. Mirrors PlanarDebugView in
# PyUniSharingFast.cs. TINT keeps the imagery legible under a colour wash (so you can
# see both the content and its provenance); FLAT discards the imagery and shows the
# partition alone, which is the clearer picture of where the seams actually fall.
DEBUG_OFF = 0
DEBUG_TINT = 1
DEBUG_FLAT = 2

# Keyed on DRONE ID, not on position in the selection: a colour that reshuffles whenever
# a drone joins or leaves the selection tells you nothing frame to frame. BGR, because
# that is the format on the wire.
DEBUG_PALETTE = [
    ("red",     (0, 0, 255)),
    ("green",   (0, 255, 0)),
    ("blue",    (255, 0, 0)),
    ("yellow",  (0, 255, 255)),
    ("magenta", (255, 0, 255)),
    ("cyan",    (255, 255, 0)),
    ("orange",  (0, 140, 255)),
    ("violet",  (255, 0, 130)),
    ("lime",    (0, 255, 140)),
    ("pink",    (180, 105, 255)),
]

# How far TINT pulls a pixel toward its view's colour.
DEBUG_TINT_STRENGTH = 0.45

# Bit 0 of the block header's poseStatus: Unity sets it on every pose it actually
# writes. Mirrors POSE_VALID in PyUniSharingFast.cs / StitcherThreading.py.
POSE_VALID = 1 << 0


def pose_is_usable(view):
    """
    Whether a block's pose can be turned into a rotation at all.

    A block slot Unity has never written reads back as zeros: flag 0 ("ready"),
    droneId 0 (a *legal* drone id, unlike the -1 Unity writes to retire a slot) and an
    all-zero quaternion.  That used to reach ``quat_to_matrix`` and raise, which aborts
    the whole frame -- one unwritten slot cost the entire panorama, and the traceback
    pointed at the geometry rather than at the wire.  The producer no longer leaves such
    slots claiming to be drone 0, but the consumer must not depend on that: this is a
    shared-memory handshake with no schema enforcement, so a torn or stale block is
    always possible and costs at most its own view.
    """
    pos, quat = view.get("pos"), view.get("quat")
    if pos is None or quat is None:
        return False
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        return False
    if float(np.dot(quat, quat)) < 1e-6:
        return False
    # Explicitly-invalid pose (a retired slot, or a v1 producer's zeroed tail).
    return bool(int(view.get("pose_status", 0)) & POSE_VALID)


def debug_colour(drone_id):
    """``(name, (B, G, R))`` for a drone id. Stable for the lifetime of that drone."""
    return DEBUG_PALETTE[int(drone_id) % len(DEBUG_PALETTE)]


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
        self._unposed = 0
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
            # Every block arriving without a usable pose is a wire/producer problem, and
            # REASON_PLANE_INVALID is the bit Unity spells as "no usable scene plane /
            # pose", so report it as that rather than as a canvas failure.
            if views and self._unposed == len(views):
                return None, False, REASON_PLANE_INVALID
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

        pano, coverage, overlap_psnr = self._render(kept, canvas_w, canvas_h, config, M)
        if pano is None:
            return None, False, REASON_CANVAS

        self._last_stats = {
            "views": len(kept),
            "dropped": len(cams) - len(kept),
            "unposed": self._unposed,
            "coverage": coverage,
            "mean_range": float(np.mean([c["range"] for c in kept])),
            "max_aniso": float(max(c["aniso"] for c in kept)),
            "overlap_psnr": overlap_psnr,
            "blend": ("nearest"
                      if config.get("blend_mode", BLEND_NEAREST) == BLEND_NEAREST
                      else "feather"),
            "debug_view": int(config.get("debug_view", DEBUG_OFF)),
            "drone_ids": [c["view"]["drone_id"] for c in kept],
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
        posed = [v for v in views if pose_is_usable(v)]
        self._unposed = len(views) - len(posed)
        if not posed:
            return None, []

        n = pg.unity_dir_to_rh(plane["plane_normal"])
        if np.linalg.norm(n) < 1e-6:
            return None, []
        n = n / np.linalg.norm(n)
        d = float(plane["plane_d"])

        # Reference view = the drone Unity nominated as the centre of the swarming plane
        # (SelectPlanarCentreCamera).  The canvas origin and axes are built from it, so a
        # reference that changes identity translates and rotates the whole mosaic -- which
        # is why it is Unity's decision rather than one re-derived here.  Taking the median
        # of the id-sorted selection instead, as this used to, picks the median *drone id*:
        # not the geometric centre, and it jumps whenever the selection gains or loses a
        # drone.  Falling back to that only when Unity publishes no centre (-1, or a centre
        # whose view was dropped) keeps a v2 producer that predates the field working.
        ref = None
        centre_id = plane.get("centre_drone_id", -1)
        if centre_id >= 0:
            ref = next((v for v in posed if v.get("drone_id") == centre_id), None)
        if ref is None:
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

            # Camera centre in the plane frame: (a, b) is its footprint on the plane and
            # h its perpendicular standoff. BLEND_NEAREST needs only these three scalars
            # per view -- see _render for why the per-pixel obliquity reduces to them.
            rel = C - frame.O
            cams.append({
                "view": v,
                "G": pg.build_G(K, R, C, frame),
                "range": rng,
                "plane_ab": (float(rel @ frame.e1), float(rel @ frame.e2)),
                "plane_h": abs(float(rel @ frame.n)),
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

    def _render(self, cams, canvas_w, canvas_h, config, M):
        """
        Warp every view into the canvas and combine.

        ``M`` is the canvas-pixel -> plane-coordinate matrix, needed by BLEND_NEAREST to
        put each canvas pixel in the same frame as the camera footprints.

        Returns ``(pano BGR uint8, coverage, worst pairwise overlap PSNR)``.
        """
        dev = self.render_device
        grid = self._canvas_grid(canvas_w, canvas_h)          # [3, HW]
        src_h, src_w = cams[0]["view"]["image"].shape[:2]
        max_range = float(config.get("max_range", 0.0)) or float("inf")
        feather_px = max(1.0, float(config.get("feather_px", 1)))
        blend_mode = int(config.get("blend_mode", BLEND_NEAREST))
        debug_view = int(config.get("debug_view", DEBUG_OFF))

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

        # Distance-to-border falloff. A homography's alpha mask has a closed form, so
        # this needs no convolution at all -- unlike a TPS mesh's, which is why
        # StabStitcher blurs and erodes instead. Measured in SOURCE pixels, so the seam
        # width in canvas pixels scales with each view's local magnification.
        # Under BLEND_FEATHER this *is* the blend weight; under BLEND_NEAREST it only
        # biases the winner away from image edges.
        du = torch.minimum(u, (src_w - 1) - u)
        dv = torch.minimum(v, (src_h - 1) - v)
        feather = (torch.minimum(du, dv) / feather_px).clamp(0.0, 1.0)
        weight = torch.where(valid, feather, torch.zeros_like(feather))    # [N,HW]

        if blend_mode == BLEND_NEAREST:
            weight = self._nearest_weights(cams, weight, grid, M)

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

        # Always measured: this is the number that quantifies pose error, and it costs
        # only a few ops given the already-warped stack. Whether it gates is the
        # caller's decision.
        #
        # Measured over geometric *coverage*, not over the blend weights: BLEND_NEAREST
        # makes the weights one-hot, so a weight-based overlap test would find no
        # overlapping pixels at all and silently retire the one diagnostic that
        # quantifies pose error. Two views still see the same ground there whether or
        # not both are drawn.
        #
        # Taken before the debug overlay, which would otherwise be measured instead of
        # the imagery -- and it stays measured while the overlay is on, so the number in
        # the log still refers to the mosaic you would get with the overlay off.
        cover = valid.view(len(cams), 1, canvas_h, canvas_w).float()
        overlap_psnr = self._overlap_psnr(warped, cover)

        # In place, so the overlay costs no extra [N,3,Hc,Wc] allocation.
        if debug_view != DEBUG_OFF:
            self._apply_debug_colours(cams, warped, debug_view)

        wgt = weight.view(len(cams), 1, canvas_h, canvas_w)
        wsum = wgt.sum(dim=0)                                              # [1,Hc,Wc]
        covered = wsum > 1e-6
        # Normalising by the per-pixel weight sum is what generalises this to any N;
        # a reference-based or pairwise scheme does not.
        pano = (warped * wgt).sum(dim=0) / wsum.clamp_min(1e-6)
        pano = torch.where(covered, pano, torch.zeros_like(pano))

        coverage = float(covered.float().mean().item())
        out = pano.clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        return out, coverage, overlap_psnr

    @staticmethod
    def _apply_debug_colours(cams, warped, debug_view):
        """
        Colour each view's contribution in place, so a patch's origin is readable off
        the panorama itself.

        Applied to the warped stack rather than to the mosaic, which means it works the
        same in both blend modes and tells you something different in each: under
        NEAREST the regions come out flat and hard-edged (that *is* the partition),
        while under FEATHER the overlaps come out as blends of two colours, which is a
        direct picture of how much of the canvas is being averaged.
        """
        colours = torch.tensor(
            [debug_colour(c["view"]["drone_id"])[1] for c in cams],
            dtype=warped.dtype, device=warped.device).view(len(cams), 3, 1, 1)

        if debug_view == DEBUG_FLAT:
            warped.copy_(colours.expand_as(warped))
        else:
            warped.lerp_(colours.expand_as(warped), DEBUG_TINT_STRENGTH)

    def _nearest_weights(self, cams, border, grid, M):
        """
        Winner-take-all weights: each covered canvas pixel goes to a single view.

        The winner is the view seeing that plane point closest to the plane normal --
        the most face-on, least foreshortened look at it. Because every canvas pixel
        lies *on* the plane, that obliquity has a closed form in three per-view scalars:
        with the camera at in-plane footprint ``(a_v, b_v)`` and perpendicular standoff
        ``h_v``, the incidence cosine at plane point ``(a, b)`` is

            cos = h_v / sqrt((a - a_v)^2 + (b - b_v)^2 + h_v^2)

        (the numerator is constant per view precisely because ``n . X == d`` everywhere
        on the canvas). Maximising it is a Voronoi partition of the plane by camera
        footprint, so each drone renders the patch of facade it is parked in front of.

        The border falloff multiplies the score rather than gating it, which keeps seams
        off the source-image edges: a view running out of frame fades below a
        neighbour's score before it runs out of pixels, so the winner changes over solid
        image on both sides instead of at a hard image boundary.
        """
        M_t = torch.as_tensor(np.asarray(M, dtype=np.float32), device=grid.device)
        ab = torch.matmul(M_t, grid)                       # [3,HW]; M's last row is [0,0,1]

        cam_a = torch.tensor([c["plane_ab"][0] for c in cams],
                             dtype=torch.float32, device=grid.device).unsqueeze(1)
        cam_b = torch.tensor([c["plane_ab"][1] for c in cams],
                             dtype=torch.float32, device=grid.device).unsqueeze(1)
        cam_h = torch.tensor([c["plane_h"] for c in cams],
                             dtype=torch.float32, device=grid.device).unsqueeze(1)

        # In-place after the first subtraction: these are [N, H*W] float32 temporaries,
        # 30 MB apiece at a 1200x800 canvas and 8 views, and the render loop shares its
        # GPU with the warp thread.
        da = ab[0].unsqueeze(0) - cam_a
        db = ab[1].unsqueeze(0) - cam_b
        d2 = da.mul_(da).add_(db.mul_(db)).add_(cam_h * cam_h)
        score = d2.sqrt_().clamp_min_(1e-6).reciprocal_().mul_(cam_h).mul_(border)

        best_score, best = score.max(dim=0)
        weight = torch.zeros_like(score)
        weight.scatter_(0, best.unsqueeze(0),
                        (best_score > 0).to(score.dtype).unsqueeze(0))
        return weight

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
        # "unposed" is a producer-side symptom, not a geometry one: blocks that arrived
        # without a usable pose (never written, or retired mid-frame). Reported
        # separately from the anisotropy drops so the two are not confused.
        unposed = s.get("unposed", 0)
        unposed_txt = f", {unposed} unposed" if unposed else ""
        print(f"[PLANAR] {s.get('views', 0)} views "
              f"(+{s.get('dropped', 0)} dropped{unposed_txt}) | blend {s.get('blend', '?')} | "
              f"coverage {s.get('coverage', 0):.0%} | "
              f"mean range {s.get('mean_range', 0):.1f} m | "
              f"max anisotropy {s.get('max_aniso', 0):.2f} | overlap PSNR {psnr_txt}")

        # A colour map is useless without the key, and the selection changes as drones
        # join, die or fall out of range -- so reprint it alongside the stats rather
        # than once at startup.
        if s.get("debug_view", DEBUG_OFF) != DEBUG_OFF:
            legend = "  ".join(f"drone {i} = {debug_colour(i)[0]}"
                               for i in s.get("drone_ids", []))
            mode = "flat" if s["debug_view"] == DEBUG_FLAT else "tint"
            print(f"[PLANAR] debug view ({mode}):  {legend}")
