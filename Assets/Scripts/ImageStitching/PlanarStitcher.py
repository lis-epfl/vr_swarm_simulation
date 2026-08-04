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

``_correction`` holds plane and delta-pose corrections applied on top of the published
pose every frame.  Both are estimated by :meth:`compute_warps` on the warp thread, and
each is independently switchable from Unity (``planarPlaneSweep`` / ``planarPoseRefine``).
That split is deliberate: poses change every frame, so the geometric solve must run
inline at frame rate, but the *corrections* drift slowly and belong at the warp thread's
cadence.

The two estimators are complementary rather than alternative, because they move
different parameters:

``_correction["plane"]`` -- **plane sweep**, one global scalar: an additive offset on the
published plane distance, chosen by scanning candidate offsets and keeping the one that
minimises photometric disagreement between views.  A plane-distance error appears in each
view as a *scale* about its own footprint, so no per-view translation can absorb it; this
is the only thing that can.  Correspondingly it can only fix errors that are common to
the whole formation -- there is one of it, so it cannot make drone 3 agree with drones 2
and 4 at the same time.

``_correction["dpose"]`` -- **pose refiner**, two DoF per view: a per-drone canvas-plane
translation, recovered by phase-correlating each view against the consensus of the
others.  Differential GNSS error and compass bias both appear as a lateral shift of that
view's footprint (a yaw error rotates the ray bundle, which at a facade is dominantly a
translation plus a second-order keystone), so one translation absorbs the bulk of both.
It cannot represent scale -- hence the sweep -- nor the keystone from a gimbal-pitch bias.

They are complementary but not fully separable.  A depth error dilates each view's
content about that view's own footprint, and averaged over the overlap that is a
*convergent* translation field -- so a convergent set of per-drone position errors is
indistinguishable from a plane-depth error by any amount of image evidence.  Measured on
a symmetric formation with such an error, the sweep moves the plane and buys nothing
(``tools/planar_selftest.py`` asserts exactly that, so it cannot be "fixed" by accident),
while the refiner still recovers most of it.  The practical guidance: turn the sweep on
when the *plane* is the thing you are unsure of -- a facade distance drawn off a map, or
a raycast onto geometry that may not be there -- and leave it off when per-drone position
error dominates.  With both on, the views are still driven into mutual agreement; what is
not identifiable is which parameter deserved the credit, which shows up as a clean mosaic
at a slightly wrong absolute scale rather than as a visible seam.

Both estimators are *gated*, not trusted: the sweep rejects candidates whose overlap
collapses, and the refiner rejects any view whose correlation peak is not clearly
dominant.  Be precise about what that second gate buys, because it is easy to overclaim:
it rejects an individual *measurement* that has no clear answer -- most usefully in the
first passes, while the consensus is still made of misaligned views -- and together with
``refine_max_shift`` it bounds how far one bad measurement can move a patch.  It is not a
guarantee against a repetitive facade: run to convergence, a periodic scene can settle
into a self-consistent solution shifted by a whole period, and at that point every view
genuinely does agree with every other, so no per-measurement confidence test can see it.
What protects against that is the clamp plus the fact that the correction is a residual
on top of a pose that is already roughly right.
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
DEBUG_TINT_STRENGTH = 0.22

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

        # Estimated corrections, applied on top of the published pose every frame.
        #   "plane" : float, additive metres on the published plane distance
        #   "dpose" : {drone_id: (dR, dC)} in right-handed world
        # Rebound as a whole new dict by compute_warps, never mutated in place: the
        # render thread reads it without a lock, so it must see one consistent
        # generation rather than a half-updated one.
        self._correction = {"dpose": None, "plane": None}

        # Cached canvas pixel grid, keyed on canvas size: the only thing that changes
        # between frames is the homography, so this is built once.
        #
        # The estimators get their OWN cache rather than sharing this one. Their canvas
        # is a different size, so a single cache would miss on every call and rebuild the
        # grid twice per frame -- and it would be doing that from two threads onto one
        # pair of attributes, which is a race as well as a waste.
        self._grid_key = None
        self._grid = None
        self._est_grid_key = None
        self._est_grid = None
        self._hann_key = None
        self._hann = None

        # Latest-wins snapshot published by planar_pano for the warp thread. One
        # attribute rebind, so the reader gets a whole frame or the previous one -- the
        # same mailbox discipline the shared-memory producers use.
        self._frame_snapshot = None

        # Estimator state (warp thread only).
        self._plane_offset = 0.0        # low-passed sweep result, metres
        # drone_id -> low-passed camera-centre correction, right-handed WORLD metres.
        # Deliberately not stored in plane coordinates: the plane frame's e1 comes from
        # the reference camera's right axis, so it rotates when the reference drone yaws
        # or is re-elected, and a stored (a, b) would then silently mean something
        # different from the value that was measured.
        self._pose_shift = {}
        self._sweep_stats = {}
        self._refine_stats = {}

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

        # Hand this frame to the warp thread. Published only once a frame has actually
        # rendered, so the estimators never run on geometry the render path itself
        # rejected. One rebind of one attribute: the reader takes a whole frame or the
        # previous one, never a mixture.
        if config.get("plane_sweep", False) or config.get("pose_refine", False):
            self._frame_snapshot = (views, K, plane, config)

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
        Estimate ``_correction`` on the warp thread.

        The geometric solve itself is *not* here: it costs microseconds and runs inline
        in :meth:`planar_pano` every frame, because the poses it consumes change every
        frame -- deferring it would reintroduce exactly the pose lag the per-block pose
        snapshot exists to remove.  What runs here is the estimation of the slowly-varying
        corrections on top of those poses.

        The two estimators are independently switchable and share one warped stack per
        pass, since both need the same thing: every view resampled into a common canvas.
        With both off this costs one sleep, matching the previous no-op behaviour.
        """
        snapshot = self._frame_snapshot
        if snapshot is None:
            time.sleep(0.1)
            return

        views, K, plane, config = snapshot
        do_sweep = bool(config.get("plane_sweep", False))
        do_refine = bool(config.get("pose_refine", False))
        if not (do_sweep or do_refine):
            # Let a correction estimated before the flag was cleared decay out rather
            # than stay frozen in the geometry: switching an estimator off in the
            # inspector should visibly return to the raw published pose.
            if self._plane_offset != 0.0 or self._pose_shift:
                self._plane_offset = 0.0
                self._pose_shift = {}
                self._sweep_stats = {}
                self._refine_stats = {}
                self._correction = {"dpose": None, "plane": None}
            time.sleep(0.1)
            return

        try:
            if do_sweep:
                self._sweep_plane(views, K, plane, config)
            else:
                self._plane_offset = 0.0
                self._sweep_stats = {}

            if do_refine:
                self._refine_poses(views, K, plane, config)
            else:
                self._pose_shift = {}
                self._refine_stats = {}
        finally:
            # One rebind, so a render mid-update sees either generation whole. Built even
            # on failure, or a half-applied estimate would persist into the next frame.
            self._correction = {
                "plane": self._plane_offset if do_sweep else None,
                "dpose": self._pose_to_dpose() if do_refine else None,
            }

        # The estimators are cheap (a handful of reduced-resolution warps); without a
        # floor this thread would spin on the same snapshot at hundreds of Hz and steal
        # the GPU from the render loop it shares a device and a GIL with.
        time.sleep(0.05)

    def _pose_to_dpose(self):
        """Per-view world corrections -> the ``{drone_id: (dR, dC)}`` shape."""
        if not self._pose_shift:
            return None
        return {did: (None, dC) for did, dC in self._pose_shift.items()}

    # ------------------------------------------------------------------ estimators

    # Canvas downscale for the estimator passes. The corrections are sub-metre
    # quantities over a canvas tens of metres across, so a quarter-resolution canvas
    # still resolves them to well under a source pixel -- and it cuts the sweep's cost
    # by 16x, which is what keeps this off the render loop's GPU budget.
    ESTIMATOR_SCALE = 0.25

    # A candidate whose multi-covered area falls below this fraction of the incumbent's
    # is rejected outright. Photometric cost is a mean over overlapping pixels, so a
    # depth that shrinks the overlap to a small well-aligned patch would otherwise win
    # by destroying the very evidence it is scored on.
    SWEEP_MIN_OVERLAP_RATIO = 0.6

    # Correlation peak must beat the best rival outside its own neighbourhood by this
    # much. Chiefly this refuses measurements taken against a consensus that is itself
    # still a blur of misaligned views -- in the first pass over a badly-posed formation
    # most views fail it, and they start passing as the correction converges. See the
    # module docstring for what it does *not* protect against.
    REFINE_MIN_PEAK_RATIO = 1.25

    def _sweep_plane(self, views, K, plane, config):
        """
        One global scalar: the additive plane-distance offset that best aligns the views.

        Scans ``sweep_steps`` candidates spanning +/- ``sweep_range`` metres around the
        *current* estimate, then fits a parabola through the best sample and its two
        neighbours so the result is not quantised to the step size.  Brute force rather
        than gradient descent on purpose: the cost is a 1-D curve over a bounded interval
        with one broad minimum, so a scan cannot diverge or find a spurious local
        optimum, and its cost is fixed and predictable -- which matters on a thread that
        shares a GPU with the render loop.
        """
        span = float(config.get("sweep_range", 0.0))
        steps = int(config.get("sweep_steps", 0))
        if span <= 0.0 or steps < 3:
            self._sweep_stats = {"skipped": "range/steps not set"}
            return

        # Odd count so the incumbent estimate is always itself a candidate: without it a
        # converged sweep dithers between the two samples straddling the true value.
        if steps % 2 == 0:
            steps += 1

        centre = self._plane_offset
        offsets = np.linspace(centre - span, centre + span, steps)

        costs, areas = [], []
        for off in offsets:
            cost, area = self._photometric_cost(views, K, plane, config, float(off))
            costs.append(cost)
            areas.append(area)

        costs = np.asarray(costs, dtype=np.float64)
        areas = np.asarray(areas, dtype=np.float64)

        # Score only candidates that still have enough overlap to have been scored
        # fairly, measured against the best-covered candidate rather than an absolute
        # area (the formation's own spread sets what "full overlap" even means).
        ref_area = float(areas.max()) if areas.size else 0.0
        usable = np.isfinite(costs) & (areas >= self.SWEEP_MIN_OVERLAP_RATIO * ref_area)
        if ref_area <= 0.0 or not np.any(usable):
            self._sweep_stats = {"skipped": "no candidate had usable overlap"}
            return

        masked = np.where(usable, costs, np.inf)
        i = int(np.argmin(masked))
        best = float(offsets[i])

        # Sub-step refinement, but only from an interior sample flanked by two usable
        # ones -- a parabola through an edge sample extrapolates outside the scanned
        # interval, which is precisely where nothing was measured.
        if 0 < i < steps - 1 and usable[i - 1] and usable[i + 1]:
            c0, c1, c2 = masked[i - 1], masked[i], masked[i + 1]
            denom = c0 - 2.0 * c1 + c2
            if denom > 1e-12:
                step = float(offsets[1] - offsets[0])
                best += 0.5 * step * float(c0 - c2) / denom

        rate = float(config.get("refine_rate", 0.25))
        rate = min(1.0, max(0.0, rate))
        self._plane_offset = (1.0 - rate) * self._plane_offset + rate * best

        self._sweep_stats = {
            "offset": self._plane_offset,
            "raw": best,
            "cost": float(masked[i]),
            "cost_span": float(np.nanmax(masked[np.isfinite(masked)]) - masked[i]),
            "usable": int(usable.sum()),
            "steps": steps,
        }

    def _refine_poses(self, views, K, plane, config):
        """
        Two DoF per view: the canvas-plane translation aligning each view to the others.

        Each view is phase-correlated against the mean of every *other* view over the
        region they share.  Leave-one-out rather than against the finished mosaic,
        because under winner-take-all a view *is* the mosaic wherever it wins, so
        correlating against the blend would compare a view largely against itself and
        report a confident zero shift.
        """
        # Explicitly the offset the sweep just produced, not the one still sitting in
        # _correction: that is only rebound at the end of compute_warps, so reading it
        # here would measure the pose residual against last pass's plane and leave the
        # two estimators permanently one generation out of step.
        frame, cams, M_s, cw, ch, mpp_s = self._estimator_geometry(
            views, K, plane, config, plane_offset=self._plane_offset)
        if frame is None or len(cams) < 2:
            self._refine_stats = {"skipped": "fewer than 2 usable views"}
            return

        grey, valid = self._warp_grey(cams, M_s, cw, ch)
        n, ch_c, cw_c = grey.shape

        # Exposure differs per aircraft (independent auto-exposure on real drones), and
        # phase correlation is only invariant to it once the DC term is removed.
        count = valid.sum(dim=0, keepdim=True)
        shared = count >= 2
        grey = self._zero_mean(grey, valid & shared)

        max_shift_m = float(config.get("refine_max_shift", 0.0))
        rate = min(1.0, max(0.0, float(config.get("refine_rate", 0.25))))

        raw, rejected = {}, 0
        for i in range(n):
            other_valid = valid.clone()
            other_valid[i] = False
            other_count = other_valid.sum(dim=0)
            both = valid[i] & (other_count > 0) & shared[0]
            if int(both.sum().item()) < 4096:
                rejected += 1
                continue

            consensus = (grey * other_valid).sum(dim=0) / other_count.clamp_min(1)

            # Apodise, do not just mask. Multiplying both inputs by the same hard mask
            # correlates the *mask* as well as the imagery, and the mask is identical in
            # both -- so it contributes a large peak at zero shift no matter what the
            # pixels say. That both biases the estimate toward "no correction needed" and
            # defeats the confidence gate, because the spurious peak is always dominant.
            # Softening the mask edge and tapering the canvas boundary leaves the peak
            # determined by image content, which is the only thing that carries the
            # answer.
            win = self._soften(both) * self._hann_window(ch_c, cw_c)
            dx, dy, ratio = self._phase_shift(grey[i] * win, consensus * win)
            if ratio < self.REFINE_MIN_PEAK_RATIO:
                rejected += 1
                continue

            # Two sign flips, both easy to get backwards -- hence the end-to-end
            # assertion in tools/planar_selftest.py rather than trust in this comment:
            #   1. _phase_shift reports where view i sits *relative to* the consensus,
            #      so the correction that moves it back is the negation.
            #   2. canvas rows grow downward while e2 points up the plane (the -s in
            #      canvas_to_plane_matrix), so the y axis flips again on the way out.
            # The two cancel on b and compose on a.
            da, db = -dx * mpp_s, dy * mpp_s
            if max_shift_m > 0.0 and (da * da + db * db) > max_shift_m * max_shift_m:
                rejected += 1
                continue
            raw[cams[i]["view"]["drone_id"]] = da * frame.e1 + db * frame.e2

        if not raw:
            self._refine_stats = {"skipped": f"all {n} views rejected"}
            return

        # Gauge fix. Every view's shift is free, so the whole set can slide together and
        # cost nothing photometrically -- the mosaic would then wander off the plane
        # frame over successive updates while every pairwise alignment stayed perfect.
        # Removing the mean pins that one unobservable direction.
        mean = sum(raw.values()) / len(raw)

        # These are RESIDUALS, not absolute corrections: _build_geometry already applied
        # the standing correction before this stack was warped, so what was just measured
        # is what is still left over. Accumulate; do not overwrite, or the correction can
        # never converge past one step's worth and instead oscillates around the error.
        shifts = {}
        for did, dC in raw.items():
            prev = self._pose_shift.get(did)
            total = (dC - mean) * rate
            if prev is not None:
                total = prev + total
            if max_shift_m > 0.0:
                mag = float(np.linalg.norm(total))
                if mag > max_shift_m:
                    total = total * (max_shift_m / mag)
            shifts[did] = total

        # Views that dropped out of the selection drop out of the correction too, or a
        # drone that returns is warped by an offset measured minutes ago against a
        # formation that has since moved.
        self._pose_shift = shifts
        self._refine_stats = {
            "accepted": len(shifts),
            "rejected": rejected,
            "worst_shift": max(float(np.linalg.norm(v)) for v in shifts.values()),
            "worst_residual": max(float(np.linalg.norm(v - mean)) for v in raw.values()),
        }

    # ------------------------------------------------------------------ estimator internals

    def _estimator_geometry(self, views, K, plane, config, plane_offset=None,
                            apply_dpose=True):
        """
        Geometry for one estimator pass, on a canvas reduced by ``ESTIMATOR_SCALE``.

        The canvas covers the same *plane extent* as the render canvas but with fewer
        pixels, so metres-per-pixel grows by the inverse of the scale -- shrinking the
        pixel count without shrinking the field of view, which is what keeps the two
        estimators looking at the same overlap the render path does.
        """
        canvas_w, canvas_h = config["canvas"]
        mpp = float(config.get("metres_per_pixel", 0.0))
        if canvas_w <= 0 or canvas_h <= 0 or mpp <= 0.0:
            return None, [], None, 0, 0, 0.0

        frame, cams = self._build_geometry(views, K, plane, config,
                                           plane_offset=plane_offset,
                                           apply_dpose=apply_dpose)
        if frame is None:
            return None, [], None, 0, 0, 0.0

        cw = max(16, int(canvas_w * self.ESTIMATOR_SCALE))
        ch = max(16, int(canvas_h * self.ESTIMATOR_SCALE))
        mpp_s = mpp * (canvas_w / float(cw))
        M_s = pg.canvas_to_plane_matrix(mpp_s, cw * 0.5, ch * 0.5)
        return frame, cams, M_s, cw, ch, mpp_s

    def _warp_grey(self, cams, M, canvas_w, canvas_h):
        """
        Warp every view into the estimator canvas as greyscale.

        Returns ``(grey [N,H,W] float32, valid [N,H,W] bool)``.  A lean sibling of
        :meth:`_render`: no blending, no debug overlay, no colour -- the estimators score
        agreement between views, and all three of those would only add cost and, in the
        blend's case, mix the very views being compared.
        """
        dev = self.render_device
        grid = self._estimator_grid(canvas_w, canvas_h)
        src_h, src_w = cams[0]["view"]["image"].shape[:2]

        H_stack = torch.from_numpy(
            np.stack([pg.homography_canvas_to_image(c["G"], M)
                      for c in cams]).astype(np.float32)).to(dev)

        uvw = torch.matmul(H_stack, grid)
        w = uvw[:, 2]
        valid_depth = w > 1e-6
        w_safe = torch.where(valid_depth, w, torch.ones_like(w))
        u, v = uvw[:, 0] / w_safe, uvw[:, 1] / w_safe
        valid = (valid_depth & (u >= 0) & (u <= src_w - 1)
                 & (v >= 0) & (v <= src_h - 1))

        imgs = torch.from_numpy(
            np.stack([c["view"]["image"] for c in cams])).to(dev)
        grey = imgs.permute(0, 3, 1, 2).float().mean(dim=1, keepdim=True)

        flow = torch.stack([2.0 * u / (src_w - 1) - 1.0,
                            2.0 * v / (src_h - 1) - 1.0], dim=-1)
        flow = flow.view(len(cams), canvas_h, canvas_w, 2)

        # float32 throughout: phase correlation takes an FFT of this, and fp16 rounding
        # on a near-flat facade costs more accuracy than the sampling speed is worth.
        warped = F.grid_sample(grey, flow, mode="bilinear",
                               padding_mode="zeros", align_corners=True)
        return warped[:, 0], valid.view(len(cams), canvas_h, canvas_w)

    # Width of the box blur that softens the overlap mask, in estimator-canvas pixels.
    # Wide enough that the mask edge stops looking like a step to the FFT, narrow enough
    # that it does not eat the overlap on a small shared region.
    MASK_SOFTEN_PX = 9

    @staticmethod
    def _soften(mask, k=MASK_SOFTEN_PX):
        """0/1 mask -> smooth window. Two box passes ~ a triangular roll-off."""
        m = mask.float().unsqueeze(0).unsqueeze(0)
        m = F.avg_pool2d(m, k, stride=1, padding=k // 2)
        m = F.avg_pool2d(m, k, stride=1, padding=k // 2)
        return m[0, 0]

    def _hann_window(self, h, w):
        """Separable Hann over the canvas, cached. Tapers the canvas boundary itself."""
        key = (h, w, str(self.render_device))
        if getattr(self, "_hann_key", None) == key:
            return self._hann
        wy = torch.hann_window(h, periodic=False, device=self.render_device)
        wx = torch.hann_window(w, periodic=False, device=self.render_device)
        self._hann_key, self._hann = key, wy[:, None] * wx[None, :]
        return self._hann

    @staticmethod
    def _zero_mean(grey, mask):
        """Per-view DC removal over ``mask``, so exposure differences do not score."""
        m = mask.float()
        count = m.sum(dim=(1, 2)).clamp_min(1.0)
        mean = (grey * m).sum(dim=(1, 2)) / count
        return (grey - mean.view(-1, 1, 1)) * m

    def _photometric_cost(self, views, K, plane, config, plane_offset):
        """
        Disagreement between views at one candidate plane offset.

        Returns ``(cost, area)``: the mean across-view variance over pixels at least two
        views cover, and how many such pixels there were.  Variance across all covering
        views rather than the pairwise PSNR ``_render`` logs -- that one is O(N^2) with a
        GPU sync per pair, and this runs once per sweep candidate.

        Each view is DC-removed over the shared region first, so the cost measures
        *misalignment* rather than the exposure differences between aircraft.
        """
        # apply_dpose=False: see _build_geometry. The plane must be measured against the
        # published poses, or the refiner's absorption of part of the depth error hides
        # exactly what this is trying to find.
        frame, cams, M_s, cw, ch, _ = self._estimator_geometry(
            views, K, plane, config, plane_offset=plane_offset, apply_dpose=False)
        if frame is None or len(cams) < 2:
            return float("inf"), 0.0

        grey, valid = self._warp_grey(cams, M_s, cw, ch)
        count = valid.sum(dim=0)
        shared = count >= 2
        area = float(shared.sum().item())
        if area < 1024:
            return float("inf"), area

        grey = self._zero_mean(grey, valid & shared.unsqueeze(0))
        m = (valid & shared.unsqueeze(0)).float()
        denom = m.sum(dim=0).clamp_min(1.0)
        mean = (grey * m).sum(dim=0) / denom
        var = ((grey - mean) ** 2 * m).sum(dim=0) / denom
        return float(var[shared].mean().item()), area

    @staticmethod
    def _phase_shift(a, b):
        """
        Shift ``(dx, dy)`` in pixels that moves ``a`` onto ``b``, plus a confidence.

        Normalised cross-power spectrum: the phase of ``A * conj(B)`` carries the
        translation alone, so a pure shift gives a single sharp delta regardless of image
        content or contrast.  Confidence is the peak divided by the largest rival outside
        its immediate neighbourhood; it reads near 1.0 when no shift explains the pair
        better than any other, which is the case worth throwing away.

        The shift is quantised to whole pixels of the estimator canvas.  That is not a
        precision limit in practice: the estimator canvas is coarse but the correction is
        re-measured as a *residual* every pass, so the accumulated value converges well
        inside one of its pixels.

        The sign convention is fixed by ``tools/planar_selftest.py``, which injects a
        known offset and asserts it is recovered; it is far too easy to get backwards by
        reasoning alone.
        """
        h, w = a.shape
        A = torch.fft.rfft2(a)
        B = torch.fft.rfft2(b)
        R = A * B.conj()
        R = R / R.abs().clamp_min(1e-9)
        corr = torch.fft.irfft2(R, s=(h, w))

        flat = corr.reshape(-1)
        peak_idx = int(torch.argmax(flat).item())
        peak = float(flat[peak_idx].item())
        py, px = divmod(peak_idx, w)

        # Second peak, with the winner's neighbourhood masked out so its own shoulder
        # does not count as the rival.
        rival = corr.clone()
        r = max(2, min(h, w) // 32)
        ys = [(py + dy) % h for dy in range(-r, r + 1)]
        xs = [(px + dx) % w for dx in range(-r, r + 1)]
        rival[torch.tensor(ys, device=corr.device).unsqueeze(1),
              torch.tensor(xs, device=corr.device).unsqueeze(0)] = float("-inf")
        second = float(rival.max().item())

        ratio = float("inf") if second <= 1e-9 else peak / max(second, 1e-9)

        # Wrap to signed: the correlation is circular, so a peak past the midpoint is a
        # negative shift, not a large positive one.
        dy = py - h if py > h // 2 else py
        dx = px - w if px > w // 2 else px
        return float(dx), float(dy), ratio

    # ------------------------------------------------------------------ geometry

    def _build_geometry(self, views, K, plane, config, plane_offset=None,
                        apply_dpose=True):
        """
        Convert Unity poses to CV convention, build the plane frame, and compute G per
        view.  Returns ``(PlaneFrame | None, [cam dicts])``.

        ``plane_offset`` overrides the stored sweep correction with an explicit additive
        offset in metres -- that is how the sweep evaluates a candidate without
        disturbing the correction the render path is currently using.  ``None`` means
        "use the stored one", which is what the render path passes.

        ``apply_dpose=False`` ignores the refiner's per-view corrections.  Only the sweep
        passes this, and it is what keeps the two estimators from fighting: a plane-depth
        error appears in any view whose footprint is off-centre as a local *translation*,
        so the refiner will happily absorb part of it: measured on already-refined
        geometry the sweep then sees an error the refiner has hidden, and can be driven
        the wrong way entirely.  Measuring the plane on raw poses keeps the sweep
        unbiased by whatever the refiner has done; the per-drone errors it does not
        correct for shift every candidate's cost about equally and so do not move the
        minimum.
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

        # Additive, not absolute: Unity's raycast keeps tracking the facade as the swarm
        # flies, and the sweep estimates only the slowly-varying residual on top of it.
        # An absolute override would freeze the plane at whatever the sweep last saw.
        if plane_offset is None:
            plane_offset = self._correction.get("plane") or 0.0
        d += float(plane_offset)

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
            if apply_dpose:
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
        """
        Apply the refiner's delta-pose.  Identity until compute_warps estimates one.

        ``dR`` may be ``None``: the current refiner estimates translation only, and a
        None here means "no rotation correction" rather than costing an identity matmul
        per view per frame.  The slot is kept so a rotation-capable refiner can fill it
        without changing this call site.
        """
        dpose = self._correction.get("dpose")
        if dpose is None:
            return R, C
        delta = dpose.get(view["drone_id"])
        if delta is None:
            return R, C
        dR, dC = delta
        if dR is not None:
            R = dR @ R
        return R, C if dC is None else C + dC

    # ------------------------------------------------------------------ render

    def _make_grid(self, canvas_w, canvas_h):
        ys, xs = torch.meshgrid(
            torch.arange(canvas_h, dtype=torch.float32, device=self.render_device),
            torch.arange(canvas_w, dtype=torch.float32, device=self.render_device),
            indexing="ij")
        return torch.stack([xs.reshape(-1), ys.reshape(-1),
                            torch.ones(canvas_h * canvas_w, dtype=torch.float32,
                                       device=self.render_device)], dim=0)

    def _canvas_grid(self, canvas_w, canvas_h):
        """Homogeneous canvas pixel grid ``[3, H*W]``, cached per canvas size."""
        key = (canvas_w, canvas_h, str(self.render_device))
        if self._grid_key == key:
            return self._grid
        grid = self._make_grid(canvas_w, canvas_h)
        self._grid_key, self._grid = key, grid
        return grid

    def _estimator_grid(self, canvas_w, canvas_h):
        """As :meth:`_canvas_grid`, on the warp thread's own cache. See ``__init__``."""
        key = (canvas_w, canvas_h, str(self.render_device))
        if self._est_grid_key == key:
            return self._est_grid
        grid = self._make_grid(canvas_w, canvas_h)
        self._est_grid_key, self._est_grid = key, grid
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

        # Estimator state. Printed only when the corresponding flag is on, so the line
        # stays short in the default pose-only configuration -- and so that "no
        # correction line" unambiguously means "the estimator is off" rather than
        # "the estimator ran and found nothing".
        sweep, refine = self._sweep_stats, self._refine_stats
        if sweep:
            if "skipped" in sweep:
                print(f"[PLANAR] plane sweep idle: {sweep['skipped']}")
            else:
                print(f"[PLANAR] plane sweep: {sweep['offset']:+.2f} m "
                      f"(raw {sweep['raw']:+.2f}) | {sweep['usable']}/{sweep['steps']} "
                      f"candidates usable | cost drop {sweep['cost_span']:.1f}")
        if refine:
            if "skipped" in refine:
                print(f"[PLANAR] pose refine idle: {refine['skipped']}")
            else:
                # Residual is the convergence read-out: it should fall toward zero as
                # the accumulated correction absorbs the error. A residual that stays
                # high while the correction grows means the two are fighting.
                print(f"[PLANAR] pose refine: {refine['accepted']} accepted, "
                      f"{refine['rejected']} rejected (weak peak) | "
                      f"worst correction {refine['worst_shift']:.2f} m | "
                      f"residual {refine['worst_residual']:.3f} m")

        # A colour map is useless without the key, and the selection changes as drones
        # join, die or fall out of range -- so reprint it alongside the stats rather
        # than once at startup.
        if s.get("debug_view", DEBUG_OFF) != DEBUG_OFF:
            legend = "  ".join(f"drone {i} = {debug_colour(i)[0]}"
                               for i in s.get("drone_ids", []))
            mode = "flat" if s["debug_view"] == DEBUG_FLAT else "tint"
            print(f"[PLANAR] debug view ({mode}):  {legend}")
