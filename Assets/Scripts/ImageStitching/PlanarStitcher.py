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

Its candidates are sampled uniformly in **disparity** (``f*B/Z``) and it runs in two modes.
Both are load-bearing rather than tuning.  Misalignment is linear in disparity, and the
basin of attraction is a roughly fixed *pixel* width (``~L*Z/B``, L = the texture's
correlation length), so one step size in pixels is right at every standoff and no step size
in metres is right at two.  A single metres-uniform scan is what made this estimator appear
to latch: at the shipped defaults it stepped 1.0 m against a basin about a metre wide, so
no candidate reliably landed inside the minimum, the argmin was noise, and the low-pass
walked the plane a metre a pass in an arbitrary direction until an operator toggled the
estimator and re-rolled it.  ACQUIRE therefore scans wide and coarse and *snaps* (it is the
initial lock, and damping it would leave the estimate outside the basin it just found);
TRACK scans a few pixels either side of the incumbent, low-passes, and moves only for a
measurable improvement, so a converged sweep sits still.  A run of unusable scans returns
to ACQUIRE -- the automatic form of that operator toggle.

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
collapses and refuses a scan whose cost curve is too flat to contain a minimum at all (a
flat window must mean "no measurement", not "argmin of noise" -- that distinction is the
whole difference between tracking and random-walking), and the refiner rejects any view
whose correlation peak is not clearly dominant.  A view the refiner rejects keeps the
correction it has already earned; only leaving the *selection* retires one.  Be precise
about what that second gate buys, because it is easy to overclaim:
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

# How the canvas gets its scale and centre. Mirrors PlanarCanvasMode in
# PyUniSharingFast.cs.
#
# FIXED is the original behaviour and the default: metres-per-pixel is the operator's
# constant and the canvas centre is the reference camera's principal-ray hit, so the
# mosaic covers the same patch of plane every frame and measurements stay comparable
# across runs. Its cost is that the framing is only right at one standoff -- a tight
# formation fills a fraction of the canvas, a wide one overflows it.
#
# AUTOFIT derives both from where the views actually land on the plane (_fit_canvas), so
# the mosaic frames itself. The scale is quantised and damped rather than continuous:
# a canvas that rescaled every frame would breathe, and the estimators would be measuring
# on a target that never sits still.
#
# Both are modified by the operator's zoom and pan, which are a pure viewing transform on
# top of whichever rest framing the mode produced -- see planar_pano.
CANVAS_MODE_FIXED = 0
CANVAS_MODE_AUTOFIT = 1

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

# ScenePlaneMode.FormationRelative. In this mode Unity publishes no usable normal or
# offset -- there is no raycast out there to produce one -- and the plane is derived here
# from the poses instead (_plane_from_formation). Mirrors StitcherThreading's copy and the
# C# const planeModeFormationRelative; check_wire_layout.py asserts the pair.
PLANE_MODE_FORMATION_RELATIVE = 4


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
        #
        # It carries the frame's DATA ONLY -- (views, K, plane). It deliberately does not
        # carry the config: compute_warps must read the estimator switches from the LIVE
        # metadata, or turning an estimator off cannot take effect. Publishing the config
        # alongside the frame is what made the "both estimators off" reset unreachable --
        # the snapshot was only published when a flag was on, so the branch that tested
        # for both being off could never see a config with both off.
        self._frame_snapshot = None
        self._snapshot_time = 0.0

        # Live estimator config, pushed by StitcherManager.update_planar_metadata on every
        # metadata read -- i.e. it keeps updating while the render path is failing, which
        # is exactly when an operator reaches for the switch.
        self._live_config = None

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

        # Sweep search mode. ACQUIRE runs a wide coarse scan to find the basin at all;
        # TRACK runs a narrow fine scan inside it. They are separate because one scan
        # cannot do both jobs: the basin of attraction is roughly L*Z/B (L = the scene
        # texture's correlation length), which on brick at 30 m is a few tenths of a
        # metre, while the plane prior can be metres wrong. A single scan wide enough to
        # capture the error steps straight over the minimum it is looking for.
        self._sweep_mode = "acquire"
        self._sweep_lost = 0            # consecutive TRACK passes with no usable measurement
        self._last_acquire = 0.0

        # Which rule _plane_from_formation last used ("positions" / "forward"), or None in
        # the raycast modes. Worth logging: the two behave differently under gimbal pitch,
        # and a wall that has collapsed to a single row switches between them silently.
        self._plane_source = None

        # Auto-fit canvas state. Written ONLY by the render path (planar_pano), read by
        # the estimators -- the same single-writer discipline _correction uses in the
        # other direction. Running the stabiliser from both threads would have them
        # fighting over one incumbent and stepping it twice per frame.
        #
        # The centre is kept in right-handed WORLD metres rather than as plane (a, b) for
        # exactly the reason _pose_shift is: the plane frame's e1 comes from the reference
        # camera's right axis, so it rotates when that drone yaws or is re-elected, and a
        # low-pass over (a, b) would then be averaging two different bases together.
        self._fit_mpp = None            # settled metres-per-pixel, or None until first fit
        self._fit_step = 0              # incumbent quantiser step, relative to the anchor
        self._fit_centre_world = None   # low-passed canvas centre, world metres
        self._fit_changed_at = 0.0
        self._fit_extent = None         # (width, height) of the fitted footprint, metres

        self._plane_invalid_since = None
        self._unposed = 0
        self._stale = 0
        self._last_stats = {}
        self._last_log = 0.0
        self._last_est_log = 0.0

    def set_live_config(self, config):
        """
        Publish the current planar metadata for the warp thread.

        Called from the metadata-reading loop rather than from the render path, because
        the switches have to keep arriving while the render path is failing.  One rebind
        of one attribute, same as the frame snapshot.
        """
        self._live_config = config

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
        frame, cams = self._build_geometry(views, K, plane, config, record=True)
        if frame is None:
            # Every block arriving without a usable pose is a wire/producer problem, and
            # REASON_PLANE_INVALID is the bit Unity spells as "no usable scene plane /
            # pose", so report it as that rather than as a canvas failure.
            if views and self._unposed == len(views):
                return None, False, REASON_PLANE_INVALID
            return None, False, REASON_CANVAS
        if len(cams) < 2:
            return None, False, REASON_CANVAS

        # Rest framing, then the operator's viewing transform on top of it.
        #
        # FIXED keeps the historical canvas exactly: the reference view's footprint at the
        # centre, at the operator's scale. AUTOFIT replaces both with a fit to where the
        # views actually land. Either way zoom and pan are applied afterwards and only
        # here -- the estimators run on the rest framing, see _estimator_geometry.
        mpp = float(config.get("metres_per_pixel", 0.0))
        if mpp <= 0.0:
            return None, False, REASON_CANVAS

        mode = int(config.get("canvas_mode", CANVAS_MODE_FIXED))
        # Computed in both modes: it is what bounds the pan, and panning is as useful on a
        # fixed canvas as on a fitted one. Cheap -- one 3x3 solve and four corners a view.
        bbox = self._footprint_bbox(cams, config)
        centre_ab = (0.0, 0.0)
        if mode == CANVAS_MODE_AUTOFIT:
            fitted = self._fit_canvas(frame, cams, config, bbox)
            if fitted is not None:
                mpp, centre_ab = fitted
            elif self._fit_mpp:
                # Nothing fittable this frame (too few views on the plane). Hold the last
                # settled framing rather than snapping back to the operator's constant --
                # a one-frame jump in scale is far more disruptive than a stale one.
                mpp, centre_ab = float(self._fit_mpp), self._centre_in_frame(frame)

        mpp_view, centre_ab = self._view_transform(config, mpp, bbox, centre_ab)
        if mpp_view <= 0.0 or not np.isfinite(mpp_view):
            return None, False, REASON_CANVAS
        M = self._canvas_matrix(mpp_view, canvas_w, canvas_h, centre_ab)

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

        # Hand this frame to the warp thread. Published on every frame that rendered,
        # unconditionally -- gating the publish on the estimator switches is what froze
        # the snapshot (config included) the moment an estimator was switched off, so the
        # estimator kept running on one stale frame and "off" never reached it.
        #
        # Data only, no config: see __init__. Timestamped so compute_warps can tell a
        # live frame from one frozen by a render failure -- every early return above
        # yields a blank panorama, so a freeze is silent unless the age is checked.
        # One rebind of one tuple: the reader takes a whole frame or the previous one.
        self._frame_snapshot = (views, K, plane)
        self._snapshot_time = time.monotonic()

        self._last_stats = {
            "views": len(kept),
            "dropped": len(cams) - len(kept),
            "unposed": self._unposed,
            "stale": self._stale,
            "coverage": coverage,
            "mean_range": float(np.mean([c["range"] for c in kept])),
            "max_aniso": float(max(c["aniso"] for c in kept)),
            "overlap_psnr": overlap_psnr,
            "canvas_mode": "autofit" if mode == CANVAS_MODE_AUTOFIT else "fixed",
            "mpp_fit": float(mpp),
            "mpp_view": float(mpp_view),
            "canvas_extent": (canvas_w * float(mpp_view), canvas_h * float(mpp_view)),
            "zoom": float(config.get("zoom", 1.0) or 1.0),
            # The scale at which the sharpest view's source pixels land on the plane. Zoom
            # past it and the canvas is resolving detail the cameras never captured, so it
            # is the honest end of "inspect closer" -- and it moves with the standoff, so
            # it has to be measured rather than assumed.
            "best_gsd": float(min(c["range"] for c in kept) / K[1, 1]),
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

        The switches and tuning come from ``_live_config`` while the frame comes from
        ``_frame_snapshot``.  Reading both from the snapshot made "off" unreachable and
        let a render failure freeze the estimator on one frame; the two now have
        independent freshness, which is the point.
        """
        snapshot = self._frame_snapshot
        config = self._live_config
        if snapshot is None or config is None:
            time.sleep(0.1)
            return

        # A frame the render path stopped refreshing. Every path that skips the publish
        # returns a blank panorama, so a stale snapshot means the render is already down;
        # continuing to estimate on it converges the correction onto a dead frame and then
        # applies that answer to live geometry once the render recovers.
        age = time.monotonic() - self._snapshot_time
        if age > self.SNAPSHOT_MAX_AGE_S:
            self._sweep_stats = {"skipped": f"snapshot stale ({age:.1f} s)"}
            self._refine_stats = {"skipped": f"snapshot stale ({age:.1f} s)"}
            self._maybe_log_estimators()
            time.sleep(0.1)
            return

        views, K, plane = snapshot
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
                self._sweep_mode = "acquire"
                self._sweep_lost = 0
                self._correction = {"dpose": None, "plane": None}
            time.sleep(0.1)
            return

        try:
            if do_sweep:
                self._sweep_plane(views, K, plane, config)
            else:
                # Only this estimator's state. The two are independently switchable by
                # design and the self-test pins a case where the refiner is the only one
                # that helps, so clearing both together would throw away a good
                # correction to fix a bad one.
                self._plane_offset = 0.0
                self._sweep_mode = "acquire"
                self._sweep_lost = 0
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

        # Logged from HERE, not from _maybe_log: that line is printed at the end of
        # planar_pano, after every early return, so it goes silent in exactly the
        # situations worth diagnosing.
        self._maybe_log_estimators()

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

    # A snapshot older than this is a frozen frame, not a slow one. Comfortably longer
    # than a render period (~20 Hz) and shorter than the plane-invalid grace, so a brief
    # render hiccup does not stop the estimators but a sustained failure does.
    SNAPSHOT_MAX_AGE_S = 0.5

    # ---- sweep sampling, in DISPARITY rather than metres --------------------------------
    #
    # Misalignment between two views of a plane is linear in disparity f*B/Z, not in Z.
    # A scan sampled uniformly in metres therefore has the wrong step everywhere except
    # at one range: at the defaults that shipped (+/-4 m over 9 candidates = 1.0 m apart)
    # the step was WIDER than the basin of attraction it was searching, so zero or one
    # candidate landed inside the minimum and the argmin was effectively a dice roll --
    # which is what made a bad lock persist until an operator toggled the estimator and
    # re-rolled it. These are the two step sizes that matter; both are in source pixels
    # of disparity and so mean the same thing at every standoff.
    SWEEP_FINE_STEP_PX = 0.5      # TRACK: comfortably inside a brick-facade basin
    SWEEP_COARSE_STEP_PX = 4.0    # ACQUIRE: coarse enough to cover metres of prior error
    SWEEP_TRACK_HALF_SPAN_PX = 3.0
    SWEEP_MAX_CANDIDATES = 41     # bounds one acquisition pass's GPU cost

    # The minimum must be a real minimum, not the lowest sample of a flat curve. Measured
    # as the fractional cost drop from the median usable candidate to the best one, so it
    # is scale-free (the cost is an across-view variance whose absolute size depends on
    # scene contrast). A flat window must mean "no measurement", not "argmin of noise" --
    # that distinction is the whole difference between tracking and random-walking.
    SWEEP_MIN_CONTRAST = 0.05
    # In TRACK the incumbent is always a candidate; require a new sample to beat it by
    # this fraction before moving, so a converged sweep sits still instead of dithering.
    SWEEP_IMPROVE_MARGIN = 0.02
    # Consecutive TRACK passes with no usable measurement before falling back to a wide
    # scan. At ~10-20 passes/s this is 1-2 s of no signal.
    SWEEP_LOST_PASSES = 20
    # Floor on how often a (much more expensive) acquisition scan may run.
    SWEEP_ACQUIRE_MIN_PERIOD_S = 1.0

    # Largest lag behind the freshest block before a view is dropped from the solve.
    # Pose/video skew is a first-order error term on real drones -- telemetry is ~5 Hz
    # and the video pipeline has its own latency -- and a re-served block (see
    # read_block_memory's cache) can repeat one frame's pixels indefinitely while the
    # formation moves. 0.25 s at the 40 deg/s yaw clamp is already ~45 px of seam.
    MAX_CAPTURE_SKEW_S = 0.25

    def _sweep_candidates(self, K, cams, config):
        """
        Candidate plane offsets for one scan, sampled uniformly in **disparity**.

        Returns ``(offsets_metres, mode, step_px)``, or ``(None, ...)`` if the formation
        is too degenerate to define a baseline.

        Why disparity and not metres: two views of a plane disagree by ``f*B/Z`` pixels
        for a depth error, so equal steps in ``f*B/Z`` are equal steps in the quantity the
        photometric cost actually measures.  Equal steps in metres are far too coarse at
        long range and pointlessly fine at short range -- and since the basin of
        attraction is itself a roughly fixed number of pixels wide, a metres-uniform scan
        can only be correctly sized at one standoff.  This is also why the clamp and the
        capture range are expressed here rather than as a metre-valued inspector field:
        a metre value tuned against the sim's raycast prior caps out exactly when a
        map-drawn plane needs it most.

        ACQUIRE covers the operator's stated uncertainty (``sweep_range``, metres, which
        remains the honest way to say "how wrong could my prior be") at a coarse step;
        TRACK covers a few pixels either side of the incumbent at a step comfortably
        inside a real facade's basin.
        """
        span_m = float(config.get("sweep_range", 0.0))
        if span_m <= 0.0:
            return None, self._sweep_mode, 0.0

        # Standoff and in-plane baseline straight off the current geometry. The baseline
        # that matters is the separation of the views *in the plane* -- that is the B in
        # f*B/Z -- and the median nearest-neighbour distance is the one that describes
        # the overlapping pairs rather than the formation's overall extent.
        Z = float(np.median([c["plane_h"] for c in cams]))
        ab = np.array([c["plane_ab"] for c in cams], dtype=np.float64)
        if len(ab) >= 2:
            d2 = ((ab[:, None, :] - ab[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(d2, np.inf)
            B = float(np.median(np.sqrt(d2.min(axis=1))))
        else:
            B = 0.0
        f = float(K[0, 0])

        fB = f * B
        if not np.isfinite(Z) or Z <= 1e-3 or fB <= 1e-6:
            # No usable baseline (one view, or a formation collapsed to a point). Fall
            # back to the old metres-uniform scan rather than failing: it is a poor
            # sampling but it is not wrong, and this configuration cannot be stitched
            # anyway.
            steps = max(3, int(config.get("sweep_steps", 9)) | 1)
            return (np.linspace(self._plane_offset - span_m,
                                self._plane_offset + span_m, steps),
                    self._sweep_mode, 0.0)

        d0 = fB / Z                                  # incumbent disparity, px

        if self._sweep_mode == "track":
            step_px = self.SWEEP_FINE_STEP_PX
            half_px = self.SWEEP_TRACK_HALF_SPAN_PX
            n = int(round(2.0 * half_px / step_px)) + 1
            dd = np.linspace(-half_px, half_px, n)
        else:
            # Convert the operator's metre range into the disparity interval it spans.
            # Asymmetric by construction -- a metre nearer costs more disparity than a
            # metre further -- which is precisely the asymmetry a metres-uniform scan
            # gets wrong.
            z_near = max(0.25 * Z, Z - span_m)
            z_far = Z + span_m
            d_hi, d_lo = fB / z_near, fB / z_far
            step_px = self.SWEEP_COARSE_STEP_PX
            n = int(np.ceil((d_hi - d_lo) / step_px)) + 1
            n = int(np.clip(n, max(3, int(config.get("sweep_steps", 9))),
                            self.SWEEP_MAX_CANDIDATES))
            dd = np.linspace(d_lo - d0, d_hi - d0, n)

        # Force the incumbent to be a candidate: a converged sweep that cannot sample its
        # own current answer dithers between the two samples straddling it.
        dd = np.unique(np.concatenate([dd, [0.0]]))

        z_cand = fB / np.clip(d0 + dd, 1e-6, None)
        return self._plane_offset + (z_cand - Z), self._sweep_mode, step_px

    def _sweep_plane(self, views, K, plane, config):
        """
        One global scalar: the additive plane-distance offset that best aligns the views.

        Two-mode search.  ACQUIRE scans wide and coarse to find which basin the truth is
        in; TRACK scans narrow and fine inside it and refuses to move without evidence.
        A single scan cannot do both, and the version that tried was the primary bug:
        with a step wider than the basin, the minimum was never resolved, so the argmin
        was noise and the low-pass then walked the plane a metre per pass in an arbitrary
        direction.  What looked like a latch was a dice roll that an operator toggle
        re-rolled.

        Brute force rather than gradient descent still: the cost over a bounded interval
        has one broad minimum, a scan cannot diverge, and its cost is fixed and
        predictable on a thread sharing a GPU with the render loop.
        """
        if float(config.get("sweep_range", 0.0)) <= 0.0:
            self._sweep_stats = {"skipped": "range not set"}
            return

        # apply_dpose=False for the same reason _photometric_cost uses it: the baseline
        # and standoff that size the scan must describe the published formation, not the
        # refiner's adjusted one.
        frame, cams = self._build_geometry(views, K, plane, config,
                                           plane_offset=self._plane_offset,
                                           apply_dpose=False)
        if frame is None or len(cams) < 2:
            self._sweep_stats = {"skipped": "fewer than 2 usable views"}
            return

        now = time.monotonic()
        if (self._sweep_mode == "acquire"
                and now - self._last_acquire < self.SWEEP_ACQUIRE_MIN_PERIOD_S):
            # A wide scan costs several times a tracking one; rate-limit it so a scene the
            # sweep cannot lock onto does not permanently occupy the GPU it shares.
            self._sweep_stats = {"held": "waiting to re-acquire", "mode": "acquire",
                                 "offset": self._plane_offset, "lost": self._sweep_lost}
            return
        offsets, mode, step_px = self._sweep_candidates(K, cams, config)
        if offsets is None or len(offsets) < 3:
            self._sweep_stats = {"skipped": "no usable baseline for a scan"}
            return
        if mode == "acquire":
            self._last_acquire = now

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
        if ref_area <= 0.0 or int(usable.sum()) < 3:
            self._note_no_measurement(mode, "no candidate had usable overlap")
            return

        masked = np.where(usable, costs, np.inf)
        i = int(np.argmin(masked))
        c_best = float(masked[i])
        interior = 0 < i < len(offsets) - 1

        # Is this a minimum, or the lowest sample of a flat curve? Fractional drop from
        # the median usable candidate, so the test is free of the scene's absolute
        # contrast. This is the gate that stops a textureless wall from being "measured".
        med = float(np.median(costs[usable]))
        contrast = (med - c_best) / max(c_best, 1e-12)
        if not np.isfinite(contrast) or contrast < self.SWEEP_MIN_CONTRAST:
            self._note_no_measurement(
                mode, f"flat cost curve (contrast {contrast:.3f} "
                      f"< {self.SWEEP_MIN_CONTRAST})")
            return

        best = float(offsets[i])

        # Sub-step refinement, but only from an interior sample flanked by two usable
        # ones -- a parabola through an edge sample extrapolates outside the scanned
        # interval, which is precisely where nothing was measured.
        if interior and usable[i - 1] and usable[i + 1]:
            c0, c1, c2 = masked[i - 1], masked[i], masked[i + 1]
            denom = c0 - 2.0 * c1 + c2
            if denom > 1e-12:
                step = float(offsets[i + 1] - offsets[i - 1]) * 0.5
                best += 0.5 * step * float(c0 - c2) / denom

        if mode == "acquire":
            # Snap. This IS the initial lock, not a refinement of one: low-passing an
            # acquisition would leave the estimate outside the basin it just found, where
            # the fine scan has no signal, and the two would fight indefinitely.
            self._plane_offset = best
            self._sweep_mode = "track"
            self._sweep_lost = 0
        else:
            # The incumbent is always a candidate; only move for a measurable improvement
            # on it, so a converged sweep sits still rather than dithering.
            c_inc = float(masked[int(np.argmin(np.abs(offsets - self._plane_offset)))])
            if np.isfinite(c_inc) and c_best > c_inc * (1.0 - self.SWEEP_IMPROVE_MARGIN):
                self._note_no_measurement(mode, "no improvement on the incumbent",
                                          contrast=contrast, keep_lock=True)
                return
            rate = min(1.0, max(0.0, float(config.get("refine_rate", 0.25))))
            self._plane_offset = (1.0 - rate) * self._plane_offset + rate * best
            # An argmin pinned to the edge means the truth is outside the fine window --
            # the scan still steps toward it, but a run of them means the lock is gone.
            self._sweep_lost = 0 if interior else self._sweep_lost + 1
            if self._sweep_lost >= self.SWEEP_LOST_PASSES:
                self._sweep_mode = "acquire"
                self._sweep_lost = 0

        self._sweep_stats = {
            "offset": self._plane_offset,
            "raw": best,
            "mode": mode,
            "cost": c_best,
            "contrast": contrast,
            "step_px": step_px,
            "interior": interior,
            "usable": int(usable.sum()),
            "steps": len(offsets),
        }

    def _note_no_measurement(self, mode, why, contrast=None, keep_lock=False):
        """
        Record a scan that produced no usable answer, and count it toward losing lock.

        Separate from a hard skip because the distinction matters operationally: the
        sweep having nothing to measure on a blank wall is normal and the right response
        is to hold the current estimate, whereas a long run of it means the estimate is
        no longer in the basin and only a wide scan can recover.
        """
        if mode == "track" and not keep_lock:
            self._sweep_lost += 1
            if self._sweep_lost >= self.SWEEP_LOST_PASSES:
                self._sweep_mode = "acquire"
                self._sweep_lost = 0
        self._sweep_stats = {
            "held": why,
            "mode": mode,
            "offset": self._plane_offset,
            "lost": self._sweep_lost,
        }
        if contrast is not None:
            self._sweep_stats["contrast"] = contrast

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

        # Start from the corrections already earned, keeping every view still in the
        # selection. A view whose measurement was gated out this pass MUST keep its
        # accumulated correction: rebuilding the set from this pass's accepted views only
        # meant a single weak correlation peak deleted that drone's whole history, snapped
        # its patch back to the raw pose, and left it to re-converge from scratch -- which
        # flaps frame to frame. Note the old behaviour was backwards as well as wrong:
        # when ALL views were rejected the early return preserved everything, and only a
        # PARTIAL rejection destroyed history.
        #
        # Genuine absence from the selection is the one thing that does retire a
        # correction, or a drone that returns is warped by an offset measured minutes ago
        # against a formation that has since moved.
        present = {v["drone_id"] for v in views}
        shifts = {did: dC for did, dC in self._pose_shift.items() if did in present}

        if not raw:
            self._pose_shift = shifts
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
        for did, dC in raw.items():
            prev = shifts.get(did)
            total = (dC - mean) * rate
            if prev is not None:
                total = prev + total
            shifts[did] = total

        # Re-gauge the ACCUMULATED set, not merely this pass's residuals. Zero-meaning the
        # residuals alone leaves the accumulated set free to drift off zero mean as
        # membership changes -- each pass is individually gauged against a different
        # subset -- and a net translation slides the whole mosaic across the canvas.
        if shifts:
            acc_mean = sum(shifts.values()) / len(shifts)
            shifts = {did: v - acc_mean for did, v in shifts.items()}

        # Clamp last, so it bounds the value actually applied rather than a pre-gauge
        # intermediate.
        if max_shift_m > 0.0:
            for did, v in shifts.items():
                mag = float(np.linalg.norm(v))
                if mag > max_shift_m:
                    shifts[did] = v * (max_shift_m / mag)

        self._pose_shift = shifts
        self._refine_stats = {
            "accepted": len(raw),
            "rejected": rejected,
            # Views carrying a correction they did not re-measure this pass. A steady
            # non-zero count is the signature of a scene the refiner can only partly
            # measure -- useful, and invisible if "accepted" is reported as len(shifts).
            "held": len(shifts) - len(raw),
            "worst_shift": max(float(np.linalg.norm(v)) for v in shifts.values()),
            "worst_residual": max(float(np.linalg.norm(v - mean)) for v in raw.values()),
        }

    # ------------------------------------------------------------------ canvas framing

    # Fraction of spare canvas left around the fitted footprint, so the mosaic does not
    # sit hard against the canvas edge and a drone drifting outward does not immediately
    # force a rescale.
    AUTOFIT_MARGIN = 1.12

    # A view whose footprint area exceeds this multiple of the median is left out of the
    # fit. The pathology is a near-grazing view, whose footprint runs away toward the
    # horizon and would pull the scale out by an order of magnitude on its own. The
    # anisotropy gate catches the same views, but it runs on H -- which needs the scale
    # this function is computing -- so the fit cannot lean on it.
    AUTOFIT_AREA_OUTLIER = 6.0

    # Scale quantisation, in steps per octave. The fitted scale is snapped to one of these
    # steps about the operator's planarMetresPerPixel, because a continuously-fitted
    # canvas visibly breathes: every frame's footprint differs slightly, and rescaling on
    # each is both distracting to look at and a moving target for the estimators.
    AUTOFIT_STEPS_PER_OCTAVE = 3.0
    # Extra dead-band either side of the incumbent step, in steps. Without it the scale
    # dithers between two steps whenever the formation sits near a boundary.
    AUTOFIT_HYSTERESIS_STEPS = 0.25
    # Floor on how often the scale may step, seconds. Hysteresis stops dithering about a
    # boundary; this stops a genuinely growing formation from ratcheting every frame.
    AUTOFIT_DWELL_S = 1.5
    # Hard bound on how far the fit may depart from the operator's value, in octaves.
    AUTOFIT_MAX_OCTAVES = 3.0
    # Low-pass rate for the canvas centre. Snapped rather than damped on the first fit --
    # the same argument as the sweep's ACQUIRE snap, since there is no incumbent to damp
    # toward and starting at the origin would make every run open with a slow slide.
    AUTOFIT_CENTRE_RATE = 0.15

    def _view_footprint(self, cam, max_range):
        """
        Where a view's four image corners land on the plane, in plane coordinates.

        Recovered by inverting ``G`` rather than by re-casting rays from the pose, which
        means it automatically reflects whatever corrections ``_build_geometry`` already
        baked in and needs no second copy of that logic.  ``G`` maps ``(a, b, 1)`` to a
        homogeneous pixel whose third component is depth in metres, so for a corner
        ``u``, ``p = G^-1 u`` satisfies ``p = (a, b, 1) / depth`` -- i.e. ``p[2]`` is the
        reciprocal depth, and every rejection the ray cast would make (parallel to the
        plane, behind the camera, past ``max_range``) is a test on that one number.

        Returns an ``(n, 2)`` array of corners that hit the plane, or ``None``.
        """
        h, w = cam["view"]["image"].shape[:2]
        corners = np.array([
            [0.0, 0.0, 1.0], [w - 1.0, 0.0, 1.0],
            [w - 1.0, h - 1.0, 1.0], [0.0, h - 1.0, 1.0],
        ], dtype=np.float64).T
        try:
            p = np.linalg.solve(np.asarray(cam["G"], dtype=np.float64), corners)
        except np.linalg.LinAlgError:
            return None

        min_recip = 1.0 / max_range if np.isfinite(max_range) and max_range > 0.0 else 0.0
        ok = np.isfinite(p).all(axis=0) & (p[2] > max(min_recip, 1e-9))
        if not ok.any():
            return None
        return (p[:2, ok] / p[2, ok]).T

    def _footprint_bbox(self, cams, config):
        """
        Axis-aligned bounds of the views' footprints on the plane, ``(lo, hi)`` in plane
        coordinates, or ``None`` when too little of the formation is looking at it.

        Views with a wildly outsized footprint are dropped first -- see
        ``AUTOFIT_AREA_OUTLIER``.
        """
        max_range = float(config.get("max_range", 0.0)) or np.inf
        spans = []
        for cam in cams:
            fp = self._view_footprint(cam, max_range)
            if fp is None or len(fp) < 3:
                continue
            lo, hi = fp.min(axis=0), fp.max(axis=0)
            spans.append((lo, hi, float((hi[0] - lo[0]) * (hi[1] - lo[1]))))

        if len(spans) < 2:
            return None

        areas = np.array([s[2] for s in spans])
        median = float(np.median(areas))
        if median > 0.0:
            keep = [s for s in spans if s[2] <= median * self.AUTOFIT_AREA_OUTLIER]
            # Never let the outlier rule empty the fit: if it would, the spread is not an
            # outlier, it is the formation, and the bbox over everything is the honest
            # answer.
            if len(keep) >= 2:
                spans = keep

        lo = np.min(np.stack([s[0] for s in spans]), axis=0)
        hi = np.max(np.stack([s[1] for s in spans]), axis=0)
        return lo, hi

    def _fit_canvas(self, frame, cams, config, bbox):
        """
        Settle the auto-fit scale and centre from a footprint bounding box.

        Render thread only -- it mutates the ``_fit_*`` incumbents.  Returns
        ``(mpp, centre_ab)``, or ``None`` when there is nothing to fit and the caller
        should hold whatever it had.
        """
        if bbox is None:
            return None
        canvas_w, canvas_h = config["canvas"]
        anchor = float(config.get("metres_per_pixel", 0.0))
        if anchor <= 0.0 or canvas_w <= 0 or canvas_h <= 0:
            return None

        lo, hi = bbox
        extent = np.maximum(hi - lo, 1e-6)
        raw = max(extent[0] / canvas_w, extent[1] / canvas_h) * self.AUTOFIT_MARGIN
        if not np.isfinite(raw) or raw <= 0.0:
            return None

        # Quantise in log space: one step is a fixed *ratio*, which is what "a scale step"
        # means perceptually, and it makes the hysteresis band symmetric about the step.
        limit = self.AUTOFIT_MAX_OCTAVES * self.AUTOFIT_STEPS_PER_OCTAVE
        q = float(np.clip(np.log2(raw / anchor) * self.AUTOFIT_STEPS_PER_OCTAVE,
                          -limit, limit))
        now = time.monotonic()
        if self._fit_mpp is None:
            self._fit_step = int(round(q))
            self._fit_changed_at = now
        elif (abs(q - self._fit_step) > 0.5 + self.AUTOFIT_HYSTERESIS_STEPS
                and now - self._fit_changed_at >= self.AUTOFIT_DWELL_S):
            self._fit_step = int(round(q))
            self._fit_changed_at = now
        mpp = anchor * (2.0 ** (self._fit_step / self.AUTOFIT_STEPS_PER_OCTAVE))

        # Centre, low-passed in world coordinates (see __init__ for why not in (a, b)).
        centre_ab = 0.5 * (lo + hi)
        measured = frame.O + centre_ab[0] * frame.e1 + centre_ab[1] * frame.e2
        if self._fit_centre_world is None:
            self._fit_centre_world = measured
        else:
            self._fit_centre_world = (self._fit_centre_world
                                      + self.AUTOFIT_CENTRE_RATE
                                      * (measured - self._fit_centre_world))

        self._fit_mpp = mpp
        self._fit_extent = (float(extent[0]), float(extent[1]))
        return mpp, self._centre_in_frame(frame)

    def _centre_in_frame(self, frame):
        """The low-passed canvas centre expressed in ``frame``'s plane coordinates."""
        if self._fit_centre_world is None:
            return 0.0, 0.0
        rel = self._fit_centre_world - frame.O
        return float(rel @ frame.e1), float(rel @ frame.e2)

    def _canvas_matrix(self, mpp, canvas_w, canvas_h, centre_ab):
        """
        ``M`` for a canvas of ``mpp`` centred on plane point ``centre_ab``.

        ``canvas_to_plane_matrix`` takes the plane origin's pixel coordinates, so placing
        a chosen plane point at the canvas centre is a shift of those: solving
        ``M (W/2, H/2, 1) == (ca, cb, 1)`` on its two rows gives the pair below.  Passing
        ``(0, 0)`` reproduces the plain centred canvas exactly.
        """
        ca, cb = centre_ab
        return pg.canvas_to_plane_matrix(mpp,
                                         canvas_w * 0.5 - ca / mpp,
                                         canvas_h * 0.5 + cb / mpp)

    def _view_transform(self, config, mpp_fit, bbox, centre_ab):
        """
        Apply the operator's zoom and pan to a rest framing.

        Returns ``(mpp_view, centre_ab)``.  Pan arrives as a fraction of the canvas, not
        as metres, so that dragging half a screen is half a screen at every zoom level;
        it is converted here, where the canvas extent is known.  A non-zero pan is then
        clamped to the footprint bounding box, because panning until the canvas holds
        nothing but blank plane is never what the operator meant -- and this is the side
        that knows where the imagery actually is.

        The clamp applies to the pan only, never to the rest framing: an unpanned canvas
        must come out exactly where its mode put it, even on the odd frame where the
        bounding box does not contain it (the reference view can be dropped by the range
        check while the others survive).  Otherwise ``pan = 0`` would silently relocate
        the FIXED canvas, which is the one thing it must never do.
        """
        canvas_w, canvas_h = config["canvas"]
        zoom = float(config.get("zoom", 1.0) or 1.0)
        if not np.isfinite(zoom) or zoom <= 0.0:
            zoom = 1.0
        mpp_view = mpp_fit / zoom

        pan = config.get("pan", (0.0, 0.0)) or (0.0, 0.0)
        pan_a, pan_b = float(pan[0]), float(pan[1])
        if not (np.isfinite(pan_a) and np.isfinite(pan_b)):
            pan_a = pan_b = 0.0
        if pan_a == 0.0 and pan_b == 0.0:
            return mpp_view, centre_ab

        ca = centre_ab[0] + pan_a * canvas_w * mpp_view
        cb = centre_ab[1] + pan_b * canvas_h * mpp_view
        if bbox is not None:
            lo, hi = bbox
            ca = float(np.clip(ca, min(lo[0], centre_ab[0]), max(hi[0], centre_ab[0])))
            cb = float(np.clip(cb, min(lo[1], centre_ab[1]), max(hi[1], centre_ab[1])))
        return mpp_view, (ca, cb)

    # ------------------------------------------------------------------ estimator internals

    def _estimator_geometry(self, views, K, plane, config, plane_offset=None,
                            apply_dpose=True):
        """
        Geometry for one estimator pass, on a canvas reduced by ``ESTIMATOR_SCALE``.

        The canvas covers the same *plane extent* as the render canvas but with fewer
        pixels, so metres-per-pixel grows by the inverse of the scale -- shrinking the
        pixel count without shrinking the field of view, which is what keeps the two
        estimators looking at the same overlap the render path does.

        **The operator's zoom and pan are deliberately not applied here.**  They are a
        viewing transform, and letting them through would resize and slide the canvas the
        sweep is scanning on: its minimum is a basin a few source-disparity pixels wide,
        and moving the measurement window mid-convergence is precisely the "sweep loses
        its lock" failure the ACQUIRE/TRACK split exists to prevent.  Pilots zoom in to
        look at things, which must not cost them their alignment.

        The auto-fit scale *is* followed, because it changes what plane region the
        overlap covers -- but it is only read here.  ``_fit_canvas`` runs on the render
        thread, and the two estimators run on the warp thread; one writer, two readers.
        Following it at a frame's lag is harmless, and the corrections themselves are
        stored in metres (``_pose_shift``) and scanned in source-disparity pixels
        (``SWEEP_*_STEP_PX``), so both are invariant to the canvas scale and a step in the
        fit needs nothing reset.
        """
        canvas_w, canvas_h = config["canvas"]
        mpp = float(config.get("metres_per_pixel", 0.0))
        if canvas_w <= 0 or canvas_h <= 0 or mpp <= 0.0:
            return None, [], None, 0, 0, 0.0
        if (int(config.get("canvas_mode", CANVAS_MODE_FIXED)) == CANVAS_MODE_AUTOFIT
                and self._fit_mpp):
            mpp = float(self._fit_mpp)

        frame, cams = self._build_geometry(views, K, plane, config,
                                           plane_offset=plane_offset,
                                           apply_dpose=apply_dpose)
        if frame is None:
            return None, [], None, 0, 0, 0.0

        cw = max(16, int(canvas_w * self.ESTIMATOR_SCALE))
        ch = max(16, int(canvas_h * self.ESTIMATOR_SCALE))
        mpp_s = mpp * (canvas_w / float(cw))
        centre_ab = ((0.0, 0.0)
                     if int(config.get("canvas_mode", CANVAS_MODE_FIXED)) == CANVAS_MODE_FIXED
                     else self._centre_in_frame(frame))
        M_s = self._canvas_matrix(mpp_s, cw, ch, centre_ab)
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

    def _drop_stale(self, posed):
        """
        Drop views whose frame is too far behind the freshest one.

        ``read_block_memory`` re-serves the previous frame for a block that is busy being
        written, and will do so indefinitely for a block that never becomes ready again --
        the pixels of one moment then get warped by a homography built for the formation's
        position now.  Both halves of the evidence are already on the wire and were
        previously decoded and discarded: ``capture_time`` (the pose's own timestamp) and
        ``cached``.

        Degrades safely: a producer that publishes no capture time leaves every view at
        the same value, so nothing is dropped.
        """
        times = [float(v.get("capture_time", 0.0) or 0.0) for v in posed]
        if not times:
            return posed, 0
        newest = max(times)
        if newest <= 0.0:
            return posed, 0
        fresh = [v for v, t in zip(posed, times)
                 if newest - t <= self.MAX_CAPTURE_SKEW_S]
        # Never let this empty the solve: if every view is old they are old *together*,
        # which is a stalled producer rather than a skew problem, and the panorama
        # freezing is a better failure than it vanishing.
        if len(fresh) < 2:
            return posed, 0
        return fresh, len(posed) - len(fresh)

    def _build_geometry(self, views, K, plane, config, plane_offset=None,
                        apply_dpose=True, record=False):
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
        unposed = len(views) - len(posed)
        posed, stale = self._drop_stale(posed)
        if record:
            # Written only by the render path. _build_geometry is called from both
            # threads, so an unconditional write here races the warp thread and the
            # count Unity is shown belongs to whichever call happened last.
            self._unposed, self._stale = unposed, stale
        if not posed:
            return None, []

        if int(plane.get("plane_mode", -1)) == PLANE_MODE_FORMATION_RELATIVE:
            n, d = self._plane_from_formation(
                posed, float(config.get("standoff", 0.0)))
            if n is None:
                return None, []
        else:
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
        # The reference gets the same correction every other view gets. Leaving it on the
        # raw pose builds the canvas frame around a camera that the render then moves,
        # so any net translation in the refiner's output slides the mosaic across a
        # canvas that does not follow it -- a whole-image drift on top of the per-view
        # alignment the refiner was asked for.
        if apply_dpose:
            R_ref, C_ref = self._apply_correction(ref, R_ref, C_ref)

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

    # Below this ratio of smallest to middle singular value, the camera positions are
    # treated as spanning a plane. A wall of drones with half a metre of GNSS scatter
    # across 20 m reads ~0.025; a single row of drones reads ~1 and is rejected.
    FORMATION_PLANARITY_MAX = 0.2

    def _plane_from_formation(self, posed, standoff):
        """
        Derive ``(n, d)`` from the camera poses alone.  Returns ``(None, 0.0)`` on failure.

        This is the only plane source that works on real drones: ``UpdateScenePlane``
        raycasts Unity colliders, and a real facade has none.  All the operator supplies is
        ``standoff`` -- the perpendicular distance from the formation to the surface -- so
        no georeferenced origin has to be agreed between Unity and the drone telemetry.

        The normal comes from a plane fitted to the camera *positions* where the formation
        actually spans a plane, and from the mean camera *forward* where it does not:

        - Fitting the positions is immune to gimbal pitch, which matters because a facade
          wall flown with the gimbal 20 deg down would otherwise yield a plane tilted 20
          deg off vertical.  It also handles nadir for free: drones spread over a
          horizontal plane fit a horizontal plane, whose normal is up.
        - It degenerates for a single row of drones (any normal perpendicular to the row
          fits equally well), which is exactly when the mean forward is reliable instead.

        The mean forward is used either way to orient the result toward the cameras, so the
        plane's "front" is unambiguous downstream -- the same convention
        ``UpdateScenePlane`` applies to its raycast hit normal.

        Uses the *published* poses, not the refiner-corrected ones: the corrections are
        zero-meaned, so they cannot move the centroid, and leaving them out keeps the plane
        from moving in lockstep with the estimator that is being measured against it.
        """
        if not posed or standoff <= 0.0:
            return None, 0.0

        centres, forwards = [], []
        for v in posed:
            R, C = pg.unity_pose_to_cv(v["pos"], v["quat"])
            centres.append(C)
            forwards.append(R[2])          # rows of a world->camera rotation are the axes
        centres = np.asarray(centres, dtype=np.float64)

        f_mean = np.asarray(forwards, dtype=np.float64).mean(axis=0)
        f_norm = np.linalg.norm(f_mean)
        if f_norm < 1e-6:
            # Cameras pointing in opposing directions average to nothing; there is no
            # single surface they are all looking at, so refuse rather than invent one.
            return None, 0.0
        f_mean = f_mean / f_norm

        centroid = centres.mean(axis=0)
        n = None
        if len(centres) >= 3:
            sv = np.linalg.svd(centres - centroid, full_matrices=False)
            s, vt = sv[1], sv[2]
            if s[1] > 1e-6 and (s[2] / s[1]) < self.FORMATION_PLANARITY_MAX:
                n = vt[2]

        source = "positions"
        if n is None:
            n = -f_mean
            source = "forward"

        # Orient toward the cameras: the surface is in front of them, so the normal must
        # oppose the direction they are looking.
        if float(np.dot(n, f_mean)) > 0.0:
            n = -n
        n = n / np.linalg.norm(n)

        # Standoff is the PERPENDICULAR distance from the formation to the surface, which
        # is what an operator means by "the facade is 30 m in front of the wall". Stepping
        # along -n rather than along f_mean is what makes that true when the two differ.
        d = float(np.dot(n, centroid)) - float(standoff)
        self._plane_source = source
        return n, d

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

    def _maybe_log_estimators(self, period=5.0):
        """
        Estimator state, printed from the WARP thread.

        Deliberately not folded into ``_maybe_log``: that runs at the end of
        ``planar_pano``, after every early return, so it falls silent in precisely the
        situations worth diagnosing -- a frozen snapshot or a plane the render path is
        rejecting produce no line at all rather than a line saying so.

        What is printed is chosen to identify the failure mode without a second
        photometric evaluation: the search mode, the fractional contrast at the minimum,
        the sampling step in disparity pixels, and whether the argmin was interior or
        pinned to an edge.  Those are properties of the cost curve's SHAPE, which means
        the same thing in the sim and in the field -- unlike the cost at zero offset,
        which on a real drone measures how good the operator's typed prior was rather
        than whether the sweep is locked.
        """
        now = time.time()
        if now - self._last_est_log < period:
            return
        self._last_est_log = now

        sweep, refine = self._sweep_stats, self._refine_stats
        if sweep:
            if "skipped" in sweep:
                print(f"[PLANAR] plane sweep idle: {sweep['skipped']}")
            elif "held" in sweep:
                print(f"[PLANAR] plane sweep holding {sweep['offset']:+.2f} m "
                      f"[{sweep['mode']}]: {sweep['held']} "
                      f"({sweep.get('lost', 0)}/{self.SWEEP_LOST_PASSES} to re-acquire)")
            else:
                print(f"[PLANAR] plane sweep [{sweep['mode']}]: {sweep['offset']:+.2f} m "
                      f"(raw {sweep['raw']:+.2f}) | {sweep['usable']}/{sweep['steps']} "
                      f"candidates @ {sweep['step_px']:.1f} px | "
                      f"contrast {sweep['contrast']:.3f} | "
                      f"argmin {'interior' if sweep['interior'] else 'EDGE'}")
        if refine:
            if "skipped" in refine:
                print(f"[PLANAR] pose refine idle: {refine['skipped']}")
            else:
                # Residual is the convergence read-out: it should fall toward zero as
                # the accumulated correction absorbs the error. A residual that stays
                # high while the correction grows means the two are fighting.
                print(f"[PLANAR] pose refine: {refine['accepted']} accepted, "
                      f"{refine['rejected']} rejected (weak peak), "
                      f"{refine.get('held', 0)} held | "
                      f"worst correction {refine['worst_shift']:.2f} m | "
                      f"residual {refine['worst_residual']:.3f} m")

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
        stale = s.get("stale", 0)
        unposed_txt = f", {unposed} unposed" if unposed else ""
        unposed_txt += f", {stale} stale" if stale else ""
        print(f"[PLANAR] {s.get('views', 0)} views "
              f"(+{s.get('dropped', 0)} dropped{unposed_txt}) | blend {s.get('blend', '?')} | "
              f"coverage {s.get('coverage', 0):.0%} | "
              f"mean range {s.get('mean_range', 0):.1f} m | "
              f"max anisotropy {s.get('max_aniso', 0):.2f} | overlap PSNR {psnr_txt}"
              # Only in FormationRelative, where the plane is derived here rather than
              # published. Which rule won matters: "forward" means the formation collapsed
              # to a row and the normal now follows the gimbal.
              + (f" | plane from {self._plane_source}" if self._plane_source else ""))

        # Framing, and where zoom stops buying detail. Printed as the canvas extent in
        # metres rather than as metres-per-pixel because the extent is the thing an
        # operator can check against the scene in front of them. "empty" flags a canvas
        # sampled finer than the sharpest view's ground sample distance: past that point
        # the mosaic is interpolating source pixels, not resolving new detail, which no
        # other number on this line would reveal.
        mpp_view = s.get("mpp_view", 0.0)
        if mpp_view > 0.0:
            gsd = s.get("best_gsd", 0.0)
            zoom = s.get("zoom", 1.0)
            ext_a, ext_b = s.get("canvas_extent", (0.0, 0.0))
            print(f"[PLANAR] canvas [{s.get('canvas_mode', '?')}] "
                  f"{ext_a:.1f} x {ext_b:.1f} m "
                  f"@ {mpp_view * 100.0:.1f} cm/px"
                  + (f" | zoom {zoom:.2f}x" if abs(zoom - 1.0) > 1e-3 else "")
                  + (f" | source GSD {gsd * 100.0:.1f} cm/px"
                     f"{' (EMPTY magnification)' if mpp_view < gsd else ''}"
                     if gsd > 0.0 else ""))

        # A colour map is useless without the key, and the selection changes as drones
        # join, die or fall out of range -- so reprint it alongside the stats rather
        # than once at startup.
        if s.get("debug_view", DEBUG_OFF) != DEBUG_OFF:
            legend = "  ".join(f"drone {i} = {debug_colour(i)[0]}"
                               for i in s.get("drone_ids", []))
            mode = "flat" if s["debug_view"] == DEBUG_FLAT else "tint"
            print(f"[PLANAR] debug view ({mode}):  {legend}")
