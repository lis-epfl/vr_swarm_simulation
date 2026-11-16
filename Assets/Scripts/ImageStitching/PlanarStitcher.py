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

By default the two switches drive :mod:`planar_align` (``_align``): learned matches
between the views, one consistent alignment chosen across the whole formation, and a
bundle adjustment through the plane that solves a rotation and a position per drone and
the plane itself.  Its answer is pinned to the frames it was solved on (``_make_anchors``)
and carried to later frames by each view's own image motion (:mod:`planar_track`), so
what is drawn between solves does not depend on when a drone's telemetry arrived.
``PLANAR_LEGACY_ESTIMATORS=1`` selects the two photometric estimators instead, which are
what the rest of this docstring describes.

The two photometric estimators are complementary rather than alternative, because they
move different parameters:

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

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from BaseStitcher import BaseStitcher
import planar_geometry as pg
import planar_pose_filter
import planar_track
from run_clock import stamp
from wire_panorama import WireReadyPanorama

try:
    import planar_align
except Exception as _align_exc:                      # pragma: no cover
    planar_align = None
    print(f"[PLANAR] learned alignment unavailable ({_align_exc}); "
          f"using the photometric estimators")

# PLANAR_LEGACY_ESTIMATORS=1 restores the photometric plane sweep and the phase-correlation
# refiner behind the same two inspector switches, for comparison. By default those
# switches drive planar_align: "pose refine" solves a rotation and a position correction
# per drone from learned matches, and "plane sweep" solves the plane (orientation as well
# as distance) in the same bundle adjustment.
LEGACY_ESTIMATORS = os.environ.get("PLANAR_LEGACY_ESTIMATORS", "0") not in ("", "0")

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
#
# MULTIBAND keeps NEAREST's partition and its guarantee -- fine detail always comes from
# exactly one view, so nothing can ghost -- but cross-fades the *low* frequencies across
# the seam, after matching each view's exposure to its neighbours'. What is left of a
# seam under NEAREST once the geometry is right is mostly photometric: every aircraft
# runs its own auto-exposure and white balance, and a hard partition draws that
# difference as an edge. See _multiband.
BLEND_FEATHER = 0
BLEND_NEAREST = 1
BLEND_MULTIBAND = 2

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

# How a view's pixels are read at the place an output pixel falls in it. Bicubic by
# default, because it is what makes a full-resolution mosaic worth drawing: resampled back
# into a drone's own frame, a mosaic drawn at twice the feeds' pixel density reproduces
# the feed to 59 dB with 99% of its contrast in every frequency band, where bilinear
# leaves 45 dB and 80% at the finest detail the feed holds (measured on the recorded
# clips; 32-35 dB and 30-40% for the canvas-then-resize path this replaced). With one
# sample per output pixel the kernel's sixteen taps cost about a millisecond a frame.
# PLANAR_SAMPLING=bilinear restores the old kernel, and a config may carry "sampling".
SAMPLING = os.environ.get("PLANAR_SAMPLING", "bicubic").lower()

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

    # Resampling of held telemetry (planar_pose_filter), created on the first frame.
    # PLANAR_POSE_FILTER=0 switches it off; it is a no-op on poses that change every
    # frame, which is every simulated one.
    _pose_filter = None
    POSE_FILTER = os.environ.get("PLANAR_POSE_FILTER", "1") not in ("", "0")

    # Image-based propagation of an alignment between solves (planar_track), created with
    # the aligner. PLANAR_VISUAL_TRACK=0 goes back to placing every frame by its published
    # pose plus the correction.
    _tracker = None
    VISUAL_TRACK = os.environ.get("PLANAR_VISUAL_TRACK", "1") not in ("", "0")
    # An anchor older than this is no longer carried forward: the aligner has stopped
    # answering, and a product of fifty inter-frame motions is not an alignment.
    MAX_ANCHOR_AGE_S = 2.5
    # A new anchor further than this (source pixels) from the one on screen replaces it
    # outright instead of being low-passed into it.
    ANCHOR_SNAP_PX = 10.0

    # Learned alignment (planar_align). Class-level defaults so an instance built without
    # __init__ -- the self-test's bare stitcher -- still has them.
    _aligner = None
    _align_stats = {}
    _align_warned = False
    # Floor on the aligner's period. One pass is 0.1-0.5 s of GPU shared with the render
    # loop, and the corrections it estimates drift over tens of seconds.
    ALIGN_MIN_PERIOD_S = 0.2
    # Winner-take-all score multiplier for a view the aligner found no support for.
    WEAK_VIEW_PRIORITY = 0.2
    # A new alignment is eased in over this long rather than switched to. Each update
    # moves the views against each other by a pixel or so (a different set of matches,
    # a facade that is not quite a plane), and applied as a step that reads as the seams
    # ticking at the update rate.
    ALIGN_BLEND_S = 0.3
    # The clock the blend runs on. An instance attribute in the stream bench, which plays
    # clips on simulated time.
    _clock = staticmethod(time.monotonic)

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
        #   "plane_fix" : the aligner's plane, replacing or riding on the published one --
        #             ("world", n, d) or ("delta", rotvec, offset); see _apply_plane_fix
        #   "rotvec" / "blend" : the aligner's rotations as vectors, and the generation
        #             this one is eased in from; see _live_correction
        self._correction = {"dpose": None, "plane": None, "plane_fix": None}

        # Cached pixel grid of the estimators' canvas, keyed on its size: the only thing
        # that changes between passes is the homography, so this is built once. It is
        # the warp thread's alone -- the render path keeps its own lattice of the output
        # raster (_lattice), a different size written from a different thread, and one
        # cache for both would miss on every call and race while doing it.
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
        self._last_geometry = None
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

    def planar_pano(self, views, intrinsics, plane, config, out_size=None):
        """
        Build the mosaic from pose alone.

        ``views``      list of block records (image + Unity pose) from read_block_memory
        ``intrinsics`` (fx, fy, cx, cy) for the block resolution
        ``plane``      dict from read_dynamic_state (Unity-world normal + offset)
        ``config``     planar_* metadata fields
        ``out_size``   ``(height, width)`` to draw the mosaic at, or ``None``

        ``config["canvas"]`` frames the mosaic; ``out_size`` is how many pixels it is
        drawn with (see the note above ``_render``).  With ``out_size`` the panorama is a
        :class:`WireReadyPanorama` -- RGB, bottom-up rows, in a buffer that is reused a
        few frames later -- which ``first_thread`` writes to Unity as it is.  Without it
        the panorama is a BGR, top-down array of the canvas size, which is what every
        tool and test asks for.

        Returns ``(panorama uint8 | None, quality_ok, quality_reason)`` -- the same
        contract as ``StabStitcher.stab_pano``.
        """
        frame_started = time.perf_counter()
        canvas_w, canvas_h = config["canvas"]
        if canvas_w <= 0 or canvas_h <= 0:
            return None, False, REASON_CANVAS
        if out_size is not None and (int(out_size[0]) <= 0 or int(out_size[1]) <= 0):
            out_size = None
        out_w, out_h = (int(out_size[1]), int(out_size[0])) if out_size is not None \
            else (canvas_w, canvas_h)

        # Sustained plane invalidity is a real failure; a transient miss is not.
        if not plane.get("plane_valid", False):
            if self._plane_invalid_since is None:
                self._plane_invalid_since = time.time()
            elif time.time() - self._plane_invalid_since > self.PLANE_INVALID_GRACE_S:
                return None, False, REASON_PLANE_INVALID
        else:
            self._plane_invalid_since = None

        # Before anything reads a pose: real telemetry is held between 5 Hz samples, and
        # the warp, the snapshot the aligner works from and the canvas frame must all see
        # the same resampled stream.
        if self.POSE_FILTER:
            if self._pose_filter is None:
                self._pose_filter = planar_pose_filter.PoseFilter()
            views = self._pose_filter.apply(views)

        # Each view's own image motion since its last frame, while the aligner is in use:
        # that is what carries an alignment from one solve to the next (planar_track).
        if self._tracker is not None and config.get("pose_refine", False):
            self._tracker.step(views, self._predicted_plane_motion(views, intrinsics,
                                                                   plane, config))

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
        H_all = np.stack([np.asarray(cam["G"], dtype=np.float64) for cam in cams]) \
            @ np.asarray(M, dtype=np.float64)
        kept = []
        for cam, H, aniso in zip(cams, H_all,
                                 pg.homography_anisotropy_batch(H_all, canvas_w, canvas_h)):
            if not np.isfinite(aniso) or aniso > aniso_max:
                # Not a hard failure: drop this view and mosaic the rest. A single
                # grazing drone should not cost the whole panorama.
                continue
            cam["H"] = H
            cam["aniso"] = float(aniso)
            kept.append(cam)

        if len(kept) < 2:
            return None, False, REASON_DISTORTION

        pano, coverage, overlap_psnr = self._render(kept, canvas_w, canvas_h, config, M,
                                                    out_size)
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

        # The geometry this frame was actually rendered with, for anything that has to
        # measure the mosaic rather than reproduce it. Rebuilding it from _last_stats is
        # not equivalent: the canvas centre never appears there, so a caller would have
        # to re-derive the framing and could silently measure a different canvas than the
        # one on screen. Read by tools/planar_clip_bench.py and by the plane-support pass.
        # In CANVAS pixels, as M and every cam["H"] are, whatever size it was drawn at.
        self._last_geometry = (frame, kept, M, canvas_w, canvas_h, float(mpp_view))

        ms = 1e3 * (time.perf_counter() - frame_started)
        self._frame_ms = ms if self._frame_ms is None else 0.9 * self._frame_ms + 0.1 * ms

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
            # What the mosaic was drawn at, and the plane distance one of ITS pixels
            # spans -- the coarser axis, should the output not share the canvas' aspect.
            "out_size": (out_w, out_h),
            "mpp_out": float(mpp_view) * max(canvas_w / float(out_w), canvas_h / float(out_h)),
            "zoom": float(config.get("zoom", 1.0) or 1.0),
            # The scale at which the sharpest view's source pixels land on the plane. An
            # output coarser than this is throwing the feeds' detail away; one much finer
            # is interpolating detail the cameras never captured. It moves with the
            # standoff, so it has to be measured rather than assumed.
            "best_gsd": float(min(c["range"] for c in kept) / K[1, 1]),
            "blend": {BLEND_NEAREST: "nearest", BLEND_MULTIBAND: "multiband"}.get(
                int(config.get("blend_mode", BLEND_NEAREST)), "feather"),
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
            if (self._plane_offset != 0.0 or self._pose_shift
                    or self._correction.get("dpose") is not None
                    or self._correction.get("plane_fix") is not None):
                self._plane_offset = 0.0
                self._pose_shift = {}
                self._sweep_stats = {}
                self._refine_stats = {}
                self._align_stats = {}
                self._sweep_mode = "acquire"
                self._sweep_lost = 0
                if self._aligner is not None:
                    self._aligner.reset()
                self._correction = {"dpose": None, "plane": None, "plane_fix": None}
            time.sleep(0.1)
            return

        if self._learned_alignment():
            started = time.monotonic()
            try:
                self._align(views, K, plane, config, do_sweep, do_refine)
            except Exception as exc:                 # never let the warp thread die
                self._align_stats = {"held": f"aligner error: {exc}"}
            self._maybe_log_estimators()
            time.sleep(max(0.02, self.ALIGN_MIN_PERIOD_S - (time.monotonic() - started)))
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
                "plane_fix": None,
            }

        # Logged from HERE, not from _maybe_log: that line is printed at the end of
        # planar_pano, after every early return, so it goes silent in exactly the
        # situations worth diagnosing.
        self._maybe_log_estimators()

        # The estimators are cheap (a handful of reduced-resolution warps); without a
        # floor this thread would spin on the same snapshot at hundreds of Hz and steal
        # the GPU from the render loop it shares a device and a GIL with.
        time.sleep(0.05)

    # ------------------------------------------------------------------ learned alignment

    def _learned_alignment(self):
        """Whether the two estimator switches drive ``planar_align`` (the default)."""
        if LEGACY_ESTIMATORS or planar_align is None:
            return False
        if self._aligner is None:
            self._aligner = planar_align.PlanarAligner(self.render_device
                                                       if self.render_device.type == "cuda"
                                                       else None)
        if self._aligner.available():
            if self._aligner.models.error and not self._align_warned:
                self._align_warned = True
                print(f"[PLANAR] {self._aligner.models.error}")
            if self._tracker is None and self.VISUAL_TRACK and planar_track.available():
                self._tracker = planar_track.ViewTracker()
            self._aligner.warm_up()                      # once; a no-op afterwards
            return True
        if not self._align_warned:
            self._align_warned = True
            print(f"[PLANAR] learned alignment off ({self._aligner.models.error}); "
                  f"using the photometric estimators")
        return False

    def _align(self, views, K, plane, config, do_sweep, do_refine):
        """
        One ``planar_align`` pass, and the correction it leaves for the render path.

        The aligner is given the *published* poses and plane and keeps its own state, so
        nothing here depends on what ``_correction`` currently holds -- which is what
        lets the render thread go on reading it while this runs.
        """
        aligner = self._aligner
        posed = [v for v in views if pose_is_usable(v)]
        posed, _ = self._drop_stale(posed)
        n_pub, d_pub = self._published_plane(posed, plane, config)
        if n_pub is None or len(posed) < 2:
            self._align_stats = {"skipped": "no usable plane or fewer than 2 posed views"}
            return

        frame_views = []
        for v in posed:
            R, C = pg.unity_pose_to_cv(v["pos"], v["quat"])
            frame_views.append({"drone_id": v["drone_id"], "image": v["image"],
                                "R": R, "C": C})

        # Switching one of the two off must release what it estimated, the same as the
        # photometric pair: they stay independently switchable.
        if not do_refine:
            aligner.pose = {}
        if not do_sweep:
            aligner.plane = None
            aligner.plane_delta = None

        formation = int(plane.get("plane_mode", -1)) == PLANE_MODE_FORMATION_RELATIVE
        sweep_range = float(config.get("sweep_range", 0.0) or 0.0)
        rate = float(config.get("refine_rate", 0.25) or 0.25)
        # With image-based propagation the aligner's own state is left unsmoothed: each
        # solve is an accurate answer for its own frames, and what has to be smooth is
        # the mapping on screen, which is low-passed where the anchors are made, below.
        # Smoothing the *bias* instead makes it lag whenever the offset between a
        # drone's picture and its telemetry changes, which is whenever it accelerates.
        visual = self._tracker is not None and do_refine
        if visual and aligner.mode == "track":
            self._seed_from_screen(aligner, posed, frame_views, K)
        solved = aligner.update(frame_views, K, n_pub, d_pub,
                                solve_pose=do_refine, solve_plane=do_sweep,
                                plane_is_prior_only=formation,
                                max_shift_m=float(config.get("refine_max_shift", 0.0) or 0.0),
                                rate=1.0 if visual else rate,
                                plane_sigma_m=sweep_range if formation else None)

        plane_fix = None
        if do_sweep and aligner.plane is not None:
            # A formation-relative plane is a guess the aligner replaces outright, and a
            # facade does not move, so the estimate is held in WORLD coordinates: it stays
            # put while the formation advances on it, where a correction to the typed
            # standoff would lag by however fast the drones are closing. A raycast plane
            # is a live measurement, so there the estimate rides on it as a delta.
            if formation:
                plane_fix = ("world", aligner.plane[0], aligner.plane[1])
            elif aligner.plane_delta is not None:
                plane_fix = ("delta", aligner.plane_delta[0], aligner.plane_delta[1])
        # One rebind, as everywhere else: the render thread sees a whole generation. It
        # carries where the blend starts from -- whatever is on screen at this instant,
        # which may itself be part-way through the previous blend.
        shown = self._live_correction()
        current = self._correction
        anchors, anchor_frame = current.get("anchors"), current.get("anchor_frame")
        if visual and solved:
            anchors, anchor_frame = self._make_anchors(
                posed, frame_views, K, aligner, n_pub, d_pub, plane_fix,
                current, rate=rate)
        elif not visual:
            anchors = anchor_frame = None
        self._correction = {
            "dpose": aligner.dpose() if do_refine else None,
            "rotvec": aligner.rotvecs() if do_refine else None,
            "plane": None, "plane_fix": plane_fix,
            "weak": frozenset(aligner.stats.get("unsupported") or ()) if do_refine else None,
            "anchors": anchors, "anchor_frame": anchor_frame,
            "blend": (self._clock(), self.ALIGN_BLEND_S,
                      {k: shown.get(k) for k in ("dpose", "rotvec", "plane_fix",
                                                 "anchors", "anchor_frame")}),
        }
        self._align_stats = dict(aligner.stats)
        self._sweep_stats = {}
        self._refine_stats = {}

    # What the poses say the plane did in each view since its last frame, for the tracker.
    _pose_maps = None                # drone_id -> (capture time, G, plane key)

    def _predicted_plane_motion(self, views, intrinsics, plane, config):
        """
        ``{drone_id: 3x3}`` taking a plane point's pixel in a view's previous frame to
        its pixel in this one, according to the (corrected) poses.

        Only its direction and rough size are relied on -- ``planar_track`` uses it to
        tell the plane's motion from that of everything nearer or further -- so the
        timing faults that make poses unfit to *place* a frame do not matter here.
        """
        posed = [v for v in views if pose_is_usable(v)]
        if not posed:
            return None
        n, d = self._published_plane(posed, plane, config)
        if n is None:
            return None
        corr = self._live_correction()
        fix = corr.get("plane_fix")
        if fix is not None:
            n, d = self._apply_plane_fix(fix, n, d, posed)
        e1, e2 = planar_align.plane_basis(n)
        frame = pg.PlaneFrame(n=n, d=float(d), O=float(d) * n, e1=e1, e2=e2)
        K = pg.intrinsics_matrix(*intrinsics)
        key = (tuple(np.round(n, 6)), round(float(d), 4))
        previous = self._pose_maps or {}
        now, out = {}, {}
        for v in posed:
            did, t = v["drone_id"], float(v.get("capture_time", 0.0) or 0.0)
            R, C = pg.unity_pose_to_cv(v["pos"], v["quat"])
            R, C = self._apply_correction(v, R, C, corr)
            G = pg.build_G(K, R, C, frame)
            before = previous.get(did)
            if before is not None and before[0] == t:
                now[did] = before                          # the same frame again
                continue
            now[did] = (t, G, key)
            if before is not None and before[2] == key and 0.0 < t - before[0] < 0.5:
                try:
                    out[did] = G @ np.linalg.inv(before[1])
                except np.linalg.LinAlgError:
                    pass
        self._pose_maps = now
        return out

    # ---- anchors: an alignment pinned to the frames it was solved on ---------------------

    @staticmethod
    def _plane_change(src, dst):
        """3x3 taking plane coordinates in frame ``src`` to those in frame ``dst``."""
        rel = src.O - dst.O
        return np.array([[src.e1 @ dst.e1, src.e2 @ dst.e1, rel @ dst.e1],
                         [src.e1 @ dst.e2, src.e2 @ dst.e2, rel @ dst.e2],
                         [0.0, 0.0, 1.0]])

    def _make_anchors(self, posed, frame_views, K, aligner, n_pub, d_pub, plane_fix,
                      current, rate):
        """
        ``({drone_id: (capture time, G)}, PlaneFrame)`` for the frames just solved on.

        ``G`` maps the anchor frame's plane coordinates to pixels of *that* frame of
        that drone, with the solve's corrections in it -- the one place the answer is
        exact.  ``planar_track`` carries it to later frames.

        Each new anchor is low-passed against the one on screen, brought forward to the
        same frame: the solutions differ from pass to pass by a pixel or so (which
        matches were found, a surface that is not quite a plane) and the mosaic should
        not tick with them.  That is the smoothing ``refine_rate`` asks for, applied to
        the mapping rather than to a pose bias.  A view whose new mapping is far from
        its old one is not noise and is taken as it comes (see ``ANCHOR_SNAP_PX``).
        """
        dpose = aligner.dpose() or {}
        if aligner.plane is not None:
            n_a, d_a = np.asarray(aligner.plane[0], dtype=np.float64), float(aligner.plane[1])
        else:
            n_a, d_a = np.asarray(n_pub, dtype=np.float64), float(d_pub)

        corrected = []
        for fv in frame_views:
            R, C = fv["R"], fv["C"]
            delta = dpose.get(fv["drone_id"])
            if delta is not None:
                R, C = delta[0] @ R, C + delta[1]
            corrected.append((R, C))
        centroid = np.mean([C for _R, C in corrected], axis=0)
        origin = centroid - (float(n_a @ centroid) - d_a) * n_a
        e1, e2 = planar_align.plane_basis(n_a)
        frame = pg.PlaneFrame(n=n_a, d=d_a, O=origin, e1=e1, e2=e2)

        prev, prev_frame = current.get("anchors"), current.get("anchor_frame")
        mix = float(np.clip(2.0 * rate, 0.3, 1.0))
        to_prev = (self._plane_change(frame, prev_frame)
                   if prev and prev_frame is not None else None)
        h_img, w_img = posed[0]["image"].shape[:2]
        probe = np.array([[0.5 * w_img, 0.2 * w_img, 0.8 * w_img, 0.2 * w_img, 0.8 * w_img],
                          [0.5 * h_img, 0.2 * h_img, 0.2 * h_img, 0.8 * h_img, 0.8 * h_img],
                          [1.0, 1.0, 1.0, 1.0, 1.0]])
        unsupported = set(aligner.stats.get("unsupported") or ())
        carry = (self._plane_change(frame, prev_frame)
                 if prev and prev_frame is not None else None)
        anchors = {}
        for v, (R, C) in zip(posed, corrected):
            did, t_a = v["drone_id"], float(v.get("capture_time", 0.0) or 0.0)
            if did in unsupported and carry is not None and did in prev:
                # Nothing was measured for this view in this pass. The mapping it already
                # has is still the images' own; a fresh one would be its published pose
                # with a correction nobody checked, which is how a view jumps a metre the
                # moment it loses its matches. Kept as it is, on the new frame's axes,
                # until it is measured again or grows too old to trust.
                t_p, G_p = prev[did]
                if t_a - t_p <= self.MAX_ANCHOR_AGE_S \
                        and self._tracker.between(did, t_p, t_a) is not None:
                    anchors[did] = (t_p, G_p @ carry)
                    continue
            G = pg.build_G(K, R, C, frame)
            if to_prev is not None and did in prev:
                t_p, G_p = prev[did]
                A = self._tracker.between(did, t_p, t_a)
                if A is not None:
                    G_old = A @ G_p @ to_prev
                    # How far apart the two mappings put this view, in its own pixels.
                    # A small difference is the pass-to-pass noise the low-pass exists
                    # for. A large one is a view being put right -- its first lock, or
                    # a re-lock after it lost its matches -- and easing that in would
                    # leave it visibly wrong for a second; only that view snaps.
                    try:
                        q = G @ np.linalg.solve(G_old, probe)
                        moved = float(np.max(np.linalg.norm(q[:2] / q[2] - probe[:2], axis=0)))
                    except np.linalg.LinAlgError:
                        moved = np.inf
                    if moved <= self.ANCHOR_SNAP_PX:
                        G = (1.0 - mix) * G_old + mix * G
            anchors[did] = (t_a, G)
        return anchors, frame

    def _seed_from_screen(self, aligner, posed, frame_views, K):
        """
        Start the next solve from the alignment on screen, not from the last pose bias.

        The aligner rectifies each view with its published pose plus the correction it
        holds, and only accepts matches within a gate of that.  The correction is a bias
        on the telemetry, and the telemetry's offset from the picture is not constant: a
        drone whose video lags moves a metre and more against its own pose when the
        formation starts or stops.  Its matches then fall outside the gate, it loses its
        support, and nothing brings it back.  What *is* continuous is the mapping the
        images have carried forward, so each view's correction is replaced by the one
        that reproduces that mapping on the snapshot's pose -- the gate is then centred
        on what the pilot is looking at.
        """
        visual, frame = self._visual_mappings(posed, self._correction)
        if not visual:
            return
        K_inv = np.linalg.inv(K)
        for fv in frame_views:
            G = visual.get(fv["drone_id"])
            if G is None:
                continue
            R_vis, C_vis = self._pose_from_G(G, K_inv, frame)
            aligner.pose[fv["drone_id"]] = (
                planar_align.matrix_to_rotvec(R_vis @ np.asarray(fv["R"]).T),
                C_vis - np.asarray(fv["C"]))

    def _visual_mappings(self, posed, corr):
        """
        ``({drone_id: G}, PlaneFrame)``: each view's mapping for the frame in hand,
        carried from its anchor by its own image motion.  Views whose chain cannot be
        vouched for are simply absent and fall back to their pose.
        """
        anchors, frame = corr.get("anchors"), corr.get("anchor_frame")
        if not anchors or frame is None or self._tracker is None:
            return None, None
        prev, prev_frame = corr.get("anchors_prev"), corr.get("anchor_frame_prev")
        b = float(corr.get("anchor_blend", 1.0))
        to_prev = (self._plane_change(frame, prev_frame)
                   if prev and prev_frame is not None and b < 1.0 else None)
        out = {}
        for v in posed:
            did, t = v["drone_id"], float(v.get("capture_time", 0.0) or 0.0)
            G = self._propagate(did, anchors.get(did), t)
            if G is None:
                continue
            if to_prev is not None:
                G_old = self._propagate(did, prev.get(did), t)
                if G_old is not None:
                    G = (1.0 - b) * (G_old @ to_prev) + b * G
            out[did] = G
        return (out, frame) if out else (None, None)

    def _propagate(self, drone_id, anchor, t_now):
        if anchor is None:
            return None
        t_a, G_a = anchor
        if t_now - t_a > self.MAX_ANCHOR_AGE_S or t_now < t_a:
            return None
        A = self._tracker.between(drone_id, t_a, t_now)
        return None if A is None else A @ G_a

    @staticmethod
    def _pose_from_G(G, K_inv, frame):
        """
        The camera ``(R, C)`` a plane-to-image mapping implies.

        ``K^-1 G = R [e1 | e2 | O - C]`` for a true pose; a propagated mapping is only
        close to that form, so the first two columns are replaced by the nearest
        orthonormal pair.  For the consumers that need rays to go with a view's pixels.
        """
        M = K_inv @ G
        U, sv, Vt = np.linalg.svd(M[:, :2], full_matrices=False)
        Q = U @ Vt
        r3 = np.cross(Q[:, 0], Q[:, 1])
        R = np.column_stack([Q[:, 0], Q[:, 1], r3]) @ np.stack([frame.e1, frame.e2, frame.n])
        scale = float(sv.mean()) if sv.mean() > 1e-9 else 1.0
        return R, frame.O - R.T @ (M[:, 2] / scale)

    def _live_correction(self):
        """
        The correction to apply *now*: ``_correction``, eased in from the one before it.

        Read-only on shared state, so either thread may call it.  The warp thread
        publishes a generation together with the values it replaces; until
        ``ALIGN_BLEND_S`` has passed, what is applied is a smoothstep between the two.
        """
        corr = self._correction
        blend = corr.get("blend")
        if blend is None:
            return corr
        t0, duration, prev = blend
        b = (self._clock() - t0) / duration if duration > 0.0 else 1.0
        if b >= 1.0:
            return corr
        b = max(0.0, b)
        b = b * b * (3.0 - 2.0 * b)

        out = {"plane": corr.get("plane"), "dpose": corr.get("dpose"),
               "rotvec": corr.get("rotvec"), "plane_fix": corr.get("plane_fix"),
               "weak": corr.get("weak"),
               # Anchors are blended where they are used, once each has been carried to
               # the frame in hand (_visual_mappings); here they only travel together.
               "anchors": corr.get("anchors"), "anchor_frame": corr.get("anchor_frame"),
               "anchors_prev": prev.get("anchors"),
               "anchor_frame_prev": prev.get("anchor_frame"), "anchor_blend": b}
        new_dp, old_dp = corr.get("dpose"), prev.get("dpose")
        new_rv, old_rv = corr.get("rotvec") or {}, prev.get("rotvec") or {}
        if new_dp and old_dp:
            dp, rv = {}, {}
            for did, (dR, dC) in new_dp.items():
                if did in old_dp and did in new_rv and did in old_rv:
                    om = old_rv[did] + b * (new_rv[did] - old_rv[did])
                    c0 = old_dp[did][1]
                    dp[did] = (planar_align.rotvec_to_matrix(om), c0 + b * (dC - c0))
                    rv[did] = om
                else:
                    # A drone the previous generation did not hold has nothing to ease
                    # from; it takes the new value at once.
                    dp[did] = (dR, dC)
                    if did in new_rv:
                        rv[did] = new_rv[did]
            out["dpose"], out["rotvec"] = dp, rv

        new_fix, old_fix = corr.get("plane_fix"), prev.get("plane_fix")
        if new_fix is not None and old_fix is not None and new_fix[0] == old_fix[0]:
            first = old_fix[1] + b * (new_fix[1] - old_fix[1])
            if new_fix[0] == "world":
                first = first / np.linalg.norm(first)
            out["plane_fix"] = (new_fix[0], first,
                                old_fix[2] + b * (new_fix[2] - old_fix[2]))
        return out

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
        try:
            # Every view's four corners in one solve; the per-view form below is the same
            # arithmetic and stays as the fallback for a singular G in the batch.
            h, w = cams[0]["view"]["image"].shape[:2]
            corners = np.array([[0.0, w - 1.0, w - 1.0, 0.0],
                                [0.0, 0.0, h - 1.0, h - 1.0],
                                [1.0, 1.0, 1.0, 1.0]])
            p = np.linalg.solve(np.stack([np.asarray(c["G"], dtype=np.float64) for c in cams]),
                                np.broadcast_to(corners, (len(cams), 3, 4)))
            min_recip = 1.0 / max_range if np.isfinite(max_range) and max_range > 0.0 else 0.0
            ok = np.isfinite(p).all(axis=1) & (p[:, 2] > max(min_recip, 1e-9))       # [N, 4]
            with np.errstate(divide="ignore", invalid="ignore"):
                ab = p[:, :2] / p[:, 2:3]                                             # [N, 2, 4]
            lo_all = np.where(ok[:, None, :], ab, np.inf).min(axis=2)
            hi_all = np.where(ok[:, None, :], ab, -np.inf).max(axis=2)
            for k in np.nonzero(ok.sum(axis=1) >= 3)[0]:
                lo, hi = lo_all[k], hi_all[k]
                spans.append((lo, hi, float((hi[0] - lo[0]) * (hi[1] - lo[1]))))
        except np.linalg.LinAlgError:
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

    def _estimator_grid(self, canvas_w, canvas_h):
        """
        Homogeneous pixel grid ``[3, H*W]`` of the estimator canvas, cached per size.

        The warp thread's own cache: the render path keeps a lattice of the output
        raster (``_lattice``), which is another size and another thread's to write.
        """
        key = (canvas_w, canvas_h, str(self.render_device))
        if self._est_grid_key == key:
            return self._est_grid
        ys, xs = torch.meshgrid(
            torch.arange(canvas_h, dtype=torch.float32, device=self.render_device),
            torch.arange(canvas_w, dtype=torch.float32, device=self.render_device),
            indexing="ij")
        grid = torch.stack([xs.reshape(-1), ys.reshape(-1),
                            torch.ones(canvas_h * canvas_w, dtype=torch.float32,
                                       device=self.render_device)], dim=0)
        self._est_grid_key, self._est_grid = key, grid
        return grid

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

        n, d = self._published_plane(posed, plane, config)
        if n is None:
            return None, []

        # One read of the correction for the whole build, so the plane and every view
        # come from the same generation and the same instant of any blend in progress.
        corr = self._live_correction()

        # The aligner's plane, when it has one. Not applied to an explicit candidate
        # (plane_offset given): that is the photometric sweep evaluating its own scan,
        # which the aligner never runs alongside.
        fix = corr.get("plane_fix") if plane_offset is None else None
        if fix is not None:
            n, d = self._apply_plane_fix(fix, n, d, posed)

        # Additive, not absolute: Unity's raycast keeps tracking the facade as the swarm
        # flies, and the sweep estimates only the slowly-varying residual on top of it.
        # An absolute override would freeze the plane at whatever the sweep last saw.
        if plane_offset is None:
            plane_offset = corr.get("plane") or 0.0
        d += float(plane_offset)

        # Mappings carried forward from the last solve by the images themselves. They
        # live on the plane of that solve, so that is the plane this frame is built on.
        visual, anchor_frame = (self._visual_mappings(posed, corr)
                                if apply_dpose and corr.get("anchors") else (None, None))
        if visual:
            n, d = anchor_frame.n, anchor_frame.d

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
            R_ref, C_ref = self._apply_correction(ref, R_ref, C_ref, corr)

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
        right_ref, up_ref = R_ref[0], -R_ref[1]
        if visual and ref["drone_id"] in visual:
            # The same two definitions -- where the reference view's principal point
            # lands, and which way its image x axis runs there -- read off its carried
            # mapping instead of its pose, so the canvas follows what that view shows and
            # not when its telemetry happened to arrive.
            src_h, src_w = ref["image"].shape[:2]
            try:
                q = np.linalg.solve(visual[ref["drone_id"]],
                                    np.array([[K[0, 2], K[0, 2] + 0.1 * src_w],
                                              [K[1, 2], K[1, 2]], [1.0, 1.0]]))
            except np.linalg.LinAlgError:
                q = None
            if q is not None and q[2, 0] > 1e-9 and q[2, 1] > 1e-9:
                ab = q[:2] / q[2]
                centre = (anchor_frame.O + ab[0, 0] * anchor_frame.e1
                          + ab[1, 0] * anchor_frame.e2)
                along = (anchor_frame.O + ab[0, 1] * anchor_frame.e1
                         + ab[1, 1] * anchor_frame.e2) - centre
                if np.linalg.norm(along) > 1e-6:
                    origin, right_ref, up_ref = centre, along, None
        frame = pg.build_plane_frame(n, d, origin, right_ref, up_ref)
        to_anchor = self._plane_change(frame, anchor_frame) if visual else None
        K_inv = np.linalg.inv(K) if visual else None

        max_range = config.get("max_range", np.inf)
        weak = (corr.get("weak") or ()) if apply_dpose else ()
        cams = []
        for v in posed:
            R, C = pg.unity_pose_to_cv(v["pos"], v["quat"])
            if apply_dpose:
                R, C = self._apply_correction(v, R, C, corr)

            # Reject views that cannot see the plane at all before doing any work.
            _, rng = pg.ray_plane_intersect(C, R[2], n, d, max_range=max_range)
            if not np.isfinite(rng):
                continue

            if visual and v["drone_id"] in visual:
                # Drawn through its carried mapping; the pose that goes with it (for the
                # partition, and for anything that casts rays) is the one it implies.
                G = visual[v["drone_id"]] @ to_anchor
                R, C = self._pose_from_G(G, K_inv, frame)
            else:
                G = pg.build_G(K, R, C, frame)

            # Camera centre in the plane frame: (a, b) is its footprint on the plane and
            # h its perpendicular standoff. BLEND_NEAREST needs only these three scalars
            # per view -- see _render for why the per-pixel obliquity reduces to them.
            rel = C - frame.O
            cams.append({
                "view": v,
                "G": G,
                "range": rng,
                "plane_ab": (float(rel @ frame.e1), float(rel @ frame.e2)),
                "plane_h": abs(float(rel @ frame.n)),
                # The pose G was built from, corrections included. Carried rather than
                # recomputed by consumers: planar_bundle triangulates across view pairs
                # and needs the ray origins to be the same ones the warp used, and
                # re-deriving them from view["pos"]/["quat"] silently drops whatever
                # _apply_correction had already applied.
                "R": R,
                "C": C,
                # A view the aligner could not anchor to the others is drawn from its
                # published pose, metres out. It still fills what nothing else sees, but
                # it should not win a pixel from a view that is known to be in place.
                "weak": v["drone_id"] in weak,
            })

        return frame, cams

    def _published_plane(self, posed, plane, config):
        """
        The plane as published, before any estimate: ``(n, d)`` in right-handed world, or
        ``(None, 0.0)``.  Formation-relative derives it from the poses; every other mode
        reads Unity's normal and offset.
        """
        if int(plane.get("plane_mode", -1)) == PLANE_MODE_FORMATION_RELATIVE:
            return self._plane_from_formation(posed, float(config.get("standoff", 0.0)))
        n = pg.unity_dir_to_rh(plane["plane_normal"])
        if np.linalg.norm(n) < 1e-6:
            return None, 0.0
        return n / np.linalg.norm(n), float(plane["plane_d"])

    def _apply_plane_fix(self, fix, n, d, posed):
        """
        Replace or adjust the published plane with the aligner's estimate.

        ``("world", n, d)`` is the estimate itself and simply wins.  ``("delta", w, off)``
        turns the published normal by the rotation vector ``w`` and moves the plane ``off`` metres along its
        old normal, about the foot of the formation's centroid -- the same pivot the
        delta was measured about, recomputed from this frame's poses so the adjustment
        travels with the published plane instead of pinning it.
        """
        if fix[0] == "world":
            self._plane_source = "aligned"
            return np.asarray(fix[1], dtype=np.float64), float(fix[2])
        centres = np.asarray([pg.unity_point_to_rh(v["pos"]) for v in posed],
                             dtype=np.float64)
        centroid = centres.mean(axis=0)
        foot = centroid - (float(n @ centroid) - d) * n
        n_new = planar_align.rotvec_to_matrix(fix[1]) @ n
        n_new = n_new / np.linalg.norm(n_new)
        return n_new, float(n_new @ (foot + float(fix[2]) * n))

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

    def _apply_correction(self, view, R, C, corr=None):
        """
        Apply the refiner's delta-pose.  Identity until compute_warps estimates one.

        ``dR`` may be ``None``: the current refiner estimates translation only, and a
        None here means "no rotation correction" rather than costing an identity matmul
        per view per frame.  The slot is kept so a rotation-capable refiner can fill it
        without changing this call site.
        """
        dpose = (self._correction if corr is None else corr).get("dpose")
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
    #
    # Two rasters, deliberately not the same thing:
    #
    # - the CANVAS (``config["canvas"]``, Unity's planarCanvasWidth/Height) is the framing.
    #   With metres-per-pixel it says which patch of plane is shown, and everything that
    #   is expressed in canvas pixels stays in those units whatever the output size: the
    #   auto-fit, the pan, the anisotropy gate, the blend width, the overlap PSNR and the
    #   estimators' reduced canvas.
    # - the OUTPUT raster is how many pixels that patch is drawn with.  It is the caller's
    #   (``out_size``, Unity's planarRenderScale x the canvas) and only the last step of
    #   the render sees it.
    #
    # They were one raster once, and the mosaic then went canvas -> cv2.resize -> wire: two
    # resamplings of the feeds, the second to a buffer of another aspect, on a canvas that
    # could not be enlarged without every per-view tensor growing with it.  Splitting them
    # is what lets the output be drawn at the feeds' own resolution while the estimators
    # go on measuring on the canvas they were tuned on.
    #
    # The render is organised so that its full-resolution work does not grow with the
    # number of views:
    #
    # - each view's fields (where the raster's pixels fall in its image, and how good a
    #   look it has of them) are evaluated over its own footprint's bounding box only, one
    #   view at a time.  The arithmetic is the same per-pixel arithmetic as ever; what
    #   changed is the working set, which now stays inside the GPU's cache instead of
    #   being N canvas-sized tensors (0.02 ms an op against 0.3);
    # - under the two winner-take-all blends every output pixel shows exactly one view, so
    #   exactly one bilinear sample is taken per pixel, out of all the frames stacked into
    #   one texture.  Warping every view across the whole raster and discarding all but
    #   one was most of the old cost;
    # - what MULTIBAND adds is low-frequency by construction, so it is computed on a grid
    #   a blend-width coarse and only looked up at full resolution;
    # - the overlap PSNR is a diagnostic, measured on the canvas and not on every frame.
    #
    # What is left is mostly the interpreter: a small tensor op costs ~10 us to dispatch
    # however little it computes, and the render thread shares its GIL with the two
    # others.  Hence the stacked ``(u w, v w, w)`` tensors and the batched row and column
    # terms below -- they are there to halve the number of ops, not the arithmetic.

    # Source u8 frames on the device and their pinned staging buffer (see _upload_sources),
    # the pinned output ring (see _out_buffer), the PSNR throttle and the render timers.
    # Class-level so a stitcher built without __init__ -- the self-test's -- has them.
    _src_cache = None
    _out_ring = None
    _out_idx = 0
    _lattice_cache = None
    _limit_cache = None
    _psnr_cache = None
    _render_ms = None
    _frame_ms = None

    # Sampling coordinate of a pixel no view covers: far outside any image, so the
    # sampler's zero padding returns black without a mask.
    _NOWHERE = -1.0e4
    # Label of such a pixel. The block section holds 24 views, so 255 is never a view.
    _NO_VIEW = 255

    @staticmethod
    def _raster_matrix(canvas_w, canvas_h, out_w, out_h, bottom_up):
        """
        3x3 taking an output pixel ``(X, Y, 1)`` to the canvas pixel it shows.

        Pixel areas are what line up, as in ``cv2.resize`` -- output pixel ``X`` covers
        canvas ``[X, X + 1) * canvas_w / out_w`` -- so the mosaic is framed exactly as
        the canvas resized to the output would be, at any output size and aspect.  With
        the output equal to the canvas this is the identity.

        ``bottom_up`` reverses the rows, which is the wire's order (Unity's textures start
        at the bottom-left): the flip costs nothing here, where it is a sign in a matrix.
        """
        sx, sy = canvas_w / float(out_w), canvas_h / float(out_h)
        if bottom_up:
            return np.array([[sx, 0.0, 0.5 * sx - 0.5],
                             [0.0, -sy, (out_h - 0.5) * sy - 0.5],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
        return np.array([[sx, 0.0, 0.5 * sx - 0.5],
                         [0.0, sy, 0.5 * sy - 0.5],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def _lattice(self, out_w, out_h):
        """
        The output raster's pixel coordinates as a row ``[1, W]`` and a column ``[H, 1]``,
        measured from the raster's centre.

        Kept as two vectors rather than a grid: every field of every view is a sum of a
        function of the column and a function of the row, so the full-size tensor only
        appears when the two are added.  Centred because float32 then spends its bits on
        the half-width rather than the width; the matrices are shifted to match, in
        float64, before they are handed to the device.
        """
        key = (out_w, out_h, str(self.render_device))
        cached = self._lattice_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        dev = self.render_device
        x = (torch.arange(out_w, dtype=torch.float32, device=dev) - 0.5 * out_w).view(1, out_w)
        y = (torch.arange(out_h, dtype=torch.float32, device=dev) - 0.5 * out_h).view(out_h, 1)
        self._lattice_cache = (key, x, y)
        return x, y

    def _frame_limit(self, src_w, src_h, dev):
        """``(src_w - 1, src_h - 1)`` as a ``[2, 1, 1]`` tensor, cached."""
        key = (src_w, src_h, str(dev))
        cached = self._limit_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        limit = torch.tensor([src_w - 1.0, src_h - 1.0], dtype=torch.float32,
                             device=dev).view(2, 1, 1)
        self._limit_cache = (key, limit)
        return limit

    @staticmethod
    def _view_rois(H_out, src_w, src_h, out_w, out_h, pad=2):
        """
        ``(y0, y1, x0, x1)`` per view: the output pixels it can possibly cover.

        The image is a rectangle and a homography takes it to a convex quadrilateral, so
        the bounding box of its four projected corners contains every pixel the view
        reaches -- provided all four corners actually land on the plane in front of the
        camera.  ``H_out^-1 (u, v, 1)`` is ``(X, Y, 1) / depth``, so that is one sign
        test on its third component; a view that fails it (the horizon is in frame)
        simply gets the whole raster, which is what every view used to get.
        """
        n = len(H_out)
        whole = (0, out_h, 0, out_w)
        corners = np.array([[0.0, src_w - 1.0, src_w - 1.0, 0.0],
                            [0.0, 0.0, src_h - 1.0, src_h - 1.0],
                            [1.0, 1.0, 1.0, 1.0]])
        try:
            p = np.linalg.solve(H_out, np.broadcast_to(corners, (n, 3, 4)))
        except np.linalg.LinAlgError:
            return [whole] * n
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = p[:, :2] / p[:, 2:3]
        good = (np.isfinite(p).all(axis=(1, 2)) & (p[:, 2] > 1e-12).all(axis=1)
                & np.isfinite(xy).all(axis=(1, 2)))
        xy = np.where(good[:, None, None], xy, 0.0)
        lo = np.floor(xy.min(axis=2)) - pad
        hi = np.ceil(xy.max(axis=2)) + pad + 1
        x0, x1 = np.clip(lo[:, 0], 0, out_w).astype(int), np.clip(hi[:, 0], 0, out_w).astype(int)
        y0, y1 = np.clip(lo[:, 1], 0, out_h).astype(int), np.clip(hi[:, 1], 0, out_h).astype(int)
        return [(int(y0[i]), int(y1[i]), int(x0[i]), int(x1[i])) if good[i] else whole
                for i in range(n)]

    @staticmethod
    def _lattice_terms(H, x, y):
        """
        ``(rows [N,3,1,W], cols [N,3,H,1])`` whose sum is, for every view, the homogeneous
        source pixel ``(u w, v w, w)`` of every point of the lattice ``x`` (a row of
        abscissae) by ``y`` (a column of ordinates).  ``H`` is ``[N,3,3]``, lattice ->
        source.  The two terms are a few thousand numbers; a view's full-size field is
        one add of a slice of each.
        """
        return (H[:, :, 0, None, None] * x + H[:, :, 2, None, None],
                H[:, :, 1, None, None] * y)

    @staticmethod
    def _view_fields(uvd, limit, max_range):
        """
        Where a lattice's points fall in a view.  Returns ``(uv, f, ok)``.

        ``uvd`` ``[..., 3, H, W]`` is the sum of :meth:`_lattice_terms` and is consumed.
        ``uv`` ``[..., 2, H, W]`` are source pixel coordinates, ``f`` the distance in
        source pixels to the nearest image border (negative outside the frame) and ``ok``
        whether the point is in front of the camera and within ``max_range`` -- the
        third component of the homography is camera depth in metres, so both tests are
        on that one number.  Where ``ok`` is false ``uv`` is finite but meaningless.
        """
        depth = uvd[..., 2, :, :]
        ok = depth > 1e-6
        if np.isfinite(max_range):
            ok &= depth < max_range
        uv = uvd[..., :2, :, :].mul_(torch.where(ok, depth, 1.0).reciprocal_().unsqueeze(-3))
        f = torch.minimum(uv, limit - uv).amin(dim=-3)
        return uv, f, ok

    def _plane_geometry(self, cams, dev):
        """
        ``[N, 4]``: each camera's footprint ``(a, b)`` on the plane, its standoff ``h``,
        and ``h`` times its priority -- the four scalars a view's score is made of.
        """
        return torch.tensor(
            [[c["plane_ab"][0], c["plane_ab"][1], c["plane_h"],
              c["plane_h"] * (self.WEAK_VIEW_PRIORITY if c.get("weak") else 1.0)]
             for c in cams], dtype=torch.float32, device=dev)

    @staticmethod
    def _plane_terms(M, x, y, geo):
        """
        ``(rows [N,1,W], cols [N,H,1])`` whose sum is the squared distance from each
        view's camera to each lattice point, the latter being on the plane.  ``M`` takes
        the lattice to plane coordinates and is axis-aligned (``canvas_to_plane_matrix``),
        which is what splits the distance into a row and a column.
        """
        if abs(M[0, 1]) > 1e-12 or abs(M[1, 0]) > 1e-12:
            raise ValueError("the canvas matrix is expected to be axis-aligned")
        rows = (x * float(M[0, 0]) + float(M[0, 2]) - geo[:, 0, None, None]).square()
        cols = (y * float(M[1, 1]) + float(M[1, 2]) - geo[:, 1, None, None]).square()
        return rows, cols + geo[:, 2, None, None].square()

    @staticmethod
    def _score(f, ok, feather_px, cos):
        """
        Winner-take-all score: how face-on a view sees a point, faded toward its border.

        Each covered pixel goes to the view seeing that plane point closest to the plane
        normal -- the least foreshortened look at it.  Because every pixel lies *on* the
        plane, that obliquity has a closed form in three per-view scalars: with the
        camera at in-plane footprint ``(a_v, b_v)`` and perpendicular standoff ``h_v``,
        the incidence cosine at plane point ``(a, b)`` is

            cos = h_v / sqrt((a - a_v)^2 + (b - b_v)^2 + h_v^2)

        (the numerator is constant per view precisely because ``n . X == d`` everywhere
        on the canvas).  Maximising it is a Voronoi partition of the plane by camera
        footprint, so each drone renders the patch of facade it is parked in front of.
        ``cos`` arrives already multiplied by the view's priority.

        The border falloff multiplies the score rather than gating it, which keeps seams
        off the source-image edges: a view running out of frame fades below a
        neighbour's score before it runs out of pixels, so the winner changes over solid
        image on both sides instead of at a hard image boundary.
        """
        s = (f * (1.0 / feather_px)).clamp_(max=1.0).mul_(cos)
        return torch.where(ok & (f > 0), s, 0.0)

    def _upload_sources(self, cams):
        """
        The views' frames as one ``[N, H, W, 3]`` uint8 tensor on the render device.

        uint8 across PCIe and converted on the GPU: a quarter of the bytes of a float
        upload.  Each frame goes through a pinned staging buffer (a DMA instead of a
        pageable copy), and only when it has changed -- ``read_block_memory`` re-serves
        the array it already handed out for a block the producer has not rewritten, and
        on the real fleet the feeds arrive slower than the render is woken.  A frame is
        recognised by being the same array; the few sampled pixels summed beside it
        catch one rewritten in place.
        """
        images = [c["view"]["image"] for c in cams]
        n = len(images)
        h, w = images[0].shape[:2]
        dev = self.render_device
        if dev.type != "cuda":
            return torch.from_numpy(np.stack(images)).to(dev)

        cache = self._src_cache
        if cache is None or cache["shape"] != (n, h, w) or cache["dev"].device != dev:
            host = torch.empty((n, h, w, 3), dtype=torch.uint8).pin_memory()
            cache = {"shape": (n, h, w), "host": host, "np": host.numpy(),
                     "dev": torch.empty((n, h, w, 3), dtype=torch.uint8, device=dev),
                     "keys": [None] * n}
            self._src_cache = cache

        synced = False
        for k, image in enumerate(images):
            mark = int(image[::61, ::67].sum())
            key = cache["keys"][k]
            if key is not None and key[0] is image and key[1] == mark:
                continue
            if not synced:
                # A copy out of the staging buffer may still be in flight if the last
                # render ended early; it must land before the buffer is written again.
                torch.cuda.current_stream().synchronize()
                synced = True
            np.copyto(cache["np"][k], image)
            cache["dev"][k].copy_(cache["host"][k], non_blocking=True)
            cache["keys"][k] = (image, mark)
        return cache["dev"]

    def _out_buffer(self, shape):
        """
        Next output buffer of the given shape from a small ring, pinned on CUDA.

        A ring because the buffer is handed to ``first_thread`` by reference and written
        to shared memory a few milliseconds later, while this thread may already be
        rendering the next frame; three is two more than that needs.
        """
        ring = self._out_ring
        if not ring or tuple(ring[0].shape) != tuple(shape):
            ring = []
            for _ in range(3):
                t = torch.empty(shape, dtype=torch.uint8)
                if self.render_device.type == "cuda":
                    t = t.pin_memory()
                ring.append(t)
            self._out_ring, self._out_idx = ring, 0
        buf = ring[self._out_idx]
        self._out_idx = (self._out_idx + 1) % len(ring)
        return buf

    def _render(self, cams, canvas_w, canvas_h, config, M, out_size=None):
        """
        Draw the mosaic.

        ``M`` is the canvas-pixel -> plane-coordinate matrix and each ``cam["H"]`` the
        canvas-pixel -> source-pixel homography, both in *canvas* pixels.  ``out_size``
        is ``(height, width)`` of the output raster; with it the result is a
        :class:`WireReadyPanorama` (RGB, bottom-up rows), without it a BGR top-down
        array of the canvas size.

        Returns ``(panorama uint8 [H, W, 3], coverage, worst pairwise overlap PSNR)``.
        """
        started = time.perf_counter()
        dev = self.render_device
        wire = out_size is not None
        out_h, out_w = (int(out_size[0]), int(out_size[1])) if wire else (canvas_h, canvas_w)
        src_h, src_w = cams[0]["view"]["image"].shape[:2]
        max_range = float(config.get("max_range", 0.0)) or float("inf")
        feather_px = max(1.0, float(config.get("feather_px", 1)))
        blend_mode = int(config.get("blend_mode", BLEND_NEAREST))
        debug_view = int(config.get("debug_view", DEBUG_OFF))
        sampling = config.get("sampling", SAMPLING)
        if sampling not in ("bilinear", "bicubic"):
            sampling = "bicubic"

        # Output pixel -> canvas pixel, and through it to each view and to the plane.
        # Composed in float64; only the finished coefficients go to the device.
        H_canvas = np.stack([np.asarray(c["H"], dtype=np.float64) for c in cams])
        S = self._raster_matrix(canvas_w, canvas_h, out_w, out_h, bottom_up=wire)
        centre = np.array([[1.0, 0.0, 0.5 * out_w], [0.0, 1.0, 0.5 * out_h], [0.0, 0.0, 1.0]])
        H_out = H_canvas @ S
        rois = self._view_rois(H_out, src_w, src_h, out_w, out_h)
        S_c = S @ centre
        M_c = np.asarray(M, dtype=np.float64) @ S_c
        x, y = self._lattice(out_w, out_h)
        limit = self._frame_limit(src_w, src_h, dev)
        # The frame's few small tensors go to the device BEFORE its images do. A copy from
        # pageable memory waits for whatever is queued ahead of it, and behind six frames
        # of pixels each of these is 0.4 ms of the render thread standing still.
        H_c = torch.as_tensor(H_out @ centre, dtype=torch.float32, device=dev)
        H_t = torch.as_tensor(H_canvas, dtype=torch.float32, device=dev)
        geo = self._plane_geometry(cams, dev)

        src = self._upload_sources(cams).permute(0, 3, 1, 2).float()       # [N,3,Hs,Ws] BGR

        # Always measured: this is the number that quantifies pose error. Taken from the
        # frames as they arrived -- before the debug overlay, which would otherwise be
        # measured instead of the imagery, and before the exposure gains.
        overlap_psnr = self._measure_overlap(cams, src, H_t, canvas_w, canvas_h,
                                             config, max_range)

        if wire:
            src = src.flip(1)                                             # RGB on the wire
        if debug_view != DEBUG_OFF:
            # A colour is the same colour wherever it is resampled to, so the overlay is
            # applied to the frames and every blend mode shows it its own way: NEAREST
            # flat and hard-edged (that *is* the partition), FEATHER as blends of two
            # colours over the overlaps -- a direct picture of how much is being averaged.
            colours = self._debug_colours(cams, src, rgb=wire)
            src = colours.expand_as(src).contiguous() if debug_view == DEBUG_FLAT \
                else torch.lerp(src, colours.expand_as(src), DEBUG_TINT_STRENGTH)

        if blend_mode == BLEND_FEATHER:
            pano, covered = self._compose_feather(H_c, x, y, rois, src, limit, max_range,
                                                  feather_px, sampling)
        else:
            low = None
            if blend_mode == BLEND_MULTIBAND:
                step = int(max(2, round(feather_px / 3.0)))
                # Not under the overlay: it replaces or tints the imagery, and gains
                # fitted to a colour wash would be fitted to the palette.
                corr, gains = self._low_bands(cams, src, H_canvas, H_t, geo, M, canvas_w,
                                              canvas_h, step, limit, max_range, feather_px,
                                              exposure=debug_view == DEBUG_OFF)
                if gains is not None:
                    src = src * gains
                low = (corr, step, S_c)
            pano, covered = self._compose_partition(cams, H_c, geo, M_c, x, y, rois, src,
                                                    limit, max_range, feather_px, sampling,
                                                    low)

        # No mask needed: a pixel nothing covers was sampled from nowhere and is already
        # black. Rounded, not truncated -- a flat patch interpolates to 199.99998 as
        # often as to 200.00002, and truncation turns that into a grey level of noise and
        # half a level of darkening. Bilinear NEAREST and FEATHER are convex combinations
        # of pixels and cannot leave [0, 255]; a bicubic kernel overshoots at an edge, and
        # MULTIBAND adds gains and a correction of either sign.
        coverage_t = covered.float().mean()
        pano = pano.add_(0.5)
        if blend_mode == BLEND_MULTIBAND or sampling == "bicubic":
            pano = pano.clamp_(0.0, 255.0)
        out_u8 = pano.to(torch.uint8).permute(1, 2, 0).contiguous()

        if wire:
            buf = self._out_buffer(out_u8.shape)
            buf.copy_(out_u8, non_blocking=True)
            coverage = float(coverage_t.item())
            if dev.type == "cuda":
                torch.cuda.current_stream().synchronize()
            out = buf.numpy().view(WireReadyPanorama)
        else:
            out = out_u8.cpu().numpy()
            coverage = float(coverage_t.item())

        ms = 1e3 * (time.perf_counter() - started)
        self._render_ms = ms if self._render_ms is None else 0.9 * self._render_ms + 0.1 * ms
        return out, coverage, overlap_psnr

    def _compose_partition(self, cams, H_c, geo, M_c, x, y, rois, src, limit, max_range,
                           feather_px, sampling, low=None):
        """
        NEAREST and MULTIBAND: one view per pixel.  Returns ``(pano [3,H,W], covered)``.

        The winner and where it samples are kept as running bests while the views are
        visited one at a time over their own bounding boxes; the image is then read in a
        single ``grid_sample`` out of all the frames stacked into one tall texture.
        ``low`` is ``(correction, step, S_c)`` from :meth:`_low_bands`, or ``None``.
        """
        dev = src.device
        n, _, src_h, src_w = src.shape
        out_h, out_w = y.shape[0], x.shape[1]
        rows, cols = self._lattice_terms(H_c, x, y)
        prow, pcol = self._plane_terms(M_c, x, y, geo)
        gain = [c["plane_h"] * (self.WEAK_VIEW_PRIORITY if c.get("weak") else 1.0)
                for c in cams]

        best = torch.zeros((out_h, out_w), dtype=torch.float32, device=dev)
        best_uv = torch.zeros((2, out_h, out_w), dtype=torch.float32, device=dev)
        best_uv[0].fill_(self._NOWHERE)
        label = torch.full((out_h, out_w), self._NO_VIEW, dtype=torch.uint8, device=dev)
        for i in range(n):
            y0, y1, x0, x1 = rois[i]
            if y1 <= y0 or x1 <= x0 or cams[i]["plane_h"] <= 1e-9:
                continue
            uv, f, ok = self._view_fields(rows[i, :, :, x0:x1] + cols[i, :, y0:y1],
                                          limit, max_range)
            score = self._score(f, ok, feather_px,
                                torch.rsqrt(prow[i, :, x0:x1] + pcol[i, y0:y1]).mul_(gain[i]))
            held = best[y0:y1, x0:x1]
            better = score > held
            torch.maximum(held, score, out=held)
            label[y0:y1, x0:x1].masked_fill_(better, i)
            held_uv = best_uv[:, y0:y1, x0:x1]
            torch.where(better, uv, held_uv, out=held_uv)
        covered = best > 0

        # Every frame in one texture, one above the other, each with its border repeated
        # as far out as the sampling kernel reaches: a sample on a frame's edge then
        # interpolates with copies of that edge, never with the next frame nor with the
        # black beyond the texture.
        pad = 2 if sampling == "bicubic" else 1
        pitch, wide = src_h + 2 * pad, src_w + 2 * pad
        atlas = F.pad(src, (pad, pad, pad, pad), mode="replicate").permute(1, 0, 2, 3) \
            .reshape(1, 3, n * pitch, wide)
        tile = label.to(torch.float32)
        # float32 throughout. Half precision resolves a normalised coordinate to 1/2048
        # of the image -- a fifth of a source pixel at 800 wide, a third at 1280 -- which
        # is invisible on a canvas that shrinks the feeds and is exactly the detail a
        # full-resolution one is there to keep. With one sample a pixel it buys no time.
        k = 2.0 / (n * pitch - 1)
        best_uv[0].add_(float(pad)).mul_(2.0 / (wide - 1)).sub_(1.0)
        best_uv[1].mul_(k).add_(tile, alpha=pitch * k).add_(pad * k - 1.0)
        pano = F.grid_sample(atlas, best_uv.permute(1, 2, 0).unsqueeze(0), mode=sampling,
                             padding_mode="zeros", align_corners=True)[0]

        if low is not None:
            # pano = winner's pixels + (blended low band - winner's own low band). The
            # bracket is what _low_bands returns per view, on its coarse grid; it is
            # looked up here the way the frames were, by label.
            corr, step, S_c = low
            _, _, low_h, low_w = corr.shape
            low_pitch = low_h + 2
            low_atlas = F.pad(corr, (0, 0, 1, 1), mode="replicate").permute(1, 0, 2, 3) \
                .reshape(1, 3, n * low_pitch, low_w)
            # Canvas pixel of each output column and row, then its place on the coarse
            # grid, whose cell j is centred on canvas pixel (j + 0.5) * step - 0.5.
            col = ((x * float(S_c[0, 0]) + (float(S_c[0, 2]) + 0.5)) / step - 0.5) \
                .clamp_(0.0, low_w - 1.0)
            row = ((y * float(S_c[1, 1]) + (float(S_c[1, 2]) + 0.5)) / step - 0.5) \
                .clamp_(0.0, low_h - 1.0)
            k = 2.0 / (n * low_pitch - 1)
            grid = torch.stack(((col * (2.0 / (low_w - 1)) - 1.0).expand(out_h, out_w),
                                tile.mul_(low_pitch * k).add_(row * k + (k - 1.0))), dim=-1)
            # Zero padding, so a pixel no view covers (its label is past the last tile)
            # stays black here too.
            pano = pano.add_(F.grid_sample(low_atlas, grid.unsqueeze(0), mode="bilinear",
                                           padding_mode="zeros", align_corners=True)[0])
        return pano, covered

    def _compose_feather(self, H_c, x, y, rois, src, limit, max_range, feather_px, sampling):
        """
        FEATHER: every view that covers a pixel, weighted by its distance to its own
        image border.  Returns ``(pano [3,H,W], covered)``.

        A homography's alpha mask has a closed form, so the weight needs no convolution
        at all -- unlike a TPS mesh's, which is why StabStitcher blurs and erodes instead.
        It is measured in SOURCE pixels, so the cross-fade's width in output pixels
        scales with each view's local magnification.  Normalising by the per-pixel weight
        sum is what generalises this to any N; a reference-based or pairwise scheme does
        not.  Accumulated a view at a time over its own bounding box: this is the one
        blend whose cost has to grow with the number of views, since it reads them all.
        """
        dev = src.device
        out_h, out_w = y.shape[0], x.shape[1]
        rows, cols = self._lattice_terms(H_c, x, y)
        to_grid = 2.0 / limit
        acc = torch.zeros((3, out_h, out_w), dtype=torch.float32, device=dev)
        wsum = torch.zeros((out_h, out_w), dtype=torch.float32, device=dev)
        for i in range(src.shape[0]):
            y0, y1, x0, x1 = rois[i]
            if y1 <= y0 or x1 <= x0:
                continue
            uv, f, ok = self._view_fields(rows[i, :, :, x0:x1] + cols[i, :, y0:y1],
                                          limit, max_range)
            weight = torch.where(ok, (f * (1.0 / feather_px)).clamp_(0.0, 1.0), 0.0)
            # Border padding: a bicubic sample just inside the frame reaches a pixel or
            # two outside it, and should find the edge there rather than black. A sample
            # that is itself outside has no weight either way.
            warped = F.grid_sample(src[i:i + 1],
                                   uv.mul_(to_grid).sub_(1.0).permute(1, 2, 0).unsqueeze(0),
                                   mode=sampling, padding_mode="border",
                                   align_corners=True)[0]
            acc[:, y0:y1, x0:x1].addcmul_(warped, weight)
            wsum[y0:y1, x0:x1].add_(weight)
        return acc.div_(wsum.clamp_min(1e-6)), wsum > 1e-6

    # ---- two-band blend and exposure -------------------------------------------------

    # Brown & Lowe's gain compensation, per colour channel. sigma_n is how far two
    # views' mean intensities over a shared patch may differ for reasons other than
    # exposure (0-255); sigma_g how far a gain is expected to sit from 1.
    GAIN_SIGMA_N = 10.0
    GAIN_SIGMA_G = 0.1
    GAIN_LIMITS = (0.6, 1.6)
    # Per-frame low-pass on the gains. Auto-exposure moves over seconds; the estimate
    # moves with every frame's noise.
    GAIN_RATE = 0.15
    _gain_state = None                 # (drone id tuple, gains [N,3] float64)

    def _low_bands(self, cams, src, H_canvas, H_t, geo, M, canvas_w, canvas_h, step, limit,
                   max_range, feather_px, exposure):
        """
        What MULTIBAND adds to the winner-take-all partition.  Returns
        ``(correction [N,3,h,w], gains [N,3,1,1] | None)``.

        Each view is split into a low band -- a blur about ``3 * step`` canvas pixels
        wide -- and the residual.  The low bands are cross-faded with the *blurred*
        partition; the residuals are taken from the winning view alone:

            pano = sum_i W~_i low_i / sum_i W~_i  +  sum_i W_i (I_i - low_i)
                 = I_w + (mix - low_w)                       with w the winner

        Exposure and vignetting differences live in the low band and fade across the
        seam; edges and texture live in the residual and are never averaged, so a
        residual misalignment still cannot double anything.  ``correction`` is the
        bracket for every view; where one view is alone within the blur radius it is
        zero and the mosaic is that view exactly.

        All of it lives on a grid ``step`` canvas pixels coarse, whatever the output
        size, because none of it has finer structure than that.  The views are sampled
        there directly, out of a box-reduced copy of the frames so that one sample
        stands for about the area between it and the next, at four samples a cell: the
        cell's value is their mean and its coverage the fraction that landed in frame.
        The old route to the same grid was to warp every view at full resolution and
        average that down.

        The low band is a *normalised* blur (blur(I * cover) / blur(cover)): a plain one
        would pull every footprint's rim toward the black outside it.

        ``gains`` are the exposure gains (see :meth:`_solve_gains`), already applied to
        the low bands here; the caller applies them to the frames.
        """
        dev = src.device
        n, _, src_h, src_w = src.shape
        low_w = max(2, -(-int(canvas_w) // step))
        low_h = max(2, -(-int(canvas_h) // step))
        half = 0.5 * step
        xs = ((torch.arange(2 * low_w, dtype=torch.float32, device=dev) + 0.5) * half - 0.5) \
            .view(1, -1)
        ys = ((torch.arange(2 * low_h, dtype=torch.float32, device=dev) + 0.5) * half - 0.5) \
            .view(-1, 1)

        rows, cols = self._lattice_terms(H_t, xs, ys)
        uv, f, ok = self._view_fields(rows + cols, limit, max_range)        # [N,2,2h,2w]
        inside = ok & (f >= 0)
        prow, pcol = self._plane_terms(M, xs, ys, geo)
        top, winner = self._score(f, ok, feather_px,
                                  torch.rsqrt(prow + pcol).mul_(geo[:, 3, None, None])).max(dim=0)
        hard = F.one_hot(winner, n).permute(2, 0, 1) * (top > 0)            # [N,2h,2w]

        # How many source pixels lie between two neighbouring samples, and so how far the
        # frames are box-reduced before they are read. A power of two, from the median
        # view, so it only changes with the zoom or the standoff.
        p = H_canvas @ np.array([0.5 * canvas_w, 0.5 * canvas_h, 1.0])      # [N,3]
        with np.errstate(divide="ignore", invalid="ignore"):
            J = (H_canvas[:, :2, :2] - (p[:, :2] / p[:, 2:3])[:, :, None]
                 * H_canvas[:, 2:3, :2]) / p[:, 2, None, None]
            scale = np.sqrt(np.abs(np.linalg.det(J)))
        scale = scale[(p[:, 2] > 0.0) & np.isfinite(scale)]
        spacing = (float(np.median(scale)) if len(scale) else 1.0) * half
        pool = int(2 ** int(np.clip(np.round(np.log2(max(spacing, 1.0))), 0, 4)))
        pool = max(1, min(pool, src_h // 8, src_w // 8))
        small = F.avg_pool2d(src, pool) if pool > 1 else src
        small_h, small_w = small.shape[2:]
        # Reduced pixel p is the mean of source pixels [p * pool, (p + 1) * pool), so
        # source pixel u sits at (u + 0.5) / pool - 0.5 in the reduced frame.
        uv[:, 0].mul_(2.0 / (pool * (small_w - 1))) \
            .add_((0.5 / pool - 0.5) * 2.0 / (small_w - 1) - 1.0)
        uv[:, 1].mul_(2.0 / (pool * (small_h - 1))) \
            .add_((0.5 / pool - 0.5) * 2.0 / (small_h - 1) - 1.0)
        samples = F.grid_sample(small, uv.permute(0, 2, 3, 1), mode="bilinear",
                                padding_mode="border", align_corners=True)  # [N,3,2h,2w]

        gains = None
        if exposure:
            # Compared only where a sample holds no pixel from outside its frame. The
            # pair sums come back in one small copy and the 3 x N x N solve is numpy's:
            # a couple of dozen device ops and a GPU LU for six unknowns cost more.
            whole = (inside & (f >= float(pool))).flatten(1).to(samples.dtype)   # [N,P]
            stats = torch.cat(((whole @ whole.T).unsqueeze(-1),
                               torch.einsum("icp,jp->ijc",
                                            samples.flatten(2) * whole.unsqueeze(1), whole)),
                              dim=-1).cpu().numpy()
            gains = torch.as_tensor(self._solve_gains(cams, stats), dtype=torch.float32,
                                    device=dev).view(n, 3, 1, 1)
            samples = samples * gains

        # One blur for the coverage, the views and the partition: [N, 1 + 3 + 1, ...].
        cover = inside.unsqueeze(1).to(samples.dtype)
        bands = self._binomial_blur(self._binomial_blur(F.avg_pool2d(
            torch.cat((cover, samples * cover, hard.unsqueeze(1).to(samples.dtype)), dim=1), 2)))
        den = bands[:, :1]
        low_img = bands[:, 1:4] / den.clamp_min(1e-3)                      # [N,3,h,w]
        # A view's low band is an extrapolation once most of the blur's support is
        # outside its footprint; fade its say out there rather than trust it.
        soft = bands[:, 4:5] * (den * 2.0).clamp_(0.0, 1.0)
        mix = (low_img * soft).sum(dim=0, keepdim=True) / soft.sum(dim=0, keepdim=True) \
            .clamp_min(1e-4)
        # ... and where next to none of it is inside, the view's own low band is not
        # known well enough to subtract: leave such a pixel as its view drew it.
        return (mix - low_img) * (den * 50.0).clamp_(0.0, 1.0), gains

    def _solve_gains(self, cams, stats):
        """
        One gain per view and channel, ``[N, 3]``, that makes the views agree where they
        overlap.

        ``stats[i, j]`` is ``(N_ij, sum over that overlap of view i's B, G, R)``, with
        ``N_ij`` the number of coarse samples views *i* and *j* both hold.  Minimises,
        per channel,
        ``sum_ij N_ij ((g_i m_ij - g_j m_ji)^2 / sigma_n^2 + (1 - g_i)^2 / sigma_g^2)``
        with ``m_ij`` view *i*'s mean over its overlap with *j*.  The second term is what
        stops the trivial answer (all gains zero) and keeps the whole mosaic from
        drifting dark.
        """
        stats = np.asarray(stats, dtype=np.float64)
        n = stats.shape[0]
        npix = stats[..., 0]
        mean = stats[..., 1:] / np.maximum(npix, 1.0)[:, :, None]     # m_ij per channel
        npix = npix * (1.0 - np.eye(n)) * (npix >= 16.0)

        sn, sg = self.GAIN_SIGMA_N ** -2, self.GAIN_SIGMA_G ** -2
        m_ij = mean.transpose(2, 0, 1)                                 # [3,N,N]
        m_ji = m_ij.transpose(0, 2, 1)
        w = npix[None]
        A = -(w * m_ij * m_ji * sn)
        idx = np.arange(n)
        A[:, idx, idx] += (w * (m_ij * m_ij * sn + sg)).sum(axis=2)
        b = np.broadcast_to((w * sg).sum(axis=2), (3, n)).copy()
        # A view with no overlap at all has an all-zero row; pin its gain to 1.
        lonely = npix.sum(axis=1) <= 0.0
        A[:, idx[lonely], idx[lonely]] += 1.0
        b[:, lonely] += 1.0
        gains = np.clip(np.linalg.solve(A, b[..., None])[..., 0].T, *self.GAIN_LIMITS)

        ids = tuple(c["view"]["drone_id"] for c in cams)
        state = self._gain_state
        if state is not None and state[0] == ids:
            gains = state[1] + self.GAIN_RATE * (gains - state[1])
        self._gain_state = (ids, gains)
        return gains

    _blur_kernels = {}                 # (device, dtype) -> the 5-tap binomial

    @classmethod
    def _binomial_blur(cls, x):
        """Separable 5-tap binomial, replicate-padded.  ``[B,C,h,w]``."""
        key = (str(x.device), x.dtype)
        k = cls._blur_kernels.get(key)
        if k is None:
            k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=x.dtype, device=x.device) / 16.0
            cls._blur_kernels[key] = k
        c = x.shape[1]
        x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"),
                     k.view(1, 1, 1, 5).expand(c, 1, 1, 5), groups=c)
        return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="replicate"),
                        k.view(1, 1, 5, 1).expand(c, 1, 5, 1), groups=c)

    @staticmethod
    def _debug_colours(cams, src, rgb):
        """``[N,3,1,1]`` palette colour of each view, in ``src``'s channel order."""
        table = [debug_colour(c["view"]["drone_id"])[1] for c in cams]       # BGR
        if rgb:
            table = [colour[::-1] for colour in table]
        return torch.tensor(table, dtype=src.dtype, device=src.device).view(len(cams), 3, 1, 1)

    # ---- overlap PSNR ----------------------------------------------------------------

    def _measure_overlap(self, cams, src, H_t, canvas_w, canvas_h, config, max_range):
        """
        The overlap PSNR, measured at most every ``config["psnr_period_s"]`` seconds.

        It is a diagnostic: it reads every view over the whole canvas, which is exactly
        the work the render no longer does, and nothing consumes it faster than the log
        line is printed.  The live stitcher therefore asks for it twice a second.  A
        config without the key -- every tool and test -- measures it on every call, and
        so does one that gates the panorama on it.
        """
        period = float(config.get("psnr_period_s", 0.0) or 0.0)
        ids = tuple(c["view"]["drone_id"] for c in cams)
        now = time.monotonic()
        cached = self._psnr_cache
        if (period > 0.0 and not config.get("psnr_gate", False) and cached is not None
                and cached[0] == ids and now - cached[1] < period):
            return cached[2]
        value = self._overlap_psnr(src, H_t, canvas_w, canvas_h, max_range)
        self._psnr_cache = (ids, now, value)
        return value

    def _overlap_psnr(self, src, H_t, canvas_w, canvas_h, max_range):
        """
        Worst pairwise PSNR over the region two views both cover, on the canvas.

        Measured over geometric *coverage*, not over the blend weights: winner-take-all
        makes the weights one-hot, so a weight-based overlap test would find no
        overlapping pixels at all and silently retire the one diagnostic that quantifies
        pose error.  Two views still see the same ground there whether or not both are
        drawn.  On the canvas rather than the output raster, so the number does not
        move when the output resolution does.
        """
        dev = src.device
        n, _, src_h, src_w = src.shape
        limit = self._frame_limit(src_w, src_h, dev)
        rows, cols = self._lattice_terms(
            H_t,
            torch.arange(canvas_w, dtype=torch.float32, device=dev).view(1, -1),
            torch.arange(canvas_h, dtype=torch.float32, device=dev).view(-1, 1))
        uv, f, ok = self._view_fields(rows + cols, limit, max_range)
        valid = ok & (f >= 0)
        warped = F.grid_sample(src, uv.mul_(2.0 / limit).sub_(1.0).permute(0, 2, 3, 1),
                               mode="bilinear", padding_mode="zeros",
                               align_corners=True)                         # [N,3,Hc,Wc]

        # Every pair's sums stay on the device and come back in one copy: a count and a
        # squared error fetched pair by pair was ninety host syncs a frame at ten views.
        counts, errors = [], []
        for i in range(n):
            for j in range(i + 1, n):
                both = valid[i] & valid[j]
                diff = (warped[i] - warped[j]) * both
                counts.append(both.sum())
                errors.append((diff * diff).sum())
        if not counts:
            return float("inf")
        sums = torch.stack([torch.stack(counts).double(), torch.stack(errors).double()]).cpu()

        worst = float("inf")
        for count, error in zip(sums[0].tolist(), sums[1].tolist()):
            if count < 1024:
                continue
            mse = error / (count * 3)
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
        align = self._align_stats
        if sweep or refine or align:
            print(stamp())
        if align:
            if "skipped" in align:
                print(f"[PLANAR] align idle: {align['skipped']}")
            elif "held" in align:
                print(f"[PLANAR] align holding [{align.get('mode', '?')}]: {align['held']}")
            else:
                ms = align.get("ms", {})
                unsup = align.get("unsupported") or []
                print(f"[PLANAR] align [{align['mode']}, {align.get('matcher', '?')}]: "
                      f"{align['inliers']}/{align['matches']} inliers over "
                      f"{align['pairs']} pairs, median {align['median_px']:.2f} px | "
                      f"worst correction {align['worst_shift']:.2f} m / "
                      f"{align['worst_rot_deg']:.2f} deg | plane tilt "
                      f"{align['plane_tilt_deg']:.1f} deg, offset "
                      f"{align['plane_offset']:+.2f} m | {ms.get('total', 0.0):.0f} ms"
                      + (f" | {align['far']} far-field matches set aside"
                         if align.get("far") else "")
                      + (f" | no support: drones {unsup}" if unsup else "")
                      + (f" | {align['note']}" if align.get("note") else ""))
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
        print(f"{stamp()} [PLANAR] {s.get('views', 0)} views "
              f"(+{s.get('dropped', 0)} dropped{unposed_txt}) | blend {s.get('blend', '?')} | "
              f"coverage {s.get('coverage', 0):.0%} | "
              f"mean range {s.get('mean_range', 0):.1f} m | "
              f"max anisotropy {s.get('max_aniso', 0):.2f} | overlap PSNR {psnr_txt}"
              # Only in FormationRelative, where the plane is derived here rather than
              # published. Which rule won matters: "forward" means the formation collapsed
              # to a row and the normal now follows the gimbal.
              + (f" | plane from {self._plane_source}" if self._plane_source else ""))

        # Framing, and whether the output resolves the feeds. Printed as the canvas extent
        # in metres rather than as metres-per-pixel because the extent is the thing an
        # operator can check against the scene in front of them. The detail figure is
        # output pixels per source pixel of the sharpest view: below 1 the mosaic is
        # coarser than the feeds it is made of, which is said in so many words because no
        # other number on this line would reveal it; "empty" is the other end, a zoom
        # past the point where it is interpolating source pixels, not resolving new ones.
        mpp_out = s.get("mpp_out", 0.0)
        if mpp_out > 0.0:
            gsd = s.get("best_gsd", 0.0)
            zoom = s.get("zoom", 1.0)
            ext_a, ext_b = s.get("canvas_extent", (0.0, 0.0))
            out_w, out_h = s.get("out_size", (0, 0))
            detail = gsd / mpp_out if gsd > 0.0 else 0.0
            verdict = ""
            if 0.0 < detail < 0.95:
                verdict = (f" (BELOW feed resolution: raise planarRenderScale "
                           f"by {1.0 / detail:.1f}x, or zoom in)")
            elif zoom > 1.0 + 1e-3 and detail > 2.0:
                verdict = " (EMPTY magnification)"
            times = ""
            if self._frame_ms is not None and self._render_ms is not None:
                times = f" | {self._frame_ms:.1f} ms a frame (render {self._render_ms:.1f})"
            print(f"[PLANAR] canvas [{s.get('canvas_mode', '?')}] "
                  f"{ext_a:.1f} x {ext_b:.1f} m drawn at {out_w}x{out_h}, "
                  f"{mpp_out * 100.0:.1f} cm/px"
                  + (f" | zoom {zoom:.2f}x" if abs(zoom - 1.0) > 1e-3 else "")
                  + (f" | source GSD {gsd * 100.0:.1f} cm/px, {detail:.1f} px per "
                     f"source px{verdict}" if gsd > 0.0 else "")
                  + times)

        # A colour map is useless without the key, and the selection changes as drones
        # join, die or fall out of range -- so reprint it alongside the stats rather
        # than once at startup.
        if s.get("debug_view", DEBUG_OFF) != DEBUG_OFF:
            legend = "  ".join(f"drone {i} = {debug_colour(i)[0]}"
                               for i in s.get("drone_ids", []))
            mode = "flat" if s["debug_view"] == DEBUG_FLAT else "tint"
            print(f"[PLANAR] debug view ({mode}):  {legend}")
