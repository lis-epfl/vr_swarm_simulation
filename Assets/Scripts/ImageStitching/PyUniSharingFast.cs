using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PyUniSharingFast : MonoBehaviour
{   
    // ------------------------------------------------------------------
    // INSPECTOR LAYOUT
    //
    // The settings that actually change between runs sit flat at the top;
    // everything else is bucketed into the [Serializable] groups below, which
    // Unity draws as one collapsible row each. Most groups apply to exactly
    // one `typeOfStitcher`, so on any given run most of them are inert.
    //
    // This is presentation only. The metadata wire layout is untouched -- the
    // grouped fields are still reached by their original names through the
    // forwarding properties further down, so WriteMetadata and every call site
    // are unchanged and tools/check_wire_layout.py still holds.
    // ------------------------------------------------------------------

    [Tooltip("Which stitcher Python runs. This picks which of the grouped settings below " +
             "are live: Classic, StabStitch and Planar are mutually exclusive.")]
    [SerializeField]
    private stitcherType typeOfStitcher = stitcherType.CLASSIC;

    [Tooltip("Which camera pose is published to the planar stitcher. GroundTruth is exact, " +
             "so with it the mosaic should be pixel-perfect on a truly planar scene -- any " +
             "seam is a bug rather than a limitation. The noisy modes exist to size how much " +
             "refinement real drones would need. PLANAR only.")]
    [SerializeField]
    private StitchPoseSourceMode poseSource = StitchPoseSourceMode.GroundTruth;

    [Tooltip("Which surface the planar stitcher treats as the scene plane. Auto picks Nadir " +
             "when the gimbal is pitched down and Facade otherwise. PLANAR only.")]
    [SerializeField]
    private ScenePlaneMode scenePlaneMode = ScenePlaneMode.Auto;

    [Tooltip("How overlapping views are combined. Nearest gives every pixel to the single " +
             "most face-on view, so nothing is averaged and pose error shows as a seam " +
             "rather than ghosting; Feather cross-fades the overlap, which is only clean " +
             "while the geometry is exact. In a facade wall the views overlap almost " +
             "everywhere, so Feather ghosts across the whole mosaic. PLANAR only.")]
    [SerializeField]
    private PlanarBlendMode planarBlendMode = PlanarBlendMode.Nearest;

    [Tooltip("Testing aid: colour each patch of the mosaic by the drone that supplied it. " +
             "Tint washes colour over the imagery, Flat replaces it entirely. The " +
             "drone-to-colour key is printed on the Python console's periodic [PLANAR] " +
             "line. Leave Off for flight. PLANAR only.")]
    [SerializeField]
    private PlanarDebugView planarDebugView = PlanarDebugView.Off;

    [Tooltip("Show the stitched panorama; unticked shows the individual drone feeds instead. " +
             "Mirrors the controller's click switch when one is connected, but can also be " +
             "toggled directly here or with vr.togglePanoramaKey -- for testing without a controller.")]
    [SerializeField]
    private bool panoramaUserEnabled = true;

    [Tooltip("Hide the individual ScreenSpawn feeds for the drones currently being stitched into " +
             "the panorama (they already appear in the panorama). Only applies while the panorama " +
             "is displayed; during quality-fallback all feeds reappear.")]
    [SerializeField]
    private bool hideStitchedDroneScreens = false;

    [Tooltip("Print the Python stitch/warp loop rate (Hz) to the console. Disable to declutter " +
             "the log while reading other per-frame diagnostics (e.g. the StabStitch quality PSNR).")]
    [SerializeField]
    private bool printStitchRate = true;

    // ---------------- grouped settings ----------------

    [Space(6)]
    [Tooltip("Shared-memory bridge sizing and pacing. Every field here is consumed once in " +
             "Start() to size a named section, so changes need a Play restart AND a Python restart.")]
    [SerializeField]
    private BridgeSettings bridge = new BridgeSettings();

    [Tooltip("PLANAR stitcher settings: scene-plane raycast, canvas framing, sanity gates and " +
             "the warp-thread estimators. Inert unless typeOfStitcher is PLANAR.")]
    [SerializeField]
    private PlanarSettings planar = new PlanarSettings();

    [Tooltip("STABSTITCH settings: fusion mode, REFERENCE_BLEND feathering and the panorama " +
             "quality fallback. Inert unless typeOfStitcher is STABSTITCH.")]
    [SerializeField]
    private StabStitchSettings stabStitch = new StabStitchSettings();

    [Tooltip("CLASSIC feature-matching settings. Inert under STABSTITCH and PLANAR, " +
             "which do not do feature matching at all.")]
    [SerializeField]
    private ClassicSettings classic = new ClassicSettings();

    [Tooltip("Curved panorama screen geometry, the HMD transform and the body-yaw controls.")]
    [SerializeField]
    private VrDisplaySettings vr = new VrDisplaySettings();

    [Tooltip("GNSS error model. Only used when poseSource is GroundTruthPlusGnss.")]
    [SerializeField]
    private StitchPoseSource.Settings gnssSettings = new StitchPoseSource.Settings();

    // ---------------- group definitions ----------------

    /// <summary>
    /// Shared-memory bridge sizing + pacing. These size the named sections in
    /// <c>Start</c> and are published in metadata for Python to size its own mappings
    /// from, so changing any of them requires restarting both Play mode and
    /// StitcherThreading.py.
    /// </summary>
    [Serializable]
    public class BridgeSettings
    {
        [Tooltip("Enable writing images to BlockSharedMemory. Must be OFF in the DJI scene: " +
                 "ImageSharing.cs is the producer there, and two producers with different " +
                 "header sizes would corrupt the map.")]
        public bool enableImageWriting = true;

        [Tooltip("Enable reading panorama from PanoramaSharedMemory")]
        public bool enablePanoramaReading = true;

        [Tooltip("Must be 800 in the DJI scene to match the 800x450 drone feed (image_stream_feed.py / ImageSharing.cs); StitcherThreading.py sizes itself from this via the metadata map.")]
        public int blockImageWidth = 800;

        [Tooltip("Must be 450 in the DJI scene to match the 800x450 drone feed (image_stream_feed.py / ImageSharing.cs); StitcherThreading.py sizes itself from this via the metadata map.")]
        public int blockImageHeight = 450;

        [Tooltip("Panorama width Python renders and Unity uploads to the curved screen.")]
        public int panoramaImageWidth = 600;

        [Tooltip("Panorama height Python renders and Unity uploads to the curved screen.")]
        public int panoramaImageHeight = 400;

        [Tooltip("Seconds between block publishes. StitcherThreading.py's RENDER_MIN_PERIOD is " +
                 "paced just above this; lowering it here without matching that starves the warp thread.")]
        public float sendInterval = 0.05f;

        [Tooltip("Seconds between panorama reads.")]
        public float readInterval = 0.05f;

        [Range(3, 12)]
        [Tooltip("Block slots published to Python in PLANAR mode. A planar mosaic generalises " +
                 "to any number of overlapping views, unlike the fixed left/centre/right triple " +
                 "the other stitchers need (they stay pinned to 3 regardless of this). The block " +
                 "map is sized from this once at startup, so restart the Python stitcher after " +
                 "changing it.")]
        public int maxStitchViews = 8;
    }

    /// <summary>
    /// CLASSIC feature-matching parameters. Published in metadata unconditionally
    /// (PlanarStitcher.py reads them off the active stitcher regardless), but they only
    /// affect the output under the CLASSIC stitcher.
    /// </summary>
    [Serializable]
    public class ClassicSettings
    {
        [Tooltip("Warp inputs to cylindrical coordinates before matching.")]
        public bool cylindrical = false;

        [Tooltip("Descriptor matcher: brute force or FLANN.")]
        public matcherType typeOfMatcher = matcherType.BF;

        [Tooltip("Refine the homography with RANSAC.")]
        public bool ransac = false;

        [Tooltip("FLANN search checks.")]
        public int checks = 50;

        [Tooltip("Lowe ratio test threshold.")]
        public float ratio_thresh = 0.7f;

        [Tooltip("Minimum keypoint score to keep a match.")]
        public float score_threshold = 0.1f;

        [Tooltip("Assumed focal length in pixels, used by the cylindrical warp.")]
        public int focal_length = 1000;
    }

    /// <summary>
    /// STABSTITCH-only settings: how the warped views are fused, how the REFERENCE_BLEND
    /// soft mask is feathered, and the quality fallback that hides a bad panorama.
    /// </summary>
    [Serializable]
    public class StabStitchSettings
    {
        [Tooltip("How warped views are fused into the panorama.")]
        public FusionMode typeOfFusion = FusionMode.REFERENCE_BLEND;

        [Tooltip("Gaussian blur kernel size for the reference-image soft mask (must be an odd integer)")]
        public int blurKernelSize = 41;

        [Tooltip("Gaussian blur sigma for the reference-image soft mask feathering width (pixels)")]
        public float blurSigma = 15f;

        [Tooltip("Width in pixels of the edge strip where LINEAR blending is applied to hide seams (REFERENCE_BLEND mode only). Interior of the reference image is left pixel-perfect.")]
        public int borderSize = 60;

        [Tooltip("When the panorama is judged bad (poor alignment / distorted warp), hide it and show the individual drone feeds (via ScreenSpawn) instead.")]
        public bool qualityFallbackEnabled = true;

        [Tooltip("Minimum overlap PSNR (dB) for the panorama to be considered good. Higher = stricter (falls back to feeds more readily).")]
        public float qualityThreshold = 18f;

        [Tooltip("ScreenSpawn style used to display the individual drone feeds while the panorama is in fallback.")]
        public ScreenSpawn.ScreenStyle fallbackScreenStyle = ScreenSpawn.ScreenStyle.OUTER_CIRCLE;
    }

    /// <summary>
    /// PLANAR-only settings. The mode selector, blend mode and debug overlay stay at the
    /// top of the inspector because they are flipped between runs; everything here is
    /// tuning that is set once for a scene.
    /// </summary>
    [Serializable]
    public class PlanarSettings
    {
        [Header("Scene plane")]
        [Tooltip("Layers the scene-plane raycast may hit. This matters more than it looks: the " +
                 "ScreenSpawn feed quads and this component's own curved panorama screen float " +
                 "in world space near the pilot, and an unfiltered raycast will happily return " +
                 "one of them, putting the 'scene plane' a few metres away. Obstacle + Default " +
                 "is the intended setting.")]
        public LayerMask scenePlaneMask = ~0;

        [Tooltip("Plane distance used when the raycast misses. A wrong distance is a uniform " +
                 "scale error and degrades gracefully, so this is preferable to blanking the " +
                 "panorama every time the ray clips a window.")]
        public float fallbackPlaneDistance = 30f;

        [Tooltip("Smoothing time constant for the plane normal and distance, so a car driving " +
                 "through the ray or a one-frame miss doesn't jerk the mosaic.")]
        public float planeFilterTime = 0.5f;

        [Tooltip("Facade mode: snap the plane normal to the swarm formation normal rather than " +
                 "trusting the raycast hit normal. Useful when the facade has ledges or mullions " +
                 "that make the hit normal flicker.")]
        public bool snapNormalToFormation = false;

        [Tooltip("Plane normal used when scenePlaneMode is Manual.")]
        public Vector3 manualPlaneNormal = Vector3.up;

        [Tooltip("Plane offset used when scenePlaneMode is Manual.")]
        public float manualPlaneDistance = 0f;

        [Header("Canvas & gates")]
        public int planarCanvasWidth = 1200;
        public int planarCanvasHeight = 800;

        [Tooltip("Fixed canvas scale. The output resolution never changes per solve (that would " +
                 "flicker), so this sets how much of the plane fits in it. Fixed rather than " +
                 "auto-fitted so measurements stay comparable across runs.")]
        public float planarMetresPerPixel = 0.05f;

        [Tooltip("Rays landing beyond this distance are rejected. Without it an oblique view's " +
                 "footprint is unbounded whenever the horizon is in frame.")]
        public float planarMaxRange = 200f;

        [Range(10f, 89f)]
        [Tooltip("Drop views whose optical axis meets the plane at a shallower angle than this " +
                 "(measured from the plane normal). A grazing view contributes a long thin " +
                 "smear and trips the anisotropy gate anyway.")]
        public float maxObliquityDeg = 70f;

        [Tooltip("Centre-drone hysteresis in metres. The incumbent centre drone is kept until " +
                 "another is closer to the formation centroid by more than this. The scene-plane " +
                 "raycast is cast from the centre drone, so a swap steps the published plane " +
                 "offset and visibly shifts the mosaic — worth resisting near a tie.")]
        public float planarCentreHysteresis = 1.5f;

        [Tooltip("Border falloff width, in SOURCE pixels — so the width in canvas pixels " +
                 "scales with each view's local magnification. Under Feather this is the " +
                 "cross-fade width; under Nearest it only keeps the seam off the source-image " +
                 "edges, by making a view lose to a neighbour before it runs out of frame.")]
        public int planarFeatherPx = 40;

        [Tooltip("Projective-sanity gate: worst local stretch ratio allowed across the canvas. " +
                 "1.0 is an isotropic similarity; near-horizon views blow up.")]
        public float planarAnisoMax = 12f;

        [Tooltip("Minimum fraction of the canvas that must be covered by some view.")]
        public float planarMinCoverage = 0.35f;

        [Tooltip("Gate the panorama on overlap PSNR. Off by default: it is logged as a " +
                 "diagnostic either way, and gating on it would hide the panorama permanently " +
                 "once pose noise is injected, which defeats the point of injecting it.")]
        public bool planarPsnrGateEnabled = false;

        [Header("Warp-thread estimators")]
        [Tooltip("Plane sweep: correct the published plane DISTANCE by scanning candidate " +
                 "offsets and keeping the one where the views agree best. Fixes the one error " +
                 "no per-view translation can absorb — a wrong plane distance appears in each " +
                 "view as a scale about its own footprint. Being a single global number it " +
                 "cannot fix per-drone error; that is what the pose refiner is for. " +
                 "Independent of it: either, both or neither may be on.")]
        public bool planarPlaneSweep = false;

        [Tooltip("Pose refiner: correct each drone's camera position by phase-correlating its " +
                 "view against the consensus of the others. Absorbs differential GNSS error " +
                 "and compass bias, which both show up at a facade as a lateral shift of that " +
                 "view's footprint. Cannot represent scale (use the plane sweep) or the " +
                 "keystone from a gimbal-pitch bias. Views whose correlation peak is not " +
                 "clearly dominant are left uncorrected rather than guessed at.")]
        public bool planarPoseRefine = false;

        [Range(0.5f, 50f)]
        [Tooltip("Plane sweep half-range, metres either side of the current estimate. Size it " +
                 "to how wrong the plane could plausibly be — a map-drawn facade is worth a few " +
                 "metres, a raycast onto real geometry much less. Too wide wastes candidates; " +
                 "too narrow and the sweep cannot reach the answer.")]
        public float planarSweepRange = 4f;

        [Range(3, 21)]
        [Tooltip("Plane sweep candidate count. Forced odd in Python so the incumbent estimate " +
                 "is always itself a candidate; the result is parabola-refined between samples, " +
                 "so this sets the capture range's resolution rather than the final precision.")]
        public int planarSweepSteps = 9;

        [Range(0.01f, 1f)]
        [Tooltip("Low-pass rate applied to both estimators, per warp update. The corrections " +
                 "are slowly-varying by premise (GNSS bias, plane distance), so heavy smoothing " +
                 "costs nothing and keeps a single bad measurement from reaching the mosaic. " +
                 "1.0 snaps to the raw estimate — useful for seeing what it actually measured.")]
        public float planarRefineRate = 0.25f;

        [Tooltip("Largest per-view correction the refiner may apply, metres. A correlation " +
                 "peak further out than the pose could plausibly be wrong is a mismatch, not a " +
                 "measurement. 0 disables the clamp.")]
        public float planarRefineMaxShift = 3f;
    }

    /// <summary>
    /// Curved panorama screen geometry plus the headset-direction controls. This is the
    /// VR display side of the component and is independent of which stitcher is running.
    /// </summary>
    [Serializable]
    public class VrDisplaySettings
    {
        [Header("Curved screen")]
        [Tooltip("Curved screen radius from the pilot, metres.")]
        public float radius = 5f;

        [Tooltip("Horizontal arc the curved screen subtends, degrees.")]
        public float angleRange = 90f;

        [Tooltip("Mesh segments across the arc.")]
        public int segments = 20;

        [Tooltip("Curved screen height, metres.")]
        public float height = 3f;

        [Tooltip("Re-derive the screen aspect from the panorama dimensions each time they change.")]
        public bool resize_dimension = false;

        [Tooltip("Vertical offset of the curved screen above the Arena centre.")]
        public float screenHeightOffset = 0f;

        [Header("Headset direction")]
        [Tooltip("HMD head transform (OVRCameraRig.centerEyeAnchor). Auto-found from the OVRPlayerController if left empty.")]
        public Transform headTransform;

        [Tooltip("Degrees/second the body yaws at full controller yaw-stick deflection. The panorama " +
                 "heading integrates this command and the OVRCameraRig is rotated by the same amount " +
                 "to mimic body motion. Head tracking is excluded, so the pilot can look around at the " +
                 "side screens without moving the panorama.")]
        public float bodyYawRate = 90f;

        [Tooltip("Rotate the OVRCameraRig by the controller yaw-rate command to mimic body motion. " +
                 "Disable to advance the panorama heading only, leaving the rig untouched.")]
        public bool driveCameraRigYaw = true;

        [Tooltip("Key that recalibrates the body heading to the current CenterEyeAnchor yaw, so the " +
                 "panorama centre and the VR velocity frame re-align with wherever the pilot is looking.")]
        public KeyCode calibrateKey = KeyCode.C;

        [Tooltip("Key that toggles the panorama on/off (for testing without a controller connected).")]
        public KeyCode togglePanoramaKey = KeyCode.T;

        [Tooltip("ScreenSpawn that shows the individual drone feeds. Auto-found if left empty.")]
        public ScreenSpawn screenSpawn;
    }

    // ---------------- forwarding properties ----------------
    //
    // Every grouped field keeps its original name here, so the ~2500 lines below this
    // block -- WriteMetadata's offset writes above all -- are untouched by the grouping.
    // Read/write throughout because a handful (the image dimensions, screenSpawn) are
    // clamped or auto-found at runtime.

    private bool enableImageWriting { get => bridge.enableImageWriting; set => bridge.enableImageWriting = value; }
    private bool enablePanoramaReading { get => bridge.enablePanoramaReading; set => bridge.enablePanoramaReading = value; }
    private int blockImageWidth { get => bridge.blockImageWidth; set => bridge.blockImageWidth = value; }
    private int blockImageHeight { get => bridge.blockImageHeight; set => bridge.blockImageHeight = value; }
    private int panoramaImageWidth { get => bridge.panoramaImageWidth; set => bridge.panoramaImageWidth = value; }
    private int panoramaImageHeight { get => bridge.panoramaImageHeight; set => bridge.panoramaImageHeight = value; }
    private float sendInterval { get => bridge.sendInterval; set => bridge.sendInterval = value; }
    private float readInterval { get => bridge.readInterval; set => bridge.readInterval = value; }
    private int maxStitchViews { get => bridge.maxStitchViews; set => bridge.maxStitchViews = value; }

    private bool cylindrical { get => classic.cylindrical; set => classic.cylindrical = value; }
    private matcherType typeOfMatcher { get => classic.typeOfMatcher; set => classic.typeOfMatcher = value; }
    private bool ransac { get => classic.ransac; set => classic.ransac = value; }
    private int checks { get => classic.checks; set => classic.checks = value; }
    private float ratio_thresh { get => classic.ratio_thresh; set => classic.ratio_thresh = value; }
    private float score_threshold { get => classic.score_threshold; set => classic.score_threshold = value; }
    private int focal_length { get => classic.focal_length; set => classic.focal_length = value; }

    private FusionMode typeOfFusion { get => stabStitch.typeOfFusion; set => stabStitch.typeOfFusion = value; }
    private int blurKernelSize { get => stabStitch.blurKernelSize; set => stabStitch.blurKernelSize = value; }
    private float blurSigma { get => stabStitch.blurSigma; set => stabStitch.blurSigma = value; }
    private int borderSize { get => stabStitch.borderSize; set => stabStitch.borderSize = value; }
    private bool qualityFallbackEnabled { get => stabStitch.qualityFallbackEnabled; set => stabStitch.qualityFallbackEnabled = value; }
    private float qualityThreshold { get => stabStitch.qualityThreshold; set => stabStitch.qualityThreshold = value; }
    private ScreenSpawn.ScreenStyle fallbackScreenStyle { get => stabStitch.fallbackScreenStyle; set => stabStitch.fallbackScreenStyle = value; }

    private LayerMask scenePlaneMask { get => planar.scenePlaneMask; set => planar.scenePlaneMask = value; }
    private float fallbackPlaneDistance { get => planar.fallbackPlaneDistance; set => planar.fallbackPlaneDistance = value; }
    private float planeFilterTime { get => planar.planeFilterTime; set => planar.planeFilterTime = value; }
    private bool snapNormalToFormation { get => planar.snapNormalToFormation; set => planar.snapNormalToFormation = value; }
    private Vector3 manualPlaneNormal { get => planar.manualPlaneNormal; set => planar.manualPlaneNormal = value; }
    private float manualPlaneDistance { get => planar.manualPlaneDistance; set => planar.manualPlaneDistance = value; }
    private int planarCanvasWidth { get => planar.planarCanvasWidth; set => planar.planarCanvasWidth = value; }
    private int planarCanvasHeight { get => planar.planarCanvasHeight; set => planar.planarCanvasHeight = value; }
    private float planarMetresPerPixel { get => planar.planarMetresPerPixel; set => planar.planarMetresPerPixel = value; }
    private float planarMaxRange { get => planar.planarMaxRange; set => planar.planarMaxRange = value; }
    private float maxObliquityDeg { get => planar.maxObliquityDeg; set => planar.maxObliquityDeg = value; }
    private float planarCentreHysteresis { get => planar.planarCentreHysteresis; set => planar.planarCentreHysteresis = value; }
    private int planarFeatherPx { get => planar.planarFeatherPx; set => planar.planarFeatherPx = value; }
    private float planarAnisoMax { get => planar.planarAnisoMax; set => planar.planarAnisoMax = value; }
    private float planarMinCoverage { get => planar.planarMinCoverage; set => planar.planarMinCoverage = value; }
    private bool planarPsnrGateEnabled { get => planar.planarPsnrGateEnabled; set => planar.planarPsnrGateEnabled = value; }
    private bool planarPlaneSweep { get => planar.planarPlaneSweep; set => planar.planarPlaneSweep = value; }
    private bool planarPoseRefine { get => planar.planarPoseRefine; set => planar.planarPoseRefine = value; }
    private float planarSweepRange { get => planar.planarSweepRange; set => planar.planarSweepRange = value; }
    private int planarSweepSteps { get => planar.planarSweepSteps; set => planar.planarSweepSteps = value; }
    private float planarRefineRate { get => planar.planarRefineRate; set => planar.planarRefineRate = value; }
    private float planarRefineMaxShift { get => planar.planarRefineMaxShift; set => planar.planarRefineMaxShift = value; }

    private float radius { get => vr.radius; set => vr.radius = value; }
    private float angleRange { get => vr.angleRange; set => vr.angleRange = value; }
    private int segments { get => vr.segments; set => vr.segments = value; }
    private float height { get => vr.height; set => vr.height = value; }
    private float screenHeightOffset { get => vr.screenHeightOffset; set => vr.screenHeightOffset = value; }
    private bool resize_dimension { get => vr.resize_dimension; set => vr.resize_dimension = value; }
    private Transform headTransform { get => vr.headTransform; set => vr.headTransform = value; }
    private float bodyYawRate { get => vr.bodyYawRate; set => vr.bodyYawRate = value; }
    private bool driveCameraRigYaw { get => vr.driveCameraRigYaw; set => vr.driveCameraRigYaw = value; }
    private KeyCode calibrateKey { get => vr.calibrateKey; set => vr.calibrateKey = value; }
    private KeyCode togglePanoramaKey { get => vr.togglePanoramaKey; set => vr.togglePanoramaKey = value; }
    private ScreenSpawn screenSpawn { get => vr.screenSpawn; set => vr.screenSpawn = value; }

    // Read-only access so ScreenSpawn can adopt the same FPV feed resolution
    // (PyUniSharingFast is the single source of truth for the block resolution).
    public int BlockImageWidth => bridge.blockImageWidth;
    public int BlockImageHeight => bridge.blockImageHeight;

    private string blockMapName = "BlockSharedMemory";
    private int blockImageCount = 0;
    private int blockImageSize = 0;   // bytes per drone image (W*H*3)
    private int blockSize = 0;        // per-drone block: header + image
    private int totalBlockSize = 0;   // blockImageCount * blockSize

    private string panoramaMapName = "PanoramaSharedMemory";
    private int panoramaImageSize = 0;
    private int totalPanoramaSize = 0;

    private string metadataMapName = "MetadataSharedMemory";
    // Total bytes WriteMetadata actually writes, including the trailing reserved gap.
    // Must equal metadataTailEnd + metadataReservedGap + 8 and must match StitcherThreading.py's
    // METADATA_SIZE, or the two processes request different section sizes.
    // (This used to read 261 while the code wrote through 325; it only survived
    // because Windows rounds a section up to a 4 KB page.)
    private const int metadataSize = 412;

    private IntPtr blockFileMap;
    private IntPtr blockPtr;
    private IntPtr panoramaFileMap;
    private IntPtr panoramaPtr;

    private IntPtr metadataFileMap;
    private IntPtr metadataPtr;

    // ---------------- debug readouts ----------------
    // Populated by FindCameras / the stitch selection, not configuration. Kept visible
    // (rather than [HideInInspector]) because they are the quickest way to confirm which
    // drones were discovered and which of them are currently going to the stitcher.

    [Header("Debug — populated at runtime")]
    [Tooltip("Read-only: every FPV camera discovered in the scene. Rebuilt every few seconds; " +
             "the block header's droneId is an index into this list.")]
    public List<Camera> camerasToCapture;

    [Tooltip("Read-only: per-camera boundary flag, index-aligned with camerasToCapture.")]
    public List<bool> camerasToStitch;

    [SerializeField]
    [Tooltip("Read-only: the drones currently sent to the stitcher, ordered left / centre / right. Updates during Play.")]
    private List<string> stitchedDrones = new List<string>();
    private List<AttitudeAlgorithm> stitchAttitudes;  // per-camera boundary estimator, index-aligned with camerasToCapture
    private List<StateFinder> stitchStates;            // per-camera state (IsAlive), index-aligned with camerasToCapture

    private RenderTexture reusableTexture;
    private Texture2D image;
    private byte[] blockImageBytes;   // reusable scratch for one converted drone image
    private float nextSendTime, nextReceiveTime = 0f;

    // Async capture: block images are read back from the GPU with
    // AsyncGPUReadback (no ReadPixels stall) and converted RGBA->BGR by a Burst
    // job in the completion callback. Each pending entry carries the slot /
    // droneId / heading recorded at request time (1-2 frames before completion).
    private struct PendingReadback
    {
        public bool inUse;
        public int slot;               // block slot [left, centre, right]
        public int droneId;            // camerasToCapture index
        public float heading;          // camera yaw at request time
        // Full camera pose, snapshotted alongside heading at request time for the same
        // reason: by completion the drone has moved. At 20 Hz and 1-3 m/s that is
        // 5-15 cm, the same order as the alignment a planar homography is chasing, and
        // it is correlated with drone motion so it would read as real misalignment.
        public Vector3 camPos;
        public Quaternion camRot;
        public float captureTime;
        public int poseStatus;
        public byte[] verifyReference; // sync-captured reference, set only during row-order calibration
    }
    private PendingReadback[] pendingReadbacks;
    private Action<AsyncGPUReadbackRequest>[] pendingCallbacks;  // one cached delegate per pool entry
    private const float readbackPoolWarnInterval = 5f;           // throttle for the exhaustion warning
    private float nextReadbackPoolWarnTime = 0f;
    private NativeArray<byte> convertedBlock;   // reusable BGR24 output of the conversion job
    // AsyncGPUReadback returns rows in the GPU-native order, which differs per
    // graphics API from the bottom-up order ReadPixels gives. The first readback
    // is compared against a synchronous capture of the same RT contents to pick
    // the flip that reproduces the existing top-down BGR block format exactly.
    private bool readbackFlipRows;
    private bool readbackFlipCalibrated = false;
    private bool readbackFlipCalibrating = false;

    // Row-wise RGBA32 -> BGR24 conversion (optionally flipping row order) into
    // the top-down BGR block layout the Python stitcher consumes.
    [BurstCompile]
    private struct ConvertRgbaToBgrJob : IJobParallelFor
    {
        [ReadOnly, NativeDisableParallelForRestriction] public NativeArray<byte> src;  // RGBA32
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<byte> dst; // BGR24
        public int width;
        public int height;
        public bool flipRows;

        public void Execute(int y)
        {
            int srcRow = y * width * 4;
            int dstRow = (flipRows ? height - 1 - y : y) * width * 3;
            for (int x = 0; x < width; x++)
            {
                int s = srcRow + x * 4;
                int d = dstRow + x * 3;
                dst[d]     = src[s + 2];  // B
                dst[d + 1] = src[s + 1];  // G
                dst[d + 2] = src[s];      // R
            }
        }
    }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr CreateFileMapping(IntPtr hFile, IntPtr lpFileMappingAttributes, uint flProtect, uint dwMaximumSizeHigh, uint dwMaximumSizeLow, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint dwDesiredAccess, uint dwFileOffsetHigh, uint dwFileOffsetLow, UIntPtr dwNumberOfBytesToMap);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr OpenFileMapping(uint dwDesiredAccess, bool bInheritHandle, string lpName);

    public enum stitcherType
    {
        // The values are pinned rather than sequential because Unity serializes the enum
        // as a plain int: the scenes hold 4 (STABSTITCH) and 5 (PLANAR), so closing the
        // gap left by the retired UDIS(1)/NIS(2)/REWARP(3) options would silently repoint
        // every scene at a different stitcher. Python matches on the name, not the value.
        CLASSIC = 0,
        STABSTITCH = 4,
        // Pose-initialized planar homography. For the vertical-plane and nadir
        // configurations, where the scene is one dominant plane and a single
        // homography per view is exact. Needs camera pose, so it requires the v2
        // block header and is unavailable in the DJI scene (real drones publish
        // yaw only).
        PLANAR = 5
    }

    public enum matcherType
    {
        BF,
        FLANN
    }

    public enum FusionMode
    {
        REFERENCE_BLEND,
        REFERENCE,
        EDGE_BLEND,
        LINEAR,
        AVERAGE
    }
    
    private bool hasStarted = false;

    // Constant values
    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    private const int FlagPosition = 0;             // panorama flag (int32) at region start
    private const int panoramaQualityPosition = 4;  // packed quality word (int32) — see masks below
    private const int panoramaDataPosition = 8;      // RGB24 panorama data

    // Packed quality word (PanoramaSharedMemory [4:8]), written by Python's
    // write_panorama_memory. bit 0 = panorama good; bits 1-3 = failing-gate
    // reason (only set when bit 0 == 0). Keep in sync with StitcherThreading.py.
    private const int QUALITY_OK_BIT = 1 << 0;       // panorama good (show panorama)
    private const int REASON_CANVAS = 1 << 1;        // degenerate warp canvas size
    private const int REASON_DISTORTION = 1 << 2;    // folded/torn mesh (inter-grid loss)
    private const int REASON_PHOTOMETRIC = 1 << 3;   // overlap PSNR below threshold
    private const int REASON_NO_OVERLAP = 1 << 4;    // selected cameras' yaw gap exceeds FOV (pre-stitch gate)
    private const int REASON_TOO_FEW_IMAGES = 1 << 5; // fewer than 3 selected feeds (pre-stitch gate)
    private const int REASON_PLANE_INVALID = 1 << 6; // planar: no usable scene plane or pose on the wire
    // Per-drone block layout. Two versions exist; Python picks between them from the
    // blockHeaderSize field in metadata, so both producers can coexist:
    //
    //   v1 (12 bytes, legacy — also what ImageSharing.cs writes in the DJI scene):
    //     int32 flag | int32 droneId | float32 heading | RGB24 image
    //   v2 (48 bytes, pose-carrying — required by the PLANAR stitcher):
    //     ... | float32 camPos[3] | float32 camRot[4] (xyzw) | float32 captureTime
    //         | int32 poseStatus  | RGB24 image
    //
    // The pose lives in the block rather than in metadata because it must be the pose
    // of *this* frame: RequestBlockCapture snapshots it 1-2 frames before the readback
    // completes, and read_block_memory may re-serve a cached block, which then needs
    // its own pose rather than the current one.
    private const int blockFlagOffset = 0;
    private const int blockDroneIdOffset = 4;
    private const int blockHeadingOffset = 8;
    private const int blockCamPosOffset = 12;      // float32 x, y, z (Unity world)
    private const int blockCamRotOffset = 24;      // float32 x, y, z, w (Unity Transform.rotation)
    private const int blockCaptureTimeOffset = 40; // float32 Time.realtimeSinceStartup
    private const int blockPoseStatusOffset = 44;  // int32 bitfield, see POSE_* below
    private const int blockLegacyHeaderSize = 12;
    private const int blockPoseHeaderSize = 48;
    // Chosen once in Start(): the pose header only when this component is the producer.
    private int blockHeaderSize = blockLegacyHeaderSize;
    private int blockImageDataOffset = blockLegacyHeaderSize;

    // poseStatus bits, so a dumped frame is self-describing about where its pose came from.
    private const int POSE_VALID = 1 << 0;
    private const int POSE_GROUND_TRUTH = 1 << 1;
    private const int POSE_NOISE_INJECTED = 1 << 2;

    private const int maxBlockWidth = 2000;
    private const int maxBlockHeight = 2000;
    private const int maxBlockImageCount = 30;
    private const int maxBlockImageSize = maxBlockWidth*maxBlockHeight * 3;
    // Reserve for the largest header, so switching stitcher mode never needs a bigger section.
    private const int maxTotalBlockSize = maxBlockImageCount * (blockPoseHeaderSize + maxBlockImageSize);
    private const int maxPanoramaWidth = 4000;
    private const int maxPanoramaHeight = 4000;
    private const int maxPanoramaSize = maxPanoramaWidth * maxPanoramaHeight * 3;
    private const int maxTotalPanoramaSize = panoramaDataPosition + maxPanoramaSize;

    // Metadata layout: the pilot heading yaw (float) is appended after
    // qualityThreshold (printStitchRate follows the yaw). This carries the
    // integrated body yaw (WriteBodyYaw), not the live HMD direction, so Python
    // selects the same views as SelectStitchCameras.
    // Offset = sizes(20) + stitcher(64) + cylindrical(1) + matcher(64) + ransac(1)
    //          + checks(4) + ratio(4) + score(4) + focal(4) + reserved(1) + fusion(64)
    //          + blurKernel(4) + blurSigma(4) + border(4) + qualityEnabled(1) + qualityThreshold(4)
    private const int metadataHeadYawOffset = 248;

    // ---- Metadata tail (wire v2) -------------------------------------------------
    // Appended after printStitchRate (which ends the v1 prefix at 253). Every offset
    // is 4-byte aligned so the seqlock counter below can be written atomically.
    // Mirrored field-for-field by readMetadataMemory / read_dynamic_state in
    // StitcherThreading.py -- change one, change the other.
    private const int metaWireVersion = 2;

    // The v1 prefix ends at 253; pad to 256 so every 4-byte field below is aligned.
    private const int metadataTailStart = 256;

    // Static: written by WriteMetadata, changes only on inspector edits.
    private const int metaBlockHeaderSizeOffset = 256;
    private const int metaWireVersionOffset = 260;
    private const int metaFxOffset = 264;
    private const int metaFyOffset = 268;
    private const int metaCxOffset = 272;
    private const int metaCyOffset = 276;
    private const int metaPlanarCanvasWidthOffset = 280;
    private const int metaPlanarCanvasHeightOffset = 284;
    private const int metaPlanarMetresPerPixelOffset = 288;
    private const int metaPlanarMaxRangeOffset = 292;
    private const int metaPlanarFeatherPxOffset = 296;
    private const int metaPlanarAnisoMaxOffset = 300;
    private const int metaPlanarMinCoverageOffset = 304;
    private const int metaPlanarPoseSourceOffset = 308;   // uint8
    private const int metaPlanarPsnrGateOffset = 309;     // uint8
    private const int metaPlanarBlendModeOffset = 310;    // uint8
    private const int metaPlanarDebugViewOffset = 311;    // uint8
    // The tail's 4-byte padding slot is now full: the next field must come out of
    // metadataReservedGap, after metadataTailEnd.

    // Dynamic: rewritten every frame by WriteDynamicState under a seqlock.
    // A bare blit is fine for a lone scalar like bodyYaw, but a torn plane normal
    // mixing an old X with a new Y/Z is neither unit-length nor perpendicular to
    // anything, and would produce one frame of geometrically garbage panorama --
    // rare enough to be very hard to reproduce. Writer bumps the counter either
    // side of the payload; the reader retries while it is odd or has changed.
    private const int metaDynSeqOffset = 312;
    private const int metaPlaneNxOffset = 316;
    private const int metaPlaneNyOffset = 320;
    private const int metaPlaneNzOffset = 324;
    private const int metaPlaneDOffset = 328;
    private const int metaPlaneValidOffset = 332;         // uint8
    private const int metaPlaneModeOffset = 333;          // uint8
    // 334-335 padding
    private const int metaGimbalPitchOffset = 336;
    // Which drone the planar canvas is framed on. Dynamic, not static: it changes when
    // the centre drone dies or the formation reshapes, and it must be read in the same
    // seqlock as the plane -- a canvas origin paired with the wrong frame's plane is the
    // same class of bug as a torn normal.
    private const int metaCentreDroneIdOffset = 340;      // int32, -1 = none

    // Warp-thread estimator settings. Static like the 256..311 block, but placed after
    // the dynamic one because that block's 4-byte padding slot was already full, so
    // these came out of metadataReservedGap instead. Ordering is irrelevant to
    // correctness -- both sides address every one of these by absolute offset -- but the
    // split is worth knowing about when adding the next field.
    private const int metaPlanarPlaneSweepOffset = 344;    // uint8
    private const int metaPlanarPoseRefineOffset = 345;    // uint8
    // 346-347 padding
    private const int metaPlanarSweepRangeOffset = 348;
    private const int metaPlanarSweepStepsOffset = 352;
    private const int metaPlanarRefineRateOffset = 356;
    private const int metaPlanarRefineMaxShiftOffset = 360;
    private const int metadataTailEnd = 364;

    // Trailing gap between the tail and the two size fields metadataSize ends with. It
    // shrinks as the tail grows so metadataSize -- and hence the mapped section size --
    // stays fixed at 412; a changed map size would strand any already-running Python.
    private const int metadataReservedGap = 40;

    // A left/centre/right panorama is always exactly 3 views; a planar mosaic can take
    // as many overlapping views as the formation offers.
    private const int STITCH_COUNT_LRC = 3;

    public enum StitchPoseSourceMode
    {
        GroundTruth,          // camera transform, exact
        NoisyState,           // StateFinder's simulated GPS/IMU noise (sigma ~0.03 m)
        GroundTruthPlusGnss,  // exact pose + injected drifting GNSS error (see StitchPoseSource.cs)
    }

    /// <summary>
    /// How the planar stitcher combines views where their footprints overlap.
    /// Mirrored by <c>BLEND_*</c> in PlanarStitcher.py.
    /// </summary>
    public enum PlanarBlendMode
    {
        // Cross-fade every contributing view, weighted by distance to its own image
        // border. Correct only while the geometry is exact; any residual misalignment
        // ghosts across the whole overlap, which in a wall formation is most of the
        // canvas.
        Feather,
        // One view per pixel: whichever sees that point closest to the plane normal.
        // Nothing is averaged, so residual pose error shows as a seam rather than a
        // doubled image.
        Nearest,
    }

    /// <summary>
    /// Diagnostic overlay that colours each patch of the planar mosaic by the drone it
    /// came from. Testing aid only — leave it Off for flight. Mirrored by <c>DEBUG_*</c>
    /// in PlanarStitcher.py; the drone-to-colour key is printed by the Python console's
    /// periodic [PLANAR] line, since the selection changes as drones join or drop out.
    /// </summary>
    public enum PlanarDebugView
    {
        Off,
        // Colour wash over the imagery: shows provenance without hiding the content.
        Tint,
        // Flat colour per view, imagery discarded. The clearest read on where the seams
        // fall and how big each drone's patch is.
        Flat,
    }

    /// <summary>
    /// Which surface the planar stitcher treats as the scene plane.
    /// Note this is <i>not</i> the swarm formation plane that SwarmPlaneController owns:
    /// that is the wall the drones fly in, this is the surface they are looking at,
    /// parallel to it and some distance in front.
    /// </summary>
    public enum ScenePlaneMode
    {
        Auto,     // Nadir when the gimbal is pitched down, Facade otherwise
        Facade,   // cast along the centre camera's forward axis
        Nadir,    // cast straight down; normal pinned to world up
        Manual,   // inspector normal + distance, no raycast
    }

    // Index into camerasToCapture of the centre stitch camera, set once per frame by
    // SelectStitchCameras or, in PLANAR mode, SelectPlanarCentreCamera. The scene-plane
    // raycast and the intrinsics are taken from it.
    private int centreStitchCameraIndex = -1;

    // Live scene-plane state, published every frame by WriteDynamicState.
    private Vector3 scenePlaneNormal = Vector3.up;
    private float scenePlaneD = 0f;
    private bool scenePlaneValid = false;
    private ScenePlaneMode resolvedPlaneMode = ScenePlaneMode.Nadir;
    private bool scenePlaneInitialised = false;
    private RaycastHit[] scenePlaneHits = new RaycastHit[8];

    private StitchPoseSource gnssNoise;
    private float lastPoseNoiseTime = -1f;
    private float poseNoiseDt = 0f;

    /// <summary>
    /// Pose of camera <paramref name="camIdx"/> to publish with its frame.
    /// Read from the camera transform, never reconstructed from StateFinder: the FPV
    /// camera is offset from the drone body and Slerps toward its target rotation
    /// (FPVCameraScript), so the body pose is neither the optical centre nor the
    /// current orientation.
    /// </summary>
    private void GetCameraPose(int camIdx, Camera camera,
                               out Vector3 pos, out Quaternion rot, out int status)
    {
        pos = camera.transform.position;
        rot = camera.transform.rotation;
        status = POSE_VALID;

        if (poseSource == StitchPoseSourceMode.GroundTruth)
        {
            status |= POSE_GROUND_TRUTH;
            return;
        }

        if (poseSource == StitchPoseSourceMode.GroundTruthPlusGnss)
        {
            if (gnssNoise == null) gnssNoise = new StitchPoseSource(gnssSettings);
            gnssNoise.Apply(camIdx, poseNoiseDt, ref pos, ref rot);
            status |= POSE_NOISE_INJECTED;
            return;
        }

        StateFinder state = (stitchStates != null && camIdx < stitchStates.Count)
            ? stitchStates[camIdx] : null;
        if (state == null)
        {
            status |= POSE_GROUND_TRUTH;   // no state to degrade with; say so honestly
            return;
        }

        // NoisyState: carry the camera's offset from the body across, so this stays the
        // optical centre; identical to camera.transform.position when the noise is zero.
        pos += state.Position - state.transform.position;
        status |= POSE_NOISE_INJECTED;
    }

    // Advances the shared common-mode GNSS bias once per frame, so every drone in a
    // frame draws against the same common-mode sample -- which is the entire point of
    // that term. Per-drone bias advances inside Apply.
    private void StepPoseNoise()
    {
        float now = Time.time;
        poseNoiseDt = (lastPoseNoiseTime < 0f) ? 0f : Mathf.Max(0f, now - lastPoseNoiseTime);
        lastPoseNoiseTime = now;

        if (poseSource != StitchPoseSourceMode.GroundTruthPlusGnss) return;
        if (gnssNoise == null) gnssNoise = new StitchPoseSource(gnssSettings);
        gnssNoise.Step(now);
    }

    // Slots in the block map. Deliberately a function of the *camera count* and the
    // stitcher mode only -- never of the per-frame selection, because CreateBlockMap
    // destroys and recreates the named section and Python holds a single mapping of it.
    // Frames where fewer cameras are selected mark the spare slots droneId = -1 instead.
    private int DesiredBlockCount()
    {
        // Not the producer (DJI scene): ImageSharing.cs owns BlockSharedMemory and creates a
        // fixed StitchSlots(=3)-slot section. The count still has to be published, because
        // Python sizes its mapping from metadata and this component owns the metadata map
        // either way. Deliberately NOT clamped by camerasToCapture here: that scene has no
        // sim FPV cameras, so the clamp would advertise 0 blocks and Python would map none
        // of the section ImageSharing is filling.
        if (!enableImageWriting) return STITCH_COUNT_LRC;

        int wanted = (typeOfStitcher == stitcherType.PLANAR)
            ? Mathf.Clamp(maxStitchViews, 3, maxBlockImageCount)
            : STITCH_COUNT_LRC;
        return Mathf.Min(wanted, camerasToCapture != null ? camerasToCapture.Count : 0);
    }

    // The pose-carrying block header is only written when this component is the
    // producer. In the DJI scene ImageSharing.cs owns BlockSharedMemory with the
    // 12-byte v1 header and enableImageWriting is off here, so the size we advertise
    // in metadata stays consistent with whatever is actually writing the blocks.
    private int ActiveBlockHeaderSize()
    {
        return enableImageWriting ? blockPoseHeaderSize : blockLegacyHeaderSize;
    }

    // Curved-screen runtime objects; the geometry itself lives in VrDisplaySettings.
    private Material curvedScreenMaterial;
    private MeshRenderer panoramaRenderer;
    private Texture2D panoTexture;

    // Body heading that drives the panorama. Seeded once from the head's initial
    // yaw, then advanced only by the controller yaw-rate command (never by head
    // tracking). cameraRigTransform is rotated by the same command so the rig
    // turns with the body while the head still yaws freely relative to it.
    private Transform cameraRigTransform;
    private float bodyYaw;
    private bool bodyYawInitialized = false;
    // Body-locked mode (non-hull attitude modes): the pilot's yaw stick spins the
    // drone(s) directly, so the body/rig heading is slaved to the followed drone's
    // ACTUAL heading rather than the raw stick, giving zero relative motion between
    // the headset view and the feed. Tracks the followed camera and its last heading
    // so we advance by the true per-frame delta (and reseed cleanly on a drone swap).
    private Camera bodyLockCamera;
    private float previousDroneYaw;
    private bool bodyLockInitialized = false;
    // Set when the calibrate key is pressed; consumed the same frame once the centre-drone yaw
    // is known, so the recentre snaps the view onto the (discrete) panorama centre.
    private bool calibrationRequested = false;

    // Controller-integrated body heading in degrees (0-360). Exposed so the swarm's VR command
    // frame can rotate velocity commands into the pilot's heading. Mirrors the private bodyYaw.
    public static float BodyYawDegrees { get; private set; }

    // The "Drone N" root of the drone at the centre of the stitch selection (camera yaw closest to
    // the body yaw), i.e. the drone the pilot is looking through. Refreshed every frame whether or
    // not stitching is running, so systems that need "whatever the pilot is facing" can anchor on it
    // — SwarmPlaneController orients the vertical swarming plane from this drone's heading.
    public static Transform CentreStitchDrone { get; private set; }

    private GameObject arena;
    private int[] selectedStitchIndices = new int[0];  // camera indices written to the 3 blocks, ordered [left, centre, right]

    // Change-detection so the stitchedDrones readout only rebuilds when the selection changes.
    private string lastStitchedDronesKey;

    // Panorama display state. The pilot toggle itself (panoramaUserEnabled) is at the top of
    // the inspector; this is the resolved state after the quality fallback has had its say.
    private bool panoramaDisplayActive = true;

    // Edge-detection for the controller's click switch, so a disconnected controller
    // (InputManager's "userSwitch" resting at its default) doesn't fight the manual
    // toggle above every frame -- only an actual change in the reading takes over.
    private float lastControllerUserSwitch;
    private bool controllerUserSwitchInitialized = false;

    private string lastHiddenScreensKey;

    // Other timing values to check the number of camera in the block
    private float cameraUpdateInterval = 3f;
    private float nextCameraUpdateTime = 0f;

    void Start()
    {
        // Always create metadata if either feature is enabled
        if (enableImageWriting || enablePanoramaReading)
        {
            metadataFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)metadataSize, metadataMapName);
            metadataPtr = MapViewOfFile(metadataFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        }

        // Discover cameras first so the block mapping can be sized to the
        // exact drone count (matches image_stream.py / StitcherThreading.py).
        // Discovery is unconditional (it only builds a few lists): the centre-drone
        // selection publishes CentreStitchDrone every frame, and consumers such as
        // SwarmPlaneController need it whether or not stitching is running.
        FindCameras();
        blockHeaderSize = ActiveBlockHeaderSize();
        blockImageDataOffset = blockHeaderSize;
        // Published unconditionally, alongside blockHeaderSize: Python sizes its
        // BlockSharedMemory mapping from these two, and in the DJI scene the section is
        // created by ImageSharing.cs rather than here. Leaving the count at 0 there (as it
        // was when this assignment sat inside the enableImageWriting branch) makes Python
        // map nothing and the real-drone panorama never appears. CreateBlockMap stays gated
        // on enableImageWriting, so this does not make us a second producer.
        blockImageCount = DesiredBlockCount();
        if (enableImageWriting)
        {
            // Two producers writing one block map with different header sizes would
            // corrupt it. In the DJI scene this component must have image writing off.
            if (FindObjectOfType<ImageSharing>() != null)
            {
                Debug.LogError("PyUniSharingFast: enableImageWriting is on while an ImageSharing " +
                               "component is present. Both write BlockSharedMemory, and they use " +
                               "different block header sizes. Turn enableImageWriting off here.");
            }
        }

        CalculateMemorySizes();
        CreateMemoryMaps();

        // Resolve the HMD head transform now (the OVRCameraRig is in the scene)
        // so the first metadata write seeds a real head yaw rather than 0.
        FindHeadTransform();

        if (enableImageWriting || enablePanoramaReading)
        {
            WriteMetadata();
        }

        if (enablePanoramaReading)
        {
            GenerateCurvedScreen();
            panoramaRenderer = GetComponent<MeshRenderer>();
            curvedScreenMaterial = panoramaRenderer.material;
            curvedScreenMaterial.SetFloat("_Glossiness", 0f);
            curvedScreenMaterial.SetColor("_EmissionColor", Color.white);
            curvedScreenMaterial.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;
            curvedScreenMaterial.EnableKeyword("_EMISSION");
            panoTexture = new Texture2D(panoramaImageWidth, panoramaImageHeight, TextureFormat.RGB24, false);
            // The texture object is stable (only its bytes change on each read),
            // so bind it to the curved-screen material once here.
            curvedScreenMaterial.mainTexture = panoTexture;
            curvedScreenMaterial.SetTexture("_EmissionMap", panoTexture);

            // ScreenSpawn drives the per-drone feed fallback when the panorama is bad.
            if (screenSpawn == null)
            {
                screenSpawn = FindObjectOfType<ScreenSpawn>();
            }
        }

        if (enableImageWriting)
        {
            reusableTexture = new RenderTexture(blockImageWidth, blockImageHeight, 24);
            image = new Texture2D(blockImageWidth, blockImageHeight, TextureFormat.RGB24, false);
            blockImageBytes = new byte[blockImageSize];
            EnsureConvertedBlockBuffer();

            EnsureReadbackPool();
        }

        hasStarted = true;
        nextSendTime = Time.time;
    }

    void Update()
    {
        if(blockImageWidth>maxBlockWidth || blockImageHeight>maxBlockHeight || panoramaImageWidth>maxPanoramaWidth || panoramaImageHeight>maxPanoramaHeight)
        {
            Debug.LogError("Problem Dimensions");
            return;
        }

        if (Time.time >= nextCameraUpdateTime)
        {
            UpdateCameras();
            nextCameraUpdateTime = Time.time + cameraUpdateInterval;
        }

        if (enablePanoramaReading && resize_dimension)
        {
            GenerateCurvedScreen();
        }

        // The body heading (not the live head direction) drives which three views
        // are stitched and where the curved panorama screen sits. It starts at the
        // initial head yaw, then only advances with the controller yaw-rate command
        // (which also rotates the camera rig to mimic body motion). The head can
        // still yaw freely via HMD tracking to look at the side screens without
        // moving the panorama. Resolve the rig/head lazily (the rig may be added to
        // the scene later).
        FindHeadTransform();

        // Recalibrate on demand. Seed the body heading from the current head yaw so this frame's
        // centre-drone selection is taken relative to where the pilot is looking; the recentre is
        // finished below (once centreYaw is known) by rotating the view onto that centre drone.
        if (Input.GetKeyDown(calibrateKey))
        {
            SeedBodyYawFromHead();
            calibrationRequested = true;
        }

        UpdateBodyYaw();
        WriteBodyYaw(bodyYaw);

        // Resolve the centre drone. Both branches publish CentreStitchDrone (which
        // SwarmPlaneController anchors on) and centreStitchCameraIndex (which the scene-plane
        // raycast and the intrinsics are taken from), and both return the centre camera's yaw,
        // which drives the curved-screen orientation so the screen snaps to the new view only
        // when the selection changes.
        //
        // The rule differs because the two configurations disagree on what "centre" means: a
        // left/centre/right panorama is centred on what the pilot is looking at, whereas a
        // planar mosaic is centred on the formation, and under the shared heading of
        // vertical-plane mode the yaw rule has no unique answer at all.
        float centreYaw;
        if (typeOfStitcher == stitcherType.PLANAR)
        {
            centreYaw = SelectPlanarCentreCamera();
        }
        else
        {
            centreYaw = SelectStitchCameras(bodyYaw, out selectedStitchIndices);
        }

        // Advance the pose-error model once per frame, before any pose is snapshotted.
        StepPoseNoise();

        // Resolve and publish the scene plane before the capture loop, so the plane
        // Python sees for this frame matches the frame's pose snapshots.
        UpdateScenePlane(centreStitchCameraIndex);
        WriteDynamicState();

        // Planar picks its contributing views separately from its centre, and must do it after
        // UpdateScenePlane: the selection tests each camera against the current scene plane.
        if (typeOfStitcher == stitcherType.PLANAR)
        {
            SelectPlanarStitchCameras(out selectedStitchIndices);
        }

        // Finish a pending calibration now that the centre drone is known: recentre the view onto
        // it so the head faces the (snapped) panorama centre and the VR velocity forward matches.
        if (calibrationRequested)
        {
            CalibrateToCentre(centreYaw);
            calibrationRequested = false;
        }

        UpdateStitchedDronesDisplay(selectedStitchIndices);
        UpdateStitchedScreenHiding(selectedStitchIndices);

        if (enablePanoramaReading)
        {
            UpdateCurvedScreenPose(centreYaw);
        }

        // Handle image writing to BlockSharedMemory (one self-describing block
        // per drone, handshaken independently — matches image_stream.py).
        if (enableImageWriting)
        {
            if (camerasToCapture.Count == 0)
            {
                FindCameras();
            }
            else if (Time.time >= nextSendTime && blockPtr != IntPtr.Zero)
            {
                // Queue an async GPU readback for the drones selected this frame. The
                // block write to shared memory happens in the completion callback, 1-2
                // frames later — no ReadPixels stall on the main thread.
                //
                // The map has a fixed slot count (sized from the camera count, not the
                // selection), so any slots the selection doesn't reach this frame are
                // explicitly marked empty. Without that they would keep serving a stale
                // frame from Python's busy-block cache indefinitely.
                for (int j = 0; j < blockImageCount; j++)
                {
                    if (j < selectedStitchIndices.Length)
                        RequestBlockCapture(j, selectedStitchIndices[j]);
                    else
                        InvalidateBlockSlot(j);
                }

                // Advance by whole intervals, but never fall more than one interval
                // behind — a long stall must not trigger a burst of catch-up sends.
                nextSendTime = Mathf.Max(nextSendTime + sendInterval, Time.time - sendInterval);
            }
        }

        if (Input.GetKeyDown(togglePanoramaKey))
        {
            panoramaUserEnabled = !panoramaUserEnabled;
        }

        // Mirror the pilot's panorama toggle from the click switch, but only on an
        // actual change in its reading -- not every frame -- so a disconnected
        // controller (userSwitch resting at InputManager's default) doesn't
        // immediately undo the manual toggle/Inspector checkbox above. readController.py
        // latches the spring-loaded switch, so userSwitch is a stable on/off level
        // (1 = show panorama, -1 = show feeds) while a controller is connected.
        if (InputManager.Instance != null)
        {
            float userSwitch = InputManager.Instance.InputStatus["userSwitch"];
            if (!controllerUserSwitchInitialized)
            {
                lastControllerUserSwitch = userSwitch;
                controllerUserSwitchInitialized = true;
            }
            else if (userSwitch != lastControllerUserSwitch)
            {
                panoramaUserEnabled = userSwitch > 0f;
                lastControllerUserSwitch = userSwitch;
            }
        }

        // Handle panorama reading from PanoramaSharedMemory
        if (enablePanoramaReading)
        {
            if (Time.time >= nextReceiveTime && panoramaPtr != IntPtr.Zero && Marshal.ReadInt32(panoramaPtr, FlagPosition) == 0)
            {
                Marshal.WriteInt32(panoramaPtr, FlagPosition, 1);

                int qualityWord = Marshal.ReadInt32(panoramaPtr, panoramaQualityPosition);

                // bit 0 = panorama good; bits 1-3 = failing-gate reason (only
                // meaningful when bit 0 is clear). When the panorama is bad (and
                // fallback is enabled) show the individual drone feeds instead.
                bool qualityOk = (qualityWord & QUALITY_OK_BIT) != 0;
                // The pilot's click-switch toggle overrides quality: if they turned
                // the panorama off, hide it and show the feeds regardless of quality.
                bool panoramaGood = (!qualityFallbackEnabled || qualityOk) && panoramaUserEnabled;
                if (panoramaGood)
                {
                    // Upload straight from the mapped view while the flag is held.
                    // Python writes RGB24 bottom-up (flipped + BGR->RGB on its
                    // side), exactly the layout the texture expects — no managed
                    // copy, no per-pixel conversion.
                    panoTexture.LoadRawTextureData(IntPtr.Add(panoramaPtr, panoramaDataPosition), panoramaImageSize);
                }
                Marshal.WriteInt32(panoramaPtr, FlagPosition, 0);

                ApplyQualityFallback(panoramaGood, qualityWord);
                if (panoramaGood)
                {
                    panoTexture.Apply(false);
                }

                nextReceiveTime += readInterval;
            }
        }
    }

    // Switch between the stitched panorama screen and the individual drone
    // feeds (ScreenSpawn). Only acts on a transition so it doesn't fight
    // ScreenSpawn's per-frame positioning. Python already debounces the
    // quality verdict (hysteresis), so the flag is stable.
    private void ApplyQualityFallback(bool panoramaGood, int qualityWord)
    {
        if (panoramaGood == panoramaDisplayActive)
        {
            return; // no change
        }
        panoramaDisplayActive = panoramaGood;

        // Log only on the transition so it doesn't spam every frame. When the
        // panorama disappears, report which stitch-quality gate(s) tripped.
        if (!panoramaGood)
        {
            string reason = !panoramaUserEnabled
                ? "toggled off by pilot (click switch / keyboard / inspector)"
                : $"stitch quality bad: {DescribeQualityReason(qualityWord)}";
            Debug.Log($"[Panorama] hidden — showing individual feeds. {reason}.");
        }
        else
        {
            Debug.Log("[Panorama] restored — stitch quality good again.");
        }

        // Show/hide the curved panorama screen.
        if (panoramaRenderer != null)
        {
            panoramaRenderer.enabled = panoramaGood;
        }

        // Show/hide the individual feed screens.
        if (screenSpawn != null)
        {
            screenSpawn.ShowFallbackFeeds(!panoramaGood, fallbackScreenStyle);
        }
    }

    // Decode the failing-gate bits of the packed quality word into a readable
    // reason, so the transition log says *why* the panorama was hidden.
    private string DescribeQualityReason(int qualityWord)
    {
        string reasons = "";
        if ((qualityWord & REASON_CANVAS) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "canvas (degenerate warp size)";
        if ((qualityWord & REASON_DISTORTION) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "distortion (folded/torn mesh)";
        if ((qualityWord & REASON_PHOTOMETRIC) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "photometric (overlap PSNR below threshold)";
        if ((qualityWord & REASON_NO_OVERLAP) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "no overlap (camera yaw gap exceeds FOV)";
        if ((qualityWord & REASON_TOO_FEW_IMAGES) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "too few feeds (fewer than 3 selected)";
        if ((qualityWord & REASON_PLANE_INVALID) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "no usable scene plane / pose (planar)";
        return reasons.Length > 0 ? reasons : "unspecified";
    }

    // Renders (if needed) and queues an async GPU readback of one selected
    // camera into the given block slot. The camera's ScreenSpawn feed RT is the
    // capture source (SpawnScreens sizes it to the block resolution), so the
    // camera is never rendered twice: an enabled camera's RT already holds this
    // frame's image; a disabled one (feed screen hidden) is rendered on demand
    // here, at the send rate only.
    private void RequestBlockCapture(int slot, int camIdx)
    {
        if (pendingReadbacks == null) return;  // image writing was off at Start
        if (camIdx < 0 || camIdx >= camerasToCapture.Count) return;
        Camera camera = camerasToCapture[camIdx];
        if (camera == null) return;

        RenderTexture rt = camera.targetTexture;
        if (rt != null && rt.width == blockImageWidth && rt.height == blockImageHeight)
        {
            if (!camera.enabled)
            {
                camera.Render();
            }
        }
        else
        {
            // Fallback (no ScreenSpawn / mismatched RT): render into our own RT.
            // The readback request below snapshots the RT at this point in the
            // GPU command stream, so reusing one RT across cameras is safe.
            RenderTexture previousRT = camera.targetTexture;
            camera.targetTexture = reusableTexture;
            camera.Render();
            camera.targetTexture = previousRT;
            rt = reusableTexture;
        }

        int p = AcquirePendingSlot();
        if (p < 0)
        {
            // Pool exhausted (readbacks piling up): this slot keeps its previous frame.
            // Warned about rather than dropped silently — sustained exhaustion starves
            // the same slots every send (AcquirePendingSlot scans from 0), and a slot
            // that has never been written still carries the empty-slot sentinel, so the
            // drone simply never appears in the mosaic.
            if (Time.time >= nextReadbackPoolWarnTime)
            {
                nextReadbackPoolWarnTime = Time.time + readbackPoolWarnInterval;
                Debug.LogWarning($"[PyUniSharingFast] GPU readback pool exhausted " +
                                 $"({pendingReadbacks.Length} entries, {blockImageCount} block slots): " +
                                 "block slots are being skipped. Readbacks are completing slower than " +
                                 "sendInterval; lower the block resolution or the send rate.");
            }
            return;
        }

        pendingReadbacks[p].slot = slot;
        pendingReadbacks[p].droneId = camIdx;
        // Heading and pose are recorded now, matching the image being read back — not
        // at completion, when the drone may have yawed or flown on.
        pendingReadbacks[p].heading = camera.transform.eulerAngles.y;
        GetCameraPose(camIdx, camera,
                      out pendingReadbacks[p].camPos,
                      out pendingReadbacks[p].camRot,
                      out pendingReadbacks[p].poseStatus);
        pendingReadbacks[p].captureTime = Time.realtimeSinceStartup;

        // One-time row-order calibration: capture the same RT synchronously so
        // the completion callback can pick the flip that reproduces the exact
        // top-down BGR bytes the old ReadPixels path produced.
        if (!readbackFlipCalibrated && !readbackFlipCalibrating)
        {
            readbackFlipCalibrating = true;
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = rt;
            image.ReadPixels(new Rect(0, 0, blockImageWidth, blockImageHeight), 0, 0, false);
            RenderTexture.active = previousActive;
            byte[] reference = new byte[blockImageSize];
            ConvertToBlockFormat(image.GetRawTextureData(), reference);
            pendingReadbacks[p].verifyReference = reference;
        }

        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, pendingCallbacks[p]);
    }

    // Marks a block slot as carrying no view this frame, using the same droneId = -1
    // sentinel the DJI feed map uses. Python drops the slot *and* evicts it from its
    // busy-block cache, so a drone that leaves the selection stops contributing rather
    // than lingering in the mosaic forever.
    private void InvalidateBlockSlot(int slot)
    {
        if (blockPtr == IntPtr.Zero || slot < 0 || slot >= blockImageCount) return;

        IntPtr block = IntPtr.Add(blockPtr, slot * blockSize);
        if (Marshal.ReadInt32(block, blockFlagOffset) != 0) return;  // consumer mid-read

        Marshal.WriteInt32(block, blockFlagOffset, 1);
        Marshal.WriteInt32(block, blockDroneIdOffset, -1);
        if (blockHeaderSize >= blockPoseHeaderSize)
        {
            Marshal.WriteInt32(block, blockPoseStatusOffset, 0);
        }
        Marshal.WriteInt32(block, blockFlagOffset, 0);
    }

    // Sizes the readback pool for two full in-flight batches, one entry per block slot.
    //
    // It used to be a flat 8, which was "a couple of in-flight 3-slot batches" back when
    // every stitcher captured exactly 3 views. PLANAR captures up to blockImageCount
    // (maxStitchViews) per send, and a readback completes 1-2 frames later, so an
    // undersized pool does not merely drop a frame: AcquirePendingSlot scans from index
    // 0, so it is always the *same* tail slots that lose the race, and a slot that never
    // wins never gets written at all. Those blocks then sit at their initial contents
    // forever, which is what surfaced on the Python side as a degenerate quaternion.
    //
    // Grow-only, and the array is replaced wholesale rather than re-indexed: a pending
    // entry keeps its index, so a readback still in flight across the resize completes
    // into the same entry it reserved.
    private void EnsureReadbackPool()
    {
        int wanted = Mathf.Clamp(blockImageCount * 2, 8, 2 * maxBlockImageCount);
        int have = pendingReadbacks != null ? pendingReadbacks.Length : 0;
        if (have >= wanted) return;

        Array.Resize(ref pendingReadbacks, wanted);
        Array.Resize(ref pendingCallbacks, wanted);
        for (int i = have; i < wanted; i++)
        {
            int idx = i;   // one cached delegate per entry, so requests never allocate
            pendingCallbacks[i] = request => OnBlockReadback(idx, request);
        }
    }

    private int AcquirePendingSlot()
    {
        for (int i = 0; i < pendingReadbacks.Length; i++)
        {
            if (!pendingReadbacks[i].inUse)
            {
                pendingReadbacks[i].inUse = true;
                return i;
            }
        }
        return -1;
    }

    // Completion callback (main thread): convert the RGBA readback to the
    // top-down BGR block format with a Burst job and write it into the slot's
    // shared-memory block under the usual flag handshake. Runs 1-2 frames after
    // the request; the handshake semantics are identical to the old synchronous
    // path (a busy consumer just drops this frame).
    private void OnBlockReadback(int p, AsyncGPUReadbackRequest request)
    {
        PendingReadback pending = pendingReadbacks[p];
        pendingReadbacks[p].inUse = false;
        pendingReadbacks[p].verifyReference = null;

        // Teardown/resize safety: drop late readbacks once the maps or buffers
        // are gone or the block resolution changed under this request.
        if (request.hasError || blockPtr == IntPtr.Zero || !convertedBlock.IsCreated)
            return;

        NativeArray<byte> data = request.GetData<byte>();
        if (data.Length < blockImageWidth * blockImageHeight * 4 || convertedBlock.Length != blockImageSize)
            return;

        if (pending.verifyReference != null)
        {
            CalibrateReadbackFlip(data, pending.verifyReference);
        }

        var job = new ConvertRgbaToBgrJob
        {
            src = data,
            dst = convertedBlock,
            width = blockImageWidth,
            height = blockImageHeight,
            flipRows = readbackFlipRows,
        };
        job.Schedule(blockImageHeight, 32).Complete();
        convertedBlock.CopyTo(blockImageBytes);

        IntPtr block = IntPtr.Add(blockPtr, pending.slot * blockSize);

        // Skip this slot if the consumer is mid-read on its block.
        if (Marshal.ReadInt32(block, blockFlagOffset) != 0)
            return;

        // Mark busy while we write the header + image.
        Marshal.WriteInt32(block, blockFlagOffset, 1);

        Marshal.WriteInt32(block, blockDroneIdOffset, pending.droneId);
        WriteFloat(block, blockHeadingOffset, pending.heading);

        if (blockHeaderSize >= blockPoseHeaderSize)
        {
            WriteFloat(block, blockCamPosOffset + 0, pending.camPos.x);
            WriteFloat(block, blockCamPosOffset + 4, pending.camPos.y);
            WriteFloat(block, blockCamPosOffset + 8, pending.camPos.z);
            WriteFloat(block, blockCamRotOffset + 0, pending.camRot.x);
            WriteFloat(block, blockCamRotOffset + 4, pending.camRot.y);
            WriteFloat(block, blockCamRotOffset + 8, pending.camRot.z);
            WriteFloat(block, blockCamRotOffset + 12, pending.camRot.w);
            WriteFloat(block, blockCaptureTimeOffset, pending.captureTime);
            Marshal.WriteInt32(block, blockPoseStatusOffset, pending.poseStatus);
        }

        Marshal.Copy(blockImageBytes, 0, IntPtr.Add(block, blockImageDataOffset), blockImageSize);

        // Ready for the consumer.
        Marshal.WriteInt32(block, blockFlagOffset, 0);
    }

    // Decides readbackFlipRows by converting the first readback both ways and
    // comparing against the synchronous ReadPixels capture of the same RT
    // contents. The winning flip reproduces the old block bytes exactly, so the
    // bridge format is provably unchanged.
    private void CalibrateReadbackFlip(NativeArray<byte> data, byte[] reference)
    {
        int noFlipMismatches = CountConversionMismatches(data, reference, false);
        int flipMismatches = noFlipMismatches == 0 ? int.MaxValue
                                                   : CountConversionMismatches(data, reference, true);
        readbackFlipRows = flipMismatches < noFlipMismatches;
        readbackFlipCalibrated = true;
        readbackFlipCalibrating = false;

        int winner = Mathf.Min(noFlipMismatches, flipMismatches);
        if (winner == 0 || noFlipMismatches == 0)
        {
            Debug.Log($"[PyUniSharingFast] AsyncGPUReadback row order calibrated: flipRows={readbackFlipRows} (byte-identical to the ReadPixels path).");
        }
        else
        {
            Debug.LogWarning($"[PyUniSharingFast] AsyncGPUReadback calibration found no exact match " +
                             $"(mismatched bytes: noFlip={noFlipMismatches}, flip={flipMismatches}); " +
                             $"using flipRows={readbackFlipRows}. Verify the panorama orientation.");
        }
    }

    // (Re)allocates the persistent conversion buffer to the current block size.
    // Any readback still in flight across a resize is dropped by the length
    // check in OnBlockReadback.
    private void EnsureConvertedBlockBuffer()
    {
        if (convertedBlock.IsCreated && convertedBlock.Length == blockImageSize) return;
        if (convertedBlock.IsCreated) convertedBlock.Dispose();
        convertedBlock = new NativeArray<byte>(blockImageSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
    }

    private int CountConversionMismatches(NativeArray<byte> data, byte[] reference, bool flipRows)
    {
        var job = new ConvertRgbaToBgrJob
        {
            src = data,
            dst = convertedBlock,
            width = blockImageWidth,
            height = blockImageHeight,
            flipRows = flipRows,
        };
        job.Schedule(blockImageHeight, 32).Complete();

        int mismatches = 0;
        for (int i = 0; i < blockImageSize; i++)
        {
            if (convertedBlock[i] != reference[i]) mismatches++;
        }
        return mismatches;
    }

    // Converts Unity's raw RGB24 texture data (bottom-left origin, RGB order)
    // into the block format the Python consumers expect: top-left origin, BGR
    // order. This matches the cv2 images written by image_stream.py and the
    // BGR->RGB read performed in ImageSharing.cs.
    private void ConvertToBlockFormat(byte[] src, byte[] dst)
    {
        int rowBytes = blockImageWidth * 3;
        for (int y = 0; y < blockImageHeight; y++)
        {
            int srcRow = y * rowBytes;                              // bottom-up source row
            int dstRow = (blockImageHeight - 1 - y) * rowBytes;     // flipped to top-down
            for (int x = 0; x < blockImageWidth; x++)
            {
                int s = srcRow + x * 3;
                int d = dstRow + x * 3;
                dst[d]     = src[s + 2];  // B
                dst[d + 1] = src[s + 1];  // G
                dst[d + 2] = src[s];      // R
            }
        }
    }

    private void GenerateCurvedScreen()
    {
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        Mesh mesh = new Mesh();
        int vertCount = (segments + 1) * 2;
        Vector3[] vertices = new Vector3[vertCount];
        Vector2[] uvs = new Vector2[vertCount];
        int[] triangles = new int[segments * 6];

        float angleStep = angleRange / segments;
        float halfHeight = height / 2f;

        for (int i = 0; i <= segments; i++)
        {
            float angle = Mathf.Deg2Rad * (-angleRange / 2 + i * angleStep);
            float x = Mathf.Sin(angle) * radius;
            float z = Mathf.Cos(angle) * radius;

            vertices[i * 2] = new Vector3(x, -halfHeight, z);
            uvs[i * 2] = new Vector2(i / (float)segments, 0);
            vertices[i * 2 + 1] = new Vector3(x, halfHeight, z);
            uvs[i * 2 + 1] = new Vector2(i / (float)segments, 1);

            if (i < segments)
            {
                int triangleOffset = i * 6;
                triangles[triangleOffset] = i * 2;
                triangles[triangleOffset + 1] = (i * 2) + 1;
                triangles[triangleOffset + 2] = (i * 2) + 2;
                triangles[triangleOffset + 3] = (i * 2) + 2;
                triangles[triangleOffset + 4] = (i * 2) + 1;
                triangles[triangleOffset + 5] = (i * 2) + 3;
            }
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uvs;
        mesh.RecalculateNormals();
        meshFilter.mesh = mesh;
    }

    // Resolve the HMD head transform (OVRCameraRig.centerEyeAnchor). Prefers the
    // serialized override; otherwise finds the OVRCameraRig that lives in the
    // scene. Resolved once in Start and cached — the Update guard short-circuits
    // every frame after, so FindObjectOfType only runs until the rig's anchors
    // are ready (then Camera.main as an editor fallback when there's no rig).
    private void FindHeadTransform()
    {
        if (headTransform != null && cameraRigTransform != null) return;

        OVRCameraRig rig = FindObjectOfType<OVRCameraRig>();
        if (rig != null)
        {
            // The rig root is the "body" we rotate; centerEyeAnchor is the HMD-
            // tracked head, which yaws relative to the rig as the pilot looks around.
            if (cameraRigTransform == null) cameraRigTransform = rig.transform;
            if (headTransform == null && rig.centerEyeAnchor != null) headTransform = rig.centerEyeAnchor;
            if (headTransform != null) return;
        }

        // Editor fallback when no OVR rig is present: drive the main camera as both
        // the head and the body so the behaviour is still testable.
        if (Camera.main != null)
        {
            if (headTransform == null) headTransform = Camera.main.transform;
            if (cameraRigTransform == null) cameraRigTransform = Camera.main.transform;
        }
    }

    // Advances the body heading that drives the panorama. Seeded once from the
    // head's initial yaw so the panorama starts where the pilot is first looking,
    // then advanced only by the controller yaw-rate command (the same normalised
    // [-1,1] "yaw" the swarm integrates). Head tracking is deliberately excluded
    // so looking around doesn't move the panorama. The same command rotates the
    // OVRCameraRig to mimic body motion; the head still yaws freely relative to it.
    private void UpdateBodyYaw()
    {
        if (!bodyYawInitialized)
        {
            SeedBodyYawFromHead();
            return;
        }

        // Vertical-plane swarming also spins the drones with the yaw stick (the anchor takes
        // it directly, the rest of the wall slaves to the anchor), so it needs the same
        // treatment as the non-hull modes below — regardless of which attitude algorithm is
        // selected, since plane mode replaces the heading rule entirely.
        SwarmPlaneController plane = SwarmPlaneController.Instance;
        if (plane != null && plane.PlaneModeActive)
        {
            bodyLockInitialized = false; // reseed the camera-follow path when plane mode ends
            UpdateBodyYawFromPlane(plane);
            return;
        }

        // Non-hull attitude modes (NONE/SIMPLE) spin the drone(s) with the yaw stick,
        // so slave the body/rig heading to the drone's ACTUAL heading instead of the
        // raw stick. Matching rate constants wouldn't cancel relative motion: the drone
        // lags its command through the yaw filter, inner rate loop and drag.
        if (IsBodyLockedToDrone())
        {
            UpdateBodyYawFromDrone();
            return;
        }

        // Hull modes: the drones hold their heading and the stick steers the view.
        bodyLockInitialized = false; // reseed if we later switch back to a body-locked mode
        float normYaw = InputManager.Instance != null ? InputManager.Instance.InputStatus["yaw"] : 0f;
        float deltaYaw = normYaw * bodyYawRate * Time.deltaTime;
        if (deltaYaw == 0f) return;

        bodyYaw = Mathf.Repeat(bodyYaw + deltaYaw, 360f);
        BodyYawDegrees = bodyYaw;

        if (driveCameraRigYaw && cameraRigTransform != null)
        {
            // Rotate the body about the world vertical at the rig's pivot. The head
            // (centerEyeAnchor) rotates with it but keeps its own HMD-tracked yaw.
            cameraRigTransform.Rotate(0f, deltaYaw, 0f, Space.World);
        }
    }

    // Lock the body heading (and the rig) onto the swarming plane's own heading while
    // vertical-plane mode is on.
    //
    // In that mode the yaw stick has two effects at once: AttitudeAlgorithm.ApplyPlaneModeAttitude
    // feeds it to the anchor drone as a yaw-rate command (and every other drone slaves its heading
    // to the anchor), while the hull path below would *also* integrate the same stick into bodyYaw.
    // Those two integrations don't agree — different gains (bodyYawRate deg/s vs
    // VelocityControl.maxYawRate rad/s), and the drones additionally lag through the yaw filter,
    // the inner rate loop, drag, and SwarmPlaneController's own low-pass — so the view drifts off
    // the wall as soon as the stick moves. In the radially-outward configuration that drift is
    // invisible (the panorama re-snaps to whichever camera is now closest to bodyYaw), but here
    // every drone shares one heading: bodyYaw is what aims the VR velocity frame at the wall, and
    // the wall's own heading is the only value that can't be wrong.
    //
    // The lock is absolute rather than incremental, so any offset already present when plane mode
    // was entered is corrected on the first frame instead of being carried forever. Following
    // AnchorYaw (the low-passed plane heading) and not the anchor drone's instantaneous heading is
    // deliberate: it is the heading the rest of the wall is being driven to, so it is what the
    // panorama actually shows.
    private void UpdateBodyYawFromPlane(SwarmPlaneController plane)
    {
        float planeYaw = plane.AnchorYaw * Mathf.Rad2Deg;
        float deltaYaw = Mathf.DeltaAngle(bodyYaw, planeYaw);
        if (deltaYaw == 0f) return;

        bodyYaw = Mathf.Repeat(planeYaw, 360f);
        BodyYawDegrees = bodyYaw;

        if (driveCameraRigYaw && cameraRigTransform != null)
        {
            cameraRigTransform.Rotate(0f, deltaYaw, 0f, Space.World);
        }
    }

    // True in the non-hull attitude modes, where the yaw stick spins the drone(s)
    // directly (AttitudeAlgorithm feeds inputYawRate into desiredYawRate). The hull
    // modes instead zero the drone yaw and steer the view by the stick, so they keep
    // the input-integrated path above. Defaults to the input path if no SwarmManager.
    private bool IsBodyLockedToDrone()
    {
        SwarmManager sm = SwarmManager.Instance;
        if (sm == null) return false;
        SwarmManager.AttitudeAlgorithm algo = sm.GetSelectedAttitudeAlgorithm();
        return algo != SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL
            && algo != SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL;
    }

    // Advance the body heading by the followed drone's true per-frame yaw change and
    // rotate the rig by the same amount, so the pilot turns exactly with the drone
    // (no relative motion between the headset view and the feed). The followed drone
    // is the alive camera whose heading is closest to the current body yaw (the view
    // centre); on a drone swap we reseed rather than emit a phantom jump.
    private void UpdateBodyYawFromDrone()
    {
        Camera lockCam = GetBodyLockCamera();
        if (lockCam == null)
        {
            bodyLockInitialized = false; // nothing to follow this frame; hold heading
            return;
        }

        float droneYaw = lockCam.transform.eulerAngles.y;
        if (!bodyLockInitialized || lockCam != bodyLockCamera)
        {
            bodyLockCamera = lockCam;
            previousDroneYaw = droneYaw;
            bodyLockInitialized = true;
            return;
        }

        float deltaYaw = Mathf.DeltaAngle(previousDroneYaw, droneYaw);
        previousDroneYaw = droneYaw;
        if (deltaYaw == 0f) return;

        bodyYaw = Mathf.Repeat(bodyYaw + deltaYaw, 360f);
        BodyYawDegrees = bodyYaw;

        if (driveCameraRigYaw && cameraRigTransform != null)
        {
            cameraRigTransform.Rotate(0f, deltaYaw, 0f, Space.World);
        }
    }

    // The alive camera whose heading is closest to the current body yaw (the drone the
    // pilot is facing). In NONE every drone yaws together, so for a single drone this is
    // just that drone, and for a swarm it stays locked to the same centre drone.
    private Camera GetBodyLockCamera()
    {
        if (camerasToCapture == null || camerasToCapture.Count == 0) return null;

        int best = -1;
        float bestDiff = float.MaxValue;
        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (camerasToCapture[i] == null || !IsAlive(i)) continue;
            float diff = Mathf.Abs(Mathf.DeltaAngle(camerasToCapture[i].transform.eulerAngles.y, bodyYaw));
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = i;
            }
        }
        return best >= 0 ? camerasToCapture[best] : null;
    }

    // Snap the body heading to the current head (CenterEyeAnchor) world yaw. Used both to
    // seed the heading on the first frame and to recalibrate on demand (the calibrateKey),
    // so the panorama centre and the VR velocity frame re-align with wherever the pilot is
    // currently looking. The rig is deliberately not rotated here — only the reference
    // heading moves; CalibrateToCentre finishes the on-demand recentre.
    private void SeedBodyYawFromHead()
    {
        bodyYaw = headTransform != null ? headTransform.eulerAngles.y : 0f;
        bodyYawInitialized = true;
        BodyYawDegrees = bodyYaw;
    }

    // Finishes an on-demand calibration. The panorama stays snapped to the centre drone's yaw,
    // so to align it with the pilot we recentre the *view*: rotate the OVRCameraRig so the head's
    // forward points at the panorama centre (the screen's local +Z direction `dir`), then set the
    // body heading to centreYaw so the stitch selection and VR velocity frame stay in the camera-yaw
    // frame the rest of the system uses. The head look direction is 90 deg off from
    // world-yaw==centreYaw because of how the curved screen is oriented, so we align the head to
    // `dir` rather than to centreYaw itself. The view is only rotated when the rig is being driven
    // (driveCameraRigYaw); otherwise the heading still aligns without moving the view. Also
    // raises/lowers the rig so the pilot's eyes end up level with the curved screen.
    private void CalibrateToCentre(float centreYaw)
    {
        if (driveCameraRigYaw && cameraRigTransform != null && headTransform != null)
        {
            // Aim the head at the panorama's centre, not at the raw camera yaw. The
            // curved screen's centre column lies along the GameObject's local +Z (see
            // GenerateCurvedScreen: the mid vertex is at (0, y, radius)), and
            // UpdateCurvedScreenPose points that +Z at `dir`. So the pilot faces the
            // panorama centre exactly when the head's forward equals `dir` — which is
            // 90 deg off from world-yaw==centreYaw. Compute `dir` identically to
            // UpdateCurvedScreenPose (same frame, same centreYaw) and rotate the rig
            // by the signed horizontal angle from the current head forward onto it.
            float radians = -centreYaw * Mathf.Deg2Rad;
            Vector3 dir = new Vector3(Mathf.Cos(radians), 0f, Mathf.Sin(radians));

            Vector3 headForward = headTransform.forward;
            headForward.y = 0f;
            float delta = Vector3.SignedAngle(headForward, dir, Vector3.up);
            cameraRigTransform.Rotate(0f, delta, 0f, Space.World);
        }

        // Raise/lower the rig so the pilot's eyes sit level with the curved screen
        // (arena centre + screenHeightOffset, the same height used in
        // UpdateCurvedScreenPose). The head (centerEyeAnchor) tracks vertically
        // relative to the rig root, so shift the rig by the gap between the current
        // head height and the target screen height. A yaw rotation about world-up
        // never changes the head's Y, so reading it after the yaw recentre is safe.
        if (cameraRigTransform != null && headTransform != null)
        {
            FindArena();
            if (arena != null)
            {
                float screenY = arena.transform.position.y + screenHeightOffset;
                float headOffsetFromRig = headTransform.position.y - cameraRigTransform.position.y;
                Vector3 rigPos = cameraRigTransform.position;
                rigPos.y = screenY - headOffsetFromRig;
                cameraRigTransform.position = rigPos;
            }
        }

        bodyYaw = Mathf.Repeat(centreYaw, 360f);
        BodyYawDegrees = bodyYaw;
        bodyYawInitialized = true;
        WriteBodyYaw(bodyYaw);
    }

    // Lazily locate the Arena (same tag ScreenSpawn uses) for screen placement.
    private void FindArena()
    {
        if (arena == null)
        {
            arena = GameObject.FindGameObjectWithTag("Arena");
        }
    }

    // Selects the three drones whose FPV cameras straddle the body yaw: the
    // centre drone (camera yaw closest to bodyYaw) plus the yaw-neighbour on
    // each side. Mirrors the Python selection in StitcherThreading.py
    // (get_drone_order + get_subsets_from_order) so both sides agree on which
    // three views form the panorama. Returns the centre drone's yaw (used to
    // place the curved screen); 'selected' holds the camera indices written to
    // the three blocks, ordered [left, centre, right].
    private float SelectStitchCameras(float bodyYaw, out int[] selected)
    {
        if (camerasToCapture == null || camerasToCapture.Count == 0)
        {
            selected = new int[0];
            CentreStitchDrone = null;
            centreStitchCameraIndex = -1;
            return bodyYaw;
        }

        // Candidate set: alive boundary drones only (convex hull). If fewer than three
        // are on the boundary, fall back to all alive drones so the panorama still forms.
        List<int> candidates = new List<int>(camerasToCapture.Count);
        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (IsAlive(i) && IsBoundary(i)) candidates.Add(i);
        }
        if (candidates.Count < 3)
        {
            candidates.Clear();
            for (int i = 0; i < camerasToCapture.Count; i++)
            {
                if (IsAlive(i)) candidates.Add(i);
            }
        }

        int n = candidates.Count;

        // Centre = the candidate whose camera yaw is closest to bodyYaw.
        int centreCam = candidates[ClosestInList(candidates, bodyYaw)];

        // Publish the centre drone (the FPV camera's parent is the "Drone N" root, see DroneName)
        // so other systems can anchor on whatever the pilot is looking at — SwarmPlaneController
        // uses it to orient the vertical swarming plane.
        CentreStitchDrone = camerasToCapture[centreCam].transform.parent;
        // Also kept as a camera index, for the scene-plane raycast and intrinsics.
        centreStitchCameraIndex = centreCam;

        // Fewer than three candidates total: send what we have.
        if (n < 3)
        {
            selected = candidates.ToArray();
            return camerasToCapture[centreCam].transform.eulerAngles.y;
        }

        // Order candidates by yaw ascending (0..360); take centre + circular neighbours.
        candidates.Sort((a, b) =>
            camerasToCapture[a].transform.eulerAngles.y.CompareTo(
            camerasToCapture[b].transform.eulerAngles.y));

        int centrePos = candidates.IndexOf(centreCam);
        int leftPos = (centrePos - 1 + n) % n;
        int rightPos = (centrePos + 1) % n;

        selected = new int[] { candidates[leftPos], candidates[centrePos], candidates[rightPos] };
        return camerasToCapture[centreCam].transform.eulerAngles.y;
    }

    /// <summary>
    /// Picks the centre camera for a planar mosaic: the alive drone nearest the swarm's
    /// centroid, measured <i>in the plane the swarm is currently constrained to</i>.
    ///
    /// Deliberately not <see cref="SelectStitchCameras"/>'s "camera yaw closest to the body
    /// yaw" rule. In vertical-plane mode <c>AttitudeAlgorithm.ApplyPlaneModeAttitude</c> drives
    /// every drone to the anchor's heading, so yaw proximity becomes a near-tie broken by
    /// residual jitter and the centre drone changes almost every frame. That is not cosmetic:
    /// the scene-plane raycast is cast from this camera, so each swap steps the published plane
    /// offset, and <c>PlanarStitcher</c> builds the canvas frame and origin from the centre of
    /// the selection — the whole mosaic shifts. Geometric centrality has a unique answer under a
    /// shared heading; yaw proximity does not.
    ///
    /// Returns the centre camera's yaw, for the curved-screen orientation.
    /// </summary>
    private float SelectPlanarCentreCamera()
    {
        if (camerasToCapture == null || camerasToCapture.Count == 0)
        {
            CentreStitchDrone = null;
            centreStitchCameraIndex = -1;
            return bodyYaw;
        }

        // In-plane basis of the swarming plane: the vertical wall while plane mode is on, the
        // horizontal plane otherwise. GetPlaneAxes already returns (X, Z) for a horizontal
        // plane, so both configurations share one code path.
        Vector3 planeRight, planeUp;
        SwarmPlaneController swarmPlane = SwarmPlaneController.Instance;
        if (swarmPlane != null)
        {
            swarmPlane.GetPlaneAxes(out planeRight, out planeUp);
        }
        else
        {
            planeRight = Vector3.right;
            planeUp = Vector3.forward;
        }

        Vector3 centroid = Vector3.zero;
        int alive = 0;
        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (!IsAlive(i) || camerasToCapture[i] == null) continue;
            centroid += camerasToCapture[i].transform.position;
            alive++;
        }
        if (alive == 0)
        {
            CentreStitchDrone = null;
            centreStitchCameraIndex = -1;
            return bodyYaw;
        }
        centroid /= alive;

        // Distance is measured in-plane, not in 3D: in vertical-plane mode the swarm's spread
        // along the plane normal is formation error the restoring term is actively removing,
        // and letting it vote would hand the centre to whichever drone happens to be lagging
        // out of the wall.
        int best = -1;
        float bestDist = float.MaxValue;
        float incumbentDist = float.MaxValue;
        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (!IsAlive(i) || camerasToCapture[i] == null) continue;

            Vector3 rel = camerasToCapture[i].transform.position - centroid;
            float dist = new Vector2(Vector3.Dot(rel, planeRight),
                                     Vector3.Dot(rel, planeUp)).magnitude;
            if (dist < bestDist)
            {
                bestDist = dist;
                best = i;
            }
            if (i == centreStitchCameraIndex) incumbentDist = dist;
        }

        // Hysteresis: two drones straddling the centroid are otherwise still free to swap on
        // noise alone, which reintroduces exactly the flicker this method exists to remove.
        // Metres, not squared metres, so the inspector value means what it says.
        if (incumbentDist < float.MaxValue && incumbentDist <= bestDist + planarCentreHysteresis)
        {
            best = centreStitchCameraIndex;
        }

        centreStitchCameraIndex = best;
        CentreStitchDrone = camerasToCapture[best].transform.parent;
        return camerasToCapture[best].transform.eulerAngles.y;
    }

    /// <summary>
    /// Picks the cameras that contribute to a planar mosaic: every alive drone actually
    /// looking at the scene plane, closest-first, capped at the block count.
    ///
    /// Deliberately does <b>not</b> filter on <c>AttitudeAlgorithm.BoundaryEstimate</c>,
    /// which is the key difference from <see cref="SelectStitchCameras"/>. In
    /// vertical-plane mode the convex hull is the <i>rim of the wall</i>, so boundary
    /// drones are precisely the wrong subset -- every drone in the wall, interior
    /// included, is looking at the facade and contributes footprint. In nadir the hull
    /// is the outline of the formation and the interior drones tile the middle of the
    /// ground mosaic. The boundary rule exists because the radially-outward config nests
    /// interior views inside other views; that reasoning does not transfer.
    /// </summary>
    private void SelectPlanarStitchCameras(out int[] selected)
    {
        selected = new int[0];
        if (camerasToCapture == null || camerasToCapture.Count == 0) return;

        float cosObliquity = Mathf.Cos(maxObliquityDeg * Mathf.Deg2Rad);
        List<int> candidates = new List<int>(camerasToCapture.Count);
        List<float> distances = new List<float>(camerasToCapture.Count);

        Vector3 centreHit = Vector3.zero;
        bool haveCentreHit = false;
        if (centreStitchCameraIndex >= 0 && centreStitchCameraIndex < camerasToCapture.Count)
        {
            haveCentreHit = TryPlaneHit(camerasToCapture[centreStitchCameraIndex],
                                        out centreHit, out _);
        }

        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (!IsAlive(i)) continue;
            Camera cam = camerasToCapture[i];
            if (cam == null) continue;

            if (!TryPlaneHit(cam, out Vector3 hit, out float range)) continue;
            if (range > planarMaxRange) continue;

            // Reject grazing views: they contribute a long thin smear and blow up the
            // anisotropy gate on the Python side anyway.
            if (Mathf.Abs(Vector3.Dot(cam.transform.forward, scenePlaneNormal)) < cosObliquity)
                continue;

            candidates.Add(i);
            distances.Add(haveCentreHit ? Vector3.SqrMagnitude(hit - centreHit) : range);
        }

        if (candidates.Count == 0) return;

        // Over-subscribed: keep the views whose footprints cluster around the centre
        // one, so the mosaic is contiguous rather than a scattered set with holes.
        if (candidates.Count > blockImageCount)
        {
            int[] order = new int[candidates.Count];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            Array.Sort(order, (a, b) => distances[a].CompareTo(distances[b]));

            List<int> trimmed = new List<int>(blockImageCount);
            for (int i = 0; i < blockImageCount; i++) trimmed.Add(candidates[order[i]]);
            candidates = trimmed;
        }

        // Sort by camera index, NOT by yaw. The slot a drone occupies must be stable
        // frame to frame, or Python's busy-block cache serves one drone's frame in
        // another's slot. Python sorts by drone_id, so index order makes both agree.
        candidates.Sort();
        selected = candidates.ToArray();
    }

    /// <summary>
    /// Draws the scene plane, the centre camera's ray, and each selected view's
    /// footprint. Worth having: a raycast that latched onto a feed screen, an inverted
    /// plane normal or a selection that dropped the interior drones all look identical
    /// from Python (a wrong-looking mosaic) but are obvious here at a glance.
    /// </summary>
    private void OnDrawGizmosSelected()
    {
        if (!Application.isPlaying || typeOfStitcher != stitcherType.PLANAR) return;
        if (camerasToCapture == null || !scenePlaneInitialised) return;

        // Plane patch, drawn in the plane's own axes around the centre camera's hit.
        Vector3 origin = scenePlaneNormal * scenePlaneD;
        if (centreStitchCameraIndex >= 0 && centreStitchCameraIndex < camerasToCapture.Count)
        {
            Camera centre = camerasToCapture[centreStitchCameraIndex];
            if (centre != null)
            {
                Gizmos.color = scenePlaneValid ? Color.cyan : new Color(1f, 0.5f, 0f);
                Vector3 dir = (resolvedPlaneMode == ScenePlaneMode.Nadir)
                    ? Vector3.down : centre.transform.forward;
                if (TryPlaneHit(centre, out Vector3 hit, out _))
                {
                    Gizmos.DrawLine(centre.transform.position, hit);
                    Gizmos.DrawWireSphere(hit, 0.4f);
                    origin = hit;
                }
                else
                {
                    Gizmos.DrawRay(centre.transform.position, dir * fallbackPlaneDistance);
                }
            }
        }

        Vector3 right = Vector3.Cross(Vector3.up, scenePlaneNormal);
        if (right.sqrMagnitude < 1e-6f) right = Vector3.right;
        right.Normalize();
        Vector3 up = Vector3.Cross(scenePlaneNormal, right).normalized;

        float half = Mathf.Max(5f, planarMetresPerPixel * planarCanvasWidth * 0.5f);
        float halfV = Mathf.Max(5f, planarMetresPerPixel * planarCanvasHeight * 0.5f);
        Gizmos.color = scenePlaneValid
            ? new Color(0f, 1f, 1f, 0.8f) : new Color(1f, 0.5f, 0f, 0.8f);
        Vector3 a = origin + right * half + up * halfV;
        Vector3 b = origin - right * half + up * halfV;
        Vector3 c = origin - right * half - up * halfV;
        Vector3 d = origin + right * half - up * halfV;
        Gizmos.DrawLine(a, b); Gizmos.DrawLine(b, c);
        Gizmos.DrawLine(c, d); Gizmos.DrawLine(d, a);

        // Plane normal, so an inverted orientation is immediately visible.
        Gizmos.color = Color.magenta;
        Gizmos.DrawRay(origin, scenePlaneNormal * 3f);

        // Each selected view's footprint on the plane.
        if (selectedStitchIndices == null) return;
        foreach (int idx in selectedStitchIndices)
        {
            if (idx < 0 || idx >= camerasToCapture.Count) continue;
            Camera cam = camerasToCapture[idx];
            if (cam == null) continue;
            DrawFootprintGizmo(cam);
        }
    }

    // Back-projects the image corners onto the scene plane. Corners that miss (behind
    // the camera, or past the horizon) leave the quad open, which is the visual cue
    // that the view is being clipped.
    private void DrawFootprintGizmo(Camera cam)
    {
        Vector3[] corners = new Vector3[4];
        bool[] ok = new bool[4];
        Vector2[] viewport = { new Vector2(0, 0), new Vector2(1, 0),
                               new Vector2(1, 1), new Vector2(0, 1) };

        for (int i = 0; i < 4; i++)
        {
            Ray ray = cam.ViewportPointToRay(viewport[i]);
            float denom = Vector3.Dot(scenePlaneNormal, ray.direction);
            if (Mathf.Abs(denom) < 1e-6f) continue;
            float t = (scenePlaneD - Vector3.Dot(scenePlaneNormal, ray.origin)) / denom;
            if (t <= 0f || t > planarMaxRange) continue;
            corners[i] = ray.origin + ray.direction * t;
            ok[i] = true;
        }

        Gizmos.color = Color.green;
        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            if (ok[i] && ok[j]) Gizmos.DrawLine(corners[i], corners[j]);
        }
    }

    // Where a camera's principal ray meets the current scene plane.
    private bool TryPlaneHit(Camera cam, out Vector3 hit, out float range)
    {
        hit = Vector3.zero;
        range = 0f;

        Vector3 origin = cam.transform.position;
        Vector3 dir = cam.transform.forward;
        float denom = Vector3.Dot(scenePlaneNormal, dir);
        if (Mathf.Abs(denom) < 1e-6f) return false;          // parallel to the plane

        float t = (scenePlaneD - Vector3.Dot(scenePlaneNormal, origin)) / denom;
        if (t <= 0f) return false;                            // plane is behind the camera

        hit = origin + dir * t;
        range = t;
        return true;
    }

    /// <summary>
    /// Resolves the scene plane from a raycast off the centre stitch camera and
    /// low-passes it. Called every frame before the capture loop so the plane published
    /// this frame matches the frame's pose snapshots.
    /// </summary>
    private void UpdateScenePlane(int centreCamIdx)
    {
        if (camerasToCapture == null || centreCamIdx < 0 || centreCamIdx >= camerasToCapture.Count)
            return;
        Camera centre = camerasToCapture[centreCamIdx];
        if (centre == null) return;

        // Auto: the gimbal being pitched hard down is what actually distinguishes a
        // ground mosaic from a facade one. Deliberately not keyed off
        // SwarmPlaneController.PlaneModeActive — that describes the formation, not the
        // surface being imaged.
        resolvedPlaneMode = scenePlaneMode;
        if (resolvedPlaneMode == ScenePlaneMode.Auto)
        {
            resolvedPlaneMode = (FPVCameraScript.SharedPitch <= -60f)
                ? ScenePlaneMode.Nadir : ScenePlaneMode.Facade;
        }

        if (resolvedPlaneMode == ScenePlaneMode.Manual)
        {
            scenePlaneNormal = manualPlaneNormal.sqrMagnitude > 1e-6f
                ? manualPlaneNormal.normalized : Vector3.up;
            scenePlaneD = manualPlaneDistance;
            scenePlaneValid = true;
            scenePlaneInitialised = true;
            return;
        }

        Vector3 origin = centre.transform.position;
        // Nadir casts along world down rather than camera forward: at a -90 gimbal the
        // FPV camera's Slerp leaves forward only *almost* down, and world down keeps the
        // ground normal exactly world up instead of wobbling with the drone.
        Vector3 dir = (resolvedPlaneMode == ScenePlaneMode.Nadir)
            ? Vector3.down : centre.transform.forward;

        dir = dir.normalized;
        bool hit = RaycastScenePlane(origin, dir, out Vector3 hitNormal, out float hitDistance);

        // Normal: the hit surface, overridden where we have a better prior.
        Vector3 targetNormal;
        if (resolvedPlaneMode == ScenePlaneMode.Nadir)
        {
            targetNormal = Vector3.up;
        }
        else if (snapNormalToFormation && SwarmPlaneController.Instance != null
                 && SwarmPlaneController.Instance.PlaneModeActive)
        {
            targetNormal = SwarmPlaneController.Instance.PlaneNormal;
        }
        else if (hit)
        {
            targetNormal = hitNormal;
        }
        else
        {
            targetNormal = scenePlaneInitialised ? scenePlaneNormal : -dir;
        }

        // Orient toward the cameras, so the plane's "front" is unambiguous downstream.
        if (Vector3.Dot(targetNormal, dir) > 0f) targetNormal = -targetNormal;
        targetNormal.Normalize();

        // Offset: through the hit point, or out at the fallback distance on a miss.
        // A wrong distance is only a uniform scale error, so this degrades gracefully.
        float range = hit ? hitDistance : fallbackPlaneDistance;
        float targetD = Vector3.Dot(targetNormal, origin + dir * range);

        if (!scenePlaneInitialised || planeFilterTime <= 0f)
        {
            scenePlaneNormal = targetNormal;
            scenePlaneD = targetD;
            scenePlaneInitialised = true;
        }
        else
        {
            float alpha = 1f - Mathf.Exp(-Time.deltaTime / planeFilterTime);
            scenePlaneNormal = Vector3.Slerp(scenePlaneNormal, targetNormal, alpha).normalized;
            scenePlaneD = Mathf.Lerp(scenePlaneD, targetD, alpha);
        }
        scenePlaneValid = hit;
    }

    // Nearest hit that is actually scenery. The layer mask alone is not enough: drones
    // and feed screens can sit on default layers, and hitting one silently puts the
    // scene plane a few metres from the camera, which then looks exactly like a
    // geometry bug rather than a raycast bug.
    private bool RaycastScenePlane(Vector3 origin, Vector3 dir,
                                   out Vector3 normal, out float distance)
    {
        normal = Vector3.zero;
        distance = 0f;
        if (dir.sqrMagnitude < 1e-6f) return false;
        dir = dir.normalized;

        int count = Physics.RaycastNonAlloc(origin, dir, scenePlaneHits, planarMaxRange,
                                            scenePlaneMask, QueryTriggerInteraction.Ignore);
        bool found = false;
        float best = float.MaxValue;
        for (int i = 0; i < count; i++)
        {
            Transform t = scenePlaneHits[i].transform;
            if (t == null) continue;
            if (t.CompareTag("Screen") || t.CompareTag("DroneBase")) continue;
            if (t.root != null && t.root.CompareTag("DroneBase")) continue;
            if (t.IsChildOf(transform)) continue;   // this component's own curved screen
            if (scenePlaneHits[i].distance < best)
            {
                best = scenePlaneHits[i].distance;
                normal = scenePlaneHits[i].normal;
                found = true;
            }
        }
        distance = best;
        return found;
    }

    /// <summary>
    /// Pinhole intrinsics for the block resolution, taken from the camera's own
    /// projection matrix rather than from a field-of-view number.
    ///
    /// ScreenSpawn owns the FOV and aspect while this component owns the block
    /// resolution, so those two can disagree; the projection matrix cannot disagree
    /// with the pixels, because it is the matrix Unity rendered them with.
    /// </summary>
    private void DeriveIntrinsics(Camera camera, out float fx, out float fy,
                                  out float cx, out float cy)
    {
        Matrix4x4 P = camera.projectionMatrix;
        fx = P.m00 * blockImageWidth * 0.5f;
        fy = P.m11 * blockImageHeight * 0.5f;
        cx = (1f + P.m02) * blockImageWidth * 0.5f;
        cy = (1f - P.m12) * blockImageHeight * 0.5f;   // NDC +y is up, image row 0 is top

        float expected = (float)blockImageWidth / blockImageHeight;
        if (!intrinsicsAspectWarned && Mathf.Abs(camera.aspect - expected) > 1e-3f)
        {
            intrinsicsAspectWarned = true;
            Debug.LogWarning($"PyUniSharingFast: FPV camera aspect {camera.aspect:F4} disagrees " +
                             $"with the block resolution {blockImageWidth}x{blockImageHeight} " +
                             $"({expected:F4}). Re-run ScreenSpawn after changing block size, or " +
                             "the planar mosaic will be stretched.");
        }
    }
    private bool intrinsicsAspectWarned = false;

    // True when camera i's drone is on the swarm boundary (convex hull). Read live
    // each frame; Unity's null check covers a missing or destroyed AttitudeAlgorithm.
    private bool IsBoundary(int i)
    {
        return stitchAttitudes != null
            && i < stitchAttitudes.Count
            && stitchAttitudes[i] != null
            && stitchAttitudes[i].BoundaryEstimate;
    }

    // True when camera i's drone is still alive (DroneHealthMonitor parks dead drones
    // far below the course). A missing/destroyed StateFinder is treated as alive so a
    // setup gap never blanks the panorama. Read live each frame, like IsBoundary.
    private bool IsAlive(int i)
    {
        return stitchStates == null
            || i >= stitchStates.Count
            || stitchStates[i] == null
            || stitchStates[i].IsAlive;
    }

    // Local position within 'indices' whose camera yaw is closest to 'yaw' using
    // circular distance (matches the wraparound handling in
    // StitcherThreading.get_subsets_from_order).
    private int ClosestInList(List<int> indices, float yaw)
    {
        int best = 0;
        float bestDiff = float.MaxValue;
        for (int k = 0; k < indices.Count; k++)
        {
            float diff = Mathf.Abs(camerasToCapture[indices[k]].transform.eulerAngles.y - yaw);
            if (diff > 180f) diff = 360f - diff;
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = k;
            }
        }
        return best;
    }

    // Anchors the curved panorama screen at the Arena centre and rotates it so
    // its arc faces the centre drone's direction (mirrors the per-drone screen
    // placement in ScreenSpawn.UpdateRealDroneScreen). The screen therefore
    // snaps to a new direction only when the head turns far enough to change
    // the selected centre view.
    private void UpdateCurvedScreenPose(float centreYaw)
    {
        FindArena();
        if (arena == null) return;

        float radians = -centreYaw * Mathf.Deg2Rad;
        Vector3 dir = new Vector3(Mathf.Cos(radians), 0f, Mathf.Sin(radians));

        transform.position = arena.transform.position + Vector3.up * screenHeightOffset;
        transform.rotation = Quaternion.LookRotation(dir, Vector3.up);
    }

    // Lightweight per-frame write of just the body yaw into the metadata block
    // (WriteMetadata is not called every frame). The Python stitcher reads this
    // to choose the head-facing views, so it must match the yaw passed to
    // SelectStitchCameras (the body heading, not the live HMD direction).
    private void WriteBodyYaw(float yaw)
    {
        if (metadataPtr == IntPtr.Zero) return;
        WriteFloat(metadataPtr, metadataHeadYawOffset, yaw);
    }

    // Little-endian float32 into a shared-memory region. Python unpacks these with
    // struct '<f', so the byte order has to be explicit rather than inherited.
    private static void WriteFloat(IntPtr basePtr, int offset, float value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
        Marshal.Copy(bytes, 0, IntPtr.Add(basePtr, offset), 4);
    }

    /// <summary>
    /// Per-frame write of the scene plane and gimbal pitch, under a seqlock.
    ///
    /// The counter is bumped to an odd value before the payload and to the next even
    /// value after it, so a reader that sees an odd counter (or a different one either
    /// side of its read) knows the data was in flux and retries. Unlike the single
    /// scalars written by <see cref="WriteBodyYaw"/>, these fields are only meaningful
    /// together: a plane normal torn across a write is not unit length and not
    /// perpendicular to anything.
    /// </summary>
    private void WriteDynamicState()
    {
        if (metadataPtr == IntPtr.Zero) return;

        int seq = Marshal.ReadInt32(metadataPtr, metaDynSeqOffset);
        Marshal.WriteInt32(metadataPtr, metaDynSeqOffset, seq | 1);      // mark in-flux
        System.Threading.Thread.MemoryBarrier();

        WriteFloat(metadataPtr, metaPlaneNxOffset, scenePlaneNormal.x);
        WriteFloat(metadataPtr, metaPlaneNyOffset, scenePlaneNormal.y);
        WriteFloat(metadataPtr, metaPlaneNzOffset, scenePlaneNormal.z);
        WriteFloat(metadataPtr, metaPlaneDOffset, scenePlaneD);
        Marshal.WriteByte(metadataPtr, metaPlaneValidOffset, (byte)(scenePlaneValid ? 1 : 0));
        Marshal.WriteByte(metadataPtr, metaPlaneModeOffset, (byte)resolvedPlaneMode);
        WriteFloat(metadataPtr, metaGimbalPitchOffset, FPVCameraScript.SharedPitch);
        // droneId in the block header is the camerasToCapture index, so the centre camera
        // index is already in Python's id space -- no mapping table to keep in sync.
        Marshal.WriteInt32(metadataPtr, metaCentreDroneIdOffset, centreStitchCameraIndex);

        System.Threading.Thread.MemoryBarrier();
        Marshal.WriteInt32(metadataPtr, metaDynSeqOffset, (seq | 1) + 1); // stable again
    }

    // Reflects the current stitch selection in the Inspector (read-only). Only
    // rebuilds the list when the selection changes so it doesn't allocate every
    // frame. Entries are ordered [left, centre, right].
    private void UpdateStitchedDronesDisplay(int[] selected)
    {
        string key = string.Join(",", selected);
        if (key == lastStitchedDronesKey) return;
        lastStitchedDronesKey = key;

        stitchedDrones.Clear();
        for (int j = 0; j < selected.Length; j++)
        {
            string role = (selected.Length == 3)
                ? (j == 0 ? "L" : j == 1 ? "C" : "R")
                : $"#{j}";
            stitchedDrones.Add($"{role}: {DroneName(selected[j])}");
        }
    }

    // Tell ScreenSpawn which stitched drones' individual feeds to hide. Only hides
    // while the flag is on AND the panorama is actually displayed (panoramaDisplayActive);
    // otherwise pushes an empty set so all feeds show. Change-detected to avoid
    // per-frame allocation.
    private void UpdateStitchedScreenHiding(int[] selected)
    {
        if (screenSpawn == null) screenSpawn = FindObjectOfType<ScreenSpawn>();
        if (screenSpawn == null) return;

        bool hide = hideStitchedDroneScreens && panoramaDisplayActive;
        string key = hide ? string.Join(",", selected) : "off";
        if (key == lastHiddenScreensKey) return;
        lastHiddenScreensKey = key;

        if (!hide)
        {
            screenSpawn.SetStitchedDronesHidden(null);
            return;
        }

        var drones = new List<GameObject>(selected.Length);
        for (int j = 0; j < selected.Length; j++)
        {
            int camIdx = selected[j];
            if (camerasToCapture == null || camIdx < 0 || camIdx >= camerasToCapture.Count) continue;
            Camera cam = camerasToCapture[camIdx];
            if (cam == null) continue;
            Transform parent = cam.transform.parent;   // the "Drone N" GameObject
            if (parent != null) drones.Add(parent.gameObject);
        }
        screenSpawn.SetStitchedDronesHidden(drones);
    }

    // Human-readable name of the drone owning a capture camera (the FPV camera's
    // parent), for the Inspector display.
    private string DroneName(int camIdx)
    {
        if (camerasToCapture == null || camIdx < 0 || camIdx >= camerasToCapture.Count)
            return "?";
        Camera cam = camerasToCapture[camIdx];
        if (cam == null) return "?";
        Transform parent = cam.transform.parent;
        return parent != null ? parent.name : cam.name;
    }

    private void FindCameras()
    {
        camerasToCapture = new List<Camera>();
        stitchAttitudes = new List<AttitudeAlgorithm>();
        stitchStates = new List<StateFinder>();

        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");

        foreach (GameObject drone in drones)
        {
            Camera camera = drone.transform.Find("FPV")?.GetComponent<Camera>();

            if (camera != null)
            {
                camerasToCapture.Add(camera);
                // Cache this drone's boundary estimator (aligned with camerasToCapture)
                // so only convex-hull boundary drones are selected for stitching.
                Transform droneParent = drone.transform.Find("DroneParent");
                AttitudeAlgorithm attitude = droneParent?.GetComponent<AttitudeAlgorithm>();
                stitchAttitudes.Add(attitude);
                // Cache this drone's state too, so dead drones are excluded from stitching.
                StateFinder state = droneParent?.GetComponent<VelocityControl>()?.State;
                stitchStates.Add(state);
            }
            if(camerasToCapture.Count >maxBlockImageCount) break;
        }
    }

    // Rebuilds the (serialized) camerasToStitch bool list aligned with
    // camerasToCapture, from the cached boundary estimators. Coarse ~3 s snapshot
    // for the inspector; the live selection uses IsBoundary directly each frame.
    private void UpdateCameraToStitch()
    {
        int count = camerasToCapture != null ? camerasToCapture.Count : 0;
        camerasToStitch = new List<bool>(count);
        for (int i = 0; i < count; i++)
        {
            camerasToStitch.Add(IsBoundary(i));
        }
    }

    private void CalculateMemorySizes()
    {
        if(blockImageCount>maxBlockImageCount)
        {
            blockImageCount = maxBlockImageCount;
            Debug.LogError("Decrease number of drones or increase maxBlockImageCount constant. Value upperbounded at maxBlockImageCount.");
        }

        if(blockImageWidth>maxBlockWidth)
        {
            blockImageWidth = maxBlockWidth;
            Debug.LogError("Decrease dimensions of images or increase maxBlockWidth constant.");
        }

        if(blockImageHeight>maxBlockHeight)
        {
            blockImageHeight = maxBlockHeight;
            Debug.LogError("Decrease dimensions of images or increase maxBlockHeight constant.");
        }

        blockImageSize = blockImageWidth*blockImageHeight*3;
        blockSize = blockHeaderSize + blockImageSize;
        totalBlockSize = blockImageCount * blockSize;

        if(panoramaImageWidth>maxPanoramaWidth)
        {
            panoramaImageWidth = maxPanoramaWidth;
            Debug.LogError("Decrease dimensions of images or increase maxPanoramaWidth constant.");
        } 

        if(panoramaImageHeight>maxPanoramaHeight)
        {
            panoramaImageHeight = maxPanoramaHeight;
            Debug.LogError("Decrease dimensions of images or increase maxPanoramaHeight constant.");
        }

        panoramaImageSize = panoramaImageWidth * panoramaImageHeight * 3;
        totalPanoramaSize = panoramaDataPosition + panoramaImageSize;
    }

    private void CreateMemoryMaps()
    {
        // Only create block memory map if image writing is enabled
        if (enableImageWriting)
        {
            CreateBlockMap();
        }

        // Only create panorama memory map if panorama reading is enabled
        if (enablePanoramaReading)
        {
            panoramaFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)maxTotalPanoramaSize, panoramaMapName);
            if (panoramaFileMap != IntPtr.Zero)
            {
                panoramaPtr = MapViewOfFile(panoramaFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
                if (panoramaPtr == IntPtr.Zero)
                {
                    int errorCode = Marshal.GetLastWin32Error();
                    Debug.LogWarning($"Failed to map view of panorama file. Error Code: {errorCode}");
                }
            }
            else
            {
                Debug.LogWarning("Unable to create panorama memory-mapped file.");
            }
        }
    }

    // Creates (or recreates) the per-drone block mapping, sized exactly to
    // blockImageCount * blockSize so it matches what image_stream.py and
    // StitcherThreading.py allocate. All per-block flags are initialised to 0.
    private void CreateBlockMap()
    {
        DestroyBlockMap();

        if (blockImageCount <= 0 || totalBlockSize <= 0)
            return;

        blockFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)totalBlockSize, blockMapName);
        if (blockFileMap == IntPtr.Zero)
        {
            Debug.LogWarning("Unable to create block memory-mapped file.");
            return;
        }

        blockPtr = MapViewOfFile(blockFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        if (blockPtr == IntPtr.Zero)
        {
            int errorCode = Marshal.GetLastWin32Error();
            Debug.LogWarning($"Failed to map view of block file. Error Code: {errorCode}");
            return;
        }

        // Initialise every block: flag 0 (ready for the consumer) and droneId -1.
        //
        // The droneId matters as much as the flag. A fresh section is zero-filled, and
        // zero is a *legal* drone id -- so until a slot's first readback lands, the
        // consumer reads a ready block claiming to be drone 0 with an all-zero pose,
        // which on the PLANAR path is a degenerate quaternion rather than an empty
        // slot. -1 is the sentinel that already means "no view here", so say that from
        // the moment the section exists rather than from the first frame that fills it.
        for (int i = 0; i < blockImageCount; i++)
        {
            Marshal.WriteInt32(blockPtr, i * blockSize + blockFlagOffset, 0);
            Marshal.WriteInt32(blockPtr, i * blockSize + blockDroneIdOffset, -1);
            if (blockHeaderSize >= blockPoseHeaderSize)
            {
                Marshal.WriteInt32(blockPtr, i * blockSize + blockPoseStatusOffset, 0);
            }
        }
    }

    private void DestroyBlockMap()
    {
        if (blockPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(blockPtr);
            blockPtr = IntPtr.Zero;
        }
        if (blockFileMap != IntPtr.Zero)
        {
            CloseHandle(blockFileMap);
            blockFileMap = IntPtr.Zero;
        }
    }

    private void DestroyMemoryMaps()
    {
        if (blockPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(blockPtr);
            blockPtr = IntPtr.Zero;
        }
        
        if (panoramaPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(panoramaPtr);
            panoramaPtr = IntPtr.Zero;
        }
        
        if (metadataPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(metadataPtr);
            metadataPtr = IntPtr.Zero;
        }
        
        if (blockFileMap != IntPtr.Zero)
        {
            CloseHandle(blockFileMap);
            blockFileMap = IntPtr.Zero;
        }
        
        if (panoramaFileMap != IntPtr.Zero)
        {
            CloseHandle(panoramaFileMap);
            panoramaFileMap = IntPtr.Zero;
        }

        if (metadataFileMap != IntPtr.Zero)
        {
            CloseHandle(metadataFileMap);
            metadataFileMap = IntPtr.Zero;
        }
    }

    private void OnValidate()
    {
        if(hasStarted)
        {
            CalculateMemorySizes();
            
            if (enableImageWriting || enablePanoramaReading)
            {
                WriteMetadata();
            }

            // Update reusable resources for image writing
            if (enableImageWriting)
            {
                reusableTexture = new RenderTexture(blockImageWidth, blockImageHeight, 24);
                image = new Texture2D(blockImageWidth, blockImageHeight, TextureFormat.RGB24, false);
                blockImageBytes = new byte[blockImageSize];
                EnsureConvertedBlockBuffer();
                CreateBlockMap();
            }

            // Update reusable resources for panorama reading
            if (enablePanoramaReading)
            {
                panoTexture = new Texture2D(panoramaImageWidth, panoramaImageHeight, TextureFormat.RGB24, false);
                if (curvedScreenMaterial != null)
                {
                    curvedScreenMaterial.mainTexture = panoTexture;
                    curvedScreenMaterial.SetTexture("_EmissionMap", panoTexture);
                }
            }
        }
    }

    private void ValidateTextures()
    {
        if (!enableImageWriting) return;

        if (reusableTexture == null || reusableTexture.width != blockImageWidth || reusableTexture.height != blockImageHeight)
        {
            reusableTexture?.Release();
            reusableTexture = new RenderTexture(blockImageWidth, blockImageHeight, 24);
        }

        if (image == null || image.width != blockImageWidth || image.height != blockImageHeight)
        {
            Destroy(image);
            image = new Texture2D(blockImageWidth, blockImageHeight, TextureFormat.RGB24, false);
        }
    }

    private void UpdateCameras()
    {
        // Re-discovery keeps CentreStitchDrone live for non-stitching consumers, so it runs
        // regardless; only the shared-memory resize below depends on image writing.
        FindCameras();
        UpdateCameraToStitch();

        if (!enableImageWriting) return;

        int newblockImageCount = DesiredBlockCount();
        if (newblockImageCount != blockImageCount)
        {
            blockImageCount = newblockImageCount;
            CalculateMemorySizes();
            CreateBlockMap();          // resize the mapping to the new drone count
            WriteMetadata();
            blockImageBytes = new byte[blockImageSize];
            EnsureConvertedBlockBuffer();
            EnsureReadbackPool();      // more slots per send needs more in-flight entries
            ValidateTextures();
        }
    }

    private void WriteMetadata()
    {
        if (metadataPtr == IntPtr.Zero)
        {
            Debug.LogError($"Problem with metadata memory.");
            return;
        }

        int offset = 0;

        Marshal.WriteInt32(metadataPtr, offset, blockImageWidth);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, blockImageHeight);
        offset += 4;
        Debug.LogWarning(blockImageCount);
        Marshal.WriteInt32(metadataPtr, offset, blockImageCount);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, panoramaImageWidth);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, panoramaImageHeight);
        offset += 4;

        byte[] stringBytes = Encoding.UTF8.GetBytes(typeOfStitcher.ToString());
        byte[] stringBuffer = new byte[64];
        Array.Copy(stringBytes, stringBuffer, Math.Min(stringBytes.Length, stringBuffer.Length));
        Marshal.Copy(stringBuffer, 0, IntPtr.Add(metadataPtr, offset), stringBuffer.Length);
        offset += 64;

        Marshal.WriteByte(metadataPtr, offset, (byte)(cylindrical ? 1 : 0));
        offset +=1;

        byte[] stringBytesMatcher = Encoding.UTF8.GetBytes(typeOfMatcher.ToString());
        byte[] stringBufferMatcher = new byte[64];
        Array.Copy(stringBytesMatcher, stringBufferMatcher, Math.Min(stringBytesMatcher.Length, stringBufferMatcher.Length));
        Marshal.Copy(stringBufferMatcher, 0, IntPtr.Add(metadataPtr, offset), stringBufferMatcher.Length);
        offset += 64;

        Marshal.WriteByte(metadataPtr, offset, (byte)(ransac ? 1 : 0));
        offset +=1;

        Marshal.WriteInt32(metadataPtr, offset, checks);
        offset += 4;

        byte[] ratioThreshBytes = BitConverter.GetBytes(ratio_thresh);
        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(ratioThreshBytes);
        }
        Marshal.Copy(ratioThreshBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        byte[] scoreThresholdBytes = BitConverter.GetBytes(score_threshold);
        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(scoreThresholdBytes);
        }
        Marshal.Copy(scoreThresholdBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        Marshal.WriteInt32(metadataPtr, offset, focal_length);
        offset += 4;

        // Reserved: this byte carried the retired NIS stitcher's onlyIHN flag. It is still
        // written (as 0) rather than removed because every field after it -- including
        // metadataHeadYawOffset -- is addressed by a position Python reaches by reading
        // sequentially, so dropping the byte would shift the whole v1 prefix.
        Marshal.WriteByte(metadataPtr, offset, 0);
        offset += 1;

        byte[] fusionModeBytes = Encoding.UTF8.GetBytes(typeOfFusion.ToString());
        byte[] fusionModeBuffer = new byte[64];
        Array.Copy(fusionModeBytes, fusionModeBuffer, Math.Min(fusionModeBytes.Length, fusionModeBuffer.Length));
        Marshal.Copy(fusionModeBuffer, 0, IntPtr.Add(metadataPtr, offset), fusionModeBuffer.Length);
        offset += 64;

        Marshal.WriteInt32(metadataPtr, offset, blurKernelSize);
        offset += 4;

        byte[] blurSigmaBytes = BitConverter.GetBytes(blurSigma);
        if (!BitConverter.IsLittleEndian) Array.Reverse(blurSigmaBytes);
        Marshal.Copy(blurSigmaBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        Marshal.WriteInt32(metadataPtr, offset, borderSize);
        offset += 4;

        // StabStitch panorama-quality fallback parameters
        Marshal.WriteByte(metadataPtr, offset, (byte)(qualityFallbackEnabled ? 1 : 0));
        offset += 1;

        byte[] qualityThresholdBytes = BitConverter.GetBytes(qualityThreshold);
        if (!BitConverter.IsLittleEndian) Array.Reverse(qualityThresholdBytes);
        Marshal.Copy(qualityThresholdBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        // Body yaw (pilot heading). Also written every frame by WriteBodyYaw at the
        // same offset; seeded here at the head's initial yaw so the start-time full
        // write matches the first body heading (UpdateBodyYaw seeds it identically).
        byte[] headYawBytes = BitConverter.GetBytes(headTransform != null ? headTransform.eulerAngles.y : 0f);
        if (!BitConverter.IsLittleEndian) Array.Reverse(headYawBytes);
        Marshal.Copy(headYawBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        // Console-verbosity toggle: gates Python's per-loop stitch/warp rate (Hz) prints.
        Marshal.WriteByte(metadataPtr, offset, (byte)(printStitchRate ? 1 : 0));
        offset += 1;

        // ---- Wire v2 tail ---------------------------------------------------------
        // Everything below is addressed by explicit constant rather than by the running
        // offset, because the dynamic block is also written per-frame by
        // WriteDynamicState and the two must agree on where each field lives.
        Debug.Assert(offset <= metadataTailStart,
                     $"metadata v1 prefix ended at {offset}, overrunning the v2 tail at {metadataTailStart}");

        Marshal.WriteInt32(metadataPtr, metaBlockHeaderSizeOffset, ActiveBlockHeaderSize());
        Marshal.WriteInt32(metadataPtr, metaWireVersionOffset, metaWireVersion);

        // Intrinsics from any FPV camera: SpawnScreens configures them identically, and
        // DeriveIntrinsics warns if the aspect has drifted from the block resolution.
        float fx = 0f, fy = 0f, cx = 0f, cy = 0f;
        if (camerasToCapture != null)
        {
            for (int i = 0; i < camerasToCapture.Count; i++)
            {
                if (camerasToCapture[i] == null) continue;
                DeriveIntrinsics(camerasToCapture[i], out fx, out fy, out cx, out cy);
                break;
            }
        }
        WriteFloat(metadataPtr, metaFxOffset, fx);
        WriteFloat(metadataPtr, metaFyOffset, fy);
        WriteFloat(metadataPtr, metaCxOffset, cx);
        WriteFloat(metadataPtr, metaCyOffset, cy);

        Marshal.WriteInt32(metadataPtr, metaPlanarCanvasWidthOffset, planarCanvasWidth);
        Marshal.WriteInt32(metadataPtr, metaPlanarCanvasHeightOffset, planarCanvasHeight);
        WriteFloat(metadataPtr, metaPlanarMetresPerPixelOffset, planarMetresPerPixel);
        WriteFloat(metadataPtr, metaPlanarMaxRangeOffset, planarMaxRange);
        Marshal.WriteInt32(metadataPtr, metaPlanarFeatherPxOffset, planarFeatherPx);
        WriteFloat(metadataPtr, metaPlanarAnisoMaxOffset, planarAnisoMax);
        WriteFloat(metadataPtr, metaPlanarMinCoverageOffset, planarMinCoverage);
        Marshal.WriteByte(metadataPtr, metaPlanarPoseSourceOffset, (byte)poseSource);
        Marshal.WriteByte(metadataPtr, metaPlanarPsnrGateOffset,
                          (byte)(planarPsnrGateEnabled ? 1 : 0));
        Marshal.WriteByte(metadataPtr, metaPlanarBlendModeOffset, (byte)planarBlendMode);
        Marshal.WriteByte(metadataPtr, metaPlanarDebugViewOffset, (byte)planarDebugView);

        Marshal.WriteByte(metadataPtr, metaPlanarPlaneSweepOffset,
                          (byte)(planarPlaneSweep ? 1 : 0));
        Marshal.WriteByte(metadataPtr, metaPlanarPoseRefineOffset,
                          (byte)(planarPoseRefine ? 1 : 0));
        WriteFloat(metadataPtr, metaPlanarSweepRangeOffset, planarSweepRange);
        Marshal.WriteInt32(metadataPtr, metaPlanarSweepStepsOffset, planarSweepSteps);
        WriteFloat(metadataPtr, metaPlanarRefineRateOffset, planarRefineRate);
        WriteFloat(metadataPtr, metaPlanarRefineMaxShiftOffset, planarRefineMaxShift);

        // Seed the dynamic block so Python never reads an uninitialised plane before
        // the first Update tick.
        WriteDynamicState();

        if(hasStarted) return;
        offset = metadataTailEnd + metadataReservedGap;

        Marshal.WriteInt32(metadataPtr, offset, maxTotalBlockSize);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, maxTotalPanoramaSize);
        offset += 4;
        Debug.Assert(offset == metadataSize,
                     $"metadata write ended at {offset}, but metadataSize is {metadataSize}");
    }

    private void CheckExistingMapping(string mapName)
    {
        IntPtr existingMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, mapName);
        if (existingMap != IntPtr.Zero)
        {
            System.Threading.Thread.Sleep(100);

            IntPtr secondCheck = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, mapName);
            if (secondCheck != IntPtr.Zero)
            {
                Debug.LogError($"Memory map '{mapName}' still exists after closing the handle.");
                CloseHandle(secondCheck);
            }
        }
    }

    void OnDestroy()
    {
        // Flush in-flight readbacks before unmapping so a late completion
        // callback can never touch a dead pointer, then free the job buffer.
        AsyncGPUReadback.WaitAllRequests();
        DestroyMemoryMaps();
        if (convertedBlock.IsCreated) convertedBlock.Dispose();
    }

    void OnApplicationQuit()
    {
        AsyncGPUReadback.WaitAllRequests();
        DestroyMemoryMaps();
        Debug.Log("Application quitting. Memory maps destroyed.");
    }
}