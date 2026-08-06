# CLAUDE.md

Unity project for flying an aerial **swarm of drones** with a **VR (Meta/OVR) interface**. Each drone
has an FPV camera; the head-facing subset of feeds is streamed to a **Python real-time image-stitching**
pipeline (StabStitch++) over **memory-mapped files**, and the panorama is rendered back onto a curved VR
screen. See [AGENTS.md](AGENTS.md) for a per-file index.

Two largely independent subsystems — touching one rarely affects the other:

```
                       Unity (C#)                         |        Python
  ┌──────────────────────────────────────────────┐       |  ┌─────────────────────────┐
  swarm flight control          VR display / capture      |   real-time stitching
  SwarmManager → SwarmAlgorithm  PyUniSharingFast  ──MMF──>|   StitcherThreading
   → OlfatiSaber/Reynolds         screenSpawn     <─MMF────|    → StabStitcher (StabStitch++)
   → AttitudeAlgorithm                                     |    → PlanarStitcher (pose-driven)
   → VelocityControl                                       |       → planar_geometry
  └──────────────────────────────────────────────┘       |  └─────────────────────────┘
```

## Shared-memory contract (keep C# and Python in sync)

Four Windows named memory maps. **If you change a layout/offset, change both
`Assets/Scripts/ImageStitching/PyUniSharingFast.cs` and
`Assets/Scripts/ImageStitching/StitcherThreading.py`** (`readMetadataMemory`/`WriteMetadata`).
Every block-flag handshake is strictly **one producer + one consumer** — never add a second
reader/writer to a map; that's why the feed and stitch maps are separate.

- `MetadataSharedMemory` — sizes + stitcher config + blur/border + quality settings + **body/pilot yaw**
  (yaw also rewritten every frame at a fixed offset). This is the integrated *body heading*
  (`PyUniSharingFast.bodyYaw`): seeded from the HMD's initial yaw, then advanced only by the controller
  yaw-rate command — **not** live HMD direction, so head-look doesn't move the panorama.
  It also carries a seqlocked **dynamic block** (scene plane, gimbal pitch, planar centre drone id,
  planar canvas zoom/pan) — see the PLANAR section. Its total size is pinned at 412 bytes: new tail
  fields come out of `metadataReservedGap`, because changing the size would strand an already-running
  Python. The static tail's padding filled up at 311, so the estimator settings (344–367) sit *after*
  the dynamic block, and the canvas zoom/pan/mode (368–380) after those — the zoom/pan are **seqlocked
  despite not being contiguous with the block**, since pan is an offset from the origin the centre
  drone defines. Both sides address every tail field by absolute offset, so the ordering is cosmetic,
  but it is why `readMetadataMemory` and `read_dynamic_state` each seek more than once rather than
  reading straight through.
- `BlockSharedMemory` — the **stitcher input**, one block per selected drone. `flag` is the handshake
  (0 = ready, 1 = busy). Images are **BGR, top-down**. Sole consumer: `StitcherThreading.py`. Sole
  producer: sim = `PyUniSharingFast`; real-drone mode (DJIScene) = `ImageSharing.cs` (so keep
  `enableImageWriting` off on `PyUniSharingFast` there). Unity publishes the header size as
  `blockHeaderSize` in metadata and Python reads that rather than assuming, so the size is declared in
  exactly one place even though two components write blocks:
  - **v1, 12 bytes** — `int flag | int droneId | float heading | RGB24 image`. Historical; **no
    producer writes it any more**. Kept named because the constant is what `check_wire_layout.py`
    asserts against, and because a v1 producer's blocks are still *readable* — they simply arrive with
    `poseStatus == 0` and are dropped by the planar solve.
  - **v2, 48 bytes** — appends `float camPos[3] | float camRot[4] (xyzw) | float captureTime |
    int poseStatus`, all in **Unity world / left-handed**. Required by `PLANAR`, written by **both**
    producers. The pose lives in the block, not in metadata, because it must be the pose of *that*
    frame: it is snapshotted in `RequestBlockCapture` 1–2 frames before the readback completes, and
    Python may re-serve a cached block, which then needs its own pose. On the real-drone path the same
    argument is why the pose is computed in the DJI producer's `frame_sink` — the image bytes and the
    telemetry come out of one `ds_wrapper` fetch there, which is the tightest pairing available.
  - `captureTime` is **seconds since the producer started, not a wall clock**. The field is float32, in
    which `time.time()` (~1.75e9) has ~128 s of resolution; it is only ever read as a difference
    (`MAX_CAPTURE_SKEW_S`), so a since-start clock is both sufficient and the only one that works.

  Slot count is `blockImageCount` (metadata): 3 for the left/centre/right stitchers, up to
  `maxStitchViews` for `PLANAR`. It is sized from the *camera count*, never the per-frame selection —
  `CreateBlockMap` recreates the named section, and Python holds a single mapping of it. Slots the
  selection doesn't reach are marked `droneId == -1` — **including at creation**, because a fresh
  section is zero-filled and `0` is a legal drone id: until its first readback lands, a never-written
  slot otherwise advertises itself as a ready block from drone 0 carrying an all-zero pose.
  The readback pool is sized to two full batches (`EnsureReadbackPool`) for the same reason: with a
  pool smaller than `blockImageCount`, `AcquirePendingSlot` starves the *same* tail slots every send,
  so they are never written at all rather than merely late.

  **`PyUniSharingFast` publishes `blockImageCount` + `blockHeaderSize` even when it is not the
  producer**, because Python sizes its mapping from them and only this component writes metadata.
  In the DJI scene the section is created by `ImageSharing.cs`, so `DesiredBlockCount()` **reads that
  component's `StitchSlotCount`** rather than assuming a number, and skips the `camerasToCapture` clamp
  — that scene has no sim FPV cameras, and clamping advertises 0 blocks, which makes Python map none of
  the section and the real-drone panorama silently never appear. `ImageSharing.stitchSlots` is a
  serialized field, not a constant, because `PLANAR` mosaics every drone that sees the facade while
  `STABSTITCH` takes exactly 3; raise it to the fleet size before flying `PLANAR`.
  `create` and `describe` being split across two files that never reference each other is the
  hazard; `tools/check_wire_layout.py` asserts the header size and the LRC count, and — when the
  `DJI_Swarm` repo is checked out beside this one — the feed header across both repos too.
- `DroneFeedSharedMemory` — **all real-drone feeds** (DJIScene only), same per-block layout as above but
  a fixed capacity of **10 blocks** indexed by zero-based drone id (must match `MAX_DRONES` in the
  DJI_Swarm repo's `image_stream_feed.py`, which is the producer). Consumer: `ImageSharing.cs`, which
  displays the feeds and re-publishes the 3 body-yaw-selected views into `BlockSharedMemory`
  (`PublishStitchBlocks`). Unity marks unwritten **and already-consumed** blocks with `droneId == -1`
  (the producer rewrites `droneId` every write) — that marker is the new-frame detection, since the
  flag alone can't distinguish a fresh frame from a re-read.
- `PanoramaSharedMemory` — `int flag | int quality_ok | RGB24 panorama`. `quality_ok == 0` ⇒ Unity shows
  individual feeds instead of the panorama. Panorama is **vertically flipped and converted BGR→RGB** by
  Python (Unity textures start bottom-left; Unity uploads the bytes straight into an RGB24 texture).

## Conventions & invariants (not enforced by code)

- **Coupled 3-view selection** (all stitchers *except* `PLANAR`)**:** the body-yaw-relative
  left/centre/right pick exists in C# (`PyUniSharingFast.SelectStitchCameras` for sim,
  `ImageSharing.PublishStitchBlocks` for real drones) and in Python (`get_drone_order` +
  `get_subsets_from_order`) and all three must agree. Those two Python functions are
  left/centre/right-only — `PLANAR` does not use them.
- **Planar selection is C#-only:** `PyUniSharingFast.SelectPlanarStitchCameras` picks the views and
  Python consumes that selection rather than re-deriving it, so there is only one rule to keep in sync.
  It deliberately does **not** filter on `BoundaryEstimate`: in vertical-plane mode the convex hull is
  the *rim of the wall*, so boundary drones are exactly the wrong subset (every drone in the wall sees
  the facade), and in nadir the interior drones tile the middle of the mosaic. The boundary rule exists
  because the radially-outward config nests interior views inside other views.
- **The planar centre drone is also C#-only**, and picked by a different rule from every other stitcher:
  `SelectPlanarCentreCamera` takes the alive drone nearest the swarm centroid *measured in the swarming
  plane* (`SwarmPlaneController.GetPlaneAxes`), with metre-valued hysteresis. It is **not**
  `SelectStitchCameras`'s "camera yaw closest to body yaw" rule, which is only meaningful for the
  radially-outward ring: in vertical-plane mode `AttitudeAlgorithm` drives every drone to the anchor's
  heading, so yaw proximity is a tie broken by jitter and the centre changes almost every frame. That
  matters because the scene-plane raycast originates at this camera and `PlanarStitcher` frames the
  canvas on it — a flickering centre both steps the published plane offset and slides the mosaic.
- **In vertical-plane mode the body/rig yaw is slaved to the plane, not to the yaw stick**
  (`PyUniSharingFast.UpdateBodyYawFromPlane`). The stick already spins the anchor drone and the rest
  of the wall follows it, so also integrating that stick into `bodyYaw` walks the view off the wall —
  different gain, none of the drone's lag. Invisible in the radially-outward ring (the panorama
  re-snaps to the nearest camera), but under a shared heading `bodyYaw` is what aims the VR velocity
  frame at the wall. The lock is absolute, so entering plane mode also clears any pre-existing offset.
- **Toggling the swarming plane also switches the stitcher and the screen layout**
  (`SwarmPlaneController.ApplyDisplayConfiguration`): vertical ⇒ `PLANAR` + `FORMATION_WALL`,
  horizontal ⇒ `STABSTITCH` + `OUTER_CIRCLE`. Neither pairing is taste — a wall is one dominant plane
  where the pose-driven homographies are exact and the shared heading collapses `OUTER_CIRCLE` onto a
  single arc position, and the ring is the parallax-heavy case StabStitch++ exists for. It fires only
  on an actual mode change, so the inspector's choices stand until the first toggle, and
  `driveDisplayConfiguration` turns it off for comparing two stitchers on one formation. Both setters
  go through the owning component (`PyUniSharingFast.SetStitcherType`,
  `InterfaceManager.SetScreenStyle`) rather than writing the fields: the stitcher switch has to resize
  `BlockSharedMemory` (`PLANAR` wants `maxStitchViews` slots, the others 3) and republish metadata for
  Python to pick up, and the layout has to stay owned by `InterfaceManager` per the source-of-truth rule
  below. The resize is a no-op in the DJI scene, where the slot count comes from
  `ImageSharing.stitchSlots` instead — raise that by hand before flying `PLANAR` there.
- **Boundary drones** = `AttitudeAlgorithm.BoundaryEstimate` (convex-hull). Left/centre/right stitching
  and the `OUTER_CIRCLE` screen layout only use boundary drones (see the planar exception above).
- **`OUTER_CIRCLE` is only meaningful for the radially-outward ring**, and `ScreenStyle.FORMATION_WALL`
  is its shared-heading counterpart (vertical plane, or nadir). Placing each screen at its own drone's
  yaw works only because the ring *spreads* the yaws; under a shared heading every screen lands on the
  same arc position and they stack. `FORMATION_WALL` keeps yaw as the thing that aims the display — the
  circular mean of the yaws sets one azimuth for the wall, so turning the formation turns the wall — and
  takes the *separation* from each drone's rank inside the swarming plane (`SwarmPlaneController.
  GetPlaneAxes`, the same basis the planar centre-drone rule uses). Consequences worth knowing:
  - **It ranks into a grid rather than scaling the true in-plane coordinates.** Proportional placement
    preserves the formation's shape but guarantees nothing about spacing — two drones a metre apart in a
    40 m wall still overlap — whereas ranking is non-overlapping by construction, and the reading that
    actually matters ("that feed is the drone up and to the left") survives either way.
  - **The whole grid is solved once per frame, before any screen is placed**, and over exactly the set
    about to be shown (`IsFeedSuppressed`) — a cell index only means something relative to the others,
    so a screen hidden into the panorama has to leave its cell rather than hold a gap.
  - Unlike `OUTER_CIRCLE` it does **not** gate on `BoundaryEstimate`, for the same reason
    `SelectPlanarStitchCameras` doesn't: in a wall the hull is the rim, and every drone in it is looking
    at the facade.
  - **Non-overlap is geometric, not a tuned constant:** the column pitch is the angle whose chord at
    `radius` is one padded screen width, the row pitch one padded screen height, both recomputed each
    frame from the live `scale`. `formationWallMaxSpanDeg` (default 120°) then caps how far the wall may
    wrap, and overflow goes into extra rows — a screen beside the pilot's ear carries information in the
    ring (a drone behind you) but none here. Lowering `scale` is what buys more columns.
  - The auto grid estimates the **row** count from the formation's aspect and divides to get the columns.
    Rounding the columns directly overshoots: a 5×2 wall reads as aspect 4, and `round(√(10·4))` is 6
    columns, splitting ten drones 6/4 across rows that are really 5 and 5.
- **`InterfaceManager.screenStyle` is the single source of truth for the layout, and the panorama
  fallback may not overwrite it.** `PyUniSharingFast.fallbackScreenStyle` is a substitute for a layout
  that shows *nothing* — `ScreenSpawn.ShowFallbackFeeds` only applies it when the configured style is
  `OFF`, and otherwise keeps the configured one. Overwriting it unconditionally was silently
  layout-changing (toggle the panorama off and `FORMATION_WALL` feeds came back as `OUTER_CIRCLE`) and
  it desyncs the two components: InterfaceManager pushes its style *into* `ScreenSpawn.screenStyle`
  (which is `[HideInInspector]`), so the inspector goes on reading the configured style while the
  screens are in the fallback's, and only nudging the style in the inspector pushes it down again.
  For the same reason the restore reads InterfaceManager's field rather than a snapshot taken when the
  fallback engaged, and `OnInterfaceParamsChanged` re-applies the substitution *after* the push.
- Image format across the bridge is **BGR + top-down** for stitch inputs; the returned panorama is
  flipped once and converted to RGB on the Python side.
- **Resolution is metadata-driven:** `StitcherThreading.py` sizes inputs/outputs from the Unity metadata
  (`blockImageWidth/Height` + `panoramaImageWidth/Height` in `PyUniSharingFast`'s inspector); set those to
  scale resolution. The StabStitch nets always run at a fixed `NET_W×NET_H`, so only the render + bridge
  costs grow with resolution — not the warp pipeline.
- **STABSTITCH render/warp are decoupled:** a ~21 Hz render loop uses cached warp params only (no neural
  net); a separate ~6.6 Hz thread runs the nets and updates the cache. The render warp is a single
  `grid_sample` over a **precomputed TPS sampling field** (`_compute_tps_flow`, cached per warp update) —
  the float64 TPS solve + per-pixel RBF live in the warp thread, not the render loop.
- **The two threads share one GPU and one GIL**, so wasted render work directly slows the warp update.
  The render signal is rate-limited to `RENDER_MIN_PERIOD` in `StitcherThreading.py`: the shared memory
  is polled much faster than Unity refills it, and re-rendering an unchanged frame measurably halved the
  warp rate. Keep that pacing just above Unity's `sendInterval` (20 Hz) rather than removing it.
- **Never use `torchvision.transforms.GaussianBlur` on canvas-sized tensors here** — it convolves with a
  dense k×k kernel. At 1690×653 the 41×41 blur cost 125 ms and the 21×21 blur 38 ms, which was the single
  largest cost in the warp update. Use `SeparableGaussianBlur` in `StabStitcher.py` (two 1-D passes,
  same result to ~5e-5 relative). Likewise erode with two 1-D `max_pool2d` passes, not one k×k pass.

## PLANAR stitcher (pose-driven, for the plane configurations)

Selected via `typeOfStitcher = PLANAR`. Intended for the **vertical-plane / facade** configuration
(`SwarmPlaneController`, key `V`) and the **nadir** one (`SwarmManager.gimbalPitch = -90`). In both the
scene is one dominant plane and parallax is structurally absent, so **one homography per view is exact**
and computable from intrinsics + pose + the plane with no image content — hence no feature matching, no
neural net, and no failure on asphalt/water/uniform facades. It is an *alternative* to `STABSTITCH`, not
an upgrade: StabStitch++'s parallax-tolerant TPS warps are what make the radially-outward config work.

- **`G = K R [e1 | e2 | (O − C)]`** (`planar_geometry.build_G`), not the textbook
  `K(R − t nᵀ/d)K⁻¹` — no reference view, no `d` in a denominator, nothing to invert. Its **third
  component is camera depth in metres**, so behind-camera rejection, max-range clipping and the sampling
  coordinate all come out of one batched matmul.
- **All handedness lives in `planar_geometry.unity_pose_to_cv` / `unity_dir_to_rh`.** Unity is
  left-handed Y-up; CV is right-handed with +Y down and image row 0 at the top. Every world quantity
  (plane normal, origin, basis) must go through the same conversion. This is the highest-risk area —
  `tools/planar_selftest.py` cross-checks it against an independent implementation derived from Unity's
  documented `worldToCameraMatrix`, which never calls into `planar_geometry`, so a shared sign error
  cannot cancel out.
- **Every block's pose is validated before use** (`PlanarStitcher.pose_is_usable`: `POSE_VALID` set,
  finite, non-degenerate quaternion) and a failing view is *dropped*, not raised on. This is a
  shared-memory handshake with no schema enforcement, so a stale or unwritten block must cost its own
  view rather than the frame — an uncaught `degenerate quaternion` out of `quat_to_matrix` kills the
  panorama and points the traceback at the geometry instead of at the producer. Dropped views are
  counted as `unposed` on the periodic `[PLANAR]` line.
- **The scene plane is a raycast** from the centre stitch drone (`UpdateScenePlane`), low-passed, sent
  as unit normal + offset under a **seqlock** (a torn normal is neither unit length nor perpendicular to
  anything). `scenePlaneMask` must exclude the feed screens and the curved panorama screen — those float
  in world space near the pilot, and hitting one puts the "scene plane" a few metres away, which then
  looks exactly like a geometry bug.
- **The canvas is framed on the centre drone Unity nominates**, whose id rides in the *same seqlock* as
  the plane (`centre_drone_id`; `-1` falls back to the median of the selection, for a pre-field v2
  producer). Same reasoning as the plane normal: the canvas origin and axes are built from that view, so
  a centre paired with another frame's plane tears exactly like a torn normal. `_build_geometry` must
  never re-derive it — the median of the id-sorted selection is the median *drone id*, not the geometric
  centre, and it jumps whenever the selection gains or loses a drone.
- **The canvas extent is a mode, not a constant** (`planarCanvasMode`, default `Fixed`). `Fixed` is the
  original framing — `planarMetresPerPixel` centred on that centre drone's principal-ray hit, so the
  canvas covers the same patch of plane every frame and measurements stay comparable across runs, at the
  cost of being correctly framed at exactly one standoff. `AutoFit` derives scale and centre from where
  the views actually land (`PlanarStitcher._footprint_bbox` inverts each view's `G` rather than
  re-casting rays, so it inherits whatever corrections `_build_geometry` already applied). Two things
  keep `AutoFit` from breathing: the scale is **quantised to 1/3-octave steps** about
  `planarMetresPerPixel` with hysteresis and a dwell time, and the centre is low-passed **in world
  coordinates, not in plane `(a, b)`** — same reason `_pose_shift` is, since `e1` rotates with the
  reference camera. `Fixed` + `zoom 1` + `pan 0` is asserted byte-identical to the pre-mode mosaic in
  `planar_selftest.py`; that is what makes putting the dropdown back a guarantee rather than a hope.
- **Zoom and pan are a viewing transform and must never reach the estimators.** `planarZoom` /
  `planarPan` ride the seqlock (368–379) and are applied only in `planar_pano`;
  `_estimator_geometry` follows the *fit* scale but never the zoom. The sweep tracks a minimum whose
  basin is a few source-disparity pixels wide, so resizing or sliding its measurement window mid-
  convergence is exactly the "sweep loses its lock" failure the ACQUIRE/TRACK split exists to prevent —
  a pilot zooming in to look at something must not cost them their alignment. The selftest pins this by
  requiring the refiner's correction to be *identical* at 1x and 4x. Following the auto-fit scale is
  safe for the reason the split is necessary: corrections are stored in metres and scanned in disparity
  pixels, so both are canvas-scale-invariant and a step in the fit resets nothing. `_fit_*` is written
  only by the render thread and read by the warp thread — one writer, as with `_correction` the other way.
  Past ~1:1 with the sharpest view's ground sample distance, zoom is **empty magnification**; the
  `[PLANAR]` line prints the GSD and says so, because that limit moves with the standoff.
- **The default blend is winner-take-all, not a cross-fade** (`planarBlendMode`, default `Nearest`).
  Each canvas pixel goes to the single view seeing that plane point closest to the plane normal —
  `cos = h_v / √((a−a_v)² + (b−b_v)² + h_v²)` from the camera's in-plane footprint `(a_v, b_v)` and
  standoff `h_v`, which is a Voronoi partition of the plane by camera footprint. Cross-fading (`Feather`)
  is only clean while the geometry is *exact*: any residual error (pose noise, a facade that isn't quite
  planar) then superimposes two offset copies, and in a wall formation the views overlap over most of the
  canvas, so the ghosting is everywhere rather than confined to a seam. Winner-take-all cannot ghost —
  no pixel ever has two contributors — at the cost of a visible seam where the winner changes.
- **Blending weights are an analytic distance-to-border feather** in *source* pixels, normalised per
  pixel across all contributing views. A homography's alpha mask has a closed form, so unlike
  `StabStitcher` this needs no blur or erosion, and it generalises to any N for free. Under `Nearest`
  that falloff *multiplies* the obliquity score instead of blending, which keeps the seam off the
  source-image edges: a view running out of frame loses to a neighbour before it runs out of pixels.
- **`planarDebugView` colours each patch by its source drone** (`Tint` washes colour over the imagery,
  `Flat` replaces it) — the quickest read on where the seams fall. The palette is keyed on **drone id**,
  not on position in the selection, so a colour means the same thing frame to frame; the key is printed
  on Python's periodic `[PLANAR]` line, because the selection changes as drones join or drop out. The
  overlay is applied to the warped stack, so it works in both blend modes, and it is applied *after*
  the PSNR measurement so the logged number still describes the real mosaic.
- **Overlap PSNR is measured over geometric coverage, not blend weight.** Under `Nearest` the weights
  are one-hot, so a weight-based overlap test finds no overlap anywhere and silently retires the one
  diagnostic that quantifies pose error.
- **The projective-sanity gate is local Jacobian anisotropy** (`homography_anisotropy`), not the SVD
  condition number of the raw 3×3 — the latter is dominated by the metres-per-pixel unit scaling and
  reads ~35 000 for a perfectly benign nadir view, where the Jacobian reads 1.000. Mesh-distortion
  metrics are meaningless here: a homography cannot fold.
- **With `poseSource = GroundTruth` the mosaic must be pixel-exact** on a truly planar scene. Nothing is
  being estimated, so a visible seam is a bug, not a limitation. `StitchPoseSource.cs` injects
  GNSS-magnitude error (Ornstein–Uhlenbeck, with a **common-mode fraction** — nearby receivers share most
  of their error, and common-mode error translates the mosaic rigidly and costs nothing) for sizing how
  much refinement real drones would need.
- **`compute_warps` runs the two photometric estimators**, not the geometric solve — that costs
  microseconds and stays inline in `planar_pano` every frame, because the poses change every frame.
  What belongs on the warp thread is the *slowly-varying corrections*, which are what `_correction`
  holds. Each is independently switchable from the inspector (`planarPlaneSweep` / `planarPoseRefine`),
  and they move different parameters, so they are complementary rather than alternative.
  In one line each: the **sweep** fixes *how far away the surface is* — get that wrong and every view is
  drawn at the wrong size, so neighbours cannot line up however good the poses are; the **refiner** fixes
  *where each drone thinks it is* — get that wrong and one view's patch lands sideways of where it
  belongs. Both default **off**, so the pose-only backbone is unchanged until they are enabled.
  - **`_correction["plane"]` — plane sweep.** One global scalar: an *additive* offset on the published
    plane distance (additive so Unity's raycast keeps tracking the facade and the sweep only estimates
    the residual), chosen by scanning candidates and keeping the lowest photometric disagreement, then
    parabola-refined between samples. A plane-distance error appears in each view as a *scale* about its
    own footprint, so no per-view translation can absorb it.
  - **The sweep samples uniformly in disparity (`f·B/Z`), not in metres, and has two modes.** This is not
    a refinement — a single metres-uniform scan was the cause of the "sweep gets stuck on a wrong plane
    until you toggle it off and on" bug. Misalignment is linear in disparity, and the basin of attraction
    is a roughly fixed *pixel* width (≈`L·Z/B`), so one step size in pixels is correct at every standoff
    and no step size in metres is correct at two. The shipped defaults scanned ±4 m over 9 candidates —
    **1.0 m apart, against a basin about a metre wide.** To be sure of landing a candidate inside a
    minimum of width `W` the step must be ≤ `W/2`, so the minimum was never resolved: the argmin was
    noise, the low-pass then walked the plane ~1 m per pass in an arbitrary direction, and toggling the
    estimator merely re-rolled the dice. `planar_selftest.py` measures the real cost curve on two
    textures and pins both halves of that (`[…] the fine scan step resolves the basin` / `… the old
    metres-uniform step did not`).
    - **ACQUIRE** — wide, coarse (`SWEEP_COARSE_STEP_PX`), spanning ±`planarSweepRange` metres, which
      stays the honest way to say "how wrong could my prior be". It **snaps** on success rather than
      low-passing: this is the initial lock, and damping it would leave the estimate outside the basin it
      just found, where the fine scan has no signal. Rate-limited, since it costs several tracking scans.
    - **TRACK** — a few pixels either side of the incumbent at `SWEEP_FINE_STEP_PX`, low-passed by
      `planarRefineRate`, and it only moves for a measurable improvement on the incumbent so a converged
      sweep sits still instead of dithering.
    - A **contrast gate** (fractional cost drop from the median candidate to the best) is what separates
      "a minimum" from "the lowest sample of a flat curve". A flat window must mean *no measurement*, not
      *argmin of noise* — that distinction is the whole difference between tracking and random-walking.
      A run of `SWEEP_LOST_PASSES` unusable scans, or an argmin pinned to the window edge, drops back to
      ACQUIRE. That is the automatic form of the operator toggle that used to be the only cure.
  - **`_correction["dpose"]` — pose refiner.** Two DoF per view: a per-drone translation recovered by
    phase-correlating each view against the **leave-one-out** consensus of the others (not against the
    finished mosaic — under winner-take-all a view *is* the mosaic where it wins, so it would correlate
    against itself and report a confident zero). Differential GNSS and compass bias both appear at a
    facade as a lateral shift, so one translation absorbs the bulk of both.
  - **The sweep measures on `apply_dpose=False` geometry.** Not incidental: the refiner readily absorbs
    part of a depth error as per-view translation, and a sweep measuring on already-refined geometry is
    then hunting an error the refiner has hidden — it gets driven the wrong way.
  - **Corrections are residuals and accumulate** (`prev + rate * measured`), because `_build_geometry`
    has already applied the standing correction by the time the estimator sees the stack. Overwriting
    instead of accumulating caps convergence at one step's worth and oscillates.
  - **A gated-out measurement must not delete the correction that view already earned.** Rebuilding
    `_pose_shift` from only the views accepted this pass meant one weak correlation peak wiped that
    drone's history, snapped its patch back to the raw pose and left it to re-converge — which flaps.
    Only genuine absence from the *selection* retires a correction. The old behaviour was backwards as
    well as wrong: rejecting **all** views hit an early return that preserved everything, so only a
    **partial** rejection destroyed anything, which is why the self-test now gates exactly one view.
  - **The gauge is re-applied to the accumulated set, not just to each pass's residuals.** Each pass is
    zero-meaned against whichever subset it accepted, so residual-only gauging lets the accumulated set
    drift off zero mean as membership changes — a net translation that slides the whole mosaic.
    Relatedly, the **reference view gets the same correction every other view gets**: building the canvas
    frame from a raw pose the render then moves is a whole-image drift on top of the per-view alignment.
  - **Estimator switches come from the live metadata, the frame comes from the snapshot.** They must have
    independent freshness. When both were read out of `_frame_snapshot`, and the snapshot was only
    published while an estimator was on, the "both off" reset branch was *unreachable by construction*
    and turning an estimator off simply froze the last frame with the flag still set — the estimator kept
    running on one dead frame forever. `StitcherManager.update_planar_metadata` now pushes the config via
    `set_live_config` on every metadata read, which is the one path that keeps running while the render
    path is failing. The snapshot is timestamped and older than `SNAPSHOT_MAX_AGE_S` stops the
    estimators: every `planar_pano` path that skips the publish also returns a *blank* panorama, so a
    frozen snapshot means the render is already down and converging onto it poisons the recovery.
  - **Estimator logging lives on the warp thread** (`_maybe_log_estimators`), not on the `[PLANAR]` line.
    That line is printed at the end of `planar_pano`, after every early return, so it goes silent in
    exactly the situations worth diagnosing. What it prints is the cost curve's *shape* — mode, contrast,
    step in disparity pixels, interior-vs-edge argmin — because shape means the same thing in the sim and
    in the field, whereas the cost at zero offset measures how good the operator's prior was.
  - **Views whose frame has stopped advancing are dropped** (`MAX_CAPTURE_SKEW_S`, measured as lag behind
    the freshest block). `read_block_memory` re-serves a busy block's previous frame indefinitely, so one
    view's pixels can freeze while the formation moves; `capture_time` and `cached` were already on the
    wire and were being decoded and discarded. Never empties the solve — if every view is old they are
    old together, which is a stalled producer, and a frozen panorama beats a vanished one.
  - **Known degeneracy, asserted in the self-test so it cannot be "fixed" by accident:** a depth error
    dilates each view about its own footprint, which over the overlap *is* a convergent translation
    field — so convergent per-drone position error is indistinguishable from a plane-depth error. On
    that case the sweep alone moves the plane and gains nothing (17.0 → 17.1 dB overlap PSNR) while the
    refiner still recovers most of it (→ 24.9 dB). Turn the sweep on when the **plane** is what you are
    unsure of (a map-drawn facade distance, a raycast onto geometry that may not be there); leave it off
    when per-drone error dominates.
  - The confidence gate rejects a *measurement* with no dominant correlation peak — most usefully in
    early passes while the consensus is still a blur of misaligned views. It is **not** a guarantee
    against a repetitive facade: run to convergence, a periodic scene can settle into a self-consistent
    solution shifted by a whole period, and no per-measurement test can see that. `planarRefineMaxShift`
    is what bounds the damage. Both estimators apodise before correlating — masking both inputs with the
    same hard mask correlates the *mask*, which pins the answer at zero shift regardless of the pixels.
  - Estimator passes run on a quarter-resolution canvas with their **own** grid cache; sharing the
    render path's cache would miss on every call (different canvas size) and race across two threads.

### PLANAR on real drones — there is no raycast out there

Everything above assumes the sim. In the DJI scene the two things `PLANAR` depends on most — a
per-frame camera pose and a scene plane — come from Unity internals that do not exist in the field, so
both are supplied differently there. **Three settings and one flag** are what make it run:

| Where | Setting |
|---|---|
| `PyUniSharingFast` | `useManualIntrinsics = true`, `manualVerticalFovDeg = 46.4` (Mini 3 Pro at 16:9) |
| `PyUniSharingFast` | `scenePlaneMode = FormationRelative`, `planarStandoffMetres` = distance to the facade |
| `ImageSharing` | `stitchSlots` ≥ fleet size (it defaults to 3, which is STABSTITCH's number) |
| DJI_Swarm | `-ImageStreamPose` / `ImageStreamPose = $true` / `--image-stream-pose` |

- **The pose comes from `dji_camera_pose.CameraPoseSolver`** (DJI_Swarm repo), called inside
  `image_stream_feed.py`'s `frame_sink` — where the image bytes and the telemetry come out of the *same*
  `ds_wrapper` fetch, which is the tightest pose/frame pairing the system can offer. Position is metres
  from a **latched** origin (the first valid fix), *not* the per-tick swarm centroid `swarm_flocking.py`
  uses: that one drifts with the formation, which is right for flocking and wrong here, because a moving
  origin puts every frame's poses in a different frame from the plane. Rotation is the **gimbal**
  attitude, not the aircraft's. Frame is Unity-world left-handed, `+X = East, +Y = Up, +Z = North`.
  `dji_pose_selftest.py` pins every sign against an independently derived construction — do not "tidy"
  those signs without running it.
- **The plane is derived from the poses, not raycast** (`ScenePlaneMode.FormationRelative` →
  `PlanarStitcher._plane_from_formation`). The operator supplies one number, `planarStandoffMetres`, the
  *perpendicular* distance from the formation to the surface. No georeferenced origin has to be agreed
  between Unity and the drone telemetry, because the whole solve is invariant to a common translation.
  The normal comes from a plane fitted to the camera **positions** when the formation spans a plane —
  immune to gimbal pitch, and correct for nadir as well as facade — and falls back to the mean camera
  **forward** when the drones are collinear (a single row), where no plane fits. The `[PLANAR]` log line
  says which rule won; `forward` means the wall has collapsed to a row and the normal now follows the
  gimbal.
- **The other plane modes still exist and still work**, and `Manual` is now reachable in that scene at
  all: it and `FormationRelative` are resolved *before* `UpdateScenePlane`'s camera guard, which used to
  return early whenever `camerasToCapture` was empty — i.e. always, in the DJI scene.
  - `ScenePlaneMode.Manual` + `manualPlaneNormal` / `manualPlaneDistance` — world coordinates, so it
    needs an origin agreed with the pose frame. Prefer `FormationRelative` unless you have one.
  - A facade traced on the DJI_Swarm GUI map (`shapes.json`) is the georeferenced route, and is **not
    implemented**. Two traps waiting there: obstacles are **axis-aligned rectangles**, so a facade on an
    arbitrary bearing needs the geofence polygon or a new shape type; and trace the **base** of the
    building, not the roofline — satellite imagery displaces the roof from the footprint by
    `height × tan(off-nadir)`, ~7 m for a 20 m building.
- **Test it without aircraft**: `python tools/planar_feed_bench.py --drones 6 --rows 2` writes real
  `DroneFeedSharedMemory` blocks from a synthetic facade, so Unity and `StitcherThreading.py` run
  unmodified. `--selftest` does the same headless, with no Unity at all, and `--pose-error` /
  `--depth-error` / `--yaw-error` reproduce the field error budget on the bench. The two error knobs are
  separate on purpose: the refiner recovers lateral error and structurally cannot recover depth error,
  and lumping them together makes it look broken when it is working exactly as claimed.
- **The plane is the least sensitive of the four error sources**, so a rough distance really is enough.
  Budget for a ≤5 px seam at 30 m standoff / 8 m baseline (f ≈ 525 px at 800×450): plane distance
  ≤ 1.07 m, differential position ≤ 0.29 m, differential yaw ≤ 0.55°, gimbal pitch ≤ 0.55°. Stock DJI
  values land at 20–60 px (1–3.5 m on the facade) — a recognisable mosaic with visibly broken seams.
  Two levers, both worth more than tightening the plane: plane error scales as `f·B·δZ/Z²`, so it is
  **quadratic in standoff** (3 m plane error = 56 px at 15 m, 14 px at 30 m, 3.5 px at 60 m) and linear
  in baseline; attitude error is `f·δθ` and therefore **range-invariant** (0.5° = 4.6 px at any
  distance). Stand back, keep the formation tight, and expect no help from range on compass error.
- **Pose/video sync is a first-order term, not a detail.** DJI telemetry is only ~5 Hz fresh and the
  video has its own pipeline latency; at the 40 °/s yaw clamp, 300 ms of skew is ~55 px — larger than
  everything else combined. Hold station and yaw slowly while capturing. `MAX_CAPTURE_SKEW_S` drops a
  view that has fallen behind the freshest one, which bounds this but cannot remove a skew common to
  every aircraft.
- **Keep `planarBlendMode = Nearest`.** At 20–60 px of residual, `Feather` superimposes two offset
  copies over most of the canvas; winner-take-all confines the error to a seam.
- **The sweep's capture range is narrower than `planarSweepRange` suggests, and the field is where that
  bites.** Its basin of attraction is roughly `L·Z/B`, where `L` is the scene texture's correlation
  length — on brick at 30 m behind a 15 m wall that is ±0.2–0.6 m, not ±4 m. The ACQUIRE/TRACK split and
  disparity-uniform sampling described above are the implemented half of the fix, and they matter *more*
  here than in the sim: with no raycast the plane is static and wrong from frame one, so the sweep starts
  outside its basin rather than being knocked out of a good one. There is no transient to ride out and
  nothing for a watchdog to notice — `ScenePlaneMode.Manual` sets `scenePlaneValid = true`
  unconditionally, so `REASON_PLANE_INVALID` and the 2 s grace never fire out there. What the field case
  needs is **capture range**, not an escape hatch.
- **Do not reason about the sweep by asking "is the correction better than zero?"** In the sim zero is
  the raycast, which is a good prior; in the field zero is whatever standoff the operator typed, so that
  test measures the guess rather than the lock. The deployment-invariant read-out is the cost curve's
  shape, which is what `_maybe_log_estimators` prints.
- **Cross-drone triangulation remains the stronger prior** and needs no new dependency: `BaseStitcher`
  already loads SuperPoint plus BF/FLANN, and the formation's 8–25 m baselines put triangulated depth at
  ~0.2 m at 30 m — inside the fine scan's basin directly, with no acquisition pass at all. **Not
  implemented**; it is the next thing to build if ACQUIRE proves too slow or too easily fooled.

Two checkers, both runnable without Unity:
`python tools/planar_selftest.py` (geometry + end-to-end render) and
`python tools/check_wire_layout.py` (asserts the C# and Python layout constants agree — a mismatch there
is silent, since neither side fails to compile, it just reads a float from the middle of another field).

## Drone prefab hierarchy (relied on by many scripts)

`Drone N` → child `DroneParent` (has `SwarmAlgorithm`, `AttitudeAlgorithm`, `VelocityControl`, `Rigidbody`)
and child `FPV` (the `Camera`). Drones are tagged `DroneBase`; the arena is tagged `Arena`. Many scripts
use `transform.Find("DroneParent")` / `Find("FPV")`.

## Running the Python stitcher

`cd Assets/Scripts/ImageStitching && python StitcherThreading.py`. **Use the `stitching` miniconda env** —
the default `python` has no torch. StabStitch++ models load from
`StabStitch2_main/Full_model_inference/full_model_ssd/*.pth`.
