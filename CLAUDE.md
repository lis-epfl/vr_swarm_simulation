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
  It also carries a seqlocked **dynamic block** (scene plane, gimbal pitch, planar centre drone id) —
  see the PLANAR section. Its total size is pinned at 412 bytes: new tail fields come out of
  `metadataReservedGap`, because changing the size would strand an already-running Python. The static
  tail's padding filled up at 311, so the estimator settings (344–363) sit *after* the dynamic block;
  both sides address every tail field by absolute offset, so the ordering is cosmetic, but it is why
  `readMetadataMemory` seeks a second time rather than reading straight through.
- `BlockSharedMemory` — the **stitcher input**, one block per selected drone. `flag` is the handshake
  (0 = ready, 1 = busy). Images are **BGR, top-down**. Sole consumer: `StitcherThreading.py`. Sole
  producer: sim = `PyUniSharingFast`; real-drone mode (DJIScene) = `ImageSharing.cs` (so keep
  `enableImageWriting` off on `PyUniSharingFast` there). Two header versions coexist; Unity publishes
  which one it writes as `blockHeaderSize` in metadata, and Python reads that rather than assuming:
  - **v1, 12 bytes** — `int flag | int droneId | float heading | RGB24 image`. What `ImageSharing.cs`
    writes (real drones publish yaw only).
  - **v2, 48 bytes** — appends `float camPos[3] | float camRot[4] (xyzw) | float captureTime |
    int poseStatus`, all in **Unity world / left-handed**. Required by `PLANAR`. The pose lives in the
    block, not in metadata, because it must be the pose of *that* frame: it is snapshotted in
    `RequestBlockCapture` 1–2 frames before the readback completes, and Python may re-serve a cached
    block, which then needs its own pose.

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
  In the DJI scene the section is created by `ImageSharing.cs` (`StitchSlots = 3`,
  `MetadataSize = 12`), so `DesiredBlockCount()` returns `STITCH_COUNT_LRC` there **without** the
  `camerasToCapture` clamp — that scene has no sim FPV cameras, and clamping advertises 0 blocks,
  which makes Python map none of the section and the real-drone panorama silently never appear.
  `create` and `describe` being split across two files that never reference each other is the
  hazard; `tools/check_wire_layout.py` now asserts the two pairs agree.
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
- **Boundary drones** = `AttitudeAlgorithm.BoundaryEstimate` (convex-hull). Left/centre/right stitching
  and the `OUTER_CIRCLE` screen layout only use boundary drones (see the planar exception above).
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
  and they move different parameters, so they are complementary rather than alternative:
  - **`_correction["plane"]` — plane sweep.** One global scalar: an *additive* offset on the published
    plane distance (additive so Unity's raycast keeps tracking the facade and the sweep only estimates
    the residual), chosen by scanning `planarSweepSteps` candidates over ±`planarSweepRange` and keeping
    the lowest photometric disagreement, then parabola-refined between samples. A plane-distance error
    appears in each view as a *scale* about its own footprint, so no per-view translation can absorb it.
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
