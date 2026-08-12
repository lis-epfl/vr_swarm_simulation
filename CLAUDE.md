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

**Every section has exactly one size, forever, and it is a compile-time constant on both sides.**
A named Windows section cannot be resized: `CreateFileMapping` opens the existing one when the name is
taken and denies a larger request. So no section size may depend on the drone count, the stitcher, or
the resolution — only on constants that `tools/check_wire_layout.py` asserts equal across all three
files. Sizes that varied at runtime are what produced the intermittent access-denied crashes.

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
  reading straight through. The tail ends with a **producer heartbeat** (384, bumped every Unity frame)
  and the **section geometry** (388–395 capacity/stride, 404–411 the two section sizes) — see the
  `BlockSharedMemory` entry for why both exist.
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

  **The section is a fixed array of fixed-stride slots, created once and never recreated.**
  `blockSlotCapacity` (24) × `blockSlotStride` (48 + 1280×720×3) = `blockSectionBytes`, all
  compile-time constants mirrored in `PyUniSharingFast.cs`, `ImageSharing.cs` and
  `StitcherThreading.py`. This is not a tuning choice — a named Windows section **cannot be
  resized**. `CreateFileMapping` (and Python's `mmap.mmap(-1, size, tagname)`, which is the same
  call) opens the *existing* section when the name is taken, and a request **larger** than it fails
  with `ERROR_ACCESS_DENIED`. Sizing the section to the live drone count, as it used to be, therefore
  deadlocked against whichever process held the old size; that was the "access denied on
  BlockSharedMemory" failure, and because Unity's failure path left `blockPtr` null it silently
  stopped publishing for the rest of the session. Three paths used to resize it — `SetStitcherType`
  (3 slots ⇄ N), the 3 s `UpdateCameras` tick, and `OnValidate` — and all three are gone.
  Consequences worth keeping:
  - Slot `i` is **always** at `i * blockSlotStride`. Neither the fleet size, the stitcher, nor the
    resolution moves a slot. Only the *payload length* (`blockImageWidth × blockImageHeight × 3`) is
    live, and it is read from metadata; the rest of each slot is padding. The image envelope is
    therefore load-bearing: `CalculateMemorySizes` clamps to 1280×720 with a `LogError`, and Python
    refuses to run on a payload that would overrun the stride.
  - `blockImageCount` (metadata) is now a **scan hint**, not a size — how many slots are worth
    polling. `droneId == -1` is the only thing that marks a slot as carrying no view, set
    **including at creation** over the whole capacity, because a fresh section is zero-filled and `0`
    is a legal drone id: an unwritten slot would otherwise advertise a ready block from drone 0 with
    an all-zero (degenerate) quaternion. Most slots stay at that sentinel for a whole session.
  - Python's sufficiency test is a **floor** (`>= MIN_STITCH_IMAGES` / `>= MIN_PLANAR_IMAGES`), not
    `len(views) == num_blocks`. The old equality only worked while the section was sized to exactly
    the three slots STABSTITCH fills; against a fixed capacity it is permanently false.
  - The readback pool is still sized to two full batches (`EnsureReadbackPool`): with a pool smaller
    than the per-send slot count, `AcquirePendingSlot` starves the *same* tail slots every send, so
    they are never written at all rather than merely late.
  - **There is no per-stitcher view knob.** `maxStitchViews` and `ImageSharing.stitchSlots` are gone;
    `PLANAR` mosaics every alive drone that passes the plane-hit/range/obliquity tests, and
    STABSTITCH still writes its three. Switching stitcher no longer touches the wire at all.

  **`PyUniSharingFast` publishes `blockImageCount` + `blockHeaderSize` even when it is not the
  producer**, because only this component writes metadata. In the DJI scene the section is created by
  `ImageSharing.cs`, so `PublishedSlotCount()` advertises the whole capacity there rather than
  clamping to `camerasToCapture` — that scene has no sim FPV cameras, and a hint of 0 makes Python
  scan no slots at all, so the real-drone panorama silently never appears.
  `create` and `describe` being split across two files that never reference each other is the
  hazard, and it now cuts twice: both components request the *same named section*, so unequal sizes
  deny whichever starts second. `tools/check_wire_layout.py` asserts the header size, the LRC count
  and the full section geometry across both files, and — when the `DJI_Swarm` repo is checked out
  beside this one — the feed header across both repos too.

  **Python fails loudly rather than limping** (`_fatal`): an unmappable section, a geometry
  disagreement with the producer, or a **heartbeat** that has not advanced for 5 s all print a
  diagnostic and exit non-zero. The heartbeat is the only signal that distinguishes "Unity is idle"
  from "Unity has crashed" — closing the producer's handle does not destroy the section while Python
  holds it, so every block keeps its last contents and the stitcher would otherwise render a frozen
  panorama forever.
- `DroneFeedSharedMemory` — **all real-drone feeds** (DJIScene only), same per-block layout as above but
  a fixed capacity of **10 blocks** indexed by zero-based drone id (must match `MAX_DRONES` in the
  DJI_Swarm repo's `image_stream_feed.py`, which is the producer). Its per-block stride is fitted to
  the 800×450 feed and is **not** `blockSlotStride` — the two maps are sized independently.
  Consumer: `ImageSharing.cs`, which displays the feeds and re-publishes the selected views into
  `BlockSharedMemory` (`PublishStitchBlocks`: the 3 body-yaw-selected ones under STABSTITCH, every
  fresh feed under PLANAR). Unity marks unwritten **and already-consumed** blocks with `droneId == -1`
  (the producer rewrites `droneId` every write) — that marker is the new-frame detection, since the
  flag alone can't distinguish a fresh frame from a re-read.
- `PanoramaSharedMemory` — `int flag | int quality_ok | RGB24 panorama`. `quality_ok == 0` ⇒ Unity shows
  individual feeds instead of the panorama. Panorama is **vertically flipped and converted BGR→RGB** by
  Python (Unity textures start bottom-left; Unity uploads the bytes straight into an RGB24 texture).
  Fixed-size for the same reason as the block section: `panoramaSectionBytes` (8 + 4000×4000×3) on
  both sides, with the live panorama written as a *prefix*. Unity always used the constant; Python
  used to ask for `w*h*3 + 8`, so whichever process created the section first denied the other — and
  unlike the block map that path had no retry, so the symptom was a curved screen that simply never
  updated for the whole session.

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
  radially-outward ring: in vertical-plane mode `AttitudeAlgorithm` drives every drone to one shared
  target heading, so yaw proximity is a tie broken by jitter and the centre changes almost every frame.
  That matters because the scene-plane raycast originates at this camera and `PlanarStitcher` frames the
  canvas on it — a flickering centre both steps the published plane offset and slides the mosaic.
- **Vertical-plane mode has no leader drone, and adding one back is a regression.** The plane's heading
  is a setpoint `SwarmPlaneController` owns (`TargetYaw`, seeded from the swarm's circular-mean heading
  on entry, then advanced only by the yaw stick), its offset along the normal is the swarm centroid
  (`PlaneOrigin`), and the vertical leash centre is latched from that centroid and moved by the climb
  stick (`ReferenceAltitude`). Every drone then converges on the heading through the *same* law —
  `plane.TargetYawRate` as feed-forward plus `YawCorrectionFactor ×` its own error, summed and clamped
  to `maxYawRate` in `VelocityControl`. This is the real fleet's scheme
  (`DJI_Swarm joystick_controller.heading_hold_rate` + `swarm_plane.py`), and it is deliberate: the
  earlier design nominated an *anchor* drone that took the stick directly, had its swarm force zeroed,
  and supplied both the plane offset and the vertical reference, while the rest P-tracked its live
  compass. That drone turned at the full stick rate and the wall trailed it through the yaw filter, the
  inner rate loop and drag — one important drone and n−1 followers. Two consequences worth keeping:
  - **`maxTargetLeadDeg` (25°), not the stick gain, is what holds the headings together through a
    turn.** `targetYawRateDegPerSec` (60) deliberately exceeds `maxYawRate` (≈57 °/s), exactly as the
    fleet's ff 60 exceeds its 40 °/s clamp, so a full-stick turn is rate-saturated. Clamping how far
    the setpoint may lead the swarm's *measured* mean heading is what stops a sustained turn banking up
    a heading debt the drones keep paying off after the stick is centred. The clamp acts only while the
    stick is deflected, so at centre stick the hold keeps full authority and a disturbance never drags
    the setpoint along with the wall.
  - **The reference altitude is latched, not the live centroid.** A live centroid leaves the mean
    altitude a free mode: the leash would bound each drone's spread about the mean while the mean drifted
    on the net vertical bias the swarm forces carry (cohesion and the plane pull are zero-sum, ground
    repulsion is not). It tracks the climb stick at the rate read off the drones' own `maxAltitudeRate`,
    so the leash cannot clip a climb the pilot is commanding.
- **In vertical-plane mode the body/rig yaw is slaved to the plane, not to the yaw stick**
  (`PyUniSharingFast.UpdateBodyYawFromPlane`). The stick already steers the plane's target heading and
  the whole wall converges on it, so also integrating that stick into `bodyYaw` walks the view off the
  wall — different gain, none of the drones' lag. Invisible in the radially-outward ring (the panorama
  re-snaps to the nearest camera), but under a shared heading `bodyYaw` is what aims the VR velocity
  frame at the wall. It follows `TargetYaw` rather than the drones' measured mean so the rig answers the
  stick 1:1 instead of lagging it, and `maxTargetLeadDeg` bounds the resulting lead. The lock is
  absolute, so entering plane mode also clears any pre-existing offset.
- **The stitcher and the screen layout follow the configuration, and there are three of them**
  (`SwarmPlaneController.RefreshDisplayConfiguration`): vertical plane ⇒ `PLANAR` +
  `FORMATION_WALL`, horizontal with the gimbal down ⇒ `PLANAR` + `FORMATION_MAP`, horizontal
  looking out ⇒ `STABSTITCH` + `OUTER_CIRCLE`. No pairing is taste — a wall (or the ground under a
  nadir view) is one dominant plane where the pose-driven homographies are exact and the shared
  heading collapses `OUTER_CIRCLE` onto a single arc position, and the ring is the parallax-heavy
  case StabStitch++ exists for. The configuration is the two bits (plane mode, gimbal ≤
  `FPVCameraScript.NadirPitch`); the plane wins when both are set. The nadir bit is **polled in
  `Update`, not taken off `swarmParamsChanged`**, because `SetGimbalPitchNormalized` — the joystick
  dial, the only one a pilot in a headset can reach — doesn't raise that event. It fires only
  on an actual change, so the inspector's choices stand until the first one, and
  `driveDisplayConfiguration` turns it off for comparing two stitchers on one formation. Both setters
  go through the owning component (`PyUniSharingFast.SetStitcherType`,
  `InterfaceManager.SetScreenStyle`) rather than writing the fields: the stitcher switch has to
  republish metadata for Python to pick up, and the layout has to stay owned by `InterfaceManager` per
  the source-of-truth rule below. `SetStitcherType` deliberately does **not** touch
  `BlockSharedMemory` any more — it used to resize it (3 slots for STABSTITCH, more for `PLANAR`), and
  this call site, fired mid-flight by the `V` key with Python attached, was the most reliable way to
  hit the `ERROR_ACCESS_DENIED` that a non-resizable named section guarantees.
- **Boundary drones** = `AttitudeAlgorithm.BoundaryEstimate` (convex-hull). Left/centre/right stitching
  and the `OUTER_CIRCLE` screen layout only use boundary drones (see the planar exception above).
- **Every screen style works off either feed source, and `ScreenStyle.REAL_DRONE` is gone.** A layout
  needs four things per drone — display yaw, world position, the image-frame yaw the map's roll undoes,
  and alive/visible — and `ScreenSpawn` reads all four through accessors (`TryGetDisplayYawRad`,
  `TryGetSamplePosition`, `SampleYawDeg`, `IsFeedSuppressed`) that dispatch on
  `DroneScreenBinding.feedIndex`: `-1` is a sim drone (Transforms and components), `>= 0` indexes
  `realFeeds`, pushed by `ImageSharing.UpdateRealDroneFeed` as it consumes `DroneFeedSharedMemory`.
  Nothing was added to the wire — the pose was already in the block beside the pixels. Points worth
  keeping:
  - **The real path only became reachable by giving it bindings.** `UpdateScreenPositions` returns
    immediately on an empty `bindings` list, and `SpawnScreens` used to build bindings *only* when
    handed a sim swarm — so in DJIScene no layout ran at all and the feeds were positioned by
    `UpdateRealDroneScreen`, a push-driven style of its own. That was `OUTER_CIRCLE` with a zero
    offset/lookAtOffset (exactly the `outerCircle*` defaults) and no boundary gate — and the gate is
    inactive anyway wherever there is no `SwarmManager`, which is every real-drone scene. Hence the
    removal rather than a deprecation: it had nothing left to express. Its enum value **5 is left
    unused**, because Unity serialises enum fields by integer and reusing it would silently re-point
    `FORMATION_WALL`/`FORMATION_MAP` in every scene that stores them.
  - **Aliveness is freshness, not a flag.** A real feed is suppressed once
    `Time.time - lastUpdateTime` exceeds the timeout `ImageSharing` pushes from `stitchFrameMaxAge`,
    so the screens and the stitch selection agree on what "still flying" means and a drone that stops
    streaming leaves the layout instead of freezing in it. The staleness test lives in `TryGetFeed`,
    so no accessor can hand a layout a dead drone's position either.
  - **An unposed feed (`poseStatus == 0`) costs its own screen, not the layout.** It has no ranking
    key, so the grid styles drop it, exactly as the planar solve drops an unposed block; the circle
    styles still show it, since they need only the heading.
  - **`headingOffsetDegrees` is applied to the pushed position and nowhere else**
    (`ImageSharing.LayoutPosition`). That offset rotates the compass heading into the HMD's yaw
    frame, and it is deliberately *not* applied to the pose on the stitcher path — there the pose
    defines its own `+Z = North` frame that the scene plane is expressed in, and turning one without
    the other yaws the mosaic off the facade. The layouts are the one place the two frames meet,
    because `FORMATION_WALL` takes its basis from the headings and then projects the positions onto it.
  - The screen material's white base colour is keyed on the **real-feed spawn path**, not on the
    style: `ImageSharing` assigns the frame to `mainTexture` (albedo) as well as the emission map, and
    a black base multiplies it away. Keyed on the style it would now never fire.
- **`OUTER_CIRCLE` is only meaningful for the radially-outward ring**, and `ScreenStyle.FORMATION_WALL`
  is its shared-heading counterpart (vertical plane; nadir gets `FORMATION_MAP`, below). Placing each screen at its own drone's
  yaw works only because the ring *spreads* the yaws; under a shared heading every screen lands on the
  same arc position and they stack. `FORMATION_WALL` keeps yaw as the thing that aims the display — the
  circular mean of the yaws sets one azimuth for the wall, so turning the formation turns the wall — and
  takes the *separation* from each drone's rank inside the swarming plane (`SwarmPlaneController.
  GetPlaneAxes`, the same basis the planar centre-drone rule uses). Consequences worth knowing:
  - **The basis has three tiers, and the second is what makes the style work without a swarm.**
    `SwarmPlaneController.Instance` is first because it is *exact* — the plane is a setpoint the
    controller owns, so it is right before the drones have converged on it. Where there is no such
    component (every real-drone scene: DJIScene contains no swarm at all) the normal is the circular
    mean of the drones' own headings, which `BuildFormationGridLayout` already accumulates for the
    wall's azimuth — the style is contracted to a *vertical* plane, and for a vertical plane
    `GetPlaneAxes` reduces to "up is world up, right is the horizontal perpendicular to the normal",
    whose horizontal direction is just the shared heading because the drones face the facade. Third
    is the world `(x, z)` pair, reached only when the yaws cancel — a radially-outward ring, where
    there is no wall to aim at and `OUTER_CIRCLE` is the right style anyway. That collapse is one
    threshold (`WallYawResultantMin`) gating both the basis and the azimuth, since they fail together.
    Two things follow. The basis is resolved **between** the grid's two loops rather than before
    them, because tier 2 is derived from what the first loop collects. And it is deliberately *not*
    `PlanarStitcher._plane_from_formation`, which fits the camera positions: that plane has to be
    metrically right to build homographies, whereas ranking drones into cells needs only the two
    axes, and re-deriving it here would be a third copy of a rule that must agree with the other two.
    `SwarmPlaneController.PlaneAxesFromNormal` is shared by tiers 1 and 2 so they cannot drift, and
    the tier-2 normal is the mean heading *direction* — matching tier 1, where `planeNormal` is
    `YawToForward(TargetYaw)` and not its opposite, so the two rank columns the same way round.
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
- **`ScreenStyle.FORMATION_MAP` is the nadir counterpart of the wall** — same
  `BuildFormationGridLayout`, same rank-into-a-grid, same span budget — and it hangs the grid on a
  **vertical panel in front of the pilot, not on the floor**. A floor layout is geometrically honest
  and useless: it puts the whole display outside the gaze cone and the pilot flies looking at their
  feet. It differs from the wall in exactly three places, all following from what the cameras see:
  - **The frame is the pilot's body yaw** (`PyUniSharingFast.BodyYawDegrees`), not the swarm's mean
    heading and not `GetPlaneAxes`. In nadir the swarming plane is horizontal, so `GetPlaneAxes`
    degenerates to the world `(X, Z)` pair and ranks the formation north-up — "ahead" then means
    nothing to the pilot — while the wall's circular mean of the yaws *collapses* on a
    radially-outward ring and merely holds its last azimuth. Body yaw has neither failure: it is
    always defined, and it is what `CalibrateToCentre` aims the head at, so the panel is in front of
    the pilot by construction. It is **snapped, not low-passed** — easing the pilot's own heading
    slides the panel out of view during a turn — and the glide a cell swap needs moves to the *cell
    offsets* instead. Rows are along-track distance, furthest ahead at the top.
  - **Columns are subtracted, not added.** In this display frame azimuth increases to the pilot's
    **left** (a screen sits at `r(cos a, 0, sin a)` with `a = −yaw`, and the head faces `a = −bodyYaw`),
    which `OUTER_CIRCLE` demonstrates: a drone yawed clockwise of the view centre gets the more
    negative azimuth and appears to the right. A mirrored map is not cosmetic — "the obstacle is on
    the right" has to mean the pilot's right. **`FORMATION_WALL` still adds**, so its starboard drone
    lands on the pilot's left; that looks like a latent mirror in the wall, unverified in a headset.
  - **Each feed is rolled into the map frame** by `droneYaw − frameYaw`. A nadir image is already a
    plan view drawn in its *own* drone's heading frame (at gimbal −90 the camera's up axis lands on
    the drone's forward), so without the roll two feeds show the same ground rotated differently the
    moment the headings disagree, and no grid placement fixes it. The wall has no such problem: its
    cameras look along the plane normal, where a shared heading already means a shared image frame.
    The roll makes each quad sweep its rotated bounding box, so the cell size is
    `w|cos δ| + h|sin δ|` maxed **per screen** — the width term peaks at `atan(h/w)`, so the largest
    roll is not always the widest cell.
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
| DJI_Swarm | `-ImageStreamPose` / `ImageStreamPose = $true` / `--image-stream-pose` |

There used to be a fourth: `ImageSharing.stitchSlots`, which defaulted to STABSTITCH's 3 and had to be
raised to the fleet size by hand — and only took effect on a Play restart, because the section was
created in `Start`. It is gone; `PublishStitchBlocks` now writes every aircraft with a fresh posed
frame into the fixed-capacity section.

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
