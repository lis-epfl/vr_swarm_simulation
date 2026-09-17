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
  (yaw **and** the PLANAR standoff + its resolved source are also rewritten every frame at fixed
  offsets — 248, 364 and 396). This is the integrated *body heading*
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
  a fixed capacity of **10 blocks** indexed by zero-based drone id, followed by a **64-byte
  scene-plane trailer**. Every constant lives in the DJI_Swarm repo's
  `utils/imageSharingUtil.py` as `FEED_*` (not in `image_stream_feed.py` — `check_wire_layout.py`
  parses the former and asserts it against `ImageSharing.cs`, and does not parse the latter).
  Its per-block stride is fitted to the 800×450 feed and is **not** `blockSlotStride` — the two maps
  are sized independently.
  Consumer: `ImageSharing.cs`, which displays the feeds and re-publishes the selected views into
  `BlockSharedMemory` (`PublishStitchBlocks`: the 3 body-yaw-selected ones under STABSTITCH, every
  fresh feed under PLANAR). Unity marks unwritten **and already-consumed** blocks with `droneId == -1`
  (the producer rewrites `droneId` every write) — that marker is the new-frame detection, since the
  flag alone can't distinguish a fresh frame from a re-read.

  **The trailer carries `planarStandoffMetres` from the PC**, seqlocked, and is the one place this
  system grows a section that was already sized — which is normally how you earn
  `ERROR_ACCESS_DENIED`. It is safe *only* because Windows compares **page-rounded** sizes: the block
  array is 10,800,480 B, which rounds to 10,801,152 (2637 × 4 KB), leaving **672 already-backed
  bytes**. Measured on Windows 11: create at the block size and open at +672 succeeds in either
  order, +673 is denied. So a Unity carrying the trailer and a DJI_Swarm predating it interoperate in
  both start orders, and the trailer just reads zero. `check_wire_layout.py` asserts
  `page(FeedSectionBytes) == page(FeedBlocksBytes)` for that reason — **past 672 bytes both cross
  orders become fatal and the feed dies**, so the bound is machine-checked, not a comment. Both sides
  also carry a two-step fallback to the pre-trailer size; on the Python side that is not decoration,
  because `ImageStreamPublisher.__init__` runs before the aircraft arm and an unhandled `OSError`
  there takes down the flight controller rather than merely the mosaic.
  Unity **never writes** the trailer (`Start`'s block-init loop stops at `MaxFeedBlocks`), and
  `PyUniSharingFast` must never map this section — `ImageSharing` is its sole consumer and hands the
  value over on `ImageSharing.FeedStandoff*` statics, the same shape as `PlanarSelected`/`BodyYawDegrees`.
- `PanoramaSharedMemory` — `int flag | int quality_ok | RGB24 panorama`. `quality_ok == 0` ⇒ Unity shows
  individual feeds instead of the panorama. Panorama is **vertically flipped and converted BGR→RGB** by
  Python (Unity textures start bottom-left; Unity uploads the bytes straight into an RGB24 texture).
  Fixed-size for the same reason as the block section: `panoramaSectionBytes` (8 + 4000×4000×3) on
  both sides, with the live panorama written as a *prefix*. Unity always used the constant; Python
  used to ask for `w*h*3 + 8`, so whichever process created the section first denied the other — and
  unlike the block map that path had no retry, so the symptom was a curved screen that simply never
  updated for the whole session.
  **The quality word's upper 16 bits are a panorama sequence number** (`PANORAMA_SEQ_SHIFT` /
  `panoramaSeqShift`, asserted by `check_wire_layout.py`), bumped on every image write and never 0.
  Unity uploads the texture only when it changes, and after `readInterval` re-checks every frame
  until it does. The old timer read beat against Python's own ~30 Hz cadence: on the bridge bench it
  re-uploaded ~10% of panoramas already on screen and never showed another ~10%, and each upload is
  a main-thread `LoadRawTextureData` + `Apply` whose RGB24→RGBA32 conversion (D3D11 has no 24-bit
  format) grows with the panorama size. `0` is a producer without the counter and gets the old
  upload-every-read behaviour, so either side can be updated first.

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
  **A steadier heading is not grounds for reverting this to the yaw-proximity rule.** The argument is
  structural, not about jitter: under a shared heading yaw proximity has no unique answer at all, and
  it still has none however well `VelocityControl`'s heading hold holds the drones on it.
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
    turn.** On the fleet, `YAW_RATE_DEG_S` (60) deliberately exceeds the PC's 40 °/s clamp, so a
    full-stick turn is rate-saturated and the debt is guaranteed. **The sim does not reproduce that
    clamp:** `maxYawRate` is the Mini 3 Pro airframe's 75 °/s (1.309 rad/s), *above*
    `targetYawRateDegPerSec` (60), so here the debt comes only from the drones lagging the setpoint —
    through every turn's entry, and indefinitely for a drone something is holding back. Clamping how far
    the setpoint may lead the swarm's *measured* mean heading is what stops a sustained turn banking up
    a heading debt the drones keep paying off after the stick is centred. The clamp acts only while the
    stick is deflected, so at centre stick the hold keeps full authority and a disturbance never drags
    the setpoint along with the wall. There is now a **second** lead clamp one level down
    (`VelocityControl.maxHeadingHoldErrorDeg`); the two do not fight, because this one bounds the
    shared setpoint against the swarm *mean* and that one bounds each drone's own setpoint against its
    own heading, and they meet only through the measured heading this clamp already tracks.
  - **The reference altitude is latched, not the live centroid.** A live centroid leaves the mean
    altitude a free mode: the leash would bound each drone's spread about the mean while the mean drifted
    on the net vertical bias the swarm forces carry (cohesion and the plane pull are zero-sum, ground
    repulsion is not). It tracks the climb stick at the rate read off the drones' own `maxAltitudeRate`,
    so the leash cannot clip a climb the pilot is commanding.
- **The altitude ceiling is a hard cap on the height *setpoint*, measured from the Terrain, and it is
  per scene.** `SwarmManager.maxHeightAboveTerrain` (0 = off) is pushed into every drone through
  `SwarmAlgorithm`'s params update, and `VelocityControl` clamps `desired_height` to
  `terrain + limit` — the symmetric counterpart of the `MinHeight` floor applied immediately after
  it, with the floor winning if the two ever cross. Points worth keeping:
  - **Clamping the setpoint rather than the stick is what keeps it free of windup.**
    `desired_height` is clamped *in place*, so holding the climb stick at the ceiling stores nothing
    and pushing back down moves the drone on the first tick. The same argument applies one level up,
    which is why `SwarmPlaneController.IntegrateReferenceAltitude` clamps the wall's reference
    altitude to the same ceiling at the centroid: left alone, a held stick walks that reference past
    a ceiling the drones cannot follow it through, and every metre of it has to be flown back before
    the stick does anything visible.
  - **The swarm's upward force is dropped above the ceiling as well, and only in the horizontal
    case.** In a tilted plane the swarm already drives the setpoint (`verticalSwarmAuthority`), which
    the clamp bounds; in a horizontal formation its vertical force goes straight into thrust, where
    the clamped setpoint does not reach it and a sustained upward pull from the lattice would simply
    sit on top of the cap. Only the upward half is dropped, so a drone held at the ceiling still
    settles back into the formation.
  - **In plane mode a climb stick is dropped once the wall's reference altitude is pinned at the
    ceiling, and the test has to be that reference rather than the drone's own setpoint.** A stick
    left in the sum is added and clamped away every tick, which cancels the swarm's vertical term
    exactly — so a wall flown up into the ceiling **pancakes flat against it** instead of re-forming
    below, and drones end up in each other's cells. Measured on a headless ScaledCityWorld wall, 60 s
    of full climb stick: plain clamp, the wall collapses to 51.24–51.33 m and **2 of 10 drones are
    lost to a drone-drone collision at the cap**; with the stick dropped it keeps 37.9–51.3 m of
    spread and loses none, the top still exactly at the ceiling. Gating on the drone's own setpoint
    instead is worth nothing (it measured *worse*, 4 lost): the swarm pushes the setpoint a tick's
    worth below the cap and that very dip re-enables the stick, which puts it straight back. The
    same flattening is what the pre-existing `MinHeight` floor does at the bottom — the first bench
    arm lost a pair that way at 1.8 m — and it is not addressed here.
  - **Measured from the Unity Terrain under the drone, not from a downward raycast.** A raycast
    ceiling rides up over the rooftops, which both defeats the point (a drone could sit at
    `limit + 50` over a building) and shoves it down hard at every roof edge. Past the edge of the
    terrain the nearest tile answers rather than nobody (`TerrainHeightSampler`), so flying off the
    map is not a way out from under the ceiling; a scene with **no** Terrain — DJIScene,
    FactoryScene — has no ground to measure from and gets no ceiling at all.
  - **The flown ceiling is ~1.3 m above the number**, because the height PD holds each drone that far
    above its setpoint (the `ForceMode.Acceleration` linear twin above). Fixing that is its own
    change; do not compensate for it here, or the compensation becomes wrong the day it is fixed.
  - The value belongs to the scene's geometry, so it is per scene: `ScaledCityWorld` 50, exactly its
    uniform skyline, so the swarm must fly *around* a building rather than over it; `CityWorld` and
    `CrowdWorld` 150, because the pack's unscaled buildings reach ~136; `RingChallenge` and
    `NBackExperiment` 80, comfortably above their content (~47 and ~56).
- **The heading loop is a cascade, and `AttitudeAlgorithm` is the PC, not the flight controller.**
  On the real fleet the PC sends a yaw *rate* (`YawControlMode.ANGULAR_VELOCITY`) and the DJI FC holds
  heading off its own IMU whenever that rate is zero. `AttitudeAlgorithm` is the analogue of the PC's
  `heading_hold_rate`; the FC's half lives in `VelocityControl` (`headingSetpoint`, `headingHoldKp`),
  and until it was added there was **no heading loop at all** below the outer one — only a rate loop.
  Anything that knocked the nose off heading came back at well over a second, which is what made every
  screen on the ring slide whenever the pilot changed the velocity command. Worth keeping:
  - **The order in `VelocityControl` is load-bearing.** `yawFilterCoefficient` is a command prefilter
    on the *outer* rate and nothing downstream of it — the heading-hold P term is disturbance rejection
    and is added after, unlagged. The setpoint integrates the **post**-filter rate, or it runs
    permanently ahead by `τ_EMA × rate` and owes all of it back at stick release. And `maxYawRate`
    clamps the **sum**, as `heading_hold_rate` does, not the feed-forward alone.
  - **The lead clamp is unconditional here and gated on the fleet.** The fleet gates on `ff_rate != 0`
    because `KP_YAW × 25° = 20 °/s` is *below* its 40 °/s clamp, so a pinned setpoint would cap its
    correction below the actuator limit. Here `headingHoldKp × maxHeadingHoldErrorDeg` (2.2 rad/s) is
    above `maxYawRate` (1.309), so the rate clamp binds first and the clamp costs nothing. The invariant is
    `headingHoldKp * maxHeadingHoldErrorDeg * Deg2Rad >= maxYawRate`; below it, restore the gate.
  - **The setpoint is back-calculated: it advances only by the part of the rate the `maxYawRate` clamp
    let through, and that correction may slow the advance but never reverse it.** Without it the
    setpoint integrates a rate the airframe cannot fly — through every spin-up, and whenever the outer
    command exceeds the limit — and the aircraft overshoots the target by the whole lead on arrival
    (12–19° in a replica of the loop at outer gains of 2–3). This is what lets `YawCorrectionFactor`
    be 3 rather than the fleet's `KP_YAW` 0.8. The one-sided limit is what keeps disturbance
    rejection intact: an unlimited back-calculation drags the setpoint toward wherever a large enough
    knock left the nose, and in a mode with no outer loop (`NONE`) that heading is then simply lost.
  - **The yaw axis flies its real 75 °/s only because `angularDrag` is compensated on it.** PhysX
    adds the torque's velocity change *before* damping by `(1 − drag·dt)` (measured in batchmode), so
    at the prefab's drag of 5 holding 75 °/s needs 7.3 rad/s² against `maxAlpha` 3.38: until this was
    added the airframe topped out at **34.9 °/s**, and below saturation delivered 0.58 of its command.
    The yaw part of α is now `clamp((effectiveYawRate − ω_v)(1/τ + drag), ±maxAlpha)`, plus
    `drag·ω_v`, divided by `(1 − drag·dt)` — exact to five decimals on a tilted body under
    simultaneous tilt torque. `maxAlpha` is therefore the *net* yaw acceleration, `1/τ + drag` keeps
    the closed-loop pole the droopy loop already had, and `headingHoldKp` went 8 → 5 because the loop
    gain rose by 1/0.58 (5 reproduces the old hold's small-step response: 0.38 s rise, 0.5% overshoot).
    Two consequences: a full-stick 60 °/s plane-mode turn is now actually flown (it used to be pinned
    at the lead clamp, turning ~130° of a 4 s stick's 240°), and with the drag gone a stopped turn or
    a knock is arrested by `maxAlpha` alone — stopping from 60 °/s passes the target by ~9°.
  - **`yawFilterCoefficient = 1` (off) is the faithful setting, not an oversight.**
    `joystick_controller` smooths pitch, roll and gimbal pitch (`STICK_SMOOTHING_ALPHA`) and feeds the
    yaw stick in **raw**, because the heading hold is what makes smoothing unnecessary.
  - **The disturbance it rejects is common-mode**, so no aggregate ever suppressed it: every drone
    takes the same world-frame velocity command, tilts the same way at the same instant and gets the
    same heading kick, and a circular mean of *n* identical excursions is the excursion. That is why
    the screens slid *as a group*, and it is the cleanest confirmation of the diagnosis — if a
    per-drone excursion and `swarmMeanYaw`'s ever differ materially, something else is going on.
  - Seed the setpoint, never ramp it (`Start`, the dead branch, `ResetToPos`). `ResetToPos` must read
    the **transform**, not `State.Angles.y`: `StateFinder.ResetToPos` has just written euler *degrees*
    into a field everything else reads as radians, and it is only corrected by the next `GetState()`.
- **`VelocityControl`'s angular gains used to be anisotropic by accident, and every yaw-vs-tilt
  asymmetry traced to it.** `desiredTorque = Scale(desiredAlphaClamped, State.Inertia)` was applied
  with `ForceMode.Acceleration`, which *ignores* the inertia tensor — so the pre-multiply was never
  cancelled and the achieved angular acceleration was `α × I`: pitch/roll ×0.3893, yaw ×0.7688. It is
  now `ForceMode.Force`, with `timeConstantAlphaRate` (one constant for all three axes, not two) and
  `maxAlpha` rescaled by 0.3893 so the pitch/roll response is **unchanged**. Two things follow:
  - **That anisotropy is what broke the `tiltHeadingLeak` cancellation.** The cancellation makes
    `Dot(desiredOmega, upBody) == effectiveYawRate` exactly, but only in *commanded*-rate space; an
    anisotropically scaled achieved rate is rotated off `upBody`, so a tilt change leaked heading
    however carefully that line was written. `tiltHeadingLeak` is nonzero only while the tilt is
    *changing*, which is exactly why the symptom appeared on velocity-command changes and never in
    steady flight. **The three body axes must keep one time constant** or it comes back. The yaw
    part's own law (drag compensation, above) does not break this: it acts on the world-vertical
    component, which isotropic damping and an `alphaTilt` perpendicular to `upBody` leave decoupled
    from the tilt part outright, whereas the rule is about the body axes the yaw command is spread over.
  - **The α clamp is split about `upBody`, not about the body axes.** The old clamp scaled the `(x, z)`
    pair circularly while clamping `y` independently, which rescales the tilt and yaw parts of
    `desiredOmega` by different factors and destroys the cancellation exactly when tilt demand is
    highest — and that is the common case, not an edge case: a full-stick step commands far more body
    rate than `maxAlpha` can deliver, so pitch/roll sits at its clamp for most of a second on every
    stick movement. The circular form is kept for the tilt part for its original rotation-invariance
    reason. Fixing the cancellation analytically instead was considered and rejected: it would hard-code
    the inertia tensor, `angularDrag` and the timestep into the yaw channel, would do nothing under
    saturation, and would break silently the day this `ForceMode` was corrected. The heading-hold
    integrator is the model-free version — which is why the real FC needs none of those numbers either.
    The yaw drag compensation is not that fix coming back: it buys *authority* the α clamp refuses,
    which no model-free loop can, and it reads `angularDrag` and the step live rather than baking them in.
  - **The linear twin is still there and is deliberately untouched:** `desiredForce = thrust * Mass` is
    also applied as an acceleration, so achieved vertical acceleration is 3× the computed thrust. The
    height PD absorbs it by sitting at an offset setpoint (~1.3 m above `desired_height`) and the 2.7 g
    clamp really bites at ~8 g. Fixing it means retuning `HeightKp`/`HeightKd` and re-reading the clamp
    — its own change, with its own altitude-hold comparison.
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
- **The FPV cameras are the entire rendering cost, and they are driven manually.** The pilot's eye
  cameras are culling-masked to the feed-screen layer (`ScreenSpawn.screenLayerName`, "UI"), so the
  stereo pass draws the screens and the curved panorama and *not* the world — every full pass over
  the city is one drone's FPV camera rendering into its feed RenderTexture. Left `enabled`, the
  visible ones redraw the city at the headset's rate: at 10 drones, 90 Hz and `STABSTITCH` (which
  hides 3), that is 7x90 + 3x30 = **720 full-city renders per second**, and the eye pass is a
  rounding error beside it. Consequences worth keeping:
  - **`ScreenSpawn.feedRenderHz` (default 30) is the dial**, not the resolution. A feed screen is
    video, so it is refreshed at `sendInterval`'s rate rather than the headset's — 720 to ~300
    renders/s, measured against a replica of the schedule. `0` restores the pre-throttle behaviour
    (cameras left `enabled`) and is the A/B control for any measurement.
  - **The stagger is not a refinement.** Each camera carries a phase offset of its share of one
    interval (`SeedFeedRenderTime`), so ~3 render per frame instead of all 10 landing together. The
    average is identical; the peak is what drops a headset frame. That is also why a screen which
    becomes visible is *re-phased* rather than made due now — seeding it to "due" would fire it
    alongside every other screen doing the same (on the first frame, all of them) and they would
    then advance in lockstep forever, which is precisely the spike the stagger removes.
  - **Two independent schedules drive the same cameras**, so the render gate lives on the camera:
    `FPVCameraScript.EnsureRenderedThisFrame` stamps `Time.frameCount` and both
    `ScreenSpawn.StepFeedRenders` and `PyUniSharingFast.RequestBlockCapture` go through it. A
    drone that is both displayed and stitched therefore costs one render, not two, and
    `RequestBlockCapture` can no longer test `!camera.enabled` to mean "nobody else will draw it"
    — under the throttle every FPV camera is disabled. A camera whose GameObject has no
    `FPVCameraScript` falls back to being left `enabled`: with nothing to drive it manually,
    disabling it would freeze that feed rather than merely slow it.
  - **Stitch captures are deliberately NOT staggered.** They stay synchronised at `sendInterval`,
    because spreading a triplet across frames introduces exactly the capture skew
    `MAX_CAPTURE_SKEW_S` exists to reject.
  - `allowHDR` is off and `stereoTargetEye` is `None` on every FPV camera, set both on
    `DroneReduced.prefab` and again in `ScreenSpawn` beside `aspect`/`fieldOfView`/`targetTexture`
    so a scene override cannot reintroduce them. There is no post-processing on these cameras and
    the stitcher consumes 8-bit BGR, so the FP16 intermediate bought nothing; `Both` on a camera
    that renders into a RenderTexture risks the built-in XR path treating the pass as stereo.
  - **Occlusion culling is worth baking and is not code.** The flight scenes ship with
    `m_OcclusionCullingData: {fileID: 0}` and a zeroed `m_SceneGUID`, so each FPV camera submits
    the whole city with nothing hidden behind anything. The static flags are already right (1,765
    objects in `ScaledCityWorld` at `m_StaticEditorFlags: 4294967295`, Occluder and Occludee
    included), so `Window > Rendering > Occlusion Culling > Bake` is the whole job. It pays at
    street level and very little at altitude.
- **Boundary drones** = `AttitudeAlgorithm.BoundaryEstimate` (convex-hull). Left/centre/right stitching
  and the `OUTER_CIRCLE` screen layout only use boundary drones (see the planar exception above).
- **The look-direction gap fill (`SwarmManager.fillLookDirectionGap`, off by default) is the one
  place the hull heading rule knows where the pilot is looking.** The outward bisector rule leaves a
  blind spot straight ahead when the swarm splits around a building: the front drone is swallowed,
  its two neighbours pass either side ahead of the lagging swarm, and the sharp corners they form
  point them 40–50° off the flight direction (the FPV camera is ~98° wide, so ±50° leaves no overlap
  at all). `AttitudeAlgorithm.UpdateLookGapShift` takes the two hull vertices whose headings bracket
  the body yaw ψ and turns each towards it by `clamp(d_near − lookGapCoverageDeg, 0,
  lookGapMaxShiftDeg)`; every other drone gets exactly zero. Points worth keeping:
  - **Body yaw (`PyUniSharingFast.BodyYawDegrees`), not the head.** Body yaw aims the panorama and
    the VR velocity frame, so it is where the pilot is flying; the head moves on every glance at a
    side screen. `BodyYawValid` exists for this consumer — 0° is north, and the bench disables
    `PyUniSharingFast`, so an unseeded body yaw would swing the swarm's front round to face north.
  - **Only the bracketing pair moves, and the one expression is its own gate.** s is 0 while a drone
    already faces within the coverage angle; s ≤ d_near, so nothing crosses ψ or reorders; and the
    pair only changes where ψ crosses a heading, i.e. where s is 0 — no new edge to dither across.
    The shift is added to `rawTargetHeading` *before* the target low-pass, which smooths a change of
    pair like any hull deformation.
  - **Computed once per tick in the shared hull pass, not per drone**, so the drones, the
    `SharedLookGap*`/`SharedMaxGapDeg` read-outs and the shape CSV columns cannot disagree.
    `SharedMaxGapDeg` measures the *shifted* headings, so with the fill on it shows the price: the
    gaps beside the pair widen by s.
  - Acts only in `GLOBAL_CONVEXHULL` (a local hull makes nearly every drone a vertex), outside
    vertical-plane mode (its own shared heading) and above `FPVCameraScript.NadirPitch` (a downward
    camera has no direction to fill). The defaults (20°, 15°) take a ±50° front pair to ±35°.
  - **It is not idle in cruise, and that is measured, not assumed.** An evenly spread ring of 9–10
    hull drones keeps ψ within 18–20° of a heading, but a flying swarm averages ~7.7 of 10 on the
    hull, so on ScaledCityWorld's 120 held-out flights (headless, pilot facing each waypoint) the
    fill turned a pair ~27% of the time, by ~7° on average. What it bought: ψ more than 30° from
    every shown drone's live heading 8.8% → 1.1% of the time, outside every shown image 0.4% → 0,
    and 35° → 23° during the large gaps. Drones lost 18 → 16 and building contacts 1 → 1, with
    every loss a single drone-drone collision in a different flight in each arm — heading does not
    feed translation, so the flights merely diverge. Raise `lookGapCoverageDeg` to keep it for the
    large gaps only.
  - Off, the shift list is all exact zeros and the hull rule is skipped past untouched. Sim-only:
    DJIScene has no `AttitudeAlgorithm`.
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
  - **The texture consumer must take its screens from `ScreenSpawn.Screens`, never from
    `FindGameObjectsWithTag("Screen")`.** That search returns only *active* objects, and a layout
    legitimately deactivates a screen whose feed has not arrived — at spawn time, all of them. The
    two form a cycle: `ImageSharing` needs the screen to bind a texture, and binding the texture is
    what lets the per-frame update push heading/position back into `ScreenSpawn` and un-hide it. Tag
    search breaks that cycle permanently and the `screens.Count == 0` retry in `Update` cannot
    recover it, because the objects it is looking for stay invisible to the search forever. The
    symptom is a DJI scene that displays nothing at all, with no error — only a debug line reporting
    0 screens found.
  - **The display yaw is low-passed so a screen's position stays in phase with its own picture.**
    `TryGetDisplayYawRad` is filtered at `InterfaceManager.displayYawSmoothTime` (0.1 s, = `1 /
    FPVCameraScript.smoothSpeed`) and `TryGetRawDisplayYawRad` is the unfiltered read. This is a phase
    fix, not cosmetic smoothing: the circle styles place a screen at its drone's yaw, read raw off a
    50 Hz physics value the Rigidbody does not interpolate, while the *pixels* in that screen come
    from the FPV camera, which Slerps onto the same heading — so the screen led its own imagery.
    Points worth keeping:
    - **The two formation grids deliberately take the raw value.** They never place a screen at a
      per-drone yaw at all (`wallAnchorAzimuth` plus a cell offset), so the per-drone yaw only feeds
      the aggregate circular mean and the tier-2 in-plane basis — where filtering moves no screen and
      only perturbs the aggregate, which already has its own low-pass at `formationWallSmoothTime`.
      Stacking the two would lengthen the wall's swing by stealth, and the tier-2 basis has to agree
      with tier 1 (`GetPlaneAxes`), an unfiltered setpoint.
    - **The filters are advanced in one pass per frame** (`StepDisplayYawFilters`), guarded on
      `Time.frameCount`, because `UpdateScreenPositions` is *not* called once per frame — `Update`
      calls it and so do `SpawnScreens`, `SetScreenStyle` and the stitch-hide setters. Filtering
      inside the accessor would advance at a rate nobody controls.
    - Circular (wrapped error, or it tears at ±π), seeded rather than ramped so a screen does not fly
      in from azimuth 0, and **invalidated rather than held** when the raw read fails — a feed that
      drops out and returns must re-seed, not sweep back from where it went quiet.
    - Real feeds use the same filter and the same constant, with no `IsRealFeed` branch. Their yaw
      arrives as a coarse staircase (pushed only on a new frame), and the gimbal has its own lag of
      broadly this order — but that lag is **not measured**, so the number is a reuse, not a
      derivation. Branching would also break the one-accessor-per-quantity rule above.
  - **Aliveness is freshness, not a flag.** A real feed is suppressed once
    `Time.time - lastUpdateTime` exceeds the timeout `ImageSharing` pushes from `stitchFrameMaxAge`,
    so the screens and the stitch selection agree on what "still flying" means and a drone that stops
    streaming leaves the layout instead of freezing in it. The staleness test lives in `TryGetFeed`,
    so no accessor can hand a layout a dead drone's position either.
  - **An unposed feed (`poseStatus == 0`) costs its own screen, not the layout.** It has no ranking
    key, so the grid styles drop it, exactly as the planar solve drops an unposed block; the circle
    styles still show it, since they need only the heading.
  - **`hideStitchedDroneScreens` is split across two components, one per feed source, because the
    toggle and the selection live in different places.** `PyUniSharingFast` owns the flag and the
    panorama-displayed state and publishes their conjunction as `HideStitchedFeeds` (static, like
    `PlanarSelected`, and set before the `ScreenSpawn` guard so it is honest in a scene with no
    layout). Membership comes from whoever chose the views: the sim half resolves its camera indices
    to `"Drone N"` GameObjects (`SetStitchedDronesHidden`), while `ImageSharing.PublishStitchBlocks`
    pushes the ids it actually wrote to the stitcher (`SetStitchedRealFeedsHidden`), keyed by feed
    index since a real aircraft has no GameObject to match on. `ScreenSpawn` keeps the two sets
    apart and `IsFeedSuppressed` consults whichever the binding is. Doing it all in the sim half is
    what used to fail: with no cameras in the DJI scene its set was always empty, so the checkbox
    did nothing there and every real feed stayed visible on top of the panorama. Hiding a real
    screen is safe in a way the tag-search deadlock above was not — the texture upload and the
    `UpdateRealDroneFeed` push both key off `ImageSharing`'s own screen dictionary, not off the
    GameObject being active, so the feed stays fresh and un-hides the moment the toggle goes off.
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
- **`ScreenStyle.AUTO` is a choice of rule, not a layout, and only `InterfaceManager` may hold it.**
  It follows the configuration — `OUTER_CIRCLE` horizontal looking out, `FORMATION_MAP` once the
  gimbal is at or below `FPVCameraScript.NadirPitch`, `FORMATION_WALL` in the vertical plane — through
  `SwarmPlaneController.StyleForConfiguration`, the *same* function `driveDisplayConfiguration` pushes
  through, so the automatic layout and the automatic stitcher cannot describe different
  configurations. Consequences worth keeping:
  - **InterfaceManager pushes the resolved style down; `AUTO` never reaches
    `ScreenSpawn.screenStyle`.** Every consumer reads `InterfaceManager.ResolvedScreenStyle`, and the
    one path that could smuggle it in (`fallbackScreenStyle`, a `ScreenStyle` field so the dropdown
    offers it) is coerced to `OUTER_CIRCLE` in `ShowFallbackFeeds`. A style with no `case` in the
    placement switches does not *fail*, it freezes the screens where they were — which is the
    fallback appearing to half-engage. `SetScreenStyle` is likewise refused while in `AUTO`: it
    writes a concrete style into the field, so the first configuration change would end `AUTO` for
    the session.
  - **`SwarmPlaneController.TryResolveDisplayConfiguration` answers for *both* feed sources**, because
    the real-drone scene has neither of the sim's signals: no `SwarmPlaneController` (the wall is
    commanded by the PC's `swarm_plane.py`) and no `FPVCameraScript`, whose swarm-wide `SharedPitch`
    would sit at 0 and report a nadir fleet as looking out. Both come off the feed blocks instead
    (`ImageSharing.LiveFeed*`), so nothing was added to the wire: the gimbal pitch is the elevation of
    the block rotation's forward axis — that rotation *is* the gimbal's — and "vertical plane" becomes
    "do the drones share a heading", hysteretic on the circular-mean resultant. That substitution is
    not an approximation of the display's needs, it *is* them: `OUTER_CIRCLE` is only meaningful while
    the yaws are spread, and a wall points every aircraft at one surface. Deliberately **not** a plane
    fitted to the positions — a single-row wall (`_plane_from_formation`'s `forward` case) fits no
    plane and is still a wall.
  - **"No aircraft has streamed yet" is a third answer, not `false`.** Both readings are "horizontal,
    looking out" on an empty fleet, so the resolver returns false and the caller *holds* rather than
    flipping a DJI scene to `OUTER_CIRCLE` and back. A sim scene is never indeterminate: with no
    controller in it, vertical-plane swarming is unreachable, so `false` is the answer and not a gap.
  - **The enum values are spelled out.** Dropping `REAL_DRONE` without pinning them closed the gap at
    5 that its comment claims to leave, moving `FORMATION_WALL` to 5 and `FORMATION_MAP` to 6 — and
    DJIScene, which stores 6, silently became a nadir map layout. `FORMATION_WALL = 6`,
    `FORMATION_MAP = 7`, `AUTO = 8`.
- Image format across the bridge is **BGR + top-down** for stitch inputs; the returned panorama is
  flipped once and converted to RGB on the Python side.
- **Resolution is metadata-driven:** `StitcherThreading.py` sizes inputs/outputs from the Unity metadata
  (`blockImageWidth/Height` + `panoramaImageWidth/Height` in `PyUniSharingFast`'s inspector); set those to
  scale resolution. The StabStitch nets always run at a fixed `NET_W×NET_H`, and the TPS field is
  evaluated on a bounded lattice (below), so the warp update is ~21 ms whether blocks are 768 or
  1280 px wide; only the uploads and copies grow with resolution.
- **Size the resolution to the headset, and move the feeds and the panorama together.**
  `blockImageWidth/Height` is also every feed screen's render-texture size (ScreenSpawn adopts it), so
  the stitch input can never exceed what the feed screens show. The Quest Pro resolves ~22 px/deg: an
  `OUTER_CIRCLE` feed (radius 2 m, scale 1) spans ~48°, so ~1024 px is its useful maximum, and the
  curved screen (radius 5 m, 90°, 3 m) wants ~1980×735. The stitch canvas carries ~1.8× the block
  width of detail, so **1024×576 blocks pair with a 1920×720 panorama** at ~21 texels/deg on both
  screens; the sim scenes used 768×432 / 1600×600, ~16 texels/deg on both. 1280×720 (the envelope
  max) would exceed the headset and only cost FPV render time. Feed render textures are mipmapped,
  because the grid layouts show the same texture on screens half `OUTER_CIRCLE`'s size. The
  `REFERENCE_BLEND` widths (`blurKernelSize`/`blurSigma`/`borderSize`) are canvas pixels, so they
  scale with the block width: 41/15/60 suit 768, the 1024 scenes use 55/20/80. DJIScene stays
  800×450, the real feed's size.
- **The TPS field is evaluated once per warp update on a lattice of at most 512 samples** along the
  canvas' longer side (`_field_lattice_size`, `STABSTITCH_FLOW_GRID`, 0 = exact) and resampled
  for each consumer: the canvas (blend masks), the panorama (`_field_to_image_grid`, with
  cv2.resize's half-pixel geometry), and the quality canvas. The field over a 7×9 control grid is
  smooth enough that the worst sampling error is 0.05 source px at every resolution (asserted), where
  a half-size lattice cost 4 ms at 768-wide blocks and 10 ms at 1280. Resample through the
  `align_corners=True` lattice helpers, never by hand, or the panorama shifts half a pixel.
- **STABSTITCH render/warp are decoupled:** the render loop (`stab_pano`, paced by Unity's
  `sendInterval`, 30 Hz) uses cached warp params only (no neural net); a separate warp thread
  (`compute_warps`) runs the nets and updates the cache. The render warp is a single `grid_sample`
  over a **precomputed TPS sampling field** (`_compute_tps_flow`, cached per warp update) — the
  float64 TPS solve + per-pixel RBF live in the warp thread, not the render loop. The render costs
  ~2 ms of CPU and <1 ms of GPU per frame; **its rate is Unity's publish rate**, so the lever is
  `sendInterval` (+ `readInterval`), and the cost of raising it is the hidden stitched FPV cameras
  being rendered on demand per send — the headset frame rate is the constraint.
- **The nets run incrementally, and the temporal window has its own cadence.** SpatialNet's output
  depends only on its frame and TemporalNet's only on a frame and its predecessor, so both are
  computed once when a frame is *admitted* to the 7-frame buffer and cached beside it
  (`_ingest`); each warp update runs the nets only on frames admitted since the last one and
  re-runs the cheap SmoothNet over the window (`_window_params`). That is the paper's online
  protocol without the 7× repetition (150 ms → ~40 ms per update, so the warp thread keeps pace
  with admissions). Frames are admitted at `NET_FRAME_PERIOD` (0.05 s, the cadence the sim always
  fed the nets at), **not** at the render rate — a faster render must not shorten the ~350 ms
  smoothing window. `STABSTITCH_LEGACY_WARP=1` restores the full-window recompute for comparison;
  `tools/stabstitch_selftest.py` asserts the two agree (fp32/deterministic: ≤0.01 px; under
  default TF32/autotuned cuDNN they drift by a few tenths of a pixel run to run, which is float
  noise, not logic).
- **A different drone triplet is a different video, and the window starts over** (`_restart_video`,
  triggered by the `view_ids` `process_stitching` passes, or by `VIDEO_GAP_S` without admissions).
  Left alone, the window kept the previous triplet's frames: TemporalNet read the jump between
  unrelated images as motion and SmoothNet pulled the new warp towards the old drones' meshes, so
  every new panorama warped for its first ~0.35 s (16–33 px off in panorama pixels, against 0.4 px
  of normal jitter). A new video's window is **padded with its first frame repeated**
  (`pad_new_video`), which is exactly what the nets compute for a still camera, so the panorama
  appears on the first frame instead of after 7 admissions of side-by-side concat. Keep
  `BUFFER_LEN` at 7: on replayed sim frames 5, 9 and 11 all jittered more than the trained value,
  and `NET_FRAME_PERIOD` 0.033 did too.
- **The nets' input is antialiased** (`lr_antialias`, `STABSTITCH_LR_ANTIALIAS=0` restores the
  cv2-identical input). The paper downsamples with `cv2.resize` INTER_LINEAR, which skips source
  pixels past 2×: harmless on lens-blurred camera footage, not on pixel-sharp renders. At 1024-wide
  blocks (2.1×) the aliasing shimmers frame to frame, the nets track it, and the displayed warp
  jittered ~30% more than at 768; antialiased, it jitters less than 768 did. No weights, layers or
  maths change — only the filter that produces the 480×360 input.
- **The quality debounce is timed** (`QUALITY_HYSTERESIS_S`, 0.35 s), not a count of warp updates:
  two updates was ~0.35 s at the old 5–6 Hz warp rate and only 0.1 s at 20 Hz, enough for a PSNR
  near the threshold to flash the panorama on and off.
- **Measure flicker on real frames, not by eye:** `tools/stabstitch_flicker_replay.py record`
  captures the three stitch slots from a playing Unity without taking a block flag, and `replay`
  runs them through the stitcher on a simulated clock and scores the displayed control points'
  motion in panorama pixels (jitter, warp shape, the paper's roughness score, the start of each
  segment; `--switch` simulates a triplet change).
- **`StabStitch2_main` is gitignored, and the perf edits to it live in
  `tools/stabstitch2_perf.patch`.** The vendored forwards used to build small constant tensors on
  the host and `.cuda()` them every call, and `torch.inverse`'s singularity check is a host sync;
  ~45 of the 99 device syncs per warp update were there. The patch caches the constants and uses
  `inv_ex` — no maths, weights or layers change, and the self-test rebuilds the pristine copy by
  reverse-applying the patch and checks the outputs match. After re-downloading StabStitch2,
  re-apply it from `Codes/` with `git apply <repo>/Assets/Scripts/ImageStitching/tools/stabstitch2_perf.patch`.
- **Never pass a 0-d CUDA tensor where a Python number will do** in the mesh maths: `float(t)` is
  a device→host sync, and `_get_norm_mesh`/`_recover_mesh` were called 16× per update with the
  canvas size still on the GPU. The canvas bounds come back in two `.tolist()` calls and the
  quality metrics in one `.cpu()`; a warp update now performs 4 syncs (budgeted in the self-test).
- **The render produces the wire layout directly** (`WireReadyPanorama`): per warp update the
  TPS field and the blend weights are resampled to the panorama size with the rows reversed,
  so the render's `grid_sample` lands RGB, bottom-up, at `panoramaImageWidth × Height`, and
  `first_thread` writes it without the `cv2.resize`/`flip`/`cvtColor`/`tobytes` pass it still
  applies to the other stitchers' BGR canvases. Inputs go up as uint8 through pinned staging and
  are converted on the GPU; the low-res net input is `F.interpolate` on the GPU, which matches
  `cv2.resize(INTER_LINEAR)` to within one uint8 step (asserted).
- **The two threads share one GPU and one GIL**, so wasted render work directly slows the warp update.
  `first_thread` wakes the render only for a **fresh** block — a slot whose header `captureTime`
  advanced (peeked without the flag handshake, so an unchanged block costs no copy and never holds
  the flag against Unity's readback callback) — floored at `RENDER_MIN_PERIOD` (0.025 s) in
  `StitcherThreading.py`. Keep that floor just under Unity's `sendInterval` (0.0333 s); it bounds
  the duplicate wake-ups when Unity's three readbacks complete on different frames.
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
| `PyUniSharingFast` | `scenePlaneMode = FormationRelative`; `planarStandoffMetres` is now only the **fallback** — see below |
| DJI_Swarm | `-ImageStreamPose` / `ImageStreamPose = $true` / `--image-stream-pose` |

**`planarStandoffMetres` is supplied by the PC and the inspector field is the fallback.**
It used to be hand-typed, and a wrong value there is the single largest error in the mosaic — the
2026-08-11 MED clips were replayed against a typed 30 m when the truth was 34.3 m, which at this
geometry's 4.9 px per metre is ~21 px of seam, more than every other term combined. The PC now
computes it (`clip_scene_plane.pick_facade` + `standoff_of`) and writes it into the feed map's
trailer; `ImageSharing.ReadFeedTrailer` validates it and `PyUniSharingFast.ResolvePlanarStandoff`
prefers it, writing the result to metadata offset 364 every frame via `WritePlanarStandoff` —
per-frame because the standoff now *tracks the formation* rather than being a constant, and
`WriteMetadata` only runs on Start/OnValidate/a stitcher switch. Four things must hold before the PC
value is used (magic, version, status, and a **heartbeat** advanced within 2 s); the heartbeat is the
load-bearing one, because closing the producer's handle leaves its last bytes readable forever and a
finished `clip_replay` would otherwise pin its clip's standoff into every later editor session.
The inspector field is still worth setting correctly: it is the only source in every **sim** scene,
in the DJI scene before a controller starts, and again the moment one exits.

**Three inspector fields carry a "plane distance" and none of them are the same quantity** —
`scenePlaneMode` selects one and the other two are dead, which is why `PlanarSettings` groups them
under per-mode headers. `fallbackPlaneDistance` (Nadir/Facade) is a distance **along the centre
camera's ray**, used *only when the raycast misses*, and is unreachable on real drones because a
facade has no collider. `manualPlaneDistance` (Manual) is the world plane offset **d** itself,
paired with `manualPlaneNormal`. `planarStandoffMetres` (FormationRelative) is the **perpendicular**
distance from the formation **centroid** to the surface. Different origins, different directions,
different roles; they merely all read about 30.

**`planarStandoffSource` (`Auto` | `Inspector`) is the operator's takeover**, and
`planarStandoffInUse` is a read-only read-out of which one is actually in force. `Inspector` is
absolute — it ignores a live PC value even when one is arriving — because an operator reaching for
it is overriding an auto-pick they believe is wrong, and silently reverting the moment the PC looked
healthy again would be the opposite of what they asked for. It is named `Inspector` rather than
`Manual` so it cannot be read as `scenePlaneMode`'s unrelated `Manual`.
**`planarStandoffMetres` deliberately does *not* mirror the live value**, which is the obvious thing
to want and is a trap: that field is the *fallback*, so writing the live value into it means the
instant the PC stops publishing the fallback becomes the last PC value rather than the operator's
number — a finished `clip_replay` would leave its clip's standoff sitting in the inspector and every
later session would inherit it silently. That is precisely the stale-value failure the heartbeat
exists to prevent, reintroduced through the inspector. Hence a separate read-out.
The **resolved** source (not the requested one — `Auto` still reads `inspector` whenever nothing is
arriving) is published at metadata offset **396**, taken out of `metadataReservedGap`, which drops to
7 with `metadataTailEnd` at 397; `metadataSize` stays 412 so no running Python is stranded. It exists
because the value at 364 is a float either way: without it `clip_replay.py` goes on announcing that
it drives the standoff while a `Manual` scene quietly ignores it, which is the same
two-sources-one-number confusion the toggle was added to remove. `unity_stitch_meta.py` reports it
(`standoff 34.26 m (from the PC)`) and watches it, so a mid-replay flip to `Manual` prints an
`[unity]` line.

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
  - A facade traced on the DJI_Swarm GUI map (`shapes.json`) is the georeferenced route, and **is now
    the default** — it is what supplies the standoff described above. The wall is chosen
    automatically (`clip_scene_plane.pick_facade`: nearest positive ray-plane hit along the mean
    camera forward, gated on look-off/range/lateral extent, with 3 m hysteresis and a 2.5 s dwell so
    it cannot flap), and which wall won is reported on the controller console and the GUI chip. Verify
    it offline against real footage with
    `python clip_scene_plane.py --clip <dir> --pick [-v]`, which replays the rule over a clip's own
    poses and prints every candidate's reject reason. Two traps remain: **trace the base** of the
    building, not the roofline — satellite imagery displaces the roof from the footprint by
    `height × tan(off-nadir)`, ~7 m for a 20 m building; and the behind-the-wall test is the
    **look-off angle, not a negative standoff**, because `facade_from_line` orients the normal toward
    the formation, so a wall traced on the far side comes back with a perfectly positive standoff.
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

## City scenes and the ScaledCity assets

Flight scenes: `CityWorld`, `ScaledCityWorld`, `CrowdWorld` (Modular City Pack cities), `FactoryScene`,
`RingChallenge`, `NBackExperiment`, and `DJIScene` (real drones, no sim swarm). All of them spawn the
same `Assets/Prefabs/DroneReduced.prefab`, so anything edited there — `maxPitch`, `maxSpeed`,
`timeConstantAcceleration` — changes every scene at once. Per-scene tuning belongs on the scene's own
`SwarmManager`, which is the only per-scene owner of those values.

**`ScaledCityWorld` is a fork of `CityWorld` with its own prefab tree**, not a scaled instance of the
original. `Assets/Prefabs/ScaledCity/` holds `City_Pack_01_Scaled`, `goal_patch_Scaled` and 48
`Patches/MC_Patch_NN_Scaled` prefabs; nothing outside that scene references them. Across those patches
sit **169 buildings, each at `localScale (0.25, 0.25, z)`** — a quarter of the pack's footprint, with a
per-building `z` chosen so **every building is exactly 50 units tall**. Footprints run 3.5–17.9 by
2.0–10.8 units. The point is a city whose streets are wide relative to its buildings and whose skyline
is flat, so obstacle behaviour is comparable between runs instead of being dominated by whichever
tower a drone happened to meet.

Pack-geometry facts scripts depend on:

- **Buildings are authored local `+Z` up, pivot at the base** — the `0.25/0.25/z` above is
  width/width/**height**, not width/height/depth. `BoxCollider.m_Size` is the *unscaled* mesh bounds,
  so world size is `m_Size` times `localScale` componentwise.
- **The pack marks every building `Batching Static`**, in the ScaledCity forks as well as the
  originals. Entering Play combines their meshes and the renderer stops reading the transform, so a
  runtime scale or move shifts the **collider** without moving anything you can see — invisible walls.
  `BuildingWidthTuner.ApplyWidth` refuses to run once batched and says so. **Tune in edit mode before
  pressing Play**; the value carries into Play without saving the scene.
- **`BuildingWidthTuner.widthScale` is relative, not absolute.** 1 always means "leave it alone", so
  the component is inert until touched. ScaledCityWorld already bakes 0.25 into its prefabs and
  CityWorld bakes 1, so an absolute scale would need a different baseline per scene and would silently
  quadruple the wrong city; in ScaledCityWorld, 4 restores the original pack width. `StreetWidthTuner`
  is the complementary knob — it widens streets by shrinking each tile's block content about its own
  kerb pivot, leaving the `Road_Structure` plate and the 90.83 grid pitch alone. Buildings are moved,
  never resized. **The widened strip is neither road nor plate**: `Road_Structure` is a *ring* from the
  tile edge in to 38.1 u (`CityTiles.BlockHalfSpan`; its area is 0.297 of its bounds, which is exactly
  the ring around a 76.2 block), empty inside where the footpath slab sits. Shrinking a block therefore
  opens a band onto whatever is under the city — in ScaledCityWorld, four grass Terrains at y −0.017 —
  so each block gains a grass verge (7.6 u wide at 0.8) and the carriageway does not widen at all.
- **A tile's transform is not always where its block is — locate a tile by its kerb (`Carbs_NN`).**
  `MC_Patch_32`, in the pack and in the fork, has its whole block baked ~318 u off its own pivot, so its
  transform sits at the city centre while its buildings stand at the edge (seven other tiles carry a
  baked vertical offset instead). `StreetWidthTuner` and `GoalPatchReplacer` both read the kerb. The
  replacer used to copy the transform, which put a goal patch across the four middle tiles — it happened
  in `AAAA_t1_SingleDrone_20260706_214605` and `ERIC_t7_Swarm_20260706_232027` (goal 2 in both) — and
  shrank the pitch its adjacency test measures to ~64 u, so diagonal goals were allowed.
- **Only the blocks live inside the tiles.** Straight under the city root sit 12 `Green_Belt_Tile`
  groups, a `Garden`, six stray trees, and 48 `Road_Structure` plates that duplicate, to within 0.6 u, the
  plate every tile already contains — and are pivoted on a tile *corner*, so assign them by mesh centre,
  never by pivot. The green belts' 96 grass strips and 672 trees are the **street medians**: authored
  44.9–45.8 u from a tile centre, i.e. on the line between two tiles, down the middle of every street;
  nothing else is further out than 34.8 u. Moving a tile leaves all of it behind.
- **Tied scenery hangs under one of three tile children, and the name is the behaviour**
  (`CityTiles.BlockShare`, which `StreetWidthTuner` applies as a per-item scale `1 + share·(s − 1)` about
  the tile centre): `Scenery` (on the block, share 1, moves with it), `Street` (the medians, share 0,
  never moves, so it stays mid-street at every block scale) and `Verge` (share ½, stays mid-verge, since
  the verge's inner edge moves with the block and its outer edge does not). Untied, the tuner scaled the
  medians with the block about the nearer tile centre, which walked each one ~9 u onto a verge at 0.8.
  `Tools/Swarm/Tie city scenery to tiles` (`CitySceneryParenter`) divides the tuning back out to get each
  object's authored position (exact to 1e-4 u against the prefab), sorts it by the kerb-line test
  (beyond `BlockHalfSpan` is street), **restores medians to their authored position**, and splits
  straddling groups per tile. A median goes to the tile west of its line (south, for an east–west line),
  never to the footprint test, which is a coin toss on the line — so a strip and its trees always share a
  tile. It re-sorts already-tied scenery too, so it is safe to run again. It has to **unpack the city
  prefab instance in the scene** first, because Unity will not reparent inside an instance (the tiles stay
  linked to their patch prefabs); doing it inside the prefab would re-read the scene's thousands of
  scenery position overrides relative to the wrong parent. `GoalPatchReplacer` hands all three containers
  to the goal that replaces a tile, or every goal would sit in a gap in the medians, visible from the air.
- **`VergeTreePlanter`** plants trees down the middle of that verge on a seeded subset of tiles (or a
  list, or all), skipping spots within `clearance` of anything on the tile and keeping interior street
  mouths clear across the verge. Deterministic per seed and tile *name*, scriptable (`Plant()`,
  `Plant(tiles)`, `Clear()`), and it refuses at block scale ≥ 1, where there is no verge. Plant in edit
  mode: it marks trees Batching Static, which a Play-mode plant cannot.
- **`CityRowOffsetter`** staggers alternate rows (kerb-derived, numbered from the south or west edge) by
  a fraction of a tile, live in edit mode like the tuners, recording what it applied. It moves tile roots
  only, so it **refuses while the city root still holds untied scenery** — tie first. A half-tile stagger
  makes T-junctions whose mouths the other row's median strip runs across, leaves a half-tile notch at one
  end of each shifted row (`centred` splits it), and stops earlier runs from replaying.
- **A tile's block content is "every Renderer under the tile", and that rule only holds for authored
  props.** An authored prop is one object carrying one renderer, so scaling the renderer's transform
  about the kerb moves the prop. A *runtime-spawned composite* is not: a walker's renderers
  (`HumanM_BodyMesh`, the hat's mesh node) are **children** of the object `WalkerPatrol` drives, so the
  same operation pulls the walker apart rather than moving it. It is invisible on the body — a skinned
  mesh renders from its bone matrices and ignores its own transform — and fully visible on the hat,
  which is how it surfaced: hats ~6 m off their walkers, being `(1 − blockScale) ×` the walker's ~30 m
  from the kerb. `StreetWidthTuner` is otherwise edit-mode only, but `MatchNewPatches` runs on the first
  `Update` of a tuned play session — deliberately after every `Start`, i.e. exactly once the walkers
  exist. Hence `SpawnedContentRoot`, which `WalkerPatrol` puts on its container and the tuner skips.
  Anything else spawned under a tile during play needs the same marker.

**The `Obstacle` layer is applied per scene, and the three city scenes disagree.** This is the largest
scene-to-scene difference for swarm behaviour, because `OlfatiSaber` has exactly one membership rule —
a collider on layer `Obstacle` — so name, tag and source prefab are all irrelevant:

| scene | objects on `Obstacle` | state |
|---|---|---|
| `ScaledCityWorld` | 1934, buildings only | **clean** — `ObstacleLayerAuditor` has been run |
| `CityWorld` | 4268, everything | roads, kerbs, lights, hydrants included |
| `CrowdWorld` | 552, everything | same |

The layer lives in scene-level `m_Layer` overrides, **not** in the prefabs: the original pack's patch
prefabs carry no `Obstacle` layer at all, while the ScaledCity forks carry it on their 212 building
objects. Inspecting a prefab therefore tells you nothing — check the scene.

Why the dirty scenes matter: `OlfatiSaber.GetObstacleCylinder` takes the **circumradius of the
axis-aligned bounds**, and a flat plate is the worst possible input to that. `Road_Structure_NNN` is
90.83 × 90.83 × ~0 and becomes a cylinder of radius ~64 units standing over a whole patch — the giant
circle in the gizmos, and a repulsion field with no gap between patches to fly through.
`Tools/Swarm/Audit obstacle layer` reports it and `Tools/Swarm/Restrict obstacle layer to buildings`
repairs it (`Assets/Scripts/Environment/Editor/ObstacleLayerAuditor.cs`). **Run the audit before
concluding anything about obstacle avoidance in `CityWorld` or `CrowdWorld`** — their obstacle gains
are the ones first tuned against `ScaledCityWorld`'s clean layer, which has since been retuned (below).

Even on a clean layer the circumradius is conservative by construction: a 20 × 20 m building presents a
14.14 m cylinder, i.e. ~4 m of phantom shell outside the facade. Every obstacle distance in the tuning
(`d_obs`, `d_shield`) is measured to that cylinder, not to the wall. `ObstacleCylinderGizmos` with
`drawBounds` shows the gap directly.

**`ScaledCityWorld`'s Olfati-Saber gains are tuned for flying the swarm straight *through* the city while
still following the spread stick quickly, and no longer match `CityWorld`/`CrowdWorld`.** Seven
`SwarmManager` values moved together (was → now): `a` 1.4 → 0.8, `delta` 0.2 → 0.08, `c_obs` 2.2 → 14,
`maxObstacleAccel` 4 → 4.57 (the tilt budget), `d_shield` 1.4 → 2.5, `c2_core` 1.6 → 0.6,
`coreRadiusFilterTime` 2 → 0.5. `b` and `c_vm` are deliberately unchanged, and no code changed. On 120
held-out flights in headless Unity (full-stick transits, waypoint tours, straight at a building, bang-bang
reversals) drones lost went 88 → 16, building contacts 160 → 0, clean flights 22% → 95%, for 1% of
progress speed. After a spread-stick step the formation settles in 4.5 s instead of 4.8 s with 7% overshoot
instead of 25%, though a contraction takes ~1.3 s longer to reach 90%. The causes, found by attributing
forces at every crash:
- **Building contacts were cohesion, not the pilot.** `r0_coh` = 20 makes the α-lattice all-to-all, so a
  drone the formation leaves behind a building is dragged through it by every other drone — a median
  6.6 m/s² in the contact events, against an obstacle force that saturates at 4. Lowering `a` is the fix;
  a lower `delta` also fades the pull beyond 16 m, and `c_obs` and the ceiling raise the other side.
- **`a` is also the formation's stiffness, and it is the one dial between crash safety and the spread
  response.** The σ₁ shape ties the slope of φ near `d_ref` to the attraction asymptote (`b` moves the ratio
  by ~5% at most), and contraction is driven by attraction alone. A first pass at 0.7 (with `b` 3.6 and
  `c_vm` 0.02) lost only 4 drones in the same Unity flights but took 7 s to reach 90% of a spread change,
  twice the old time. With every other value above held, the replica measures (drones lost / contacts per
  km, contraction t90, expansion overshoot): old tuning 1.65 / 2.25, 3.4 s, 27% — `a` 0.8: 0.36 / 0.008,
  4.4 s, 8% — 0.9: 0.39 / 0.08, 4.0 s, 17% — 1.0: 0.46 / 0.27, 3.7 s, 24% — 1.2: 0.58 / 0.64, 3.1 s, 35%.
- **The hollow core resists contraction.** Its radius is low-passed from the measured ring, so when the
  pilot pulls the spread in, the core is still sized to the wider ring and pushes out. A 0.5 s filter cuts
  settling by about a third with no change to crashes or the ring. A smaller `coreRadiusFraction` or a
  lower `maxCoreAccel` does not help: both make the ring flip configuration slowly after an expansion,
  and the lower ceiling also drops the hover hull fraction from 1.0 to 0.9.
- **Drone-drone kills are momentum.** At the kill tick nothing pushes the pair together; the closing speed
  (median 2 m/s, p90 5) was built up beside a building a second earlier, by the shield stopping one drone
  while its neighbour flies on. The wider `d_shield` and the lower `delta` make that braking gradual. What
  remains is mostly abrupt full-stick reversals (12 of the 16 Unity losses). `c_vm` would damp it but hands
  a braking drone back the command the shield removed (0.1 cut drone-drone kills 4× and nearly tripled
  contacts) and slows the spread response, so it stays 0. `d_shield` 3.0 saves a few more drones for 5% of
  progress speed.
- **The hollow core's velocity match brakes the swarm in translation** (`vel_obs` is built from absolute
  velocity, but the core travels with the swarm): about 0.7 m/s² at 1.6, switched off beside buildings
  only. `c2_core` 0.6 buys back the speed the wider shield costs; the hover ring still forms.
- **In Unity a contact is worse than a bump.** PhysX friction (default material, μ 0.6) pins a drone that
  cohesion is pressing into a facade; the swarm leaves it and it dies as `TooFarFromSwarm`, or flips as
  `Crashed`. Baseline flights with no contact never split; flights with one were split 39% of the time.
  `DroneHealthMonitor` has no contact check, so count contacts as well as deaths when tuning.
- **Two limits no tuning here removes.** Full stick at the joystick's tightest spread (`d_ref` 0.4, ~1.8 m
  spacing) loses drones to each other even in open sky, under the old values as much as the new; 0.7 is
  safe. And the values are sized for 10 drones: cohesion sums over neighbours, so at 15 contacts come back
  (0.8/km) until `a` is scaled by about 9/(N−1), and drone-drone losses still rise with N.

## Drone prefab hierarchy (relied on by many scripts)

`Drone N` → child `DroneParent` (has `SwarmAlgorithm`, `AttitudeAlgorithm`, `VelocityControl`, `Rigidbody`)
and child `FPV` (the `Camera`). Drones are tagged `DroneBase`; the arena is tagged `Arena`. Many scripts
use `transform.Find("DroneParent")` / `Find("FPV")`.

## Running the Python stitcher

`cd Assets/Scripts/ImageStitching && python StitcherThreading.py`. **Use the `stitching` miniconda env** —
the default `python` has no torch. StabStitch++ models load from
`StabStitch2_main/Full_model_inference/full_model_ssd/*.pth`.

Checkers: `python tools/stabstitch_selftest.py [--frame 1024x576 --wire 1920x720]` (STABSTITCH
equivalence + timings, needs the GPU and the `debug_input_drone_*.jpg` frames),
`python tools/planar_selftest.py`, `python tools/check_wire_layout.py`. `STABSTITCH_TIMING=1` prints the
per-stage warp/render breakdown; the per-thread rate lines print once a second, and the stitch line
carries **Unity's frame rate**, measured from the per-frame heartbeat — the number every bridge
setting trades against, visible with the headset on.

`python tools/stabstitch_bridge_bench.py --frame 1024x576 --wire 1920x720` times the whole Python side
end to end with a stand-in for Unity (metadata, posed blocks at 30 Hz, heartbeat, panorama reads) and
the real `StitcherThreading.py` as a subprocess. It uses sections of its own via
`STITCH_SHM_SUFFIX` — named sections are machine-global, so on the real names it would be a second
producer inside a live editor's session — and so it can run beside Unity in Play mode. It does not
model Unity's frame cost; for that, read the `PyUniSharingFast.*` markers in the Unity Profiler.
