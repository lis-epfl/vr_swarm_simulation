# A single drone with a 360 camera

Status, 2026-11-05: a proposal. Nothing is built in the project. The measurements come from a throwaway probe run
headless on a copy of the project, and nothing has been flown in the headset.

Update, 2026-11-06: B, the cube and the sphere, is built (step 4 below; option A was not built). It is
`InterfaceManager.DisplayMode.SPHERE = 3` with the recorder condition `SingleSphere`, and its recording and analysis
are done too (step 5). It passes a headless check, `ProceduralCityProbe` with `PROCEDURAL_PROBE_SPHERE=1`, in
BlockShapedPatchCityWorld. It has not been flown in the headset, and its frame rate there is unmeasured. CLAUDE.md
holds its invariants.

## Summary

- **Why.** The trials compare the swarm with one drone carrying the same camera. The swarm differs from that drone in
  two ways at once: its cameras face every way (all-round coverage), and they sit in different places (a distributed
  perspective). One drone with a 360 camera has the first without the second, and two products now sell exactly that:
  the DJI Avata 360 and the Antigravity A1.
- **Unity can do it, but has no ready-made 360 camera.** It has the parts (`Camera.RenderToCubemap`,
  `RenderTexture.ConvertToEquirect`), and nothing in the project uses them yet. A probe in BlockShapedPatchCityWorld
  rendered a full 360 view in 6.9 ms of main thread at 2048 pixels per cube face. That should fit the 13.9 ms a frame
  has at 72 Hz if the view refreshes every other frame.
- **There are two ways to show the pilot the view.**
  - **A:** put the headset rig (`OVRCameraRig`) on the drone, so the headset renders the city from the drone.
  - **B:** render a cube camera on the drone and show it on a sphere round the pilot, who stays in the arena.
- **B is closer to the products.** Both products send the goggles a mono 360 video at 30–60 frames a second, a tenth
  of a second or more late, and far less sharp than what they record. B is that kind of camera by construction. A, as
  built, is an ideal 360 camera and better than either product: stereo, 72 Hz, no delay, anti-aliased, and it follows
  the head's movement as well as its turning. Settings can move A part of the way towards the products, one at a time,
  and some of them are untried.
- **Which is better depends on the question.**
  - *Would a commercial 360 drone do as well as the swarm?* B, shown as a sphere, at the same image quality as the
    feeds.
  - *How much of the swarm's lead is coverage, and how much its spread-out viewpoints?* Neither. The cleanest arm is
    one drone carrying a ring of cameras, shown on the swarm's own screens (section 7).
- **Suggested order.** A first, as a half-day test of whether a pilot can fly and search this way and tolerate it.
  Then the ring arm if separating coverage from perspective is the aim, and B for trials of a product-like arm.

## 1. The question

| Arm | Cameras | Display |
|---|---|---|
| Swarm (today) | ten drones, one forward camera each, spread round a formation | one screen per drone on a ring round the pilot (`OUTER_CIRCLE`) |
| SingleDrone (today) | one drone, one forward camera | one screen |
| 360 drone (proposed) | one drone, a camera that sees every way | a sphere round the pilot, or the swarm's ring of screens (section 7) |

| Comparison | What it isolates |
|---|---|
| One camera → a ring of cameras on one drone | all-round coverage |
| A ring on one drone → the swarm | distributed perspective, plus the formation morphing round buildings |
| A ring on one drone → a 360 sphere | the product-style interface |

S2 and S4 found that the swarm's lead over one drone came from its cameras rather than from its morphing, but neither
separated coverage from distribution. `citysim` could predict that split before anyone flies: several of its
visibility functions already take the camera's horizontal field of view as an argument, though others hard-code it.

## 2. What the products show the pilot

Both carry two fisheye lenses, one on top of the drone and one underneath, stitch them into a sphere, and let the pilot
look round it in goggles by turning their head while the drone flies on. The table is drawn from reviews and the
makers' pages (listed under Sources); a figure from a single source says so.

| | DJI Avata 360 | Antigravity A1 |
|---|---|---|
| On sale | reviewed from March 2026 | launched December 2025 |
| Camera | two 200° fisheyes, top and bottom; 1/1.1-inch sensors | two fisheyes, top and bottom; 1/1.28-inch sensors |
| Recording | 8K at up to 60 fps | 8K at 30 fps, 5.2K at 60 |
| Live view | head-tracked, with Goggles 3 or N3 and the RC Motion 3; 1080p at up to 60 fps in 360 mode (DJI) | head-tracked, in the Vision goggles; "2K at 30 fps" (one review) |
| Delay | about 130 ms at 1080p/60 to the RC 2 or a phone (DJI); no figure for the goggles | "over 100 ms" (one review) |
| Goggles | not stated in the pages read | 2560 × 2560 per eye, 90°, 72 Hz |
| Flying and looking | the drone keeps its course while the view follows the head | the drone flies where the Grip controller points, whatever the head does |
| Controls | wrist tilt (RC Motion 3); sticks (FPV Remote Controller 3, or the RC 2 with its own screen and no goggles) | the Grip only: point-to-fly (forward only) or wrist tilt; no sticks |
| Flight-direction cue | one review mentions a window showing where you are flying, without saying on which screen | a small floating window or marker while the pilot looks away from the direction of travel |
| Drone body | erased, in the goggles too (one review) | erased in the live view (two reviews) |
| Seam in the live view | "visible", with an occasional warp (one review) | not stated; one review saw the picture stutter or blur when the head moved |
| Stereo | not stated | not stated |
| Comfort | one reviewer "didn't feel too good after a full battery" | one reviewer wore the goggles for two hours "without feeling any discomfort" |

What this means for a simulated 360 drone:

- **The view is mono and answers turning, not moving.** Two lenses facing opposite ways see each direction from one
  point, so nothing comes from stereo, and like any 360 video the view changes when the head turns but not when it
  leans.
- **The world arrives as video**: 30 (A1) to 60 (Avata) frames a second, a tenth of a second or more late.
- **The live picture is far below the recording**: 1080p or "2K" for the whole live stream.
- **The seam runs round the horizon.** With the lenses top and bottom, the two images meet in a band at the drone's
  own level, which is where distant streets and facades appear.
- **What nobody says:**
  - whether the live stream carries the whole sphere or only the part being looked at. That decides how sharp the
    live view is (1080p spread over the whole sphere is about 5 pixels per degree, a quarter of a feed here; over a
    crop it could match a feed) and whether turning the head waits for the radio;
  - whether the view turns with the drone when it yaws;
  - the goggles' head-tracking delay, and whether the Avata's head tracking works with its stick controller.
- DroneXL says the Avata's preview field of view can be set to 104°, 134° or 272°, without saying how a 272° view is
  drawn. A squeezed all-round view, like the swarm's ring of screens, may be something a product offers too.
- An earlier note quoted 1088 × 1088 at 50 fps as the A1's live view. In the review that lists it, that figure sits in
  the goggles' table as "vista real" (real view), probably the goggles' passthrough camera, so it is left out here.

## 3. Does a 360 camera already exist in Unity?

- **Not in this project.** Nothing renders a cube map or a 360 (equirectangular) image, or uses a compositor overlay.
  `InterfaceManager.DisplayMode` has `SINGLE` and `BIRDSEYE` values, but only `SCREENS` is ever used.
- **In Unity, the parts exist but not the component.**
  - `Camera.RenderToCubemap(RenderTexture, faceMask)` renders a camera's surroundings into a cube texture. Unity
    documents it for real-time reflections, and the face mask lets a caller refresh only some faces.
  - `RenderTexture.ConvertToEquirect` unwraps a cube into the familiar 2:1 360 image.
  - Realtime reflection probes render cubes too, but their time slicing spreads one refresh over 9–14 frames.
- **Nothing in the installed packages works live.** Unity Recorder's 360 View records from the editor only, and its
  known-issues page says 360 View recording isn't fully supported. The Meta SDK's `OVRCubemapCapture` takes stills.
  `OVROverlay` can show a cube map or a 360 image as a compositor layer, but doesn't make one.

## 4. What the probe measured

The probe ran BlockShapedPatchCityWorld headless in batchmode, on a copy of the project outside the repo
(`D:\claude_feedres_bench`), on the RTX 4090, with no XR. It took 24 positions over the streets at 10 m and 25 m, with
every walker awake (their sleep distance set to 0, so the costs are an upper bound). The GPU was drained before and
after each render, so main-thread time and total time are measured separately. A frame at 72 Hz has 13.9 ms.

| One refresh | Pixels per degree | Main thread | Main thread + GPU | Batches | Fine detail |
|---|---|---|---|---|---|
| One 1920 × 1080 feed | 22.1 | 0.71 ms | 1.0 ms | 63 | – |
| Stand-in for one eye (2160 × 2224, 2× MSAA, 106° × 96°), for cost only | – | 0.78 ms | 1.26 ms | 96 | – |
| Cube, 2048 px faces | 17.9 | 6.9 ms (p95 8.0) | 8.8 ms (p95 10.1) | 371 | 0.55 |
| Six 90° cameras, 2530 px | 22.1 | 7.6 ms | 10.0 ms | 403 | not tested |
| Cube, 4096 px faces | 35.7 | 7.6 ms | 13.2 ms (p95 15.1) | 409 | 0.86 |

Times are medians, with the 95th percentile (p95) where it was recorded. Pixels per degree are at the centre of an
image or a cube face (a face is twice as dense at its edges); the Quest Pro shows about 22. Fine detail is the
high-frequency energy of a view sampled from the cube, against a camera rendered directly at the headset's 22 pixels per
degree (1 is as sharp); where the view straddles two faces it is 0.67 at 2048 and 0.97 at 4096.

What the checks showed:

- **It uses the baked occlusion**: 371 batches and 2.4 M triangles with it, 1,365 and 8.0 M without (6.9 ms of main
  thread against 9.5).
- **The faces cost unevenly**: +X 1.6 ms, −X 1.7, +Z 1.1, −Z 2.1, up 0.5, down 0.4 of main thread. Mipmaps add about
  1 ms of GPU.
- **A cube texture must be a power of two.** At 2530 or 1536 the texture fails to create (`RenderTexture.Create
  failed`), yet `RenderToCubemap` still returns true and draws black. That spoiled the first run, so the code must check
  `IsCreated()`.
- **The faces stay aligned with the world** whatever the camera's rotation (the output was identical with the camera
  turned), so the display has to do the turning.
- **One `OnPreCull`/`OnPostRender` pair fires per face**, so `FPVCameraScript`'s way of hiding the drone's own body
  during its render works unchanged.
- **The image is right.** A view sampled from the cube needs no flips and matches a camera rendered directly from the
  same pose to about 0.7/255 at 2048 and 0.5–0.6/255 at 4096, with no visible seam between faces. Unwrapping a cube
  into a 360 image takes 0.24 ms.

What it means for the frame rate:

- **A cube costs about what ten feeds cost, but all in one frame.** A 2048 cube refreshed at 30 Hz is 208 ms of main
  thread a second (249 at 36 Hz), against 213 for ten feeds at 30 Hz. With the 3.9–4.3 ms the patch city's frame
  already takes, a refresh frame would come to about 9 of its 13.9 ms (an estimate).
- **The rig is the cheapest**: one eye at 72 Hz is about 56 ms a second. The headset renders both eyes in one pass
  (single-pass instanced), so the main thread should stay about the same and the GPU roughly double.
- **A 4096 cube doesn't fit one frame** (13.2 ms before the eyes are drawn), so it would be split, three faces a frame.
  At cruise (7.6 m/s) the two halves are then 0.11 m apart, about 0.6° (13 headset pixels) on a facade 10 m away.
- **Video memory**, colour and depth: 0.20 GB for a 2048 cube, 0.31 GB for six 2530 cameras, 0.81 GB for a 4096 cube.

The limits: headless, so no Link compositor or video encode; the rig was not built, only its stand-in measured; and
only the two baked block cities are covered (BlockShapedPatchCityWorld and PracticeBlockWorld). A render in
ScaledCityWorld costs about four times as much and has no bake, so a cube there would miss frames unless spread over
several.

## 5. Two ways to build it

**A. The headset rig on the drone.** The pilot's rig follows the drone, and the eye cameras, which today draw only the
screens, draw the city. It is less of a hack than it sounds: a headset only ever shows one direction, and rendering
that direction from the drone is exactly what a perfect 360 camera would show there.

**B. A cube camera shown on a sphere.** A hidden camera at the drone renders all six faces into a cube texture 36 times
a second. The pilot stays in the arena inside a sphere whose shader looks up each pixel's direction in the cube, so the
head turns freely at 72 Hz inside the latest cube. That is how a 360 video player works. The probe's lookup, for a
pixel's view ray in the camera's frame:

```hlsl
float3 d = float3((uv.x * 2 - 1) * tanHalfH, (uv.y * 2 - 1) * tanHalfV, 1);
return texCUBE(_Cube, mul((float3x3)_ViewToWorld, d));
```

On the sphere, the world direction is simply the vertex's direction from the head, turned as the screens are turned
(the display puts a world heading ψ at (cos ψ, −sin ψ), i.e. the world turned by 90°).

| | A: rig on the drone | B: cube and sphere |
|---|---|---|
| Main thread per second | ~56 ms | ~210–250 ms |
| Picture | stereo, 72 Hz, 2× MSAA; follows head movement and turning | mono, 30–36 Hz, no MSAA (like the feeds); turning only |
| The pilot stays in the arena | no | yes |
| Can look like a product's live view | partly, through several settings (section 6) | yes |
| Records the 360 view | no | yes |
| Can feed a ring of screens (section 7) | no | yes |
| New code | ~120 lines and a shader fix | ~200 lines and a new shader |

Leaving the arena matters because several things assume the pilot is in it: `CalibrateToCentre`, `BodyYawCompass` and
`CockpitRoom` work in the arena's frame, and `_head.csv` and the analysis read head positions as positions in the arena.

## 6. Which comes closer to the products, and which is better

### 6.1 Feature by feature

| | Avata 360 and A1 | A as built | A adjusted | B |
|---|---|---|---|---|
| Viewpoint | one point on the drone | same | same | same |
| Depth | mono | stereo | mono, through `OVRManager.monoscopic` or `Camera.SetStereoViewMatrix`; neither tried over Link | mono |
| Head | turning only | turning and moving | turning only: pin the eye to the drone, or turn off `OVRManager.usePositionTracking` | turning only |
| How often the world updates | 30 (A1) to 60 (Avata) fps | 72 Hz | step the drone's pose at 30–36 Hz | 30–36 Hz |
| World delay | a tenth of a second or more | none | delay the drone's pose | delay the drone's pose |
| Delay in turning the head | unknown | none | none | none |
| Live sharpness | 1080p or "2K" for the stream; per degree unknown | as sharp as the headset allows | lower the eye texture's resolution | choose the cube size |
| Anti-aliasing | none (video) | 2× MSAA | off | none, like the feeds |
| Seam | visible on the Avata | none | none | none (could be drawn) |
| Drone body | erased | hidden | hidden | hidden |
| Coverage | the full sphere | full | full, or masked | full, or masked |

**B is closer to the products, by construction.** It is a mono 360 video from one point, at a frame rate, sharpness
and delay chosen to suit, which is what both products deliver. A starts as an ideal 360 camera, better than either
product on almost every row. It can be moved most of the way towards them, but each row is a separate approximation,
the mono switch is untried over Link, and A never makes a 360 image to record or reuse. Even adjusted, A's walkers
would move at 72 Hz while its camera stepped at 36, where a video steps both together; that difference is small.

### 6.2 Copy the products' interface, not their radio link

The products' live picture is worse than the simulator's in several ways: less sharp, a tenth of a second late, a
seam, stutter round obstacles. A 360 arm that copied all that, while the swarm and single-drone arms kept clean
simulated feeds, would test the camera's geometry and the radio's quality at once, against the 360 drone. A real
swarm's links are no better: the real-drone scene's feeds are 800 × 450. So:

- **copy the interface**: one viewpoint, mono, turning only, the full sphere, the body erased, the world at video rate;
- **match the image to the other arms** (below);
- **keep the products' link quality as a setting** (a smaller cube and a delay) for a sensitivity check, once their
  live sharpness is known.

Matching the image (arithmetic, not measured, taking the headset's limit as the ~22 pixels per degree the project's
resolution sizing uses). Today's feeds sit on screens at scale 1.3, which show the world at 0.76× its true size, so a
walker spanning θ degrees covers 0.76θ of the headset's view: about 17θ headset pixels, which the feed has more than
enough pixels to fill. A sphere has to show the world at true size, or turning the head would not
match the view, so at 2048 the walker gets 18θ cube pixels (more near the face edges) on 22θ headset pixels. **A 2048
cube therefore puts about as many pixels on a walker as a feed does, and shows it 1.3× larger.** A 4096 cube, six 2530
cameras or the rig would put about 1.3× more pixels on it than a feed.

The larger size cannot be avoided on a sphere, and it matters: for walkers 100–200 m away, the 2026-11-05 measure put
the share barely clearing the clutter round them at 15% at scale 1.3 and 10% at true size (scale 1.71). Equal size as
well would mean every arm's screens at 1.71, which was turned down on 2026-11-05 to keep the `CockpitRoom` anchors in
view. So 2048 is the fairer cube on detail, and the remaining difference in size should be reported with the results.

### 6.3 What does not depend on the choice

- **Coverage beyond the swarm's.** The swarm's all-round view is a band ±23° about the gimbal pitch. With the gimbal
  level, today's scoring (a hat in the central 80% of a shown image, within 40 m, with a clear line) can only count a
  hat seen from about 15 m up or lower, because the central 80% reaches about 19° below the horizon. A full sphere can
  count one straight below from 40 m, so pilots might fly higher and look down. For the decoupling, either approach can
  mask the view to the swarm's band; for the product comparison, show the whole sphere.
- **Sickness.** Both fill the whole field of view with a moving world, and `CockpitRoom` cannot show, since the city or
  the sphere covers it. One Avata 360 reviewer felt unwell after a battery. Nothing in the project records sickness
  yet, and a questionnaire before and after each run (the Simulator Sickness Questionnaire, for example) should come
  before any trial. A, at 72 Hz with smooth motion, is the gentlest version, so it is the gate: if A is sickening, B will
  be worse. A faint room-fixed overlay drawn on top, a grid or a frame, might bring the anchor back; untried.
- **Heading.** The view can turn with the drone when the stick yaws it, as the single drone's screen does now, or stay
  fixed to the world and leave the pilot to turn their head. The reviews don't say which the products do. Turning with
  the drone keeps the stick's forward in front of a seated pilot and matches the other arms. A world-fixed view may be
  gentler, since stick-driven turning is the lead suspect in the sickness assessment, but it leaves the flight direction
  behind the pilot after a half turn (the A1 answers that with point-to-fly and its marker). Either approach can do
  either.
- **Controls.** Keep the RC sticks in every arm. The Avata 360 can be flown on sticks and the A1 can't, but point-to-fly
  would change the controls between arms.
- **Depth.** Stereo matters little for finding a walker 40 m away, where the two eyes' views differ by about two
  headset pixels, and more for judging the gap beside a facade.

### 6.4 Verdict

- **Closer to the products: B.**
- **Better for trials: B.** It uses the feeds' own video path (a camera rendering into a texture at 30–36 Hz,
  mipmapped, no MSAA), its sharpness is set by its cube size, the pilot stays in the arena so calibration, the compass,
  `CockpitRoom`, `_head.csv` and the analysis keep working, and it can record what the camera saw.
- **Better as a first step: A.** Half a day, the cheapest to run, and it answers whether a pilot can fly and search
  with a head-tracked view all round at all, in its most comfortable form.
- **For separating coverage from perspective: the ring arm**, below.

## 7. The ring arm

One drone carrying 8–10 cameras at fixed world headings, each shown on its own screen on the existing `OUTER_CIRCLE`
ring. The screens, their scale, the scoring and the render schedule all match the swarm's, so only the cameras'
positions differ. It renders what the swarm arm renders (ten feeds at 30 Hz), which the patch city already holds at
72 fps. Estimated at half a day, mostly because `ScreenFor`, `HatOnShownFeed` and `FindLoneFeedScreen` assume one screen
per drone.

Its one bias: the fixed ring has no gaps, while a flying swarm averages about 7.7 of its 10 feeds on the hull, at uneven
spacing, so the ring slightly flatters the single drone. B's cube could supply the ring too, but separate cameras match
the swarm's feeds exactly.

## 8. Plan

The times are estimates, and each step ends where the next can be dropped.

**Step 1. A: the rig on the drone (about half a day)**

1. Add `DisplayMode.ONBOARD = 3` to `InterfaceManager`, with the value spelled out, since Unity stores enums as
   integers. `InterfaceManager` only spawns screens in `SCREENS`, so none appear.
2. Add `OnboardRig.cs`, which:
   - gives the three eye cameras the FPV camera's culling mask, its 2000 m far clip (the SDK's 1000 m would cut off
     the landmarks 1.25–1.45 km out), and occlusion culling whenever `ProceduralCityBuilder.OcclusionUsable`;
   - disables `CockpitRoom`, which restores the skybox;
   - disables the drone's FPV camera, which the prefab leaves enabled with no target texture, so it would otherwise
     redraw the city every frame;
   - hides the drone's body.
3. Add a public way to turn off `PyUniSharingFast`'s `driveCameraRigYaw`, which sits in a private settings block. Body
   yaw goes on aiming the velocity frame; it just stops turning the rig.
4. Follow the drone without judder. Physics runs at 50 Hz with no interpolation, so a rig copied from the drone each
   frame would move in uneven steps on a 72 Hz display: 0.8° (about 18 pixels) per step in a 40 °/s turn, and 0.15 m
   at 7.6 m/s.
   - Each physics step, after `VelocityControl` has run, record the body's position and
     `VelocityControl.HeadingSetpointRad`. Use the body, not the camera's mount 0.4 m forward, which would swing the
     eye round on every turn.
   - In `LateUpdate`, interpolate between the last two records by `(Time.time − Time.fixedTime) / Time.fixedDeltaTime`
     and set the rig's position and yaw only: no pitch or roll, as the A1 keeps the horizon level.
   - Interpolating in the follower leaves the flight bit for bit as it is, which `Rigidbody.interpolation` would not.
     The follower also overwrites the height shift `CalibrateToCentre` applies.
5. Recentre so the head faces the heading, at the start and on the calibrate key.
6. Add the stereo macros to `WalkerGroundShadow.shader`, the one city shader without them, or under single-pass
   instanced it draws in one eye only: `UNITY_VERTEX_INPUT_INSTANCE_ID` in the input struct,
   `UNITY_VERTEX_OUTPUT_STEREO` in the output struct, and `UNITY_SETUP_INSTANCE_ID` and
   `UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO` in the vertex function.
7. Optional: bring `BodyYawCompass` back as the A1's direction marker. The city's culling mask leaves out the UI layer
   it sits on, which can't simply be added back because the curved panorama screen lives there, so it needs a layer of
   its own and no 90° display offset.
8. Optional: the product-like settings of section 6.1 (mono, the eye pinned to the drone, the drone's pose stepped to
   36 Hz and delayed).

Check it compiles with `run_bench.ps1 -CompileOnly`.

**Step 2. A headset check (one flight, about 15 minutes)**

Fly A in BlockShapedPatchCityWorld and read `_perf.csv` (frame rate, longest frame). Watch for judder in turns and
beside facades, note any sickness (there is no room anchor), and see whether searching this way works. If it is
sickening at 72 Hz, B at 30–36 Hz will be worse, so stop or fix that first.

**Step 3. The ring arm (about half a day)**

Give the lone drone 8–10 FPV cameras at fixed, evenly spread world headings and the gimbal's pitch, and show each on
its own `OUTER_CIRCLE` screen. Teach `ScreenFor`, `HatOnShownFeed` and `FindLoneFeedScreen` that one drone can have
several screens. The throttled, staggered feed renders then apply to these cameras as they do to the swarm's.

**Step 4. B: the cube camera and sphere (about a day)**

Built on 2026-11-06 as `Sphere360Display` and `Assets/Shaders/Sphere360View.shader`, with `SPHERE = 3` rather than 4,
since A was not built. It differs from the sketch below in four ways:

- the sphere looks up each eye's own view ray, so the view is mono and at infinity;
- the camera position and the heading are interpolated, and the sphere takes back the rig's 50 Hz yaw steps, so the
  rig itself is not interpolated;
- `splitFaces` renders both halves from one position;
- the scoring and analysis of step 5 are in place.

The headless check measured:

- 37 refreshes a second at 36 Hz, each counted as six renders;
- the view through the sphere matching a direct render of the city to 1.2–2.0/255, against 26–65/255 for the views
  a sign or 90° slip would give;
- a hat straight below the drone scoring, and one 94 m away or behind a building not scoring.

Item 4, the frame-rate bench, replayed the recorded single-drone headset flight
`internal_4/ERKS_t1_SingleDrone_20261106_140556` in BlockShapedPatchCityWorld, on the 72 Hz grid with a stand-in
eye pass. Measured from 35 to 320 s:

| Arm | fps | 10 Hz rows at 72 fps | Main thread, mean / p95 / p99 (ms) |
|---|---|---|---|
| The flight as recorded in the headset (feed) | 71.8 | 100% | – |
| Today's single drone, 1920 feed | 71.9 | 100% | 1.3 / 2.3 / 3.2 |
| Sphere, 2048 cube, all faces at once | 71.7 | 98% | 5.3 / 11.1 / 12.3 |
| Sphere, 2048 cube, split over two frames | 72.0 | 100% | 5.4 / 6.9 / 7.9 |
| Sphere, 4096 cube, split over two frames | 72.0 | 100% | 5.9 / 7.7 / 8.8 |

No arm missed a frame in flight. Splitting is therefore the default, and with it 4096 fits too, so the choice between
2048 and 4096 (section 6.2) is free on cost. These runs were headless, with no Link compositor or video encode, so the
headset's `_perf.csv` remains the test.

1. `Camera360.cs`: one hidden camera at the drone body's position, with the FPV camera's settings (`stereoTargetEye`
   None, `allowHDR` off, the far clip, the culling mask, occlusion). It renders a 2048 cube, all six faces every second
   headset frame (36 Hz, because 30 Hz lands unevenly on 72). It fails loudly if the texture wasn't created
   (`IsCreated()`), and adds its renders to `FPVCameraScript`'s count, which needs a public method, so `_perf.csv`
   stays honest.
2. `Sphere360.shader`, stereo-instanced, on an inside-out sphere centred on the head each frame: on the UI layer, no
   depth writes, drawn after `CockpitRoom`, and turned by the display's 90°. The rig's body-yaw turning does the rest,
   as it does for the screens. Options: a mask to the swarm's band (±23° about the gimbal pitch), a cap on sharpness,
   and a delay.
3. `DisplayMode.SPHERE = 4`: move the rig to the arena, as `SpawnScreens` does for the screens, disable the FPV camera,
   and reuse step 1.4's interpolated yaw.
4. On the bench, replay a recorded single-drone flight and compare its frame rate with today's single-drone arm. Settle
   2048 against 4096 (section 6.2); a 4096 cube is split over two frames.
5. Later, if sharpness matters: the Meta compositor's cube-map layer (`OVROverlay`) could show the cube without first
   resampling it into the eye buffers. Untried here.

**Step 5. Recording and analysis (about half a day, before any data is collected)**

- `ExperimentRecorder`: new conditions `Single360 = 2` and `SingleRing = 3` (values spelled out), counted as one drone
  in the condition-mismatch check. The session JSON records the display mode, the cube size, and any mask, cap or
  delay.
- Scoring for B: `HatOnShownFeed` drops the image-frame test and keeps the range and the clear line (and the band, if
  masked), since the ring's scoring doesn't check where the pilot is looking either. `FeedShown` returns true. Log the
  angle between the head and the hat, so whether the pilot was looking can be analysed afterwards.
- Scoring for the ring: one camera per screen, rather than one drone per screen.
- `_head.csv`: under A, the head's logged position follows the drone, so the "head never moves, so a desktop run" check
  must use `headsetActive` instead. Under B, the logged head and gaze directions are in the arena's frame; log the
  sphere's rotation so they can be turned into directions in the city.
- `analyse.py`: add the conditions to `CONDS` and `COLOURS`, and skip the camera-wedge tests in `citysim/replay.py` and
  `citysim/discovery.py` for the sphere.

## 9. Open questions

- How sharp is a product's live view? Neither maker says whether the live stream carries the whole sphere or only the
  view, and DJI's FAQ and download pages were not checked. A flight with a resolution chart would settle it.
- Does a product's view turn with the drone when it yaws?
- Can a pilot tolerate a full-field view for a whole search? Nothing records sickness yet.
- What do A and B cost in the headset? Every figure here is headless.
- 2048 or 4096 for B (section 6.2)?

## 10. Files

The probe, on the bench copy and not in the repo:

- `D:\claude_feedres_bench\probe\FeedResCube.cs`: the timings and the checks.
- `D:\claude_feedres_bench\probe\FeedResCubeCount.cs`: counts the camera callbacks.
- `D:\claude_feedres_bench\probe\FeedResCubeView.shader`: the per-pixel lookup a sphere would use.
- `D:\claude_feedres_bench\probe\run_cube.ps1`: installs the probe on the bench copy and runs it.
- `D:\claude_feedres_bench\out\cube\`: `cube.jsonl`, `equirect_0.jpg` to `equirect_2.jpg` (whole frames, unwrapped),
  and `view2048_*` / `view4096_*` (a direct camera beside the same view sampled from the cube).

The code the plan touches:

- [InterfaceManager.cs](../Assets/Scripts/Interface/InterfaceManager.cs): `DisplayMode`, and when screens are spawned.
- [screenSpawn.cs](../Assets/Scripts/Interface/screenSpawn.cs): `SpawnScreens`, `StepFeedRenders`, `ScreenFor`,
  `FindLoneFeedScreen`.
- [PyUniSharingFast.cs](../Assets/Scripts/ImageStitching/PyUniSharingFast.cs): `driveCameraRigYaw`,
  `UpdateBodyYawFromLoneDrone`, `CalibrateToCentre`.
- [VelocityControl.cs](../Assets/Scripts/VelocityControl/VelocityControl.cs): `HeadingSetpointRad`.
- [FPVCameraScript.cs](../Assets/Scripts/Camera/FPVCameraScript.cs): `EnsureRenderedThisFrame`, `TotalRenders`, and
  the body hiding.
- [CockpitRoom.cs](../Assets/Scripts/Interface/CockpitRoom.cs) and
  [BodyYawCompass.cs](../Assets/Scripts/Interface/BodyYawCompass.cs).
- [WalkerGroundShadow.shader](../Assets/Shaders/WalkerGroundShadow.shader).
- [ExperimentRecorder.cs](../Assets/Scripts/Experiment/ExperimentRecorder.cs): `Condition`, `HatOnShownFeed`,
  `FeedShown`.
- [analyse.py](../Assets/Scripts/Experiment/analyse.py),
  [citysim/replay.py](../Assets/Scripts/Experiment/citysim/replay.py) and
  [citysim/discovery.py](../Assets/Scripts/Experiment/citysim/discovery.py).

## Sources

The products:

- [DJI: Avata 360 specs](https://www.dji.com/avata-360/specs)
- [Oscar Liang: DJI Avata 360, and how it compares to the Antigravity A1](https://oscarliang.com/dji-avata-360/) (17 March 2026)
- [Engadget: DJI Avata 360 review](https://www.engadget.com/cameras/dji-avata-360-drone-review-360-video-hits-new-highs-120014666.html) (26 March 2026)
- [heliguy: DJI Avata 360 review](https://www.heliguy.com/blogs/posts/dji-avata-360-review/) (26 March 2026)
- [DroneXL: DJI Avata 360 review](https://dronexl.co/2026/06/26/dji-avata-360-review-my-honest-take) (26 June 2026)
- [Oscar Liang: Antigravity A1 review](https://oscarliang.com/antigravity-a1/) (11 December 2025)
- [Antigravity A1 press release](https://newsroom.notified.com/news/393663/antigravity-a1---360-drone-with-8k-video-and) (4 December 2025)
- [DroneXL: Antigravity A1 launch](https://dronexl.co/2025/12/04/antigravity-a1-launches-8k-360-drone) (4 December 2025)
- [Maison du Drone: Antigravity A1 review (Spanish)](https://www.maisondudrone.com/es/antigravity-a1-analisis-completo-el-fpv-entra-en-la-era-de-los-360/)

Unity:

- [Camera.RenderToCubemap](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Camera.RenderToCubemap.html)
- [RenderTexture.ConvertToEquirect](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/RenderTexture.ConvertToEquirect.html)
- [ReflectionProbeTimeSlicingMode](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rendering.ReflectionProbeTimeSlicingMode.html)
- [Recorder: 360 View](https://docs.unity3d.com/Packages/com.unity.recorder@4.0/manual/RecorderImage.html) and
  [Recorder: known issues](https://docs.unity3d.com/Packages/com.unity.recorder@4.0/manual/KnownIssues.html)
- [Camera.SetStereoViewMatrix](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Camera.SetStereoViewMatrix.html)
