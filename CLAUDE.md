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
   → AttitudeAlgorithm                                     |
   → VelocityControl                                       |
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
- `BlockSharedMemory` — the **stitcher input**: 3 slots ordered left/centre/right, one block per selected
  drone: `int flag | int droneId | float heading | RGB24 image`. `flag` is the handshake (0 = ready,
  1 = busy). Images are **640×360 BGR, top-down**. Sole consumer: `StitcherThreading.py`. Sole producer:
  sim = `PyUniSharingFast`; real-drone mode (DJIScene) = `ImageSharing.cs` (so keep `enableImageWriting`
  off on `PyUniSharingFast` there).
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

- **Coupled 3-view selection:** the body-yaw-relative left/centre/right pick exists in C#
  (`PyUniSharingFast.SelectStitchCameras` for sim, `ImageSharing.PublishStitchBlocks` for real drones)
  and in Python (`get_drone_order` + `get_subsets_from_order`) and all three must agree.
- **Boundary drones** = `AttitudeAlgorithm.BoundaryEstimate` (convex-hull). Stitching and the
  `OUTER_CIRCLE` screen layout only use boundary drones.
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

## Drone prefab hierarchy (relied on by many scripts)

`Drone N` → child `DroneParent` (has `SwarmAlgorithm`, `AttitudeAlgorithm`, `VelocityControl`, `Rigidbody`)
and child `FPV` (the `Camera`). Drones are tagged `DroneBase`; the arena is tagged `Arena`. Many scripts
use `transform.Find("DroneParent")` / `Find("FPV")`.

## Running the Python stitcher

`cd Assets/Scripts/ImageStitching && python StitcherThreading.py`. **Use the `stitching` miniconda env** —
the default `python` has no torch. StabStitch++ models load from
`StabStitch2_main/Full_model_inference/full_model_ssd/*.pth`.
