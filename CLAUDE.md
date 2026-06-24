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

Three Windows named memory maps. **If you change a layout/offset, change both
`Assets/Scripts/ImageStitching/PyUniSharingFast.cs` and
`Assets/Scripts/ImageStitching/StitcherThreading.py`** (`readMetadataMemory`/`WriteMetadata`).

- `MetadataSharedMemory` — sizes + stitcher config + blur/border + quality settings + **live HMD yaw**
  (yaw also rewritten every frame at a fixed offset).
- `BlockSharedMemory` — one block per selected drone: `int flag | int droneId | float heading | RGB24 image`.
  `flag` is the handshake (0 = ready, 1 = busy). Images are **640×360 BGR, top-down**.
- `PanoramaSharedMemory` — `int flag | int quality_ok | RGB24 panorama`. `quality_ok == 0` ⇒ Unity shows
  individual feeds instead of the panorama. Panorama is **vertically flipped** by Python (Unity textures start bottom-left).

## Conventions & invariants (not enforced by code)

- **Coupled 3-view selection:** the head-relative left/centre/right pick exists in *both* C#
  (`SelectStitchCameras`) and Python (`get_drone_order` + `get_subsets_from_order`) and must agree.
- **Boundary drones** = `AttitudeAlgorithm.BoundaryEstimate` (convex-hull). Stitching and the
  `OUTER_CIRCLE` screen layout only use boundary drones.
- Image format across the bridge is **BGR + top-down**; the panorama is flipped once on the Python side.
- `StitcherThreading.py` currently **hardcodes 640×360 input / 1920×1080 output**, overriding metadata
  sizes (search `TODO: Remove hardcoding`). Keep `blockImageWidth/Height` at 640/360 in the C# inspector.
- **STABSTITCH render/warp are decoupled:** a ~15 fps render loop uses cached warp params only (no neural
  net); a separate ~3 Hz thread runs the nets and updates the cache.

## Drone prefab hierarchy (relied on by many scripts)

`Drone N` → child `DroneParent` (has `SwarmAlgorithm`, `AttitudeAlgorithm`, `VelocityControl`, `Rigidbody`)
and child `FPV` (the `Camera`). Drones are tagged `DroneBase`; the arena is tagged `Arena`. Many scripts
use `transform.Find("DroneParent")` / `Find("FPV")`.

## Running the Python stitcher

`cd Assets/Scripts/ImageStitching && python StitcherThreading.py`. **Use the `stitching` miniconda env** —
the default `python` has no torch. StabStitch++ models load from
`StabStitch2_main/Full_model_inference/full_model_ssd/*.pth`.
