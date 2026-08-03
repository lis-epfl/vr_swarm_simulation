# Agent File Index

Architecture, the shared-memory contract, and the cross-cutting invariants live in
[CLAUDE.md](CLAUDE.md) (auto-loaded). This file is just a map of *where things live* — read the
actual file for behavior, since these pointers are intentionally not kept in lock-step with the code.

## Swarm flight control (per-drone, Unity)

Chain: **SwarmManager (config) → SwarmAlgorithm (dispatch) → flocking force → VelocityControl**;
yaw is separate via AttitudeAlgorithm.

- [SwarmManager.cs](Assets/Scripts/swarm/SwarmManager.cs) — singleton holding all swarm/attitude params; fires `swarmParamsChanged` on edit; selects active algorithms.
- [SwarmAlgorithm.cs](Assets/Scripts/swarm/SwarmAlgorithm.cs) — per-drone dispatcher; writes a world-frame accel to `VelocityControl.swarmAcceleration` each `FixedUpdate`.
- [OlfatiSaber.cs](Assets/Scripts/swarm/OlfatiSaber.cs) — main flocking law (consensus + cohesion + obstacle avoidance). `Reynolds` is the simpler alternative.
- [AttitudeAlgorithm.cs](Assets/Scripts/swarm/AttitudeAlgorithm.cs) — per-drone yaw/heading; sets `BoundaryEstimate` (convex-hull edge flag).
- [VelocityControl.cs](Assets/Scripts/VelocityControl/VelocityControl.cs) — low-level controller (tilt/yaw/height loops, circular clamps); reads limits from a `FlightProfile`.
- [SwarmSpawn.cs](Assets/Scripts/swarm/SwarmSpawn.cs) — spawns the drone grid, wires the `swarm` list into each drone, plus reset/reposition/health helpers.

## VR display & camera capture (Unity ↔ Python bridge)

- [PyUniSharingFast.cs](Assets/Scripts/ImageStitching/PyUniSharingFast.cs) — captures the selected feeds → `BlockSharedMemory` (3 head-facing boundary drones normally; up to `maxStitchViews` plane-facing ones in `PLANAR`, with camera pose in the block header), writes metadata/HMD yaw, resolves and publishes the scene plane, reads the panorama, renders the curved screen, and drives the quality fallback.
- [StitchPoseSource.cs](Assets/Scripts/ImageStitching/StitchPoseSource.cs) — injects drifting GNSS-magnitude error (Ornstein–Uhlenbeck + common-mode + yaw bias) into the published camera poses. Deliberately separate from `StateFinder`, whose noise feeds the flight controller.
- [ImageSharing.cs](Assets/Scripts/dji/ImageSharing.cs) — real-drone mode (DJIScene): reads all live feeds from `DroneFeedSharedMemory` (written by the DJI_Swarm repo's `image_stream_feed.py`), shows them on the feed screens, and re-publishes the 3 body-yaw-selected views into `BlockSharedMemory` for the stitcher.
- [screenSpawn.cs](Assets/Scripts/Interface/screenSpawn.cs) — per-drone feed quads with several layouts; `ShowFallbackFeeds(on, style)` is the fallback entry point.

## Python stitching pipeline

- [StitcherThreading.py](Assets/Scripts/ImageStitching/StitcherThreading.py) — orchestrator + 3 threads (block read, fast render, slow warp); `StitcherManager` holds the backends.
- [StabStitcher.py](Assets/Scripts/ImageStitching/StabStitcher.py) — StabStitch++ backend: rolling 7-frame buffer, Spatial/Temporal/Smooth nets, fusion modes, and the panorama-quality estimate.
- [PlanarStitcher.py](Assets/Scripts/ImageStitching/PlanarStitcher.py) — pose-driven planar backend for the facade/nadir configurations: builds one exact homography per view from pose, warps and feather-blends N views, and owns the projective/coverage/plane gates. `compute_warps` is the (empty) refiner slot.
- [planar_geometry.py](Assets/Scripts/ImageStitching/planar_geometry.py) — pure-numpy geometry for the above: Unity→CV frame conversion, plane frame, `G`, footprints, Jacobian anisotropy. No Unity, torch or shared-memory dependency, so it is testable in isolation.
- [tools/planar_selftest.py](Assets/Scripts/ImageStitching/tools/planar_selftest.py) — offline geometry + end-to-end render checks; cross-checks projection against an independent Unity-derived reference.
- [tools/check_wire_layout.py](Assets/Scripts/ImageStitching/tools/check_wire_layout.py) — parses `PyUniSharingFast.cs` and `StitcherThreading.py` and asserts their shared-memory layout constants agree.

## Environment / scenery

- [GoalPatchReplacer.cs](Assets/Scripts/Environment/GoalPatchReplacer.cs) — at play-mode Start, randomly swaps `n` of the `MC_Patch_*` tiles under `City_Pack_01` for a `goal_patch` prefab (X/Z from the tile, Y pinned to 0; non-adjacent placement).
