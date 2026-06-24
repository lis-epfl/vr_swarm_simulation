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

- [PyUniSharingFast.cs](Assets/Scripts/ImageStitching/PyUniSharingFast.cs) — captures the 3 head-facing boundary feeds → `BlockSharedMemory`, writes metadata/HMD yaw, reads the panorama, renders the curved screen, and drives the quality fallback.
- [screenSpawn.cs](Assets/Scripts/Interface/screenSpawn.cs) — per-drone feed quads with several layouts; `ShowFallbackFeeds(on, style)` is the fallback entry point.

## Python stitching pipeline

- [StitcherThreading.py](Assets/Scripts/ImageStitching/StitcherThreading.py) — orchestrator + 3 threads (block read, fast render, slow warp); `StitcherManager` holds the backends.
- [StabStitcher.py](Assets/Scripts/ImageStitching/StabStitcher.py) — StabStitch++ backend: rolling 7-frame buffer, Spatial/Temporal/Smooth nets, fusion modes, and the panorama-quality estimate.
