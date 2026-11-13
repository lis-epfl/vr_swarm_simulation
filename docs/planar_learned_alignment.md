# PLANAR learned alignment: what changed, and the case for end-to-end learning

2026-11-13. The PLANAR stitcher mosaics the drones' views of one plane, a facade or the ground, with one
homography per view built from each drone's pose. That pose comes from GNSS, the compass and a gimbal, and it is
metres and degrees out. The stitcher therefore left 10–40 px seams on the recorded real-drone clips. This note
summarises what was built to fix that and how it was measured. It then assesses end-to-end learning as the next
step. The engineering detail is in [CLAUDE.md](../CLAUDE.md) under "PLANAR learned alignment".

## Results

Each clip is streamed at frame rate through `tools/planar_stream_bench.py`, and the estimators' answers arrive as
late as they really took to compute. The table gives the plane-supported seam in source pixels, median per clip.
It is measured with ORB, a detector the new method does not use.

| Recorded clip | Pose only | Previous estimators | New |
|---|---|---|---|
| MED facade, 3 drones (1) | 13.4 | 19.6 | **2.1** |
| MED facade, 3 drones (2) | 9.8 | 39.4 | **2.1** |
| MED facade, 5 drones | 41.7 | 14.9 | **4.3** |
| Grass nadir, flown | 37.0 | 5.6 | **1.8** |
| Grass nadir, contracting | 40.7 | 9.0 | **1.8** |
| Nadir, 6 drones, 30 m | 42.8 | 6.4 | **1.8** |
| Nadir, 6 drones, 40 m | 35.3 | 6.1 | **1.5** |

- **Shimmer:** the views moving against each other from frame to frame fell from 0.4–1.9 px to 0.03–0.35 px on
  six of the seven clips.
- **5-drone clip:** it is 3.7 px with 0.9 px of shimmer while the drones face the wall. It degrades in its last
  4 s, when the formation climbs past the roofline and the facade leaves the frame.
- **MED floor:** about 2 px there is the building itself. Its louvre panels stand 0.5–1 m proud, and even a free
  homography per pair cannot do better.
- **Simulated clips with injected error:** with 1 m / 1.5° of pose error the seam goes from 26–31 px with pose
  only, and 7–12 px with the previous estimators, to 2–5 px. These clips are new; they were recorded headless in
  ScenicPatchCityWorld.
- **Cost:** one update takes 0.2 s for 3 drones and 0.4–0.5 s for 6, on the warp thread. The render path grows
  from 5 to 11 ms a frame for 3 views and from 11 to 17 ms for 6 (RTX 4090).
- **Not yet tested:** none of this has run live in Unity or on aircraft.

## Contributions

1. **Learned matching and a planar bundle adjustment** (`planar_align.py`).
   - Each view is rectified onto the prior plane first, which removes viewpoint, scale and rotation. SuperPoint and
     LightGlue then match the overlapping pairs.
   - One solve recovers a rotation and a position correction per drone **and the plane's orientation**. The plane
     tilt (5–6° on MED) was the largest real error, and the old estimators could only change the plane's distance.
   - The published poses enter as Gaussian priors.
   - The robust kernel is split along and across each match's epipolar direction. Off-plane content then cannot
     drag the poses.
2. **A view graph for repeating facades.** The drones are nodes, GPS is the prior on each node, and every
   overlapping pair is an edge scored by the views' dense learned descriptors. One shift per drone is chosen so
   that all the pairs agree, and the loops resolve what single pairs cannot.
   - Rare descriptors outweigh repeated ones.
   - A shift pattern proportional to the drones' positions is recognised as "a different surface", not as pose
     error.
   - The published geometry wins unless the images clearly overrule it. On a wall of identical windows the images
     cannot.
3. **Alignment carried by the images, not the telemetry** (`planar_track.py`).
   - The problem: one drone's video ran about 1 s behind the others on a weak link. Each frame was drawn at its
     telemetry pose, so that view slid 1.5 m off whenever the formation moved.
   - The fix: each solve is pinned to the frames it used, then carried forward by each view's own frame-to-frame
     image motion. This motion is restricted to points moving as the plane does.
   - This is MViS's structure: inter-view transforms tie the views together and temporal transforms tie each view to
     its past. Here the temporal half is measured instead of fitted.
4. **Telemetry resampling** (`planar_pose_filter.py`). The pose is sampled at about 5 Hz and held in between; it is
   interpolated to each frame's time.
5. **`Multiband` blend.** Each drone's exposure is matched to its neighbours', and the low frequencies are
   cross-faded. Detail still comes from one view, so nothing ghosts. This is DJIScene's setting.
6. **Measurement tools.**
   - `tools/planar_stream_bench.py` plays real or simulated clips in real time, with `--pose-noise` and `--video`.
   - `PlanarCaptureProbe.cs` records simulated clips with exact poses, headless.
   - Self-test section 13 covers the solver, the graph, the blend, a synthetic repeating facade and a lagging video.

**Switching it on:** the existing inspector switches `planarPoseRefine` and `planarPlaneSweep` now drive all of this.
`PLANAR_LEGACY_ESTIMATORS=1` restores the photometric estimators.

**Known limits:**
- With exact simulated poses the aligner can only add noise (1.4–3.7 px), so leave the switches off in the sim.
- Content off the plane still breaks at seams.
- A wall of identical windows with pose error beyond half a window spacing cannot be resolved from the images. The
  aligner then keeps the published geometry.

## End-to-end learning: feasibility, pros and cons

**Formulation.** The unknowns are small:
- a correction `(ω_i, t_i)` per drone;
- the plane `(n, d)`;
- optionally a residual height field `h(a, b)` on the plane, for parallax.

The training objective, with no labels, is

    L = Σ_(i,j) Σ_overlap ρ( I_j(H_j⁻¹ H_i x) − I_i(x) )          photometric / feature agreement
      + Σ_i ‖(ω_i, t_i)‖²_Σ_GNSS                                   noisy GPS and compass as a prior
      + λ_t Σ_i ‖θ_i(t) − θ_i(t−1) − Δ_i(t)‖²                      temporal consistency with measured image motion
      + λ_h ‖∇h‖²                                                    a smooth surface

Here `H_i = K R_i [e1 | e2 | O − C_i]` is built from the predicted parameters. Every output is therefore a valid
homography, and all of them are consistent by construction. MViS's per-view free homographies are neither, and
StabStitch++'s meshes are neither.

A network `f(images, GPS) → (ω, t, n, d, h)` trained on L generalises what this pipeline now does with hand-set rules.

**The most practical form** keeps the current solver and learns only its parts:
- the per-match and per-pair weights;
- the "which alignment" decision on repeating facades;
- a few unrolled Gauss–Newton steps with learned features.

The geometry stays exact and inspectable, and only the judgement calls are learned.

**Pros**
- One forward pass at a fixed cost. It could run every frame and drop the anchor bookkeeping.
- It can learn what makes a facade unambiguous (windows, edges, the roofline) instead of relying on tuned
  thresholds.
- It could output a height field or mesh, which is the one thing the plane model cannot express. That would remove
  the parallax seams on roofs and background.
- Graph neural networks fit naturally: drones are nodes, GPS baselines are edge features, and overlap is the
  adjacency. Inference already scales to N views, and LightGlue's attention is itself a graph network.

**Cons**
- **Data is the binding constraint.** There are seven usable real clips from two sites. Any end-to-end model would
  memorise them.
- It fails silently on an unfamiliar site, and when it is wrong there is nothing to inspect.
- Sim-to-real gap: the simulated facades (rendered rooms behind each window) are harder than the real one in some
  ways and easier in others.
- It competes with the VR renders for the same GPU. The current per-frame budget is already 11–17 ms.

**Distilling StabStitch++ or MViS instead**
- *MViS:* it is per-video test-time optimisation (Adam over a small network, 0.5–0.8 s per frame per view). The new
  pipeline is in effect its real-time counterpart, so nothing geometric is left to distil. Its slow RoMa matcher
  could serve as an offline teacher for a fast matcher.
- *StabStitch++:* its pairwise mesh network is the missing parallax model. To extend it to N views, apply it to the
  view graph's edges and solve for per-view meshes that agree across the loops, the same way the shift graph is
  solved. Then distil that into a single network. That is weeks of work plus training data.

**Recommended path**
1. **Fly and record more:** more sites and formations, logging the video latency. Every new clip yields
   pseudo-labels for free from the current solver, and `PlanarCaptureProbe` gives unlimited exact labels in
   simulation.
2. **Replace the hand-set selection rules with small learned models**, trained on the current solver's outputs.
3. **Add a height field or mesh head** for off-plane content.
4. **Attempt full end-to-end only once a multi-site dataset exists.**
