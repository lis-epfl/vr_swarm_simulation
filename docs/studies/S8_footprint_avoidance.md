# S8 Footprint Avoidance

**Code name:** the shape study · **Dates:** 2026-11-01 → 11-02 · **Status:** done; `obstacleShape` on `SwarmManager`,
Cylinder (as before) in every scene, EnclosingCircle with the slide and d_obs 0.35 recommended, the adaptive spread the
user's call ·
**Builds on:** S7 (its harness, its slide, its confirmation re-flown), S6 (the flights and measures, T3), S4 (the
replica's closest-point model)

**Question.** S7 ended by naming the avoidance shape: every building is avoided as the circle round its axis-aligned
bounding box, which bulges into BlockShapedPatchCityWorld's streets, and the user suspected it was a bottleneck there
and a poor start for the lateral slide. Which shape, close to Olfati-Saber and easy to put into the scene, does better?
The user's conditions: S7's slide on, its straggler routing off, the adaptive spread a secondary factor; and the history
behind the cylinder: the first city avoided the bounding box itself, and a swarm flown square at a wall was slow to
split and surround it for want of a sideways force.

**Design.** One enum, `obstacleShape`: Cylinder (today, bit for bit: 120 of S7's stored flights re-flown identically),
EnclosingCircle (the smallest circle round the footprint) and Footprint (the footprint itself, Olfati-Saber's wall on
each face). The footprint is read from the collider, so any scene works; no margin (it would only rename d_obs). S7's
harness plus a fourth flight, square on at a wall from open ground, at d_ref 2.0 and at a compact 1.2. S7's confirmation
re-flown with every loss located. Two screens in the scene's city; a rule fixed before the confirmation; the
confirmation (ten rows, four cities, ~104 km and 144 square-on approaches each) and one drone; Unity, 100 flights per
row with replica twins. Manual: [docs/swarm_shape/README.md](../swarm_shape/README.md).

**Findings.**
- **The bounds circles take half the street.** They leave a median 13.9 m of the 25 m between two facades (3 streets
  closed, 69 of 794 under 6 m). S7's drones were lost there: half of the 38 lost in its confirmation went down where the
  circles leave under 10 m, a place the drones spent a tenth of their time.
- **The footprint does not work, for the cylinder's own reason.** Without the slide a compact formation flown square at a
  wall split round it in 20.5 s (60% within 40 s) against 3.8 s for the bounds circles with the slide: the first city's
  stall. With the slide it moves, but drones slide along flat faces in single file and collide: 1.55 drones lost per km
  against 0.23 (Unity: 54 against 2 in 100 flights), 21 times the contacts.
- **The enclosing circle keeps the curvature and gives back most of the street** (20.1 m left). With the slide: the
  streets 9–15% faster, a tenth of the losses (0.02 per km), the split round a wall as quick at d_ref 2.0 and about 1 s
  later at 1.2; contacts 0.6–1.3 per km against 0.25 (A0, the scene today, 0.39). In Unity at d_obs 0.35 it lost no drone
  and touched buildings 5 times in 100 flights (bounds circles with the slide: 2 and 14).
- **In the replica it lets the adaptive spread pay.** The rule's pick, C35a (enclosing circle, slide, d_obs 0.35,
  adaptive spread): the streets 20% faster than the bounds circles with the slide (5.87 against 4.89 m/s), cohesive
  speed +12.5%, 0.13 drones lost per km against 0.23, contacts at the scene's present rate (0.39); ahead in all four
  cities. **In Unity it lost 3 pairs in 100 flights** where C35, the same without the adaptive spread, lost none (A1 one):
  28% faster in the streets than A1 against C35's 17%. Too few to settle, but in S7's direction.
- One drone shares the shape: 8% faster with the enclosing circle (Unity 6.12 against 5.67 m/s).

**For the trials.** Set BlockShapedPatchCityWorld's `obstacleShape` to EnclosingCircle, `d_obs` 0.35, and tick the slide
(C35), on the user's go-ahead; it changes both conditions alike. Add the adaptive spread (C35a, the rule's pick) only
if its extra street speed outweighs the collisions Unity hints at. Route stragglers stays optional, as in S7.

**Limits and refinements.** Scripted pilots at full stick; rare losses (a tenth of a drone per km is near the noise of
~100 km); the rule's pick sits exactly at its contact bound; this confirmation's routes were harder than S7's for every
row. Not tried: a capsule or rounded rectangle for long buildings (the enclosing circle still bulges a median 6.9 m),
route stragglers with the new shape. Lessons 43–44.

**Where.**
- Unity: `swarm/ObstacleFootprint.cs` (the footprint cache, enclosing circle), `OlfatiSaber.cs` (`ComputeObstacleFrame`
  with a shape, `MakeFootprintFrame`, `GetObstacleAimCircle`, the slide's local gate), `SwarmPlaneController.cs`
  (`SpreadTarget`), `SwarmManager.cs` (`obstacleShape`), `SwarmAlgorithm.cs`, `ObstacleCylinderGizmos.cs`.
- Replica: `swarm/Bench/swarm_replica.py` (`enclosing_circle`, `aim_circle`, the `footprint` and `enclosing` branches,
  `slide_round`'s gate and side under Footprint, the `fp_side` extra).
- Code: `citysim/swarmshape.py` (gaps, screens, confirmation, rule, the face flight, S7's events, bench plan and gate),
  `swarmshape_replays.py`, `swarmshape_report.py` (+ `.html`).
- Outputs: `experiment_results/swarm_shape/` (`gaps.csv`, `rule.json`, `screen1/`, `screen1c/`, `confirm/` with
  `verdicts.json`, `confirmc/`, `s7_events/`, `bench_*/`, the candidate files). Unity runs:
  `D:\claude_obstacle_bench\runs\swarmshape_*`.
- Page: [S8 Footprint Avoidance](https://claude.ai/artifact/BSzcnGUAjtXe7w23jBBqVc) (`python -B -m citysim.swarmshape_replays`, then
  `python -B -m citysim.swarmshape_report`, then publish `experiment_results/swarm_shape/swarm_shape_bare.html` with the
  files `swarm_shape_files.json` lists).
