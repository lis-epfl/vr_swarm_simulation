# S10 Ellipse Avoidance

**Code name:** the ellipse study · **Dates:** 2026-11-02, adopted 2026-11-03 · **Status:** done; the ellipse
(E15_35a: the scene's law with `obstacleShape` Ellipse, cap 1.5) is in BlockShapedPatchCityWorld · **Builds on:** S8 (the shape
setting, its harness, the head-on flight), S7 (the slide and the rules that aim by a building), S9 (the two spread
settings, the loss statistics)

**Question.** S8 made BlockShapedPatchCityWorld avoid each building as the smallest circle round its footprint. A
circle fits a long building badly: it lies very close to some of the building's walls and far from others, so the
distance at which a drone is pushed off a wall depends on which wall it is. Does an ellipse fitted to each footprint
(`SwarmManager.obstacleShape` Ellipse, its aspect capped at `ellipseMaxAspect`) keep the swarm at a more even distance
from every wall, how much street does it give back, and what does it cost? If it is safe, adopting it should change
only the scene's values.

**Design.**
- **The shape.** The minimum-area ellipse round the footprint, its aspect capped: Khachiyan's algorithm gives the
  axis and aspect, and the ellipse is the smallest circle round the hull squashed along that axis. So it always
  encloses the footprint, and a cap of 1 is the enclosing circle exactly.
- **The frame.** The beta-agent at the closest point (mu = rc / (rc + d), R / rho on a circle; inside, the gradient at
  the drone).
- **The rules.** S7's rules aim by the ellipse in its unit space, except the slide's head-on gate, which stays local:
  measured in unit space it starves the slide on a flat side, lesson 43's stall.
- **Regression.** Shapes 0–2 are untouched (224 stored S8 flights re-flown bit for bit). Unity's fit matches the
  replica's to 0.01 cm on all 289 of the scene's buildings.
- **Geometry** (exact, 4 cities): each wall's distance to the shape; street gaps and the corridor left after `d_obs`.
- **The question in flight:** one drone flown along interior streets 6, 9 and 12.5 m off a wall, and at edge faces
  square on and at 60° and 30°.
- **Formation flights:** S8's flights and measures at d_ref 2.0, and the square-on flight at 1.2.
- **Screen:** 26 rows on the scene's city, 6,820 flights: the circle and the ellipse at caps 1.25, 1.5 and 2, ×
  `d_obs` 0.25 / 0.35 / 0.45, × the spread held (S9's F20) or adaptive (the scene).
- **Confirmation:** 7 rows on four cities, 14,904 flights, about 390 km of formation flight a row.
- **Rule** (`rule.json`, written before the confirmation): safe against the row's partner and the scene; delivers
  when the one-drone standoff spread is at most 0.8× its partner's with nothing worse than the scene. The pick is by
  measured fit, then the roundest cap.
- **Unity:** 1,024 bench flights in one build, with the scene and F20 re-flown as controls; twins gated.

**Findings.**

| | the scene (C35a) | the ellipse pick (E15_35a) |
|---|---|---|
| building's farthest wall − nearest, median (geometry) | 5.7 m | 3.3 m |
| buildings with a wall > 6 m from the shape | 173 of 289 | 47 |
| one drone's standoff spread across walls (replica / Unity) | 2.85 / 3.18 m | 1.58 / 1.78 m |
| corridor left after d_obs, median (streets under 6 m), four cities | 12.8 m (376) | 14.3 m (57) |
| street cohesive speed (replica / Unity) | 5.78 / 5.83 m/s | 6.21 / 6.13 m/s |
| loss events per km (95%) | 0.133 (0.100–0.175) | 0.097 (0.069–0.134) |
| contacts per km (replica; Unity count) | 0.55 (12) | 0.40 (17) |
| square-on split, d_ref 2.0 · 1.2 (Unity at 1.2) | 2.41 · 4.12 s (6.0) | 2.37 · 3.89 s (4.0) |

- **It answers the question.** The ellipse removes about 40% of the wall-to-wall spread, and a drone flown along the
  walls keeps a distance that varies about half as much (×0.55 replica, ×0.56 Unity). Its push off a wall follows the
  shape's distance from that wall (correlation 0.96–0.99).
- **It gives back the narrow streets at the scene's own standoff.** The median gap rises only 1.2 m, but streets under
  10 m go from 44 to 0. It is faster in the streets in every one of the four cities.
- **No cost found.** Fewer loss events and contacts in the replica; no flat-side stall (faster to split at d_ref 1.2).
  It needed no thicker standoff, so the corridor trade-off the user reserved never arose. With the spread held it
  gains as much (+8% in the streets), so the pick stands whichever way S9's decision goes.
- **Unity agrees except on contacts with the adaptive spread** (17 against 12, ×1.42, p ≈ 0.23; with the spread
  held, 22 against 27).

**Limits and refinements.**
- **Pilots and model:** scripted pilots at full stick, and the replica is 2-D.
- **Unity's sample:** 37 km per set is too short for rare losses or contacts.
- **The narrowing cost:** the ellipse narrows 22–23% of streets past the short ends of long buildings, about 3% by more
  than 2 m (the worst to 8.4 m on seed 4). No measure showed it costing anything.
- **Scope:** BlockShapedPatchCityWorld only.
- **Lessons 49–51**, and notes under 24 and 43, cover:
  - the one-drone probe built for the question;
  - the route filter that included its own target;
  - picking by the measured effect rather than the cap;
  - a censored split counting a formation that stayed whole;
  - a rule carried into unit space bringing back the stall.

**Adopted** (2026-11-03, the user's word). `swarmtune apply` changed `obstacleShape` 1 → 3 and nothing else
(`ellipseMaxAspect: 1.5` was already serialised). Nine stored E15_35a flights re-flew identical from the scene with an
empty override, which now means the ellipse; `swarmellipse.PRE_S10` gives back the circle (C35a).

**Open decisions.**
- Watch Unity's contacts in the first trial sessions.

**Where.**
- Manual: [docs/swarm_ellipse/](../swarm_ellipse/README.md).
- Code: `citysim/swarmellipse.py`, `swarmellipse_bench.py`, `swarmellipse_replays.py`, `swarmellipse_report.py` (+
  `.html`), `tests/test_swarmellipse.py`.
- The replica's ellipse functions, and in Unity `ObstacleFootprint.cs`, `OlfatiSaber.cs`, `SwarmPlaneController.cs`,
  `SwarmAlgorithm.cs`, `SwarmManager.cs`, `ObstacleCylinderGizmos.cs`, `Bench/Editor/EllipseFitProbe.cs` (commit
  `b1f826f9`).
- Outputs: `experiment_results/swarm_ellipse/`.
- Page: https://claude.ai/artifact/QZHj23gYuV4xhfS7uwb6At. Its Cities tab has the four cities drawn with circles or
  ellipses, and seven replayed routes flown by Unity with the replica's twins: one drone along a wall, the street the
  circles pinch, the street the ellipse narrows, square on and at 60° to a long building, a typical straight line, and
  the ellipse's worst loss.
