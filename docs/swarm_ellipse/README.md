# The ellipse study (S10 Ellipse Avoidance)

The registry entry is [docs/studies/S10_ellipse_avoidance.md](../studies/S10_ellipse_avoidance.md). This manual has
the method, every measure and decision, and how to run it. Code: `citysim/swarmellipse.py` (on S8's
`swarmshape.py`, S7's `swarmflow.py`, S6's `swarmtune.py`), the replica's ellipse functions, and in Unity
`ObstacleFootprint.cs`, `OlfatiSaber.cs`, `SwarmPlaneController.cs`, `SwarmAlgorithm.cs`, `SwarmManager.cs` and
`ObstacleCylinderGizmos.cs`. Outputs: `experiment_results/swarm_ellipse/`.

## 1. The question

S8 made BlockShapedPatchCityWorld avoid each building as the smallest circle round its footprint. The user's
concern with it: the circle lies very close to some of a building's walls and far from others, so the distance at
which a drone is pushed off a wall depends on which wall it is. The aim is a shape that fits each building more
closely, so that every wall is avoided at a similar distance, and that, by fitting closer, gives back the narrow
streets. It should then be one setting away from the scene.

## 2. The setting: `SwarmManager.obstacleShape` Ellipse and `ellipseMaxAspect`

`ObstacleShape.Ellipse = 3`, the fourth member (swarm_params maps the members by position), and a cap on its aspect,
`ellipseMaxAspect` (1–2). Shapes 0–2 are untouched: every ellipse rule is its own branch, and 224 of S8's stored
confirmation flights re-flown with the ellipse code in the replica came out identical.

**The fit** (`ObstacleFootprint.MveeAxis` / `EllipseFit`, `swarm_replica.mvee_axis` / `ellipse_fit`). The
minimum-area ellipse round the footprint (Khachiyan's algorithm with Todd and Yildirim's away steps, in doubles, the
3×3 inverse by cofactors) supplies only the axis and the aspect. The ellipse itself is the smallest circle round the
hull squashed along that axis by k = min(cap, aspect), stretched back: it contains the footprint exactly whatever the
iteration's tolerance, it is the minimum-area ellipse while the footprint's own aspect is within the cap, and at a cap
of 1 (or an aspect under 1.02) it is the enclosing circle itself, flown on the circle's own code path. The axis and
aspect are kept per collider and the ellipse refitted when the cap changes, so the bench's parameter sets, flown one
after another in one process, each fly their own cap. Unity's fits agree with the replica's to 0.01 cm on all 289 of
the scene's buildings at every cap (`swarmellipse fitcheck`, from `Bench/Editor/EllipseFitProbe.cs`).

**The frame** (`OlfatiSaber.MakeEllipseFrame`, `swarm_replica.ell_frame`). Outside the ellipse, the beta-agent at the
closest point (a fixed five-step trig-free solve, within 2e-6 m of dense sampling up to aspect 3.6), with the normal
there and mu = rc / (rc + d), rc the radius of curvature there: R / rho on a circle. Inside it, the gradient at the
drone itself, mu 1 and distance 0: continuous with the outside at the surface, the cylinder frame's inside on a
circle, and with no closest point to flip between the two sides of the major axis, where a saturated force would
otherwise chatter.

**S7's rules aim by the ellipse**, in its unit space, where it is the unit circle and a segment is still a segment:
the split aim, the hidden-centroid test, `FirstBlockingFrame` (the route-stragglers test: an affine map keeps a line's
parameter, so entries stay comparable with the circles') and the adaptive spread's aim test; its "among buildings"
test uses the true distance. Two exceptions, both found while designing it:
- **The slide's head-on gate stays local**: 1 − F |t·u| / slideHeadOnFraction while the stick points into the
  ellipse, F the drone's unit-space radius. On a circle F |t·u| is exactly S7's miss / R, so it is S7's gate there; at
  a flat side it is S8's local gate. Measured in unit space the line gate starves the slide where the stick meets a
  flat side square on (0.52 at cap 1.5, 0.25 at cap 2), which is lesson 43's stall.
- **The side rule's last branch is S8's lean rule** (slide the way the stick leans along the surface), which on a
  circle equals the own-side rule wherever the gate is open, and along a long side does not point back down the
  street.

An ellipse flown through this code with a = b equals the circle to 1e-15 in every term (`test_swarmellipse`).

## 3. Geometry (deterministic, `swarmellipse gaps`)

`walls.csv` and `gaps.csv`, the four cities. On the scene's city (seeds 2–4 agree):

| shape | wall's distance to the shape, median / p90 | building's farthest wall − nearest, median / p90 | buildings with a wall > 6 m from the shape | street gap median / p10 / p1 | streets under 10 m |
|---|---|---|---|---|---|
| bounds circle | 5.8 / 9.9 m | 5.0 / 8.5 m | 271 of 289 | 13.9 / 6.6 / 0.8 m | 161 |
| enclosing circle | 2.7 / 7.4 m | 5.7 / 9.8 m | 173 | 20.1 / 12.7 / 4.8 m | 44 |
| ellipse, cap 1.25 | 2.3 / 5.4 m | 3.7 / 6.3 m | 76 | 21.1 / 15.6 / 10.6 m | 6 |
| ellipse, cap 1.5 | 2.2 / 4.9 m | 3.3 / 5.3 m | 47 | 21.3 / 16.1 / 12.7 m | 0 |
| ellipse, cap 2 | 2.2 / 4.7 m | 3.2 / 5.2 m | 40 | 21.3 / 16.5 / 13.1 m | 0 |

- A wall's distance is measured at its middle along its outward normal (walls of 4 m or more).
- An enclosing shape must touch the footprint's corners, so the spread cannot reach zero. The median street is near
  its ceiling for the same reason (+1.2 m), and the narrow streets are where the ellipse gains.
- **It narrows some streets**, past the short ends of long buildings: at cap 1.5, 22–23% of streets are narrower than
  under the circle, about 3% by more than 2 m. The worst loses 4.7 m on the scene's city and 7.3 m on seed 4 (leaving
  8.4 m), while about 57% widen by more than 1 m.
- **At an unchanged d_obs the drones fly nearer the walls** (the shell thins by ~0.5 m at the median and ~2.5 m at
  p90). The corridor left once d_obs is taken off both sides (scaleFactor 10), median / p10 / streets under 6 m, scene's
  city:

  | | d_obs 0.25 | 0.35 | 0.45 |
  |---|---|---|---|
  | enclosing circle | 15.1 / 7.7 / 55 | 13.1 / 5.7 / 87 (the scene) | 11.1 / 3.7 / 144 |
  | ellipse, cap 1.5 | 16.3 / 11.1 / 0 | 14.3 / 9.1 / 12 | 12.3 / 7.1 / 38 |

## 4. How it was flown

S8's harness: `ContactSim` on the prism city (`PolyCity`), settled starts, the bench's waypoint pilot at full stick,
d_ref 2.0, the scene's city (seed 1) for the screen and seeds 1–4 for the confirmation. Every candidate names every
law field (`obstacleShape`, `ellipseMaxAspect`, `d_obs`, the slide, route stragglers, the adaptive spread, its factor
and filter): since S8 an empty override is the scene, C35a.

Flights (route indices: screen from 15000, confirmation from 16000, the probes from 17000 / 17500, the bench from
18000; S6–S9 used 0–14299):
- S8's four: `cross`, `pass`, `street`, `face` (square on at an edge face; also flown at d_ref 1.2, where the ring
  cannot simply enclose the building).
- `angle`: S8's head-on flight at 0°, 30° and 60° from the face's normal (90°, 60° and 30° to the face), for the
  formation and for one drone. The run-in must clear every other building by 2 m on lines ±18 m across the corridor
  -- not the target's own: at an angle the lines beside the aim point run into its face, which is the approach being
  flown (the first version tested the target too and found no 60° run-in at all; its rows were dropped and re-flown).
  For one drone, `onset`: the facade distance where its speed towards the building first falls below 0.8 of its
  run-in speed, and `closest`.
- `along`: one drone flown down an interior street parallel to one of its walls, 6, 9 or 12.5 m off it, from 20 m (or
  the longest run that keeps 2 m off every footprint) before the wall to as far past it. While alongside: `standoff`
  (its median distance from the wall), `closest`, `push` (how far it was pushed off its line), its speed, and the
  shape's own distance from that wall (`shell`). **This is the in-flight form of the question**: a drone 6 m off a
  wall the circle stands 7 m off is pushed out to ~10.5 m; one 6 m off a wall the circle stands 1 m off is not. Free
  formation flights cannot show it, because most closest approaches to a facade are at its corners, where every
  enclosing shape touches.

Rows (`screen_cands.json`): the enclosing circle and the ellipse at caps 1.25, 1.5 and 2, × d_obs 0.25 / 0.35 / 0.45,
× the spread held at 2.0 (S9's F20) or adaptive at 0.85 (the scene, A85): 24 rows, each ellipse paired with the circle
at its own d_obs and spread mode; and two arms (E15_35c, E15_35ac) that fly the ellipse but aim the S7 rules by the
enclosing circle, as S8's shapes do. d_ref 1.2 square-on rows in `screen12_cands.json`; the one-drone probes in
`probe_cands.json` (the law alone: no spread to vary).

## 5. The head-on and angled flights (stage 4, screened first)

Lesson 43 says a tighter shape fails at flat faces, so these were flown before anything else, 36 square-on and 36
angled flights per row, the one-drone angled flights for each probe row and the d_ref 1.2 square-on rows (2,556
flights).

- **No flat-side stall.** Square on at d_ref 2.0 every row split every time, in 2.5–2.9 s for the circles and the
  ellipses alike. At d_ref 1.2 the ellipse splits faster: 3.9 s (cap 1.5, all 36) against the circle's 4.9 s (35 of
  36); adaptive 3.9 against the scene's 5.8 (34 of 36).
- At 30° the ellipse splits slightly sooner (3.0–3.2 s against 3.2–3.4).
- **At 60° one route never splits under any ellipse** (route 15002, building 274, the 0.6 offset): the formation goes
  round one side whole, one group throughout and past the building in 7.5–8.0 s, against the circle's 8.0 s with the
  formation split (one group 78–96% of the flight). The censored split time (40 s) counts that as a failure; it is the
  formation keeping together on a glancing pass. So **the rule judges the split on square-on flights, as S8's did,
  and angled flights on time past the building and one group.**
- **One drone, the user's question at a face**: the spread across faces (p90 − p10) of the facade distance where
  avoidance takes hold, square on / 30° / 60°: circle 6.0 / 5.9 / 5.0 m, ellipse cap 1.5 1.5 / 4.4 / 2.9 m, cap 2
  1.7 / 4.2 / 2.3 m. Its median moves little (11.9 → 11.3 m square on), and the closest approach not at all (~4.2 m).

## 6. The screen

`screen1/` (the scene's city, 6,820 flights in all, no errors): every row on S8's screen mix (48 straight lines, 32
blocks, 24 street routes, 36 square-on, 36 angled; about 30 km a row), the d_ref 1.2 square-on rows, and the
one-drone probes (120 along, 36 angled, per law). It passed the stop gate by a distance:

- **One drone, along a wall** (the question): the standoff's spread across walls, averaged over the three offsets,
  3.06–3.16 m under the circle at every d_obs, 1.61–1.68 m at cap 1.25, 1.34–1.37 m at 1.5 and 1.38–1.42 m at 2: ×0.44
  at cap 1.5. How far the drone is pushed off a wall correlates 0.96–0.99 with the shape's distance from that wall.
- **The streets**: street cohesive speed at d_obs 0.35, 5.50 (circle) against 5.78 (cap 1.5) held, 5.76 against 6.25
  adaptive, one group 95% against 99%.
- **Costs**: surrounding a building 3.7 points less often adaptive (0.885 → 0.848); contacts 1.1–1.3× the circle's
  at d_obs 0.35, near 2× at d_obs 0.25 with cap 2; loss events 0–5 pairs a row, too few at 30 km to compare (lesson 46).

## 7. The rule

`rule.json`, written 2026-11-02 17:44:48, after the screen and before the confirmation flew (`swarmellipse rule`):

- **Base**: the scene, C35a. **Partner**: the circle at the row's own d_obs and spread mode.
- **Safe**, against both: loss events (drones lost less the partner of each collision) not more frequent at one-sided
  exact p ≥ 0.10 (`swarmspread.worse_p`, per km); contacts at most 1.25×; ending in pieces at most 2 points more;
  square-on split at most 1.1× at d_ref 2.0 and 1.2; hull share at least 0.5.
- **Delivers**: one drone's standoff spread at most 0.8× its partner's; street cohesive speed at least 98% of the
  scene's; surrounding within 5 points of the scene's.
- **Pick**: among rows safe and delivering, those within 5% of the evenest standoff, then the roundest (smallest cap),
  then the best score. The plan had "the largest cap"; it was changed before the rule was written because in the
  screen cap 2 fitted no closer than 1.5 and has flatter sides (lesson 51).
- **The corridor is reported, not judged**: the user's call, had the ellipse needed a thicker standoff to be safe.

The confirmation's rows were chosen from the screen: E15_35a (the scene's law with the ellipse), E125_35a (rounder),
E15_45a (thicker standoff: the corridor trade-off, if it had been needed) and E15_35 (spread held, S9's F20 open
decision), with partners C35a, C45a and C35.

## 8. The confirmation

`confirm/` (the scene's city and seeds 2–4, 14,904 flights, no errors): 40 straight lines, 30 blocks, 200 street routes,
36 square-on and 36 angled per city for each row (about 390 km of formation flights a row), the d_ref 1.2 square-on
rows (144 approaches each), and per law 720 along probes and 144 angled one-drone run-ins.

| row | loss events (per km, 95%) | contacts /km | street cohesive speed | surrounded | split, d_ref 2.0 · 1.2 (s) | one drone: standoff spread (m) | corridor (m, under 6 m) | rule |
|---|---|---|---|---|---|---|---|---|
| C35a (the scene) | 52 (0.133, 0.100–0.175) | 0.55 | 5.78 | 86.5% | 2.41 · 4.12 | 2.85 | 12.8 (376) | base |
| **E15_35a** | 38 (0.097, 0.069–0.134) | 0.40 | **6.21** | 89.2% | 2.37 · 3.89 | **1.58** | **14.3 (57)** | **pick** |
| E125_35a | 34 (0.087, 0.060–0.122) | 0.38 | 6.19 | 89.0% | 2.37 · 3.87 | 1.68 | 14.1 (128) | safe, delivers |
| C45a | 70 (0.179, 0.140–0.227) | 0.57 | 5.63 | 87.6% | 2.54 · 4.31 | 2.90 | 10.8 (584) | partner |
| E15_45a | 50 (0.128, 0.095–0.169) | 0.44 | 6.10 | 88.3% | 2.48 · 3.86 | 1.66 | 12.3 (164) | safe, delivers |
| C35 (S9's F20) | 33 (0.085, 0.058–0.119) | 0.63 | 5.36 | 89.2% | 2.31 · 4.07 | 2.85 | 12.8 (376) | partner |
| E15_35 | 29 (0.074, 0.050–0.107) | 0.62 | 5.81 | 92.9% | 2.28 · 3.59 | 1.58 | 14.3 (57) | safe, delivers |

The corridors are over all four cities (street gap less d_obs each side). The rule picks **E15_35a**: every ellipse
row is safe and delivers, and cap 1.5 is the evenest. Points worth keeping:
- **Every ellipse row is faster in the streets than its circle in every one of the four cities** (E15_35a 6.20–6.22
  against 5.72–5.84 m/s), and loses fewer drones in three of the four.
- **The one-drone standoff spread is ×0.55** under the ellipse (6 m off a wall 2.0 against 3.2 m; 9 m off 1.7 against
  2.8; the centre line 1.0 against 2.5), and the facade distance where an approach to a face slows varies 4.7 m from
  face to face against 7.3. The drone keeps 0.3 m nearer a wall 6 m off it on average; one drone touched no building.
- **The spread held** gains as much from the ellipse (E15_35 against C35: +8% in the streets), so S10 holds whichever
  way S9's spread decision goes.
- **The square-on split at d_ref 1.2 is faster with the ellipse** in every row (3.6–3.9 against 4.1–4.3 s): its flatter
  sides do not stall a formation with the slide's local gate.

## 9. Unity

The headless bench in BlockShapedPatchCityWorld as baked, the ellipse law at `b1f826f9` (uncommitted then, the same
code), routes from 18000, every set in one build with F20 and A85 re-flown as controls:

- `swarmellipse_bench1`: E15_35a, C35a, E15_35, C35 × 172 flights (30 straight lines, 20 blocks, 50 street routes, 36
  square-on, 36 angled; 37.5 km of formation flight each). Drones lost 4 / 6 / 7 / 7, contacts 17 / 12 / 22 / 27;
  street cohesive speed 6.13 / 5.83 / 5.77 / 5.29 (ellipse +5.2% adaptive, +9.1% held); surrounded 90% against 84%
  adaptive; in pieces 5.4% against 8.1%.
- `swarmellipse_bench12`: the same four at d_ref 1.2, 36 square-on approaches each: split 4.0 s with every approach
  splitting, against the scene's 6.0 s (94%); drones lost 0 / 2 / 0 / 0.
- `swarmellipse_bench1d`: one drone, E15_35 and C35, 60 along probes and 36 angled run-ins each: standoff spread 1.78
  against 3.18 m (×0.56), onset spread 4.9 against 8.9 m; no contacts.
- **The twins** (`swarmellipse_bench gate`) match Unity to 1–2% on cohesive and street speed for every set.
- **Contacts are the one measure where Unity leans the other way** with the adaptive spread (17 against 12, ×1.42),
  where the replica, at ten times the distance, has 0.72×; with the spread held Unity has 22 against 27. At these counts
  1.4× is not distinguishable from no difference (one-sided p ≈ 0.23); watch it in the first trial sessions.

## 10. What it means

Fly the ellipse at cap 1.5 with everything else as the scene has it. It does what it was built for: the shape round
each building is nearly the same distance from each of its walls, and a drone then meets every wall at nearly the same
distance, about half as variable as under the circle in the replica and in Unity. The narrow streets open as a
consequence (57 corridors under 6 m in four cities against 376) and the formation is faster in them, with no cost the
rule or Unity found beyond a handful of contacts in Unity. It needed no thicker standoff, so the corridor trade-off
never arose. One drone shares the shape, so both trial conditions get it.

**Limits.** Scripted pilots at full stick, as every study since S6; the replica is 2-D; Unity's 37 km a set is too short
to rank rare losses; the ellipse narrows some streets past the short ends of long buildings (3% by more than 2 m, the
worst to 8.4 m on seed 4), which no measure here showed costing anything; BlockShapedPatchCityWorld only (the C# works
for any collider, box buildings included, but no other scene was flown).

## Adopting it

Everything is in the code (`b1f826f9`), off by default, so adoption is scene values only:

```
python -B -m citysim.swarmtune apply --cand ../../../experiment_results/swarm_ellipse/confirm_cands.json \
    --label "E15_35a ellipse, aspect cap 1.5, d_obs 0.35, adaptive spread 0.85" [--dry-run]
```

The dry run shows exactly `obstacleShape: 1 -> 3` and `ellipseMaxAspect` inserted after it at 1.5 (the field's C#
default, so a scene without the key flies it too); every other field already matches. Run it with the scene not held
open in the editor (an open scene is re-serialised from memory on save), or pick Ellipse in SwarmManager's inspector
and save. After adoption an empty override in the harnesses means the ellipse: merge `swarmellipse.PRE_S10` under an
old candidate's `over` to fly it against the scene as it was.

## Running it

From `Assets/Scripts/Experiment` in the `stitching` env (the law is committed and `FORCE_LAW_COMMIT` bumped, so no
`SWARM_REPLICA_ALLOW_STALE` is needed):

```
python -B -m citysim.swarmellipse gaps                                   # walls.csv, gaps.csv
python -B -m citysim.swarmellipse screen --name screen1 --cand ../../../experiment_results/swarm_ellipse/screen_cands.json \
    --probes ../../../experiment_results/swarm_ellipse/probe_cands.json
python -B -m citysim.swarmellipse screen --name screen1 --cand .../screen12_cands.json --cross 0 --pass 0 --street 0 \
    --angle 0                                                            # the d_ref 1.2 square-on rows
python -B -m citysim.swarmellipse table --name screen1
python -B -m citysim.swarmellipse rule                                    # rule.json (refuses to overwrite)
python -B -m citysim.swarmellipse confirm --name confirm --cand .../confirm_cands.json --probes .../confirm_probe_cands.json
python -B -m citysim.swarmellipse confirm --name confirm --cand .../confirm12_cands.json --cross 0 --pass 0 --street 0 \
    --angle 0 --face 36                                                  # d_ref 1.2 square on
python -B -m citysim.swarmellipse choose --name confirm                   # verdicts.json
python -B -m citysim.swarmellipse_bench plan --name bench1 --cand .../bench_cands.json
python -B -m citysim.swarmellipse_bench plan --name bench12 --cand .../bench_cands.json --d 1.2 --cross 0 --pass 0 \
    --street 0 --angle 0 --face 36
python -B -m citysim.swarmellipse_bench plan --name bench1d --cand .../bench_single_cands.json --drones 1 --cross 0 \
    --pass 0 --street 0 --face 0 --angle 36 --along 60
#   then run_bench.ps1 as each plan prints (tags swarmellipse_bench1, _bench12, _bench1d), and
python -B -m citysim.swarmellipse_bench gate --name bench1 --runs D:\claude_obstacle_bench\runs\swarmellipse_bench1
#   the fit check: Unity.exe -batchmode -quit -projectPath <bench copy> -executeMethod EllipseFitProbe.Run with
#   ELLIPSE_PROBE_OUT=experiment_results/swarm_ellipse/unity_fits.json, then
python -B -m citysim.swarmellipse fitcheck
python -B -m citysim.swarmellipse_replays                                 # the page's replays, each twin checked
python -B -m citysim.swarmellipse_report                                  # the page; publish swarm_ellipse_bare.html
python -B Assets/Scripts/Experiment/citysim/tests/run_tests.py -k swarmellipse    # from the repo root
```

The page: https://claude.ai/artifact/QZHj23gYuV4xhfS7uwb6At (publish `swarm_ellipse_bare.html` with the files
`swarm_ellipse_files.json` lists, root `experiment_results/swarm_ellipse`).
