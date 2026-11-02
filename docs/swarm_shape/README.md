# The shape study (S8 Footprint Avoidance)

What the swarm's Olfati-Saber law avoids each building as, for flying **BlockShapedPatchCityWorld**: the circle round
the building's axis-aligned bounding box (as always), the smallest circle round its footprint, or the footprint itself.
Asked for on 2026-11-01, after S7: the bounds circles bulge into the streets, and with them the swarm is squeezed in
the streets and may not start its slide round a building as well as it could. The user's conditions: a method easy to
put into the latest scene, not far from Olfati-Saber, flown with S7's slide on and its straggler routing off, the
adaptive spread as a secondary factor; and the history behind the cylinder, which this study had to answer. The first
city avoided the bounding box itself, and a swarm flown square at a wall took a long time to split and surround the
building, since a flat face gives almost no sideways force. The cylinder fixed that.

Registry entry: [S8_footprint_avoidance.md](../studies/S8_footprint_avoidance.md). Page: [S8 Footprint Avoidance](https://claude.ai/artifact/BSzcnGUAjtXe7w23jBBqVc).
Code: `swarm/ObstacleFootprint.cs`, `OlfatiSaber.cs`, `SwarmPlaneController.cs`, `SwarmAlgorithm.cs`, `SwarmManager.cs`
(Unity); `swarm/Bench/swarm_replica.py` (`enclosing_circle`, `aim_circle`, the obstacle branches, `slide_round`);
`citysim/swarmshape.py` (the runs), `swarmshape_replays.py`, `swarmshape_report.py` (+ `.html`). Outputs:
`experiment_results/swarm_shape/`.

**The answer.** Avoid each building as **the smallest circle round its footprint** (`obstacleShape` EnclosingCircle),
with S7's slide and `d_obs` 0.35 (C35). The rule's pick (C35a) adds the adaptive spread, and in the replica that costs
nothing. In Unity it lost 3 colliding pairs in 100 flights where C35 lost none, so I would add it only if the extra
street speed matters more than that risk.
- **The bounds circles take about half of every street.** In the scene's city they leave a median 13.9 m of the 25 m
  between two facades, close 3 streets and narrow 69 of 794 below 6 m. S7's drones were lost there: re-flying S7's
  confirmation with every event located, 20 of its 38 lost drones went down where the circles leave under 10 m, a place
  the drones spent a tenth of their time.
- **The footprint itself does not work, for the reason the cylinder was chosen.** Without the slide, a formation flown
  square at a wall split round it in 20.5 s against 3.8 s for the bounds circles with the slide, and only 60% of the
  time split within 40 s. That is the user's first city again. With the slide it moves, but drones slide along the flat
  faces in single file and collide: 1.55 drones lost per km against 0.23, and 54 against 2 in 100 Unity flights.
- **The enclosing circle keeps the curvature and gives back most of the street**: 20.1 m left at the median. With the
  slide and `d_obs` 0.35 it is 9% faster in the streets in the replica (17% in Unity) and loses a tenth of the drones
  (0.02 against 0.23 per km). In Unity it lost none and touched buildings 5 times in 100 flights, against the bounds
  circles' 2 and 14: the best row there.
- **In the replica it also lets the adaptive spread pay**, which S7 had to leave off: the streets are 20% faster than
  the bounds circles with the slide (5.87 against 4.89 m/s), contacts are at the scene's present rate, and fewer drones
  are lost (0.13 against 0.23 per km). That is the rule's pick, C35a. Unity is less sure: 28% faster in the streets,
  but 3 pairs lost in 100 flights against C35's none.
- One drone shares the shape and is 7–8% faster too, so the trial comparison stays fair.
- **The cost** is that the drones fly nearer the facades: a split round a wall about 1 s later with the compact
  formation, and, at the thinner standoffs, more building contacts.

## 1. The setting: `SwarmManager.obstacleShape`

One enum on `SwarmManager`, values spelled out because Unity serialises enums by integer. `Cylinder` (0) is the law as
it was on every path: the replica re-flew 120 of S7's stored confirmation flights bit for bit with it.

| value | what each building is avoided as | how |
|---|---|---|
| `Cylinder` | the circle round the collider's axis-aligned bounds | `GetObstacleCylinder`, as always |
| `EnclosingCircle` | the smallest circle containing the footprint | the same `MakeCylinderFrame`, on that circle |
| `Footprint` | the footprint itself | `MakeFootprintFrame`: Olfati-Saber's wall beta-agent at the footprint's closest point (distance to the facade, mu 1, the normal turning round each corner as the distance field does) |

- **The footprint comes from the collider** (`ObstacleFootprint`, a static per-collider cache): the convex hull of a
  convex `MeshCollider`'s `sharedMesh` in world XZ (the scene's 289 prisms; static batching swaps the MeshFilter's mesh,
  never the collider's), a `BoxCollider`'s eight corners, otherwise the bounds rectangle. The smallest enclosing circle
  is a brute-force search over the hull's vertices (exact for the handful a footprint has), the same loop in
  `swarm_replica.enclosing_circle`. So it works in every scene, not only this one, with no layout file.
- **No margin.** The user asked whether the footprint needs a margin; it does not. A margin only shifts the distance,
  and in φ_β the σ₁ term is identical (σ₁((d − m) − d_obs) = σ₁(d − (d_obs + m))): only the ρ_h taper differs, and
  c_obs 14 saturates the force at the 4.57 m/s² ceiling within a few tenths of a metre of the field's edge. The
  standoff is Olfati-Saber's own `d_obs`, screened against the new shapes.
- **S7's rules under the new shapes take their centre and radius from the enclosing circle** (the split in halves, the
  own side, the hidden-centroid test, straggler routing's blocking test, the adaptive spread's aim):
  `OlfatiSaber.GetObstacleAimCircle`, and the frame's `axisCentre` / `radius`.
- **Under `Footprint`, two rules of the slide change**:
  - *the head-on gate is measured where the stick meets the surface*, `1 − |sin(incidence)| / slideHeadOnFraction`
    (`ObstacleFrame.localGate`): at a circle's surface that is S7's `1 − miss/(f R)` exactly, and on a flat face the
    whole slide wherever the stick meets the wall square on;
  - *a drone neither splitting round the building nor hidden from the centroid slides the way its stick leans along
    the wall*, the own-side rule only when it meets the wall square on. S7's own-side test can point back down a street
    along a long facade; screen 2 found this changed nothing measurable, and it is kept because it cannot do harm.
- **Scoring is not affected**: `swarmtune.score_flight` (met, surrounded, seconds past a building) and the head-on
  measures' arrival keep using the bounds circles whatever shape the law flies, so every shape is measured alike.
- The gizmos (`ObstacleCylinderGizmos`) draw the shape in force, and the d_obs line round it.

## 2. What the bounds circles take, and where S7 lost its drones

`swarmshape gaps` (→ `gaps.csv`) measures the narrowest clear width each shape leaves along every street with a
building on each side (facade to facade 25 m: 18 m of street and a 3.5 m footpath each side).

| city | shape | median | p10 | closed | under 6 m | under 10 m |
|---|---|---|---|---|---|---|
| the scene's (seed 1, 794 streets) | bounds circle | 13.9 | 6.6 | 3 | 69 | 161 |
| | enclosing circle | 20.1 | 12.7 | 0 | 15 | 44 |
| | the footprint, field at 1.6 m | 21.8 | 21.8 | 0 | 0 | 0 |
| seeds 2–4 (791–824 streets) | bounds circle | 13.8–14.2 | 6.4–7.6 | 3–8 | 46–70 | 160–185 |
| | enclosing circle | 19.7–19.9 | 11.4–12.9 | 0–2 | 8–13 | 39–55 |

The bounds circle reaches a median 8.5–9.1 m past the nearest facade (p90 12–13 m): across the 3.5 m footpath and about
halfway to the street's centre line. The enclosing circle's median radius is 12.8 m against 15.4.

**S7's losses were where the circles narrow the streets** (`swarmshape events`: S7's confirmation re-flown for T3, the
slide, the adaptive spread and both, 1440 flights, every one identical to its stored row, with each contact and loss
located). The drones spent 10–11% of their time where the two nearest bounds circles leave under 10 m; half of every
configuration's drone losses happened there (2 of 4, 2 of 4, 10 of 18, 6 of 12), five times the rate elsewhere, at a
median circle gap of 6–10 m where the facades stood 25 m apart. Contacts are on facades and so always inside a circle;
they carry no such signal.

## 3. How it was flown

S7's harness unchanged (`citysim/swarmtune.py`, `swarmflow.py`): the replica's `ContactSim` on the scene's prism city,
settled starts, the bench's waypoint pilot at full stick, S6's straight lines, blocks and street routes and their
measures, d_ref 2.0, **the slide on and route stragglers off in every arm** (the user's design; A0 is the slide off
too, the scene as it flies). `citysim/swarmshape.py` adds:

- **A fourth flight, `face`: square on at a wall.** From open ground 50 m out along the outward normal of a straight
  face (≥ 12 m) of a building on the city's edge, with a clear corridor ±18 m wide, offset 0, 0.3 or 0.6 of the
  half-face, aimed through the building at the first point beyond it clear of every footprint by 6 m. 52–57 faces
  qualify per city. Measured from arrival (the centroid within the bounds circle's R + 15 m, the same for every shape),
  censored at 40 s: `t_side` (the first drone beside the building beyond either end of the face), `t_split` (drones
  beyond both ends), `t_wrap` (surrounded), `t_through` (past its back, or the point behind it taken), the longest stall.
  It is flown at d_ref 2.0 within the main runs and, as its own run, **at d_ref 1.2**, where the ring (about 19 m) cannot
  simply enclose the building and has to split round it: the user's case.
- Route ranges: screens from 7000, the confirmation from 8000, the Unity bench from 9000 (S6 and S7 used 0–6999).

Runs: screens 1 and 2 in the scene's city (140 flights per row: 48 straight lines, 32 blocks, 24 street routes, 36
square-on; and 60 square-on at d_ref 1.2), the rule (`rule.json`, written 2026-11-02 00:00, before the confirmation),
the confirmation in the scene's city and S6's three held-out ones (504 flights per row: 40 / 30 / 20 / 36 in each, ~104
km), one drone for six of its rows, 144 square-on approaches per row at d_ref 1.2, and Unity (§7).

## 4. What the screens found

Screen 1 (scene's city, d_ref 2.0; d_obs 0.16 unless given):

| row | cohesive speed | streets | surrounded | lost /km | contacts /km | split round a wall (s) |
|---|---|---|---|---|---|---|
| A0 bounds circles, no slide | 3.97 | 4.17 | 97% | 0.07 | 0.53 | 3.3 |
| A1 bounds circles + slide | 4.86 | 4.99 | 88% | 0.00 | 0.20 | 2.6 |
| B0 footprint, no slide, d_obs 0.25 | 3.27 | 3.68 | 93% | 0.00 | 4.00 | 4.5 |
| B1 footprint + slide, d_obs 0.16 / 0.25 / 0.35 | 5.19 / 5.29 / 5.20 | 5.64 / 5.53 / 5.49 | 87–91% | 0.85 / 1.25 / 1.57 | 13.9 / 4.9 / 2.0 | 2.8–3.2 |
| B2 footprint + slide, leaning side rule, d_obs 0.25 / 0.35 / 0.5 | 5.21 / 5.17 / 5.02 | 5.35 / 5.30 / 4.99 | 84–89% | 1.51 / 1.57 / 2.30 | 5.0 / 2.8 / 1.8 | 3.0–4.4 |
| C enclosing circle + slide, d_obs 0.16 / 0.25 / 0.35 | 5.58 / 5.45 / 5.42 | 5.57 / 5.44 / 5.30 | 88–91% | 0.00 / 0.07 / 0.13 | 1.25 / 0.98 / 0.53 | 2.5–2.6 |
| C enclosing circle + slide, d_shield 2.0 | 5.28 | 5.25 | 90% | 0.00 | 1.61 | 2.6 |

At d_ref 1.2, square on (60 approaches): the bounds circles split round the building in 6.1 s without the slide and
3.8 s with it; **the footprint without the slide took 22.4 s, only 53% split within 40 s, its longest stall 6.6 s**;
with the slide 6.1–7.6 s, losing 0.4–1.0 drones per approach; the enclosing circle 3.7–3.9 s, none lost.

What carried forward:
- **The footprint's losses are the slide in single file along flat faces.** Every one was a drone-drone pair, nearly
  all with the slide on (B0 lost none). Attributing each colliding pair's forces (the closing term is the slid command,
  at closing speeds up to 9.6 m/s): one drone pinned 1.4 m off a facade with almost no command left, in a pocket
  between two faces, and its neighbour sliding along the same wall into it; or two drones sliding onto the same lane,
  pushed together by the fields of two converging faces. Sliding the way the stick leans along the wall instead of
  S7's own-side rule (which along a long facade can turn a drone back down the street) changed nothing (B2 against B1).
  A larger standoff only moves the lane out: d_obs 0.5 lost more.
- **The enclosing circle took the speed with none of that**: it is a cylinder, so the slide, the split and the field
  work as they always did.

## 5. The rule

`rule.json`, fixed before the confirmation, against A1 (the bounds circles with the slide, S7's pick): **safe** =
losses at most A1's + 0.05 per km, contacts at most **A0's** (the scene's law as it flies: the bound S7's rule used, not
the slide's own lower count, which would rule out any shape that lets drones nearer the facades), ending in pieces at
most 2 points above A1's, hull share ≥ 0.5, the split round a wall (d_ref 2.0) at most 1.1× A1's; **helps** = the
streets 5% faster than A1, cohesive speed at least 98% of A1's, surrounding within 10 points. Pick: the best S6
penalised score among safe and helping rows.

## 6. The confirmation

Four cities, 504 flights per row (~104 km of straight lines, blocks and streets, plus 144 square-on approaches).
Cohesive speed, one group and surrounded over the straight lines and blocks; streets over the street routes; the split
over the square-on approaches at d_ref 2.0.

| row | cohesive speed | streets | one group | surrounded | lost /km | contacts /km | in pieces | split (s) | rule |
|---|---|---|---|---|---|---|---|---|---|
| A0 bounds, no slide (the scene today) | 4.00 | 4.30 | 84% | 98% | 0.12 | 0.39 | 5.0% | 3.1 | contact bound |
| A1 bounds + slide | 4.93 | 4.89 | 91% | 91% | 0.23 | 0.25 | 0.8% | 2.5 | baseline |
| B2 footprint + slide, d_obs 0.25 | 5.25 | 5.20 | 88% | 92% | 1.55 | 5.32 | 6.4% | 4.8 | not safe |
| C16 enclosing + slide | 5.63 | 5.61 | 95% | 94% | 0.02 | 1.33 | 0.0% | 2.4 | not safe (contacts) |
| C25 enclosing + slide, d_obs 0.25 | 5.51 | 5.54 | 94% | 93% | 0.02 | 0.74 | 0.6% | 2.4 | not safe (contacts) |
| C35 enclosing + slide, d_obs 0.35 | 5.39 | 5.33 | 93% | 93% | 0.02 | 0.62 | 1.7% | 2.5 | not safe (contacts) |
| A1a bounds + slide + adaptive | 5.06 | 5.34 | 93% | 90% | 0.15 | 0.23 | 1.1% | 2.6 | safe, helps |
| C16a enclosing + slide + adaptive | 5.75 | 6.03 | 96% | 92% | 0.00 | 1.21 | 0.6% | 2.5 | not safe (contacts) |
| C25a enclosing + slide + adaptive, d_obs 0.25 | 5.67 | 5.84 | 96% | 91% | 0.06 | 0.70 | 0.6% | 2.5 | not safe (contacts) |
| **C35a enclosing + slide + adaptive, d_obs 0.35** | **5.55** | **5.87** | 95% | 90% | 0.13 | 0.39 | 1.4% | 2.6 | **the pick** |

- **The rule picks C35a** (score 5.03; A1a, the other safe and helping row, 4.59). It sits exactly at the contact bound
  (0.393 against A0's 0.393). Every other enclosing-circle row passes every bar but contacts, with a tenth of A1's
  losses.
- **By city** (cohesive speed / streets, m/s): A1 4.90/4.92, 4.78/4.71, 5.12/5.07, 4.92/4.85; C35a 5.45/5.89,
  5.58/5.82, 5.69/5.82, 5.48/5.96: ahead in all four, 11–13% overall and 17–23% in the streets.
- **These routes were harder than S7's**: A1 lost 0.23 drones per km here against the slide's 0.04 in S7's
  confirmation, and A0 0.12 against T3's 0.04. The comparison is paired (every row flew the same routes).
- **One drone** (progress speed, m/s; the streets): bounds circles 4.86 without the slide, 5.68 with it (6.55); the
  enclosing circle with the slide 6.11–6.13 (6.94–6.95), 8% faster; the footprint with the slide 6.24 (7.32) but its
  longest stall on a straight line 5.2 s against 0.5. No one-drone row lost or touched anything.

## 7. Square on at a wall (d_ref 1.2, four cities, 144 approaches per row)

| row | first round an end | split | split within 40 s | surrounded | past it | longest stall | lost per approach |
|---|---|---|---|---|---|---|---|
| A0 bounds, no slide | 3.4 | 7.4 | 94% | 6.9 | 9.1 | 0.5 | 0.01 |
| A1 bounds + slide | 3.0 | 3.8 | 100% | 4.9 | 7.3 | 0.5 | 0.01 |
| B0 footprint, no slide | 6.2 | **20.5** | **60%** | 11.1 | 15.6 | **4.8** | 0.00 |
| B2 footprint + slide | 3.2 | 6.7 | 94% | 7.6 | 7.5 | 0.6 | **0.39** |
| C16 / C25 / C35 enclosing + slide | 2.9 | 4.6–4.8 | 97% | 5.7–5.8 | 6.9–7.1 | 0.5 | 0.00 |
| C35a (+ adaptive) | 2.9 | 4.8 | 97% | 5.8 | 7.1 | 0.5 | 0.00 |

Seconds from arriving. With the compact formation the enclosing circle starts round sooner and gets past sooner than the
bounds circles but completes the split about 1 s later (the smaller circle turns the drones nearer the wall); in the
scene's city alone (screen) it split 3.7 s against 3.8.

## 8. Unity

`bench_shape` (+ `bench_extra` for C35a, flown afterwards on the same scenarios): BlockShapedPatchCityWorld as baked,
100 flights per row (30 straight lines, 20 blocks, 14 street routes, 36 square-on approaches from 9000; ~21 km),
scored by `bench-gate` beside replica twins from Unity's recorded starts. Drones lost and contacts are Unity's own.

Over the 64 straight lines, blocks and street routes (in brackets the replica's twins), and Unity's own counts over all
100 flights (losses are drone-drone pairs, each pair two drones):

| set | cohesive speed | streets | one group | lost /km | contacts /km | split round a wall (s, d_ref 2.0) | Unity, 100 flights: lost / contacts |
|---|---|---|---|---|---|---|---|
| A0 bounds, no slide | 3.82 (3.90) | 4.39 (4.24) | 81% | 0.00 | 0.58 | 3.4 | 0 / 15 |
| A1 bounds + slide | 4.88 (4.98) | 4.51 (4.70) | 90% | 0.11 | 0.53 | 2.7 | 2 / 14 |
| B0 footprint, no slide | 3.59 (3.63) | 3.46 (3.48) | 68% | 0.05 | 4.83 | 8.0 | 1 / 98 |
| B2 footprint + slide | 5.03 (5.26) | 5.25 (5.54) | 85% | 1.81 | 5.63 | 5.3 | **54 / 116** |
| C16 enclosing + slide | 5.64 (5.66) | 5.67 (5.73) | 96% | 0.11 | 1.12 | 2.5 | 2 / 22 |
| C25 enclosing + slide, d_obs 0.25 | 5.47 (5.66) | 5.41 (5.49) | 94% | 0.11 | 0.58 | 2.6 | 2 / 11 |
| **C35 enclosing + slide, d_obs 0.35** | **5.31 (5.33)** | **5.29 (5.41)** | 92% | **0.00** | **0.27** | 2.6 | **0 / 5** |
| A1a bounds + slide + adaptive | 4.98 (5.12) | 4.94 (5.13) | 91% | 0.11 | 0.58 | 2.8 | 2 / 12 |
| C25a enclosing + adaptive, d_obs 0.25 | 5.67 (5.71) | 5.76 (5.83) | 96% | 0.21 | 0.53 | 2.7 | 4 / 12 |
| C35a enclosing + adaptive, d_obs 0.35 (the rule's pick) | 5.49 (5.60) | **5.79** (5.69) | 94% | 0.32 | 0.27 | 2.7 | 6 / 5 |

- **The twins match Unity** within 0–4% on cohesive speed (B2 5%) and the split round a wall within 0.1 s, as in S6
  and S7.
- **The footprint fails in Unity as in the replica**: 54 drones lost and 116 contacts with the slide, 98 contacts
  without it.
- **C35, the enclosing circle with the slide at d_obs 0.35, is the best row in Unity**: 9% faster while together and
  17% faster in the streets than the bounds circles with the slide, no drone lost and the fewest contacts of any row
  (5 against 14).
- **The rule's pick, C35a, is the fastest in the streets (+28%) but lost 3 pairs in Unity** against C35's none and A1's
  one. The replica's ~100 km gave it 0.13 drones per km against A1's 0.23. Three pairs in 100 flights is too few to
  settle, but it points the same way as S7's Unity flights with the adaptive spread, which lost 4–8 drones in 64.
- **Square on at a wall with the compact formation (d_ref 1.2, 36 approaches, Unity / twin)**: the bounds circles with
  the slide split round the building in 4.0 / 4.0 s; **the footprint without the slide in 21.8 / 21.8 s, only 56%
  within 40 s** (the user's first city, reproduced in Unity); the footprint with the slide 7.6 / 7.5 s, losing 26 drones
  in 36 approaches; the enclosing circle 3.8 / 3.8 s (C25, 2 colliding pairs) and 5.0 / 5.0 s (C35a, none lost).
- **One drone (100 flights)**: the enclosing circle 6.12 m/s (C35; C25 6.13) against 5.67 with the bounds circles, the
  streets 6.97 against 6.51; nothing lost or touched. With the footprint one drone stalled 7.2 s on average at a wall
  even with the slide (bounds circles 0.8).

## 9. What it means

- **BlockShapedPatchCityWorld now flies C35a** (the user's choice, 2026-11-02): `obstacleShape` EnclosingCircle,
  `d_obs` 0.35, the slide and the adaptive spread on, route stragglers off, d_ref 2.0. I had recommended C35 (without
  the adaptive spread), adding it only if its extra street speed outweighed the collisions Unity hinted at. The rule
  picked C35a on the replica's ~100 km, where it is the fastest safe row; Unity's 100 flights lost 3 pairs with it and
  none without. A longer Unity run, or the trials' own flights, would settle it. Both trial conditions change together
  (one drone shares the shape and the field). Every other scene is still `Cylinder` with S7's switches off.
- **The harnesses read the scene as their baseline**, so an empty override is now C35a. Every S7 and S8 candidate file
  with an empty `over` (S7's T3, S8's A0) was flown against T3; `swarmshape.PRE_S8` (`obstacleShape=0, d_obs=0.16`,
  slide and adaptive spread off) gives that scene back.
- **Keep any future shape curved.** The cylinder's turning normal is not only what starts a formation round a wall
  (the user's first city); with the slide it is also what spreads drones round a building instead of lining them up
  along it.
- **Contacts are the enclosing circle's price at a thin standoff**, and `d_obs` is its dial: in the replica 1.3 per km
  at 0.16, 0.6–0.7 at 0.25–0.35, 0.39 with the adaptive spread at 0.35; in Unity 22, 11 and 5 in 100 flights at 0.16,
  0.25 and 0.35 (the bounds circles with the slide 14). S7's route stragglers (left off here as asked) is the measured
  cure if fewer matter.
- **Not done**: a capsule or rounded rectangle for long buildings (the enclosing circle still reaches a median 6.9 m
  past a facade), real pilots, what the screens show.

## Running it

From `Assets/Scripts/Experiment`, the `stitching` env.

```
python -B -m citysim.swarmshape gaps                          # what each shape leaves of the streets -> gaps.csv
python -B -m citysim.swarmshape screen --name screen1 --cand ../../../experiment_results/swarm_shape/screen1_cands.json
python -B -m citysim.swarmshape screen --name screen1c --cand ../../../experiment_results/swarm_shape/screen1c_cands.json --cross 0 --pass 0 --street 0 --face 60
python -B -m citysim.swarmshape rule                          # rule.json, before the confirmation
python -B -m citysim.swarmshape confirm --name confirm --cand ../../../experiment_results/swarm_shape/confirm_cands.json
python -B -m citysim.swarmshape confirm --name confirm --cand ../../../experiment_results/swarm_shape/confirm_single_cands.json --single
python -B -m citysim.swarmshape confirm --name confirmc --cand ../../../experiment_results/swarm_shape/confirmc_cands.json --cross 0 --pass 0 --street 0 --face 36
python -B -m citysim.swarmshape choose --name confirm         # -> confirm/verdicts.json
python -B -m citysim.swarmshape bench-plan --name bench_shape --cand ../../../experiment_results/swarm_shape/bench_cands.json
python -B -m citysim.swarmshape bench-plan --name bench_single --cand ../../../experiment_results/swarm_shape/bench_single_cands.json --drones 1
python -B -m citysim.swarmshape bench-plan --name bench_face12 --cand ../../../experiment_results/swarm_shape/bench_face12_cands.json --cross 0 --pass 0 --street 0 --face 36
#   then run_bench.ps1 as printed
python -B -m citysim.swarmshape bench-gate --name bench_shape --runs D:\claude_obstacle_bench\runs\swarmshape_bench_shape
python -B -m citysim.swarmshape events                        # S7's confirmation re-flown, its events located
python -B -m citysim.swarmshape_replays                       # the page's flights, each twin identical to the gate
python -B -m citysim.swarmshape_report                        # then publish swarm_shape_bare.html with swarm_shape_files.json
```

A candidates file is S6's: `{"label", "d", "over"}`, the label's first word the configuration's code (A0, A1, B0, B2,
C16, C25, C35; a trailing `a` adds the adaptive spread), `over` SwarmManager fields and `x:` replica-only knobs
(`x:fp_side`: `lean`, the C#'s and the default, or `s7`, S7's own-side rule, which screen 1's footprint rows flew).
Screens 1 and 2 share the folder `screen1/` (the same routes), and so do their square-on runs (`screen1c/`).
