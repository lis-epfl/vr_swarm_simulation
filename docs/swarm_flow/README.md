# The flow study (S7 Building Flow)

Three changes to how the swarm's Olfati-Saber law treats buildings, each behind its own `SwarmManager` switch, for
flying **BlockShapedPatchCityWorld**: a drone flown at a building slides round it instead of stalling; a drone left
behind a building is drawn round it to the swarm instead of through its facade; and the formation narrows in the
streets and widens again when it flies at a building. Asked for on 2026-10-30, after S6, from the two sets of
recommendations that followed it (the first set's 1, 2 and 4, the second set's 4, 6 and 7), with the instruction to stay
with the Olfati-Saber law (no path finding) and to give each change a binary switch in the Inspector.

Registry entry: [S7_building_flow.md](../studies/S7_building_flow.md). Page: [S7 Building Flow](https://claude.ai/artifact/UkwGfhkWqnBAqF4iuBiz9a).
Code: `swarm/OlfatiSaber.cs`, `SwarmPlaneController.cs`, `SwarmAlgorithm.cs`, `SwarmManager.cs` (Unity);
`swarm/Bench/swarm_replica.py` (`slide_round`, `route_round`, `aim_miss`, `spread_target`); `citysim/swarmflow.py`
(the runs), `swarmflow_replays.py`, `swarmflow_report.py` (+ `.html`). Outputs: `experiment_results/swarm_flow/`.

**The answer.** Switch on **slide round buildings**, the rule's pick. In the confirmation (four cities, about 100 km per
setting) the swarm moves 22% faster while together (cohesive speed 4.98 against 4.10 m/s; 23% in Unity) and gets past a
building in 10.6 s instead of 12.6, with 15% fewer building contacts, the same losses and no flight ending in pieces,
while still surrounding 90% of the buildings it flies at (98% before). One drone shares the shield and slides too: 17%
faster, its longest stall in a flight 0.5 s instead of 3.1. **Route stragglers** cuts building contacts (28% alone; with
the slide 58% fewer than the slide alone, one contact in 64 Unity flights) and changes nothing else; with the slide the
replica lost a little more (6 colliding pairs against 2 in ~100 km, past the rule's bar; Unity 2 pairs each), so it is
optional. **The adaptive spread** makes the streets 14–23% faster but costs drones (0.18 lost per km against 0.04; Unity
8 against 2 in 64 flights): a narrower formation is squeezed together in the tight gaps between these buildings. It now
ships at a gentler setting (0.85 of d_ref, a 3 s filter), which halved its Unity losses alone, but leave it off for the
trials until the avoidance circles are tightened.

## 1. The three switches

All three are on `SwarmManager`, off by default; off, each is the force law exactly as before (the replica re-flew 24
stored S6 flights bit for bit with them off). They act in horizontal swarming only, not in vertical-plane mode, and need
the swarm-wide aggregates `SwarmPlaneController` computes, which it now does whenever any of them is on. Their settings
are the shipped defaults below; BlockShapedPatchCityWorld saves none of them, so ticking a switch flies these values.

### 1.1 Slide round buildings (`slideRoundBuildings`)

**The problem.** Near a building the obstacle shield (`OlfatiSaber.RemoveInwardComponent`, range `d_shield`) removes the
part of the pilot's stick pointing into it. Flown straight at a building, that is the whole stick: the drone stops
against the facade with nothing to push it sideways, and only the cylinder's own curve (for an off-centre approach) or
cohesion (the rest of the swarm dragging it) gets it round. S6 measured one drone's longest stall at 9–10 s a flight.

**The change** (`OlfatiSaber.SlideRoundObstacles`, called from `ProjectCommandVelocity`; the replica's `slide_round`).
Frame by frame, as the shield always worked: the inward part is faded out by the shield's bump exactly as before, and
the same amount, times a head-on gate, is added along the building's avoidance cylinder (its tangent). The gate is
`1 − miss / (slideHeadOnFraction × R)`, `miss` being how far the stick's line from the drone passes from the building's
centre: a drone passing beside a building is untouched, and a glancing one keeps the slide the cylinder's curve already
gave it. The result is held to the stick's length: the command is turned, never lengthened.

**Which way round** decides whether this helps or tears the swarm apart, and took most of the screens (section 3):

1. When the swarm flies squarely at the building (the stick's line from its centroid, `surroundLookAhead` long, passes
   within `slideSplitAimFraction × R` of the building's centre), **the formation splits in halves**: a drone left of the
   centroid's track goes left. That is the surround.
2. Otherwise, when the building hides the swarm's centroid from the drone, it goes round **on the centroid's side**
   (then downstream of the swarm's mean velocity): a straggler, or a drone that has strayed into the next street,
   comes back instead of flying on round the far side.
3. Otherwise **its own side** of the building's centreline, the short way.

Splitting by halves everywhere, as first suggested, is wrong at street corners: when the route turns, the trailing half
of the formation sees the corner building head-on and goes round its far side, into the wrong street.

| setting | shipped | meaning |
|---|---|---|
| `slideHeadOnFraction` | 0.8 | the slide acts when the stick's line passes within this share of the building's avoidance radius of its centre |
| `slideSplitAimFraction` | 0.6 | the formation splits in halves only round a building the centroid's track passes within this share of the radius of |
| `surroundLookAhead` | 4 (40 m) | shared with the adaptive spread: how far ahead along the stick a building counts as flown at |

The single drone shares the shield, so it slides too (rules 1 and 2 do not arise for it: its centroid is itself).

### 1.2 Route stragglers round buildings (`routeStragglers`)

**The problem.** Cohesion is never shielded. A drone left on the far side of a building is pulled towards the rest of
the swarm straight through it, by a pull (up to ~6 m/s² from nine neighbours) stronger than the obstacle field's ceiling
(`maxObstacleAccel`, 4.57): S6 found cohesion the largest push into the facade in every contact it logged.

**The change** (`OlfatiSaber.FirstBlockingFrame` and `RouteBlockedPull`; the replica's `route_round`). In the neighbour
loop, **attraction** towards a neighbour whose straight line from the drone passes through the avoidance cylinder of a
building within `d_shield` is set aside, against the first such cylinder the line enters. Per cylinder, that hidden
pull may still press the drone towards the building by up to `stragglerPressShare × maxObstacleAccel`, which the field
can always resist; the excess is faded out by the shield's bump and the same amount added along the building, on the side
the hidden pull leans to (a pull straight through goes to the centroid's side, then downstream). Held to the hidden
pull's own length. The lattice's **repulsion**, and everything between drones that can see each other, is untouched.

Two earlier forms failed (section 3): turning cohesion's whole push into the building also turned the drones' repulsion
of each other, so neighbours beside a facade collided; turning all of the hidden pull left nothing holding the two halves
of a formation wrapped round a building it surrounds, and the swarm broke up.

| setting | shipped | meaning |
|---|---|---|
| `stragglerPressShare` | 0.7 | the share of the obstacle field's ceiling the hidden pull may still press with; the rest is turned |

### 1.3 Adaptive spread (`adaptiveSpread`)

**The problem.** S6 held d_ref constant at 2.0: the ring (31 m) wraps a building's avoidance circle, but it does not fit
the streets (25 m facade to facade, less where the circles bulge), where 1.6 is about a fifth quicker.

**The change** (`SwarmPlaneController.UpdateSpreadScale`; the replica's `spread_target` in `Sim.step`). One swarm-wide
scale on d_ref, the pilot's dial or the scene's: 1 while a building lies on the swarm's track within
`surroundLookAhead` (the stick's line from the centroid passes within `surroundAimFraction × R` of its centre; with no
stick, the centroid itself) or while no building stands within that look-ahead at all; `streetSpreadFactor`
otherwise. Seeded, then low-passed with `spreadFilterTime`. `OlfatiSaber.EffectiveDRef` (d_ref × scale) then replaces
d_ref everywhere the law uses it: the lattice, the core's standoff, a d_ref-relative cohesion range. The scale in force
is logged per sample as `spreadScale` in a run's `shape.csv`. The look-ahead is what buys time to widen: the lattice
takes about 5 s to follow a change of d_ref.

| setting | shipped | meaning |
|---|---|---|
| `streetSpreadFactor` | 0.85 | the share of d_ref held among buildings with none ahead |
| `surroundLookAhead` | 4 (40 m) | how far ahead of the centroid, along the stick, a building counts as flown at |
| `surroundAimFraction` | 0.8 | the swarm widens for a building its track passes within this share of the radius of |
| `spreadFilterTime` | 3 | time constant of the scale's low-pass (s) |

## 2. How it was flown

S6's harness, unchanged (`citysim/swarmtune.py`): the replica's `ContactSim` on the scene's prism city (Unity's
avoidance through each prism's bounding circle, PhysX-like contacts with friction 0.6, pair and isolation losses), every
flight from a settled formation (lesson 39), the bench's waypoint pilot at full stick, and S6's three flights (a straight
line across the city, a pass through one block, a street route) and measures (cohesive speed the headline; one group,
contacts, losses, buildings surrounded, seconds past a building, ending in pieces, hull share). `citysim/swarmflow.py`
adds the adaptive spread's share of d_ref per flight (`sigma_mean`, `narrow_frac`) and the S7 route ranges: screens from
route 3000, the confirmation from 4000, the Unity bench from 5000, the adaptive tuning from 6000 (S6 used 0–2999).

- **Screens** (`screen1`–`screen6`): the scene's city, 52 (screens 1–3) or 104 (4–6) flights per variant, the same routes
  for every variant of a run; T3 (all off) in each.
- **The rule** (`rule.json`, fixed after the screens and before the confirmation): safe = losses at most T3's + 0.05 per
  km, contacts at most T3's, ending in pieces at most 2 points above T3's, hull share ≥ 0.5; helps = cohesive speed (straight
  lines and blocks) ≥ 5% above T3's, the streets ≥ 98% of T3's, buildings surrounded within 10 points of T3's. Pick: the
  safe and helping configuration with the best S6 penalised score; none → recommend none.
- **Confirmation** (`confirm`): all eight on/off combinations, fresh routes in the scene's city and S6's three held-out
  cities, 40 straight lines, 30 blocks and 20 street routes in each (360 flights, ~100 km per combination); and one drone
  with the slide off and on.
- **Unity** (`bench_flow`, `bench_single`): BlockShapedPatchCityWorld as baked, all eight combinations and the one-drone
  pair, 64 fresh flights each (30 straight lines, 20 blocks, 14 street routes), scored with the same measures by
  `bench-gate` beside replica twins flown from Unity's recorded starts.

## 3. What the screens found

Each line against its own screen's T3 (same routes): cohesive speed, the streets' cohesive speed, buildings surrounded on
the straight lines, drones lost and contacts per km.

| variant | screen | cohesive speed | streets | surrounded | lost /km | contacts /km | outcome |
|---|---|---|---|---|---|---|---|
| Slide: every drone round its own side of the building | 1 | 4.71 (4.02) | 4.67 (4.55) | 81% (95%) | 0.00 (0.00) | 0.19 (0.64) | the base rule |
| Slide: the formation in halves everywhere (the suggestion as written) | 1 | 2.83 (4.02) | 1.85 (4.55) | 100% (95%) | 0.13 (0.00) | 0.64 (0.64) | dropped: at a street corner the trailing half goes round the far side of the corner building |
| Slide: always round on the swarm's side | 1 | 4.75 (4.02) | 5.23 (4.55) | 54% (95%) | 1.03 (0.00) | 0.32 (0.64) | dropped: stops surrounding |
| Slide: halves only round a building the swarm flies at (within 1.0 R) | 2 | 4.70 (4.02) | 3.78 (4.55) | 96% (95%) | 0.00 (0.00) | 0.19 (0.64) | dropped: the split fires at street bends |
| … within 0.6 R | 3 | 4.76 (4.02) | 4.68 (4.55) | 94% (95%) | 0.00 (0.00) | 0.32 (0.64) | kept |
| … and the swarm's side when the building hides the centroid | 4 | 4.76 (4.00) | 4.83 (4.34) | 88% (93%) | 0.13 (0.00) | 0.41 (0.54) | kept: the shipped rule, steadier in the streets |
| Slide for anything the stick line hits (head-on fraction 1.0) | 5 | 4.98 (4.00) | 4.62 (4.34) | 85% (93%) | 0.25 (0.00) | 0.22 (0.54) | not taken: faster, but loses drones in the streets |
| Route: cohesion's whole inward part turned along the building | 1 | 3.17 (4.02) | 4.22 (4.55) | 92% (95%) | 2.05 (0.00) | 0.00 (0.64) | dropped: turned the drones' mutual repulsion too, so neighbours collided |
| Route: only attraction to hidden neighbours, all of it turned | 2 | 3.41 (4.02) | 4.49 (4.55) | 97% (95%) | 0.39 (0.00) | 0.13 (0.64) | dropped: nothing left holding a surround's two halves together |
| Route: keep up to 0.5 of the field's ceiling, turn the rest | 3 | 4.00 (4.02) | 4.60 (4.55) | 95% (95%) | 0.00 (0.00) | 0.26 (0.64) | fewer contacts |
| … 0.7 | 3 | 4.13 (4.02) | 4.63 (4.55) | 95% (95%) | 0.00 (0.00) | 0.39 (0.64) | kept |
| Adaptive spread: 0.7 of d_ref in the streets | 1 | 4.23 (4.02) | 5.04 (4.55) | 91% (95%) | 0.00 (0.00) | 0.45 (0.64) | faster, but loses drones in tight gaps in later screens |
| … with a stronger close-range damper (c_damp 4, d_damp 0.5) | 4 | 4.42 (4.00) | 5.31 (4.34) | 80% (93%) | 0.57 (0.00) | 0.66 (0.54) | dropped: the damper does not stop these collisions |
| Adaptive spread: 0.8 of d_ref in the streets | 5 | 4.32 (4.00) | 5.16 (4.34) | 90% (93%) | 0.19 (0.00) | 0.63 (0.54) | confirmed |
| … and 1.2 of d_ref at a building | 6 | 4.14 (4.00) | 4.40 (4.34) | 91% (93%) | 0.06 (0.00) | 0.51 (0.54) | dropped: no better surrounding, slower streets |

What carried forward:
- **The slide's way round is the whole difference between a gain and a mess.** Each drone its own side was already +17%
  and cut contacts 70%, but surrounded only 81% of buildings; the formation in halves everywhere surrounded every one and
  shattered the street flights (one group 37%); always the swarm's side held together and stopped surrounding. Halves
  only round a building the swarm flies squarely at (within 0.6 of its radius; 0.8–1.0 also fires at street bends), with
  the swarm's side for a drone the building hides the centroid from, kept the gain and the surround.
- **Route needs both limits**: attraction only (never repulsion), and only the excess over a share of the field's
  ceiling. Diagnosed with per-term force attribution at each collision (`diag_pair`): the first form turned a pair's
  mutual repulsion (−4.5 m/s² apart) into a closing push.
- **The adaptive spread's losses are pairs squeezed together by the fields of two buildings in a tight gap** (the obstacle
  term +7–8 m/s² closing), S5's and S6's collision regime, and 5 of the confirmation's 9 pair losses under it came while the
  formation was narrowing or widening. A stronger close-range damper (`c_damp` 4, `d_damp` 0.5) did not stop them, and
  widening past d_ref at a building (×1.1, ×1.2) did not surround more.

## 4. The confirmation

The rule (`confirm/verdicts.json`) picks **slide**: the only combination both safe and helping. Every combination
that adds route or adaptive to it is faster or touches fewer buildings, but loses more drones than the bar allows (T3's
0.04 + 0.05 per km); route alone is safe but, as designed, no faster. The adaptive rows flew 0.8 of d_ref with a 1 s
filter, the defaults then (it ships at 0.85 and 3 s since section 6).

| switches on | cohesive speed | progress | one group | streets | surrounded | past a building (s) | lost /km | contacts /km | ended in pieces | rule |
|---|---|---|---|---|---|---|---|---|---|---|
| none (T3) | 4.10 | 4.78 | 86% | 4.38 | 98% | 12.6 | 0.04 | 0.39 | 4% | baseline |
| slide | 4.98 | 5.44 | 92% | 4.98 | 90% | 10.6 | 0.04 | 0.33 | 0% | **the pick**: safe, helps |
| route | 4.11 | 4.76 | 86% | 4.38 | 97% | 12.5 | 0.02 | 0.28 | 5% | safe, does not help |
| adaptive | 4.22 | 4.87 | 86% | 5.00 | 95% | 12.6 | 0.18 | 0.50 | 5% | not safe, does not help |
| slide + route | 4.89 | 5.42 | 90% | 4.87 | 90% | 10.6 | 0.12 | 0.14 | 0% | not safe, helps |
| slide + adaptive | 5.15 | 5.50 | 93% | 5.38 | 90% | 10.7 | 0.12 | 0.14 | 1% | not safe, helps |
| route + adaptive | 4.30 | 4.88 | 88% | 4.99 | 95% | 12.6 | 0.16 | 0.16 | 3% | not safe, helps |
| slide + route + adaptive | 5.08 | 5.49 | 92% | 5.41 | 89% | 10.7 | 0.18 | 0.09 | 0% | not safe, helps |
| one drone, all off | 4.87 | 4.87 | – | 6.53 | – | 13.0 | 0 | 0 | – | |
| one drone, slide on | 5.69 | 5.69 | – | 6.54 | – | 10.0 | 0 | 0 | – | |

Cohesive speed, progress, one group, surrounded and past a building are over the straight lines and blocks; "streets" is
the street routes' cohesive speed; losses, contacts and ending in pieces are over every flight. 360 flights, 101 km per
row. Every loss was a colliding pair (no isolations). One drone's longest stall in a straight-line flight: 3.1 s
switches off, 0.5 s slide on; a block passed in 12.3 s instead of 17.1.

By city (cohesive speed, m/s): the slide is ahead of T3 by 20–22% in every one.

| switches on | the scene's city | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| T3 | 4.03 | 4.07 | 4.33 | 3.97 |
| slide | 4.90 | 5.01 | 5.23 | 4.78 |
| route | 3.98 | 4.15 | 4.35 | 3.94 |
| adaptive | 4.13 | 4.19 | 4.40 | 4.17 |
| slide + route | 4.77 | 4.93 | 5.14 | 4.72 |
| slide + adaptive | 5.11 | 5.13 | 5.34 | 5.01 |
| route + adaptive | 4.24 | 4.31 | 4.55 | 4.12 |
| slide + route + adaptive | 5.03 | 5.04 | 5.32 | 4.93 |

## 5. Unity

`bench_flow` (all eight combinations), `bench_tune2` (the gentler adaptive setting) and `bench_single` (one drone):
BlockShapedPatchCityWorld as baked, 64 fresh flights per row (routes from 5000: 30 straight lines, 20 blocks, 14 street
routes; 18.5 km), scored by `bench-gate`. In brackets, the replica's twin flown from Unity's recorded start: it matches
Unity's cohesive speed within 1–3%, and its one group, surrounding and seconds per building within a few percent, as in
S6. Drones lost and contacts are Unity's own counts; every loss was a drone-drone pair (`CollidingWithAnotherDrone` with
its partner), bar one crash under slide + adaptive.

| switches on | cohesive speed (twin) | progress | one group | streets | surrounded | past a building (s) | drones lost | contacts (over 2 m/s) | ended in pieces |
|---|---|---|---|---|---|---|---|---|---|
| none (T3) | 3.94 (3.92) | 4.70 | 83% | 4.22 | 99% | 12.9 | 2 | 8 (6) | 2% |
| slide | 4.83 (4.95) | 5.35 | 90% | 4.89 | 93% | 10.9 | 4 | 3 (1) | 0% |
| route | 3.99 (3.88) | 4.69 | 85% | 4.34 | 98% | 12.9 | 0 | 7 (3) | 0% |
| adaptive (0.8, 1 s) | 4.22 (4.34) | 4.81 | 88% | 5.15 | 93% | 12.9 | 8 | 6 (2) | 2% |
| slide + route | 4.78 (4.87) | 5.35 | 89% | 5.03 | 93% | 11.0 | 4 | 1 (0) | 2% |
| slide + adaptive (0.8, 1 s) | 4.96 (5.07) | 5.40 | 92% | 5.40 | 90% | 10.9 | 3 | 14 (2) | 0% |
| route + adaptive (0.8, 1 s) | 4.22 (4.36) | 4.80 | 88% | 5.12 | 92% | 12.9 | 10 | 4 (3) | 2% |
| all three (adaptive 0.8, 1 s) | 4.87 (5.01) | 5.38 | 90% | 5.38 | 89% | 10.9 | 2 | 4 (3) | 2% |
| adaptive (0.85, 3 s) | 4.07 (4.23) | 4.77 | 85% | 4.89 | 96% | 12.8 | 4 | 10 (0) | 5% |
| slide + adaptive (0.85, 3 s) | 4.99 (5.05) | 5.38 | 93% | 5.36 | 90% | 10.9 | 2 | 3 (2) | 2% |
| one drone, all off | 4.80 (4.87) | 4.80 | – | 6.57 | – | 13.4 | 0 | 0 | – |
| one drone, slide on | 5.62 (5.68) | 5.62 | – | 6.57 | – | 10.2 | 0 | 0 | – |

- **The slide holds in Unity**: 23% faster while together, 63% fewer contacts, no flight ending in pieces; one drone 17%
  faster, a block passed in 12.6 s instead of 18.7. It lost 2 pairs against T3's 1, within the noise of 64 flights (the
  replica's 100 km had the same losses for both).
- **Slide + route leaves one contact in 64 flights** (T3 8), for the same losses as the slide alone.
- **The adaptive spread's cost is clearer in Unity than in the replica**: 8 and 10 drones lost alone and with route. With
  the slide on it lost about as many as T3 (3 and 2), but the replica, over five times the distance, lost 0.12–0.18 per km
  that way against the slide's 0.04; why the slide would help here is untested.
- The 64 flights per row are one Unity run in the scene's city; the losses are counts of 0–10 and differences of one or
  two pairs are chance.

## 6. Tuning the adaptive spread after the confirmation

The confirmation's adaptive spread (0.8 of d_ref, a 1 s filter) failed the rule on losses, and five of its nine
collisions came while the formation was narrowing or widening. `tune2` flew gentler settings on fresh routes (from 6000,
the same four cities, 24 straight lines, 16 blocks and 12 street routes in each: 208 flights, ~60 km per row):

| setting | cohesive speed | streets | one group (streets) | surrounded | lost /km | contacts /km | share of d_ref in force |
|---|---|---|---|---|---|---|---|
| none (T3) | 4.08 | 4.13 | 81% | 98% | 0.07 | 0.35 | 1.00 |
| slide | 5.10 | 4.81 | 88% | 89% | 0.03 | 0.17 | 1.00 |
| slide + adaptive 0.8, filter 1 s | 5.16 | 5.30 | 94% | 88% | 0.10 | 0.30 | 0.90 |
| slide + adaptive 0.8, filter 3 s | 5.17 | 5.36 | 95% | 88% | 0.27 | 0.17 | 0.90 |
| slide + adaptive 0.85, filter 1 s | 5.15 | 5.26 | 94% | 88% | 0.20 | 0.43 | 0.93 |
| slide + adaptive 0.85, filter 3 s | 5.16 | 5.31 | 95% | 88% | 0.10 | 0.17 | 0.93 |
| adaptive 0.85, filter 3 s | 4.19 | 4.73 | 89% | 96% | 0.07 | 0.45 | 0.93 |

All four keep the streets' gain (+10–11% over the slide alone, +15% over T3 without it), and none removes the extra
losses with the slide on (0.10–0.27 per km against 0.03, with no consistent pattern between them: chance around a real
cost). Alone, 0.85 with a 3 s filter lost no more than T3 here, and in Unity (`bench_tune2`) it lost 4 drones alone
against the confirmed setting's 8 (T3 2) and 2 with the slide (3), for a street gain of 16% and 27%. **So it now ships at
0.85 and 3 s**, as the better starting point, and the adaptive spread stays off for the trials: the squeeze in the
narrowed streets is the cause, and a gentler setting only dilutes it.

## 7. What it means

- **Tick Slide Round Buildings** on BlockShapedPatchCityWorld's `SwarmManager` for the trials. It is shared with the
  single-drone condition, which gets as much from it (one drone 17% faster, no longer stalling at buildings), so the
  comparison stays fair: both conditions get quicker. The swarm surrounds slightly fewer of the buildings it flies at (90%
  against 98%), because it gets past them sooner.
- **Route Stragglers is optional**: tick it if building contacts matter (with the slide, one in 64 Unity flights). It costs
  no speed, and its slightly higher losses with the slide in the replica are within chance.
- **Leave Adaptive Spread off** until the avoidance shape is tightened: the circles round each building's bounds narrow
  the streets, and a narrower formation is squeezed together there.
- None of the three changes the spread response in open sky (no building near: the slide and routing have nothing to
  act on, and the adaptive scale is 1), so S6's step responses stand.

### 7.1 The other recommendations

- **Tighter avoidance shape (first set, 3): do next.** The bounding circles bulge into the streets; that squeeze is what
  kills drones whenever the formation narrows (S5, S6, and the adaptive spread here). A per-building radius closer to
  the footprint would widen the streets, let the adaptive spread pay, and probably remove more contacts.
- **Realistic pilots (first set, 5) and a view measure (first set, 6): before the trials.** S7's pilots fly straight
  lines and street routes at full stick; fly the switches on S1's pilot paths or recorded inputs, and score what the
  screens show (S4's look-down), since the slide changes which drones lead round a building.
- **More Unity checks (first set, 7): partly done** (64 flights for each of eleven settings); losses still need a larger
  run, and a headset flight should check the frame-rate log (`_perf.csv`).
- **Look-ahead shield and the stronger damper (second set, 1 and 3): not now.** The slide removes the head-on stall the
  look-ahead was after (its other effect, not braking beside a street, raised street collisions in S6's prototype), and
  the damper did not stop the collisions here.
- **A stretched formation (second set, 2)** is the alternative to the adaptive spread if street cohesion matters; it
  reshapes the lattice's metric, a larger departure from Olfati-Saber.
- **Re-forming after a building (second set, 5)** is confirmed fine and better with the slide (2.6 s in pieces per building
  against 5.0).

## 8. Running it

From `Assets/Scripts/Experiment`, the `stitching` env. The replica's force-law guard refuses to fly while the C# it
mirrors has changed since `swarm_replica.FORCE_LAW_COMMIT` (S7 set it to its own commit, 7bf41e8c);
`SWARM_REPLICA_ALLOW_STALE=1` is its `--allow-stale` for these harnesses, for flying a change before it is committed.

```
python -B -m citysim.swarmflow screen --name screen4 --cand ../../../experiment_results/swarm_flow/screen4_cands.json --cross 48 --pass 32 --street 24
python -B -m citysim.swarmflow confirm --name confirm --cand ../../../experiment_results/swarm_flow/confirm_cands.json
python -B -m citysim.swarmflow confirm --name confirm --cand ../../../experiment_results/swarm_flow/confirm_single_cands.json --single
python -B -m citysim.swarmflow choose --name confirm                 # rule.json applied -> confirm/verdicts.json
python -B -m citysim.swarmflow confirm --name tune2 --cand ../../../experiment_results/swarm_flow/tune2_cands.json --k0 6000 --cross 24 --pass 16 --street 12
python -B -m citysim.swarmflow bench-plan --name bench_flow --cand ../../../experiment_results/swarm_flow/bench_cands.json
python -B -m citysim.swarmflow bench-plan --name bench_single --cand ../../../experiment_results/swarm_flow/bench_single_cands.json --drones 1
#   then run_bench.ps1 as printed
python -B -m citysim.swarmflow bench-gate --name bench_flow --runs D:\claude_obstacle_bench\runs\swarmflow_bench_flow
python -B -m citysim.swarmflow bench-gate --name bench_single --runs D:\claude_obstacle_bench\runs\swarmflow_bench_single
python -B -m citysim.swarmflow_replays          # the page's flights: each twin re-flown with a tape, kept only if identical to the gate
python -B -m citysim.swarmflow_report           # then publish swarm_flow_bare.html with the files swarm_flow_files.json lists
```

A candidates file is S6's: a JSON list of `{"label", "d", "over"}`, `over` holding SwarmManager fields (the switches
included) and `x:` replica-only knobs. The screens' side rules are replica extras: `x:slide_side` (`hidden`, the C#'s and
the default; `auto`, without the hidden-centroid case, which screens 2–5's unmarked slide variants flew; `own`, which
screen 1's flew; `formation`; `group`) and `x:route_side` (`lean`, the C#'s; `group`). Screen 1's route variants flew the
first form (whole cohesion), which is no longer in the code; screen 2's the second, all of the hidden pull, which is
`stragglerPressShare` 0. Screens 2-4 took the slide's split from `surroundAimFraction` (`slideSplitAimFraction` arrived in
screen 5), and the confirmation's and `tune2`'s adaptive candidates relied on the defaults of the time (0.8, 1 s): add
those values to their `over` to re-fly them under today's defaults.
