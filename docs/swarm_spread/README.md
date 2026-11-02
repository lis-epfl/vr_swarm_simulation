# The spread study (S9 Street Spread)

How far S7's adaptive spread should narrow the swarm in **BlockShapedPatchCityWorld**'s streets, and what the same
spreads do held fixed. Asked for on 2026-11-02, after the evaluation of S7 and S8, as a short study (about 70 minutes):
"investigate the spread values; I wonder if the adaptive spread simply narrows too much in the streets and should stay
a bit wider; also investigate the fixed d_ref". No change to the force law or the scene.

Registry entry: [S9_street_spread.md](../studies/S9_street_spread.md). Page: [S9 Street Spread](https://claude.ai/artifact/3tbJxGdDjKzmRVxqEyXmNx).
Code: `citysim/swarmspread.py` (the run, the rule, the analysis, the bench), `swarmspread_replays.py`,
`swarmspread_report.py` (+ `.html`). Outputs: `experiment_results/swarm_spread/`.

**The answer (revised after the follow-up, §8).** The adaptive spread is neither the problem nor the fix: what matters
is how narrow the swarm flies the streets.
- **Narrower is faster, beyond doubt.** Route by route over 1,000 street routes, the adaptive spread at 0.95 / 0.90 / 0.85
  is 2.7% / 5.3% / 8.0% faster in the streets than holding d_ref 2.0, 1.8 held 8.0% and 1.7 held 11.4%, in every city
  (p < 1e-15 each).
- **In the replica, narrower costs drones, held or adaptive alike.** Over 431 km per row: 24 loss events at 2.0 held,
  36 / 35 / 48 at 0.95 / 0.90 / 0.85, 43 at 1.8 held and 48 at 1.7 held (0.85 and 1.7 against 2.0: p = 0.003). Every
  replica run of 0.85 and 2.0 held (S8 and S9): 68 flights with losses against 34.
- **Unity does not reproduce the loss difference.** Two benches, 156 flights a row: 12 flights with losses at 0.85, 9 at
  1.7 held, 10 at 2.0 held. Unity also loses about twice as many as the replica's twins on the same routes (31 flights
  against 18 in those three rows; across every bench row the twin lost drones in only 3 of Unity's 39), while matching
  its speeds and contacts. So whether the
  narrower street spread costs drones in Unity is not settled.
- **The adaptive spread's changes cost nothing; the first run's two readings were wrong.** Held at 1.7–1.8 the swarm
  loses as many drones as with the adaptive spread (first run: "held 1.7–2.0 lose alike" was noise at ~10 events a row).
  And the losses cluster as the swarm leaves a building into a street in *every* row, held or adaptive: 2.0 held,
  whose spread never changes, loses 2.05 an hour in those moments against 0.41 while the shadow is held narrow, and the
  adaptive 0.85's excess over 2.0 held is the same there as elsewhere (×1.96 against ×2.18, p = 0.83). The first
  run's "33 of 63 losses during a change" was the place, not the change.
- **The adaptive spread flies like 1.8 held** (the same street speed, p = 0.68; 48 against 43 loss events); 1.7 held is
  3.1% faster than it for the same losses.
- **For the trials** the choice is the street spread: 2.0 held (adaptive off) for the fewest losses in the replica, about
  8% slower in the streets; the scene as it is (≈ 1.8 held) or 1.7 held for the speed. The first write-up's advice to
  hold 1.7–1.8 "to avoid the narrowing" is withdrawn. A Unity run of 2.0 held against the scene's setting, a few hundred
  street routes each, would settle the losses. The scene is unchanged (C35a).

§4–§7 are the first run, as written before the follow-up; read their loss findings through §8.

## 1. Why

Since S8 the scene flies C35a: the enclosing circle, S7's slide, `d_obs` 0.35 and the adaptive spread at
`streetSpreadFactor` 0.85 (d_ref 2.0 narrowed to 1.7 in the streets, low-passed at 3 s). Pooled over S8's replica and
Unity runs, C35a lost drones in 10 flights or pairs where C35 (the same without the adaptive spread) lost them in 1
(p ≈ 0.01), and across S7 and S8 the adaptive spread roughly doubled the flights that lost drones. S7 had found the
same cost and two candidate causes without separating them: the narrowed spread itself (pairs squeezed in the gaps the
avoidance circles leave) and the narrowing and widening (5 of 9 of its collisions came while the formation changed).
And `OlfatiSaber.EffectiveDRef` is the dial times the scale with no floor, so the dial's lowest setting in the trials
(1.4, `-SpreadRange 1.4,2.6`) becomes 1.19 in the streets, which no study had flown.

## 2. What was flown

The scene's force law in every row (enclosing circle, slide, `d_obs` 0.35, route stragglers off, the filter 3 s), and
only the spread changed (`run_cands.json`):

| code | d_ref (the dial) | adaptive spread | spread in the streets | asks |
|---|---|---|---|---|
| F20 | 2.0 | off | 2.0 | the base (S8's C35) |
| A85 | 2.0 | 0.85 | 1.7 | the scene (S8's C35a) |
| A90, A95 | 2.0 | 0.90, 0.95 | 1.8, 1.9 | does it narrow too much? |
| F19, F18, F17 | 1.9, 1.8, 1.7 | off | the same | the narrowed spread held fixed: the spread, or the narrowing? |
| F14, L85, L95 | 1.4 | off, 0.85, 0.95 | 1.4, 1.19, 1.33 | the dial at its lowest |
| F12 | 1.2 | off | 1.2 | what L85 narrows to, held |
| F26, H85 | 2.6 | off, 0.85 | 2.6, 2.21 | the dial at its highest |

- **The replica** (`run1`): S8's harness unchanged (`swarmtune` / `swarmflow`: ContactSim on the prism cities, settled
  starts, the bench's waypoint pilot at full stick, S6's measures), routes from 10000 (S6–S8 used 0–9999), in the scene's
  city and S6's three held-out ones, **12 straight lines, 12 blocks and 60 street routes in each** (336 flights, ~120 km
  per row, 96 km of it in the streets). The street routes are weighted five to one because the adaptive spread acts
  there and that is where S8's losses were (6 of C35a's 7). Every row flies the same routes.
- **Each loss is placed against the spread** (`swarmspread.SpreadEventLog`): the spread's share of d_ref at the loss and
  3 s before it. A loss with a change of 0.02 or more over those 3 s happened while narrowing or widening, otherwise in
  a steady spread; the share of flight time that was so is logged beside it.
- **Unity** (`bench_spread`): BlockShapedPatchCityWorld as baked, six of the rows (F20, A85, A90, A95, F17, L85), 56
  fresh flights each from 11000 (8 straight lines, 8 blocks, 40 street routes), scored by `bench-gate` beside replica
  twins flown from Unity's recorded starts. Flown at the same time as the replica, on the same machine.
- **Counted as events, not drones**: a loss event is a moment drones are lost, nearly always a colliding pair, so a
  pair is one event. Each count gets its exact 95% Poisson interval, and a comparison between two rows on the same
  routes is the exact conditional binomial test on their two counts (one-sided). The S7/S8 evaluation found that losses
  fall on different routes in every row, so pairing the routes buys no power for losses, and the counts are treated as
  independent. A factor's losses are reported beside S8's for the two settings both studies flew (F20 = C35,
  A85 = C35a), which are independent replicates on different routes.

## 3. The rule

Fixed before the run (`rule.json`, written 2026-11-02 10:03): against F20 (the adaptive spread off), a street factor
(A85, A90, A95) is **safe** when its loss events are not more frequent than F20's at one-sided exact p < 0.10 and its
contacts are at most 1.25 × F20's; it **helps** when the streets' cohesive speed is at least 3% above F20's and it
surrounds within 5 points of F20's share of buildings. Pick: the narrowest safe and helping factor (the most street
speed); none → leave the adaptive spread off. The fixed spreads and the dial ends are descriptive: they are there to say
why, not to be picked.

## 4. The replica

`run1`: 4368 flights in 23.5 min, no errors, ~117 km per row (~96 km in the streets), the same routes for every row.
Streets and street one group over the street routes; cohesive speed and surrounded over the straight lines and blocks;
losses and contacts over every flight; p is the one-sided exact test against F20.

| code | streets (m/s) | street one group | cohesive speed | surrounded | contacts /km | loss events (95%) | drones lost | p vs F20 |
|---|---|---|---|---|---|---|---|---|
| F20 2.0 held | 5.39 | 91% | 5.48 | 93% | 0.83 | 9 (4.1–17.1) | 18 | – |
| A95 | 5.53 | 92% | 5.58 | 90% | 0.67 | 6 (2.2–13.1) | 12 | 0.85 |
| A90 | 5.63 | 93% | 5.62 | 90% | 0.81 | 10 (4.8–18.4) | 20 | 0.50 |
| **A85 (the scene, the rule's pick)** | 5.84 | 96% | 5.63 | 90% | 0.60 | 14 (7.7–23.5) | 28 | 0.20 |
| F19 1.9 held | 5.59 | 93% | 5.60 | 92% | 0.77 | 13 (6.9–22.2) | 26 | 0.26 |
| F18 1.8 held | 5.77 | 95% | 5.65 | 90% | 0.57 | 6 (2.2–13.1) | 12 | 0.85 |
| F17 1.7 held | 5.95 | 96% | 5.65 | 92% | 0.35 | 12 (6.2–21.0) | 24 | 0.33 |
| F14 1.4 held | 6.37 | 99% | 5.78 | 81% | 0.21 | 17 (9.9–27.2) | 34 | 0.08 |
| L85 1.4, adaptive 0.85 | 6.47 | 100% | 5.82 | 81% | 0.09 | 14 (7.7–23.5) | 28 | 0.20 |
| L95 1.4, adaptive 0.95 | 6.39 | 99% | 5.81 | 81% | 0.21 | 9 (4.1–17.1) | 18 | 0.59 |
| F12 1.2 held | 6.53 | 100% | 5.76 | 79% | 0.07 | 11 (5.5–19.7) | 22 | 0.41 |
| F26 2.6 held | 3.64 | 64% | 4.84 | 97% | 0.90 | 2 (0.2–7.2) | 4 | 0.99 |
| H85 2.6, adaptive 0.85 | 4.44 | 78% | 5.18 | 96% | 0.82 | 10 (4.8–18.4) | 20 | 0.50 |

- **The rule** (`run1/analysis.json`): A85, A90 and A95 are all safe (p 0.20, 0.50, 0.85); A85 and A90 help (8% and 4%
  faster in the streets), A95 does not (3%, under the bar). The pick is the narrowest, **A85**.
- **By city** the street speed orders the same way in all four: A85 above F20 and A95 in every one, and **F17 above A85 in
  every one** (5.83 / 6.03 / 6.08 / 5.84 against 5.70 / 5.95 / 5.94 / 5.75).
- **Where the adaptive spread's losses fell** (a change = the spread moved 0.02 or more in the 3 s before):

  | row | loss events | while narrowing | while widening | steady | time spent changing |
  |---|---|---|---|---|---|
  | A85 | 14 | 8 | 2 | 4 | 47% |
  | A90 | 10 | 4 | 2 | 4 | 34% |
  | A95 | 6 | 0 | 0 | 6 | 9% |
  | L85 | 14 | 7 | 2 | 5 | 46% |
  | L95 | 9 | 2 | 0 | 7 | 8% |
  | H85 | 10 | 5 | 1 | 4 | 49% |

  33 of 63 in a change against 22.6 expected from the time shares (z = 2.9, one-sided p = 0.002), 26 of them while
  narrowing: the formation pulling in as the swarm leaves a building, which S7 saw in five of its nine. A change only
  happens near buildings, so part of this is where rather than what; the fixed rows fly the same places at a steady
  spread and lose no more at 1.7 than at 2.0.
- **Adaptive against the same narrowed spread held** (A85/F17, A90/F18, A95/F19, L85/F12, H85/F26): 54 events against
  44 (p = 0.18); only H85 against F26 is clear on its own (10 against 2, p = 0.02), where the narrowing is largest.
- **Surrounding** barely moves down to 1.7 (93, 92, 90, 92%) and falls at 1.4 (81%) and 1.2 (79%); building contacts fall
  steadily as the formation narrows (0.83 at 2.0 to 0.07 at 1.2). Seconds past a building (10.2–10.4) and flights
  ending in pieces (0–2%) do not change.

## 5. Unity

`bench_spread`: BlockShapedPatchCityWorld as baked, 56 flights per row (19.2 km; 8 straight lines, 8 blocks, 40 street
routes from 11000), the twins from Unity's recorded starts within 3% on street speed. Every loss was a colliding pair.

| row | streets (twin) | cohesive speed | drones lost | contacts |
|---|---|---|---|---|
| F20 2.0 held | 5.36 (5.32) | 5.41 | 2 | 19 |
| A95 | 5.44 (5.46) | 5.46 | 2 | 15 |
| A90 | 5.42 (5.42) | 5.39 | 6 | 15 |
| A85 (the scene) | 5.54 (5.58) | 5.53 | 8 | 7 |
| F17 1.7 held | 5.70 (5.84) | 5.41 | 2 | 10 |
| L85 1.4, adaptive 0.85 | 6.43 (6.51) | 5.63 | 8 | 4 |

The replica's pattern: the narrower factors lose the most pairs (4 at 0.85, 3 at 0.90, 1 at 0.95), and holding 1.7
loses no more than holding 2.0 while being the fastest in the streets at the dial's 2.0. A pair or two either way is
chance at 56 flights.

## 6. Pooled with S8

S8 flew the same two settings on other routes (C35 = F20, C35a = A85). Treating every run as an independent
replicate, flights that lost drones in the replica and colliding pairs in Unity:

| | S8 replica | S9 replica | S8 Unity | S9 Unity | all |
|---|---|---|---|---|---|
| A85 (adaptive 0.85) | 7 | 14 | 3 | 4 | **28** |
| F20 (2.0 held) | 1 | 9 | 0 | 1 | **11** |

28 against 11, one-sided exact p = 0.005: the adaptive spread at 0.85 costs about twice the collisions of holding 2.0,
which no single run showed.

## 7. What it meant after the first run (superseded by §8)

- **The adaptive spread does narrow more than it needs to, and the cure is not a wider factor.** 0.90 and 0.95 lose
  fewer drones only by giving back the street speed. Holding 1.7–1.8 has the speed and none of the narrowing, and the
  widening at a building, which the adaptive spread exists for, bought no surrounding (A85 90%, F17 92%).
- **Recommendation for the trials:** adaptive spread off, d_ref held about 1.7–1.8. The pilot's dial sets d_ref every
  frame, so centre the dial there (`launch_sim_scripts.ps1 -SpreadRange 1.2,2.4` keeps its sensitivity) and set the
  scene's `d_ref` for the keyboard; consider a narrower range whose low end stops nearer 1.5, since 1.4 held lost the
  most of any row. Not applied: the scene still flies C35a.
- **A floor on `EffectiveDRef` is not needed** on these counts: the dial's low end loses as many drones without the
  adaptive spread as with it.
- **Not settled:** people (the dial is a pilot's, and pilots turn it), and small differences between single rows (about
  ten loss events each); the pooled counts and the trends are the evidence.

## 8. The follow-up: is any of it significant, and when do the losses happen?

Asked the same day: "Is there really any significance in the findings? Conduct more tests, and do temporal analysis to
see if the crashes really occur during the change of d_ref. Visually I don't see much difference in the examples."

**What was flown.**
- **`run2`** (replica): the six rows that matter (F20, A85, A90, A95, F18, F17), 1,100 fresh flights each from route
  12000 (250 street routes and 25 straight lines in each of the four cities; 431 km per row, four times run1's street
  distance), 41.6 min, no errors. Every flight carries **`swarmspread.ShadowLog`**: the adaptive spread's *shadow*, the
  target `swarm_replica.spread_target` would set from the swarm's own centroid, stick and buildings, low-passed at 3 s
  with the factor 0.85, whatever the row flies. In an adaptive row it equals the spread (checked exactly on A85); in a
  held row it marks the same moments while the spread stays put. Every moment and every loss is placed in a phase of
  the shadow (`sh_phase`: narrowing / widening = moved 0.02 or more over the 3 s before, else held narrow / held wide),
  with the seconds spent in each phase, and also by the seconds since the target last switched; each loss carries the
  spread and shadow from 15 s before to 3 s after, beside three random moments of the same flight.
- **`bench_spread2`** (Unity): F20, A85, F17, 100 fresh street routes each from 13000, gated with twins.
- **`swarmspread_temporal`**: loss rates per phase (events per hour in that phase, exact intervals), the exact
  conditional test per phase, and the likelihood-ratio test that one rate ratio holds in the narrowing moments and
  elsewhere; for Unity, the shadow rebuilt from the trajectories (approximate).

**Street speed: significant at every step** (route-by-route paired differences; `paired` in `swarmspread_report`):

| comparison | first run (240 routes) | follow-up (1,000 routes) | faster on |
|---|---|---|---|
| A85 against F20 | +8.2% (6.7 to 9.7) | +8.0% (7.2 to 8.7) | 85% |
| A90 against F20 | +4.4% | +5.3% (4.6 to 6.0) | 78% |
| A95 against F20 | +2.5% | +2.7% (2.0 to 3.3) | 68% |
| F18 against F20 | +7.0% | +8.0% (7.3 to 8.7) | 86% |
| F17 against F20 | +10.2% | +11.4% (10.6 to 12.2) | 91% |
| F17 against A85 | +1.9% | +3.1% (2.6 to 3.7) | 80% |
| F18 against A85 | −1.1% (p = 0.13) | +0.0% (−0.5 to 0.6; p = 0.68) | 50% |

Every follow-up difference but the last is p < 1e-15 over routes and p < 0.005 over the four cities' means.

**Losses: in the replica, the narrower street spread doubles them, held or adaptive; Unity does not show it.**

| row | run1 events (117 km) | run2 events (431 km) | run2 p vs F20 | Unity flights with losses (both benches) | the twins |
|---|---|---|---|---|---|
| F20 2.0 held | 9 | 24 | – | 10 of 156 | 6 |
| F18 1.8 held | 6 | 43 | 0.014 | – | – |
| F17 1.7 held | 12 | 48 | 0.003 | 9 of 156 | 8 |
| A95 | 6 | 36 | 0.078 | 1 of 56 | 3 |
| A90 | 10 | 35 | 0.096 | 3 of 56 | 2 |
| A85 (the scene) | 14 | 48 | 0.003 | 12 of 156 | 4 |

- The first run's "held 1.7–2.0 lose alike" was noise: ~10 events a row cannot show a doubling.
- The rule fixed before the first run, applied unchanged to run2, finds **no factor safe** (A85 and A90 help but are not
  safe; A95 is neither), i.e. the adaptive spread off.
- All replica runs of A85 and F20 (S8 confirm, run1, run2): 68 flights with losses against 34. Unity, S8's bench and
  both of S9's: 15 against 10.
- Over every row of both benches Unity lost drones in 39 flights and the twins in 25 (F20, A85 and F17 alone: 31
  against 18), and the two agree on only 3 flights: the replica matches Unity's flow and contacts (street speed within
  3%, contacts within 15%) and under-counts its collisions, which do not repeat from one start to its twin.

**When the losses happen: in the place, not in the change** (`run2/temporal.json`):

| row | losses / hour narrowing | widening | held narrow | held wide | all |
|---|---|---|---|---|---|
| F20 2.0 held | 2.05 (12) | 0.76 (4) | 0.40 (4) | 1.69 (4) | 1.03 (24) |
| F18 1.8 held | 3.93 (22) | 1.18 (6) | 1.33 (13) | 0.85 (2) | 1.88 (43) |
| F17 1.7 held | 4.00 (22) | 2.20 (11) | 1.13 (11) | 1.74 (4) | 2.13 (48) |
| A95 | 3.40 (19) | 0.72 (4) | 1.10 (11) | 0.99 (2) | 1.55 (36) |
| A90 | 3.45 (19) | 1.63 (9) | 0.70 (7) | 0 (0) | 1.52 (35) |
| A85 | 4.04 (22) | 1.64 (9) | 1.62 (16) | 0.50 (1) | 2.10 (48) |

The rows spend the same share of time in each phase (about 24% narrowing, 23% widening, 43% held narrow, 10% held
wide), because the shadow is the geometry's.
- **2.0 held, whose spread never changes, loses most in the "narrowing" moments**: 2.05 an hour against 0.40 held
  narrow. Those moments are the swarm leaving a building's side into a street.
- **The adaptive spread's excess is the same in every phase**: A85 against F20 ×1.96 narrowing and ×2.18 elsewhere
  (equal-ratio test p = 0.83); pooled adaptive rows against held rows ×1.10 and ×0.99 (p = 0.69), ×1.03 overall
  (p = 0.43). Held 1.7 against A85: ×1.01 in the narrowing moments.
- **By seconds since the swarm entered a street**, the first 4 s hold 50% of F20's losses and 65% of both F17's and
  A85's, against 35% of the time.
- **Round a loss** the shadow rises to about 0.93 two seconds before it and falls (F20 0.93, F18 0.92, F17 0.93, A85
  0.92), against a flat 0.89–0.90 round random moments: a building on the track, then the turn into a street, in every
  row alike. In the adaptive rows the real spread follows the shadow, which is why the first run read the change as
  the cause.
- Unity, rebuilt: losses while narrowing F20 3 of 10, F17 3 of 9, A85 6 of 12, against ~25% of the time. Too few.

**Why the replays look alike.** Per flight the differences are a few percent of speed (0.43 m/s A85 against F20, the ring
31 m across at 2.0 and 26 m at 1.7), and losses are rare: 2% of F20's and 4% of A85's replica flights, about one Unity
flight in 16. Four replayed routes cannot show them; it takes hundreds of flights.

**What it means now.**
- The choice is the spread in the streets, not adaptive against held. The adaptive 0.85 ≈ 1.8 held in speed and losses;
  its changes add no losses and its widening at buildings buys no surrounding (89–91% in every row).
- If the replica's loss doubling is real, 2.0 held (adaptive off) is the safe setting, ~8% slower in the streets. If
  Unity is right that it is not, the scene's setting or 1.7 held (3% faster than it) costs nothing.
- **To settle it**: Unity, 2.0 held against A85 (or 1.7 held), ~300 street routes each (~35 min alone on the bench).
- Lessons 45 (rewritten) and 47.

## Running it

From `Assets/Scripts/Experiment`, the `stitching` env.

```
python -B -m citysim.swarmspread rule                          # rule.json, before the run
python -B -m citysim.swarmspread run --name run1 --cand ../../../experiment_results/swarm_spread/run_cands.json
python -B -m citysim.swarmspread analyse --name run1           # -> run1/analysis.json
python -B -m citysim.swarmspread bench-plan --name bench_spread --cand ../../../experiment_results/swarm_spread/bench_cands.json
#   then run_bench.ps1 as printed, with -Tag swarmspread_bench_spread
python -B -m citysim.swarmspread bench-gate --name bench_spread --runs D:\claude_obstacle_bench\runs\swarmspread_bench_spread
# the follow-up: four times the street distance, every loss timed against the adaptive spread's shadow
python -B -m citysim.swarmspread run --name run2 --cand ../../../experiment_results/swarm_spread/run2_cands.json --cross 25 --pass 0 --street 250 --k0 12000 --shadow
python -B -m citysim.swarmspread_temporal replica --name run2  # -> run2/temporal.json
python -B -m citysim.swarmspread bench-plan --name bench_spread2 --cand ../../../experiment_results/swarm_spread/bench2_cands.json --cross 0 --pass 0 --street 100 --k0 13000
python -B -m citysim.swarmspread bench-gate --name bench_spread2 --runs D:\claude_obstacle_bench\runs\swarmspread_bench_spread2
python -B -m citysim.swarmspread_temporal unity --runs swarmspread_bench_spread swarmspread_bench_spread2
python -B -m citysim.swarmspread_replays                       # the page's flights, each twin identical to the gate
python -B -m citysim.swarmspread_report                        # then publish swarm_spread_bare.html with swarm_spread_files.json
```

A candidates file is S6's: a JSON list of `{"label", "d", "over"}`; the label's first word is the row's code. Every row
names the law's fields explicitly (`obstacleShape`, `d_obs`, the slide, route stragglers, the filter), so a later change
to the scene's SwarmManager does not change what the file flies.
