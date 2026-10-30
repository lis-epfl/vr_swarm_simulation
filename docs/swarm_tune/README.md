# The patch-city tuning (S6 Patch City Tuning)

Retuning the Olfati-Saber swarm for **BlockShapedPatchCityWorld**: the spread (d_ref) that suits its buildings, and the
obstacle-avoidance and lattice gains that let the formation flow round a building and close up again beyond it, quickly
and without breaking into pieces. d_ref is held constant through every flight, as if the pilot leaves the dial alone.

Registry entry: [S6_patch_city_tuning.md](../studies/S6_patch_city_tuning.md). Page: [S6 Patch City Tuning](https://claude.ai/artifact/EXW3359CjhLNfHg6SpaBvF).
Code: `citysim/swarmtune.py` (the page: `swarmtune_replays.py`, `swarmtune_report.py` + `.html`). Outputs:
`experiment_results/swarm_tune/`. Run 2026-10-30.

**The answer.** d_ref 2.0, `d_shield` 1.6 (was 2.5), `d_obs` 0.16 (was 0.4), every other gain as the scene had it;
written into the scene's SwarmManager. In Unity, against the scene's gains at the same spread: 27% more route per second
with the swarm together, 26% faster progress, the same losses, 98% of the buildings met surrounded, at the cost of two and a
half times the (harmless, so far) building contacts. Against the scene as it was (d_ref 1.08), it stops the drone-drone
collisions (32 drones lost in 64 flights, then 2) and surrounds 98% of buildings instead of 67%.

## 1. The city

BlockShapedPatchCityWorld flies `layouts/BlockShaped__bbB_p40_s18_patch150_j0.15_sight487_seed1.json`: 289 convex prism
buildings, each filling its Voronoi block to the back of a 3.5 m footpath, block seeds 40 m apart in hexagonal patches
turned at random, 18 m streets. What matters for the swarm:

| | |
|---|---|
| a building's bounds | about 21.5 m × 21.7 m (median) |
| its avoidance cylinder (the circle round its bounds, which OlfatiSaber flies round) | radius 15.4 m median (12.4–19.8 m, 10th–90th percentile), so 31 m across |
| facade to facade across a street | 25 m (18 m street + two footpaths) |
| streets the cylinders close / narrow below 6 m | 3 / 72 of 794 |

The formation's own size, at hover in open sky (the hollow core's ring, ten drones):

| d_ref | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 | **2.0** | 2.2 | 2.6 |
|---|---|---|---|---|---|---|---|---|
| ring diameter (m) | 14 | 17 | 20 | 24 | 28 | **31** | 35 | 43 |
| nearest-neighbour spacing (m) | 4.4 | 5.2 | 6.3 | 7.5 | 8.8 | 10.1 | 11.4 | 14.0 |

**That is the scale argument for 2.0**: its ring is as wide as a typical building's avoidance cylinder, so the formation
can close round a building without stretching its lattice, and it is still narrower than the 35–43 m of the wider spreads
that no longer fit the 25 m between facades.

The same design at seeds 2–4 (`blockcity one --seeds patch --sight-tries 487`, about 25 minutes each; 287–298
buildings, 5.0–5.1 street sides per block, median view 245–251 m) is held out: the screens fly only seed 1, the
confirmation all four.

## 2. How it is flown

The replica (`swarm_replica.ContactSim` on a `PolyCity`) under the default obstacle model, which is what Unity flies in
this scene: every prism avoided through the circle round its axis-aligned bounds, contacts on the true prism as PhysX
resolves them (friction 0.6), pairs closer than 0.5 m lost, a drone with no live neighbour within the cohesion cutoff
lost. That is S5's model. The gains are the scene's SwarmManager (`swarm_params.resolve_swarm`) plus each candidate's
overrides.

**Every flight starts settled**, as the bench's do: the hover formation shrunk about the start junction until no drone
is within 2 m of a building, then 15 s at zero stick, and only then do the clock, the contacts and the losses start.
The first screens skipped this (section 5.1), which biased them.

The pilot is the bench's waypoint pilot at full stick, a waypoint taken when the live drones' centroid comes within 10 m,
then 10 s at zero stick once the last is taken. Three flights, on the same routes for every candidate (common random
numbers):

| flight | what it is |
|---|---|
| **cross** | a straight line between two interior junctions 250–420 m apart. The formation meets building after building head on or glancing, and has to flow round each and close up again. The primary flight. |
| **pass** | S4's pass round: along a street to a block's corner, aimed through the block at a point beyond its opposite corner, on to that street's far junction. One building, met on purpose. |
| **street** | the shortest street route between a cross flight's ends, junction by junction: keeping to the streets, which a retune must not make worse. |

## 3. The measures

Every measure is per flight; a candidate's numbers are means over its flights (rates per km over the distance flown).

| measure | definition |
|---|---|
| **progress speed** `v_prog` | the route's length over the time taken to fly it (m/s). An unfinished flight counts the route it covered by the cap. |
| **one group** `one_frac` | share of the flight the live drones were one group: two drones are linked when the free-space path between them, round the buildings, is at most 30 m (streetnet's cluster metric, S4's). A single drone 31 m away counts as a fragment. |
| **cohesive speed** `v_coh` | `v_prog × one_frac`: route covered per second with the swarm together. **The headline.** |
| **fragment episodes** | each run of samples that are not one group: per km, and their median, 90th-percentile and longest durations. An episode's duration is the time to re-form. |
| **ended split** `open_end` | still not one group 10 s after the last waypoint, hovering there (`reform_end`: how long it took). |
| **surrounded** `wrapped` | of the buildings the pilot aimed at (the route passes within 0.6 of the cylinder radius of the centre), the share whose centre was inside the drones' convex hull at some sample while the centroid was within R + 15 m of it. |
| **coverage** | 360° minus the widest gap between the drones' bearings from a met building's centre, at the best sample. |
| **per building** | `bld_pass_s`: seconds the centroid spent within R + 15 m of it (getting past it); `bld_frag_s`: seconds not one group from arriving there to 15 s after leaving; `bld_reform_s`: from leaving to one group again. |
| **losses, contacts** | drones lost (drone-drone pairs, isolation) and building-contact events, per km; the closest pair. |
| **stall** | the longest stretch the centroid moved under 1 m/s. |
| **hull share** | mean share of live drones on the convex hull: the boundary drones whose feeds `OUTER_CIRCLE` shows, which the hollow core exists to keep high. |
| **spread steps** | at hover in open sky: d_ref → 1.3 d_ref (expand), back (contract), → 0.75 d_ref; time to 50% and 90% of the change, overshoot (the standing requirement that a retune keeps the spread response quick and stable). |

## 4. The rule (fixed before the sweep's results were read)

Recorded in `experiment_results/swarm_tune/rule.json`. Applied to the confirmation run (fresh routes in the scene's city
and the three held-out cities), not to the screens:

- **Safe**: at most 0.05 drones lost and 0.1 building contacts per km, at most 5% of flights ending fragmented, at least
  half the drones on the hull, over cross, pass and street flights together.
- **Pick**: among safe candidates, the highest cohesive speed over cross and pass flights.
- **None safe**: the highest penalised score, `v_coh − 2.0 × lost/km − 0.5 × contacts/km − 4.0 × open-end share` (also
  what the screens rank by).

## 5. What the screens found

### 5.1 The first screens were biased, and are kept only as a record

`sweep1`, `search1` (stopped at 40 of 400 random candidates) and `grid1` started each flight from the open-sky formation
placed straight on the start junction. Drones overlapping a building were pushed out to its far side, and cohesion
dragged them back through the facade in the first seconds: logging every contact showed most at 2–4 s, with cohesion the
largest push into the facade and the nearest neighbour 17–35 m away. That was about 70% of the contacts at d_ref 1.8
(0.87 → 0.25 per km once fixed), more the bigger the formation, so those screens counted against wide spreads and for any
gain that shrinks the formation. The core's size, which looked like a lever at 2.2, was entirely this. The paired check
against Unity (section 6) was not affected, since its twins start from Unity's own settled positions. Everything below is
flown from a settled start (`baseline2`, `grid2`–`grid4`, `sweep2`, `confirm`). Lesson 39.

### 5.2 The scene's gains, at every spread (`baseline2`, 52 flights each, the scene's city)

| d_ref | lost /km | contacts /km | one group | surrounded | cohesive speed | progress speed |
|---|---|---|---|---|---|---|
| 1.0 | 1.79 | 0.13 | 0.93 | 51% | 3.74 | 3.98 |
| 1.2 | 1.02 | 0.38 | 0.91 | 62% | 3.64 | 3.95 |
| 1.4 | 0 | 0.13 | 0.89 | 71% | 3.53 | 3.92 |
| 1.6 | 0 | 0.13 | 0.91 | 83% | 3.54 | 3.88 |
| 1.8 | 0.13 | 0.06 | 0.89 | 94% | 3.42 | 3.85 |
| 2.0 | 0 | 0.06 | 0.84 | 95% | 3.23 | 3.81 |
| 2.2 | 0 | 0.19 | 0.83 | 98% | 3.11 | 3.74 |
| 2.6 | 0 | 0.06 | 0.75 | 99% | 2.79 | 3.72 |
| 3.0 | 0.13 | 0 | 0.61 | 99% | 2.26 | 3.73 |
| one drone | 0 | 0 | — | — | 4.01 | 4.01 |

- **Below 1.4 the formation is squeezed between the avoidance circles and drones collide** (S5's finding).
- **The wider the spread, the more buildings it surrounds**, and the more of the flight it spends in pieces.
- **Progress speed is stuck near 3.8 m/s at every spread, and one drone is no faster (4.0, 43% of its top speed).** Flown
  straight at a building, a drone stalls against it: the shield removes the part of the stick pointing into the building
  and leaves nothing to slide it round (one drone's longest stall averages 8.9 s a flight). The swarm gets past a building slightly
  quicker than one drone (16.5 s against 17.8 s) because cohesion drags the stuck drones round.

### 5.3 What each gain does (`sweep2`: one at a time at d_ref 1.8, ×0.5/0.7/1.4/2, 52 flights each; and the grids)

- **`d_shield` (the range over which the stick into a building is removed) is the speed lever.** 1.6–1.8 instead of 2.5
  adds 0.6–0.7 m/s of progress: the drones brake later and get round sooner. Below about 1.6 drones arrive too fast and
  collide with each other at the building (×0.5 lost 2.2 drones per km; 1.4 lost 0.2–0.5 per km in `grid3`).
- **`d_obs` (how far outside the cylinder the repulsion starts) is the cohesion lever among the obstacle gains.** A
  thinner field (0.16–0.2 instead of 0.4) lets the two halves of the formation stay closer round a building and in the
  streets the cylinders narrow: +0.2–0.4 m/s of cohesive speed, fewer losses, slightly more contacts. A thicker one
  (×2) costs speed and drones.
- **`a` (cohesion strength) trades fragmentation against contacts, as in ScaledCityWorld.** Every remaining contact is a
  straggler on the far side of a building being pulled through its facade by the rest of the swarm (cohesion the
  largest push into the facade in every logged contact, the obstacle field saturated against it). 0.65 removes them
  (≈0.1 per km) and costs 3–7 points of one-group share; 1.0 adds 1–2 per km. 0.8 (the scene's) gives the highest
  cohesive speed. `r0_coh` (cohesion's reach) acts the same way.
- **No measurable effect at this spread**: the close-range damper, the core's gains, size and standoff (once the start
  was fixed), `c2_beta`, `c_obs` (14 or 10), `delta` (0.08 or 0.05). Velocity consensus (`c_vm`) adds contacts steeply,
  as the ScaledCityWorld notes say.

Main effects in `grid3` (d_ref 1.6–2.2 × `d_shield` 1.4/1.6/1.8 × `d_obs` 0.12/0.16/0.2 × `a` 0.65/0.72/0.8) and
`grid2` (d_ref 1.4–2.2 × `d_shield` 1.75/2.1/2.5 × `d_obs` 0.2/0.3/0.4 × core 0.3/0.5 × `a` 0.65/0.8) are in their
run folders (`python -B -m citysim.swarmtune table --name grid3`).

## 6. Does the replica fly this city as Unity does?

`bench_base` flew the scene's gains at d_ref 1.2, 1.6 and 2.0 in Unity (BlockShapedPatchCityWorld as baked, 26 flights
each); `bench-gate` scored Unity's trajectories with the same measures beside replica twins flown from Unity's recorded
(settled) starts:

| d_ref | | cohesive speed | progress speed | one group | surrounded | hull | lost /km | contacts /km |
|---|---|---|---|---|---|---|---|---|
| 1.2 | Unity | 3.63 | 3.93 | 0.91 | 70% | 0.65 | 1.06 | 0.53 |
| | replica | 3.64 | 3.97 | 0.91 | 70% | 0.66 | 1.86 | 0.27 |
| 1.6 | Unity | 3.48 | 3.87 | 0.90 | 85% | 0.60 | 0.27 | 0.40 |
| | replica | 3.53 | 3.91 | 0.90 | 90% | 0.62 | 1.06 | 0.13 |
| 2.0 | Unity | 3.21 | 3.80 | 0.84 | 95% | 0.57 | 0 | 0.27 |
| | replica | 3.11 | 3.85 | 0.80 | 95% | 0.57 | 0 | 0.40 |

Speed, cohesion, surrounding and the hull agree within a few percent. The replica lost more drones than Unity at the
tighter spreads (22 against 10), so it is pessimistic about collisions here; contacts are of the same order.

## 7. The confirmation

Fresh routes (from route 1000) in the scene's city and the three held-out cities: 40 cross, 30 pass and 20 street
flights in each, about 104 km per candidate. Five tuned candidates from the grids, the scene's gains at five spreads, and
one drone under each set of gains (`confirm`, rule applied by `confirm` itself):

| candidate | d_ref | cohesive speed | progress speed | one group | lost /km | contacts /km | ended split | surrounded | per building: pass / split (s) |
|---|---|---|---|---|---|---|---|---|---|
| scene gains (the scene's d_ref) | 1.08 | 3.74 | 3.98 | 0.94 | 1.86 | 0.14 | 2.8% | 52% | 16.0 / 2.9 |
| scene gains (S5's) | 1.2 | 3.68 | 3.96 | 0.93 | 0.86 | 0.29 | 1.9% | 63% | 15.9 / 3.5 |
| scene gains | 1.6 | 3.55 | 3.90 | 0.91 | 0.17 | 0.21 | 3.9% | 84% | 15.8 / 3.9 |
| scene gains | 1.8 | 3.46 | 3.86 | 0.90 | 0.15 | 0.12 | 3.9% | 91% | 15.6 / 4.4 |
| scene gains | 2.0 | 3.34 | 3.81 | 0.87 | 0.06 | 0.15 | 4.4% | 94% | 15.6 / 5.0 |
| T1: `d_shield` 1.8, `d_obs` 0.16, `a` 0.65 | 1.8 | 3.94 | 4.64 | 0.85 | 0.13 | 0.12 | 6.1% | 93% | 12.9 / 5.9 |
| T2: `d_shield` 1.8, `d_obs` 0.2, `a` 0.72 | 1.8 | 3.93 | 4.62 | 0.85 | 0.07 | 0.12 | 6.1% | 93% | 13.1 / 5.4 |
| **T3: `d_shield` 1.6, `d_obs` 0.16** | **2.0** | **4.03** | **4.77** | 0.84 | **0.02** | 0.37 | 3.6% | **96%** | **12.5** / 5.1 |
| T4: `d_shield` 1.6, `d_obs` 0.16, `a` 0.72 | 2.0 | 3.88 | 4.75 | 0.81 | 0.02 | 0.105 | 8.9% | 96% | 12.4 / 5.5 |
| T5: `d_shield` 1.8, `d_obs` 0.16, `a` 0.72 | 1.6 | 4.14 | 4.71 | 0.88 | 0.29 | 0.15 | 4.4% | 87% | 13.1 / 5.1 |
| one drone, scene gains | — | 4.03 | 4.03 | — | 0 | 0 | — | — | 17.3 / — |
| one drone, T3's gains | — | 4.96 | 4.96 | — | 0 | 0 | — | — | 12.8 / — |

- **No candidate met every bar** (contacts above 0.1 per km in all; T4 came closest, 0.105, but ended 8.9% of its flights
  in pieces), so **the rule's fallback picked T3**, the highest penalised score (3.66; T2 3.49, T4 3.44).
- T3 against the scene's gains at the same spread: cohesive speed +21%, progress +25%, a building passed in 12.5 s instead
  of 15.6, losses 0.02 per km instead of 0.06, contacts 0.37 instead of 0.15. **It beat them in every one of the four
  cities** (3.89–4.15 m/s against 3.26–3.48), and lost 2 drones in 104 km where the scene as it was lost 194.
- **The obstacle gains are shared with the single-drone condition, and help it more**: one drone goes from 4.03 to 4.96
  m/s, its longest stall in a flight from 10.3 s to 2.9 s. The trials' two conditions both get faster.
- **Spread steps** (`steps.csv`, six seeds): T3 keeps `a`, and the shield and standoff do nothing in open sky, so its
  response is the scene's at the same spread: 90% of an expansion in 5.3 s, of a contraction in 6.4 s, overshoot ≤ 1.2%.
  It is slower than at 1.08 (3.5 / 3.9 s) only because the formation is twice as wide. The `a` 0.65–0.72 candidates are
  15–20% slower.

## 8. Unity

`bench_tuned`: BlockShapedPatchCityWorld as baked, 64 fresh flights per set (30 cross, 20 pass, 14 street; routes from
2000), 18.6 km each, scored by `bench-gate` beside replica twins:

| set | cohesive speed | progress speed | one group | lost /km | contacts /km (over 2 m/s) | ended split | surrounded |
|---|---|---|---|---|---|---|---|
| scene gains, d_ref 1.08 | 3.71 | 3.94 | 0.94 | 1.72 (32 drones) | 0.27 (0) | 3% | 67% |
| scene gains, d_ref 1.8 | 3.26 | 3.79 | 0.86 | 0.22 | 0.11 (0) | 5% | 92% |
| scene gains, d_ref 2.0 | 3.10 | 3.76 | 0.82 | 0.11 | 0.22 (1) | 8% | 95% |
| T1, d_ref 1.8 | 3.66 | 4.59 | 0.79 | 0.32 | 0.22 (1) | 9% | 95% |
| T2, d_ref 1.8 | 3.92 | 4.60 | 0.85 | 0.22 | 0.48 (2) | 8% | 96% |
| **T3, d_ref 2.0** | **3.94** | **4.74** | 0.83 | 0.11 | 0.54 (6) | 3% | **98%** |

T3's replica twins: cohesive speed 3.94, contacts 0.54, losses 0.11 per km, the same as Unity's. Unity's only losses
under T3 were one pair colliding (`CollidingWithAnotherDrone`); none of its contacts cost a drone. Against the scene's
gains at 2.0, T3 is 27% quicker with the swarm together and 26% quicker in progress, for two and a half times the
contacts (six of them over 2 m/s against one).

`bench_single`: one drone (the single-drone condition shares these gains) on the same 64 routes. Under the scene's old
gains 3.96 m/s, under T3's 4.93 (+25%; straight lines 3.75 → 4.66, streets 5.04 → 6.40, 19.7 → 13.3 s per building met),
no contacts or losses in either; the twins 4.01 and 4.96. So in Unity the swarm and one drone are about equally quick
through the city on these routes, as in the replica.

The gate's twins fly every SwarmManager value the bench recorded as in force for the set (its `params` line), not the
live scene's: since S6 the live scene is T3, and a twin built from it would fly the wrong gains for any run made before.

## 9. What it means

- **Fly BlockShapedPatchCityWorld at d_ref 2.0 with `d_shield` 1.6 and `d_obs` 0.16** (now the scene's values). The RC's
  spread dial overrides `SwarmManager.d_ref` every frame and the sim profile maps it to 0.4–1.6, so **launch with
  `.\launch_sim_scripts.ps1 -SpreadRange 1.4,2.6`**: the same dial sensitivity, centred on 2.0, never into the collision
  regime below 1.4. Without the RC (keyboard) the scene's 2.0 applies until A/D is pressed.
- **The cost is building contacts**: T3 touches about one building every 2 km (Unity), two and a half times the scene's rate, a few of
  them faster than 2 m/s. In the sim they are harmless (`DroneHealthMonitor` has no contact rule) and none led to a loss.
  If they matter, **T4 (`a` 0.72) cuts them to the scene's rate** at about 4% of cohesive speed and more flights ending
  split; it was not flown in Unity.
- **The remaining contacts are a force-law limit, not a gain to tune**: cohesion is not shielded, so a straggler is pulled
  through a facade whenever the rest of the swarm is on the other side (`OlfatiSaber.RemoveInwardComponent` is applied to
  the stick and the damper only). Projecting cohesion the same way would be the change to try; it needs the C#, the
  replica and the bench.
- **The swarm and one drone are now equally quick through the city on straight lines** (4.8 and 5.0 m/s); the swarm's
  edge in the trials stays its cameras, as S4 and S5 found.

### 9.1 A note on varying d_ref

A constant spread is a compromise between the streets and the buildings. Under the tuned gains the spread barely moves
speed on the flights that meet buildings (cohesive speed 3.6–3.8 m/s from 1.6 to 2.2), but it decides how many it
surrounds (86–88% at 1.6, 95–97% at 2.0). In the streets a narrower formation is much better: cohesive speed 5.1–5.3 m/s
and one group 94% of the time at 1.6, against 3.4–4.4 and 69–85% at 2.0. So **a spread that narrows to about 1.6 in the
streets and widens to about 2.0 when flying at a building would beat any constant one**, by roughly a fifth in the
streets, if it did not reintroduce collisions: at 1.6 the confirmation still lost 0.43 drones per km in the streets
(T5). Worth trying as a pilot technique (the dial) once the collision problem at small spreads is solved, or as an
automatic rule; not tested here.

## 10. Running it

From `Assets/Scripts/Experiment`, the `stitching` env:

```
python -B -m citysim.swarmtune baseline --name baseline2 --single           # the scene's gains at d_ref 1.0-3.0, one drone
python -B -m citysim.swarmtune sweep --name sweep2 --d 1.8                  # one gain at a time
python -B -m citysim.swarmtune grid --name grid3 --d 1.6 1.8 2.0 2.2 --vary d_shield=1.4,1.6,1.8 d_obs=0.12,0.16,0.2 a=0.65,0.72,0.8
python -B -m citysim.swarmtune confirm --name confirm --cand ../../../experiment_results/swarm_tune/confirm_cands.json --single
python -B -m citysim.swarmtune steps --cand ../../../experiment_results/swarm_tune/confirm_cands.json --out ../../../experiment_results/swarm_tune/steps.csv
python -B -m citysim.swarmtune bench-plan --name bench_tuned --cand ../../../experiment_results/swarm_tune/bench_cands.json --cross 30 --pass 20 --street 14
#   then run_bench.ps1 as printed (or --drones 1 for the single-drone condition)
python -B -m citysim.swarmtune bench-gate --name bench_tuned --runs D:\claude_obstacle_bench\runs\swarmtune_bench_tuned
python -B -m citysim.swarmtune apply --cand ../../../experiment_results/swarm_tune/chosen.json --label T3   # into the scene
python -B -m citysim.swarmtune table --name <run>
```

The page: `python -B -m citysim.swarmtune_replays` (re-flies each replay pick's replica twin with a tape and keeps it only if it scores identically to the gate's row) then `python -B -m citysim.swarmtune_report`, and publish `swarm_tune_bare.html` with the replay files `swarm_tune_files.json` lists.

`grid2`, `grid4` (`--base` + `--vary`) and `refine` (local random steps round a run's best) work the same way. A
candidates file is a JSON list of `{"label", "d", "over": {SwarmManager field: value}}`. Every run appends per flight
and resumes. The flights per candidate default to 24 cross, 16 pass, 12 street (`--cross/--pass/--street`).
