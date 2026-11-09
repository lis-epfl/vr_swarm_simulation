# The search study

**S2 Hidden Goal Search** in the study registry ([docs/studies/](../studies/README.md)); "the search study" is its code name.

Does the swarm find a goal it has not been told about sooner than a single drone does, and is the swarm's habit of
**morphing round buildings** (the pilot holds the stick, the formation flows round whatever stands in its way) part of
the reason? The density study ([../density_study/README.md](../density_study/README.md)) predicts what each interface
sees *at* a goal and flies *directed* transits, with a pilot who knows where the goal is. It reaches the search only
through an analytic sweep-width model. This study simulates the search itself: a pilot who does not know where the
goal is flies the city until the goal is first on screen.

Everything runs from `Assets/Scripts/Experiment` with the `stitching` env's Python, as
`python -B -m citysim.vissim <command>`. Outputs go to `experiment_results/search_study/`. The code is
`citysim/search.py` (the search), `citysim/pilots.py` (the pilots) and `citysim/pilotfit.py` (the fitted pilot).

## 1. Where it stands (2026-10-15)

**The swarm finds a hidden goal 10–20% sooner than a single drone, and almost none of that is morphing.** The extra
cameras do it. A swarm flown along the single drone's paths, steered round every building, finds the goal as soon as
the morphing swarm everywhere except in the packed city C2. There, steering ten drones through the single drone's
narrow gaps splits the formation and costs drones, and letting it morph avoids that.

5,040 replica searches (360 scenarios: 15 cities × 3 goals × 8 starts), at spread 1.0, unless named. Each ratio is the
restricted mean time to first sight of one arm over another's, over the same scenarios, with its 95% interval; below 1
means the first arm finds the goal sooner.

| | C0 open | C1p porous | C2p porous | C1 packed | C2 packed |
|---|---|---|---|---|---|
| **swarm / single**, random walker | 0.89 [0.83, 0.94] | 0.89 [0.84, 0.94] | 0.94 [0.87, 1.00] | 0.82 [0.78, 0.87] | 0.85 [0.78, 0.92] |
| … internal_3's pilots (fitted) | 0.85 [0.73, 0.99] | 0.79 [0.67, 0.93] | 0.85 [0.76, 0.96] | 0.92 [0.77, 1.08] | 0.77 [0.59, 0.96] |
| **morphing**: swarm / swarm_steer, random walker | 1.00 [0.99, 1.03] | 1.01 [0.95, 1.08] | 1.00 [0.97, 1.02] | 0.98 [0.90, 1.07] | **0.84 [0.76, 0.91]** |
| … fitted | 1.04 [0.97, 1.11] | 0.95 [0.91, 0.99] | 0.98 [0.93, 1.04] | 1.00 [0.88, 1.15] | 0.94 [0.87, 1.00] |
| **cameras**: swarm_steer / single, random walker | 0.88 [0.82, 0.94] | 0.89 [0.84, 0.93] | 0.94 [0.89, 0.98] | 0.84 [0.77, 0.91] | 1.02 [0.90, 1.12] |
| … fitted | 0.82 [0.70, 0.98] | 0.83 [0.72, 0.95] | 0.86 [0.76, 1.01] | 0.92 [0.77, 1.11] | 0.82 [0.65, 1.01] |
| swarm / single at 200 m, random walker | 0.85 [0.75, 0.94] | 0.83 [0.75, 0.90] | 0.85 [0.77, 0.91] | 0.76 [0.65, 0.88] | 0.72 [0.63, 0.82] |
| swarm / single in the head's view (fitted head), random walker | 0.94 [0.90, 0.97] | 0.91 [0.86, 0.95] | 0.94 [0.88, 0.98] | 0.92 [0.88, 0.98] | 0.86 [0.79, 0.92] |
| **swarm / single at spread 0.4**, random walker | 0.94 [0.91, 0.97] | 0.98 [0.87, 1.10] | **1.23 [1.14, 1.32]** | **1.12 [1.06, 1.19]** | **1.34 [1.23, 1.48]** |
| single drone: found within 25 min / median, random walker | 82% / 9.9 min | 64% / 15.9 min | 64% / 17.1 min | 57% / 21.3 min | 64% / 15.6 min |

- **Why morphing does not help:** the morphing swarm does see round corners more often in dense cities (its first sight
  came round a building from the formation's centre 15–21% of the time against 3–6% for the steered swarm in C1/C2,
  fitted pilot). But a goal patch is a crowd in a cleared plaza, visible for a long way down a street, so a sight round
  a corner is a few seconds early, not a goal the steered swarm would have missed. Only 0–6% of first sights came while
  a building stood inside the formation (16% in C2p, fitted).
- **What the swarm has is reach.** 73–97% of its first sights were out of range or band of a camera on the formation's
  centre: its rim drones are 7 m further out and look outward, so the hull's feeds sweep wider. That is the density
  study's "twice the sweep width", now as time.
- **The tight spread loses in dense cities.** At 0.4 the swarm loses 5.4 drones a search on average, and half its
  searches (61–83% in the dense cities) end with fewer than four drones, which counts as not finding the goal. It is
  then slower than the single drone (1.12–1.34). At 1.0 it loses 2.4, and 25–30% of searches end that way in the
  packed cities.
- **Looking round adds little to the single drone** (`single_scan` / `single` 0.95–0.99), and **straighter search
  favours the swarm**: with 240 m runs the swarm's ratio is 0.75–0.84, with 60 m runs 0.88–0.95. A walker that turns
  often re-covers ground the swarm's wider sweep has already seen.
- **Random search is slow, and people are not random.** Flown from internal_3's own leg starts in its own city, the
  fitted pilot took 2.3× (swarm) and 10× (single drone) as long as the people did (§4). The ratios above describe
  undirected search. Because internal_3's single-drone pilots out-searched the model far more than its swarm pilots,
  the swarm's advantage with real pilots may well be smaller.

## 2. The design

**Scenario.** One of a layout's goal patches is the hidden goal: today's patch (G0: a 7 m occluder ringed by eight
patrolled buildings 26 m out, 64 walkers), the only one whose walkers walk. The cities are the capability study's
(`specs/capability.json`): open C0 (today's coverage, α 0.024), porous C1p and C2p (`citygen` core "shell": every two
avoidance cylinders at least 6 m apart, the narrowest gap the formation flows through, so it can flow between any two
buildings) and packed C1 and C2 (4 m facade gaps, closed to the avoidance law), three seeds each. Each of a layout's
three goals is searched for from eight starts: free points inside the target mask (below), at least 260 m from the goal
(so not even the 200 m range sees it at the start), with as much clearance from the buildings as the city allows,
facing a random way. That gives 360 scenarios, 72 per kind of city.

**Arms.** Every arm of a scenario flies the same intent chain from the same start, so the comparison is paired:

| arm | what is flown | how the pilot gets from one intent target to the next |
|---|---|---|
| `single` | one drone, its view slaved to its heading | round every building (`routing` "single": avoidance cylinder + 1 m) |
| `single_scan` | the same flight, its view sweeping ±60° every 20.6 s (internal_3's yaw duty) | as `single` |
| `swarm` | ten drones, `OUTER_CIRCLE` feeds | round **walls** only (`routing` "swarm": gaps under `w_pass` closed, a formation radius kept off them); the stick is held through lone buildings and the formation morphs round them |
| `swarm_steer` | ten drones | the single drone's rule: round every building |

So `single → swarm_steer` is more cameras on the same paths, and `swarm_steer → swarm` is the morphing. The swarm flies
at spread 1.0 (primary) and 0.4.

**Pilots** (`citysim.pilots`). A pilot who does not know where the goal is flies straight for a while, turns and flies
on. The intent is a chain of targets, each a run from the last at a turned heading, laid target to target from the
start. It is independent of the flight, so every arm draws the same chain and a chain can be flown again in Unity.
A target that leaves the arena is reflected back in, and every target is snapped to the **target mask**: free ground
the single drone's planner can stand on, in the arena's largest connected open region. Consecutive targets are at
least 20 m apart, so none is reached the moment it is set.

- **`rw`, the random walker.** Run and tumble: runs of 20 m + Exp(100 m) (mean 120 m), turns uniform on (−180°, 180°],
  full stick, no pauses. Sensitivities `rw60` and `rw240` change the mean run, on four starts per goal.
- **`fit`, internal_3's pilots** (`vissim pilot-fit`, `citysim.pilotfit`). Fitted per interface on every leg's
  *searching window*: from the leg's start to the first instant the next goal was on a shown feed at 80 m. Before it
  the pilot had nothing to steer for; after it they flew at the goal. The commanded course is cut into straight runs
  at turns of at least 20°; smaller course changes are steering round a building or along a street, which the planner
  does, so the runs either side of one merge. Pauses are the stick centred for at least a second.
  ([pilot_model_internal_3.json](../../experiment_results/internal_3/results/pilot_model_internal_3.json))

  | | single drone | swarm |
  |---|---|---|
  | searching time / legs / distance | 779 s / 11 / 4.6 km | 549 s / 10 / 2.7 km |
  | mean run between turns (censored exponential) | 90 m | 75 m |
  | turn angle, median magnitude (resampled with a random sign) | 35° | 28° |
  | pauses per turn / share of the time paused | 0.12 / 7.6% | 0.22 / 17.7% |
  | stick while moving | full 99.5% of the time | full 99.3% |
  | heading turn rate in a turn, p50 / p90 | 53 / 66°/s | 70 / 89°/s (body yaw) |

  Both pilots flew a persistent walk: a turn every 75–90 m, a median of 30°, so the fitted walk wanders far less than
  the random walker's isotropic turns. The data are two pilots and 22 legs, so this is a description of them, not a
  population.

**Heading.** The single drone's heading turns to the bearing it is flown along, at the airframe's 75°/s (random
walker) or its pilots' p90 turn rate (fitted, 66°/s). The swarm's body yaw turns to the bearing at the rig's 90°/s
(fitted 89°/s). internal_3's single-drone pilots flew forward and steered by yawing: their course was more than 30°
off the heading 3% of the time. So looking round is a sensitivity arm (`single_scan`), not the fitted behaviour.

**The swarm's feeds** come from `display.FeedLatch`, AttitudeAlgorithm's rule with its timing:

- the hull membership is debounced over 0.5 s;
- the target heading is low-passed at 0.3 s;
- each drone turns onto its target at 75°/s;
- a feed is shown only once its drone is within 5° of the target (`BoundaryFeedReady`).

The density study's dynamic tier used the instant rule, an upper bound. `--instant-feeds` restores it.

**Head** (scoring only: it changes what counts as seen, not the flight). Relative to the body yaw:

- `ahead`: straight ahead throughout.
- `fit`: internal_3's swarm pilots' head dwells, resampled as (angle, duration) pairs and joined by moves at their
  median head speed. In their searching windows they held the head still 94% of the time, in dwells of a median
  1.8 s, mostly within 15° of ahead.

The single drone's one feed is always in front.

**Discovery** is `citysim.discovery`'s definition, the trials' co-primary: at least one of the goal's walkers expected
on a shown feed, within range, in the field of view and the gimbal's vertical band, with a clear sight line.

- Bases: `any` (every shown feed: perfect attention), and `head_ahead` / `head_fit` (the feeds inside the HMD's ±50°
  about the head model's yaw).
- Ranges: 80 m (the definition) and 200 m (what internal_3's turns onto goals suggest).
- Each interface flies at its own internal_3 transit band: the swarm at 19.5 m and −15°, the single drone at 17.0 m and
  −14°.

A flight is checked every 0.5 s. It stops when every basis has seen the goal, at the 25 min cap, or when the swarm has
fewer than four drones (the protocol's rule for a run going on). A swarm stopped that way **counts as not having found
the goal**, the conservative reading; `times(..., lost="end")` censors it where it stopped instead.

**At the first sight** (any feed, 80 m) the flight records how it happened:

- would an ideal all-round camera on the formation's centre have seen it too (`centre`), was its line blocked by a
  building (`corner`), or was the goal out of its range or band (`reach`);
- was a building inside the alive drones' hull in the last 5 s (the formation morphing);
- was the drone whose feed saw it behind a building from the centre.

**Measures.**

- Per cell (pilot × city × arm × spread): the share found within the cap, the Kaplan–Meier median and the restricted
  mean time to first sight (RMST, the area under the survival curve up to the cap).
- The contrasts: RMST ratios over the paired scenarios, with a 95% interval from a bootstrap over (layout, goal)
  clusters, and the share of scenarios the first arm found first. Labels: swarm / single, *morphing* (swarm /
  swarm_steer), *cameras* (swarm_steer / single), scan / single.

## 3. Results ([search/](../../experiment_results/search_study/search))

**Time to first sight**, any shown feed at 80 m: the share found within 25 min / the Kaplan–Meier median, spread 1.0.

| | C0 | C1p | C2p | C1 | C2 |
|---|---|---|---|---|---|
| random walker: single | 82% / 9.9 min | 64% / 15.9 | 64% / 17.1 | 57% / 21.3 | 64% / 15.6 |
| swarm | 90% / 9.8 | 72% / 12.3 | 65% / 13.5 | 72% / 13.7 | 72% / 13.1 |
| swarm_steer | 90% / 9.8 | 74% / 12.8 | 69% / 14.7 | 64% / 12.8 | 57% / 15.6 |
| fitted: single | 89% / 7.1 | 82% / 11.9 | 81% / 10.9 | 89% / 8.4 | 69% / 16.4 |
| swarm | 92% / 5.2 | 90% / 7.8 | 88% / 9.6 | 86% / 6.9 | 86% / 8.2 |
| swarm_steer | 93% / 5.1 | 92% / 8.8 | 86% / 9.9 | 82% / 7.0 | 75% / 9.4 |

The survival curves are [km_rw_any.png](../../experiment_results/search_study/search/figures/km_rw_any.png) and
[km_fit_any.png](../../experiment_results/search_study/search/figures/km_fit_any.png) (and `_head_fit` for the head's
view). Every contrast, every pilot and both bases are in
[contrasts.png](../../experiment_results/search_study/search/figures/contrasts.png) and `contrasts.csv`.

- **The contrasts hold across pilots.** Swarm / single at spread 1.0 is below 1 for every pilot in every city on any
  feed (0.75–0.95), with an interval that excludes 1 in all but two cells. For the random walkers it is also below 1
  in the head's view (0.79–0.97). With the fitted pilot in the head's view the advantage shrinks (0.81–1.03) and is
  clear only in C1p: its swarm pilots held their heads near straight ahead.
- **The morphing contrast is about 1.** Its interval excludes 1 in four cells of twenty:
  - C2 for the random walker (0.84);
  - C2p at 240 m runs (0.91);
  - C1 at 60 m runs (0.92);
  - C1p for the fitted pilot (0.95).

  Everywhere else it straddles 1 (0.94–1.05). In C2 the steered swarm, flown through the single drone's narrow gaps,
  is split 76% of its flight against 37% for the morphing one. It loses 4.4 drones a search against 2.8, and moves at
  3.5 m/s against 4.2. Morphing pays where the alternative is squeezing the formation, not through what it sees.
- **The cameras contrast carries the advantage:** fitted 0.82–0.92 in every city, random walker 0.84–0.94 in every
  city but C2 (1.02).
- **How the swarm first saw the goal** ([mechanism.png](../../experiment_results/search_study/search/figures/mechanism.png)):

  | | share of the swarm's first sights |
  |---|---|
  | out of the formation centre's reach (range or band) | 73–97% |
  | round a corner from the centre | 3–23%, most in C2 |
  | the centre would have seen it too | 0–9% |

  At spread 1.0 the morphing swarm sees round a corner more often than the steered one in the dense cities (fitted:
  15–21% against 3–6% in C1/C2), and its discovering drone was behind a building from the centre 12–40% of the time.
  The first sight came a median 112–121 m from the goal centre, against 98–106 m for the single drone.
- **Losses** (swarm, per search): spread 1.0, 0.3–0.6 in C0, 2.2–3.6 in the other cities (random walker); spread
  0.4, 2.3 in C0 and 5.7–7.1 elsewhere. The protocol's four-drone floor ends 13% of searches at 1.0 (25–30% in the
  packed cities) and 49% at 0.4. These are the replica's losses over 10–25 min of flight; the density study's Tier C
  measured the same order in Unity (5.8–7.0 per goal flight at 0.4 in packed cities).

### 3.1 Streets or blocks ([search_streets/](../../experiment_results/search_study/search_streets))

Does the morphing swarm keep to the streets, and is it faster? The first four starts of every goal (both pilots,
every condition, both spreads; 1,980 flights) were flown again with two street measures. "Street" is analyse.py's
classification of a generated layout: within half a street's width of a centreline. Every first sight is identical
to the full run's, so these are the study's own searches.

- **street time**: the share of drone-time, every alive drone at every sample, over a street.
- **centre path**: the share of the formation centre's path that ran along a street.

Random walker, spread 1.0:

| | C0 | C1p | C2p | C1 | C2 |
|---|---|---|---|---|---|
| street time: swarm, morphing / steered / single | 29 / 29 / 29% | 36 / 36 / 37% | 37 / 37 / 38% | 43 / 38 / 41% | 58 / 45 / 53% |
| centre path along streets, same order | 30 / 30 / 30% | 35 / 38 / 40% | 33 / 37 / 42% | 46 / 40 / 48% | 65 / 45 / 61% |
| speed, m/s, same order | 6.5 / 6.5 / 6.6 | 5.1 / 5.1 / 5.4 | 4.4 / 4.4 / 4.8 | 4.7 / 4.3 / 4.6 | 4.3 / 3.5 / 4.0 |
| formation split, morphing / steered | 21 / 24% | 55 / 47% | 63 / 56% | 60 / 71% | 39 / 77% |

- **The morphing swarm is not faster than the single drone.** With the random walker it matches the single drone's
  speed in the open and packed cities and is slower in the porous ones, where it squeezes through gaps. With the
  fitted pilot the single drone is faster everywhere (3.9–6.3 m/s against 3.6–5.6), because its pilots paused less.
- **It keeps to the streets about as much as the single drone**, a few points more in the packed cities. In the
  porous ones its centre runs along streets less, because every gap is wide enough and it flies straight across
  blocks.
- **What stands out is the steered swarm in the packed cities**: slower, and less of its time over streets than either
  of the others. Its pilot flies the single drone's route, which cuts through the gaps between a block's buildings
  wherever one drone fits. Ten drones do not fit: the formation splits (77% of the time in C2, against 39% for the
  morphing swarm), slows down, and its drones and its centre end up over the block. The morphing swarm's pilot treats
  a packed block as a wall and goes round it by the streets. That is the mechanism behind the one city where morphing
  helps (C2, ratio 0.84): it is the steered swarm's route that costs it, not anything the morphing swarm sees.
- The packed C2 sample replay is an extreme case. That search's single drone cut through blocks for most of its route
  (35% of its time over streets, against 53% on average).
- `search_streets_fresh/` is an independent replicate: the same starts with fresh headings and target chains, flown
  by a `--starts` that first redrew the scenarios (see §6). It gives the same picture to within 2–3 points.

## 4. Checks

### 4.1 The Unity spot check ([bench/spot/](../../experiment_results/search_study/bench/spot))

`search-bench-plan` laid each arm's random-walker chain for two scenarios in each of three cities (C0, C1p, C2; seed
0) as bench waypoints, 480 s a flight. The headless bench flew them in ProceduralCityWorld (18 flights, `--record-all`,
pilot heading on, each interface at its band altitude). `search-bench` scores Unity's recorded trajectories with
Unity's own feeds (BoundaryFeedReady) and compares them with the replica flying the same waypoints (two seeds each):

| | Unity | replica |
|---|---|---|
| swarm: found within 480 s at 80 m / RMST | 50% / 351 s | 50% / 349 s |
| swarm, 200 m: found / RMST | 100% / 115 s | 100% / 114 s |
| swarm / single, 80 m / 200 m | 0.91 / 0.38 | 0.90 / 0.37 |
| morphing (swarm / swarm_steer), 80 m / 200 m | 0.98 / 0.95 | 1.00 / 0.96 |
| cameras (swarm_steer / single), 80 m / 200 m | 0.92 / 0.40 | 0.90 / 0.39 |

Flight by flight, first sights agree within a few seconds. The shown feeds agree (5.0–8.5 against 5.3–8.5), and so do
the paths (within about 4%) and the losses. The exceptions are in packed C2:

- a steered-swarm flight first saw the goal 18 s later in Unity;
- another flight found it late in one replica seed (437 s) and not at all in Unity;
- a single-drone path ran 40% longer in the replica (1,498 m against 1,075 m).

The bases that depend on the swarm's body yaw agree less well at 200 m (head ahead: swarm / single 0.61 in Unity
against 0.83). The bench turns its pilot's yaw toward each waypoint, which is not quite the replica's rate-limited
bearing. Six scenarios are a spot check, not a gate. What it shows is that the replica's feed timing and flight are
Unity's to within the scatter, in the open and porous cities and mostly in the packed one.

### 4.2 The fitted pilot in internal_3's own city ([check/internal_3/](../../experiment_results/search_study/check/internal_3))

`search-check` flies the fitted pilot from every searching leg's start (21 legs with a window of at least 2 s), with
the leg's goal hidden, in DiamondCityWorld, eight seeds per leg. Its centroid track is scored the same way as the
pilots' searching windows (`search.track_stats`):

| | single drone: people / model | swarm: people / model |
|---|---|---|
| path speed | 6.4 / 5.8 m/s | 5.4 / 5.1 m/s |
| share of time under 1 m/s | 7.9% / 5.5% | 18.3% / 13.4% |
| turns of 20° or more per km | 13.6 / 21.8 | 10.0 / 10.1 |
| straightness over 30 s | 0.85 / 0.81 | 0.94 / 0.84 |
| time to first sight (80 m): people's median / model's median | 37 s / 351 s | 59 s / 135 s |

- **The model flies like the people.** Speed, stops and the swarm's turn rate match. The single drone turns 60% more
  than its pilots did: its planner rounds every building with a string-pulled corner, where a pilot's turn is a smooth
  arc that is often too slow to count.
- **It does not search like them.** From the same starts the people saw the goal 2.3× (swarm) and 10× (single drone)
  sooner. Both pilots had flown the city before. The density study's search model put their efficiency at 0.20 and
  0.23 of random search; here the model's own runs say 0.44 and 0.10. Either way, the people's single-drone search
  beats undirected search by far more than their swarm search did. So this study's swarm / single ratios are for
  pilots who do not know where to look; with pilots who do, the single drone closes some of the gap.
- In the same city and starts, the model's swarm / single RMST ratio is 0.70–0.72, and morphing again adds nothing (228 s
  against 232 s).

## 5. Running it

```
cd Assets/Scripts/Experiment
python -B -m citysim.vissim pilot-fit                        # internal_3's searching pilots -> internal_3/results/pilot_model_internal_3.json
python -B -m citysim.vissim search [--quick] [--pilots rw fit] [--clutter C1p C2p]   # -> search_study/search/ (~1-2 h)
python -B -m citysim.vissim search-tables --name search      # the tables and figures again from runs.csv
python -B -m citysim.vissim search --name search_streets --pilots rw fit --starts 4   # §3.1, a subset of the same searches
python -B -m citysim.vissim search-check                     # the fitted pilot in DiamondCityWorld against internal_3
python -B -m citysim.vissim search-bench-plan                # Unity spot check: scenarios, configs, runner on the bench root
powershell -File D:\claude_obstacle_bench\cfg\search\spot\run_search.ps1   # fly them (~40 min, resumable)
python -B -m citysim.vissim search-bench                     # score them against the replica on the same chains
python -B citysim/tests/run_tests.py -k test_search          # tests
```

| output | |
|---|---|
| `search_study/search/runs.csv` | one row per flight: scenario, pilot, arm, spread, why it stopped, first-sight time per basis and range, losses, contacts, path, speed, targets reached / timed out / stalled, pauses, morph and split measures, and the first sight's mechanism |
| `…/cells.csv` | per pilot × city × arm × spread: found share, KM median and RMST per basis, losses, speed, morph, mechanism shares |
| `…/contrasts.csv` | the paired RMST ratios with their intervals and the share found first |
| `…/figures/` | `km_<pilot>_<basis>.png` (survival curves per city), `contrasts.png`, `mechanism.png` |
| `search_study/check/internal_3/` | `check_runs.csv`, `check_summary.csv` |
| `search_study/bench/spot/` | `plan.json`, `bench_flights.csv`, `bench_summary.csv` |
| `search_study/search/dashboard/` | the results dashboard's four data files (below) |

**The results dashboard is a light rebuild** (2026-10-21, https://claude.ai/artifact/9KcgYMcAH4RRWWSKdFbU59). The
first one (2026-10-16) was a Dashboard-type artifact whose data was built by scratchpad scripts and lived only in its
own store; it was found gone on 2026-10-21. The rebuild is deliberately less in depth: headline ratios, swarm ÷ single
drone and the cameras-or-morphing split per city with 95% intervals, every condition's searches, and the two checks,
under filters for pilot, screens counted and range. It leaves out the first one's city maps, the replay of one search
per city in every condition, the pilots' target chains and the method tab; this manual has the method. It is built
from the repo now, so it can be rebuilt: `python -B -m citysim.search_dashboard` writes
`search/dashboard/{contrasts,conditions,unity_check,pilot_check}.csv` from the tables above; upload each as the
dashboard's file, point its `datasets/<name>` at it, and set its `files/index.html` to
`citysim/search_dashboard.html` (the Dashboard type's own instructions say how, when the artifact is read).

## 6. Limits

- **The replica.** Its ten drones fly the 2-D force law with contacts and friction, gated against Unity in open and
  porous cities. In the dense ones (C2p, C1, C2) it fails the gate on spacing and losses, and in packed cities it flies
  25–50% too fast (density study §5.5). Spread 0.4 there is the least trustworthy cell of the study, and the Unity spot
  check is the only direct test.
- **The pilots are models.** The random walker is a null model: no pilot flies run-and-tumble. The fitted pilot rests
  on 22 searching legs from two pilots who had flown the city before. Its turn statistics include whatever they knew,
  and the 20° split between intent and steering is a modelling choice the check (§4) can only partly test.
- **Discovery is not identification.** The clock stops at first sight of the goal's walkers. The hat, the press and the
  pilot's attention are the trials' to measure. The head model bounds attention from both sides (`ahead`, `any`) and
  puts internal_3's head behaviour between them.
- **The panorama is not modelled**, as in the density study.

Two operational notes:

- **A subset of searches is taken by start index, never by drawing fewer starts.** A goal's start points, headings
  and seeds come off one random stream, so drawing four starts instead of eight keeps the points but changes every
  search. `--starts` filters the full list since 2026-10-16; the first version redrew it, which is how
  `search_streets_fresh/` came to be a replicate rather than a subset.
- **Every run so far has stranded one to four jobs of its first batch**, always in C2p seed 0, the first layout in
  the list. No error was raised and no worker died. The pool's 30-minute stall rerun would eventually recover them,
  but each was quicker to fly on its own with the same job function and seeds (13–91 s). The flights are
  deterministic, so the merged rows are the ones the pool would have produced, and each run's `manifest.json` names
  the scenarios merged this way. The cause is not known; it looks like a worker hanging at start-up.
