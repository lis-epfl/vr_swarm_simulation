# Trial protocol: interface × city density

The human trials the density study prepares. Pilots search generated cities for a target pedestrian with one
of two interfaces, at three city levels chosen beforehand by a fixed rule from simulation. This page is the
protocol to pre-register; [README.md](README.md) explains how the levels were chosen and what each
prediction rests on.

**Status (2026-10-13):** levels chosen and their trial layouts generated; the predictions of all four tiers
are written below (§11). The pilot block has not been run, so the sample size and the layout pool's size are
not fixed yet (see *Power and the pilot block*).

## 1. Question

Does the swarm's advantage over a single drone, in finding a target among distractors, follow the
occlusion geometry the simulation says drives it? The swarm's feed ring sees round a goal's central
occluder (envelopment) and from several vantages at once (parallax); a single drone has to fly round it.
The levels are chosen so the prediction differs in kind across them:

- **L1, today's regime**: small occluder, open city. The swarm wins by coverage only.
- **L2, envelopment**: a wide occluder in an open city. The swarm's predicted advantage is about twice L1's.
- **L3, collapse**: a cramped goal in a dense city. The formation cannot surround the occluder, and the
  single drone is predicted to do *better*.

## 2. Conditions

**Interface** (within subject):
- *Swarm*: 10 drones, `OUTER_CIRCLE` feed ring (hull drones' feeds at their headings), the stitched
  panorama available, the look-direction gap fill on, spread stick free (d_ref 0.4–1.6).
- *Single drone*: one drone, its feed in front, the view slaved to its heading.

**Level** (within subject): three generated cities in `ProceduralCityWorld`, irregular blocks of the
study's building design at twice its pack size, 50 m tall, with three goals each. All three use the same
generator; only clutter and goal geometry change.

| | L1 | L2 | L3 |
|---|---|---|---|
| sweep level | C0-G0 | C0-G10 | C2-G7 |
| block spacing p / street width s | 90.83 m / 15 m | 90.83 m / 15 m | 60 m / 10 m |
| coverage α (buildings / ground) | 0.024 | 0.024 | 0.17 |
| free-sight length ℓ, and ℓ / R (R = 40 m) | 355 m, 8.9 | 386 m, 9.7 | 53 m, 1.3 |
| buildings in the city | 115 | 102 | 680 |
| central occluder w_c (long side) | 7 m | 28 m | 10 m |
| ring radius r_g, patrolled buildings n_b | 26 m, 8 | 48 m, 8 | 17 m, 6 |
| clearance c_g (centre to nearest other building) | 22–23 m | 44–45 m | 13–14 m |
| walkers per goal | 64 | 64 | 48 |
| narrowest street | 15.1–15.7 m | 15.3–16.0 m | 10.0–10.1 m |

L1's goal stands in for internal_3's goal patch: a 7 m occluder (the patch's centre building is 6.96 m
long) with eight patrolled buildings 26 m out (the patch's stand 24–34 m from its centre). The trial
layouts are in
`Assets/Scripts/Experiment/layouts/` (`L1_…`, `L2_…`, `L3_…`, seeds 101–104); generate more seeds with
`python -B -m citysim.citygen trial --seeds …`.

## 3. Design

2 (interface) × 3 (level), within subject, **blocked by interface**. Block order is counterbalanced; within
each block the three levels follow a **Williams design**. For three conditions that takes both a Latin
square and its mirror image, six orders, in which each level follows each other level equally often:

| order | 1st | 2nd | 3rd |
|---|---|---|---|
| A | L1 | L2 | L3 |
| B | L2 | L3 | L1 |
| C | L3 | L1 | L2 |
| D | L3 | L2 | L1 |
| E | L1 | L3 | L2 |
| F | L2 | L1 | L3 |

A participant uses the same level order in both blocks, so a level's two trials sit at the same position
within their blocks and the interface contrast is not confounded with position. Crossed with the two block
orders this gives **12 sequences**; recruit in multiples of 12 (6 balance the level orders but not their
pairing with block order).

*Pre-register this design or the alternative below, not both.* The alternative is fully interleaved: the
six cells in a 6 × 6 Williams order, interfaces alternating. Blocking is recommended. The two interfaces
need different skills and are practised separately, and changing interface means re-spawning the rig.

**Trials.** Six main trials (one per cell), three goals each. **Every trial is a fresh layout.** At level ℓ,
participant k (0-based, in recruitment order) flies layout `pool_ℓ[k]` in the swarm block and
`pool_ℓ[(k + 1) mod n]` in the single block. Each layout is then flown exactly once under each interface,
by different participants, so the layout is a crossed random effect and the interface contrast is within
layout. A pool holds n layouts per level, 3n in all. The goal seed (which blocks hold goals, which hat each
goal gets, which walker wears it) is the layout's seed, so everyone who flies a layout searches for the
same targets.

## 4. Session (≈ 85 min)

| | min |
|---|---|
| consent, briefing, headset fit, eye-tracking calibration | 10 |
| PracticeWorld: the hats on their signs, the RC, the press | 5 |
| block 1: practice to criterion (below), then its three trials (≈ 6 min each) | 30 |
| NASA-TLX for the interface; break | 5 |
| block 2: the same with the other interface | 30 |
| debrief | 5 |

**Practice to criterion, in each interface**, on practice layouts at L1 geometry (their own seed range,
e.g. 900 onwards, never in a trial pool): **two goals correctly identified within 240 s of entering each
goal's zone**. Up to three practice runs; a participant who does not reach criterion in an interface is
excluded (pre-registered). The bound is a proposal: internal_3's per-goal search times were mostly under
two minutes.

**Run cap.** A run ends when all three goals have a correct press, or on the experimenter's end key at
12 min; goals not found by then are right-censored at that time.

**Task start.** The pilot takes off and looks round untimed; the task clock starts when the first drone crosses the
start line 30 m ahead of the spawn (a tinted sheet over the view ahead that disappears as it is crossed).

## 5. Task and identification (auto-scored)

At each goal the pilot looks for the one walker wearing a hat (bucket, cap or cowboy; 64 walkers per goal at
L1/L2, 48 at L3) and presses **C2** on the RC when the hat is on screen. `ExperimentRecorder`
(`identifyMode: AutoScore`) scores the press on the spot:

- the goal is the one whose zone (the patch grown by 15 m) holds the live drones' centroid; with none,
  the press is `out_of_zone`;
- it is **correct** when that goal's hat is inside the central 80 % of a shown feed's image, within 40 m of
  that feed's camera, with no building in the way; otherwise a **false alarm**, logged, with the goal left
  open and the search going on;
- every press is logged with the scoring camera's distance, its feed and the feed the pilot's gaze was on.

The verbal hat report is a secondary check only (a third of guesses are right by chance). The
experimenter's 1/2/3 keys override the latest press's score and are logged as overrides; they never
create an identification of their own. The 40 m range is the analysis constant: five hats were
identified in internal_3 whose closest approach was 27–34 m.

## 6. Operator instructions (read to every participant)

- **Altitude.** Buildings are 50 m tall and the drones cannot climb above about 51 m. Search below the
  rooftops: internal_3's pilots searched at 5–24 m (swarm median 12 m, single drone 7 m).
- **Camera pitch.** *The gimbal dial's centre is 15° down, not level.* At 12 m a walker 40 m away sits
  about 15° below the horizon, so the centred dial puts a distant walker mid-frame; turn the dial down for
  walkers nearer the drone and up for distant ones. In internal_3 swarm pilots used −8° to −40° (median
  −21°) and single-drone pilots left the dial near its centre.
- **Spread (swarm).** Free. It is logged as a covariate and is not coached.
- **Panorama (swarm).** Available, toggled with the usual control. It is not required and not modelled;
  its on-time is logged.
- **Losses.** Drones can be lost to collisions (with each other below 0.5 m, a flip, the ground), and a
  drone left far behind is lost. A swarm run continues while four or more drones are alive.

## 7. Outcomes

- **Co-primary (two):**
  1. **Search time** per goal: zone entry → correct press.
  2. **Discovery time** per goal: leg start (leaving the previous goal's zone, or the start line for the first goal)
     → the first instant the goal is on a shown feed. "On a shown feed" is `citysim.discovery`'s definition, the one
     the predictions use: at least one of the goal's walkers expected on a shown feed, within 80 m slant range,
     inside the camera's field of view and the gimbal's vertical band, with a clear sight line. analyse.py computes it
     from the poses, the `shown` masks and the layout, with no new logging (`discoverySec` in `legs.csv`); a goal
     already seen on an earlier leg scores 0 (`preSeen`). This is where the swarm's transit advantage enters the
     task: its feeds cover twice the width of city a single drone's does as it flies, and its leading drones see
     round the buildings it passes (README §2.1). The task itself is unchanged: goals are not announced, so the
     transit is a search for the next one.
- **Confirmatory secondaries, with the directions in §11:** flown / straight-line path ratio in transit, stick
  adjustments per 100 m of straight-line progress (analyse.py's command measures), share of the transit path over
  the streets, transit time between goals.
- **Secondary:** first sight → press gap (utilisation; first sight on the shown-feed basis); false alarms;
  yaw duty; drones lost; NASA-TLX per interface. Discovery at the 40, 120 and 200 m sensitivities of the range.
- **Manipulation check:** measured hat exposure per cell on the shown-feed and head bases
  (`analyse.py`'s `inViewShownSec`, `inViewHeadSec` over search time) against the predicted band below.
- **Covariates:** alive and hull-feed counts (time-varying), panorama on-time, the spread at the goal,
  frame rate (`_perf.csv`; the headset drops to 36 fps when frames are missed), trial index.

## 8. Losses and exclusions

- Drone deaths follow the simulator's rules. A swarm run with **3 or fewer drones alive** is ended as
  *swarm failed*, and its unfound goals are censored at that time. The rate of failed runs per level is
  reported against Tier C's prediction.
- **L3 loss threshold.** L3 had to keep Tier C's predicted loss rate at or below 2.5 drones per run
  (today's rate: 10 of 40 swarm drones in internal_3). It does, narrowly: 2.21 ± 0.38 at internal_3's
  spread use. Nearly all of it comes at the tight spread (5.0 per run at d_ref 0.4, 0.83 at 1.0), so it
  depends on how much pilots squeeze the formation in transit. If the pilot block's L3 loss rate exceeds
  2.5, re-run the selection with the measured rate before the main block.
  **With the transits on the pilots' paths it does not** (2026-10-13): 2.99 ± 0.88 per run (7.0 per flight at 0.4,
  1.0 at 1.0), and every packed level flown is over the bar. Which option to take (README §8) is undecided, and the
  levels are on hold until it is.
- **Exclusions (pre-register):** practice criterion not met in an interface; a technical failure that
  ends a run early (crash, tracking loss); a run spent mostly at 36 fps is kept and flagged.

## 9. Analysis

Goal-level mixed models, fitted on found goals:

```
log(search_time) ~ interface * level + trial_index + (1 + interface | participant) + (1 | layout)
```

The discovery co-primary is modelled per leg:

```
log(discovery_s + 1) ~ interface * level + log(leg_m) + trial_index + leg + (1 + interface | participant) + (1 | layout)
```

`leg_m` is the straight-line distance from the leg's start to the goal. It is in the model because the pilots
choose the order they visit goals in: in internal_3 the swarm's legs were half as long again as the single
drone's (median 290 against 192 m), which made the swarm look slower in raw seconds while it was faster per
100 m (14.0 against 20.1 s at 80 m). Pre-seen legs enter at 0, with a sensitivity analysis that excludes them.
**The two co-primaries are tested with Holm's correction at α = 0.05 overall.**

The same structure as search time is used for the first sight → press gap and the transit outcomes, and a logistic
model for whether a goal was found. Censored goals (run cap, swarm failed) enter a censored-data sensitivity
model. **Predicted exposure is a covariate only in a secondary model**: it is a mediator of the level
effect, not a confounder. The primary hypothesis is the direction of the interface effect across
levels (H5); the interface × level interaction in the predicted direction is secondary.

## 10. Power and the pilot block

Simulation-based power analysis (Kumle, Võ & Draschkow 2021; Westfall, Kenny & Judd 2014 for the crossed
participant × layout design):

- participant variance from the **pilot block** (6–8 participants, not confirmatory); internal_3's two
  pilots are a prior only;
- layout variance from Tier B's across-seed spread;
- smallest effect of interest: **15 s per goal of search time**, about 25 % of internal_3's median.
- **The discovery co-primary is simulated beside search time, from the same participants.** Its leg-level
  variance and its correlation with search time come from the pilot block. internal_3's 24 legs are the prior:
  after `log(leg_m)` and the interface, the residual SD of `log(discovery_s + 1)` is 0.54, discovery time grows
  as leg length to the power 1.33, and the swarm's coefficient is −0.21 (a ratio of 0.81 at equal leg length;
  two pilots, so a prior, not an estimate). Its smallest effect of interest is
  proposed as **a 20 % difference in discovery time at equal leg length** (a ratio of 0.8 on the model's log
  scale). Power is computed for the Holm-corrected pair: n is the smallest at which both co-primaries reach
  0.8, each at its Holm-adjusted α.

The pilot block fixes n, expected at 20–30 participants. Six trials each, every layout flown once per
interface, means 3n layouts: 60–90 for 20–30 participants. n, the hypotheses, the levels, the exclusion rules
and the model are pre-registered before the main block.

## 11. Predictions, written before the pilot block

From the tiers (README §4). Exposure is the share of walker-time on a feed during a 60 s hold at the goal,
averaged over the operator band (altitude 5–24 m, pitch −8° to −40°); the swarm row is its best spread.

| | L1 | L2 | L3 |
|---|---|---|---|
| swarm E_head (head-visible feeds) — Tier B | 0.119 | 0.140 | 0.050 |
| single-drone exposure (best realistic policy) — Tier B | 0.072 | 0.035 | 0.100 |
| swarm − single (Δ), ± SD over seeds — Tier B | +0.048 ± 0.002 | +0.106 ± 0.003 | −0.050 ± 0.006 |
| swarm E_any (any shown feed) — Tier B | 0.42 | 0.49 | 0.16 |
| swarm first sight of a walker — Tier B | 21 s | 16 s | 41 s |
| useful feeds at the goal (of 10) — Tier B | 10 | 10 | 5.3 |
| envelops the occluder at spread 1.0 — Tier B / B' | yes / yes | yes / yes | no / no |
| nearest-neighbour spacing in transit, spread 1.0 — Tier B' | 4.4 m | 5.0 m | 6.6 m |
| drones lost per run, transit spread 1.0 / 0.4 — Tier B' | 0 / 2.0 | 0 / 2.0 | 0.7 / 6.0 |
| progress speed in transit — Tier B' | 6.1 m/s | 6.0 m/s | 4.4 m/s |
| drones lost per run, at internal_3's spread use — Tier C (headless Unity) | 0.78 ± 0.20 | 1.11 ± 0.77 | 2.21 ± 0.38 |
| drones alive / on the hull at a goal — Tier C | 9.5 / 8.4 | 9.1 / 8.6 | 7.6 / 5.8 |
| swarm E_head / single exposure at the dial centre (−15°, 13.7 m) — Tier C | 0.081 / 0.090 | 0.156 / 0.034 | 0.038 / 0.044 |
| swarm − single with the drones still alive — Tier B at Tier C's count | +0.047 | +0.103 | −0.046 |

Directions to test: swarm search time shorter than single at L1, more so at L2 (H1, H4); the advantage
gone or reversed at L3 (H1's collapse); the first sight → press gap longer for the swarm and growing with
feed and distractor load (H-util); losses per level as Tier C predicts (H-redundancy).

**Transit and discovery, with every transit on the pilots' paths** (README §2.2, §4; added 2026-10-13). Tier C is
the headless Unity bench on pilot paths, the swarm at spread 1.0 for the transit measures; the discovery rows are
the pass-by model (README §4) with internal_3's search efficiency and Tier C's speeds.

| | L1 | L2 | L3 |
|---|---|---|---|
| discovery time of an unannounced goal (80 m, shown feeds), swarm / single — pass-by | 51 / 81 s | 37 / 54 s | 80 / 138 s |
| swarm ÷ single discovery time: shown feeds / head-visible feeds (200 m: shown) | 0.64 / 0.71 (0.54) | 0.69 / 0.84 (0.68) | 0.58 / 0.73 (0.53) |
| sweep width for a goal (80 m, shown feeds), swarm / single — pass-by | 218 / 145 m | 277 / 200 m | 208 / 127 m |
| next goal first seen when its position is known (directed bound), swarm / single — Tier C | 46 / 49 s | 39 / 44 s | 71 / 84 s |
| path ÷ straight line in transit, swarm / single — Tier C | 1.05 / 1.09 | 1.06 / 1.11 | 1.12 / 1.17 |
| stick adjustments per 100 m, swarm / single — Tier C | 0.3 / 2.0 | 0.3 / 2.1 | 2.2 / 4.7 |
| share of the transit path over the streets, swarm / single — Tier C | 0.24 / 0.26 | 0.24 / 0.30 | 0.70 / 0.66 |
| progress in transit, swarm / single — Tier C | 5.9 / 5.4 m/s | 6.3 / 5.8 m/s | 3.7 / 3.4 m/s |
| sweep width in transit, any feed / head view / single (40 m) — Tier C | 83 / 66 / 44 m | 80 / 65 / 45 m | 77 / 54 / 38 m |
| drones lost per run at internal_3's spread use, pilot paths — Tier C | 0.55 ± 0.38 | 0.61 ± 0.25 | **2.99 ± 0.88** (fails G3) |
| swarm E_head / single exposure at the dial centre, pilot paths — Tier C | 0.074 / 0.096 | 0.154 / 0.043 | 0.000 / 0.042 |

Directions, from these rows:
- **Discovery (co-primary): shorter for the swarm at every level.** The model's ratio is 0.58–0.69 on the trials'
  shown-feed basis, larger than the smallest effect of interest (0.8, §10); 0.71–0.84 on the head-visible basis.
  Unlike exposure at the goal it does not collapse at L3, because it comes from the sweep in transit, which clutter
  does not take away. The absolute times are scaled by internal_3's two pilots and are the weak part.
- **Stick adjustments per 100 m (confirmatory): fewer for the swarm at every level**, by 6–7 times at L1/L2 and
  twice at L3 in the scripted flights. Human pilots adjust far more than the scripted stick (README §9), so only the
  direction is predicted.
- **Path / straight-line ratio (confirmatory): shorter for the swarm, two-sided test proposed.** The scripted flights
  give the swarm a 4–5% shorter path in Unity (in the replica the two are equal; Unity's single drone strays further
  from its planned line), which is small. internal_3's
  pilots flew 1.11× with the swarm against 1.58× with the single drone, a gap the model cannot produce: its pilots
  know where the goals are, and that gap is most likely the search (README §9).
- **Street share and transit time (confirmatory): no difference predicted** on the street share at any level;
  transit time shorter for the swarm, following discovery.
- **L3's exposure at the goal reverses more strongly on the pilot paths**: the formation now reaches the goal centre
  inside the 17 m ring (enveloped 56–67% of holds, against 0–11% on the street route), and from 13.7 m at −15° its
  cameras look over the walkers there.

**The levels are on hold (README §1).** On the pilot paths L3 loses 2.99 drones per run, above G3's bar, and the
selection rule run on these flights picks no L3. The rows above are for the levels as selected on the street route.

**L2 is the robust prediction; L1's advantage may be near zero.** Two checks point the same way at
today's geometry. Tier C's scripted single drone matches the swarm at L1 (0.090 against 0.081). Real
single-drone pilots in internal_3 also saw the target 2.7 times longer than the scripted policies Tier B
gives the single drone (README §5.2). Tier B's L1 margin (+0.048) is therefore an upper bound, while
L2's (+0.106 in Tier B, +0.12 in Tier C) comes from the single drone's exposure halving behind a wide
occluder, which no vantage on one side can undo.

**One level choice is close, and this is recorded before any data.** L3 is C2-G7 by the selection rule.
Its neighbour C2-G6 (13 m occluder, r_g 20 m) has a swarm advantage of +0.003 ± 0.002, which misses the
rule's "vanished within one SD" test by 0.16 SD (it passed by 0.05 SD before the replica's contact
model gained friction). Both are envelopment failures, but C2-G6 is a vanishing advantage and C2-G7 a
reversal. Tier C flies both.

## 12. Pre-registration checklist

- [ ] Hypotheses H1, H4, H5, H-util, H-redundancy, with their directions (§11)
- [ ] The three levels with their parameters and the selection rule that produced them (README §3)
- [ ] The predictions table (§11), including Tier C's rows
- [ ] Design: blocked or interleaved (one), the 12 sequences, the layout assignment rule, layouts per level
- [ ] Practice criterion, run cap, *swarm failed* rule, L3 loss threshold, exclusions (§4, §8)
- [ ] Auto-scored identification rule and constants (central 80 %, 40 m, 15 m zone) (§5)
- [ ] Primary and secondary outcomes, covariates, manipulation check (§7)
- [ ] The discovery definition: `citysim.discovery` at the commit the trials run on, its range (80 m, or 200 m
      if the pilot block settles it there; README §5.6) and the 40 / 120 / 200 m sensitivities (§7)
- [ ] The confirmatory secondaries' directions, taken from §11's transit rows before any trial data
- [ ] Model specifications, Holm across the two co-primaries, the pre-seen-leg sensitivity and the censoring
      sensitivity analysis (§9)
- [ ] n from the pilot block's power simulation for both co-primaries, and the stopping rule (§10)
- [ ] The software version (commit) the trials run on

## 13. Running a session

- Scene `ProceduralCityWorld`. Set the trial's layout on the `ProceduralCity` root's `ProceduralCityBuilder`
  (`layoutPath`, a file name under `Assets/Scripts/Experiment/layouts/`, or the `CITY_LAYOUT` environment
  variable). `swarmSpawn` count 10 (swarm) or 1 (single drone).
- `ExperimentRecorder`: participant, trial number, condition; `identifyMode` AutoScore. Each run's session
  JSON records the layout (id, seed, parameters), the goal seed and each goal's geometry and hat deal.
- `launch_sim_scripts.ps1` as usual (it starts `rcjoy bridge --profile sim`, which carries the C2 presses,
  and turns ASW off). Game view on *Play Maximized*.
- After each test series: group the runs (`analyse.py group`) and run `analyse.py run <test>`. A
  procedural run needs no `"city"` in `test.json`: its city is the layout its session names. The exposure
  columns the manipulation check uses are in `goals.csv`.
