# The views study (S12 Distributed Views) — proposal

**Status: a proposal for review (2026-11-17/18).** Nothing here has been run at full size. The initial tests below were
run to develop the design and to justify its settings; the main runs wait on the experimenter's review (§10). The
review page is https://claude.ai/artifact/CpzLU87XCffSTVdQDopp6j, built by
`python -B -m citysim.views_report --calib calib1 --pilot pilot1 --probe probe1`.

**The initial tests in four lines** (revised 2026-11-18, §6.6). (1) A hat is recognised at 0.5–1.6/s up close, half
that at 24–31 m, a tenth by ~46 m (credited at the press or at the first sight before it: some pilots flew in to check);
a goal's crowd is noticed at ~0.1/s out to ~240 m. (2) Wrapped round the goal building, the swarm's spread-out views
recognise hats ~1.5× as fast as one 360 camera on its most central drone (1.56, 95% 1.24–1.98); circling the block,
only 0.63×. (3) Over the whole task the wrapped swarm and the 360 sphere come out even (1.04, 0.93–1.16): the swarm
gains at the goal and loses between goals; against one drone it is faster or level (0.90, 0.77–1.04). (4) Credited at
first sight, the model's time at a goal matches the people's for the swarm and the sphere; between goals it is still
slower than them.

The registry entry is [docs/studies/S12_distributed_views.md](../studies/S12_distributed_views.md). Code:
`citysim/views_human.py` (what internal_4's pilots tell the simulation), `citysim/views.py` (the task simulator, the
conditions, the counterfactual bases, the calibration), `citysim/views_figs.py` (the figures). Outputs:
`experiment_results/views_study/` (`human/` the fits from internal_4, `<run>/` each run's `runs.jsonl`, `calib.csv` and
`figures/`). Everything runs from `Assets/Scripts/Experiment` in the `stitching` env.

## 1. The question

The swarm differs from one drone with the trials' camera in two ways at once: its ten cameras face every way (all-round
coverage), and they stand in ten places (distributed viewpoints). internal_4's third condition, one drone whose 360
camera is shown on a sphere round the pilot (`SingleSphere`), has the first without the second. So:

1. **How much faster does the swarm do the trials' task than one drone with the ordinary camera, and than one drone
   with a 360 camera?** The second ratio is the benefit of distributed views beyond all-round coverage.
2. **Where does the difference come from**: between goals (finding the next crowd) or at a goal (finding the hat),
   and, at the goal, from the viewpoints themselves (the same cameras at one point), from coverage, or from how each
   is flown?
3. **What should the human retest (swarm against the 360 sphere) expect, and how many pilots does it need?**

The single drone can also look round: its pilots yaw it (internal_4: 39% of the time at a goal) and the sphere's pilot
turns their head. Every condition is flown as internal_4's pilots flew it (§4), not as an idealised camera.

## 2. The task: internal_4's, not a point

**Decision: the simulation flies the whole trial task**, three goals drawn from the city's own buildings exactly as
`BlockGoals` draws them at Play, with the ring's pedestrians walking and the hat on one of them, the pilot not knowing
where any goal is. The clock runs from the start line to the press on the third hat. Each goal's time is split at its
zone (the patch disc grown by 15 m, analyse.py's phase zone) into **between** (from the previous press to the zone)
and **at the goal** (from the zone to the press).

Why not a point hidden at random (S3's target):

- **Most of the task is at the goals.** In internal_4 the time at a goal was 65% of the single drone's per-goal time,
  41% of the swarm's and 57% of the sphere's (per-goal medians: single 100 s at the goal against 54 s between, swarm
  41 s against 56 s, sphere 46 s against 37 s). A point target measures only the between-goal sweep, which S2 and S3
  have already measured.
- **The goal is where distributed views and a 360 camera differ most.** A goal is a building with nobody round it
  inside a ring of buildings whose pedestrians walk round them. A 360 camera at one point sees the ring buildings on
  its own side; the centre building hides the rest. The swarm wrapped round the centre building (which internal_4's
  swarm pilots did: three quarters of their at-goal time inside the ring, a median 22 m from its centre) looks out at
  every ring building at once, each from nearer.
- **It is what the retest measures.** A prediction for the task the pilots fly can be checked against them.

The between-goal half is kept, not dropped: the crowds are what a pilot searches for between goals, and there the
swarm's reach (S2) and the sphere's all-round view both count.

## 3. What the pilots can see: two distances, fitted to internal_4

S1–S11 used two a-priori distances: a hat recognised within `camera.R_ID_M` (40 m; the trials' recorder scores at
80 m since 2026-11-06) and a goal noticed when one walker is expected on a feed within `discovery.R_GOAL_M` (80 m; 200 m
as a sensitivity). Both are replaced here by rates fitted to what internal_4's pilots did, on the head basis (what was
on a shown screen inside the headset's ±50°). Those constants stay as they are in `camera.py` and `discovery.py`,
because S1–S11 reproduce only while they do; S12 carries its own.

### 3.1 The hat: recognised at about 0.5 per second within 20 m, half that at 31 m, almost never past 50 m

`views_human recognition`. For every goal of the 14 runs (42 goals, 41 correct presses, overrides included), every
0.1 s sample of the run: the hat's distance from the nearest camera that had it in the central 80% of a shown image
(both axes, at the recorded gimbal pitch; the sphere has no frame), with a clear 3-D line through the prisms, and
inside the head's view. That is the recorder's own test, measured instead of thresholded. The time the hat spent on
screen before each press is exposure; the press (credited to the hat's distance 1 s before it, the latency) is an
event. The hazard by distance, events per second on screen, is fitted by maximum likelihood as
`lambda(d) = lambda0 / (1 + (d / d50)^k)` (an inhomogeneous Poisson process in the time on screen):

| | value |
|---|---|
| events / exposure | 41 presses / 350 s on screen (1 press without the hat on screen in the 6 s before it) |
| lambda0, the rate up close | 0.49 per second |
| **d50, where the rate halves** | **31 m** (95% profile interval 20–39 m) |
| k, the steepness | 5.1 |
| where the rate falls to a tenth | 47 m |
| binned: 10–20 m / 20–30 / 30–40 / 40–50 / 50–60 / 60–80 / 80–130 m | 0.57 / 0.31 / 0.17 / 0.09 / 0 / 0.014 / 0 per s |

Past 50 m, 1 press in 145 s of exposure (SIMI's sphere, at 57 m). **Decision: the simulation recognises the hat at
this rate (pooled over the conditions) wherever it is on a screen in the head's view; where a single distance is
needed (a threshold measure, a range in a table), it is 47 m, with 31 m and 60 m beside it.** The recorder's scoring
limit (`identifyMaxRangeM`, 80 m) should stay: it is a validity bound on presses, not a model, no correct press in
internal_4 was past 57 m, and lowering it can only refuse genuine ones (the overrides show how often the bound bites).

Checks of the fit:

- **Latency** 0.5, 1 or 2 s changes d50 by under 2 m.
- **Per condition** (one shape, a rate per condition: proportional hazards), the rate up close is 0.37/s for the single
  drone, 0.58/s for the swarm and 1.18/s for the sphere, d50 30 m; one rate for all is not rejected (likelihood ratio
  p = 0.07). The sphere's six events are one pilot.
- **Scene.** SIMI flew the scenic city, whose pedestrians the bench found a quarter more visible: within the swarm, SIMI's
  rate is the plain city's (0.59 against 0.58), so the scene does not explain the sphere's.
- **Pilots differ**: ERKS recognised only up close (d50 12 m), ALEX at 27 m, SIMI at 41 m. With three pilots that is
  the size of the person-to-person spread the retest has to absorb.
- **On the screens' basis** (every shown feed, wherever the head was: perfect attention) the pooled d50 is 39 m and the
  rate up close 0.16/s; the swarm's per-second rate there is a third of the others' (it has the most screens showing
  the hat while the pilot looks at one).

### 3.2 The crowd: noticed at about 0.1 per second out to ~240 m

`views_human discovery`. For every leg between goals (42), the distance within which at least one of the goal's
walkers was expected on the head's screens (discovery's expected walkers round the ring's footprints, a clear line,
every 0.5 s), and the moment the pilot turned onto the goal for the final approach (analyse.py's commit: from there to
the zone the course stayed within 45° of the goal). 13 legs began already on the final approach (the next crowd had
been noticed during the previous search), so they are interval-censored; the 15 others are events at the turn less
1 s. The noticing hazard, fitted the same way with the censoring in the likelihood, is flat at **about 0.1 per second
out to ~200 m and halves at 236 m** (95% 154–381 m; k 14). Turned the other way: on the 20 legs with a turn, the turn
came a median 1 s after the crowd was first within 120 m and 24 s after it was first within 200 m; 40% of the turns
came before it was ever within 120 m.

**Decision: the simulation notices a goal's crowd at this rate; the single-distance equivalent is 240 m (150 and 300 m
beside it), against S1–S2's 80 m.** Crowds past 350 m are not drawn at all (`WalkerPatrol.SleepDistance`).

## 4. The conditions and the simulated pilots

`views.py`. Every condition flies the replica (ContactSim, the trial scene's own SwarmManager: since S11
BlockShapedPatchCityWorld's law with d_ref 2.0, the ellipse, the slide and the adaptive spread with Hold), in the trial
city (`BlockShaped__bbB_p40_s18_patch150_j0.15_sight487_seed1`, the layout both BlockShapedPatchCityWorld and
ScenicPatchCityWorld fly), from the spawn, with the start line 30 m ahead.

| | cameras | what the pilot sees | altitude, gimbal (between / at a goal) |
|---|---|---|---|
| **swarm** | ten drones; the hull drones' feeds (`display.FeedLatch`: the debounce, the heading low-pass, the 75°/s turn, the ready latch) | each feed on a 60° screen at its drone's yaw round the pilot; the head sees ±50° | 12.2 m, −2° / 10.4 m, −5° |
| **single** | one drone, one forward camera | its one screen in front | 7.9 m, −4° / 7.0 m, −14° |
| **sphere** | one drone, a 360 camera | the sphere; the head sees ±50° of the city round where it faces | 19.8 m / 14.3 m (no gimbal) |

The altitudes and pitches are internal_4's medians per condition and phase. A session is the whole task (§2): three
goals drawn by `BlockGoals`' rule from a session seed, 16 walkers per ring building at 2 m/s 1.1–1.5 m off the facade,
the hat on a random walker of a random ring building. **Every condition of a session faces the same goals, hats,
walkers and the same two random thresholds per goal** (an exponential draw that the integrated noticing and
recognition hazards must reach): common random numbers, so every comparison is paired.

**The pilots** are built from what internal_4's pilots did in each condition (`views_human behaviour`, `head`), as
the trajectories show (every at-goal track of the series is in `figures/tracks.png` beside the model's):

- **Between goals, every condition**: a walk on the street network (the people flew over the streets 95–100% of the
  time), persistent (a turn's weight `exp(2 cos θ)`) and drawn towards streets not yet flown near (exploration), until a
  crowd is noticed; then the street route to that goal's nearest corner. A goal noticed while another is being searched
  is remembered and flown to next (13 of internal_4's 42 legs began already on the final approach).
- **At the goal, by condition** (each goal's search runs until its hat is recognised):
  - *swarm*: to the centre building and a hold on it (the formation wraps it), 12–22 s, the rig panned 45–135° now and
    then; then the circuit of the goal block's corners at full stick, pausing beside the building half the time, and
    so on (S11's `search` flight, calibrated on ERKS). internal_4: 75% of the at-goal time inside the ring, a median
    22 m from its centre, stopped 48%.
  - *single*: round the ring buildings one by one in order: half of them orbited part of the way (half to all of the
    loop, 9 m off the facade, the camera turned in on the building 60% of the time: ALEX's way), the rest passed along
    one or two of their block's streets looking ahead (ERKS's way); after half the visits a 6–14 s stop turning on the
    spot. internal_4: 47% inside the ring, a median 48 m from the centre, stopped 21%, yawing (over 10°/s) 39%.
  - *sphere*: a walk over the goal's own junctions, least visited first, stopping 3–8 s at 15% of them. internal_4
    (SIMI): 48% inside the ring, 45 m from the centre, stopped 9%, the head a median 33° off the drone's heading.
- **The head**: a recorded head trace of that condition and phase (its offset from the view: the single drone's or
  the sphere drone's heading, the swarm's body yaw), played from a random point and on into another at its end (a block
  bootstrap of internal_4's 0.1 s traces), so the head keeps the timing it was flown with. The yaw stick turns every
  view at 40°/s.

**Counterfactual bases** (no extra flying): at every 0.2 s check during a goal's search, the recognition hazard is
also evaluated, and integrated over the same window, for the same flight seen another way:

| condition | basis | what it isolates |
|---|---|---|
| swarm | `centre`: the same shown feeds (yaws, screens, head) moved to the formation's centroid | **distributed viewpoints**: the swarm's cameras and display without their spread |
| swarm | `sphere_c`: a 360 camera at the centroid, the sphere's head | one drone with a 360 camera flying the swarm's path |
| swarm | `screen`: every shown feed, wherever the head was | attention (perfect) |
| sphere | `fwd`: one forward camera on the sphere drone's heading | all-round coverage on the sphere's own flight |
| single | `sphere_s`: a 360 camera on the single drone's flight | coverage on the single drone's flight |

Read as recognitions per minute at the goal. `real` against `centre` on the swarm's flights is what spreading the
same cameras adds; `real` against `sphere_c` is what the swarm's views add beyond one 360 camera on the same path,
the study's question in its cleanest form. **Score them with the probe** (`--probe 120`: every goal searched for a
fixed 120 s): integrated over the search as flown, the window closes when the condition's own view recognises the
hat, which biases every comparison towards it (§6.4).

## 5. Measures and statistics

- **Primary: the task time**, start line to the third press, as the restricted mean to the 900 s cap (sessions not
  done by then are censored; a swarm under four drones is censored where it stopped, the conservative reading of S2),
  swarm ÷ sphere and swarm ÷ single, paired over sessions, with a 95% bootstrap interval over sessions.
- **Secondary**: per goal, the time at the goal and the time between goals (geometric means over goals found, paired
  by goal); the share of sessions done within the cap; drone losses as events (a colliding pair is one) with exact
  Poisson intervals (lesson 46); the recognition distance and the noticing distance.
- **Mechanism**: the counterfactual bases' recognitions per minute at the goal (§4), ratios over the same windows.
- **Units**: a session is the unit (its three goals share a city and a pilot); the held-out cities are reported per
  city (lesson 13), never pooled into the trial city's verdict.

## 6. Initial tests

All in the trial city, on this machine (`stitching` env, 26–28 worker processes; a swarm session costs ~130–200 s of
one core, 90% of it the replica's physics).

### 6.1 What internal_4 says (`views_human`)

The fits of §3, and per condition (`behaviour`, `head`, `goal_times`):

| internal_4, per goal | one drone | swarm | 360 sphere |
|---|---|---|---|
| goals (runs, pilots) | 15 (5, 2) | 21 (7, 3) | 6 (2, 1) |
| at the goal: median / geometric mean (s) | 100 / 72 | 41 / 42 | 46 / 36 |
| between goals: median / geometric mean (s) | 54 / 44 | 56 / 60 | 37 / 34 |
| at a goal: speed, stopped, yawing over 10°/s | 4.7 m/s, 21%, 39% | 2.5 m/s, 48%, 16% | 5.1 m/s, 9%, 38% |
| at a goal: inside the ring, median distance from its centre | 47%, 48 m | 75%, 22 m | 48%, 45 m |
| altitude, gimbal at a goal | 7.0 m, −14° | 10.4 m, −5° | 14.3 m, (none) |
| the head's offset from the view at a goal: median, share beyond 50° | 3°, 0% | 19°, 23% | 31°, 29% |

Geometric means count a per-goal time under 10 s as 10 s (a press made from outside the zone is 0 s at it). The
people's swarm ÷ one-drone ratio at the goal is 0.59 (95% 0.36–0.95, resampling runs) and swarm ÷ sphere 1.20
(0.77–1.81). Run-to-run, the SD of log task time within a pilot and condition is 0.34 (8 df).

### 6.2 Calibration (`views calib --name calib1`): no scripted model passes

24 sessions in every condition under each of the three recognition models (§3.1; 216 sessions):

| model | at the goal (s): swarm / one drone / sphere | swarm ÷ one drone, ÷ sphere | between (s) | the gate (8 checks) |
|---|---|---|---|---|
| internal_4 (95%) | 42 (30–55) / 72 (49–110) / 36 (28–45) | 0.59 (0.36–0.95), 1.20 (0.77–1.81) | 60 / 44 / 34 | |
| one rate for every display (geometry) | 81 / 68 / 80 | 1.19, 1.02 | 68 / 79 / 63 | 3 of 8 |
| a rate per condition (proportional hazards) | 71 / 94 / 51 | 0.75, 1.40 | 69 / 76 / 63 | 4 of 8 |
| every shown screen, any head | 43 / 133 / 43 | 0.32, 1.00 | 69 / 74 / 71 | 4 of 8 |

What the gap is (and is not):

- **Not the process.** At a goal the simulated pilots fly like the people: one drone 4.4 m/s, stopped 22% (people
  4.7, 21%); swarm 2.9 m/s, 36%, 17 m from the centre (2.5, 48%, 17 m); sphere 5.3 m/s, 10%, 45 m (5.1, 9%, 46 m).
  The single drone yaws more (64% against 39%), the swarm sits inside the ring more (95% against 75%).
- **Not attention, for the swarm.** The share of the hat's on-screen time that fell inside the head's view is 0.30
  for internal_4's swarm pilots and 0.31 for the model's (their recorded head traces, replayed); the sphere's pilot
  did better than chance (0.49 against the model's 0.34).
- **It is how fast the hat is brought into close view.** Per goal, the people's head-basis recognition rate (the
  fitted hazard integrated over the at-goal time) was a median 1.2/min with the swarm and 1.1/min with one drone; the
  model's is ~0.5/min in both. People steer and look towards what they are searching for; a resampled track does not.
  So the same per-second rate gives the model too long a search, and every model is slower than the people mostly
  from its tail (sessions that hit the 900 s cap: one drone 50%, swarm 17–42%).
- **Only the per-condition model orders the conditions as the people did** (one drone slowest at the goal, the
  sphere fastest), because its rates absorb the per-condition efficiency the people showed.

**Two variants are carried into the pilot run** (§6.3): `pooled` (*geometry*: the same eyes for every display, the
scripted pilots: what the views offer, the mechanistic answer) and `cal` (*calibrated*: the per-condition model with
its recognition rate multiplied by 1.67 / 1.31 / 1.43 and its noticing rate by 1.15 / 1.72 / 1.88 for the swarm, one
drone and the sphere, the ratios of its times to the people's; `views.cal_factors`). The calibrated variant matches
internal_4 by construction, so its condition ratios are internal_4's carried to many more goal draws, not an independent
prediction. **The view decomposition (§4's counterfactual bases) does not depend on the variant**: every basis on a
flight shares the flight's per-second rate, so a scale factor cancels from their ratios.

### 6.3 The pilot run (`views pilot --name pilot1`): what the comparison looks like

40 fresh sessions in the trial city (k 200–239), every condition, both variants; and the calibrated variant on 10
sessions in each of the three held-out cities (k 300–309). Restricted mean task time to the 900 s cap, 95% bootstrap
over sessions; at and between goals, geometric means over the goals found.

| swarm ÷ … | calibrated | geometry |
|---|---|---|
| **360 sphere: the whole task** | **1.10 (0.99–1.23)** | **1.11 (0.99–1.25)** |
| sphere: at the goal | 1.22 (1.01–1.48) | 1.06 (0.84–1.33) |
| sphere: between goals | 1.11 (0.97–1.29) | 1.05 (0.88–1.24) |
| **one drone: the whole task** | **0.82 (0.71–0.93)** | **1.01 (0.89–1.14)** |
| one drone: at the goal | 0.78 (0.65–0.96) | 0.86 (0.69–1.10) |
| one drone: between goals | 1.12 (0.98–1.31) | 1.11 (0.94–1.32) |

| | calibrated: swarm / one drone / sphere | geometry |
|---|---|---|
| task, restricted mean (s) | 535 / 653 / 487 | 654 / 649 / 589 |
| sessions done within 900 s | 88% / 68% / 88% | 75% / 70% / 83% |
| drones lost (40 sessions) | 14 | 26 |
| gate checks passed | 7 of 8 (the sphere's between-goal time, whose people's interval rests on 2 runs, 32–36 s) | 5 of 8 |

Held-out cities (calibrated, 10 sessions each), swarm ÷ sphere and swarm ÷ one drone: seed 2 1.24 and 0.77, seed 3
1.40 and 0.93, seed 4 1.15 and 0.86: the same direction as the trial city.

**Reading.** Flown the way internal_4's pilots flew them, the swarm and the 360 sphere are close on the whole task,
with the sphere ahead by about a tenth (the interval just reaches 1). The swarm beats one drone with the ordinary
camera by about a fifth in the calibrated variant, all of it at the goal; with the same eyes for every display (geometry)
it does not. Between goals the swarm is the slowest of the three in both variants: it flies slower (internal_4: 5.2 m/s
in transit against 6.7–7.0 for the single drones) and loses drones (14–26 in 40 sessions, none for one drone).

### 6.4 The view probe (`views pilot --name probe1 --probe 120`): what distributed views add, without bias

The counterfactual bases of §4, integrated over each goal's search *as flown*, are biased: the search ends when the
condition's own view recognises the hat, so its window always closes on a burst of that view's exposure while the
other views' bursts are cut off. On the pilot run that said the swarm's feeds gave 2.0× the same feeds at the centroid
and 1.6× a 360 camera there, and that the sphere's 360 view gave no more than a forward camera (0.95×). **The probe
removes the stopping rule**: every goal is searched for exactly 120 s at the goal, whatever is recognised, so every
view is scored over the same window (one per-second rate for all of them, the geometry variant). 30 sessions per
condition (k 600–629), 262 goals searched for at least 100 s:

| hat recognitions per minute at a goal (95%) | |
|---|---|
| **swarm, as flown** | **0.69 (0.56–0.82)** |
| its ten feeds moved to the formation's centroid | 0.43 (0.37–0.49) |
| **one 360 camera at its centroid** (the sphere's head) | **0.72 (0.59–0.86)** |
| 360 sphere, as flown | 1.01 (0.85–1.19) |
| its forward camera only | 0.64 (0.52–0.78) |
| one drone, as flown | 0.77 (0.62–0.93) |
| a 360 camera on its flight | 1.15 (0.98–1.34) |

| ratio on identical flights | |
|---|---|
| swarm: distributed feeds ÷ the same feeds at the centroid | 1.60 (1.26–2.02) |
| **swarm: distributed feeds ÷ a 360 camera at the centroid** | **0.96 (0.71–1.30)** |
| swarm: feeds at the centroid ÷ a 360 camera there | 0.60 (0.50–0.71) |
| sphere: 360 ÷ forward camera | 1.56 (1.41–1.76) |
| single drone: 360 ÷ its forward camera | 1.51 (1.36–1.68) |

**Reading.** On the swarm's own flights its ten spread-out viewpoints recognise the hat no faster than one 360 camera
at the formation's centre would. They do recognise it 1.6 times as fast as the same ten feeds from one point: spreading
the cameras round the centre building recovers exactly what a ring of ten screens loses against a sphere at one point
(the screens leave gaps between the hull drones' fields of view, show each feed's central 80% and compress the view by
0.8, where the sphere shows every bearing at full size). On a single drone's flight, all-round coverage is worth ~1.5×
over the forward camera, whether that flight is the sphere pilot's or the single-drone pilot's. As flown, the sphere
recognises the most per minute at a goal (1.0 against the swarm's 0.7): its pilot tours the ring buildings at 5 m/s and
gets nearer to more of them than a swarm holding at the centre building.

**So, from the initial tests: the swarm's distributed views buy no more than a 360 camera at its centre, and the swarm
is flown in a way that costs it time between goals and drones.** The comparison the human retest makes is predicted to
be close (§6.3: the sphere ahead by about a tenth).

### 6.6 Follow-ups after review (2026-11-18): the wrapped swarm, the 360 stand-in, verifying before pressing

Three questions from the experimenter, each answered with a run (`follow1`–`follow3`; `views_report --follow`):

**Should the swarm always wrap round the centre building while searching a goal?** It was not doing so: the scripted
pilot held on the centre for 12–22 s and then circled the goal block's corners, wrapped (no gap of 180° or more in the
drones' bearings round the building) ~30–70% of its at-goal time; internal_4's pilots wrapped 35% of it (ERKS 51%, SIMI
24%, ALEX 3%). A `wrap` pattern (`swarm_pattern="wrap"`: to the centre and held there, the rig panned) stays wrapped
85–90% of the time. In the probe it recognises **1.33 hats a minute at a goal against the circling swarm's 0.69** (the
360 sphere as flown: 1.01).

**The 360 stand-in was in the wrong place.** `sphere_c` sat on the formation's centroid, which for a swarm wrapped
round a building is inside it (and so sees nothing), and for the circling swarm a median 17 m from the building's
centre. `sphere_d` puts the same 360 camera on the drone nearest the centroid, always outside a building:

| hat recognitions per minute at a goal (probe, 120 s) | swarm's own screens | a 360 camera on its most central drone | ratio (95%) |
|---|---|---|---|
| circling the goal's block | 0.69 | 1.10 | 0.63 (0.49–0.81) |
| **wrapped round the goal building** | **1.33** | **0.85** | **1.56 (1.24–1.98)** |
| circling, first-sight recognition (below) | 1.48 | 2.49 | 0.59 (0.46–0.76) |
| **wrapped, first-sight recognition** | **2.82** | **1.94** | **1.45 (1.16–1.82)** |

**So the swarm's spread-out views are worth about 1.5× one 360 viewpoint at a goal, but only while it wraps the goal
building; circling instead, its screens do worse than one 360 camera on one of its own drones.** §6.4's "0.96×" was the
circling pattern compared with a stand-in that was partly inside the building, and is superseded.

**Did pilots spot the hat, then fly in to check before pressing?** Sometimes. In 11 of 41 presses the hat stayed on the
head's screens for 3 s or more while the camera closed in by 10 m or more (ERKS's single drone 102 → 36 m, SIMI's
sphere 100 → 58 m); most presses were quick (the final sighting a median 2.2 s, 30 → 24 m). The press-credited fit reads
an approach as unsuccessful exposure and the recognition as made up close. Credited instead to the start of that final
sighting (`views_human.exposure_events_start`, recognition model `start`): 1.57/s up close, half at 24 m (95% 11–35),
k 3.5, a tenth at 46 m; about three times the rate at every distance, a little less steep, the range the same. The two
credits bound the truth. With `start` the model's time at a goal matches the people's without any per-condition factor
for the swarm (40–45 s against 42) and the sphere (41 against 36); one drone is now faster than its pilots (50 against
72), and between goals every condition is still slower than the people (69–77 s against 34–60).

**A flatter distance weighting** (the press-credited d50 × 1.5, half rate at 46 m) leaves the ratios where they were
(the circling swarm against the centroid camera 0.96, the wrapped swarm ~1.3× the sphere as flown): the steepness of
the curve is not what decides the comparison; where each is flown is.

**The whole task with the swarm wrapped** (40 sessions, paired with the pilot run's sphere and single sessions):

| swarm (wrapped) ÷ … | first sight | press-credited (geometry) | calibrated |
|---|---|---|---|
| **360 sphere, task** | **1.04 (0.93–1.16)** | 1.03 (0.91–1.16) | 1.02 (0.87–1.22) |
| sphere: at the goal / between | 0.97 / 1.12 | 0.91 / 1.11 | 0.93 / 1.16 |
| **one drone, task** | **0.90 (0.77–1.04)** | 0.93 (0.82–1.06) | 0.76 (0.66–0.86) |

The wrapped swarm and the sphere come out even over the task: the swarm gains a little at the goal and loses it between
goals, where it is slower and is the only condition that loses drones. Against one drone the swarm is faster or level.

**What changes in the proposal.** The simulated swarm wraps the goal building (the `wrap` pattern) as the main
condition, with circling as the sensitivity; the 360 stand-in is `sphere_d`; recognition is carried in both credits
(press, first sight), and first sight is the better-calibrated one. For the human trials it suggests the instructions
should keep telling pilots to wrap the centre building (the participant video already does), and that the retest's
analysis should log how much of each goal's search was wrapped, since that is what decides the swarm's advantage there.

### 6.7 The 360 drone flown the single drone's way (2026-11-18, `follow4`)

The experimenter's suggestion: fly the 360 drone exactly as the single drone is flown, so that only the camera
differs (`fly_as="single"`: the single drone's ring-building orbits and passes, its stops). At the sphere's own
altitude (14 m at a goal) and at the single drone's (7 m, `band_as="single"`); first-sight recognition, the same 40
goal draws as the swarm and single drone of `follow2`; and the 120 s probe.

| ratio of task times (95%) | sphere flown as single, 14 m | sphere flown as single, 7 m |
|---|---|---|
| wrapped swarm ÷ it | 0.92 (0.81–1.05) | 0.98 (0.87–1.11) |
| it ÷ the single drone (the camera alone) | 0.97 (0.84–1.11); at the goal 0.88 | 0.92 (0.77–1.07); at the goal 0.78 |
| it ÷ the sphere as SIMI flew it | 1.13 (1.03–1.23) | 1.06 (0.95–1.17) |
| probe, recognitions a minute at a goal: it / its forward camera | 1.07 / 0.82 (first sight 2.63 / 2.14) | 1.18 / 0.83 (3.10 / 2.16) |

The wrapped swarm at the probe: 1.33 a minute (first sight 2.82); the single drone 0.77 (1.97).

**Reading.** On the single drone's own paths the 360 camera is worth ~1.3–1.45× in recognitions at a goal and 12–22%
of the time at a goal, but only 3–8% of the whole task, because between goals nothing changes. Flown this way the 360
drone is no faster than SIMI's own style (SIMI's touring of the goal's junctions was as good or better), and lower is
better for it (7 m beats 14 m: the hats are nearer). Against the wrapped swarm it is even at 7 m (0.98) and the swarm is
slightly ahead at 14 m (0.92). **Proposed for the main study: the 360 drone flies the single drone's pattern at the
single drone's altitude as its primary condition (a pure camera swap), with SIMI's style as the sensitivity.**

### 6.5 What the initial tests cost, and what the main study will

The calibration batch (216 sessions) took 20 min, the pilot run (330) 52 min, the probe (90) 8 min, on 26–28 workers.
The main study of §7 (300 + 100 probe sessions per condition in the trial city, both variants, the sensitivities and
three held-out cities) is ~2,600 sessions, about 7 h on this machine.

## 7. The main study (simulation), as proposed

0. **Rework the simulated at-goal search first.** The initial tests' scripted pilots fly like internal_4's pilots but
   find the hat at about half their rate with the swarm and one drone (§6.2). The candidate is a pilot that steers and
   looks towards the ring buildings whose walkers it has not yet seen (the attention the people show, which a resampled
   head trace lacks), checked against the gate below. Until then the calibrated variant stands in.
1. **Calibration gate, before anything else** (`views calib`): the model passes when, per condition, its geometric-mean
   time at the goal and between goals lie inside internal_4's 95% bootstrap intervals (over runs), and its swarm ÷
   sphere and swarm ÷ single ratios at the goal inside theirs. The recognition model the main run uses is the one that
   passes (§6.2 says which did in the initial tests); if none does, the closest, with the gap reported (lesson 38's
   named fallback). `rule.json` holds the gate and the verdict rule below, written before the main run.
2. **The trial city**: 300 fresh sessions (goal draws k = 1000–1299, none used by the initial tests), the three
   conditions, both variants (geometry and calibrated, or whichever passes the gate after the at-goal search is
   reworked); and **the view probe** on 100 more (k 1300–1399), which decides the distributed-views question (§6.4).
   ~4 h on this machine (§6.5).
3. **Sensitivities** on 100 of those sessions each: the other two recognition models; the hat's d50 × 0.75 and × 1.33;
   the crowd's d50 at 150 and 300 m; the walk without its exploration term.
4. **Held-out cities** (the same design at seeds 2, 3 and 4, S6's held-out set): 100 sessions each, reported per city.
5. **A Unity spot check**: 12 sessions per condition re-flown on the headless bench from the replica's own pilot
   routes (waypoints and holds, as S11's bench flew its searches), Unity's recorded tracks scored with the same hazards,
   beside the replica's twins (lesson 48).
6. **The verdict rule** (distributed views beyond 360 coverage), on the trial city's task time: swarm ÷ sphere with its
   95% interval wholly below 0.9 — the swarm is at least 10% faster; wholly above 1/0.9 — the sphere is; inside
   [0.9, 1/0.9] — equivalent to within 10%; otherwise undecided at this size. The same for swarm ÷ single. The
   decomposition (`real` ÷ `centre`, `centre` ÷ `sphere_c`) is reported beside it, not tested.
7. **The page** follows the study pages' conventions: the purpose and the answer first, a Cities tab with the trial
   city, a goal drawn with what each term means, and replays of representative sessions per condition (people's and
   the model's at-goal tracks beside each other), the key to the codes, the back link to the Swarm Study Index.

## 8. The human retest (swarm against the 360 sphere), as proposed

- **Scene**: ScenicPatchCityWorld for both conditions (the sphere's only flights are there, and its pedestrians stand
  out more than in the plain scene, so the series must not mix scenes); practice in ScenicPracticeBlockWorld, to
  criterion in each condition (two hats within 240 s of entering their zones, protocol §4), unrecorded as now.
- **Design**: within subject, four runs per pilot in an ABBA or BAAB order (half each), which balances a linear
  practice effect inside every pilot. SIMI flew the swarm first and both swarm runs before both sphere runs, which is
  the order confound internal_4's page warns of. Fresh goals each run (BlockGoals draws them at every Play; the
  session records the seed).
- **Primary outcome**: the task time, log-transformed, in a mixed model `log T ~ condition + position + (1 | pilot)`;
  the effect is exp(β) with its 95% interval. Runs ended at the cap are censored (a censored-normal fit). A swarm run
  that falls under four drones is kept and flagged; a sensitivity drops it.
- **Secondary**: the time at the goal and between goals per goal; the press distance; drone losses; NASA-TLX and a
  simulator-sickness questionnaire after each condition (the sphere's comfort is unknown: a reviewer of the DJI Avata
  360 "didn't feel too good after a full battery").
- **How many pilots**: from internal_4, the run-to-run SD of log task time within a pilot and condition is 0.34
  (8 df) and the pilots' own swarm ÷ single ratios spread by ~0.3 in log beyond that. With two runs per condition the
  per-pilot SD of the log ratio is ~0.46, so 80% power at α 0.05 (a t test over pilots) needs **35 pilots for a 1.25×
  difference, 17 for 1.4×, 12 for 1.5×, 8 for 1.75×** (one run per condition: 54, 25, 18, 11; three: 29, 14, 11, 7);
  The simulation predicts swarm ÷ sphere ≈ 1.10 (§6.3), which would need ~180 pilots to detect: **no retest of a
  feasible size can resolve a difference that small.** Size it for the smallest difference worth acting on (1.4×: 17
  pilots with two runs per condition), report the ratio with its interval rather than a test, and read it beside the
  simulation's decomposition, which says where the time goes.
- **Swarm losses** are this series' other open problem (internal_4's page: the stick shield braking one drone at a
  facade while its partner flies on). SIMI's second swarm run lost six drones before its first goal. The retest
  should state in advance whether a run is restarted after a technical failure (protocol §8) and keep the losses as a
  measure.

## 9. Lessons applied

| lesson | how |
|---|---|
| 2 a horizon saturates | task time as a restricted mean to a cap no session-to-be-compared reaches often (§6), not "found within" |
| 3 the view from the start | the goals are BlockGoals', ≥ 100 m from the spawn, and the clock starts at the line |
| 4, 37 ranges set the effect | both ranges fitted to the trials, for what each stands for (a hat, a crowd); sensitivities on both |
| 5 a glimpse is generous | recognition is a rate per second on screen, not a first glimpse |
| 6 the goal and the transit | every goal split into between and at; the mechanism measured at the goal |
| 8 scripted pilots are not people | the pilots are built from internal_4 per condition, and the calibration gate checks them against it |
| 9 pilot paths | street walks: the people flew over the streets 95–100% of the time |
| 11, 48 validate in the city | the trial city, which S6–S11's twins validated; a Unity spot check with twins |
| 13, 14 units, pairing | sessions are the units; common random numbers across conditions |
| 15 the rule first | `rule.json` before the main run |
| 16–18 this machine | `parallel.run_jobs`, resumable `runs.jsonl`, no edits under a running pool |
| 27 seeing is not finding | the pilot acts on what it sees (it searches a goal until the hat is recognised); the mechanism decomposed |
| 30, 31, 36 pages | §7.7 |
| 46, 53 losses | events with exact intervals, located |
| 49 a probe for the question | the counterfactual bases put the distributed-views question to the same flights |

## 10. Decisions for the experimenter

1. **The task**: the whole internal_4 task (§2), not a point target. Agree?
2. **The model**: carry both variants (geometry, the mechanistic answer; calibrated, which matches internal_4 by
   construction) into the main run, or first rework the at-goal search until one passes the gate (§7 step 0). The
   probe's view decomposition does not depend on the choice.
3. **The ranges**: 47 m for the hat and 240 m for the crowd wherever a single distance is needed, replacing S1–S11's
   40 m and 80 m in this study only. The recorder's 80 m scoring limit unchanged.
4. **The human retest**: ScenicPatchCityWorld, ABBA/BAAB, the sample size of §8 and §6.3.
5. **Whether the single drone joins the retest** as a third arm (it would make the swarm ÷ single ratio a human result
   too, at a third more session time).
