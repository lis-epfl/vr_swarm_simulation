# S12 Distributed Views

**Code name:** the views study · **Dates:** 2026-11-17 (proposal and initial tests) · **Status:** proposed; the
initial tests are done, the main runs wait on the experimenter's review · **Builds on:** S2 (searching a city for a
goal it does not know, the paired arms), S11 (the goal search flight, calibrated on ERKS), S6–S11 (the trial scene's
law and the replica's validation in its city), internal_4 (every fitted parameter), and the 360 camera
([docs/single_drone_360_camera.md](../single_drone_360_camera.md))

**Question.** How much faster does the swarm do the trials' task than one drone with the ordinary camera, and than one
drone with a 360 camera shown on a sphere round the pilot, and where does the difference come from? The second ratio is
what the swarm's distributed viewpoints add beyond all-round coverage; it is the comparison the next human trials
(the swarm against the sphere) will measure.

**Design.** The manual is [docs/views_study/README.md](../views_study/README.md).
- **The task is internal_4's whole task**, not a point: three goals drawn from the trial city's buildings by
  BlockGoals' rule, their rings' pedestrians walking, the hat on one; the clock from the start line to the third press,
  each goal split into between and at the goal. At the goal is where the 360 camera's single viewpoint and the swarm's
  ring differ, and it was 41–65% of internal_4's per-goal time.
- **What a pilot sees is fitted to internal_4** (`views_human`): the hat is recognised at a rate that is 0.49/s up
  close and halves at 31 m (95% 20–39 m), a tenth by 47 m, on screen and in the head's view (41 presses, 350 s of
  exposure); a goal's crowd is noticed at about 0.1/s out to ~200 m, halving at 236 m. These replace S1–S11's 40 m and
  80 m in this study.
- **Three conditions** on the replica in the trial city, each flown as internal_4's pilots flew it (street walks
  between goals; at a goal the swarm wraps the centre building, the single drone works round the ring buildings, the
  sphere walks the goal's junctions; recorded head traces), with common random numbers; **counterfactual bases** on
  the same flights (the swarm's feeds moved to its centroid; a 360 camera at its centroid; the sphere with only a
  forward camera).
- A calibration gate against internal_4 before the main run; 300 sessions in the trial city, sensitivities, three
  held-out cities, a Unity spot check; the human retest's design and size.

**Findings, revised after review (2026-11-18; manual §6.6)**, which supersede the first list where they differ:
- **Wrapped round the goal building, the swarm's spread-out views recognise hats ~1.5× as fast as one 360 camera on its
  most central drone** (1.56, 95% 1.24–1.98; 1.45 with first-sight recognition). Circling the goal's block they manage
  0.63×. The first list's "0.96×, nothing over a 360 camera" compared the circling swarm with a stand-in partly inside
  the building.
- **Over the whole task the wrapped swarm and the 360 sphere come out even** (1.04, 0.93–1.16, first sight); the swarm
  gains at the goal and loses between goals. Against one drone it is faster or level (0.90, 0.77–1.04).
- **11 of 41 presses followed an approach to check the hat.** Crediting recognition at first sight triples the rate
  at every distance, flattens it a little and leaves the range (~46 m to a tenth); it brings the model's time at a goal
  to the people's for the swarm and the sphere.

**Findings** (the initial tests as first reported; manual §6):
- **On identical flights the swarm's spread-out viewpoints add nothing measurable over one 360 camera at its centre**:
  0.96× (95% 0.71–1.30) its hat recognitions per minute at a goal (the view probe: 30 sessions, 262 goals each searched
  for exactly 120 s). They are worth 1.6× (1.3–2.0) over the same ten feeds from one point, which is what a ring of
  screens loses against a sphere. On one drone's own flights, all-round coverage is worth ~1.5× a forward camera.
- **Flown as each condition was flown in internal_4**, the swarm is about a tenth slower than the 360 sphere on the
  whole task (1.10, 95% 0.99–1.23, calibrated; geometry 1.11) and about a fifth faster than one drone (0.82, 0.71–0.93,
  calibrated; geometry 1.01). Three held-out cities agree in direction. The swarm is slowest between goals and the only
  condition that loses drones (14–26 in 40 sessions).
- **The scripted pilots fail the calibration gate**: they fly like internal_4's (speed, stops, where at a goal) but
  bring the hat into close view at about half the people's rate with the swarm and one drone. A calibrated variant
  (per-condition efficiency factors) passes 7 of 8 checks, by construction.
- **A retest of feasible size cannot resolve a 10% difference** (~180 pilots); 17 pilots with two runs per condition
  detect 1.4×.
- The as-flown view comparison is biased by the stopping rule (it said 1.6× over the 360 camera); lessons 54–56 and the
  probe are the corrections.

**Limits and refinements.** internal_4 is three pilots and 42 goals; the sphere is one pilot (six goals) in the scenic
city. The simulated head is a recorded trace, not attention to what is on screen. The replica is 2-D (right for 50 m
prisms and drones under the roofs).

**Open decisions.** The manual's §10.

**Where.** Manual [docs/views_study/README.md](../views_study/README.md); code `citysim/views_human.py`,
`citysim/views.py`, `citysim/views_figs.py`, `citysim/views_report.py` (+ `.html`); outputs
`experiment_results/views_study/` (`human/`, `calib1/`, `pilot1/`, `probe1/`, `views_proposal.html`); the proposal page
https://claude.ai/artifact/CpzLU87XCffSTVdQDopp6j (`python -B -m citysim.views_report --calib calib1 --pilot pilot1
--probe probe1`, then publish `views_proposal.html`).
