# S1 Goal Visibility

**Formerly / code name:** the density study (with its task-independent capability study) · **Dates:** 2026-10-12 →
10-19 · **Status:** done; trial level L3 on hold (G3 fails on pilot paths) · **Builds on:** internal_3 (human runs)

**Question.** Which cities make the ten-drone swarm better, or worse, than one drone at spotting a target pedestrian
among walkers at a goal? Pick three trial levels for the human trials by a rule fixed in advance.

**Design.** Generated Voronoi cities (`ProceduralCityWorld`) of one ScaledCity design at twice its size, 50 m tall,
crossing clutter (C0 open, C1/C2 packed; porous C1p/C2p for the capability part) with goal geometry (G0–G10: an
occluder of width w_c ringed by patrolled buildings at radius r_g). Four model tiers: A, a Boolean free-sight model;
B, exposure at a goal (share of walker-time recognisable within 40 m on the shown feeds, the headset's ±50° view
weighted by internal_3's head directions), the swarm settled by the 2-D flight-controller replica, the single drone
on scripted policies; B', the replica flying transits; C, the headless Unity bench. Transits fly pilot paths since
2026-10-13 (the street route was 1.35–1.64× the straight line). Gates G1–G3 and a selection rule picked the levels.
Discovery (an unannounced goal's first sight on a shown feed within 80 m) was added as a co-primary for the trials.

**Findings.**
- **At a goal, its geometry decides who sees more and clutter barely matters.** Swarm − single exposure: +0.048 at
  G0 (7 m occluder, 26 m ring; today's goal patch), +0.067 at G2 (13 m, 30 m; VoronoiCityWorld's goals), +0.106 at
  G10 (28 m, 48 m), all with the swarm surrounding the occluder; +0.002 at G6 and −0.047 at G7, cramped goals
  (13–16 m clearance) it cannot get round, where about 4 of 10 feeds are useful. G0's advantage is +0.048 / +0.047 /
  +0.050 in the open, medium and dense city.
- **Trial levels** (selection rule on the street route): L1 C0-G0, L2 C0-G10 (advantage 2.2× L1's), L3 C2-G7
  (reverses). On the pilots' paths L3 loses 2.99 drones per run against G3's bar of 2.5 (7.0 per flight at spread
  0.4, 1.0 at 1.0), so the rule picks no L3 there: on hold (manual §1, §8).
- **Unity agrees for wide goals and is closer for small ones**: at the dial-centre pitch the swarm and single drone
  tie at G0 (0.081 vs 0.090) and the swarm stays far ahead at G10 (0.156 vs 0.034).
- **What each sees, task-independent** (the explorer's "What each sees" tab): the swarm puts 5.7–6.3× the ground on
  screen, 1.6–1.8× in the headset's view and 1.9–2.1× within one second, and its sweep width is 2.1–2.3× in flight.
  **Superseded by S3**, which measures this over 63 city designs with statistics, and whose sweep-width finding
  (below) applies here too.
- **Discovery** (pass-by model, 80 m, shown feeds): predicted swarm ÷ single discovery time 0.58–0.69 at the three
  levels, 0.71–0.84 on the headset basis.

**Limits and refinements** (in [lessons.md](lessons.md)):
- Scripted single-drone policies under-predict real pilots' exposure about 2.7× (internal_3), so the L1 margin is an
  upper bound; L2 is the robust prediction.
- The sweep widths in "What each sees" use a ±42 m lateral window and saturate for the swarm (found in S3): the swarm's
  ratios are understated, most at wide spreads.
- The street route misrepresented how pilots fly; scripted transits now fly pilot paths.
- Spread 0.4 drives the losses everywhere (replica and Unity alike).
- The x4 building-size sensitivity was not run; the power simulation waits on a pilot block.

**Open decisions.** L3: keep C2-G7 and let the pilot block decide, apply the rule as written (no L3), or recalibrate
the loss bar (manual §8). S3 adds a fourth path: floor the spread dial near 1.0, which removes most of the losses.

**Where.** Manual [docs/density_study/README.md](../density_study/README.md), trial protocol
[protocol.md](../density_study/protocol.md); code `citysim/` (`vissim.py` tiers, `capability.py`, `transit.py`,
`passby.py`, `select_levels.py`, `explorer.py`); outputs `experiment_results/density_study/`; trial layouts
`Assets/Scripts/Experiment/layouts/L1_…`, `L2_…`, `L3_…`; explorer https://claude.ai/artifact/NLNeGtKvo5vYW11Rj563qc.
