# S5 Block-Shaped Surround

**Code name:** the block-shaped study · **Dates:** 2026-10-28 to 29 · **Status:** done, in the replica ·
**Builds on:** S4 (the question, Phase 3's measures and pilots), BlockShapedCityWorld (session vr-swarm-simulation-91)

**Question.** In BlockShapedCityWorld's city, where each building fills its block to the back of a footpath, does
surrounding a block (to look down every street leaving it before choosing) give a pilot flying the swarm an edge over
one drone on a visual search? First: which spread (d_ref) gets the formation round a block efficiently. Then: do
footpaths make streets visible to one drone; which layout of the blocks puts more streets round each building, hidden
from one drone; do block size (p 35, p 30) and street width (s 20) change it.

**Design.** S4 Phase 3's flights (one block per trial, every way; a search for a goal at a building) on nine designs,
under today's force law as Unity flies the scene (each prism avoided as the circle round its bounding box), at a 250 m
sight range. Step 1 picks the spread and surround by a rule fixed in advance; every later step uses them. Manual:
[docs/shaped_study/README.md](../shaped_study/README.md).

**Findings.**
- **The scene's streets are too tight for the swarm at the usual spread.** Each building's avoidance circle bulges
  past its facade, and the formation is squeezed until drones collide: the swarm in the streets loses 0.33 drones a
  block at d_ref 1.0 (S4's catalogue city on the same streets: none), 0.13 at 1.2, none from 1.6.
- **d_ref 1.2 with a 20 s firm hold on the block's centre** is step 1b's pick: it encloses the block in 97% of trials
  and loses nothing. Flowing round (aim and turn) does not get round these blocks within 12 s at any spread. Step 1's
  first rule, with the streets held at 1.0, found nothing safe and its fallback pick was not used.
- **Block-shaped buildings hide more, and surrounding shows more.** At the decision one drone has looked down 28% of a
  block's exits (S4: 33%), the swarm in the streets 74% (S4: 87%), surrounding 95%.
- **The footpath does show one drone more** (28% at 3.5 m, 17% at 1.5 m, 11% with none), but under today's law a
  narrower footpath lets the circles close 12–33% of streets, and the swarm loses 1.3–1.9 drones a block.
- **Even layouts** (Lloyd, hexagons) put up to six streets round every building against 4.5, but have 6–7.6 exits
  against 7.9; one drone sees the same share on all (27–33%), so the scene's random layout leaves the most exits hidden
  from it. Even layouts lose fewer drones.
- **Smaller blocks and wider streets show one drone more** (45–52%).
- **Search: the swarm in the streets finds a goal at a building in 46% of one drone's time** on the scene's city
  (46–71% across six designs); **surrounding is slower than the streets on all six** (×1.33 on the scene's city,
  ×1.19–1.66): it saves 29% of the decisions but each takes 86% longer.

**For the trials.** Fly BlockShapedCityWorld as built; start the spread at 1.2 or more there (no street losses from
1.6); compare the swarm in the streets with one drone; don't ask pilots to surround blocks.

**Limits and refinements.** Scripted pilots, the 2-D replica (S5's settings not flown on the Unity bench; the scene's
smoke test lost a pair in 16 flights), Part A in three cities, Part B in eight per design. A tighter avoidance shape
in the force law (the footprint's own circle or closest point) would allow a narrower footpath and perhaps a quicker
surround; untested. Lessons 37–38.

**Where.**
- Code: `citysim/shaped.py`, `shaped_replays.py`, `shaped_report.py` (+ `.html`); S4's `lookdown.py`,
  `lookdown_search.py`, `split_probe.city_from` and `split_replays`' workers take S5's city, spread and conditions
  (defaults unchanged).
- Outputs: `experiment_results/shaped_study/` (`geometry.csv`, `dref/`, `dref2/`, `chosen.json`, `partA/`,
  `partB_<design>/`, `replay/`).
- Page: [S5 Block-Shaped Surround](https://claude.ai/artifact/PrHZmczN5CnhAvZLQTCkb9) (`python -B -m
  citysim.shaped_replays`, then `python -B -m citysim.shaped_report`; publish `shaped_study_bare.html` with the files
  `shaped_study_files.json` lists). Answer first, then the cities (every design, with the avoidance circles) and four
  replayed flight sets, every table, the terms.
