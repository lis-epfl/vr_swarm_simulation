# S11 Wrap Hold

**Code name:** the wrap study · **Dates:** 2026-11-06 → 11-08 · **Status:** done; Hold adopted into BlockShapedPatchCityWorld and PracticeBlockWorld 2026-11-09 · **Builds on:** S7 (the adaptive spread),
S9 (its settings, the loss statistics, the shadow-phase method), S10 (the scene's law, E15_35a, and its harness), and
internal_4's first flights in the trial scene (ERKS, 2026-11-06)

**Question.** The user's, from the map of ERKS's first swarm trial: as the swarm reached its second goal it widened to
surround the building, then narrowed while it was still wrapped round it. Replayed on ERKS's three swarm flights, the
adaptive spread as BlockShapedPatchCityWorld flies it held a narrowed spread for 39% of the time a building lay inside
the formation. It keeps the full spread only while the stick's line 40 m ahead, or with the stick centred the
centroid itself, passes within 0.8 of a building's radius of its centre: a centroid drifted off the building's middle,
or a pilot steering round it, narrows a formation that is still wrapped round it. Does holding the full spread while
a building lies inside the formation (`SwarmManager.spreadWrapRule`: Hold, or HoldUntilLeaving, which releases it while
the pilot steers away from that building) keep the swarm wide round the buildings it surrounds, and what does that cost
in the streets it enters next?

**Design.**
- **The rule** (`SwarmPlaneController.WrapHolds` / `InsideFormation`, `swarm_replica.wrap_holds` /
  `inside_formation`). A building is wrapped when its centre (the ellipse's under the scene's shape, else the circle
  S7's rules aim by) lies inside the alive drones' convex hull, tested as no gap of 180° or more in the drones'
  bearings round it. Hold returns the full spread while any building is wrapped; HoldUntilLeaving except while a stick
  above 0.1 points away from it (its centre behind the centroid along the stick). Off is S7's rule exactly: the replica
  re-flew 40 of S10's stored confirmation flights bit for bit.
- **ERKS's flights, open loop** (`swarmwrap human`): the rule replayed on the drones as they flew, the 3 s filter
  included, reproduces the logged scale to 0.0005. Narrowed while wrapped: 39% as flown, 9% under Hold, 23% under
  HoldUntilLeaving; narrowed in the streets 86%, 74% and 84%. That is the question and the cost the closed loop must
  settle.
- **The flights**: S8's four (straight line, through a block, along the streets, square on at a wall) and one added,
  `search`, a goal search round one building calibrated on ERKS's wrapped moments: the street circuit of the building's
  block at full stick, pausing beside it 0.4–1.2 of its radius off its centre, a hold on it, and away by the streets.
  Scripted under the scene's rule it narrows for 44% of its wrapped time (ERKS 39%), with the stick held 68% of it
  (ERKS 52%).
- **Rows** (each names every law field): the scene (E15_35a, Off), Hold (H15_35a), HoldUntilLeaving (L15_35a), and
  the spread held at 2.0 (F20, S9's), the end point of holding.
- **Measures**: on every flight, every 0.1 s, the share of the wrapped time at full spread (the question), the scale on
  leaving (the 10 s after the last building left the formation) and loss events and contacts placed wrapped / leaving
  / elsewhere with each phase's exposure (lesson 47: the phases are geometric, so every row, F20 included, carries the
  same moments); S8's speed, cohesion, losses, contacts and split; for the search, its hold window at full spread and
  the speed it leaves on.
- **Screen** on the scene's city; **rule** (`rule.json`) written after it, before the confirmation; **confirmation** on
  the scene's city and the three held out; **Unity** bench with the replica's twins gated.
- **Lessons applied**: 15 (the rule before the confirmation), 17–18 (resumable runs, no edits under a pool), 39
  (settled starts), 40 (a formation-level rule screened on turning routes: the street and search flights turn), 42 and
  47 (losses located against the spread's phases in every row), 44 (safety bounded by the scene as it flies), 45 (the
  value it moves to, held: F20), 46 (loss events with exact intervals), 48 (twins before trusting a cost), 49 (the
  question measured in its own flight: the search and ERKS's own flights).

**Findings.**
- **Hold is the rule's pick (H15_35a), and it does what was asked of the setting.** Over 4,352 replica flights in four
  cities (30 h of flight a row), the mean spread scale while a building is wrapped went 0.941 → 0.982 and the share of
  that time spent narrowed 48% → 15%. Unity's bench (142 flights a set) agrees to 0.003 (0.938 → 0.979). On ERKS's own
  flights, replayed open loop, narrowed while wrapped goes 39% → 9%.
- **It costs nothing measurable.** Loss events 45 against 45 in the replica, contacts 7% fewer, street speed 0.1%
  lower, the speed leaving a search 0.3% lower; split, hull and surrounding unchanged. Unity's Hold lost 9 events
  against 6 (p 0.30), and every one is located: none while Hold was acting (the scale at each was the scene's own to
  0.01), 7 before the flight had wrapped any building. The twins lose 8 against 6 on the same starts.
- **The formation itself barely changes.** Round a wrapped building the ring is 0.30 m wider under Hold (1.9%), 0.38 m
  in the searches (2.5%), and only 0.56 m even with the spread held at 2.0. Wrapped, the drones sit at the standoff on
  the building's avoidance shape, so the building sets the formation's size. Hold mostly removes a pulse in the
  setting; the swarm also stays round a building a little longer (36.6 → 40.0 s a search).
- **HoldUntilLeaving** (released while the pilot steers away) delivers a little over half of Hold and missed both bars
  by a hair. **The spread held at 2.0** (F20) costs 6.5% of street speed, which fails the rule.

**Limits and refinements.**
- The search flight is scripted. Calibrated against ERKS's wrapped moments it narrows a little more often and holds
  further off the building's middle than ERKS did (§4 of the manual), so it is somewhat harder on the scene's rule.
- One participant's three flights set the question and the calibration; internal_4 is the only trial data in this
  scene.
- The ring is measured as the formation's mean radius; nothing here measures what the cameras see of the building
  (S4's look-down measures would), so "what the pilot would see changes little" rests on the geometry.

**Open decisions.**
- **Adopt Hold into BlockShapedPatchCityWorld?** Decided 2026-11-09: adopted, in BlockShapedPatchCityWorld and
  PracticeBlockWorld (the manual's "Adopting it"). An empty harness override now means Hold; `swarmwrap.PRE_S11`
  restores Off. Watch the first trial sessions' swarm maps: the setting should stay at full while a building is
  wrapped, and the ring should look much as before.

**Where.**
- Page: https://claude.ai/artifact/658t8HQXHxpyqU6T7VX7CD.
- Manual: [docs/swarm_wrap/](../swarm_wrap/README.md).
- Code: `citysim/swarmwrap.py`, `swarmwrap_bench.py`; the replica's `inside_formation`, `wrap_holds`,
  `spread_target`; in Unity `SwarmManager.cs` (`SpreadWrapRule`, `spreadWrapRule`), `SwarmPlaneController.cs`
  (`WrapHolds`, `InsideFormation`), and the bench's `spread` trajectory column (`SwarmBenchRunner.cs`).
- Outputs: `experiment_results/swarm_wrap/`.
