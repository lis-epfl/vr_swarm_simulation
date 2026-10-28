# The split study

**S4 Split and Surround** in the study registry ([docs/studies/](../studies/README.md)). "The split study" is its code
name.

**Question.** Take a city of small Voronoi blocks, each holding one large building. The swarm surrounds a block, as
S1's swarm surrounds a goal's central building. Can it then show the pilot every street round the block, and reform
down the one the pilot picks, quickly enough that exploring block by block is efficient? Does that find things
faster than the same swarm kept in the streets, or than one drone? What does it cost in drones lost, in time to
reform, and in views that no longer relate to each other?

It answers S2's open question. S2 found that the swarm's search advantage came from its cameras, not from its
formation morphing round buildings, but none of S2's cities made going round a building the way to travel. It also
extends S3's trend in coverage beyond the point where S3's building mix jams.

Everything runs from `Assets/Scripts/Experiment` with the `stitching` env's Python. The code is:
- `citysim/blockcity.py`: the cities;
- `citysim/polyfoot.py`: polygon footprints;
- `citysim/streetnet.py`: the street network and the cluster metric;
- `citysim/split_probe.py` and `split_tables.py`: Phase 1;
- `citysim/split.py` and `split_stats.py`: Phase 2 (walks, informed search, H1/H2);
- `citysim/split_bench.py`: the Unity bench configs and gates (Phase 1's manoeuvres, Phase 2's walks and routes);
- `citysim/lookdown.py`, `lookdown_search.py` and `lookdown_contacts.py`: Phase 3, the look-down pilot (section 9);
- `citysim/correspondence.py`: the view-correspondence measures;
- `citysim/split_replays.py`: the page's Cities tab, every design's city and the flights it replays (section 8),
  recorded with the shared `citysim/playback.py` and played by `citysim/replay_view.js`;
- `citysim/split_report.py` + `split_report.html`: the results page;
- `citysim/tests/test_split.py`, `test_playback.py` and `test_lookdown.py`: the tests.

The replica's polygon city and closest-point obstacle model are in `Assets/Scripts/swarm/Bench/swarm_replica.py`.
Outputs go to `experiment_results/split_study/`.

## 1. Where it stands

Phases 1 and 2 done (2026-10-27); Phase 3, a short pilot, done 2026-10-28 for the user to review before any further
full study. The plan was approved on 2026-10-27, and this manual records each phase's method before its results.
- **Phase 1**, the initial test ([section 6](#6-phase-1-results)), chose variant R, with Phase 2 on p 40 with 18 and
  24 m streets. Unity's bench agreed.
- **Phase 2**, the comparison ([section 7](#7-phase-2-results)):
  - **H1 holds**: surrounding a block shows the pilot 7.1 points more of its streets at the decision than a pause
    of the same length.
  - **H2 is not supported**: an informed pilot finds a hidden target no sooner than with the swarm kept in the
    streets (×1.02, 0.92–1.12).
  - The swarm's advantage over one drone (it finds in 62% of the time) comes from its cameras.
- **Phase 3**, a short pilot after the user's review (2026-10-28; [section 9](#9-phase-3-looking-down-the-streets-a-short-pilot-2026-10-28)):
  - **Scored on looking down each exit to the building at its end**, the street-bound swarm knows 41% of a block's
    exits at p 40 and 9% at p 75. The old preview said 71% and 26%. A 12 s surround at d_ref 1.8 looks down 93% and
    17%.
  - **d_ref is a first-order factor.** The surround looks down more the wider it flies, up to where its drones touch
    the building it surrounds (1.8, or 2.2 at p 75). That is wider than the trials' dial and wider than Phase 1's d*.
  - **In a search for a goal at a building, the surround still finds it about 1.5 times as slowly as the swarm in
    the streets** (×1.48, 1.18–1.87).

Results page: [S4 Split and Surround](https://claude.ai/artifact/C1PtAjZybB7JdAkoJy2xXD), built by
`python -B -m citysim.split_report phase1`.

## 2. The cities (`citysim.blockcity`)

**The streets** are drawn the way `citygen` draws a goal-free blocks city from the same seed:
- RSA block seeds 0.8 p apart;
- Voronoi cells;
- kerbs s/2 inside each street.

So both variants of a seed share every street, and they share them with the `bl_` city of the same p, s and seed.
There are no goals. The designs are block pitch p {40, 50, 60, 75} m × street width s {12, 18, 24} m: the user's
coupling of block size, street width and separation.

**Two building variants on the same streets** (the user's two options):

| | R, cylinder-fit | B, block-shaped |
|---|---|---|
| building | `BnP_Large_Building_B` (15.2 × 17.7 m at size 1), uniformly scaled so its avoidance cylinder fits inside the block's inscribed circle less 1 m, on the block's Chebyshev centre | the kerb polygon inset 3.5 m (CityDressing's footpath): one convex prism |
| share of the kerbed area built (3 seeds, plazas counted empty) | 21–40% | 23–79% |
| share of the arena (α) | 0.03–0.28 | 0.04–0.56 |
| obstacle force | today's law, unchanged; every street keeps a cylinder gap ≥ its width | closest point (below) |

How "one building per block" breaks down at the corner of the grid:
- At p 40 with 24 m streets, about 60% of blocks are too small for a building of at least 8 m. Those are left as
  plazas, so that design is a mostly open city.
- The small-building control is therefore p 40 with 18 m streets (about 11% plazas). Its buildings are about 13 m
  across, near the formation's own width, which is the regime of S2's null morphing result.

**Why a block-shaped building needs a new obstacle model.** The force law sees each building as the smallest
upright cylinder round its world bounding box (`OlfatiSaber.GetObstacleCylinder`). Even the tightest single circle
round a block-shaped building bulges past its long faces. At p 50 with 12 m streets, only 63% of streets keep a
6 m gap between the cylinders; at p 75, 39% do.

The B variant therefore uses a replica-only option, `obstacle_model = "closest"`:
- the β-agent sits on the footprint's true closest point;
- μ = 1;
- `face_bend` κ tilts a face's normal away from the building's centre in proportion to the drone's offset along the
  face over the bounds radius, so an off-centre approach still slides as it would on a cylinder.

It reaches the C# only if Phase 1 chooses B. Today's law is untouched: with the option off, `swarm_replica.py step`
and an S3 quick run reproduce bit for bit.

**The predicted surround separation.** A ring of ten drones wrapped at 3 m round the building needs commanded
spacing (10·d_ref m) of about a tenth of its perimeter, so d* ≈ ring perimeter / 100. The median over blocks:

| p \ s | 12 m: R / B | 18 m: R / B | 24 m: R / B |
|---|---|---|---|
| 40 | 0.96 / 1.07 | 0.77 / 0.84 | 0.65 / 0.70 |
| 50 | 1.25 / 1.49 | 1.06 / 1.26 | 0.88 / 1.03 |
| 60 | 1.56 / 1.87 | 1.37 / 1.65 | 1.19 / 1.42 |
| 75 | 2.05 / 2.52 | 1.86 / 2.30 | 1.67 / 2.08 |

Blocks within one city differ by 15–20%, measured as the median against the p90. That is why a separation matched
to each block is tested beside one matched to the design.

## 3. The S4 profile

The profile starts from ProceduralCityWorld's `SwarmManager`, the scene the bench flies generated layouts in, with
**`r0_coh` 40**. Under the coded law, cohesion between two drones turns repulsive beyond about 75 m at `r0_coh` 20.
At 40 it stays attractive to about 140 m:

| r0_coh | 16 m | 24 m | 40 m | 60 m | 80 m | 100 m |
|---|---|---|---|---|---|---|
| 20 | 0.42 | 0.62 | 0.57 | 0.26 | −0.11 | −0.35 |
| 40 | 0.42 | 0.62 | 0.73 | 0.69 | 0.57 | 0.39 |

The values are m/s² per pair at d_ref 1.0. δ is left at 0.08, because it also shapes the obstacle, shield and core
fields. The near field is unchanged, so the spread response is checked (M5) rather than assumed.

**As it turned out** ([6.4](#64-cohesions-range-the-core-and-bs-face-bend-p1_fu_)), 40 drags drones into buildings, and
R's confirmed profile, which Phase 2 and `BlockCityWorld` fly, sets `r0_coh` to **30**: aim-and-turn for 6 s, the exit
at full stick, the core on, and the design's d* at the block.

## 4. Phase 1: the manoeuvre suite (`citysim.split_probe`)

**How every manoeuvre is flown.**
- It is flown by the replica under the bench's own pilot, so the bench can fly the same scenarios:
  - a waypoint is taken when the centroid comes within 10 m of it, or when its timeout runs out;
  - a negative stick is a centring hold at 0.25/s, capped at 0.5;
  - the spread is set per waypoint (`wpSpread`, added to `SwarmBenchRunner` for this study).
- Every manoeuvre approaches an interior block along two interior streets, from a settled formation pushed clear of
  the buildings at the start junction.

| | flight | committed when |
|---|---|---|
| M1 street run | ~200 m of interior streets at full stick, and one drone on the same waypoints | — |
| M3 pass round | approach the block's entrance junction; aim through the block at a point 15 m down the opposite exit street; on to its far junction | the aim point is taken |
| M3b surround and exit | the same approach; a centring hold on the block's centre for t_hop; out by a side exit (neither the entrance nor the opposite street) to its far junction; plus the same formation round the block by the streets to the same exit (the time price) | the hold ends |
| M4 surround hold | the approach; a 45 s centring hold on the block's centre at a given spread | — |
| M5 spread steps | open sky, `swarm_replica`'s step test | — |
| M6 head-on face (B) | the city's median building alone, flown at from 60 m through its centre | — |

**The spread.** It is 1.0 in the streets. At the block it is one of three things:
- a fixed value;
- the design's d* from M4;
- k × that block's ring perimeter / 100.

**Measures.**
- **Preview.** A candidate street is any radiating street of the block other than the entrance. It counts as
  previewed when 60% of its 2 m centreline samples within 80 m of its corner have been seen on the shown feeds,
  within 80 m slant, through a clear sight line.
  - The shown feeds are `display.FeedLatch`'s: hull membership debounced 0.5 s, heading low-passed, 75°/s yaw, and
    the ready latch.
  - It is scored at the search band (12.4 m, −20.8°) and at the transit band (19.5 m, −15°). At the transit band a
    camera sees no ground nearer than 22.6 m.
- **Clusters.** Two drones are linked when the free-space path between them is at most 30 m; a straight-line test
  would call a formation split whenever it turned a corner. A *split* is two or more clusters of at least two
  drones each.
- **Reform.** It succeeds when there is one cluster before the next decision point, censored there.
- **Envelopment.** S1's test: no drone lost, the centre inside the hull, no bearing gap over 150°, and at least 70%
  of the drones inside the block's clearance.
- **Also:** stall (the centroid under 1 m/s for 3 s while not holding), drones lost, and contacts (`ContactSim`:
  PhysX friction 0.6).

**Criteria, fixed before any flight.** They are judged per design, and must hold for both M3 and M3b:

| criterion | bar |
|---|---|
| safety | drones lost per manoeuvre ≤ 0.03, the Wilson 95% upper bound on the share of manoeuvres losing any drone ≤ 0.06, contacts per manoeuvre ≤ 0.1 |
| reform | ≥ 90% |
| preview | M3b: ≥ 80% of the streets in ≥ 80% of trials. M3: the exit reached with no stall over 3 s in ≥ 95% |
| time price | M3b's median hop time ≤ 1.5× round by the streets |
| streets | M1: ≥ 80% of one drone's speed, ≤ 0.05 contacts per 100 m |
| spread response | M5 t90 ≤ 1.25× the bench references, overshoot ≤ 15% |

**The variant.**
- **B** if it passes in at least 9 of the 12 designs.
- Otherwise **R** if it passes in at least 6.
- Otherwise whichever passes in more designs, and Phase 2 is restricted to those designs.
- **Stop** if neither is safe in at least 4 designs.

**The profile.** It is one set of values for every design. Only the separation may vary, and only with the design
or the block.

**The tuning order.** Each step keeps the best setting of the steps before it:
1. the separation;
2. `r0_coh` 40 against 20;
3. the hollow core on or off;
4. t_hop and the exit stick;
5. other gains, only if needed.

## 5. Phase 2: the comparison (fixed before it runs)

**Cities.** The designs that pass Phase 1, in the chosen variant, with 8 cities each. As run (`citysim/specs/split.json`,
written after Phase 1's decision and before any Phase 2 flight): R at p 40 s 18 and p 40 s 24, **24 cities per
design**, because the plan's 8 was sized for 12 designs and 16 cities would resolve the pooled primaries only to about
±13%; on seeds 100–123, apart from the cities Phase 1 tuned on. A 2% sample of each run is flown again and must match
bit for bit (`recheck.csv`).

**Plazas.** At p 40 s 24 over half the blocks are plazas, and a chain that stepped only onto blocks with a building
broke after 12–24 blocks, short of the walk. A chain may therefore step onto a plaza, which every condition crosses by
the streets, as one drone or the street-bound swarm would; the per-block measures (H1, reform) are taken over the
blocks with a building.

**Conditions.** Each flies the same paired block-hop chain, or the same informed scenario, from the same start and
seed. Swarm conditions fly the profile's separation rule (primary) and a constant 1.0 (the trials' dial centre).
Spreads 0.4 and 1.6 are not flown.

| condition | at each block | commit |
|---|---|---|
| `single` | street centreline, round the block by the shorter side to the exit | at the entrance corner |
| `single_look` | as `single`, then at the entrance corner it holds and yaws to each street there (75°/s, 0.5 s dwell). Replica only: on the bench a lone drone does not yaw at zero stick | end of the look |
| `swarm_street` | the single drone's path | at the entrance corner |
| `swarm_hold` | as `swarm_street`, plus a centring hold of t_hop at the entrance corner at the block spread | end of the hold (H1's time-matched comparator) |
| `swarm_pass` | aim through the block at the exit corner (M3 when the exit is opposite) | the aim point taken |
| `swarm_hop` | Phase 1's surround (a hold of t_hop on the block centre, or the aim-and-turn mode, whichever Phase 1 chose), then the exit | end of the surround |

**Pilots.**
- **The block-hop chain.** From the start block, each exit is drawn uniformly among the block's interior radiating
  streets other than the entrance and those at the entrance corner. The next block is the one ahead where that
  street ends. The chain never reads the flight. Time-based outcomes run to 600 s; block-based ones are taken over
  the blocks every condition reached.
- **The informed (frontier) pilot,** replica only (`split search`).
  - Each condition keeps its own knowledge: the street samples it has seen within 80 m on its shown feeds.
  - It decides where the condition decides in the chain: on arrival at the entrance corner (`single`,
    `swarm_street`), after the looks, the hold or the surround (`single_look`, `swarm_hold`, `swarm_hop`), and at the
    entrance for `swarm_pass`, whose pass is aimed by the decision.
  - It takes the exit with the most unseen street length within two edges (the exit street and every street at its
    far junction) per metre of travel to that junction, among the chain's exits (interior, not at the entrance
    corner, leading to an interior block). When no exit has any, it takes the one whose far junction is nearest by
    street to an interior street with at least 20 m unseen. Ties are broken by the seed.
  - **When the chain's rule leaves no exit at all** (every other exit leads off the interior), the pilot may leave by
    a street at the entrance corner, and failing that turn back down the street it came in by. Without this, 12 of
    the first 40 routes flown (most at p 40 s 24, where over half the blocks are plazas) simply stopped early and
    would have counted as unfound at the cap. Fixed before the search ran; the chain walks never use it.
  - The target is a hidden point on open ground at least 60 m inside the arena's edge, at least 300 m by street from
    the start and out of clear sight of it within 200 m.
  - It is found on a shown feed within 80 m through a clear line, and the flight stops there; the first sight at
    200 m is read off the same flight. The cap is 1,200 s, and an unfound target counts as found at the cap (RMST).
    A swarm under four drones has not found it.
  - On every other target, `swarm_street` and `swarm_hop` also fly on head-ahead knowledge (only what falls in the
    headset's view with the head facing travel).
  - **Two changes from the plan, made before any search was flown.** The plan named `single_look` for the
    head-ahead variant; a single drone's one feed is always in view, so its head-ahead flight is its any-feed flight
    again, and H2's two conditions are flown instead. And there is no 40 m first sight: the search stops at its 80 m
    find.

**Measures.**
- **Benefit:**
  - preview at each commit;
  - street clearance;
  - S3's time to find (40/80/200 m; any feed, head ahead, fitted head; a 1 s dwell variant);
  - informed-search time.
  - *As run:* the time to find was computed at 40, 80 and 200 m on any feed and head ahead. The fitted head and the
    1 s dwell variant were not computed, so lessons 4 and 5 stand for S4 too.
- **Formation:** clusters, reform before the next decision, losses, contacts, hull feeds shown.
- **Correspondence:** `citysim/correspondence.py`, the report's shortlist (A1 ground and facades, A2, A3, A4, B1,
  B2's order test, C1). No combined headline.

**Hypotheses.**
- **H1 (primary):** the pooled difference in the share of a block's radiating streets previewed at the commit,
  `swarm_hop` minus the time-matched `swarm_hold`, at the profile's separation, is greater than 0.
- **H2 (primary):** the pooled city-mean log RMST ratio of the informed search (80 m, any feed), `swarm_hop` ÷
  `swarm_street`, is less than 0.
- Both are tested with cities as units, with Holm over the two.
- **Secondary:**
  - every other contrast;
  - per-design estimates (Benjamini–Hochberg);
  - trends over p and s;
  - a ±10% equivalence test of H2's ratio at p 40 / s 18 (the small-building control); as run, at both Phase 2
    designs, since both are small-building cities;
  - H3: costs rise with block size and fall with street width.

## 6. Phase 1 results

Outputs: `experiment_results/split_study/phase1/<run>/` (`results.jsonl` per run; `split_probe tables --name <run>`
writes the criteria, the coupling map and T_hop beside it). Every run below flew 50 trials per design × variant ×
manoeuvre on city seeds 0–2 unless it says otherwise.

### 6.1 The surround map d*(p, s) (`p1_coupling`)

M4 at spreads 0.8–2.8, 30 holds per cell, 7,920 holds, no errors. d* is the smallest spread that enveloped the block
in at least 80% of holds with no drone lost and at most 0.1 contacts per hold; the prediction is the ring 3 m off the
building, divided by 100.

| p \ s | 12 m: R / B (predicted) | 18 m | 24 m |
|---|---|---|---|
| 40 | 1.0 / 1.2 (0.96 / 1.07) | 1.0 / 1.0 (0.77 / 0.84) | 0.8 / 0.8 (0.65 / 0.70) |
| 50 | 1.2 / 1.4 (1.25 / 1.49) | 1.0 / 1.4 (1.06 / 1.26) | 1.0 / 1.2 (0.88 / 1.03) |
| 60 | 1.4 / 1.8 (1.56 / 1.87) | 1.2 / 1.4 (1.37 / 1.65) | 1.2 / 1.4 (1.19 / 1.42) |
| 75 | 2.0 / none (2.05 / 2.52) | 1.8 / 2.0 (1.86 / 2.30) | 1.6 / 1.8 (1.67 / 2.08) |

- **The prediction holds.** R's d* is within one 0.2 step of perimeter/100 everywhere; B's sits at or a step below it.
  B at p 75 with 12 m streets has no safe spread: the best within the contact bar (1.6) envelops 17% of holds.
- Under S1's hold the surround takes 20–39 s to envelop at d*, and no hold at d* lost a drone.

### 6.2 The spread response (M5, `p1_steps`)

`r0_coh` 40 against the scene's 20, six seeds each, open sky: expansion t90 3.1 s for both, contraction t90 5.2 against
5.25 s, overshoot 0.1–0.2%, no kills. The bars (1.25× the bench references, 15% overshoot) are 4.4 s, 7.0 s and 15%.
R's confirmed `r0_coh` 30 (`p1_steps_r30`, flown after the decision) gives the same: 3.1 s, 5.2 s, 0.1–0.2%, no kills.
Cohesion's range changes nothing within the formation's own width, which is all a spread step exercises.

### 6.3 The screens: six ways of flying the surround (`p1_scr_*`)

Each flew M1, M3, M3b and M6 with a 30 s surround, so the time each block's streets took to preview was measured
rather than cut off. They differ from `p1_scr_soft` (S1's hold, stick 0.5 and gain 0.25, at d*) in one thing each:
the hold firm (stick 1.0, gain 0.5); aim-and-turn (full stick at the opposite exit for the surround, then off down the
side exit); the spread per block (perimeter/100, and × 1.25); 1.0 at every block.

- **Safety cannot pass at 50 trials.** Zero losses in 50 trials has a Wilson upper bound of 0.071, over the 0.06 bar;
  it takes 61 trials. The confirmation flies 100.
- **S1's hold is too slow to be a brief surround.** The surround needed for ≥ 80% of a block's streets in ≥ 80% of
  trials was 10–24 s at p 40 (R) and 18–34 s or never at p ≥ 50.
- **The firm hold and aim-and-turn cut it to 3–16 s for R**, aim fastest at p ≤ 50. Aim-and-turn flown for 30 s lost
  drones (0.04–0.24 per manoeuvre at p ≤ 60) where no other mode did: the surround outlasted the pass, and the
  formation reversed at full stick to reach a side corner behind it.
- **The pass round (M3) previews 88–99% of the block's streets by its commit**, in 12–31 s (R) with no hold, and
  reaches the exit without a stall in 98–100%.
- **The pass's reform fails at p ≥ 50 with 12–18 m streets (38–92%).** Flown on past the next junction, nearly every
  failure is one drone that went round the far side alone and is still more than 30 m of free-space path from the rest
  when the next junction comes (0–7 s after the commit, these streets being short). At p 50 it rejoins within about
  half a second in most flights; at p 75 it takes 3–12 s, or longer than 20 s.
- **B touches.** B's contacts were 0.2–2.6 per manoeuvre at p ≥ 50 in every screen, and nearly all are with the
  building the formation is taking. Re-flown with every contact located (`split_replays`' tape, 2026-10-28): 183 of
  188 in the screens (97%) and 85 of 98 at B's profile (87%) were with the block's own building, the drone pressed into
  its facade with the building across the street 18–31 m away. They come while the formation passes round the block
  (M3, 77 of the screens' 85) or turns off at its corner (M3b's exit, 102 of 103). **This bullet first said
  three-quarters were with the buildings across the street; that was wrong**, and so was the explanation it gave (a
  corridor of the street plus two footpaths, 19–31 m, too narrow for a formation at d*). Why the formation presses
  into the facade is not established: the face bend, which gives the closest-point model the sideways push a cylinder
  has, made it worse (6.4). R's buildings, steered round as cylinders, stay near the contact bar.
- **The separation stays the design's d*.** The per-block rule previews faster at p 50–60 but adds contacts; 1.0 at
  every block cannot surround p ≥ 50 blocks (20–24% of trials never previewed 80%).

### 6.4 Cohesion's range, the core and B's face bend (`p1_fu_*`)

One parameter changed from `p1_scr_soft` each.

| contacts per manoeuvre, M3 / M3b | `r0_coh` 20 | 30 | 40 (the screens) |
|---|---|---|---|
| R | 0.00–0.04 / 0.00–0.02 | 0.00–0.12 / 0.00–0.12 | 0.02–0.18 / 0.06–0.32 |
| B | 0.02–0.32 / 0.00–0.26 | 0.06–0.92 / 0.00–1.04 | 0.06–1.72 / 0.32–1.66 |

- **Raising cohesion's range trades contacts for reform.** The stronger pull at 30–60 m is what drags a drone on one
  side of a block toward the drones on the other, through the building: ScaledCityWorld's finding that building
  contacts are cohesion. At 20 the straggler rejoins more slowly: R's pass at p 75 reforms in 18–40% against 38–66% at
  40.
- **R's best is 30**: contacts near 20's and reform as good as 40's (90–100% at p 40 s 18/24, p 50 s 24, p 60 s 24).
  **B's is 20**, the only range that holds its contacts near the bar, at the cost of reform at p ≥ 60. Each variant
  is taken at its own, and every design within a variant shares one profile.
- The hollow core off changes contacts and reform by no more than the trial noise; it stays on, as in the scene.
- **B's face bend κ raises contacts and barely helps head on.** M3's contacts at p ≥ 50 were 0.38–1.02 per
  manoeuvre at κ 0, 0.64–1.72 at 0.5 (the screens) and 1.2–5.1 at 1; the head-on face (M6) passed in 57%, 69% and
  60%, with 0.09, 0.34 and 2.5 contacts. The bend tilts the repulsion off the face's normal, which slides a drone along
  the face into the corner and the neighbours rather than holding it off. B's candidates are flown at κ 0 as well.

### 6.5 The surround's length, the exit stick and the candidate profiles (`p1_t4_*`, `p1_c_*`)

M3b flown at its real length, 6, 8, 10 and 12 s, by the two quick modes (`r0_coh` 40, d*):

| R, share of trials with ≥ 80% of the streets previewed / time price | 6 s | 8 s | 10 s | 12 s |
|---|---|---|---|---|
| p 40 s 18, aim | 0.92 / 1.40 | 0.94 / 1.60 | 0.98 / 1.75 | 1.00 / 2.07 |
| p 50 s 18, aim | 0.66 / 1.30 | 0.84 / 1.40 | 0.92 / 1.60 | 0.94 / 1.52 |
| p 60 s 18, aim | 0.36 / 1.37 | 0.54 / 1.54 | 0.68 / 1.69 | 0.91 / 1.81 |
| p 75 s 18, aim | 0.08 / 1.28 | 0.22 / 1.39 | 0.34 / 1.46 | 0.43 / 1.51 |

- **The surround that previews a block's streets grows with the block, and the time price caps it.** p 40 meets both
  bars at 6 s and loses the time price past about 8 s, because its street route is short; p 50 meets them only at
  8–10 s, and mostly on the price's line; p 60–75 never does. Matching the surround's length to the block, as the
  spread is matched, would pass at most five designs, and the pass round's reform cuts that to two or three. The plan
  fixes one length for every design; the decision does not hinge on it.
- **The exit stick costs more than it buys.** At 0.5 rather than full stick, the pass round reforms 0–12 points more
  often (p 75 s 12: 38% → 50%), and the surround's time price doubles, 1.35–1.48 → 2.0–2.3. The exit stays at full
  stick.
- **The candidates** (every manoeuvre, 50 trials): R at `r0_coh` 30 by firm hold or aim-and-turn at 6 or 8 s passes
  every criterion, the loss bound aside, at the same two designs, p 40 s 18 and p 40 s 24; aim at 6 s previews the
  most per second, the plan's third key after safety and reform. B at `r0_coh` 20 passes at one design at best
  (aim at 6 s: p 40 s 18), and at none with κ 0, which holds its contacts down but stalls more (its street runs fall
  to 85–97% of one drone's speed) and loses drones at p 40. B's surround previews far more slowly than R's (40–70% at
  p 40, at most 40% at p ≥ 50): its building fills the block, so the ring round it is longer.
- **The confirmation profiles** are therefore R: `r0_coh` 30, aim-and-turn for 6 s, exit at full stick, the core on,
  d* per design; and B: `r0_coh` 20, aim-and-turn for 6 s, κ 0.5.

### 6.6 The confirmation and the decision (`p1_confirm`)

Each variant at its confirmation profile, 100 fresh trials per design and manoeuvre (trials 1000–1099, their own
random streams) on fresh cities (seeds 3–5), 8,400 manoeuvres, no errors:

| | designs passing every criterion | designs safe |
|---|---|---|
| R | 2: p 40 s 18, p 40 s 24 | 12 |
| B | 0 | 4 |

- R is safe everywhere: contacts 0.00–0.07 per manoeuvre (64 in 2,400 pass rounds and surrounds), and one of those
  2,400 lost drones (a pair); B lost drones in 10 and touched 282 times. (The street runs, M1, lost 8 of R's drones in
  1,200 runs; they are not in the safety bar.) R fails on the surround's
  preview at every p ≥ 50 (0.00–0.63 of trials) and on the pass round's reform at p ≥ 50 except p 50 s 24 (0.91), and
  at p 40 s 12 (0.83) on reform alone.
- The surround each design needs, by aim-and-turn on 100 trials (the smallest T at which 80% of trials had seen 80% of
  the block's streets): 4.2–5.6 s at p 40, 9.0 / 10.1 / 13.7 s at p 50 (s 18 / 24 / 12), 20 s at p 60 s 24 and never
  within the flight elsewhere at p ≥ 60.
- **The decision, by the rule fixed in advance:** neither variant reaches its bar (B 9 designs, R 6), so it is the
  variant passing in more designs, R, with **Phase 2 on p 40 s 18 and p 40 s 24 only**. R is safe in 12 designs, so the
  stop rule does not apply.
- What the decision means: the two passing designs are S4's most open cities (α 0.10 and 0.03; at p 40 s 24 more
  than half the blocks are plazas) and its smallest buildings (about 13 m across, near the formation's own width),
  the regime S2 measured. Surrounding a block is quick and pays for itself there, and only there.
- **Open for the user:** the surround's length is one value for every design, as the plan fixed. Matched to the block,
  as the spread is, it would pass the preview at p 50 as well; with the pass round's reform, at most p 50 s 24 would
  then pass every criterion.

### 6.7 The Unity bench check (`p1_bench`)

R's confirmed profile flown in headless Unity (ProceduralCityWorld with the `s4` set, `r0_coh` 30) at four designs, two
that pass and two that do not (p 40 s 18, p 50 s 12, p 50 s 24, p 75 s 18), on city seeds 0–2: 30 trials each of M3,
M3b with its street flight, and M4, recorded (480 flights, no errors). `split_bench plan` writes the configs from the
replica's own trials; `split_bench gate` scores Unity's trajectories with the replica's measures and re-flies each
flight in the replica from Unity's recorded start.

- **`swarm_replica.py validate` passes on all 12 runs**: track within 0.4–0.8% (limit 10%), spacing 0.4–1.7% (15%),
  losses 0–2 per 40 flights on both sides (the limit is a difference of 2 per 10), contacts 0–5 on the bench against
  0–3 in the replica.
- **The S4 gate agrees on everything it can resolve** (51 of 56 comparisons within 15%; the other 5, two contact rates
  and three split shares under 2%, are too rare to judge from 30 flights; none differ): the streets
  previewed at the commit within 0–7%, reform within 0–14%, envelopment shares identical and the time to envelop
  within 3%, losses equal.
- **Contacts are too rare to judge from 30 flights** (the bench's standard error is 40–100% of its mean), and at p 40
  s 18 Unity shows a few more than the replica: 0.17 against 0.07 per surround, 0.07 against 0 per pass. The rule's
  "more flights until the standard error is 7.5%" is a rule for shares near the middle; for a count this rare it would
  take thousands of flights, so the rates are reported instead. Phase 2's walks re-check contacts on the bench.

## 7. Phase 2 results

Outputs, under `experiment_results/split_study/`:
- `split/`: the walks (`results.jsonl`; from `split tables --name split`: `flights.csv`, `hops.csv`, `contrasts.csv`,
  `costs.csv`, `trends.csv`, `primary.json`, `h1_by_design.csv`);
- `split_search/`: the informed search (`split tables --name split_search`: `search_flights.csv`,
  `search_summary.csv`, `search_contrasts.csv`, `primary_h2.json`; and `split primaries --walks split --search
  split_search`: `primaries.json`, H1 and H2 with Holm);
- `p2_bench/bench/` and `p2_routes/bench/`: the Unity spot checks.

### 7.1 The block-hop walks (`split`)

384 walks of 600 s (24 cities per design × 8), each flown in ten flights (the four swarm conditions at d* and at 1.0,
and the two single-drone conditions): 3,840 flights, no errors. The 2% recheck matched 8 of 8 bit for bit. The run
took 2.7 h on 24 workers.

City means at the profile's spread:

| condition | blocks per walk | streets previewed at the commit | street km seen per min | time to find, 80 m (s) | drones lost per walk | contacts per walk | neighbouring screens linked (A1) |
|---|---|---|---|---|---|---|---|
| `single` | 38.3 | 58% | 1.00 | 475 | 0 | 0 | – |
| `single_look` | 26.1 | 79% | 0.89 | 489 | 0 | 0 | – |
| `swarm_street` | 37.4 | 84% | 1.32 | 424 | 0.03 | 0.10 | 81% |
| `swarm_hold` | 28.7 | 88% | 1.17 | 451 | 0.04 | 0.13 | 87% |
| `swarm_pass` | 38.6 | 98% | 1.32 | 424 | 0.07 | 0.40 | 70% |
| `swarm_hop` | 31.0 | 96% | 1.27 | 433 | 0.29 | 0.82 | 59% |

- **H1 holds.** The surround previews 7.1 points more of a block's radiating streets at the commit than the
  time-matched hold: 95.5% against 88.4% (t over 48 cities 33.0, one-sided p 2×10⁻³⁴). That is +6.1 points at p 40
  s 18 and +8.2 at s 24, each q < 10⁻²⁰.
- **The pass round previews the most, at no cost in time.** It previews 97.8% of a block's streets, takes as many
  blocks per walk as the street-bound swarm, and finds a hidden point as fast (×1.000). It loses more drones (0.07
  against 0.03 per walk) and touches buildings more (0.40 against 0.10).
- **On a chain that ignores what it sees, any pause costs time.** The chain walks measure what each way of taking a
  block shows, not whether it pays. Time to find a random hidden point at 80 m (the RMST ratio):
  - the surround against the street-bound swarm: ×1.028 at s 18 and ×1.013 at s 24 (1–3% later);
  - the hold against the street-bound swarm: 5–8% later;
  - the surround against the hold: ×0.954 and ×0.967 (3–5% sooner), because its 6 s go round the block rather than
    waiting at the corner;
  - one drone stopping to look against one that doesn't: ×1.03.
- **The swarm's cameras.** The street-bound swarm against one drone is ×0.89 at 80 m, 0.92–0.93 at 40 m and 0.76–0.79
  at 200 m; it sees 32% more street per minute.
- **The surround is the costly one.** It lost 110 drones in 384 walks (0.29 per 10-minute walk: 0.40 at s 18, 0.17 at
  s 24) and touched buildings 313 times. That is about 0.009 drones and 0.026 contacts per block, inside Phase 1's
  per-manoeuvre bars.
  - 106 of the 110 were drone-drone pairs (53 collisions); 4 were drones left behind.
  - These are the losses Phase 1 saw when aim-and-turn reversed at full stick.
  - The other swarm conditions lost 10 (street), 14 (hold) and 27 (pass).
- **Reform.** Every swarm condition was one cluster before the next junction after 98.5–100% of blocks.
- **The block spread barely matters at these blocks.** At p 40 s 24, where d* is 0.8, flying 1.0 instead changes the
  surround's preview by +0.4 points and its time to find by −0.3%. It loses 0.09 fewer drones per walk and touches
  0.10 more. At s 18, d* is 1.0, so the two are the same flights.
- **Walks cut short by the arena's edge.** A chain ends where it reaches the city's edge, and a flight that has ended
  sees nothing more, so a condition that finishes its chain sooner is charged for the rest of the 600 s. 12% of walks
  ended early in some condition. On the 88% that ran the whole time in every condition, the time-to-find ratios move by
  at most 0.003; for example, surround ÷ street goes from 1.021 to 1.022.

### 7.2 The informed search (`split_search`)

288 targets (48 cities × 6). Every condition searched for each target from the same start; on the even targets
`swarm_street` and `swarm_hop` also flew on head-ahead knowledge. That is 2,016 flights, no errors; the 2% recheck
matched 6 of 6. The run took 1.4 h on 24 workers.

Means over targets:

| condition | found within 1,200 s | time to find, 80 m (RMST, s) | decisions | none unseen near | path (m) | drones lost | contacts |
|---|---|---|---|---|---|---|---|
| `single` | 80% | 571 | 38.2 | 2.4 | 3,743 | 0 | 0 |
| `single_look` | 78% | 597 | 24.7 | 1.7 | 2,615 | 0 | 0 |
| `swarm_street` | 90% | 362 | 22.4 | 5.2 | 2,289 | 0.03 | 0.09 |
| `swarm_hold` | 91% | 409 | 19.2 | 3.4 | 2,041 | 0.01 | 0.12 |
| `swarm_pass` | 90% | 351 | 22.7 | 5.3 | 2,142 | 0.02 | 0.27 |
| `swarm_hop` | 94% | 361 | 18.3 | 3.6 | 2,080 | 0.15 | 0.63 |

The ratios below are city-mean RMST ratios at 80 m on any feed, over 48 cities with 95% t intervals.

- **H2 is not supported.**
  - The surround against the street-bound swarm is ×1.015 (0.922–1.119), one-sided p 0.62, Holm 0.62.
  - Per design: ×1.070 at s 18 (0.94–1.22) and ×0.964 at s 24 (0.83–1.11).
  - The ±10% equivalence test does not pass either (TOST p 0.34 and 0.22). The data can rule out neither a gain nor
    a loss of about 10%.
  - On head-ahead knowledge the surround is ×1.157 (0.994–1.346), leaning slower.
  - The first third of the targets read ×1.12–1.39; the full run did not bear that out.
- **Where the surround's preview goes.** On the 250 targets both conditions found, against the street-bound swarm:
  - the surround took 10% fewer decisions (×0.899, 0.823–0.981): its knowledge does steer the pilot better;
  - each decision took 21% longer (×1.213, 1.190–1.236), over the same path (×0.985);
  - so it found those targets 9% later (×1.090, 1.001–1.187).
  - It also found 21 targets within the cap that the street-bound swarm missed, against 8 the other way (exact
    McNemar p 0.02, over targets, so indicative only).
  - Those two effects cancel in the capped mean.
- **Surrounding beats pausing.** The surround finds sooner than the time-matched hold, ×0.878 (0.810–0.952). The hold
  takes 8% fewer decisions than the street-bound swarm (×0.918), but each costs 31% more time (×1.305).
- **The pass round is free and changes nothing** (×0.974, 0.886–1.070): it previews more at each block, but takes as
  many decisions as the street-bound swarm (×1.004).
- **The swarm's cameras are the effect.**
  - The street-bound swarm finds a target in 62% of one drone's time (×0.617, 0.554–0.687).
  - On targets both found it takes 39% fewer decisions over 35% less path.
  - The swarm finds 90–94% of targets within 20 minutes, against 78–80% for one drone.
- **Looking does not help one drone** (×1.050, 0.958–1.152). It saves 29% of its decisions, but each takes 63% longer.
- **Losses.** The surround lost drones in 6.9% of its searches (42 drones in 288 searches); the other swarm conditions
  lost drones in 0.7–1.4% (4–8 drones). No search ended under four drones.

### 7.3 The screens: what surrounding costs the pilot's view (`correspondence`)

The options report's measures, sampled each second on the chain walks at the profile's spread. Every screen is treated
alike, the panorama is left out, and open ground and facades both count. No headline is chosen; that is the user's
call.

| | street | hold | pass | surround |
|---|---|---|---|---|
| A1: neighbouring screens sharing content | 81% | 87% | 70% | 59% |
| A2: sharing a building | 42% | 46% | 36% | 31% |
| A3: islands round the ring | 1.79 | 1.48 | 2.30 | 2.77 |
| A3: orphan screens (no linked neighbour) | 7% | 4% | 15% | 24% |
| A3: largest island, share of the screens | 84% | 90% | 75% | 67% |
| A4: neighbours whose views overlap before a building | 88% | 93% | 79% | 70% |
| B1: viewpoint difference on a shared building | 5.5° | 5.3° | 5.9° | 6.8° |
| B2: placement error from the centroid | 5.1° | 4.8° | 5.7° | 6.6° |
| B2: pairs shown in reversed order | 2% | 2% | 2% | 2% |
| C1: ground seen on a linked screen | 100% | 100% | 98% | 95% |
| screens shown | 8.7 | 9.1 | 8.0 | 7.3 |

- **Every measure ranks the conditions the same way**, with the order test the exception: hold, street, pass,
  surround.
- **The surround buys its preview with linkage.** A quarter of its screens have no linked neighbour, against 7% in
  the streets.
- **The pass sits between.** It previews more than the surround while keeping more of the screens linked.
- The order test does not separate them: 2% of pairs are reversed in every condition.

### 7.4 The Unity spot checks (`p2_bench`, `p2_routes`)

**The walks.** Phase 2's own chain walks were flown for 600 s in headless Unity (ProceduralCityWorld with the `s4` set):
- 4 cities per design (seeds 100–103) × 4 walks;
- every condition the bench can fly, which excludes `single_look` (a lone drone does not yaw at zero stick there);
- 160 flights, no errors.

`split_bench gate-walks` scores Unity's trajectories with the replica's measures and sets each beside the replica's
own flight of the same walk. Unity / replica, means per walk:

| condition | blocks | streets previewed | street km seen | time to find, 80 m (s) | drones lost | contacts |
|---|---|---|---|---|---|---|
| `single` | 37.3 / 37.5 | 64% / 62% | 9.46 / 9.42 | 478 / 479 | 0 / 0 | 0 / 0 |
| `swarm_street` | 36.2 / 36.6 | 88% / 86% | 12.34 / 12.35 | 430 / 429 | 0.19 / 0 | 0.16 / 0.06 |
| `swarm_hold` | 27.9 / 28.2 | 90% / 89% | 10.92 / 10.93 | 453 / 453 | 0.19 / 0.06 | 0.09 / 0.13 |
| `swarm_pass` | 37.3 / 37.7 | 99% / 99% | 12.30 / 12.33 | 431 / 430 | 0.09 / 0 | 0.47 / 0.31 |
| `swarm_hop` | 29.9 / 30.3 | 96% / 96% | 11.78 / 11.79 | 436 / 436 | 0.44 / 0.22 | 1.44 / 0.84 |

- **Unity agrees on everything the hypotheses rest on:**

  | contrast (8 cities, 95% intervals) | Unity | replica, same walks |
  |---|---|---|
  | surround minus hold, streets previewed (H1 on the bench) | +6.6 points (5.6–7.7) | +6.8 (5.7–7.8) |
  | surround ÷ street-bound swarm, time to find | ×1.014 | ×1.015 |
  | surround ÷ hold, time to find | ×0.962 | ×0.962 |
  | street-bound swarm ÷ one drone, time to find | ×0.898 | ×0.895 |

  Per condition, Unity agrees with the replica to within:
  - 1.5% on blocks reached;
  - 0.4% on street seen;
  - 0.15% on time to find;
  - 2 points on the streets previewed.
- **Unity's costs are higher.**
  - Drones lost in 128 swarm walks: 29 against the replica's 9. By condition: surround 14 against 7, hold 6 against
    2, street 6 against 0, pass 3 against 0.
  - Contacts: 69 against 43.
  - So the replica understates the costs, which only strengthens 7.1's: the surround still costs the most, and every
    swarm condition loses some drones in Unity.
  - Phase 1's check hinted the same at p 40 s 18 (0.17 contacts per surround against 0.07). The losses are too few to
    split by cause.

**The routes.** 40 informed searches were re-flown open-loop: 2 designs × 2 cities × 2 targets, in each of the 5
bench-flyable conditions.
- The waypoints were the ones each replica flight's decisions laid. Unity's force law flew them, and the target's
  first sight was scored with the replica's rule (`split_bench plan-routes` / `gate-routes`).
- **At 80 m**, 26 routes found the target in both, all within 5 s of each other (median 1.8 s).
- **At 200 m**, all 40 found it in both, within 11 s (median 1.5 s).
- Two late finds in the replica (902 and 942 s) were not made in Unity, and 12 routes found the target at 80 m in
  neither.
- Unity lost 7 drones (the surround 5, the pass 2) against the replica's 4, and touched buildings 5 times, 4 of them
  in the surround.
- The routes are open-loop, so this checks the flying and the seeing, not the decisions. The decisions are code the
  two share.

### 7.5 What Phase 2 means

- **The answer to S4's question.**
  - Surrounding a block does show the pilot more of the streets round it (H1), and in the search it beats spending
    the same time paused at the corner (×0.88).
  - For finding things it is no better than keeping the swarm in the streets (H2).
  - It costs:
    - about ten times the drones lost (0.29 against 0.03 per 10-minute walk, nearly all in drone-drone collisions);
    - eight times the building contacts (0.82 against 0.10 per walk);
    - a quarter of its screens with no linked neighbour.
  - In the only cities where the surround passed Phase 1 (p 40, buildings about 13 m across), surrounding block after
    block is not an efficient way to explore.
- **It answers S2's open question.** S2 found that the swarm's search advantage came from its cameras and not from its
  formation flowing round buildings, but no S2 city made going round buildings the way to travel. S4's cities do, and
  the answer is the same:
  - deliberately surrounding each block adds nothing to the search;
  - the pass round, where the formation flows round the block on its way through, adds nothing either (×0.97);
  - the swarm kept in the streets already finds a target in 62% of one drone's time.
- **If the pilot wants the preview, the pass round is the way to get it.** It shows the most (98% of a block's
  streets), costs no time, and keeps more screens linked than the surround. It loses more drones and touches buildings
  more than staying in the streets: 0.07 against 0.03 drones and 0.40 against 0.10 contacts per walk.
- **Limits.**
  - **Two designs, both of small buildings.** The large-block regime, where surrounding might matter most, could not
    be flown: there the surround needs 9–20 s or more, and the pass round strands a drone (Phase 1).
  - **Scripted pilots.** The informed pilot uses everything any feed has shown, and decides at fixed points by a
    fixed rule. With only the head-ahead feed it leaned slower with the surround (×1.16). A person would use less of
    the preview, and would look round at a corner at their own pace.
  - **One surround length** (6 s) and one way of flying it (aim and turn).
  - **The replica is 2-D** (lesson 12); the Unity spot checks are in 7.4.
  - **A single glimpse within 80 m counts as a find** (lesson 5).

## 8. The page's Cities tab: the cities and the flights (`citysim.split_replays`)

The results page opens on a **Cities** tab (added 2026-10-28 at the user's request; every study's page now has one,
lessons.md 30). Under the masthead a key defines the design codes and the two variants, and a **Terms** tab defines
every other term (lesson 31).

- **The cities.**
  - One block of the selected design drawn to scale, with p (centre to centre), s (kerb to kerb), R's building,
    cylinder and inscribed circle (or B's building), and the formation's ring 3 m outside, beside the design's
    numbers.
  - The twelve designs at one scale (the same 240 m of each city's middle, seed 0), for R or B.
  - The selected city whole, with every block's sizes on hover.
- **The flights.** 73 in all (51 before Phase 3):
  - 7 of Phase 1's manoeuvres. These are each rule's trial nearest its design's median time, from the confirmation
    run; the surround hold is from the coupling run at d*:
    - a clean pass and a stranded drone (R at p 40 and p 75);
    - a surround that previews its block, and one too short for a p 75 block;
    - a 45 s surround hold at p 50;
    - B's pass and surround, with their contacts.
  - Two of Phase 2's block-hop walks, in all six conditions:
    - p 40 s 18 seed 101 walk 3, where the surround lost drones;
    - p 40 s 24 seed 103 walk 2, a typical one with half the blocks plazas.
  - Two informed searches, in all six conditions (p 40 s 18 seed 100 target 0; p 40 s 24 seed 101 target 1).
  - Unity's flights of the same walks and routes, which every flyable condition has. One drone looking round has
    none: the bench cannot fly it.
  - Phase 3's (added 2026-10-28, bringing the total to 75):
    - one Part A block each at p 40 and p 75 (the trial whose surround-minus-street difference in exits looked
      down is nearest its design's median), flown six ways: one drone (`lookdown_A1`'s job for the same trial,
      re-flown and checked identical too), in the street, held at the corner, passed round, Phase 2's surround,
      and the rule's. The buildings at the ends of the block's other streets are squares,
      filled once looked down.
    - one Part B search each at p 40 and p 60 (the target whose surround-over-street time is nearest the median),
      in all six conditions, the dots showing the footpath each pilot's feeds have seen.

  Every flight's summary gives its block and street spreads.

How it is made, in two stages (`python -B -m citysim.split_replays [--stage refly|assemble]`):
- **refly** (~10 min on six E-core workers). Each pick's job is run again exactly as its run planned it
  (`split_probe.run_job`, `split.walk_job`, `split.search_job`; Phase 3's `lookdown.run_job` and
  `lookdown_search.run_job`). `split_probe.Flight` is swapped for a subclass that
  puts a `citysim.playback.Tape` on the simulation. The tape wraps `sim.step` and only reads, recording every drone's
  position, heading and shown feed, the pilot's heading and waypoint, and each loss and contact as it happens, every
  0.1 s (manoeuvres) or 0.2 s. For the searches, every `split.choose_exit` call is logged beside the real one: the
  exits weighed, their scores and the choice. **A pick is kept only if the job's scored rows come out identical to the
  ones its run stored**: all 15 picks did. A Phase 3 block re-flies one job per spread it shows, and is kept only if
  every one matches.
- **assemble**:
  - Adds Unity's flights from the bench copy's recorded trajectories (`D:\claude_obstacle_bench\runs\s4_p2_bench_*`,
    `s4_p2_routes_*`), scored with the replica's rules.
  - Writes each city's map (blocks, buildings, R's cylinders, street samples) and `index.json`.
  - Takes the captions from the pick tables (`P1_PICKS`, `WALK_PICKS`, `SEARCH_PICKS`). Each caption was checked
    against its flight's numbers.

Beside each tape the player gets:
- each street sample's first sight on a shown feed (the search band, 80 m);
- the decisions: each block's arrival, its commit, and its candidate streets with the time each was previewed;
- which drones were apart from the formation's main group (the cluster metric, every 0.5 s);
- the scored row's headline numbers.

`split_report` embeds `index.json` in the page and inlines the player. The published page fetches the cities and
flights from `replay/` beside it (11 MB, listed in `split_phase1_files.json`); the local `split_phase1.html` inlines
them all.

**The replays corrected a Phase 1 finding.** B's drones touch the building they flow round, not the buildings across
the street (6.3; lesson 32).

## 9. Phase 3: looking down the streets (a short pilot, 2026-10-28)

**Why.** The user's review of the Cities tab (2026-10-28) raised two points:
- **The preview measure is too generous.** In the p 40 · s 18 search the surround behaved as intended: the formation
  wrapped the building and could look down both halves of the block. But Phases 1 and 2 counted a street as
  previewed once a feed had seen the start of it. Seeing the start of a street is not looking down it, to the
  building at its end, which is what a searcher needs ("does the next building down this street have the goal?").
  That is where surrounding should pay.
- **d_ref is a key parameter, and must be reported and studied.** In the Phase 1 surround at p 75 · s 18 the spread
  was 1.8, and the block is clearly too large for it: the swarm cannot easily surround the building.

The user asked for a shorter study first, to check its results before another full one. Everything is variant R,
with R's confirmed profile (cohesion 30, street spread 1.0, exit stick 1.0). Code: `citysim/lookdown.py` (Part A),
`citysim/lookdown_search.py` (Part B), `citysim/tests/test_lookdown.py`. Outputs: `experiment_results/split_study/
lookdown_A/` and `lookdown_B/`.

### 9.1 The measure

Every exit of the block (a radiating street other than the entrance) is scored at three nested levels. Each counts
what a shown feed (FeedLatch) saw within `LOOK_RANGE_M` 80 m slant, through a clear 2-D line of sight, at the search
band (12.4 m, −20.8°):

| level | the exit counts once a shown feed has seen |
|---|---|
| entered (the old preview) | 60% of the street's centreline samples within 80 m of its corner V |
| seen to its end | 60% of its centreline samples within `STREET_END_M` 10 m of its far junction F |
| **looked down** (the primary) | `END_FRACTION` 0.5 of its **end targets**: the footpath ring (`FOOTPATH_OFF_M` 2 m outside the footprint, a point every 2 m) of the building at the street's end, keeping the points within `END_REACH_M` 12 m of the ring's point nearest F. That building belongs to the block ahead at F (`streetnet.Network.block_ahead`); when that block is a plaza, the targets are its lawn (a 3 m grid) near F. |

The per-block measure is the share of the block's exits looked down at the **decision**. The pass is aimed by its
decision and decides on arrival at the corner, as the street-bound swarm does, so at its decision it has seen what that
swarm has. What it sees going round counts for the next block, and is measured by the next junction (`look_F`).
`look_c_far` restricts to the exits not at the entrance corner.

### 9.2 Part A: one block, every way, every spread (`lookdown manoeuvres`)

- **Designs:** p 40, 50, 60, 75 at s 18. **Trials:** 30 per design, in Phase 1's cities (seeds 0–2, ten each).
- **Pairing:** every condition flies the same block, entrance, start, seed and exit. The exit is a side street drawn by
  Phase 2's chain rule (`split.exit_options`).
- **Conditions**, each at every block spread d_ref ∈ {1.0, 1.4, 1.8, 2.2, 2.6} (the street flown at 1.0):
  - street: decides on arrival at the corner;
  - hold T: a centring hold on the corner for T s at the block spread (Phase 2's `swarm_hold`);
  - pass: aimed through the block at its exit at the block spread (Phase 2's `swarm_pass`);
  - surround, aim T: full stick at the opposite exit for T s, then off down the exit (Phase 2's surround);
  - surround, firm T: a centring hold on the block's centre at stick 1.0, gain 0.5 for T s (Phase 1's firm hold).
- **T** ∈ {6, 12, 20} s. 4 designs × 5 spreads × 30 trials × 11 conditions = 6,600 flights.
- **Every row records its block spread (`d_block`) and street spread (`d_street`)**, and the page reports both for
  every flight it replays, Phases 1 and 2 included. The trials' dial runs from 0.4 to 1.6: the spreads above it are
  wider than a pilot can dial today.

### 9.3 Part B: searching for a goal at a building (`lookdown_search`)

Fixed before Part A's results were read:
- **The surround Part B flies, by rule** (`lookdown_search.choose`). Among the surround settings (mode, T, d) that are
  safe (drones lost ≤ 0.03 and contacts ≤ 0.1 per manoeuvre) and whose time price is ≤ 1.5, the one with the most exits
  looked down at the decision. Ties go to the shorter T, then the smaller d. The time price is the median over trials of
  the time from the block to the next junction, over the street-bound swarm's in the same trial. If no safe setting is
  within the price, the safe setting with the most looked down per second of that time; if none is safe, the least
  unsafe.
- **The target** stands on the footpath ring of a random interior building, as a goal does. It is at least 300 m by
  street from the start, out of the start's clear sight within 200 m, and at least 60 m inside the arena. It is found at
  its first sight on a shown feed within 80 m (40 and 200 m recorded too), with a 1,200 s cap. A swarm under four drones
  has not found it.
- **The pilot** is informed, as Phase 2's is, but its map is of where the target can be: the footpath points its own
  feeds have seen within 80 m. At each decision it takes the exit with the most unseen footpath of the blocks along it
  and at its far junction, per metre of travel. With none unseen nearby, it heads for the nearest junction whose blocks
  still have 20 m unseen (Phase 2's threshold). The code is the same in every condition.
  - **Checked during the run, and left as fixed (2026-10-28).** In the first rounds the surrounds' searches at p 40
    often had no exit with unseen footpath nearby: about twice a search, against 0.25 for the street-bound swarm.
    Their unfound searches had seen 95% of the footpath when the 20 minutes ran out. That looked like a pilot choosing
    at random once only slivers under 20 m were unseen, which would charge the conditions that see most of every
    building. It was not. Re-flying those searches with every decision logged showed each such move heading for a
    junction with 20 m unseen, and no random choice at all; the counter (`blind_decisions`) counts both.
  - On the misreading, the run was stopped after 23 searches and restarted with a one-point threshold. That was
    stopped after 30 searches (`lookdown_B_v2`, kept, not analysed): it changed individual flights both ways and left
    more targets unfound (11 flights against 7), by chasing slivers its cameras could not see. The first run then
    resumed where it had stopped, unchanged, so `lookdown_B` is the rule as fixed in advance.
- **Conditions:**
  - `single`;
  - `swarm_street`;
  - `swarm_hold` (the rule's T and d);
  - `swarm_pass` (the rule's d);
  - `swarm_hop` (the rule's setting);
  - `swarm_hop_p2` (Phase 2's surround: aim 6 s at Phase 1's d*, 1.0 at p 40 and 1.2 at p 60).
- **Designs:** p 40 · s 18 and p 60 · s 18, on 8 fresh cities each (seeds 200–207), 4 targets per city.
- **Read-out:** city-mean log ratios of the restricted mean time to find (80 m), with t intervals over the 8 cities. This
  is a pilot: read the direction and the size, not significance.

### 9.4 Part A results (`lookdown_A`, 120 trials, 6,600 flights, none failed)

Shares of a block's exits at the decision, means over 30 trials per design. Intervals are t intervals over trials (3
cities per design, so they are optimistic about city-to-city variation).

- **The old preview overstated what the street-bound swarm knows, and more so the larger the block.**

  | design | entered (the old preview) | seen to its far end | looked down | exits per block |
  |---|---|---|---|---|
  | p 40 · s 18 | 71% | 59% | 41% | 7.1 |
  | p 50 · s 18 | 53% | 40% | 28% | 6.1 |
  | p 60 · s 18 | 37% | 28% | 21% | 5.6 |
  | p 75 · s 18 | 26% | 19% | 9% | 5.5 |

- **One drone looks down almost none of them** (`lookdown_A1`, added 2026-10-28 for the redesigned page: the lone
  drone over the same 120 blocks, from the same start with the same exit, round the block by the shorter side;
  `python -B -m citysim.lookdown single`). Phase 3 had compared the ways of flying the swarm only, and the trials
  compare the swarm with one drone.

  | design | one drone | one drone, stopping to look at the corner | swarm in the streets |
  |---|---|---|---|
  | p 40 · s 18 | 6% | 26% | 41% |
  | p 50 · s 18 | 5% | 11% | 28% |
  | p 60 · s 18 | 3% | 6% | 21% |
  | p 75 · s 18 | 1% | 1% | 9% |

  By the next junction one drone has looked down 34 / 30 / 25 / 22% of the block's exits, the swarm in the streets
  80 / 57 / 58 / 35%. Most of the swarm's lead in what it sees before choosing is there without surrounding.

- **Surrounding the block is what looks down its exits; pausing at the corner barely does.** At d_ref 1.8:

  | design | street | corner hold (any T) | surround, aim 6 s | aim 12 s | aim 20 s | firm 12 s |
  |---|---|---|---|---|---|---|
  | p 40 | 41% | 51% | 71% | 93% | 95% | 83% |
  | p 50 | 28% | 34–35% | 49% | 79% | 85% | 71% |
  | p 60 | 21% | 25% | 32% | 50% | 70% | 46% |
  | p 75 | 9% | 9% | 10% | 17% | 34% | 18% |

  - **Against a corner hold of the same length at the same spread**, the surround looks down 20 / 15 / 7 / 1 points more
    at 6 s, 42 / 44 / 26 / 8 at 12 s and 44 / 49 / 45 / 25 at 20 s (p 40 / 50 / 60 / 75; every interval above 0 except
    p 75 at 6 s).
  - A hold longer than 6 s adds nothing: the formation sees from the corner what it can see from there.
  - Aim and turn beats the firm hold from 12 s on: it moves the formation round the block, where the hold parks it on
    the centre.
- **d_ref matters, as the user expected: the wider the block spread, the more the surround looks down, up to the
  point where it starts touching buildings.**
  - Aim and turn for 12 s, against the same trial at d_ref 1.0: +7 / +12 / +13 / +5 points at 1.8, +11 / +23 / +25 /
    +11 at 2.6.
  - Contacts per surround (every T and mode): at most 0.06 up to d_ref 1.8, 0.12–0.27 at 2.2, 0.48–0.71 at 2.6.
    The pass and the hold touch buildings from 2.2 too, less often (0.10–0.43 at 2.6).
  - **The contacts are with the building being surrounded, not the buildings across the street.** Re-flying 20 trials
    at 2.6 (p 40 and 60, every condition; `python -B -m citysim.lookdown_contacts`, which puts each contact on the
    nearest footprint and writes `lookdown_A/contacts.csv`) located 85 contacts. 84 were with the block's own building, and 76 of those came between
    arriving at the block and the decision, during the surround or the corner hold. What sets the safe spread here is
    the building the formation is wrapped round, not the street width. Why a wider formation presses into it is not
    established.
  - **So with 18 m streets the widest safe spread is 1.8 at p 40–60, and 2.2 at p 75** (where only the aim-and-turn
    surrounds stay under the bar). Both are wider than the trials' dial allows (0.4–1.6), and at p 40–60 wider than
    Phase 1's d* (1.0–1.2). d* measured whether the formation encloses the block; looking down its exits pays for more
    spread than that.
- **At p 75 · s 18 no safe setting surrounds the block well enough to look down its exits.** This is the user's
  observation. At 1.8, Phase 1's d* there, a 20 s aim-and-turn surround encloses the block (S1's test) in only 40% of
  trials and looks down 34% of its exits. At 2.2, the widest safe spread, it encloses 67% and looks down 38%; at 2.6
  it encloses 83% but touches buildings (0.43 contacts per manoeuvre) and still looks down only 42%. Ten drones are
  too few for a ring that size at 18 m streets.
- **The cost is time.** From arriving at the block to the next junction, at d_ref 1.8 (median s, p 40 / 50 / 60 / 75):

  | way | p 40 | p 50 | p 60 | p 75 |
  |---|---|---|---|---|
  | street | 15.0 | 14.6 | 17.8 | 22.2 |
  | pass | 14.5 | 14.5 | 18.2 | 21.5 |
  | hold 6 s | 21.3 | 20.8 | 24.0 | 28.5 |
  | surround, aim 6 s | 18.0 | 19.8 | 21.8 | 25.8 |
  | surround, aim 12 s | 23.8 | 26.9 | 26.2 | 30.2 |
  | surround, aim 20 s | 26.3 | 32.8 | 34.5 | 38.0 |

  An aim-and-turn surround of 6 s takes less time than a 6 s corner hold and looks down more.
- **The pass's sight going round the block is mostly what the street-bound swarm sees anyway by the next junction.**
  - By the next junction the pass has looked down 83 / 62 / 59 / 33% of the block's exits, against the street-bound
    swarm's 80 / 57 / 58 / 35%.
  - The 12 s surround has looked down 97 / 91 / 74 / 42% by then.
  - Phase 2's finding that the pass "previews 98% of its streets" was the old measure, scored at its aim point.
- **The rule's picks for Part B:**
  - p 40 · s 18: aim and turn 6 s at 1.8. It looks down 71% at the decision, at a time price of 1.24.
  - p 60 · s 18: aim and turn 12 s at 1.8. It looks down 50%, at 1.44.
  - p 50 · s 18 and p 75 · s 18 (not searched): a firm hold of 6 s at 1.8 (51%, 1.35) and aim and turn 12 s at 2.2
    (21%, 1.36).

  The 1.5 time price is what holds the pick to short surrounds. With no time bar the most a safe surround looks down is
  95% at p 40 (aim 20 s at 1.8, price 1.96), 79% at p 50 (aim 12 s at 1.8, 1.75), 63% at p 60 (aim 20 s at 1.4, 1.73)
  and 38% at p 75 (aim 20 s at 2.2, 1.68).

### 9.5 Part B results (`lookdown_B`, 64 searches, 384 flights, none failed)

Eight cities per design, four targets each. Found: the share of searches that saw the target within 80 m by the
20-minute cap. Mean time: the restricted mean time to find, a target not found counting at the cap.

| condition (block spread, surround) | p 40 found | p 40 mean time | p 60 found | p 60 mean time |
|---|---|---|---|---|
| swarm in the streets (1.0) | 97% | 301 s | 97% | 250 s |
| surround, the rule's pick (1.8; aim 6 s at p 40, 12 s at p 60) | 88% | 409 s | 94% | 406 s |
| Phase 2's surround (d*: 1.0 / 1.2; aim 6 s) | 91% | 410 s | 100% | 332 s |
| hold at the corner (1.8; 6 / 12 s) | 94% | 457 s | 97% | 418 s |
| pass round (1.8) | 94% | 356 s | 97% | 285 s |
| one drone | 78% | 688 s | 97% | 480 s |

City-mean ratios of that time, with t intervals over the cities (8 per design, 16 pooled):

| contrast | p 40 | p 60 | pooled |
|---|---|---|---|
| surround ÷ street | ×1.27 (1.03–1.56) | ×1.74 (1.17–2.58) | **×1.48 (1.18–1.87)** |
| surround ÷ hold | ×0.83 (0.72–0.96) | ×0.98 (0.63–1.52) | ×0.90 (0.72–1.13) |
| surround ÷ Phase 2's surround | ×0.99 (0.60–1.63) | ×1.13 (0.93–1.37) | ×1.06 (0.82–1.37) |
| pass ÷ street | ×1.20 (0.98–1.46) | ×1.11 (0.98–1.26) | ×1.15 (1.03–1.30) |
| street ÷ one drone | ×0.43 (0.30–0.62) | ×0.45 (0.35–0.58) | ×0.44 (0.36–0.55) |

- **Looking down more does not find a goal at a building sooner. The surround takes about half as long again as the
  swarm kept in the streets.** The interval excludes 1 pooled and at each design.
- **The decomposition is Phase 2's (lesson 27).** Over the searches both found, the surround needed 6% fewer
  decisions at p 40 and 3% fewer at p 60, but each took 27% and 74% longer, so the time was ×1.20 and ×1.68. Its lead
  at each decision (71% of the exits looked down against 41% at p 40) is information the street-bound swarm gets one
  block later, by flying the exit it chose: by the next junction that swarm has looked down 80% of the block's exits.
- **The surround still beats a pause of the same length at the corner at p 40** (×0.83), as H1 found on the old
  measure. The wider spread and longer surround the rule picked do no better than Phase 2's surround (×1.06).
- **Costs per search:** the surround lost 0.09 drones at p 40 and 0.31 at p 60, with 1.31 and 0.13 building
  contacts, against none lost and 0.09 and 0 contacts for the street-bound swarm.
- The cameras remain the effect: the swarm in the streets finds the goal in 44% of one drone's time.

### 9.6 What Phase 3 means

- **The user's two points hold, and the stricter measure changes the picture of what each way of taking a block
  shows.** Scored on looking down, the street-bound swarm knows much less at each decision than Phase 2's preview said
  (41% of a block's exits at p 40, 9% at p 75), and surrounding the block is what looks down the rest. d_ref is a
  first-order factor: the surround looks down more the wider it flies, up to where its drones touch the building,
  and that is beyond both Phase 1's d* and the trials' dial.
- **It does not change S4's answer for search.** With the goal at a building, where looking down should pay most,
  the surround finds it about 1.5 times as slowly as the swarm in the streets. Seeing down every exit before choosing
  saves a few decisions; the street-bound swarm sees down the exit it takes as it flies it, and each surround costs
  more per block (in the search, a median 19.5 s against 15.3 s per decision at p 40, and 29.9 s against 17.4 s at
  p 60).
- **For the next full study**, if there is one:
  - make looking down the preview measure;
  - fly d_ref as a factor, and widen BlockCityWorld's dial if people are to fly it;
  - fly a surround long enough for its block as a condition of its own, rather than letting a time bar choose
    against it;
  - use more cities (Part A's intervals are over trials in three cities; eight per design in Part B give a pilot's
    width);
  - try 12 and 24 m streets, since the contacts that cap the spread are with the building being surrounded;
  - check the wide spreads on the Unity bench.

  The place a surround could still pay is a task where the decision itself is the cost: a pilot who must commit
  without flying down an exit, or a goal that has to be confirmed from more than one side. Neither was flown.

### 9.7 At a 250 m sight range (2026-10-28)

**Why.** Every S4 measure counted a feed as seeing a point within 80 m, a range inherited from the goal's
recognition distance. The user judged that too short to mean anything for looking down a street to the building at its
end, and asked for 250 m across every condition. With 80 m, half the exits' far ends on the smallest blocks were out
of range of the corner where the pilot chooses (median 89 m away; 154 m on the largest blocks), so the range, not the
formation, set most of the numbers.

**What changed.** One recorded setting per run, `sight_m` (Part A's `look_range_m`), default 80 so every earlier run
reproduces bit for bit (checked on a stored job of each). It sets:
- Part A: the range every level (entered, seen to its end, looked down) is scored at;
- the searches (Part B and Phase 2's informed search): the pilot's map of what it has seen, the first sight that
  stops the search (recorded beside 40 / 80 / 200 m), and the radius within which the target must start out of sight
  of the start (at least 200 m, so a target is never in view at second zero).

The flights are the same at either range; only what counts as seen changes. New runs, beside the 80 m ones:
`lookdown_A_r250` and `lookdown_A1_r250` (`lookdown manoeuvres|single --sight 250`), `lookdown_B_r250` (`lookdown_search
choose --partA lookdown_A_r250 --sight 250`, then `search`), `split_search_r250` (`split search --sight 250`; recheck
6 of 6 identical). Phase 1 and Phase 2's walks were not re-run: Phase 1 decided on whether the manoeuvre works, and
the walks' range-dependent measures are the old preview and S3's hidden-point find. The page's answer and Look down
tab use the 250 m runs (`split_report.SIGHT_M`); Phase 2's own tabs stay its pre-registered 80 m record, and say so.

**Part A** (exits looked down at the decision, %, p 40 / 50 / 60 / 75):

| way | at 80 m | at 250 m |
|---|---|---|
| one drone | 6 / 5 / 3 / 1 | 33 / 33 / 27 / 23 |
| one drone, stopping to look | 26 / 11 / 6 / 1 | 64 / 55 / 43 / 39 |
| swarm in the streets | 41 / 28 / 20 / 9 | 87 / 75 / 63 / 51 |
| surround, aim 12 s at d_ref 1.8 | 93 / 79 / 50 / 17 | 99 / 97 / 92 / 81 |
| surround, aim 20 s at d_ref 1.8 | 95 / 85 / 70 / 33 | 99 / 98 / 99 / 92 |

- The swarm in the streets already looks down most exits; the surround's lead is 12 points at p 40 and 30 at p 75.
  By the next junction the swarm in the streets has looked down 99 / 97 / 95 / 90%, one drone 55 / 54 / 48 / 48%.
- **d_ref barely matters for seeing at 250 m**: a 12 s surround looks down 99% at p 40 at every spread from 1.0 to 2.6,
  89–95% at p 60, 80–82% at p 75. It still matters for enclosing the block (S1's test, unchanged by range) and the
  safe spread is still 1.8 (2.2 at p 75), set by contacts with the building surrounded.
- The rule's picks: a 6 s firm hold at 1.8 at p 40 (96%), aim 6 s at 1.8 at p 50 (91%), aim 12 s at 1.8 at p 60 (92%),
  aim 12 s at 1.4 at p 75 (82%).
- The pass by the next junction: 100 / 96 / 93 / 91%, the street-bound swarm 99 / 97 / 95 / 90%.

**The searches** (time to find at 250 m as a share of one drone's, city means, 95% intervals):

| condition | Part B: a goal at a building (16 cities) | Phase 2: a target anywhere (48 cities) |
|---|---|---|
| swarm in the streets | 0.51 (0.40–0.66) | 0.53 (0.44–0.64) |
| swarm passing round | 0.58 (0.42–0.79) | 0.53 (0.44–0.64) |
| swarm surrounding | 0.65 (0.46–0.90) | 0.58 (0.48–0.69) |
| swarm holding at the corner | 0.73 (0.55–0.95) | 0.67 (0.57–0.79) |
| surrounding ÷ streets | ×1.26 (1.01–1.58); ×1.21 at p 40, ×1.32 at p 60 | ×1.08 (0.97–1.21) |

- The decomposition is the same as at 80 m. Against the swarm in the streets, on searches both found with at least
  one decision: one drone needs 2.0–2.3× the decisions (Phase 2: 1.96×), each 9–19% shorter; surrounding needs 14–15%
  fewer, each 23–60% longer.
- At 250 m a few searches see the target before the first decision (7 of 384 flights in Part B, 68 of 2,016 in
  Phase 2); they count in the times and are left out of the decomposition.
- One drone stopping to look at each corner is slower than one that doesn't (×1.14, 1.01–1.30).

**So the answer does not change with the range.** Everyone sees far more at 250 m, the swarm still finds a target in
about half of one drone's time, and surrounding each block is still no quicker than flying the streets.
