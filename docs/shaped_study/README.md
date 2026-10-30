# The block-shaped study (S5 Block-Shaped Surround)

S4's question asked again in BlockShapedCityWorld's city: in a city whose buildings fill their blocks, does
surrounding a block (to look down every street leaving it before choosing) help a pilot flying the swarm beat a pilot
flying one drone on a visual search? With the spread that gets the formation round a block efficiently, a footpath
comparison (the concern that a footpath lets one drone see more), block size and street width, and four ways of
laying out the blocks (the hope that some layout puts more streets round each building, hidden from one drone).

Page: [S5 Block-Shaped Surround](https://claude.ai/artifact/PrHZmczN5CnhAvZLQTCkb9). Registry entry:
[S5_block_shaped_surround.md](../studies/S5_block_shaped_surround.md). Run 2026-10-28/29.

## 1. The city and how it is flown

- **BlockShapedCityWorld** (session vr-swarm-simulation-91, committed 9cb451a8) is S4's variant B made flyable in
  Unity: S4's own `bbB_p40_s18_seed1` layout (random RSA block seeds, p 40, s 18), each block built over to the back of
  a 3.5 m footpath as one convex prism (`PrismBuilding`, a convex MeshCollider), on BlockCityWorld's SwarmManager
  (`r0_coh` 30).
- **Unity flies it with today's force law.** OlfatiSaber wraps each prism in the circumradius of its axis-aligned
  bounds (centre at the bounds' centre); contacts meet the true prism. The replica's `PolyCity` does exactly that under
  the default obstacle model, so every S5 flight uses `split_probe.model(st, "R")` on a `PolyCity`, never S4's
  closest-point model (which the C# never had). A Unity smoke test of the scene (91: S4's 16 manoeuvres) lost one
  colliding pair, in line with what the replica shows below.
- **The flights are S4 Phase 3's**, with the city named per job: `lookdown.run_job` (Part A, one block per trial, every
  way from the same start with the same exit) and `lookdown_search.run_job` (Part B, a goal on the footpath in front of
  a building). Both now take an optional `city` (`blockcity.Params` fields: variant, inset, seeds, jitter, relax), an
  optional street spread (`d_street`) and a condition subset; their defaults are S4's, which still reproduce bit for
  bit. A camera counts as seeing within 250 m (lesson 37).

## 2. The designs

| name | p · s | footpath | layout | what it varies |
|---|---|---|---|---|
| base | 40 · 18 | 3.5 m | random (RSA) | the scene's city |
| foot15 | 40 · 18 | 1.5 m | RSA | a narrow footpath |
| foot0 | 40 · 18 | none | RSA | the facade on the kerb |
| p35 | 35 · 18 | 3.5 m | RSA | smaller blocks |
| p30 | 30 · 18 | 1.5 m | RSA | smaller still (at 3.5 m only 47 of 574 blocks would hold a building) |
| s20 | 40 · 20 | 3.5 m | RSA | wider streets |
| lloyd2 | 40 · 18 | 3.5 m | RSA relaxed twice (Lloyd) | more even blocks |
| hex15 | 40 · 18 | 3.5 m | hexagonal lattice, seeds moved up to 15% | near-regular blocks |
| hex0 | 40 · 18 | 3.5 m | exact hexagonal lattice | regular hexagons |

Geometry (`shaped geometry`, means over Part A's cities and Part B's, 11 seeds; `geometry.csv`):

| | base | foot15 | foot0 | p35 | p30 | s20 | lloyd2 | hex15 | hex0 |
|---|---|---|---|---|---|---|---|---|---|
| blocks left green | 1% | 0% | 0% | 43% | 55% | 12% | 1% | 0% | 0% |
| block sides that are streets | 4.45 | 4.44 | 4.44 | 4.28 | 3.78 | 4.26 | 4.63 | 5.68 | 6.00 |
| streets leaving its corners | 7.93 | 7.93 | 7.93 | 8.60 | 9.98 | 8.40 | 7.59 | 6.31 | 6.00 |
| streets the avoidance circles close | 1% | 12% | 33% | 0% | 0% | 0% | 0% | 0% | 0% |

A side shorter than its junctions' setback is not a street (its junctions merge), so random blocks have fewer street
sides but more exits; even blocks have six of each. Two circles that overlap across a street close it to the force
law; a narrower footpath grows each building and its circle.

## 3. Step 1: the spread (d_ref) that passes round a block efficiently

**Step 1** (`shaped dref`, 2,310 flights on the base design): every way of taking a block (S4 Part A's eleven) at block
spreads 0.8–2.2, the streets at S4's 1.0. Its rule, fixed in advance: among surrounds that are safe (at most 0.03 drones
lost and 0.1 contacts per block), reform before the next junction in 90% of trials and enclose the block (S1's test)
in 70%, the quickest. **Nothing was safe.** The swarm lost drones in drone-drone collisions everywhere, the swarm in the
streets too: 0.33 per block (5 of 30 trials, each 10–20 s after reaching the block, on the block's sides, with the
formation's centre 2–11 m outside the nearest avoidance circle). On the same streets S4's catalogue buildings lost
none in 30 trials. The rule's fallback ("the least unsafe") picked a 12 s firm hold at 1.2 that never encloses the
block; **it is not used**, and is kept in `chosen.json` as `step1`.

**Step 1b** (`shaped dref2`, 1,680 flights), added after step 1 because it held the street spread at 1.0 and so could
not find a spread that fits these streets: one spread for the whole flight (streets and block), 0.6–1.8. Its rule,
fixed before it ran: among surrounds that enclose the block in at least 70% of trials, the fewest drones lost per
block, then the quickest, then the smaller spread; the safety bar is reported, not required.

| d_ref | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 |
|---|---|---|---|---|---|---|---|
| drones lost per block, swarm in the streets | 1.13 | 0.47 | 0.33 | 0.13 | 0.07 | 0 | 0 |
| 20 s firm hold: encloses / lost | 0% / 1.13 | 27% / 0.40 | 80% / 0.13 | **97% / 0** | 97% / 0.20 | 100% / 0 | 100% / 0 |
| 20 s aim and turn: encloses | 20% | 33% | 30% | 50% | 63% | 67% | 77% |
| 12 s aim and turn: encloses | 0% | 0% | 0% | 0% | 3% | 20% | 33% |
| block to next junction (s): firm hold 20 / streets | 35 / 18 | 35 / 19 | 35 / 20 | 35 / 20 | 35 / 21 | 35 / 21 | 36 / 23 |

- **The collisions come from a formation squeezed between the circles**: the wider the spread, the fewer.
- **The rule picks a 20 s firm hold on the block's centre at d_ref 1.2**: it encloses 97%, loses nothing, touches
  nothing, and costs 15 s a block over the streets. At 1.2 the swarm in the streets still loses 0.13 drones a block;
  none from 1.6.
- **Aim and turn (flowing round) does not get round these blocks** within 12 s at any spread, and within 20 s at most
  77% of the time. Every later step flies d_ref 1.2 in the streets and at the block, and the 20 s firm hold (and holds
  at the corner for 20 s).

## 4. Part A: what each way looks down (`shaped partA`, 1,620 flights)

30 trials per design in cities 0–2, each one block flown five swarm ways at d_ref 1.2 plus the lone drone's two.
Exits looked down at the decision (by the next junction in brackets), drones lost per block in the streets, and the
surround's enclosure:

| design | one drone | one drone, looking | swarm in the streets | holding | surrounding | lost, streets | surround encloses |
|---|---|---|---|---|---|---|---|
| base | 28% (49%) | 55% | 74% (98%) | 82% | 95% | 0.13 | 97% |
| foot15 | 17% (41%) | 37% | 51% (91%) | 63% | 78% | 1.33 | 27% |
| foot0 | 11% (23%) | 22% | 41% (77%) | 47% | 57% | 1.87 | 10% |
| p35 | 51% (72%) | 73% | 88% (99%) | 91% | 99% | 0 | 100% |
| p30 | 52% (69%) | 76% | 90% (100%) | 93% | 99% | 0 | 100% |
| s20 | 45% (64%) | 64% | 83% (99%) | 86% | 96% | 0 | 100% |
| lloyd2 | 30% (53%) | 53% | 71% (98%) | 79% | 95% | 0 | 100% |
| hex15 | 27% (65%) | 64% | 69% (100%) | 82% | 99% | 0 | 100% |
| hex0 | 33% (63%) | 65% | 65% (100%) | 80% | 100% | 0 | 100% |

S4's catalogue city on the same streets, same range: one drone 33%, one drone looking 64%, swarm in the streets 87%, a
12 s aim-and-turn surround at 1.8 99%.

- **Block-shaped buildings hide more from everyone**, and surrounding uncovers 21 points over the streets (S4: 12).
- **The footpath shows one drone more** (28% at 3.5 m, 17% at 1.5 m, 11% with none), as the user expected, but a
  narrower one lets the circles close streets: the swarm loses 1.3–1.9 drones a block and cannot surround.
- **Even layouts** put six streets round every building against 4.5, but have six exits against 7.9. One drone sees
  the same share on every layout (27–33%), so the random layout leaves the most exits hidden from it (5.7 against
  4.0 on exact hexagons). The even layouts lost no drones here, and the surround enclosed every block.
- **Smaller blocks and wider streets show one drone more** (45–52%): at p 35 and p 30 many blocks are greens.

## 5. Part B: a search for a goal at a building (`shaped partB`)

Six designs, eight cities each (200–207), four targets each: one drone, the swarm in the streets, holding 20 s,
passing round and surrounding (20 s firm hold), at d_ref 1.2, found at the first sight within 250 m, 20-minute cap.
Time to find as a share of one drone's (city means, 95% intervals), and surrounding against the streets:

| design | in the streets | passing round | surrounding | holding | surrounding ÷ streets | found: one drone / streets / surround | lost per search, streets |
|---|---|---|---|---|---|---|---|
| base | 46% (32–67) | 50% | 61% (44–86) | 79% | ×1.33 (1.19–1.49) | 72% / 97% / 94% | 1.3 |
| foot15 | 52% | 57% | 61% | 72% | ×1.19 (0.85–1.67) | 31% / 69% / 66% | 4.1 |
| p35 | 66% | 69% | 110% | 108% | ×1.66 (1.26–2.20) | 84% / 91% / 84% | 0.3 |
| p30 | 59% | 55% | 78% | 92% | ×1.32 (1.00–1.72) | 78% / 91% / 88% | 0.6 |
| s20 | 67% | 73% | 110% | 115% | ×1.64 (1.16–2.32) | 91% / 94% / 88% | 0.1 |
| hex15 | 71% (35–143) | 72% | 96% | 121% | ×1.35 (1.00–1.81) | 91% / 91% / 97% | 0.9 |

- **The swarm in the streets beats one drone on every design** (46–71% of its time), and **surrounding is slower than
  the streets on every design** (×1.19–1.66).
- **Why, on the base design**: surrounding saves 29% of the decisions (S4: 15%) but each takes 86% longer (the 20 s
  hold). One drone needs 2.0 times the swarm's decisions.
- **The losses carry over**: at 1.2 the swarm in the streets loses 1.3 drones a search on the scene's city; with a 1.5 m
  footpath 4.1, and it then finds only 69% of the goals.

## 6. What S5 means

- The swarm's edge over one drone holds in the scene's city (46% of one drone's time), as in S4 (51–53% at the same
  range), and it is still the cameras. Surrounding is not how to use it: getting round a block that fills its plot
  takes a 20 s hold, more than the better choice saves.
- For the trials: fly BlockShapedCityWorld as built (it hides the most from one drone of everything flown here and
  still flies), start the spread dial at 1.2 or more there (no losses in the streets from 1.6), and compare the swarm
  in the streets with one drone. A narrower footpath, which would hide more from one drone, needs a tighter avoidance
  shape in the force law first (the footprint's own circle or the closest point), a change neither Unity nor the replica
  has. A hexagonal layout loses fewer drones at the cost of fewer exits per block.
- Limits: scripted pilots, the 2-D replica, Part A's three cities and Part B's eight; S5's settings have not been
  flown on the Unity bench.

## 7. Running it

From `Assets/Scripts/Experiment`, the `stitching` env:

```
python -B -m citysim.shaped geometry
python -B -m citysim.shaped dref        # step 1 (kept: its result is why step 1b exists)
python -B -m citysim.shaped dref2       # step 1b
python -B -m citysim.shaped choose      # step 1b's rule -> chosen.json
python -B -m citysim.shaped partA
python -B -m citysim.shaped partB
python -B -m citysim.shaped_replays     # the Cities tab's replays, each re-flown and checked identical
python -B -m citysim.shaped_report      # -> shaped_study_bare.html + shaped_study_files.json
```

Outputs in `experiment_results/shaped_study/`: `geometry.csv`, `dref/`, `dref2/`, `chosen.json`, `partA/`,
`partB_<design>/`, `replay/`, the page. Every run is resumable (results are appended per job). The page's styles are
S4's page's own, read at build time.
