# The density study

How much of the swarm's advantage in finding a target depends on the city it flies in, predicted before
the human trials so the trials can be designed around it. Cities are generated with controlled clutter
and controlled goal geometry. Four tiers of model, each checked against the next, predict what each
interface can see at a goal. Three trial levels are then picked by a rule fixed in advance. The trial
protocol built on them is [protocol.md](protocol.md).

Everything runs from `Assets/Scripts/Experiment` with the `stitching` env's Python
(`%LOCALAPPDATA%\miniconda3\envs\stitching\python.exe`), as `python -B -m citysim.<module>`. Outputs go to
`experiment_results/density_study/`.

To see the study rather than read it, open
[explorer/density_study.html](../../experiment_results/density_study/explorer/density_study.html) in a
browser. It decodes the level codes, maps one generated city of every level and the three trial cities,
animates a goal scene for each interface, plays the Unity flights and charts every tier's results and
checks. It is one file with the
data inside, but it loads d3 and its fonts from CDNs, so it needs a network connection.

## 1. Where it stands (2026-10-13)

| | result |
|---|---|
| Levels | **L1 = C0-G0, L2 = C0-G10, L3 = C2-G7** ([levels.json](../../experiment_results/density_study/levels.json)), selected with the transits on the street route. **On hold since the pilot paths:** L3 fails G3 there (below) |
| G1 (Tier B: the levels exist) | **pass.** L2's swarm − single exposure advantage is 2.22× L1's (+0.106 vs +0.048), 40 times its seed SD; L3's reverses (−0.050). |
| Replica gate (Tier B' usable) | **pass** with buildings as contacts with friction; fails with them as kills (§5.1). On the pilot paths: passes in L1 and the porous C1p, fails on spacing and losses in the porous C2p and in L3; the transit measures agree everywhere except the stick adjustment count and, in L3, speed (§5.5) |
| G2 (Tier B': spacing at L3) | **pass** on both routings: 6.6 m mean nearest-neighbour spacing in transit at spread 1.0 on the street route, 9.5 m on the pilot paths, against the 3 m internal_3's losses came at |
| G3 (Tier C: losses at L3) | street route: **pass, narrowly**, 2.21 ± 0.38 drones per run at internal_3's transit-spread use against a bar of 2.5 (2.92 as flown). **Pilot paths: fail, 2.99 per run** (7.0 per flight at spread 0.4, 1.0 at 1.0); the rule then has no L3 ([levels_recheck.json](../../experiment_results/density_study/levels_recheck.json)) |
| What each interface sees (task-independent) | swarm ÷ single drone: 5.7–6.3× the ground on screen, 1.6–1.8× in the headset's view, 2.1–2.3× the sweep width in flight; 0.8% of its ground first seen round a corner in the open city, 3–8% in dense ones (§2.1) |
| Transit on pilot paths | 1.05–1.09× the straight line for both interfaces (1.33–1.53× on the street route); the swarm needs a fifth (L1) to a twelfth (L2) of the single drone's stick adjustments, and 0.6 of them at L3 (§4, Tier B') |
| Goal discovery (co-primary) | internal_3 scored (§5.6); predicted swarm ÷ single discovery time for an unannounced goal 0.61–0.69 at all three levels on shown feeds (§4, discovery) |
| internal_3 reproduced / predicted | replay exact (24 goals within 0.04 s); swarm displayed exposure predicted (median 0.42 vs 0.44 measured); head-visible and single-drone exposure under-predicted (§5.2) |
| Unity procedural city | built from a layout, round-trip exact; Play-mode probe passes all seven checks (§5.4) |

**A decision is needed before the pilot block: G3 fails for L3 once the drones fly through the city.** The levels
were picked with every transit on the street route. Flown the way pilots fly, through the blocks, the packed L3 loses
more drones at the tight spread (7.0 per flight at 0.4, against 5.0 on the street route) and a little more at 1.0
(1.0 against 0.83). That puts it at 2.99 per run against G3's bar of 2.5. Every other packed level flown fails too (C1-G0
and C1-G7 2.71, C2-G6 2.93), while the open levels lose fewer than before (L1 0.55, L2 0.61). The losses are
drone–drone collisions at the tight spread, between walls. The rule's only L3 candidate was C2-G7, so
re-run on the pilot paths it picks no L3 at all; L1 and L2 are unchanged. Nothing has been changed: `levels.json`
still holds the street-route picks, and the re-run is in `levels_recheck.json`. The options are in §8.

**Read the levels with one caveat.** L3 is a knife-edge between two goal geometries. C2-G7 (10 m
occluder, ring at 17 m, 6 patrolled buildings) reverses the advantage. Its neighbour C2-G6 (13 m, 20 m, 8)
has +0.003 ± 0.002, which misses the rule's "vanished within one SD" test by 0.16 SD. Both are
envelopment failures. Before the replica's contact model gained friction C2-G6 passed by 0.05 SD and was
the pick. Tier C flies both.

## 2. The question and the mechanism

A swarm of ten drones shows its hull drones' feeds on a ring round the pilot (`OUTER_CIRCLE`). At a goal, a
pedestrian wearing a hat walks a loop round one of the buildings ringing a central occluder. A single drone
sees one side of the occluder at a time and has to fly round it. A swarm wide enough to **envelop** the
occluder sees all sides at once, and its rim sees round corners its centre cannot (**parallax**). The
study manipulates what decides whether that happens:

- **Goal geometry G**: occluder width w_c, ring radius r_g, number of patrolled buildings n_b. The
  clearance c_g (goal centre to the nearest building other than the occluder) follows from them.
- **Clutter C**: the city's block spacing p, street width s, coverage α (building footprint over ground).
  Its free-sight length ℓ = π⟨A⟩ / (α⟨P⟩) is the mean distance a sight line travels before a building
  (the Boolean blockage model); compare it with the 40 m recognition range R.

The swarm's advantage should grow with w_c while the formation can still envelop the occluder (the
envelopment limit, set by the spread and c_g), and collapse where it cannot. Clutter matters less at the
goal than in transit, where it costs spacing, speed and drones.

**Exposure** is the measure throughout. It is the share of walker-time a walker is recognisable on a
feed: within R slant range, inside the camera's field of view (74.4° × 46.2°) and vertical band, with a
clear line of sight. E_any counts any shown feed. **E_head** counts only feeds inside the pilot's ±50°
field of view, averaged over internal_3's recorded head-yaw distribution. For the single drone the two are
equal. Every walker is scored, because the target is a random one of the goal's walkers.

### 2.1 What each interface can see, task-independent ([capability/](../../experiment_results/density_study/capability/capability))

No goals and no walkers: how much ground each interface has on the pilot's screens, standing anywhere in a city
and flying across it, from today's sparse city to dense ones (`citysim.capability`, `vissim capability`). The cities
are the study's three clutter levels as generated (C0 open; C1 and C2 with **packed** blocks, whose 4 m facade gaps
close them to the avoidance law) and two **porous** ones (C1p, C2p: `citygen` core "shell", every two avoidance
cylinders at least 6 m apart, the narrowest gap the formation flows through, §2.2), three seeds each. Standing: 12
random points per city, the formation flown in and settled there by the replica, the single drone on its centre facing
the same way, at the search band's median altitude and pitch (12.4 m, −21°). Flying: six crossings of every city
(origin and destination at least 400 m apart) by each pilot on its own path (§2.2), at the transit band's (19.5 m,
−15°). Recognition is the hat's 40 m; 80 m (a person) is beside it in the tables.

Swarm ÷ single drone, the swarm at spread 1.0 ([headline.csv](../../experiment_results/density_study/capability/capability/headline.csv),
[figure](../../experiment_results/density_study/figures/capability.png)):

| city | built over | building encounters / km | on screen now | in the headset's view | within 1 s | could be seen at all | sweep width | … in the headset's view | swarm's ground first seen round a corner |
|---|---|---|---|---|---|---|---|---|---|
| C0, open | 2.4% | 2.9 | 6.3× | 1.8× | 2.1× | 1.3× | 2.1× | 1.6× | 0.8% |
| C1p, porous | 8.0% | 9.1 | 6.0× | 1.7× | 2.0× | 1.2× | 2.2× | 1.4× | 3.2% |
| C2p, porous | 11% | 13.0 | 6.0× | 1.7× | 2.0× | 1.2× | 2.3× | 1.4× | 4.4% |
| C1, packed | 12% | 13.4 | 5.8× | 1.7× | 1.9× | 1.2× | 2.3× | 1.4× | 4.6% |
| C2, packed | 17% | 19.9 | 5.7× | 1.6× | 1.9× | 1.2× | 2.3× | 1.4× | 7.9% |

- **The swarm's advantage is simultaneity, not reach.** Its hull feeds put about six times the ground on the pilot's
  screens (3,100–5,200 m² against 550–830 m²), 1.6–1.8 times as much inside the headset's ±50°, and twice as much
  within a second, because a head turn and the rig's body yaw bring every feed into view while a single drone has to
  turn its whole view at 75°/s. What could be seen at all, given time to turn (its feeds' union against the single
  drone's 360° isovist), is only 1.2–1.3×: the plan's "the potential viewshed is a wash".
- **Flying, the swarm clears 2.1–2.3 times the width of city** per metre flown (sweep width W: 75–83 m against
  32–40 m), 1.35–1.6 times inside the headset's view, and the ratio grows with clutter, as buildings cut the single
  drone's one forward cone shorter than the spread of the swarm's. A single drone that looks round as it flies (yaw
  sweeps at internal_3's single-drone yaw duty) clears only 7–9% more, at two to three and a half times the stick
  adjustments.
- **Seeing behind buildings grows with clutter.** The share of the swarm's ground that its feeds saw first while a
  camera at the formation's centre could not, because a building blocked it, rises from 0.8% in today's city to 3–4%
  in the porous ones and 4.6–7.9% in the packed ones (more at the tight spread, less at 1.6): 0.8, 2.8–3.7 and
  3.7–5.4 thousand m² per km flown. The rest of what the rim adds is reach: 78–91% of the swarm's first sightings came
  while the cell was out of the centre camera's range or vertical band, and 16–24% of its ground the centre's camera
  never saw on the whole crossing. Of the ground both saw, the feeds saw 76–87% first, a median 1.4–2.4 s sooner: the
  leading drones see the street ahead, and round the building beside it, before the formation gets there.
- **Standing, rim parallax recovers 12–18%** of the ground hidden from the formation's centre inside the feeds'
  wedges (`E_par`), and 8–9.5 of the ten feeds stay useful (not a wall).

### 2.2 How the pilots fly a city: pilot paths, not the street route

Tier B' and Tier C used to fly every transit along the shortest street route, 1.35–1.64× the straight line in the
trial cities, where internal_3's human swarm pilots flew 1.06–1.12× in three of four runs. Since 2026-10-13 every
scripted flight flies the path its pilot would (`citysim.routing`; the street route stays as the comparison):

- **The swarm's pilot steers round only what the formation cannot flow through.** A gate probe (`vissim gates`,
  [gates.csv](../../experiment_results/density_study/transit/gates.csv)) flies the formation at full stick through a
  wall with one gap and straight at a lone building. It flows round a lone building at every spread, and through a
  gap of 6 m between avoidance cylinders at spreads 0.4 and 1.0 (12 m at 1.6) with no drone left behind and no stall
  over 5 s (`w_pass`, `citysim/data/w_pass.json`). Buildings closer than that are walls; the pilot keeps the
  formation's centre a formation radius off them (a per-metre surcharge, so it flies the middle of a corridor), and
  holds the stick through everything else.
- **The single drone's pilot steers round every building** (its avoidance cylinder plus 1 m). The probe says why: a
  lone drone flown straight at a building stalls against it for 9–21 s before its noise slides it off.
- **The swarm morphs round 1.5–4.7 buildings per km at spread 1.0** on these paths (a building inside the live
  drones' hull while the formation is still compact; most in the porous cities, up to 7.8 at spread 1.6). At 0.4 the
  formation is too small to wrap a building: it passes beside one whole, or stretches. Threading the porous cities'
  6 m gaps at 1.0 it spends about half its transit more than 2.5 formation diameters across (14–31% in the packed
  ones). It holds one heading throughout: no stick adjustment at all in the open and porous cities at spreads 0.4 and
  1.0 against 1.0–3.5 per 100 m for the single drone, and 1.2–2.6 against 3.6–4.4 in the packed ones. Its paths are 1.00–1.02× the straight line in
  open and porous cities and 1.05–1.09× in packed ones (it keeps further off walls than the single drone, whose paths
  are 1.02–1.10×); the street route is 1.18–1.31× for both
  ([capability_paths.png](../../experiment_results/density_study/figures/capability_paths.png)).
- **Flying through clutter at the tight spread costs drones.** At spread 0.4 the formation lost 2.0–4.0 drones per
  crossing in the porous and packed cities, against 0.2–0.6 at spread 1.0 and 1.6. Morphing round buildings is cheap
  at 1.0 and expensive at 0.4.
- **What a wall-grazing path cost:** before the pilot kept the formation off walls, its path through L3's packed
  blocks lost 2.5 drones per flight at spread 1.0 on the bench (the replica: 2.3), where the street route lost 0.8
  (0.7) and was no slower. With the surcharge, over all of Tier C's L3 flights, it loses 1.0 per flight at spread
  1.0 against the street route's 0.83, and 7.0 at 0.4 against 5.0: through packed blocks the tight spread is
  costly whatever the path (§1, G3).

## 3. The levels and the rule that picked them

`select_levels` applies the plan's §8 rule to the tiers' tables:

- **L1**, today's regime: C0-G0 (today's coverage, a goal like today's patch). A control that the effect is
  geometry-driven.
- **L2**: the level with the largest swarm − single advantage (Tier B, E_head) among those that pass G1
  against L1 (≥ 2× L1's advantage, > 3 SD), are envelopment-feasible at spread 1.0 (Tier B and the Tier B'
  arrivals, each ≥ 0.9), keep the useful-feed count U ≥ 0.7 of L1's, and are feasible layouts (not jammed,
  ≤ 56 buildings within 50 m of any free point, `OlfatiSaber`'s 64-collider buffer with margin).
- **L3**: at the densest clutter level, a level whose advantage vanished (≤ 1 SD) or reversed, whose
  envelopment fails (≤ 0.1 in Tier B and B') or whose useful feeds fall to ≤ 0.6 of L1's, that passes G2 (Tier
  B' transit at spread 1.0 keeps ≥ 3 m spacing) and G3 (Tier C's loss rate ≤ 2.5 drones per run). Of those,
  the fewest predicted losses, then the smallest advantage.

| | L1 C0-G0 | L2 C0-G10 | L3 C2-G7 |
|---|---|---|---|
| p / s / α | 90.83 m / 15 m / 0.024 | 90.83 m / 15 m / 0.024 | 60 m / 10 m / 0.17 |
| ℓ (ℓ / R) | 355 m (8.9) | 386 m (9.7) | 53 m (1.3) |
| w_c / r_g / n_b / c_g | 7 m / 26 m / 8 / 22–23 m | 28 m / 48 m / 8 / 44–45 m | 10 m / 17 m / 6 / 13–14 m |
| swarm − single E_head (Tier B), ± seed SD | +0.048 ± 0.002 | +0.106 ± 0.003 | −0.050 ± 0.006 |
| swarm / single E_head | 0.119 / 0.072 | 0.140 / 0.035 | 0.050 / 0.100 |
| envelops at spread 1.0 (Tier B / B') | yes / yes | yes / yes | no / no |
| useful feeds at the goal (of 10) | 10 | 10 | 5.3 |
| drones lost per run, spread 1.0 / 0.4 (Tier B') | 0 / 2.0 | 0 / 2.0 | 0.7 / 6.0 |
| drones lost per run, at internal_3's spread use (Tier C) | 0.78 ± 0.20 | 1.11 ± 0.77 | 2.21 ± 0.38 |
| drones alive / on the hull at a goal (Tier C) | 9.5 / 8.4 | 9.1 / 8.6 | 7.6 / 5.8 |
| swarm − single with the drones still alive (Tier B at Tier C's count) | +0.047 | +0.103 | −0.046 |

All buildings are the study design (`BnP_Small_Building_C`, 6.53 × 5.22 m in the pack) at twice its size
and 50 m tall, yawed 0° or 90°. The goal's ring buildings are at 0.9 of that. The full table for every level
of the sweep is [levels_table.csv](../../experiment_results/density_study/levels_table.csv).

## 4. The tiers

| tier | what it computes | assumes | cannot tell | checked against | gate |
|---|---|---|---|---|---|
| **A** analytic (`theory`) | ℓ, P(free sight to R), the regime (range- vs occlusion-limited), walker lap and hidden arc | independent (Boolean) buildings, one camera | formations, display, dynamics | Tier B's free-sight on hard-core scatter layouts | none |
| **B** static Monte Carlo (`vissim sweep`) | E_any, E_head, first sight, episodes, the envelopment/parallax split of occlusion escape, V(t), useful feeds, per (G, C, seed) × the operator band | the formation as the replica settles it on the goal; the C# display rule ported (hull, headings, look-gap fill); ideal cameras | deformation in transit, losses, strategy | A at low α; internal_3 (§5.2) | **G1** |
| **B'** dynamic replica (`vissim dynamic`, `envmap`) | the formation flown spawn → every goal: spacing, losses, contacts, progress, useful feeds in transit, envelopment on arrival; the envelopment map | the 2-D force law as coded, scripted pilot, buildings as contacts with friction | heading timing, altitude, flips | Tier C, flight for flight (`swarm_replica.py validate`) | replica gate, **G2** |
| **C** headless Unity (`vissim tierC`) | the same goal flights in ProceduralCityWorld: losses by Unity's rules, PhysX contacts, the real hull/heading/look-gap code, exposure from real camera poses at real altitude | scripted pilots, no VR | attention and strategy | the human runs | **G3** |

### Tier A: the Boolean model ([tierA.csv](../../experiment_results/density_study/tierA/tierA.csv))

At ×2 buildings ℓ is 380 m at today's α 0.024, 91 m at 0.1 and 46 m at 0.2. A sight line to R = 40 m is free
90%, 65% and 42% of the time. Today's cities are range-limited, and the occlusion-limited regime starts
near α 0.3 (×2). Against hard-core scatter layouts the model agrees as α → 0 (0.008 off at α 0.01, 20 m)
and is optimistic at high α (0.16 off at α 0.2, 40 m). Non-overlapping buildings block more per unit
coverage than overlapping ones.

### Tier B: exposure at the goal

Per goal of every layout: the replica flies the settled formation in from the spawn side and lets it settle
on the goal under a centring pilot (`formation.envelop`, start "arrive"), at spreads 0.4, 1.0 and 1.6; the
display rule gives the shown feeds at 8 body yaws; every walker is scored over 60 s at each altitude ×
pitch of the operator band (5.1, 12.4, 23.8 m × −40°, −15°, −8°, from internal_3's search phase). The
single drone flies three policies (an orbit inside the ring facing out, one outside facing in, a scan
from the approach side) plus the best static vantage. "Best realistic" excludes the static vantage,
which needs knowledge of where the target is.

The second sweep, `tierB2`, is the one the levels come from. The first, `tierB`, failed G1. Its best
feasible level reached 1.79× L1 (a 20 m occluder with the ring at 36 m), still on the rising part of the
curve, and its densest clutter, aimed at α 0.25, jammed at 0.17. `probe_large` (open city only) then showed
the advantage still rising to 2.24× L1 at a 28 m occluder **once the ring moves out with it** (48 m). A
large occluder with the ring kept close does worse: 24 m at 36 m gives 1.6×, 20 m at 32 m 1.3×, because
the annulus the swarm needs to fill disappears. `tierB2` therefore extends the goal range to 20–28 m with
the ring moved out, and targets α 0.17 at C2. (All three sweeps are re-flown with the final contact model.)

**What drives the advantage**: the single drone's exposure falls as the occluder grows (0.072 at L1,
0.035 at L2), because it sees one side at a time, while the swarm's barely moves (0.119 → 0.140). At
cramped goals (G6, G7) the formation cannot get inside the ring, its feeds face walls, and the single
drone's precise placement wins. Clutter barely changes exposure *at* the goal (the goal is cleared to
c_g): within a goal level the three clutter levels' advantages agree to within 0.006. Clutter's cost
is in transit (Tier B'), which is why L1 and L2 share C0.

### Tier B': the formation in flight ([dynamic.csv](../../experiment_results/density_study/tierBprime/tierB2/dynamic.csv), [envmap.csv](../../experiment_results/density_study/tierBprime/envmap.csv))

Each layout is flown from its spawn to every goal at full stick, arriving within 25 m, then held 30 s at spread 1.6
with the centring pilot; transits at spreads 0.4 and 1.0. Since 2026-10-13 the transits fly each pilot's own path
(§2.2), and the street route the study used to fly is flown beside them as the comparison (`dynamic.csv`'s `routing`
column; [transit.csv](../../experiment_results/density_study/tierBprime/tierB2/transit.csv) scores every flight). On
the street route the speed along the route falls with clutter (6.1, 5.3 and 4.6 m/s at C0, C1, C2) and spread 0.4 is
where drones are lost (mean 1.4, 2.5 and 4.7 per flight); at 1.0 every level keeps 4–10 m spacing.

At the trial levels, swarm / single drone on the pilot paths (the street route in brackets, swarm):

| | L1 | L2 | L3 |
|---|---|---|---|
| path ÷ straight line | 1.05 / 1.04 (1.53) | 1.05 / 1.08 (1.48) | 1.09 / 1.09 (1.33) |
| share of the path over the streets | 0.27 / 0.27 (0.85) | 0.28 / 0.32 (0.86) | 0.64 / 0.62 (0.92) |
| stick adjustments per 100 m | 0.3 / 1.3 (2.6) | 0.2 / 2.4 (2.6) | 2.4 / 3.9 (3.6) |
| progress (straight-line distance ÷ time), m/s | 6.1 / 5.9 (4.4) | 6.5 / 6.0 (4.5) | 4.0 / 4.0 (3.6) |
| sweep width, any feed / in the headset's view | 82 / 58 m against 40 m | 80 / 60 m against 40 m | 70 / 43 m against 33 m |
| drones lost per flight, spread 1.0 / 0.4 | 0.7 / 0.7 (0 / 2.0) | 0 / 0.7 (0 / 2.0) | 1.0 / 5.3 (0.7 / 6.0) |
| next goal first seen, s into the leg (80 m) | 42 / 45 (56) | 35 / 35 (46) | 61 / 63 (64) |

- **Flying through the city is nearly a third shorter and needs a tenth of the stick work** at L1 and L2: the swarm holds
  one heading through the blocks where the street route turned at every junction. At L3 the packed blocks leave
  little but the streets, and the gain is a fifth of the path.
- **The single drone's pilot path is as short as the swarm's**, because the scripted pilot knows the city. What the
  swarm saves is the steering: 4–12 times fewer adjustments at L1/L2, 1.6 times at L3. internal_3's single-drone
  pilots flew 1.04–1.87× the straight line (median 1.58, against 1.11 for the swarm), so real single-drone paths are
  longer than the model's (§9).
- **Discovery when the goal is known is a tie.** Flying straight at a goal they know, both interfaces see it at about
  the same moment, because the walkers come into the forward cameras first. The swarm's advantage in finding a goal
  is in searching for one it does not know, through its twice-as-wide sweep: the pass-by model (below).
- **L3's numbers here are the replica's in a packed city**, where it flies too fast and splits differently from Unity
  (§5.5); Tier C's are the ones to use.

The envelopment map (`envmap`, an isolated goal over w_c × r_g ×
spread × drones alive) is monotone, as it should be: a bigger occluder needs a wider spread or a larger
ring. A 7 m occluder is enveloped from spread 0.7 with a 24 m ring; a 20 m one needs spread 1.0 with a
32 m ring, or 1.3 with 28 m.

### Tier C: the bench ([tierC/tierC/](../../experiment_results/density_study/tierC/tierC))

The headless Unity bench flies the same goal flights as B' (`bench.py scenarios --goals`), with centring
holds of 60 s at spread 1.6 so the exposure window matches Tier B's. It flies the trial layouts of the three
picks (seeds 101–103) and the sweep layouts of their six neighbours (seeds 0–2): one step of clutter either
way, and the nearest goal geometry. Every level is flown with ten drones (four flights per layout: both
goal orders × transit spread 0.4 and 1.0), the picks with one drone too. The lone drone's heading is steered
to face its travel, and it scans at 75°/s while holding. Pilot heading is on, so the look-gap fill acts
as in the scenes, and the drones fly at 12.4 m (internal_3's median search altitude; flown ~1.3 m higher).
36 runs, 126 flights.

**Losses per run** are reported at internal_3's spread use. Its swarm pilots spent 33% of transit time
nearest spread 0.4 and 67% nearest 1.0 (`operator_band`'s `spread_share`), so a level's figure is
0.33 × its loss per flight at 0.4 + 0.67 × its loss at 1.0. G3's bar, 2.5 per run, is internal_3's own rate
under that same spread use, which makes this the like-for-like comparison. The plain mean over the flights
as flown (half at 0.4) is beside it.

| level | lost per run (± seed SD) | as flown | at 0.4 / 1.0 | alive / hull drones at a goal | enveloped (20–30 s / end of hold) | swarm E_head / single, at −15° |
|---|---|---|---|---|---|---|
| **L1 C0-G0** | 0.78 ± 0.20 | 1.00 | 1.67 / 0.33 | 9.5 / 8.4 | 1.0 / 1.0 | 0.081 / 0.090 |
| C1-G0 | 1.33 ± 0.33 | 1.83 | 3.33 / 0.33 | 8.7 / 7.7 | 1.0 / 1.0 | 0.075 |
| C0-G2 | 0.11 ± 0.19 | 0.17 | 0.33 / 0.00 | 9.9 / 9.1 | 1.0 / 1.0 | 0.138 |
| **L2 C0-G10** | 1.11 ± 0.77 | 1.33 | 2.00 / 0.67 | 9.1 / 8.6 | 1.0 / 1.0 | 0.156 / 0.034 |
| C1-G10 | 1.16 ± 0.50 | 1.75 | 3.50 / 0.00 | 9.0 / 8.4 | 1.0 / 1.0 | 0.153 |
| C0-G9 | 1.33 ± 0.34 | 1.83 | 3.33 / 0.33 | 8.6 / 8.0 | 1.0 / 1.0 | 0.161 |
| **L3 C2-G7** | 2.21 ± 0.38 | 2.92 | 5.00 / 0.83 | 7.6 / 5.8 | 0.0 / 0.11 | 0.038 / 0.044 |
| C1-G7 | 0.89 ± 0.35 | 1.17 | 2.00 / 0.33 | 9.5 / 6.7 | 0.0 / 0.28 | 0.045 |
| C2-G6 | 2.49 ± 0.44 | 3.33 | 5.83 / 0.83 | 7.4 / 5.4 | 0.0 / 0.53 | 0.060 |

What Tier C adds:
- **G3 holds for L3, narrowly**: 2.21 against 2.5, less than one seed SD below the bar. Its neighbour C2-G6
  sits at 2.49. Nearly every loss is a drone-drone pair at the tight spread: at spread 1.0 the densest level
  loses 0.83 per run. The single drone lost nothing and touched nothing on any level.
- **Losses rank with clutter, as B' predicted, and are of the same size.** At the tight spread Unity loses
  a little less than the replica (L1 1.67 against 2.0, L3 5.0 against 6.0). At spread 1.0 it loses a
  little more (L1 0.33 against none, L3 0.83 against 0.67).
- **Envelopment agrees with B' level for level.** The open levels envelop every goal. The cramped ones
  (G6, G7) envelop none 20–30 s into the hold (B's measure), and 11–53% by its end. **L3 really is the
  collapse**: fewer drones alive, fewer on the hull, and no envelopment.
- **The swarm's exposure keeps Tier B's ranking** (L2 0.156 > L1 0.081 > L3 0.038 at the dial centre).
  L2's advantage is the largest (+0.12) and L3's is gone (−0.006).
- **At L1 the scripted single drone matches the swarm** (0.090 against 0.081), where Tier B gave the swarm
  +0.048. The single drone's hold parks it at the small occluder's edge, scanning, and from there it sees
  most of a 26 m ring. This is one policy at one pitch, not Tier B's grid average, but it says the same
  thing as §5.2: at today's geometry the swarm's predicted margin is small and policy-dependent.
- **With the drones still alive at a goal**, Tier B's advantages barely move (+0.047, +0.103, −0.046
  against +0.048, +0.106, −0.050 with ten; `vissim tierC-alive`).

### Tier C on the pilot paths ([tierC/tierC_pilot/](../../experiment_results/density_study/tierC/tierC_pilot))

The same 36 runs (126 flights) flown again with every transit on the pilot paths (`tierC-plan --routing pilot`).
Five of them flew on a second copy of the bench project after a Unity start crashed and left a process holding the
first copy's lock. Pilot paths, the street route in brackets
([figure](../../experiment_results/density_study/figures/tierC_losses.png)):

| level | runs | lost per run, pilot (street) | at 0.4 / 1.0 per flight, pilot (street) | alive / hull at a goal | enveloped 20–30 s / end | swarm E_head / single, −15° |
|---|---|---|---|---|---|---|
| **L1 C0-G0** | 3 | 0.55 ± 0.38 (0.77) | 1.67 / 0.00 (1.67 / 0.33) | 9.4 / 8.2 | 1.00 / 0.97 | 0.074 / 0.096 |
| C1-G0 | 3 | 2.71 ± 0.54 (1.33) | 5.83 / 1.17 (3.33 / 0.33) | 6.9 / 6.5 | 0.94 / 0.94 | 0.042 |
| C0-G2 | 3 | 0.44 ± 0.19 (0.11) | 1.33 / 0.00 (0.33 / 0.00) | 9.6 / 9.0 | 1.00 / 1.00 | 0.130 |
| **L2 C0-G10** | 3 | 0.61 ± 0.25 (1.11) | 0.83 / 0.50 (2.00 / 0.67) | 9.8 / 8.8 | 1.00 / 0.97 | 0.154 / 0.043 |
| C1-G10 | 3 | 1.88 ± 0.51 (1.16) | 4.67 / 0.50 (3.50 / 0.00) | 8.1 / 8.0 | 0.94 / 0.94 | 0.163 |
| C0-G9 | 3 | 0.88 ± 0.51 (1.33) | 2.00 / 0.33 (3.33 / 0.33) | 9.2 / 8.6 | 1.00 / 1.00 | 0.161 |
| **L3 C2-G7** | 3 | 2.99 ± 0.88 (2.21) | 7.00 / 1.00 (5.00 / 0.83) | 6.5 / 5.1 | 0.56 / 0.67 | 0.000 / 0.042 |
| C1-G7 | 3 | 2.71 ± 0.54 (0.88) | 6.17 / 1.00 (2.00 / 0.33) | 7.2 / 5.7 | 0.56 / 0.75 | 0.001 |
| C2-G6 | 3 | 2.93 ± 0.75 (2.49) | 6.50 / 1.17 (5.83 / 0.83) | 6.8 / 6.0 | 0.53 / 0.58 | 0.004 |

- **Through packed blocks the tight spread costs drones.** At spread 0.4 the packed levels lose 5.8–7.0 drones per
  flight on the pilot paths against 2.0–5.8 on the street route; at 1.0 the pilot paths lose 0.2–0.8 more.
  The losses are drone–drone collisions (1.6–1.8 colliding pairs per flight at the packed levels, against 0.4–1.5 on
  the street route), not buildings (0–0.3 contacts per flight): the formation squeezed between walls at its tightest
  spread. Every packed level is now over G3's bar; the open levels lose fewer drones than before.
- **The formation reaches cramped goals it used to stop short of.** On the street route the swarm halted 10–14 m from
  the centre of L3's goal, outside its 17 m ring. Through the blocks it arrives on the centre, and envelops 53–75% of
  the holds at the cramped goals (G6, G7) where it enveloped 0–53%. From inside a tight ring at 13.7 m and −15° its
  cameras look over the walkers: exposure at the dial centre falls to 0.000–0.004 (0.038–0.060 on the street route;
  at −40° it keeps 0.034). **The collapse at L3 is stronger on the pilot paths, not weaker.**
- **The open levels are unchanged at the goal** (L1 E_head 0.074 against 0.081, L2 0.154 against 0.156) and lose
  fewer drones (0.55 and 0.61 per run, against 0.77 and 1.11).
- **The transit, in Unity**, is protocol §11's table: swarm / single paths 1.05–1.12 / 1.09–1.17 of the straight line,
  a sixth to a seventh of the single drone's stick adjustments at L1/L2 and half at L3, about twice its sweep width, and the
  next goal seen 3–13 s sooner when its position is known.

### Discovery: finding a goal nobody announced ([transit/](../../experiment_results/density_study/transit))

The goals are not announced, so in transit the pilot is searching, and the co-primary is how long the next goal
takes to appear on a shown feed. Flying straight at a known goal, the two interfaces see it at much the same moment
(the directed bound: Tier C 46 / 49 s at L1, 39 / 44 s at L2, 71 / 84 s at L3, swarm / single), because the walkers
enter the forward cameras first. The swarm's advantage is in the search, through the width of city it clears.

**The pass-by probe** (`vissim passby`) flies each pilot on its own path straight past every trial goal (27) and
every goal of internal_3's runs (24): six headings, lateral offsets 0–300 m, 600 m passes, the swarm at spread 1.0.
A pass detects the goal when one of its walkers is expected on a shown feed (`citysim.discovery`). The share of
passes that detect it at offset x is the goal's lateral range curve, and twice its integral the sweep width W_goal
([passby_W.csv](../../experiment_results/density_study/transit/passby_W.csv),
[figure](../../experiment_results/density_study/figures/passby.png)):

| W_goal, swarm / single | L1 | L2 | L3 | internal_3's city |
|---|---|---|---|---|
| 80 m, shown feeds | 218 / 145 m | 277 / 200 m | 208 / 127 m | 237 / 164 m |
| 80 m, head-visible | 196 / 145 m | 228 / 200 m | 165 / 127 m | 198 / 164 m |
| 200 m, shown feeds | 517 / 291 m | 520 / 372 m | 444 / 250 m | 520 / 315 m |

(At 200 m the swarm's curve is still above zero at the probe's last offsets, so its W is a lower bound.)

**The search model** (`vissim discovery-predict`): random search finds the next of k goals in an area A at the
rate k W v / A, so a trial's three legs average 0.611 A / (W v). internal_3's pilots found goals 5.0 times
(swarm) and 4.4 times (single drone) faster than random search would in their city, at 80 m (η 0.20 and 0.23):
nearly the same search skill with both interfaces, which is what lets the swarm / single ratio stand without it.
The predictions use the mean η. Scaled by η,
with Tier C's speeds ([discovery_predictions.csv](../../experiment_results/density_study/transit/discovery_predictions.csv)):

| next goal on a shown feed (80 m), swarm / single | L1 | L2 | L3 |
|---|---|---|---|
| predicted discovery time | 51 / 81 s | 37 / 54 s | 80 / 138 s |
| swarm ÷ single, shown / head-visible feeds | 0.64 / 0.71 | 0.69 / 0.84 | 0.58 / 0.73 |

The ratio is the robust output: unlike exposure at the goal it holds at L3, because it comes from the sweep in
transit, which clutter shortens for both interfaces. The absolute times rest on η from two pilots. With the
replica's speeds instead of Unity's the ratios are within 0.03.

## 5. Validation

### 5.1 The replica against Unity ([replica_validation/](../../experiment_results/density_study/replica_validation))

`swarm_replica.py validate` flies 17 recorded bench flights per city again (6 transits, a tour, 4 rams, 3
zigzags, 3 spread steps): DiamondCityWorld, an L1 layout and a C2 (α 0.17) layout. Limits: median track
error ≤ 10% of the path, median spacing error ≤ 15%, losses within 2 per 10 flights, step t90 within
1.5 s.

| building model | DiamondCityWorld | L1 | C2 (α 0.17) |
|---|---|---|---|
| kill on entering a footprint (`Sim`) | pass | pass | **fail**: 18 lost vs Unity's 5, spacing 20% off |
| contact, no friction | pass | pass | **fail**: losses 5 vs 5, spacing 15.1% off |
| contact, friction 0.6 (`ContactSim`, adopted) | pass | pass | pass: losses 3 vs 5, contacts 16 vs 23, spacing 10%, track 3% |

In the dense city Unity's drones bump facades 23 times in 17 flights, and friction pins some until the swarm
leaves them (3 of its 5 losses were `TooFarFromSwarm`). Kill-on-entry turns every bump into a death, and
frictionless contact lets a pinned drone slide free. PhysX's default friction is the only constant
`ContactSim` adds, and it is the value every collider in the project has. Tier B and B' were re-flown
with it; their tables above are the re-flown ones. It moved L3, as §1 says, and nothing else.

### 5.2 internal_3 (`vissim validate`, [validate/internal_3/](../../experiment_results/density_study/validate/internal_3))

**Replay.** Re-scoring internal_3's recorded poses through citysim's line-of-sight and camera model
reproduces `analyse.py`'s per-goal in-view seconds on all 24 goals (largest difference 0.04 s), so the
study and the analysis share one definition of exposure.

**Prediction.** Tier B's machinery is run on internal_3's own city and goals (DiamondCityWorld, the
patches at their runtime width), arriving from where each run arrived. Its output is compared with each
goal's measured in-view share of the search phase. The predicted range is the spread, over the operator
grid (altitude × pitch × spread, or the single drone's three policies), of the per-goal mean:

| | predicted median [range] | measured median | goals inside the range |
|---|---|---|---|
| swarm, shown feeds | 0.42 [0, 0.71] | 0.44 | 12 of 12 |
| swarm, head-visible feeds | 0.12 [0, 0.20] | 0.22 | 5 of 12 |
| swarm, first sight of the hat | 14 s [0.7, 60] | 5.5 s | 10 of 12 |
| single drone | 0.05 [0, 0.10] | 0.13 | 4 of 12 |
| single drone, first sight | 37 s [11, 60] | 25 s | 7 of 12 |

The model gets the geometry right: the swarm's displayed exposure, medians 0.42 against 0.44 (the range
is wide, so the medians are the stronger test). It under-predicts two things that are about the pilot,
not the city:
- **Head-visible exposure** (0.22 measured against 0.12). Pilots look at the feed the target is on more
  often than their overall head-yaw distribution says.
- **The single drone** (0.13 against 0.05). Its pilots found better vantages than the scripted orbit and
  scan.

The second matters most for the trials. The levels rest on how the single drone's exposure *changes* with
the occluder (0.072 at L1 to 0.035 at L2 in Tier B), and the scripted policies capture that direction, but
in absolute terms the predicted swarm advantage is an upper bound. Its real size is what the trials
measure.

### 5.3 The empirical anchor (`vissim anchor`)

From internal_3's recorded poses: of the 179 s in which some swarm camera had the hat in range and in
front but blocked, another camera saw it 14% of the time (11% on shown feeds, 6.6% on head-visible
ones). For the single drone the figure is 0. That is occlusion escape at today's geometry: real, and small.

### 5.4 The Unity side

- `CityObstacleExport.RunProceduralCityWorld` + `vissim roundtrip`: every collider of a built layout matches
  its layout box (|Δc| < 1e-3 m, axes equal up to sign). The builder puts buildings on `Obstacle` and nothing
  else with a collider (streets have none), so the obstacle audit has nothing else to find.
- `ProceduralCityProbe` (headless Play mode on the bench copy, the recorder redirected to a scratch folder),
  all checks passed on L2 seed 101: three goals with three different hats, 64 walkers per goal, none inside
  a building, the swarm held at 51.3 m after a reset to 70 m, the session record's layout / goal seed / goal
  geometry / auto-score fields, `hull;shown` and the gaze columns, a press away from every goal scored
  `out_of_zone` and one beside a goal scored with its camera distance and feed.
- Bench smoke runs at 10 and 1 drones in a layout, and a goal flight replayed by the replica to within
  4.9 m RMS over 1.4 km.

### 5.5 The replica on the pilot paths ([pilot_paths2/](../../experiment_results/density_study/replica_validation/pilot_paths2))

Pilot paths are a new kind of path, flown in two new kinds of city, so the replica was gated again before any B' or
capability number was used. The bench flew the goal flights on pilot paths with every flight recorded: the three L1
and three L3 trial layouts, and three layouts each of C1p and C2p. The swarm flew at spreads 0.4 and 1.0 in both goal
orders (12 flights per kind of city), and the single drone in one layout per kind (2 flights). Two checks:

**`swarm_replica.py validate`**, its usual limits, pooled per kind of city (four flights per run are too few for a
loss limit of 2 per 10):

| city | swarm verdict | drones lost, Unity / replica | spacing off by | track off by |
|---|---|---|---|---|
| L1, open | **pass** | 10 / 10 | 3.5% | 0.3% |
| C1p, porous | **pass** | 30 / 32 | 14.1% | 0.8% |
| C2p, porous | **fail** (spacing, losses) | 32 / 37 | 20.7% | 1.0% |
| L3, packed | **fail** (spacing, losses) | 50 / 44 | 38.9% | 1.8% |

The single drone passes everywhere (no losses on either side; track within 2%). By spread, L1 passes at both, C1p at
1.0, C2p fails on spacing at 0.4 and on losses at 1.0 (6 against Unity's 2), and L3 fails on spacing at both (33–41%)
while its losses at 1.0 agree (8 against 9). The tracks agree everywhere. What fails is the formation's inside in the
dense cities: threading walls it stretches and splits, and the replica's groups sit at a different spacing from
Unity's.

**`vissim transit-gate`** scores the same Unity flights, and two replica re-flights of each, with the study's transit
measures at one altitude (13.7 m, the flights' median) and the transit pitch, so only the flying differs. A cell agrees
when within 15% or two standard errors of Unity
([transit_gate.csv](../../experiment_results/density_study/replica_validation/pilot_paths2/transit_gate.csv)):

- **Agree in every city:** path ratio, street share (except the single drone in C2p: 0.44 against 0.37), sweep
  width (any feed and head-visible), area seen per metre, the share seen round a corner, the share of time stretched
  or split, and losses within two standard errors (the validate gate's absolute limit, above, is the stricter test).
  So do the swarm − single contrasts on all of these ([transit_gate_contrast.csv](../../experiment_results/density_study/replica_validation/pilot_paths2/transit_gate_contrast.csv)).
- **Stick adjustments are under-counted** by a third to a half: 0.15–0.20 per 100 m for the swarm against Unity's
  0.29–0.31 in L1 and the porous cities, and 1.3 against 2.1 for the single drone in L1. Both interfaces are under-counted, and the
  contrast holds: the swarm needs a fraction of the single drone's adjustments in every city, in both.
- **In packed L3 the replica flies too fast:** 4.3–4.6 m/s against Unity's 2.8–3.7 for the swarm, 4.7 against 3.6 for
  the single drone. It reaches goals sooner (swarm discovery at spread 0.4: 47 s against 93 s), counts fewer morph
  events (2.4 against 5.5 per km at 1.0), and the swarm − single discovery contrast changes sign there.

What follows: §2.1's capability measures rest on quantities the gate confirms, in the cities it passed and in the
porous ones, apart from the adjustment counts, which are low for both interfaces (their ratio holds). In packed cities
the replica's speeds and discovery times are optimistic, so **L3's transit speed, losses and discovery come from Tier
C**, and the replica's packed-city numbers (C1, C2 in §2.1, L3 in Tier B') are a relative comparison only.

### 5.6 Goal discovery in internal_3 ([legs.csv](../../experiment_results/internal_3/results/legs.csv), [discovery_range.csv](../../experiment_results/internal_3/results/discovery_range.csv))

`analyse.py`'s discovery family scores the recorded runs by the co-primary's definition (`citysim.discovery`): leg
start to the first instant the next goal's walkers are on a shown feed. Median per leg (12 legs per interface):

| | single drone | swarm |
|---|---|---|
| discovery time at 80 m (40 / 120 / 200 m) | 31.0 s (36.8 / 24.5 / 18.8) | 55.6 s (62.0 / 43.2 / 23.3) |
| straight-line leg | 192 m | 290 m |
| discovery time per 100 m of leg, at 80 m / 200 m | 20.1 / 10.3 s | 14.0 / 6.7 s |
| distance from the goal at first sight (80 m) | 109 m | 113 m |

In raw seconds the swarm was slower, but its pilots chose legs half as long again; per 100 m of leg it found the next
goal faster at every range. That is why the co-primary's model carries the leg's length (protocol §9).

**The range.** A pilot with no map cannot turn onto a goal before seeing it. On the 11 legs where a pilot visibly
turned onto the goal for the final approach (`commitSec` > 1 s), first sight came no later than that turn in 9% of
them at 80 m (the turn came a median 19 s before sight) and in 55% at 200 m (median 0.4 s): pilots seem to notice a
goal's crowd from about 200 m. Two pilots who had flown the same city before may also have known where goals could
be, so 80 m stays the definition and 200 m is reported beside it; the pilot block, in fresh cities, can settle it.

## 6. Running it

```
cd Assets/Scripts/Experiment
python -B -m citysim.catalogue                                # the designs and the study design
python -B -m citysim.coverage --city DiamondCityWorld         # α, ℓ(θ) of a city (--layout <file> for a layout)
python -B -m citysim.operator_band calibrate internal_3       # the operator band -> internal_3/results
python -B -m citysim.vissim tierA                             # Tier A, and the Boolean model vs scatter
python -B -m citysim.vissim sweep --spec citysim/specs/tierB2.json    # Tier B (~6 min here)
python -B -m citysim.vissim dynamic --sweep tierB2 --levels all       # Tier B' flights (~4 min)
python -B -m citysim.vissim envmap                            # Tier B' envelopment map (~5 min)
python -B -m citysim.vissim gate1 --sweep tierB2              # G1 and the candidates
python -B -m citysim.select_levels --sweep tierB2             # the levels -> levels.json
python -B -m citysim.citygen trial --seeds 101 102 103 104    # their trial layouts -> layouts/
python -B -m citysim.citygen mixcity --city DiamondCityWorld  # not the study: DiamondCityWorld's buildings on Voronoi streets (VoronoiCityWorld)
python -B -m citysim.vissim tierC-plan                        # Tier C configs + runner on the bench root
powershell -File D:\claude_obstacle_bench\cfg\tierC\tierC\run_tierC.ps1   # fly them (~2.5 h, resumable)
python -B -m citysim.vissim tierC                             # score them -> tierC/tierC/
python -B -m citysim.vissim sweep --spec citysim/specs/tierB2.json --levels C0-G0 C0-G10 C2-G7 \
       --drones 7 8 9 10 --name tierB2_alive                  # Tier B for the picks with fewer drones
python -B -m citysim.vissim tierC-alive                       # ... read at Tier C's alive-at-goal count
python -B -m citysim.select_levels --sweep tierB2 --tierC tierC       # the levels with G3 applied
python -B -m citysim.vissim validate                          # replay + predict internal_3
python -B -m citysim.vissim anchor                            # internal_3's empirical occlusion escape
python -B -m citysim.vissim gates                             # the pilots' gate probe -> citysim/data/w_pass.json (~15 s)
python -B -m citysim.vissim capability                        # §2.1: what each interface sees, standing and flying (~10 min)
python -B -m citysim.vissim dynamic --sweep tierB2            # Tier B', both routings, every pilot, scored (~25 min)
python -B -m citysim.vissim tierC-plan                        # Tier C on the pilot paths (name tierC_pilot) + runner
powershell -File D:\claude_obstacle_bench\cfg\tierC\tierC_pilot\run_tierC.ps1   # fly them (~2 h, resumable)
python -B -m citysim.vissim tierC                             # score them -> tierC/tierC_pilot/
python -B -m citysim.vissim transit-gate                      # the replica against Unity on the transit measures (~40 min)
python -B -m citysim.vissim passby                            # goal lateral range curves and W_goal (~15 min)
python -B -m citysim.vissim discovery-predict --speeds-from tierC_pilot   # the discovery predictions (Unity's speeds)
python -B -m citysim.vissim tierC-alive --name tierC_pilot    # Tier B at the pilot paths' alive-at-goal count
python -B -m citysim.select_levels --sweep tierB2 --tierC tierC_pilot --out levels_recheck   # the levels re-checked
                                                              # on the pilot paths (levels.json is left alone)
python -B -m citysim.report --sweep tierB2 --tierC tierC_pilot          # figures
python -B -m citysim.explorer                                 # the explorer page (~35 s)
python -B citysim/tests/run_tests.py                          # tests (or pytest where installed)
```

The discovery co-primary of recorded runs is `analyse.py run <test>` (its discovery family, `legs.csv`).

The replica's gate is `python -B swarm_replica.py validate <bench runs> --out <dir>` from
`Assets/Scripts/swarm/Bench`, on runs flown with `bench.py config --record-all`. In Unity:
`Tools/Swarm/Export building catalogue`, `Tools/Swarm/Build procedural city world`,
`Tools/Swarm/Build procedural city from layout…`.

## 7. Files

| | |
|---|---|
| `Assets/Scripts/Experiment/citysim/` | the package (its `__init__` lists every module), `specs/` (sweep specs), `tests/` |
| `Assets/Scripts/Experiment/layouts/` | trial layouts, committed; `sweep~/` holds sweep layouts (git-ignored, invisible to Unity) |
| `Assets/Scripts/Experiment/building_catalogue.json` | the ScaledCity building designs and their sizes |
| `Assets/Scripts/Environment/ProceduralCityBuilder.cs` and friends | the Unity side (see AGENTS.md) |
| `experiment_results/density_study/tierA/` | the analytic table and the Boolean-vs-scatter check |
| `…/tierB/<sweep>/` | `exposure.csv` (per goal × interface × policy × altitude × pitch), `geometry.csv` (E, V(t), U), `envelop.csv`, `layouts.csv`, the spec and a manifest |
| `…/tierBprime/` | `envmap.csv`, and per sweep `dynamic.csv` / `dynamic_goals.csv` |
| `…/tierC/<name>/` | `plan.json`, `tierC_flights.csv`, `tierC_goals.csv`, `tierC_levels.csv` |
| `…/gate1/` | G1 per sweep: the deltas, the candidates, the verdict |
| `…/levels.json`, `levels_table.csv` | the picks, every level's predictions and the gates' margins (selected on the street route) |
| `…/levels_recheck.json`, `levels_recheck_table.csv` | the same rule on the pilot paths: no L3 (§1) |
| `…/tierC/tierC_pilot/` | Tier C re-flown on the pilot paths: as `tierC/`, plus each flight's transit and discovery columns; `cache/` (git-ignored) holds per-flight scores so a re-score only scores re-flown runs |
| `…/replica_validation/` | the replica gate (contact with friction; `kill/` and `contact_nofriction/` hold the two failing models' runs) |
| `…/validate/internal_3/` | the replay and the prediction of internal_3 |
| `…/anchor/` | internal_3's empirical occlusion escape |
| `…/figures/` | `report.py`'s figures |
| `…/explorer/density_study.html` | the explorer page (`citysim/explorer.py` filling `citysim/explorer.html`) |
| `…/capability/capability/` | §2.1: `static.csv` (per point × spread × body yaw), `moving.csv` (per crossing × pilot, both camera bands), `pairs.csv` (swarm against single on the same crossing), `layouts.csv` (porosity), `headline.csv` |
| `…/transit/` | `gates.csv` (the gate probe), `passby.csv` / `passby_W.csv` (goal lateral range curves, W_goal), `discovery_predictions.csv` |
| `…/tierBprime/tierB2/transit.csv` | every Tier B' flight scored by `citysim.transit` (path, commands, coverage, look-behind, discovery per goal) |
| `…/replica_validation/pilot_paths2/` | the replica's gate on the final pilot paths (`gate_pooled.csv`, `gate_by_spread.csv`) and its transit gate (`transit_gate.csv` per city × pilot, `transit_gate_contrast.csv` swarm against single drone, `transit_gate_flights.csv`); `pilot_paths/` is the first pilot's, which grazed walls |

## 8. Decisions made while implementing

- **The envelopment test requires the formation inside the ring**, not merely round the goal: centre inside
  the hull, no bearing gap over 150°, and ≥ 70% of the drones within c_g. A formation surrounding the whole
  patch from outside the ring looks at the ring buildings' outer faces, which is not envelopment.
- **Tier B's templates arrive** rather than being placed on the goal. A formation placed on the centre
  overstated envelopment at small spreads, because the replica never had to fly it in round the occluder.
- **Buildings in the replica are contacts with friction** (§5.1), by the gate.
- **Tier C's levels are the picks and their nearest neighbours** in clutter and in goal geometry. That
  is the plan's "the candidate levels and their neighbours", made explicit, and it covers the close L3 call.
- **Tier C flies at 12.4 m**, internal_3's median swarm search altitude. Exposure is scored at each pitch
  of the operator band (pitch does not change the flight), at the recorded altitude.
- **The lone drone's heading is steered on the bench.** The bench's yaw stick used to be zero, which a swarm
  ignores but a lone drone obeys, so its feed never turned.
- **G3 reads losses per run at internal_3's spread use, not the plain mean over the flights.** The flights
  bracket the spread stick (half at 0.4, half at 1.0) so that per-spread rates can be combined, and G3's bar
  (2.5 per run) is internal_3's rate under its pilots' spread use. Weighting the two by that use is the
  comparison the bar was set for. This choice was made after the runs, and it decides G3 for L3: 2.21
  passes, while the plain mean, 2.92, would not. Both are reported, and the protocol treats the bar as
  provisional until the pilot block measures it.
- **Tier B was re-read with the drones Tier C keeps alive** (plan §7's "B's exposure re-run with C's
  expected alive hull count"): the picks re-flown with 7–10 drones (`vissim sweep --levels … --drones …`)
  and interpolated at Tier C's alive-at-goal count. Reported, not a gate.
- **Sweeps cannot hang any more.** Twice a pool worker died silently and its job was lost, leaving the
  sweep waiting forever with every worker idle. `vissim` now ends a sweep after 15 min without a result,
  saying how many jobs were outstanding; the commands added since (`gates`, `capability`, `dynamic`,
  `passby`) rerun the lost jobs in a fresh pool instead (`citysim.parallel`).
- **Transits fly pilot paths, not the street route** (2026-10-13, §2.2). The street route made every scripted flight
  a third to two thirds longer than the straight line and kept the swarm out of the blocks, which is the opposite of
  how internal_3's pilots flew it, and it hid the swarm's transit advantage: holding one heading while the formation
  morphs round buildings, and its leading drones seeing round them. The pilot model has three parts, each pinned by a
  measurement: `w_pass` by the gate probe, the swarm keeping a formation radius off walls by the bench (a path that
  grazed L3's walls lost three times the drones), and the single drone steering round every building by the probe
  (a lone drone flown at a building stalls). The street route is flown beside it as the comparison.
- **Porous cities exist only in the capability study.** Without them a through-the-city path in a dense city *is*
  a street path, so morphing would only ever have been measured in today's sparse city. The trial levels are
  unchanged: L3 stays packed.
- **G3 on the pilot paths is open, and it is the experimenter's call** (2026-10-13, §1). Nothing was changed:
  `levels.json` holds the street-route picks, `levels_recheck.json` the rule re-run on the pilot paths (no L3). The
  options:
  1. *Keep L3 = C2-G7 and let the pilot block decide*, as protocol §8 already provides ("if the pilot block's L3
     loss rate exceeds 2.5, re-run the selection with the measured rate"). The scripted pilot spends a third of its
     transit at spread 0.4 at full stick through packed blocks, internal_3's share of tight spread from a sparser
     city; whether people fly a packed city that way is what the pilot block measures. Everything else about L3
     holds, and its collapse at the goal is stronger on the pilot paths.
  2. *Apply the rule as written to the pilot paths*: there is no L3, and relaxing "the densest clutter level" does
     not produce one among the flown levels: every packed level is over the bar (C1-G0 2.71, C1-G7 2.71, C2-G6
     2.93, C2-G7 2.99), and the one C1 level under it (C1-G10, 1.88) keeps the swarm's advantage. A different goal
     geometry or a sparser L3 would have to be flown first.
  3. *Calibrate the bar like for like*: 2.5 is human pilots in DiamondCityWorld; the scripted pilot's rate on pilot
     paths in that city has not been measured (goal flights need a generated layout, which a tile city is not). If
     the scripted pilot loses more than people there, 2.99 may be within today's experience.
- **Discovery is a co-primary outcome** (the protocol, §7). It needs no new logging: analyse.py and the simulator
  share `citysim.discovery`. Its range is 80 m a priori; internal_3 suggests pilots notice a goal's crowd from about
  200 m (§5.6), so 200 m is reported beside it, and the choice is the pre-registration's.

## 9. Limits

- **The panorama is not modelled**, only the feed ring. Its on-time is a covariate in the trials.
- **The operator band comes from internal_3's two pilots**: four runs per interface. It is a prior,
  not an estimate.
- **Walkers follow the patrol rule exactly**, and recognition is a fixed 40 m slant range. The hat's
  visibility on the display and the pilot's attention are not modelled; that gap is what the trials
  measure (utilisation).
- **Tier B' has no heading, altitude or flips**, and its friction is one constant. Tier C has all three,
  with scripted pilots that do not search the way people do.
- **Building size is ×2 throughout.** The plan's ×4 size sensitivity was not run.
- **The pilots are scripted and informed.** The pilot paths know the city: they are what a pilot who can see where
  the walls are would fly, not a search. How humans search a city for an unannounced goal is modelled only through
  the sweep width (random search, calibrated on internal_3), which is crude in absolute terms; the swarm / single
  ratio is the robust part. The single drone's look-round variant flies only in the replica.
- **So the model's paths cannot show the path-ratio gap internal_3 showed.** Scripted, the two interfaces fly paths of
  the same length (1.04–1.09× at the trial levels); internal_3's pilots flew 1.11× with the swarm and 1.58× with the
  single drone (medians). Their paths include the search, since goals are not announced, so the likelier source of
  that gap is wandering until the next goal is seen, which the swarm's wider sweep shortens, rather than steering.
  The directed flights put a floor under both interfaces' path ratios; the gap is the trials' to measure.
- **The replica under-counts stick adjustments** by a third to a half against Unity flying the same paths, for both
  interfaces (§5.5); their ratio holds, their absolute values are low. Human pilots also make many adjustments the
  scripted stick never does (internal_3: 3.8–9.8 per 100 m over both interfaces, against the model's 0–4.4).
- **Look-behind is measured against an ideal camera at the formation's centre**, which sees all round; against the
  single drone it is the sweep width and who saw a cell first (`pairs.csv`).
