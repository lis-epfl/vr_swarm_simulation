# Relating the swarm's views: metric options

How the simulation studies could measure whether a pilot can relate the swarm's camera views to one another, and
so charge the swarm for spreading out among buildings. Written 2026-10-23 for review: nothing here is implemented.
Already decided: shared content counts open ground **and facades**; no screen is privileged (no anchor feed); the
stitched panorama is out of scope (it re-presents what the screens already show); "a larger swarm" means a wider
**spread** (d_ref 0.4 / 1.0 / 1.6, 10 drones).

## 1. Summary

- The studies score the swarm on how much it **sees**. A pilot learning the environment from the views also has to
  **relate** them to each other, **place** what they see relative to themselves, and **keep** it as the swarm moves.
  Spread and building clutter hurt those steps, and no headline measure charges for them.
- Overlap between neighbouring views is redundancy, and redundancy costs coverage, so a pure coverage measure
  always favours spreading out. That is what S3 found ("1.6 beats 1.0"). The metric decides what "the best spread"
  means.
- Neighbouring drones face apart, so their views only start to overlap some distance out: about 6–12 m at spread
  0.4, 15–28 m at 1.0 and 26–48 m at 1.6. A facade nearer than that ends both views before they meet. That is the
  mechanism behind "zero overlap near buildings", and because it is geometry it can be measured directly.
- S3 already measures whether neighbouring views share **open ground**. It does not count shared **facades**, which
  are most of what a view shows near buildings.
- Section 5 gives twelve options in four families, section 7 five ways to check them with pilots, and section 8 a
  shortlist and the decisions needed.

## 2. The pilot's problem

**What the pilot has.** In `OUTER_CIRCLE`, each shown feed is a screen on a 2 m ring round the pilot, placed at its
drone's heading relative to the body yaw. A 74.4° feed fills 47.9° of the ring (compressed by 0.64), and the
headset shows ±50°; internal_3's swarm pilots looked within 25° of straight ahead 87% of the transit time. Screens
are placed **by heading only** and drone positions are never shown. Under the hull rule each drone faces outward
from where it sits, so its heading is the pilot's only cue to where it is.

**Four steps from views to knowledge.** Coverage measures only the first.

1. **See**: the content appears on some screen.
2. **Relate**: the pilot can tell how one screen joins its neighbours (the same building, the street continuing).
3. **Place**: the pilot can put what they see in the right place relative to themselves, to fly to it or avoid it.
4. **Keep**: the pilot can follow it as the swarm moves and content passes from screen to screen.

Human-factors work calls the ability to integrate across separate views *visual momentum* (Woods, 1984). Views that
share no content, no landmarks and no continuity give none, and each screen becomes a snapshot to be placed alone.

**Where neighbouring views meet.** Neighbouring screens point apart by the gap between their headings, about
51–60° with six or seven screens, while each camera sees 74.4°. Their fields of view therefore only start to
overlap where their inner edges cross, at roughly (baseline ÷ 2) ÷ tan((74.4° − gap) ÷ 2). The baseline grows with
the spread, and so does that distance:

| screens shown (heading gap) | spread 0.4 | 1.0 | 1.6 |
|---|---|---|---|
| 7 (51°) | 6 m | 15 m | 26 m |
| 6 (60°) | 12 m | 28 m | 48 m |

Distances from the cameras, for an evenly spread ring with formations about 6 / 14 / 24 m across. Nearer in, a
wedge between the two views is seen by neither camera.

- **Spread alone pushes the overlap out of range.** At 1.6 with six screens the views do not meet within the 40 m
  recognition range even in an empty city. That is S3's "no buildings" column below: only 0.46 of pairs linked at
  1.6 in the most open cities.
- **Clutter stops the views before they meet.** The nearest facade is 17 m from a camera at coverage 0.02 and 7 m
  at 0.14 (S3 medians). A building nearer than the overlap distance appears on both screens as two separate
  pieces, seen at angles about one heading gap apart, with the middle missing. This is the "parallax and zero
  overlap" a pilot meets near buildings.
- **What two screens do share looks alike.** For content inside both fields of view, the angle between the two
  cameras' lines of sight cannot exceed 74.4° minus the heading gap: about 15–23° with six or seven screens, at any
  spread. Spread moves the shared region outward rather than making shared content look different. Large viewpoint
  differences arise on buildings that both screens show *without* sharing content.

## 3. How coverage is measured today

Code: `transit.Cells` and `transit.seen_from`.

- **Ground** is a 2 m grid of **open** cells; cells inside building footprints are removed, so a denser city has
  less ground to see.
- A cell is **seen** when a shown feed would see a point at hat height (1.8 m) on it: within 40 m slant range,
  inside the 74.4° horizontal field of view, inside the vertical field of view at the flight altitude and gimbal
  pitch, and on a straight line from the camera that crosses no building footprint.
- **Buildings only block.** The 2-D model treats every building as taller than the drones (50 m slabs, about 19.5 m
  of flight). Facades are never counted as content.
- **Each camera's counted ground is a far strip.** At S3's flight band (19.5 m, −15°) the bottom of the frame cuts
  off ground nearer than about 22.6 m, so each camera counts only ground 22.6–35.9 m out (about 500 m² before
  occlusion).
- **Discovery at 200 m** uses a 4 m grid and rays every 0.5° to the first facade.

So the content neighbouring screens could share near buildings, a facade 15–22 m out at spread 1.0, is exactly what
the ground measure cannot count.

## 4. What the studies measure about correspondence now

**S3** pairs each shown screen with its ring neighbours (`clutter.RingTally`). A pair is linked when it shares
≥ 20 m² of open ground within 40 m through clear lines of sight; "no buildings" repeats the test with buildings
ignored. Averaged over block sizes:

| coverage | pairs linked, 0.4 / 1.0 / 1.6 | 1.6, no buildings | whole ring linked, 1.0 / 1.6 |
|---|---|---|---|
| 0.02 | 0.77 / 0.66 / 0.43 | 0.46 | 13.7% / 0.6% |
| 0.06 | 0.41 / 0.38 / 0.24 | 0.32 | 1.0% / 0 |
| 0.10 | 0.36 / 0.30 / 0.16 | 0.27 | 0.2% / 0 |
| 0.14 | 0.39 / 0.28 / 0.13 | 0.31 | 0.1% / 0 |

These are reported as levels only. Sweep width, ground per second and time to find count ground from any screen.
The one cost S3 reports (partial r −0.42) is geometric: wide formations lose coverage in small dense blocks. It is
not a cost to the pilot. **S1** counts a feed as "useful" only if it adds ≥ 10% unique ground, so it rewards
non-overlap. **S2** has no such measure. `lessons.md` #5 lists the gap.

## 5. Options

Every option treats all screens alike. "Neighbours" means screens next to each other on the ring.

### A. Do neighbouring screens connect?

**A1. Shared content (ground + facades).**
- *What.* For each pair of neighbours: the open ground both see (as now), plus the facade both see, meaning the
  same face of the same building matched in 2 m stretches. A pair is linked above a minimum, for example 20 m² of
  ground or 4 m of facade. Reported as the share of pairs linked, the time every pair is linked, and the amount
  shared.
- *For the pilot.* Whether two screens show some of the same thing, so they can be lined up.
- *Pros.*
  - The direct reading of "zero overlap".
  - Extends a measure that already exists.
  - Facades fill its main gap: at 1.0 the overlap begins 15–28 m out, and a facade 15–22 m away lies inside it but
    nearer than the counted ground strip.
- *Cons.*
  - Needs a threshold.
  - "The same stretch" is strict: pilots can also match screens by a building seen in two pieces (A2).
  - Facades are matched by horizontal stretch, not by height band.
- *Cost.* Low to medium: `Footprints.first_hit` must also report which face it hits.

**A2. Shared building (a looser link).**
- *What.* A pair is linked when both screens show any part of the same building within range, even as two separate
  pieces.
- *For the pilot.* Matching screens by a landmark ("that's the same tower") rather than by overlapping pixels,
  which is how people relate photographs taken from different spots.
- *Pros.*
  - Closer to how people relate views.
  - It survives the blind wedge between near views.
  - It pairs naturally with B1 (how differently the building is seen).
- *Cons.*
  - It assumes the building is recognisable. DiamondCityWorld's mix repeats 37 designs, so different buildings can
    look alike. A long block seen at both ends also counts, though the pilot may not realise it is one building.
  - More generous than A1, so it understates the penalty if pilots cannot in fact make these matches.
- *Cost.* Low, once A1 knows which building each ray hits.

**A3. Fragmentation of the ring.**
- *What.* From the A1 (or A2) links: the number of islands (runs of linked screens), the share of orphan screens (no
  linked neighbour), and the largest island as a share of the screens shown. One gap only opens the ring into a
  line, which is still one picture, so islands = max(1, gaps).
- *For the pilot.* How many separate pictures they must hold and place independently.
- *Pros.*
  - One number per moment.
  - Separates one gap from five gaps.
  - Cheap once the links exist.
- *Cons.*
  - Inherits the link threshold.
  - Treats all islands alike, whatever their size or direction, and counts screens rather than content.
- *Cost.* Low.

**A4. Overlap reach.**
- *What.* For each pair of neighbours: the distance at which their views start to overlap (from the two cameras'
  positions and headings, as in section 2), against the distance to the first facade in the direction they would
  overlap. Reported as the share of pairs whose overlap is in reach, and both distances.
- *For the pilot.* Whether the two screens *can* share anything here, or whether the street is too narrow for this
  spread.
- *Pros.*
  - Explains both causes in one comparison: spread sets the overlap distance, the city sets the facade distance.
  - Cheap, with no threshold on shared area.
  - Gives a design rule: the widest spread whose views still meet at a typical facade distance.
- *Cons.*
  - One ray stands for a region.
  - Reaching the overlap is necessary but not sufficient, since the shared area can still be tiny.
  - A diagnostic, not an outcome.
- *Cost.* Low.

### B. Can the pilot match and place what they see?

**B1. Viewpoint difference on a shared building.**
- *What.* For a building two neighbours both show (A2), the angle between the two cameras' lines of sight to the
  parts each sees.
- *For the pilot.* How differently the same building looks on the two screens: much the same at 10°, a different
  face at 60°.
- *Pros.*
  - Measures the "parallax" pilots meet near buildings.
  - Continuous, with no threshold, and read in degrees.
- *Cons.*
  - Defined only where a building is shared.
  - For strictly shared content (A1) it is capped at 74.4° minus the heading gap and says little, so it belongs
    with A2.
  - No threshold for "too different to match" is known. Mental-rotation cost grows with angle (Shepard & Metzler,
    1971), but its form for street scenes is unknown.
- *Cost.* Low.

**B2. Placement error (does the screen show it where it is?).**
- *What.* For everything on a screen, the angle between its direction from the camera and its direction from the
  formation centroid, which is where the pilot effectively sits. Zero for the single drone. Also an order test: the
  share of pairs of objects whose left-to-right order on the ring is the reverse of their true order.
- *For the pilot.* Whether "on my left screen" means on my left, and whether they can turn towards something they
  saw.
- Rough sizes, for content at the edge of a feed:

  | content distance | spread 0.4 | 1.0 | 1.6 |
  |---|---|---|---|
  | 30 m | 3° | 7° | 10° |
  | 7 m | 11° | 19° | 24° |

- *Pros.*
  - Measured per piece of content, and about acting on what is seen.
  - Works on orphan screens too.
  - Grows with clutter.
  - The order test needs no threshold.
- *Cons.*
  - Assumes the pilot reads each screen as seen from the centroid; a trained pilot may correct for that.
  - The display's 0.64 compression shifts content the other way. Content about 1.8 formation radii from its camera
    happens to be shown at its true direction, so what the ring shows differs from this camera-only figure. That
    is worth checking separately as a display question.
- *Cost.* Low.

### C. Penalising coverage (the headline measures)

**C1. Linked coverage.**
- *What.* Content counts only if its screen is linked to at least one neighbour (by A1, or by A2 for a looser
  version). Sweep width, ground per second and time to find are reported on that basis beside "any feed" and
  "headset".
- *For the pilot.* Content on an orphan screen counts as not usefully seen.
- *Pros.*
  - Simple, symmetric, and explained in one sentence.
  - Gives a "best spread" that includes the cost.
  - The single drone is unaffected, so the swarm ÷ single ratios take the penalty automatically.
- *Cons.*
  - All or nothing at the threshold.
  - An isolated linked pair counts fully, though it cannot be placed relative to the rest.
  - A target on an orphan screen can still be spotted.
- *Cost.* Low.

**C2. Graded coverage.**
- *What.* Each screen's content is weighted by its links: by the share of its two neighbours it links to (0, ½, 1),
  or by its island's size ÷ the screens shown.
- *Pros.* Smoother than C1, and the island version charges for fragmentation directly.
- *Cons.*
  - The weights are a modelling choice with no first-principles defence.
  - Harder to explain.
  - The island version punishes a linked pair at the edge heavily.
- *Cost.* Low.

**C3. Placeable coverage.**
- *What.* Content counts only where its placement error (B2) is under a tolerance (for example 15°), or is weighted
  by that error.
- *For the pilot.* Ties "seen" to "knows where it is".
- *Pros.*
  - Measured per piece of content, and continuous.
  - Captures clutter directly.
  - Independent of the overlap threshold.
- *Cons.*
  - Needs the tolerance, and inherits B2's centroid assumption.
  - An orphan screen with small error counts fully, so pair it with C1 or A3.
- *Cost.* Low.

**C4. Side by side, no combined measure.**
- *What.* Coverage stays as it is, and the trade-off is plotted: coverage gained against linkage lost (A1–A4, B1) per
  spread and coverage level.
- *Pros.* No arbitrary combination, and honest about the trade-off.
- *Cons.* No single "best spread": the reader does the weighing, and the cost is easy to overlook.
- *Cost.* None beyond A and B.

### D. Over time and in the task

**D1. Hand-off continuity.**
- *What.* When content leaves one screen while still in range, it counts as handed off if it appears on a
  neighbouring screen within about 1 s. Reported as the share of exits handed on.
- *For the pilot.* Following something from screen to screen. Continuity can link two screens over time even if
  they never overlap at one instant.
- *Pros.* Matches how a moving pilot builds a map, and credits wide formations that still hand off.
- *Cons.*
  - Zero while hovering.
  - Depends on how the scripted walker turns, which is not human-like.
  - More computation.
- *Cost.* Medium.

**D2. Acting on a sighting.**
- *What.* A simulated pilot heads for something seen, along the direction its screen shows. The cost is the extra
  path or time to reach it, or to bring it to the centre of view.
- *Pros.* Task units (seconds and metres), which are the most persuasive.
- *Cons.* Needs a model of how pilots read the screens, with heavy assumptions. Best built after human data show how
  pilots turn after a sighting.
- *Cost.* High.

## 6. Side by side

| option | shows spread | shows clutter | free parameter | cost | answers |
|---|---|---|---|---|---|
| A1 shared content | yes | yes | minimum shared | low–medium | do two screens show the same thing? |
| A2 shared building | yes | yes | range | low | can they be matched by a landmark? |
| A3 fragmentation | yes | yes | via A1/A2 | low | how many separate pictures? |
| A4 overlap reach | yes | yes | none | low | can the views meet here at all? |
| B1 viewpoint difference | yes | strongly | none | low | how differently is a shared building seen? |
| B2 placement error | yes | yes | none (order test) | low | is it where the screen says? |
| C1 linked coverage | yes | yes | via A1/A2 | low | headline with the penalty |
| C2 graded coverage | yes | yes | weights | low | smoother headline |
| C3 placeable coverage | yes | yes | tolerance | low | knowledge-based headline |
| C4 side by side | — | — | none | none | the trade-off, no single number |
| D1 hand-off | yes | yes | time window | medium | continuity while moving |
| D2 acting on a sighting | yes | yes | pilot model | high | cost in task units |

Every replica option needs the walks flown again, because the per-sample views were not saved. S3's manual gives
about 3.5 h for every design at 4 walks per city, or about 12 h for the full grid. A targeted run on S3's cities at
one or two block sizes would take about 1–2 h (estimate).

## 7. Checking it with pilots

The replica cannot say whether these quantities matter to people; the trials can. `citysim/replay.py` already
rebuilds which screens were shown on a recorded flight, so any A–C measure can be computed for a flight and set
beside what the pilot did.

- **H1. Pointing task.** At a pause or at the end of a run, the pilot points (head or controller) to a landmark or
  goal seen earlier, and the error is scored in degrees. A direct measure of environment knowledge, standard in
  spatial cognition. It needs a protocol change and interrupts the task.
- **H2. Gaze switching between adjacent screens** (the gazed feed is logged on runs since 2026-10-12). Cheap and
  already logged, but ambiguous: more switching may simply be good scanning.
- **H3. The turn after first sight.** The direction the pilot turns or flies after a goal first appears on a screen,
  against its true bearing, compared with B2 at that moment. Uses existing data, but has few goals (internal_3 has 4
  swarm runs) and is confounded by strategy.
- **H4. Outcomes already scored.** Time from first sight to the identify press (`sightToPressSec`) and false alarms
  (AutoScore), related to A1–A3 at the moment of sighting.
- **H5. A questionnaire item**, such as "I could tell how the views fitted together", or a situation-awareness
  rating. Cheap but subjective, and one number per trial.

The runs since 2026-10-12, which carry gaze and the recorded shown feeds, are not yet grouped into a test.

## 8. Shortlist and decisions

**Shortlist:**
- **Describe:** A1 (ground + facades) and A3, extending S3's ring measure, with A4 to explain them.
- **Matching and placing:** A2 with B1 (the near-building case), and B2's order test.
- **Headline:** C1 on A1 links, with C1 on A2 links and C3 as sensitivity checks. Add C4's trade-off plots either
  way.
- **People:** H1 and H3 at the next trial block, with A1–A4 computed on the recorded flights.
- **Defer:** D1 and D2.

**Decisions:**
1. Which headline: C1, C2, C3 or C4?
2. Strict links (A1, the same content), landmark links (A2, the same building), or both?
3. The minimums (20 m² of ground / 4 m of facade).
4. For B2 and C3, is the centroid the pilot's reference point?
5. Whether to add a pointing task (H1) to the trial protocol.
6. A targeted run now, or fold into the refined S3 re-run (about 12 h, which also corrects the sweep widths)?

## 9. Caveats

- The replica is 2-D: every building is a 50 m slab, facades are matched by horizontal stretch rather than height
  band, and there are no textures. "Shared" means shared geometry, not recognisably the same.
- The overlap distances and placement errors in sections 2 and 5 are rough geometry for an evenly spread ring, not
  measurements. The metrics would compute them from the actual cameras.
- None of the thresholds has been validated with people.
- The walks are a scripted random walker, not people. Attention is modelled only by the headset basis.
