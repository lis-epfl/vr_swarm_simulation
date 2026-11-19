# internal_4 · The block-shaped patch city

- **Dates:** 6, 10, 16 and 18 November 2026
- **Pilots:** ERKS, ALEX, SIMI, YEGO
- **City:** BlockShapedPatchCityWorld (SIMI and YEGO: ScenicPatchCityWorld, the same city in a realistic look)

## Summary

Four pilots in the block-shaped patch city: 289 prism buildings that fill their blocks, on streets laid in turned
patches so that no view runs straight across the city. ERKS (6 November) flew two single-drone runs, three swarm runs
and a third single-drone run, and ALEX (10 November) two single-drone runs, then two swarm runs, both in
BlockShapedPatchCityWorld. SIMI (16 November) flew two swarm runs, then the first two runs of a third condition, one
drone whose 360 camera is shown on a sphere all round the pilot, in ScenicPatchCityWorld: the same city, colliders and
swarm law with a realistic look. YEGO (18 November) flew two swarm runs, then two single-drone runs, also in
ScenicPatchCityWorld: the first pilot to fly the swarm before the single drone. Each run's three goals were drawn afresh
from the city's own buildings. Practice was not recorded, so every run here counts.

## Changes since internal_3

### City

- **BlockShapedPatchCityWorld replaces DiamondCityWorld:** windowed prism buildings, each filling its block to the back
  of a 3.5 m footpath, on streets laid in turned patches so that no long straight view crosses the city (`504f068a`).
- **Goals drawn from the city at every Play:** a goal is one building and the ring of buildings across the streets from
  it, with pedestrians walking round the ring; the goals, the hats and the target walkers change every session
  (`a662bc31`).
- **Landmarks on all four sides,** for orientation: a mountain range to the south, a lake to the north, wind turbines
  to the east, a TV mast to the west in line with the start road; made taller twice for this city's narrow streets
  (`56d24e45`, `980da54b`, `a58decfd`).
- **Denser street furniture:** border trees, benches and bins, bus stops, hydrants and payphones (`37ba79bf`).

### Swarm

- **A wider formation that flows round buildings:** S6's spread (d_ref 2.0) and shield (`f5d7d01d`); S7's slide round
  buildings and adaptive spread, which narrows the formation to 0.85 of d_ref in the streets (`7bf41e8c`, `e545e7f7`);
  S10's ellipse round each building's footprint in place of a circle (`37a6ccf0`).
- **The swarm starts as a hollow ring** of ten drones, none at the centre (`d736f6b9`).

### Controls

- **The spread dial does nothing** while the adaptive spread is on, and **the gimbal starts level**, moving only once its
  dial is turned (`9b809bfc`).
- **The yaw stick turns the pilot at 40 °/s with expo in both conditions,** where it was 90 °/s with the swarm and 75 °/s
  with one drone (`7d1d0f28`).

### Display

- **No stitching:** the panorama is gone and every feed is 1920×1080 (`d2bb65e0`, `83b3ad63`).
- **The screens are 1.3× larger,** 60° wide at 2 m (flown from ERKS's runs on; committed in `333b53bb`).
- **A dim room fixed to the real room** behind the screens, a grid floor and ceiling, against motion sickness
  (`c096133a`).
- **Steadier frames:** baked occlusion and walkers asleep far from every camera halve the main thread's time
  (`e33b8e92`).

### Walkers

- **Taller walkers (1.83 m) with larger hats** relative to the head (`5933e617`), and **hats 10% darker** (`dc9a4240`).

### Task and scoring

- **A start line:** the task clock starts when the first drone crosses a line 30 m ahead of the spawn, so take-off and
  looking round are not timed (`732fb769`).
- **Presses scored on the spot:** a press is correct when the hat is in the central 80% of a shown feed, in range and
  unobstructed; otherwise a false alarm, or outside every goal's zone (`c6b85655`).
- **Each run records what it flew:** its commit, its settings and every drone lost with its reason (`602484db`).

## Within the series

- **Scoring:** ERKS's presses were scored at 40 m with no look-back, each counting for the goal whose zone held the
  formation's centre; ALEX's at 80 m, looking back 3 s, for any goal (`abef940b`); SIMI's at 80 m, looking back 5 s
  (`8974165a`); YEGO's as SIMI's. Every press the recorder refused in this series was the right hat, as the experimenter
  judged it: five of ERKS's, one of ALEX's, one of SIMI's and three of YEGO's count as correct here (test.json
  `identifyOverrides`). ERKS's fell to rules loosened on 6 November (the range, the look-back, the zone) as well as to the
  central 80% of the frame the hat had to be in; ALEX's, SIMI's and YEGO's to the central 80% alone, so after YEGO's
  runs the recorder accepts the hat anywhere in a shown feed.
- **YEGO pressed two or three times per hat** in the first swarm run: the recorder drops a press within 1 s of the last,
  so only the first press of each burst was scored, and the repeats are in the event log as ignored.
- **The wrap rule:** for ALEX, SIMI and YEGO the swarm kept its full spread while a building was inside the formation
  (S11's Hold, `1a4877ad`); for ERKS it did not.
- **The screens:** ERKS's hung 0.3 m below eye level, the others' at eye level, where the layout also no longer
  turned into a map when the camera pointed down (`333b53bb`).
- **SIMI's and YEGO's city looked real:** ScenicPatchCityWorld draws the same buildings with photographed facades,
  gentle sun shadows, a realistic ground and restyled street furniture, and unlit feed screens (`78be29af`); its
  colliders, goals and swarm settings are the trial scene's, and the run files agree field for field. Its pedestrians
  stand out more against the street there (about a quarter more clearly, measured on the bench before it was flown), so
  identifying a hat is easier in it and SIMI's and YEGO's times are not comparable with ERKS's and ALEX's. Each pilot's
  swarm ÷ single-drone ratio compares two conditions in one city, so YEGO's still pools with ERKS's and ALEX's. SIMI's
  were its first flights in the headset, and it held the headset's 72 Hz for both pilots.
- **Walkers asleep past 500 m, not 350 m,** from YEGO's last single-drone run (`9d9d3079`, committed during the
  session): a patrol farther than that from every drone camera stops and hides its walkers' bodies, though never the
  hat. The run files do not record the distance, so YEGO's earlier runs most likely flew 350 m, as every earlier pilot did.
- **A third condition, the 360 sphere, flown by SIMI only:** one drone's 360 camera drawn on a sphere round the pilot, with
  no feed screens and no cockpit room (`b563e408`, `65b7481c`); the rig follows the drone's heading, smoothed between
  physics steps (`b600544b`). SIMI flew no single-drone run, so SIMI appears only in the swarm ÷ 360 sphere comparison.
- **Practice:** ALEX practised in PracticeBlockWorld, SIMI in its scenic copy, ScenicPracticeBlockWorld (`e1164cc5`),
  with the start line behind the hat walkers (`965a1fda`). Practice scenes have no recorder, so no practice is in the data.

## Setup

| | internal_3 | internal_4 |
|---|---|---|
| City | DiamondCityWorld: tile blocks and four diamond parks | BlockShapedPatchCityWorld: 289 prism buildings (SIMI, YEGO: the same city in ScenicPatchCityWorld's realistic look) |
| Conditions | swarm, single drone | swarm, single drone; SIMI: swarm and the 360 sphere |
| Goals | three tiles, each a patch of nine buildings | three buildings with their rings, drawn each Play |
| Spread | the dial, d_ref 0.4 to 1.6 | d_ref 2.0, narrowed to 0.85 of it in the streets |
| Avoidance | a circle round each building's bounds | an ellipse round each building's footprint |
| Screens | feeds 1280×720 at scale 1, and the StabStitch panorama | feeds 1920×1080 at scale 1.3, no panorama |
| Yaw stick | 90 °/s with the swarm, 75 °/s with one drone | 40 °/s with expo, both |
| Behind the screens | black | a room-fixed grid floor and ceiling |
| Answers | an RC press, its outcome not keyed | an RC press, scored on the spot |
| Task clock | from Play (here, from the start line, derived) | from the start line |

## Notable

### setups: Each pilot's setup

The four sessions span twelve days and the software changed between them; the shaded rows are where the pilots' own
run files disagree.

### legs: The longest legs

A single leg can decide a run's total: ERKS's third swarm run spent most of its time between goals on the way to one
goal, and YEGO's second single-drone run lost its way to its last goal. Most of the longest legs at a goal are the
single drone's, searching one goal's ring with one camera.

### losses: Why the swarm lost drones

Every drone lost in this series was one of a colliding pair, and the four pairs came about the same way. One drone of
the pair reached a building and was braked there by the stick shield, which takes away the part of the pilot's command
that points into a building; its partner, still clear of the building and about 20 m behind it on the line of the
stick, kept the full command and flew into it, closing at up to 9 m/s in SIMI's three. The shield stops every drone
short of a facade at about the same distance, so a drone arriving second on the same line is stopped where the first is
already parked. At full stick the pilot's command outweighs the formation's own repulsion between drones, and the
close-range damper reaches only 3 m, too short to shed that speed. The replica's force-by-force account of the recorded
states (`swarm_replica.py forces`) shows the same sequence in all four: the shield's difference between the two drones
builds the closing speed over about two seconds, the building pushes the braked drone back towards its partner, and
both drones are at their tilt limit when the damper acts in the last fifth of a second.

SIMI's second swarm run lost three pairs within about a minute of entering the city. Five times in that minute two
drones closing fast came within a metre of each other, which no other run in the series did. SIMI did not push the
formation into buildings more than ERKS or ALEX, measured by how much of the stick the shield took away. What differed
is how SIMI steered: by swinging the stick's direction at full deflection rather than with the yaw stick, which turns
the command at once where yaw turns it at 40 °/s and can stop a drone that was sliding along a facade in another's path.
ALEX steered the same way and lost the series' other pair; ERKS turned with yaw and lost none. YEGO then swung the
stick's direction more than any of them and lost none either, though two fast closings in YEGO's second swarm run
passed within a metre. So steering by the stick is at most one ingredient, not the cause.

A second look (19 November) found nothing that made SIMI's second run riskier, only that its risky moments ended worse.
It did not meet the danger more often: it had the fewest arrivals at a facade at full stick pointing straight into it,
3.7 per km against 5.9 to 20 in the other runs, and the least time with one drone 15 to 30 m straight behind another on
the line of the stick, a quarter of its full-stick time against 45% to 57%. What set its three collisions apart is the
run-up: each time the drone that hit started 16 to 22 m behind the braked one, within about 1.5 m of the line of the
stick, with no drone between them, so it had the whole distance to build speed and nothing to deflect it. ALEX's pair
was two neighbours closing sideways. Elsewhere in the series a drone stopped at a facade with another that far back and
that close to the line once, in YEGO's second run, and the two passed 3.4 m apart. The formation was as flat and as
round as in every other run (the drones within a few centimetres of one height throughout, so any near miss in plan is
a collision). Where it happened cannot be tested: two of the three were in a part of the city no other run came within
29 m of, and ERKS's third run passed the first within 5 m and lost nothing. If losses fell in proportion to the
distance flown, one of the nine swarm runs would collect three of the four pairs about one time in fifteen. The best
reading is that SIMI's run met the worst form of a hazard every run carries three times in a minute, not that SIMI flew
differently. SIMI's swarm practice in ScenicPracticeBlockWorld lost one more pair, which only the editor log records.

## Reading

Against one drone with an ordinary camera, all three pilots who flew both finished sooner with the swarm, ERKS only
just: it found each goal's hat in about a third of the time once there, which more than paid for slower travel between
goals. YEGO flew the swarm first and was still quicker with it, so practice no longer explains the lead away. Against one
drone with a 360 camera, which only SIMI flew, the swarm was slower at both.

At the goals the swarm took between a quarter and a half of the single drone's time for every pilot, which is what ten
cameras looking round a ring of buildings at once should buy. Between goals it took longer for every pilot, the
formation's price in narrow streets, as the simulation studies predicted, but by little except for ERKS, whose third
swarm run spent most of its travel on one leg. Until YEGO, each pilot was quicker in the condition flown later, and the
swarm had been flown later; YEGO's runs are the first against that order, and the swarm's lead held. Three pilots make
this a direction worth testing, not a result.

The 360 sphere was added to pull apart two things the swarm gives at once: a view all round, and many points of view. The
sphere has the first without the second, and without the formation's price between goals. In SIMI's runs it took about
three fifths of the swarm's time at the goals and half its time between them. Three things keep that from being a
result. It is one pilot with two runs each, flown swarm first, so practice favours the sphere. SIMI's second swarm run
lost six drones before its first goal and searched the rest of the city with four; it is kept in, but without it the
sphere's share rises to about four fifths at the goals and three quarters between them (Data notes). And SIMI flew the scenic city, so
these times compare with YEGO's but not with ERKS's or ALEX's. What the sphere needs next is pilots who fly it first,
and the single drone as well.

The swarm's drone losses are this series' other finding (Why the swarm lost drones, under Other results): all four
pairs came from the stick shield braking one drone at a facade while its partner flew on into it. The pilots who lost
them steered by swinging the stick rather than with yaw, but YEGO steered that way more than anyone and lost none.

## Data notes

- Practice was flown in PracticeBlockWorld or its scenic copy, ScenicPracticeBlockWorld, which have no recorder: every
  run here counts.
- Trial numbers count within each condition in this series (in internal_2 and internal_3 they run across both).
- ERKS's third swarm run was labelled t3 after a 7 s false start, which is archived.
- SIMI's second swarm run lost six of its ten drones on the way to its first goal, so the rest of it was flown with four.
  **It is kept in for now, but read it as a degraded run:** with four drones the formation shows four feeds, not ten,
  and the run took 632 s against 323 s for SIMI's first swarm run (353 s between goals against 145, 279 s at the goals
  against 178). Excluding it was considered (19 November) and left for later. It carries most of SIMI's swarm-against-
  sphere difference: with it, the sphere took 0.54 of the swarm's time in all, 0.64 at the goals and 0.44 between them;
  against the first swarm run alone, 0.79, 0.82 and 0.75. The sphere was quicker either way. Why it lost so many drones
  is under Why the swarm lost drones. It was
  quit after the bucket hat as if all three goals were found: the press at the cap, which the recorder scored a false
  alarm, had the cap on a shown feed at 20 m with a clear line, 0.85 of the way from the centre to the edge, outside
  the central 80% the recorder then accepted; it counts as correct here.
- SIMI's swarm practice lost one colliding pair 9 minutes in; with no recorder in the practice scene, the editor log is
  its only record.
- YEGO's second swarm run was recorded as t1 and is relabelled t2 (its session file says so). Four short restarts of
  YEGO's second single-drone run, none of which reached a goal, are archived.
- Both of YEGO's swarm runs were quit as if all three goals were found, and three of YEGO's presses the recorder scored
  as false alarms: the cap in the first run, the bucket and the cap in the second. Each had the hat on a shown feed with
  a clear line, in the pilot's head view, but outside the central 80% the recorder then accepted: the first cap close to
  the edge of the frame for four seconds, the bucket at the very edge, the second cap for under half a second as it came
  round a building. All three were the right hat, and they count as correct here.
- The time at a goal ends at the press that answered it, and what follows is time between goals (since 18 November
  2026). It used to end at the last exit from the goal's zone before the next goal's zone was entered, which counted
  flying out of the patch after the answer as time at the goal, and any pass back through it while looking for the next
  goal: in YEGO's second swarm run, nearly two minutes after the bucket was answered counted at the bucket, because the
  formation passed back through its zone on the way to the cap.
