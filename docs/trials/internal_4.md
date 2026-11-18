# internal_4 · The block-shaped patch city

- **Dates:** 6, 10 and 16 November 2026
- **Pilots:** ERKS, ALEX, SIMI
- **City:** BlockShapedPatchCityWorld (SIMI: ScenicPatchCityWorld, the same city in a realistic look)

## Summary

Three pilots in the block-shaped patch city: 289 prism buildings that fill their blocks, on streets laid in turned
patches so that no view runs straight across the city. ERKS (6 November) flew two single-drone runs, three swarm runs
and a third single-drone run, and ALEX (10 November) two single-drone runs, then two swarm runs, both in
BlockShapedPatchCityWorld. SIMI (16 November) flew two swarm runs, then the first two runs of a third condition, one
drone whose 360 camera is shown on a sphere all round the pilot, in ScenicPatchCityWorld: the same city, colliders and
swarm law with a realistic look. Each run's three goals were drawn afresh from the city's own buildings. Practice was
flown and not recorded, so every run here counts.

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
  (`8974165a`). Presses the recorder refused that the experimenter judged correct (five of ERKS's, one of ALEX's, and one
  of SIMI's, pending the experimenter's confirmation) count as correct here (test.json `identifyOverrides`).
- **The wrap rule:** for ALEX and SIMI the swarm kept its full spread while a building was inside the formation (S11's
  Hold, `1a4877ad`); for ERKS it did not.
- **The screens:** ERKS's hung 0.3 m below eye level, ALEX's and SIMI's at eye level, where the layout also no longer
  turned into a map when the camera pointed down (`333b53bb`).
- **SIMI's city looked real:** ScenicPatchCityWorld draws the same buildings with photographed facades, gentle sun
  shadows, a realistic ground and restyled street furniture, and unlit feed screens (`78be29af`); its colliders, goals and
  swarm settings are the trial scene's, and the run files agree field for field. Its pedestrians stand out more against
  the street there (about a quarter more clearly, measured on the bench before it was flown), so identifying a hat is
  easier in it and SIMI's times are not comparable with ERKS's and ALEX's. These were its first flights in the headset,
  and it held the headset's 72 Hz.
- **A third condition, the 360 sphere, flown by SIMI only:** one drone's 360 camera drawn on a sphere round the pilot, with
  no feed screens and no cockpit room (`b563e408`, `65b7481c`); the rig follows the drone's heading, smoothed between
  physics steps (`b600544b`). SIMI flew no single-drone run, so SIMI appears only in the swarm ÷ 360 sphere comparison.
- **Practice:** ALEX practised in PracticeBlockWorld, SIMI in its scenic copy, ScenicPracticeBlockWorld (`e1164cc5`),
  with the start line behind the hat walkers (`965a1fda`).

## Setup

| | internal_3 | internal_4 |
|---|---|---|
| City | DiamondCityWorld: tile blocks and four diamond parks | BlockShapedPatchCityWorld: 289 prism buildings (SIMI: the same city in ScenicPatchCityWorld's realistic look) |
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

The two sessions were four days apart and the trial scene changed between them; the shaded rows are where the two
pilots' own run files disagree.

### legs: The longest legs

A single leg can decide a run's total: ERKS's third swarm run spent most of its time between goals on the way to one
goal. The swarm's long legs are between goals, the single drone's at them.

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
ALEX steered the same way and lost the series' other pair; ERKS turned with yaw and lost none. Four pairs cannot settle
this. Nothing else was found: the swarm's settings, the frame rate and the editor log are as in the trial scene's runs.
SIMI's swarm practice in ScenicPracticeBlockWorld lost one more pair, which only the editor log records.

## Reading

Against one drone with an ordinary camera, the swarm searched the goals faster and travelled between them more slowly,
and for ERKS and ALEX the first outweighed the second. Against one drone with a 360 camera, which only SIMI flew, the
swarm was slower at both. That is one pilot who flew the sphere second, after a swarm run that lost six drones.

At the goals the swarm took between a third and a half of the single drone's time, which is what ten cameras looking round
a ring of buildings at once should buy; between goals it took longer, the formation's price in narrow streets, as the
simulation studies predicted. In total it came out ahead for both pilots, narrowly for ERKS and clearly for ALEX. With two
pilots this is a direction, not a result: one long leg (ERKS's third swarm run) carries much of the swarm's travel time,
and ALEX, the pilot with the larger lead, flew the swarm second, so practice and condition are confounded.

The 360 sphere was added to pull apart two things the swarm gives at once: a view all round, and many points of view. The
sphere has the first without the second, and without the formation's price between goals. In SIMI's runs it found the
goals in about half the swarm's time at them and was quicker between them too. Three things keep that from being a
result. It is one pilot with two runs each, flown swarm first, so practice favours the sphere as it favoured the later
condition for every pilot so far. SIMI's second swarm run lost six drones before its first goal and searched the rest
of the city with four. And SIMI flew the scenic city, so these times compare with each other but not with ERKS's or
ALEX's. What the sphere needs next is pilots who fly it first, and the single drone as well.

The swarm's drone losses are this series' other finding (Why the swarm lost drones, under Other results): all four
pairs came from the stick shield braking one drone at a facade while its partner flew on into it, and the pilots who
lost them steered by swinging the stick rather than with yaw.

## Data notes

- Practice was flown in PracticeBlockWorld (SIMI: ScenicPracticeBlockWorld) and not recorded: every run here counts.
- Trial numbers count within each condition in this series (in internal_2 and internal_3 they run across both).
- ERKS's third swarm run was labelled t3 after a 7 s false start, which is archived.
- SIMI's second swarm run lost six of its ten drones on the way to its first goal, so the rest of it was flown with four. It was
  quit after the bucket hat as if all three goals were found: the press at the cap, which the recorder scored a false
  alarm, had the cap on a shown feed at 20 m with a clear line, 0.85 of the way from the centre to the edge, outside
  the central 80% the recorder accepts; it counts as correct here, pending the experimenter's confirmation.
- SIMI's swarm practice lost one colliding pair 9 minutes in; with no recorder in the practice scene, the editor log is
  its only record.
