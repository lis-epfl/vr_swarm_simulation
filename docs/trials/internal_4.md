# internal_4 · The block-shaped patch city

- **Dates:** 6 and 10 November 2026
- **Pilots:** ERKS, ALEX
- **City:** BlockShapedPatchCityWorld

## Summary

Two pilots, ERKS on 6 November and ALEX on 10 November, in BlockShapedPatchCityWorld: 289 prism buildings that fill
their blocks, on streets laid in turned patches so that no view runs straight across the city. ERKS flew two
single-drone runs, three swarm runs and a third single-drone run; ALEX two single-drone runs, then two swarm runs. Each
run's three goals were drawn afresh from the city's own buildings. Practice was flown in PracticeBlockWorld and not
recorded, so every run here counts.

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
  formation's centre; ALEX's at 80 m, looking back 3 s, for any goal (`abef940b`). Presses the recorder refused that the
  experimenter judged correct (five of ERKS's, one of ALEX's) count as correct here (test.json `identifyOverrides`).
- **The wrap rule:** for ALEX the swarm kept its full spread while a building was inside the formation (S11's Hold,
  `1a4877ad`); for ERKS it did not.
- **The screens:** ERKS's hung 0.3 m below eye level, ALEX's at eye level, where the layout also no longer turned into a
  map when the camera pointed down (`333b53bb`).

## Setup

| | internal_3 | internal_4 |
|---|---|---|
| City | DiamondCityWorld: tile blocks and four diamond parks | BlockShapedPatchCityWorld: 289 prism buildings |
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

## Reading

The swarm searched the goals faster and travelled between them more slowly, and for both pilots the first outweighed the
second.

At the goals the swarm took between a third and a half of the single drone's time, which is what ten cameras looking round
a ring of buildings at once should buy; between goals it took longer, the formation's price in narrow streets, as the
simulation studies predicted. In total it came out ahead for both pilots, narrowly for ERKS and clearly for ALEX. With two
pilots this is a direction, not a result: one long leg (ERKS's third swarm run) carries much of the swarm's travel time,
and ALEX, the pilot with the larger lead, flew the swarm second, so practice and condition are confounded.

## Data notes

- Practice was flown in PracticeBlockWorld and not recorded: every run here counts.
- Trial numbers count within each condition in this series (in internal_2 and internal_3 they run across both).
- ERKS's third swarm run was labelled t3 after a 7 s false start, which is archived.
