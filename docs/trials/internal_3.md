# internal_3 · DiamondCityWorld

- **Dates:** 1 October 2026
- **Pilots:** BBAA, BBCC
- **City:** DiamondCityWorld

## Summary

Two pilots on 1 October, two runs in each condition: BBAA flew the single drone first, BBCC the swarm. The city was
DiamondCityWorld, the tile city laid out eight by eight with its rows staggered and four diamond-shaped parks, each a
superblock of four tiles. Practice was flown separately and not recorded, so every run here counts. The pilot pressed the
RC's button at each identify; the outcomes were not keyed.

## Changes since internal_2

### City

- **DiamondCityWorld replaces ScaledCityWorld:** an eight-by-eight grid of tiles with staggered rows and four diamond parks
  of four tiles each (`88aa94e7`, `fe38e13a`, `57559907`), the spawn road twice as long (`9490e308`).
- **Verge trees on every goal patch** (`2d2ce82c`).

### Swarm

- **The fix for internal_2's collisions at the tightest spread:** the hollow core now matches velocity relative to the
  swarm instead of braking every ring drone, and a close-range damper opposes two drones closing on each other
  (`fe3a448b`).

### Display

- **72 fps held in the swarm condition, feeds 1280×720 and the panorama 2400×900,** with feeds out of the pilot's view
  refreshed at 5 Hz (`efe9209d`).
- **A compass arrow below the pilot** showing the body heading (`360c2d63`).

### Walkers

- **Smaller bucket and cowboy hats;** the cap unchanged (`e622f617`).

### Task and scoring

- **The pilot presses the RC's C2 button to identify,** logged with its time; the outcome of each press was not keyed in
  this series (`e622f617`).

## Setup

| | internal_2 | internal_3 |
|---|---|---|
| City | ScaledCityWorld: a regular tile grid, buildings 50 m tall | DiamondCityWorld: staggered rows, four diamond parks |
| Swarm law | hollow core braking on absolute velocity, no damper | core relative to the swarm, close-range damper |
| Spread | the dial, d_ref 0.4 to 1.6 | the same |
| Screens | feeds 1152×648, panorama 2160×810 | feeds 1280×720, panorama 2400×900, 72 fps held |
| Hats | three hats | bucket and cowboy smaller, cap unchanged |
| Answers | said aloud, entered afterwards | an RC press, its outcome not keyed |
| Practice | the first run in each condition, recorded | flown separately, not recorded |
| Task clock | from Play (here, from the start line, derived) | from Play (here, from the start line, derived) |

## Notable

### losses: Drones lost

Every loss was a pair. BBCC's were drones colliding at the tightest spread the dial allows (d_ref 0.4), the failure the
damper was added for: it slowed the slow closings, not the fast ones that a building's shield or a hard stop forced.
BBAA's one pair, at d_ref 1.0, began with a drone pinned on a street-light pole in a goal patch, a collider on the Default
layer that the force law cannot see; another drone then ran into it.

### panorama: The stitched panorama

The panorama replaced the three front feeds for only a small part of each swarm run: its quality gate hid it whenever the
views did not overlap well enough.

## Reading

The two pilots disagree: the swarm was quicker in total for BBAA and slower for BBCC, and each was quicker in the
condition flown second.

That makes this series a measure of practice as much as of the swarm. The one phase both pilots agree on is the way
between goals, where the swarm was quicker for both; at the goals BBAA's swarm took about half the single drone's time and
BBCC's about twice. BBCC flew the swarm at the tightest spread and lost eight drones in its two swarm runs, while BBAA kept
the spread near 1.0. With two pilots and two runs each, no ratio here could reach p 0.05 however one-sided.

## Data notes

- Practice was flown separately and not recorded: every run here counts.
- Answers were pressed on the RC, so the total time ends at the last press; the presses' outcomes were not keyed, so the
  identification card shows presses, not correct answers.
- The editor log of that day is not in the series folder, so the losses carry no recorded reason here; the log read on
  1 October named every one a drone-drone collision (one pair after a street-light pole).
- The task is timed from the start line, found in the drone logs; the series itself was timed from Play.
