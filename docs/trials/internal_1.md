# internal_1 · CityWorld, the first trials

- **Dates:** 6 and 7 July 2026
- **Pilots:** ERIC, ERIP
- **City:** CityWorld

## Summary

The first user trials, on the evening of 6 July and into the night: ERIC flew three single-drone and three swarm trials,
ERIP one of each, in CityWorld, the city pack's city at full width. The recorder was a day old and wrote no answers or
presses; the times were taken by hand and are not on this machine, so this page has the recordings and what they show,
not task times. Several trials were restarted, and some recordings are false starts of a few seconds.

## The starting setup

### City

- **CityWorld,** the Modular City Pack's city with its buildings at full width and of mixed height (`856ccb90`), with goal
  patches added the day before (`8795f166`, `6a94956f`).
- **The buildings were not obstacles:** they stood on the Default layer, so the swarm's avoidance never saw them (fixed in
  September, for ScaledCityWorld: `287cf27e`).
- **The goal patches stood off the tile grid,** about half a tile from where a tile would be, a placement bug fixed later.

### Swarm

- **Ten drones under the Olfati-Saber law,** the spread on a dial, and no hollow core.
- **The drones turned at about 35 °/s at most,** a drag bug found and fixed in September (`37094462`, `1f612c27`).

### Display

- **The ring of feeds, and a StabStitch panorama** switched on and off from the controller (`a24209a1`); the head
  calibrated and the gimbal tilted from the controller (`51e28945`, `f7bfe31d`).

### Walkers

- **One special walker at each goal, in a cowboy hat** (`0ca67a48`, `214a3155`).

### Task and scoring

- **The first experiment recorder** (`1041eab5`): drone, head, walker and event logs, but no answers and no presses. The
  times were taken by hand at the trials.

## Notable

### restarts: Recordings and restarts

A trial restarted shows as several recordings; the shortest are false starts.

### smoothness: How the pilot flew

The stick measures need only the head log, so they exist for these runs too, wherever the formation reached the goals'
zones.

## Reading

Read this series as the pilot study it was: it found the problems the next series fixed.

The buildings were invisible to the swarm's avoidance, the drones turned at half their rate, and the answers were not
recorded, so nothing here compares the swarm with one drone on equal terms. What it does show is how the task was flown
before any of the later fixes: restarts, false starts, and the formation's paths through a city it could not see.

## Data notes

- No task times exist in the run files: every session has totalTaskTime 0 and no identify events. The hand-timed list, if
  found, goes in the series folder as manual_times.csv (run;timeSec;source) and replaces the recording lengths.
- One run was labelled SingleDrone but flew ten drones (ERIP_t1_SingleDrone_20260706_222450); it is analysed under its
  label.
- CityWorld has no export of its buildings, so the replays draw the tile grid only, and hat visibility and the discovery
  measures are not computed.
- The practice list in test.json marks ERIC's and ERIP's first trial in each condition, which leaves ERIP no counted run.
- The start line here is derived from the drone logs, as for internal_2 and internal_3; some recordings never cross it.
  The recording lengths on this page run from Play regardless.
