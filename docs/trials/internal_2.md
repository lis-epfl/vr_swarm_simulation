# internal_2 · ScaledCityWorld

- **Dates:** 23 September 2026
- **Pilots:** AABB, AABC
- **City:** ScaledCityWorld

## Summary

Two pilots on 23 September, three runs in each condition: AABB flew the swarm first, AABC the single drone. The first run
in each condition was practice, which leaves two counted runs per pilot and condition. The city was ScaledCityWorld, the
city pack's tiles with every building a quarter as wide and 50 m tall. Answers were said aloud and entered after the
session, so each run ends where the experimenter stopped the recording.

## Changes since internal_1

### City

- **ScaledCityWorld replaces CityWorld:** the pack's buildings at a quarter of their width and all 50 m tall, so the
  streets are wide and the skyline flat (`287cf27e`), with the streets widened further (`a588d2fd`).
- **The buildings are obstacles at last:** in CityWorld they stood on the wrong layer, and the swarm's avoidance never saw
  them (`287cf27e`, `d3cae643`).
- **Goal patches with buildings twice as wide and a centre building** (`fa1b302f`, `512f49fd`), and **an altitude
  ceiling** 50 m above the ground (`518f532e`).

### Swarm

- **Each building avoided as a cylinder,** damped and shielded (`23e49a1e`, `eaffb2e1`); **a hollow core,** so more drones
  stand on the formation's edge with their feeds shown (`e0d9d30b`); **the gains retuned** for flying through the city
  (`1cb148ec`).
- **A heading loop:** each drone turns at its full 75 °/s, where a drag bug had held it near 35 °/s (`37094462`,
  `1f612c27`), and a feed waits until its drone faces its heading (`5ea29c90`).

### Display

- **Feeds 1152×648 and the panorama 2160×810** (`0f65d325`, `1afd5a23`), StabStitch faster (`6250581b`), and the feeds
  rendered at 30 Hz to lift the frame rate (`d130fcfb`).

### Walkers

- **Three hats,** a cowboy hat, a cap and a bucket hat, a different one at each goal (`87138574`, `21cfafe4`), hung on the
  walker's head, with 8 walkers per patrol (`38143432`).

### Task and scoring

- **The single-drone condition** shows its one feed in front of the pilot (`041ed4fd`), and **a practice world**
  (`a36371b8`).
- **Answers said aloud and entered after the session** from the experimenter's notes, matched to the goals in the order
  visited.

## Setup

| | internal_1 | internal_2 |
|---|---|---|
| City | CityWorld: full-width buildings of mixed height | ScaledCityWorld: quarter-width buildings, all 50 m |
| Obstacles | the buildings unseen by the swarm's avoidance | every building a cylinder the swarm avoids |
| Swarm law | no hollow core | hollow core, gains retuned for the city |
| Drone yaw | about 35 °/s at most | 75 °/s, with a heading loop |
| Spread | the dial | the dial, d_ref 0.4 to 1.6 |
| Screens | the feed ring, a panorama on a switch | feeds 1152×648, panorama 2160×810 |
| Hats | one cowboy hat | three hats, one per goal |
| Answers | none recorded (timed by hand) | said aloud, entered afterwards |
| Task clock | from Play (by hand) | from Play (here, from the start line, derived) |

## Notable

### smoothness: How the pilot flew

The stick measures are what this series was first analysed for: whether pilots hold one command and let the swarm find its
way, or keep re-steering.

### losses: Drones lost

Nearly every loss was a pair of drones colliding at the tightest spread the dial allows (d_ref 0.4): the hollow core
braked each ring drone by a different amount, and the lattice could not push a close pair apart hard enough. The next
series flew the fix.

### panorama: The stitched panorama

How long the stitched panorama was on screen in place of the three front feeds. These runs predate the recorder's
panorama events: the editor log of the day records each switch, but not always when, so most of each run is unknown and
the share is a wide range.

## Reading

In total the swarm was level with one drone for AABB and quicker for AABC, and neither the goals nor the way between them
separate the two conditions.

AABC flew the swarm second, so its lead is also a lead of practice; AABB, who flew the swarm first, was level. The clearest
difference is in how the pilots flew: AABC steered the swarm far less than the single drone (less stick movement, fewer
adjustments, much less turning), AABB more. Both of AABB's counted swarm runs lost drones to collisions at the
tightest spread (d_ref 0.4), the failure the next series' damper was built for; AABC's did not.

## Data notes

- The first run of each pilot in each condition was practice (test.json): drawn, counted in no summary.
- The answers were entered after the session, so no run records when a hat was named: the total time ends where the
  experimenter stopped the recording, at the last answer said aloud.
- The task is timed from the start line, found in the drone logs; the series itself was timed from Play.
- Drone losses are attributed from the editor log kept that day.
