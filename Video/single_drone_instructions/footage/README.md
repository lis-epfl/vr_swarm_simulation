# Footage for the single-drone instruction video

Unity recordings go here, under the names `script.yaml` uses (`<shot>_overhead.mp4`, `<shot>_interface.mp4`,
`<shot>_log.csv` and so on). The files are git-ignored; the takes stay in `D:\claude_scenic_bench\video_shots\single<N>`.

Recorded headlessly in **ScenicPracticeBlockWorld** with one drone (`drones: 1` in the shot list) by `VideoShotRunner`
([Video/README.md](../../README.md)): `python Video/single_drone_instructions/shots.py <take>` writes the shot list,
and `python Video/single_drone_instructions/import_takes.py <take>` copies the take in, with the swarm video's
drone-free clips beside it. The overhead camera draws the drone's camera frustum in green.

| shot | cameras | what | used by |
|---|---|---|---|
| `goalview` | overhead (held over the goal) | the goal and its ring, the hat wearer ringed in yellow | goal_hat |
| `search` | overhead (held over the goal, east up), interface, front feed | in along the street from the start, round the centre building until the hat is in view, stop and turn to it | circle_tip, search_* |
| `avoid` | overhead, interface | full stick straight at a building, no steering | avoidance |
| `hover` | overhead, interface (raised) | hovering before the city | interface |
| `ctl_forward` | interface (high, sees the floor arrow), overhead | the right stick up and down | ctl_forward |
| `ctl_sideways` | overhead (east up), interface | the right stick left and right | ctl_sideways |
| `ctl_updown`, `ctl_yaw`, `ctl_pitch` | interface (+ overhead east up for yaw) | the left stick and the gimbal dial | ctl_updown, ctl_yaw, ctl_pitch |
| `landmarks_views`, `hats_closeup` | still cameras | **the swarm video's** (no drone in them), copied by `import_takes.py` | landmarks, hats |

`events.txt` says when the search saw the hat and when it was in the middle of the screen; `search_front_hat.csv`
is where the hat is in the drone's own feed. A new take moves the times marked "take" in `script.yaml`.
