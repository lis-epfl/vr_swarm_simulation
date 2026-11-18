# Footage for the participant instruction video

Unity recordings go here, under the names `script.yaml` uses, which are the headless shot names:
`<shot>_overhead.mp4`, `<shot>_interface.mp4`, `<shot>_log.csv` and so on. After each build, `../build/shots.md`
lists every shot and whether it has been found. The files are git-ignored (they are large); the takes stay in
`D:\claude_scenic_bench\video_shots\take<N>`.

All of it is recorded headlessly in **ScenicPracticeBlockWorld** by `VideoShotRunner`
([Video/README.md](../../README.md)) and copied in with `python Video/participant_instructions/import_takes.py <take>`.
The overhead camera draws each drone's frustum in the Scene-view gizmo's colours: green on the edge (its feed is on a
screen), red inside the swarm.

| shot | cameras | what | used by |
|---|---|---|---|
| `goalview` | overhead (held over the goal) | the goal and its ring, the hat wearer ringed in yellow | goal_hat |
| `hats` | close-up | the cap, bucket and cowboy stations by the start line, 3.2 s each | hats |
| `search` | overhead, interface, front feed | start to goal, wrap the centre building, turn until the hat is in front | surround_tip, search_* |
| `hover` | overhead, interface (raised) | hovering before the city | interface, boundary |
| `gap` | overhead, interface (raised) | down a street narrower than the swarm | avoidance, hidden_screens |
| `split` | overhead | full stick straight at a building | into_buildings |
| `spread` | overhead | down a street, then held on the building at its end | swarm_size |
| `ctl_forward`, `ctl_sideways` | overhead | the right stick, each way | ctl_forward, ctl_sideways |
| `ctl_updown`, `ctl_yaw`, `ctl_pitch` | interface | the left stick and the gimbal dial | ctl_updown, ctl_yaw, ctl_pitch |

`<shot>_log.csv` is the flight, frame by frame (sticks, gimbal dial, height, swarm size, the overhead camera's pose,
every drone's position and edge state): the controller photo's sticks, the height and swarm-size gauges and the
drone marker read it. `search_front_hat.csv` is where the hat is in the front feed; `events.txt` says when the hat
came in front. A new take moves the times marked "take" in `script.yaml`.
