# Videos

Instruction and explainer videos, one folder each, built from a script by one shared tool.

| folder | what it is |
|---|---|
| [participant_instructions/](participant_instructions/) | The swarm, its controls and the hat search, shown to participants before the practice run. |
| [single_drone_instructions/](single_drone_instructions/) | The same for one drone: its screen, controls and the hat search. Reuses the swarm video's pictures (its `visuals.py` loads theirs) and its drone-free clips. |
| [tools/](tools/) | `build_video.py` (the builder) and `vidkit.py` (drawing, narration, captions, rendering). |
| `_recordings/` | Raw Unity takes from `Tools/Video` (git-ignored). |

## Building a video

Use the `stitching` miniconda env (numpy, OpenCV, Pillow, PyYAML) with ffmpeg on PATH:

```
python Video/tools/build_video.py Video/participant_instructions             # 1920x1080, 30 fps
python Video/tools/build_video.py Video/participant_instructions --preview   # half size, 15 fps, fast
python Video/tools/build_video.py Video/participant_instructions --stills    # one PNG per cue
python Video/tools/build_video.py Video/participant_instructions --only ctl_yaw,ctl_pitch
```

Output goes to `<video>/build/` (git-ignored): the MP4, `subtitles_<name>.srt` with the same subtitles (named so a player does not load it over the burned-in ones), and
`shots.md`, the Unity recordings the script asks for and which of them are still missing.

## A video folder

| path | |
|---|---|
| `script.yaml` | The cues: narration, picture, footage, timing. **This is the file to edit.** Its header lists every field. |
| `visuals.py` | That video's generated pictures (schematics, maps, the controller), named from `script.yaml`. |
| `footage/` | Unity recordings, named as `script.yaml` asks. A cue uses its file as soon as it exists; until then it shows its schematic with a red PLACEHOLDER banner. Git-ignored except its README. |
| `narration/` | Optional: your own voice. `narration/<cue id>.wav` (or .mp3/.m4a/.flac) replaces the computer voice for that cue, and the cue stretches to fit it. |

How a cue's length is decided: the narration plus `lead` and `tail`, or `duration` if longer, or the
whole clip with `duration: clip`. So rewording a line re-times the video by itself.

Narration comes from ElevenLabs when `script.yaml` says `tts: elevenlabs` (voice, model and settings under
`elevenlabs:`) and a key is set: the **secret** key, `sk_...`, in `$ELEVENLABS_API_KEY` or in
`~/.elevenlabs_api_key` (never in the repo). Without one, or if the API refuses, the build says so and
uses Windows' own voices instead (SAPI, offline: `Microsoft Hazel Desktop`, en-GB;
`powershell -File Video/tools/tts_sapi.ps1 -List` shows what is installed). Clips are cached per text,
voice and settings in `build/tts/`, so only changed lines are sent again.

## Recording Unity footage

`Tools/Video` in the Unity editor (`Assets/Scripts/Video/`):

1. **Capture rig at Play** (tick it once; it is remembered per user). Every Play then adds two cameras,
   which change nothing about the flight:
   - **Overhead** (`VideoOverhead`): straight down over the swarm, following it, turned so the pilot's
     forward points up. Each drone's camera frustum is drawn for this camera only, as the Scene-view
     gizmo draws it (`Tools/Swarm/Drone Frustum Gizmos`): green while the drone is on the edge (its feed
     is on a screen), red while it is inside the swarm. Gizmos never reach a camera, hence the copy.
   - **Interface** (`VideoInterface`): just behind and above the pilot's head, looking at the ring of
     screens, with a simple head and shoulders where the pilot sits.

   A small preview of both shows in the Game view's corner (not recorded). Keys: **F5** preview on/off,
   **F6** overhead follows / holds still, **F7** north-up / heading-up, **PageUp / PageDown** zoom the
   overhead view. The rig's inspector has the rest (marker size, cone length, camera offsets, and
   `highlightHats`, which rings the hat wearers: useful when editing, never in footage of the task).
2. **Record at Play**: also records both cameras with the Unity Recorder from the start of Play to the
   end, into `Video/_recordings/<time>_<scene>_overhead.mp4` and `..._interface.mp4`: one take, two
   angles, in sync. Or leave it off and use **Start recording now / Stop recording now** during Play for
   several takes in one session. *Record the interface / overhead camera* choose which.
3. Trim each take (any editor; `ffmpeg -ss <start> -to <end> -i in.mp4 -c copy out.mp4` is enough) and
   save it in the video's `footage/` under the name in `build/shots.md`.

Recording uses the game's own frame rate (variable), so the flight runs at normal speed. It does cost
two extra renders a frame, so flying in the headset while recording may drop to 36 fps; footage for
the video does not need the headset, and a desktop session flown with the RC is fine for every
overhead shot. The interface camera follows the eye camera, so without a headset it sits at the
calibrated seat looking straight ahead.

The first time, run **Add capture tags** (the toggles also do it): it adds the two camera tags to
`ProjectSettings/TagManager.asset`, which the Recorder finds the cameras by.

## Recording headlessly (no headset, no editor)

`VideoShotRunner` flies a scripted shot list on a copy of the project and writes each camera straight to
MP4 through ffmpeg, at a fixed 30 fps game clock, so the footage plays at real speed however slowly the
copy renders. The pilot writes `InputManager.InputStatus` as the bench's does and turns the view with the
yaw stick, so the screens turn round the seat as a participant's would.

```
powershell -File Assets/Scripts/swarm/Bench/run_bench.ps1 -BenchRoot D:\claude_scenic_bench -SyncOnly
set VIDEO_SHOTS_CONFIG=D:\claude_scenic_bench\video_shots\take5\shots.json
set STITCH_SHM_SUFFIX=_videoshots
Unity.exe -batchmode -projectPath D:\claude_scenic_bench\vr_swarm_simulation -executeMethod VideoShotLauncher.Run -logFile <log>
python Video/participant_instructions/import_takes.py D:/claude_scenic_bench/video_shots/take5
```

The shot list's fields are in `VideoShotRunner.cs` (`Config`, `Shot`, `Seg`); `drones` in the config spawns that many drones whatever the scene says (the single-drone video's `shots.py` sets 1). Segments: `hold` (centre on a
point), `goto` (fly there, optionally turning to face the way), `stick` (a body-frame stick, eased in and out),
`gimbal` (ramp the dial), `findHat` (hold, turn until the goal's hat wearer is in the central 80 % of an edge
drone's feed within `hatRange` with no building in the way, then bring that screen to the front and log the
moment to `events.txt`), `patrol` (fly a loop of waypoints facing the way it flies until the hat is in view, then stop and turn the drone to it; a lone drone's feed always counts as shown). A shot may hold the overhead view over a point (`centre`), fix its up direction (`headingUp: false`, `upYaw`), ring the hat wearers
(`highlightHats`), film named objects close up (`closeups`), and record the front screen's own feed with the
hat's place in it (`front`). Every shot also writes `<shot>_log.csv`, the flight frame by frame, which the
video's overlays read.

Trees are not colliders, so "no building in the way" does not mean the hat is unobstructed: look at the find
before using a take. Do not pass `-nographics` (the cameras must render), and keep `STITCH_SHM_SUFFIX` set:
PyUniSharingFast runs, and named sections are machine-wide. Batchmode never resumes `WaitForEndOfFrame` and
renders no camera by itself, which is why `VideoShotCapture` renders and reads the cameras in a late `LateUpdate`.
