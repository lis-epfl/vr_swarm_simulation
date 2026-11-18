"""The headless Unity shot list for the single-drone video: writes <take>/shots.json for VideoShotRunner.

    python Video/single_drone_instructions/shots.py D:/claude_scenic_bench/video_shots/single2 [--only avoid,hover]

then record it on the bench copy (Video/README.md, "Recording headless"), and copy the take in with
`import_takes.py`. Every shot flies ONE drone (`drones: 1`), whatever the scene's swarmSpawn says.

The landmark views and the hat close-ups are the swarm video's (footage/README.md): neither has a drone in it.
"""
import json
import os
import sys

GOAL = [-265.2, -11.5]          # the practice goal's centre building (practice layout, building 28)
# The middle of the streets round the centre building, junction to junction (12.5 m out from its walls),
# starting at the junction west of it and going round by the south.
LOOP = [[-294.6, -12.1], [-284.7, -32.4], [-241.5, -28.8], [-234.5, -9.7], [-252.9, 9.3], [-283.3, 8.3]]

# Interface camera positions (VideoCaptureRig): behind and above the drawn pilot's head. Locked to the head's yaw
# (ifaceSmooth 0): one drone's screen turns with the pilot, and a camera that trails the turn shows it swinging away.
NEAR = dict(ifaceBack=0.8, ifaceUp=0.4, ifaceFov=60, ifaceLookAhead=2.0, ifaceLookDown=0.1, ifaceSmooth=0)
ARROW = dict(ifaceBack=1.7, ifaceUp=3.2, ifaceFov=55, ifaceLookAhead=1.6, ifaceLookDown=1.0, ifaceSmooth=0)  # floor arrow


def seg(kind, dur, **kw):
    d = dict(kind=kind, dur=dur, target=[], speed=1, arrive=8, face=False, stick=[], after=3, ramp=0.5,
             gimbalTo=999, path=[], untilHat=True, approach=0)
    d.update(kw)
    return d


def shot(name, start, segs, alt=12, yaw=90, settle=6, ortho=25, iface=True, overhead=True, view=NEAR, gimbal=0, **kw):
    d = dict(name=name, start=start, alt=alt, yaw=yaw, settle=settle, orthoSize=ortho, headingUp=True, upYaw=0,
             follow=True, coneLength=12, overhead=overhead, iface=iface, goal=GOAL, gimbal=gimbal, front=False,
             centre=[], highlightHats=False, closeups=[], closeupDur=3, closeupDist=2.6, closeupHeight=1.75,
             closeupFov=32, hatRange=80, hideStartLine=False, segs=segs)
    d.update(view)
    d.update(kw)
    return d


def sticks(axis, value, push=3.5, rest=1.5):
    v = [0, 0, 0, 0]
    v[axis] = value
    w = [0, 0, 0, 0]
    w[axis] = -value
    return [seg("hold", 1.0), seg("stick", push, stick=v), seg("stick", rest, stick=[0, 0, 0, 0]),
            seg("stick", push, stick=w), seg("stick", 2.0, stick=[0, 0, 0, 0])]


def flat(points):
    return [c for p in points for c in p]


EAST_UP = dict(headingUp=False, upYaw=90)


def search(name, wait, patrol=150):
    return shot(name, [-425, 0], [seg("hold", wait),
                                  seg("goto", 30, target=[-330, -3], arrive=6, face=True),
                                  seg("goto", 15, target=LOOP[0], arrive=5, face=True),
                                  seg("patrol", patrol, path=flat(LOOP[1:] + LOOP[:1]), speed=0.7, arrive=5, face=True,
                                      after=6)],
                alt=10, gimbal=-0.15, ortho=60, centre=GOAL, front=True, hatRange=30, view=NEAR, **EAST_UP)

shots = [
    # the goal from above with the hat wearer ringed (the drone waits at the start)
    shot("goalview", [-440, 0], [seg("hold", 9)], iface=False, ortho=58, centre=GOAL, highlightHats=True, settle=4),
    # one lap round the centre building, for the tip (no stopping for the hat)
    shot("loop", LOOP[0], [seg("patrol", 45, path=flat(LOOP[1:] + LOOP[:1]), speed=0.7, arrive=5, face=True,
                               untilHat=False)],
         alt=10, yaw=150, gimbal=-0.15, ortho=60, centre=GOAL, view=NEAR, **EAST_UP),
    # the search: in along the street from the start, then round the centre building until the hat is in view
    # (within 30 m, in the middle 80% of the picture), then stop and turn the drone towards it. Three of them, setting
    # off at different moments, so the hat wearer is somewhere else on their walk each time: pick the clearest find.
    search("search", wait=2),
    search("search_b", wait=14),
    search("search_c", wait=27),
    # shorter searches (the user: two laps before the find was too long): one lap at most, setting off every 6 s, so
    # one of them meets the hat wearer early on the loop. Record them with the goalview in the same take.
    *[search(f"search_w{w}", wait=w, patrol=40) for w in range(0, 48, 6)],
    # full stick straight at a building, no steering: the drone slows and slides round it
    # (held still over the building it meets, the one with the dark roof at about (-355, -8), so the drone is seen
    # sliding round it rather than the city sliding under it)
    shot("avoid", [-425, -15.6], [seg("stick", 2, stick=[0, 0, 0, 0]),
                                  seg("goto", 30, target=[-288, -15.6], arrive=6), seg("hold", 2)],
         ortho=34, centre=[-350, -8], view=NEAR, hideStartLine=True),
    shot("hover", [-400, 0], [seg("hold", 14)], ortho=18, view=NEAR),
    # the controls, one per shot, in the open before the city so nothing gets in the way
    shot("ctl_forward", [-445, 0], sticks(1, 1.0), view=ARROW),
    shot("ctl_sideways", [-445, 0], sticks(0, 1.0), view=NEAR, **EAST_UP),
    shot("ctl_updown", [-445, 0], [seg("hold", 1.0), seg("stick", 5.0, stick=[0, 0, 1, 0]),
                                   seg("stick", 1.0, stick=[0, 0, 0, 0]), seg("stick", 4.0, stick=[0, 0, -1, 0]),
                                   seg("stick", 1.5, stick=[0, 0, 0, 0])], alt=8, view=NEAR),
    shot("ctl_yaw", [-445, 0], sticks(3, 1.0, push=3.0, rest=1.0), view=NEAR, **EAST_UP),
    shot("ctl_pitch", [-445, 0], [seg("hold", 1.0), seg("gimbal", 3.0, gimbalTo=-0.6), seg("gimbal", 1.0),
                                  seg("gimbal", 3.0, gimbalTo=0.3), seg("gimbal", 1.5, gimbalTo=0.0)], view=NEAR),
]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    chosen = [sh for sh in shots if only is None or sh["name"] in only]
    os.makedirs(out, exist_ok=True)
    cfg = dict(scene="Assets/Scenes/ScenicPracticeBlockWorld.unity", outDir=out.replace("\\", "/"),
               ffmpeg="C:/utils/ffmpeg.exe", width=1920, height=1080, fps=30, initialSettle=10, drones=1,
               shots=chosen)
    with open(os.path.join(out, "shots.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print(len(chosen), "shots ->", out)


if __name__ == "__main__":
    main()
