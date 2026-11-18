"""Copy a headless take (VideoShotRunner's output folder) into footage/, under the names script.yaml uses.

    python Video/participant_instructions/import_takes.py D:/claude_scenic_bench/video_shots/take5

Every <shot>_<camera>.mp4, <shot>_log.csv (sticks, gimbal, height, swarm size, overhead camera, every drone,
frame by frame), <shot>_front_hat.csv and events.txt (when the hat came in front, what each close-up shows) is
copied as it is named. Old footage of the same names is replaced; anything else in footage/ is left alone.
Trim in script.yaml with a cue's `start` / `end` / `speed`.
"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    take = sys.argv[1]
    out = os.path.join(HERE, "footage")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(take, "*.mp4")) + glob.glob(os.path.join(take, "*.csv")) +
                   glob.glob(os.path.join(take, "events.txt")))
    for p in files:
        if p.endswith(".mp4") and os.path.getsize(p) < 1000:
            print(f"skip {os.path.basename(p)} (empty)")
            continue
        shutil.copy2(p, os.path.join(out, os.path.basename(p)))
        print(f"{os.path.basename(p)}")


if __name__ == "__main__":
    main()
