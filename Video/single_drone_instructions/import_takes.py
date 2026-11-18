"""Copy a headless take (VideoShotRunner's output folder) into footage/, under the names script.yaml uses.

    python Video/single_drone_instructions/import_takes.py D:/claude_scenic_bench/video_shots/single1

Every <shot>_<camera>.mp4, <shot>_log.csv, <shot>_front_hat.csv and events.txt is copied as it is named, and
then the swarm video's clips that show no drone (the landmark views and the hat close-ups, SHARED below) are
copied from ../participant_instructions/footage, so both videos show the same ones. Old footage of the same
names is replaced; anything else in footage/ is left alone.
"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM_FOOTAGE = os.path.join(os.path.dirname(HERE), "participant_instructions", "footage")
SHARED = ["landmarks_views.mp4", "landmarks_log.csv", "hats_closeup.mp4", "hats_log.csv"]


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
        print(os.path.basename(p))
    for name in SHARED:
        src = os.path.join(SWARM_FOOTAGE, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, name))
            print(f"{name} (from the swarm video)")
        else:
            print(f"missing {name}: not in {SWARM_FOOTAGE}")


if __name__ == "__main__":
    main()
