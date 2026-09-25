"""Back-fill identify answers into recorded experiment runs.

Used when the experimenter's identify keys were not pressed during a run and the participant's
answers were noted by hand instead, as the hat named at each goal patch in the order the patches
were visited (e.g. "Cap, Cowboy, Bucket").

    python apply_answers.py template [--dir DIR] [--pid AABB AABC]   # write answers.csv
    python apply_answers.py apply    [--dir DIR]                     # apply answers.csv

answers.csv (';'-delimited, one row per run) has columns
    run;answer1;answer2;answer3;visitOrder (auto, do not edit)
`answerK` is the hat the participant named at the K-th patch they visited: Cap / Cowboy / Bucket
(case-insensitive, "Walker" optional), `skip` if they gave no answer there, or blank if they never
got that far. `visitOrder` is filled in by `template` from the logs so the order can be checked.

Visit order is reconstructed from the trajectories: a patch is visited when the centroid of the
alive drones first enters its tile (a square of half-width TILE_HALF about the goal patch centre).
A patch whose tile was never entered is ordered by its closest approach instead and marked `~` in
`visitOrder` -- check those by hand.

`apply` sets, per goal in _session.json: answered / outcome (correct iff the named hat is the hat on
that patch) plus the reconstruction (visitOrder, visitEnterSec, visitExitSec, visitApprox), and
recomputes nCorrect. decisionTimeSec, swarmToWalkerDist and the centroid stay unset (-1 / 0) and no
identify events are written: the moment of each answer was not recorded, and the visit times are
stored separately so they are not mistaken for it. Safe to re-run after editing answers.csv.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

DEFAULT_DIR = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\UAVS@BERKELEY\DroneSim\experiment")
TILE_HALF = 90.83 / 2   # city tile pitch / 2: the goal patch replaces one whole tile
N_ANSWERS = 3


def read_rows(path):
    # utf-8-sig: Excel (and PowerShell) save answers.csv with a BOM.
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter=";"))


def norm_hat(h):
    h = h.strip().lower()
    return h[:-len("walker")] if h.endswith("walker") else h


def centroid_track(d, stem):
    acc = defaultdict(list)
    for r in read_rows(os.path.join(d, stem + "_drones.csv"))[1:]:
        if r[7] == "1":
            acc[float(r[0])].append((float(r[3]), float(r[5])))
    return [(t, sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v))
            for t, v in sorted(acc.items())]


def visits(d, stem, goals):
    """[(goal, enterSec, exitSec, approx)] in visit order."""
    track = centroid_track(d, stem)
    out = []
    for g in goals:
        inside = [abs(x - g["goalX"]) <= TILE_HALF and abs(z - g["goalZ"]) <= TILE_HALF for _, x, z in track]
        if any(inside):
            i = inside.index(True)
            j = next((k for k in range(i, len(track)) if not inside[k]), len(track) - 1)
            out.append((g, track[i][0], track[j][0], False))
        else:
            k = min(range(len(track)),
                    key=lambda k: (track[k][1] - g["goalX"]) ** 2 + (track[k][2] - g["goalZ"]) ** 2)
            out.append((g, track[k][0], track[k][0], True))
    out.sort(key=lambda v: v[1])
    return out


def order_string(vs):
    return " > ".join(("~" if a else "") + g["hat"].replace("Walker", "") for g, _, _, a in vs)


def cmd_template(d, pids):
    out = os.path.join(d, "answers.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; delete it first if you want a fresh template")
    rows = [["run"] + [f"answer{k + 1}" for k in range(N_ANSWERS)] + ["visitOrder (auto, do not edit)"]]
    for s in sorted(glob.glob(os.path.join(d, "*_session.json"))):
        stem = os.path.basename(s)[: -len("_session.json")]
        if pids and stem.split("_")[0].upper() not in pids:
            continue
        j = json.load(open(s, encoding="utf-8"))
        rows.append([stem] + [""] * N_ANSWERS + [order_string(visits(d, stem, j["goals"]))])
    with open(out, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter=";").writerows(rows)
    print(f"wrote {out} ({len(rows) - 1} runs)")


def cmd_apply(d):
    rows = read_rows(os.path.join(d, "answers.csv"))[1:]
    for r in rows:
        if not r or not r[0].strip():
            continue
        stem = r[0].strip()
        answers = [a.strip() for a in r[1:1 + N_ANSWERS]]
        answers += [""] * (N_ANSWERS - len(answers))
        if not any(answers):
            continue
        sj = os.path.join(d, stem + "_session.json")
        if not os.path.exists(sj):
            raise SystemExit(f"{stem}: no session file (renamed or excluded since the template was made?)")
        j = json.load(open(sj, encoding="utf-8"))
        hats = {norm_hat(g["hat"]) for g in j["goals"]}
        for a in answers:
            if a and a.lower() != "skip" and norm_hat(a) not in hats:
                raise SystemExit(f"{stem}: answer '{a}' is not one of {sorted(hats)} or 'skip'")

        vs = visits(d, stem, j["goals"])
        for k, (g, enter, exit_, approx) in enumerate(vs):
            a = answers[k] if k < len(answers) else ""
            if not a:
                outcome = ""
            elif a.lower() == "skip":
                outcome = "skip"
            else:
                outcome = "correct" if norm_hat(a) == norm_hat(g["hat"]) else "incorrect"
            g.update(answered=bool(outcome), outcome=outcome, answeredHat=a,
                     decisionTimeSec=-1.0, swarmToWalkerDist=-1.0,
                     centroidAtAnswerX=0.0, centroidAtAnswerY=0.0, centroidAtAnswerZ=0.0,
                     visitOrder=k + 1, visitEnterSec=round(enter, 4), visitExitSec=round(exit_, 4),
                     visitApprox=approx)
        j["nCorrect"] = sum(g["outcome"] == "correct" for g in j["goals"])
        j["answersSource"] = "manual (apply_answers.py, matched by visit order)"
        with open(sj, "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps(j, indent=4))
        detail = ", ".join(f"{g['hat'].replace('Walker', '')}<-{g['answeredHat'] or '-'}:{g['outcome'] or '-'}"
                           for g, *_ in vs)
        print(f"{stem}: nCorrect={j['nCorrect']}  [{detail}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["template", "apply"])
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--pid", nargs="*", default=[], help="only these participants (template mode)")
    a = p.parse_args()
    if a.mode == "template":
        cmd_template(a.dir, {x.upper() for x in a.pid})
    else:
        cmd_apply(a.dir)
