"""S12 Distributed Views: the page's Cities tab data (lessons 30-31).

    python -B -m citysim.views_replays [--run main_main] [--spot spot] [--jobs 12]
    -> experiment_results/views_study/replay/ (cities/, flights/, index.json)

The city is the trial city (BlockShapedPatchCityWorld's layout, seed 1). The flights are main-study sessions, each
flown in all three conditions on the same goal draw, re-flown here with a tape and kept only if the re-flight's task
time and per-goal presses equal the stored row's (the replay is the study's own flight). Every flight carries its
three goals (the building's patch and its zone), each recognition as a press where the formation was, and the start
line. The picks follow rules fixed before the main run was read (pick):

  typical   the session whose swarm / 360-drone task ratio is nearest the median of the sessions all three finished
  swarm     the session where the swarm's task time is shortest relative to the 360 drone's
  sphere    the session where the 360 drone's is shortest relative to the swarm's
  lost      the session where the swarm lost the most drones (first such goal draw), if any did

The Unity spot check (views_bench), when its gate has run, adds a group per route: Unity's own flight beside the
replica twin flown from Unity's start.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import playback as PB  # noqa: E402
from citysim import views as V  # noqa: E402

OUT = V.OUT / "replay"
DT = 0.2
LABEL = {"swarm": "Swarm", "single": "One drone", "sphere": "360 drone"}


def city_record(seed=V.CITY_SEED):
    from citysim import layout as lay
    path = lay.resolve("BlockShaped__" + V.T.params(seed).ident())
    with open(path, encoding="utf-8-sig") as f:
        d = json.load(f)
    blocks = [np.reshape(b["kerb_xz"], (-1, 2)) if len(b["kerb_xz"]) >= 6 else np.zeros((0, 2)) for b in d["blocks"]]
    flags = [(1 if b.get("interior") else 0) | (2 if b.get("building", 0) < 0 else 0) for b in d["blocks"]]
    blds = [np.reshape(b["poly_xz"], (-1, 2)) for b in d["boxes"] if b.get("type") == "ConvexPrism"]
    cid = f"views_city{seed}"
    return cid, PB.city_json(cid, list(map(float, d["layout"]["arena"])), blocks, blds, flags,
                             note="The trial city: BlockShapedPatchCityWorld's layout (seed 1).")


class Hook:
    """views.fly's tape hook: a playback.Tape on the simulation, with the swarm's headings and shown feeds from its
    feed latch (the pilot's view for one drone), and the clock offset (the swarm's settle)."""

    def __init__(self):
        self.tape, self.t_off, self.feeds = None, 0.0, None

    def __call__(self, sim, pilot, latch):
        self.t_off = float(sim.t)
        n = len(sim.P)
        if latch is not None:
            orig = latch.step

            def step(*a, **k):
                self.feeds = orig(*a, **k)
                return self.feeds
            latch.step = step
            heading = lambda: latch.yaw.copy()                                  # noqa: E731
            shown = lambda: (np.asarray(self.feeds.drones, int) if self.feeds is not None else np.zeros(0, int))  # noqa: E731
        else:
            heading = lambda: np.full(n, pilot.view)                            # noqa: E731
            shown = lambda: np.flatnonzero(sim.alive)                           # noqa: E731
        self.tape = PB.Tape(sim, DT, heading=heading, shown=shown, body=lambda: float(pilot.view))


def _same(a, b):
    def key(r):
        return (r["reason"], None if r["task_s"] is None else round(r["task_s"], 2),
                tuple((g["goal"], g["press"]) for g in r["goals"]))
    return key(a) == key(b)


def flight(cond, k, opts, stored, seed=V.CITY_SEED):
    """(flight JSON, ok) of session k flown in cond with a tape; ok if it re-flies as stored."""
    h = Hook()
    o = dict(opts, tape=h, keep_log=True)
    r = V.fly(cond, seed, k, o)
    ok = _same(r, stored)
    rec = h.tape.record()
    off = h.t_off + (r["line"] or 0.0)                  # the session's times (from the line) on the tape's clock
    geo = r["goal_geo"]
    goals, presses = [], []
    for g in r["goals"]:
        gg = geo[g["goal"]]
        found = None if g["press"] is None else round(g["press"] + off, 2)
        goals.append(dict(x=round(gg["centre"][0], 1), z=round(gg["centre"][1], 1), shape="disc",
                          half=round(g["patch"], 1), zone=V.ZONE_EXTRA_M, label=f"goal {g['goal'] + 1}", found=found))
        if found is not None:
            tt = found - off + (r["line"] or 0.0)
            lg = min(r["log"], key=lambda e: abs(e[0] - tt))
            presses.append(dict(t=found, x=lg[1], z=lg[2], outcome="correct",
                                label=f"goal {g['goal'] + 1}, recognised at {g['rec_d']:.0f} m" if g["rec_d"] else
                                f"goal {g['goal'] + 1}"))
    spawn = np.array(V.T.city(seed)["L"].layout["spawn"], float)[[0, 2]]
    hd = float(V.T.city(seed)["L"].layout.get("spawn_yaw", 90.0))
    start = dict(t=round(off, 2), x=round(float(spawn[0] + V.LINE_M * math.sin(math.radians(hd))), 1),
                 z=round(float(spawn[1] + V.LINE_M * math.cos(math.radians(hd))), 1), heading=hd)
    bands = [dict(label="at a goal (zone entry to its press)", tone="good",
                  spans=[[round(g["zone"] + off, 1), round(g["press"] + off, 1)] for g in r["goals"]
                         if g["zone"] is not None and g["press"] is not None])]
    summ = dict(time_s=r["task_s"], reason=r["reason"], lost=r["lost"], contacts=r["contacts"],
                at=[g["at_s"] for g in r["goals"]], between=[g["between_s"] for g in r["goals"]])
    fid = f"views_{cond}_{k}"
    f = PB.flight_json(rec, id=fid, title=LABEL[cond], city=f"views_city{seed}", cond=cond, source="replica",
                       source_label="the study's flight", summary=PB.rounded(summ), unit="session",
                       marks=dict(goals=goals, presses=presses, start=start), bands=bands,
                       span=300 if cond == "swarm" else 220)
    return f, ok


def _job(job):
    cond, k, opts, stored = job
    try:
        f, ok = flight(cond, k, opts, stored)
        return dict(cond=cond, k=k, f=f, ok=ok)
    except Exception as e:                            # noqa: BLE001
        return dict(cond=cond, k=k, error=repr(e), trace=traceback.format_exc()[-1500:])


def pick(run, variant=V.REC_MAIN):
    """{name: k} by the module docstring's rules."""
    import pandas as pd
    S, _, _ = V.frames(V.read(run))
    S = S[(S.variant == variant) & (S.city_seed == V.CITY_SEED)]
    w = S.pivot_table(index="k", columns="cond", values="task_s")
    lost = S[S.cond == "swarm"].set_index("k").lost
    both = w.dropna()
    rat = (both["swarm"] / both["sphere"])
    out = dict(typical=int((rat - rat.median()).abs().idxmin()), swarm=int(rat.idxmin()), sphere=int(rat.idxmax()))
    if lost.max() > 0:
        out["lost"] = int(lost[lost == lost.max()].index.min())
    return out, {k: round(float(rat.get(v, math.nan)), 2) for k, v in out.items()}, pd


ABOUT = {
    "typical": "The typical session: its swarm ÷ 360-drone task ratio is the median one.",
    "swarm": "The session where the swarm did best against the 360 drone.",
    "sphere": "The session where the 360 drone did best against the swarm.",
    "lost": "The session where the swarm lost the most drones.",
}


def cmd(a):
    from citysim import parallel
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cities").mkdir(exist_ok=True)
    (OUT / "flights").mkdir(exist_ok=True)
    cid, city = city_record()
    (OUT / "cities" / f"{cid}.json").write_text(json.dumps(city, separators=(",", ":")), encoding="utf-8")
    picks, ratios, _ = pick(a.run)
    rows = {(r["cond"], r["k"]): r for r in V.read(a.run) if r.get("variant") == V.REC_MAIN and "error" not in r}
    jobs = []
    for name, k in picks.items():
        for cond in V.CONDITIONS:
            o = dict(V.MAIN[cond], rec_model=V.REC_MAIN)
            jobs.append((cond, k, o, rows[(cond, k)]))
    res = parallel.run_jobs(_job, jobs, "views replays", processes=a.jobs)
    bad = [r for r in res if "error" in r or not r["ok"]]
    for r in bad:
        print("[views_replays] NOT the stored flight:", r.get("cond"), r.get("k"), r.get("error", ""), r.get("trace", "")[-600:])
    got = {(r["cond"], r["k"]): r["f"] for r in res if "error" not in r and r["ok"]}
    groups = []
    for name, k in picks.items():
        fl = []
        for cond in V.CONDITIONS:
            f = got.get((cond, k))
            if f is None:
                continue
            (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":")), encoding="utf-8")
            fl.append(dict(id=f["id"], cond=cond, city=cid, source="replica", title=LABEL[cond]))
        if fl:
            groups.append(dict(id=f"pick_{name}", title=f"{name.capitalize()} (goal draw {k})", short=name, kind="session",
                               first="swarm", flights=fl,
                               about=ABOUT[name] + f" Swarm ÷ 360 drone: {ratios[name]:.2f}. Every condition faced the "
                                     "same goals and hats; the swarm wraps each goal building, the 360 drone flies the "
                                     "single drone's searches with the 2K stream."))
    spot = V.OUT / a.spot / "bench" / "gate.jsonl" if a.spot else None
    if spot and spot.exists():
        groups += spot_groups(a.spot, cid, a.spot_n)
    (OUT / "index.json").write_text(json.dumps(dict(groups=groups, picks=picks, ratios=ratios), indent=1),
                                    encoding="utf-8")
    print(f"[views_replays] {len(got)} flights in {len(groups)} groups; {len(bad)} not kept")


def spot_groups(name, cid, n):
    """The spot check's first n routes per condition: Unity's flight and the replica twin from Unity's start."""
    import pandas as pd
    from citysim import tierc, views_bench as VB
    out = V.OUT / name / "bench"
    gate = pd.read_json(out / "gate.jsonl", lines=True)
    routes = json.loads((out / "routes.json").read_text(encoding="utf-8"))
    runs = json.loads((out / "runs.json").read_text(encoding="utf-8")) if (out / "runs.json").exists() else {}
    groups = []
    for k in sorted(gate.k.unique())[:n]:
        fl = []
        for cond in VB.CONDS:
            rt = routes.get(f"{cond}|{k}")
            run = runs.get(cond)
            if not rt or not run:
                continue
            cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
            idx = next((i for i, sc in enumerate(cfg["scenarios"]) if sc["kind"] == f"views{cond}_{k}"), None)
            label = cfg["paramSets"][0]["label"]
            p = Path(run) / f"traj_{label}_{idx}.csv"
            if idx is None or not p.exists():
                continue
            tr = tierc.read_traj(p)
            u = PB.unity_tape(tr, DT, single=cond != "swarm")
            fu = PB.flight_json(u, id=f"spot_unity_{cond}_{k}", title=f"{LABEL[cond]}, Unity", city=cid, cond=cond,
                                source="unity", source_label="Unity (headless bench)", unit="route")
            P0 = np.c_[tr["X"][0], tr["Z"][0]][tr["alive"][0]]
            m = V.T.model({})
            C = V.T.city(V.CITY_SEED)
            sim = V.rep.ContactSim(m, C["rcity"], np.array(P0, float), np.zeros((len(P0), 2)), 7000 + int(k))
            tape = PB.Tape(sim, DT, heading=lambda s=sim: np.zeros(len(s.P)), shown=lambda s=sim: np.flatnonzero(s.alive))
            orig = V.rep.ContactSim
            try:                                       # the twin, taped: views_bench.fly_twin on this sim
                V.rep.ContactSim = lambda *a_, **k_: sim
                VB.fly_twin(m, C, P0, rt["wps"], V.D_REF, 7000 + int(k))
            finally:
                V.rep.ContactSim = orig
            fr = PB.flight_json(tape.record(), id=f"spot_twin_{cond}_{k}", title=f"{LABEL[cond]}, replica twin",
                                city=cid, cond=cond, source="replica", source_label="the replica twin", unit="route")
            for f in (fu, fr):
                (OUT / "flights" / f"{f['id']}.json").write_text(json.dumps(f, separators=(",", ":")), encoding="utf-8")
                fl.append(dict(id=f["id"], cond=cond, city=cid, source=f["source"], title=f["title"]))
        if fl:
            groups.append(dict(id=f"spot_{k}", title=f"Unity spot check, route {k}", short=f"Unity {k}", kind="session",
                               first="swarm", flights=fl,
                               about="One of the study's routes flown by Unity on the headless bench and by the replica "
                                     "from Unity's start: the swarm wrapped round each goal building, the single "
                                     "drone's searches (which the 360 drone also flies)."))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="main_main")
    ap.add_argument("--spot", default="spot")
    ap.add_argument("--spot-n", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    cmd(ap.parse_args())


if __name__ == "__main__":
    main()
