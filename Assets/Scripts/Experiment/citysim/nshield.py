"""The neighbour shield (SwarmManager.neighbourShield, 2026-11-19): the trial scene's law with the shield off and on, on
S9's street routes and S8's head-on flights, in the replica and in Unity. Outputs experiment_results/neighbour_shield/.

    python -B -m citysim.nshield plan                     # Unity config (300 street routes + 40 head-on a row, from 22000)
    #   then run_bench.ps1 -Config <printed> -Tag nshield_bench1 -TimeoutMin 240
    python -B -m citysim.nshield gate D:/claude_obstacle_bench/runs/nshield_bench1   # score Unity, fly the replica twins
    python -B -m citysim.nshield unity D:/claude_obstacle_bench/runs/nshield_bench1  # pair events, contacts, paired speeds
    python -B -m citysim.nshield replica --name rep_street --street 150              # 600 street routes a row (4 cities)
    python -B -m citysim.nshield stats --name rep_street

The rows are bench_cands.json in the output folder (SCN: the scene's law named field by field, shield off; NSH: the same
with it on). Built on swarmshape's bench plan and gate and swarmspread's replica job, with their output folder pointed
here. The replica runs on the working tree's law; SWARM_REPLICA_ALLOW_STALE=1 while a change to it is uncommitted."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from citysim import swarmflow as F, swarmshape as SS, swarmspread as SP, swarmtune as T

OUT = Path(__file__).resolve().parents[4] / "experiment_results" / "neighbour_shield"
K_BENCH, K_REPLICA = 22000, 23000          # S6-S11 used 0-21999


def rate_ratio(x1, x0, km1, km0):
    """Rate ratio of two Poisson counts over their exposures, with the exact (conditional binomial) 95% interval."""
    res = stats.binomtest(int(x1), int(x1 + x0), km1 / (km1 + km0))
    ci = res.proportion_ci(0.95, "exact")
    conv = lambda q: (q / (1 - q)) * (km0 / km1) if q < 1 else np.inf
    return ((x1 / km1) / (x0 / km0) if x0 else np.inf), conv(ci.low), conv(ci.high), res.pvalue


def cmd_plan(a):
    SS.OUT = OUT
    SS.cmd_bench_plan(SimpleNamespace(name=a.name, cand=str(OUT / "bench_cands.json"), cross=0, pass_=0,
                                      street=a.street, face=a.face, k0=K_BENCH, drones=10, timeout=240))


def cmd_gate(a):
    SS.OUT = OUT
    SS.cmd_bench_gate(SimpleNamespace(name=a.name, runs=a.runs, jobs=a.jobs))


def cmd_unity(a):
    recs = [json.loads(line) for line in open(Path(a.run) / "results.jsonl", encoding="utf-8")]
    fl = pd.DataFrame([r for r in recs if r.get("event") == "flight"])
    fl["kind0"] = fl.kind.str.rsplit("_", n=1).str[0]
    fl["pairs"] = fl.deaths.map(lambda d: (d or {}).get("CollidingWithAnotherDrone", 0))
    fl["km"] = fl.path_len / 1000

    def summ(g):
        return pd.Series(dict(flights=len(g), km=g.km.sum(), pair_events=g.pairs.sum(),
                              contacts=g.contacts.sum(), hard=g.hard_contacts.sum(), speed=g.mean_speed.mean(),
                              hull=g.hull_frac.mean(), split=g.split_frac.mean(), min_pair_med=g.min_pair.median()))
    pd.set_option("display.width", 220)
    print(fl.groupby(["kind0", "set"]).apply(summ).round(3).to_string())
    tot = fl.groupby("set").apply(summ)
    print(tot.round(3).to_string())
    for k in ("pair_events", "contacts", "hard"):
        r = rate_ratio(tot.loc["NSH", k], tot.loc["SCN", k], tot.loc["NSH", "km"], tot.loc["SCN", "km"])
        print(f"{k}: NSH/SCN x{r[0]:.2f} (95% {r[1]:.2f}-{r[2]:.2f}), p {r[3]:.3g}")
    pv = fl.pivot_table(index="scenario", columns="set", values=["mean_speed", "hull_frac", "split_frac", "sim_s"])
    for v in ("mean_speed", "hull_frac", "split_frac", "sim_s"):
        d = (pv[(v, "NSH")] - pv[(v, "SCN")]).dropna()
        base = pv[(v, "SCN")].mean()
        p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
        print(f"{v}: paired {d.mean():+.4f} ({d.mean() / base:+.1%} of {base:.3f}), "
              f"95% {d.mean() - 1.96 * d.sem():+.4f}..{d.mean() + 1.96 * d.sem():+.4f}, Wilcoxon p {p:.2g}")


def cmd_replica(a):
    spec = F.load_cands(str(OUT / "bench_cands.json"))
    seeds = (T.TUNE_SEED,) + T.HELD_SEEDS
    cands, jobs = F.jobs_for(spec, seeds, dict(cross=0, pass_=0, street=a.street), K_REPLICA, False)
    jobs.sort(key=lambda j: (j["k"], j["seed"]))
    T.run(a.name, cands, jobs, a.jobs, f"neighbour shield {a.name}", root=OUT, fn=SP.run_job)
    cmd_stats(a)


def cmd_stats(a):
    rows = [json.loads(line) for line in open(OUT / a.name / "results.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows)
    arm = {cid: ("SCN" if i == 0 else "NSH") for i, cid in enumerate(dict.fromkeys(df.cand))}
    df["arm"] = df.cand.map(arm)
    df["events"] = df.loss_events.map(lambda e: len(e or []))
    out = {}
    for name, g in df.groupby("arm"):
        out[name] = dict(flights=len(g), km=round(g.prog.sum() / 1000, 1), loss_events=int(g.events.sum()),
                         contacts=int(g.contacts.sum()), split_eps=int(g.n_eps.sum()), v_prog=round(g.v_prog.mean(), 3),
                         v_coh=round(g.v_coh.mean(), 3), hull=round(g.hull_frac.mean(), 3),
                         min_pair_med=round(g.min_pair.median(), 2))
    print(pd.DataFrame(out).T.to_string())
    for k in ("loss_events", "contacts", "split_eps"):
        r = rate_ratio(out["NSH"][k], out["SCN"][k], out["NSH"]["km"], out["SCN"]["km"])
        print(f"{k}: NSH/SCN x{r[0]:.2f} (95% {r[1]:.2f}-{r[2]:.2f}), p {r[3]:.3g}")
    pv = df.pivot_table(index=["seed", "k"], columns="arm", values=["v_prog", "v_coh", "hull_frac"])
    for v in ("v_prog", "v_coh", "hull_frac"):
        d = (pv[(v, "NSH")] - pv[(v, "SCN")]).dropna()
        print(f"{v}: paired {d.mean():+.4f} ({d.mean() / pv[(v, 'SCN')].mean():+.1%}), "
              f"95% {d.mean() - 1.96 * d.sem():+.4f}..{d.mean() + 1.96 * d.sem():+.4f}, n {len(d)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--name", default="bench1")
    p.add_argument("--street", type=int, default=300)
    p.add_argument("--face", type=int, default=40)
    p = sub.add_parser("gate")
    p.add_argument("runs", nargs="+")
    p.add_argument("--name", default="bench1")
    p.add_argument("--jobs", type=int, default=16)
    p = sub.add_parser("unity")
    p.add_argument("run")
    p = sub.add_parser("replica")
    p.add_argument("--name", default="rep_street")
    p.add_argument("--street", type=int, default=150, help="street routes per city (4 cities)")
    p.add_argument("--jobs", type=int, default=12)
    p = sub.add_parser("stats")
    p.add_argument("--name", default="rep_street")
    a = ap.parse_args()
    dict(plan=cmd_plan, gate=cmd_gate, unity=cmd_unity, replica=cmd_replica, stats=cmd_stats)[a.cmd](a)


if __name__ == "__main__":
    main()
