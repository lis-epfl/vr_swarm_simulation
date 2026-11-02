"""S9 Street Spread, the follow-up: are the adaptive spread's losses the narrowing's, or the place's?

    python -B -m citysim.swarmspread_temporal replica --name run2
    python -B -m citysim.swarmspread_temporal unity --runs swarmspread_bench_spread swarmspread_bench_spread2
        -> experiment_results/swarm_spread/<name>/temporal.json, unity_temporal.json

The spread only changes near buildings, so a loss "during a change" can be the change's or the place's. run2 logs the
adaptive spread's shadow in every row (swarmspread.ShadowLog): what the adaptive spread would be doing at that moment,
computed from the swarm's own centroid, stick and buildings. In an adaptive row it is the spread itself; in a fixed row
it marks the same moments without the spread moving. Each moment of every flight, and each loss, is placed in a phase
of the shadow (swarmspread.sh_phase: narrowing, widening, held narrow, held wide), with the time spent in each phase
beside it, so a loss rate per phase can be compared row against row. If the adaptive spread's extra losses are the
narrowing's, its rate exceeds the fixed rows' in the narrowing phase and not elsewhere; if they are the place's, every
row's rate is raised there alike.

Tests: per phase, the exact conditional binomial on two rows' counts given their exposures (one-sided, the first row
loses more); and the likelihood-ratio test that one rate ratio holds in every phase (df = phases - 1), the ratio fitted
by maximum likelihood on the conditional binomials. Profiles: the spread and the shadow from 15 s before each loss to
3 s after, against the same round N_CONTROL random moments of every flight.

Unity has no shadow log, so `unity` rebuilds it from Unity's own trajectories (0.1 s rows): the centroid of the live
drones, the bench pilot's stick towards the current waypoint, the replica's spread_target on the scene's city with the
set's recorded SwarmManager values, low-passed at its filter; a loss is a drop in the alive mask. Approximate (Unity
queries buildings through PhysX), and it places so few losses that it is a check on the direction, not a test.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import swarmspread as X, swarmtune as T  # noqa: E402

PHASES = X.SH_PHASES
FIXED, ADAPT = ("F20", "F18", "F17"), ("A85", "A90", "A95")
RUNS = Path(r"D:\claude_obstacle_bench\runs")


def binom_worse(k, k0, e, e0):
    """One-sided exact p that k events over exposure e are no more frequent than k0 over e0."""
    from scipy.stats import binom
    n = k + k0
    if n == 0 or e <= 0 or e0 <= 0:
        return 1.0
    return float(binom.sf(k - 1, n, e / (e + e0)))


def lr_homogeneity(strata):
    """strata [(k_a, k_b, e_a, e_b)]: the likelihood-ratio test that one rate ratio a/b holds in every stratum.
    -> (common ratio, per-stratum ratios, LR statistic, df, p)."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import chi2
    S = [s for s in strata if s[0] + s[1] > 0 and s[2] > 0 and s[3] > 0]

    def ll(lth, s):
        ka, kb, ea, eb = s
        th = math.exp(lth)
        p = th * ea / (th * ea + eb)
        return ka * math.log(max(p, 1e-300)) + kb * math.log(max(1 - p, 1e-300))

    common = minimize_scalar(lambda x: -sum(ll(x, s) for s in S), bounds=(-8, 8), method="bounded").x
    l0 = sum(ll(common, s) for s in S)
    l1, per = 0.0, []
    for s in S:
        ka, kb, ea, eb = s
        if ka == 0 or kb == 0:            # the saturated fit puts all the mass on one side
            l1 += 0.0
            per.append(0.0 if ka == 0 else math.inf)
        else:
            th = (ka / ea) / (kb / eb)
            l1 += ll(math.log(th), s)
            per.append(th)
    lr = max(0.0, 2 * (l1 - l0))
    df = max(1, len(S) - 1)
    return math.exp(common), per, lr, df, float(chi2.sf(lr, df))


def poisson_ci(k):
    return X.poisson_ci(k)


def collect(name):
    """Per row: exposure (s) and loss events per phase, the losses' windows and the controls' windows."""
    out_dir = X.OUT / name
    cands = json.loads((out_dir / "cands.json").read_text(encoding="utf-8"))
    code = {c: v["label"].split()[0] for c, v in cands.items()}
    rows = {}
    for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or "sh_exposure" not in r:
            continue
        c = code[r["cand"]]
        o = rows.setdefault(c, dict(exp={p: 0.0 for p in PHASES}, ev={p: 0 for p in PHASES}, flights=0,
                                    flight_s=0.0, ev_win_sig=[], ev_win_sh=[], ctl_win_sig=[], ctl_win_sh=[],
                                    ev_since=[], since_exp={}, since_ev={}, events=[]))
        o["flights"] += 1
        o["flight_s"] += r["flight_s"]
        for p, v in r["sh_exposure"].items():
            o["exp"][p] += v
        for k, v in (r.get("exposure") or {}).items():
            o["since_exp"][k] = o["since_exp"].get(k, 0.0) + v
        for e in r.get("loss_events") or []:
            o["ev"][e["sh_phase"]] += 1
            o["since_ev"][e["phase"]] = o["since_ev"].get(e["phase"], 0) + 1
            o["ev_win_sig"].append(e["win_sig"])
            o["ev_win_sh"].append(e["win_sh"])
            o["events"].append(dict(kind=r["kind"], seed=r["seed"], k=r["k"], t=e["t"], sh_phase=e["sh_phase"],
                                    phase=e["phase"], since=e["since"], sig=e["sig"], sh=e["sh"]))
        for c2 in r.get("controls") or []:
            o["ctl_win_sig"].append(c2["win_sig"])
            o["ctl_win_sh"].append(c2["win_sh"])
    return rows


def profile(wins):
    if not wins:
        return None
    a = np.array(wins, float)
    return dict(n=len(a), mean=[round(float(x), 4) for x in a.mean(0)],
                lo=[round(float(x), 4) for x in np.percentile(a, 25, axis=0)],
                hi=[round(float(x), 4) for x in np.percentile(a, 75, axis=0)])


def summarise(rows, fixed=FIXED, adapt=ADAPT):
    out = dict(phases=list(PHASES), rows={}, compare={}, pooled={})
    for c, o in rows.items():
        hz = {}
        for p in PHASES:
            k, e = o["ev"][p], o["exp"][p]
            lo, hi = poisson_ci(k)
            hz[p] = dict(events=k, hours=e / 3600, per_hour=k / (e / 3600) if e > 0 else None,
                         lo=lo / (e / 3600) if e > 0 else None, hi=hi / (e / 3600) if e > 0 else None,
                         share=e / max(sum(o["exp"].values()), 1e-9))
        k, e = sum(o["ev"].values()), sum(o["exp"].values())
        out["rows"][c] = dict(flights=o["flights"], hours=e / 3600, events=k, per_hour=k / (e / 3600) if e else None,
                              phase=hz, since_exp=o["since_exp"], since_ev=o["since_ev"],
                              prof_ev_sig=profile(o["ev_win_sig"]), prof_ev_sh=profile(o["ev_win_sh"]),
                              prof_ctl_sig=profile(o["ctl_win_sig"]), prof_ctl_sh=profile(o["ctl_win_sh"]),
                              events_list=o["events"])

    def compare(a_rows, b_rows):
        A = [rows[c] for c in a_rows if c in rows]
        B = [rows[c] for c in b_rows if c in rows]
        if not A or not B:
            return None
        ka = {p: sum(o["ev"][p] for o in A) for p in PHASES}
        kb = {p: sum(o["ev"][p] for o in B) for p in PHASES}
        ea = {p: sum(o["exp"][p] for o in A) for p in PHASES}
        eb = {p: sum(o["exp"][p] for o in B) for p in PHASES}
        per = {p: dict(ka=ka[p], kb=kb[p], ratio=(ka[p] / ea[p]) / (kb[p] / eb[p]) if kb[p] and ea[p] and eb[p] else None,
                       p_worse=binom_worse(ka[p], kb[p], ea[p], eb[p])) for p in PHASES}
        tot = binom_worse(sum(ka.values()), sum(kb.values()), sum(ea.values()), sum(eb.values()))
        common, _, lr, df, p_lr = lr_homogeneity([(ka[p], kb[p], ea[p], eb[p]) for p in PHASES])
        # the narrowing phase against the rest
        rest = [p for p in PHASES if p != "narrowing"]
        s2 = [(ka["narrowing"], kb["narrowing"], ea["narrowing"], eb["narrowing"]),
              (sum(ka[p] for p in rest), sum(kb[p] for p in rest), sum(ea[p] for p in rest), sum(eb[p] for p in rest))]
        c2, per2, lr2, df2, p2 = lr_homogeneity(s2)
        return dict(a=list(a_rows), b=list(b_rows), per_phase=per, p_worse_all=tot,
                    ratio_all=(sum(ka.values()) / sum(ea.values())) / (sum(kb.values()) / sum(eb.values()))
                    if sum(kb.values()) else None,
                    homogeneity=dict(common=common, lr=lr, df=df, p=p_lr),
                    narrowing_vs_rest=dict(ratio_narrowing=per2[0] if per2 else None,
                                           ratio_rest=per2[1] if len(per2) > 1 else None, lr=lr2, p=p2))

    for a in rows:
        if a != "F20":
            out["compare"][f"{a}>F20"] = compare([a], ["F20"])
    for a, b in (("A85", "F17"), ("A90", "F18"), ("A85", "F18")):
        out["compare"][f"{a}>{b}"] = compare([a], [b])
    out["pooled"]["adaptive>fixed"] = compare(adapt, fixed)
    out["pooled"]["A85+A90>F20+F18+F17"] = compare(("A85", "A90"), fixed)
    return out


def cmd_replica(a):
    rows = collect(a.name)
    out = summarise(rows)
    out["window"] = dict(before=X.WIN_BEFORE, after=X.WIN_AFTER, dt=X.WIN_DT, trans_s=X.TRANS_S, ds=X.TRANS_DS,
                         shadow_factor=X.SHADOW_FACTOR)
    (X.OUT / a.name / "temporal.json").write_text(json.dumps(T_clean(out), indent=1), encoding="utf-8")
    report(out)


def T_clean(x):
    from citysim.swarmtune_report import _clean
    return _clean(x)


def report(out):
    print("loss events per hour of flight, by the shadow's phase (events / hours):")
    head = f"{'row':5s} {'all':>14s} " + " ".join(f"{p:>20s}" for p in PHASES)
    print(head)
    for c in ("F20", "F18", "F17", "A95", "A90", "A85", "L85", "F14"):
        r = out["rows"].get(c)
        if not r:
            continue
        cells = " ".join(f"{(r['phase'][p]['per_hour'] or 0):6.2f} ({r['phase'][p]['events']:3d}/{r['phase'][p]['hours']:5.1f}h)"
                         for p in PHASES)
        print(f"{c:5s} {r['per_hour'] or 0:6.2f} ({r['events']:3d})  {cells}")
    print("time share by phase:", {c: {p: round(r["phase"][p]["share"], 3) for p in PHASES} for c, r in out["rows"].items()})
    for k, v in list(out["compare"].items()) + list(out["pooled"].items()):
        if not v:
            continue
        pp = {p: f"{v['per_phase'][p]['ka']}/{v['per_phase'][p]['kb']} x{(v['per_phase'][p]['ratio'] or 0):.2f} p{v['per_phase'][p]['p_worse']:.3f}" for p in PHASES}
        nv = v["narrowing_vs_rest"]
        print(f"{k:22s} all x{(v['ratio_all'] or 0):.2f} p{v['p_worse_all']:.4f} | {pp} | homogeneity p {v['homogeneity']['p']:.3f}"
              f" | narrowing x{(nv['ratio_narrowing'] or 0):.2f} vs rest x{(nv['ratio_rest'] or 0):.2f}, p {nv['p']:.3f}")


# ============================================================================= Unity, rebuilt from its trajectories

def unity_flight(m, C, sc, tr):
    """(phase exposure s, [(loss phase, t)]) for one Unity flight: the shadow rebuilt from the trajectory."""
    rep = T.rep
    rc = C["rcity"]
    w = np.array(sc["wps"], float).reshape(-1, 4)
    t, al, Xs, Zs, wpi = tr["t"], tr["alive"], tr["X"], tr["Z"], tr["wp"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    sh, hist, exp, ev = None, [], {p: 0.0 for p in PHASES}, []
    prev_alive = None
    for i in range(len(t)):
        a = al[i]
        if not a.any():
            break
        P = np.c_[Xs[i], Zs[i]][a]
        cen = P.mean(0)
        k = int(wpi[i])
        cmd = np.zeros(2)
        if k < len(w) and w[k][2] > 0:
            dv = w[k][:2] - cen
            cmd = dv / max(float(np.linalg.norm(dv)), 1e-9) * w[k][2]
        cand = rc.near(P, m.r0_obs * m.S + rep.QUERY_MARGIN_M)
        miss = rep.aim_miss(m, rc, cand, cen, m.stick_to_world(cmd, 0.0))
        street = rep.spread_target(m, rc, cand, cen, miss) < 1.0
        tg = X.SHADOW_FACTOR if street else 1.0
        sh = tg if sh is None else sh + (1 - math.exp(-dt / m.spread_filter)) * (tg - sh)
        hist.append(sh)
        j = max(0, i - int(round(X.TRANS_S / dt)))
        ph = X.sh_phase(sh, hist[j])
        exp[ph] += dt
        if prev_alive is not None and (prev_alive & ~a).any():
            ev.append((ph, float(t[i]), int((prev_alive & ~a).sum())))
        prev_alive = a
    return exp, ev


def cmd_unity(a):
    from citysim import tierc, use_bench_modules
    use_bench_modules()
    import bench
    C = T.city(T.TUNE_SEED)
    rows = {}
    for rn in a.runs:
        run = RUNS / rn
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        _, _, params, flights, _, _ = bench.read_results(run)
        for ps in cfg["paramSets"]:
            lab = ps["label"]
            eff = {k: float(f"{v:.7g}") if isinstance(v, float) else v for k, v in params[lab]["effective"].items()}
            over = dict(eff)
            over.pop("d_ref", None)
            m = T.model(over)
            o = rows.setdefault(lab, dict(exp={p: 0.0 for p in PHASES}, ev={p: 0 for p in PHASES}, flights=0,
                                          drones=0))
            for idx, sc in enumerate(cfg["scenarios"]):
                if not sc["kind"].startswith("street"):
                    continue
                f = run / f"traj_{lab}_{idx}.csv"
                if not f.exists():
                    continue
                e, ev = unity_flight(m, C, sc, tierc.read_traj(f))
                o["flights"] += 1
                for p in PHASES:
                    o["exp"][p] += e[p]
                for ph, _, n in ev:
                    o["ev"][ph] += 1
                    o["drones"] += n
    out = dict(runs=a.runs, rows={})
    for c, o in rows.items():
        out["rows"][c] = dict(flights=o["flights"], drones=o["drones"], events=sum(o["ev"].values()),
                              hours=sum(o["exp"].values()) / 3600,
                              phase={p: dict(events=o["ev"][p], hours=o["exp"][p] / 3600,
                                             share=o["exp"][p] / max(sum(o["exp"].values()), 1e-9)) for p in PHASES})
    (X.OUT / "unity_temporal.json").write_text(json.dumps(T_clean(out), indent=1), encoding="utf-8")
    for c, r in out["rows"].items():
        print(c, r["flights"], "flights", r["events"], "loss moments", {p: (r["phase"][p]["events"], round(r["phase"][p]["share"], 3)) for p in PHASES})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replica")
    p.add_argument("--name", default="run2")
    p = sub.add_parser("unity")
    p.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    dict(replica=cmd_replica, unity=cmd_unity)[a.cmd](a)


if __name__ == "__main__":
    main()
