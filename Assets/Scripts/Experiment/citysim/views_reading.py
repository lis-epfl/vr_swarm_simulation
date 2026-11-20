"""S12 Distributed Views: the author's reading of the main study, for the results page (citysim/views_results.py).

Written on 2026-11-18 after main_main, main_probe, main_sens, main_held, main_calib and the spot check were read. The
prose is the interpretation; every number in it is read from the assembled results (`A`, views_results.assemble), so a
rebuild after a re-run cannot leave a stale figure in the text. If a re-run moves a number past a statement's premise
(a verdict changes, the cube sensitivity stops tying), the statement itself has to be revisited by hand.
"""
import math

import numpy as np
import pandas as pd

from citysim import views as V
from citysim import views_report as R

PRIMARY = V.REC_MAIN
OTHER = "start" if PRIMARY == "pooled" else "pooled"


def _r(x, nd=2):
    return f"{x['est']:.{nd}f} ({x['lo']:.{nd}f}–{x['hi']:.{nd}f})"


def _e(x, nd=2):
    return f"{x['est']:.{nd}f}"


def _sens(A, prefix):
    return next(r for r in A["S"]["rows"] if r["label"].startswith(prefix))


def _people(A):
    """internal_4's task times and at-goal ratio (unpaired geometric means over runs), cached on A."""
    if "_people" in A:
        return A["_people"]
    t = pd.read_csv(V.REPO / "experiment_results" / "trials" / "internal_4" / "tables" / "runs.csv", sep=";")
    gm = t.groupby("condition").time_s.apply(lambda x: float(np.exp(np.log(x).mean())))
    human = pd.read_csv(V.HUMAN / "goal_times.csv")
    at = R.people_ratio_ci(human, "swarm", "single", "at_s")
    A["_people"] = dict(swarm=gm.get("Swarm"), single=gm.get("SingleDrone"), sphere=gm.get("SingleSphere"),
                        at_single=at, n_runs=len(t))
    return A["_people"]


def _pilots(A, ratio, runs=2):
    """Pilots for 80% power at a true ratio (either way round), with internal_4's within-pilot spread."""
    ratio = max(ratio, 1 / ratio)
    for r in R.power(sd_w=A["Pw"]["sd_w"], sd_int=0.30, runs=(runs,)):
        return next((n for rr, n in r["pts"] if rr >= ratio - 1e-9), None)


def answer(A):
    s = A["M"]["rat"][PRIMARY]
    cube = _sens(A, "the 360 drone shows the cube")
    pp = _people(A)
    done = A["M"]["rows"][PRIMARY]["sphere"]["done"]
    return [
        f"<b>The swarm beats one drone: it takes {_e(s['single'])}× the time</b>, all of the gain at the goal. "
        f"internal_4's pilots measured {pp['swarm'] / pp['single']:.2f}×.",
        f"<b>Against a 360 drone shown at full resolution, the swarm only ties ({_e(cube['sphere'])}×).</b> Seeing every "
        f"way from one place is as good as ten spread-out views, provided the picture is sharp.",
        f"<b>The swarm's large lead over the 360 drone ({_e(s['sphere'])}×) comes from resolution, not from the "
        f"viewpoints.</b> The main study showed that drone through a 2K live stream, about a third of a feed screen's "
        f"pixels per degree. Its pilot could not make out a hat until close to it, and it finished the task in "
        f"{done:.0%} of sessions.",
    ]


def short(A):
    return ("So the 360 comparison measures the display: the swarm beats a blurry 360 view and ties a sharp one. Nobody "
            "has flown the 2K stream, so the human trial will be decided by how sharp a real 360 drone's live view is.")


def long(A):
    M, P, H, Pw, C, Sp = A["M"], A["P"], A["H"], A["Pw"], A["C"], A["Sp"]
    s, so = M["rat"][PRIMARY], M["rat"][OTHER]
    rp = M["rows"][PRIMARY]
    pr, prr = P["by"][PRIMARY]["rates"], P["by"][PRIMARY]["rat"]
    cube = _sens(A, "the 360 drone shows the cube")
    soft = _sens(A, "the stream costs half")
    crowd = _sens(A, "the stream scales the hat but not the crowd")
    fs = V.sphere_factor()
    others = [r for r in A["S"]["rows"] if r["sphere"] and r is not cube]
    lo_s = min(r["sphere"]["est"] for r in others)
    hi_s = max(r["sphere"]["est"] for r in others)
    sing = [r["single"]["est"] for r in A["S"]["rows"] if r.get("single")]
    held_s = [h["sphere"]["est"] for h in H if h["sphere"]]
    held_g = [h["single"]["est"] for h in H if h["single"]]
    held_hi = max(max(h["sphere"]["hi"] for h in H if h["sphere"]), max(h["single"]["hi"] for h in H if h["single"]))
    pp = _people(A)
    at_p = pp["at_single"]
    n_single = _pilots(A, 1 / s["single"]["est"])
    n_sphere = Pw["n_pred"]
    agree = Sp.get("agree") or {}
    spot_ok = bool(agree) and all(v is not False for d in agree.values() for v in d.values())
    out = [
        f"<b>Both verdicts are the swarm's, on either credit.</b> Press-credited, swarm ÷ 360 drone {_r(s['sphere'])} and "
        f"swarm ÷ one drone {_r(s['single'])}; first sight, {_r(so['sphere'])} and {_r(so['single'])}. The 360 drone "
        f"finished within the {V.CAP_S:.0f} s cap in only {rp['sphere']['done']:.0%} of sessions (the swarm "
        f"{rp['swarm']['done']:.0%}, one drone {rp['single']['done']:.0%}), so its restricted mean sits against the cap "
        f"and the ratio against it is, if anything, kind to it.",

        f"<b>The swarm's lead is at the goal and nowhere else.</b> Per goal at the goal the swarm takes "
        f"{rp['swarm']['at']['est']:.0f} s, one drone {rp['single']['at']['est']:.0f} s, the 360 drone "
        f"{rp['sphere']['at']['est']:.0f} s. Between goals the swarm is level with the 360 drone "
        f"({_e(s['sphere']['between'])}×) and slower than one drone ({_e(s['single']['between'])}×): ten drones fly "
        f"no faster than one, and they notice a crowd about as far off ({rp['swarm']['notice_d']:.0f} m median against "
        f"{rp['single']['notice_d']:.0f} m).",

        f"<b>It is the spread of the viewpoints, not the number of feeds.</b> At a goal the wrapped swarm recognises at "
        f"{pr['swarm']['cf_real']['est']:.2f} hats a minute. The same ten feeds moved to the formation's centre manage "
        f"{pr['swarm']['cf_centre']['est']:.2f} ({_e(prr['spread_vs_centre'], 0)}× fewer; partly because the centre of a "
        f"wrapped swarm is inside the building), and a 360 camera on its most central drone, at the swarm's height, "
        f"{pr['swarm']['cf_sphere_h']['est']:.3f} ({_e(prr['spread_vs_360_low'], 0)}× fewer). Round a building the "
        f"drones stand on every side of it, close to whoever is walking there; one camera, however wide, stands on one "
        f"side.",

        f"<b>Why the stream does so badly.</b> A 2K equirectangular frame spreads 2048 pixels over 360°, "
        f"{2048 / 360:.1f} per degree against the feed screens' 17.7 reaching the eye, so a hat counts {fs:.1f}× as far "
        f"away. The hazard is steep (k {A['rec']['d_head|lat1.0|all']['fit']['k']:.1f}), so a hat 20 m off, an easy "
        f"recognition on a feed, becomes one at {20 * fs:.0f} m, almost none. The 360 drone's recognitions came at a "
        f"median {rp['sphere']['rec_d']:.1f} m against one drone's {rp['single']['rec_d']:.1f} m: it had to fly up to "
        f"each hat. On its own flights a forward camera would have recognised "
        f"{1 / prr['sphere_vs_fwd']['est']:.0f}× as often; on one drone's flights a 2K 360 camera would have recognised "
        f"{prr['single_360']['est']:.2f}× as often as the forward camera did.",

        f"<b>The resolution is the whole question about the 360 drone.</b> Every sensitivity but the display leaves swarm ÷ "
        f"360 drone between {lo_s:.2f} and {hi_s:.2f} (the hat's distance ×0.75 or ×1.33, the crowd's at 200 or 400 m, "
        f"the walk without exploring, the 360 drone flown SIMI's way or at SIMI's altitudes, the swarm circling instead "
        f"of wrapping). Halve the stream's cost and leave the crowd unscaled: {_r(soft['sphere'])}, still the swarm's. "
        f"Show the cube SIMI flew: {_r(cube['sphere'])}, equivalent within 10%, with the 360 drone then "
        f"{_e(cube['cam'])}× one drone's time, a better camera than the forward one. Scaling the hat but not the crowd "
        f"changes little ({_e(crowd['sphere'])}), so it is recognising the hat, not noticing the crowd, that the "
        f"stream loses.",

        f"<b>Against one drone the answer does not move.</b> Swarm ÷ one drone stays between {min(sing):.2f} and "
        f"{max(sing):.2f} across the sensitivities, and in the three held-out cities (per city, never pooled) it is "
        f"{', '.join(f'{x:.2f}' for x in held_g)}, with swarm ÷ 360 drone {', '.join(f'{x:.2f}' for x in held_s)}; "
        f"every interval's upper end is at most {held_hi:.2f}.",

        f"<b>The model is slower than the people, between goals.</b> internal_4's geometric-mean task times were "
        f"{pp['swarm']:.0f} s with the swarm and {pp['single']:.0f} s with one drone; the model's restricted means are "
        f"{rp['swarm']['task']['est']:.0f} s and {rp['single']['task']['est']:.0f} s. The calibration gate (passed "
        f"{C['passes'].get(PRIMARY + '_wrap', 0)} of 8, press-credited) placed the gap between goals: the scripted pilot "
        f"wanders where people head for the next crowd. The ratios are what the model is trusted for, and they agree: "
        f"swarm ÷ one drone {_e(s['single'])} against the people's {pp['swarm'] / pp['single']:.2f}, at the goal "
        f"{_e(s['single']['at'])} against {at_p[0]:.2f} ({at_p[1]:.2f}–{at_p[2]:.2f}).",

        f"<b>The swarm pays for it in drones.</b> It lost {rp['swarm']['lost']} drones and touched buildings "
        f"{rp['swarm']['contacts']} times over {M['n']} sessions ({rp['swarm']['lost'] / M['n']:.2f} drones a session), "
        f"no session ending under four drones. "
        + ("Unity's headless bench agreed with the replica on route time, wrapping and losses (the spot check)."
           if spot_ok else "Unity's spot check did not agree on every rule; see the details tab."),

        f"<b>What the human retest should expect, and fly.</b> Swarm ÷ one drone near {_e(s['single'])}: with two runs "
        f"per condition and internal_4's run-to-run spread (SD {Pw['sd_w']:.2f} in log time), about {n_single} pilots "
        f"for 80% power. Swarm ÷ the 360 stream near {_e(s['sphere'])}: about {n_sphere} pilots. But the stream's "
        f"cost is this study's one unmeasured input, and at the cube's resolution the answer flips to a tie, which no "
        f"feasible number of pilots could separate from 10%. So the retest should fly the 360 drone on the stream, as "
        f"the sphere now shows it, and the first thing to read from it is the distance at which pilots name a hat through "
        f"the stream: about {A['rec']['d_head|lat1.0|all']['fit']['d50'] / fs:.0f} m at half-rate if this model is right, "
        f"{A['rec']['d_head|lat1.0|all']['fit']['d50']:.0f} m if the stream costs nothing.",
    ]
    return out


READING = dict(answer=answer, short=short, long=long)
