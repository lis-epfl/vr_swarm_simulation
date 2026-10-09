"""Horizontal-plane replica of the swarm, for fast A/B tests on recorded pilot input.

    python swarm_replica.py sim    --test internal_3 [--variant "core off"] [--set c_damp=0] [--seeds 3]
    python swarm_replica.py forces --test internal_3 [--crash RUN@T[:i,j]]
    python swarm_replica.py step   [--scene ScaledCityWorld] [--variant ...]
    python swarm_replica.py params --test internal_3 [--run STEM]

  sim     closed loop. Each recorded swarm run's pilot inputs (stick, body yaw, spread) drive a simulated swarm
          from that run's take-off state through that run's city -- the edit-mode city with the run's goal
          patches swapped in, as GoalPatchReplacer did at runtime. Reports drone-drone and building kills per
          minute, split at the tightest spread (d_ref <= 0.45), plus spacing, hull fraction and speed.
  forces  open loop. The force law evaluated on the RECORDED states before each drone-drone crash in
          results/crashes.csv (`analyse.py run` writes it), attributing the pair's closing acceleration to
          each term. Positive = pushes the pair together.
  step    spread-stick step response at hover in open sky (the spread-response requirement: any retune must
          keep the swarm following d_ref quickly and stably), then a cruise / stop / reversal block at d_ref 0.4.
  params  the resolved parameters and where each one came from.

Run with the `stitching` env. Output goes to <test>/results/replica/ (or --out).

WHAT IT MIRRORS: OlfatiSaber.GetSwarmAcceleration (cohesion, velocity consensus, close-range damper, obstacle
beta-agents, hollow core, the shield's projection), SwarmPlaneController's core state (centroid, mean
velocity, filtered radius), VelocityControl's horizontal loop (stick resolution, shield, velocity P-loop,
swarm-force low-pass, circular tilt clamp, tilt -> rate -> alpha cascade, linear and angular drag) and
StateFinder's noise -- AS OF FORCE_LAW_COMMIT. Every number comes from the C# initialisers, the scene, the
prefab or the experiment's own records (swarm_params.py); the STRUCTURE of the law is written out here, so it
goes stale the next time the C# changes. sim, forces and step check git for changes to FORCE_LAW_FILES since
FORCE_LAW_COMMIT and refuse (exit 3) unless --allow-stale (params just reports it): review the diff, bring
this file into line, then bump the constant.

WHAT IT DOES NOT MODEL -- check before trusting a result that depends on one of these:
  - Heading/yaw. No hull heading rule, no look-direction gap fill, no FPV view. Heading does not feed
    translation, so kills and spacing are unaffected; anything about what the pilot sees is out of reach.
  - Altitude. No climb or descent, no altitude ceiling, no MinHeight floor, no plane mode. In `sim` every
    drone is taken to be within every building's height (obstacles are queried by footprint). `forces` uses
    the recorded 3D positions, so its obstacle query does see altitude.
  - Street furniture. Obstacles are the Obstacle-layer buildings in city_obstacles_<city>.json plus each goal
    patch's nine buildings. Lamps, poles and props sit on Default -- invisible to the swarm in Unity too, but
    here they cannot be HIT either: internal_3's street-light crash cannot happen in the replica.
  - PhysX contact. A drone dies on entering a building footprint (+0.25 m) or coming within 0.5 m of another.
    No bounce, no friction pinning against a facade (which in Unity often ends as TooFarFromSwarm).
  - DroneHealthMonitor. No TooFarFromSwarm / Crashed / stuck rules: a straggler simply flies on.
  - The shield uses this tick's obstacle frames; in Unity they can be one tick stale (execution order).
For building contacts and anything physical, use the headless Unity bench (run_bench.ps1, README.md).
"""
import sys

sys.dont_write_bytecode = True   # nothing under Assets/ may grow a __pycache__ for Unity to import

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import swarm_params as sp  # noqa: E402

sys.path.insert(0, str(sp.EXPERIMENT_DIR))
import analyse  # noqa: E402  (scene_boxes: the city as a run saw it, goal tiles swapped in)

FORCE_LAW_COMMIT = "fe3a448b"
FORCE_LAW_FILES = [
    "Assets/Scripts/swarm/OlfatiSaber.cs",
    "Assets/Scripts/swarm/SwarmAlgorithm.cs",
    "Assets/Scripts/swarm/SwarmPlaneController.cs",
    "Assets/Scripts/VelocityControl/VelocityControl.cs",
    "Assets/Scripts/VelocityControl/StateFinder.cs",
]

SMALL_DREF = 0.45      # "the tightest spread": the stick's minimum is 0.4
PAIR_KILL_M = 0.5      # centre distance at which two drones are lost together
DRONE_HALF_M = 0.25    # added to a building footprint for the building-kill test
QUERY_EVERY = 10       # ticks between refreshes of the buildings worth querying
QUERY_MARGIN_M = 60.0  # how far beyond OverlapSphere's reach those are taken from


# ============================================================================= staleness

def force_law_changes(since=FORCE_LAW_COMMIT):
    """(commits, uncommitted) touching FORCE_LAW_FILES since `since`, or (None, reason) if git cannot tell."""
    def git(*a):
        return subprocess.run(["git", "-C", str(sp.REPO), *a], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return git("log", "--oneline", f"{since}..HEAD", "--", *FORCE_LAW_FILES), \
               git("status", "--porcelain", "--", *FORCE_LAW_FILES)
    except (OSError, subprocess.CalledProcessError) as e:
        return None, str(e).strip()


def check_force_law(allow_stale, since=FORCE_LAW_COMMIT):
    log, dirty = force_law_changes(since)
    if log is None:
        print(f"[replica] WARNING: cannot check the mirrored C# against git ({dirty}); assuming it is as of {since}")
        return
    if not log and not dirty:
        return
    msg = [f"[replica] The C# this replica mirrors has changed since FORCE_LAW_COMMIT {since}:"]
    if log:
        msg += ["  " + line for line in log.splitlines()]
    if dirty:
        msg += ["  uncommitted: " + line.strip() for line in dirty.splitlines()]
    msg += [f"  Review `git diff {since} -- {' '.join(FORCE_LAW_FILES)}`, bring swarm_replica.py into line with any",
            "  change to the horizontal force law, then bump FORCE_LAW_COMMIT. A change that leaves the law alone",
            "  (yaw, altitude, plane mode) needs only the bump."]
    if allow_stale:
        print("\n".join(msg + ["  --allow-stale: continuing on the replica's law as it stands."]))
        return
    print("\n".join(msg + ["  Refusing to run (exit 3); --allow-stale overrides."]))
    sys.exit(3)


# ============================================================================= the force law

def bump(x, d, delta):
    """OlfatiSaber.GetBetaBump: rho_h(x / d), and 0 everywhere when d <= 0 (switched off)."""
    x = np.asarray(x, float)
    if d <= 0:
        return np.zeros_like(x)
    r = x / d
    cosine = 0.5 * (1.0 + np.cos(np.pi * (r - delta) / (1.0 - delta)))
    return np.where(r < delta, 1.0, np.where(r >= 1.0, 0.0, cosine))


def sigma1(z):
    return z / np.sqrt(1.0 + z * z)


def sat(f, m):
    """The smooth vector saturation OlfatiSaber applies to the obstacle field, core and damper; m <= 0 = none."""
    if m <= 0:
        return f
    return f / np.sqrt(1.0 + (f * f).sum(-1, keepdims=True) / (m * m))


def clamp_norm(v, m):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > m, v * m / np.maximum(n, 1e-12), v)


def cohesion(r, dref, r0, a, b, delta):
    """OlfatiSaber.GetCohesionForce: (1/r0) rho'(r/r0) psi(r) + rho(r/r0) phi(r), rho = rho_h squared."""
    c = (b - a) / (2.0 * np.sqrt(a * b))
    z = r - dref
    psi = (a + b) / 2 * (np.sqrt(1 + (z + c) ** 2) - np.sqrt(1 + c * c)) + (a - b) * z / 2
    phi = (a + b) / 2 * (z + c) / np.sqrt(1 + (z + c) ** 2) + (a - b) / 2
    x = r / r0
    arg = np.pi * (x - delta) / (1 - delta)
    inside = (x >= delta) & (x < 1)
    rho = np.where(x < delta, 1.0, np.where(inside, (0.5 * (1 + np.cos(arg))) ** 2, 0.0))
    drho = np.where(inside, 0.5 * (-np.pi) / (1 - delta) * (1 + np.cos(arg)) * np.sin(arg), 0.0)
    return drho / r0 * psi + rho * phi


class Model:
    """The resolved constants under the names of the C# fields they come from, plus what
    SwarmAlgorithm.UpdateOlfatiSaberParameters derives from them."""

    def __init__(self, swarm, air):
        self.swarm, self.air = swarm, air
        if swarm["swarmAlgorithm"] != 2:
            raise ValueError(f"swarmAlgorithm is {swarm['swarmAlgorithm']}; the replica models OLFATI_SABER (2) only")
        if swarm["is3D"]:
            raise ValueError("is3D is on; the replica models the horizontal (2D) formation only")
        for k in ("d_ref", "delta", "a", "b", "c_vm", "d_obs", "r0_obs", "c_obs", "c2_beta", "d_shield",
                  "c_core", "c2_core", "coreStandoffRatio", "coreRadiusFraction", "coreRadiusFilterTime",
                  "c_damp", "d_damp", "r0_coh", "r0CohRatio"):
            setattr(self, k, float(swarm[k]))
        self.S = float(swarm["scaleFactor"])
        self.hollow = bool(swarm["hollowSwarmCore"])
        self.core_relative = bool(swarm["coreRelativeVelocity"])
        self.g = air["gravity"]
        self.dt = air["fixedDeltaTime"]
        self.max_tilt = min(air["maxPitch"], air["maxRoll"])
        budget = self.g * np.tan(self.max_tilt)    # the drone's tilt budget, which caps all three ceilings
        self.max_obs = min(swarm["maxObstacleAccel"], budget)
        self.max_core = min(swarm["maxCoreAccel"], budget)
        self.max_damp = min(swarm["maxDampAccel"], budget)
        self.max_speed = air["maxSpeed"]
        self.tau_acc = air["timeConstantAcceleration"]
        self.swarm_filter = air["SwarmAccelFilterCoefficient"]
        self.tau_w = air["timeConstantOmegaXYRate"]
        self.tau_a = air["timeConstantAlphaRate"]
        self.max_alpha = air["maxAlpha"]
        self.drag, self.ang_drag = air["drag"], air["angularDrag"]
        noise = bool(air["enableStateNoise"])
        self.pos_noise = air["positionNoiseSigma"] if noise else 0.0
        self.vel_noise = air["velocityNoiseSigma"] if noise else 0.0
        self.att_noise = air["attitudeNoiseSigma"] if noise else 0.0

    def r0_eff(self, dref):
        """OlfatiSaber.EffectiveR0Coh."""
        return self.r0CohRatio * dref if self.hollow and self.r0CohRatio > 0 else self.r0_coh

    def stick_to_world(self, stick, body_yaw_deg):
        """VelocityControl.SetNormalisedVelocity + the VR command frame (forced in the hull attitude modes):
        world = Euler(0, bodyYaw, 0) * (roll, 0, pitch) * maxSpeed, the stick magnitude-limited to 1."""
        s = np.asarray(stick, float)
        n = np.linalg.norm(s)
        if n > 1:
            s = s / n
        y = np.radians(body_yaw_deg)
        return np.array([s[0] * np.cos(y) + s[1] * np.sin(y), -s[0] * np.sin(y) + s[1] * np.cos(y)]) * self.max_speed


class City:
    """Buildings as OlfatiSaber sees them. Each box is a collider's oriented box (centre, three half-axis
    vectors); the obstacle is the smallest cylinder about world up containing its world AABB
    (OlfatiSaber.GetObstacleCylinder over Collider.bounds), and OverlapSphere finds it by the box itself."""

    def __init__(self, boxes):
        self.n = len(boxes)
        if self.n == 0:
            self.c3 = np.zeros((0, 3)); self.unit = np.zeros((0, 3, 3)); self.half = np.zeros((0, 3))
            self.c = np.zeros((0, 2)); self.R = np.zeros(0); self.flat = np.zeros((0, 3), bool)
            return
        self.c3 = np.array([b[1] for b in boxes], float)
        ax = np.array([b[2] for b in boxes], float)
        self.half = np.linalg.norm(ax, axis=2)
        self.unit = ax / np.maximum(self.half, 1e-9)[..., None]
        ext = np.abs(ax).sum(axis=1)                      # bounds.extents
        self.c = self.c3[:, [0, 2]]                       # bounds.center, horizontally
        self.R = np.hypot(ext[:, 0], ext[:, 2])
        self.flat = np.abs(self.unit[..., 1]) < 0.5       # the box axes that lie in the horizontal

    @classmethod
    def for_run(cls, goals, city_name):
        return cls(analyse.scene_boxes(goals, city_name))

    def _proj(self, P, B, alt):
        d = np.zeros((len(P), len(B), 3))
        d[..., 0] = P[:, None, 0] - self.c3[B][None, :, 0]
        d[..., 2] = P[:, None, 1] - self.c3[B][None, :, 2]
        if alt is not None:
            d[..., 1] = alt[:, None] - self.c3[B][None, :, 1]
        return np.einsum("nbk,bjk->nbj", d, self.unit[B])

    def distance(self, P, B, alt=None):
        """(n, len(B)) distance from each drone to each box. alt=None takes every drone to be within every
        box's height, so only the box's horizontal axes count."""
        excess = np.maximum(np.abs(self._proj(P, B, alt)) - self.half[B][None], 0.0)
        if alt is None:
            excess = np.where(self.flat[B][None], excess, 0.0)
        return np.linalg.norm(excess, axis=-1)

    def footprint_hit(self, P, B, margin):
        """(n,) whether each drone's centre is inside some box's footprint grown by margin."""
        if len(B) == 0:
            return np.zeros(len(P), bool)
        inside = (np.abs(self._proj(P, B, None)) <= self.half[B][None] + margin) | ~self.flat[B][None]
        return inside.all(-1).any(-1)

    def near(self, P, reach):
        """Buildings whose cylinder comes within reach of any of the points P (n, 2)."""
        if self.n == 0:
            return np.arange(0)
        d = np.linalg.norm(self.c[None] - P[:, None], axis=-1).min(0) - self.R
        return np.flatnonzero(d < reach)

    def min_distance(self, P, alt):
        if self.n == 0:
            return np.full(len(P), np.inf)
        return self.distance(P, np.arange(self.n), alt).min(1)


def remove_inward(v, shield):
    """OlfatiSaber.RemoveInwardComponent, for the pilot's command and the damper: fade out the part of v
    heading into each obstacle within d_shield, frame by frame in query order."""
    for w, out in shield:
        inward = -(v * out).sum(-1)
        v = v + (w * np.maximum(inward, 0.0))[:, None] * out
    return v


def swarm_terms(m, city, cand, Pm, Vm, cen, vmean, core_r, dref, cmd, fallback, alt=None):
    """Every horizontal term OlfatiSaber.GetSwarmAcceleration sums, and the shielded command, per drone.

    Pm, Vm     (n, 2) measured positions and velocities (StateFinder, noisy in Unity)
    cen        swarm centroid (SwarmPlaneController, from TRUE positions)
    vmean      core reference velocity: the mean measured velocity, or zeros (coreRelativeVelocity off)
    core_r     core radius in metres, 0 when the core is off this tick
    cmd        the pilot's world command before the shield, (2,)
    fallback   (n, 2) outward direction for a drone exactly on the core axis (C#: its own forward)
    alt        (n,) altitudes for the obstacle query, or None (within every building's height)
    """
    n, S, dl = len(Pm), m.S, m.delta
    zero = np.zeros((n, 2))

    # ---- alpha lattice: cohesion, velocity consensus, close-range damper (all in swarm units)
    D = Pm[None, :, :] - Pm[:, None, :]                    # [i, j] = j - i
    dist = np.linalg.norm(D, axis=-1)
    pair = ~np.eye(n, dtype=bool) & (dist > 0)
    unit = np.where(pair[..., None], D / np.where(pair, dist, 1.0)[..., None], 0.0)
    r = dist / S
    coh = (np.where(pair, cohesion(r, dref, m.r0_eff(dref), m.a, m.b, dl), 0.0)[..., None] * unit).sum(1)
    vc = m.c_vm * (Vm[None, :, :] - Vm[:, None, :]).sum(1)
    damp = zero
    if m.c_damp > 0:
        rate = ((Vm[None, :, :] - Vm[:, None, :]) * unit).sum(-1)       # range rate, < 0 approaching
        w = np.where(pair & (r < m.d_damp) & (rate < 0), m.c_damp * bump(r, m.d_damp, dl) * rate, 0.0)
        damp = (w[..., None] * unit).sum(1)

    # ---- obstacles: beta-agents on each building's cylinder, and the shield's frames
    obs, prox, shield = zero, np.zeros(n), []
    if len(cand):
        rel = Pm[:, None, :] - city.c[cand][None]
        rho = np.linalg.norm(rel, axis=-1)
        R = city.R[cand][None]
        found = (city.distance(Pm, cand, alt) < m.r0_obs * S) & (rho >= 1e-4)   # OverlapSphere, valid frame
        out = rel / np.maximum(rho, 1e-4)[..., None]
        tan = np.stack([out[..., 1], -out[..., 0]], -1)                         # world up x outward
        mu = np.minimum(R / np.maximum(rho, 1e-4), 1.0)
        dd = np.maximum(rho - R, 0.0) / S
        bo = bump(dd, m.d_obs, dl) * found
        prox = bo.max(1)
        rep = (bo * (sigma1(dd - m.d_obs) - 1.0))[..., None] * (-out)
        vel_obs = (mu * (Vm[:, None, :] * tan).sum(-1))[..., None] * tan
        vmatch = bo[..., None] * (vel_obs - Vm[:, None, :])
        obs = sat(m.c_obs * rep.sum(1) + m.c2_beta * vmatch.sum(1), m.max_obs)
        if m.d_shield > 0:
            ws = bump(dd, m.d_shield, dl) * (found & (dd < m.d_shield))
            for k in np.flatnonzero(ws.max(0) > 0):
                shield.append((ws[:, k], out[:, k]))

    # ---- hollow core: a virtual beta-agent cylinder on the centroid, moving with the swarm
    core = zero
    if core_r > 0 and m.c_core > 0:
        rel = Pm - cen
        rho = np.linalg.norm(rel, axis=1)
        axial = rho < 1e-4
        out = np.where(axial[:, None], fallback, rel / np.maximum(rho, 1e-4)[:, None])
        tan = np.stack([out[:, 1], -out[:, 0]], -1)
        mu = np.where(axial, 1.0, np.minimum(core_r / np.maximum(rho, 1e-4), 1.0))
        dd = np.where(axial, 0.0, np.maximum(rho - core_r, 0.0) / S)
        so = max(m.coreStandoffRatio * dref, 1e-3)
        b = bump(dd, so, dl)
        vrel = Vm - vmean
        rep = (b * (sigma1(dd - so) - 1.0))[:, None] * (-out)
        vel_obs = (mu * (vrel * tan).sum(-1))[:, None] * tan
        vmatch = b[:, None] * (vel_obs - vrel)
        core = sat((m.c_core * rep + m.c2_core * vmatch) * (1.0 - prox)[:, None], m.max_core)

    damp_raw = damp
    damp = remove_inward(sat(damp, m.max_damp), shield)
    shielded = remove_inward(np.broadcast_to(np.asarray(cmd, float), (n, 2)).copy(), shield)
    return dict(vc=vc, coh=coh, damp=damp, damp_raw=damp_raw, obs=obs, core=core, cmd=shielded, prox=prox)


# ============================================================================= closed loop

class Sim:
    """The swarm's horizontal state, stepped as VelocityControl and PhysX step it."""

    def __init__(self, m, city, P0, V0, seed):
        self.m, self.city = m, city
        n = len(P0)
        self.rng = np.random.default_rng(seed)
        self.P, self.V = np.array(P0, float), np.array(V0, float)
        self.th, self.w, self.fs = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
        self.alive = np.ones(n, bool)
        ang = self.rng.uniform(0, 2 * np.pi, n)
        self.forward = np.stack([np.sin(ang), np.cos(ang)], 1)   # no heading is modelled; any distinct one
        self.core_r = None
        self.cand = np.arange(0)
        self.tick = 0
        self.kills = []            # (t, "drone" | "building", d_ref)

    @property
    def t(self):
        return self.tick * self.m.dt

    def step(self, stick, body_yaw_deg, dref):
        m, rng = self.m, self.rng
        idx = np.flatnonzero(self.alive)
        n = len(idx)
        if n == 0:
            return
        P, V = self.P[idx], self.V[idx]
        Pm = P + rng.normal(0, m.pos_noise, P.shape)
        Vm = V + rng.normal(0, m.vel_noise, V.shape)
        cen = P.mean(0)

        # SwarmPlaneController.UpdateCoreRadius: a fraction of the mean distance from the centroid, low-passed,
        # seeded on the first tick it is wanted and cleared when it is not.
        core_r = 0.0
        if m.hollow and n > 0:
            target = m.coreRadiusFraction * np.linalg.norm(P - cen, axis=1).mean()
            if self.core_r is None or m.coreRadiusFilterTime <= 0:
                self.core_r = target
            else:
                self.core_r += (1 - np.exp(-m.dt / m.coreRadiusFilterTime)) * (target - self.core_r)
            core_r = self.core_r
        else:
            self.core_r = None
        vmean = Vm.mean(0) if m.core_relative else np.zeros(2)

        if self.tick % QUERY_EVERY == 0:
            self.cand = self.city.near(P, m.r0_obs * m.S + QUERY_MARGIN_M)
        cmd = m.stick_to_world(stick, body_yaw_deg)
        T = swarm_terms(m, self.city, self.cand, Pm, Vm, cen, vmean, core_r, dref, cmd, self.forward[idx])

        # VelocityControl: low-passed swarm force + velocity P-loop on the shielded command, circular tilt
        # clamp, tilt -> rate -> alpha; then PhysX (velocity change before damping) and a = g tan(tilt).
        swarm = T["vc"] + T["coh"] + T["damp"] + T["obs"] + T["core"]
        self.fs[idx] += m.swarm_filter * (swarm - self.fs[idx])
        user = (T["cmd"] - Vm) / m.tau_acc
        theta = clamp_norm((user + self.fs[idx]) / m.g, m.max_tilt)
        th_meas = self.th[idx] + rng.normal(0, m.att_noise, theta.shape)
        w_cmd = (theta - th_meas) / m.tau_w
        alpha = clamp_norm((w_cmd - self.w[idx]) / m.tau_a, m.max_alpha)
        w = (self.w[idx] + alpha * m.dt) * (1 - m.ang_drag * m.dt)
        th = self.th[idx] + w * m.dt
        tm = np.linalg.norm(th, axis=1, keepdims=True)
        acc = m.g * np.tan(tm) * th / np.maximum(tm, 1e-9)
        V = (V + acc * m.dt) * (1 - m.drag * m.dt)
        P = P + V * m.dt
        self.w[idx], self.th[idx], self.V[idx], self.P[idx] = w, th, V, P
        self.tick += 1

        # ---- losses: pairs closer than PAIR_KILL_M, and drones inside a building footprint
        dead = set()
        if n > 1:
            Dt = np.linalg.norm(P[None] - P[:, None], axis=-1)
            for i, j in zip(*np.where(np.triu(Dt < PAIR_KILL_M, 1))):
                if i in dead or j in dead:
                    continue
                dead |= {i, j}
                self.kills.append((self.t, "drone", dref))
        for i in np.flatnonzero(self.city.footprint_hit(P, self.cand, DRONE_HALF_M)):
            if i not in dead:
                dead.add(i)
                self.kills.append((self.t, "building", dref))
        self.alive[idx[list(dead)]] = False


def formation_sample(P):
    """(min pair distance, median nearest-neighbour distance, hull fraction) of the points P."""
    from scipy.spatial import ConvexHull
    Dm = np.linalg.norm(P[None] - P[:, None], axis=-1)
    np.fill_diagonal(Dm, np.inf)
    try:
        hull = len(ConvexHull(P).vertices) / len(P) if len(P) >= 3 else 1.0
    except Exception:
        hull = np.nan
    return Dm.min(), np.median(Dm.min(1)), hull


def load_run(test_dir, stem):
    test_dir = Path(test_dir)
    dr = pd.read_csv(test_dir / f"{stem}_drones.csv", sep=";")
    hd = pd.read_csv(test_dir / f"{stem}_head.csv", sep=";")
    sh = pd.read_csv(test_dir / f"{stem}_shape.csv", sep=";")
    session = json.loads((test_dir / f"{stem}_session.json").read_text(encoding="utf-8-sig"))
    piv = {k: dr.pivot_table(index="t", columns="droneId", values=k) for k in ["gtX", "gtY", "gtZ", "alive", "yawDeg"]}
    T = piv["gtX"].index.values
    P3 = np.stack([piv["gtX"].values, piv["gtY"].values, piv["gtZ"].values], -1)
    return dict(stem=stem, T=T, P3=P3, alive=piv["alive"].values > 0.5, yaw=piv["yawDeg"].values,
                hd=hd, sh=sh, goals=session.get("goals", []))


def recorded_core(sh):
    """Whether the run flew with the hollow core on, by its own shape.csv."""
    return bool(sh.hollowCore.mean() > 0.5) if "hollowCore" in sh else None


def run_sim(job):
    stem, label, swarm, air, city_name, test_dir, seed = job
    R = load_run(test_dir, stem)
    m = Model(swarm, air)
    city = City.for_run(R["goals"], city_name)
    T, P3 = R["T"], R["P3"]
    alt_mean = np.nanmean(P3[..., 1], 1)
    k0 = int(np.argmax(alt_mean > min(10.0, 0.5 * np.nanmax(alt_mean))))   # airborne
    k0 = max(k0, 1)
    P0 = P3[k0][:, [0, 2]]
    V0 = (P3[k0 + 1] - P3[k0 - 1])[:, [0, 2]] / (T[k0 + 1] - T[k0 - 1])
    sim = Sim(m, city, P0, V0, seed)
    hd, sh = R["hd"], R["sh"]
    ht, st = hd.t.values, sh.t.values
    roll, pitch, yaw = hd.inRoll.values, hd.inPitch.values, hd.bodyYaw.values
    drefs = sh.dRef.values
    t, tend = T[k0], min(ht[-1], st[-1])
    rec = []
    while t < tend and sim.alive.sum() > 1:
        h = min(max(np.searchsorted(ht, t, "right") - 1, 0), len(ht) - 1)
        s = min(max(np.searchsorted(st, t, "right") - 1, 0), len(st) - 1)
        sim.step((roll[h], pitch[h]), yaw[h], drefs[s])
        t += m.dt
        if sim.tick % 5 == 0:
            Pa = sim.P[sim.alive]
            if len(Pa) > 1:
                dmin, nn, hull = formation_sample(Pa)
                rec.append((drefs[s], dmin, nn, np.linalg.norm(sim.V[sim.alive].mean(0)), hull, len(Pa)))
    warning = None
    flown = recorded_core(sh)
    if flown is not None and flown != m.hollow:
        warning = (f"{stem}: shape.csv says the hollow core was {'on' if flown else 'off'}, but hollowSwarmCore "
                   f"resolved to {m.hollow} -- give the test a swarmParams entry in test.json")
    return dict(stem=stem, variant=label, seed=seed, kills=sim.kills, rec=np.array(rec), dt=m.dt, warning=warning)


def summarise(results):
    rows = []
    for label in dict.fromkeys(r["variant"] for r in results):
        rs = [r for r in results if r["variant"] == label]
        rec = np.concatenate([r["rec"] for r in rs if len(r["rec"])])
        dt5 = 5 * rs[0]["dt"] / 60.0
        small = rec[:, 0] <= SMALL_DREF
        min_small, min_all = small.sum() * dt5, len(rec) * dt5
        kills = [k for r in rs for k in r["kills"]]
        dd_small = sum(1 for _, kind, d in kills if kind == "drone" and d <= SMALL_DREF)
        dd_large = sum(1 for _, kind, d in kills if kind == "drone" and d > SMALL_DREF)
        rows.append(dict(
            variant=label, runs=len({r["stem"] for r in rs}), seeds=len({r["seed"] for r in rs}),
            min_small=round(min_small, 1), min_all=round(min_all, 1),
            dd_small=dd_small, dd_large=dd_large,
            dd_per_min_small=round(dd_small / max(min_small, 1e-9), 3),
            dd_per_min_large=round(dd_large / max(min_all - min_small, 1e-9), 3),
            building=sum(1 for _, kind, _ in kills if kind == "building"),
            nn_small=round(float(np.median(rec[small, 2])), 2) if small.any() else np.nan,
            nearmiss_small=round(float((rec[small, 1] < 1.0).mean()), 3) if small.any() else np.nan,
            hull_small=round(float(np.nanmean(rec[small, 4])), 3) if small.any() else np.nan,
            hull_all=round(float(np.nanmean(rec[:, 4])), 3),
            speed=round(float(rec[:, 3].mean()), 2)))
    return pd.DataFrame(rows)


# ============================================================================= open loop: forces at crashes

TERMS = ["user", "shield", "coh", "obs", "core", "damp", "vc", "clamp"]


def crash_terms(m, city, R, k, sh, hd, ids):
    """All terms on every alive drone at recorded sample k. Measured = recorded: no noise, no filter (the
    swarm-force low-pass is left out, so each term is what the force law commands at that instant)."""
    alive = R["alive"][k]
    idx = np.flatnonzero(alive)
    P3 = R["P3"][k, idx]
    T = R["T"]
    V3 = R["V3"][k, idx]
    Pm, Vm = P3[:, [0, 2]], V3[:, [0, 2]]
    s = sh.iloc[int((sh.t - T[k]).abs().argmin())]
    h = hd.iloc[int((hd.t - T[k]).abs().argmin())]
    core_r = float(s.coreRadiusM) if s.hollowCore else 0.0      # the radius the run actually had
    vmean = Vm.mean(0) if m.core_relative else np.zeros(2)
    yaw = np.radians(R["yaw"][k, idx])
    fallback = np.stack([np.sin(yaw), np.cos(yaw)], 1)
    cmd = m.stick_to_world((h.inRoll, h.inPitch), h.bodyYaw)
    cand = city.near(Pm, m.r0_obs * m.S + QUERY_MARGIN_M)
    t_ = swarm_terms(m, city, cand, Pm, Vm, Pm.mean(0), vmean, core_r, float(s.dRef), cmd, fallback, alt=P3[:, 1])
    user_ideal = (cmd - Vm) / m.tau_acc
    user = (t_["cmd"] - Vm) / m.tau_acc
    swarm = t_["vc"] + t_["coh"] + t_["damp"] + t_["obs"] + t_["core"]
    tot = user + swarm
    amax = m.max_tilt * m.g                    # the tilt clamp, as an acceleration (a/g is the commanded angle)
    clamped = clamp_norm(tot, amax)
    out = dict(user=user_ideal, shield=user - user_ideal, coh=t_["coh"], obs=t_["obs"], core=t_["core"],
               damp=t_["damp"], vc=t_["vc"], clamp=clamped - tot, tot=clamped,
               sat=np.linalg.norm(tot, axis=1) / amax,
               damp_raw=np.linalg.norm(t_["damp_raw"], axis=1),
               dobs=city.min_distance(Pm, P3[:, 1]), dref=float(s.dRef))
    pos = {d: i for i, d in enumerate(idx)}
    return out, [pos.get(d) for d in ids]


def run_forces(test_dir, city_name, crashes, swarm_for, air, window_samples):
    rows = []
    for stem, tcrash, ids in crashes:
        R = load_run(test_dir, stem)
        T = R["T"]
        R["V3"] = np.gradient(R["P3"], T, axis=0)
        acc = np.gradient(R["V3"], T, axis=0)
        m = Model(swarm_for(stem), air)
        city = City.for_run(R["goals"], city_name)
        i, j = ids
        ti = int(np.searchsorted(T, tcrash)) - 2       # last sample with both alive and a clean central difference
        print(f"\n==== {stem} t={tcrash:.2f} pair {i},{j}   closing acceleration, m/s^2 (+ = pushes the pair together)")
        print("  t_rel  dist vclose " + " ".join(f"{x:>7s}" for x in TERMS) + "   model   meas  sat_i sat_j  dObs_i dObs_j")
        for k in range(max(1, ti - window_samples), ti + 1):
            if not (R["alive"][k, i] and R["alive"][k, j]):
                continue
            out, (pi, pj) = crash_terms(m, city, R, k, R["sh"], R["hd"], (i, j))
            e = (R["P3"][k, j] - R["P3"][k, i])[[0, 2]]
            e = e / np.linalg.norm(e)
            closing = {x: -float(np.dot(out[x][pj] - out[x][pi], e)) for x in TERMS}
            model = -float(np.dot(out["tot"][pj] - out["tot"][pi], e))
            meas = -float(np.dot((acc[k, j] - acc[k, i])[[0, 2]], e))
            vclose = -float(np.dot((R["V3"][k, j] - R["V3"][k, i])[[0, 2]], e))
            dist = float(np.linalg.norm(R["P3"][k, j] - R["P3"][k, i]))
            rows.append(dict(run=stem, tcrash=tcrash, i=i, j=j, trel=T[k] - tcrash, dist=dist, vclose=vclose,
                             dref=out["dref"], model=model, meas=meas, sat_i=out["sat"][pi], sat_j=out["sat"][pj],
                             dobs_i=out["dobs"][pi], dobs_j=out["dobs"][pj],
                             damp_raw_i=out["damp_raw"][pi], damp_raw_j=out["damp_raw"][pj], **closing))
            if (ti - k) % 2 == 0:
                print(f" {T[k] - tcrash:6.2f} {dist:5.2f} {vclose:5.2f} " + " ".join(f"{closing[x]:7.2f}" for x in TERMS)
                      + f"  {model:6.2f} {meas:6.2f}  {out['sat'][pi]:5.2f} {out['sat'][pj]:5.2f}"
                      + f"  {out['dobs'][pi]:6.1f} {out['dobs'][pj]:6.1f}")
    return pd.DataFrame(rows)


def crash_list(test_dir, picks):
    """[(stem, t, (i, j))]: --crash picks, or every two-drone crash in results/crashes.csv."""
    out = []
    if picks:
        for p in picks:
            run, rest = p.split("@", 1)
            t, _, pair = rest.partition(":")
            out.append((run, float(t), tuple(int(x) for x in pair.split(",")) if pair else None))
    table = Path(test_dir) / "results" / "crashes.csv"
    need_table = not picks or any(ids is None for _, _, ids in out)
    if need_table and not table.exists():
        sys.exit(f"[replica] {table} does not exist: run `analyse.py run {Path(test_dir).name}` first, or name "
                 "crashes with --crash RUN@T:i,j")
    cr = pd.read_csv(table, sep=";") if need_table else None
    if not picks:
        for (run, t), g in cr.groupby(["run", "t"]):
            ids = sorted(g.droneId.tolist())
            if len(ids) == 2:
                out.append((run, float(t), tuple(ids)))
            else:
                print(f"[replica] skipping {run} t={t}: {len(ids)} drone(s) lost at that instant, not a pair")
        return out
    resolved = []
    for run, t, ids in out:
        if ids is None:
            g = cr[(cr.run == run) & ((cr.t - t).abs() < 0.2)]
            ids = tuple(sorted(g.droneId.tolist()))
            if len(ids) != 2:
                sys.exit(f"[replica] {run}@{t}: crashes.csv has {len(ids)} drones there; give the pair as :i,j")
        resolved.append((run, t, ids))
    return resolved


# ============================================================================= step response

STEP_SPREAD = [(0, 1.0), (40, 1.6), (70, 0.4), (100, 1.0), (130, 0.4)]
STEP_STICK = [(0, (0, 0)), (140, (0, 1)), (155, (0, 0)), (162, (0, 1)), (170, (0, -1)), (178, (1, 0)), (184, (0, 0))]
STEP_END = 190.0


def _held(table, t):
    v = table[0][1]
    for t0, x in table:
        if t >= t0:
            v = x
    return v


def run_step(job):
    label, swarm, air, seed, drones = job
    m = Model(swarm, air)
    rng = np.random.default_rng(1000 + seed)
    sim = Sim(m, City([]), rng.normal(0, 6.0, (drones, 2)), np.zeros((drones, 2)), seed)
    rec = []
    while sim.t < STEP_END:
        sim.step(np.array(_held(STEP_STICK, sim.t), float), 0.0, _held(STEP_SPREAD, sim.t))
        if sim.tick % 5 == 0:
            Pa = sim.P[sim.alive]
            if len(Pa) > 1:
                dmin, nn, hull = formation_sample(Pa)
                rec.append((sim.t, np.linalg.norm(Pa - Pa.mean(0), axis=1).mean(), nn, dmin, hull))
    return label, seed, np.array(rec), len(sim.kills)


def step_metrics(rec, t_step, t_end):
    """Settled ring radius after the step, time to 50% / 90% of the change, overshoot %."""
    t, R = rec[:, 0], rec[:, 1]
    r0 = R[(t > t_step - 3) & (t <= t_step)].mean()
    r1 = R[(t > t_end - 5) & (t <= t_end)].mean()
    w = (t > t_step) & (t <= t_end)
    tt, frac = t[w] - t_step, (R[w] - r0) / (r1 - r0)
    return r1, tt[np.argmax(frac >= 0.5)], tt[np.argmax(frac >= 0.9)], max(0.0, frac.max() - 1.0) * 100


# ============================================================================= command line

def presets():
    """Named variants. Values that mean "as shipped" are read from the C# defaults, so they track retunes."""
    f = sp.swarm_manager_fields()
    damper = {k: f[k].default for k in ("c_damp", "d_damp", "maxDampAccel")}
    return {
        "as flown": {},            # sim/forces: what the test flew; no overrides
        "scene": {},               # step: the scene as saved; no overrides
        "core off": {"hollowSwarmCore": False},
        "core absolute": {"hollowSwarmCore": True, "coreRelativeVelocity": False},
        "core relative": {"hollowSwarmCore": True, "coreRelativeVelocity": True},
        "no damper": {"c_damp": 0.0},
        "damper": damper,
        "core relative + damper": {"hollowSwarmCore": True, "coreRelativeVelocity": True, **damper},
        "core absolute, no damper": {"hollowSwarmCore": True, "coreRelativeVelocity": False, "c_damp": 0.0},
    }


def parse_variants(items, default="as flown"):
    """['core off', 'mine:c_damp=1,d_damp=0.4', 'damper:d_damp=0.4'] -> [(label, overrides)]."""
    known = presets()
    out = []
    for item in items or [default]:
        label, _, assigns = item.partition(":")
        label = label.strip()
        extra = sp.parse_assignments([assigns]) if assigns else {}
        if label not in known and not extra:
            sys.exit(f"[replica] unknown variant {label!r}: presets are {list(known)}, or give LABEL:field=value,...")
        out.append((label, {**known.get(label, {}), **extra}))
    return out


def test_dir_of(a):
    d = Path(a.root) / a.test
    if not (d / "test.json").exists():
        sys.exit(f"[replica] {d} is not a test folder (no test.json); see analyse.py status")
    return d


def swarm_stems(test_dir, selectors):
    stems = sorted(p.name[:-len("_drones.csv")] for p in test_dir.glob("*_Swarm_*_drones.csv"))
    if selectors:
        stems = [s for s in stems if any(analyse.matches(s, x) for x in selectors)]
    if not stems:
        sys.exit(f"[replica] no swarm runs in {test_dir}" + (f" matching {selectors}" if selectors else ""))
    return stems


def out_dir(a, test_dir):
    d = Path(a.out) if a.out else test_dir / "results" / "replica"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def cmd_sim(a):
    check_force_law(a.allow_stale)
    td = test_dir_of(a)
    city = a.city or sp.test_city(td)
    stems = swarm_stems(td, a.runs)
    air = dict(sp.resolve_airframe())
    base = sp.parse_assignments(a.set)
    jobs = []
    for label, over in parse_variants(a.variant):
        for stem in stems:
            swarm = dict(sp.resolve_swarm(city, td, stem, {**base, **over}))
            for seed in range(a.seeds):
                jobs.append((stem, label, swarm, air, city, str(td), seed))
    print(f"[replica] {a.test} in {city}: {len(stems)} swarm runs x {a.seeds} seeds x "
          f"{len(jobs) // max(len(stems) * a.seeds, 1)} variants; force law as of {FORCE_LAW_COMMIT}")
    t0 = time.time()
    with Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(run_sim, jobs)
    for w in sorted({r["warning"] for r in res if r["warning"]}):
        print("[replica] WARNING:", w)
    table = summarise(res)
    print(f"{len(jobs)} simulated runs in {time.time() - t0:.0f} s\n")
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    d = out_dir(a, td)
    table.insert(0, "force_law", FORCE_LAW_COMMIT)
    table.insert(0, "test", a.test)
    path = d / f"sim_{stamp()}.csv"
    table.to_csv(path, index=False)
    if a.save:
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(res, f)
    print(f"\nwritten {path}")


def cmd_forces(a):
    check_force_law(a.allow_stale)
    td = test_dir_of(a)
    city = a.city or sp.test_city(td)
    base = sp.parse_assignments(a.set)
    air = dict(sp.resolve_airframe())
    crashes = crash_list(td, a.crash)
    df = run_forces(td, city, crashes, lambda stem: dict(sp.resolve_swarm(city, td, stem, base)), air,
                    int(round(a.window * 10)))
    if df.empty:
        print("[replica] no drone-drone crashes to replay")
        return
    last = df[df.trel >= -a.last]
    cols = ["vclose"] + TERMS + ["model", "meas", "sat_i", "sat_j"]
    pd.set_option("display.width", 250)
    print(f"\nMean over the last {a.last} s before contact:")
    print(last.groupby(["run", "tcrash"])[cols].mean().round(2).to_string())
    path = out_dir(a, td) / f"forces_{stamp()}.csv"
    df.to_csv(path, index=False)
    print(f"\nwritten {path}")


def cmd_step(a):
    check_force_law(a.allow_stale)
    air = dict(sp.resolve_airframe())
    base = sp.parse_assignments(a.set)
    variants = parse_variants(a.variant, default="scene")
    jobs = [(label, dict(sp.resolve_swarm(a.scene, overrides={**base, **over})), air, s, a.drones)
            for label, over in variants for s in range(a.seeds)]
    with Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(run_step, jobs)
    rows = []
    for label, _ in variants:
        rs = [r for r in res if r[0] == label]
        mm = np.array([[*step_metrics(r[2], 40, 70), *step_metrics(r[2], 70, 100), *step_metrics(r[2], 100, 130)]
                       for r in rs]).mean(0)
        hover = np.concatenate([r[2][(r[2][:, 0] > 90) & (r[2][:, 0] < 100)] for r in rs])
        cruise = np.concatenate([r[2][r[2][:, 0] > 140] for r in rs])
        rows.append({"variant": label,
                     "expand R": round(mm[0], 1), "t50": round(mm[1], 1), "t90": round(mm[2], 1), "over%": round(mm[3]),
                     "contract R": round(mm[4], 1), "t50 ": round(mm[5], 1), "t90 ": round(mm[6], 1),
                     "over% ": round(mm[7]), "re-expand t90": round(mm[10], 1),
                     "NN@0.4 hover": round(float(np.median(hover[:, 2])), 2),
                     "hull@0.4": round(float(np.nanmean(hover[:, 4])), 2),
                     "dmin<1m cruise": round(float((cruise[:, 3] < 1.0).mean()), 3),
                     "kills": sum(r[3] for r in rs)})
    pd.set_option("display.width", 250)
    print(f"[replica] spread steps in {a.scene}'s gains, {a.drones} drones, {a.seeds} seeds: d_ref 1.0 -> 1.6 at 40 s "
          "(expand), -> 0.4 at 70 s (contract), -> 1.0 at 100 s; cruise / stop / reversal at 0.4 from 140 s")
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_params(a):
    if a.test:
        td = test_dir_of(a)
        city = a.city or sp.test_city(td)
        stem = a.run
        if stem:
            matches = [s for s in swarm_stems(td, None) if analyse.matches(s, stem)]
            stem = matches[0] if matches else stem
        p = sp.resolve_swarm(city, td, stem, sp.parse_assignments(a.set))
        print(f"SwarmManager for {a.test}" + (f" / {stem}" if stem else "") + f" ({city}):")
    else:
        p = sp.resolve_swarm(a.scene, overrides=sp.parse_assignments(a.set))
        print(f"SwarmManager in {a.scene}:")
    print(p.describe())
    print("\nAirframe:")
    print(sp.resolve_airframe().describe())
    log, dirty = force_law_changes()
    state = "unknown (git unavailable)" if log is None else ("current" if not log and not dirty else "STALE")
    print(f"\nForce law mirrored as of {FORCE_LAW_COMMIT}: {state}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, test=True):
        if test:
            p.add_argument("--test", required=p.prog.split()[-1] != "params", help="test folder, e.g. internal_3")
            p.add_argument("--root", default=str(sp.DEFAULT_ROOT), help="experiment folder (default %(default)s)")
            p.add_argument("--city", help="override test.json's city")
            p.add_argument("--out", help="output folder (default <test>/results/replica)")
        p.add_argument("--set", action="append", metavar="FIELD=VALUE[,...]",
                       help="SwarmManager override applied to every variant")
        p.add_argument("--allow-stale", action="store_true", help="run even if the mirrored C# has changed")

    p = sub.add_parser("sim", help="closed-loop replica on a test's recorded pilot inputs")
    common(p)
    p.add_argument("--variant", action="append", metavar="LABEL[:FIELD=VALUE,...]",
                   help=f"repeatable; presets: {', '.join(presets())} (default: as flown)")
    p.add_argument("--runs", nargs="+", metavar="SELECTOR", help="run stems or prefixes (PID, PID_tN)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--save", action="store_true", help="also pickle every run's raw result")
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("forces", help="force attribution on the recorded states before each crash")
    common(p)
    p.add_argument("--crash", action="append", metavar="RUN@T[:i,j]",
                   help="replay this crash (default: every two-drone crash in results/crashes.csv)")
    p.add_argument("--window", type=float, default=4.0, help="seconds before contact to evaluate")
    p.add_argument("--last", type=float, default=1.5, help="seconds before contact the summary averages")
    p.set_defaults(func=cmd_forces)

    p = sub.add_parser("step", help="spread-stick step response at hover in open sky")
    common(p, test=False)
    p.add_argument("--scene", default=sp.DEFAULT_CITY, help="whose SwarmManager values (default %(default)s)")
    p.add_argument("--variant", action="append", metavar="LABEL[:FIELD=VALUE,...]")
    p.add_argument("--drones", type=int, default=10)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.set_defaults(func=cmd_step)

    p = sub.add_parser("params", help="print the resolved parameters and their sources")
    common(p)
    p.add_argument("--scene", default=sp.DEFAULT_CITY, help="scene to read when no --test is given")
    p.add_argument("--run", help="a run stem or prefix: include its <stem>_swarm.json if it has one")
    p.set_defaults(func=cmd_params)

    a = ap.parse_args()
    try:
        a.func(a)
    except ValueError as e:
        sys.exit(f"[replica] {e}")


if __name__ == "__main__":
    main()
