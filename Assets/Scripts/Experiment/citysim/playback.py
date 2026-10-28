"""Flight replays for the study pages: a city's map and flights flown in it, recorded compactly enough to ship beside
a page and played back by citysim/replay_view.js. Every study page has a Cities tab built from these (lessons.md 30).

    tape = playback.Tape.on_flight(f, dt=0.2)        # a split_probe.Flight before it runs: records it as it flies
    f.run(T)
    rec = tape.record()                            # frames, positions, headings, shown feeds, losses, contacts
    rec = playback.unity_tape(tr, dt=0.2)            # the same from a bench trajectory (tierc.read_traj)
    city = playback.city_json(cid, arena, blocks, buildings, ...)
    flight = playback.flight_json(rec, id=..., city=cid, ...)      # JSON, the bulky arrays in base64

The tape only reads the simulation: it wraps sim.step and copies state around each step, so a taped flight is the
flight flown without it, bit for bit. Re-fly a study's flight with a tape and compare its scored row with the stored
one before trusting the replay (split_replays does).

Encoding (replay_view.js decodes it): positions int16 decimetres; headings uint8, 256 steps per turn; shown-feed
masks uint16 (at most 16 drones); first-seen times uint16 in SEEN_STEP_S steps, 65535 never; every typed array
little-endian, in base64. Frames are uniform: frame k is at t0 + k dt. Everything else is plain JSON.
"""
import sys

sys.dont_write_bytecode = True

import base64  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

SEEN_STEP_S = 0.5
NEVER = 65535


# ============================================================================= encoding

def b64(a, dtype):
    """A typed array, little-endian, in base64."""
    return base64.b64encode(np.ascontiguousarray(np.asarray(a), dtype=np.dtype(dtype).newbyteorder("<"))
                            .tobytes()).decode("ascii")


def dm16(a):
    """Metres to int16 decimetres (NaN to 0: a dead drone's position is never drawn)."""
    a = np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(a * 10.0), -32767, 32767).astype("<i2")


def yaw8(a):
    """Degrees (0 = +z, clockwise) to uint8, 256 steps per turn."""
    a = np.nan_to_num(np.asarray(a, float))
    return (np.round(np.mod(a, 360.0) / 360.0 * 256.0) % 256).astype("u1")


def mask16(b):
    """(frames, n) booleans to one uint16 bit mask per frame."""
    b = np.asarray(b, bool)
    if b.shape[1] > 16:
        raise ValueError(f"{b.shape[1]} drones: the masks hold 16")
    return (b.astype(np.int64) << np.arange(b.shape[1])).sum(1).astype("<u2")


def times16(t, step=SEEN_STEP_S):
    """Seconds to uint16 steps of `step`, NEVER for inf / NaN."""
    t = np.asarray(t, float)
    out = np.full(t.shape, NEVER, np.int64)
    ok = np.isfinite(t)
    out[ok] = np.clip(np.round(t[ok] / step), 0, NEVER - 1)
    return out.astype("<u2")


def polys_json(polys):
    """A list of (k, 2) polygons: vertex counts (uint16) and vertices (int16 dm), flat."""
    polys = [np.asarray(p, float).reshape(-1, 2) for p in polys]
    n = np.array([len(p) for p in polys], "<u2")
    xz = np.vstack(polys) if polys else np.zeros((0, 2))
    return dict(count=len(polys), n=b64(n, "<u2"), xz=b64(dm16(xz).reshape(-1), "<i2"))


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ============================================================================= the replica's flights

class Tape:
    """Records a replica flight (swarm_replica.Sim or ContactSim) every dt as it flies: positions, alive, headings,
    shown feeds, the pilot's heading and waypoint, and each loss and contact as it happens. heading / shown / body /
    wp are callables reading the flight's own state (shown returns drone indices)."""

    def __init__(self, sim, dt, heading, shown, body=None, wp=None):
        self.sim, self.dt = sim, float(dt)
        self.every = max(1, int(round(self.dt / sim.m.dt)))
        self.dt = self.every * sim.m.dt
        self.heading, self.shown, self.body, self.wp = heading, shown, body, wp
        self.t, self.P, self.alive, self.yaw, self.mask, self.bodies, self.wps = [], [], [], [], [], [], []
        self.losses, self.contacts = [], []
        orig = sim.step

        def step(*a, **k):
            s = self.sim
            if s.tick % self.every == 0:
                self._snap()
            alive0 = s.alive.copy()
            touch0 = s.touching.copy() if hasattr(s, "touching") else None
            nk = len(s.kills)
            out = orig(*a, **k)
            died = np.flatnonzero(alive0 & ~s.alive)
            if len(died):
                self._died(died, [k[1] for k in s.kills[nk:]])
            if touch0 is not None:
                for i in np.flatnonzero(s.touching & ~touch0 & s.alive):
                    self.contacts.append(dict(t=_r(s.t, 2), i=int(i), x=_r(s.P[i, 0]), z=_r(s.P[i, 1])))
            return out

        sim.step = step

    @classmethod
    def on_flight(cls, f, dt=0.2):
        """A split_probe.Flight's tape: headings from its feed latch (each drone's actual yaw), the shown feeds, the
        pilot's body yaw and the waypoint it is flying to."""
        return cls(f.sim, dt, heading=lambda: f.latch.yaw.copy(),
                   shown=lambda: (np.asarray(f.feeds.drones, int) if f.feeds is not None else np.zeros(0, int)),
                   body=lambda: float(f.body or 0.0), wp=lambda: int(f.wp))

    def _snap(self):
        s = self.sim
        self.t.append(float(s.t))
        self.P.append(s.P.copy())
        self.alive.append(s.alive.copy())
        self.yaw.append(np.asarray(self.heading(), float).copy())
        m = np.zeros(len(s.P), bool)
        m[np.asarray(self.shown(), int)] = True
        self.mask.append(m & s.alive)
        self.bodies.append(self.body() if self.body else 0.0)
        self.wps.append(self.wp() if self.wp else 0)

    def _died(self, died, kinds):
        """Each lost drone with its cause: a pair kill ("drone": both die), a building entry, or isolation."""
        s = self.sim
        pairs = kinds.count("drone")
        left = list(died)
        why = {}
        while pairs and len(left) >= 2:                    # the closest two of the drones lost this step
            D = np.linalg.norm(s.P[left][:, None] - s.P[left][None], axis=-1)
            np.fill_diagonal(D, np.inf)
            a, b = np.unravel_index(int(np.argmin(D)), D.shape)
            i, j = left[a], left[b]
            why[i] = why[j] = "drone"
            left = [x for x in left if x not in (i, j)]
            pairs -= 1
        rest = [k for k in kinds if k != "drone"]
        for k, i in enumerate(left):
            why[i] = rest[k] if k < len(rest) else "lost"
        for i in died:
            self.losses.append(dict(t=_r(s.t, 2), i=int(i), x=_r(s.P[i, 0]), z=_r(s.P[i, 1]), why=why[i]))

    def record(self):
        if not self.t:
            raise ValueError("the tape recorded nothing: was it attached before the flight ran?")
        return dict(dt=self.dt, t=np.array(self.t), P=np.array(self.P), alive=np.array(self.alive),
                    yaw=np.array(self.yaw), shown=np.array(self.mask), body=np.array(self.bodies),
                    wp=np.array(self.wps), losses=self.losses, contacts=self.contacts)


# ============================================================================= Unity's flights

def unity_tape(tr, dt=0.2, single=False):
    """A tape from a bench trajectory (tierc.read_traj): every dt (a whole number of the trajectory's rows). A lone
    drone's feed is always shown (the bench's mask is the hull rule's). The bench records contacts as a count only,
    so contacts is empty; a loss is the first row a drone is no longer alive, where it was last seen."""
    step = max(1, int(round(dt / max(float(np.median(np.diff(tr["t"]))), 1e-6))))
    rows = np.arange(0, len(tr["t"]), step)
    alive = tr["alive"]
    shown = (alive if single else (tr["shown"] & alive))[rows]
    losses = []
    for i in range(tr["n"]):
        gone = np.flatnonzero(alive[:-1, i] & ~alive[1:, i])
        if len(gone):
            j = int(gone[0])
            losses.append(dict(t=_r(tr["t"][j + 1], 2), i=i, x=_r(tr["X"][j, i]), z=_r(tr["Z"][j, i]), why="lost"))
    body = tr["body"] if np.isfinite(tr["body"]).all() else np.zeros(len(tr["t"]))
    return dict(dt=float(np.median(np.diff(tr["t"][rows]))) if len(rows) > 1 else dt, t=tr["t"][rows],
                P=np.stack([tr["X"], tr["Z"]], -1)[rows], alive=alive[rows], yaw=tr["yaw"][rows], shown=shown,
                body=body[rows], wp=tr["wp"][rows], losses=losses, contacts=[])


# ============================================================================= JSON

def flight_json(rec, **meta):
    """One flight for replay_view.js. meta: id, title, city, source, ... (anything JSON); the tape's arrays become
    xz / yaw / shown / body / wp in base64, alive_to the first frame each drone is gone (frames if never)."""
    t = np.asarray(rec["t"], float)
    if len(t) > 2 and np.abs(np.diff(t) - rec["dt"]).max() > 0.02:
        raise ValueError("the tape's frames are not uniform in time")
    alive = np.asarray(rec["alive"], bool)
    frames, n = alive.shape
    alive_to = []
    for i in range(n):
        gone = np.flatnonzero(~alive[:, i])
        alive_to.append(int(gone[0]) if len(gone) else frames)
    return dict(meta, n=int(n), dt=round(float(rec["dt"]), 4), frames=int(frames), t0=round(float(t[0]), 3),
                xz=b64(dm16(rec["P"]).reshape(-1), "<i2"), yaw=b64(yaw8(rec["yaw"]).reshape(-1), "u1"),
                shown=b64(mask16(rec["shown"]), "<u2"), alive_to=alive_to, body=b64(yaw8(rec["body"]), "u1"),
                wp=b64(np.clip(np.asarray(rec["wp"]), 0, 65535).astype("<u2"), "<u2"),
                losses=rec["losses"], contacts=rec["contacts"])


def city_json(cid, arena, blocks, buildings, block_flags=None, cylinders=None, samples=None, streets=None, **meta):
    """One city for replay_view.js. blocks / buildings: lists of (k, 2) polygons; block_flags: per block, bit 1
    interior and bit 2 empty (a plaza); cylinders: (m, 3) cx, cz, r of what the force law sees round each building;
    samples: (q, 2) street centreline points (a flight's first-seen times index them); streets: (r, 4) segments."""
    out = dict(meta, id=cid, arena=[float(x) for x in arena], blocks=polys_json(blocks), buildings=polys_json(buildings))
    if block_flags is not None:
        out["block_flags"] = b64(np.asarray(block_flags, "u1"), "u1")
    if cylinders is not None:
        out["cylinders"] = b64(dm16(np.asarray(cylinders, float).reshape(-1)), "<i2")
    if samples is not None:
        out["samples"] = b64(dm16(np.asarray(samples, float).reshape(-1)), "<i2")
        out["n_samples"] = int(len(samples))
    if streets is not None:
        out["streets"] = b64(dm16(np.asarray(streets, float).reshape(-1)), "<i2")
    return out


def rounded(x, nd=1):
    """Nested lists / arrays of numbers rounded for JSON (NaN / inf to None)."""
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rounded(v, nd) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd) if math.isfinite(float(x)) else None
    return x
