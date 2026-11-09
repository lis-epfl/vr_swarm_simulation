"""The page replays (citysim.playback, S4's split_replays): what the compact encoding keeps, that a tape changes
nothing about the flight it records, and the formation's ring drawn round a block-shaped building."""
import sys

sys.dont_write_bytecode = True

import base64  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import playback  # noqa: E402


def _dec(s, dtype):
    return np.frombuffer(base64.b64decode(s), dtype)


def test_flight_json_round_trips_the_tape():
    rng = np.random.default_rng(4)
    frames, n = 50, 10
    P = rng.uniform(-360, 360, (frames, n, 2))
    yaw = rng.uniform(-720, 720, (frames, n))
    shown = rng.random((frames, n)) < 0.6
    alive = np.ones((frames, n), bool)
    alive[30:, 3] = False
    rec = dict(dt=0.2, t=np.arange(frames) * 0.2, P=P, alive=alive, yaw=yaw, shown=shown,
               body=rng.uniform(0, 360, frames), wp=np.arange(frames) // 5, losses=[], contacts=[])
    f = json.loads(json.dumps(playback.flight_json(rec, id="x")))
    assert np.abs(_dec(f["xz"], "<i2").reshape(frames, n, 2) / 10 - P).max() <= 0.05 + 1e-9
    y = _dec(f["yaw"], "u1").reshape(frames, n).astype(float) * 360 / 256          # (uint8 x 360 overflows uint16)
    assert np.abs((y - np.mod(yaw, 360) + 180) % 360 - 180).max() <= 360 / 512 + 1e-9
    m = _dec(f["shown"], "<u2").astype(int)
    assert ((m[:, None] >> np.arange(n)) & 1).astype(bool).tolist() == shown.tolist()
    assert f["alive_to"][3] == 30 and f["alive_to"][0] == frames
    assert (_dec(f["wp"], "<u2") == rec["wp"]).all()


def test_first_seen_times_mark_never():
    t = np.array([0.0, 0.26, 10.0, np.inf, np.nan])
    assert playback.times16(t).tolist() == [0, 1, 20, playback.NEVER, playback.NEVER]


def test_a_tape_leaves_the_flight_unchanged():
    """The replays are only worth showing if they are the flights the study scored: the same job flown with and
    without a tape must give the same row."""
    from citysim import split_probe as S, split_replays as R
    st = dict(r0_coh=30.0, t_hop=6.0, m3b_mode="aim")
    for trial in range(6):
        j = dict(kind="M3b", variant="R", p=40.0, s=18.0, city_seed=0, trial=trial, setting=st)
        plain = S.run_job(j)
        if plain.get("skipped") or "error" in plain:
            continue
        taped, made = R.taped(lambda: S.run_job(j), 0.1)
        assert R._strip(taped) == R._strip(plain)
        rec = made[0][1].record()
        assert rec["P"].shape[1] == S.N_DRONES and len(rec["t"]) > 50
        assert np.allclose(np.diff(rec["t"]), rec["dt"])
        return
    raise AssertionError("no M3b trial flew at p 40 s 18 seed 0")


def test_ring_round_a_block_shaped_building():
    """The ring 3 m outside a convex building, rounded at its corners: its perimeter is the building's plus 2 pi d,
    the d* prediction's ring."""
    from citysim import geometry as geo, split_replays as R
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    ring = R.ring_round(sq, 3.0, step_deg=1.0)
    assert abs(geo.perimeter(ring) - (40.0 + 2 * math.pi * 3.0)) < 0.05
    assert np.all(np.abs(np.hypot(*(ring - [5, 5]).T)) >= 8.0 - 1e-9)
