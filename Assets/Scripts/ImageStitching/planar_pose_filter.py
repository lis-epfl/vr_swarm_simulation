"""
Rebuild a continuous camera pose from telemetry that is held between samples.

On the real fleet the video arrives at 20 fps and the telemetry behind each frame's pose
at about 5 Hz: the recorded clips show the position changing on a quarter of the frames
and standing still on the rest.  Every frame is therefore warped with a pose that is
between zero and 200 ms stale, a different amount for every drone on every frame.  At
2 m/s that is a sawtooth of up to 0.4 m on the plane -- seven source pixels at 30 m --
and during a yaw at the 40 deg/s clamp the heading alone steps eight degrees at a time.
The aligner cannot remove it: it estimates biases that last seconds, and this error
changes five times a second.

So the pose stream is resampled before anything uses it.  Each drone's distinct pose
values are kept as timed samples, and the pose handed on is the one interpolated at

    t_frame - delay

``delay`` is how much later the picture is than the telemetry it arrives with.  It is
not zero and its sign is the helpful one: video is compressed, sent and decoded, so a
frame is older than the newest telemetry sample by more than a sample period, and the
right pose for it lies *between* two samples already received.  Interpolating there is
both more accurate than the held value and free -- nothing is being delayed that was not
already late.  Where the query does run past the newest sample the pose is extrapolated
at the last measured rate, briefly.

A stream that is not held is passed through untouched.  The filter only acts on a drone
whose pose has actually been repeating from frame to frame, so the simulator's poses --
exact, and new on every render -- never see it.
"""

import os
from collections import deque

import numpy as np

# How much older a frame's picture is than the telemetry published with it, seconds.
# Measured on the recorded clips as the value that minimises how much the views move
# against each other from frame to frame (tools/planar_stream_bench.py --pose-delay).
DEFAULT_DELAY_S = float(os.environ.get("PLANAR_POSE_DELAY_S", "0.15"))

# A stream counts as held once this fraction of recent frames repeated the last pose.
HELD_FRACTION = 0.3
WINDOW = 40
# Extrapolate at most this far past the newest sample, and only from two samples no
# further apart than MAX_SAMPLE_GAP_S: beyond that the rate is stale and holding is safer.
MAX_EXTRAPOLATION_S = 0.25
MAX_SAMPLE_GAP_S = 0.8
HISTORY = 12
# Samples, and the span they may cover, behind the rate a position is extrapolated at.
RATE_SAMPLES = 5
RATE_WINDOW_S = 1.2


def _slerp(q0, q1, u):
    """Unit quaternions (x, y, z, w); ``u`` may leave [0, 1] to extrapolate."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:                      # the short way round
        q1, dot = -q1, -dot
    if dot > 0.999999:
        q = q0 + u * (q1 - q0)
    else:
        th = np.arccos(np.clip(dot, -1.0, 1.0))
        q = (np.sin((1.0 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


class _Channel:
    """One quantity of one drone (position or rotation): its samples and whether it is held."""

    def __init__(self, interpolate, linear=False):
        self.samples = deque(maxlen=HISTORY)          # (time first seen, value)
        self.repeats = deque(maxlen=WINDOW)
        self.interpolate = interpolate
        self.linear = linear

    def push(self, t, value):
        value = np.asarray(value, dtype=np.float64)
        if self.samples and np.array_equal(self.samples[-1][1], value):
            self.repeats.append(1)
        else:
            self.repeats.append(0)
            self.samples.append((t, value))

    def held(self):
        return (len(self.repeats) >= 10
                and sum(self.repeats) / float(len(self.repeats)) >= HELD_FRACTION)

    def at(self, t):
        s = self.samples
        if len(s) < 2:
            return s[-1][1]
        if t >= s[-1][0]:
            (t0, v0), (t1, v1) = s[-2], s[-1]
            if t1 - t0 > MAX_SAMPLE_GAP_S or t1 <= t0:
                return v1
            ahead = min(t - t1, MAX_EXTRAPOLATION_S)
            if self.linear:
                # A sample's time is only known to the frame it first appeared in, so a
                # rate from two samples 0.2 s apart is a quarter out. A line through the
                # last few is steadier, and a drone does not change velocity much in the
                # second they span.
                recent = [(ts, v) for ts, v in list(s)[-RATE_SAMPLES:]
                          if t1 - ts <= RATE_WINDOW_S]
                if len(recent) >= 3:
                    ts = np.array([r[0] for r in recent]) - t1
                    vs = np.stack([r[1] for r in recent])
                    tc = ts - ts.mean()
                    rate = (tc[:, None] * (vs - vs.mean(axis=0))).sum(axis=0) / max(
                        float((tc * tc).sum()), 1e-9)
                    return v1 + rate * ahead
            return self.interpolate(v0, v1, 1.0 + ahead / (t1 - t0))
        for k in range(len(s) - 1, 0, -1):
            (t0, v0), (t1, v1) = s[k - 1], s[k]
            if t >= t0:
                return self.interpolate(v0, v1, (t - t0) / max(t1 - t0, 1e-6))
        return s[0][1]


class PoseFilter:
    """Per-drone resampling of held telemetry.  ``apply`` is called once per frame."""

    def __init__(self, delay_s=None):
        self.delay = DEFAULT_DELAY_S if delay_s is None else float(delay_s)
        self.tracks = {}                 # drone_id -> (last time, position, rotation)
        self.active = 0

    def apply(self, views):
        """
        Views with ``pos`` / ``quat`` resampled where their stream is held.

        Returns the same list object when nothing is held, so the common simulator case
        costs a few comparisons and allocates nothing.
        """
        out, changed, active = [], False, 0
        for v in views:
            pos, quat = v.get("pos"), v.get("quat")
            t = float(v.get("capture_time", 0.0) or 0.0)
            if pos is None or quat is None or t <= 0.0 \
                    or not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))) \
                    or float(np.dot(quat, quat)) < 1e-6:
                out.append(v)
                continue
            track = self.tracks.get(v["drone_id"])
            if track is None or t < track[0] - 5.0:
                # New drone, or the producer's clock restarted.
                track = [t - 1.0, _Channel(lambda a, b, u: a + u * (b - a), linear=True),
                         _Channel(_slerp)]
                self.tracks[v["drone_id"]] = track
            if t > track[0]:                               # a new frame, not a re-serve
                track[0] = t
                track[1].push(t, pos)
                track[2].push(t, quat)

            new = None
            if track[1].held():
                new = dict(v, pos=tuple(float(c) for c in track[1].at(t - self.delay)))
            if track[2].held():
                new = dict(new if new is not None else v,
                           quat=tuple(float(c) for c in track[2].at(t - self.delay)))
            if new is None:
                out.append(v)
            else:
                out.append(new)
                changed = True
                active += 1
        self.active = active
        return out if changed else views
