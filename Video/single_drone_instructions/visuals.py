"""Pictures for the single-drone instruction video.

Everything is the swarm video's (participant_instructions/visuals.py: the controller photo, the city maps, the
overlays, the clip grid), pointed at this video's own footage/. Only what a swarm has no use for is added here.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

import vidkit as vk
from vidkit import ACCENT, INK, MUTED, PANEL, View, ease

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(os.path.dirname(HERE), "participant_instructions", "visuals.py")
_spec = importlib.util.spec_from_file_location("swarm_video_visuals", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
base.FOOTAGE_DIR = os.path.join(HERE, "footage")


def v_one_screen(ctx):
    """The pilot from above with the one feed screen in front, and the drone with its camera's view."""
    fr = ctx.frame()
    c = np.array([620, 560])
    fr.circle(tuple(c), 300, PANEL, alpha=0.6)
    base.draw_pilot(fr, c, 1.4, forward=False)
    # the screen: an arc 60 degrees wide in front of the pilot (scale 1.3 at 2 m)
    fr.arc(tuple(c), 230, -120, -60, ACCENT, 14)
    fr.text("one screen, always in front of you", (c[0], c[1] - 300), 30, INK, "semibold", anchor="ma")
    fr.text("you, in the headset", (c[0], c[1] + 320), 26, MUTED, anchor="ma")
    view = View((0, 0), 6.0, (1380, 560))
    a = ease(ctx.u / 0.2)
    base.draw_drone(fr, view, np.array([0.0, -20.0]), 0.0, (235, 238, 245), alpha=1.0, cone=40,
                    cone_alpha=0.25 * a, r=12)
    fr.text("the drone, from above:\nthe green outline is what its camera sees", (1380, 820), 26, MUTED,
            anchor="ma", alpha=a, max_width=520)
    return fr


VISUALS = dict(base.VISUALS)
VISUALS.update({"one_screen": v_one_screen})
OVERLAYS = dict(base.OVERLAYS)
