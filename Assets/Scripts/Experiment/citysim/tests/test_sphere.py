"""A SPHERE run (one drone's 360 camera on a sphere round the pilot, InterfaceManager.DisplayMode.SPHERE): the hat
bases and the discovery measure read it as a camera that sees every way, the head as facing a bearing in the city
(headYaw - sphereYaw), and leave a feed run exactly as before."""
import sys

sys.dont_write_bytecode = True

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import conftest  # noqa: E402,F401  (paths, skip)
from citysim import discovery, replay  # noqa: E402

FAR_BOX = [("t", np.array([300.0, 25.0, 300.0]), np.diag([5.0, 25.0, 5.0]))]   # a city that hides nothing here
CAM = (0.0, 10.0, 0.0)          # the drone, heading 0 (+Z)


def _run(hat_xz, head_city_yaw, sphere=True, band=0.0):
    """A one-drone run, 1 s at 10 Hz, its hat fixed at hat_xz (1.8 m up), the head facing head_city_yaw in the city.
    The sphere sits at the display's 90 deg (a lone drone turned with, heading 0), so headYaw = city yaw + 90."""
    t = np.round(np.arange(11) * 0.1, 3)
    n = len(t)
    drones = pd.DataFrame(dict(t=t, droneId=0, gtX=CAM[0], gtY=CAM[1], gtZ=CAM[2], yawDeg=0.0, alive=1, hull=0,
                               shown=1, stitched=0))
    head = pd.DataFrame(dict(t=t, bodyYaw=0.0, headYaw=(head_city_yaw + 90.0) % 360.0, gimbalPitch=0.0,
                             gazeFeed=-1, sphereYaw=90.0 if sphere else np.nan))
    walkers = pd.DataFrame(dict(t=t, goalIndex=0, hatX=hat_xz[0], hatY=1.8, hatZ=hat_xz[1]))
    session = dict(displayMode="SPHERE", sphere360=dict(bandHalfDeg=band)) if sphere else {}
    return SimpleNamespace(drones=drones, head=head, walkers=walkers, goals=[dict(goalIndex=0)], hat_logged=True,
                           ct=t, cdt=np.full(n, 0.1), session=session,
                           condition="SingleSphere" if sphere else "SingleDrone")


def _bases(run):
    (b,) = replay.hat_bases(run, FAR_BOX, r_id=40.0)
    return {k: bool(np.all(b[k])) for k in ("all", "shown", "head", "band")}, b


def test_a_hat_behind_the_drone_is_seen_on_the_sphere_and_not_on_a_feed():
    behind = (0.0, -20.0)
    seen, b = _bases(_run(behind, head_city_yaw=180.0))
    assert seen == dict(all=True, shown=True, head=True, band=True), seen
    assert "gaze" not in b                 # no quads to hit: not measured rather than zero
    feed, _ = _bases(_run(behind, head_city_yaw=180.0, sphere=False))
    assert not feed["all"] and not feed["shown"]


def test_the_head_sees_it_only_when_facing_its_bearing_in_the_city():
    behind = (0.0, -20.0)
    assert _bases(_run(behind, head_city_yaw=180.0))[0]["head"]
    assert _bases(_run(behind, head_city_yaw=140.0))[0]["head"]        # 40 deg off: inside the HMD's +-50
    assert not _bases(_run(behind, head_city_yaw=0.0))[0]["head"]      # facing the drone's heading, away from it
    assert not _bases(_run(behind, head_city_yaw=120.0))[0]["head"]    # 60 deg off


def test_a_band_masks_the_sphere_as_it_does_the_view():
    below = (0.0, -8.0)                    # 46 deg below the horizon from 10 m up
    assert _bases(_run(below, 180.0, band=0.0))[0]["band"]
    assert not _bases(_run(below, 180.0, band=23.1))[0]["band"]
    assert _bases(_run((0.0, -20.0), 180.0, band=23.1))[0]["band"]      # 22 deg below: inside


def test_out_of_range_is_out_on_the_sphere_too():
    seen, _ = _bases(_run((0.0, -45.0), 180.0))
    assert not seen["all"]


def test_discovery_on_the_sphere():
    pts, w = np.array([[0.0, -20.0]]), np.array([1.0])
    _, a, h, _ = discovery.run_expected(_run((0.0, -20.0), 180.0), FAR_BOX, pts, w, r_goal=40.0, every_s=0.5)
    assert np.all(a == 1.0) and np.all(h == 1.0)
    _, a, h, _ = discovery.run_expected(_run((0.0, -20.0), 0.0), FAR_BOX, pts, w, r_goal=40.0, every_s=0.5)
    assert np.all(a == 1.0) and np.all(h == 0.0)
    _, a, h, _ = discovery.run_expected(_run((0.0, -20.0), 180.0, sphere=False), FAR_BOX, pts, w, r_goal=40.0,
                                        every_s=0.5)
    assert np.all(a == 0.0) and np.all(h == 0.0)


def test_the_sphere_view_reads_the_session():
    assert replay.sphere_view(_run((0, 0), 0.0)) == (True, 0.0)
    assert replay.sphere_view(_run((0, 0), 0.0, band=23.1)) == (True, 23.1)
    assert replay.sphere_view(_run((0, 0), 0.0, sphere=False)) == (False, 0.0)
    assert replay.sphere_view(SimpleNamespace()) == (False, 0.0)
