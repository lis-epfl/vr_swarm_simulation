"""The ScaledCity building designs (building_catalogue.json, from Tools/Swarm/Export building catalogue) and the
one design the study builds its cities from.

    python -B -m citysim.catalogue                          # the table and the pick
    python -B -m citysim.catalogue --mix DiamondCityWorld   # a tile city's building mix (citygen --mix)

A design's w and d are its footprint along the building's local X and Y at the ScaledCity scale (metres), h its
height (50 m for every design). The study uses ONE design everywhere -- city, goal occluder, goal ring -- at a
few sizes, so the footprint distribution is one number per level and "building size" is a factor rather than
a mixture: the design closest to the median footprint of the 210 buildings in the 48 patches, in log(long side)
and log(short side) together, among the BnP_Apartment_ and BnP_Small_Building_ designs with at least four
instances (ordinary mid-rise blocks, well represented in the pack). Ties break by name, so the pick is stable.

A MIX (city_mix) is the other way to build: every building drawn from a tile city's own designs, in proportion to
how often each stands in that city. It is counted off the city's city_obstacles_<city>.json (Tools/Swarm/Export
city obstacles), so it is the city as exported, diamonds' copies included -- re-export after restyling it.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

from citysim import EXPERIMENT_DIR  # noqa: E402

CATALOGUE_JSON = EXPERIMENT_DIR / "building_catalogue.json"
STUDY_FAMILIES = ("BnP_Apartment_", "BnP_Small_Building_")
MIN_INSTANCES = 4


@dataclass(frozen=True)
class Design:
    design: str
    family: str
    w: float            # along the building's local X, metres at x1
    d: float            # along its local Y
    h: float
    count: int

    @property
    def long(self):
        return max(self.w, self.d)

    @property
    def short(self):
        return min(self.w, self.d)

    def footprint(self, size, yaw_deg=0):
        """(half x, half z) of the footprint at horizontal scale `size`, yawed 0 or 90 degrees."""
        hx, hz = 0.5 * self.w * size, 0.5 * self.d * size
        return (hz, hx) if int(round(yaw_deg)) % 180 == 90 else (hx, hz)


def load(path=CATALOGUE_JSON):
    d = json.loads(path.read_text(encoding="utf-8"))
    return [Design(e["design"], e["family"], float(e["w"]), float(e["d"]), float(e["h"]), int(e["count"]))
            for e in d["designs"]]


def median_footprint(designs):
    """(long, short) median over every building instance (designs weighted by their count)."""
    longs = np.repeat([x.long for x in designs], [x.count for x in designs])
    shorts = np.repeat([x.short for x in designs], [x.count for x in designs])
    return float(np.median(longs)), float(np.median(shorts))


def study_design(designs=None):
    """The design the study builds with (see the module docstring for the rule)."""
    designs = designs or load()
    ml, ms = median_footprint(designs)
    cands = [x for x in designs if x.family in STUDY_FAMILIES and x.count >= MIN_INSTANCES]
    if not cands:
        raise ValueError("no candidate design in the catalogue")
    return min(cands, key=lambda x: (math.hypot(math.log(x.long / ml), math.log(x.short / ms)), x.design))


def get(name, designs=None):
    for x in designs or load():
        if x.design == name:
            return x
    raise KeyError(f"no design {name!r} in {CATALOGUE_JSON.name}")


def design_of(box_name):
    """The catalogue design a tile city's building is an instance of: its name less any ' (label)' suffix (the
    diamonds' copies, e.g. 'Skyscraper_D_003 (plaza A_West)') and its instance number."""
    return re.sub(r"_\d+$", "", re.sub(r"\s*\(.*\)$", "", box_name))


def city_mix(city, designs=None):
    """[(Design, count)] of a tile city's buildings, by design name: how often each catalogue design stands in
    city_obstacles_<city>.json. Raises if a building is not a catalogue design (re-export the catalogue)."""
    path = EXPERIMENT_DIR / f"city_obstacles_{city}.json"
    if not path.exists():
        raise FileNotFoundError(f"no {path.name}: Tools/Swarm/Export city obstacles in {city} writes it")
    by_name = {x.design: x for x in designs or load()}
    counts = Counter(design_of(b["name"]) for b in json.loads(path.read_text(encoding="utf-8-sig"))["boxes"])
    unknown = sorted(n for n in counts if n not in by_name)
    if unknown:
        raise KeyError(f"{path.name} has buildings of no catalogue design: {', '.join(unknown)}")
    return [(by_name[n], counts[n]) for n in sorted(counts)]


if __name__ == "__main__":
    if "--mix" in sys.argv:
        city = sys.argv[sys.argv.index("--mix") + 1]
        mix = city_mix(city)
        n = sum(c for _, c in mix)
        area = sum(x.w * x.d * c for x, c in mix) / n
        print(f"{city}: {n} buildings in {len(mix)} designs, mean footprint {area:.1f} m^2")
        for x, c in sorted(mix, key=lambda e: (-e[1], e[0].design)):
            print(f"  {x.design:24s} {x.w:6.2f} x {x.d:5.2f}  {c:3d}  {c / n:6.1%}")
        sys.exit(0)
    ds = load()
    ml, ms = median_footprint(ds)
    pick = study_design(ds)
    print(f"{len(ds)} designs, {sum(x.count for x in ds)} buildings; median footprint {ml:.2f} x {ms:.2f} m")
    for x in sorted(ds, key=lambda x: (x.family not in STUDY_FAMILIES, x.design)):
        dist = math.hypot(math.log(x.long / ml), math.log(x.short / ms))
        flag = " <- study design" if x is pick else ""
        print(f"  {x.design:24s} {x.long:6.2f} x {x.short:5.2f}  n {x.count:2d}  dist {dist:.3f}{flag}")
    print(f"study design: {pick.design} ({pick.w:.2f} x {pick.d:.2f} m at x1; x2 {2 * pick.long:.1f} x "
          f"{2 * pick.short:.1f}, x4 {4 * pick.long:.1f} x {4 * pick.short:.1f})")
