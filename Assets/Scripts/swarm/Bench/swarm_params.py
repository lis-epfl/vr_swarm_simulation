"""The swarm's parameters, read from where Unity reads them, so no tool here hardcodes a gain.

SwarmManager settings resolve in the order Unity applies them, then the experiment's own records:
  1. the C# field initialisers in SwarmManager.cs   (what a scene that never saved a field gets)
  2. the scene's SwarmManager block                 (Assets/Scenes/<scene>.unity)
  3. test.json "swarmParams"                        (what a test flew, written by hand for old tests)
  4. <stem>_swarm.json                              (ExperimentRecorder's record of a run, where it exists)
  5. explicit overrides                             (--set on the command line)
Each value carries its source, so `swarm_replica.py params` can show where a number came from.

The airframe (VelocityControl, StateFinder, Rigidbody) comes from DroneReduced.prefab over the C#
initialisers, which is the prefab every scene spawns.

Run with the `stitching` env (numpy/pandas). Nothing here imports Unity or writes anything.
"""
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ASSETS = REPO / "Assets"
SWARM_MANAGER_CS = ASSETS / "Scripts" / "swarm" / "SwarmManager.cs"
VELOCITY_CONTROL_CS = ASSETS / "Scripts" / "VelocityControl" / "VelocityControl.cs"
STATE_FINDER_CS = ASSETS / "Scripts" / "VelocityControl" / "StateFinder.cs"
DRONE_PREFAB = ASSETS / "Prefabs" / "DroneReduced.prefab"
EXPERIMENT_DIR = ASSETS / "Scripts" / "Experiment"
DEFAULT_ROOT = Path.home() / "AppData" / "LocalLow" / "UAVS@BERKELEY" / "DroneSim" / "experiment"
DEFAULT_CITY = "ScaledCityWorld"   # analyse.py's default for a test.json without "city"


class Field:
    def __init__(self, name, kind, default, enum=None):
        self.name, self.kind, self.default, self.enum = name, kind, default, enum

    def convert(self, value, where=""):
        """A value from YAML, JSON or the command line, as this field's Python type."""
        ctx = f" ({where})" if where else ""
        if self.kind == "float":
            return float(value)
        if self.kind == "bool":
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "1"):
                    return True
                if v in ("false", "0"):
                    return False
                raise ValueError(f"{self.name} is a bool, not {value!r}{ctx}")
            if value in (0, 1, True, False):
                return bool(value)
            raise ValueError(f"{self.name} is a bool, not {value!r}{ctx}")
        if self.kind == "enum" and isinstance(value, str) and not _is_number(value):
            if value not in self.enum:
                raise ValueError(f"{self.name}: {value!r} is not one of {self.enum}{ctx}")
            return self.enum.index(value)
        f = float(value)
        if f != int(f):
            raise ValueError(f"{self.name} takes an integer, not {value!r}{ctx}")
        i = int(f)
        if self.kind == "enum" and not 0 <= i < len(self.enum):
            raise ValueError(f"{self.name}: {i} is not one of {list(enumerate(self.enum))}{ctx}")
        return i


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------------------------------ C# source

_ENUM_RE = re.compile(r"enum\s+(\w+)\s*\{([^}]*)\}", re.S)
_FIELD_RE = re.compile(r"^\s*(public|private|protected)?\s*(?:\[[^\]]*\]\s*)*(float|int|bool|\w+)\s+(\w+)\s*(?:=\s*([^;]+))?;",
                       re.M)


def _strip_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _literal(text):
    t = text.strip()
    if t in ("true", "false"):
        return t == "true"
    t = t.rstrip("fFdD")
    try:
        return float(t)
    except ValueError:
        return None


def csharp_fields(path, public_only=True):
    """{name: Field} for the numeric/bool/enum instance fields a C# component declares, with their
    initialisers. Fields with no literal initialiser default as C# does (0, false, the first enum value)."""
    src = _strip_comments(Path(path).read_text(encoding="utf-8-sig"))
    enums = {}
    for name, body in _ENUM_RE.findall(src):
        members = [m.split("=")[0].strip() for m in body.split(",") if m.strip()]
        enums[name] = members
    out = OrderedDict()
    for access, typ, name, init in _FIELD_RE.findall(src):
        if public_only and access != "public":
            continue
        if typ in ("float", "int", "bool"):
            kind = typ
        elif typ in enums:
            kind = "enum"
        else:
            continue
        if kind == "enum":
            default = 0
            if init:
                member = init.strip().split(".")[-1]
                default = enums[typ].index(member) if member in enums[typ] else 0
        else:
            lit = _literal(init) if init else None
            default = {"float": 0.0, "int": 0, "bool": False}[kind] if lit is None else lit
        f = Field(name, kind, None, enums.get(typ))
        f.default = f.convert(default)
        out[name] = f
    return out


def swarm_manager_fields():
    return csharp_fields(SWARM_MANAGER_CS)


# ------------------------------------------------------------------------------------------ Unity YAML

def _meta_guid(cs_path):
    text = Path(str(cs_path) + ".meta").read_text(encoding="utf-8")
    return re.search(r"guid:\s*([0-9a-f]+)", text).group(1)


def _yaml_docs(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return text.split("\n--- ")


def _top_level_scalars(doc):
    """The component's own fields: two-space-indented `key: value` lines with a scalar value."""
    out = {}
    for line in doc.splitlines():
        m = re.match(r"^  (\w+): (.*)$", line)
        if m and not m.group(2).startswith(("{", "[")) and m.group(2) != "":
            out[m.group(1)] = m.group(2).strip()
    return out


def component_values(path, cs_path):
    """Scalar fields of the first MonoBehaviour in a scene/prefab whose script is cs_path."""
    guid = _meta_guid(cs_path)
    for doc in _yaml_docs(path):
        if f"guid: {guid}" in doc and "m_Script:" in doc:
            return _top_level_scalars(doc)
    raise LookupError(f"{Path(path).name} has no {Path(cs_path).stem} component")


def rigidbody_values(path):
    for doc in _yaml_docs(path):
        if "\nRigidbody:" in doc:
            return _top_level_scalars(doc)
    raise LookupError(f"{Path(path).name} has no Rigidbody")


def scene_path(scene):
    p = Path(scene)
    return p if p.suffix == ".unity" else ASSETS / "Scenes" / f"{scene}.unity"


# ------------------------------------------------------------------------------------------ resolution

class Params(OrderedDict):
    """Resolved values; .source[name] says which layer set each one."""

    def __init__(self):
        super().__init__()
        self.source = {}
        self.fields = {}

    def set(self, name, value, source):
        self[name] = value
        self.source[name] = source

    def describe(self, names=None):
        rows = []
        for n in names or self:
            rows.append(f"  {n:28s} {str(self[n]):>12s}   {self.source[n]}")
        return "\n".join(rows)


def resolve_swarm(scene, test_dir=None, stem=None, overrides=None):
    """SwarmManager settings for `scene`, then a test's and a run's records, then overrides."""
    fields = swarm_manager_fields()
    p = Params()
    p.fields = fields
    for n, f in fields.items():
        p.set(n, f.default, "SwarmManager.cs default")

    raw = component_values(scene_path(scene), SWARM_MANAGER_CS)
    for n, v in raw.items():
        if n in fields:
            p.set(n, fields[n].convert(v, scene), f"{Path(scene).stem}.unity")

    if test_dir is not None:
        manifest = Path(test_dir) / "test.json"
        if manifest.exists():
            m = json.loads(manifest.read_text(encoding="utf-8-sig"))
            for n, v in (m.get("swarmParams") or {}).items():
                if n not in fields:
                    raise ValueError(f"{manifest}: swarmParams names {n!r}, which SwarmManager does not have")
                p.set(n, fields[n].convert(v, str(manifest)), "test.json swarmParams")
        if stem:
            rec = Path(test_dir) / f"{stem}_swarm.json"
            if rec.exists():
                d = json.loads(rec.read_text(encoding="utf-8-sig"))
                for n, v in (d.get("swarmManager") or {}).items():
                    if n in fields:
                        if isinstance(v, float):
                            v = float(f"{v:.7g}")   # JsonUtility writes float32 as 0.800000011920929
                        p.set(n, fields[n].convert(v, rec.name), f"{rec.name}")

    for n, v in (overrides or {}).items():
        if n not in fields:
            raise ValueError(f"SwarmManager has no setting {n!r}")
        p.set(n, fields[n].convert(v, "override"), "override")
    return p


# The airframe fields the replica needs, from the component that owns each. Private fields are read from
# the C# initialiser (they are not serialised, so the prefab cannot change them).
_VC_FIELDS = ["maxPitch", "maxRoll", "maxAlpha", "maxSpeed", "timeConstantAcceleration",
              "SwarmAccelFilterCoefficient", "timeConstantOmegaXYRate", "timeConstantAlphaRate", "gravity"]
_SF_FIELDS = ["enableStateNoise", "positionNoiseSigma", "velocityNoiseSigma", "attitudeNoiseSigma"]


def resolve_airframe(prefab=DRONE_PREFAB):
    p = Params()
    for cs, names, values in (
            (VELOCITY_CONTROL_CS, _VC_FIELDS, component_values(prefab, VELOCITY_CONTROL_CS)),
            (STATE_FINDER_CS, _SF_FIELDS, component_values(prefab, STATE_FINDER_CS))):
        fields = csharp_fields(cs, public_only=False)
        for n in names:
            f = fields[n]
            if n in values:
                p.set(n, f.convert(values[n]), Path(prefab).name)
            else:
                p.set(n, f.default, f"{Path(cs).name} default")
    rb = rigidbody_values(prefab)
    p.set("drag", float(rb["m_Drag"]), f"{Path(prefab).name} Rigidbody")
    p.set("angularDrag", float(rb["m_AngularDrag"]), f"{Path(prefab).name} Rigidbody")
    tm = (REPO / "ProjectSettings" / "TimeManager.asset").read_text(encoding="utf-8")
    p.set("fixedDeltaTime", float(re.search(r"Fixed Timestep:\s*([\d.]+)", tm).group(1)), "TimeManager.asset")
    return p


def test_city(test_dir):
    m = Path(test_dir) / "test.json"
    if not m.exists():
        return DEFAULT_CITY
    return json.loads(m.read_text(encoding="utf-8-sig")).get("city", DEFAULT_CITY)


def parse_assignments(items):
    """['a=0.8', 'hollowSwarmCore=true'] -> {'a': '0.8', ...} (converted later, per field)."""
    out = {}
    for it in items or []:
        for part in it.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"expected name=value, got {part!r}")
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    import sys
    scene = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CITY
    print(f"SwarmManager in {scene}:")
    print(resolve_swarm(scene).describe())
    print("\nAirframe (DroneReduced.prefab):")
    print(resolve_airframe().describe())
