"""Levels (spec §7): a YAML file loads into a Level, which has a canonical form, a SHA-256 id, and an MJCF compile."""

import base64
import hashlib
import struct
import zlib
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from . import motion as M
from .marker import crosses
from .model import ROLL, TORSION

GEOM_SIZES = {"plane": 3, "box": 3, "sphere": 1, "capsule": 2}
MARKER_KEYS = {"sphere": {"pos", "radius"}, "box": {"pos", "size", "euler"}, "capsule": {"fromto", "radius"}}
TASK_MARKERS = {
    "reach_goal": ("goal",),
    "waypoint_race": ("waypoints",),
    "collect_items": ("items",),
    "unlock_goal": ("keys", "goal"),
    "survive": ("region",),
}
MARKER_LISTS = {"waypoints", "items", "keys"}  # task keys holding an ordered, non-empty list of markers
# Look constants (spec §11): colour by marker kind; the env hides collected markers and dims the locked goal
MARKER_RGBA = {
    "goal": (0.2, 0.9, 0.3, 0.4),
    "waypoints": (0.2, 0.5, 1.0, 0.4),
    "items": (1.0, 0.85, 0.1, 0.4),
    "keys": (1.0, 0.4, 0.1, 0.4),
    "region": (0.6, 0.3, 0.9, 0.25),
}
HAZARD_RGBA = (0.9, 0.1, 0.1, 0.5)  # level-wide, so the env never hides a hazard
LOCKED_RGBA = (0.5, 0.5, 0.5, 0.15)  # a marker that can't trigger yet: the unlock goal, a later waypoint
PASSED_RGBA = (0.2, 0.9, 0.3, 0.2)  # a passed waypoint: the goal's green, fainter
CONTACT = {"friction": 2, "solref": 2, "solimp": 5}  # optional per-geom contact values and their lengths
HFIELD_BASE = 1.0  # depth (m) of the solid slab under each hfield; MuJoCo needs one
# A kinematic body's mass and inertia: heavy, so a contact barely moves it within the one step before the core poses it
# again (a measured carry converged by about 1000 kg, see step.md)
BODY_MASS, BODY_INERTIA = 1000.0, 1000.0
BODY_GEOMS = {"box", "sphere", "capsule"}  # a plane or hfield can't move
TILE_WARN_RGBA = (0.9, 0.25, 0.1, 1.0)  # look constant: a touched tile's colour blends to this as its fall nears


@dataclass
class Level:
    kill_plane_z: float
    geoms: list
    tasks: dict
    hazards: list = field(default_factory=list)
    follow_yaw_deg: float = 0.0
    version: int = 1
    generator: dict | None = None  # provenance of a generated level; hand-authored files omit it
    bodies: list = field(default_factory=list)  # kinematic bodies; a level without them omits the key
    collapse: dict | None = None  # collapsing tiles {delay, tiles}; a level without them omits the key

    @classmethod
    def from_yaml(cls, text, base="."):
        """Load and validate a level, filling every default; raises ValueError on a malformed level.

        An hfield's `file` is a 16-bit grayscale PNG, relative to `base`."""
        return cls.from_dict(yaml.safe_load(text), base)

    @classmethod
    def from_dict(cls, d, base="."):
        """`from_yaml` on an already-parsed level."""
        _keys(d, {"kill_plane_z", "geoms", "tasks"}, {"version", "follow_yaw_deg", "hazards", "generator", "bodies", "collapse"}, "level")
        if d.get("version", 1) != 1:
            raise ValueError(f"level version {d['version']} is not 1")
        if not d["tasks"]:
            raise ValueError("a level needs at least one task")
        return cls(
            kill_plane_z=_num(d["kill_plane_z"]),
            geoms=[_geom(g, Path(base)) for g in d["geoms"]],
            tasks={name: _task(name, t) for name, t in d["tasks"].items()},
            hazards=[_marker(h, moving=True) for h in d.get("hazards") or []],
            follow_yaw_deg=_num(d.get("follow_yaw_deg", 0)),
            generator=_provenance(d["generator"]) if "generator" in d else None,
            bodies=[_body(b, Path(base)) for b in d.get("bodies") or []],
            collapse=_collapse(d["collapse"], Path(base)) if "collapse" in d else None,
        )

    @classmethod
    def load(cls, path):
        return cls.from_yaml(Path(path).read_text(), Path(path).parent)

    def canonical(self):
        return self._canonical

    # Cached on first read: a 256x256 hfield takes ~0.1 s to serialise, and the env reads both on every reset.
    # So edit a Level only before its first canonical() or id.
    @cached_property
    def _canonical(self):
        d = asdict(self)
        if d["generator"] is None:  # hand-authored: no block, so the id is the same as before provenance existed
            del d["generator"]
        if not d["bodies"]:  # likewise for a level without bodies
            del d["bodies"]
        if d["collapse"] is None:  # and without tiles
            del d["collapse"]
        return yaml.safe_dump(d, sort_keys=True)

    @cached_property
    def id(self):
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def movers(self):
        """(geom name, marker) of every marker that has a motion, named as `mjcf()` names its geom."""
        return [(f"hazard{i}", h) for i, h in enumerate(self.hazards) if "motion" in h] + [(f"{name}/{k}", task[k]) for name, task in self.tasks.items() for k in TASK_MARKERS[name]
                if k not in MARKER_LISTS and "motion" in task[k]]

    def mjcf(self):
        """MJCF sections for model.build: hfield assets, then worldbody geoms: the level's solids, the hazards, then
        each task's markers (translucent, no contact)."""
        assets, out = [], []
        for i, g in enumerate(self.geoms):
            if g["type"] == "hfield":
                samples = np.frombuffer(base64.b64decode(g["samples"]), "<u2")
                assets.append(
                    f'<hfield name="hfield{i}" nrow="{g["nrow"]}" ncol="{g["ncol"]}" size="{_s(g["size"])} {HFIELD_BASE!r}" '
                    f'elevation="{" ".join(map(str, samples.tolist()))}"/>'
                )
            out.append(_geom_mjcf(g, f'hfield="hfield{i}"' if g["type"] == "hfield" else None))
        # hazard names carry no "/", so the env's per-task marker looks pass them by
        out += [_marker_mjcf(h, f"hazard{i}", HAZARD_RGBA) for i, h in enumerate(self.hazards)]
        for name, task in self.tasks.items():
            for k in TASK_MARKERS[name]:
                if k in MARKER_LISTS:
                    out += [_marker_mjcf(m, f"{name}/{k}/{i}", MARKER_RGBA[k]) for i, m in enumerate(task[k])]
                else:
                    out.append(_marker_mjcf(task[k], f"{name}/{k}", MARKER_RGBA[k]))
        return "<asset>{}</asset>\n<worldbody>\n{}\n</worldbody>".format("\n".join(assets), "\n".join(out))

    def body_mjcf(self):
        """MJCF sections for model.build's `bodies`, after the view mocaps: each kinematic body `body<i>` (a free
        joint, gravity compensated, heavy) with its geoms, and a contact exclude from the world and from each other
        (the core poses them, so only the shell should touch them). "" for a level without bodies."""
        if not self.bodies:
            return ""
        names = [f"body{i}" for i in range(len(self.bodies))]
        out = [
            f'<body name="{n}" pos="{_s(b["pos"])}" gravcomp="1"><freejoint/>'
            f'<inertial pos="0 0 0" mass="{BODY_MASS!r}" diaginertia="{_s([BODY_INERTIA] * 3)}"/>'
            + "".join(_geom_mjcf(g) for g in b["geoms"]) + "</body>"
            for n, b in zip(names, self.bodies)
        ]
        ex = [f'<exclude body1="world" body2="{n}"/>' for n in names]
        ex += [f'<exclude body1="{a}" body2="{b}"/>' for i, a in enumerate(names) for b in names[i + 1:]]
        return "<worldbody>\n{}\n</worldbody>\n<contact>{}</contact>".format("\n".join(out), "".join(ex))

    def tile_mjcf(self):
        """MJCF sections for model.build's `bodies`, after `body_mjcf()`'s: each collapsing tile `tile<i>`, a kinematic
        body (as a kinematic body is) at its geom's pos, excluded from the world, the bodies, and each other. "" for a
        level without tiles."""
        if self.collapse is None:
            return ""
        tiles = [f"tile{i}" for i in range(len(self.collapse["tiles"]))]
        out = [
            f'<body name="{n}" pos="{_s(g["pos"])}" gravcomp="1"><freejoint/>'
            f'<inertial pos="0 0 0" mass="{BODY_MASS!r}" diaginertia="{_s([BODY_INERTIA] * 3)}"/>'
            + _geom_mjcf({**g, "pos": [0.0, 0.0, 0.0]}) + "</body>"
            for n, g in zip(tiles, self.collapse["tiles"])
        ]
        ex = [f'<exclude body1="{a}" body2="{t}"/>' for t in tiles
              for a in ["world", *(f"body{i}" for i in range(len(self.bodies)))]]
        ex += [f'<exclude body1="{a}" body2="{b}"/>' for i, a in enumerate(tiles) for b in tiles[i + 1:]]
        return "<worldbody>\n{}\n</worldbody>\n<contact>{}</contact>".format("\n".join(out), "".join(ex))


def _geom_mjcf(g, shape=None):
    """A level solid: group 0, contype and conaffinity 1 (the shell's outer sphere), contact values with priority."""
    rgba = " ".join(map(repr, g.get("rgba", [0.45, 0.5, 0.45, 1.0])))
    shape = shape or f'size="{_s(g["size"])}"'
    if CONTACT.keys() & g.keys():
        # priority over the shell's outer sphere, so these values apply as written; condim 6 keeps rolling
        slide, roll = g.get("friction", [1.0, ROLL])
        shape += f' priority="1" condim="6" friction="{slide!r} {TORSION!r} {roll!r}"'
        shape += "".join(f' {k}="{_s(g[k])}"' for k in ("solref", "solimp") if k in g)
    return (f'<geom type="{g["type"]}" pos="{_s(g["pos"])}" {shape} euler="{_s(g["euler"])}" '
            f'rgba="{rgba}" contype="1" conaffinity="1" group="0"/>')


def _marker_mjcf(m, name, rgba):
    if m["type"] == "sphere":
        shape = f'pos="{_s(m["pos"])}" size="{m["radius"]!r}"'
    elif m["type"] == "box":
        shape = f'pos="{_s(m["pos"])}" size="{_s(m["size"])}" euler="{_s(m["euler"])}"'
    else:
        shape = f'fromto="{_s(m["fromto"])}" size="{m["radius"]!r}"'
    return f'<geom name="{name}" type="{m["type"]}" {shape} rgba="{_s(rgba)}" contype="0" conaffinity="0" group="0"/>'


def _s(v):
    return " ".join(map(repr, v))


def _keys(d, required, optional, what):
    if not isinstance(d, dict):
        raise ValueError(f"{what} must be a mapping")
    missing, unknown = required - d.keys(), d.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"{what}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{x!r} is not a number")
    return float(x)


def _vec(v, n):
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"{v!r} is not a list of {n} numbers")
    return [_num(x) for x in v]


def _geom(g, base):
    if isinstance(g, dict) and g.get("type") == "hfield":
        return _hfield(g, base)
    _keys(g, {"type", "size"}, {"pos", "euler", "rgba", *CONTACT}, "geom")
    if g["type"] not in GEOM_SIZES:
        raise ValueError(f"geom type {g['type']!r} is not one of {sorted(GEOM_SIZES)}")
    out = {
        "type": g["type"],
        "pos": _vec(g.get("pos", [0, 0, 0]), 3),
        "size": _vec(g["size"], GEOM_SIZES[g["type"]]),
        "euler": _vec(g.get("euler", [0, 0, 0]), 3),
    }
    if "rgba" in g:  # a look value, so an unset colour stays unset rather than becoming part of the id
        out["rgba"] = _vec(g["rgba"], 4)
    for k, n in CONTACT.items():  # unset stays unset: the geom then mixes with the shell as MuJoCo does
        if k in g:
            out[k] = _vec(g[k], n)
    return out


def _body(b, base):
    """A kinematic body: `pos` (its frame's origin, which a spin turns about), `geoms` in that frame, a `motion`."""
    _keys(b, {"geoms", "motion"}, {"pos"}, "body")
    if not isinstance(b["geoms"], list) or not b["geoms"]:
        raise ValueError("a body needs a non-empty list of geoms")
    geoms = [_geom(g, base) for g in b["geoms"]]
    if bad := {g["type"] for g in geoms} - BODY_GEOMS:
        raise ValueError(f"body geom types {sorted(bad)} are not in {sorted(BODY_GEOMS)}")
    return {"pos": _vec(b.get("pos", [0, 0, 0]), 3), "geoms": geoms, "motion": M.load(b["motion"])}


def _collapse(c, base):
    """Collapsing tiles: `tiles` (box, sphere or capsule geoms in world coordinates), each falling `delay` s after the
    shell first touches it."""
    _keys(c, {"delay", "tiles"}, set(), "collapse")
    if not isinstance(c["tiles"], list) or not c["tiles"]:
        raise ValueError("collapse needs a non-empty list of tiles")
    tiles = [_geom(g, base) for g in c["tiles"]]
    if bad := {g["type"] for g in tiles} - BODY_GEOMS:
        raise ValueError(f"tile types {sorted(bad)} are not in {sorted(BODY_GEOMS)}")
    if (delay := _num(c["delay"])) < 0:
        raise ValueError(f"collapse delay {delay} is negative")
    return {"delay": delay, "tiles": tiles}


def _marker(m, moving=False):
    """A marker; `moving` lets it take a `motion` (absent stays absent, so a still marker keeps its id)."""
    kind = m.get("type") if isinstance(m, dict) else None
    if kind not in MARKER_KEYS:
        raise ValueError(f"marker type {kind!r} is not one of {sorted(MARKER_KEYS)}")
    optional = {"euler"} if kind == "box" else set()
    _keys(m, MARKER_KEYS[kind] - optional | {"type"}, optional | ({"motion"} if moving else set()), f"{kind} marker")
    out = {"type": kind}
    for k in MARKER_KEYS[kind]:
        if k == "radius":
            out[k] = _num(m[k])
        else:
            out[k] = _vec(m.get(k, [0, 0, 0]), 6 if k == "fromto" else 3)
    if "motion" in m:
        out["motion"] = M.load(m["motion"])
    return out


def _task(name, t):
    if name not in TASK_MARKERS:
        raise ValueError(f"task {name!r} is not one of {sorted(TASK_MARKERS)}")
    _keys(t, {"spawn", "time_limit_s", *TASK_MARKERS[name]}, set(), name)
    _keys(t["spawn"], {"pos"}, {"yaw_deg"}, f"{name} spawn")
    out = {
        "spawn": {"pos": _vec(t["spawn"]["pos"], 3), "yaw_deg": _num(t["spawn"].get("yaw_deg", 0))},
        "time_limit_s": _num(t["time_limit_s"]),
    }
    for k in TASK_MARKERS[name]:
        if k in MARKER_LISTS:
            if not isinstance(t[k], list) or not t[k]:
                raise ValueError(f"{name} {k} must be a non-empty list of markers")
            out[k] = [_marker(m) for m in t[k]]
        else:
            out[k] = _marker(t[k], moving=k == "region")
    if name == "survive" and not crosses(out["region"], out["spawn"]["pos"], out["spawn"]["pos"], t=0.0):
        raise ValueError(f"survive spawn {out['spawn']['pos']} is outside its region")
    return out


def _provenance(p):
    """`generator: {name, version, task, seed, difficulty}` of a generated level."""
    _keys(p, {"name", "version", "task", "seed", "difficulty"}, set(), "generator")
    if not isinstance(p["name"], str) or not all(type(p[k]) is int for k in ("version", "seed")):
        raise ValueError("generator name must be a string, and version and seed integers")
    if p["task"] not in TASK_MARKERS:
        raise ValueError(f"generator task {p['task']!r} is not one of {sorted(TASK_MARKERS)}")
    return {"name": p["name"], "version": p["version"], "task": p["task"], "seed": p["seed"],
            "difficulty": _num(p["difficulty"])}


def _hfield(g, base):
    """An hfield geom; the canonical form holds its samples inline (`nrow`, `ncol`, base64 little-endian uint16)."""
    common = {"type", "size", "pos", "euler", "rgba", *CONTACT}
    if "file" in g:
        _keys(g, {"type", "size", "file"}, common, "hfield")
        rows = _png16(base / g["file"])
    else:
        _keys(g, {"type", "size", "nrow", "ncol", "samples"}, common, "hfield")
        if not all(type(g[k]) is int for k in ("nrow", "ncol")) or not isinstance(g["samples"], str):
            raise ValueError("hfield nrow and ncol must be integers and samples a base64 string")
        rows = np.frombuffer(base64.b64decode(g["samples"], validate=True), "<u2").reshape(g["nrow"], g["ncol"])
    out = _geom({k: v for k, v in g.items() if k in common} | {"type": "box"}, base) | {"type": "hfield"}
    nrow, ncol = rows.shape
    if nrow > 256 or ncol > 256:
        raise ValueError(f"hfield grid {nrow}x{ncol} is over 256x256")
    hx, hy, top = out["size"]
    if top <= 0:
        raise ValueError(f"hfield max height {top} is not positive")
    if min(nrow, ncol) < 2 or 2 * hx / (ncol - 1) < 0.25 or 2 * hy / (nrow - 1) < 0.25:
        raise ValueError(f"hfield cells of a {nrow}x{ncol} grid over {2 * hx}x{2 * hy} m are under 0.25 m")
    return out | {"nrow": nrow, "ncol": ncol, "samples": base64.b64encode(rows.astype("<u2").tobytes()).decode()}


def _png16(path):
    """The samples of a non-interlaced 16-bit grayscale PNG, as (rows, columns) uint16."""
    b = Path(path).read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    i, idat = 8, b""
    while i < len(b):
        n, kind = struct.unpack(">I4s", b[i:i + 8])
        body = b[i + 8:i + 8 + n]
        if kind == b"IHDR":
            w, h, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, colour, interlace) != (16, 0, 0):
                raise ValueError(f"{path} is not a non-interlaced 16-bit grayscale PNG")
        elif kind == b"IDAT":
            idat += body
        i += 12 + n
    raw, stride = zlib.decompress(idat), 2 * w
    out, prev = bytearray(), bytearray(stride)
    for r in range(h):
        f, row = raw[r * (stride + 1)], bytearray(raw[r * (stride + 1) + 1:(r + 1) * (stride + 1)])
        for x in range(stride):  # undo the PNG filter, 2 bytes per pixel
            a = row[x - 2] if x >= 2 else 0
            c = prev[x - 2] if x >= 2 else 0
            if f == 1:
                row[x] = (row[x] + a) & 255
            elif f == 2:
                row[x] = (row[x] + prev[x]) & 255
            elif f == 3:
                row[x] = (row[x] + (a + prev[x]) // 2) & 255
            elif f == 4:
                p = a + prev[x] - c
                pa, pb, pc = abs(p - a), abs(p - prev[x]), abs(p - c)
                row[x] = (row[x] + (a if pa <= pb and pa <= pc else prev[x] if pb <= pc else c)) & 255
        out += row
        prev = row
    return np.frombuffer(bytes(out), ">u2").reshape(h, w)
