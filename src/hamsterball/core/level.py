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
LOCKED_GOAL_RGBA = (0.5, 0.5, 0.5, 0.15)
CONTACT = {"friction": 2, "solref": 2, "solimp": 5}  # optional per-geom contact values and their lengths
HFIELD_BASE = 1.0  # depth (m) of the solid slab under each hfield; MuJoCo needs one


@dataclass
class Level:
    kill_plane_z: float
    geoms: list
    tasks: dict
    hazards: list = field(default_factory=list)
    follow_yaw_deg: float = 0.0
    version: int = 1

    @classmethod
    def from_yaml(cls, text, base="."):
        """Load and validate a level, filling every default; raises ValueError on a malformed level.

        An hfield's `file` is a 16-bit grayscale PNG, relative to `base`."""
        d = yaml.safe_load(text)
        _keys(d, {"kill_plane_z", "geoms", "tasks"}, {"version", "follow_yaw_deg", "hazards"}, "level")
        if d.get("version", 1) != 1:
            raise ValueError(f"level version {d['version']} is not 1")
        if not d["tasks"]:
            raise ValueError("a level needs at least one task")
        return cls(
            kill_plane_z=_num(d["kill_plane_z"]),
            geoms=[_geom(g, Path(base)) for g in d["geoms"]],
            tasks={name: _task(name, t) for name, t in d["tasks"].items()},
            hazards=[_marker(h) for h in d.get("hazards") or []],
            follow_yaw_deg=_num(d.get("follow_yaw_deg", 0)),
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
        return yaml.safe_dump(asdict(self), sort_keys=True)

    @cached_property
    def id(self):
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def mjcf(self):
        """MJCF sections for model.build: hfield assets, then worldbody geoms: the level's solids, the hazards, then
        each task's markers (translucent, no contact)."""
        assets, out = [], []
        for i, g in enumerate(self.geoms):
            rgba = " ".join(map(repr, g.get("rgba", [0.45, 0.5, 0.45, 1.0])))
            if g["type"] == "hfield":
                samples = np.frombuffer(base64.b64decode(g["samples"]), "<u2")
                assets.append(
                    f'<hfield name="hfield{i}" nrow="{g["nrow"]}" ncol="{g["ncol"]}" size="{_s(g["size"])} {HFIELD_BASE!r}" '
                    f'elevation="{" ".join(map(str, samples.tolist()))}"/>'
                )
                shape = f'hfield="hfield{i}"'
            else:
                shape = f'size="{_s(g["size"])}"'
            if CONTACT.keys() & g.keys():
                # priority over the shell's outer sphere, so these values apply as written; condim 6 keeps rolling
                slide, roll = g.get("friction", [1.0, ROLL])
                shape += f' priority="1" condim="6" friction="{slide!r} {TORSION!r} {roll!r}"'
                shape += "".join(f' {k}="{_s(g[k])}"' for k in ("solref", "solimp") if k in g)
            out.append(
                f'<geom type="{g["type"]}" pos="{_s(g["pos"])}" {shape} euler="{_s(g["euler"])}" '
                f'rgba="{rgba}" contype="1" conaffinity="1" group="0"/>'
            )
        # hazard names carry no "/", so the env's per-task marker looks pass them by
        out += [_marker_mjcf(h, f"hazard{i}", HAZARD_RGBA) for i, h in enumerate(self.hazards)]
        for name, task in self.tasks.items():
            for k in TASK_MARKERS[name]:
                if k in MARKER_LISTS:
                    out += [_marker_mjcf(m, f"{name}/{k}/{i}", MARKER_RGBA[k]) for i, m in enumerate(task[k])]
                else:
                    out.append(_marker_mjcf(task[k], f"{name}/{k}", MARKER_RGBA[k]))
        return "<asset>{}</asset>\n<worldbody>\n{}\n</worldbody>".format("\n".join(assets), "\n".join(out))


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


def _marker(m):
    kind = m.get("type") if isinstance(m, dict) else None
    if kind not in MARKER_KEYS:
        raise ValueError(f"marker type {kind!r} is not one of {sorted(MARKER_KEYS)}")
    optional = {"euler"} if kind == "box" else set()
    _keys(m, MARKER_KEYS[kind] - optional | {"type"}, optional, f"{kind} marker")
    out = {"type": kind}
    for k in MARKER_KEYS[kind]:
        if k == "radius":
            out[k] = _num(m[k])
        else:
            out[k] = _vec(m.get(k, [0, 0, 0]), 6 if k == "fromto" else 3)
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
            out[k] = _marker(t[k])
    if name == "survive" and not crosses(out["region"], out["spawn"]["pos"], out["spawn"]["pos"]):
        raise ValueError(f"survive spawn {out['spawn']['pos']} is outside its region")
    return out


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
