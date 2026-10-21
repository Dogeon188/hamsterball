"""Motion: a mover's pose as a pure function of its motion and sim time, so the core state (which holds the time)
is enough to know where every mover is.

A motion is the canonical dict `load` returns: `path` (a closed polyline of offsets, walked at constant speed) or
`oscillate` (a sine offset along a unit axis), each with a `period` (s), a `phase` (a fraction of the period), and a
`spin` about z (deg/s). The pose is an offset from the mover's authored pose plus a yaw about its authored centre.
"""

import math

import numpy as np

FORMS = {"path": {"points"}, "oscillate": {"axis", "amplitude"}}


def load(m):
    """Validate a motion and fill its defaults; raises ValueError on a malformed one."""
    from .level import _keys, _num, _vec  # level imports this module

    kind = m.get("type") if isinstance(m, dict) else None
    if kind not in FORMS:
        raise ValueError(f"motion type {kind!r} is not one of {sorted(FORMS)}")
    _keys(m, FORMS[kind] | {"type", "period"}, {"phase", "spin"}, f"{kind} motion")
    out = {"type": kind, "period": _num(m["period"]), "phase": _num(m.get("phase", 0)) % 1.0, "spin": _num(m.get("spin", 0))}
    if out["period"] <= 0:
        raise ValueError(f"motion period {out['period']} is not positive")
    if kind == "path":
        if not isinstance(m["points"], list) or len(m["points"]) < 2:
            raise ValueError("a path needs at least two points")
        out["points"] = [_vec(p, 3) for p in m["points"]]
        if _length(out["points"]) == 0:
            raise ValueError("a path's points are all the same")
    else:
        axis = np.array(_vec(m["axis"], 3))
        if not np.linalg.norm(axis):
            raise ValueError("an oscillate axis is zero")
        out["axis"] = (axis / np.linalg.norm(axis)).tolist()  # a direction, so [2, 0, 0] and [1, 0, 0] share an id
        out["amplitude"] = _num(m["amplitude"])
    return out


def pose(motion, t):
    """(offset, yaw): the mover's offset (3-vector, m) from its authored pose and its yaw (rad) at sim time t."""
    u = (t / motion["period"] + motion["phase"]) % 1.0
    yaw = math.radians(motion["spin"] * t)
    if motion["type"] == "oscillate":
        return motion["amplitude"] * math.sin(2 * math.pi * u) * np.array(motion["axis"]), yaw
    pts = np.array(motion["points"] + motion["points"][:1])  # closed
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return np.array([np.interp(u * s[-1], s, pts[:, i]) for i in range(3)]), yaw


def _length(points):
    pts = np.array(points + points[:1])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
