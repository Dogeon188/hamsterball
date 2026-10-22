"""Motion: a mover's pose as a pure function of its motion and sim time, so the core state (which holds the time)
is enough to know where every mover is.

A motion is the canonical dict `load` returns: `path` (a closed polyline of offsets, walked at constant speed) or
`oscillate` (a sine offset along a unit axis), each with a `period` (s), a `phase` (a fraction of the period), and a
`spin` about z (deg/s). The pose is an offset from the mover's authored pose plus a yaw about its authored centre.
A body's motion can also `tilt`: a sine rocking about a horizontal axis of the body's frame.
"""

import math

import mujoco
import numpy as np

FORMS = {"path": {"points"}, "oscillate": {"axis", "amplitude"}}


# Validators, shared with level (which imports this module)
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


def load(m, tilt=False):
    """Validate a motion and fill its defaults; raises ValueError on a malformed one. `tilt` lets it take a `tilt`
    (only a body can turn off z; absent stays absent, so a motion without one keeps its id)."""
    kind = m.get("type") if isinstance(m, dict) else None
    if kind not in FORMS:
        raise ValueError(f"motion type {kind!r} is not one of {sorted(FORMS)}")
    _keys(m, FORMS[kind] | {"type", "period"}, {"phase", "spin"} | ({"tilt"} if tilt else set()), f"{kind} motion")
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
        out["axis"] = _unit(axis)  # a direction, so [2, 0, 0] and [1, 0, 0] share an id
        out["amplitude"] = _num(m["amplitude"])
    if "tilt" in m:
        out["tilt"] = _tilt(m["tilt"], out["period"])
    return out


def _tilt(t, period):
    """`{axis: [x, y, 0], amplitude (deg), period (default the motion's), phase: 0}`."""
    _keys(t, {"axis", "amplitude"}, {"period", "phase"}, "tilt")
    axis = np.array(_vec(t["axis"], 3))
    if axis[2] or not np.linalg.norm(axis):
        raise ValueError(f"a tilt axis {axis.tolist()} is not a non-zero horizontal vector")
    out = {"axis": _unit(axis), "amplitude": _num(t["amplitude"]),
           "period": _num(t.get("period", period)), "phase": _num(t.get("phase", 0)) % 1.0}
    if out["period"] <= 0:
        raise ValueError(f"tilt period {out['period']} is not positive")
    return out


def _unit(v):
    """v normalised, left as is when it already is one: a replay reloads the canonical level, and normalising a
    unit vector again can move its last bits."""
    n = np.linalg.norm(v)
    return (v if abs(n - 1) < 1e-12 else v / n).tolist()


def quat(motion, t):
    """The mover's orientation at sim time t: its yaw about z, then (a body's) tilt about its own horizontal axis."""
    yaw = math.radians(motion["spin"] * t)
    q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    if "tilt" not in motion:
        return q
    tl = motion["tilt"]
    a = math.radians(tl["amplitude"]) * math.sin(2 * math.pi * ((t / tl["period"] + tl["phase"]) % 1.0))
    out = np.empty(4)
    mujoco.mju_mulQuat(out, q, np.array([math.cos(a / 2), *(math.sin(a / 2) * np.array(tl["axis"]))]))
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
