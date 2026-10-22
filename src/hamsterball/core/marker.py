"""The marker test (spec §9): does the shell centre's segment over one env step cross a marker's volume?

Markers are the canonical dicts of level.py. A segment of zero length is a point test. A marker attached to a body
is authored in that body's frame, which the caller passes as `frame`, the body's world (pos, quat).
"""

import math

import mujoco
import numpy as np

from .motion import pose


def centre(marker, t=0.0, frame=None):
    """The point a task steers toward at sim time t: a sphere's or box's position, a capsule's midpoint, plus the
    marker's motion offset. It is also the pivot the marker's spin turns about. With `frame`, in world coordinates."""
    if marker["type"] == "capsule":
        f = marker["fromto"]
        c = np.array([(f[0] + f[3]) / 2, (f[1] + f[4]) / 2, (f[2] + f[5]) / 2])
    else:
        c = np.array(marker["pos"])
    if frame is not None:
        return frame[0] + _mat(frame[1]) @ c
    return c + pose(marker["motion"], t)[0] if "motion" in marker else c


def crosses(marker, p0, p1, t=0.0, t0=None, frame=None):
    """The swept test from p0 at sim time t0 (default t) to p1 at sim time t. A moving marker is tested on the
    relative path: p0 in its frame at t0, p1 in its frame at t, joined by a straight segment. An attached marker
    (`frame`, its body's pose now) takes both points into that one frame: a point test, as the survive region's is."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if frame is not None:
        rot = _mat(frame[1])
        p0, p1 = (p0 - frame[0]) @ rot, (p1 - frame[0]) @ rot  # R^T (p - pos)
    if "motion" in marker:  # test in the marker's authored frame: undo its offset, then its spin about its centre
        p0, p1 = _authored(marker, p0, t if t0 is None else t0), _authored(marker, p1, t)
    kind = marker["type"]
    if kind == "sphere":
        return _seg_seg_dist(p0, p1, np.array(marker["pos"]), np.array(marker["pos"])) <= marker["radius"]
    if kind == "capsule":
        f = marker["fromto"]
        return _seg_seg_dist(p0, p1, np.array(f[:3]), np.array(f[3:])) <= marker["radius"]
    return _seg_box(p0, p1, marker)


def _authored(marker, p, t):
    """Point p at sim time t in the marker's authored frame."""
    offset, yaw = pose(marker["motion"], t)
    c, cy, sy = centre(marker, 0.0), math.cos(yaw), math.sin(yaw)
    rot = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])  # Rz(-yaw)
    return c + (p - c - offset) @ rot.T


def _mat(q):
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, np.asarray(q, float))
    return rot.reshape(3, 3)


def _seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection, 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            s, t = min(max(-c / a, 0.0), 1.0), 0.0
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                s, t = min(max(-c / a, 0.0), 1.0), 0.0
            elif t > 1:
                s, t = min(max((b - c) / a, 0.0), 1.0), 1.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def _seg_box(p0, p1, box):
    """Slab test in the box's frame (euler in degrees, MuJoCo's xyz sequence)."""
    q, rot = np.empty(4), np.empty(9)
    mujoco.mju_euler2Quat(q, np.radians(box["euler"]), "xyz")
    mujoco.mju_quat2Mat(rot, q)
    rot = rot.reshape(3, 3)
    a = (p0 - box["pos"]) @ rot  # R^T (p - pos)
    d = (p1 - p0) @ rot
    lo, hi = 0.0, 1.0
    for i, h in enumerate(box["size"]):
        if abs(d[i]) < 1e-12:
            if abs(a[i]) > h:
                return False
            continue
        t0, t1 = (-h - a[i]) / d[i], (h - a[i]) / d[i]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if lo > hi:
            return False
    return True
