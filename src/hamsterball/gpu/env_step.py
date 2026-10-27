"""The env-step kernels (spec §9, §10) for reach_goal: what `HamsterballEnv._advance` and `_observe` do, one thread per
world, so the vector env runs them inside the step graph. See env_step.md."""

import math

import mujoco
import numpy as np
import warp as wp

from ..core.model import R
from ..env import PHI_SCALE, TIME_PENALTY

FAILURE_CODES = ("none", "hazard", "kill_plane", "left_region")  # info["failure_reason"] holds the index

SPHERE, CAPSULE, BOX = 0, 1, 2  # marker kinds in the table (a sphere is a capsule of zero length)
_R, _PHI_SCALE, _TIME_PENALTY = (wp.constant(x) for x in (R, PHI_SCALE, TIME_PENALTY))
_BOX = wp.constant(BOX)
_G = wp.constant((math.sqrt(5) - 1) / 2)  # the golden section's ratio
_TOL = wp.constant(1e-6)  # the box test's bracket, in metres
_FACING, _PITCH = wp.constant(0), wp.constant(1)  # model.FACING, model.PITCH
_SHELL, _HAM, _SHELL_DOF = wp.constant(1), wp.constant(2), wp.constant(0)  # model.SHELL, HAM, SHELL_DOF


class Markers:
    """The marker tables on the device, one block of `stride` rows per bank entry: row 0 the goal, then the hazards, each
    a sphere, capsule, or box in the kernels' form (`a`, `b` the segment's ends, `rot` and `half` the box's frame and
    half-sizes). `entries` is one list of marker dicts per entry, goal first; `n[e]` counts entry e's rows, and the rest
    of its block is padding."""

    def __init__(self, entries, device):
        self.stride = max(len(rows) for rows in entries)
        self.n = wp.array([len(rows) for rows in entries], dtype=int, device=device)
        kind, a, b, rad, rot, half, centre = [], [], [], [], [], [], []
        pad = {"type": "sphere", "pos": [0, 0, 0], "radius": 0.0}
        for m in (m for rows in entries for m in [*rows, *[pad] * (self.stride - len(rows))]):
            kind.append({"sphere": SPHERE, "capsule": CAPSULE, "box": BOX}[m["type"]])
            if m["type"] == "capsule":
                a.append(m["fromto"][:3]), b.append(m["fromto"][3:])
            else:
                a.append(m["pos"]), b.append(m["pos"])
            centre.append(np.mean([a[-1], b[-1]], axis=0))
            rad.append(m.get("radius", 0.0))
            half.append(m.get("size", [0, 0, 0]))
            q, mat = np.empty(4), np.empty(9)  # the box's frame, as marker._seg_box builds it
            mujoco.mju_euler2Quat(q, np.radians(m.get("euler", [0, 0, 0])), "xyz")
            mujoco.mju_quat2Mat(mat, q)
            rot.append(mat.reshape(3, 3))
        f = lambda x, t: wp.array(np.array(x, np.float32), dtype=t, device=device)  # noqa: E731
        self.kind = wp.array(kind, dtype=int, device=device)
        self.a, self.b, self.half, self.centre = (f(x, wp.vec3) for x in (a, b, half, centre))
        self.rad, self.rot = f(rad, float), f(rot, wp.mat33)


@wp.func
def seg_seg_dist(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    """Closest distance between segments p1q1 and p2q2 (marker._seg_seg_dist)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = wp.dot(d1, d1), wp.dot(d2, d2), wp.dot(d2, r)
    eps = 1.0e-12
    s = float(0.0)
    t = float(0.0)
    if a <= eps and e <= eps:
        return wp.length(r)
    if a <= eps:
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= eps:
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denom = a * e - b * b
            if denom > eps:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                s = wp.clamp(-c / a, 0.0, 1.0)
                t = 0.0
            elif t > 1.0:
                s = wp.clamp((b - c) / a, 0.0, 1.0)
                t = 1.0
    return wp.length(p1 + d1 * s - (p2 + d2 * t))


@wp.func
def box_dist(a: wp.vec3, d: wp.vec3, h: wp.vec3, s: float):
    """Distance from the point a + s d to the box of half-sizes h at the origin."""
    p = a + s * d
    return wp.length(wp.vec3(wp.max(wp.abs(p[0]) - h[0], 0.0), wp.max(wp.abs(p[1]) - h[1], 0.0), wp.max(wp.abs(p[2]) - h[2], 0.0)))


@wp.func
def seg_box_dist(a: wp.vec3, d: wp.vec3, h: wp.vec3):
    """marker._seg_box_dist: a golden-section search along the segment a + s d, s in [0, 1], to a 1e-6 m bracket."""
    lo = float(0.0)
    hi = float(1.0)
    span = _TOL / wp.max(wp.length(d), _TOL)
    x1 = hi - _G * (hi - lo)
    x2 = lo + _G * (hi - lo)
    f1 = box_dist(a, d, h, x1)
    f2 = box_dist(a, d, h, x2)
    for _ in range(64):  # the bracket shrinks by 0.62 per pass; this only bounds a degenerate one
        if hi - lo <= span:
            break
        if f1 <= f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - _G * (hi - lo)
            f1 = box_dist(a, d, h, x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + _G * (hi - lo)
            f2 = box_dist(a, d, h, x2)
    return wp.min(wp.min(box_dist(a, d, h, lo), box_dist(a, d, h, hi)), wp.min(f1, f2))


@wp.func
def touches(i: int, kind: wp.array(dtype=int), a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3),
            rad: wp.array(dtype=float), rot: wp.array(dtype=wp.mat33), half: wp.array(dtype=wp.vec3),
            p0: wp.vec3, p1: wp.vec3):
    """marker.crosses for a still marker i with reach R: the shell touches it somewhere on the swept segment p0 to p1."""
    if kind[i] == _BOX:
        rt = wp.transpose(rot[i])
        return seg_box_dist(rt @ (p0 - a[i]), rt @ (p1 - p0), half[i]) <= _R
    return seg_seg_dist(p0, p1, a[i], b[i]) <= rad[i] + _R


@wp.kernel
def advance(
    qpos: wp.array2d(dtype=float),
    kind: wp.array(dtype=int),
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    rad: wp.array(dtype=float),
    rot: wp.array(dtype=wp.mat33),
    half: wp.array(dtype=wp.vec3),
    centre: wp.array(dtype=wp.vec3),
    entry: wp.array(dtype=int),
    stride: int,
    nmarkers: wp.array(dtype=int),
    kill_z: wp.array(dtype=float),
    max_steps: wp.array(dtype=int),
    truncate: int,
    shaped: int,
    gamma: float,
    steps: wp.array(dtype=int),
    prev_pos: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=float),
    reward: wp.array(dtype=float),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    success: wp.array(dtype=wp.bool),
    failure: wp.array(dtype=int),
    event_out: wp.array(dtype=float),
    time_out: wp.array(dtype=float),
    shaping_out: wp.array(dtype=float),
):
    """HamsterballEnv._advance for reach_goal: the swept touch test of the goal (row 0) and the hazards (rows 1..),
    success over failure, the time penalty, the shaping term, truncation. Updates the episode's step count, previous
    shell position, and potential."""
    w = wp.tid()
    e = entry[w]
    row = e * stride
    steps[w] = steps[w] + 1
    pos = wp.vec3(qpos[w, 0], qpos[w, 1], qpos[w, 2])
    prev = prev_pos[w]
    event = float(0.0)
    fail = int(0)
    won = touches(row, kind, a, b, rad, rot, half, prev, pos)
    if won:
        event = 1.0
    else:
        for i in range(1, nmarkers[e]):
            if fail == 0 and touches(row + i, kind, a, b, rad, rot, half, prev, pos):
                fail = 1  # hazard
        if fail == 0 and pos[2] < kill_z[e]:
            fail = 2  # kill_plane
        if fail != 0:
            event = -1.0
    prev_pos[w] = pos
    done = won or fail != 0
    new_phi = float(0.0)
    if not done:
        new_phi = _PHI_SCALE * wp.length(centre[row] - pos)
    shaping = float(0.0)
    if shaped != 0:
        shaping = gamma * new_phi - phi[w]
    phi[w] = new_phi
    success[w] = won
    failure[w] = fail
    terminated[w] = done
    truncated[w] = truncate != 0 and not done and steps[w] >= max_steps[e]
    event_out[w] = event
    time_out[w] = _TIME_PENALTY
    shaping_out[w] = shaping
    reward[w] = event + _TIME_PENALTY + shaping


@wp.func
def to_facing(v: wp.vec3, sy: float, cy: float):
    return wp.vec3(v[0] * cy + v[1] * sy, -v[0] * sy + v[1] * cy, v[2])


@wp.kernel
def observe(
    mask: wp.array(dtype=wp.bool),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    qvel: wp.array2d(dtype=float),
    subtree_com: wp.array2d(dtype=wp.vec3),
    subtree_linvel: wp.array2d(dtype=wp.vec3),
    userdata: wp.array2d(dtype=float),
    wall_contact: wp.array(dtype=wp.bool),
    target: wp.array(dtype=wp.vec3),
    entry: wp.array(dtype=int),
    stride: int,
    kill_z: wp.array(dtype=float),
    task: wp.array(dtype=int),
    success: wp.array(dtype=wp.bool),
    state: wp.array2d(dtype=float),
    goal: wp.array2d(dtype=float),
):
    """HamsterballEnv._observe's `state` and `goal` for the worlds in `mask`; needs fresh xpos, xmat, subtree_com and
    subtree_linvel, which the caller refreshes. The goal's target is the first row of the world's entry in `target`; its
    fraction remaining is 1, reach_goal's one target left."""
    w = wp.tid()
    if not mask[w]:
        return
    facing = userdata[w, _FACING]
    sy = wp.sin(facing)
    cy = wp.cos(facing)
    c = xpos[w, _SHELL]
    v = wp.vec3(qvel[w, _SHELL_DOF], qvel[w, _SHELL_DOF + 1], qvel[w, _SHELL_DOF + 2])
    wb = wp.vec3(qvel[w, _SHELL_DOF + 3], qvel[w, _SHELL_DOF + 4], qvel[w, _SHELL_DOF + 5])
    ws = to_facing(xmat[w, _SHELL] @ wb, sy, cy)  # free-joint angular velocity is in the body frame
    vs = to_facing(v, sy, cy)
    rel = to_facing(subtree_com[w, _HAM] - c, sy, cy)
    relv = to_facing(subtree_linvel[w, _HAM] - v, sy, cy)
    for i in range(3):
        state[w, i] = c[i]
        state[w, 8 + i] = vs[i]
        state[w, 11 + i] = ws[i]
        state[w, 14 + i] = rel[i]
        state[w, 17 + i] = relv[i]
    state[w, 3] = c[2] - kill_z[entry[w]]
    state[w, 4] = sy
    state[w, 5] = cy
    state[w, 6] = wp.sin(userdata[w, _PITCH])
    state[w, 7] = wp.cos(userdata[w, _PITCH])
    state[w, 20] = float(0.0)
    if wall_contact[w]:
        state[w, 20] = 1.0
    for i in range(10):
        goal[w, i] = 0.0
    goal[w, 5 + task[w]] = 1.0
    if not success[w]:
        g = to_facing(target[entry[w] * stride] - c, sy, cy)
        for i in range(3):
            goal[w, i] = g[i]
        goal[w, 3] = 1.0
        goal[w, 4] = 1.0


@wp.kernel
def stash_final(
    done: wp.array(dtype=wp.bool),
    state: wp.array2d(dtype=float),
    goal: wp.array2d(dtype=float),
    final_state: wp.array2d(dtype=float),
    final_goal: wp.array2d(dtype=float),
):
    """Copy the observation of each finished world aside before its reset overwrites it."""
    w = wp.tid()
    if not done[w]:
        return
    for i in range(21):
        final_state[w, i] = state[w, i]
    for i in range(10):
        final_goal[w, i] = goal[w, i]


@wp.kernel
def reset_episode(
    mask: wp.array(dtype=wp.bool),
    qpos: wp.array2d(dtype=float),
    centre: wp.array(dtype=wp.vec3),
    entry: wp.array(dtype=int),
    stride: int,
    steps: wp.array(dtype=int),
    prev_pos: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=float),
):
    """HamsterballEnv.reset's episode state: no steps, the swept segment a point at the spawn, Φ there."""
    w = wp.tid()
    if not mask[w]:
        return
    pos = wp.vec3(qpos[w, 0], qpos[w, 1], qpos[w, 2])
    steps[w] = 0
    prev_pos[w] = pos
    phi[w] = _PHI_SCALE * wp.length(centre[entry[w] * stride] - pos)


@wp.kernel
def set_spawn(mask: wp.array(dtype=wp.bool), entry: wp.array(dtype=int), t_spawn: wp.array(dtype=wp.vec3),
              t_yaw: wp.array(dtype=float), spawn: wp.array(dtype=wp.vec3), facing: wp.array(dtype=float)):
    """The masked worlds' spawn pose from their entry's table."""
    w = wp.tid()
    if mask[w]:
        spawn[w] = t_spawn[entry[w]]
        facing[w] = t_yaw[entry[w]]


@wp.kernel
def either(x: wp.array(dtype=wp.bool), y: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.bool)):
    w = wp.tid()
    out[w] = x[w] or y[w]

