"""The env-step kernels (spec §9, §10): what `HamsterballEnv._advance` and `_observe` do for the five tasks, moving
markers included, one thread per world, so the vector env runs them inside the step graph. See env_step.md."""

import math

import mujoco
import numpy as np
import warp as wp

from ..core.level import LOCKED_RGBA, MARKER_RGBA, PASSED_RGBA, TASK_MARKERS
from ..core.marker import centre as marker_centre
from ..core.model import R
from ..env import PHI_SCALE, TIME_PENALTY

FAILURE_CODES = ("none", "hazard", "kill_plane", "left_region")  # info["failure_reason"] holds the index

SPHERE, CAPSULE, BOX = 0, 1, 2  # marker kinds in the table (a sphere is a capsule of zero length)
_R, _PHI_SCALE, _TIME_PENALTY = (wp.constant(x) for x in (R, PHI_SCALE, TIME_PENALTY))
_BOX = wp.constant(BOX)
# how a task's targets become live (`_live`): all of them; only the first left (waypoints); the goal once no key is left
ALL, IN_ORDER, GOAL_LAST = 0, 1, 2
MODES = {"reach_goal": ALL, "waypoint_race": IN_ORDER, "collect_items": ALL, "unlock_goal": GOAL_LAST, "survive": ALL}
# what a target's look does as the episode goes (`_look`): nothing; vanish once collected; locked, live, passed; locked until the keys are in
PLAIN, VANISH, WAYPOINT, UNLOCK = 0, 1, 2, 3
_ALL, _IN_ORDER, _GOAL_LAST = (wp.constant(x) for x in (ALL, IN_ORDER, GOAL_LAST))
_VANISH, _WAYPOINT, _UNLOCK = (wp.constant(x) for x in (VANISH, WAYPOINT, UNLOCK))
_LOCKED, _PASSED = (wp.constant(wp.vec4(*c)) for c in (LOCKED_RGBA, PASSED_RGBA))
_G = wp.constant((math.sqrt(5) - 1) / 2)  # the golden section's ratio
_TOL = wp.constant(1e-6)  # the box test's bracket, in metres
_FACING, _PITCH = wp.constant(0), wp.constant(1)  # model.FACING, model.PITCH
_SHELL, _HAM, _SHELL_DOF = wp.constant(1), wp.constant(2), wp.constant(0)  # model.SHELL, HAM, SHELL_DOF
STILL, OSCILLATE, PATH = 0, 1, 2  # a marker row's motion kind
_STILL, _OSCILLATE = wp.constant(STILL), wp.constant(OSCILLATE)
_TWO_PI = wp.constant(2.0 * math.pi)


def targets(task, block):
    """The task's targets in task order, `(marker key, marker)`: keys before the goal, waypoint i before i + 1."""
    out = []
    for k in TASK_MARKERS[task]:
        out += [(k, m) for m in block[k]] if isinstance(block[k], list) else [(k, block[k])]
    return out


@wp.struct
class Motion:
    """The marker rows' motions (`motion.pose`): per row its kind, period, phase, spin (rad/s), the offset's
    amplitude vector (oscillate: axis times amplitude), the point its spin turns about (`pivot`: `marker.centre` at
    sim time 0, so its authored centre plus its offset then), and a path's slice (`start`, `count`) of `pts` and `cum` (its
    closed polyline's points and cumulative lengths)."""

    kind: wp.array(dtype=int)
    period: wp.array(dtype=float)
    phase: wp.array(dtype=float)
    spin: wp.array(dtype=float)
    vec: wp.array(dtype=wp.vec3)
    pivot: wp.array(dtype=wp.vec3)
    start: wp.array(dtype=int)
    count: wp.array(dtype=int)
    pts: wp.array(dtype=wp.vec3)
    cum: wp.array(dtype=float)


class Markers:
    """The marker tables on the device, one block of `stride` rows per bank entry: the entry's task's targets in task
    order, then its level's hazards, each a sphere, capsule, or box in the kernels' form (`a`, `b` the segment's ends,
    `rot` and `half` the box's frame and half-sizes), and the rest of the block padding. `role` and `base` say how each
    target looks (see `look_markers`) and `motion` (a `Motion`) how it moves. Per entry: `ntargets`, `n` (targets and hazards), `mode`, `goal_row`, `survive`,
    `time_penalty`. `entries` is a list of `(task, block, hazards)`; `row_geom` maps each row to its geom in the model,
    `row_slot` to its marker slot."""

    def __init__(self, entries, device):
        self.stride = max(len(targets(t, b)) + len(h) for t, b, h in entries)
        self.max_targets = max(len(targets(t, b)) for t, b, _ in entries)
        pad = {"type": "sphere", "pos": [0, 0, 0], "radius": 0.0}
        kind, a, b, rad, rot, half, centre, role, base = [], [], [], [], [], [], [], [], []
        ntargets, n, mode, goal_row, survive, tp = [], [], [], [], [], []
        mo = {k: [] for k in ("kind", "period", "phase", "spin", "vec", "pivot", "start", "count")}
        pts, cum = [[0, 0, 0]], [0.0]  # a path's closed polylines end to end (one padding point so the arrays are never empty)
        for task, block, hazards in entries:
            tgt = targets(task, block)
            ntargets.append(len(tgt)), n.append(len(tgt) + len(hazards)), mode.append(MODES[task])
            goal_row.append({"reach_goal": 0, "unlock_goal": len(tgt) - 1}.get(task, -1))  # the row that pays no +1 of its own
            survive.append(int(task == "survive")), tp.append(0.0 if task == "survive" else TIME_PENALTY)
            rows = [m for _, m in tgt] + list(hazards)
            for m in rows + [pad] * (self.stride - len(rows)):
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
                motion = m.get("motion")
                mo["kind"].append(STILL if motion is None else OSCILLATE if motion["type"] == "oscillate" else PATH)
                mo["period"].append(motion["period"] if motion else 1.0), mo["phase"].append(motion["phase"] if motion else 0.0)
                mo["spin"].append(math.radians(motion["spin"]) if motion else 0.0)
                mo["vec"].append(np.array(motion["axis"]) * motion["amplitude"] if motion and motion["type"] == "oscillate" else [0, 0, 0])
                mo["pivot"].append(marker_centre(m, 0.0))
                mo["start"].append(len(pts)), mo["count"].append(0)
                if motion and motion["type"] == "path":
                    closed = np.array(motion["points"] + motion["points"][:1])
                    pts += closed.tolist()
                    cum += np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]).tolist()
                    mo["count"][-1] = len(closed)
            keys = [k for k, _ in tgt]
            role += [VANISH if k in ("items", "keys") else WAYPOINT if k == "waypoints" else UNLOCK if MODES[task] == GOAL_LAST and k == "goal"
                     else PLAIN for k in keys] + [PLAIN] * (self.stride - len(keys))
            base += [MARKER_RGBA[k] for k in keys] + [(0, 0, 0, 0)] * (self.stride - len(keys))
        f = lambda x, t: wp.array(np.array(x, np.float32), dtype=t, device=device)  # noqa: E731
        i = lambda x: wp.array(x, dtype=int, device=device)  # noqa: E731
        self.kind, self.role = i(kind), i(role)
        self.a, self.b, self.half, self.centre = (f(x, wp.vec3) for x in (a, b, half, centre))
        self.rad, self.rot, self.base = f(rad, float), f(rot, wp.mat33), f(base, wp.vec4)
        self.ntargets, self.n, self.mode, self.goal_row, self.survive = i(ntargets), i(n), i(mode), i(goal_row), i(survive)
        self.time_penalty = f(tp, float)
        self.motion = Motion()
        self.motion.kind, self.motion.start, self.motion.count = i(mo["kind"]), i(mo["start"]), i(mo["count"])
        self.motion.period, self.motion.phase, self.motion.spin = (f(mo[k], float) for k in ("period", "phase", "spin"))
        self.motion.vec, self.motion.pivot = f(mo["vec"], wp.vec3), f(mo["pivot"], wp.vec3)
        self.motion.pts, self.motion.cum = f(pts, wp.vec3), f(cum, float)


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


@wp.func
def inside(i: int, kind: wp.array(dtype=int), a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3),
           rad: wp.array(dtype=float), rot: wp.array(dtype=wp.mat33), half: wp.array(dtype=wp.vec3), p: wp.vec3):
    """marker.crosses for a still marker i as a centre test (reach 0, a point): p is in its volume."""
    if kind[i] == _BOX:
        q = wp.transpose(rot[i]) @ (p - a[i])
        return wp.abs(q[0]) <= half[i][0] and wp.abs(q[1]) <= half[i][1] and wp.abs(q[2]) <= half[i][2]
    return seg_seg_dist(p, p, a[i], b[i]) <= rad[i]


@wp.func
def live(w: int, i: int, todo: wp.array2d(dtype=wp.bool), mode: int, ntargets: int):
    """HamsterballEnv._live: target i can trigger now. Only the next waypoint; the goal (the last row) only once no key
    is left. A target's liveness depends on earlier rows only, so one pass in row order reaches the fixed point."""
    if not todo[w, i]:
        return False
    if mode == _IN_ORDER:
        for j in range(i):
            if todo[w, j]:
                return False
    if mode == _GOAL_LAST and i == ntargets - 1:
        for j in range(i):
            if todo[w, j]:
                return False
    return True


@wp.func
def offset_of(r: int, t: float, mo: Motion):
    """motion.pose's offset for marker row r at sim time t: a sine along an axis, or a point on the closed path at the
    fraction of its length the phase gives. A still row has none."""
    off = wp.vec3()
    k = mo.kind[r]
    if k != _STILL:
        u = t / mo.period[r] + mo.phase[r]
        u = u - wp.floor(u)  # Python's % 1.0
        if k == _OSCILLATE:
            off = mo.vec[r] * wp.sin(_TWO_PI * u)
        else:
            s0 = mo.start[r]
            n = mo.count[r]
            x = u * mo.cum[s0 + n - 1]
            j = int(0)
            for i in range(n - 2):
                if x > mo.cum[s0 + i + 1]:
                    j = i + 1
                else:
                    break
            seg = mo.cum[s0 + j + 1] - mo.cum[s0 + j]
            f = float(0.0)
            if seg > 0.0:
                f = wp.clamp((x - mo.cum[s0 + j]) / seg, 0.0, 1.0)
            off = mo.pts[s0 + j] + (mo.pts[s0 + j + 1] - mo.pts[s0 + j]) * f
    return off


@wp.func
def to_authored(r: int, p: wp.vec3, t: float, mo: Motion):
    """marker._authored: the point p at sim time t in row r's authored frame, undoing its offset and then its spin
    about its pivot (a still row's frame is the world's)."""
    if mo.kind[r] == _STILL:
        return p
    yaw = mo.spin[r] * t
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    v = p - mo.pivot[r] - offset_of(r, t, mo)
    return mo.pivot[r] + wp.vec3(cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1], v[2])


@wp.func
def nearest(w: int, p: wp.vec3, todo: wp.array2d(dtype=wp.bool), mode: int, ntargets: int, centre: wp.array(dtype=wp.vec3),
            row: int, t: float, mo: Motion):
    """HamsterballEnv._target: the centre of the live target nearest p at sim time t (the first of a tie), a moving
    target's by its offset then."""
    best = wp.vec3()
    d_best = float(1.0e30)
    for i in range(ntargets):
        if live(w, i, todo, mode, ntargets):
            c = centre[row + i] + offset_of(row + i, t, mo)
            d = wp.length(c - p)
            if d < d_best:
                d_best = d
                best = c
    return best


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
    t_ntargets: wp.array(dtype=int),
    t_n: wp.array(dtype=int),
    t_mode: wp.array(dtype=int),
    t_goal_row: wp.array(dtype=int),
    t_survive: wp.array(dtype=int),
    t_time_penalty: wp.array(dtype=float),
    t_kill_z: wp.array(dtype=float),
    t_max_steps: wp.array(dtype=int),
    truncate: int,
    shaped: int,
    gamma: float,
    todo: wp.array2d(dtype=wp.bool),
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
    time: wp.array(dtype=float),
    prev_t: wp.array(dtype=float),
    mo: Motion,
):
    """HamsterballEnv._advance: rows 0..ntargets-1 are the task's targets, the rest hazards. Survive tests the region
    (row 0, which may move) at the centre and clears it at the time limit; every other task removes the live targets
    the swept segment touches (targets are still). A hazard is tested on its relative path: the previous centre in its
    frame at the previous sim time, the current one in its frame now. Then success over failure (hazard, kill plane,
    left region), the time penalty (none in survive), the shaping term, truncation. Updates the targets left, step
    count, previous shell position and sim time, potential."""
    w = wp.tid()
    e = entry[w]
    row = e * stride
    ntargets = t_ntargets[e]
    nmarkers = t_n[e]
    mode = t_mode[e]
    goal_row = t_goal_row[e]
    survive = t_survive[e]
    time_penalty = t_time_penalty[e]
    kill_z = t_kill_z[e]
    max_steps = t_max_steps[e]
    steps[w] = steps[w] + 1
    pos = wp.vec3(qpos[w, 0], qpos[w, 1], qpos[w, 2])
    prev = prev_pos[w]
    now = time[w]
    event = float(0.0)
    fail = int(0)
    in_region = bool(True)
    if survive != 0:
        in_region = inside(row, kind, a, b, rad, rot, half, to_authored(row, pos, now, mo))
        if in_region and steps[w] >= max_steps:
            todo[w, 0] = False
    else:
        for i in range(ntargets):
            if live(w, i, todo, mode, ntargets) and touches(row + i, kind, a, b, rad, rot, half, prev, pos):
                todo[w, i] = False
                if i != goal_row:
                    event += 1.0  # +1 per item, key, or waypoint
    left = int(0)
    for i in range(ntargets):
        if todo[w, i]:
            left += 1
    won = left == 0  # success wins over failure
    if won:
        event += 1.0
    else:
        for i in range(ntargets, nmarkers):
            if fail == 0 and touches(row + i, kind, a, b, rad, rot, half, to_authored(row + i, prev, prev_t[w], mo),
                                     to_authored(row + i, pos, now, mo)):
                fail = 1  # hazard
        if fail == 0 and pos[2] < kill_z:
            fail = 2  # kill_plane
        if fail == 0 and not in_region:
            fail = 3  # left_region
        if fail != 0:
            event -= 1.0
    prev_pos[w] = pos
    prev_t[w] = now
    done = won or fail != 0
    new_phi = float(0.0)
    if not done:
        new_phi = _PHI_SCALE * wp.length(nearest(w, pos, todo, mode, ntargets, centre, row, now, mo) - pos)
    shaping = float(0.0)
    if shaped != 0:
        shaping = gamma * new_phi - phi[w]
    phi[w] = new_phi
    success[w] = won
    failure[w] = fail
    terminated[w] = done
    truncated[w] = truncate != 0 and not done and steps[w] >= max_steps
    event_out[w] = event
    time_out[w] = time_penalty
    shaping_out[w] = shaping
    reward[w] = event + time_penalty + shaping


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
    centre: wp.array(dtype=wp.vec3),
    todo: wp.array2d(dtype=wp.bool),
    steps: wp.array(dtype=int),
    entry: wp.array(dtype=int),
    stride: int,
    t_ntargets: wp.array(dtype=int),
    t_mode: wp.array(dtype=int),
    t_survive: wp.array(dtype=int),
    t_max_steps: wp.array(dtype=int),
    t_kill_z: wp.array(dtype=float),
    task: wp.array(dtype=int),
    success: wp.array(dtype=wp.bool),
    state: wp.array2d(dtype=float),
    goal: wp.array2d(dtype=float),
    time: wp.array(dtype=float),
    mo: Motion,
):
    """HamsterballEnv._observe's `state` and `goal` for the worlds in `mask`; needs fresh xpos, xmat, subtree_com and
    subtree_linvel, which the caller refreshes. The goal's target is the nearest live one; its fraction remaining is
    the targets left over all of them, or in survive the time left over the time limit."""
    w = wp.tid()
    if not mask[w]:
        return
    e = entry[w]
    row = e * stride
    ntargets = t_ntargets[e]
    mode = t_mode[e]
    survive = t_survive[e]
    max_steps = t_max_steps[e]
    kill_z = t_kill_z[e]
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
    state[w, 3] = c[2] - kill_z
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
        g = to_facing(nearest(w, c, todo, mode, ntargets, centre, row, time[w], mo) - c, sy, cy)
        for i in range(3):
            goal[w, i] = g[i]
        goal[w, 3] = 1.0
        left = int(0)
        for i in range(ntargets):
            if todo[w, i]:
                left += 1
        goal[w, 4] = float(left) / float(ntargets)
        if survive != 0:
            goal[w, 4] = 1.0 - float(steps[w]) / float(max_steps)


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
    t_ntargets: wp.array(dtype=int),
    t_mode: wp.array(dtype=int),
    todo: wp.array2d(dtype=wp.bool),
    steps: wp.array(dtype=int),
    prev_pos: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=float),
    time: wp.array(dtype=float),
    prev_t: wp.array(dtype=float),
    mo: Motion,
):
    """HamsterballEnv.reset's episode state: every target left, no steps, the swept segment a point at the spawn (and the
    sim time then), Φ there."""
    w = wp.tid()
    if not mask[w]:
        return
    e = entry[w]
    ntargets = t_ntargets[e]
    pos = wp.vec3(qpos[w, 0], qpos[w, 1], qpos[w, 2])
    for i in range(todo.shape[1]):
        todo[w, i] = i < ntargets  # a padded target is never left
    steps[w] = 0
    prev_pos[w] = pos
    prev_t[w] = time[w]
    phi[w] = _PHI_SCALE * wp.length(nearest(w, pos, todo, t_mode[e], ntargets, centre, e * stride, time[w], mo) - pos)


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



@wp.kernel
def look_markers(
    todo: wp.array2d(dtype=wp.bool),
    entry: wp.array(dtype=int),
    stride: int,
    t_ntargets: wp.array(dtype=int),
    t_mode: wp.array(dtype=int),
    row_geom: wp.array(dtype=int),
    role: wp.array(dtype=int),
    base: wp.array(dtype=wp.vec4),
    geom_rgba: wp.array2d(dtype=wp.vec4),
):
    """The episode task's marker colours (HamsterballEnv's `_look`), derived from the targets left: a collected item or
    key vanishes (alpha 0), a waypoint is locked until it is next and passed after, the unlock goal is locked until
    every key is in. A world's marker slots hold only its own task's markers, so the other tasks' are never drawn."""
    w = wp.tid()
    e = entry[w]
    ntargets = t_ntargets[e]
    mode = t_mode[e]
    for i in range(ntargets):
        r = e * stride + i
        c = base[r]
        if role[r] == _VANISH and not todo[w, i]:
            c = wp.vec4(c[0], c[1], c[2], 0.0)
        elif role[r] == _WAYPOINT:
            if not todo[w, i]:
                c = _PASSED
            elif not live(w, i, todo, mode, ntargets):
                c = _LOCKED
        elif role[r] == _UNLOCK and todo[w, i] and not live(w, i, todo, mode, ntargets):
            c = _LOCKED
        geom_rgba[w, row_geom[r]] = c


@wp.kernel
def pose_movers(
    entry: wp.array(dtype=int),
    stride: int,
    t_n: wp.array(dtype=int),
    row_slot: wp.array(dtype=int),
    row_geom: wp.array(dtype=int),
    slot_pos: wp.array2d(dtype=wp.vec3),
    slot_quat: wp.array2d(dtype=wp.quat),
    time: wp.array(dtype=float),
    mo: Motion,
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
):
    """HamsterballEnv._pose_movers: each moving marker's geom at its pose at the world's sim time, its authored pose
    moved by the offset and turned by the yaw about z (quaternions are MuJoCo's w, x, y, z). The geom is static, so
    its world pose (`geom_xpos`, `geom_xmat`), which forward never recomputes, is written too."""
    w, i = wp.tid()
    e = entry[w]
    r = e * stride + i
    if i >= t_n[e] or mo.kind[r] == _STILL:
        return
    g = row_geom[r]
    s = row_slot[r]
    t = time[w]
    yaw = mo.spin[r] * t
    cz = wp.cos(0.5 * yaw)
    sz = wp.sin(0.5 * yaw)
    q0 = slot_quat[e, s]
    # (cz, 0, 0, sz) * q0
    q = wp.quat(cz * q0[0] - sz * q0[3], cz * q0[1] - sz * q0[2], cz * q0[2] + sz * q0[1], cz * q0[3] + sz * q0[0])
    p = slot_pos[e, s] + offset_of(r, t, mo)
    geom_pos[w, g] = p
    geom_quat[w, g] = q
    geom_xpos[w, g] = p
    a, b, c, d = q[0], q[1], q[2], q[3]
    geom_xmat[w, g] = wp.mat33(
        a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c),
        2.0 * (b * c + a * d), a * a - b * b + c * c - d * d, 2.0 * (c * d - a * b),
        2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a - b * b - c * c + d * d)
