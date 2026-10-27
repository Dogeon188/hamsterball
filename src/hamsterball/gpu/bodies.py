"""Kinematic bodies on the device: each body slot's pose and velocity from its motion and the world's sim time. See
bodies.md."""

import math

import numpy as np
import warp as wp

from ..core.model import DT

_DT = wp.constant(wp.float64(DT))
_TWO_PI = wp.constant(wp.float64(2.0 * math.pi))
_PI = wp.constant(wp.float64(math.pi))
PATH, OSCILLATE = 0, 1
_OSCILLATE = wp.constant(OSCILLATE)


@wp.struct
class BodyMotion:
    """Per (entry, body slot) row `entry * nbodies + slot`, all float64 (a secant over one physics step cancels in
    float32): the motion's kind, period, phase, spin (rad/s), oscillation vector (axis times amplitude), authored
    position, a path's slice (`start`, `count`) of `pts` and `cum` (its closed polyline's points and cumulative
    lengths), and the tilt (`tilt` 0 or 1, its horizontal axis, amplitude in rad, period, phase)."""

    kind: wp.array(dtype=int)
    period: wp.array(dtype=wp.float64)
    phase: wp.array(dtype=wp.float64)
    spin: wp.array(dtype=wp.float64)
    vec: wp.array(dtype=wp.vec3d)
    pos: wp.array(dtype=wp.vec3d)
    start: wp.array(dtype=int)
    count: wp.array(dtype=int)
    pts: wp.array(dtype=wp.vec3d)
    cum: wp.array(dtype=wp.float64)
    tilt: wp.array(dtype=int)
    tilt_axis: wp.array(dtype=wp.vec3d)
    tilt_amp: wp.array(dtype=wp.float64)
    tilt_period: wp.array(dtype=wp.float64)
    tilt_phase: wp.array(dtype=wp.float64)


class BodyTables:
    """The bank's bodies' motions on the device (`motion`, a `BodyMotion`), `nbodies` per entry, and the addresses of the
    body slots' qpos and dofs in the compiled model (slots are consecutive free joints)."""

    def __init__(self, bank, cpu_model, device):
        self.K = K = bank.nbodies
        self.qadr = [int(cpu_model.jnt_qposadr[cpu_model.body_jntadr[cpu_model.body(f"body{k}").id]]) for k in range(K)]
        self.dadr = [int(cpu_model.jnt_dofadr[cpu_model.body_jntadr[cpu_model.body(f"body{k}").id]]) for k in range(K)]
        # every body slot's qpos address, driven and seesaw, in `bank.body_names` order (what an attached marker reads)
        self.slot_qadr = [int(cpu_model.jnt_qposadr[cpu_model.body_jntadr[cpu_model.body(n).id]]) for n in bank.body_names]
        E = len(bank.entries)
        f = {k: [] for k in ("kind", "period", "phase", "spin", "vec", "pos", "start", "count", "tilt", "axis", "amp", "tperiod", "tphase")}
        pts, cum = [[0, 0, 0]], [0.0]  # one padding point so the arrays are never empty
        nb = []
        for i, _ in bank.entries:
            bodies = [b for b in bank.levels[i].bodies if "motion" in b]  # a seesaw is not driven
            nb.append(len(bodies))
            for k in range(K):
                b = bodies[k] if k < len(bodies) else None
                m = b["motion"] if b else {"type": "oscillate", "axis": [0, 0, 0], "amplitude": 0.0, "period": 1.0, "phase": 0.0, "spin": 0.0}
                tl = m.get("tilt")
                f["kind"].append(OSCILLATE if m["type"] == "oscillate" else PATH)
                f["period"].append(m["period"]), f["phase"].append(m["phase"]), f["spin"].append(math.radians(m["spin"]))
                f["vec"].append(np.array(m["axis"]) * m["amplitude"] if m["type"] == "oscillate" else [0, 0, 0])
                f["pos"].append(b["pos"] if b else [0, 0, 0])
                f["start"].append(len(pts)), f["count"].append(0)
                if m["type"] == "path":
                    closed = np.array(m["points"] + m["points"][:1])
                    pts += closed.tolist()
                    cum += np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]).tolist()
                    f["count"][-1] = len(closed)
                f["tilt"].append(int(tl is not None))
                f["axis"].append(tl["axis"] if tl else [0, 0, 0]), f["amp"].append(math.radians(tl["amplitude"]) if tl else 0.0)
                f["tperiod"].append(tl["period"] if tl else 1.0), f["tphase"].append(tl["phase"] if tl else 0.0)
        d = lambda x, t: wp.array(np.array(x, np.float64 if t in (wp.float64, wp.vec3d) else np.int32), dtype=t, device=device)  # noqa: E731
        self.nbodies = d(nb or [0], int)
        mo = self.motion = BodyMotion()
        mo.kind, mo.start, mo.count, mo.tilt = (d(f[k] or [0], int) for k in ("kind", "start", "count", "tilt"))
        mo.period, mo.phase, mo.spin = (d(f[k] or [1.0], wp.float64) for k in ("period", "phase", "spin"))
        mo.tilt_amp, mo.tilt_period, mo.tilt_phase = (d(f[k] or [1.0], wp.float64) for k in ("amp", "tperiod", "tphase"))
        mo.vec, mo.pos, mo.tilt_axis = (d(f[k] or [[0, 0, 0]], wp.vec3d) for k in ("vec", "pos", "axis"))
        mo.pts, mo.cum = d(pts, wp.vec3d), d(cum, wp.float64)
        self.dev_qadr, self.dev_dadr = d(self.qadr or [0], int), d(self.dadr or [0], int)

    def drive(self, mask, entry, time, qpos, qvel):
        """`drive_bodies` for the masked worlds."""
        if self.K:
            wp.launch(drive_bodies, (entry.shape[0], self.K), [mask, entry, time, self.K, self.nbodies, self.dev_qadr,
                                                               self.dev_dadr, self.motion, qpos, qvel])


@wp.func
def offset_at(r: int, t: wp.float64, mo: BodyMotion):
    """motion.pose's offset at sim time t (float64): a sine along the axis, or the point on the closed path at the
    fraction of its length the phase gives."""
    u = t / mo.period[r] + mo.phase[r]
    u = u - wp.floor(u)  # Python's % 1.0
    off = wp.vec3d()
    if mo.kind[r] == _OSCILLATE:
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
        fr = wp.float64(0.0)
        if seg > wp.float64(0.0):
            fr = wp.clamp((x - mo.cum[s0 + j]) / seg, wp.float64(0.0), wp.float64(1.0))
        off = mo.pts[s0 + j] + (mo.pts[s0 + j + 1] - mo.pts[s0 + j]) * fr
    return off


@wp.func
def mul_quat(a: wp.vec4d, b: wp.vec4d):
    """Quaternion product, (w, x, y, z) in components 0 to 3 (mju_mulQuat)."""
    return wp.vec4d(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


@wp.func
def quat_at(r: int, t: wp.float64, mo: BodyMotion):
    """motion.quat at sim time t: the yaw about z, then the tilt about the body's own horizontal axis."""
    yaw = mo.spin[r] * t
    q = wp.vec4d(wp.cos(yaw / wp.float64(2.0)), wp.float64(0.0), wp.float64(0.0), wp.sin(yaw / wp.float64(2.0)))
    if mo.tilt[r] != 0:
        u = t / mo.tilt_period[r] + mo.tilt_phase[r]
        u = u - wp.floor(u)
        a = mo.tilt_amp[r] * wp.sin(_TWO_PI * u)
        s = wp.sin(a / wp.float64(2.0))
        ax = mo.tilt_axis[r]
        q = mul_quat(q, wp.vec4d(wp.cos(a / wp.float64(2.0)), s * ax[0], s * ax[1], s * ax[2]))
    return q


@wp.func
def sub_quat(qa: wp.vec4d, qb: wp.vec4d):
    """mju_subQuat: the turn w with qb * quat(w) = qa, as a 3-vector (the body frame's angular displacement)."""
    d = mul_quat(wp.vec4d(qb[0], -qb[1], -qb[2], -qb[3]), qa)
    axis = wp.vec3d(d[1], d[2], d[3])
    n = wp.length(axis)
    if n < wp.float64(1.0e-15):
        axis = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    else:
        axis = axis / n
    speed = wp.float64(2.0) * wp.atan2(n, d[0])
    if speed > _PI:
        speed = speed - _TWO_PI
    return axis * speed


@wp.kernel
def drive_bodies(
    mask: wp.array(dtype=wp.bool),
    entry: wp.array(dtype=int),
    time: wp.array(dtype=float),
    K: int,
    nbodies: wp.array(dtype=int),
    qadr: wp.array(dtype=int),
    dadr: wp.array(dtype=int),
    mo: BodyMotion,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    """core.step.drive_bodies: each body slot a masked world's level uses is posed at its motion's pose at the world's
    sim time, with the velocity that carries it to the pose one physics step later (the secant, angular part in the
    body frame). The slots beyond the level's own bodies are left parked."""
    w, k = wp.tid()
    e = entry[w]
    if not mask[w] or k >= nbodies[e]:
        return
    r = e * K + k
    t = wp.float64(time[w])
    o0 = offset_at(r, t, mo)
    o1 = offset_at(r, t + _DT, mo)
    q0 = quat_at(r, t, mo)
    q1 = quat_at(r, t + _DT, mo)
    p = mo.pos[r] + o0
    v = (o1 - o0) / _DT
    om = sub_quat(q1, q0) / _DT
    qa = qadr[k]
    da = dadr[k]
    for i in range(3):
        qpos[w, qa + i] = float(p[i])
        qvel[w, da + i] = float(v[i])
        qvel[w, da + 3 + i] = float(om[i])
    for i in range(4):
        qpos[w, qa + 3 + i] = float(q0[i])
