"""The abstract tier's Warp `drive` and `obs` kernels (spec §5), ported line for line from `core.abstract_tier`."""

import warp as wp

from ..core import abstract_tier as A
from ..core.model import FACING, HAM, HAM_DOF, HAM_QPOS, R, SHELL, SHELL_DOF, T

_REST, _TOUCH, _GRACE = wp.constant(A.REST), wp.constant(A.TOUCH), wp.constant(A.GRACE)
_KP, _KD, _F_MAX, _K_GRIP = wp.constant(A.KP), wp.constant(A.KD), wp.constant(A.F_MAX), wp.constant(A.K_GRIP)
_LEAN, _SIDE, _BACK = wp.constant(A.LEAN), wp.constant(A.SIDE), wp.constant(A.BACK)
_LAST_TOUCH, _FACING, _SHELL, _HAM, _SHELL_DOF, _HAM_DOF = (wp.constant(x) for x in (A.LAST_TOUCH, FACING, SHELL, HAM, SHELL_DOF, HAM_DOF))
_R_WALL, _HAM_QPOS = wp.constant(R - T), wp.constant(HAM_QPOS)


@wp.kernel
def drive(
    normals: wp.array(dtype=wp.vec3),  # core.model.TILE_NORMALS
    action: wp.array2d(dtype=float),  # the tier's two columns first
    time: wp.array(dtype=float),
    qvel: wp.array2d(dtype=float),
    xpos: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    userdata: wp.array2d(dtype=float),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    """Foot force pair and grip couple into xfrc_applied. Needs fresh xpos/xmat/xipos for the current qpos."""
    w = wp.tid()
    c = xpos[w, _SHELL]
    h = xipos[w, _HAM]
    r = h - c
    sm = xmat[w, _SHELL]
    p = wp.transpose(sm) @ r  # the hamster in the shell frame
    t = time[w]
    best = float(-1.0e30)
    for i in range(normals.shape[0]):
        best = wp.max(best, wp.dot(normals[i], p))
    if _REST - best <= _TOUCH:  # airborne cut
        userdata[w, _LAST_TOUCH] = t
    if t - userdata[w, _LAST_TOUCH] > _GRACE:
        return

    facing = userdata[w, _FACING]
    ax = action[w, 0]
    ay = action[w, 1]
    mag = wp.sqrt(ax * ax + ay * ay)
    e = wp.min(mag, 1.0)
    if mag > 0.0:  # camera frame -> world: rotate by the facing
        ax = ax / mag
        ay = ay / mag
        back = float(1.0)
        if ay < 0.0:
            back = _BACK
        e = e * (ay * ay * back + ax * ax * _SIDE)
        cyw = wp.cos(facing)
        syw = wp.sin(facing)
        ax, ay = ax * cyw - ay * syw, ax * syw + ay * cyw
    sl = wp.sin(_LEAN * e)
    cl = wp.cos(_LEAN * e)
    target = wp.vec3(sl * ax, sl * ay, -cl)  # target lean

    dist = wp.length(r) + 1.0e-9
    n = r / dist
    v = wp.vec3(
        qvel[w, _HAM_DOF] - qvel[w, _SHELL_DOF],
        qvel[w, _HAM_DOF + 1] - qvel[w, _SHELL_DOF + 1],
        qvel[w, _HAM_DOF + 2] - qvel[w, _SHELL_DOF + 2],
    )
    f = _KP * _REST * (target - n) - _KD * v
    f = f - wp.dot(f, n) * n
    fl = wp.length(f)
    if fl > _F_MAX:
        f = f * (_F_MAX / fl)

    # +F on the hamster and -F on the shell, both at the foot on the inner wall, as forces plus torques about each COM
    foot = c + _R_WALL * n
    a = foot - h
    b = foot - xipos[w, _SHELL]

    # grip couple: -K (w_h - w_s), world-frame angular velocities from the free joints' local ones
    hm = xmat[w, _HAM]
    wh = wp.vec3(qvel[w, _HAM_DOF + 3], qvel[w, _HAM_DOF + 4], qvel[w, _HAM_DOF + 5])
    ws = wp.vec3(qvel[w, _SHELL_DOF + 3], qvel[w, _SHELL_DOF + 4], qvel[w, _SHELL_DOF + 5])
    g = -_K_GRIP * (hm @ wh - sm @ ws)

    xfrc_applied[w, _HAM] = xfrc_applied[w, _HAM] + wp.spatial_vector(f, wp.cross(a, f) + g)
    xfrc_applied[w, _SHELL] = xfrc_applied[w, _SHELL] - wp.spatial_vector(f, wp.cross(b, f) + g)


@wp.kernel
def obs(time: wp.array(dtype=float), userdata: wp.array2d(dtype=float), wall_contact: wp.array(dtype=wp.bool)):
    """wall_contact: the shell wall was touched within the grace window."""
    w = wp.tid()
    wall_contact[w] = time[w] - userdata[w, _LAST_TOUCH] <= _GRACE


@wp.kernel
def reset(
    mask: wp.array(dtype=wp.bool),
    spawn: wp.array(dtype=wp.vec3),
    time: wp.array(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    userdata: wp.array2d(dtype=float),
):
    """Hamster 1 cm above the bottom of a shell centred at spawn; the foot force starts inside its grace."""
    w = wp.tid()
    if not mask[w]:
        return
    s = spawn[w]
    qpos[w, _HAM_QPOS + 0] = s[0]
    qpos[w, _HAM_QPOS + 1] = s[1]
    qpos[w, _HAM_QPOS + 2] = s[2] - (_REST - 0.01)
    qpos[w, _HAM_QPOS + 3] = 1.0
    qpos[w, _HAM_QPOS + 4] = 0.0
    qpos[w, _HAM_QPOS + 5] = 0.0
    qpos[w, _HAM_QPOS + 6] = 0.0
    for i in range(6):
        qvel[w, _HAM_DOF + i] = 0.0
    userdata[w, _LAST_TOUCH] = time[w]


class WarpAbstractTier:
    """What `BatchCore` needs from a tier: its CPU twin, its kernels, and the arrays they read."""

    def __init__(self, tier: A.AbstractTier, device):
        from ..core.model import TILE_NORMALS

        self.tier = tier
        self.normals = wp.array(TILE_NORMALS, dtype=wp.vec3, device=device)

    def reset(self, nworld, mask, spawn, d):
        wp.launch(reset, nworld, [mask, spawn, d.time, d.qpos, d.qvel, d.userdata])

    def drive(self, nworld, action, d):
        wp.launch(drive, nworld, [self.normals, action, d.time, d.qvel, d.xpos, d.xipos, d.xmat, d.userdata, d.xfrc_applied])

    def obs(self, nworld, d, out):
        wp.launch(obs, nworld, [d.time, d.userdata, out["wall_contact"]])

    def obs_arrays(self, nworld, device):
        return {"wall_contact": wp.zeros(nworld, dtype=wp.bool, device=device)}
