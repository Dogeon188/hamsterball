"""The abstract control tier (spec §5): the hamster as one sphere, driven by run direction and effort."""

import math

import gymnasium as gym
import numpy as np

from .model import CORE_SLOTS, HAM, HAM_DOF, HAM_QPOS, M_HAM, R, SHELL, SHELL_DOF, SOLREF, T, TILE_NORMALS

R_HAM = 0.12
REST = R - T - R_HAM  # resting radius, hamster centre to shell centre (0.35 m)
S = M_HAM / 2.0  # gains below are per 2 kg of hamster
KP, KD, F_MAX = 200.0 * S, 28.0 * S, 30.0 * S
K_GRIP = 2.0 * S  # N m s/rad: damps hamster spin relative to the shell
LEAN = math.radians(70)  # lean ceiling at full effort
SIDE, BACK = 0.7, 0.5  # effort scale running sideways and backward; forward is 1, between blends by cos^2/sin^2
TOUCH = 0.002 + 0.003  # wall gap (m) that counts as touching: the 2 mm margin + 3 mm
GRACE = 0.045  # s after the last touch before the foot force cuts out
LAST_TOUCH = CORE_SLOTS  # userdata slot, the first after the core's


class AbstractTier:
    nuserdata = 1
    resting_radius = REST
    sections = ""  # no actuators
    fragment = (
        f'<body name="hamster" pos="0 0 {T + R_HAM + 0.01!r}"><freejoint/>'
        f'<geom type="sphere" size="{R_HAM}" mass="{M_HAM}" solref="{SOLREF}" margin="0.002" priority="1" condim="1" '
        f'contype="2" conaffinity="2" group="2" rgba="1 .6 .2 1"/>'
        f'<site name="hamster_eye"/></body>'
    )

    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, model, data, spawn):
        """Hamster 1 cm above the bottom of a shell centred at spawn; the foot force starts inside its grace."""
        data.qpos[HAM_QPOS:HAM_QPOS + 3] = spawn[0], spawn[1], spawn[2] - (REST - 0.01)
        data.qpos[HAM_QPOS + 3:HAM_QPOS + 7] = 1, 0, 0, 0
        data.qvel[HAM_DOF:HAM_DOF + 6] = 0
        data.userdata[LAST_TOUCH] = data.time

    def obs(self, model, data, facing):
        return {"wall_contact": data.time - data.userdata[LAST_TOUCH] <= GRACE}

    def drive(self, model, data, action, facing):
        """Foot force pair and grip couple into xfrc_applied. Needs fresh xpos/xmat/xipos for the current qpos."""
        cx, cy, cz = data.xpos[SHELL].tolist()
        hx, hy, hz = data.xipos[HAM].tolist()
        rx, ry, rz = hx - cx, hy - cy, hz - cz
        sm = data.xmat[SHELL].tolist()  # row-major shell rotation; p = R^T r is the hamster in the shell frame
        p = (sm[0] * rx + sm[3] * ry + sm[6] * rz, sm[1] * rx + sm[4] * ry + sm[7] * rz, sm[2] * rx + sm[5] * ry + sm[8] * rz)
        ud = data.userdata
        if REST - float((TILE_NORMALS @ p).max()) <= TOUCH:  # airborne cut
            ud[LAST_TOUCH] = data.time
        if data.time - ud[LAST_TOUCH] > GRACE:
            return

        ax, ay = float(action[0]), float(action[1])
        mag = math.hypot(ax, ay)
        e = min(mag, 1.0)
        if mag > 0:  # camera frame -> world: rotate by the facing
            ax, ay = ax / mag, ay / mag
            e *= ay * ay * (1.0 if ay >= 0 else BACK) + ax * ax * SIDE
            cyw, syw = math.cos(facing), math.sin(facing)
            ax, ay = ax * cyw - ay * syw, ax * syw + ay * cyw
        sl, cl = math.sin(LEAN * e), math.cos(LEAN * e)
        tx, ty, tz = sl * ax, sl * ay, -cl  # target lean

        dist = math.sqrt(rx * rx + ry * ry + rz * rz) + 1e-9
        nx, ny, nz = rx / dist, ry / dist, rz / dist
        qv = data.qvel
        vx, vy, vz = (qv[HAM_DOF] - qv[SHELL_DOF]), (qv[HAM_DOF + 1] - qv[SHELL_DOF + 1]), (qv[HAM_DOF + 2] - qv[SHELL_DOF + 2])
        fx = KP * REST * (tx - nx) - KD * vx
        fy = KP * REST * (ty - ny) - KD * vy
        fz = KP * REST * (tz - nz) - KD * vz
        fn = fx * nx + fy * ny + fz * nz
        fx, fy, fz = fx - fn * nx, fy - fn * ny, fz - fn * nz
        f = math.sqrt(fx * fx + fy * fy + fz * fz)
        if f > F_MAX:
            fx, fy, fz = fx * F_MAX / f, fy * F_MAX / f, fz * F_MAX / f

        # +F on the hamster and -F on the shell, both at the foot on the inner wall, as forces plus torques about each COM
        px, py, pz = cx + (R - T) * nx, cy + (R - T) * ny, cz + (R - T) * nz
        ax_, ay_, az_ = px - hx, py - hy, pz - hz
        sx, sy, sz = data.xipos[SHELL].tolist()
        bx, by, bz = px - sx, py - sy, pz - sz

        # grip couple: -K (w_h - w_s), world-frame angular velocities from the free joints' local ones
        hm = data.xmat[HAM].tolist()
        w0, w1, w2 = qv[HAM_DOF + 3], qv[HAM_DOF + 4], qv[HAM_DOF + 5]
        u0, u1, u2 = qv[SHELL_DOF + 3], qv[SHELL_DOF + 4], qv[SHELL_DOF + 5]
        gx = -K_GRIP * (hm[0] * w0 + hm[1] * w1 + hm[2] * w2 - sm[0] * u0 - sm[1] * u1 - sm[2] * u2)
        gy = -K_GRIP * (hm[3] * w0 + hm[4] * w1 + hm[5] * w2 - sm[3] * u0 - sm[4] * u1 - sm[5] * u2)
        gz = -K_GRIP * (hm[6] * w0 + hm[7] * w1 + hm[8] * w2 - sm[6] * u0 - sm[7] * u1 - sm[8] * u2)

        xf = data.xfrc_applied
        h, s = xf[HAM], xf[SHELL]
        h[0] += fx
        h[1] += fy
        h[2] += fz
        h[3] += ay_ * fz - az_ * fy + gx
        h[4] += az_ * fx - ax_ * fz + gy
        h[5] += ax_ * fy - ay_ * fx + gz
        s[0] -= fx
        s[1] -= fy
        s[2] -= fz
        s[3] -= by * fz - bz * fy + gx
        s[4] -= bz * fx - bx * fz + gy
        s[5] -= bx * fy - by * fx + gz
