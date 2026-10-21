"""The core step (spec §4): camera yaw, shell drag, containment, then the tier's drive, once per physics step.

Every per-step function is a pure function of model and data fields plus the action. All state lives in qpos,
qvel and userdata (slot 0 the camera yaw, the tier's slots after it), so one mj_getState(mjSTATE_INTEGRATION)
vector is the core state.
"""

import math
from typing import Protocol

import gymnasium as gym
import mujoco
import numpy as np

from .abstract_tier import AbstractTier
from .model import DT, FLAT, GAME_SIZE, HAM, HAM_DOF, HAM_QPOS, M_HAM, M_SHELL, R, SHELL, SHELL_DOF, SHELL_QPOS, build

FRAME_SKIP = 10  # physics steps per env step: 60 Hz
DRAG_K = 0.059  # N s^2/m^2 on the shell: sets the full-effort top speed
CHASE_ALPHA = 1 - math.exp(-DT / 0.55)  # chase yaw, tau 0.55 s
CHASE_MIN_SPEED2 = 0.8 ** 2  # (m/s)^2, horizontal shell speed below which the chase yaw holds
ORBIT_RATE = math.radians(120)  # rad/s at cam_yaw = 1
FIELD_START, FIELD_HARD = 0.005, 0.020  # m past the resting radius: containment starts, and stops hard
MU = M_HAM * M_SHELL / (M_HAM + M_SHELL)  # the pair's reduced mass
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
TAU = 2 * math.pi


class Tier(Protocol):
    """What the core needs from a control tier (spec §6)."""

    fragment: str  # MJCF: a free body named "hamster", 2.4 kg in total, geoms in group 2, a site "hamster_eye"
    nuserdata: int  # its userdata slots, after the core's slot 0
    resting_radius: float  # m, shell centre to hamster subtree centre of mass at rest

    def action_space(self) -> gym.spaces.Box: ...
    def reset(self, model, data, spawn) -> None: ...
    def drive(self, model, data, action, yaw: float) -> None: ...
    def obs(self, model, data, yaw: float) -> dict: ...


def wrap(a):
    """Angle into (-pi, pi]."""
    a = math.remainder(a, TAU)
    return math.pi if a == -math.pi else a


def camera_yaw(data, cam):
    """Chase the shell's heading, add the orbit input, store and return the yaw."""
    yaw = data.userdata[0]
    vx, vy = data.qvel[SHELL_DOF], data.qvel[SHELL_DOF + 1]
    if vx * vx + vy * vy > CHASE_MIN_SPEED2:
        yaw += CHASE_ALPHA * wrap(math.atan2(-vx, vy) - yaw)
    yaw = wrap(yaw + ORBIT_RATE * cam * DT)
    data.userdata[0] = yaw
    return yaw


def drag(data):
    """-k |v| v on the shell, grounded or not."""
    vx, vy, vz = data.qvel[SHELL_DOF:SHELL_DOF + 3].tolist()
    k = -DRAG_K * math.sqrt(vx * vx + vy * vy + vz * vz)
    f = data.xfrc_applied[SHELL]
    f[0] += k * vx
    f[1] += k * vy
    f[2] += k * vz


def contain(model, data, resting_radius):
    """Keep the hamster inside: an inelastic impulse past FIELD_START, a position stop at FIELD_HARD.

    Conserves momentum, angular momentum, and the pair's centre of mass. Needs fresh xpos and subtree_com.
    """
    cx, cy, cz = data.xpos[SHELL].tolist()
    hx, hy, hz = data.subtree_com[HAM].tolist()
    rx, ry, rz = hx - cx, hy - cy, hz - cz
    dist = math.sqrt(rx * rx + ry * ry + rz * rz)
    depth = dist - resting_radius - FIELD_START
    if depth <= 0:
        return
    ux, uy, uz = rx / dist, ry / dist, rz / dist
    qv = data.qvel
    w = (qv[HAM_DOF] - qv[SHELL_DOF]) * ux + (qv[HAM_DOF + 1] - qv[SHELL_DOF + 1]) * uy + (qv[HAM_DOF + 2] - qv[SHELL_DOF + 2]) * uz
    if w > 0:
        j = min(1.0, depth / (FIELD_HARD - FIELD_START)) * MU * w
        jh, js = j / M_HAM, j / M_SHELL
        qv[HAM_DOF] -= jh * ux
        qv[HAM_DOF + 1] -= jh * uy
        qv[HAM_DOF + 2] -= jh * uz
        qv[SHELL_DOF] += js * ux
        qv[SHELL_DOF + 1] += js * uy
        qv[SHELL_DOF + 2] += js * uz
    over = depth - (FIELD_HARD - FIELD_START)
    if over > 0:  # a sustained load still creeps past the impulse, so stop it in position
        qp = data.qpos
        oh, os_ = over * M_SHELL / (M_HAM + M_SHELL), over * M_HAM / (M_HAM + M_SHELL)
        qp[HAM_QPOS] -= oh * ux
        qp[HAM_QPOS + 1] -= oh * uy
        qp[HAM_QPOS + 2] -= oh * uz
        qp[SHELL_QPOS] += os_ * ux
        qp[SHELL_QPOS + 1] += os_ * uy
        qp[SHELL_QPOS + 2] += os_ * uz
        mujoco.mj_forward(model, data)


def physics_step(model, data, tier, action, cam):
    """One physics step. action is the tier's part; cam is the orbit input in [-1, 1]."""
    data.xfrc_applied.fill(0)
    # mj_step leaves positions computed for the state before it, so refresh them: the step then reads only the state
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    yaw = camera_yaw(data, cam)
    drag(data)
    contain(model, data, tier.resting_radius)
    tier.drive(model, data, action, yaw)
    mujoco.mj_step(model, data)


class Core:
    """A compiled hamsterball and its data. Holds constants only; the state is in data."""

    def __init__(self, tier: Tier | None = None, level=FLAT, camera_action=False, view_size=GAME_SIZE):
        self.tier = tier or AbstractTier()
        self.camera_action = camera_action
        self.model = build(self.tier, level, view_size)
        self.data = mujoco.MjData(self.model)
        self.reset()

    @property
    def action_space(self):
        box = self.tier.action_space()
        if not self.camera_action:
            return box
        return gym.spaces.Box(np.append(box.low, -1).astype(box.dtype), np.append(box.high, 1).astype(box.dtype), dtype=box.dtype)

    @property
    def yaw(self):
        return float(self.data.userdata[0])

    def reset(self, spawn=(0.0, 0.0, R), yaw=0.0):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[SHELL_QPOS:SHELL_QPOS + 3] = spawn
        self.tier.reset(m, d, spawn)
        d.userdata[0] = yaw
        mujoco.mj_forward(m, d)

    def step(self, action, n=FRAME_SKIP):
        """Hold one action over n physics steps."""
        m, d, tier = self.model, self.data, self.tier
        cam = 0.0
        if self.camera_action:
            action, cam = action[:-1], float(action[-1])
        for _ in range(n):
            physics_step(m, d, tier, action, cam)

    def state(self):
        s = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, s, STATE)
        return s

    def set_state(self, s):
        mujoco.mj_setState(self.model, self.data, s, STATE)
        mujoco.mj_forward(self.model, self.data)
