"""The core step (spec §4): camera yaw, shell drag, containment, then the tier's drive, once per physics step.

Every per-step function is a pure function of model and data fields plus the action (and a level's constant body
motions). All state lives in time, qpos, qvel and userdata (the core's slots, `model.YAW` first, then the tier's, then
a level's collapsing tiles'), so one mj_getState(mjSTATE_INTEGRATION) vector is the core state.
"""

import math
from typing import NamedTuple, Protocol

import gymnasium as gym
import mujoco
import numpy as np

from .abstract_tier import AbstractTier
from .level import Level
from .motion import pose, quat
from .model import (CORE_SLOTS, DT, FLAT, GAME_SIZE, HAM, HAM_DOF, HAM_QPOS, M_HAM, M_SHELL, R, SHELL, SHELL_DOF,
                    SHELL_QPOS, PITCH, YAW, build)

FRAME_SKIP = 10  # physics steps per env step: 60 Hz
DRAG_K = 0.059  # N s^2/m^2 on the shell: sets the full-effort top speed
CHASE_ALPHA = 1 - math.exp(-DT / 0.55)  # chase yaw, tau 0.55 s
CHASE_MIN_SPEED2 = 0.8 ** 2  # (m/s)^2, horizontal shell speed below which the chase yaw holds
ORBIT_RATE = math.radians(120)  # rad/s at cam_yaw = 1
# camera pitch, below the horizon; needs play-test (see step.md)
PITCH_DEFAULT, PITCH_MIN, PITCH_MAX = math.radians(15), math.radians(-10), math.radians(60)
PITCH_RATE = math.radians(90)  # rad/s at cam_pitch = 1
FIELD_START, FIELD_HARD = 0.005, 0.020  # m past the resting radius: containment starts, and stops hard
MU = M_HAM * M_SHELL / (M_HAM + M_SHELL)  # the pair's reduced mass
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
TAU = 2 * math.pi
G = 9.81  # m/s^2, a fallen tile's acceleration (MuJoCo's default gravity)
TILE_DROP = 1.0  # m under the kill plane where a fallen tile stays


class Body(NamedTuple):
    """A level body's addresses in the compiled model, its authored pos, and its motion (None for a seesaw)."""

    qpos: int
    dof: int
    pos: np.ndarray
    motion: dict | None


class Tile(NamedTuple):
    """A collapsing tile's addresses, authored pos, and the z offset it stops falling at (under the kill plane)."""

    qpos: int
    dof: int
    pos: np.ndarray
    bottom: float


class Tiles(NamedTuple):
    """A level's collapsing tiles: the first trigger-time userdata slot, the first tile's body id, the delay."""

    slot: int
    body: int
    delay: float
    tiles: list


class Tier(Protocol):
    """What the core needs from a control tier (spec §6)."""

    fragment: str  # MJCF: a free body named "hamster", 2.4 kg in total, geoms in group 2, a site "hamster_eye"
    nuserdata: int  # its userdata slots, after the core's CORE_SLOTS
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
    """Turn toward the shell's direction of travel, add the orbit input, store and return the yaw."""
    yaw = data.userdata[YAW]
    vx, vy = data.qvel[SHELL_DOF], data.qvel[SHELL_DOF + 1]
    if vx * vx + vy * vy > CHASE_MIN_SPEED2:
        yaw += CHASE_ALPHA * wrap(math.atan2(-vx, vy) - yaw)
    yaw = wrap(yaw + ORBIT_RATE * cam * DT)
    data.userdata[YAW] = yaw
    return yaw


def camera_pitch(data, cam):
    """Add the pitch input and clamp to [PITCH_MIN, PITCH_MAX]. Moves no physics."""
    data.userdata[PITCH] = min(max(data.userdata[PITCH] + PITCH_RATE * cam * DT, PITCH_MIN), PITCH_MAX)


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


def drive_bodies(data, bodies):
    """Pose each kinematic body at its motion's pose at data.time, with the velocity that carries it to the pose one
    step later (the secant, so friction sees the body's real motion). bodies: `Body`s; a seesaw (motion None) its
    joint moves, and this leaves it alone."""
    t = data.time
    for qa, da, pos, motion in bodies:
        if motion is None:
            continue
        o0, o1 = pose(motion, t)[0], pose(motion, t + DT)[0]
        q0, q1 = quat(motion, t), quat(motion, t + DT)
        data.qpos[qa:qa + 3] = pos + o0
        data.qpos[qa + 3:qa + 7] = q0
        data.qvel[da:da + 3] = (o1 - o0) / DT
        w = np.empty(3)
        mujoco.mju_subQuat(w, q1, q0)  # q0 * quat(w) = q1: the turn in the body frame, as a free joint's qvel is
        data.qvel[da + 3:da + 6] = w / DT


def _fall(s, bottom):
    """A tile's z offset s seconds into its fall: free fall at g, held at `bottom` (under the kill plane)."""
    return 0.0 if s <= 0 else max(-0.5 * G * s * s, bottom)


def drive_tiles(data, tiles):
    """Pose each collapsing tile from data.time and its trigger time (its userdata slot, +inf until the shell touches
    it): at rest until `delay` after the trigger, then falling, with the secant velocity as `drive_bodies`.
    tiles: `Tiles`, or None."""
    if tiles is None:
        return
    slot, _, delay, adrs = tiles
    t = data.time
    for i, (qa, da, pos, bottom) in enumerate(adrs):
        start = data.userdata[slot + i] + delay
        z0, z1 = _fall(t - start, bottom), _fall(t + DT - start, bottom)
        data.qpos[qa:qa + 3] = pos
        data.qpos[qa + 2] += z0
        data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[da:da + 6] = (0.0, 0.0, (z1 - z0) / DT, 0.0, 0.0, 0.0)


def touch_tiles(model, data, tiles):
    """Store data.time in the slot of every untouched tile the shell touches. Call right after mj_step: its contacts
    come from the state the step started from, so this reads only the core state."""
    if tiles is None or not data.ncon:
        return
    slot, b0, _, adrs = tiles
    for b1, b2 in model.geom_bodyid[data.contact.geom[:data.ncon]].tolist():
        i = (b2 if b1 == SHELL else b1 if b2 == SHELL else -1) - b0
        if 0 <= i < len(adrs) and data.userdata[slot + i] == math.inf:
            data.userdata[slot + i] = data.time


def physics_step(model, data, tier, action, cam, bodies=(), tiles=None, cam_pitch=0.0):
    """One physics step. action is the tier's part; cam and cam_pitch are the orbit and pitch inputs in [-1, 1];
    bodies as `drive_bodies`, tiles as `drive_tiles`."""
    data.xfrc_applied.fill(0)
    drive_bodies(data, bodies)
    drive_tiles(data, tiles)
    # mj_step leaves positions computed for the state before it, so refresh them: the step then reads only the state
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    yaw = camera_yaw(data, cam)
    camera_pitch(data, cam_pitch)
    drag(data)
    contain(model, data, tier.resting_radius)
    tier.drive(model, data, action, yaw)
    mujoco.mj_step(model, data)
    touch_tiles(model, data, tiles)


class Core:
    """A compiled hamsterball and its data. Holds constants only (the level's body motions among them); the state is
    in data."""

    def __init__(self, tier: Tier | None = None, level=FLAT, camera_action=False, view_size=GAME_SIZE):
        """`level` is a Level, or MJCF sections as `model.build` takes them (no kinematic bodies)."""
        self.tier = tier or AbstractTier()
        self.camera_action = camera_action
        if isinstance(level, Level):
            c = level.collapse
            self.model = m = build(self.tier, level.mjcf(), view_size, level.body_mjcf() + level.tile_mjcf(),
                                   len(c["tiles"]) if c else 0)

            def adr(name):  # the body's first joint's qpos and dof addresses
                j = m.body_jntadr[m.body(name).id]
                return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])

            self.bodies = [Body(*adr(f"body{i}"), np.array(b["pos"]), b.get("motion"))
                           for i, b in enumerate(level.bodies)]
            self.tiles = None
            if c:
                self.tiles = Tiles(CORE_SLOTS + self.tier.nuserdata, m.body("tile0").id, c["delay"],
                                   [Tile(*adr(f"tile{i}"), np.array(g["pos"]),
                                         level.kill_plane_z - TILE_DROP - g["pos"][2]) for i, g in enumerate(c["tiles"])])
        else:
            self.model, self.bodies, self.tiles = build(self.tier, level, view_size), [], None
        self.data = mujoco.MjData(self.model)
        self.reset()

    @property
    def action_space(self):
        box = self.tier.action_space()
        if not self.camera_action:
            return box
        low, high = np.append(box.low, [-1, -1]), np.append(box.high, [1, 1])  # cam_yaw, cam_pitch
        return gym.spaces.Box(low.astype(box.dtype), high.astype(box.dtype), dtype=box.dtype)

    @property
    def yaw(self):
        return float(self.data.userdata[YAW])

    @property
    def pitch(self):
        return float(self.data.userdata[PITCH])

    def reset(self, spawn=(0.0, 0.0, R), yaw=0.0):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[SHELL_QPOS:SHELL_QPOS + 3] = spawn
        self.tier.reset(m, d, spawn)
        d.userdata[YAW], d.userdata[PITCH] = yaw, PITCH_DEFAULT
        if self.tiles:
            d.userdata[self.tiles.slot:] = math.inf  # untouched
        drive_bodies(d, self.bodies)
        drive_tiles(d, self.tiles)
        mujoco.mj_forward(m, d)

    def step(self, action, n=FRAME_SKIP):
        """Hold one action over n physics steps."""
        m, d, tier = self.model, self.data, self.tier
        cam = cam_pitch = 0.0
        if self.camera_action:
            action, cam, cam_pitch = action[:-2], float(action[-2]), float(action[-1])
        for _ in range(n):
            physics_step(m, d, tier, action, cam, self.bodies, self.tiles, cam_pitch)

    def state(self):
        s = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, s, STATE)
        return s

    def set_state(self, s):
        mujoco.mj_setState(self.model, self.data, s, STATE)
        mujoco.mj_forward(self.model, self.data)
