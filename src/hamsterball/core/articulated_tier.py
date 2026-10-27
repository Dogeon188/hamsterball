"""The articulated control tier (spec: articulated-tier): a jointed quadruped driven by one normalised torque per joint."""

import math

import gymnasium as gym
import mujoco
import numpy as np

from .model import FACING, HAM, HAM_DOF, HAM_QPOS, M_HAM, R, SOLREF, T

N_JOINTS = 8  # per leg: hip, knee; legs front-left, front-right, rear-left, rear-right
INNER = R - T  # inner wall radius (m)

# Skeleton (m, kg, N m, rad): the prototype's `fast` candidate, see articulated_tier.md. The torso's forward axis is +x.
TORSO_R, TORSO_HALF = 0.05, 0.0746  # spine capsule radius and half length (carries the torso's mass)
BAR_R = 0.04  # the pelvis and shoulder bars: capsules along y through the hips, so every leg attaches to solid body
HEAD_R, HEAD_POS = 0.045, (0.1925, 0, 0.025)
HIP_XF, HIP_XR, HIP_Y, HIP_Z = 0.0544, 0.1177, 0.1137, -0.0005  # hip offsets from the torso centre (rear hips sit at -HIP_XR)
THIGH_F, SHIN_F, THIGH_R, SHIN_R, FOOT_R = 0.06, 0.066, 0.1477, 0.06, 0.0246  # front and rear leg segments, foot radius
MU = 2.075  # foot friction (the tiles' is 1; the larger of the two applies)
M_HEAD, M_THIGH, M_SHIN, M_FOOT = 0.8, 0.0599, 0.0742, 0.02
M_TORSO = M_HAM - M_HEAD - 4 * (M_THIGH + M_SHIN + M_FOOT)  # the rest of the 2.4 kg
HIP_RANGE, KNEE_RANGE = (-0.508, 1.675), (-2.199, 0.0194)
GEAR_HIP, GEAR_KNEE = 17.15, 21.09  # peak torques
DAMPING, ARMATURE = 0.23, 0.005
STAND = (0.7128, -1.7116) * 2 + (0.5091, -1.7597) * 2  # hip, knee per leg (fl fr rl rr)
# Resting radius: the subtree COM's largest distance from the shell centre, limp on the wall (see the doc).
RESTING_RADIUS = 0.44

LEGS = [("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)]  # name, x sign, y sign
_GEOM = f'solref="{SOLREF}" contype="2" conaffinity="0" group="2"'
_FROZEN = f'{_GEOM} priority="1" condim="1"'  # frictionless against the tiles
_RGBA = 'rgba="1 .6 .2 1"'  # torso
_LEG = 'rgba=".62 .33 .12 1"'
_HEAD = 'rgba="1 .88 .62 1"'
_VISUAL = 'mass="0" contype="0" conaffinity="0" group="2"'  # marker geoms: no mass, no contact


DEG = math.degrees  # MJCF's compiler angle is degrees (the levels rely on it); the ranges are written in degrees


def _dims(sx):
    """(hip x offset, thigh, shin) of the front (sx > 0) or rear legs."""
    return (HIP_XF, THIGH_F, SHIN_F) if sx > 0 else (-HIP_XR, THIGH_R, SHIN_R)


def _leg(name, sx, sy):
    hx, thigh, shin = _dims(sx)
    return (
        f'<body name="{name}_thigh" pos="{hx} {sy * HIP_Y} {HIP_Z}">'
        f'<joint name="{name}_hip" axis="0 1 0" range="{DEG(HIP_RANGE[0])} {DEG(HIP_RANGE[1])}" damping="{DAMPING}" armature="{ARMATURE}"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {-thigh}" size="0.015" mass="{M_THIGH}" {_FROZEN} {_LEG}/>'
        f'<body name="{name}_shin" pos="0 0 {-thigh}">'
        f'<joint name="{name}_knee" axis="0 1 0" range="{DEG(KNEE_RANGE[0])} {DEG(KNEE_RANGE[1])}" damping="{DAMPING}" armature="{ARMATURE}"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {-shin}" size="0.012" mass="{M_SHIN}" {_FROZEN} {_LEG}/>'
        f'<geom name="{name}_foot" type="sphere" pos="0 0 {-shin}" size="{FOOT_R}" mass="{M_FOOT}" {_GEOM} '
        f'margin="0.002" condim="3" friction="{MU} .005 .0001" rgba=".25 .15 .1 1"/>'
        f'</body></body>'
    )


def _stand_height():
    """Torso centre height over the shell centre with every foot just touching the inner wall, torso level."""
    z = -1e9
    for i, (_, sx, sy) in enumerate(LEGS):
        hx, thigh, shin = _dims(sx)
        h, k = STAND[2 * i], STAND[2 * i + 1]
        fx = hx + thigh * math.sin(h) + shin * math.sin(h + k)
        fz = HIP_Z - thigh * math.cos(h) - shin * math.cos(h + k)
        z = max(z, -math.sqrt((INNER - FOOT_R) ** 2 - fx * fx - HIP_Y ** 2) - fz)
    return z


def _sections():
    motors = "".join(
        f'<motor name="{n}_{j}" joint="{n}_{j}" gear="{g}" ctrlrange="-1 1"/>'
        for n, _, _ in LEGS for j, g in (("hip", GEAR_HIP), ("knee", GEAR_KNEE))
    )
    z = R + _stand_height()
    key = " ".join(repr(float(v)) for v in (0, 0, R, 1, 0, 0, 0, 0, 0, z, 1, 0, 0, 0, *STAND))
    return f'<actuator>{motors}</actuator><keyframe><key name="stand" qpos="{key}"/></keyframe>'


def _fragment():
    return (
        f'<body name="hamster" pos="0 0 {R + _stand_height()!r}"><freejoint/>'
        f'<geom type="capsule" fromto="{-TORSO_HALF} 0 0 {TORSO_HALF} 0 0" size="{TORSO_R}" mass="{M_TORSO}" '
        f'{_FROZEN} {_RGBA}/>'
        + "".join(  # shoulder and pelvis bars, massless: they only widen the collision shape and the look
            f'<geom type="capsule" fromto="{x} {-HIP_Y} {HIP_Z} {x} {HIP_Y} {HIP_Z}" size="{BAR_R}" mass="0" {_FROZEN} {_RGBA}/>'
            for x in (HIP_XF, -HIP_XR))
        + f'<body name="head" pos="{HEAD_POS[0]} {HEAD_POS[1]} {HEAD_POS[2]}">'
        f'<geom type="sphere" size="{HEAD_R}" mass="{M_HEAD}" {_FROZEN} {_HEAD}/>'
        # nose and eyes, so the head reads as facing +x
        f'<geom type="sphere" pos="{HEAD_R * 0.95} 0 -0.005" size="0.012" {_VISUAL} rgba=".15 .08 .08 1"/>'
        f'<geom type="sphere" pos="{HEAD_R * 0.7} 0.025 0.025" size="0.01" {_VISUAL} rgba="0 0 0 1"/>'
        f'<geom type="sphere" pos="{HEAD_R * 0.7} -0.025 0.025" size="0.01" {_VISUAL} rgba="0 0 0 1"/>'
        f'<site name="hamster_eye" pos="0.03 0 0.01"/></body>'
        + "".join(_leg(*leg) for leg in LEGS)
        + '</body>'
)


class ArticulatedTier:
    nuserdata = 0
    features = False  # no tail or ears: its own head and torso show which way it points
    resting_radius = RESTING_RADIUS
    fragment = _fragment()
    _feet = None  # foot geom ids, found on first obs (the model is not known here)
    sections = _sections()  # top-level MJCF after the worldbody: the motors and the standing keyframe

    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)

    def reset(self, model, data, spawn):
        """Standing at the spawn (the keyframe's pose relative to the shell), torso forward along the facing."""
        key = model.key_qpos[0]
        data.qpos[HAM_QPOS:HAM_QPOS + 3] = np.asarray(spawn) + key[HAM_QPOS:HAM_QPOS + 3] - key[:3]
        yaw = float(data.userdata[FACING]) + math.pi / 2  # facing 0 looks along +y; the torso's forward is +x
        data.qpos[HAM_QPOS + 3:HAM_QPOS + 7] = math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
        data.qpos[HAM_QPOS + 7:HAM_QPOS + 7 + N_JOINTS] = key[HAM_QPOS + 7:HAM_QPOS + 7 + N_JOINTS]
        data.qvel[HAM_DOF:HAM_DOF + 6 + N_JOINTS] = 0
        data.ctrl[:N_JOINTS] = 0

    def drive(self, model, data, action, facing):
        data.ctrl[:N_JOINTS] = action

    def obs(self, model, data, facing):
        """wall_contact: any hamster geom touches a tile; proprio: 8 joint positions, 8 joint velocities, 4 foot-contact
        flags (fl fr rl rr, a foot on a tile), the torso's forward then up axis in the facing frame (26 floats).
        Runs collision on the current state (see step.md)."""
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        if self._feet is None:  # one tier object per compiled model
            self._feet = [model.geom(f"{n}_foot").id for n, _, _ in LEGS]
        root = model.body_rootid
        wall, feet = False, np.zeros(4, np.float32)
        for g1, g2 in data.contact.geom[:data.ncon].tolist():
            b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
            if (b1 == 1 and root[b2] == HAM) or (b2 == 1 and root[b1] == HAM):
                wall = True
                for i, name in enumerate(self._feet):
                    feet[i] = feet[i] or name in (g1, g2)
        axes = data.xmat[HAM].reshape(3, 3)[:, [0, 2]]  # torso forward (+x) and up (+z), world
        cy, sy = math.cos(facing), math.sin(facing)  # world -> facing frame: rotate by -facing about z
        axes = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]]) @ axes
        proprio = np.concatenate([
            data.qpos[HAM_QPOS + 7:HAM_QPOS + 7 + N_JOINTS], data.qvel[HAM_DOF + 6:HAM_DOF + 6 + N_JOINTS],
            feet, axes.T.ravel()], dtype=np.float32)
        return {"wall_contact": wall, "proprio": proprio}
