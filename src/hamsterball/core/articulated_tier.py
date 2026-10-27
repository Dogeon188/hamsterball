"""The articulated control tier (spec: articulated-tier): a jointed quadruped driven by one normalised torque per joint."""

import math

import gymnasium as gym
import mujoco
import numpy as np

from .model import FACING, HAM, HAM_DOF, HAM_QPOS, M_HAM, R, SOLREF, T

N_JOINTS = 8  # per leg: hip, knee; legs front-left, front-right, rear-left, rear-right
INNER = R - T  # inner wall radius (m)

# Skeleton (m, kg, N m, rad); first-cut constants, see articulated_tier.md. The torso's forward axis is +x.
TORSO_R, TORSO_HALF = 0.05, 0.08  # torso capsule radius and half length
HEAD_R, HEAD_POS = 0.045, (0.17, 0, 0.025)
HIP_X, HIP_Y, HIP_Z = 0.08, 0.085, -0.02  # hip offsets from the torso centre (x sign: front +, rear -; y: left +)
THIGH, SHIN, FOOT_R = 0.09, 0.09, 0.02
MASS = {"torso": 1.38, "head": 0.30, "thigh": 0.10, "shin": 0.06, "foot": 0.02}  # sums to M_HAM over 4 legs
HIP_RANGE, KNEE_RANGE = (-0.9, 0.9), (-1.8, 0.05)
GEAR_HIP, GEAR_KNEE = 5.0, 4.0  # peak torques
DAMPING, ARMATURE = 0.05, 0.002
STAND = (0.4, -0.8) * 4  # hip, knee per leg (hip swings the foot backward, knee bends the shin forward)
# Resting radius: the subtree COM's largest distance from the shell centre, lying on the wall (see the doc).
RESTING_RADIUS = 0.40

LEGS = [("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)]  # name, x sign, y sign
_GEOM = f'solref="{SOLREF}" contype="2" conaffinity="0" group="2"'
_FROZEN = f'{_GEOM} priority="1" condim="1"'  # frictionless against the tiles
_RGBA = 'rgba="1 .6 .2 1"'  # torso
_LEG = 'rgba=".62 .33 .12 1"'
_HEAD = 'rgba="1 .88 .62 1"'
_VISUAL = 'mass="0" contype="0" conaffinity="0" group="2"'  # marker geoms: no mass, no contact


def _leg(name, sx, sy):
    return (
        f'<body name="{name}_thigh" pos="{sx * HIP_X} {sy * HIP_Y} {HIP_Z}">'
        f'<joint name="{name}_hip" axis="0 1 0" range="{HIP_RANGE[0]} {HIP_RANGE[1]}" damping="{DAMPING}" armature="{ARMATURE}"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {-THIGH}" size="0.015" mass="{MASS["thigh"]}" {_FROZEN} {_LEG}/>'
        f'<body name="{name}_shin" pos="0 0 {-THIGH}">'
        f'<joint name="{name}_knee" axis="0 1 0" range="{KNEE_RANGE[0]} {KNEE_RANGE[1]}" damping="{DAMPING}" armature="{ARMATURE}"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {-SHIN}" size="0.012" mass="{MASS["shin"]}" {_FROZEN} {_LEG}/>'
        f'<geom name="{name}_foot" type="sphere" pos="0 0 {-SHIN}" size="{FOOT_R}" mass="{MASS["foot"]}" {_GEOM} '
        f'margin="0.002" condim="3" rgba=".25 .15 .1 1"/>'
        f'</body></body>'
    )


def _stand_height():
    """Torso centre height over the shell centre with every foot just touching the inner wall, torso level."""
    z = -1e9
    for i, (_, sx, sy) in enumerate(LEGS):
        h, k = STAND[2 * i], STAND[2 * i + 1]
        fx = sx * HIP_X + THIGH * math.sin(h) + SHIN * math.sin(h + k)
        fz = HIP_Z - THIGH * math.cos(h) - SHIN * math.cos(h + k)
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


class ArticulatedTier:
    nuserdata = 0
    features = False  # no tail or ears: its own head and torso show which way it points
    resting_radius = RESTING_RADIUS
    fragment = (
        f'<body name="hamster" pos="0 0 {R + _stand_height()!r}"><freejoint/>'
        f'<geom type="capsule" fromto="{-TORSO_HALF} 0 0 {TORSO_HALF} 0 0" size="{TORSO_R}" mass="{MASS["torso"]}" '
        f'{_FROZEN} {_RGBA}/>'
        f'<body name="head" pos="{HEAD_POS[0]} {HEAD_POS[1]} {HEAD_POS[2]}">'
        f'<geom type="sphere" size="{HEAD_R}" mass="{MASS["head"]}" {_FROZEN} {_HEAD}/>'
        # nose and eyes, so the head reads as facing +x
        f'<geom type="sphere" pos="{HEAD_R * 0.95} 0 -0.005" size="0.012" {_VISUAL} rgba=".15 .08 .08 1"/>'
        f'<geom type="sphere" pos="{HEAD_R * 0.7} 0.025 0.025" size="0.01" {_VISUAL} rgba="0 0 0 1"/>'
        f'<geom type="sphere" pos="{HEAD_R * 0.7} -0.025 0.025" size="0.01" {_VISUAL} rgba="0 0 0 1"/>'
        f'<site name="hamster_eye" pos="0.03 0 0.01"/></body>'
        + "".join(_leg(*leg) for leg in LEGS)
        + '</body>'
    )
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
        """wall_contact: any hamster geom touches a tile. Runs collision on the current state (see step.md)."""
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        root = model.body_rootid
        for g1, g2 in data.contact.geom[:data.ncon].tolist():
            b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
            if (b1 == 1 and root[b2] == HAM) or (b2 == 1 and root[b1] == HAM):
                return {"wall_contact": True}
        return {"wall_contact": False}
