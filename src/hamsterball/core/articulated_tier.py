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
HIP_RANGE, KNEE_RANGE = (-1.5, 1.5), (-1.8, 0.05)
GEAR_HIP, GEAR_KNEE = 16.0, 12.8  # peak torques
DAMPING, ARMATURE = 0.5, 0.01  # damping and armature keep a 60 Hz PD (kd up to 1) stable on 0.0003 kg m^2 legs
STAND = (0.28, -0.87) * 4  # hip, knee per leg (hip swings the foot backward, knee bends the shin forward)
# Resting radius: the subtree COM's largest distance from the shell centre, lying on the wall (see the doc).
RESTING_RADIUS = 0.40

LEGS = [("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)]  # name, x sign, y sign
_GEOM = f'solref="{SOLREF}" contype="2" conaffinity="0" group="2"'
_FROZEN = f'{_GEOM} priority="1" condim="1"'  # frictionless against the tiles
_RGBA = 'rgba="1 .6 .2 1"'  # torso
_LEG = 'rgba=".62 .33 .12 1"'
_HEAD = 'rgba="1 .88 .62 1"'
_VISUAL = 'mass="0" contype="0" conaffinity="0" group="2"'  # marker geoms: no mass, no contact


DEG = math.degrees  # MJCF's compiler angle is degrees (the levels rely on it); the ranges are written in degrees


def _leg(name, sx, sy):
    return (
        f'<body name="{name}_thigh" pos="{sx * HIP_X} {sy * HIP_Y} {HIP_Z}">'
        f'<joint name="{name}_hip" axis="0 1 0" range="{DEG(HIP_RANGE[0])} {DEG(HIP_RANGE[1])}" damping="{DAMPING}" armature="{ARMATURE}"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {-THIGH}" size="0.015" mass="{MASS["thigh"]}" {_FROZEN} {_LEG}/>'
        f'<body name="{name}_shin" pos="0 0 {-THIGH}">'
        f'<joint name="{name}_knee" axis="0 1 0" range="{DEG(KNEE_RANGE[0])} {DEG(KNEE_RANGE[1])}" damping="{DAMPING}" armature="{ARMATURE}"/>'
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


def _fragment():
    return (
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
