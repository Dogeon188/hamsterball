"""Views (spec §11): a model <camera> on a mocap body per view, posed from the core state once per env step."""

import math

import mujoco
import numpy as np

from .model import FACING, HAM, PITCH, PITCH_DEFAULT, SHELL

VIEWS = ("chase", "follow", "first_person")

# Look constants, picked by eye and awaiting play-test (see views.md). They change pixels, not trajectories.
CHASE_DIST = 3.0  # m from the shell centre; the camera pitch sets its angle below the horizon
FOLLOW_DIST, FOLLOW_PITCH = 6.0, math.radians(35)
FOVY = {"chase": 60, "follow": 50, "first_person": 75}  # vertical, degrees
TAIL_OFFSET, TAIL_LEN, TAIL_R = 0.13, 0.17, 0.06  # m: base behind the hamster centre, cone length, base radius
TAIL_RGBA = ".35 .18 .08 1"

# Mocap bodies with no geoms: no mass and no contacts, so they never touch the physics.
FRAGMENT = "\n".join(f'<body name="view_{v}" mocap="true"><camera name="{v}" fovy="{FOVY[v]}"/></body>' for v in VIEWS)
# The tail: a cone (a mesh, as MuJoCo has no cone primitive) along its body's -y, base at the origin. A mocap body,
# massless and contactless like the views; group 2, the hamster's, so first person hides it.
_RING = " ".join(f"{TAIL_R * math.cos(a)!r} 0 {TAIL_R * math.sin(a)!r}" for a in np.linspace(0, 2 * math.pi, 8, endpoint=False))
ASSET = f'<asset><mesh name="tail" vertex="{_RING} 0 {-TAIL_LEN} 0"/></asset>'
FRAGMENT += (f'\n<body name="tail" mocap="true"><geom name="tail" type="mesh" mesh="tail" mass="0" contype="0" '
             f'conaffinity="0" group="2" rgba="{TAIL_RGBA}"/></body>')


def _quat(yaw, pitch):
    """Orientation of a camera facing yaw `yaw` (0 = +y, as the facing) tilted `pitch` below the horizon.

    A MuJoCo camera looks along its -z with +y up.
    """
    sy, cy, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    fwd = (-sy * cp, cy * cp, -sp)
    right = (cy, sy, 0.0)
    up = (-sy * sp, cy * sp, cp)  # right x fwd
    mat = np.array([right, up, [-f for f in fwd]]).T.flatten()
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, mat)
    return q, fwd


def pose(model, data, follow_yaw):
    """Pose the view and tail mocaps from fresh positions (after mj_kinematics); a later mj_kinematics moves the cameras."""
    facing, pitch = float(data.userdata[FACING]), float(data.userdata[PITCH])
    c = data.xpos[SHELL]
    for v, at, yw, p, dist in (
        ("chase", c, facing, pitch, CHASE_DIST),
        ("follow", c, follow_yaw, FOLLOW_PITCH, FOLLOW_DIST),
        ("first_person", data.site("hamster_eye").xpos, facing, pitch - PITCH_DEFAULT, 0.0),  # level at the default
    ):
        mocap = model.body_mocapid[model.body(f"view_{v}").id]
        q, fwd = _quat(yw, p)
        data.mocap_pos[mocap] = at - dist * np.array(fwd)
        data.mocap_quat[mocap] = q
    # the tail: TAIL_OFFSET behind the hamster centre, its body's -y turned to -facing
    mocap = model.body_mocapid[model.body("tail").id]
    data.mocap_pos[mocap] = data.xpos[HAM] + TAIL_OFFSET * np.array([math.sin(facing), -math.cos(facing), 0.0])
    data.mocap_quat[mocap] = (math.cos(facing / 2), 0.0, 0.0, math.sin(facing / 2))


def option(view):
    """The MjvOption a view renders with: group 3 (inner tiles) hidden everywhere; first person also hides 1 and 2."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    if view != "first_person":
        opt.geomgroup[1] = opt.geomgroup[2] = 1
    return opt
