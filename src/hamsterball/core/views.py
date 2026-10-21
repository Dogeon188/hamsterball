"""Views (spec §11): a model <camera> on a mocap body per view, posed from the core state once per env step."""

import math

import mujoco
import numpy as np

from .model import SHELL

VIEWS = ("chase", "follow", "first_person")

# Look constants, picked by eye and awaiting play-test (see views.md). They change pixels, not trajectories.
CHASE_DIST, CHASE_PITCH = 3.0, math.radians(15)  # m from the shell centre; degrees below the horizon
FOLLOW_DIST, FOLLOW_PITCH = 6.0, math.radians(35)
FOVY = {"chase": 60, "follow": 50, "first_person": 75}  # vertical, degrees

# Mocap bodies with no geoms: no mass and no contacts, so they never touch the physics.
FRAGMENT = "\n".join(f'<body name="view_{v}" mocap="true"><camera name="{v}" fovy="{FOVY[v]}"/></body>' for v in VIEWS)


def _quat(yaw, pitch):
    """Orientation of a camera facing yaw `yaw` (0 = +y, as the camera yaw) tilted `pitch` below the horizon.

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
    """Pose the view mocaps from fresh positions (after mj_kinematics); a later mj_kinematics moves the cameras."""
    yaw = float(data.userdata[0])
    c = data.xpos[SHELL]
    for v, at, yw, pitch, dist in (
        ("chase", c, yaw, CHASE_PITCH, CHASE_DIST),
        ("follow", c, follow_yaw, FOLLOW_PITCH, FOLLOW_DIST),
        ("first_person", data.site("hamster_eye").xpos, yaw, 0.0, 0.0),
    ):
        mocap = model.body_mocapid[model.body(f"view_{v}").id]
        q, fwd = _quat(yw, pitch)
        data.mocap_pos[mocap] = at - dist * np.array(fwd)
        data.mocap_quat[mocap] = q


def option(view):
    """The MjvOption a view renders with: group 3 (inner tiles) hidden everywhere; first person also hides 1 and 2."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    if view != "first_person":
        opt.geomgroup[1] = opt.geomgroup[2] = 1
    return opt
