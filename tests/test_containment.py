"""Containment reads the hamster subtree's centre-of-mass velocity, so a swinging limb counts."""

import mujoco
import numpy as np

from hamsterball.core import Core
from hamsterball.core.abstract_tier import AbstractTier
from hamsterball.core.model import HAM_DOF, HAM_QPOS, SHELL_DOF, R
from hamsterball.core.step import contain


class TwoBodyTier(AbstractTier):
    """A light root and a heavy child on a vertical slide: the root can sit still while the centre of mass moves."""

    nuserdata = 1
    fragment = (
        '<body name="hamster" pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.05" mass="0.4" contype="0" conaffinity="0" group="2"/>'
        '<site name="hamster_eye"/>'
        '<body name="limb"><joint type="slide" axis="0 0 1"/>'
        '<geom type="sphere" size="0.05" mass="2.0" contype="0" conaffinity="0" group="2"/></body></body>'
    )


def test_a_swinging_limb_moving_the_com_outward_gets_the_containment_impulse():
    core = Core(TwoBodyTier())
    m, d = core.model, core.data
    d.qpos[HAM_QPOS:HAM_QPOS + 3] = (0, 0, R + 0.4)  # COM 0.4 m above the shell centre, past the 0.35 m rest + field
    d.qvel[:] = 0
    d.qvel[HAM_DOF + 6] = 3.0  # the limb rises; the root stays still
    mujoco.mj_forward(m, d)
    contain(m, d, 0.35)
    assert d.qvel[SHELL_DOF + 2] > 0  # the shell took the impulse
    assert d.qvel[HAM_DOF + 2] < 0  # and the root gave it up
