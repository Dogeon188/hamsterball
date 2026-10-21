"""The core state vector restores a run bit for bit (spec §4)."""

import numpy as np

from hamsterball.core import Core


def drive(core, start, stop):
    """Env steps start..stop of a run that turns and orbits the camera."""
    out = []
    for i in range(start, stop):
        a = np.array([np.sin(i / 20), 1.0, 1.0 if i % 60 < 20 else 0.0], np.float32)
        core.step(a)
        out.append(core.state())
    return out


def test_setting_a_stored_core_state_reproduces_the_run_bit_for_bit():
    core = Core(camera_action=True)
    d = core.data
    d.qpos[7:10] = d.qpos[0:3]
    d.qvel[8] = -12.0  # thrown down from the shell centre
    states = drive(core, 0, 120)  # the throw hits containment's hard stop early on

    replay = Core(camera_action=True)
    replay.set_state(states[49])
    assert np.array_equal(drive(replay, 50, 120)[-1], states[-1])
    # yaw and the airborne-cut timer ride along in the state
    assert replay.yaw == core.yaw != 0


def test_camera_flag_appends_cam_yaw_to_the_tier_action_space():
    assert Core().action_space.shape == (2,)
    assert Core(camera_action=True).action_space.shape == (3,)
