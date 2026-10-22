"""The core state vector restores a run bit for bit (spec §4)."""

import math

import numpy as np
import pytest

from hamsterball.core import Core


def drive(core, start, stop):
    """Env steps start..stop of a run that turns and orbits the camera."""
    out = []
    for i in range(start, stop):
        a = np.array([np.sin(i / 20), 1.0, 1.0 if i % 60 < 20 else 0.0, np.cos(i / 15)], np.float32)
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
    # yaw, pitch and the airborne-cut timer ride along in the state
    assert replay.yaw == core.yaw != 0
    assert replay.pitch == core.pitch != Core().pitch


def test_camera_flag_appends_cam_yaw_and_cam_pitch_to_the_tier_action_space():
    assert Core().action_space.shape == (2,)
    space = Core(camera_action=True).action_space
    assert space.shape == (4,) and (space.low == -1).all() and (space.high == 1).all()


def test_cam_pitch_turns_the_pitch_at_90_deg_per_s_within_minus_10_to_60_deg():
    core = Core(camera_action=True)
    assert math.degrees(core.pitch) == pytest.approx(15)  # the default: 15 deg below the horizon
    for _ in range(15):  # a quarter second at 60 Hz
        core.step(np.array([0, 0, 0, -1], np.float32))
    assert math.degrees(core.pitch) == pytest.approx(15 - 22.5)
    for _ in range(60):
        core.step(np.array([0, 0, 0, 1], np.float32))
    assert math.degrees(core.pitch) == pytest.approx(60)
    for _ in range(60):
        core.step(np.array([0, 0, 0, -1], np.float32))
    assert math.degrees(core.pitch) == pytest.approx(-10)
    core.reset()
    assert math.degrees(core.pitch) == pytest.approx(15)


def test_without_the_camera_flag_the_pitch_holds_its_default():
    core = Core()
    for _ in range(30):
        core.step(np.array([0, 1], np.float32))
    assert math.degrees(core.pitch) == pytest.approx(15)


def test_cam_pitch_moves_no_physics():
    runs = []
    for sign in (1, -1):
        core = Core(camera_action=True)
        for i in range(120):
            core.step(np.array([np.sin(i / 20), 1.0, 0.5, sign * np.cos(i / 15)], np.float32))
        d = core.data
        runs.append((d.qpos.copy(), d.qvel.copy(), d.time, core.pitch))
    (q0, v0, t0, p0), (q1, v1, t1, p1) = runs
    assert np.array_equal(q0, q1) and np.array_equal(v0, v1) and t0 == t1 and p0 != p1
