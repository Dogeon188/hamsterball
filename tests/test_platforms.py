"""Driven platforms (survive-movers ticket 04): bodies that tilt, and a survive region that rides a body."""

import copy
import math

import gymnasium as gym
import mujoco
import numpy as np
import pytest
import yaml

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core import Core
from hamsterball.core.level import Level
from hamsterball.core.marker import centre, crosses
from hamsterball.core.motion import load
from hamsterball.env import LEVELS_DIR
from test_core_state import drive
from test_motion import run, track

TILT = {"type": "oscillate", "axis": [1, 0, 0], "amplitude": 0, "period": 4,
        "tilt": {"axis": [1, 0, 0], "amplitude": 10, "period": 8}}
BASE = yaml.safe_load("""
version: 1
kill_plane_z: -5
geoms:
  - {type: box, pos: [0, 0, -20], size: [1, 1, 0.1]}
bodies:
  - pos: [0, 0, -0.1]
    geoms: [{type: box, size: [2, 2, 0.1]}]
    motion: {type: oscillate, axis: [1, 0, 0], amplitude: 0, period: 4}
tasks:
  survive:
    spawn: {pos: [0, 0, 0.5]}
    region: {type: box, pos: [0, 0, 0.6], size: [2, 2, 0.5], attach: 0}
    time_limit_s: 10
""")


def level(edit=lambda d: None):
    d = copy.deepcopy(BASE)
    edit(d)
    return Level.from_dict(d)


def test_a_tilt_loads_with_its_defaults_filled_and_a_motion_without_one_is_unchanged():
    m = load(TILT, tilt=True)
    assert m["tilt"] == {"axis": [1.0, 0.0, 0.0], "amplitude": 10.0, "period": 8.0, "phase": 0.0}
    assert load({**TILT, "tilt": {"axis": [3, 0, 0], "amplitude": 10}}, tilt=True)["tilt"]["period"] == 4.0  # the motion's
    assert "tilt" not in load({k: v for k, v in TILT.items() if k != "tilt"}, tilt=True)


def test_a_loaded_tilt_loads_again_unchanged():
    # a replay reloads the canonical level, so normalising an axis twice must give the same bits
    m = load({**TILT, "axis": [1, 1, 0], "tilt": {"axis": [1, 1, 0], "amplitude": 8}}, tilt=True)
    assert load(m, tilt=True) == m


@pytest.mark.parametrize("tilt", [
    {"axis": [0, 0, 1], "amplitude": 5},  # not horizontal
    {"axis": [0, 0, 0], "amplitude": 5},
    {"axis": [1, 0, 0]},
    {"axis": [1, 0, 0], "amplitude": 5, "period": 0},
    {"axis": [1, 0, 0], "amplitude": 5, "spin": 1},
])
def test_a_malformed_tilt_is_rejected(tilt):
    with pytest.raises(ValueError):
        load({**TILT, "tilt": tilt}, tilt=True)


def test_only_a_body_motion_can_tilt():
    def edit(d):
        d["tasks"]["survive"]["region"] = {"type": "sphere", "pos": [0, 0, 0.5], "radius": 3, "motion": TILT}
    with pytest.raises(ValueError):
        level(edit)


def test_a_body_tilts_about_its_axis_by_a_sine_of_sim_time():
    def edit(d):
        d["bodies"][0]["motion"] = TILT
        d["bodies"][0]["pos"] = [0, 0, -3]  # out of the shell's way
        d["tasks"]["survive"]["region"] = {"type": "sphere", "pos": [0, 0, 0.5], "radius": 3}
    core = Core(level=level(edit))
    m, d = core.model, core.data
    b = m.body("body0").id
    core.step(np.zeros(2), n=1200)  # 2 s: a quarter of the tilt period, the full 10 degrees
    mujoco.mj_kinematics(m, d)
    np.testing.assert_allclose(d.xquat[b], [math.cos(math.radians(5)), math.sin(math.radians(5)), 0, 0], atol=1e-6)


def test_an_attached_region_loads_in_its_bodys_frame():
    lv = level()
    assert lv.tasks["survive"]["region"]["attach"] == 0
    assert "attach" not in Level.load(LEVELS_DIR / "first_survive.yaml").tasks["survive"]["region"]


@pytest.mark.parametrize("edit", [
    lambda r: r.update(attach=1),  # no such body
    lambda r: r.update(attach=0.0),
    lambda r: r.update(attach=True),
    lambda r: r.update(motion={"type": "oscillate", "axis": [1, 0, 0], "amplitude": 1, "period": 4}),
])
def test_a_malformed_attach_is_rejected(edit):
    with pytest.raises(ValueError):
        level(lambda d: edit(d["tasks"]["survive"]["region"]))


def test_the_spawn_check_uses_the_attached_bodys_pose_at_time_zero():
    def moved(d):  # the body starts 3 m along +x, so the region (its frame) leaves the origin spawn behind
        d["bodies"][0]["motion"] = {"type": "oscillate", "axis": [1, 0, 0], "amplitude": 3, "period": 4, "phase": 0.25}
    with pytest.raises(ValueError, match="outside its region"):
        level(moved)

    def spawn_on_it(d):
        moved(d)
        d["tasks"]["survive"]["spawn"]["pos"] = [3, 0, 0.5]
    level(spawn_on_it)


def env(lv, **kw):
    return gym.make("Hamsterball-v1", tasks=["survive"], levels=[lv], **kw)


def test_a_spawned_hamsterball_rests_on_the_platform_at_reset():
    e = env(level())
    e.reset(seed=0)
    for _ in range(60):
        _, _, terminated, _, info = e.step(np.zeros(2, np.float32))
        assert not terminated
    assert abs(info["state"][2] - 0.5) < 0.005


def test_the_region_test_and_goal_vector_follow_the_attached_body():
    def far(d):  # the platform slides 8 m along +x in 2 s and back; the shell, on a floor beside it, stays put
        d["geoms"] = [{"type": "box", "pos": [0, 0, -0.5], "size": [20, 20, 0.5]}]
        d["bodies"][0] = {"pos": [0, 0, -3], "geoms": [{"type": "box", "size": [2, 2, 0.1]}],
                          "motion": {"type": "path", "points": [[0, 0, 0], [8, 0, 0]], "period": 4}}
        d["tasks"]["survive"]["region"] = {"type": "box", "pos": [0, 0, 3.5], "size": [2, 2, 0.5], "attach": 0}
    e = env(level(far))
    obs, _ = e.reset(seed=0)
    np.testing.assert_allclose(obs["goal"][:3], [0, 0, 0], atol=1e-3)  # the region centre (0, 0, 0.5) at t = 0
    for i in range(1, 61):
        obs, _, terminated, _, info = e.step(np.zeros(2, np.float32))
        if i == 15:  # 0.25 s: the body is 1 m along +x (facing 0, so camera x is world x)
            np.testing.assert_allclose(obs["goal"][:3], [1, 0, 0], atol=1e-3)
        if terminated:
            break
    assert info["failure_reason"] == "left_region" and 29 <= i <= 31  # the region's -x face passes x = 0 at 0.5 s


def test_an_attached_region_turns_with_its_bodys_tilt():
    region = level().tasks["survive"]["region"]  # a box at (0, 0, 0.6) in the body frame, half-height 0.5
    s = math.sqrt(0.5)
    frame = (np.zeros(3), np.array([s, s, 0, 0]))  # 90 degrees about x: body +z is world -y
    assert crosses(region, [0, -0.6, 0], [0, -0.6, 0], frame=frame)
    assert not crosses(region, [0, 0, 0.6], [0, 0, 0.6], frame=frame)
    np.testing.assert_allclose(centre(region, frame=frame), [0, -0.6, 0], atol=1e-12)


def test_the_drawn_region_rides_its_body():
    def sliding(d):
        d["bodies"][0]["motion"] = {"type": "path", "points": [[0, 0, 0], [4, 0, 0]], "period": 8}
    e = env(level(sliding))
    e.reset(seed=0)
    for _ in range(60):
        e.step(np.zeros(2, np.float32))
    m, d = e.unwrapped.core.model, e.unwrapped.core.data
    mujoco.mj_kinematics(m, d)
    np.testing.assert_allclose(d.geom_xpos[m.geom("survive/region").id], [1, 0, 0.5], atol=1e-3)  # 1 s at 1 m/s


def test_a_platform_at_a_steady_speed_carries_an_inputless_hamsterball_and_its_region():
    def steady(d):  # 0.25 m/s along +x for the whole 8 s episode
        d["bodies"][0]["motion"] = {"type": "path", "points": [[0, 0, 0], [8, 0, 0]], "period": 64}
        d["tasks"]["survive"]["time_limit_s"] = 8
    e = env(level(steady))
    e.reset(seed=0)
    core, rel = e.unwrapped.core, []
    qa = core.bodies[0][0]
    for i in range(1, 481):
        _, _, terminated, truncated, info = e.step(np.zeros(2, np.float32))
        if i % 120 == 0:
            rel.append(core.data.qpos[0] - core.data.qpos[qa])
    assert info["success"] and core.data.qpos[qa] == pytest.approx(2.0, abs=1e-3)
    # measured: the ball rolls back at the instant start, then keeps pace; offset from the platform centre at 2, 4,
    # 6, 8 s: -0.093, -0.090, -0.085, -0.076 m (step.md)
    assert all(-0.2 < r < 0 for r in rel) and max(rel) - min(rel) < 0.03


def raft_env(**kw):
    return env("survive_raft", **kw)  # by name: a Collector records the env's kwargs


def test_an_inputless_hamsterball_rolls_off_survive_raft():
    info = run(raft_env(), lambda obs: np.zeros(2, np.float32))
    assert info["failure_reason"] == "left_region"  # measured: at env step 177, 2.95 s into the 30 s


def test_steering_toward_the_region_centre_rides_survive_raft_out():
    assert run(raft_env(), track)["success"]


def test_setting_a_stored_core_state_on_a_tilting_platform_reproduces_the_run_bit_for_bit():
    lv = Level.load(LEVELS_DIR / "survive_raft.yaml")
    core = Core(level=lv, camera_action=True)
    states = drive(core, 0, 240)
    replay = Core(level=lv, camera_action=True)
    replay.set_state(states[119])
    assert np.array_equal(drive(replay, 120, 240)[-1], states[-1])
