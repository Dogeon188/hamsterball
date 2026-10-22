"""Kinematic bodies (survive-movers ticket 03): solid geoms that move by a motion and shove the hamsterball."""

import copy

import gymnasium as gym
import mujoco
import numpy as np
import pytest
import yaml

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core import Core
from hamsterball.core.level import Level
from hamsterball.core.model import HAM, SHELL
from test_core_state import drive

BASE = yaml.safe_load("""
version: 1
kill_plane_z: -5
geoms:
  - {type: box, pos: [0, 0, -0.5], size: [10, 10, 0.5]}
tasks:
  survive:
    spawn: {pos: [0, 0, 0.5]}
    region: {type: sphere, pos: [0, 0, 0.5], radius: 3}
    time_limit_s: 10
""")
BAR = {"pos": [0, 4, 0.5], "geoms": [{"type": "box", "size": [3, 0.25, 0.5]}],
       "motion": {"type": "oscillate", "axis": [0, 1, 0], "amplitude": 3, "period": 4}}


def level(*bodies):
    d = copy.deepcopy(BASE)
    if bodies:
        d["bodies"] = list(bodies)
    return Level.from_dict(d)


def test_a_body_loads_with_its_defaults_filled():
    b = level(BAR).bodies[0]
    assert b["pos"] == [0.0, 4.0, 0.5]
    assert b["geoms"] == [{"type": "box", "pos": [0.0, 0.0, 0.0], "size": [3.0, 0.25, 0.5], "euler": [0.0, 0.0, 0.0]}]
    assert b["motion"]["type"] == "oscillate" and b["motion"]["phase"] == 0.0
    assert level({**BAR, "pos": [0, 4.0, 0.5]}).id == level(BAR).id


def test_a_level_without_bodies_keeps_its_id_and_mjcf():
    lv = level()
    assert lv.bodies == []
    assert "bodies" not in lv.canonical()
    assert lv.body_mjcf() == ""


@pytest.mark.parametrize("edit", [
    lambda b: b.pop("motion"),
    lambda b: b.pop("geoms"),
    lambda b: b.update(geoms=[]),
    lambda b: b.update(geoms=[{"type": "box", "size": [1, 1]}]),
    lambda b: b.update(geoms=[{"type": "plane", "size": [1, 1, 1]}]),
    lambda b: b.update(spin=3),
    lambda b: b["motion"].update(period=0),
])
def test_a_malformed_body_is_rejected(edit):
    b = copy.deepcopy(BAR)
    edit(b)
    with pytest.raises(ValueError):
        level(b)


def test_bodies_compile_after_the_view_mocaps_and_never_touch_the_world():
    core = Core(level=level(BAR, {**BAR, "pos": [0, -4, 0.5]}))
    m, d = core.model, core.data
    assert m.body("shell").id == SHELL and m.body("hamster").id == HAM
    assert m.body("body0").id > m.body("view_first_person").id
    assert m.body("body1").id == m.body("body0").id + 1
    core.step(np.zeros(2), n=60)  # the bar sits on the floor and the pair overlap nothing but the floor
    pairs = {tuple(sorted((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]))) for c in d.contact[:d.ncon]}
    assert all(SHELL in p for p in pairs)


def test_a_body_follows_its_motion_from_sim_time_alone():
    core = Core(level=level(BAR))
    m, d = core.model, core.data
    b = m.body("body0").id
    np.testing.assert_allclose(d.xpos[b], [0, 4, 0.5])  # posed at reset
    core.step(np.zeros(2), n=600)  # 1 s: a quarter period, the far end of the swing
    mujoco.mj_kinematics(m, d)
    np.testing.assert_allclose(d.xpos[b], [0, 7, 0.5], atol=1e-6)


def test_setting_a_stored_core_state_on_a_level_with_bodies_reproduces_the_run_bit_for_bit():
    lv = level(BAR, {**BAR, "pos": [0, -4, 0.5],
                     "motion": {"type": "path", "points": [[0, 0, 0], [0, 6, 0]], "period": 3, "spin": 40}})
    core = Core(level=lv, camera_action=True)
    states = drive(core, 0, 240)  # the bars sweep through the shell's spot
    assert not np.array_equal(states[0][:3], states[-1][:3])
    replay = Core(level=lv, camera_action=True)
    replay.set_state(states[119])
    assert np.array_equal(drive(replay, 120, 240)[-1], states[-1])


def test_a_sweeping_bar_shoves_a_resting_hamsterball_off_its_spot():
    # the bar's near face sweeps from y = 3.75 to y = -1.25, through the shell at the origin, and back
    core = Core(level=level({**BAR, "motion": {"type": "oscillate", "axis": [0, -1, 0], "amplitude": 5, "period": 8}}))
    core.step(np.zeros(2), n=3000)
    assert core.data.qpos[1] < -4.0  # measured -4.35 m after 5 s (step.md)


def test_a_platform_moving_under_the_shell_carries_it_by_friction():
    d = copy.deepcopy(BASE)
    d["geoms"] = [{"type": "box", "pos": [0, 0, -20], "size": [1, 1, 0.1]}]  # no floor: only the platform holds it up
    d["tasks"]["survive"]["region"]["radius"] = 30
    d["bodies"] = [{"pos": [0, 0, -0.1], "geoms": [{"type": "box", "size": [2, 2, 0.1]}],
                    "motion": {"type": "path", "points": [[0, 0, 0], [4, 0, 0]], "period": 16}}]  # 0.5 m/s along +x
    core = Core(level=Level.from_dict(d))
    core.step(np.zeros(2), n=2400)  # 4 s: the platform is at x = 2
    x, z = core.data.qpos[0], core.data.qpos[2]
    assert 1.6 < x < 2.0 and abs(z - 0.5) < 0.005  # measured x 1.772, z 0.499 (step.md); a mocap body is not carried


def test_an_inputless_hamsterball_on_survive_pushers_is_shoved_out():
    env = gym.make("HamsterballSurvive-v0", levels=["survive_pushers"])
    env.reset(seed=0)
    for _ in range(1200):
        _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
        if terminated or truncated:
            break
    assert info["failure_reason"] in ("left_region", "kill_plane")
