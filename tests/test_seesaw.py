"""Seesaw platforms (survive-movers ticket 05): a passive body on a hinge or ball joint that tips under the shell."""

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
from hamsterball.core.model import HAM, HAM_DOF, HAM_QPOS, SHELL, SHELL_DOF, SHELL_QPOS
from test_core_state import drive
from test_motion import run, track

BASE = yaml.safe_load("""
version: 1
kill_plane_z: -5
geoms: []
bodies:
  - pos: [0, 0, -0.1]
    geoms: [{type: box, size: [3, 3, 0.1]}]
    joint: {type: hinge, axis: [0, 1, 0], stiffness: 200, damping: 50, range: 15}
tasks:
  survive:
    spawn: {pos: [0, 0, 0.5]}
    region: {type: box, pos: [0, 0, 0.6], size: [3, 3, 0.5], attach: 0}
    time_limit_s: 10
""")


def level(edit=lambda d: None):
    d = copy.deepcopy(BASE)
    edit(d)
    return Level.from_dict(d)


def test_a_seesaw_loads_with_its_defaults_filled():
    joint = level().bodies[0]["joint"]
    assert joint == {"type": "hinge", "axis": [0.0, 1.0, 0.0], "stiffness": 200.0, "damping": 50.0, "range": 15.0,
                     "mass": 20.0}
    ball = level(lambda d: d["bodies"][0].update(joint={"type": "ball", "stiffness": 200, "range": 10}))
    assert ball.bodies[0]["joint"] == {"type": "ball", "stiffness": 200.0, "damping": 0.0, "range": 10.0, "mass": 20.0}


@pytest.mark.parametrize("edit", [
    lambda b: b.update(motion={"type": "oscillate", "axis": [1, 0, 0], "amplitude": 1, "period": 4}),  # both
    lambda b: b.pop("joint"),  # neither
    lambda b: b["joint"].update(type="slide"),
    lambda b: b["joint"].update(axis=[0, 0, 1]),  # a hinge tilts: its axis is horizontal
    lambda b: b["joint"].pop("axis"),
    lambda b: b["joint"].update(range=0),
    lambda b: b["joint"].update(range=90),
    lambda b: b["joint"].update(stiffness=-1),
    lambda b: b["joint"].update(mass=0),
    lambda b: b.update(joint={"type": "ball", "axis": [0, 1, 0], "stiffness": 1, "range": 5}),  # a ball has no axis
])
def test_a_malformed_seesaw_is_rejected(edit):
    with pytest.raises(ValueError):
        level(lambda d: edit(d["bodies"][0]))


@pytest.mark.parametrize("kind", ["hinge", "ball"])
def test_a_seesaw_compiles_after_the_view_mocaps_so_the_fixed_ids_hold(kind):
    def edit(d):
        if kind == "ball":
            d["bodies"][0]["joint"] = {"type": "ball", "stiffness": 200, "damping": 50, "range": 15}
    m = Core(level=level(edit)).model
    assert (m.body("shell").id, m.body("hamster").id) == (SHELL, HAM)
    assert m.jnt_qposadr[m.body_jntadr[HAM]] == HAM_QPOS and m.jnt_dofadr[m.body_jntadr[HAM]] == HAM_DOF
    assert m.jnt_qposadr[m.body_jntadr[SHELL]] == SHELL_QPOS and m.jnt_dofadr[m.body_jntadr[SHELL]] == SHELL_DOF
    saw = m.body("body0").id
    assert saw > m.body("view_first_person").id and m.body_jntnum[saw] == 1
    assert m.body_jntadr[saw] == m.njnt - 1 and m.nq == 14 + (1 if kind == "hinge" else 4)


@pytest.mark.parametrize("kind", ["hinge", "ball"])
def test_setting_a_stored_core_state_on_a_seesaw_reproduces_the_run_bit_for_bit(kind):
    def edit(d):
        if kind == "ball":
            d["bodies"][0]["joint"] = {"type": "ball", "stiffness": 200, "damping": 50, "range": 15}
    lv = level(edit)
    core = Core(level=lv, camera_action=True)
    states = drive(core, 0, 240)
    assert np.ptp([s[1 + 14] for s in states]) > 1e-3  # the seesaw's joint moves (its qpos follows the shell's 14)
    replay = Core(level=lv, camera_action=True)
    replay.set_state(states[119])
    assert np.array_equal(drive(replay, 120, 240)[-1], states[-1])


def tilt_after(x, seconds=1.0, kind="hinge"):
    """The seesaw's tilt (deg, positive when it tips down toward +x) after an inputless shell rests at x for `seconds`."""
    def edit(d):
        d["tasks"]["survive"]["spawn"]["pos"] = [x, 0, 0.5]
        if kind == "ball":
            d["bodies"][0]["joint"] = {"type": "ball", "stiffness": 200, "damping": 50, "range": 15}
    core = Core(level=level(edit))
    core.reset(spawn=(x, 0.0, 0.5))
    core.step(np.zeros(2), n=round(600 * seconds))
    m, d = core.model, core.data
    mujoco.mj_kinematics(m, d)
    z = d.xmat[m.body("body0").id].reshape(3, 3)[:, 2]  # the platform's normal leans downhill
    return math.degrees(math.atan2(z[0], z[2]))


@pytest.mark.parametrize("kind", ["hinge", "ball"])
def test_a_hamsterball_off_the_pivot_tips_the_seesaw_toward_it_and_at_the_pivot_it_stays_level(kind):
    # measured (level.md): at the pivot 0.00 deg; 1 m off, 8.6 deg after 1 s (the static m g x / k is 9.0); 2 m off,
    # the 15 deg limit
    assert abs(tilt_after(0.0, kind=kind)) < 0.05
    assert 7 < abs(tilt_after(1.0, kind=kind)) < 10
    assert tilt_after(1.0) > 0 > tilt_after(-1.0)  # down on the shell's side
    assert 14.5 < abs(tilt_after(2.0, kind=kind)) < 15.5


@pytest.mark.parametrize("kind", ["hinge", "ball"])
def test_the_attached_region_tilts_with_the_seesaw(kind):
    def edit(d):
        d["tasks"]["survive"]["spawn"]["pos"] = [1, 0.5, 0.5]
        if kind == "ball":
            d["bodies"][0]["joint"] = {"type": "ball", "stiffness": 200, "damping": 50, "range": 15}
    e = gym.make("Hamsterball-v1", tasks=["survive"], levels=[level(edit)])
    e.reset(seed=0)
    for _ in range(60):
        obs, *_ = e.step(np.zeros(2, np.float32))
    m, d = e.unwrapped.core.model, e.unwrapped.core.data
    mujoco.mj_kinematics(m, d)
    drawn = d.geom_xpos[m.geom("survive/region").id]
    assert abs(drawn[0]) > 0.05  # the 0.7 m-high region centre swings off the pivot as the seesaw tips
    yaw = d.userdata[0]
    rel = drawn - d.xpos[SHELL]
    cam = [rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw), -rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw), rel[2]]
    np.testing.assert_allclose(obs["goal"][:3], cam, atol=1e-5)


def seesaw_env(**kw):
    return gym.make("Hamsterball-v1", tasks=["survive"], levels=["survive_seesaw"], **kw)  # by name, for a Collector


def test_an_inputless_hamsterball_tips_survive_seesaw_and_rolls_off():
    info = run(seesaw_env(), lambda obs: np.zeros(2, np.float32))
    assert info["failure_reason"] == "left_region"  # measured: at env step 183


def test_steering_toward_the_region_centre_balances_survive_seesaw():
    assert run(seesaw_env(), track)["success"]
