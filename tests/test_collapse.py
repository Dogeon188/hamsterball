"""Collapsing tiles (survive-movers ticket 06): floor tiles that fall a fixed delay after the shell first touches them."""

import copy

import gymnasium as gym
import mujoco
import numpy as np
import pytest
import yaml

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core import Core
from hamsterball.core.level import TILE_WARN_RGBA, Level
from hamsterball.core.model import HAM, SHELL
from test_core_state import drive

BASE = yaml.safe_load("""
version: 1
kill_plane_z: -5
geoms:
  - {type: box, pos: [0, 0, -20], size: [1, 1, 0.1]}
tasks:
  survive:
    spawn: {pos: [0, 0, 0.5]}
    region: {type: sphere, pos: [0, 0, 0.5], radius: 30}
    time_limit_s: 10
""")
# two 2x2 m tiles side by side, the shell resting on the first; no floor under them
TILES = {"delay": 1, "tiles": [{"type": "box", "pos": [0, 0, -0.1], "size": [1, 1, 0.1]},
                               {"type": "box", "pos": [4, 0, -0.1], "size": [1, 1, 0.1]}]}


def level(collapse=None):
    d = copy.deepcopy(BASE)
    if collapse is not None:
        d["collapse"] = collapse
    return Level.from_dict(d)


def test_a_collapse_block_loads_with_its_defaults_filled():
    c = level(TILES).collapse
    assert c["delay"] == 1.0
    assert c["tiles"][0] == {"type": "box", "pos": [0.0, 0.0, -0.1], "size": [1.0, 1.0, 0.1], "euler": [0.0, 0.0, 0.0]}


def test_a_level_without_tiles_keeps_its_id():
    assert level().collapse is None
    assert "collapse" not in level().canonical()


@pytest.mark.parametrize("edit", [
    lambda c: c.pop("delay"),
    lambda c: c.update(delay=-1),
    lambda c: c.update(tiles=[]),
    lambda c: c.update(tiles=[{"type": "plane", "size": [1, 1, 1]}]),
    lambda c: c.update(fall=3),
])
def test_a_malformed_collapse_block_is_rejected(edit):
    c = copy.deepcopy(TILES)
    edit(c)
    with pytest.raises(ValueError):
        level(c)


def tile_z(core, i):
    m, d = core.model, core.data
    mujoco.mj_kinematics(m, d)
    return d.xpos[m.body(f"tile{i}").id][2]


def test_a_touched_tile_falls_after_its_delay_and_takes_the_hamsterball_with_it():
    core = Core(level=level(TILES))
    core.step(np.zeros(2), n=540)  # 0.9 s: the shell has rested on tile 0 since it settled
    assert abs(core.data.qpos[2] - 0.5) < 0.005 and tile_z(core, 0) == pytest.approx(-0.1, abs=1e-3)
    core.step(np.zeros(2), n=900)  # 2.4 s
    assert core.data.qpos[2] < -5 and tile_z(core, 0) < -5  # under the kill plane
    assert tile_z(core, 1) == pytest.approx(-0.1, abs=1e-6)  # never touched, never falls


def test_tiles_compile_after_the_bodies_and_touch_only_the_shell():
    core = Core(level=level(TILES))
    m, d = core.model, core.data
    assert m.body("shell").id == SHELL and m.body("hamster").id == HAM
    assert m.body("tile0").id > m.body("view_first_person").id
    core.step(np.zeros(2), n=60)
    pairs = {tuple(sorted((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]))) for c in d.contact[:d.ncon]}
    assert pairs and all(SHELL in p for p in pairs)


def test_setting_a_stored_core_state_on_a_collapse_level_reproduces_the_run_bit_for_bit():
    xs = range(-12, 13, 4)
    grid = {"delay": 1.5, "tiles": [{"type": "box", "pos": [x, y, -0.1], "size": [1.99, 1.99, 0.1]} for x in xs for y in xs]}
    core = Core(level=level(grid), camera_action=True)
    states = drive(core, 0, 180)  # tiles 0 and 1 along the run fall at 1.5 and 2.6 s, under a shell gone past them
    triggers = lambda s: int(np.isfinite(s[-49:]).sum())  # noqa: E731  the tile slots end the state
    assert 0 < triggers(states[99]) < triggers(states[-1])  # tiles trigger both before and after the restore
    assert states[-1][2] > 0  # the shell is still on the tiles
    replay = Core(level=level(grid), camera_action=True)
    replay.set_state(states[99])
    assert np.array_equal(drive(replay, 100, 180)[-1], states[-1])


def test_a_touched_tile_warns_by_colour_before_it_falls():
    lv = level(TILES)
    env = gym.make("HamsterballSurvive-v1", levels=[lv]).unwrapped
    env.reset(seed=0)
    m = env.core.model
    rgba = lambda i: m.geom_rgba[m.body_geomadr[m.body(f"tile{i}").id]].copy()  # noqa: E731
    still = rgba(1)
    for _ in range(30):  # 0.5 s of the 1 s delay
        env.step(np.zeros(2, np.float32))
    half = rgba(0)
    for _ in range(25):
        env.step(np.zeros(2, np.float32))
    assert np.allclose(rgba(1), still)
    assert not np.allclose(half, still) and np.abs(rgba(0) - TILE_WARN_RGBA).sum() < np.abs(half - TILE_WARN_RGBA).sum()
    env.reset(seed=0)  # a cached model: the look resets with the episode
    assert np.allclose(rgba(0), still)


def test_an_inputless_hamsterball_on_survive_collapse_falls_before_the_time_limit():
    env = gym.make("HamsterballSurvive-v1", levels=["survive_collapse"])
    env.reset(seed=0)
    for _ in range(10000):
        _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
        if terminated or truncated:
            break
    assert info["failure_reason"] in ("left_region", "kill_plane")
