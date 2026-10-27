"""The batched core on a mixed bank (GPU effort, seam 1): worlds on different levels, plane and hfield, one padded model."""

import os

import numpy as np
import pytest

from hamsterball.core import Core
from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from hamsterball.env import LEVELS_DIR

pytestmark = pytest.mark.gpu

DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")  # "cuda:0" runs the same tests on a GPU
LOOSE = dict(rtol=1e-4, atol=1e-5)  # shell and hamster dofs after a physics step; see batch_core.md
FORWARD = np.array([0.0, 1.0], np.float32)


def level(geoms, tasks=None):
    return Level.from_dict({
        "kill_plane_z": -5, "geoms": geoms,
        "tasks": tasks or {"reach_goal": {"spawn": {"pos": [0, 0, 0.5]}, "goal": {"type": "sphere", "pos": [0, 8, 0.5], "radius": 1},
                                          "time_limit_s": 30}}})


PLANE = level([{"type": "plane", "size": [0, 0, 0.1]}])
FLOOR = [{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}]
# a rotated box and a sphere and capsule on the way forward: the shell touches them within a second
BUMPS = level(FLOOR + [{"type": "box", "pos": [0.2, 1.3, 0.2], "size": [0.4, 0.2, 0.3], "euler": [0, 0, 30]},
                       {"type": "sphere", "pos": [-3, 3, 0], "size": [1]},
                       {"type": "capsule", "pos": [3, 2, 0.1], "size": [0.3, 1.5], "euler": [0, 90, 0]}])
HILL = Level.load(LEVELS_DIR / "hill.yaml")
ROUGH = ProceduralGenerator()("reach_goal", 1, 0.5)  # a generated hfield level: noise and slopes under the spawn
BANK = [PLANE, BUMPS, HILL, level(FLOOR), ROUGH]


def batch(levels, **kw):
    from hamsterball.gpu.batch_core import BatchCore

    return BatchCore(levels=levels, num_envs=len(levels), device=DEVICE, **kw)


def cpu_state(lv, env_steps):
    """A Core on the level, run forward for env steps from its spawn: its state and the core."""
    core = Core(level=lv)
    core.reset(lv.tasks["reach_goal"]["spawn"]["pos"])
    for _ in range(env_steps):
        core.step(FORWARD)
    return core.state(), core


def test_every_level_of_a_mixed_bank_round_trips_its_cpu_state():
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        s = cpu_state(lv, 20)[0].astype(np.float32).astype(np.float64)
        b.reset([w], bank_index=w)
        b.set_state(w, s)
        assert np.array_equal(b.state(w), s), lv.id[:8]


QPOS, QVEL = slice(1, 15), slice(15, 27)
HFIELD_QVEL = dict(rtol=1e-4, atol=2e-2)  # a rough hfield's contacts: Warp's solver leaves up to 7e-3 here; see batch_core.md


@pytest.mark.parametrize("env_steps", [0, 40, 70])
def test_one_step_on_worlds_of_different_levels_matches_the_cpu_core(env_steps):
    """Each world is on its own level (a plane, solids, an hfield, a floor): the unused slots are parked, so a world's
    ball touches only its own level."""
    b = batch(BANK)
    cpu = []
    for w, lv in enumerate(BANK):
        s, core = cpu_state(lv, env_steps)
        b.reset([w], bank_index=w)
        b.set_state(w, s)
        cpu.append(core)
    b.step(np.tile(FORWARD, (len(BANK), 1)), 1)
    for w, core in enumerate(cpu):
        core.step(FORWARD, 1)
        want, got = core.state(), b.state(w)
        np.testing.assert_allclose(got[QPOS], want[QPOS], **LOOSE, err_msg=f"world {w} qpos")
        np.testing.assert_allclose(got[QVEL], want[QVEL], **(HFIELD_QVEL if BANK[w] is ROUGH else LOOSE), err_msg=f"world {w} qvel")


@pytest.mark.parametrize("env_steps", [0, 40, 70])
def test_padding_does_not_change_a_worlds_step(env_steps):
    """A world in the mixed bank steps as it does in a bank of its own level alone, whatever the other levels add."""
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        s = cpu_state(lv, env_steps)[0]
        alone = batch([lv])
        for core, i in ((b, w), (alone, 0)):
            core.reset([i], bank_index=i)
            core.set_state(i, s)
        for core in (b, alone):
            core.step(np.tile(FORWARD, (core.num_envs, 1)), 1)
        np.testing.assert_allclose(b.state(w), alone.state(0), rtol=0, atol=1e-6, err_msg=f"world {w}")


def test_a_geom_with_contact_values_raises_until_it_is_batched():
    slippery = level(FLOOR)
    slippery.geoms = [{**FLOOR[0], "friction": [0.1, 0.01], "euler": [0, 0, 0]}]
    with pytest.raises(NotImplementedError, match="friction"):
        batch([PLANE, slippery])
