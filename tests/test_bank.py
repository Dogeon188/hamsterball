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
DIAGONAL = np.array([1.0, 0.3], np.float32)
# the raft's driven body moves by a float32 secant velocity and the ball rests on it under contacts: 60 physics steps leave
# 1.9e-4 against `Core`; the same world in a bank of its own differs from it by under 5.3e-5 (the padding test below), so
# this is float32, not the padding
RAFT = dict(rtol=1e-4, atol=1e-3)


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
LEVEL_NAMES = ["first_goal", "survive_collapse", "survive_seesaw", "survive_pushers", "survive_raft"]
MIXED = [Level.load(LEVELS_DIR / f"{n}.yaml") for n in LEVEL_NAMES]  # a collapse level, seesaws and driven bodies in one bank
COLLAPSE = MIXED[:2]  # a bank with tiles and no seesaw: the padded model's value was ~48,000x first_goal's here


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


def first_task_state(lv, env_steps):
    """A Core on the level's first task, run forward for env steps from its spawn: its state and the core."""
    core = Core(level=lv)
    core.reset(next(iter(lv.tasks.values()))["spawn"]["pos"])
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
# ponytail: Warp caps a hfield contact at 4 points per pair and the CPU core keeps every one it finds, so a rough hfield's qvel
# differs by up to 7e-3 (see batch_core.md); this bound absorbs it. Tighten it if mujoco-warp lifts the cap.
HFIELD_QVEL = dict(rtol=1e-4, atol=2e-2)


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


def test_the_solver_scale_of_every_world_is_its_levels_own():
    """Warp's Newton solver scales its tolerance by `stat.meaninertia * nv` with the padded nv; the product a world reads
    must be the CPU model's for its own level, with collapse tiles and a seesaw in the bank (the padded value was ~48,000x
    first_goal's)."""
    b = batch(MIXED)
    for w, lv in enumerate(MIXED):
        b.reset([w], bank_index=w)
        cpu = Core(level=lv).model
        got = float(b.model.stat.meaninertia.numpy()[w]) * b.cpu_model.nv
        assert got == pytest.approx(cpu.stat.meaninertia * cpu.nv, rel=1e-4), LEVEL_NAMES[w]


@pytest.mark.parametrize("bank, w", [(COLLAPSE, 0), (COLLAPSE, 1)] + [(MIXED, w) for w in range(len(MIXED))], ids=lambda p: f"{len(p)}" if isinstance(p, list) else str(p))
def test_sixty_physics_steps_in_a_bank_with_collapse_and_seesaw_levels_match_the_cpu_core(bank, w):
    """The padded solver scale bent a world's trajectory over many steps; one step (seam 1) missed it."""
    lv = bank[w]
    b = batch(bank)
    s, core = first_task_state(lv, 0)  # from the spawn: the ball settles under contacts the solver's tolerance decides
    b.reset([w], bank_index=w)
    b.set_state(w, s)
    b.step(np.tile(DIAGONAL, (b.num_envs, 1)), 60)
    core.step(DIAGONAL, 60)
    want, got = core.state(), b.state(w)
    drift = max(np.abs(got[QPOS] - want[QPOS]).max(), np.abs(got[QVEL] - want[QVEL]).max())
    print(f"{len(bank)} levels, {LEVEL_NAMES[w]}: drift over 60 physics steps {drift:.3g}")
    bound = RAFT if LEVEL_NAMES[w] == "survive_raft" else LOOSE
    np.testing.assert_allclose(got[QPOS], want[QPOS], **bound, err_msg=f"{LEVEL_NAMES[w]} qpos")
    np.testing.assert_allclose(got[QVEL], want[QVEL], **bound, err_msg=f"{LEVEL_NAMES[w]} qvel")


@pytest.mark.parametrize("w", range(len(MIXED)))
def test_padding_does_not_change_sixty_physics_steps(w):
    """A world in the mixed bank steps as it does in a bank of its own level alone, over many steps, whatever the other
    levels pad the model with (the solver scale included)."""
    lv, s = MIXED[w], first_task_state(MIXED[w], 0)[0]
    banks = [(batch(MIXED), w), (batch([lv]), 0)]
    for b, i in banks:
        b.reset([i], bank_index=i)
        b.set_state(i, s)
        b.step(np.tile(DIAGONAL, (b.num_envs, 1)), 60)
    got, alone = (b.state(i) for b, i in banks)
    print(f"{LEVEL_NAMES[w]}: mixed bank against its own over 60 physics steps {np.abs(got[1:27] - alone[1:27]).max():.3g}")
    np.testing.assert_allclose(got[QPOS], alone[QPOS], rtol=0, atol=1e-5, err_msg=f"{LEVEL_NAMES[w]} qpos")
    np.testing.assert_allclose(got[QVEL], alone[QVEL], rtol=0, atol=1e-5, err_msg=f"{LEVEL_NAMES[w]} qvel")


def contact_level(**contact):
    return level([{**FLOOR[0], **contact}])


# a slippery floor with a wide rolling term (a hard stop for the ball), a soft floor, and a floor with all three set
CONTACT_BANK = [PLANE, contact_level(friction=[0.1, 0.2]), contact_level(solref=[0.1, 0.8]),
                contact_level(friction=[0.3, 0.01], solref=[0.06, 1.5], solimp=[0.8, 0.9, 0.002, 0.4, 3]), level(FLOOR)]


@pytest.mark.parametrize("env_steps", [0, 40])
def test_geom_friction_solref_and_solimp_of_each_level_match_the_cpu_core(env_steps):
    """A world takes its own level's contact values and its neighbours' don't leak: the ball rolls (or settles) on each
    world's floor as on `Core`, over one env step and over three."""
    b = batch(CONTACT_BANK)
    cores = []
    for w, lv in enumerate(CONTACT_BANK):
        s, core = cpu_state(lv, env_steps)
        b.reset([w], bank_index=w)
        b.set_state(w, s)
        cores.append(core)
    for _ in range(3):
        b.step(np.tile(FORWARD, (len(CONTACT_BANK), 1)))
        for w, core in enumerate(cores):
            core.step(FORWARD)
            want, got = core.state(), b.state(w)
            np.testing.assert_allclose(got[QPOS], want[QPOS], **LOOSE, err_msg=f"world {w} qpos")
            np.testing.assert_allclose(got[QVEL], want[QVEL], **LOOSE, err_msg=f"world {w} qvel")
