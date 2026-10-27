"""Seesaws (hinge and ball) on the batched core and the Warp vector env (GPU effort): seam 1 against `Core` (the CPU-layout
state, one physics step, per-world joint fields, the reset pose) and seam 2 against the CPU env (recorded episodes of
seesaw levels replayed through `step_to`, a tip-off and a held balance among them)."""

import copy
import os

import mujoco
import numpy as np
import pytest
import torch
import yaml

from hamsterball.core import Core
from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from test_warp_bodies import (FLOAT32, LOOSE, STILL, batch, bundled, chase, cpu_state, record, replay, still, venv_for)

pytestmark = pytest.mark.gpu

DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")
GEN = ProceduralGenerator()

SEESAW = LOOSE  # a seesaw joint after a physics step, as the shell and hamster dofs (measured below)
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
    spawn: {pos: [0.5, 0.3, 0.5]}
    region: {type: box, pos: [0, 0, 0.6], size: [3, 3, 0.5], attach: 0}
    time_limit_s: 12
""")
PILLAR = {"pos": [6, 0, -0.1], "geoms": [{"type": "box", "size": [1, 1, 0.1]}],
          "motion": {"type": "oscillate", "axis": [0, 1, 0], "amplitude": 1.0, "period": 4.0}}


def seesaw(joint, spawn=(0.5, 0.3, 0.5), before=(), after=(), geoms=None, attach=0, edit=lambda d: None):
    """The base seesaw level with this joint, spawn, and driven bodies before and after the seesaw (the region follows
    it at index `attach`)."""
    d = copy.deepcopy(BASE)
    if geoms:
        d["bodies"][0]["geoms"] = geoms
    d["bodies"][0]["joint"] = joint
    d["bodies"] = [copy.deepcopy(b) for b in before] + d["bodies"] + [copy.deepcopy(b) for b in after]
    d["tasks"]["survive"]["spawn"]["pos"] = list(spawn)
    d["tasks"]["survive"]["region"]["attach"] = len(before)
    edit(d)
    return Level.from_dict(d)


HINGE = seesaw({"type": "hinge", "axis": [0, 1, 0], "stiffness": 300, "damping": 30, "range": 12, "mass": 30})
HINGE_X = seesaw({"type": "hinge", "axis": [1, 0, 0], "stiffness": 150, "damping": 10, "range": 20, "mass": 12}, spawn=(0.3, 0.8, 0.5))
BALL = bundled("survive_seesaw")
BALL2 = seesaw({"type": "ball", "stiffness": 120, "damping": 15, "range": 25, "mass": 45}, spawn=(-0.7, 0.2, 0.5),
               geoms=[{"type": "box", "size": [2, 2, 0.1]}, {"type": "sphere", "size": [0.4], "pos": [1, 0, 0.3]}])
SEESAW_FIRST = seesaw({"type": "ball", "stiffness": 250, "damping": 40, "range": 10, "mass": 20}, after=[PILLAR])
SEESAW_LAST = seesaw({"type": "hinge", "axis": [0, 1, 0], "stiffness": 250, "damping": 40, "range": 10, "mass": 20}, before=[PILLAR])
FLOOR = Level.from_dict({
    "kill_plane_z": -5, "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}],
    "tasks": {"survive": {"spawn": {"pos": [0, 0, 0.5]}, "region": {"type": "sphere", "pos": [0, 0, 0.5], "radius": 5},
                          "time_limit_s": 10}}})
RAFT = bundled("survive_raft")
GEN_SEESAW = GEN("survive", 5, 0.5)
assert "joint" in GEN_SEESAW.bodies[0]
BANK = [HINGE, HINGE_X, BALL, BALL2, SEESAW_FIRST, SEESAW_LAST, RAFT, FLOOR]


def body_slices(core):
    """Where each level body's qpos and qvel sit in the CPU core-state vector: (qpos slice, qvel slice, driven)."""
    m, out = core.model, []
    for b in core.bodies:
        j = int(np.flatnonzero(m.jnt_qposadr == b.qpos)[0])
        nq, nv = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3)}.get(m.jnt_type[j], (1, 1))
        out.append((slice(1 + b.qpos, 1 + b.qpos + nq), slice(1 + m.nq + b.dof, 1 + m.nq + b.dof + nv), b.motion is not None))
    return out


def test_every_level_of_a_bank_with_seesaws_round_trips_its_cpu_state():
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        s = cpu_state(lv, 60).state().astype(np.float32).astype(np.float64)
        b.reset([w], bank_index=w)
        b.set_state(w, s)
        assert np.array_equal(b.state(w), s), w
    states = [cpu_state(lv, 5).state().astype(np.float32) for lv in BANK]
    assert len({len(s) for s in states}) > 1
    b.set_states(states)
    for w, s in enumerate(states):
        assert np.array_equal(b.state(w), s), w
    assert all(np.array_equal(a, s) for a, s in zip(b.cpu_states(), states))


@pytest.mark.parametrize("env_steps", [0, 30, 150])
def test_a_step_matches_the_cpu_core_on_every_seesaw_level_of_a_bank(env_steps):
    """The seesaw's joint, the shell and the hamster after one physics step, from a state a few env steps into a tilt."""
    b = batch(BANK)
    cores = []
    for w, lv in enumerate(BANK):
        core = cpu_state(lv, env_steps)
        b.reset([w], bank_index=w)
        b.set_state(w, core.state())
        cores.append(core)
    b.step(np.zeros((len(BANK), 2), np.float32), 1)
    worst = 0.0
    for w, core in enumerate(cores):
        core.step(STILL, 1)
        want, got = core.state(), b.state(w)
        assert len(got) == len(want)
        nq = core.model.nq
        for qp, qv, driven in body_slices(core):
            tol = FLOAT32 if driven else SEESAW
            np.testing.assert_allclose(got[qp], want[qp], **tol, err_msg=f"world {w} body qpos")
            np.testing.assert_allclose(got[qv], want[qv], **tol, err_msg=f"world {w} body qvel")
            worst = max(worst, float(np.abs(got[qp] - want[qp]).max()), float(np.abs(got[qv] - want[qv]).max()))
        np.testing.assert_allclose(got[1:15], want[1:15], **LOOSE, err_msg=f"world {w} shell and hamster qpos")
        np.testing.assert_allclose(got[1 + nq:13 + nq], want[1 + nq:13 + nq], **LOOSE, err_msg=f"world {w} shell and hamster qvel")
    print(f"env_steps {env_steps}: worst seesaw or driven body difference {worst:.3g}")


def test_each_world_takes_its_own_levels_joint_fields():
    """The model's per-world fields at a reset are the CPU model's of the world's level, on the seesaw's slot."""
    b = batch([HINGE, BALL2, BALL, FLOOR])
    m = b.model
    for w, lv in enumerate([HINGE, BALL2, BALL, FLOOR]):
        b.reset([w], bank_index=w)
    for w, lv in enumerate([HINGE, BALL2, BALL]):
        want = Core(level=lv).model
        kind = lv.bodies[0]["joint"]["type"]
        name = "hinge0" if kind == "hinge" else "ball0"
        slot = b.cpu_model.body(name)
        j, bid = int(slot.jntadr[0]), slot.id
        da = int(b.cpu_model.jnt_dofadr[j])
        ref = want.body("body0")
        rj, rd = int(ref.jntadr[0]), int(want.jnt_dofadr[int(ref.jntadr[0])])
        nd = 1 if kind == "hinge" else 3
        np.testing.assert_allclose(m.body_mass.numpy()[w, bid], want.body_mass[ref.id], rtol=1e-6)
        np.testing.assert_allclose(m.body_inertia.numpy()[w, bid], want.body_inertia[ref.id], rtol=1e-5)
        np.testing.assert_allclose(m.body_ipos.numpy()[w, bid], want.body_ipos[ref.id], atol=1e-6)
        np.testing.assert_allclose(m.body_pos.numpy()[w, bid], want.body_pos[ref.id], atol=1e-6)
        np.testing.assert_allclose(m.body_subtreemass.numpy()[w, bid], want.body_subtreemass[ref.id], rtol=1e-6)
        np.testing.assert_allclose(m.body_invweight0.numpy()[w, bid], want.body_invweight0[ref.id], rtol=1e-4)
        np.testing.assert_allclose(m.dof_invweight0.numpy()[w, da:da + nd], want.dof_invweight0[rd:rd + nd], rtol=1e-4)
        np.testing.assert_allclose(m.dof_damping.numpy()[w, da:da + nd], want.dof_damping[rd:rd + nd], rtol=1e-6)
        np.testing.assert_allclose(m.jnt_stiffness.numpy()[w, j], want.jnt_stiffness[rj], rtol=1e-6)
        np.testing.assert_allclose(m.jnt_range.numpy()[w, j], want.jnt_range[rj], rtol=1e-6)
        if kind == "hinge":
            np.testing.assert_allclose(m.jnt_axis.numpy()[w, j], want.jnt_axis[rj], atol=1e-6)
    # a world on a level without the slot parks it: a wide range, so the limit never acts
    j = int(b.cpu_model.body("ball0").jntadr[0])
    assert m.jnt_range.numpy()[3, j][1] > np.radians(90)
    assert int(mujoco.mjtJoint.mjJNT_BALL) == b.cpu_model.jnt_type[j]


def test_a_seesaw_level_steps_as_it_does_in_a_bank_of_its_own():
    """Padding changes nothing: a world in the mixed bank (other levels' joint values in the same arrays) equals the same
    world in a bank of its level alone."""
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        s = cpu_state(lv, 60).state()
        alone = batch([lv])
        for core, i in ((b, w), (alone, 0)):
            core.reset([i], bank_index=i)
            core.set_state(i, s)
        for core in (b, alone):
            core.step(np.zeros((core.num_envs, 2), np.float32), 3)
        np.testing.assert_allclose(b.state(w), alone.state(0), rtol=0, atol=1e-5, err_msg=f"world {w}")


def test_a_reset_leaves_the_seesaw_level_at_the_cpu_cores_reset():
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        spawn = lv.tasks["survive"]["spawn"]["pos"]
        b.reset([w], bank_index=w, spawn=spawn)
        core = Core(level=lv)
        core.reset(spawn)
        np.testing.assert_allclose(b.state(w)[1:], core.state()[1:], atol=1e-6, err_msg=f"world {w}")


# ---- seam 2: the Warp vector env against the CPU env on recorded episodes -------------------------------------------

SCENARIOS = {
    "ball_seesaw_tips_the_still_ball_off": (BALL, still, "left_region"),
    "ball_seesaw_balanced_chasing_the_region": (BALL, chase, "success"),
    "hinge_seesaw_tips_the_still_ball_off": (HINGE, still, "left_region"),
    "hinge_seesaw_balanced_chasing_the_region": (HINGE, chase, "success"),
    "hinge_seesaw_on_the_x_axis_tips_the_still_ball_off": (HINGE_X, still, "left_region"),
    "generated_seesaw_tips_off": (GEN_SEESAW, still, "left_region"),
    "generated_seesaw_balanced": (GEN_SEESAW, chase, "success"),
    "seesaw_after_a_driven_body_balanced": (SEESAW_LAST, chase, "success"),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_step_to_matches_the_cpu_env_on_recorded_episodes_on_seesaw_levels(name):
    from test_warp_env import OBS

    lv, policy, want = SCENARIOS[name]
    ep = record(lv, policy, {"time_limit_s": 12})
    assert ep["outcome"] == want
    venv = venv_for([lv], 1, time_limit_s=12)
    obs, _ = venv.reset(seed=0)
    for k in ("state", "goal"):
        np.testing.assert_allclose(obs[k][0].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
    worst = replay(venv, [ep], [name])
    print(f"{name}: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}, region pose {worst[2]:.3g}")


def test_episodes_on_different_seesaw_levels_of_one_bank_replay_together():
    """A ball and a hinge seesaw, a seesaw beside a driven body in either order, and a level with neither, forced onto
    one padded bank."""
    lvs = [BALL, HINGE, SEESAW_FIRST, SEESAW_LAST, BALL2, FLOOR]
    pols = [chase, still, chase, still, still, still]
    eps = [record(lv, p, {"time_limit_s": 12}) for lv, p in zip(lvs, pols)]
    venv = venv_for(lvs, len(lvs), time_limit_s=12)
    venv.reset(seed=0, options={"level_index": torch.arange(len(lvs))})
    worst = replay(venv, eps, [f"level{i}" for i in range(len(lvs))])
    print(f"mixed bank: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}, region pose {worst[2]:.3g}")


class SeesawsAndOthers:
    """The procedural generator, with a seesaw layout for every even seed."""

    def __init__(self):
        self.tasks, self.version = GEN.tasks, GEN.version

    def __call__(self, task, seed, difficulty):
        want = seed % 2 == 0
        while want and not any("joint" in b for b in GEN(task, seed, difficulty).bodies):
            seed += 1
        return GEN(task, seed, difficulty)


def test_a_generator_pool_with_the_seesaw_layout_trains_on_the_device():
    venv = venv_for(SeesawsAndOthers(), 8, bank_size=8, difficulty=0.5)
    on_seesaw = [i for i, lv in enumerate(venv.levels) if any("joint" in b for b in lv.bodies)]
    assert on_seesaw and venv.core.bank.nball >= 1
    venv.reset(seed=3, options={"level_index": torch.tensor([on_seesaw[w % len(on_seesaw)] if w % 2 else -1 for w in range(8)])})
    for _ in range(300):
        obs, r, term, trunc, info = venv.step(torch.zeros(8, 2))
    assert torch.isfinite(obs["state"]).all() and torch.isfinite(obs["goal"]).all()
