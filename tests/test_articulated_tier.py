"""The articulated tier in the core: the fragment contract, standing, containment, and the core-state round trip."""

import math

import mujoco
import numpy as np
import pytest

from hamsterball.core import Core
from hamsterball.core.articulated_tier import ArticulatedTier
from hamsterball.core.model import HAM, HAM_DOF, HAM_QPOS, SHELL_DOF
from hamsterball.core.step import FIELD_HARD, FIELD_START, contain

ZERO = np.zeros(8, np.float32)
# the types MJX supports (mujoco-mjx 3.16 types.py), the limit on geoms that touch tiles included
JOINTS = {mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE}
GEOMS = {mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE}  # sphere/capsule-box exist in MJX; cylinder/ellipsoid-box don't
ACTUATORS = {mujoco.mjtTrn.mjTRN_JOINT}


def com_offset(core):
    d = core.data
    return d.subtree_com[HAM] - d.xpos[1]


def hamster_geoms(m):
    return [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == HAM]


def test_fragment_honours_the_tier_contract():
    m = Core(ArticulatedTier()).model
    assert m.body("hamster").id == HAM and m.body("head").id > HAM
    assert m.site_bodyid[m.site("hamster_eye").id] == m.body("head").id
    assert m.body_subtreemass[HAM] == pytest.approx(2.4, abs=1e-9)
    geoms = hamster_geoms(m)
    assert len(geoms) > 8 and (m.geom_group[geoms] == 2).all()
    solid = [g for g in geoms if m.geom_contype[g]]  # the rest are massless markers (nose, eyes)
    assert len(solid) > 8 and (m.geom_contype[solid] == 2).all()  # they hit tiles (tile conaffinity 2)...
    assert not (m.geom_contype[geoms] & m.geom_conaffinity[geoms]).any()  # ...and never each other
    assert {mujoco.mjtGeom(t) for t in m.geom_type[geoms]} <= GEOMS
    hamster_joints = [j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == HAM]
    assert {mujoco.mjtJoint(m.jnt_type[j]) for j in hamster_joints} <= JOINTS and len(hamster_joints) == 9
    assert m.nu == 8 and {mujoco.mjtTrn(t) for t in m.actuator_trntype} <= ACTUATORS
    assert (m.actuator_ctrlrange == [-1, 1]).all() and (m.actuator_gear[:, 0] > 0).all()
    assert Core(ArticulatedTier()).action_space.shape == (8,)


@pytest.mark.parametrize("facing", [0.0, 1.0, -2.5])
def test_reset_stands_with_its_feet_on_the_inner_wall_and_the_torso_along_the_facing(facing):
    core = Core(ArticulatedTier())
    core.reset(spawn=(3.0, -2.0, 0.5), facing=facing)
    m, d = core.model, core.data
    assert (d.qpos[HAM_QPOS + 7:HAM_QPOS + 15] == m.key_qpos[0][HAM_QPOS + 7:HAM_QPOS + 15]).all()
    assert core.tier.obs(m, d, facing)["wall_contact"]
    fwd = d.xmat[HAM].reshape(3, 3)[:, 0]  # the torso's forward axis
    assert fwd[:2] == pytest.approx([-math.sin(facing), math.cos(facing)], abs=1e-9)
    assert (d.xmat[HAM].reshape(3, 3)[2, 2]) == pytest.approx(1)  # upright
    feet = [d.geom(f"{n}_foot").xpos for n in ("fl", "fr", "rl", "rr")]
    assert all(abs(np.linalg.norm(f - d.xpos[1]) - 0.47 + 0.02) < 0.004 for f in feet)  # on the 0.47 m inner wall
    assert np.linalg.norm(com_offset(core)) < core.tier.resting_radius - 0.1  # standing, deep inside the radius
    # the hamster holds the pose for a moment on its own (no gait: only the contact), and does not explode
    core.step(ZERO, n=30)
    assert np.isfinite(d.qpos).all() and d.qpos[HAM_QPOS + 2] > 0.1


def test_a_stored_core_state_reproduces_the_run_bit_for_bit_ctrl_included():
    def run(core, a, b):
        out = []
        for i in range(a, b):
            core.step(np.sin(np.arange(8) + i / 7).astype(np.float32))
            out.append(core.state())
        return out

    core = Core(ArticulatedTier())
    states = run(core, 0, 40)
    assert core.data.ctrl.any()
    other = Core(ArticulatedTier())
    other.set_state(states[19])
    assert np.array_equal(run(other, 20, 40)[-1], states[-1])


@pytest.mark.parametrize("speed", [0, 4, 8, 12])
def test_thrown_hamster_sinks_no_further_than_the_hard_stop_plus_a_step(speed):
    worst = 0.0
    for direction in [(0, 0, -1), (0, 0, 1), (1, 0, 0)]:
        core = Core(ArticulatedTier())
        d, m = core.data, core.model
        d.qpos[HAM_QPOS:HAM_QPOS + 3] = d.qpos[0:3]  # from the shell centre
        d.qvel[HAM_DOF:HAM_DOF + 3] = np.array(direction) * speed
        mujoco.mj_forward(m, d)
        for _ in range(600):
            core.step(ZERO, n=1)
            assert np.isfinite(d.qpos).all()
            worst = max(worst, np.linalg.norm(com_offset(core)) - core.tier.resting_radius)
    print(speed, round(worst * 1000, 1), "mm")
    assert worst < FIELD_HARD + 0.006  # abstract tier: 23 mm at 12 m/s


def test_a_fallen_hamster_lying_on_the_wall_gets_no_containment_impulse():
    core = Core(ArticulatedTier())
    m, d = core.model, core.data
    rng = np.random.default_rng(1)
    for _ in range(4):
        core.reset()
        q = rng.normal(size=4)
        d.qpos[HAM_QPOS + 3:HAM_QPOS + 7] = q / np.linalg.norm(q)  # any way up
        mujoco.mj_forward(m, d)
        for _ in range(300):  # limp: it collapses and settles
            core.step(ZERO, n=1)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            before = d.qvel.copy()
            assert np.linalg.norm(com_offset(core)) < core.tier.resting_radius + FIELD_START
            contain(m, d, core.tier.resting_radius)
            assert (d.qvel == before).all()


def test_the_playtest_posture_loop_holds_the_hamster_standing_and_a_shove_topples_it_onto_the_wall():
    from hamsterball.game.articulated_playtest import shove, torques

    core = Core(ArticulatedTier())
    d = core.data
    for _ in range(180):
        core.step(torques(d, d.time))
    assert d.xmat[HAM].reshape(3, 3)[2, 2] > 0.95 and np.linalg.norm(com_offset(core)) < 0.3
    shove(core)
    for _ in range(240):
        core.step(torques(d, d.time, limp=True))
    assert d.xmat[HAM].reshape(3, 3)[2, 2] < 0.5 and np.linalg.norm(com_offset(core)) > 0.32  # down on the wall


def test_the_articulated_model_has_no_tail_or_ears_and_the_abstract_one_keeps_them():
    from hamsterball.core import views
    from hamsterball.core.abstract_tier import AbstractTier

    m = Core(ArticulatedTier()).model
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "features") < 0
    assert mujoco.mj_name2id(Core(AbstractTier()).model, mujoco.mjtObj.mjOBJ_BODY, "features") >= 0
    core = Core(ArticulatedTier())
    views.pose(core.model, core.data, 0.0)  # poses the views without a features body


def test_the_ground_floors_and_sky_render_a_checker_and_a_gradient():
    import yaml
    from hamsterball.core.level import Level
    from hamsterball.env import LEVELS_DIR

    core = Core(ArticulatedTier())
    m = core.model
    assert m.geom_matid[0] >= 0  # the flat ground
    assert {mujoco.mjtTexture(t) for t in m.tex_type} >= {mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtTexture.mjTEXTURE_2D}
    for name in ("first_goal", "hill"):  # a box floor and an hfield
        level = Level.from_dict(yaml.safe_load((LEVELS_DIR / f"{name}.yaml").read_text()), LEVELS_DIR)
        lm = Core(level=level).model
        floors = [g for g in range(lm.ngeom) if lm.geom_group[g] == 0 and lm.geom_contype[g] == 1 and lm.geom_matid[g] >= 0]
        assert floors, name
