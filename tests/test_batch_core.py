"""The batched core against the CPU core (GPU effort, seam 1): shared states, one physics step, the CPU layout."""

import math
import os

import mujoco
import numpy as np
import pytest

from hamsterball.core import Core
from hamsterball.core.abstract_tier import GRACE, LAST_TOUCH, REST
from hamsterball.core.model import FACING, PITCH, R
from hamsterball.core.step import FIELD_HARD, FIELD_START, PITCH_MAX, PITCH_MIN, TURN_STEP

pytestmark = pytest.mark.gpu

TIGHT = dict(rtol=1e-5, atol=1e-6)  # what the core-step kernels write: xfrc_applied and the userdata slots
LOOSE = dict(rtol=1e-4, atol=1e-5)  # shell and hamster dofs after a physics step; see batch_core.md for the measurement


DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")  # "cuda:0" runs the same tests on a GPU


def batch(**kw):
    from hamsterball.gpu.batch_core import BatchCore

    return BatchCore(num_envs=2, device=DEVICE, camera_action=True, **kw)


def fields(core):
    m = core.model
    sizes = dict(time=1, qpos=m.nq, qvel=m.nv, warmstart=m.nv, qfrc_applied=m.nv, xfrc_applied=6 * m.nbody,
                 mocap_pos=3 * m.nmocap, mocap_quat=4 * m.nmocap, userdata=m.nuserdata)
    out, at = {}, 0
    for k, v in sizes.items():
        out[k] = slice(at, at + v)
        at += v
    assert at == mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
    return out


def resting(d):
    pass


def running(core):
    for _ in range(30):
        core.step(np.array([0.0, 1.0, 0.0, 0.0], np.float32))


def inside_the_field(d):  # hamster 12 mm past its resting radius, thrown outward
    d.qpos[7:10] = d.qpos[0:3] + [REST + 0.012, 0, 0]
    d.qvel[6] = 3.0


def past_the_hard_stop(d):
    d.qpos[7:10] = d.qpos[0:3] + [0, REST + FIELD_HARD + 0.01, 0]
    d.qvel[7] = 2.0


def airborne(d):  # 3 m up, 0.5 s since the last wall touch, hamster off the wall: the foot force is cut
    d.qpos[2] += 3.0
    d.qpos[9] += 3.0
    d.time = 1.0
    d.userdata[LAST_TOUCH] = 0.5


def near_facing_wrap(d):
    d.userdata[FACING] = math.pi - 0.01


def pitch_at_max(d):
    d.userdata[PITCH] = PITCH_MAX - 0.001


def pitch_at_min(d):
    d.userdata[PITCH] = PITCH_MIN + 0.001


CASES = {  # name: (state setup, run before sharing, action)
    "resting": (resting, None, [0, 0, 0, 0]),
    "running": (resting, running, [0.3, 1.0, 0.0, 0.0]),
    "sideways": (resting, running, [1.0, 0.0, 0.0, 0.0]),
    "containment_field": (inside_the_field, None, [0, 1.0, 0, 0]),
    "hard_stop": (past_the_hard_stop, None, [0, 1.0, 0, 0]),
    "airborne": (airborne, None, [0, 1.0, 0, 0]),
    "turn_wraps": (near_facing_wrap, None, [0, 0, 1.0, 0]),
    "pitch_max": (pitch_at_max, None, [0, 0, 0, 1.0]),
    "pitch_min": (pitch_at_min, None, [0, 0, 0, -1.0]),
}


def shared(case):
    """A Core stepped one physics step, the state before it, and the action."""
    setup, run, action = CASES[case]
    core = Core(camera_action=True)
    setup(core.data)
    mujoco.mj_forward(core.model, core.data)
    if run:
        run(core)
    action = np.array(action, np.float32)
    before = core.state()
    core.step(action, 1)
    return before, core.state(), action, core


@pytest.mark.parametrize("case", CASES)
def test_one_physics_step_matches_the_cpu_core(case):
    before, after, action, core = shared(case)
    b = batch()
    b.set_state(1, before)
    b.step(np.stack([np.zeros(4, np.float32), action]), 1)
    got, f = b.state(1), fields(core)
    for k in ("xfrc_applied", "userdata"):
        np.testing.assert_allclose(got[f[k]], after[f[k]], **TIGHT, err_msg=k)
    for k in ("qpos", "qvel", "time"):
        np.testing.assert_allclose(got[f[k]], after[f[k]], **LOOSE, err_msg=k)


def test_an_env_step_of_ten_physics_steps_stays_within_the_bound():
    before, _, action, core = shared("running")
    core.set_state(before)
    b = batch()
    b.set_state(1, before)
    for _ in range(3):  # one graph per n, reused
        core.step(action)
        b.step(np.stack([np.zeros(4, np.float32), action]))
    got, ref, f = b.state(1), core.state(), fields(core)
    np.testing.assert_allclose(got[f["xfrc_applied"]], ref[f["xfrc_applied"]], **TIGHT)
    for k in ("qpos", "qvel"):
        np.testing.assert_allclose(got[f[k]], ref[f[k]], **LOOSE, err_msg=k)


def test_the_captured_graph_matches_eager_launches():
    """On CUDA the n steps run as one captured graph; on the CPU device both are eager, so this is trivial there."""
    before, _, action, core = shared("running")
    a, e = batch(), batch(graph=False)
    for b in (a, e):
        b.set_state(1, before)
        for _ in range(3):
            b.step(np.stack([np.zeros(4, np.float32), action]))
    f = fields(core)
    ga, ge = a.state(1), e.state(1)
    np.testing.assert_allclose(ga[f["xfrc_applied"]], ge[f["xfrc_applied"]], **TIGHT)
    for k in ("qpos", "qvel"):
        np.testing.assert_allclose(ga[f[k]], ge[f[k]], **LOOSE, err_msg=k)
    print("graph vs eager max |diff|:", float(np.abs(ga - ge).max()))


def test_set_state_then_state_returns_the_cpu_vector():
    before, _, _, _ = shared("running")
    b = batch()
    b.set_state(0, before)
    assert np.array_equal(b.state(0), before.astype(np.float32))  # Warp stores float32
    f32 = before.astype(np.float32).astype(np.float64)
    b.set_state(1, f32)
    assert np.array_equal(b.state(1), f32)  # a float32-exact vector round-trips exactly
    assert np.array_equal(b.state(0), before.astype(np.float32))  # and one world's load leaves the other's alone


def test_the_airborne_hamster_is_not_on_the_wall():
    b = batch()
    before, _, _, _ = shared("airborne")
    b.set_state(0, before)
    b.step(np.zeros((2, 4), np.float32), 1)
    assert not b.obs()["wall_contact"][0] and b.obs()["wall_contact"][1]


def test_reset_puts_the_given_worlds_back_at_their_spawn():
    b = batch()
    before, _, _, _ = shared("running")
    b.set_state(0, before)
    b.set_state(1, before)
    b.reset([1], spawn=(2.0, 3.0, R), facing=1.0)
    ref = Core(camera_action=True)
    ref.reset((2.0, 3.0, R), 1.0)
    np.testing.assert_allclose(b.state(1), ref.state(), atol=1e-6)
    assert np.array_equal(b.state(0), before.astype(np.float32))


def test_the_articulated_tier_is_not_batched_yet():
    from hamsterball.gpu.batch_core import BatchCore

    class Articulated:
        pass

    with pytest.raises(NotImplementedError):
        BatchCore(tier=Articulated(), device=DEVICE)


def test_a_plain_import_does_not_load_warp_or_torch():
    import subprocess
    import sys

    code = "import sys, hamsterball, hamsterball.env, hamsterball.core; assert 'warp' not in sys.modules and 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
