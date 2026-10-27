"""Kinematic bodies on the batched core and the Warp vector env (GPU effort): seam 1 against `Core` (driven bodies and
the CPU-layout state) and seam 2 against the CPU env (recorded episodes of body levels replayed through `step_to`)."""

import os

import numpy as np
import pytest
import torch

from hamsterball.core import Core
from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from hamsterball.core.model import DT
from hamsterball.env import LEVELS_DIR

pytestmark = pytest.mark.gpu

DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")  # "cuda:0" runs the same tests on a GPU
FLOAT32 = dict(rtol=1e-5, atol=1e-6)  # what the spec asks of the core-step kernels' writes
LOOSE = dict(rtol=1e-4, atol=1e-5)  # shell and hamster dofs after a physics step; see batch_core.md
HFIELD = (dict(rtol=1e-4, atol=3e-5), dict(rtol=1e-4, atol=2e-2))  # the same on a generated hfield: Warp's contacts; see batch_core.md
GEN = ProceduralGenerator()
STILL = np.zeros(2, np.float32)


def bundled(name):
    return Level.load(LEVELS_DIR / f"{name}.yaml")


FLOOR = Level.from_dict({
    "kill_plane_z": -5, "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}],
    "tasks": {"survive": {"spawn": {"pos": [0, 0, 0.5]}, "region": {"type": "sphere", "pos": [0, 0, 0.5], "radius": 5},
                          "time_limit_s": 10}}})
PUSHERS, RAFT = bundled("survive_pushers"), bundled("survive_raft")  # two oscillating bars; a tilting raft on a path
PILLARS = GEN("survive", 18, 0.9)  # three pillars on a hfield
PLATFORM = GEN("survive", 0, 0.5)  # an oscillating, rocking platform with a region attached
assert [len(lv.bodies) for lv in (FLOOR, PUSHERS, RAFT, PILLARS, PLATFORM)] == [0, 2, 1, 3, 1]
BANK = [FLOOR, PUSHERS, RAFT, PILLARS, PLATFORM]


def batch(levels, **kw):
    from hamsterball.gpu.batch_core import BatchCore

    return BatchCore(levels=levels, num_envs=len(levels), device=DEVICE, tasks=["survive"], **kw)


def cpu_state(lv, env_steps):
    """A Core on the level, run still from its spawn for env_steps: the core."""
    core = Core(level=lv)
    core.reset(lv.tasks["survive"]["spawn"]["pos"])
    for _ in range(env_steps):
        core.step(STILL)
    return core


def body_slices(core):
    """Where each driven body's qpos and qvel sit in the CPU core-state vector (time, qpos, qvel, ...)."""
    nq = core.model.nq
    return [(slice(1 + b.qpos, 8 + b.qpos), slice(1 + nq + b.dof, 7 + nq + b.dof)) for b in core.bodies]


def test_every_level_of_a_bank_with_bodies_round_trips_its_cpu_state():
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        s = cpu_state(lv, 30).state().astype(np.float32).astype(np.float64)
        b.reset([w], bank_index=w)
        b.set_state(w, s)
        assert np.array_equal(b.state(w), s), w
    # and all worlds at once, as step_to loads them: a list of vectors of different lengths
    states = [cpu_state(lv, 5).state().astype(np.float32) for lv in BANK]
    assert len({len(s) for s in states}) > 1
    b.set_states(states)
    for w, s in enumerate(states):
        assert np.array_equal(b.state(w), s), w
    assert all(np.array_equal(a, s) for a, s in zip(b.cpu_states(), states))


@pytest.mark.parametrize("env_steps", [0, 30, 400])
def test_drive_bodies_matches_the_cpu_core_after_one_physics_step(env_steps):
    """Each world is on a level with bodies (or none): the bodies' qpos and qvel after a step are the CPU's at float32
    tolerance, whatever the padded slots beyond the level's own bodies hold."""
    b = batch(BANK)
    cores = []
    for w, lv in enumerate(BANK):
        core = cpu_state(lv, env_steps)
        b.reset([w], bank_index=w)
        b.set_state(w, core.state())
        cores.append(core)
    b.step(np.zeros((len(BANK), 2), np.float32), 1)
    for w, core in enumerate(cores):
        core.step(STILL, 1)
        want, got = core.state(), b.state(w)
        assert len(got) == len(want)
        np.testing.assert_allclose(got[0], want[0], atol=1e-6)
        for qp, qv in body_slices(core):
            np.testing.assert_allclose(got[qp], want[qp], **FLOAT32, err_msg=f"world {w} body qpos")
            np.testing.assert_allclose(got[qv], want[qv], **FLOAT32, err_msg=f"world {w} body qvel")
        nq = core.model.nq
        tol_q, tol_v = HFIELD if BANK[w] is PILLARS else (LOOSE, LOOSE)
        np.testing.assert_allclose(got[1:15], want[1:15], **tol_q, err_msg=f"world {w} shell and hamster qpos")
        np.testing.assert_allclose(got[1 + nq:13 + nq], want[1 + nq:13 + nq], **tol_v, err_msg=f"world {w} shell and hamster qvel")


def test_a_bodies_level_steps_as_it_does_in_a_bank_of_its_own():
    """Padding changes nothing: a world in the mixed bank equals the same world in a bank of its level alone."""
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        s = cpu_state(lv, 60).state()
        alone = batch([lv])
        for core, i in ((b, w), (alone, 0)):
            core.reset([i], bank_index=i)
            core.set_state(i, s)
        for core in (b, alone):
            core.step(np.zeros((core.num_envs, 2), np.float32), 3)
        np.testing.assert_allclose(b.state(w), alone.state(0), rtol=0, atol=1e-6, err_msg=f"world {w}")


def test_a_reset_poses_the_bodies_as_the_cpu_core_does():
    b = batch(BANK)
    b.reset()
    for w, lv in enumerate(BANK):
        b.reset([w], bank_index=w)
        core = Core(level=lv)
        core.reset(lv.tasks["survive"]["spawn"]["pos"])
        want, got = core.state(), b.state(w)
        for qp, qv in body_slices(core):
            np.testing.assert_allclose(got[qp], want[qp], **FLOAT32, err_msg=f"world {w}")
            np.testing.assert_allclose(got[qv], want[qv], **FLOAT32, err_msg=f"world {w}")


# ---- seam 2: the Warp vector env against the CPU env on recorded episodes -------------------------------------------


from test_warp_env import CODES, OBS  # noqa: E402

from hamsterball.env import HamsterballEnv  # noqa: E402


def chase(obs, _):
    """Steer at the region's centre, slowing within 3 m of it."""
    g = obs["goal"][:2]
    n = float(np.linalg.norm(g))
    return (g / max(n, 1e-6) * min(1.0, n / 3)).astype(np.float32)


def still(obs, _):
    return STILL


PILLARS2 = GEN("survive", 18, 0.5)
# id: (level, policy, env kwargs, expected outcome)
SCENARIOS = {
    "raft_rolls_off_its_low_edge": (RAFT, still, {"time_limit_s": 12}, "left_region"),
    "raft_chased_and_lost": (RAFT, chase, {"time_limit_s": 12}, "left_region"),
    "pusher_shoves_a_still_ball_out": (PUSHERS, still, {"time_limit_s": 12}, "left_region"),
    "platform_ridden_to_time_up_shaped": (PLATFORM, chase, {"time_limit_s": 12, "shaped": True}, "success"),
    "pillars_ridden_across_to_time_up": (PILLARS2, chase, {"time_limit_s": 12}, "success"),
    "three_pillars_shove_a_still_ball_out": (PILLARS, still, {"time_limit_s": 12}, "left_region"),
}


def region_pose(model, data, geom, w=None):
    """World pose (xpos, xmat) of a geom, from the CPU data or world w of the Warp data."""
    if w is None:
        return data.geom_xpos[geom].copy(), data.geom_xmat[geom].copy().reshape(3, 3)
    return data.geom_xpos.numpy()[w, geom], data.geom_xmat.numpy()[w, geom].reshape(3, 3)


def record(lv, policy, kw):
    """One CPU episode: the reset, then per step the core state, what step returns and the attached region's pose."""
    env = HamsterballEnv(levels=[lv], tasks=["survive"], **kw)
    obs, info = env.reset(seed=0)
    attached = "attach" in lv.tasks["survive"]["region"]
    g = env.core.model.geom("survive/region").id
    pose = lambda: region_pose(env.core.model, env.core.data, g) if attached else None  # noqa: E731
    ep = {"reset_obs": obs, "reset_pose": pose(), "steps": [], "attached": attached}
    for i in range(env._max_steps + 5):
        obs, r, term, trunc, info = env.step(policy(obs, i))
        ep["steps"].append((info["core_state"], obs, r, term, trunc, info, pose()))
        if term or trunc:
            break
    last = ep["steps"][-1]
    ep["outcome"] = "success" if last[5]["success"] else last[5]["failure_reason"] or ("truncated" if last[4] else "running")
    return ep


def warp_region(venv, w):
    e = int(venv.core.entry.numpy()[w])
    return region_pose(None, venv.core.data, int(venv.row_geom.numpy()[e * venv.markers.stride]), w)


def venv_for(levels, n, **kw):
    from hamsterball.gpu.vector_env import WarpVectorEnv

    return WarpVectorEnv(n, levels=levels, tasks=["survive"], device=DEVICE, **kw)


def replay(venv, eps, labels):
    """Feed each world the core states of its CPU episode; compare observations, rewards, flags, codes and the attached
    region's pose. Returns the worst observation, reward and pose differences."""
    worst = worst_r = worst_pose = 0.0

    def same_pose(w, want, where):
        got = warp_region(venv, w)
        np.testing.assert_allclose(got[0], want[0], atol=1e-5, err_msg=f"{where} region pos")
        np.testing.assert_allclose(got[1], want[1], atol=1e-5, err_msg=f"{where} region xmat")
        return max(float(np.abs(got[0] - want[0]).max()), float(np.abs(got[1] - want[1]).max()))

    for w, ep in enumerate(eps):
        if ep["attached"]:
            worst_pose = max(worst_pose, same_pose(w, ep["reset_pose"], f"{labels[w]} reset"))
    for t in range(max(len(e["steps"]) for e in eps)):
        # a world whose episode is over has reset onto a level of the draw (and a layout of its own): hold its state
        held = venv.core.cpu_states()
        states = [e["steps"][t][0] if t < len(e["steps"]) else held[w] for w, e in enumerate(eps)]
        obs, reward, terminated, truncated, info = venv.step_to(states)
        for w, ep in enumerate(eps):
            if t >= len(ep["steps"]):
                continue
            cpu_obs, cpu_r, cpu_term, cpu_trunc, cpu_info, cpu_pose = ep["steps"][t][1:]
            where = f"{labels[w]} step {t}"
            assert bool(terminated[w]) == cpu_term and bool(truncated[w]) == cpu_trunc, where
            assert bool(info["success"][w]) == cpu_info["success"], where
            assert int(info["failure_reason"][w]) == CODES[cpu_info["failure_reason"]], where
            done = cpu_term or cpu_trunc
            got = info["final_obs"] if done else obs  # a finished world has already reset
            for k in ("state", "goal"):
                g = got[k][w].cpu().numpy()
                np.testing.assert_allclose(g, cpu_obs[k], **OBS, err_msg=f"{where} {k}")
                worst = max(worst, float(np.abs(g - cpu_obs[k]).max()))
            np.testing.assert_allclose(float(reward[w]), cpu_r, **OBS, err_msg=where)
            worst_r = max(worst_r, abs(float(reward[w]) - cpu_r))
            if ep["attached"] and not done:  # a finished world's markers are already at its next episode's start
                worst_pose = max(worst_pose, same_pose(w, cpu_pose, where))
    return worst, worst_r, worst_pose


@pytest.mark.parametrize("name", SCENARIOS)
def test_step_to_matches_the_cpu_env_on_recorded_episodes_on_body_levels(name):
    lv, policy, kw, want = SCENARIOS[name]
    ep = record(lv, policy, kw)
    assert ep["outcome"] == want  # the scenario does what it says
    venv = venv_for([lv], 1, **kw)
    obs, _ = venv.reset(seed=0)
    for k in ("state", "goal"):
        np.testing.assert_allclose(obs[k][0].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
    worst = replay(venv, [ep], [name])
    print(f"{name}: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}, region pose {worst[2]:.3g}")


def test_episodes_on_different_body_levels_of_one_bank_replay_together():
    """Worlds forced onto a raft, two bars, a platform and three pillars of one padded bank (and a level with no body):
    each replays its own CPU episode, so the other worlds' bodies change nothing."""
    lvs = [RAFT, PUSHERS, PLATFORM, PILLARS, bundled("first_survive")]
    pols = [still, still, chase, still, still]
    eps = [record(lv, p, {"time_limit_s": 12}) for lv, p in zip(lvs, pols)]
    venv = venv_for(lvs, len(lvs), time_limit_s=12)
    venv.reset(seed=0, options={"level_index": torch.arange(len(lvs))})
    worst = replay(venv, eps, [f"level{i}" for i in range(len(lvs))])
    print(f"mixed bank: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}, region pose {worst[2]:.3g}")


def test_live_physics_drives_each_world_from_its_own_clock_and_the_ball_rides():
    """Through `step` on a platform bank: a world's body follows its motion at its sim time, and a ball that is chased
    to the region stays inside it for many steps (it rides the platform)."""
    venv = venv_for([PLATFORM, PUSHERS], 2, time_limit_s=6)
    venv.reset(seed=0, options={"level_index": torch.tensor([0, 1])})
    for _ in range(120):
        obs, r, term, trunc, info = venv.step(torch.stack([torch.as_tensor(chase({"goal": venv._obs["goal"][w].cpu().numpy()}, 0)) for w in range(2)]))
        assert not bool(info["failure_reason"][0]), "the ridden platform keeps the ball in its region"


def test_a_generator_pool_with_body_layouts_trains_on_the_device():
    """A bank of generated survive levels (every layout the generator draws) pads to the most bodies any holds; worlds
    on every kind step, finish and reset in the same step."""
    venv = venv_for(GEN, 8, bank_size=16, difficulty=0.8)
    assert venv.core.bank.nbodies >= 1
    obs, info = venv.reset(seed=3)
    for i in range(150):
        obs, r, term, trunc, info = venv.step(torch.zeros(8, 2))
    assert torch.isfinite(obs["state"]).all() and torch.isfinite(obs["goal"]).all()
    kinds = {len(venv.levels[i].bodies) for i in info["level_index"].tolist()}
    assert len(kinds) > 1  # worlds were on levels with different numbers of bodies


def test_set_levels_swaps_between_banks_with_different_numbers_of_bodies():
    """Running worlds keep their state across a rebuild whose padded layout differs (a bank with no body, then one
    with three, then one with one)."""
    first = bundled("first_survive")
    venv = venv_for([first, FLOOR], 2, time_limit_s=12)
    venv.reset(seed=0)
    for _ in range(5):
        venv.step(torch.zeros(2, 2))
    before = [s.copy() for s in venv.core.cpu_states()]
    for pool in ([PILLARS, PUSHERS], [RAFT]):
        venv.set_levels(pool)
        assert all(np.array_equal(a, b) for a, b in zip(venv.core.cpu_states(), before)), "worlds keep their state"
        for _ in range(3):
            obs, *_ = venv.step(torch.zeros(2, 2))
        assert torch.isfinite(obs["state"]).all()
        before = [s.copy() for s in venv.core.cpu_states()]
