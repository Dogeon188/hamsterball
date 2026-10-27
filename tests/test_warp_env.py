"""The Warp vector env against the CPU env (GPU effort, seam 2): recorded CPU episodes replayed through `step_to`."""

import os

import gymnasium as gym
import numpy as np
import pytest

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level
from hamsterball.env import HamsterballEnv

pytestmark = pytest.mark.gpu

DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")  # "cuda:0" runs the same tests on a GPU
# float32 tolerances for observations and rewards, measured against the CPU env; see vector_env.md
OBS = dict(rtol=1e-5, atol=1e-5)
CODES = {None: 0, "hazard": 1, "kill_plane": 2, "left_region": 3}  # env_step.FAILURE_CODES


def level(goal, hazards=()):
    return Level.from_dict({
        "kill_plane_z": -5,
        "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}],
        "hazards": list(hazards),
        "tasks": {"reach_goal": {"spawn": {"pos": [0, 0, 0.5], "yaw_deg": 0}, "goal": goal, "time_limit_s": 30}},
    })


SPHERE = {"type": "sphere", "pos": [0, 8, 0.5], "radius": 1}
CAPSULE = {"type": "capsule", "fromto": [-3, 8, 0.5, 3, 8, 0.5], "radius": 0.5}
BOX = {"type": "box", "pos": [0.5, 8, 0.5], "size": [1, 0.5, 1], "euler": [0, 0, 30]}
AHEAD = {"type": "sphere", "pos": [0, 4, 0.5], "radius": 0.3}
FORWARD, SIDEWAYS, STILL = [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]

# id: (level, [(action held all episode, expected outcome)], env kwargs)
SCENARIOS = {
    "sphere_goal": (level(SPHERE), [(FORWARD, "success"), (STILL, "truncated")], {"time_limit_s": 6}),
    "capsule_goal": (level(CAPSULE), [(FORWARD, "success"), ([0.4, 1.0], "success")], {"time_limit_s": 6}),
    "box_goal": (level(BOX), [(FORWARD, "success"), ([0.1, 1.0], "success")], {"time_limit_s": 6, "shaped": True}),
    "sphere_hazard": (level(SPHERE, [AHEAD]), [(FORWARD, "hazard")], {"time_limit_s": 6, "shaped": True, "shaping_gamma": 0.9}),
    "capsule_hazard": (
        level(SPHERE, [{"type": "capsule", "fromto": [-1, 4, 0.5, 1, 4, 0.5], "radius": 0.2}]),
        [(FORWARD, "hazard")], {"time_limit_s": 6}),
    "box_hazard": (
        level(SPHERE, [{"type": "box", "pos": [0.3, 4, 0.5], "size": [0.4, 0.2, 0.5], "euler": [0, 0, 40]}]),
        [(FORWARD, "hazard")], {"time_limit_s": 6}),
    "falls_off": (level(SPHERE), [(SIDEWAYS, "kill_plane")], {"time_limit_s": 30, "shaped": True}),
    "no_truncation": (level(SPHERE), [(STILL, "running")], {"time_limit_s": 1, "truncate": False}),
}


def record(lv, action, kw):
    """One CPU episode under a constant action: the reset observation, then per step the core state and what step returns."""
    env = HamsterballEnv(levels=[lv], tasks=["reach_goal"], **kw)
    obs, info = env.reset(seed=0)
    ep = {"reset_obs": obs, "reset_state": info["core_state"], "steps": []}
    for _ in range(env._max_steps + 5):
        obs, r, term, trunc, info = env.step(np.array(action, np.float32))
        ep["steps"].append((info["core_state"], obs, r, term, trunc, info))
        if term or trunc:
            break
    last = ep["steps"][-1]
    ep["outcome"] = "success" if last[5]["success"] else last[5]["failure_reason"] or ("truncated" if last[4] else "running")
    return ep


def warp_env(lv, n, **kw):
    from hamsterball.gpu.vector_env import WarpVectorEnv

    return WarpVectorEnv(n, levels=[lv], tasks=["reach_goal"], device=DEVICE, **kw)


@pytest.mark.parametrize("name", SCENARIOS)
def test_step_to_matches_the_cpu_env_on_recorded_episodes(name):
    lv, episodes, kw = SCENARIOS[name]
    eps = [record(lv, action, kw) for action, _ in episodes]
    assert [e["outcome"] for e in eps] == [want for _, want in episodes]  # the scenario does what it says
    venv = warp_env(lv, len(eps), **kw)
    obs, _ = venv.reset(seed=0)
    for w, ep in enumerate(eps):  # the reset observation is the CPU's
        for k in ("state", "goal"):
            np.testing.assert_allclose(obs[k][w].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
    worst = worst_r = 0.0
    for t in range(max(len(e["steps"]) for e in eps)):
        states = np.stack([e["steps"][min(t, len(e["steps"]) - 1)][0] for e in eps])
        obs, reward, terminated, truncated, info = venv.step_to(states)
        for w, ep in enumerate(eps):
            if t >= len(ep["steps"]):
                continue
            _, cpu_obs, cpu_r, cpu_term, cpu_trunc, cpu_info = ep["steps"][t]
            where = f"{name} world {w} step {t}"
            assert bool(terminated[w]) == cpu_term and bool(truncated[w]) == cpu_trunc, where
            assert bool(info["success"][w]) == cpu_info["success"], where
            assert int(info["failure_reason"][w]) == CODES[cpu_info["failure_reason"]], where
            done = cpu_term or cpu_trunc
            got = info["final_obs"] if done else obs  # a finished world has already reset: its last obs is final_obs
            assert bool(info["_final_obs"][w]) == done, where
            for k in ("state", "goal"):
                g = got[k][w].cpu().numpy()
                np.testing.assert_allclose(g, cpu_obs[k], **OBS, err_msg=f"{where} {k}")
                worst = max(worst, float(np.abs(g - cpu_obs[k]).max()))
            np.testing.assert_allclose(float(reward[w]), cpu_r, **OBS, err_msg=where)
            worst_r = max(worst_r, abs(float(reward[w]) - cpu_r))
            for term_name, v in cpu_info["reward_terms"].items():
                np.testing.assert_allclose(float(info["reward_terms"][term_name][w]), v, **OBS, err_msg=f"{where} {term_name}")
    print(f"{name}: worst observation difference {worst:.3g}, reward {worst_r:.3g}")


def test_a_finished_world_is_already_at_its_spawn_when_step_returns():
    lv, _, _ = SCENARIOS["sphere_goal"]
    venv = warp_env(lv, 2, time_limit_s=1 / 6)  # 10 env steps
    first, _ = venv.reset(seed=0)
    for _ in range(10):
        obs, _, terminated, truncated, info = venv.step(np.zeros((2, 2), np.float32))
    assert truncated.all() and not terminated.any()
    np.testing.assert_allclose(obs["state"].cpu().numpy(), first["state"].cpu().numpy(), atol=1e-5)  # the next episode
    assert not np.allclose(info["final_obs"]["state"].cpu().numpy(), first["state"].cpu().numpy(), atol=1e-5)
    assert info["_final_obs"].all() and info["_final_info"].all()
    assert info["final_info"]["success"].shape == (2,)
    obs, _, terminated, truncated, info = venv.step(np.zeros((2, 2), np.float32))  # and the new episode runs from step 0
    assert not truncated.any() and not info["_final_obs"].any()


def test_physics_steps_reach_the_goal_and_return_torch_tensors_on_the_device():
    import torch

    lv, _, _ = SCENARIOS["sphere_goal"]
    venv = warp_env(lv, 3, time_limit_s=10)
    venv.reset(seed=0)
    action = torch.tensor([FORWARD, SIDEWAYS, STILL], dtype=torch.float32, device=venv.torch_device)
    won = None
    for _ in range(400):
        obs, reward, terminated, truncated, info = venv.step(action)
        for t in (obs["state"], obs["goal"], reward, terminated, truncated, info["success"]):
            assert isinstance(t, torch.Tensor) and t.device == venv.torch_device
        assert obs["state"].shape == (3, 21) and obs["goal"].shape == (3, 10) and reward.dtype == torch.float32
        if info["success"][0]:
            won = reward[0].item()
            break
    assert won is not None and won > 0.9  # +1 for the goal, less the time penalty
    assert not info["success"][1:].any()


def test_spaces_match_the_cpu_envs():
    lv, _, _ = SCENARIOS["sphere_goal"]
    for kw in ({}, {"camera_action": True}):
        cpu = HamsterballEnv(levels=[lv], tasks=["reach_goal"], **kw)
        venv = warp_env(lv, 4, **kw)
        assert venv.single_action_space == cpu.action_space
        assert venv.single_observation_space == cpu.observation_space
        assert venv.num_envs == 4
        assert venv.observation_space == gym.vector.utils.batch_space(cpu.observation_space, 4)
        assert venv.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP


def test_make_vec_builds_the_warp_backend_and_the_cpu_backend_is_unchanged():
    from hamsterball.gpu.vector_env import WarpVectorEnv

    venv = gym.make_vec("HamsterballReachGoal-v2", num_envs=2, vectorization_mode="vector_entry_point", backend="warp",
                        levels=["first_goal"], device=DEVICE, bank_size=8, model_cache=4)  # the last two are accepted
    assert isinstance(venv, WarpVectorEnv) and isinstance(venv, gym.vector.VectorEnv)
    cpu = gym.make_vec("HamsterballReachGoal-v2", num_envs=1, levels=["first_goal"])
    try:
        assert type(cpu) is gym.vector.AsyncVectorEnv
    finally:
        cpu.close()


def test_an_hfield_level_runs():
    """An hfield level compiles into the one model like any other."""
    venv = gym.make_vec("HamsterballReachGoal-v2", num_envs=2, vectorization_mode="vector_entry_point", backend="warp",
                        levels=["hill"], device=DEVICE)
    venv.reset(seed=0)
    obs, *_ = venv.step(np.zeros((2, 2), np.float32))
    assert np.isfinite(obs["state"].cpu().numpy()).all()


@pytest.mark.parametrize("kw, error", [
    ({"tier": "articulated"}, NotImplementedError),
    ({"obs_state": False}, ValueError),
    ({"difficulty": 0.7}, ValueError),
    ({"no_such_kwarg": 1}, TypeError),
])
def test_unsupported_kwargs_raise_a_clear_error(kw, error):
    from hamsterball.gpu.vector_env import WarpVectorEnv

    with pytest.raises(error):
        WarpVectorEnv(2, **{"levels": ["first_goal"], "tasks": ["reach_goal"], "device": DEVICE, **kw})


def test_a_missing_cuda_device_is_a_runtime_error():
    from hamsterball.gpu.vector_env import WarpVectorEnv

    with pytest.raises(RuntimeError):
        WarpVectorEnv(1, levels=["first_goal"], tasks=["reach_goal"], device="cuda:99")


def test_without_the_gpu_extra_the_error_names_it():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['warp'] = None\n"
        "try:\n    import hamsterball.gpu.vector_env\nexcept ImportError as e:\n    assert 'gpu' in str(e), e\nelse:\n    raise SystemExit('no error')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
