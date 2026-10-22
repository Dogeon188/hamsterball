"""Vector envs (spec §10 Vector envs, §15): the registered vector_entry_point and its stock vector envs."""

import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

import hamsterball  # noqa: F401  registers the env ids


def make_vec(n=2, **kw):
    return gym.make_vec("HamsterballReachGoal-v1", num_envs=n, **{"levels": ["first_goal"], **kw})


def test_make_vec_gives_a_stock_async_env_with_same_step_autoreset():
    venv = make_vec()
    try:
        assert type(venv) is AsyncVectorEnv
        assert venv.context == ("spawn" if sys.platform == "darwin" else "forkserver")
        assert venv.shared_memory is False  # no views
        assert venv.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
        obs, _ = venv.reset(seed=0)
        assert obs["state"].shape == (2, 21) and obs["goal"].shape == (2, 10)
    finally:
        venv.close()


def test_an_episode_end_resets_in_the_same_step_with_the_final_obs_in_info():
    venv = make_vec(time_limit_s=1 / 6)  # 10 env steps
    try:
        first, _ = venv.reset(seed=0)
        for _ in range(10):
            obs, _, terminated, truncated, info = venv.step(np.zeros((2, 2), np.float32))
        assert truncated.all() and not terminated.any()
        np.testing.assert_array_equal(obs["state"], first["state"])  # already the next episode's spawn
        assert not np.array_equal(info["final_obs"][0]["state"], first["state"][0])
        assert info["_final_obs"].all()
    finally:
        venv.close()


def test_linux_uses_forkserver(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    venv = make_vec(1)
    try:
        assert venv.context == "forkserver"
    finally:
        venv.close()


def test_sync_mode_gives_a_sync_vector_env():
    venv = make_vec(vectorization_mode="sync")
    assert type(venv) is SyncVectorEnv
    venv.reset(seed=0)


def test_the_warp_backend_is_not_built_in_v1():
    with pytest.raises(NotImplementedError):
        make_vec(backend="warp")


def test_views_render_in_spawned_workers():
    venv = make_vec(views=("chase", "first_person"), view_size=(32, 48))
    try:
        assert venv.shared_memory is True
        obs, _ = venv.reset(seed=0)
        obs, *_ = venv.step(np.zeros((2, 2), np.float32))
        for v in ("chase", "first_person"):
            assert obs[v].shape == (2, 32, 48, 3) and obs[v].dtype == np.uint8
            assert obs[v].any(axis=-1).mean() > 0.1  # drawn, not a black frame
    finally:
        venv.close()


WORKER = """
import os, pickle, sys
fn = pickle.loads(sys.stdin.buffer.read())
assert "mujoco" not in sys.modules  # unpickling a worker's env fn must leave mujoco unloaded
fn()
print(os.environ.get("MUJOCO_EGL_DEVICE_ID"))
"""


def run_worker(fn, **env):
    import os
    import pickle
    import subprocess

    base = {k: v for k, v in os.environ.items() if k != "MUJOCO_EGL_DEVICE_ID"}
    out = subprocess.run(
        [sys.executable, "-c", WORKER], input=pickle.dumps(fn), env={**base, **env}, capture_output=True, check=True
    )
    return out.stdout.decode().strip()


def test_each_worker_takes_an_egl_device_round_robin_before_mujoco_loads(monkeypatch):
    from hamsterball import env as E
    from hamsterball.core.generator import ProceduralGenerator

    monkeypatch.setattr(E, "egl_devices", lambda: 2)
    monkeypatch.setattr(gym.vector, "AsyncVectorEnv", lambda fns, **_: fns)  # the env fns, no workers started
    fns = E.make_vec(3, tasks=["reach_goal"], levels=ProceduralGenerator())  # kwargs that import mujoco on unpickle
    assert [run_worker(fn) for fn in fns] == ["0", "1", "0"]
    assert run_worker(fns[1], MUJOCO_EGL_DEVICE_ID="0") == "0"  # one set by hand wins
    monkeypatch.setattr(E, "egl_devices", lambda: 1)  # one GPU: left to MuJoCo
    assert run_worker(E.make_vec(1, tasks=["reach_goal"])[0]) == "None"
