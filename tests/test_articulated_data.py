"""Recording an articulated episode: torques as actions, proprio, the run vector, and the low-level policy's name."""

import gymnasium as gym
import numpy as np
import pytest

minari = pytest.importorskip("minari")

import hamsterball  # noqa: F401  registers the env ids
from hamsterball import lowlevel
from hamsterball.data import Collector, replay, rerender, to_lerobot

STEPS = 40


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


def play(dataset_id="test/gait-v0", **kw):
    """Record a gait-driven articulated episode of STEPS env steps."""
    env = Collector(gym.make("HamsterballReachGoal-v2", levels=["first_goal"], tier="articulated", **kw))
    gait = lowlevel.get(lowlevel.DEFAULT)
    obs, _ = env.reset(seed=0)
    run = np.array([0.0, 1.0])
    for _ in range(STEPS):
        torques = gait.act(obs, run)
        obs, *_ = env.step(torques, run_vector=run)
    return env.create_dataset(dataset_id, algorithm_name="scripted", low_level_policy=gait.name)


def test_a_gait_run_stores_torques_proprio_run_vector_and_policy_name():
    ds = play()
    ep = next(ds.iterate_episodes())
    assert ep.actions.shape == (STEPS, 8)
    assert ep.observations["proprio"].shape == (STEPS + 1, 26)
    assert ds.observation_space["proprio"].shape == (26,)
    rv = ep.infos["run_vector"]
    assert rv.shape == (STEPS + 1, 2)
    np.testing.assert_array_equal(rv[1:], np.tile([0.0, 1.0], (STEPS, 1)))
    assert ds.storage.metadata["low_level_policy"] == lowlevel.DEFAULT
    assert ds.recover_environment().observation_space == ds.observation_space


def test_an_articulated_dataset_replays_bit_for_bit():
    replay(play())


def test_abstract_datasets_store_no_proprio_and_no_run_vector():
    env = Collector(gym.make("HamsterballReachGoal-v2", levels=["first_goal"]))
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.array([0.0, 1.0]))
    ds = env.create_dataset("test/abstract-v0", algorithm_name="scripted")
    ep = next(ds.iterate_episodes())
    assert "proprio" not in ds.observation_space and "run_vector" not in ep.infos
    assert "low_level_policy" not in ds.storage.metadata


@pytest.mark.gl
def test_rerender_recomputes_proprio_wall_flag_and_progress(monkeypatch):
    src = play()
    def no_physics(*a, **k):
        raise AssertionError("re-render stepped physics")
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase",), (24, 32), "test/gait-rerendered-v0")
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    for k in ("proprio", "state", "goal"):
        np.testing.assert_array_equal(b.observations[k], a.observations[k])
    np.testing.assert_array_equal(b.infos["run_vector"], a.infos["run_vector"])
    assert out.storage.metadata["low_level_policy"] == lowlevel.DEFAULT


@pytest.mark.gl
def test_to_lerobot_adds_the_proprio_feature(tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    ds = rerender(play(), ("chase",), (24, 32), "test/gait-rerendered-v0")
    to_lerobot(ds, "hamsterball/test", tmp_path / "lerobot", use_videos=False)
    out = lerobot_dataset.LeRobotDataset("hamsterball/test", root=tmp_path / "lerobot")
    assert out.features["observation.proprio"]["shape"] == (26,)
    assert out.features["action"]["shape"] == (8,)
