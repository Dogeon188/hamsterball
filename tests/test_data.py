"""Recording and replay (spec §12): the Collector and the re-step determinism check."""

import importlib.metadata
import platform

import gymnasium as gym
import numpy as np
import pytest

minari = pytest.importorskip("minari")

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.data import Collector, ReplayMismatch, replay
from hamsterball.env import TASKS


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


def record(env_id="HamsterballReachGoal-v0", episodes=1, steps=20, seed=0, **kw):
    """Record episodes of seeded random actions, each cut after `steps` env steps, into a new dataset."""
    env = Collector(gym.make(env_id, **kw))
    env.action_space.seed(seed)
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        for _ in range(steps):
            # float64, as a policy might give: the Collector steps with the float32 action it stores
            if any(env.step(env.action_space.sample().astype(np.float64))[2:4]):
                break
    return env


def test_each_episode_stores_the_envs_reset_options():
    env = record(episodes=2, levels=["first_goal", "hill"])
    _, info = gym.make("HamsterballReachGoal-v0", levels=["first_goal", "hill"]).reset(seed=1)
    ds = env.create_dataset("test/reset-options-v0", algorithm_name="random")
    meta = list(ds.storage.get_episode_metadata([0, 1]))
    assert meta[1]["options"] == info["reset_options"]
    assert meta[0]["seed"] == 0 and meta[1]["seed"] == 1


def test_discard_episode_drops_the_episode_in_progress():
    # this also guards the Minari internal discard relies on (DataCollector._buffer): if it changes, this fails
    env = record(episodes=2, steps=5, levels=["first_goal"])
    env.discard_episode()
    env.reset(seed=7)
    env.step(env.action_space.sample())
    ds = env.create_dataset("test/discard-v0", algorithm_name="random")
    assert ds.total_episodes == 2 and ds.total_steps == 5 + 1
    assert [m["seed"] for m in ds.storage.get_episode_metadata([0, 1])] == [0, 7]


def test_the_dataset_records_versions_platform_and_every_space_shaping_kwarg():
    env = record(levels=["first_goal"], camera_action=True, frame_skip=5, shaped=True)
    ds = env.create_dataset("test/meta-v0", algorithm_name="random")
    meta = minari.load_dataset("test/meta-v0").storage.metadata
    assert meta["hamsterball_version"] == importlib.metadata.version("hamsterball")
    assert meta["mujoco_version"] == "3.16.0"
    assert meta["platform"] == f"{platform.system()}-{platform.machine()}"
    kw = ds.env_spec.kwargs
    assert kw["tier"] == "abstract" and kw["camera_action"] and kw["frame_skip"] == 5 and kw["shaped"]
    rebuilt = ds.recover_environment()
    assert rebuilt.action_space == env.action_space and rebuilt.unwrapped.frame_skip == 5


def test_episode_metadata_holds_the_outcome_return_and_length():
    env = Collector(gym.make("HamsterballSurvive-v0"))
    for action in ([0, 1], [0, 0]):  # run out of the region, then stand still to the time limit
        env.reset(seed=0)
        while not any(env.step(np.array(action, np.float32))[2:4]):
            pass
    ds = env.create_dataset("test/outcome-v0", algorithm_name="scripted")
    left, stayed = ds.storage.get_episode_metadata([0, 1])
    assert (left["success"], left["failure_reason"], left["return"]) == (False, "left_region", -1.0)
    assert (stayed["success"], stayed["failure_reason"], stayed["return"], stayed["length"]) == (True, None, 1.0, 600)
    assert left["length"] == left["total_steps"]


@pytest.mark.parametrize("task", TASKS)
def test_an_episode_of_every_task_replays_bit_for_bit(task):
    env = record("Hamsterball-v0", episodes=2, steps=60, tasks=[task], camera_action=True)
    replay(env.create_dataset(f"test/replay-{task}-v0", algorithm_name="random"))


def test_replay_reports_the_first_step_that_diverges(monkeypatch):
    monkeypatch.setattr("hamsterball.core.step.DRAG_K", 0.06)  # record under other physics
    ds = record(steps=10, levels=["first_goal"]).create_dataset("test/diverge-v0", algorithm_name="random")
    monkeypatch.undo()
    with pytest.raises(ReplayMismatch) as e:
        replay(ds)
    assert (e.value.episode, e.value.step) == (0, 1) and e.value.diff > 0
    assert e.value.field.split("[")[0] in ("qpos", "qvel", "warmstart")


def test_replay_warns_when_the_recording_build_differs():
    ds = record(steps=3, levels=["first_goal"]).create_dataset("test/old-v0", algorithm_name="random")
    ds.storage.update_metadata({"mujoco_version": "3.15.0", "platform": "Windows-AMD64"})
    with pytest.warns(UserWarning, match="mujoco_version") as w:
        replay(ds)
    assert any("platform" in str(x.message) for x in w)


def test_a_pixel_env_records_state_and_goal_and_its_view_kwargs():
    env = record(steps=5, levels=["first_goal"], views=("chase",), view_size=(16, 24), obs_state=False)
    ds = env.create_dataset("test/pixels-v0", algorithm_name="random")
    assert set(ds.observation_space.keys()) == {"state", "goal"}
    kw = ds.env_spec.kwargs
    assert (list(kw["views"]), list(kw["view_size"]), kw["obs_state"]) == (["chase"], [16, 24], False)
    replay(ds)
