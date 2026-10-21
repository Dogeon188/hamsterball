"""Re-render, video, and LeRobot export (spec §12): pixels from stored core states, with no physics stepped."""

import gymnasium as gym
import numpy as np
import pytest

minari = pytest.importorskip("minari")

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.data import Collector, rerender, to_video
from test_tasks import toward_target


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


def play(task="unlock_goal", level="locked_goal", dataset_id="test/played-v0", **kw):
    """Record one episode of the scripted controller to success: keys hide, the goal unlocks, then goes inactive."""
    env = Collector(gym.make("Hamsterball-v0", tasks=[task], levels=[level], **kw))
    obs, _ = env.reset(seed=0)
    for _ in range(5000):
        obs, _, terminated, truncated, info = env.step(toward_target(obs))
        if terminated or truncated:
            break
    assert info["success"]
    return env.create_dataset(dataset_id, algorithm_name="scripted")


def live_frames(ds, view, view_size):
    """Re-step the recorded actions in a pixel env: the frames a live recording with that view would have stored."""
    ep = next(ds.iterate_episodes())
    (meta,) = ds.storage.get_episode_metadata([ep.id])
    env = gym.make(ds.env_spec, views=(view,), view_size=view_size)
    obs, _ = env.reset(seed=meta["seed"], options=meta["options"])
    frames = [obs[view]] + [env.step(a)[0][view] for a in ep.actions]
    return np.stack(frames)


def no_physics(*a, **kw):
    raise AssertionError("re-render stepped physics")


def test_rerender_recomputes_progress_and_adds_lossless_views(monkeypatch):
    src = play()
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase", "first_person"), (24, 32), "test/rerendered-v0")
    assert minari.load_dataset("test/rerendered-v0").total_steps == src.total_steps
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    # progress (fraction remaining, the active flag, the target) recomputed from stored states equals the live record
    np.testing.assert_array_equal(b.observations["goal"], a.observations["goal"])
    assert b.observations["goal"][-1, 3] == 0  # success: no active target
    np.testing.assert_array_equal(b.observations["state"], a.observations["state"])
    np.testing.assert_array_equal(b.actions, a.actions)
    np.testing.assert_array_equal(b.rewards, a.rewards)
    # lossless, and with hidden keys and the unlocked goal: bit for bit the frames a live pixel env renders
    np.testing.assert_array_equal(b.observations["chase"], live_frames(src, "chase", (24, 32)))
    assert b.observations["first_person"].shape == (src.total_steps + 1, 24, 32, 3)
    # the new dataset rebuilds an env with its own observation space, and keeps the episode metadata
    assert out.recover_environment().observation_space == out.observation_space
    (m0,), (m1,) = src.storage.get_episode_metadata([0]), out.storage.get_episode_metadata([0])
    assert m1 == m0 and m1["success"]
    assert out.storage.metadata["algorithm_name"] == "scripted"
    assert out.storage.metadata["platform"] == src.storage.metadata["platform"]


def test_to_video_writes_an_mp4_at_the_env_step_rate(tmp_path):
    av = pytest.importorskip("av")
    src = play("reach_goal", "first_goal", frame_skip=20)  # 30 env steps per second
    to_video(src, 0, "follow", tmp_path / "ep0.mp4", view_size=(48, 64))
    with av.open(str(tmp_path / "ep0.mp4")) as f:
        stream = f.streams.video[0]
        frames = list(f.decode(stream))
        assert stream.average_rate == 30
    assert len(frames) == src.total_steps + 1  # the reset frame and one per step
    assert (frames[0].height, frames[0].width) == (48, 64)
