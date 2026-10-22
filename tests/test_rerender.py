"""Re-render, video, and LeRobot export (spec §12): pixels from stored core states, with no physics stepped."""

import gymnasium as gym
import numpy as np
import pytest

minari = pytest.importorskip("minari")

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.data import Collector, replay, rerender, to_lerobot, to_video
from test_motion import drift_env, run, track
from test_moving_hazards import dodge, sweepers_env
from test_platforms import raft_env
from test_seesaw import seesaw_env
from test_tasks import toward_target

pytestmark = pytest.mark.gl


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


def play(task="unlock_goal", level="locked_goal", dataset_id="test/played-v0", **kw):
    """Record one episode of the scripted controller to success: keys hide, the goal unlocks, then goes inactive."""
    env = Collector(gym.make("Hamsterball-v2", tasks=[task], levels=[level], **kw))
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


def test_rerender_reproduces_the_camera_pitch():
    env = Collector(gym.make("Hamsterball-v2", tasks=["reach_goal"], levels=["first_goal"], camera_action=True))
    env.reset(seed=0)
    for i in range(45):  # pitch down, then back up past the default
        env.step(np.array([0, 1, 0, 1 if i < 20 else -1], np.float32))
    src = env.create_dataset("test/pitch-v0", algorithm_name="scripted")
    out = rerender(src, ("chase", "first_person"), (24, 32), "test/pitch-rerendered-v0")
    sin_pitch = next(src.iterate_episodes()).observations["state"][:, 6]
    assert sin_pitch[20] > sin_pitch[0] > sin_pitch[45]
    b = next(out.iterate_episodes())
    for v in ("chase", "first_person"):
        np.testing.assert_array_equal(b.observations[v], live_frames(src, v, (24, 32)))


def test_rerender_draws_the_tail_where_it_was():
    env = Collector(gym.make("Hamsterball-v2", tasks=["reach_goal"], levels=["first_goal"], camera_action=True))
    env.reset(seed=0)
    for i in range(12):  # turn, so the features swing off their reset pose
        env.step(np.array([0, 1, 1 if i < 6 else 0, 0], np.float32))
    src = env.create_dataset("test/tail-v0", algorithm_name="scripted")
    out = rerender(src, ("chase",), (48, 64), "test/tail-rerendered-v0")  # the features cover a few of these pixels
    frames = next(out.iterate_episodes()).observations["chase"]
    np.testing.assert_array_equal(frames, live_frames(src, "chase", (48, 64)))

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


def test_to_lerobot_exports_a_rerendered_dataset(tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")  # the lerobot extra, Python 3.12+
    src = play("reach_goal", "first_goal")
    ds = rerender(src, ("chase", "follow"), (24, 32), "test/rerendered-v0")
    to_lerobot(ds, "hamsterball/test", tmp_path / "lerobot", use_videos=False)  # images: lossless, comparable
    out = lerobot_dataset.LeRobotDataset("hamsterball/test", root=tmp_path / "lerobot")
    ep = next(ds.iterate_episodes())
    n = len(ep.actions)  # LeRobot frames pair an observation with the action taken from it: the last one has none
    assert (out.num_episodes, out.num_frames, out.fps) == (1, n, 60)
    last = out[n - 1]
    assert last["task"] == "reach_goal"
    np.testing.assert_array_equal(last["action"].numpy(), ep.actions[-1])
    np.testing.assert_array_equal(last["observation.state"].numpy(), ep.observations["state"][n - 1])
    np.testing.assert_array_equal(last["observation.goal"].numpy(), ep.observations["goal"][n - 1])
    img = (last["observation.images.chase"].numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)
    np.testing.assert_array_equal(img, ep.observations["chase"][n - 1])
    assert last["next.reward"].item() == pytest.approx(ep.rewards[-1]) and last["next.done"].item()


def test_a_survive_drift_episode_replays_and_rerenders_bit_for_bit(monkeypatch):
    env = Collector(drift_env())
    assert run(env, track)["success"]
    src = env.create_dataset("test/drift-v0", algorithm_name="scripted")
    replay(src)
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase",), (24, 32), "test/drift-rerendered-v0")
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    np.testing.assert_array_equal(b.observations["goal"], a.observations["goal"])
    np.testing.assert_array_equal(b.observations["chase"], live_frames(src, "chase", (24, 32)))


def test_a_survive_sweepers_episode_replays_and_rerenders_bit_for_bit(monkeypatch):
    env = Collector(sweepers_env())
    assert run(env, dodge)["success"]
    src = env.create_dataset("test/sweepers-v0", algorithm_name="scripted")
    replay(src)
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase",), (24, 32), "test/sweepers-rerendered-v0")
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    np.testing.assert_array_equal(b.observations["goal"], a.observations["goal"])
    np.testing.assert_array_equal(b.rewards, a.rewards)
    np.testing.assert_array_equal(b.observations["chase"], live_frames(src, "chase", (24, 32)))


def test_a_survive_pushers_episode_replays_and_rerenders_bit_for_bit(monkeypatch):
    env = Collector(gym.make("HamsterballSurvive-v2", levels=["survive_pushers"]))
    env.reset(seed=0)
    while not any(env.step(np.array([0.3, 0.2], np.float32))[2:4]):
        pass
    src = env.create_dataset("test/pushers-v0", algorithm_name="scripted")
    replay(src)
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase",), (24, 32), "test/pushers-rerendered-v0")
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    np.testing.assert_array_equal(b.observations["state"], a.observations["state"])
    np.testing.assert_array_equal(b.observations["chase"], live_frames(src, "chase", (24, 32)))


def test_a_survive_raft_episode_replays_and_rerenders_bit_for_bit(monkeypatch):
    env = Collector(raft_env())
    assert run(env, track)["success"]
    src = env.create_dataset("test/raft-v0", algorithm_name="scripted")
    replay(src)
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase",), (24, 32), "test/raft-rerendered-v0")
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    np.testing.assert_array_equal(b.observations["goal"], a.observations["goal"])
    np.testing.assert_array_equal(b.rewards, a.rewards)
    np.testing.assert_array_equal(b.observations["chase"], live_frames(src, "chase", (24, 32)))


def test_a_survive_collapse_episode_replays_and_rerenders_bit_for_bit(monkeypatch):
    env = Collector(gym.make("HamsterballSurvive-v2", levels=["survive_collapse"]))
    env.reset(seed=0)
    while not any(env.step(np.array([0.3, 0.6], np.float32))[2:4]):  # rolls over several tiles, then off the floor
        pass
    src = env.create_dataset("test/collapse-v0", algorithm_name="scripted")
    replay(src)
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase",), (24, 32), "test/collapse-rerendered-v0")
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    np.testing.assert_array_equal(b.observations["state"], a.observations["state"])
    np.testing.assert_array_equal(b.observations["chase"], live_frames(src, "chase", (24, 32)))


def test_a_survive_seesaw_episode_replays_and_rerenders_bit_for_bit(monkeypatch):
    env = Collector(seesaw_env())
    assert run(env, track)["success"]
    src = env.create_dataset("test/seesaw-v0", algorithm_name="scripted")
    replay(src)
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        out = rerender(src, ("chase",), (24, 32), "test/seesaw-rerendered-v0")
    a, b = next(src.iterate_episodes()), next(out.iterate_episodes())
    np.testing.assert_array_equal(b.observations["goal"], a.observations["goal"])
    np.testing.assert_array_equal(b.rewards, a.rewards)
    np.testing.assert_array_equal(b.observations["chase"], live_frames(src, "chase", (24, 32)))
