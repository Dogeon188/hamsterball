"""The dataset pose stream and `hamsterball-data poses` (pose-stream ticket 03)."""

import json

import gymnasium as gym
import numpy as np
import pytest

minari = pytest.importorskip("minari")

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level
from hamsterball.data import Collector, pose_stream
from hamsterball.datacli import main
from click.testing import CliRunner
from test_motion import run, track
from test_moving_hazards import dodge, sweepers_env
from test_platforms import raft_env
from test_seesaw import seesaw_env
from test_tasks import toward_target


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


class Live:
    """A Collector that also collects `pose()` after reset and each step."""

    def __init__(self, env):
        self.env, self.poses = Collector(env), []

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.poses.append(self.env.unwrapped.pose())
        return out

    def step(self, action):
        out = self.env.step(action)
        self.poses.append(self.env.unwrapped.pose())
        return out

    def __getattr__(self, name):
        return getattr(self.env, name)


def until_done(env, action):
    env.reset(seed=0)
    while not any(env.step(np.array(action, np.float32))[2:4]):
        pass


def scripted(env):
    obs, _ = env.reset(seed=0)
    for _ in range(5000):
        obs, _, terminated, truncated, _ = env.step(toward_target(obs))
        if terminated or truncated:
            break


def survive(env, controller):
    assert run(env, controller)["success"]


CASES = {
    "locked_goal": (lambda: gym.make("Hamsterball-v2", tasks=["unlock_goal"], levels=["locked_goal"]), scripted),
    "waypoint_loop": (lambda: gym.make("Hamsterball-v2", tasks=["waypoint_race"], levels=["waypoint_loop"]), scripted),
    "survive_raft": (raft_env, lambda e: survive(e, track)),
    "survive_seesaw": (seesaw_env, lambda e: survive(e, track)),
    "survive_collapse": (
        lambda: gym.make("HamsterballSurvive-v2", levels=["survive_collapse"]),
        lambda e: until_done(e, [0.3, 0.6]),
    ),
    "survive_sweepers": (sweepers_env, lambda e: survive(e, dodge)),
}


def no_physics(*a, **kw):
    raise AssertionError("the pose stream stepped physics")


@pytest.mark.parametrize("name", CASES)
def test_the_dataset_stream_equals_the_live_poses_without_stepping_physics(name, monkeypatch):
    make, drive = CASES[name]
    env = Live(make())
    drive(env)
    ds = env.create_dataset(f"test/{name}-v0", algorithm_name="scripted")
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.physics_step", no_physics)
        header, *poses = pose_stream(ds, 0)
    assert poses == env.poses
    assert header["pose_format"] == 1 and header["task"] in ("unlock_goal", "waypoint_race", "survive")
    assert Level.from_yaml(header["level"]).id == header["level_id"]
    assert header["rate"] == pytest.approx(60) and header["step_limit"] > 0
    assert "outcome" in poses[-1]


def test_the_cli_writes_one_header_line_then_one_pose_per_line(tmp_path):
    env = Live(gym.make("Hamsterball-v2", tasks=["reach_goal"], levels=["first_goal"]))
    env.reset(seed=0)
    for _ in range(7):
        env.step(np.array([0, 1], np.float32))
    ds = env.create_dataset("test/cli-v0", algorithm_name="scripted")
    path = tmp_path / "poses.jsonl"
    result = CliRunner().invoke(main, ["poses", "test/cli-v0", "0", str(path)])
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == list(pose_stream(ds, 0)) and len(lines) == 1 + 8
