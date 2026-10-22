"""Human baselines (spec §13): per (level, task) and (task, difficulty) group statistics over Minari datasets."""

import gymnasium as gym
import pytest
from click.testing import CliRunner

pytest.importorskip("minari")

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.baseline import Stats, baselines, main
from hamsterball.data import Collector


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


def synthetic(dataset_id, episodes, **kw):
    """A dataset of one-step episodes whose metadata then says `episodes`: (options, success, length, return)."""
    env = Collector(gym.make("HamsterballReachGoal-v1", levels=["first_goal"], **kw))
    for i in range(len(episodes)):
        env.reset(seed=i)
        env.step(env.action_space.sample())
    ds = env.create_dataset(dataset_id, author="tester")
    ds.storage.update_episode_metadata(
        [{"options": o, "success": s, "length": n, "return": r} for o, s, n, r in episodes], range(len(episodes))
    )
    return dataset_id


def file_level(task, level_id="ab" * 32):
    return {"level": "version: 1\n", "level_id": level_id, "task": task}


def generated(task, difficulty):
    return {"level": "version: 1\n", "level_id": "cd" * 32, "task": task, "difficulty": difficulty}


@pytest.fixture
def two_datasets():
    # frame_skip 10 is 1/60 s per env step; frame_skip 5 with camera_action is 1/120 s, another action space
    a = synthetic("test/human-a-v0", [
        (file_level("reach_goal"), True, 120, 1.0),  # 2 s
        (file_level("reach_goal"), True, 60, 0.5),  # 1 s
        (file_level("reach_goal"), False, 30, -1.0),
        (file_level("survive"), False, 90, -1.0),
        (generated("waypoint_race", 0.34), True, 600, 3.0),  # 10 s
        (generated("waypoint_race", 0.26), False, 300, 0.0),
        (generated("waypoint_race", 0.36), True, 300, 2.0),  # 5 s
    ])
    b = synthetic("test/human-b-v0", [(file_level("reach_goal"), True, 360, 1.0)], frame_skip=5, camera_action=True)
    return [a, b]


def test_groups_merge_across_datasets_by_level_and_task_or_task_and_difficulty(two_datasets):
    got = baselines(two_datasets)
    assert got[("reach_goal", "ab" * 32)] == Stats(4, 0.75, 2.0, 1.0, pytest.approx(0.375))  # 3 s from b
    assert got[("survive", "ab" * 32)] == Stats(1, 0.0, None, None, -1.0)
    assert got[("waypoint_race", 0.3)] == Stats(2, 0.5, 10.0, 10.0, 1.5)
    assert got[("waypoint_race", 0.4)] == Stats(1, 1.0, 5.0, 5.0, 2.0)
    assert len(got) == 4


def test_the_cli_prints_a_row_per_group(two_datasets):
    result = CliRunner().invoke(main, two_datasets)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert len(lines) == 5  # a header and four groups
    row = next(line for line in lines if "reach_goal" in line).split()
    assert row == ["reach_goal", "abababababab", "4", "75%", "2.00", "1.00", "0.38"]
    assert next(line for line in lines if "d=0.3" in line).split()[-3:] == ["10.00", "10.00", "1.50"]
    assert next(line for line in lines if "survive" in line).split()[-3:] == ["-", "-", "-1.00"]


def test_without_minari_the_cli_says_which_extra_to_install(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "minari", None)  # import minari now raises ImportError
    result = CliRunner().invoke(main, ["hamsterball/human-ana-v0"])
    assert result.exit_code == 1 and "install hamsterball[data]" in result.output
