"""hamsterball-data (later efforts §7): each subcommand wraps its data function and turns its errors into exit 1."""

import pytest
from click.testing import CliRunner

minari = pytest.importorskip("minari")

from hamsterball.datacli import main
from test_data import record


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


def run(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


def test_replay_passes_a_clean_dataset_and_counts_its_episodes():
    record(episodes=2, steps=5, levels=["first_goal"]).create_dataset("test/clean-v0", algorithm_name="random")
    result = run("replay", "test/clean-v0")
    assert result.exit_code == 0 and "2 episodes replay bit for bit" in result.output
    assert "1 episodes" in run("replay", "test/clean-v0", "--episode", 1).output


def test_replay_exits_1_with_the_step_that_diverges(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr("hamsterball.core.step.DRAG_K", 0.06)  # record under other physics
        record(steps=5, levels=["first_goal"]).create_dataset("test/diverge-v0", algorithm_name="random")
    result = run("replay", "test/diverge-v0")
    assert result.exit_code == 1 and "episode 0 diverges at step 1" in result.output


def test_a_size_that_is_not_hxw_is_a_usage_error():
    result = run("rerender", "test/any-v0", "test/out-v0", "--view", "chase", "--size", "84")
    assert result.exit_code == 2 and "is not HxW" in result.output


@pytest.mark.gl
def test_rerender_then_video_then_lerobot_export(tmp_path):
    record(steps=5, levels=["first_goal"]).create_dataset("test/src-v0", algorithm_name="random")
    result = run("rerender", "test/src-v0", "test/px-v0", "--view", "chase", "--view", "follow", "--size", "24x32")
    assert result.exit_code == 0, result.output
    assert {"chase", "follow"} <= set(minari.load_dataset("test/px-v0").observation_space.keys())

    assert run("video", "test/px-v0", 0, "follow", tmp_path / "ep0.mp4").exit_code == 0
    assert (tmp_path / "ep0.mp4").stat().st_size > 0

    # the raw recording has no views: to_lerobot's ValueError, or the missing extra, both exit 1
    assert run("lerobot", "test/src-v0", "hamsterball/test", tmp_path / "lr").exit_code == 1
    pytest.importorskip("lerobot.datasets.lerobot_dataset")  # the lerobot extra, Python 3.12+
    result = run("lerobot", "test/px-v0", "hamsterball/test", tmp_path / "lr", "--images")
    assert result.exit_code == 0, result.output


def test_without_minari_the_cli_says_which_extra_to_install(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "minari", None)  # import minari now raises ImportError
    result = run("replay", "test/any-v0")
    assert result.exit_code == 1 and "install hamsterball[data]" in result.output
