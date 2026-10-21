"""The game's recording session (spec §13): finished episodes append to the player's dataset, abandoned runs don't."""

import numpy as np
import pytest

minari = pytest.importorskip("minari")

from hamsterball.data import replay  # noqa: E402
from hamsterball.core.generator import ProceduralGenerator  # noqa: E402
from hamsterball.env import LEVELS_DIR  # noqa: E402
from hamsterball.game.session import Session  # noqa: E402

SURVIVE = str(LEVELS_DIR / "first_survive.yaml")
OUT = np.array([0, 1, 0], np.float32)  # run forward out of first_survive's region: a failure in about a second


@pytest.fixture(autouse=True)
def datasets_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "datasets"))


def play_out(session, overruns=0):
    """Step until the episode ends; return its length."""
    n = 1
    while not any(session.step(OUT, overruns)[2:4]):
        n += 1
    return n


def test_each_finished_episode_appends_to_the_players_dataset_with_its_overrun_count():
    s = Session("survive", SURVIVE, player="eric")
    assert s.dataset_id == "hamsterball/human-eric-v0"
    for overruns in (3, 0):
        s.start()
        play_out(s, overruns)
    ds = minari.load_dataset("hamsterball/human-eric-v0")
    first, second = ds.storage.get_episode_metadata([0, 1])
    assert ds.total_episodes == 2
    assert (first["overruns"], second["overruns"]) == (3, 0)
    assert first["failure_reason"] == second["failure_reason"] == "left_region"
    replay(ds)
    s.close()


def test_the_dataset_replays_without_the_level_files_and_across_tasks(tmp_path):
    level = tmp_path / "mine.yaml"
    level.write_text((LEVELS_DIR / "first_survive.yaml").read_text())
    s = Session("survive", str(level), player="eric")
    s.start()
    play_out(s)
    s.close()
    g = Session("reach_goal", "generated", difficulty=0.2, seed=4, player="eric")
    g.start()
    for _ in range(5):
        g.step(OUT)
    g.start(retry=True)  # Retry abandons the run; the next episode is the same level
    play_out(g)
    g.close()
    level.unlink()
    ds = minari.load_dataset("hamsterball/human-eric-v0")
    assert "levels" not in ds.env_spec.kwargs  # a level path only works on the recording machine
    options = [m["options"] for m in ds.storage.get_episode_metadata([0, 1])]
    assert [o["task"] for o in options] == ["survive", "reach_goal"]
    assert options[1]["difficulty"] == 0.2
    assert options[1]["level"] == ProceduralGenerator()("reach_goal", 4, 0.2).canonical()  # the seed, retried
    assert ds.total_episodes == 2  # not the abandoned run
    replay(ds)


def test_a_crash_loses_only_the_run_in_progress():
    s = Session("survive", SURVIVE, player="eric")
    s.start()
    n = play_out(s)
    s.start()
    for _ in range(5):
        s.step(OUT)
    del s  # the process dies here: no close, no flush
    ds = minari.load_dataset("hamsterball/human-eric-v0")
    assert (ds.total_episodes, ds.total_steps) == (1, n)


def test_quitting_mid_episode_records_nothing_and_no_record_never_writes():
    s = Session("survive", SURVIVE, player="eric")
    s.start()
    s.step(OUT)
    s.close()  # Menu, Quit, or closing the window
    off = Session("survive", SURVIVE, record=False)  # --no-record needs no player name
    assert not off.recording
    off.start()
    play_out(off)
    off.close()
    assert minari.list_local_datasets() == {}


def test_recording_needs_a_valid_player_name():
    with pytest.raises(ValueError, match="a-z0-9_"):
        Session("survive", SURVIVE, player="Eric")
