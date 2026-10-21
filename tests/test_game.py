"""The game's headless logic (spec §13): input mapping, pacing, the pause/result flow, and the HUD text."""

import math

import numpy as np
import pytest

from hamsterball.game.play import action


def test_keys_run_in_the_camera_frame_at_half_effort():
    np.testing.assert_allclose(action({"w"}), [0, 0.5, 0])
    np.testing.assert_allclose(action({"d"}), [0.5, 0, 0])
    h = 0.5 / math.sqrt(2)
    np.testing.assert_allclose(action({"w", "a"}), [-h, h, 0])
    np.testing.assert_allclose(action({"w", "s"}), [0, 0, 0])
    np.testing.assert_allclose(action(set()), [0, 0, 0])


def test_shift_walks_ctrl_sprints_both_walk():
    np.testing.assert_allclose(action({"s", "shift"}), [0, -0.25, 0])
    np.testing.assert_allclose(action({"s", "ctrl"}), [0, -1.0, 0])
    np.testing.assert_allclose(action({"s", "shift", "ctrl"}), [0, -0.25, 0])


def test_q_turns_left_e_turns_right():
    # +cam_yaw raises the yaw, turning camera forward from +y toward -x: left
    assert action({"q"})[2] == 1
    assert action({"e"})[2] == -1
    assert action({"q", "e"})[2] == 0


def test_left_stick_deadzone_rescales_effort_and_overrides_keys():
    # GLFW sticks: y is down, so pushing up is y = -1
    np.testing.assert_allclose(action({"w"}, left=(0.1, -0.1)), [0, 0.5, 0])  # inside the deadzone: keys win
    np.testing.assert_allclose(action({"w"}, left=(0.0, 1.0)), [0, -1.0, 0])
    np.testing.assert_allclose(action(set(), left=(0.575, 0.0)), [0.5, 0, 0])  # (0.575 - 0.15) / 0.85 = 0.5
    np.testing.assert_allclose(action(set(), left=(2.0, 0.0)), [1.0, 0, 0])  # capped at effort 1


def test_right_stick_x_sets_cam_yaw_and_overrides_q_e():
    assert action({"q"}, right_x=0.1)[2] == 1  # inside the deadzone
    assert action({"q"}, right_x=1.0)[2] == pytest.approx(-1.0)  # right stick right turns right
    assert action(set(), right_x=-0.575)[2] == pytest.approx(0.5)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_pacer_holds_real_time_and_counts_overruns_without_catching_up():
    from hamsterball.game.play import Pacer

    c = FakeClock()
    p = Pacer(1 / 60, clock=c, sleep=c.sleep)
    p.tick()  # the first frame starts now
    starts = [c.t]
    for work in [0.005, 0.005, 0.05, 0.005, 0.005]:  # the third frame's work takes 3 periods
        c.t += work
        p.tick()
        starts.append(c.t)
    gaps = np.diff(starts)
    np.testing.assert_allclose(gaps[[0, 1, 3, 4]], 1 / 60)
    assert gaps[2] == pytest.approx(0.05)  # slowed down: the next frame starts when the late one ends, no burst
    assert p.overruns == 1


def test_pacer_tolerates_a_late_wake_and_reset_forgets_a_pause():
    from hamsterball.game.play import Pacer

    c = FakeClock()
    p = Pacer(1 / 60, clock=c, sleep=c.sleep)
    p.tick()
    c.t += 1 / 60 + 0.001  # within the slack
    p.tick()
    assert p.overruns == 0
    c.t += 5.0  # paused
    p.reset()
    p.tick()
    c.t += 0.005
    p.tick()
    assert p.overruns == 0


def test_esc_pauses_and_resumes_and_the_pause_menu_leaves_or_restarts():
    from hamsterball.game.play import Flow

    f = Flow()
    assert f.state == "play"
    assert f.press("esc") is None and f.state == "paused"
    assert f.press("enter") is None and f.state == "paused"  # the result keys do nothing while paused
    assert f.press("esc") is None and f.state == "play"
    f.press("esc")
    assert f.press("resume") is None and f.state == "play"
    for button in ("retry", "menu", "quit"):
        f = Flow()
        f.press("esc")
        assert f.press(button) == button
    assert f.press("retry") is None  # a button outside its screen does nothing


def test_result_card_keys_start_the_next_or_same_episode_or_go_to_the_menu():
    from hamsterball.game.play import Flow

    for key, command in (("enter", "next"), ("r", "retry"), ("esc", "menu")):
        f = Flow()
        assert f.press("r") is None  # R does nothing during play
        f.end("success")
        assert f.state == "result" and f.outcome == "success"
        assert f.press(key) == command
    f = Flow()
    f.end("success")
    f.press("enter")
    assert f.state == "play" and f.outcome is None


def test_closing_the_window_quits_from_any_screen():
    from hamsterball.game.play import Flow

    f = Flow()
    assert f.press("close") == "quit"


@pytest.mark.parametrize(
    "level, task, progress",
    [
        ("first_goal", "reach_goal", "reach the goal"),
        ("waypoint_loop", "waypoint_race", "waypoint 1/4"),
        ("item_scatter", "collect_items", "items 0/3"),
        ("locked_goal", "unlock_goal", "keys 0/2"),
        ("first_survive", "survive", "survive: 10.0 s left"),
    ],
)
def test_hud_at_reset_shows_time_progress_recording_effort_speed(level, task, progress):
    from hamsterball.env import HamsterballEnv
    from hamsterball.game.play import hud

    env = HamsterballEnv([task], levels=[level], camera_action=True)
    env.reset(options={"task": task})
    limit = {"first_goal": 30, "first_survive": 10}.get(level, 45)
    assert hud(env, np.zeros(3), recording=False) == [
        f"0.0 / {limit:.1f} s",
        progress,
        "not recording",
        "effort 0.00",
        "speed 0.0 m/s",
    ]


def test_hud_tracks_sim_time_effort_and_the_unlock_goal_once_the_keys_are_in():
    from hamsterball.env import HamsterballEnv
    from hamsterball.game.play import hud

    env = HamsterballEnv(["unlock_goal"], levels=["locked_goal"], camera_action=True)
    env.reset()
    a = np.array([0, 1.0, 0])
    for _ in range(60):
        env.step(a)
    lines = hud(env, a, recording=True)
    assert lines[0] == "1.0 / 45.0 s" and lines[2] == "REC" and lines[3] == "effort 1.00"
    assert float(lines[4].split()[1]) > 0.5  # one second of sprinting
    env.unwrapped._todo[:] = [("goal", None)]  # every key in
    assert hud(env, a, recording=True)[1] == "keys 2/2: reach the goal"


@pytest.mark.parametrize(
    "terminated, truncated, info, text",
    [
        (True, False, {"success": True, "failure_reason": None}, "Success"),
        (True, False, {"success": False, "failure_reason": "kill_plane"}, "Failed: fell off"),
        (True, False, {"success": False, "failure_reason": "hazard"}, "Failed: hazard"),
        (True, False, {"success": False, "failure_reason": "left_region"}, "Failed: left the region"),
        (False, True, {"success": False, "failure_reason": None}, "Time up"),
        (False, False, {"success": False, "failure_reason": None}, None),
    ],
)
def test_outcome_names_how_the_episode_ended(terminated, truncated, info, text):
    from hamsterball.game.play import outcome

    assert outcome(terminated, truncated, info) == text


def test_the_window_module_imports_and_the_cli_prints_help():
    pytest.importorskip("imgui_bundle")
    from click.testing import CliRunner

    from hamsterball.game.window import main

    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    for flag in ("--levels", "--player", "--no-record", "--task", "--level", "--seed", "--difficulty"):
        assert flag in r.output


def test_the_cli_skips_the_menu_given_a_task_and_level_and_prefills_it_otherwise(tmp_path, monkeypatch):
    pytest.importorskip("imgui_bundle")
    from click.testing import CliRunner

    from hamsterball.game import menu, window

    monkeypatch.setattr(menu, "CONFIG", tmp_path / "game.json")
    menu.save_config({"player": "eric", "task": "survive", "level": "first_survive", "difficulty": 0.5}, menu.CONFIG)
    calls = []
    monkeypatch.setattr(window, "run", lambda *a: calls.append(a))

    def start(*args):
        assert CliRunner().invoke(window.main, list(args)).exit_code == 0
        return calls.pop()

    settings, levels, record, skip = start("--task", "reach_goal", "--level", "generated", "--seed", "3")
    assert skip and record and str(levels) == str(window.LEVELS_DIR)
    assert settings == {"player": "eric", "task": "reach_goal", "level": "generated", "difficulty": 0.5, "seed": 3}
    settings, _, _, skip = start("--difficulty", "0.8")
    assert not skip and settings["task"] == "survive" and settings["difficulty"] == 0.8
    _, _, record, skip = start("--task", "survive", "--level", "first_survive", "--player", "Eric")
    assert not skip  # recording needs a valid name: the menu asks for one
    _, _, record, skip = start("--task", "survive", "--level", "first_survive", "--player", "Eric", "--no-record")
    assert skip and not record
