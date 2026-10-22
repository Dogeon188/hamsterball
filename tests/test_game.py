"""The game's headless logic (spec §13): input mapping, pacing, the pause/result flow, and the HUD text."""

import math

import numpy as np
import pytest

from hamsterball.game.play import STICK_PITCH, STICK_TURN, Mouse, action


def test_keys_run_in_the_camera_frame_at_half_effort():
    np.testing.assert_allclose(action({"w"}), [0, 0.5, 0, 0])
    np.testing.assert_allclose(action({"d"}), [0.5, 0, 0, 0])
    h = 0.5 / math.sqrt(2)
    np.testing.assert_allclose(action({"w", "a"}), [-h, h, 0, 0])
    np.testing.assert_allclose(action({"w", "s"}), [0, 0, 0, 0])
    np.testing.assert_allclose(action(set()), [0, 0, 0, 0])


def test_shift_walks_ctrl_sprints_both_walk():
    np.testing.assert_allclose(action({"s", "shift"}), [0, -0.25, 0, 0])
    np.testing.assert_allclose(action({"s", "ctrl"}), [0, -1.0, 0, 0])
    np.testing.assert_allclose(action({"s", "shift", "ctrl"}), [0, -0.25, 0, 0])


def test_q_turns_left_e_turns_right():
    # +turn raises the facing, turning camera forward from +y toward -x: left
    assert action({"q"})[2] == pytest.approx(STICK_TURN)
    assert action({"e"})[2] == pytest.approx(-STICK_TURN)
    assert action({"q", "e"})[2] == 0


def test_left_stick_deadzone_rescales_effort_and_overrides_keys():
    # GLFW sticks: y is down, so pushing up is y = -1
    np.testing.assert_allclose(action({"w"}, left=(0.1, -0.1)), [0, 0.5, 0, 0])  # inside the deadzone: keys win
    np.testing.assert_allclose(action({"w"}, left=(0.0, 1.0)), [0, -1.0, 0, 0])
    np.testing.assert_allclose(action(set(), left=(0.575, 0.0)), [0.5, 0, 0, 0])  # (0.575 - 0.15) / 0.85 = 0.5
    np.testing.assert_allclose(action(set(), left=(2.0, 0.0)), [1.0, 0, 0, 0])  # capped at effort 1


def test_right_stick_x_sets_turn_and_overrides_q_e():
    assert action({"q"}, right=(0.1, 0.0))[2] == pytest.approx(STICK_TURN)  # inside the deadzone
    assert action({"q"}, right=(1.0, 0.0))[2] == pytest.approx(-STICK_TURN)  # right stick right turns right
    assert action(set(), right=(-0.575, 0.0))[2] == pytest.approx(0.5 * STICK_TURN)


def test_right_stick_y_sets_cam_pitch_up_looks_up():
    assert action(set(), right=(0.0, -1.0))[3] == pytest.approx(-STICK_PITCH)  # GLFW y down: pushed up lowers the pitch
    assert action(set(), right=(0.0, 0.575))[3] == pytest.approx(0.5 * STICK_PITCH)
    assert action(set(), right=(0.1, 0.1))[2:] == pytest.approx([0, 0])  # radial deadzone covers both axes
    assert action({"q"}, right=(0.0, 1.0))[2:] == pytest.approx([STICK_TURN, STICK_PITCH])  # y alone leaves Q's turn


def test_q_e_and_the_full_right_stick_turn_120_and_pitch_90_deg_per_s_at_60_hz():
    # turn 1 turns the facing 30 deg in one env step, cam_pitch 1 pitches 15 deg; the game runs 60 env steps a second
    assert action({"q"})[2] * 30 * 60 == pytest.approx(120)
    assert action({"e"})[2] * 30 * 60 == pytest.approx(-120)
    assert action(set(), right=(1.0, 0.0))[2] * 30 * 60 == pytest.approx(-120)
    assert action(set(), right=(0.0, -1.0))[3] * 15 * 60 == pytest.approx(-90)


def test_the_mouse_turns_and_pitches_a_tenth_of_a_degree_per_pixel_and_carries_over_the_cap():
    m = Mouse()
    turn, pitch = m.move(10, 10)
    assert (turn * 30, pitch * 15) == (pytest.approx(-1), pytest.approx(1))  # degrees
    assert [m.move(d, 0)[0] * 30 for d in (400, 0, 0)] == pytest.approx([-30, -10, 0])  # 40 deg: capped at 30 a frame


def test_a_mouse_flick_turns_at_the_max_rate_over_frames_until_spent():
    m = Mouse(sens=0.1)
    assert [m.move(d, 0)[0] for d in (25, 0, 0, 0)] == pytest.approx([-1, -1, -0.5, 0])  # right turns right
    assert m.move(-4, 0)[0] == pytest.approx(0.4)
    m.move(30, 30)
    m.clear()
    assert m.move(0, 0) == (0, 0)


def test_mouse_up_looks_up_and_each_axis_carries_on_its_own():
    m = Mouse(sens=0.1)
    # screen y is down; the pitch is below the horizon, so moving up (dy < 0) lowers it
    assert [m.move(0, dy)[1] for dy in (-15, 0, 0)] == pytest.approx([-1, -0.5, 0])
    assert [m.move(d, 0) for d in (25, 0)] == [pytest.approx((-1, 0)), pytest.approx((-1, 0))]
    assert m.move(0, 25) == pytest.approx((-0.5, 1))  # the turn carry spends on while the pitch's starts


def test_mouse_motion_overrides_only_its_own_stick_axis_and_keys():
    assert action({"q"}, right=(1.0, 0.0), mouse=(0.3, 0.0))[2] == pytest.approx(0.3)
    assert action({"q"}, mouse=(0.0, 0.0))[2] == pytest.approx(STICK_TURN)  # no motion and nothing carried: the keys win
    assert action(set(), right=(1.0, 0.0), mouse=(0.0, 0.0))[2] == pytest.approx(-STICK_TURN)
    assert action(set(), right=(0.0, 1.0), mouse=(0.0, -0.4))[3] == pytest.approx(-0.4)
    assert action({"q"}, right=(0.0, 1.0), mouse=(0.0, -0.4))[2] == pytest.approx(STICK_TURN)  # a pitch flick leaves Q's turn
    assert action(set(), right=(1.0, 1.0), mouse=(0.3, 0.0))[2:] == pytest.approx([0.3, STICK_PITCH])  # and the stick's pitch


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
    p.wait()  # the first frame starts now
    starts = [c.t]
    for work in [0.005, 0.005, 0.05, 0.005, 0.005]:  # the third frame's work takes 3 periods
        c.t += work
        p.wait()
        starts.append(c.t)
    gaps = np.diff(starts)
    np.testing.assert_allclose(gaps[[0, 1, 3, 4]], 1 / 60)
    assert gaps[2] == pytest.approx(0.05)  # slowed down: the next frame starts when the late one ends, no burst
    assert p.overruns == 1


def test_pacer_tolerates_a_late_wake_and_reset_forgets_a_pause():
    from hamsterball.game.play import Pacer

    c = FakeClock()
    p = Pacer(1 / 60, clock=c, sleep=c.sleep)
    p.wait()
    c.t += 1 / 60 + 0.001  # within the slack
    p.wait()
    assert p.overruns == 0
    c.t += 5.0  # paused
    p.reset()
    p.wait()
    c.t += 0.005
    p.wait()
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
    assert hud(env, np.zeros(4), recording=False) == [
        "0.0 / 10.0 s" if task == "survive" else "0.0 s",  # no time limit for a human player, except survive's
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
    a = np.array([0, 1.0, 0, 0])
    for _ in range(60):
        env.step(a)
    lines = hud(env, a, recording=True)
    assert lines[0] == "1.0 s" and lines[2] == "REC" and lines[3] == "effort 1.00"
    assert float(lines[4].split()[1]) > 0.5  # one second of sprinting
    env.unwrapped._todo[:] = [("goal", None)]  # every key in
    assert hud(env, a, recording=True)[1] == "keys 2/2: reach the goal"


def test_waypoint_progress_stops_at_the_last_waypoint_after_a_win():
    from hamsterball.env import HamsterballEnv
    from hamsterball.game.play import progress

    env = HamsterballEnv(["waypoint_race"], levels=["waypoint_loop"], camera_action=True)
    env.reset()
    env.unwrapped._todo.clear()  # every waypoint passed
    assert progress(env) == "waypoint 4/4"


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


def test_the_window_draws_the_kill_plane_into_its_scene_not_the_model():
    import mujoco

    from hamsterball.core import Core
    from hamsterball.core.level import Level
    from hamsterball.game import window

    lv = Level.load(window.LEVELS_DIR / "first_goal.yaml")
    core = Core(level=lv)
    scene = mujoco.MjvScene(core.model, maxgeom=1000)
    mujoco.mjv_updateScene(core.model, core.data, mujoco.MjvOption(), None, mujoco.MjvCamera(),
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    n = scene.ngeom
    window._kill_plane(scene, lv.kill_plane_z)
    assert scene.ngeom == n + 1
    plane = scene.geoms[n]
    assert plane.type == mujoco.mjtGeom.mjGEOM_PLANE and plane.pos[2] == pytest.approx(lv.kill_plane_z)
    assert 0 < plane.rgba[3] < 1
    full = mujoco.MjvScene(core.model, maxgeom=0)
    window._kill_plane(full, lv.kill_plane_z)  # a full scene: no plane, no overrun
    assert full.ngeom == 0


def test_the_window_module_imports_and_the_cli_prints_help():
    pytest.importorskip("imgui_bundle")
    from click.testing import CliRunner

    from hamsterball.game.window import main

    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "mouse" in r.output
    assert "F3" in r.output
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


def test_the_result_card_text_draws_white_over_the_chase_view(monkeypatch):
    """Regression: mjr_render left texture unit 1 active, so imgui drew every glyph as a dim solid block."""
    pytest.importorskip("imgui_bundle")
    from OpenGL import GL

    from hamsterball.game import window

    try:
        win = window._open()
    except RuntimeError as e:
        pytest.skip(f"no window: {e}")
    frames, shot = [], []
    monkeypatch.setattr(window, "outcome", lambda *a: "Failed: hazard")  # the first step ends the episode

    def end_frame(w):
        window.imgui.render()
        window.imgui.backends.opengl2_render_draw_data(window.imgui.get_draw_data())
        frames.append(1)
        if len(frames) == 3:  # the result card is up
            fw, fh = window.glfw.get_framebuffer_size(w)
            px = GL.glReadPixels(0, 0, fw, fh, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            shot.append(np.frombuffer(px, np.uint8).reshape(fh, fw, 3))
            window.glfw.set_window_should_close(w, True)
        window.glfw.swap_buffers(w)

    monkeypatch.setattr(window, "_end_frame", end_frame)
    try:
        window._play(win, window.Session("reach_goal", str(window.LEVELS_DIR / "first_goal.yaml"), record=False))
    finally:
        window._close(win)
    img = shot[0]
    h, w = img.shape[:2]
    card = img[int(h * 0.42):int(h * 0.58), int(w * 0.4):int(w * 0.6)]  # the card is centred on the window
    assert (card.min(axis=-1) > 220).sum() > 500  # white text; the broken draw left nothing above ~150
