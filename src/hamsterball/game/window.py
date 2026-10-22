"""The game window (spec §13): one GLFW window, `mjr_render` draws the chase view, Dear ImGui draws the HUD on top.

A thin shell over play.py, menu.py, and session.py: it draws the menu, reads keys and sticks, steps the session once
per frame, and draws. Needs the `game` extra.
"""

import ctypes
import time

import click

from imgui_bundle import imgui  # before glfw: imgui_bundle points pyglfw at the GLFW library it ships

import glfw  # noqa: E402
import mujoco  # noqa: E402
from OpenGL import GL  # noqa: E402

from .. import ENV_VERSION  # noqa: E402
from ..core import views as V  # noqa: E402
from ..env import LEVELS_DIR, TASKS  # noqa: E402
from . import menu  # noqa: E402
from .play import Flow, Mouse, Pacer, action, hud, outcome, sim_time  # noqa: E402
from .session import Session  # noqa: E402

PERIOD = 1 / 60  # one env step (frame_skip 10 × 1/600 s) per frame, real time
KEYS = {
    "w": [imgui.Key.w],
    "a": [imgui.Key.a],
    "s": [imgui.Key.s],
    "d": [imgui.Key.d],
    "q": [imgui.Key.q],
    "e": [imgui.Key.e],
    "shift": [imgui.Key.left_shift, imgui.Key.right_shift],
    "ctrl": [imgui.Key.left_ctrl, imgui.Key.right_ctrl],
}
PRESSES = {"esc": [imgui.Key.escape], "enter": [imgui.Key.enter, imgui.Key.keypad_enter], "r": [imgui.Key.r]}


def _sticks():
    """((left x, left y), right x) from the first connected gamepad, or zeros."""
    for j in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
        if glfw.joystick_is_gamepad(j) and (state := glfw.get_gamepad_state(j)):
            ax = state.axes
            return (ax[glfw.GAMEPAD_AXIS_LEFT_X], ax[glfw.GAMEPAD_AXIS_LEFT_Y]), ax[glfw.GAMEPAD_AXIS_RIGHT_X]
    return (0.0, 0.0), 0.0


class _Chase:
    """The chase view drawn into the window's framebuffer, with shadows on; rebuilt when the level's model changes."""

    def __init__(self, model):
        self.model = model
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = model.camera("chase").id
        self.opt = V.option("chase")

    def draw(self, data, width, height):
        mujoco.mj_kinematics(self.model, data)  # the env posed the view mocaps; carry them to the camera
        mujoco.mj_camlight(self.model, data)
        mujoco.mjv_updateScene(self.model, data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self.scene, self.ctx)
        # mjr_render leaves texture unit 1 active; imgui would bind its font there with no texcoords, drawing each
        # glyph as a dim solid block
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def free(self):
        self.ctx.free()


def _hud(lines):
    imgui.set_next_window_pos((10, 10))
    flags = imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_inputs
    imgui.begin("hud", None, flags)
    for line in lines:
        if line == "REC":
            imgui.text_colored((1, 0.2, 0.2, 1), line)
        else:
            imgui.text(line)
    imgui.end()


def _centered(name):
    vp = imgui.get_main_viewport()
    imgui.set_next_window_pos(vp.get_center(), imgui.Cond_.always, (0.5, 0.5))
    flags = imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move
    imgui.begin(name, None, flags)


def _open():
    """The GLFW window with imgui on its GL context."""
    if not glfw.init():
        raise RuntimeError("GLFW failed to initialise")
    window = glfw.create_window(1280, 720, "Hamsterball", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW could not open a window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # the Pacer holds 60 Hz; vsync on top of it would beat against the schedule
    imgui.create_context()
    imgui.get_io().set_ini_filename("")  # windows are placed each frame; no imgui.ini in the working directory
    imgui.backends.glfw_init_for_opengl(ctypes.cast(window, ctypes.c_void_p).value, True)
    imgui.backends.opengl2_init()  # fixed-function, so it shares MuJoCo's compatibility context
    return window


def _close(window):
    imgui.backends.opengl2_shutdown()
    imgui.backends.glfw_shutdown()
    imgui.destroy_context()
    glfw.destroy_window(window)
    glfw.terminate()


def _new_frame():
    glfw.poll_events()
    imgui.backends.opengl2_new_frame()
    imgui.backends.glfw_new_frame()
    imgui.new_frame()


def _end_frame(window):
    imgui.render()
    imgui.backends.opengl2_render_draw_data(imgui.get_draw_data())
    glfw.swap_buffers(window)


def _menu(window, settings, levels_dir, record):
    """The menu screen: returns the chosen settings on Play, or None on Quit or closing the window."""
    s = dict(settings)
    choices = {t: menu.level_choices(levels_dir, t) for t in TASKS}
    seed_text = "" if s["seed"] is None else str(s["seed"])
    while not glfw.window_should_close(window):
        _new_frame()
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(vp.pos)
        imgui.set_next_window_size(vp.size)
        imgui.begin("menu", None, imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move)
        imgui.text("Hamsterball")
        imgui.separator()
        _, s["player"] = imgui.input_text("player", s["player"])
        _, t = imgui.combo("task", TASKS.index(s["task"]) if s["task"] in TASKS else 0, TASKS)
        s["task"] = TASKS[t]
        levels = choices[s["task"]]
        if s["level"] not in levels:
            s["level"] = levels[0]
        _, i = imgui.combo("level", levels.index(s["level"]), levels)
        s["level"] = levels[i]
        problems = []
        if s["level"] == menu.GENERATED:
            _, s["difficulty"] = imgui.slider_float("difficulty", s["difficulty"], 0.0, 1.0, "%.2f")
            _, seed_text = imgui.input_text("seed (blank: random)", seed_text)
            try:
                s["seed"] = menu.parse_seed(seed_text)
            except ValueError:
                problems.append("The seed is a whole number, 0 or more, or blank for a random one.")
        if record and not menu.name_ok(s["player"]):
            problems.append("Player names use a-z, 0-9, and _ only.")
        imgui.text(f"recording to hamsterball/human-{s['player']}-v{ENV_VERSION}" if record else "not recording")
        for p in problems:
            imgui.text_colored((1, 0.3, 0.3, 1), p)
        imgui.begin_disabled(bool(problems))
        play = imgui.button("Play")
        imgui.end_disabled()
        imgui.same_line()
        quit_ = imgui.button("Quit")
        imgui.end()
        _end_frame(window)
        if quit_:
            return None
        if play and not problems:
            if s["level"] != menu.GENERATED:
                s["seed"] = None
            return s
        time.sleep(PERIOD)
    return None


def _play(window, session):
    """Play episodes of `session` until Menu, Quit, or closing the window. Returns "menu" or "quit"."""
    chase, flow, pacer, mouse = None, Flow(), Pacer(PERIOD), Mouse()
    act = action(set())
    env = session.env
    cursor_x = None  # None: no reading since the cursor was locked, so the lock's jump isn't a turn

    def start(retry=False):
        session.start(retry)
        pacer.reset()
        pacer.overruns = 0
        mouse.clear()

    start()
    command = None
    try:
        while command not in ("menu", "quit"):
            _new_frame()
            presses = [n for n, ks in PRESSES.items() if any(imgui.is_key_pressed(k, False) for k in ks)]
            if glfw.window_should_close(window):
                presses = ["close"]
            # the cursor is locked and hidden only while playing, so the pause and result buttons stay clickable
            playing = flow.state == "play"
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED if playing else glfw.CURSOR_NORMAL)
            if playing:
                held = {n for n, ks in KEYS.items() if any(imgui.is_key_down(k) for k in ks)}
                left, right_x = _sticks()
                x = glfw.get_cursor_pos(window)[0]
                cam = mouse.move(0.0 if cursor_x is None else x - cursor_x)
                cursor_x = x
                act = action(held, left, right_x, cam)
                pacer.wait()
                _, _, terminated, truncated, info = session.step(act, pacer.overruns)
                if text := outcome(terminated, truncated, info):
                    flow.end(text)
            else:
                pacer.reset()
                mouse.clear()
                cursor_x = None
                time.sleep(PERIOD)

            core = env.unwrapped.core
            if chase is None or chase.model is not core.model:
                if chase:
                    chase.free()
                chase = _Chase(core.model)
            chase.draw(core.data, *glfw.get_framebuffer_size(window))

            _hud(hud(env, act, session.recording))
            if flow.state == "paused":
                _centered("paused")
                imgui.text("Paused")
                presses += [b for b in ("resume", "retry", "menu", "quit") if imgui.button(b.capitalize())]
                imgui.end()
            elif flow.state == "result":
                _centered("result")
                imgui.text(flow.outcome)
                imgui.text(f"time {sim_time(env)[0]:.2f} s")
                imgui.text(f"overruns {pacer.overruns}")
                imgui.text(f"saved to {session.dataset_id}" if session.recording else "not recorded")
                imgui.text("Enter: next   R: retry   Esc: menu")
                imgui.end()
            _end_frame(window)

            for p in presses:
                command = flow.press(p)
                if command in ("retry", "next"):
                    start(retry=command == "retry")  # a retry abandons a run in progress
                elif command in ("menu", "quit"):
                    break
    finally:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)  # the menu needs it
        session.close()  # abandons a run in progress
        if chase:
            chase.free()
    return command


def run(settings, levels_dir, record, skip_menu):
    """The menu, then episodes with its choices, then the menu again, until Quit or closing the window."""
    window = _open()
    try:
        while True:
            if not skip_menu:
                settings = _menu(window, settings, levels_dir, record)
                if settings is None:
                    break
            skip_menu = False
            menu.save_config(settings, menu.CONFIG)
            level = menu.resolve(settings["level"], levels_dir)
            session = Session(settings["task"], level, settings["difficulty"], settings["seed"], settings["player"],
                              record)
            if _play(window, session) == "quit":
                break
    finally:
        _close(window)


@click.command()
@click.option("--levels", "levels_dir", type=click.Path(exists=True, file_okay=False), default=str(LEVELS_DIR),
              help="The directory of level YAMLs the menu lists.")
@click.option("--player", help="The player's name, [a-z0-9_]+; recording goes to hamsterball/human-<player>-vN.")
@click.option("--no-record", is_flag=True, help="Practise: record nothing.")
@click.option("--task", type=click.Choice(TASKS), help="The task; with --level, skips the menu.")
@click.option("--level", help="A level YAML, a level in --levels by name, or 'generated'.")
@click.option("--seed", type=click.IntRange(min=0), help="A generated level's seed (default random).")
@click.option("--difficulty", type=click.FloatRange(0, 1), help="A generated level's difficulty (default 0.5).")
def main(levels_dir, player, no_record, task, level, seed, difficulty):
    """Play Hamsterball: the chase view at 60 Hz, keyboard (WASD, Shift, Ctrl, Q/E), mouse to turn, or gamepad, Esc to pause."""
    flags = {"player": player, "task": task, "level": level, "difficulty": difficulty, "seed": seed}
    settings = menu.prefill(flags, menu.load_config(menu.CONFIG))
    record = not no_record
    skip = task is not None and level is not None and (not record or menu.name_ok(settings["player"]))
    run(settings, levels_dir, record, skip)
