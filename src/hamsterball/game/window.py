"""The game window (spec §13): one GLFW window, `mjr_render` draws the chase view, Dear ImGui draws the HUD on top.

A thin shell over play.py: it reads keys and sticks, steps the env once per frame, and draws. Needs the `game` extra.
"""

import ctypes
import time

import click

from imgui_bundle import imgui  # before glfw: imgui_bundle points pyglfw at the GLFW library it ships

import glfw  # noqa: E402
import mujoco  # noqa: E402

from ..core import views as V  # noqa: E402
from ..env import TASKS, HamsterballEnv  # noqa: E402
from .play import Flow, Pacer, action, hud, outcome, sim_time  # noqa: E402

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


def run(task, level, recording=False):
    """Play episodes of `task` on `level` until Quit, Menu, or closing the window. Returns the last command."""
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

    env = HamsterballEnv([task], levels=[level], camera_action=True)
    chase, flow, pacer = None, Flow(), Pacer(PERIOD)
    act = action(set())

    def start(options):
        nonlocal retry
        _, info = env.reset(options=options)
        retry = {"level": info["reset_options"]["level"], "task": task}  # same level and seed
        pacer.reset()
        pacer.overruns = 0

    retry = None
    start({"task": task})
    command = None
    try:
        while command not in ("menu", "quit"):
            glfw.poll_events()
            imgui.backends.opengl2_new_frame()
            imgui.backends.glfw_new_frame()
            imgui.new_frame()

            presses = [n for n, ks in PRESSES.items() if any(imgui.is_key_pressed(k, False) for k in ks)]
            if glfw.window_should_close(window):
                presses = ["close"]
            if flow.state == "play":
                held = {n for n, ks in KEYS.items() if any(imgui.is_key_down(k) for k in ks)}
                left, right_x = _sticks()
                act = action(held, left, right_x)
                pacer.tick()
                _, _, terminated, truncated, info = env.step(act)
                if text := outcome(terminated, truncated, info):
                    flow.end(text)
            else:
                pacer.reset()
                time.sleep(PERIOD)

            core = env.unwrapped.core
            if chase is None or chase.model is not core.model:
                if chase:
                    chase.free()
                chase = _Chase(core.model)
            chase.draw(core.data, *glfw.get_framebuffer_size(window))

            _hud(hud(env, act, recording))
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
                imgui.text("Enter: next   R: retry   Esc: menu")
                imgui.end()

            imgui.render()
            imgui.backends.opengl2_render_draw_data(imgui.get_draw_data())
            glfw.swap_buffers(window)

            for p in presses:
                command = flow.press(p)
                if command == "retry":
                    start(retry)
                elif command == "next":
                    start({"task": task})
                elif command in ("menu", "quit"):
                    break
    finally:
        env.close()
        if chase:
            chase.free()
        imgui.backends.opengl2_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.destroy_window(window)
        glfw.terminate()
    return command


@click.command()
@click.option("--task", required=True, type=click.Choice(TASKS), help="The episode's task.")
@click.option("--level", required=True, help="A level YAML file, or a bundled level's name.")
def main(task, level):
    """Play Hamsterball: the chase view at 60 Hz, keyboard (WASD, Shift, Ctrl, Q/E) or gamepad, Esc to pause."""
    run(task, level)
