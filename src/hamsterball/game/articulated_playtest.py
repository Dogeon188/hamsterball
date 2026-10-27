"""Play-test viewer for the articulated tier's skeleton: a Core on a flat level, no env, no task.

Run: `uv run --extra game python -m hamsterball.game.articulated_playtest [--skeleton NAME] [--slope DEG | --level hill]`.
Needs the `game` extra. PROTOTYPE: `--skeleton` picks `best`, `fast` or `sprint` from proto_candidates.json (see proto_skeletons.py), or `base`.
"""

import json
import math
import time
from pathlib import Path

import click
import mujoco
import numpy as np

from ..core import Core
from ..core.articulated_tier import N_JOINTS
from ..core.model import HAM, HAM_DOF, HAM_QPOS, SHELL
from . import proto_skeletons as P

PERIOD = 1 / 60
SHOVE = 2.5  # m/s of the space-bar shove, sideways with a roll
CANDIDATES = Path(__file__).with_name("proto_candidates.json")
SLOPE_KEY = 20.0  # degrees the K key tilts the world to (uphill is +y, the start facing)


def candidates():
    return json.loads(CANDIDATES.read_text()) if CANDIDATES.exists() else {}


def load(name):
    """(tier, gait factory) of a named candidate; 'base' is the first-cut skeleton with the ticket-04 trot."""
    if name == "base":
        skel, gait, rest = {}, {}, None
    else:
        c = candidates()[name]
        skel, gait, rest = c["skel"], c["gait"], c.get("rest")
    tier = P.make_tier(skel)
    tier.resting_radius = rest if rest else P.measure_rest(tier) + 0.01
    return tier, lambda: P.Gait(tier, **gait)


def shove(core):
    """Throw the hamster sideways and roll it over, so it topples against the wall."""
    d = core.data
    d.qvel[HAM_DOF + 1] += SHOVE
    d.qvel[HAM_DOF + 3] += 6.0


def run(name, slope, level):
    from imgui_bundle import imgui  # before glfw, as in window.py

    import glfw
    from .window import _close, _end_frame, _hud, _new_frame, _open
    from ..core import views as V
    from ..core.level import Level

    tier, make_gait = load(name)
    gait = make_gait()
    lvl = Level.load(Path(__file__).parents[1] / "levels" / f"{level}.yaml") if level else None
    core = Core(tier, level=lvl, camera_action=False) if lvl else Core(tier, camera_action=False)
    m, d = core.model, core.data
    window = _open()
    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)
    scene = mujoco.MjvScene(m, maxgeom=10000)
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 1.6, 90.0, -15.0
    opt = V.option("chase")
    on = limp = False
    deg = slope
    held = lambda *keys: any(imgui.is_key_down(k) for k in keys)
    pressed = lambda k: imgui.is_key_pressed(k, False)
    last = time.perf_counter()
    P.tilt(core, deg)

    def reset():
        core.reset()
        P.tilt(core, deg)
        gait.reset()

    try:
        while not glfw.window_should_close(window):
            _new_frame()
            if pressed(imgui.Key.escape):
                break
            if pressed(imgui.Key.r):
                reset()
                on = limp = False
            on ^= pressed(imgui.Key.g)
            limp ^= pressed(imgui.Key.f)
            if pressed(imgui.Key.k):
                deg = 0.0 if deg else SLOPE_KEY
                P.tilt(core, deg)
            if pressed(imgui.Key.space):
                shove(core)
            cam.azimuth += 2.0 * (held(imgui.Key.e) - held(imgui.Key.q))
            cam.distance = min(max(cam.distance + 0.03 * (held(imgui.Key.x) - held(imgui.Key.z)), 0.8), 5.0)
            fw = held(imgui.Key.w, imgui.Key.up_arrow) - held(imgui.Key.s, imgui.Key.down_arrow)
            rt = held(imgui.Key.d, imgui.Key.right_arrow) - held(imgui.Key.a, imgui.Key.left_arrow)
            az = math.radians(cam.azimuth)  # camera forward on the ground is (cos az, sin az); right is (sin az, -cos az)
            run_v = np.array([fw * math.cos(az) + rt * math.sin(az), fw * math.sin(az) - rt * math.cos(az)])
            if np.hypot(*run_v) > 1:
                run_v /= np.hypot(*run_v)
            obs = core.tier.obs(m, d, core.facing)
            if limp:
                act = np.zeros(N_JOINTS, np.float32)
            else:
                act = gait.act(obs["proprio"], run_v if on else (0.0, 0.0))
            core.step(act)

            cam.lookat[:] = d.xpos[SHELL]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            w, h = glfw.get_framebuffer_size(window)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), scene, ctx)
            from OpenGL import GL
            GL.glActiveTexture(GL.GL_TEXTURE0)  # see window.py: mjr_render leaves unit 1 active
            off = d.subtree_com[HAM] - d.xpos[SHELL]
            _hud([
                f"articulated tier play-test: skeleton {name}",
                "G gait on/off  WASD/arrows run vector (camera-relative)  K 20 deg slope on/off  F limp  Space shove  R reset  Q/E orbit  Z/X zoom  Esc quit",
                f"mode: {'LIMP' if limp else 'GAIT' if on else 'stand'}   slope {deg:.0f} deg (uphill = start facing)   shell speed {np.hypot(*d.qvel[:2]):.2f} m/s",
                f"hamster COM {np.linalg.norm(off):.3f} m from shell centre (resting radius {core.tier.resting_radius:.3f})   run {run_v.round(2)}",
            ])
            _end_frame(window)
            now = time.perf_counter()
            time.sleep(max(0.0, PERIOD - (now - last)))
            last = time.perf_counter()
    finally:
        ctx.free()
        _close(window)


@click.command()
@click.option("--skeleton", default="base", help="a candidate name from proto_candidates.json, or base")
@click.option("--slope", default=0.0, help="start with the world tilted this many degrees (uphill is +y)")
@click.option("--level", default=None, help="a bundled level name, e.g. hill")
def main(skeleton, slope, level):
    """Articulated tier play-test: G switches on a gait that follows the WASD run vector; K tilts the world to 20 degrees."""
    run(skeleton, slope, level)


if __name__ == "__main__":
    main()
