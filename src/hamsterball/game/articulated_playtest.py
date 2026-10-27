"""Play-test viewer for the articulated tier's skeleton: a Core on a flat level, no env, no task.

Run: `uv run --extra game python -m hamsterball.game.articulated_playtest`. Needs the `game` extra.
"""

import math
import time

import click
import mujoco
import numpy as np

from ..core import Core
from ..core.articulated_tier import GEAR_HIP, GEAR_KNEE, N_JOINTS, STAND, ArticulatedTier
from ..core.model import HAM, HAM_DOF, HAM_QPOS, SHELL

PERIOD = 1 / 60
KP, KD = 80.0, 1.0  # the posture loop, N m/rad and N m s/rad, in joint torque before the gear
GEAR = np.array([GEAR_HIP, GEAR_KNEE] * 4)
TROT_AMP, TROT_LIFT, TROT_HZ, TROT_BIAS = 1.15, 0.5, 7.5, -0.3  # hip swing and knee lift (rad), stride Hz, hip offset
PITCH_KNEE, PITCH_HIP = 3.5, -0.8  # nose-down feedback: front legs lengthen, rear shorten; front hips swing (rad per unit)
OFFSET = 0.5  # rad the held W/S (all hips) and A/D (all knees) keys move the posture target
SHOVE = 2.5  # m/s of the space-bar shove, sideways with a roll


def torques(data, t, hips=0.0, knees=0.0, trot=False, limp=False):
    """The normalised action: zero when limp, else a PD loop to the standing pose plus the key offsets, plus a trot
    (diagonal legs in antiphase, the knee flexing in the swing half) when `trot`."""
    if limp:
        return np.zeros(N_JOINTS, np.float32)
    q, qd = data.qpos[HAM_QPOS + 7:HAM_QPOS + 7 + N_JOINTS], data.qvel[HAM_DOF + 6:HAM_DOF + 6 + N_JOINTS]
    target = np.array(STAND, float)
    target[0::2] += hips
    target[1::2] += knees
    fz = data.xmat[HAM][6]  # the torso's forward axis height: below 0 is nose down
    if trot:
        target[0::2] += TROT_BIAS
        for leg in range(4):
            p = 2 * math.pi * TROT_HZ * t + (0.0 if leg in (0, 3) else math.pi)
            target[2 * leg] += TROT_AMP * math.sin(p)
            target[2 * leg + 1] -= TROT_LIFT * max(0.0, -math.cos(p))
            front = 1.0 if leg < 2 else -1.0
            target[2 * leg + 1] -= front * PITCH_KNEE * fz
            target[2 * leg] -= front * PITCH_HIP * fz
    return np.clip((KP * (target - q) - KD * qd) / GEAR, -1, 1).astype(np.float32)


def shove(core):
    """Throw the hamster sideways and roll it over, so it topples against the wall."""
    d = core.data
    d.qvel[HAM_DOF + 1] += SHOVE
    d.qvel[HAM_DOF + 3] += 6.0


def run():
    from imgui_bundle import imgui  # before glfw, as in window.py

    import glfw
    from .window import _close, _end_frame, _hud, _new_frame, _open
    from ..core import views as V

    core = Core(ArticulatedTier(), camera_action=False)
    m, d = core.model, core.data
    window = _open()
    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)
    scene = mujoco.MjvScene(m, maxgeom=10000)
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 1.6, 90.0, -15.0
    opt = V.option("chase")
    trot = limp = False
    held = lambda *keys: any(imgui.is_key_down(k) for k in keys)
    pressed = lambda k: imgui.is_key_pressed(k, False)
    last = time.perf_counter()
    try:
        while not glfw.window_should_close(window):
            _new_frame()
            if pressed(imgui.Key.escape):
                break
            if pressed(imgui.Key.r):
                core.reset()
                trot = limp = False
            trot ^= pressed(imgui.Key.t)
            limp ^= pressed(imgui.Key.f)
            if pressed(imgui.Key.space):
                shove(core)
            cam.azimuth += 2.0 * (held(imgui.Key.e) - held(imgui.Key.q))
            cam.distance = min(max(cam.distance + 0.03 * (held(imgui.Key.x) - held(imgui.Key.z)), 0.8), 5.0)
            hips = OFFSET * (held(imgui.Key.w) - held(imgui.Key.s))
            knees = OFFSET * (held(imgui.Key.d) - held(imgui.Key.a))
            core.step(torques(d, d.time, hips, knees, trot, limp))

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
                "articulated tier play-test",
                "W/S hips  A/D knees  T trot (push)  F limp (fall)  Space shove  R reset  Q/E orbit  Z/X zoom  Esc quit",
                f"mode: {'LIMP' if limp else 'trot' if trot else 'stand'}   shell speed {np.hypot(*d.qvel[:2]):.2f} m/s",
                f"hamster COM {np.linalg.norm(off):.3f} m from shell centre (resting radius {core.tier.resting_radius})",
            ])
            _end_frame(window)
            now = time.perf_counter()
            time.sleep(max(0.0, PERIOD - (now - last)))
            last = time.perf_counter()
    finally:
        ctx.free()
        _close(window)


@click.command()
def main():
    """Articulated tier play-test: keys apply posture offsets, a canned trot pushes the shell, and F or Space makes the hamster fall."""
    run()


if __name__ == "__main__":
    main()
