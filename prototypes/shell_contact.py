# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "glfw", "numpy"]
# ///
"""PROTOTYPE, throwaway: compare shell colliders, tile counts, and contact softness under model C.

Answers the "Shell wall contact quality" ticket. Not production code. Model C and its constants are
copied from hamster_feel.py at 2307efc (the chosen feel); only the shell's contacts vary.

Run:   uv run prototypes/shell_contact.py
Sweep: uv run prototypes/shell_contact.py --sweep

Keys: O outer collider, N tile count, K softness, M margin, WASD run (camera-relative),
Shift full effort, Q/E orbit camera, R reset, Esc quit. Gamepad: left stick.
"""

import itertools
import math
import sys
import time

import glfw
import mujoco
import numpy as np

R, T = 0.5, 0.03
M_SHELL, M_HAM = 1.0, 2.0
R_HAM = 0.12
L_ARM = R - T - R_HAM
TIMESTEP = 0.002
LEAN_MAX = math.radians(70)
C_KP, C_KD, C_F_MAX = 200.0, 28.0, 30.0
C_TOUCH, C_GRACE = 0.003, 0.045
KEY_EFFORT, SHIFT_EFFORT = 0.5, 1.0
DEADZONE = 0.15

# Variant axes. The first value of each is the 2307efc baseline.
OUTERS = ["tiles", "sphere", "sphere+roll"]  # what meets the world
TILE_COUNTS = [180, 80, 320]
SOFTS = {"default": "0.02 1", "stiff": "0.004 1", "soft": "0.04 2"}  # solref on the hamster-tile pair
MARGINS = [0.0, 0.002]  # hamster geom margin (m): contact acts before touching
ROLL = 0.04  # rolling friction (m) for sphere+roll; ~0.7 m/s^2 coast loss on paper, the tiles' facet drag


def shell_wall(outer, n_tiles, soft):
    side = math.sqrt(4 * math.pi * R * R / n_tiles) * 1.25
    golden = math.pi * (3 - math.sqrt(5))
    # tiles on bit 2 touch only the hamster when a sphere meets the world on bit 1
    bits = "" if outer == "tiles" else ' contype="2" conaffinity="2"'
    geoms = []
    for i in range(n_tiles):
        z = 1 - 2 * (i + 0.5) / n_tiles
        r = math.sqrt(1 - z * z)
        n = np.array([r * math.cos(i * golden), r * math.sin(i * golden), z])
        p = (R - T / 2) * n
        geoms.append(
            f'<geom type="box" size="{side/2:.4f} {side/2:.4f} {T/2}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'zaxis="{n[0]:.4f} {n[1]:.4f} {n[2]:.4f}" mass="{M_SHELL/n_tiles:.5f}" solref="{SOFTS[soft]}"{bits} rgba=".6 .8 1 .25"/>'
        )
    if outer == "sphere":
        geoms.append(f'<geom type="sphere" size="{R}" mass="0" rgba="0 0 0 0"/>')
    elif outer == "sphere+roll":  # friction mixes by max, so the plane's tiny rolling term loses
        geoms.append(f'<geom type="sphere" size="{R}" mass="0" condim="6" friction="1 0.005 {ROLL}" rgba="0 0 0 0"/>')
    return "\n".join(geoms)


def course():
    g = ['<geom type="plane" size="40 40 .1" rgba=".35 .4 .35 1"/>']
    for x, deg in [(-4, 10), (0, 20), (4, 30)]:
        a = math.radians(deg)
        g.append(f'<geom type="box" size="1.5 3 .1" pos="{x} {6 + 3*math.cos(a):.3f} {3*math.sin(a):.3f}" '
                 f'euler="{deg} 0 0" rgba=".7 .6 .4 1"/>')
    for k in range(8):
        g.append(f'<geom type="capsule" fromto="-3 {-3 - .7*k} 0 3 {-3 - .7*k} 0" size=".08" rgba=".5 .5 .5 1"/>')
    return "\n".join(g)


def build(v):
    outer, n_tiles, soft, margin = v
    bits = "" if outer == "tiles" else ' contype="2" conaffinity="2"'
    ham = (f'<body name="ham" pos="0 0 {T + R_HAM + .01}"><freejoint/>'
           f'<geom type="sphere" size="{R_HAM}" mass="{M_HAM}" solref="{SOFTS[soft]}" margin="{margin}"{bits} rgba="1 .6 .2 1"/></body>')
    xml = f"""<mujoco><option timestep="{TIMESTEP}" integrator="implicitfast"/><size nuserdata="1"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<worldbody><light pos="0 0 10" dir="0 0 -1" directional="true"/>
{course()}
<body name="shell" pos="0 0 {R}"><freejoint/>
{shell_wall(outer, n_tiles, soft)}
</body>
{ham}
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def body_vel(m, d, b):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
    return res[3:]


def wall_gap(m, d, n_tiles):
    """Hamster surface to the nearest inner tile face (m), negative when pressed in."""
    s, h = m.body("shell").id, m.body("ham").id
    a = m.body_geomadr[s]
    normals = m.geom_pos[a:a + n_tiles] / (R - T / 2)  # tiles come first in the shell body
    r = d.xmat[s].reshape(3, 3).T @ (d.xipos[h] - d.xpos[s])
    return R - T - R_HAM - (normals @ r).max()


def control(m, d, n_tiles, v, e):
    """Model C from 2307efc. v: unit run direction in world xy, e: effort in [0,1]."""
    s, h = m.body("shell").id, m.body("ham").id
    d.xfrc_applied[:] = 0
    if wall_gap(m, d, n_tiles) <= C_TOUCH:
        d.userdata[0] = d.time
    if d.time - d.userdata[0] > C_GRACE:
        return
    c, hp = d.xpos[s], d.xipos[h]
    down = np.array([0, 0, -1.0])
    n = (hp - c) / (np.linalg.norm(hp - c) + 1e-9)
    target = math.sin(e * LEAN_MAX) * v + math.cos(e * LEAN_MAX) * down
    rel_v = body_vel(m, d, h) - body_vel(m, d, s)
    f = C_KP * L_ARM * (target - n) - C_KD * rel_v
    f -= np.dot(f, n) * n
    norm = np.linalg.norm(f)
    if norm > C_F_MAX:
        f *= C_F_MAX / norm
    p = c + n * (R - T)
    d.xfrc_applied[h, :3] = f
    d.xfrc_applied[s, :3] = -f
    d.xfrc_applied[s, 3:] = np.cross(p - d.xipos[s], -f)


def penetration(d):
    return max([0.0] + [-c.dist for c in d.contact[: d.ncon]])


def broken(m, d):
    return not np.isfinite(d.qpos).all() or np.linalg.norm(d.xipos[m.body("ham").id] - d.xpos[m.body("shell").id]) > R


def measure(v):
    """Full effort +x on flat ground for 6 s, then a 3 s coast; separately, a 3 s climb up a 20 deg tilt."""
    m, d = build(v)
    n_tiles = v[1]
    fwd = np.array([1.0, 0, 0])
    rattle, zs, pen, step_t, steps = 0, [], 0.0, 0.0, 0
    while d.time < 6.0:
        control(m, d, n_tiles, fwd, 1.0)
        t = time.perf_counter()
        mujoco.mj_step(m, d)
        step_t += time.perf_counter() - t
        steps += 1
        if broken(m, d):
            return dict(broken=True)
        if d.time > 2.0:  # at speed
            rattle += wall_gap(m, d, n_tiles) > v[3] + C_TOUCH  # margin holds the hamster that far off
            zs.append(d.qpos[2])
            pen = max(pen, penetration(d))
    v_run = np.linalg.norm(d.qvel[:2])
    while d.time < 9.0:
        control(m, d, n_tiles, fwd, 0.0)
        mujoco.mj_step(m, d)
        if abs(d.time - 7.0) < TIMESTEP / 2:  # the hamster swings back to the bottom in the first second and pushes the shell on
            v7 = np.linalg.norm(d.qvel[:2])
    coast = (v7 - np.linalg.norm(d.qvel[:2])) / 2.0

    m, d = build(v)
    a = math.radians(20)
    m.opt.gravity[:] = [-9.81 * math.sin(a), 0, -9.81 * math.cos(a)]  # uphill is +x
    while d.time < 3.0:
        control(m, d, n_tiles, fwd, 1.0)
        mujoco.mj_step(m, d)
    return dict(broken=False, v6=v_run, rattle=rattle / len(zs), jitter=np.std(zs) * 1000, pen=pen * 1000,
                coast=coast, climb=d.qpos[0], sps=steps / step_t)


def sweep():
    print(f"{'outer':12s} {'tiles':>5s} {'soft':8s} {'margin':>6s} | {'v@6s':>5s} {'rattle':>6s} {'z jit':>6s} {'pen':>5s} {'coast':>6s} {'climb':>6s} {'mj_step/s':>9s}")
    print(f"{'':12s} {'':>5s} {'':8s} {'mm':>6s} | {'m/s':>5s} {'>3mm':>6s} {'mm':>6s} {'mm':>5s} {'m/s2':>6s} {'m@3s':>6s} {'':>9s}")
    for v in itertools.product(OUTERS, TILE_COUNTS, SOFTS, MARGINS):
        r = measure(v)
        head = f"{v[0]:12s} {v[1]:5d} {v[2]:8s} {v[3]*1000:6.0f} |"
        if r["broken"]:
            print(head, "BROKEN (NaN or hamster escaped)")
            continue
        print(head, f"{r['v6']:5.1f} {r['rattle']*100:5.0f}% {r['jitter']:6.1f} {r['pen']:5.1f} {r['coast']:6.2f} {r['climb']:6.2f} {r['sps']:9.0f}")


def read_input(win, fwd):
    right = np.array([fwd[1], -fwd[0], 0.0])
    fwd3 = np.array([fwd[0], fwd[1], 0.0])
    if glfw.joystick_is_gamepad(glfw.JOYSTICK_1):
        st = glfw.get_gamepad_state(glfw.JOYSTICK_1)
        x, y = st.axes[0], -st.axes[1]
        mag = min(1.0, math.hypot(x, y))
        if mag > DEADZONE:
            v = x * right + y * fwd3
            return v / np.linalg.norm(v), (mag - DEADZONE) / (1 - DEADZONE)
    k = lambda key: glfw.get_key(win, key) == glfw.PRESS
    v = (k(glfw.KEY_W) - k(glfw.KEY_S)) * fwd3 + (k(glfw.KEY_D) - k(glfw.KEY_A)) * right
    if np.linalg.norm(v) < 1e-6:
        return np.zeros(3), 0.0
    shift = k(glfw.KEY_LEFT_SHIFT) or k(glfw.KEY_RIGHT_SHIFT)
    return v / np.linalg.norm(v), SHIFT_EFFORT if shift else KEY_EFFORT


def play():
    glfw.init()
    win = glfw.create_window(1280, 720, "PROTOTYPE shell contact", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    axes = [OUTERS, TILE_COUNTS, list(SOFTS), MARGINS]
    idx = [0, 0, 0, 0]
    pending = {"reset": False}

    def on_key(_w, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        cycle = {glfw.KEY_O: 0, glfw.KEY_N: 1, glfw.KEY_K: 2, glfw.KEY_M: 3}
        if key in cycle:
            i = cycle[key]
            idx[i] = (idx[i] + 1) % len(axes[i])
            pending["reset"] = True
        elif key == glfw.KEY_R:
            pending["reset"] = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    glfw.set_key_callback(win, on_key)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    cam.distance, cam.elevation, cam.azimuth = 4.0, -20.0, 90.0

    def load():
        v = tuple(ax[i] for ax, i in zip(axes, idx))
        m, d = build(v)
        return v, m, d, mujoco.MjvScene(m, maxgeom=2000), mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)

    v_, m, d, scn, ctx = load()
    t0, rate, gaps = time.perf_counter(), 1.0, []
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if pending["reset"]:
            pending["reset"] = False
            ctx.free()
            v_, m, d, scn, ctx = load()
            t0, gaps = time.perf_counter(), []

        s = m.body("shell").id
        a = math.radians(cam.azimuth)
        v, e = read_input(win, (math.cos(a), math.sin(a)))

        vel = d.qvel[:2]
        if np.linalg.norm(vel) > 0.8:
            err = (math.degrees(math.atan2(vel[1], vel[0])) - cam.azimuth + 180) % 360 - 180
            cam.azimuth += 0.03 * err
        cam.azimuth += 2.0 * ((glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_E) == glfw.PRESS))
        cam.lookat[:] = d.xpos[s]

        wall = time.perf_counter() - t0
        steps = 0
        while d.time < wall and steps < 50:
            control(m, d, v_[1], v, e)
            mujoco.mj_step(m, d)
            gaps.append(wall_gap(m, d, v_[1]) > v_[3] + C_TOUCH)
            steps += 1
        gaps = gaps[-500:]  # last second
        if wall > 0.5:
            rate = 0.9 * rate + 0.1 * (d.time / wall)

        left = "\n".join(["outer (O)", "tiles (N)", "softness (K)", "margin (M)", "", "speed", "rattle, last 1 s", "penetration", "sim/real"])
        right = "\n".join([v_[0], str(v_[1]), f"{v_[2]} ({SOFTS[v_[2]]})", f"{v_[3]*1000:.0f} mm", "",
                           f"{np.linalg.norm(d.qvel[:3]):.2f} m/s", f"{100*np.mean(gaps) if gaps else 0:.0f}% >3 mm past margin",
                           f"{penetration(d)*1000:.1f} mm", f"{rate:.2f}" + ("  BROKEN" if broken(m, d) else "")])

        w, hgt = glfw.get_framebuffer_size(win)
        vp = mujoco.MjrRect(0, 0, w, hgt)
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp, left, right, ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
                           "O outer  N tiles  K softness  M margin  WASD run  Shift effort  Q/E orbit  R reset  Esc quit", "", ctx)
        glfw.swap_buffers(win)
    ctx.free()
    glfw.terminate()


if __name__ == "__main__":
    sweep() if "--sweep" in sys.argv else play()
