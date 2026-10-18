# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "glfw", "numpy"]
# ///
"""PROTOTYPE, throwaway: compare how abstract-hamster models feel to play.

Answers the "Abstract hamster model" ticket. Not production code.

Run:   uv run prototypes/hamster_feel.py
Check: uv run prototypes/hamster_feel.py --smoke

Starts on model C with the speed cascade off.
Keys: 1/2/3 switch model (A direct torque, B pendulum, C contact), WASD run
(camera-relative), Shift full effort, T toggle speed cascade, Q/E orbit camera,
R reset, Esc quit.
Gamepad: left stick sets direction and effort.
"""

import math
import sys
import time

import glfw
import mujoco
import numpy as np

# Tunables. Barycentric slope limit: sin(a) = m*l*sin(lean max) / ((M+m)*R) = (m/(M+m)) * (l/R) * sin(lean max):
# ~19 deg for C (m/M 1.5, its gravity feed-forward holds the full 52 deg lean), ~26 for B (m/M 3, 58 deg). C still climbs 20 deg:
# its foot-force pair acts R_HAM apart (hamster centre vs wall), a small extra drive couple.
# Real reference: Syrian hamster 110-140 g in a Kaytee 7 in ball of 222 g gives m/M ~0.56 (limit ~11 deg here);
# dwarfs (20-50 g) and the 11.5 in ball (907 g) give far less.
R, T = 0.5, 0.03          # shell radius (game scale, a real ball is 0.09-0.15 m), wall thickness
N_TILES = 180             # boxes in the compound shell wall; facet impacts act as rolling resistance (~0.6 m/s^2 coasting)
M_SHELL, M_HAM = 1.2, 1.8   # A and C: m/M 1.5, ~2.7x the real ratio; C climbs 20 deg from rest at any shell orientation from 1.33 (1.27 sticks)
B_M_SHELL, B_M_HAM = 0.75, 2.25   # B keeps m/M 3: at 1.5 its cascade tops out at 4.7 m/s against the drag
R_HAM = 0.12              # hamster sphere radius: R_HAM/R 0.24, a 125 g hamster (3.1 cm sphere) in a ~10 in ball
L_ARM = R - T - R_HAM     # pendulum arm (B) = contact radius (C)
TIMESTEP = 0.002
B_KP, B_KD, B_TAU_MAX = 40.0, 4.0, 12.0   # B_KD was 3: B whirled at e=0.5 with the 2.25 kg hamster
B_LEAN_MAX = math.radians(58)   # lean at full effort; past ~60 B whirls around the shell centre (KV_ALONG 0.65 is near its edge)
C_KP, C_KD, C_F_MAX = 200.0, 32.0, 60.0   # foot-force lean PD and cap (cap stops binding above ~90); C_KD was 28 at 2 kg
C_LEAN_MAX = math.radians(52)   # C's lean at full effort (drive torque ~ sin(lean)); whirls at 90 with KV_ALONG >= 0.8
A_TAU_MAX = 6.0           # direct shell torque at full effort
KEY_EFFORT, SHIFT_EFFORT = 0.5, 1.0
V_MAX = 6.0              # speed cascade: input sets target velocity, lean tracks it
K_DRAG = 0.5              # N/(m/s)^2 horizontal drag on the shell while it touches the world
V_RUN = 50.0              # C's running-speed fade: its lean (drive) falls linearly to 0 as speed along the input nears V_RUN;
                          # K_DRAG alone holds C near 6.3 m/s, so this only trims it to ~6 (K_DRAG 0.1 + V_RUN 8 lets the fade set the cap)
KV_ALONG, KV_LAT = 0.65, 0.25  # gains on velocity error along / across the held direction; KV_LAT >= 0.5 shakes C over bumps
DEADZONE = 0.15
RAMPS = [(-4, 10), (0, 20), (4, 30)]  # (x, slope deg): 6 m ramps up +y from y=6, top edges are ledges

NAMES = {"A": "A direct torque on shell", "B": "B lean-controlled pendulum", "C": "C contact body + foot force"}


def shell_wall(mass):
    side = math.sqrt(4 * math.pi * R * R / N_TILES) * 1.25
    golden = math.pi * (3 - math.sqrt(5))
    geoms = []
    for i in range(N_TILES):
        z = 1 - 2 * (i + 0.5) / N_TILES
        r = math.sqrt(1 - z * z)
        n = np.array([r * math.cos(i * golden), r * math.sin(i * golden), z])
        p = (R - T / 2) * n
        geoms.append(
            f'<geom type="box" size="{side/2:.4f} {side/2:.4f} {T/2}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'zaxis="{n[0]:.4f} {n[1]:.4f} {n[2]:.4f}" mass="{mass/N_TILES:.5f}" rgba=".6 .8 1 .25"/>'
        )
    return "\n".join(geoms)


def course():
    g = ['<geom type="plane" size="40 40 .1" rgba=".35 .4 .35 1"/>']
    for x, deg in RAMPS:
        a = math.radians(deg)
        g.append(f'<geom type="box" size="1.5 3 .1" pos="{x} {6 + 3*math.cos(a):.3f} {3*math.sin(a):.3f}" '
                 f'euler="{deg} 0 0" rgba=".7 .6 .4 1"/>')
    for k in range(8):  # bump field
        g.append(f'<geom type="capsule" fromto="-3 {-3 - .7*k} 0 3 {-3 - .7*k} 0" size=".08" rgba=".5 .5 .5 1"/>')
    return "\n".join(g)


def masses(kind):
    return (B_M_SHELL, B_M_HAM) if kind == "B" else (M_SHELL, M_HAM)


def build(kind):
    m_shell, m_ham = masses(kind)
    if kind == "C":
        ham = (f'<body name="ham" pos="0 0 {T + R_HAM + .01}"><freejoint/>'
               f'<geom type="sphere" size="{R_HAM}" mass="{m_ham}" rgba="1 .6 .2 1"/></body>')
        inner = ""
    else:  # A and B: pendulum on a ball joint at the shell centre, no contacts
        ham = ""
        inner = (f'<body name="ham"><joint type="ball" damping=".001"/>'
                 f'<geom type="sphere" size="{R_HAM}" pos="0 0 {-L_ARM}" mass="{m_ham}" contype="0" conaffinity="0" rgba="1 .6 .2 1"/>'
                 f'<geom type="capsule" fromto="0 0 0 0 0 {-L_ARM}" size=".01" mass=".001" contype="0" conaffinity="0" rgba="1 .6 .2 1"/></body>')
    xml = f"""<mujoco><option timestep="{TIMESTEP}" integrator="implicitfast"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<worldbody><light pos="0 0 10" dir="0 0 -1" directional="true"/>
{course()}
<body name="shell" pos="0 0 {R}"><freejoint/>
{shell_wall(m_shell)}
{inner}
</body>
{ham}
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def body_vel(m, d, b):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
    return res[3:]


def grounded(m, d, s):
    return any(0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) and s in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
               for c in d.contact[: d.ncon])


def control(m, d, kind, v, e, v_run=V_RUN):
    """v: unit run direction in world xy (3-vector, z=0), e: effort in [0,1]. Also applies the ground drag."""
    s, h = m.body("shell").id, m.body("ham").id
    d.xfrc_applied[:] = 0
    if grounded(m, d, s):
        d.xfrc_applied[s, :2] = -K_DRAG * d.qvel[:2] * np.linalg.norm(d.qvel[:2])
    c, hp = d.xpos[s], d.xipos[h]
    down = np.array([0, 0, -1.0])
    w_h = d.cvel[h, :3]
    if kind == "C":
        # foot force tracks the same lean target as B; without this PD the hamster sloshes
        n = (hp - c) / (np.linalg.norm(hp - c) + 1e-9)
        e *= min(1.0, max(0.0, 1 - np.dot(d.qvel[:2], v[:2]) / v_run))  # can only run so fast; braking and coasting keep full strength
        target = math.sin(e * C_LEAN_MAX) * v + math.cos(e * C_LEAN_MAX) * down
        rel_v = body_vel(m, d, h) - body_vel(m, d, s)
        f = C_KP * L_ARM * (target - n) - C_KD * rel_v
        f -= M_HAM * 9.81 * (down - np.dot(down, target) * target)  # gravity feed-forward: hold the hamster at the target lean
        f -= np.dot(f, n) * n  # tangential only: the wall carries the normal load
        norm = np.linalg.norm(f)
        if norm > C_F_MAX:
            f *= C_F_MAX / norm
        p = c + n * (R - T)  # foot on the inner wall
        d.xfrc_applied[h, :3] = f
        d.xfrc_applied[s, :3] -= f
        d.xfrc_applied[s, 3:] = np.cross(p - d.xipos[s], -f)
        return
    u = (hp - c) / (np.linalg.norm(hp - c) + 1e-9)
    target = down if kind == "A" else math.sin(e * B_LEAN_MAX) * v + math.cos(e * B_LEAN_MAX) * down
    tau = B_KP * np.cross(u, target) - B_KD * w_h
    norm = np.linalg.norm(tau)
    if norm > B_TAU_MAX:
        tau *= B_TAU_MAX / norm
    if kind == "A":  # hamster held hanging (external, cosmetic); shell driven directly
        d.xfrc_applied[h, 3:] = tau
        d.xfrc_applied[s, 3:] = A_TAU_MAX * e * np.cross([0, 0, 1.0], v)
    else:  # internal couple: hamster and shell push on each other
        d.xfrc_applied[h, 3:] = tau
        d.xfrc_applied[s, 3:] = -tau


def cascade(d, v, e):
    """Sphero-style outer loop: (direction, effort) is a target velocity; returns the lean command.
    No input coasts (hamster hangs); overspeed along the held direction is not braked, only reversing is."""
    if e <= 0:
        return np.zeros(3), 0.0
    err = v[:2] * e * V_MAX - d.qvel[:2]
    along = np.dot(err, v[:2])
    cmd = KV_ALONG * max(along, 0.0) * v[:2] + KV_LAT * (err - along * v[:2])  # no along-direction brake
    mag = np.linalg.norm(cmd)
    if mag < 1e-6:
        return np.zeros(3), 0.0
    return np.array([cmd[0] / mag, cmd[1] / mag, 0.0]), min(1.0, mag)


def penetration(d):
    return max([0.0] + [-c.dist for c in d.contact[: d.ncon]])


def escaped(m, d):
    return np.linalg.norm(d.xipos[m.body("ham").id] - d.xpos[m.body("shell").id]) > R


def smoke():
    """Headless check: each model rolls on flat ground without whirling; B climbs 20 deg from a run-up, C from rest on the ramp;
    C settles at a drag-limited top speed, coasts and runs on downhill."""
    for kind in "ABC":
        m, d = build(kind)
        v = np.array([1.0, 0, 0])
        worst = 0.0
        while d.time < 3.0:
            control(m, d, kind, v, 1.0)
            mujoco.mj_step(m, d)
            worst = max(worst, penetration(d))
            assert not escaped(m, d), f"{kind}: hamster left the shell"
        speed = d.qvel[0]  # shell free joint is first: vx
        print(f"{NAMES[kind]:30s} x={d.qpos[0]:6.2f} m  vx={speed:5.2f} m/s  max penetration={worst*1000:5.1f} mm")
        assert d.qpos[0] > 0.5, f"{kind}: shell did not roll forward"
        runs = [(e, (-1.0, 0.0), (0.0, 0.0)) for e in (0.5, 1.0)]
        if kind == "C":  # diagonal from x=-4 so the ball misses the bump field
            runs.append((1.0, (-1.0, -1.0), (-4.0, 0.0)))
        for e, dirn, start in runs:  # cascade on flat ground: no whirl, shell spin near the no-slip rate
            spin, speed = whirl(kind, e, dirn, start)
            print(f"{'':30s} whirl e={e} dir={dirn}: worst spin/limit={spin:4.2f}  speed={speed:4.2f} m/s (target {e*V_MAX})")
            assert spin < 1, f"{kind}: hamster whirls, shell spins too fast at e={e} dir={dirn}"
            assert abs(speed - e * V_MAX) < 0.2 * e * V_MAX, f"{kind}: cascade speed off at e={e} dir={dirn}"
        for casc in ([True] if kind == "B" else [False, True] if kind == "C" else []):
            t = climb(kind, s0=2.0 if kind == "C" else None, casc=casc)
            print(f"{'':30s} climbs the 20 deg ramp {'from rest 2 m up' if kind == 'C' else 'from a run-up'}, cascade {'on' if casc else 'off'}: {t:4.2f} s")
            assert t < math.inf, f"{kind}: could not climb the 20 deg ramp"
    v6, v10 = top_speed("C")
    print(f"{'C top speed':30s} cascade off, full effort: {v6:4.2f} m/s at 6 s, {v10:4.2f} at 10 s")
    assert abs(v10 - v6) < 0.1 * v6 and v10 < 8, "C: drag does not settle the top speed"
    assert abs(v10 - 6) < 0.6, "C: cascade-off top speed is not near 6 m/s"
    lean = steady_lean("C")
    print(f"{'C steady lean':30s} cascade off, full effort, no speed fade: {lean:4.1f} deg (C_LEAN_MAX {math.degrees(C_LEAN_MAX):.0f})")
    assert abs(lean - math.degrees(C_LEAN_MAX)) < 3, "C: lean sags off its target"
    v0, v3 = coast("C")
    print(f"{'C coast':30s} released at {v0:4.2f} m/s, 3 s later {v3:4.2f} m/s")
    assert 0.3 < v3 < v0, "C: coast stopped too fast or did not slow"  # was 2 < v3 before the drag
    vb = downhill("C", 0)
    print(f"{'C downhill':30s} full effort down {dict(RAMPS)[0]} deg: {vb:4.2f} m/s at the foot (V_MAX {V_MAX})")
    assert vb > 0.9 * V_MAX, "C: cascade capped downhill speed"  # flat cascade speed is ~5 under drag; was vb > V_MAX
    print(f"{'C bumps':30s} worst spin/limit={bumps('C'):4.2f} (info: hops over the bumps)")
    std, flips = shake("C")
    print(f"{'C bump shake':30s} hamster side offset std={std*100:3.1f} cm, side lean flips={flips} (KV_LAT 0.65: 2.9 cm, 31)")
    assert std < 0.012 and flips < 20, "C: hamster shakes sideways over the bumps"
    tt = turn("C")
    print(f"{'C turn':30s} -x at 5 m/s, then hold +y: heading turns 60 deg in {tt:4.2f} s")
    assert tt < 2, "C: steering too slow"
    print("smoke ok")


def place(m, d, kind, pos):
    """Move the shell centre to pos, hamster keeping its offset (C only has a free hamster)."""
    off = d.qpos[7:10] - d.qpos[0:3]
    d.qpos[0:3] = pos
    if kind == "C":
        d.qpos[7:10] = d.qpos[0:3] + off


def spin_ratio(d):
    """Shell spin over its limit 1.5 |v|/R + 1 rad/s (no-slip spin is |v|/R)."""
    return np.linalg.norm(d.qvel[3:6]) / (1.5 * np.linalg.norm(d.qvel[:2]) / R + 1.0)


def whirl(kind, e, v=(-1.0, 0.0), start=(0.0, 0.0), t_end=8.0, settle=3.0):
    """Cascade toward v on flat ground: worst spin ratio after settling, and final speed."""
    m, d = build(kind)
    place(m, d, kind, (*start, R))
    v, worst = np.array([*v, 0.0]) / np.linalg.norm(v), 0.0
    while d.time < t_end:
        control(m, d, kind, *cascade(d, v, e))
        mujoco.mj_step(m, d)
        if d.time > settle:
            worst = max(worst, spin_ratio(d))
    return worst, np.linalg.norm(d.qvel[:2])


def top_speed(kind, v=(-1.0, 0, 0)):
    """Cascade off, full effort on flat ground: speed at 6 s and at 10 s."""
    m, d = build(kind)
    out = []
    for t in (6.0, 10.0):
        while d.time < t:
            control(m, d, kind, np.array(v), 1.0)
            mujoco.mj_step(m, d)
        out.append(np.linalg.norm(d.qvel[:2]))
    return out


def steady_lean(kind, v=(-1.0, 0, 0), t_end=10.0, settle=6.0):
    """Cascade off, full effort on flat ground with the speed fade off: mean hamster lean from vertical (deg) after settling."""
    m, d = build(kind)
    s, h, lean = m.body("shell").id, m.body("ham").id, []
    while d.time < t_end:
        control(m, d, kind, np.array(v), 1.0, v_run=math.inf)
        mujoco.mj_step(m, d)
        if d.time > settle:
            u = d.xipos[h] - d.xpos[s]
            lean.append(math.degrees(math.acos(-u[2] / np.linalg.norm(u))))
    return float(np.mean(lean))


def on_ramp(x, s0, lift=0.1):
    """Shell centre resting s0 m up the surface of the ramp at x (ramp box top is 0.1 above its slope line)."""
    a = math.radians(dict(RAMPS)[x])
    return x, 6 + s0 * math.cos(a) - (R + lift) * math.sin(a), s0 * math.sin(a) + (R + lift) * math.cos(a)


def coast(kind, t_drive=12.0, t_coast=3.0):
    """Full effort toward -x until 0.8 V_MAX (drag holds the cascade near 5 m/s), then release: (speed at release, speed t_coast later)."""
    m, d = build(kind)
    v = np.array([-1.0, 0, 0])
    while d.time < t_drive and np.linalg.norm(d.qvel[:2]) < 0.8 * V_MAX:
        control(m, d, kind, *cascade(d, v, 1.0))
        mujoco.mj_step(m, d)
    v0, t0 = np.linalg.norm(d.qvel[:2]), d.time
    while d.time < t0 + t_coast:
        control(m, d, kind, *cascade(d, v, 0.0))
        mujoco.mj_step(m, d)
    return v0, np.linalg.norm(d.qvel[:2])


def downhill(kind, x=4, s0=5.5, t_end=6.0):
    """Full effort down the ramp at x (-y) from s0 m up its surface; speed when the shell reaches its foot."""
    m, d = build(kind)
    place(m, d, kind, on_ramp(x, s0))
    v = np.array([0, -1.0, 0])
    while d.time < t_end and d.qpos[1] > 6:
        control(m, d, kind, *cascade(d, v, 1.0))
        mujoco.mj_step(m, d)
    return np.linalg.norm(d.qvel[:3])


def bumps(kind, e=1.0, t_end=4.0):
    """Cascade toward -y across the bump field: worst spin ratio."""
    m, d = build(kind)
    v, worst = np.array([0, -1.0, 0]), 0.0
    while d.time < t_end:
        control(m, d, kind, *cascade(d, v, e))
        mujoco.mj_step(m, d)
        worst = max(worst, spin_ratio(d))
    return worst


def shake(kind="C", y0=0.0, t_end=5.0):
    """Cascade at full effort from (0, y0) toward -y across the bump field, after t > 0.8 s:
    std of the hamster's sideways offset from the shell centre (m), and sign flips of the sideways lean command."""
    m, d = build(kind)
    place(m, d, kind, (0, y0, R))
    s, h = m.body("shell").id, m.body("ham").id
    v, off, flips, last = np.array([0, -1.0, 0]), [], 0, 0
    while d.time < t_end:
        cv, ce = cascade(d, v, 1.0)
        control(m, d, kind, cv, ce)
        mujoco.mj_step(m, d)
        if d.time > 0.8:
            off.append(d.xipos[h, 0] - d.xpos[s, 0])
            sg = np.sign(round(cv[0] * ce, 2))
            flips += bool(sg and last and sg != last)
            last = sg or last
    return float(np.std(off)), flips


def turn(kind="C", v_start=5.0, t_end=4.0):
    """Full effort toward -x until v_start, then hold +y: seconds until the velocity heading has turned 60 deg."""
    m, d = build(kind)
    v = np.array([-1.0, 0, 0])
    while np.linalg.norm(d.qvel[:2]) < v_start and d.time < 12:
        control(m, d, kind, *cascade(d, v, 1.0))
        mujoco.mj_step(m, d)
    v, t0 = np.array([0, 1.0, 0]), d.time
    while d.time < t0 + t_end:
        control(m, d, kind, *cascade(d, v, 1.0))
        mujoco.mj_step(m, d)
        if math.degrees(math.atan2(d.qvel[1], -d.qvel[0])) >= 60:
            return d.time - t0
    return math.inf


def climb(kind, x=0, s0=None, casc=True, t_end=10.0):
    """Full effort up the ramp at x, from rest s0 m up its surface (None: 1.5 m before its foot); seconds until the shell
    passes the top edge, inf if never."""
    m, d = build(kind)
    place(m, d, kind, (x, 4.5, R) if s0 is None else on_ramp(x, s0))
    v = np.array([0, 1.0, 0])
    while d.time < t_end:
        control(m, d, kind, *(cascade(d, v, 1.0) if casc else (v, 1.0)))
        mujoco.mj_step(m, d)
        if d.qpos[1] > 6 + 6 * math.cos(math.radians(dict(RAMPS)[x])):
            return d.time
    return math.inf


def read_input(win, fwd):
    """Returns (unit world direction, effort) from gamepad or keys, camera-relative."""
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
    win = glfw.create_window(1280, 720, "PROTOTYPE hamster feel", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    kind = "C"
    pending = {"kind": None, "reset": False, "cascade": False}

    def on_key(_w, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            pending["kind"] = "ABC"[key - glfw.KEY_1]
        elif key == glfw.KEY_T:
            pending["cascade"] = not pending["cascade"]
        elif key == glfw.KEY_R:
            pending["reset"] = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    glfw.set_key_callback(win, on_key)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    cam.distance, cam.elevation, cam.azimuth = 4.0, -20.0, 90.0

    def load(k):
        m, d = build(k)
        return m, d, mujoco.MjvScene(m, maxgeom=2000), mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)

    m, d, scn, ctx = load(kind)
    t0, sim0, rate = time.perf_counter(), 0.0, 1.0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if pending["kind"] or pending["reset"]:
            kind = pending["kind"] or kind
            pending.update(kind=None, reset=False)
            ctx.free()
            m, d, scn, ctx = load(kind)
            t0, sim0 = time.perf_counter(), 0.0

        s = m.body("shell").id
        a = math.radians(cam.azimuth)
        fwd = (math.cos(a), math.sin(a))
        v, e = read_input(win, fwd)
        raw_e = e

        # chase cam: trail the shell's horizontal velocity; Q/E orbit
        vel = d.qvel[:2]
        if np.linalg.norm(vel) > 0.8:
            err = (math.degrees(math.atan2(vel[1], vel[0])) - cam.azimuth + 180) % 360 - 180
            cam.azimuth += 0.03 * err
        cam.azimuth += 2.0 * ((glfw.get_key(win, glfw.KEY_E) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS))
        cam.lookat[:] = d.xpos[s]

        wall = time.perf_counter() - t0
        steps = 0
        while d.time < wall and steps < 50:  # cap so a slow frame cannot spiral
            cv, ce = cascade(d, v, raw_e) if pending["cascade"] else (v, raw_e)
            control(m, d, kind, cv, ce)
            mujoco.mj_step(m, d)
            steps += 1
        e = ce if steps else raw_e
        if wall > 0.5:
            rate = 0.9 * rate + 0.1 * (d.time / wall)

        h = m.body("ham").id
        u = d.xipos[h] - d.xpos[s]
        lean = math.degrees(math.acos(np.clip(-u[2] / (np.linalg.norm(u) + 1e-9), -1, 1)))
        left = "\n".join(["model", "cascade", "speed", "lean cmd", "hamster lean", "penetration", "sim/real", "", "masses M/m", "R / arm", "B kp/kd/taumax", "KV along/lat"])
        right = "\n".join([NAMES[kind], "on (target speed)" if pending["cascade"] else "off (direct lean/force)", f"{np.linalg.norm(d.qvel[:3]):.2f} m/s", f"{e:.2f}", f"{lean:.0f} deg",
                           f"{penetration(d)*1000:.1f} mm", f"{rate:.2f}" + ("  ESCAPED" if escaped(m, d) else ""), "",
                           "%s / %s kg" % masses(kind), f"{R} / {L_ARM:.2f} m", f"{B_KP}/{B_KD}/{B_TAU_MAX}", f"{KV_ALONG}/{KV_LAT}"])

        w, hgt = glfw.get_framebuffer_size(win)
        vp = mujoco.MjrRect(0, 0, w, hgt)
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp, left, right, ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
                           "1/2/3 model  WASD run  Shift effort  T cascade  Q/E orbit  R reset  Esc quit", "", ctx)
        glfw.swap_buffers(win)
    ctx.free()
    glfw.terminate()


if __name__ == "__main__":
    smoke() if "--smoke" in sys.argv else play()
