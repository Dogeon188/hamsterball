# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "glfw", "numpy"]
# ///
"""PROTOTYPE, throwaway: reproduce model C's feel with an honest (zero net wrench) foot-force pair, and cap its top speed.

Answers the "Honest physics for the chosen feel" ticket. Not production code. Model C is copied from shell_contact.py
with the issue 12 contacts fixed (sphere+roll outer collider, 320 tiles, solref 0.04 2, 2 mm margin).

Run:   uv run prototypes/honest_drive.py
Sweep: uv run prototypes/honest_drive.py --sweep

Keys: P pair, H masses, L lean ceiling, F feed-forward, C speed cap, V top speed, B containment (each reloads), G drop lane, WASD run (camera-relative),
Shift full effort, Q/E orbit camera, R reset, Esc quit. Gamepad: left stick.
"""

import itertools
import math
import sys
import time
from typing import NamedTuple

import glfw
import mujoco
import numpy as np

R, T = 0.5, 0.03
R_HAM = 0.12
L_ARM = R - T - R_HAM
TIMESTEP = 0.002
C_KP, C_KD, C_F_MAX = 200.0, 28.0, 30.0  # per 2 kg of hamster: scaled with its mass so every mass ratio leans alike
K_GRIP = 2.0  # N m s/rad per 2 kg, foot+grip only: damps hamster spin relative to the shell (199be5c; 5 went unstable)
C_TOUCH, C_GRACE = 0.003, 0.045
N_TILES, SOLREF, MARGIN = 320, "0.04 2", 0.002  # issue 12 verdict
ROLL, TORSION = 0.04, 0.05  # outer sphere's rolling and torsional friction (m), issue 12 verdict
FIELD_START, FIELD_HARD = 0.005, 0.02  # m past the resting radius L_ARM where the containment field starts, and where it is fully hard
SOLIMP = "0.9 0.9999 0.02 0.5 2"  # impedance climbs to near-hard over 2 cm of penetration (MuJoCo default 0.9 0.95 0.001)
A_ROLL = 0.55  # m/s^2 coast loss from ROLL, measured in issue 12; calibrates the fade
KEY_EFFORT, SHIFT_EFFORT = 0.5, 1.0
DEADZONE = 0.15

# Variant axes. The first value of each is the issue 12 chosen version.
PAIRS = ["offset", "centre", "foot+grip"]  # offset: +F at the hamster centre, -F at the foot (2307efc, net couple |F| R_HAM)
MASSES = [(2.0, 1.0), (2.4, 0.8), (2.7, 0.6)]  # hamster, shell (kg)
LEANS = [70, 78, 85]  # lean ceiling at full effort (deg)
FFS = [False, True]  # gravity feed-forward (73c1914): holds the commanded lean, which the PD alone sags ~20 deg below
CAPS = ["none", "fade", "drag"]  # fade: effort falls to 0 as speed along the input nears V_RUN. drag: k v^2 on the shell
TOPS = [8.0, 12.0, 16.0]  # target full-effort top speed (m/s) a cap is calibrated to
CONTAINS = ["off", "field", "solimp"]  # keeps the hamster in on hard hits: the soft tile contact lets it sink ~1.5 cm per m/s


class Var(NamedTuple):
    pair: str
    masses: tuple
    lean: int
    ff: bool
    cap: str
    top: float
    contain: str = "off"


PICK = Var("foot+grip", (2.4, 0.8), 70, False, "drag", 16.0, "field")  # play-tested choice; play mode starts here


def shell_wall(m_shell, roll, imp=""):
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
            f'zaxis="{n[0]:.4f} {n[1]:.4f} {n[2]:.4f}" mass="{m_shell/N_TILES:.5f}" solref="{SOLREF}"{imp} contype="2" conaffinity="2" rgba=".6 .8 1 .25"/>'
        )
    if roll:  # friction mixes by max, so the plane's tiny rolling term loses
        geoms.append(f'<geom type="sphere" size="{R}" mass="0" condim="6" friction="1 {TORSION} {ROLL}" rgba="0 0 0 0"/>')
    else:  # frictionless roll: exposes energy fed by the drive at zero input
        geoms.append(f'<geom type="sphere" size="{R}" mass="0" rgba="0 0 0 0"/>')
    return "\n".join(geoms)


# Drop lane at y = -20: a 30 deg ramp with a 20 m drop runs down -x, bends to flat over a curved runout (a straight
# 30 deg kink bounced the ball over the bumps), and enters a 0.3 m platform of 8 cm then 15 cm bumps, which ends in
# a 0.3 m step down at x = -4. G spawns the ball near the top.
LANE_Y, DROP_DEG, DROP_LEN, DECK = -20.0, 30.0, 40.0, 0.3
CURVE_R, CURVE_SEGS = 6.0, 8


def lane_segments():
    """(start xz, angle rad, length) from the deck edge at x = 10 up the curved runout and the straight ramp."""
    segs, p = [], np.array([10.0, DECK])
    step = math.radians(DROP_DEG) / CURVE_SEGS
    for i in range(CURVE_SEGS):
        segs.append((p, (i + 0.5) * step, 2 * CURVE_R * math.sin(step / 2)))
        p = p + segs[-1][2] * np.array([math.cos(segs[-1][1]), math.sin(segs[-1][1])])
    segs.append((p, math.radians(DROP_DEG), DROP_LEN))
    return segs


def course():
    g = ['<geom type="plane" size="80 80 .1" rgba=".35 .4 .35 1"/>']
    for p, a, ln in lane_segments():  # boxes overlap 5 cm at each joint; in a concave bend the overlap hides under the next box
        nrm = np.array([-math.sin(a), math.cos(a)])
        mid = p + ln / 2 * np.array([math.cos(a), math.sin(a)]) - 0.1 * nrm
        g.append(f'<geom type="box" size="{ln/2 + .05:.3f} 3 .1" pos="{mid[0]:.3f} {LANE_Y} {mid[1]:.3f}" '
                 f'euler="0 {-math.degrees(a):.3f} 0" rgba=".7 .6 .4 1"/>')
    g.append(f'<geom type="box" size="7 3 {DECK/2}" pos="3 {LANE_Y} {DECK/2}" rgba=".6 .55 .45 1"/>')
    bumps = [(9 - .7 * k, .08) for k in range(8)] + [(3 - 1.0 * k, .15) for k in range(6)]
    for x, r in bumps:
        g.append(f'<geom type="capsule" fromto="{x:.2f} {LANE_Y - 3} {DECK} {x:.2f} {LANE_Y + 3} {DECK}" size="{r}" rgba=".5 .5 .5 1"/>')
    for x, deg in [(-4, 10), (0, 20), (4, 30)]:
        a = math.radians(deg)
        g.append(f'<geom type="box" size="1.5 3 .1" pos="{x} {6 + 3*math.cos(a):.3f} {3*math.sin(a):.3f}" '
                 f'euler="{deg} 0 0" rgba=".7 .6 .4 1"/>')
    for k in range(8):
        g.append(f'<geom type="capsule" fromto="-3 {-3 - .7*k} 0 3 {-3 - .7*k} 0" size=".08" rgba=".5 .5 .5 1"/>')
    return "\n".join(g)


def spawn_top(m, d):
    """Shell at rest on the drop ramp, 38 m up its slope, hamster at the bottom of the shell."""
    mujoco.mj_resetData(m, d)
    p, a, _ = lane_segments()[-1]
    c = np.array([p[0] + 38 * math.cos(a) - R * math.sin(a), LANE_Y, p[1] + 38 * math.sin(a) + R * math.cos(a)])
    d.qpos[0:3], d.qpos[7:10] = c, c - [0, 0, L_ARM - 0.01]
    mujoco.mj_forward(m, d)


def build(var, roll=True):
    m_ham, m_shell = var.masses
    imp = f' solimp="{SOLIMP}"' if var.contain == "solimp" else ""
    # foot+grip: frictionless hamster-wall contact, so the foot force is the only traction (199be5c); with friction it blows up
    slip = ' priority="1" condim="1"' if var.pair == "foot+grip" else ""
    ham = (f'<body name="ham" pos="0 0 {T + R_HAM + .01}"><freejoint/>'
           f'<geom type="sphere" size="{R_HAM}" mass="{m_ham}" solref="{SOLREF}"{imp} margin="{MARGIN}"{slip} contype="2" conaffinity="2" rgba="1 .6 .2 1"/></body>')
    xml = f"""<mujoco><option timestep="{TIMESTEP}" integrator="implicitfast"/><size nuserdata="1"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<worldbody><light pos="0 0 10" dir="0 0 -1" directional="true"/>
{course()}
<body name="shell" pos="0 0 {R}"><freejoint/>
{shell_wall(m_shell, roll, imp)}
</body>
{ham}
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def body_vel(m, d, b):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
    return res[3:]


def wall_gap(m, d):
    """Hamster surface to the nearest inner tile face (m), negative when pressed in."""
    s, h = m.body("shell").id, m.body("ham").id
    a = m.body_geomadr[s]
    normals = m.geom_pos[a:a + N_TILES] / (R - T / 2)  # tiles come first in the shell body
    r = d.xmat[s].reshape(3, 3).T @ (d.xipos[h] - d.xpos[s])
    return R - T - R_HAM - (normals @ r).max()


def control(m, d, var, cal, v, e):
    """Model C with a choice of force pair and speed cap. v: unit run direction in world xy, e: effort in [0,1].
    cal: V_RUN (fade) or drag k (drag) from calibrate()."""
    s, h = m.body("shell").id, m.body("ham").id
    d.xfrc_applied[:] = 0
    if var.cap == "drag":  # not gated on ground contact: at speed the outer sphere touches the ground on only 8-20% of steps
        u = d.qvel[:3]
        d.xfrc_applied[s, :3] = -cal * np.linalg.norm(u) * u
    if var.cap == "fade":  # the hamster's legs can only run so fast; braking and coasting keep full strength
        e *= min(1.0, max(0.0, 1 - np.dot(d.qvel[:2], v[:2]) / cal))
    if var.contain == "field":  # velocity clamp, not a force: forces stiff enough go unstable at 2 ms whether or not the shell is grounded
        r = d.xipos[h] - d.xpos[s]
        dist = np.linalg.norm(r)
        depth = dist - L_ARM - FIELD_START
        if depth > 0:
            u = r / dist
            out = np.dot(d.qvel[6:9] - d.qvel[0:3], u)  # free-joint linear velocities, world frame
            m_ham, m_shell = var.masses
            if out > 0:  # inelastic impulse along the line of centres: momentum and angular momentum conserved
                j = min(1.0, depth / (FIELD_HARD - FIELD_START)) * m_ham * m_shell / (m_ham + m_shell) * out
                d.qvel[6:9] -= j / m_ham * u
                d.qvel[0:3] += j / m_shell * u
            over = depth - (FIELD_HARD - FIELD_START)
            if over > 0:  # hard stop: a sustained load (a 30 g hit) still creeps ~1 mm per step past the velocity clamp
                d.qpos[7:10] -= over * m_shell / (m_ham + m_shell) * u  # pair's centre of mass stays put
                d.qpos[0:3] += over * m_ham / (m_ham + m_shell) * u
                mujoco.mj_forward(m, d)
    if wall_gap(m, d) <= MARGIN + C_TOUCH:
        d.userdata[0] = d.time
    if d.time - d.userdata[0] > C_GRACE:
        return
    scale = var.masses[0] / 2.0
    lean = math.radians(var.lean)
    c, hp = d.xpos[s], d.xipos[h]
    down = np.array([0, 0, -1.0])
    n = (hp - c) / (np.linalg.norm(hp - c) + 1e-9)
    target = math.sin(e * lean) * v + math.cos(e * lean) * down
    rel_v = body_vel(m, d, h) - body_vel(m, d, s)
    f = scale * (C_KP * L_ARM * (target - n) - C_KD * rel_v)
    if var.ff:
        f -= var.masses[0] * 9.81 * (down - np.dot(down, target) * target)
    f -= np.dot(f, n) * n
    norm = np.linalg.norm(f)
    if norm > C_F_MAX * scale:
        f *= C_F_MAX * scale / norm
    p = c + n * (R - T)  # foot on the inner wall
    d.xfrc_applied[h, :3] += f
    d.xfrc_applied[s, :3] -= f
    if var.pair == "offset":  # -F at the foot, a hamster radius off +F's line: net couple |F| R_HAM
        d.xfrc_applied[s, 3:] += np.cross(p - d.xipos[s], -f)
    elif var.pair == "centre":  # -F on +F's line through the hamster centre: zero net wrench, no hamster spin
        d.xfrc_applied[s, 3:] += np.cross(hp - d.xipos[s], -f)
    else:  # both at the foot, plus a grip couple so the foot force doesn't spin the hamster like a slipping wheel
        d.xfrc_applied[h, 3:] += np.cross(p - hp, f)
        d.xfrc_applied[s, 3:] += np.cross(p - d.xipos[s], -f)
        grip = -K_GRIP * scale * (d.cvel[h, :3] - d.cvel[s, :3])
        d.xfrc_applied[h, 3:] += grip
        d.xfrc_applied[s, 3:] -= grip


def broken(m, d):
    return not np.isfinite(d.qpos).all() or np.linalg.norm(d.xipos[m.body("ham").id] - d.xpos[m.body("shell").id]) > R


def lean_deg(m, d):
    u = d.xipos[m.body("ham").id] - d.xpos[m.body("shell").id]
    return math.degrees(math.acos(max(-1.0, min(1.0, -u[2] / (np.linalg.norm(u) + 1e-9)))))


class Broken(Exception):
    pass


def sim(m, d, var, cal, until, v, e, each=None):
    while d.time < until - TIMESTEP / 2:
        control(m, d, var, cal, v, e)
        mujoco.mj_step(m, d)
        if broken(m, d):
            raise Broken
        if each:
            each()


X, Y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])


def speed(d):
    return float(np.linalg.norm(d.qvel[:2]))


def accel(var):
    """Full-effort acceleration from 1 s to 3 s on flat ground, uncapped (m/s^2)."""
    var = var._replace(cap="none")
    m, d = build(var)
    sim(m, d, var, 0.0, 1.0, X, 1.0)
    v1 = speed(d)
    sim(m, d, var, 0.0, 3.0, X, 1.0)
    return (speed(d) - v1) / 2.0


def calibrate(var, a=None):
    """V_RUN (fade) or drag k (drag) that puts the full-effort top speed near var.top, from the uncapped acceleration."""
    if var.cap == "none":
        return 0.0
    a = accel(var) if a is None else a
    m_ham, m_shell = var.masses
    imp = f' solimp="{SOLIMP}"' if var.contain == "solimp" else ""
    if var.cap == "drag":  # drag cancels the net accel at the top speed; thin shell rolls like 5/3 its mass
        return (5 / 3 * m_shell + m_ham) * a / var.top ** 2
    lean = math.radians(var.lean)  # drive ~ sin(lean); the fade stops where it drops to the rolling loss
    x = math.asin(min(1.0, A_ROLL / (a + A_ROLL) * math.sin(lean))) / lean
    return var.top / (1 - x)


def sink(var, cal, vr):
    """Throw the hamster from the shell centre at vr m/s (down, up, side), shell at rest: worst sink past L_ARM (m), inf if it escapes."""
    worst = [0.0]
    for dirn in [(0, 0, -1), (0, 0, 1), (1, 0, 0)]:
        m, d = build(var)
        h, s = m.body("ham").id, m.body("shell").id
        d.qpos[7:10] = d.qpos[0:3]
        d.qvel[6:9] = np.array(dirn) * vr
        try:
            sim(m, d, var, cal, 1.0, X, 0.0,
                lambda: worst.__setitem__(0, max(worst[0], np.linalg.norm(d.xipos[h] - d.xpos[s]) - L_ARM)))
        except Broken:
            return math.inf
    return worst[0]


def contain_row(var):
    """Escape throws, then rattle and worst wall penetration over 2-6 s at full effort on flat ground."""
    cal = calibrate(var)
    r = {f"sink{vr}": sink(var, cal, vr) for vr in (0, 4, 8, 12)}
    m, d = build(var)
    rattle, pen = [], 0.0
    def each():
        nonlocal pen
        if d.time > 2.0:
            rattle.append(wall_gap(m, d) > MARGIN + C_TOUCH)
            pen = max([pen] + [-c.dist for c in d.contact[: d.ncon]])
    sim(m, d, var, cal, 6.0, X, 1.0, each)
    r.update(rattle=np.mean(rattle), pen=pen, v6=speed(d))
    return r


def measure(var):
    a = accel(var)
    cal = calibrate(var, a)
    capped = var.cap != "none"
    r = dict(a=a, cal=cal)

    m, d = build(var)
    sim(m, d, var, cal, 3.0, X, 0.5)
    r["v3h"] = speed(d)

    m, d = build(var)
    leans = []
    sim(m, d, var, cal, 3.0, X, 1.0, lambda: d.time > 2.0 and leans.append(lean_deg(m, d)))
    r["v3f"], r["lean"] = speed(d), np.mean(leans)
    t_top = 20.0 if capped else 10.0
    sim(m, d, var, cal, t_top, X, 1.0)
    r["top"] = speed(d)
    sim(m, d, var, cal, t_top + 1.0, X, 0.0)  # the hamster swings back to the bottom in the first second and pushes the shell on
    v1 = speed(d)
    sim(m, d, var, cal, t_top + 3.0, X, 0.0)
    r["coast"] = (v1 - speed(d)) / 2.0

    m, d = build(var, roll=False)  # frictionless roll: any speed change at zero input comes from the drive
    while speed(d) < 4.2 and d.time < 5.0:
        sim(m, d, var, cal, d.time + 0.1, X, 1.0)
    t0, v0 = d.time, speed(d)
    sim(m, d, var, cal, t0 + 40.0, X, 0.0)
    r["drift"] = speed(d) - v0

    m, d = build(var)  # a 90 deg turn at speed, then release: leftover yaw after 8 s
    sim(m, d, var, cal, 3.0, X, 1.0)
    sim(m, d, var, cal, 5.0, Y, 1.0)
    sim(m, d, var, cal, 13.0, X, 0.0)
    r["yaw"] = abs(d.cvel[m.body("shell").id, 2])

    m, d = build(var)
    g = math.radians(20)
    m.opt.gravity[:] = [-9.81 * math.sin(g), 0, -9.81 * math.cos(g)]  # uphill is +x
    sim(m, d, var, cal, 3.0, X, 1.0)
    r["climb"] = d.qpos[0]
    return r


def sweep():
    """Drive grid uncapped (pair x masses x lean x ff), then each cap at each top speed on the 2/1 kg, 70 deg drive."""
    drives = [Var(p, ms, l, ff, "none", 0.0) for p, ms, l, ff in itertools.product(PAIRS, MASSES, LEANS, FFS)]
    caps = [Var(p, MASSES[0], LEANS[0], False, c, t) for p, c, t in itertools.product(PAIRS, CAPS[1:], TOPS)]
    print(f"{'pair':9s} {'ham/shell':>9s} {'lean':>4s} {'ff':>2s} {'cap':>9s} | {'a':>5s} {'v3 h':>5s} {'v3 f':>5s} {'lean':>4s} {'top':>5s} {'coast':>5s} {'drift':>6s} {'yaw':>5s} {'climb':>5s}")
    print(f"{'':9s} {'kg':>9s} {'deg':>4s} {'':>2s} {'(cal)':>9s} | {'m/s2':>5s} {'m/s':>5s} {'m/s':>5s} {'deg':>4s} {'m/s':>5s} {'m/s2':>5s} {'m/s':>6s} {'rad/s':>5s} {'m@3s':>5s}")
    for var in drives + caps:
        cap = "none" if var.cap == "none" else f"{var.cap} {var.top:.0f}"
        head = f"{var.pair:9s} {var.masses[0]:4.1f}/{var.masses[1]:3.1f} {var.lean:4d} {'on' if var.ff else '':>2s} {cap:>9s} |"
        try:
            r = measure(var)
        except Broken:
            print(head, "BROKEN (NaN or hamster escaped)")
            continue
        cal = "" if var.cap == "none" else f"  {'V_RUN' if var.cap == 'fade' else 'k'} {r['cal']:.3g}"
        print(head, f"{r['a']:5.2f} {r['v3h']:5.1f} {r['v3f']:5.1f} {r['lean']:4.0f} {r['top']:5.1f} "
              f"{r['coast']:5.2f} {r['drift']:+6.2f} {r['yaw']:5.2f} {r['climb']:5.1f}{cal}")

    print(f"\ncontainment on {PICK.pair} {PICK.masses[0]}/{PICK.masses[1]} kg {PICK.lean} deg, {PICK.cap} {PICK.top:.0f}")
    print(f"{'contain':7s} | worst sink (mm) at throw 0 / 4 / 8 / 12 m/s | {'rattle':>6s} {'pen mm':>6s} {'v@6s':>5s}")
    for c in CONTAINS:
        r = contain_row(PICK._replace(contain=c))
        sinks = " ".join(f"{r[f'sink{vr}']*1000:5.0f}" if r[f"sink{vr}"] < math.inf else "  OUT" for vr in (0, 4, 8, 12))
        print(f"{c:7s} | {sinks:>42s} | {r['rattle']*100:5.0f}% {r['pen']*1000:6.1f} {r['v6']:5.1f}")


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
    win = glfw.create_window(1280, 720, "PROTOTYPE honest drive", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    axes = [PAIRS, MASSES, LEANS, FFS, CAPS, TOPS, CONTAINS]
    idx = [ax.index(x) for ax, x in zip(axes, PICK)]
    pending = {"reset": False, "spawn": False}

    def on_key(_w, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        cycle = {glfw.KEY_P: 0, glfw.KEY_H: 1, glfw.KEY_L: 2, glfw.KEY_F: 3, glfw.KEY_C: 4, glfw.KEY_V: 5, glfw.KEY_B: 6}
        if key in cycle:
            i = cycle[key]
            idx[i] = (idx[i] + 1) % len(axes[i])
            pending["reset"] = True
        elif key == glfw.KEY_R:
            pending["reset"] = True
        elif key == glfw.KEY_G:
            pending["spawn"] = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    glfw.set_key_callback(win, on_key)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    cam.distance, cam.elevation, cam.azimuth = 4.0, -20.0, 90.0

    def load():
        var = Var(*(ax[i] for ax, i in zip(axes, idx)))
        cal = calibrate(var)
        m, d = build(var)
        return var, cal, m, d, mujoco.MjvScene(m, maxgeom=2000), mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)

    var, cal, m, d, scn, ctx = load()
    t0, rate, acc, v_prev, worst = time.perf_counter(), 1.0, 0.0, 0.0, 0.0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if pending["reset"]:
            pending["reset"] = False
            ctx.free()
            var, cal, m, d, scn, ctx = load()
            t0, acc, v_prev, worst = time.perf_counter(), 0.0, 0.0, 0.0
        if pending["spawn"]:
            pending["spawn"] = False
            spawn_top(m, d)
            t0, acc, v_prev, worst = time.perf_counter(), 0.0, 0.0, 0.0
            cam.azimuth = 180.0

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
        steps, t_before = 0, d.time
        while d.time < wall and steps < 50:
            control(m, d, var, cal, v, e)
            mujoco.mj_step(m, d)
            steps += 1
            worst = max(worst, np.linalg.norm(d.xipos[m.body("ham").id] - d.xpos[s]) - L_ARM)
        if d.time > t_before:
            acc = 0.9 * acc + 0.1 * (speed(d) - v_prev) / (d.time - t_before)
            v_prev = speed(d)
        if wall > 0.5:
            rate = 0.9 * rate + 0.1 * (d.time / wall)

        cap = "none" if var.cap == "none" else f"{var.cap} to {var.top:.0f} m/s ({'V_RUN' if var.cap == 'fade' else 'k'} {cal:.3g})"
        left = "\n".join(["pair (P)", "hamster/shell (H)", "lean ceiling (L)", "feed-forward (F)", "speed cap (C, V)", "containment (B)", "",
                          "speed", "accel", "lean", "shell yaw rate", "worst sink (G/R resets)", "sim/real"])
        right = "\n".join([var.pair, f"{var.masses[0]} / {var.masses[1]} kg", f"{var.lean} deg", "on" if var.ff else "off", cap, var.contain, "",
                           f"{speed(d):.2f} m/s", f"{acc:+.2f} m/s2", f"{lean_deg(m, d):.0f} deg",
                           f"{d.cvel[s, 2]:+.2f} rad/s",
                           f"{worst*1000:.0f} mm" + ("  ESCAPED" if worst > R - L_ARM else ""), f"{rate:.2f}" + ("  BROKEN" if broken(m, d) else "")])

        w, hgt = glfw.get_framebuffer_size(win)
        vp = mujoco.MjrRect(0, 0, w, hgt)
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp, left, right, ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
                           "P pair  H masses  L lean  F feed-fwd  C cap  V top speed  B contain  G drop lane  WASD run  Shift effort  Q/E orbit  R reset  Esc quit", "", ctx)
        glfw.swap_buffers(win)
    ctx.free()
    glfw.terminate()


if __name__ == "__main__":
    sweep() if "--sweep" in sys.argv else play()
