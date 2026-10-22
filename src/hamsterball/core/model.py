"""The hamsterball model (spec §3): the shell, a tier's hamster fragment, and a level, compiled into one MjModel."""

import math

import mujoco
import numpy as np

DT = 1 / 600
R, T = 0.5, 0.03  # shell radius and wall thickness (m)
M_SHELL, M_HAM = 0.8, 2.4  # kg; every tier's hamster weighs M_HAM
N_TILES = 320
SOLREF = "0.04 2"  # shell tiles and the abstract hamster
TORSION, ROLL = 0.05, 0.04  # outer sphere's torsional and rolling friction (m)

# Fixed ids: the compiler puts the shell first and the hamster second, each a free body, so the per-step code
# can hard-code them (as a GPU kernel would). Level geoms add no bodies; later bodies (view mocaps, then a level's
# kinematic bodies) come after.
SHELL, HAM = 1, 2
SHELL_QPOS, HAM_QPOS = 0, 7
SHELL_DOF, HAM_DOF = 0, 6
# The core's userdata slots; the tier's follow them, then a level's collapsing tiles'.
YAW = 0
CORE_SLOTS = 1


def _tile_normals():
    phi = math.pi * (3 - math.sqrt(5))
    out = []
    for i in range(N_TILES):
        z = 1 - 2 * (i + 0.5) / N_TILES
        r = math.sqrt(1 - z * z)
        out.append((r * math.cos(i * phi), r * math.sin(i * phi), z))
    return np.array(out)


TILE_NORMALS = _tile_normals()  # (320, 3) in the shell frame

FLAT = '<worldbody><geom type="plane" size="0 0 .1" rgba=".35 .4 .35 1"/></worldbody>'


def _shell_xml():
    side = 1.25 * math.sqrt(4 * math.pi * R * R / N_TILES)
    tiles = "\n".join(
        f'<geom type="box" size="{side / 2!r} {side / 2!r} {T / 2!r}" pos="{(R - T / 2) * x!r} {(R - T / 2) * y!r} {(R - T / 2) * z!r}" '
        f'zaxis="{x!r} {y!r} {z!r}" mass="{M_SHELL / N_TILES!r}" solref="{SOLREF}" contype="2" conaffinity="2" group="3"/>'
        for x, y, z in TILE_NORMALS.tolist()
    )
    # friction mixes by max, so a level geom's small rolling term loses to the outer sphere's
    outer = f'<geom type="sphere" size="{R}" mass="0" condim="6" friction="1 {TORSION} {ROLL}" group="1" rgba=".6 .8 1 .25"/>'
    return f'<body name="shell" pos="0 0 {R}"><freejoint/>\n{tiles}\n{outer}\n</body>'


GAME_SIZE = (720, 1280)  # (height, width) of the game window, the smallest offscreen buffer a model gets


def build(tier, level=FLAT, view_size=GAME_SIZE, bodies="", slots=0):
    """Compile the shell, the tier's fragment, the view mocaps, and the level (top-level MJCF sections: assets,
    worldbody geoms), with an offscreen buffer that holds both the game window and `view_size` (height, width).
    `bodies` (top-level MJCF sections too, from `Level.body_mjcf()`) goes last, so its bodies follow the view mocaps.
    `slots` more userdata slots go after the tier's (a level's collapsing tiles)."""
    from .views import FRAGMENT as VIEWS  # views imports this module's ids

    h, w = max(view_size[0], GAME_SIZE[0]), max(view_size[1], GAME_SIZE[1])
    xml = f"""<mujoco><option timestep="{DT!r}" integrator="implicitfast"/>
<size nuserdata="{CORE_SLOTS + tier.nuserdata + slots}"/>
<visual><global offwidth="{w}" offheight="{h}"/><map znear="0.001"/></visual>
<worldbody><light pos="0 0 10" dir="0 0 -1" directional="true"/></worldbody>
{level}
<worldbody>
{_shell_xml()}
{tier.fragment}
{VIEWS}
</worldbody>
{bodies}</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.body("shell").id == SHELL and m.body("hamster").id == HAM
    assert m.jnt_qposadr[m.body_jntadr[HAM]] == HAM_QPOS and m.jnt_dofadr[m.body_jntadr[HAM]] == HAM_DOF
    assert abs(m.body_subtreemass[HAM] - M_HAM) < 1e-9
    return m
