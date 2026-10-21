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
# can hard-code them (as a GPU kernel would). Level geoms add no bodies; later bodies (view mocaps) come after.
SHELL, HAM = 1, 2
SHELL_QPOS, HAM_QPOS = 0, 7
SHELL_DOF, HAM_DOF = 0, 6


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


def build(tier, level=FLAT):
    """Compile the shell, the tier's fragment, the view mocaps, and the level (top-level MJCF sections: assets,
    worldbody geoms)."""
    from .views import FRAGMENT as VIEWS  # views imports this module's ids

    xml = f"""<mujoco><option timestep="{DT!r}" integrator="implicitfast"/>
<size nuserdata="{1 + tier.nuserdata}"/>
<visual><global offwidth="1280" offheight="720"/><map znear="0.001"/></visual>
<worldbody><light pos="0 0 10" dir="0 0 -1" directional="true"/></worldbody>
{level}
<worldbody>
{_shell_xml()}
{tier.fragment}
{VIEWS}
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.body("shell").id == SHELL and m.body("hamster").id == HAM
    assert m.jnt_qposadr[m.body_jntadr[HAM]] == HAM_QPOS and m.jnt_dofadr[m.body_jntadr[HAM]] == HAM_DOF
    assert abs(m.body_subtreemass[HAM] - M_HAM) < 1e-9
    return m
