"""Level files (spec §7): YAML into a Level, its canonical form, and its SHA-256 id."""

import hashlib

import numpy as np
import pytest
import yaml

from hamsterball.core.level import Level

LEVEL = """
version: 1
kill_plane_z: -5
geoms:
  - {type: box, pos: [0, 0, -0.5], size: [10, 10, 0.5]}
  - {type: box, pos: [4, 0, 0.5], size: [1, 2, 0.5], euler: [0, 15, 0], rgba: [1, 0, 0, 1]}
tasks:
  reach_goal:
    spawn: {pos: [0, 0, 0.5]}
    goal: {type: sphere, pos: [0, 8, 0.5], radius: 1}
    time_limit_s: 30
"""

# The same level with comments, other key order, floats for ints, and the defaults written out.
SAME = """
# a comment
tasks:
  reach_goal:
    time_limit_s: 30.0
    goal: {radius: 1.0, type: sphere, pos: [0.0, 8.0, 0.5]}
    spawn: {yaw_deg: 0, pos: [0, 0, 0.5]}
geoms:
  - {type: box, size: [10, 10, 0.5], pos: [0, 0, -0.5], euler: [0, 0, 0]}
  - {type: box, pos: [4, 0, 0.5], size: [1, 2, 0.5], euler: [0, 15, 0], rgba: [1, 0, 0, 1]}
follow_yaw_deg: 0
kill_plane_z: -5.0
version: 1
"""


def test_a_level_file_loads_with_its_defaults_filled_in():
    level = Level.from_yaml(LEVEL)
    assert level.kill_plane_z == -5.0
    assert level.follow_yaw_deg == 0.0
    assert level.geoms[0]["euler"] == [0.0, 0.0, 0.0]
    assert level.tasks["reach_goal"]["spawn"] == {"pos": [0.0, 0.0, 0.5], "yaw_deg": 0.0}


def test_canonical_form_is_sorted_yaml_with_float_values():
    canonical = Level.from_yaml(LEVEL).canonical()
    assert canonical.startswith("follow_yaw_deg: 0.0\ngeoms:\n")
    assert "kill_plane_z: -5.0\n" in canonical
    assert "time_limit_s: 30.0\n" in canonical
    assert canonical.endswith("version: 1\n")
    assert Level.from_yaml(canonical).canonical() == canonical


def test_level_id_is_the_sha256_of_the_canonical_form():
    level = Level.from_yaml(LEVEL)
    assert level.id == hashlib.sha256(level.canonical().encode()).hexdigest()
    assert Level.from_yaml(SAME).id == level.id
    moved = yaml.safe_load(LEVEL)
    moved["tasks"]["reach_goal"]["goal"]["pos"][1] = 9
    assert Level.from_yaml(yaml.safe_dump(moved)).id != level.id


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.pop("kill_plane_z"),
        lambda d: d["geoms"][0].update(type="mesh"),
        lambda d: d["geoms"][0].update(colour=[1, 0, 0, 1]),
        lambda d: d["tasks"].update(teleport={}),
        lambda d: d["tasks"]["reach_goal"].pop("goal"),
        lambda d: d["tasks"]["reach_goal"]["goal"].update(type="cone"),
    ],
)
def test_a_malformed_level_is_rejected(edit):
    d = yaml.safe_load(LEVEL)
    edit(d)
    with pytest.raises(ValueError):
        Level.from_yaml(yaml.safe_dump(d))


def test_a_level_compiles_its_geoms_and_markers_into_the_model():
    from hamsterball.core import Core

    m = Core(level=Level.from_yaml(LEVEL).mjcf()).model
    world = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
    assert len(world) == 3  # two level geoms, one goal marker
    solid, marker = world[:2], world[2]
    assert set(m.geom_contype[solid]) == {1} and set(m.geom_conaffinity[solid]) == {1}
    assert set(m.geom_group[world]) == {0}
    assert m.geom_contype[marker] == 0 and m.geom_conaffinity[marker] == 0
    assert 0 < m.geom_rgba[marker][3] < 1
    assert m.geom_pos[marker].tolist() == [0, 8, 0.5] and m.geom_size[marker][0] == 1


def png16(path, rows):
    """Write a 16-bit grayscale PNG, row r with PNG filter type r % 5 (spec: PNG 2nd ed., section 9)."""
    import struct
    import zlib

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    def paeth(a, b, c):
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        return a if pa <= pb and pa <= pc else b if pb <= pc else c

    raw, up = b"", bytes(2 * len(rows[0]))
    for r, row in enumerate(rows):
        x, f = struct.pack(f">{len(row)}H", *row), r % 5
        out = bytearray([f])
        for i, v in enumerate(x):
            a, c = (x[i - 2], up[i - 2]) if i >= 2 else (0, 0)
            pred = [0, a, up[i], (a + up[i]) // 2, paeth(a, up[i], c)][f]
            out.append((v - pred) % 256)
        raw, up = raw + bytes(out), x
    ihdr = struct.pack(">IIBBBBB", len(rows[0]), len(rows), 16, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


HILLS = [[0, 1000, 65535], [300, 0, 20000]]  # 2 rows x 3 columns
HFIELD_LEVEL = """
version: 1
kill_plane_z: -5
geoms:
  - {type: hfield, pos: [0, 10, 0], euler: [0, 0, 30], size: [8, 4, 2], file: hills.png}
  - {type: hfield, size: [2, 2, 1], file: hills.png}
tasks:
  reach_goal:
    spawn: {pos: [0, 0, 0.5]}
    goal: {type: sphere, pos: [0, 8, 0.5], radius: 1}
    time_limit_s: 30
"""


def test_an_hfield_level_round_trips_through_its_canonical_form(tmp_path):
    png16(tmp_path / "hills.png", HILLS)
    (tmp_path / "level.yaml").write_text(HFIELD_LEVEL)
    level = Level.load(tmp_path / "level.yaml")
    canonical = level.canonical()
    assert "file" not in canonical and "hills.png" not in canonical  # the samples are inlined
    again = Level.from_yaml(canonical)
    assert again.id == level.id and again.canonical() == canonical
    png16(tmp_path / "hills.png", [[0, 1000, 65535], [300, 1, 20000]])  # one sample changes the id
    assert Level.load(tmp_path / "level.yaml").id != level.id


def test_an_hfield_png_decodes_to_its_samples_whatever_the_row_filters(tmp_path):
    import base64

    import numpy as np

    rows = [[(r * 7919 + c * 104729 + r * c * 31) % 65536 for c in range(9)] for r in range(10)]  # rows use filters 0-4
    png16(tmp_path / "hills.png", rows)
    d = yaml.safe_load(HFIELD_LEVEL)
    d["geoms"] = d["geoms"][1:]
    level = Level.from_yaml(yaml.safe_dump(d), tmp_path)
    g = level.geoms[0]
    assert (g["nrow"], g["ncol"]) == (10, 9)
    got = np.frombuffer(base64.b64decode(g["samples"]), "<u2").reshape(10, 9)
    np.testing.assert_array_equal(got, rows)


@pytest.mark.parametrize(
    "rows, size",
    [
        ([[0] * 3] * 2, [0.24, 4, 1]),  # 3 columns over 0.48 m: 0.24 m cells
        ([[0] * 2] * 33, [4, 3.99, 1]),  # 33 rows over 7.98 m: 0.249 m cells
        ([[0] * 257] * 2, [100, 4, 1]),  # 257 columns
        ([[0] * 2] * 257, [4, 100, 1]),  # 257 rows
        ([[0] * 2], [4, 4, 1]),  # one row has no cells
        ([[0] * 2] * 2, [4, 4, 0]),  # no height
    ],
)
def test_an_hfield_with_small_cells_or_a_big_grid_is_rejected(tmp_path, rows, size):
    png16(tmp_path / "hills.png", rows)
    d = yaml.safe_load(HFIELD_LEVEL)
    d["geoms"] = [{"type": "hfield", "size": size, "file": "hills.png"}]
    with pytest.raises(ValueError):
        Level.from_yaml(yaml.safe_dump(d), tmp_path)


def test_the_largest_grid_with_the_smallest_cells_loads(tmp_path):
    png16(tmp_path / "hills.png", [[0] * 256] * 256)
    d = yaml.safe_load(HFIELD_LEVEL)
    d["geoms"] = [{"type": "hfield", "size": [31.875, 31.875, 1], "file": "hills.png"}]  # 255 cells of 0.25 m
    assert Level.from_yaml(yaml.safe_dump(d), tmp_path).geoms[0]["nrow"] == 256


@pytest.mark.parametrize(
    "edit",
    [
        lambda g: g.update(nrow="2"),
        lambda g: g.update(nrow=3),  # samples don't fill the grid
        lambda g: g.update(samples="not base64!"),
        lambda g: g.update(file="hills.png"),  # a file and inline samples
        lambda g: g.pop("samples"),
    ],
)
def test_a_malformed_inline_hfield_is_rejected(tmp_path, edit):
    png16(tmp_path / "hills.png", HILLS)
    (tmp_path / "level.yaml").write_text(HFIELD_LEVEL)
    d = yaml.safe_load(Level.load(tmp_path / "level.yaml").canonical())
    edit(d["geoms"][0])
    with pytest.raises(ValueError):
        Level.from_yaml(yaml.safe_dump(d), tmp_path)


def test_hfields_compile_into_the_model(tmp_path):
    import mujoco
    import numpy as np

    from hamsterball.core import Core

    png16(tmp_path / "hills.png", HILLS)
    (tmp_path / "level.yaml").write_text(HFIELD_LEVEL)
    m = Core(level=Level.load(tmp_path / "level.yaml").mjcf()).model
    hf = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD]
    assert len(hf) == 2 and m.nhfield == 2
    assert set(m.geom_bodyid[hf]) == {0} and set(m.geom_group[hf]) == {0}
    assert set(m.geom_contype[hf]) == {1} and set(m.geom_conaffinity[hf]) == {1}
    first = m.geom_dataid[hf[0]]
    np.testing.assert_allclose(m.hfield_size[first][:3], [8, 4, 2])
    np.testing.assert_allclose(m.geom_pos[hf[0]], [0, 10, 0])
    assert (m.hfield_nrow[first], m.hfield_ncol[first]) == (2, 3)
    # MuJoCo scales samples to [0, 1] of the max height; the PNG's top row is the +y edge
    grid = m.hfield_data[m.hfield_adr[first]:][:6].reshape(2, 3)
    np.testing.assert_allclose(grid[::-1] * 65535, HILLS, atol=0.01)


def coast_distance(**contact):
    """How far the hamsterball coasts from 3 m/s along +x with no input, on a ground box with these contact values."""
    from hamsterball.core import Core

    d = yaml.safe_load(LEVEL)
    d["geoms"] = [{"type": "box", "pos": [0, 0, -0.5], "size": [50, 50, 0.5], **contact}]
    core = Core(level=Level.from_yaml(yaml.safe_dump(d)).mjcf())
    core.reset(yaw=-1.5707963)
    core.data.qvel[[0, 6]] = 3.0  # shell and hamster
    core.data.qvel[4] = 3.0 / 0.5  # rolling, not sliding
    core.step(np.zeros(2), n=1200)
    return core.data.qpos[0]


def test_authored_friction_overrides_the_shells():
    default = coast_distance()
    assert coast_distance(friction=[1, 0.04]) == pytest.approx(default, rel=0.01)  # the shell's own values
    assert coast_distance(friction=[1, 0.004]) > 1.3 * default  # under max-mixing it would equal the default
    assert coast_distance(friction=[1, 0.2]) < 0.5 * default


def test_contact_parameters_compile_with_priority_over_the_shell():
    from hamsterball.core import Core

    d = yaml.safe_load(LEVEL)
    d["geoms"][1].update(solref=[0.05, 1.5], solimp=[0.8, 0.9, 0.002, 0.5, 2])
    m = Core(level=Level.from_yaml(yaml.safe_dump(d)).mjcf()).model
    plain, tuned = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0][:2]
    (outer,) = [g for g in range(m.ngeom) if m.geom_condim[g] == 6 and m.geom_bodyid[g] == 1]
    assert m.geom_priority[plain] == m.geom_priority[outer] < m.geom_priority[tuned]
    assert m.geom_condim[tuned] == 6  # the higher-priority geom sets condim; 3 would drop rolling friction
    np.testing.assert_allclose(m.geom_solref[tuned], [0.05, 1.5])
    np.testing.assert_allclose(m.geom_solimp[tuned], [0.8, 0.9, 0.002, 0.5, 2])
    np.testing.assert_allclose(m.geom_friction[tuned], m.geom_friction[outer])  # unset friction: the shell's
