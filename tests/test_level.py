"""Level files (spec §7): YAML into a Level, its canonical form, and its SHA-256 id."""

import hashlib

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


MULTI = """
kill_plane_z: -5
geoms: [{type: box, pos: [0, 0, -0.5], size: [10, 10, 0.5]}]
tasks:
  waypoint_race:
    spawn: {pos: [0, 0, 0.5]}
    waypoints: [{type: sphere, pos: [0, 3, 0.5], radius: 1}, {type: capsule, fromto: [0, 6, 0, 2, 6, 0], radius: 1}]
    time_limit_s: 45
  collect_items:
    spawn: {pos: [0, 0, 0.5]}
    items: [{type: box, pos: [3, 0, 0.5], size: [1, 1, 1]}]
    time_limit_s: 45
  unlock_goal:
    spawn: {pos: [0, 0, 0.5]}
    keys: [{type: sphere, pos: [-3, 0, 0.5], radius: 1}]
    goal: {type: sphere, pos: [0, -3, 0.5], radius: 1}
    time_limit_s: 45
"""


def test_waypoint_items_and_unlock_blocks_load_their_marker_lists():
    tasks = Level.from_yaml(MULTI).tasks
    assert [w["type"] for w in tasks["waypoint_race"]["waypoints"]] == ["sphere", "capsule"]
    assert tasks["collect_items"]["items"][0]["euler"] == [0.0, 0.0, 0.0]
    assert tasks["unlock_goal"]["keys"][0]["pos"] == [-3.0, 0.0, 0.5]
    assert tasks["unlock_goal"]["goal"]["radius"] == 1.0


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["tasks"]["waypoint_race"].update(waypoints=[]),
        lambda d: d["tasks"]["collect_items"].update(items={"type": "sphere", "pos": [0, 0, 0], "radius": 1}),
        lambda d: d["tasks"]["unlock_goal"].pop("keys"),
        lambda d: d["tasks"]["unlock_goal"]["keys"][0].update(type="cone"),
    ],
)
def test_a_malformed_marker_list_is_rejected(edit):
    d = yaml.safe_load(MULTI)
    edit(d)
    with pytest.raises(ValueError):
        Level.from_yaml(yaml.safe_dump(d))


def test_each_marker_compiles_to_a_named_geom():
    from hamsterball.core import Core

    m = Core(level=Level.from_yaml(MULTI).mjcf()).model
    names = ["waypoint_race/waypoints/0", "waypoint_race/waypoints/1", "collect_items/items/0", "unlock_goal/keys/0", "unlock_goal/goal"]
    for n in names:
        g = m.geom(n)
        assert m.geom_contype[g.id] == 0 and m.geom_group[g.id] == 0
    assert m.geom("collect_items/items/0").pos.tolist() == [3, 0, 0.5]
