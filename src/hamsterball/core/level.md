# level.py

## Purpose

Levels of spec §7: a project-owned YAML file loads into one in-memory `Level`, which has a canonical form, a SHA-256 id, and an MJCF compile that `model.build` merges with the shell and the tier's fragment.

## Interface

- `Level.from_yaml(text)`, `Level.load(path)`: validate and fill every default; a malformed level raises `ValueError`.
- `Level` fields: `kill_plane_z`, `geoms` (list of dicts), `tasks` (task id to block), `follow_yaw_deg` (default 0), `version` (1).
- `level.canonical()`: `yaml.safe_dump(asdict(level), sort_keys=True)`. `level.id`: SHA-256 hex of the canonical form.
- `level.mjcf()`: worldbody geoms for `model.build(tier, level)`, or `Core(level=...)`.

## How it works

- Loading normalises every number in a numeric field to a Python float (except `version`), so `30` and `30.0` share an id, and fills `pos` and `euler` of geoms, `euler` of box markers, the spawn's `yaw_deg`, and `follow_yaw_deg`. PyYAML writes floats with `repr`, Python's shortest form.
- Geoms: `plane`, `box`, `sphere`, `capsule` with `type`, `size` (3, 3, 1, 2 numbers, MuJoCo's meaning), `pos`, `euler` (degrees, MuJoCo's default xyz sequence and degree angles), and optional `rgba`. They compile to group 0 on contype and conaffinity 1.
- Markers: `sphere {pos, radius}`, `box {pos, size, euler}` (half-sizes), `capsule {fromto, radius}`. Each compiles to a translucent group-0 geom with contype and conaffinity 0, named `<task>/<key>` (a lone marker, such as `unlock_goal/goal`) or `<task>/<key>/<i>` (a list entry, such as `waypoint_race/waypoints/2`) so the env can find it, and coloured by kind from `MARKER_RGBA`: goal green, waypoints blue, items yellow, keys orange. `LOCKED_GOAL_RGBA` is the unlock goal's look before every key is in (the env sets it). Colours are look constants (spec §11).
- Task blocks: `spawn {pos, yaw_deg}`, the task's markers, `time_limit_s`. `TASK_MARKERS` lists each task's marker keys in task order: `reach_goal` `goal`; `waypoint_race` `waypoints`; `collect_items` `items`; `unlock_goal` `keys`, `goal`. The keys in `MARKER_LISTS` (`waypoints`, `items`, `keys`) hold a non-empty list of markers, in order.

## Invariants and checks

- Two files that differ only in comments, key order, int-versus-float spelling, or written-out defaults share an id.
- Level geoms add no bodies, so the shell and hamster keep body ids 1 and 2 (see [model.md](model.md)).
- `uv run pytest tests/test_level.py`.

## Known quirks

- Unknown keys raise rather than pass through, so a key that a later ticket adds (`hazards`, `friction`, `solref`, `solimp`, `hfield`, `survive`, `generator`) is rejected until that ticket teaches the loader.
- An unset geom `rgba` stays unset in the canonical form and compiles to a fixed grey: colour is a look value, so it isn't filled into the level id. Every other default is filled.
- A `plane` collides infinitely in MuJoCo whatever its `size`, so a level with a kill plane needs finite ground (boxes) for the hamsterball to fall off.

## Decisions

- 2026-10-22, ticket hamsterball-v1/02: `Level` is a dataclass over plain dicts and lists, the canonical tree itself, in place of a typed class per geom and marker.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/02: added with primitive geoms, the kill plane, `follow_yaw_deg`, and the `reach_goal` block.
- 2026-10-22, ticket hamsterball-v1/03: added the `waypoint_race`, `collect_items`, and `unlock_goal` blocks, marker lists, named marker geoms, and colour by kind.
