# level.py

## Purpose

Levels of spec §7: a project-owned YAML file loads into one in-memory `Level`, which has a canonical form, a SHA-256 id, and an MJCF compile that `model.build` merges with the shell and the tier's fragment.

## Interface

- `Level.from_yaml(text)`, `Level.load(path)`: validate and fill every default; a malformed level raises `ValueError`.
- `Level` fields: `kill_plane_z`, `geoms` (list of dicts), `tasks` (task id to block), `hazards` (list of markers, default empty), `follow_yaw_deg` (default 0), `version` (1).
- `level.canonical()`: `yaml.safe_dump(asdict(level), sort_keys=True)`. `level.id`: SHA-256 hex of the canonical form.
- `level.mjcf()`: worldbody geoms for `model.build(tier, level)`, or `Core(level=...)`.

## How it works

- Loading normalises every number in a numeric field to a Python float (except `version`), so `30` and `30.0` share an id, and fills `pos` and `euler` of geoms, `euler` of box markers, the spawn's `yaw_deg`, and `follow_yaw_deg`. PyYAML writes floats with `repr`, Python's shortest form.
- Geoms: `plane`, `box`, `sphere`, `capsule` with `type`, `size` (3, 3, 1, 2 numbers, MuJoCo's meaning), `pos`, `euler` (degrees, MuJoCo's default xyz sequence and degree angles), and optional `rgba`. They compile to group 0 on contype and conaffinity 1.
- Markers: `sphere {pos, radius}`, `box {pos, size, euler}` (half-sizes), `capsule {fromto, radius}`. Each compiles to a translucent group-0 geom with contype and conaffinity 0, coloured by kind: goal green, survive region blue, hazard red. Hazards compile after the level geoms, then each task's markers.
- Hazards: a level-wide list of markers under `hazards`, shared by every task; a missing or empty list loads as `[]` (filled, so it is part of the id).
- Task blocks: `spawn {pos, yaw_deg}`, the task's markers, `time_limit_s`. `reach_goal` has marker `goal`, `survive` has marker `region`; the other three tasks are not taught yet. A survive spawn outside its region (the marker point test of [marker.md](marker.md)) raises.

## Invariants and checks

- Two files that differ only in comments, key order, int-versus-float spelling, or written-out defaults share an id.
- Level geoms add no bodies, so the shell and hamster keep body ids 1 and 2 (see [model.md](model.md)).
- `uv run pytest tests/test_level.py`.

## Known quirks

- Unknown keys raise rather than pass through, so a key that a later ticket adds (`friction`, `solref`, `solimp`, `hfield`, the other four tasks, `generator`) is rejected until that ticket teaches the loader.
- An unset geom `rgba` stays unset in the canonical form and compiles to a fixed grey: colour is a look value, so it isn't filled into the level id. Every other default is filled.
- A `plane` collides infinitely in MuJoCo whatever its `size`, so a level with a kill plane needs finite ground (boxes) for the hamsterball to fall off.

## Decisions

- 2026-10-22, ticket hamsterball-v1/02: `Level` is a dataclass over plain dicts and lists, the canonical tree itself, in place of a typed class per geom and marker.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/02: added with primitive geoms, the kill plane, `follow_yaw_deg`, and the `reach_goal` block.
- 2026-10-22, ticket hamsterball-v1/04: added `hazards` and the `survive` block with its spawn-inside-region check. Filling `hazards: []` changed the canonical form, so every level id from ticket 02 changed.
