# level.py

## Purpose

Levels of spec §7: a project-owned YAML file loads into one in-memory `Level`, which has a canonical form, a SHA-256 id, and an MJCF compile that `model.build` merges with the shell and the tier's fragment.

## Interface

- `Level.from_yaml(text, base=".")`, `Level.load(path)`, `Level.from_dict(d, base=".")` (an already-parsed level, as the generator builds it): validate and fill every default; a malformed level raises `ValueError`. An hfield's `file` resolves against `base`; `load` passes the level file's directory.
- `Level` fields: `kill_plane_z`, `geoms` (list of dicts), `tasks` (task id to block), `hazards` (list of markers, default empty), `follow_yaw_deg` (default 0), `version` (1), `generator` (the provenance block, or `None`), `bodies` (list of kinematic bodies, default empty).
- `level.canonical()`: `yaml.safe_dump(asdict(level), sort_keys=True)`, leaving out `generator` when it is `None` and `bodies` when it is empty. `level.id`: SHA-256 hex of the canonical form. Both are computed on first read and cached, so edit a `Level` only before then.
- `level.mjcf()`: top-level MJCF sections (an `<asset>` with the hfields, then a `<worldbody>` with the geoms and markers) for `model.build(tier, level)`. `level.body_mjcf()`: the kinematic bodies' sections for `model.build(..., bodies=)`, or `""`. `Core(level=level)` takes the `Level` and calls both.

## How it works

- Loading normalises every number in a numeric field to a Python float (except `version`), so `30` and `30.0` share an id, and fills `pos` and `euler` of geoms, `euler` of box markers, the spawn's `yaw_deg`, and `follow_yaw_deg`. PyYAML writes floats with `repr`, Python's shortest form.
- Geoms: `plane`, `box`, `sphere`, `capsule` with `type`, `size` (3, 3, 1, 2 numbers, MuJoCo's meaning), `pos`, `euler` (degrees, MuJoCo's default xyz sequence and degree angles), and optional `rgba`. They compile to group 0 on contype and conaffinity 1. A `plane` floor collides infinitely whatever its `size`, so a level whose edge should drop the hamsterball to the kill plane needs a box floor.
- Heightfields: `{type: hfield, size: [x half-extent, y half-extent, max height], pos, euler, file}` where `file` is a non-interlaced 16-bit grayscale PNG (read with a stdlib decoder, all five row filters). Loading replaces `file` with `nrow`, `ncol`, and `samples` (base64 of little-endian uint16, row-major, the PNG's top row first), so the canonical form and the id cover the terrain, and the inline form loads as-is. The loader rejects grids over 256×256 and cells (`2·x / (ncol−1)`, `2·y / (nrow−1)`) under 0.25 m. Each compiles to an `<hfield>` asset `hfield<i>` (i = geom index) with `elevation` = the raw samples; the PNG's top row is the +y edge.
- Contact values: any geom may set `friction: [sliding, rolling]`, `solref` (2), `solimp` (5). A geom that sets any of them compiles with `priority="1"` (the shell's outer sphere has 0), `condim="6"`, and friction `sliding 0.05 rolling`, so its values replace the shell's rather than mixing; an unset `friction` there takes the shell's `1` and `0.04`, and an unset `solref`/`solimp` MuJoCo's defaults (the shell sets neither). A geom with none of them stays unset in the canonical form and mixes with the shell as before.
- Markers: `sphere {pos, radius}`, `box {pos, size, euler}` (half-sizes), `capsule {fromto, radius}`. Each compiles to a translucent group-0 geom with contype and conaffinity 0, named `<task>/<key>` (a lone marker, such as `unlock_goal/goal`) or `<task>/<key>/<i>` (a list entry, such as `waypoint_race/waypoints/2`) so the env can find it, and coloured by kind from `MARKER_RGBA`: goal green, waypoints blue, items yellow, keys orange, survive region purple. `LOCKED_RGBA` is the look of a marker that can't trigger yet (the unlock goal before every key is in, a waypoint after the next), and `PASSED_RGBA` a passed waypoint's (the env sets both). Colours are look constants (spec §11).
- Task blocks: `spawn {pos, yaw_deg}`, the task's markers, `time_limit_s`. `TASK_MARKERS` lists each task's marker keys in task order: `reach_goal` `goal`; `waypoint_race` `waypoints`; `collect_items` `items`; `unlock_goal` `keys`, `goal`; `survive` `region`. The keys in `MARKER_LISTS` (`waypoints`, `items`, `keys`) hold a non-empty list of markers, in order.
- Hazards: a level-wide list of markers under `hazards`, shared by every task; a missing or empty list loads as `[]` (filled, so it is part of the id). Each compiles after the level geoms as `hazard<i>` in `HAZARD_RGBA` (red); the name has no `/`, so the env's per-task looks never hide it.
- Provenance: a generated level carries `generator: {name, version, task, seed, difficulty}` (string, int, task id, int, number); the id covers it. A hand-authored level has `generator = None`, and its canonical form has no `generator` key, so ids from before the block existed stand. See [generator.md](generator.md).
- Motion: the survive region may carry a `motion` ([motion.md](motion.md)), loaded with its defaults filled; absent stays absent, so a still level's canonical form and id are unchanged. A hazard may carry one too (ticket survive-movers/02); task markers other than the region don't yet (an unknown key raises). `level.movers()` lists `(geom name, marker)` for every marker with a motion, named as `mjcf()` names its geom; the env poses those geoms. `mjcf()` compiles a moving marker at its authored pose.
- Kinematic bodies: a level-wide `bodies` list of `{pos, geoms, motion}`. `pos` (default origin) is the body frame's origin, which the motion offsets and a spin turns about; `geoms` (non-empty) are in that frame, validated as level geoms but only `box`, `sphere`, `capsule` (a plane or hfield can't move); `motion` is required ([motion.md](motion.md)). Body `i` compiles as `body<i>`: a free joint, `gravcomp="1"`, an explicit inertial of `BODY_MASS` 1000 kg and `BODY_INERTIA` 1000 kg·m² (geom masses unused), its geoms on contype and conaffinity 1 like level geoms, and `<contact><exclude>` from the world and from every other body. The core drives it ([step.md](step.md)).
- A survive spawn outside its region (the marker point test of [marker.md](marker.md), with the region at its pose at time 0) raises.
- Attaching (ticket survive-movers/04): the survive region may set `attach: <i>`, the index (an int) of a body in `bodies`; it is then authored in that body's frame and rides it, tilt included. It can't also have a `motion`. It compiles inside `body<i>` (in `body_mjcf()`, not `mjcf()`), so MuJoCo draws it on the body in every view; its contype 0 and the body's explicit inertial keep it out of contacts and mass. `body_frame(body, t)` is the body's world `(pos, quat)` from its motion. The survive spawn check uses the attached body's frame at time 0. A body's motion may `tilt` ([motion.md](motion.md)).

## Invariants and checks

- Two files that differ only in comments, key order, int-versus-float spelling, or written-out defaults share an id.
- Level geoms add no bodies, so the shell and hamster keep body ids 1 and 2 (see [model.md](model.md)).
- The canonical form round-trips: file → canonical YAML → `Level` → same id, hfield samples included.
- `uv run pytest tests/test_level.py`; `test_authored_friction_overrides_the_shells` checks the priority by coast distance from 3 m/s over 2 s on a box: 3.78 m with no contact values or the shell's own, 5.30 m with rolling 0.004 (max-mixing would give 3.78 m), 1.28 m with rolling 0.2.

## Known quirks

- Unknown keys raise rather than pass through.
- An unset geom `rgba` stays unset in the canonical form and compiles to a fixed grey: colour is a look value, so it isn't filled into the level id. Every other default is filled.
- A `plane` collides infinitely in MuJoCo whatever its `size`, so a level with a kill plane needs finite ground (boxes) for the hamsterball to fall off.
- MuJoCo scales hfield samples so the lowest is 0 and the highest is the max height: only relative heights in the PNG matter, and a constant PNG is flat at `pos` z. An hfield sits on a solid slab `HFIELD_BASE` = 1 m deep (MuJoCo needs a positive base; spec §7's `size` has three numbers).
- The torsional friction of a contact-tuned geom is the shell's 0.05: spec §7 lets authors set only sliding and rolling.

- `canonical()` and `id` are cached on the instance: a 256×256 hfield takes about 0.13 s to serialise on the M2 (PyYAML's pure-Python emitter), and the env read them five times per reset. A `Level` edited after its first read keeps its old id.

## Decisions

- 2026-10-22, ticket hamsterball-v1/02: `Level` is a dataclass over plain dicts and lists, the canonical tree itself, in place of a typed class per geom and marker.
- 2026-10-22, ticket hamsterball-v1/05: a 16-bit PNG is decoded by a stdlib reader (zlib and the five PNG filters) rather than Pillow, to add no dependency; MuJoCo's own PNG loader is not used, as the canonical form needs the raw uint16 samples.
- 2026-10-22, ticket hamsterball-v1/05: `mjcf()` returns top-level MJCF sections rather than bare worldbody geoms, so hfield assets can ride along; `model.build` places them between two worldbody sections.

- 2026-10-22, ticket survive-movers/01: a marker's `motion` is a key on the marker itself (offsets from its authored pose), omitted when absent, so no existing level id changes.

- 2026-10-22, ticket hamsterball-v1/10: a hand-authored level's canonical form omits `generator` rather than writing `generator: null`, so the provenance block changes no existing level id.
- 2026-10-22, ticket survive-movers/03: `bodies` is omitted from the canonical form when empty, like `generator`, so no existing level id changes. A body's geoms sit in its own frame (MuJoCo's form), so a spin or a later tilt (ticket 04) turns about `pos`.
- 2026-10-22, ticket survive-movers/04: a region names its body by index, not by a new body `name` key: bodies compile as `body<i>` already, and an index needs no uniqueness check. Ticket 05's seesaw, if it joins the same `bodies` list, is named the same way.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/02: added with primitive geoms, the kill plane, `follow_yaw_deg`, and the `reach_goal` block.
- 2026-10-22, ticket hamsterball-v1/03: added the `waypoint_race`, `collect_items`, and `unlock_goal` blocks, marker lists, named marker geoms, and colour by kind.
- 2026-10-22, ticket hamsterball-v1/04: added `hazards` and the `survive` block with its spawn-inside-region check. Filling `hazards: []` changed the canonical form, so every level id from ticket 02 changed.
- 2026-10-22, coordinator relaying the user, during the ticket 03/04 merge: spec §7's `plane` geom stays allowed; the doc states that a plane floor collides infinitely, so a level whose edge should reach the kill plane needs a box floor.
- 2026-10-22, ticket hamsterball-v1/05: added hfields (PNG beside the file, samples inlined in the canonical form) and per-geom `friction`, `solref`, `solimp` with priority over the shell.
- 2026-10-22, ticket hamsterball-v1/14, agent finding while profiling the throughput gate: on a 256×256 hfield level each env reset spent about 0.63 s in `yaml.safe_dump` (five `id`/`canonical()` reads at 0.126 s each), which cut a reset-every-470-steps run from about 2,000 to 870 env steps/s. `canonical()` and `id` are now cached; `tests/test_level.py::test_a_large_hfield_levels_id_is_cheap_to_read_again`.
- 2026-10-22, ticket hamsterball-v1/10: added the optional `generator` provenance block and `Level.from_dict`.
- 2026-10-22, user play-test request: passed waypoints turn dim green; added the look constant `PASSED_RGBA` (the goal's green at alpha 0.2), which the env sets.
- 2026-10-22, user play-test request: waypoints after the next render grey; `LOCKED_GOAL_RGBA` is renamed `LOCKED_RGBA`, since it now colours locked waypoints too.
- 2026-10-22, ticket survive-movers/02: hazards take an optional `motion`; `movers()` lists them as `hazard<i>`; bundled `survive_sweepers`.
- 2026-10-22, ticket survive-movers/01: added the optional `motion` on the survive region, `Level.movers()`, and the bundled level `survive_drift`.
- 2026-10-22, ticket survive-movers/03: added `bodies`, `body_mjcf()`, and the bundled level `survive_pushers`. The static-geom MJCF moved into `_geom_mjcf`, shared with body geoms (same output).
- 2026-10-22, ticket survive-movers/04: `attach` on the survive region, `tilt` on body motions, `body_frame`; bundled `survive_raft`. Levels without either keep their ids.
