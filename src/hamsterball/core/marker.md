# marker.py

## Purpose

The marker test of spec §9: whether the segment the shell centre swept over one env step crosses a marker's volume, so a fast shell can't skip a marker between env steps.

## Interface

- `crosses(marker, p0, p1, t=0.0, t0=None) -> bool`: `marker` is a canonical marker dict from [level.md](level.md); `p0 == p1` is a point test (the reset step, the survive region). `p0` is at sim time `t0` (default `t`) and `p1` at `t`. A marker with a `motion` (see [motion.md](motion.md)) is tested on the relative path: `p0` in its frame at `t0`, `p1` in its frame at `t`. With `t0 == t` it is posed at `t` and held there.
- `centre(marker, t=0.0, frame=None)`: the point a task steers toward: a sphere's or box's `pos`, a capsule's midpoint, plus the motion's offset at `t`. It is also the pivot a marker's spin turns about.
- `frame=(pos, quat)` on either: the marker is attached to a body (`attach`, [level.md](level.md)) and authored in its frame; `frame` is that body's world pose now. `crosses` maps both points into that one frame (a point test, which is all the attached survive region needs), and `centre` returns the world point.

## How it works

- Sphere: point-to-segment distance ≤ radius (the segment–segment routine with a degenerate second segment).
- Capsule: segment–segment distance ≤ radius (Ericson, *Real-Time Collision Detection* §5.1.9).
- Box: the segment moves into the box frame (`mju_euler2Quat` with the `xyz` sequence, as MuJoCo compiles the marker geom), then a slab test against the half-sizes.

A moving marker is tested in its authored frame: each point is shifted by minus the motion offset at its own time, then turned by minus the yaw at that time about the authored centre, so the three shape tests stay as they are. The straight segment between the two mapped points is the relative path, so a crossing within one env step is caught whether the shell moves, the marker moves, or both (a fast hazard can't jump the shell, nor the shell a hazard).

The relative path is exact when the marker's offset and the shell both move in a straight line at constant speed over the step and the marker doesn't spin. Otherwise the true relative path bends away from the chord. Spin: a point at distance r from the pivot turns by Δθ = spin·Δt and its true path is an arc, which the chord misses by at most its sagitta r(1 − cos(Δθ/2)) ≈ r·Δθ²/8. An oscillation's offset bends from its chord by at most A(2πΔt/period)²/8, and a path's by at most v·Δt/2 at a corner. At 60 Hz these are sub-millimetre for the bundled levels (`survive_sweepers`: 45°/s at r ≤ 1.7 m gives 4e-5 m; amplitude 3.5 m at period 6 s gives 1.3e-4 m).

It reads only the marker, two shell positions, and the sim time, so re-render can rerun it from stored core states.

## Invariants and checks

- `uv run pytest tests/test_marker.py`; `tests/test_env.py` checks a goal crossed entirely within one env step.

## Known quirks

## Decisions

- 2026-10-22, ticket survive-movers/01: a moving marker is tested by moving the points into its authored frame rather than moving the marker dict, so a spinning box needs no euler composition.
- 2026-10-22, ticket survive-movers/04: the caller passes an attached marker's frame, so the test stays a pure function of its arguments: the env reads it from the body's qpos, the loader from the body's motion at time 0.

## Tried and rejected

- 2026-10-22, ticket survive-movers/02: sub-stepping the rotation (several chords per env step) to tighten the spin bound. The sagitta is already sub-millimetre at the spins a level uses, so one chord does.

## History

- 2026-10-22, ticket hamsterball-v1/02: added for the reach-goal goal; all three marker shapes, since later tasks use each.
- 2026-10-22, ticket survive-movers/02: `crosses` takes `t0`, the time of `p0`, and tests a moving marker on the relative path. Used for hazards; the survive region is a point test and needs no `t0`.
- 2026-10-22, ticket survive-movers/01: `crosses` and `centre` take the sim time `t` for a marker with a `motion`. The swept test still holds the marker still over the segment; ticket 02 (moving hazards) makes it relative.
- 2026-10-22, ticket survive-movers/04: `frame` for markers attached to a body (the region of `survive_raft`).
