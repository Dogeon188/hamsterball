# motion.py

## Purpose

A mover's pose as a pure function of its motion and sim time, so the core state (which holds `time`) fixes where every mover is: replay and re-render need nothing beyond the stored core states.

## Interface

- `load(m) -> dict`: validates a motion and fills its defaults; a malformed one raises `ValueError`. [level.md](level.md) calls it on a marker's `motion`.
  - `{type: path, points: [[x, y, z], ...], period, phase: 0, spin: 0}`: a closed polyline of at least two offsets (m), not all equal, walked at constant speed (by arc length) once per `period`, back from the last point to the first.
  - `{type: oscillate, axis: [x, y, z], amplitude, period, phase: 0, spin: 0}`: an offset of `amplitude · sin(2π u)` along `axis`, which loads normalised to a unit vector (non-zero).
  - `period` (s) is positive; `phase` is a fraction of the period and loads mod 1; `spin` is a constant turn about z in degrees per second (positive counter-clockwise from above).
- `pose(motion, t) -> (offset, yaw)`: the offset (3-vector, m) from the mover's authored pose and the yaw (rad) about its authored centre at sim time `t`, with `u = (t / period + phase) mod 1`.

## How it works

Offsets, not absolute positions: a path's points and an oscillation are added to the mover's authored pose, so one motion form serves a sphere's `pos`, a capsule's `fromto`, and (ticket 03) a body of several geoms. The spin is about the mover's centre (`marker.centre` for a marker), whatever its offset.

## Invariants and checks

- `uv run pytest tests/test_motion.py`: path and oscillate poses at worked times, phase, spin, equivalent spellings, malformed motions; the region in the loader, marker test, and env.

## Known quirks

- `load` imports `level`'s validation helpers inside the function, since `level` imports this module.
- `pose` gives no velocity. A kinematic body (ticket 03) that needs one for contact friction adds it here, beside `pose`, from the same formulas.

## Decisions

- 2026-10-22, ticket survive-movers/01: one pure function of (motion, sim time) for every mover (region now, hazards and bodies in tickets 02 and 03); nothing about a mover is Python-side state.
- 2026-10-22, ticket survive-movers/01: `phase` is a fraction of the period for both forms (rather than degrees for oscillate), and a path moves at constant speed by arc length rather than equal time per segment, so a drift speed is the path length over the period.

## Tried and rejected

## History

- 2026-10-22, ticket survive-movers/01: added, for the drifting survive region of `survive_drift`.
