# marker.py

## Purpose

The marker test of spec §9: whether the segment the shell centre swept over one env step crosses a marker's volume, so a fast shell can't skip a marker between env steps.

## Interface

- `crosses(marker, p0, p1, t=0.0) -> bool`: `marker` is a canonical marker dict from [level.md](level.md); `p0 == p1` is a point test (the reset step, the survive region). A marker with a `motion` is posed at sim time `t` (see [motion.md](motion.md)) and held there over the segment.
- `centre(marker, t=0.0)`: the point a task steers toward: a sphere's or box's `pos`, a capsule's midpoint, plus the motion's offset at `t`. It is also the pivot a marker's spin turns about.

## How it works

- Sphere: point-to-segment distance ≤ radius (the segment–segment routine with a degenerate second segment).
- Capsule: segment–segment distance ≤ radius (Ericson, *Real-Time Collision Detection* §5.1.9).
- Box: the segment moves into the box frame (`mju_euler2Quat` with the `xyz` sequence, as MuJoCo compiles the marker geom), then a slab test against the half-sizes.

A moving marker is tested in its authored frame: both points are shifted by minus the motion offset, then turned by minus its yaw about the authored centre, so the three shape tests stay as they are.

It reads only the marker, two shell positions, and the sim time, so re-render can rerun it from stored core states.

## Invariants and checks

- `uv run pytest tests/test_marker.py`; `tests/test_env.py` checks a goal crossed entirely within one env step.

## Known quirks

## Decisions

- 2026-10-22, ticket survive-movers/01: a moving marker is tested by moving the points into its authored frame rather than moving the marker dict, so a spinning box needs no euler composition.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/02: added for the reach-goal goal; all three marker shapes, since later tasks use each.
- 2026-10-22, ticket survive-movers/01: `crosses` and `centre` take the sim time `t` for a marker with a `motion`. The swept test still holds the marker still over the segment; ticket 02 (moving hazards) makes it relative.
