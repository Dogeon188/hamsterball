# marker.py

## Purpose

The marker test of spec §9: whether the segment the shell centre swept over one env step crosses a marker's volume, so a fast shell can't skip a marker between env steps.

## Interface

- `crosses(marker, p0, p1) -> bool`: `marker` is a canonical marker dict from [level.md](level.md); `p0 == p1` is a point test (the reset step, the survive region).
- `centre(marker)`: the point a task steers toward: a sphere's or box's `pos`, a capsule's midpoint.

## How it works

- Sphere: point-to-segment distance ≤ radius (the segment–segment routine with a degenerate second segment).
- Capsule: segment–segment distance ≤ radius (Ericson, *Real-Time Collision Detection* §5.1.9).
- Box: the segment moves into the box frame (`mju_euler2Quat` with the `xyz` sequence, as MuJoCo compiles the marker geom), then a slab test against the half-sizes.

It reads only the marker and two shell positions, so re-render can rerun it from stored core states.

## Invariants and checks

- `uv run pytest tests/test_marker.py`; `tests/test_env.py` checks a goal crossed entirely within one env step.

## Known quirks

## Decisions

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/02: added for the reach-goal goal; all three marker shapes, since later tasks use each.
