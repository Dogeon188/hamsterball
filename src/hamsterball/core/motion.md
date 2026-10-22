# motion.py

## Purpose

A mover's pose as a pure function of its motion and sim time, so the core state (which holds `time`) fixes where every mover is: replay and re-render need nothing beyond the stored core states.

## Interface

- `load(m, tilt=False) -> dict`: validates a motion and fills its defaults; a malformed one raises `ValueError`. [level.md](level.md) calls it on a marker's `motion`, and with `tilt=True` on a body's.
  - `{type: path, points: [[x, y, z], ...], period, phase: 0, spin: 0}`: a closed polyline of at least two offsets (m), not all equal, walked at constant speed (by arc length) once per `period`, back from the last point to the first.
  - `{type: oscillate, axis: [x, y, z], amplitude, period, phase: 0, spin: 0}`: an offset of `amplitude · sin(2π u)` along `axis`, which loads normalised to a unit vector (non-zero).
  - `period` (s) is positive; `phase` is a fraction of the period and loads mod 1; `spin` is a constant turn about z in degrees per second (positive counter-clockwise from above).
- `pose(motion, t) -> (offset, yaw)`: the offset (3-vector, m) from the mover's authored pose and the yaw (rad) about its authored centre at sim time `t`, with `u = (t / period + phase) mod 1`.
  - `tilt: {axis: [x, y, 0], amplitude, period, phase: 0}` (a body's motion only): a rocking of `amplitude · sin(2π u_tilt)` degrees about `axis` (horizontal, non-zero, loads normalised), with its own `u_tilt = (t / period + phase) mod 1`; `period` defaults to the motion's. Absent stays absent, so a motion without one keeps its id.
- `quat(motion, t)`: the orientation at `t`, the spin's yaw about z times the tilt about its axis (so the tilt axis is in the spun frame). Without a tilt it is the yaw quaternion `drive_bodies` wrote before, bit for bit.

## How it works

Offsets, not absolute positions: a path's points and an oscillation are added to the mover's authored pose, so one motion form serves a sphere's `pos`, a capsule's `fromto`, and (ticket 03) a body of several geoms. The spin is about the mover's centre (`marker.centre` for a marker), whatever its offset.

## Invariants and checks

- `uv run pytest tests/test_motion.py`: path and oscillate poses at worked times, phase, spin, equivalent spellings, malformed motions; the region in the loader, marker test, and env.

## Known quirks

- The loader's validation helpers `_keys`, `_num`, `_vec` (and `_unit`) live here, and `level` imports them, since `level` imports this module anyway.
- `pose` gives no velocity. A kinematic body (ticket 03) that needs one for contact friction adds it here, beside `pose`, from the same formulas.
- An axis loads normalised only when its norm isn't already 1 (within 1e-12): a replay reloads the canonical level, and dividing a unit vector by its norm again moved the last bit of `[1, 1, 0]`'s components, so a `survive_raft` replay diverged at reset by 1.4e-17 in the body's angular velocity before this (ticket survive-movers/04).
- `_unit`'s norm is `sqrt(fsum(squares))`, not `np.linalg.norm`: NumPy computes a vector's norm as a BLAS `ddot`, and OpenBLAS's AVX512 kernel (CI's AMD EPYC 9V45 and 9V74 runners) rounded the last bit of a 3-decimal heading's norm differently from macOS arm64 and AVX2 hosts, so the stored hazard axis, and with it the level id, differed by host.

## Decisions

- 2026-10-22, ticket survive-movers/01: one pure function of (motion, sim time) for every mover (region now, hazards and bodies in tickets 02 and 03); nothing about a mover is Python-side state.
- 2026-10-22, ticket survive-movers/01: `phase` is a fraction of the period for both forms (rather than degrees for oscillate), and a path moves at constant speed by arc length rather than equal time per segment, so a drift speed is the path length over the period.
- 2026-10-22, ticket survive-movers/04: a tilt is a sine about one horizontal body-frame axis with its own period and phase, not a general orientation path: enough to rock a raft, and it composes with spin. Only a body's motion takes it (a marker test turns points by yaw only); a region rides a tilting body by attaching to it instead.
- 2026-10-22, ticket survive-movers/01, recorded in the survive-movers code review: a path's points and an oscillation are offsets from the mover's authored pose, not absolute positions (the ticket said "positions"), so one motion serves a sphere's `pos`, a capsule's `fromto` and a body of several geoms; and a path is walked at constant speed by arc length, not linearly in time per segment (the ticket's "interpolates linearly by sim time"), as the entry above states.

## Tried and rejected

## History

- 2026-10-22, ticket survive-movers/01: added, for the drifting survive region of `survive_drift`.
- 2026-10-22, ticket survive-movers/02: hazards use it too (`survive_sweepers`).
- 2026-10-22, ticket survive-movers/03: kinematic bodies reuse `pose`. Their velocity is the secant over one physics step (in `step.drive_bodies`), not an analytic derivative beside `pose`, which the Known quirks entry anticipated: it lands the integrator exactly on the next pose and needs no second set of formulas.
- 2026-10-22, ticket survive-movers/04: added `tilt` and `quat` for `survive_raft`; `_unit` keeps axis normalisation idempotent (see Known quirks).
- 2026-10-22, code review of survive-movers: `_keys`, `_num`, `_vec` moved here from `level`, which imports them, in place of function-level imports of `level`.
- 2026-10-23, fix-ci: `_unit` normalises with `math.fsum`, so an axis, and the level id it lands in, is the same on every CPU (see Known quirks). Same values on macOS arm64 and AVX2 x86_64, so no id or golden hash changed.
