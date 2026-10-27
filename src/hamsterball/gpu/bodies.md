# bodies.py (gpu)

## Purpose

Kinematic bodies (pillars, driven platforms, a tilting raft) on the device: the kernel that poses each body slot of the padded model from its motion and the world's sim time. It is the batched twin of `drive_bodies` in [step.py](../core/step.md), and [batch_core.py](batch_core.md) launches it in the physics step's order. The slots themselves (their geom slots, parking, the CPU-layout state map) are in [bank.md](bank.md) and [batch_core.md](batch_core.md).

## Interface

Needs the `gpu` extra.

- `BodyTables(bank, cpu_model, device)`: the bank's bodies' motions on the device (`motion`, a `BodyMotion` struct), `nbodies` per entry, and `qadr` / `dadr`, the qpos and dof address of each body slot in the compiled model (slots are consecutive free joints). `BodyTables.drive(mask, entry, time, qpos, qvel)` launches `drive_bodies` for the masked worlds; a bank with no body launches nothing.
- Kernel `drive_bodies` (one thread per world and body slot), and its float64 helpers `offset_at`, `quat_at`, `mul_quat`, `sub_quat`, each named for the `motion` or MuJoCo function it ports.
- `BodyMotion` rows are `entry * nbodies + slot`: motion kind (path or oscillate), period, phase, spin (rad/s), oscillation vector (axis times amplitude), authored position, a path's slice of the closed polyline and its cumulative lengths, and the tilt (axis, amplitude, period, phase).

## How it works

A body slot a world's level uses is posed at `pos + offset(t)` with orientation `quat(t)` (yaw about z, then the tilt about the body's own horizontal axis), and given the secant velocity to the pose one physics step later: `(offset(t + dt) - offset(t)) / dt`, and `subQuat(q1, q0) / dt` for the angular part, in the body frame as a free joint's qvel is. Slots beyond the level's own bodies are not touched: they rest at the origin with a unit quaternion and their geoms are parked ([bank.md](bank.md)). The kernel runs for every world at the start of each physics step (after `xfrc_applied` is cleared, before the position refresh, as on the CPU) and for the masked worlds at each reset, where the CPU's `reset` also drives the bodies at sim time 0.

All of it is float64, including the time (`data.time` is float32, widened). The secant is a difference of two poses a physics step apart, so in float32 it would carry a velocity error of about 6e-4 m/s on a pose of 10 m; in float64 the error is the float32 storage of the result.

## Invariants and checks

- `uv run --extra gpu pytest tests/test_warp_bodies.py`: seam 1 (a mixed bank of a floor, two bars, a raft, three pillars and a platform; round trips of the CPU-layout state, one physics step against `Core` at `rtol=1e-5, atol=1e-6` on each body's qpos and qvel, padding invariance, the reset pose) and seam 2 (recorded CPU episodes replayed through `step_to`, see [env_step.md](env_step.md)). `HAMSTERBALL_WARP_DEVICE=cuda:0` runs it on a GPU.
- Measured (Warp CPU device, one physics step from states 0, 400 and 2000 env steps into a run, on all five levels): the largest difference from `Core` in a body's qpos was 1.2e-7, 5.4e-7 and 3.3e-6, in its qvel 1.5e-8, 2.3e-7 and 8.4e-7. The growth with sim time is the float32 time: a state's time loads rounded to float32 (an ulp of 3.8e-6 s at 33 s), which moves a body moving at 1 m/s by about that much times its speed.

- CUDA run (2026-10-27, Warp 1.18.0, an NVIDIA GeForce RTX 2080 Ti, sm_75, on a GPU cluster): `HAMSTERBALL_WARP_DEVICE=cuda:0 pytest -m gpu` passed 117 of 117 on the merged tree (the 16 tests of `tests/test_warp_bodies.py` among them), the body, observation and region-pose differences within the CPU device's bounds (observation 1.4e-6 at most, region pose 3.7e-7).

## Known quirks

- A world's sim time is Warp's float32 `data.time`, which accumulates one physics step's rounding at a time, so a long episode's bodies drift in phase from the CPU's (the CPU's time is float64). At the episode lengths in use (under a minute) it is below the test's tolerance on a loaded state, and Warp's own time keeps the drift ([batch_core.md](batch_core.md)).
- Only free-joint bodies. A seesaw (a body with a `joint`) raises in the bank; ticket 08 adds it, and tiles (ticket 09) join the same slot pool.

## Decisions

- 2026-10-27, ticket gpu-backend/07: the motion formulas are re-implemented in float64 here and not shared with the marker kernels' float32 `offset_of`: markers need no velocity, and the body's secant does.
- 2026-10-27, ticket gpu-backend/07: one thread per (world, slot) with a per-entry body count, rather than a loop over slots in a world's thread, so a bank of many bodies (pillars, later tiles) runs in parallel.

## Tried and rejected

- Float32 offsets and an analytic derivative of the offset in place of the secant: the spec fixes the secant (the CPU's friction sees it), and an analytic velocity differs from it by O(dt) on a path's corners.

## History

- 2026-10-27, ticket gpu-backend/07: first version. CUDA run on an NVIDIA GeForce RTX 2080 Ti on a GPU cluster: 117 of 117 `gpu` tests passed (see Invariants and checks).
- 2026-10-27, ticket gpu-backend/08: tiles joined as a pool of their own after the body pool, not inside it (see tiles.md and bank.md); the Known quirks line above that says tiles join the body pool is superseded.
