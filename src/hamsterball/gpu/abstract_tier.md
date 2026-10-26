# abstract_tier.py (gpu)

## Purpose

The abstract tier's batched twin: Warp kernels for `drive` and `obs`, line for line from [`core/abstract_tier.py`](../core/abstract_tier.md), plus the tier's part of a masked reset. Constants (gains, grace, lean, tile normals) are read from the CPU module, so one definition serves both.

## Interface

`WarpAbstractTier(tier, device)` is what [batch_core.py](batch_core.md) builds from `AbstractTier` via `TIERS`: `reset(nworld, mask, spawn, data)`, `drive(nworld, action, data)`, `obs(nworld, data, out)`, `obs_arrays(nworld, device)`. The kernels are `drive`, `obs`, `reset`; one thread per world, the action's first two columns are the tier's.

## How it works

`drive` reads `xpos`, `xipos`, `xmat`, `qvel`, `time` and the userdata slots and adds the foot-force pair and grip couple to `xfrc_applied` as `spatial_vector`s (force on top, torque below), after the airborne cut over the 320 tile normals (a device array). It needs positions the core refreshed. `obs` writes `wall_contact`. `reset` puts the hamster 1 cm above the bottom of the shell at `spawn` and sets the last touch to now.

## Invariants and checks

Checked through the batched core's parity test (`tests/test_batch_core.py`): `xfrc_applied` and the userdata slots match the CPU's at `rtol=1e-5, atol=1e-6` for resting, running, sideways, and airborne states.

## Known quirks

- The airborne cut compares a gap against `TOUCH` (5 mm): a state within float32 rounding of that edge can cut on one backend and not the other. The test states stay about 5 mm clear of it.

## History

- 2026-10-27, ticket gpu-backend/02: first version.
