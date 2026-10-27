# tiles.py (gpu)

## Purpose

Collapsing tiles on the device: the kernels that pose each tile slot of the padded model from its trigger time, store the trigger when the shell touches a tile, and blend a touched tile's colour toward the warning colour. They are the batched twins of `drive_tiles` and `touch_tiles` in [step.py](../core/step.md) and of `HamsterballEnv._look_tiles` in [env.py](../env.md). [batch_core.py](batch_core.md) launches the first two in the physics step's order; the vector env launches the third with the marker looks ([vector_env.md](vector_env.md)). The slots (geom slots, parking, the state map) are in [bank.md](bank.md) and [batch_core.md](batch_core.md).

## Interface

Needs the `gpu` extra.

- `TileTables(bank, cpu_model, tier, device)`: per entry the tile count and delay, per (entry, slot) the authored position and the z offset a tile stops falling at, the tile slots' qpos and dof addresses, the first tile body's id (`body0`), the first trigger-time userdata slot (`first_slot`, after the core's and the tier's) and each tile's geom slot (`tile_slot`). Methods, each a no-op for a bank with no tile: `init(mask, entry, userdata)` (every tile of a masked world's level untouched, `+inf`), `drive(mask, entry, time, userdata, qpos, qvel)`, `touch(model, data, entry)`, `look(entry, time, userdata, slot_geom, t_rgba, geom_rgba)`.
- Kernels `init_tiles`, `drive_tiles` (one thread per world and tile slot), `touch_tiles` (one thread per contact slot of Warp's contact array), `look_tiles` (per world and tile slot), and the float64 helper `fall`.

## How it works

A tile slot a world's level uses sits at rest until `delay` after its trigger time (the userdata slot, `+inf` until touched), then falls at g, held `TILE_DROP` under the kill plane; its velocity is the secant to the pose one physics step later, as `drive_bodies`. It is float64 like `bodies.drive_bodies`, with `data.time` and the float32 trigger widened. The kernel runs at the start of each physics step (after `xfrc_applied` is cleared, with the bodies' drive, before the position refresh) and for the masked worlds at each reset, after `init_tiles`, as the CPU's `reset` does. Slots beyond the level's own tiles rest at the origin with a unit quaternion and parked geoms; their userdata slots stay 0.

`touch_tiles` runs right after `mjw.step` and reads the contacts the step leaves, which are those of the state the step started from: for each contact between the shell (body 1) and a tile body of the world's level whose slot is `+inf`, it stores the world's sim time (after the step, as the CPU does). Every contact of a tile writes the same value, so concurrent writes agree.

`look_tiles` writes each tile's `geom_rgba`: its authored colour (the bank's table) plus `clamp((t - trigger) / delay, 0, 1)` of the way to `TILE_WARN_RGBA`; an untouched tile is its own colour (`t - inf` is `-inf`), and a delay of 0 gives the warning colour throughout, as the CPU's `_look_tiles` does.

## Invariants and checks

- `uv run --extra gpu pytest tests/test_warp_tiles.py`: seam 1 (a bank of a 4-tile floor, boxes with a sphere and a capsule, the bundled 81-tile floor, a generated 63-tile floor and a level with no tile: CPU-layout round trips including `inf` triggers, one physics step against `Core` from the state just before a tile triggers and from states with tiles falling, the live fall against `Core` and the hold under the kill plane, a reset against `Core`) and seam 2 (recorded CPU episodes of collapse levels replayed through `step_to`, tile colours included, see [env_step.md](env_step.md)). `HAMSTERBALL_WARP_DEVICE=cuda:0` runs it on a GPU.

## Known quirks

- A ball that rests exactly on a tile (the spawn on a level of tiles at z = R, tile top at 0) touches it at distance 0.0. MuJoCo lists that as a contact and triggers the tile on the first physics step; Warp lists only contacts with distance below the margin, so the tile triggers on the next step, once gravity has sunk the ball a micron: a 1.7 ms later trigger. Tests that compare live physics start the ball 5 cm above the tiles.
- The secant velocity over the step where a tile reaches its bottom amplifies the float32 rounding of its stored trigger (`g * s * 2e-9 s / dt`, about 1e-5 m/s at a fall of 1.1 s); the tests' tile qvel bound is `atol=3e-5` for that reason.
- A world's tiles are the first tile slots of the pool, in the level's order (the CPU's `tile<i>`); a level with fewer tiles than the pool parks the rest.

## Decisions

- 2026-10-27, ticket gpu-backend/08: tiles are a second free-joint pool after the body pool, with a geom slot per type per tile slot (one geom per tile, so at most one slot of one type per tile), in the CPU's body order (bodies, then tiles), so the CPU-layout state is the padded one with each pool's unused tail cut ([batch_core.md](batch_core.md)).
- 2026-10-27, ticket gpu-backend/08: one trigger-time userdata slot per tile slot, `+inf` set at reset; userdata is the model's last segment and tile slots its last slots, so a level's userdata is a tail cut too.
- 2026-10-27, ticket gpu-backend/08: the warning colours are written by a kernel in `_look`, not on the host, from the state alone (sim time and trigger), so a re-render or `step_to` draws them the same.

## Tried and rejected

- Nothing tried and dropped yet.

## History

- 2026-10-27, ticket gpu-backend/08: first version.
