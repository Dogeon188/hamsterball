# bank.py (gpu)

## Purpose

The **level bank** of the GPU-batched backend: a pool of levels compiled into one padded Warp model, the per-(level, task) tables each world reads when it resets, and the kernels that load a world's slots and draw its (level, task). [batch_core.py](batch_core.md) owns one; [vector_env.py](vector_env.md) draws from it.

## Interface

Needs the `gpu` extra (importing without it raises `ImportError` naming it).

- `Bank(levels, tasks=TASK_NAMES, keep=())` (host): `levels` is the pool, indexed by bank index; `tasks` limits the entries to those tasks; `keep` is a list of `(Level, task)` pairs appended after the pool that a running world still needs (they can't be drawn). An **entry** is one (level, task) pair; `entries`, `entry_of[bank index, task index]` (-1 for none), `first_entry`, `keep_entry`, and `n_pool` describe them; after `tables()`, `slot_ids` (each slot's geom id) and `row_slots` (per entry, the slot of each marker row) say where the env-step kernels write marker colours. A pool with seesaws raises `NotImplementedError` (ticket 09), as does a geom with `friction`, `solref` or `solimp` (see Decisions). Kinematic bodies are built (ticket 07): `nbodies` and `body_counts` are the body pool's size. Collapsing tiles are built (ticket 08): `ntiles` and `tile_counts` are the tile pool's size, and after `tables()` `tile_slots` (per entry, the geom slot of each tile) says where the tile kernels write colours.
- `Bank.mjcf()`, `Bank.body_mjcf()`, `Bank.tile_mjcf()`: the padded MJCF sections for `core.model.build`'s level and `bodies` arguments (the second is body_mjcf then tile_mjcf; each is "" without any). `Bank.tables(model)`: the slot tables as numpy.
- `DeviceTables(bank, model, device)`: the tables on the device, `load(mask, entry, model, data)` (the kernel that writes each masked world's slots), and the draw's candidate tables.
- Kernels `load_slots`, `seed_rng`, `draw`; constants `STATIC_TYPES`, `MARKER_TYPES`, `TASK_NAMES` (the goal one-hot's order), `BATCHED` (the model fields that carry a world axis), `PARK_DEPTH`, `PARK_MARGIN`.

## How it works

- **Slots.** One MJCF compiled from the bank, with one geom per slot, named `s/<type>/<i>` for a level's solids and `m/<type>/<i>` for its markers. The counts are the largest any entry holds: per solid type (`plane`, `hfield`, `box`, `sphere`, `capsule`) and per marker type (`sphere`, `capsule`, `box`; a task's markers, then the level's hazards, as `marker_rows`). Nothing is declared by level authors or the generator. Solids have contact on, markers off, as `Level.mjcf()` writes them.
- **Body pool** (ticket 07). `nbodies` free-joint body slots `body<k>` (the most bodies any entry's level has), compiled as the CPU compiles a body (`level._free_mjcf`: gravity compensated, 1000 kg, the same inertia) at the origin, excluded from the world and from each other. Slot k holds geom slots `b<k>/<type>/<j>` per type (box, sphere, capsule) at the most any entry's body k holds; their table rows are the body's geoms in the body's frame, and `load_slots` writes them like any slot. An entry whose level has fewer bodies or fewer geoms of a type parks the rest: margin -1000, alpha 0, and a position 1000 m below the body (a body's geom is dynamic, so it moves with its body; the body slot itself rests at the origin, unused, with a unit quaternion). The poses are driven by [bodies.py](bodies.md). Tiles join it as a second pool (below).
- **Tile pool** (ticket 08). `ntiles` free-joint slots `tile<k>` after the body slots (the CPU's order: bodies, then tiles), the most tiles any entry's level has, compiled as `Level.tile_mjcf` compiles a tile (`_free_mjcf`) at the origin, excluded from the world, the body slots and each other. A tile is one geom, so slot k holds one geom slot `t<k>/<type>/0` for each type any entry's tile k has; its table row is the tile's geom at its body's origin (the body's authored position is in [tiles.py](tiles.md)'s tables). Unused slots park as a body's do. A tile slot is `t<k>` in `slots`, so `slot[0][0] in "bt"` marks a dynamic slot.
- **Attached markers.** A marker attached to a body (survive's region on a raft or platform) is an ordinary marker row, with a marker slot in the world, authored in its body's frame; the env-step kernels pose that slot on the body from its qpos each step ([env_step.md](env_step.md)). The CPU draws it inside the body instead; the geom's world pose is the same.
- **Hfields.** Every distinct hfield in the bank is a model asset (deduplicated by size and samples); the hfield slots start on asset 0, and each world's `geom_dataid` points the slot at its own. Warp reads the hfield's size from `hfield_size[dataid]`, and the slot's `geom_size` only for bounds.
- **Per-world fields** (`BATCHED`, each `(num_envs, ngeom)`): `geom_pos`, `geom_quat`, `geom_size`, `geom_rgba`, `geom_margin`, `geom_dataid`, `geom_aabb`, `geom_rbound`. `load_slots` writes one world's row for every slot from its entry's table. A world-attached geom's world pose is computed once at `make_data` and skipped by forward, so the kernel also writes `data.geom_xpos` and `geom_xmat`. The bounds are the compiler's formulas (box: the half-diagonal; capsule: radius plus half-length; hfield: `sqrt(hx^2 + hy^2 + max(top, base)^2)` and the box of the slab; a plane keeps its compiled infinite ones), checked against the compiler's own values for every geom type in a scratch comparison (worst difference 8e-7).
- **Parked slots** (unused by an entry): the pose is `(0, 0, kill_plane_z - 1000)`, `geom_margin` is -1000, alpha is 0, the compiled size and bounds stay. Warp shares `geom_contype` across worlds, so the margin is what keeps a parked slot from touching the ball (ticket 01); setting an hfield's `geom_dataid` to -1 would not.
- **Draw.** `draw` runs one thread per masked world on a per-world `wp.uint32` RNG state (`seed_rng` seeds it from the host's seed). A forced level or task (`force_level`, `force_task`, -1 for free) is taken as given; a free level is uniform over the pool levels with an entry for the forced task (any entry if the task is free), a free task uniform over that level's entries. The result is the world's bank index, task index and entry.
- **Memory.** The shared model holds each distinct hfield once; the per-world cost is the batched fields (about 30 KB per world for the padded static and marker slots plus the shell's tile geoms; 38 KB with 81 tile slots) plus the Data (about 31 KB per world). Numbers at the default bank are in [batch_core.md](batch_core.md).

## Invariants and checks

- CUDA: the same tests passed on an NVIDIA GeForce RTX 2080 Ti on a GPU cluster (see [batch_core.md](batch_core.md)).
- `tests/test_bank.py`: round trips and one-step parity on worlds of different levels (plane, solids, hfield, floor, generated hfield) against `Core`; a world in the mixed bank steps as it does in a bank of its own level alone (`atol=1e-6`), which is what shows padding and parking change nothing.
- `tests/test_warp_levels.py`: draws cover the pool, a seed replays them, forced draws, errors, set_levels.

## Known quirks

- A moving marker's slot holds its authored pose in the tables; the env-step kernel `pose_movers` overwrites that slot's `geom_pos`, `geom_quat`, `geom_xpos` and `geom_xmat` each env step from the world's sim time ([env_step.md](env_step.md)), and the world's next reset reloads the authored pose. A capsule's table quaternion points its axis from the `fromto` start to its end, the compiler's the other way: the same capsule.

- A pool whose levels have no entry for the env's tasks (a survive-only level in a reach_goal bank) still has a bank index, and `env.levels[i]` is it, but it is never drawn and a forced draw onto it is an error.
- The caps grow with the largest level: one level with many boxes pads every world to that many.

## Decisions

- 2026-10-27, ticket gpu-backend/08: tile slots are a pool of their own after the body pool, not part of it: a tile's geom slot count per type is 0 or 1, its body is posed by [tiles.py](tiles.md) from a trigger slot, and the state map cuts the two pools separately.

- 2026-10-27, ticket gpu-backend/07: body slots are sized by the most bodies any level has, with geom slots per type per slot, and slot k is the same body slot in every level (bodies keep their order), so a level's CPU-layout state is the padded state with the unused slots' tails cut (see [batch_core.md](batch_core.md)).
- 2026-10-27, ticket gpu-backend/07: an attached marker's geom is a world marker slot posed on the body each step, not a geom inside the body slot: it needs no per-body marker geom slots, and the env-step kernels already pose marker slots.

- 2026-10-27, ticket gpu-backend/05: slots are per solid type at the bank's maximum, including `plane` and `hfield` (a plane slot and an hfield slot for today's levels), rather than special-casing terrain.
- 2026-10-27, ticket gpu-backend/05: tables are per entry (a level and a task), with the level's solids repeated for each of its tasks, so one kernel loads solids and markers together. A level of five tasks costs five rows of a few slots each.
- 2026-10-27, ticket gpu-backend/05: a geom with contact values raises. `geom_priority` and `geom_condim` are shared across worlds, so a slot can't be plain in one level and contact-valued in another; slots per (type, contact) would fix it, and no bundled or generated level has such a geom.
- 2026-10-27, ticket gpu-backend/05: `keep` entries let `set_levels()` leave a running world on its old level until it resets, instead of resetting every world.

## Tried and rejected

- Reading rbound and aabb from a compile of each level: it costs a full model compile per level, which a 1024-level bank can't afford.

## History

- 2026-10-27, ticket gpu-backend/07: the body pool and attached markers as rows of the marker table.

- 2026-10-27, ticket gpu-backend/05: first version.
