# bank.py (gpu)

## Purpose

The **level bank** of the GPU-batched backend: a pool of levels compiled into one padded Warp model, the per-(level, task) tables each world reads when it resets, and the kernels that load a world's slots and draw its (level, task). [batch_core.py](batch_core.md) owns one; [vector_env.py](vector_env.md) draws from it.

## Interface

Needs the `gpu` extra (importing without it raises `ImportError` naming it).

- `Bank(levels, tasks=TASK_NAMES, keep=())` (host): `levels` is the pool, indexed by bank index; `tasks` limits the entries to those tasks; `keep` is a list of `(Level, task)` pairs appended after the pool that a running world still needs (they can't be drawn). An **entry** is one (level, task) pair; `entries`, `entry_of[bank index, task index]` (-1 for none), `first_entry`, `keep_entry`, and `n_pool` describe them; after `tables()`, `slot_ids` (each slot's geom id) and `row_slots` (per entry, the slot of each marker row) say where the env-step kernels write marker colours. A pool with bodies, seesaws or collapsing tiles raises `NotImplementedError` (tickets 07 to 09), as does a geom with `friction`, `solref` or `solimp` (see Decisions).
- `Bank.mjcf()`: the padded MJCF sections for `core.model.build`. `Bank.tables(model)`: the slot tables as numpy.
- `DeviceTables(bank, model, device)`: the tables on the device, `load(mask, entry, model, data)` (the kernel that writes each masked world's slots), and the draw's candidate tables.
- Kernels `load_slots`, `seed_rng`, `draw`; constants `STATIC_TYPES`, `MARKER_TYPES`, `TASK_NAMES` (the goal one-hot's order), `BATCHED` (the model fields that carry a world axis), `PARK_DEPTH`, `PARK_MARGIN`.

## How it works

- **Slots.** One MJCF compiled from the bank, with one geom per slot, named `s/<type>/<i>` for a level's solids and `m/<type>/<i>` for its markers. The counts are the largest any entry holds: per solid type (`plane`, `hfield`, `box`, `sphere`, `capsule`) and per marker type (`sphere`, `capsule`, `box`; a task's markers, then the level's hazards, as `marker_rows`). Nothing is declared by level authors or the generator. Solids have contact on, markers off, as `Level.mjcf()` writes them.
- **Hfields.** Every distinct hfield in the bank is a model asset (deduplicated by size and samples); the hfield slots start on asset 0, and each world's `geom_dataid` points the slot at its own. Warp reads the hfield's size from `hfield_size[dataid]`, and the slot's `geom_size` only for bounds.
- **Per-world fields** (`BATCHED`, each `(num_envs, ngeom)`): `geom_pos`, `geom_quat`, `geom_size`, `geom_rgba`, `geom_margin`, `geom_dataid`, `geom_aabb`, `geom_rbound`. `load_slots` writes one world's row for every slot from its entry's table. A world-attached geom's world pose is computed once at `make_data` and skipped by forward, so the kernel also writes `data.geom_xpos` and `geom_xmat`. The bounds are the compiler's formulas (box: the half-diagonal; capsule: radius plus half-length; hfield: `sqrt(hx^2 + hy^2 + max(top, base)^2)` and the box of the slab; a plane keeps its compiled infinite ones), checked against the compiler's own values for every geom type in a scratch comparison (worst difference 8e-7).
- **Parked slots** (unused by an entry): the pose is `(0, 0, kill_plane_z - 1000)`, `geom_margin` is -1000, alpha is 0, the compiled size and bounds stay. Warp shares `geom_contype` across worlds, so the margin is what keeps a parked slot from touching the ball (ticket 01); setting an hfield's `geom_dataid` to -1 would not.
- **Draw.** `draw` runs one thread per masked world on a per-world `wp.uint32` RNG state (`seed_rng` seeds it from the host's seed). A forced level or task (`force_level`, `force_task`, -1 for free) is taken as given; a free level is uniform over the pool levels with an entry for the forced task (any entry if the task is free), a free task uniform over that level's entries. The result is the world's bank index, task index and entry.
- **Memory.** The shared model holds each distinct hfield once; the per-world cost is the batched fields (about 30 KB per world for the padded static and marker slots plus the shell's tile geoms) plus the Data (about 31 KB per world). Numbers at the default bank are in [batch_core.md](batch_core.md).

## Invariants and checks

- CUDA: the same tests passed on an NVIDIA GeForce RTX 2080 Ti on a GPU cluster (see [batch_core.md](batch_core.md)).
- `tests/test_bank.py`: round trips and one-step parity on worlds of different levels (plane, solids, hfield, floor, generated hfield) against `Core`; a world in the mixed bank steps as it does in a bank of its own level alone (`atol=1e-6`), which is what shows padding and parking change nothing.
- `tests/test_warp_levels.py`: draws cover the pool, a seed replays them, forced draws, errors, set_levels.

## Known quirks

- A pool whose levels have no entry for the env's tasks (a survive-only level in a reach_goal bank) still has a bank index, and `env.levels[i]` is it, but it is never drawn and a forced draw onto it is an error.
- The caps grow with the largest level: one level with many boxes pads every world to that many.

## Decisions

- 2026-10-27, ticket gpu-backend/05: slots are per solid type at the bank's maximum, including `plane` and `hfield` (a plane slot and an hfield slot for today's levels), rather than special-casing terrain.
- 2026-10-27, ticket gpu-backend/05: tables are per entry (a level and a task), with the level's solids repeated for each of its tasks, so one kernel loads solids and markers together. A level of five tasks costs five rows of a few slots each.
- 2026-10-27, ticket gpu-backend/05: a geom with contact values raises. `geom_priority` and `geom_condim` are shared across worlds, so a slot can't be plain in one level and contact-valued in another; slots per (type, contact) would fix it, and no bundled or generated level has such a geom.
- 2026-10-27, ticket gpu-backend/05: `keep` entries let `set_levels()` leave a running world on its old level until it resets, instead of resetting every world.

## Tried and rejected

- Reading rbound and aabb from a compile of each level: it costs a full model compile per level, which a 1024-level bank can't afford.

## History

- 2026-10-27, ticket gpu-backend/05: first version.
