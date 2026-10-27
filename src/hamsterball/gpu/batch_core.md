# batch_core.py

## Purpose

The batched core: `num_envs` worlds of the CPU core ([step.md](../core/step.md)) on one Warp device, with each world's state read and written in the CPU core-state layout. It is the base of the GPU-batched backend (spec `gpu-backend`); this slice covers a bank of levels with planes, hfields, boxes, spheres, capsules and kinematic bodies and seesaws and collapsing tiles padded into one model ([bank.md](bank.md)), the abstract tier, and the camera flag.

## Interface

Needs the `gpu` extra; importing `hamsterball.gpu.batch_core` without it raises `ImportError` naming the extra, and nothing else in the package imports it, so a plain install never loads `warp` or `torch`.

- `BatchCore(tier=None, levels=(FLAT_LEVEL,), num_envs=1, camera_action=False, device="cpu", graph=True, tasks=all, keep=())`: `levels` is the bank, a list of `Level`s (the default is one plane, `FLAT_LEVEL`); kinematic bodies are driven by [bodies.py](bodies.md), collapsing tiles by [tiles.py](tiles.md), seesaws' fields set per world by [seesaws.py](seesaws.md). `tasks` and `keep` are for the vector env (see [bank.md](bank.md)). `bank` is the host `Bank`, `entry` the device array of each world's (level, task) entry. `device` is `"cpu"` or `"cuda:N"`; a missing CUDA device raises `RuntimeError` (the core maps Warp's `ValueError` for an unknown device). A tier other than `AbstractTier` raises `NotImplementedError` until effort 1 adds its batched twin to `TIERS`.
- `reset(worlds=None, bank_index=None, spawn=(0, 0, R), facing=0.0)`: reset the given worlds like `Core.reset` onto bank level `bank_index` (stored in `level_index`; default the level they are on), taking that level's first task's markers.
- `reset_masked(mask, spawns, facings)`: `reset` with Warp arrays on the device (a bool mask, vec3 spawns, facings), so it can run inside a captured graph; the vector env's same-step autoreset calls it. Each masked world loads the slots of its `entry` first (`load_levels(mask)`, also public: it rewrites the geometry and nothing else).
- `states()`: every world's state in the padded layout, `(num_envs, padded state size)`; the CPU layout when no level in the bank has bodies. `cpu_states()`: a list of each world's state in the CPU layout of its level (vectors differ in length when the levels' bodies do).
- `step(action, n=FRAME_SKIP, post=None)`: `action` is `(num_envs, adim)`, numpy or torch, with `turn` and `cam_pitch` last under the camera flag; one action per world, held over `n` physics steps. `post`, a callable that launches kernels, runs after the `n` steps inside the same graph (the vector env's env-step kernels); pass the same one every call, since the graph is kept per `n`. On CUDA the `n` steps are captured once per `n` as one graph and relaunched; on the CPU device they run eagerly.
- `launch(key, fn)`: runs `fn`, a callable that launches kernels, as one graph kept under `key` on CUDA and eagerly on the CPU device. The vector env uses it for the autoreset when views make it run apart from the step graph ([vector_env.md](vector_env.md)).
- `drag_k`: the shell's drag coefficient, `core.step.DRAG_K` unless set (a plain attribute read when a step launches; the feel test sets 0 to measure acceleration uncapped, before the first step, because a CUDA graph keeps the value it captured).
- `obs()`: the tier's observation after the last step, `{"wall_contact": bool array}` for the abstract tier.
- `set_states(states)`: `set_state` for every world: a `(num_envs, size)` array or a list of one vector per world, each in the CPU layout of the level that world is on (a wrong length raises `ValueError`).
- `state(world)` and `set_state(world, s)`: the `mjSTATE_INTEGRATION` vector `Core.state()` returns for the world's level, at float32 precision, because Warp stores float32. A float32-exact vector round-trips exactly; any other comes back rounded to float32. `set_state` carries the solver warm start, as the CPU vector does, and the positions refresh at the next step.

Kernels, each named for its CPU function and in `physics_step`'s order: clear `xfrc_applied`, `drive_bodies`, `drive_tiles`, the position refresh (`mjw.kinematics`, `mjw.com_pos`), `turn_facing`, `camera_pitch`, `drag`, `contain` (impulse and hard stop), the tier's `drive` (see [abstract_tier.md](abstract_tier.md)), `mjw.step`, `touch_tiles`; then the tier's `obs` once per env step. Constants are `core.step`'s and `core.model`'s, read from there.

## How it works

The padded model is compiled once (`core.model.build` on `Bank.mjcf()`, so the ids, the view mocaps, and the userdata layout are the CPU's) and loaded with `mjw.put_model`, with the geom fields of [bank.md](bank.md) given a world axis; `mjw.make_data` allocates the worlds. Every kernel is one thread per world. The state vector is Warp's own `mjw.get_state`/`set_state` with `State.INTEGRATION`, whose layout matches C's here (time, qpos, qvel, warm start, `qfrc_applied`, `xfrc_applied`, mocap, userdata; 124 numbers on a flat level), so a bank of static levels needs no mapping: the padded slots are geoms, which the integration state does not hold. A level with bodies has more qpos, qvel, warm start, `qfrc_applied` and `xfrc_applied` numbers (13 each driven body: 7 qpos, 6 dofs, and 6 more in each of the other three vectors; a hinge seesaw 1 qpos and 1 dof, a ball seesaw 4 and 3, and 6 more `xfrc_applied` each). The body slots are the last bodies, joints and dofs of the model (driven, then hinge, then ball), and a level's own bodies take the slots of their kind in order (`Bank.slot_of`), so a level's CPU vector is the padded one gathered by an index array: each segment's prefix (the shell, the hamster, the views), then each of the level's bodies at its slot's address. `state`, `set_state`, `cpu_states` and `set_states` map by these arrays (`_layout`, cached per sequence of body kinds and the tile count); an unused slot rests at the model's `qpos0` (a unit quaternion), with zero velocity and warm start. The tile slots (ticket 08) follow the body slots (free joints, 7 qpos, 6 dofs, 6 `xfrc_applied` each) and are the last userdata slots, one trigger time each; a level's tiles take the first tile slots in order, and the gather ends each segment with them (userdata: the tier's slots, then the level's tiles'). An unused tile's trigger slot is 0, a used one's is `+inf` from the reset until touched.

## Invariants and checks

- One physics step from a shared state matches `Core`: `xfrc_applied` and the userdata slots at `rtol=1e-5, atol=1e-6`; qpos, qvel and time at the looser bound below. Cases: resting, running forward and sideways, inside the containment field, past the hard stop, airborne past the grace, facing wrapping at pi, pitch at both limits.
- Three env steps (30 physics steps) from a running state stay within the same bounds.
- Feel (`tests/test_warp_feel.py`, ticket 11): the spec §5 measurements on the batched core within 3% of the spec values and the containment sink rows within 3 mm of the CPU core's, measured in the same test; about 2 minutes on the Warp CPU device, 10 s on a GPU.
- `uv run pytest tests/test_batch_core.py` (skipped with a stated reason when the extra is missing); `HAMSTERBALL_WARP_DEVICE=cuda:0` runs the same tests on a GPU. `BatchCore(graph=False)` runs CUDA eagerly, for the graph-versus-eager test.

## Known quirks

- **Mixed-bank parity** (Warp CPU device, 2026-10-27, one physics step from states 0, 20, 40, 70 and 120 env steps into a run, tests plus a sweep over six generated levels and the bundled hill): on plane, box, sphere, capsule and the bundled hill levels the largest difference from `Core` was 1.4e-6 in qpos and qvel. On rough generated hfield levels with 2 to 5 contacts it was 1.2e-5 in qpos and 7.0e-3 in qvel (shell linear 2.3e-3, angular 7.0e-3, hamster 7.0e-3), the same on a bank of that level alone, so it is Warp's hfield contact solution against the CPU's (issue 20's hfield drift), not padding; a world in a mixed bank equals the same world alone within 1e-6. The test bound for hfield qvel is `atol=2e-2`.
- **CUDA run of the bank tests** (2026-10-27, Warp 1.18.0, CUDA 13.4, an NVIDIA GeForce RTX 2080 Ti, sm_75, on a GPU cluster): `HAMSTERBALL_WARP_DEVICE=cuda:0 pytest -m gpu` passed, 79 of 79 (all gpu tests, with `tests/test_bank.py` and `tests/test_warp_levels.py` among them). The mixed-bank parity tests held at the same bounds as on the CPU device, including the rough generated hfield world, and a world in the mixed bank matched the same world alone within 1e-6. Largest observation difference against the CPU env: 5.7e-6 on the six-level bank, 3.8e-6 on the five-task bank.
- **Memory at the default bank** (1024 generated reach_goal levels, one hfield each of about 40 by 123 samples, 64 worlds): the padded model has one hfield slot and a goal sphere and a hazard box marker slot, 1024 hfield assets (33 MB of shared model), and per world about 31 KB of Data plus 30 KB of batched model fields, 61 KB, so 65,536 worlds take about 4 GB. Building the bank took 1.7 s plus 13 s for the generator to make the 1024 levels. These are CPU-side figures; a bank with more geoms per level pads every world to the largest.
- **Measured looser bound** (Warp CPU device, Warp 1.18.0, `mujoco-warp` 3.16.0, 2026-10-27): after one physics step the largest absolute error against `Core` was 1.6e-7 in qpos and 7.0e-7 in qvel (relative, 1e-6 or less) across the cases above; after 30 physics steps of running 1.1e-7 and 2.0e-6, and `xfrc_applied` 3e-5 on forces of about 20 N (1.5e-6 relative). The test bound for shell and hamster dofs is `rtol=1e-4, atol=1e-5`, 5 to 15 times the measurement. float32 against float64 and the solver's iteration account for it.
- `contain` calls no forward pass past the hard stop. The CPU calls `mj_forward` there; the only thing the rest of the step reads from it is fresh positions (`xpos`, `xipos`, `xmat`, `subtree_com`), and `mj_step` recomputes everything else. So the kernel refreshes positions again for every world after `contain`, which gives a world that did not move the same values back. It is two cheap kernels, not a per-world masked forward.
- **CUDA run** (2026-10-27, Warp 1.18.0, CUDA 13.4, an NVIDIA GeForce RTX 2080 Ti, sm_75, on a GPU cluster): `HAMSTERBALL_WARP_DEVICE=cuda:0 pytest tests/test_batch_core.py` passed, 16 of 16. The one-physics-step deviations from `Core` matched the CPU device's: qpos 1.6e-7 and qvel 7.0e-7 at most, `xfrc_applied` 1.1e-5 on forces of about 20 N, userdata 1.8e-7. The captured graph (three env steps of 10 physics steps) matched eager launches with a max difference of 0.0 in the state vector. Not bitwise against the CPU, as expected.
- `float32` time: `data.time` is float32 on Warp, so very long episodes lose resolution (at 1000 s a step is about 3 ulps). The vector env's episodes are far shorter.
- **Feel on the batched core** (2026-10-27, ticket gpu-backend/11; `tests/test_warp_feel.py`, the runs of `tests/test_feel.py` stepped one physics step at a time along +x, worlds side by side; Warp CPU device, and an NVIDIA GeForce RTX 2080 Ti, sm_75, Warp 1.18.0, on a GPU cluster). Spec §5 values against the measured:

  | measurement | spec | Warp CPU device | RTX 2080 Ti |
  |---|---|---|---|
  | acceleration 1 to 3 s, uncapped (m/s^2) | 4.06 | 4.090 (+0.7%) | 4.083 (+0.6%) |
  | speed at 3 s, half effort (m/s) | 5.6 | 5.598 (0.0%) | 5.576 (-0.4%) |
  | speed at 3 s, full effort (m/s) | 9.6 | 9.804 (+2.1%) | 9.820 (+2.3%) |
  | mean lean, 2 to 3 s (deg) | 54 | 53.94 (-0.1%) | 53.94 (-0.1%) |
  | top speed at 20 s (m/s) | 14.3 | 14.25 (-0.3%) | 14.30 (0.0%) |
  | coast loss (m/s^2) | 2.49 | 2.513 (+0.9%) | 2.503 (+0.5%) |
  | 20 degree climb in 3 s (m) | 5.1 | 5.168 (+1.3%) | 5.168 (+1.3%) |

  The containment sink after a throw at 0, 4, 8 and 12 m/s (worst of three directions, mm): 13.8, 16.5, 19.8, 22.8 on the CPU device and the same on CUDA, against 14, 17, 20, 23 on `Core` (within 1.2 mm; the test bound is 3 mm). Speed at 3 s on full effort is the row nearest its 3% bound; the CPU test holds every row to 2% and measured it within 1.3% on an M2.
- **Hfield contact-cap drift** (2026-10-27, ticket gpu-backend/11; `tests/measure_hfield_drift.py`, an NVIDIA GeForce RTX 2080 Ti on a GPU cluster, Warp 1.18.0, 200 drops per level, the shell dropped 0.6 m above the terrain's highest point at random spots inside 90% of the hfield and left still for 3 s, rest position on `Core` against the batched core). CUDA is not deterministic, so counts move by a few between runs (two runs: 16 and 13 spots over 1 cm at difficulty 0.3).

  | level | median dz | 90th percentile dz | spots with dz over 1 cm | spots with xy over 1 cm | dz over 1 cm where xy matched |
  |---|---|---|---|---|---|
  | `hill` (bundled) | 0.00 mm | 0.01 mm | 0 | 38 | 0 of 162 |
  | rolling bench terrain (120 m, 0.6 m high) | 0.04 mm | 0.68 mm | 0 | 45 | 0 of 155 |
  | generated reach_goal, difficulty 0.3 | 0.13 mm | 4.4 to 5.1 mm | 13 to 16 | 141 | 0 of 59 |
  | generated, difficulty 0.6 | 0.03 mm | 21 to 25 mm | 26 to 27 | 187 | 0 of 13 |
  | generated, difficulty 0.9 | 0.08 mm | 100 to 105 mm | 40 to 42 | 197 | 0 of 3 |

  Reading: on the one bundled hfield level and the smooth bench terrain the cap does nothing visible: rest heights agree to under 4 mm at every spot. On rough generated terrain most balls have not come to rest after 3 s (they roll down the slope), and a ball that rolls to a different place ends at a different height: those are the spots over 1 cm (up to 17 m, a ball that rolled off a different edge). Where both backends left the ball within 1 cm of each other in xy, no spot differed in height by over 1 cm. So the contact cap shows as trajectories that part on rough slopes, as issue 20 found (32 of 200 over 1 cm on Warp), not as a rest-height offset. Rerun: `uv run --extra gpu python tests/measure_hfield_drift.py --device cuda:0`.

## Decisions

- 2026-10-27, ticket gpu-backend/08: the tiles extend ticket 09's gather: `_layout` is keyed by the body kinds and the tile count, and appends the level's tiles at their slots' addresses after its bodies, with the first tile slots of userdata.

- 2026-10-27, ticket gpu-backend/09: with seesaws a level's body is not at its own index among the slots, so the tail cut is replaced by a gather per sequence of body kinds (see above); unused slots rest at `qpos0`.
- 2026-10-27, ticket gpu-backend/07: the CPU-layout state of a world with bodies is the padded state with each segment's tail cut (the body slots are the model's last), mapped by precomputed index arrays per body count; `states()` stays padded and `cpu_states()` is the portable form (the vector env's `set_levels` carries worlds across banks of different padding with it).

- 2026-10-27, ticket gpu-backend/02: the tier hook is a class per tier in `TIERS` (CPU class to batched twin) supplying `drive`, `obs`, `reset`, and the arrays they read, launched by the core.
- 2026-10-27, ticket gpu-backend/02: state goes through `mjw.get_state`/`set_state` (the official layout), not a hand-built vector.
- 2026-10-27, ticket gpu-backend/03: the vector env's env-step kernels join the physics graph through `step(post=)`, and its autoreset uses `reset_masked`, a device-array form of `reset`, so no host sync sits between a finished world and its reset.
- 2026-10-27, ticket gpu-backend/05: the model is padded from a `Bank` of `Level`s (the older MJCF-sections form of `levels` is gone), and `reset_masked` loads each world's level slots before the physics resets, so a level draw inside the graph needs no host step.
- 2026-10-27, ticket gpu-backend/11: the drag coefficient is an argument of the `drag` kernel (the core's `drag_k`), not a baked-in constant, so the feel test can measure acceleration uncapped as the CPU test does by patching `DRAG_K`; the value is read at launch, so it must be set before the first step on CUDA.

## Tried and rejected

- A masked forward pass for the hard stop (see Known quirks): refreshing positions is equivalent for the step and cheaper than learning whether Warp can mask `forward` per world.
- Zeroing the warm start in `set_state`: the CPU vector carries the warm start, so loading it is what keeps the two cores' first step comparable.

## History

- 2026-10-27, ticket gpu-backend/08: collapsing tiles: `drive_tiles` in the physics step and at reset, `touch_tiles` after `mjw.step`, tiles in the state gather. Memory per world with the bundled 81-tile floor in the bank (Warp CPU device, one world, arrays' allocated bytes): about 158 KB of Data (498 dofs, 408 geoms; 22 KB with no tiles) and 38 KB of batched model fields (30 KB with none), 196 KB against 53 KB, so 65,536 worlds of that bank take about 12.8 GB. The tile slots' dofs dominate: each adds 6 dofs and its geom.
- 2026-10-27, ticket gpu-backend/07: kinematic bodies: `drive_bodies` in the physics step and at reset, the CPU-layout state map, `cpu_states()`, list-form `set_states`.

- 2026-10-27, ticket gpu-backend/02: first version, a flat level, the abstract tier, the camera flag.
- 2026-10-27, ticket gpu-backend/03: added `post`, `reset_masked` and `set_states`; a nonexistent CUDA device now raises `RuntimeError` as documented (Warp raises `ValueError`, which the first version let through).
- 2026-10-27, ticket gpu-backend/05: the bank: padded slots, per-world level loading, `states()`, `keep` and `tasks`.
- 2026-10-27, ticket gpu-backend/10: added `launch`.
- 2026-10-27, ticket gpu-backend/11: the §5 feel test on the batched core, the hfield drift measurement, and the CUDA run (159 `gpu` tests passed on an NVIDIA GeForce RTX 2080 Ti; see Known quirks).
