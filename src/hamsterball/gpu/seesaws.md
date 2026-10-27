# seesaws.py (gpu)

## Purpose

Seesaws (a body on a passive hinge or ball joint, `level.py`'s `_seesaw_mjcf`) in the padded batch model: the per-world joint and mass fields of the bank's seesaw slots, and the frames an attached marker reads from a seesaw's joint. The slots themselves (the body pool, geom slots, the CPU-layout state map) are in [bank.md](bank.md) and [batch_core.md](batch_core.md); the free-joint bodies are [bodies.md](bodies.md).

## Interface

Needs the `gpu` extra.

- `SeesawTables(bank, cpu_model, device)`: each entry's seesaw values on the device, `load(mask, entry, model)` (the kernel `load_seesaws`, one thread per world and slot; launches nothing for a bank without seesaws), and `meaninertia` per entry.
- `attach_frames(bank, slot_qadr)`: per entry, per level body, `(qadr, frame, pivot, axis)`: where an attached marker finds its body in qpos and how to read it (`env_step.FREE`, `BALL`, `HINGE`). `vector_env` gives it to `env_step.Markers`.
- `SEESAW_FIELDS`: the model fields that get a world axis (`jnt_axis`, `jnt_stiffness`, `jnt_range`, `dof_damping`, `body_pos`, `body_mass`, `body_inertia`, `body_ipos`, `body_iquat`, `body_subtreemass`, `body_invweight0`, `dof_invweight0`); `BatchCore` adds them to `put_model`'s `batch_sizes` only when the bank has a seesaw.

## How it works

- **Slots.** One hinge slot per hinge seesaw a level can have at most, then one ball slot per ball seesaw, after the free-joint slots (`hinge0`, `ball0`; `Bank.body_names`). A level's n-th hinge seesaw takes the n-th hinge slot (`Bank.slot_of`). Joint type and `jnt_limited` are shared across worlds (ticket 01), so a hinge and a ball are separate slots and a slot a level doesn't use keeps the compiled joint: a range of 170 degrees, so the limit never acts, and a unit mass.
- **Per-world fields.** Every field a seesaw needs batches per world (ticket 01). At a reset `load_seesaws` writes the world's entry row into the slot's joint (axis, stiffness, range in radians), dofs (damping on each) and body (pivot `body_pos`, mass, inertia, `body_ipos`, `body_iquat`). Nothing is shared except `jnt_type` and `jnt_limited`.
- **Derived fields.** After writing mass or inertia ticket 01 says to call `mjw.set_const`. The port instead computes the derived `body_subtreemass`, `body_invweight0` and `dof_invweight0` on the host when the bank is built, and loads them like the rest: `mjw.set_const` also writes `qpos` from `qpos0` and runs a forward pass over all worlds, which can't run inside the autoreset's graph. Each distinct seesaw is compiled once alone in a scratch `MjModel` (a seesaw's tree is its own, so its mass, inertia and invweight0 don't depend on the rest of the level; attached markers weigh nothing and are left out), which also gives the inertia's centre and principal axes exactly as the CPU compiles them. `body_subtreemass[0]` (the world) is not rewritten; nothing reads it for a world-attached tree.
- **Mean inertia.** MuJoCo scales the solver's tolerance by the mean of the mass matrix's diagonal at qpos0 (`stat.meaninertia`), which depends on the level's bodies. Warp keeps one per world (`model.stat.meaninertia` is replaced by a `(num_envs,)` array when the bank has seesaws), and `load_seesaws` writes the entry's value: the shell and hamster dofs, the level's driven bodies and its seesaws, as the CPU model sums them.
- **Ball slot centre of mass.** The ball slot is compiled with its centre of mass off the joint. MuJoCo calls a ball body with its centre at the joint "simple" (a diagonal 3 by 3 block in the mass matrix) and Warp builds the matrix's sparsity once from that; a level whose seesaw has an off-centre mass (a sphere on a plate) then ran with a diagonal block and a velocity error of 1e-4 after one step. With the slot non-simple, the dense block holds zeros for a centred seesaw and the right values for the rest.
- **Attached markers.** A marker attached to a seesaw rides it as `level.body_frame` says: the pose is the fixed pivot, turned by the ball's quaternion in qpos, or by the hinge angle about its unit axis (`(cos(a/2), sin(a/2) axis)`); a driven body's is its free joint. `env_step`'s `frame_pos` and `frame_quat` read it by the row's `frame`; `to_body`, `centre_at` and `pose_movers` go through them.

## Invariants and checks

- `uv run --extra gpu pytest tests/test_warp_seesaws.py` (`HAMSTERBALL_WARP_DEVICE=cuda:0` runs it on a GPU). Seam 1, on a bank of two hinge levels (different axis, stiffness, damping, range and mass), a ball seesaw level, one with a two-geom seesaw whose centre of mass is off its pivot, a ball seesaw before a driven body and a hinge seesaw after one, a raft and a floor: every level's CPU state round trips exactly, one physics step matches `Core` (`rtol=1e-4, atol=1e-5`), a world equals the same level in a bank of its own, each world's model fields equal the CPU model's, a reset equals the CPU reset. Seam 2: CPU episodes of the bundled ball seesaw, a hinge seesaw, a generated seesaw level, each tipped off (still ball, `left_region`) and balanced to time-up (chasing the region), and a seesaw after a driven body, replayed through `step_to`, alone and six together on one bank; a generator pool (seesaw layout among the others) steps and resets on the device.
- Measured (Warp CPU device): one physics step from states 0, 30 and 150 env steps into a run, the largest difference from `Core` in a seesaw's qpos and qvel was 2.8e-11, 9.2e-8 and 7.1e-7; the observation in seam 2 at most 1.9e-6, the reward 0, the attached region's world pose 1.6e-7 (eight scenarios of up to 720 steps, flags, success and failure codes exact).
- CUDA: see History.

## Known quirks

- A pool whose levels hold several seesaws pads every world to the most of each kind; `Bank.slot_of` hands out slots in body order.
- The attached frame of a seesaw uses float32 trigonometry for a hinge angle (the CPU's is float64): within 1.6e-7 on the region's pose.

## Decisions

- 2026-10-27, ticket gpu-backend/09: derived invweight0, subtree mass and mean inertia come from the host, per distinct seesaw, not from `mjw.set_const` at each reset (see How it works).
- 2026-10-27, ticket gpu-backend/09: hinge and ball seesaws take slots of their own after the free-joint slots, and a level's body k is not slot k any more: `Bank.slot_of` maps level bodies to slots by kind, and the state map is built per sequence of body kinds (`BatchCore._layout`), not by cutting tails.

## Tried and rejected

- A ball slot compiled with its centre at the joint, as the free-joint slots are: Warp's mass-matrix sparsity then had a diagonal ball block, and a seesaw with an off-centre mass stepped with a 1e-4 velocity error.
- Calling `mjw.set_const` per reset: see How it works.

## History

- 2026-10-27, ticket gpu-backend/09: first version.
