# batch_core.py

## Purpose

The batched core: `num_envs` worlds of the CPU core ([step.md](../core/step.md)) on one Warp device, with each world's state read and written in the CPU core-state layout. It is the base of the GPU-batched backend (spec `gpu-backend`); this slice covers one model for every world (a plane, no bodies or tiles), the abstract tier, and the camera flag.

## Interface

Needs the `gpu` extra; importing `hamsterball.gpu.batch_core` without it raises `ImportError` naming the extra, and nothing else in the package imports it, so a plain install never loads `warp` or `torch`.

- `BatchCore(tier=None, levels=(FLAT,), num_envs=1, camera_action=False, device="cpu")`: `levels` is the bank (a `Level` or MJCF sections each); this slice needs them to compile to one model, else `NotImplementedError`. `device` is `"cpu"` or `"cuda:N"`; a missing CUDA device raises `RuntimeError`. A tier other than `AbstractTier` raises `NotImplementedError` until effort 1 adds its batched twin to `TIERS`.
- `reset(worlds=None, bank_index=None, spawn=(0, 0, R), facing=0.0)`: reset the given worlds like `Core.reset`; `bank_index` is stored in `level_index`.
- `step(action, n=FRAME_SKIP)`: `action` is `(num_envs, adim)`, numpy or torch, with `turn` and `cam_pitch` last under the camera flag; one action per world, held over `n` physics steps. On CUDA the `n` steps are captured once per `n` as one graph and relaunched; on the CPU device they run eagerly.
- `obs()`: the tier's observation after the last step, `{"wall_contact": bool array}` for the abstract tier.
- `state(world)` and `set_state(world, s)`: the `mjSTATE_INTEGRATION` vector `Core.state()` returns, at float32 precision, because Warp stores float32. A float32-exact vector round-trips exactly; any other comes back rounded to float32. `set_state` carries the solver warm start, as the CPU vector does, and the positions refresh at the next step.

Kernels, each named for its CPU function and in `physics_step`'s order: clear `xfrc_applied`, the position refresh (`mjw.kinematics`, `mjw.com_pos`), `turn_facing`, `camera_pitch`, `drag`, `contain` (impulse and hard stop), the tier's `drive` (see [abstract_tier.md](abstract_tier.md)), `mjw.step`; then the tier's `obs` once per env step. Constants are `core.step`'s and `core.model`'s, read from there.

## How it works

The CPU model is compiled once (`core.model.build`, so the ids, the view mocaps, and the userdata layout are the CPU's) and loaded with `mjw.put_model`; `mjw.make_data` allocates the worlds. Every kernel is one thread per world. The state vector is Warp's own `mjw.get_state`/`set_state` with `State.INTEGRATION`, whose layout matches C's here (time, qpos, qvel, warm start, `qfrc_applied`, `xfrc_applied`, mocap, userdata; 124 numbers on a flat level), so no mapping is needed yet. Padding and per-world slot mapping arrive with levels that differ.

## Invariants and checks

- One physics step from a shared state matches `Core`: `xfrc_applied` and the userdata slots at `rtol=1e-5, atol=1e-6`; qpos, qvel and time at the looser bound below. Cases: resting, running forward and sideways, inside the containment field, past the hard stop, airborne past the grace, facing wrapping at pi, pitch at both limits.
- Three env steps (30 physics steps) from a running state stay within the same bounds.
- `uv run pytest tests/test_batch_core.py` (skipped with a stated reason when the extra is missing); `HAMSTERBALL_WARP_DEVICE=cuda:0` runs the same tests on a GPU. `BatchCore(graph=False)` runs CUDA eagerly, for the graph-versus-eager test.

## Known quirks

- **Measured looser bound** (Warp CPU device, Warp 1.18.0, `mujoco-warp` 3.16.0, 2026-10-27): after one physics step the largest absolute error against `Core` was 1.6e-7 in qpos and 7.0e-7 in qvel (relative, 1e-6 or less) across the cases above; after 30 physics steps of running 1.1e-7 and 2.0e-6, and `xfrc_applied` 3e-5 on forces of about 20 N (1.5e-6 relative). The test bound for shell and hamster dofs is `rtol=1e-4, atol=1e-5`, 5 to 15 times the measurement. float32 against float64 and the solver's iteration account for it.
- `contain` calls no forward pass past the hard stop. The CPU calls `mj_forward` there; the only thing the rest of the step reads from it is fresh positions (`xpos`, `xipos`, `xmat`, `subtree_com`), and `mj_step` recomputes everything else. So the kernel refreshes positions again for every world after `contain`, which gives a world that did not move the same values back. It is two cheap kernels, not a per-world masked forward.
- **CUDA run** (2026-10-27, Warp 1.18.0, CUDA 13.4, an NVIDIA GeForce RTX 2080 Ti, sm_75, on a GPU cluster): `HAMSTERBALL_WARP_DEVICE=cuda:0 pytest tests/test_batch_core.py` passed, 16 of 16. The one-physics-step deviations from `Core` matched the CPU device's: qpos 1.6e-7 and qvel 7.0e-7 at most, `xfrc_applied` 1.1e-5 on forces of about 20 N, userdata 1.8e-7. The captured graph (three env steps of 10 physics steps) matched eager launches with a max difference of 0.0 in the state vector. Not bitwise against the CPU, as expected.
- `float32` time: `data.time` is float32 on Warp, so very long episodes lose resolution (at 1000 s a step is about 3 ulps). The vector env's episodes are far shorter.

## Decisions

- 2026-10-27, ticket gpu-backend/02: the tier hook is a class per tier in `TIERS` (CPU class to batched twin) supplying `drive`, `obs`, `reset`, and the arrays they read, launched by the core.
- 2026-10-27, ticket gpu-backend/02: state goes through `mjw.get_state`/`set_state` (the official layout), not a hand-built vector.

## Tried and rejected

- A masked forward pass for the hard stop (see Known quirks): refreshing positions is equivalent for the step and cheaper than learning whether Warp can mask `forward` per world.
- Zeroing the warm start in `set_state`: the CPU vector carries the warm start, so loading it is what keeps the two cores' first step comparable.

## History

- 2026-10-27, ticket gpu-backend/02: first version, a flat level, the abstract tier, the camera flag.
