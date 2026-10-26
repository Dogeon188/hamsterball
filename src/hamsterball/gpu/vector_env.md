# vector_env.py (gpu)

## Purpose

The Warp vector env: `gymnasium.make_vec(<id>, num_envs=N, vectorization_mode="vector_entry_point", backend="warp", ...)` returns a `gymnasium.vector.VectorEnv` that steps N worlds of one level on one Warp device, with actions, observations, rewards and flags as float32 torch tensors on that device. It wraps the batched core ([batch_core.md](batch_core.md)) with the env-step kernels ([env_step.md](env_step.md)). The CPU env ([env.md](../env.md)) stays the reference.

## Interface

Needs the `gpu` extra (`mujoco-warp`, `warp-lang`, `torch`); importing the module without it raises `ImportError` naming the extra, and `make_vec(backend="warp")` imports it lazily, so a plain install never loads Warp or torch.

- `WarpVectorEnv(num_envs, tasks=TASKS, levels=None, tier="abstract", camera_action=False, views=(), view_size=(84, 84), obs_state=True, frame_skip=10, time_limit_s=None, truncate=True, shaped=False, shaping_gamma=0.99, model_cache=16, difficulty=0.5, bank_size=1024, device="cuda:0")`: the CPU env's kwargs plus `bank_size` and `device`. `model_cache` and `bank_size` are accepted and unused so far (registered ids pass `model_cache`; the bank comes with generator pools). `device` is `"cuda:N"`, or `"cpu"` for Warp's CPU device (tests); a CUDA device that doesn't exist raises `RuntimeError`.
- `single_action_space`, `single_observation_space`, and the batched `action_space` and `observation_space` equal the CPU env's for the same kwargs (`Dict(state=Box(21), goal=Box(10))`). `metadata["autoreset_mode"]` is `SAME_STEP`.
- `reset(seed=None, options=None)`: resets every world to the level's spawn; returns `(obs, info)` with `info["level_index"]` and `info["task_index"]`. `options` raises `NotImplementedError` until the level bank exists.
- `step(action)`: `action` is `(num_envs, action dim)`, a torch tensor or array, with `turn` and `cam_pitch` last under the camera flag. Returns `(obs, reward, terminated, truncated, info)` as tensors: `obs` a dict `state` `(N, 21)` and `goal` `(N, 10)`, `reward` float32, the flags bool. `info`: `success` (bool), `failure_reason` (int32, the code table in [env_step.md](env_step.md)), `reward_terms` (`event`, `time`, `shaping`), `level_index`, `task_index`, and for the same-step autoreset `final_obs` (a dict of `(N, ...)` tensors, valid where the mask is true), `final_info` (`success`, `failure_reason`, `reward_terms` of the step that ended), and the bool masks `_final_obs` and `_final_info`. A world that finished has already reset when `step` returns: `obs` is its next episode's first observation, while `reward`, the flags and `info` describe the step that ended it.
- `step_to(core_states)`: `(num_envs, state size)` CPU-layout core states (numpy or torch), loaded into the worlds, then the env-step logic and autoreset with no physics; returns what `step` returns. It is the batched twin of `HamsterballEnv.step_to`, and the seam the parity tests use.
- The returned tensors are views of buffers the next `reset`, `step` or `step_to` overwrites; clone what you keep.
- Supported now, and the rest raises: one level (a file or `Level`), task `reach_goal`, the abstract tier, no views, still hazards. A generator pool, several levels, other tasks, views, `obs_state=False`, `tier="articulated"`, a body or tile level, and a moving hazard raise `NotImplementedError` (`ValueError` for `obs_state=False` and a non-default `difficulty`, as on the CPU); an unknown kwarg is a `TypeError`.
- `env.level` and `env.levels` (`[level]`): the bank, one entry so far.

## How it works

`BatchCore` holds the physics. `step` copies the action, then `BatchCore.step(..., post=self._env_step)` launches the n physics steps and `_env_step` in one captured graph on CUDA (eager on the CPU device): refresh the derived quantities (`mjw.kinematics`, `com_pos`, `com_vel`, `subtree_vel`, which the CPU's `_observe` also recomputes), the tier's `obs`, the `advance` kernel, `observe` into `state` and `goal`, `stash_final` for finished worlds, `BatchCore.reset_masked` on the finished worlds, `reset_episode`, and `observe` again for those worlds. Nothing syncs with the host. The step limit is `ceil(time_limit_s / (frame_skip * dt))`, as on the CPU. Marker looks, pixel views and the level draw are not built in this slice.

## Invariants and checks

- Seam 2: `tests/test_warp_env.py` records CPU episodes (goals and hazards as sphere, capsule and box, falling off, truncation, `truncate=False`, shaping), replays their core states through the CPU-recorded and Warp `step_to`, and compares observations and rewards at `rtol=1e-5, atol=1e-5` and flags, success and failure code exactly; a finished world's `final_obs` equals the CPU's last observation.
- `uv run --extra gpu pytest tests/test_warp_env.py` (skipped with a stated reason without the extra); `HAMSTERBALL_WARP_DEVICE=cuda:0` runs it on a GPU.

## Known quirks

- **Parity measured** (Warp CPU device, 2026-10-27, eight scenarios, 1 to 360 steps each): the largest observation difference was 5.7e-6 (float32 against the CPU's float64 on positions near 10 m), the largest reward difference 4.7e-8; flags, success and failure codes matched on every step.
- `info["failure_reason"]` is a tensor of codes, not the CPU's string.
- The tensors are views, see Interface.
- **CUDA run** (2026-10-27, Warp 1.18.0, an NVIDIA GeForce RTX 2080 Ti, sm_75, on a GPU cluster): `HAMSTERBALL_WARP_DEVICE=cuda:0 pytest -m gpu tests/test_warp_env.py tests/test_batch_core.py` passed, 38 of 38 (22 here, 16 of the batched core). The env step ran as one captured graph (physics, env-step kernels, autoreset). Parity differences matched the CPU device's: largest observation difference 5.7e-6, largest reward difference 4.7e-8.

## Decisions

- 2026-10-27, ticket gpu-backend/03: `info` and `final_info` hold the step that ended an episode; `final_obs` is a dict of `(N, ...)` tensors with masks rather than the CPU vector env's object array, since a per-world object array would force a host copy.
- 2026-10-27, ticket gpu-backend/03: one level and one task, so there is no device RNG and no reset options yet; both arrive with the level bank (ticket 05).
- 2026-10-27, ticket gpu-backend/03: a CUDA device that doesn't exist raises `RuntimeError` from `BatchCore`, which maps Warp's `ValueError` for an unknown device (ticket 02's doc had assumed it raised `RuntimeError`).

## Tried and rejected

- Nothing yet.

## History

- 2026-10-27, ticket gpu-backend/03: first version.
- 2026-10-27, ticket gpu-backend/03: CUDA run on an NVIDIA GeForce RTX 2080 Ti passed (see Known quirks).
