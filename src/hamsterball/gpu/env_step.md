# env_step.py (gpu)

## Purpose

The env-step kernels: what `HamsterballEnv._advance` and `_observe` ([env.md](../env.md)) do after the physics, one thread per world, so the Warp vector env ([vector_env.md](vector_env.md)) runs them inside the step graph. It covers all five tasks on still markers; moving markers, attached markers and the previous sim time they need come with later tickets of the `gpu-backend` effort.

## Interface

Needs the `gpu` extra. Nothing but [vector_env.py](vector_env.md) calls it.

- `targets(task, block)`: the task's targets in task order, `(key, marker)`: keys before the goal, waypoint i before i + 1, survive's region.
- `Markers(entries, device)`: the marker tables on the device, one block of `stride` rows per bank entry (`entries` is a list of `(task, block, hazards)`): the task's targets in task order, then the hazards, the rest padding. Per entry: `ntargets`, `n` (targets plus hazards), the task's `mode` (`ALL`, `IN_ORDER`, `GOAL_LAST`: how targets become live), `goal_row`, `survive`, `time_penalty`; per row, its look (`role`, `base` colour). Each row is a sphere, a capsule (a sphere is a capsule of zero length), or a box, in the form `core.marker` reads (the box's frame is built with the same `mju_euler2Quat` and `mju_quat2Mat` calls).
- `FAILURE_CODES = ("none", "hazard", "kill_plane", "left_region")`: the table for `info["failure_reason"]`, an int32 per world, the index into this tuple. `left_region` is survive's.
- Kernels, named for the CPU function they port: `advance` (`_advance`), `observe` (`_observe`'s `state` and `goal`), `look_markers` (`_look`), `stash_final` and `reset_episode` (autoreset), `set_spawn` (the spawn pose of the masked worlds from their entry's table), `either` (an or of two masks).
- Helpers: `seg_seg_dist`, `box_dist`, `seg_box_dist`, `touches` (`core.marker`'s `_seg_seg_dist`, `_seg_box_dist` and `crosses` with reach `R`), `inside` (`crosses` at reach 0 on a point: survive's region test), `live` (`_live`), `nearest` (`_target`).
- `advance`, `observe`, `reset_episode` and `look_markers` read each world's entry (the bank's (level, task) pair, see [bank.md](bank.md)) for its marker rows, mode, kill plane height and step limit; the targets-left array is `(num_envs, max targets)`, padded with False.

## How it works

- `advance` adds a step. Survive tests the region (row 0) at the shell centre (`inside`) and, once the step count reaches the limit with the centre inside, removes it: the time-up success. Every other task visits the targets in row order and removes each live one that the segment from the previous env step's shell centre to the current one touches with reach `R` (a waypoint only when it is the first left, the unlock goal only when no key is left; +1 per item, key or waypoint, the goal's own +1 comes with success). The CPU reruns until nothing new triggers so one step can complete several targets; here a target's liveness depends on earlier rows only, so one pass in row order reaches the same fixed point. No targets left is success, `+1`. Otherwise the hazards (`fail = 1`), then the centre's height against the kill plane (`fail = 2`), then survive's centre outside its region (`fail = 3`); any is `-1`. Success wins over failure, as on the CPU. The time penalty is `-0.001` (0 in survive), the shaping term `gamma * phi' - phi` with `phi' = 0` at a terminated state and else the potential to the nearest live target, and `truncated` is `truncate and not terminated and steps >= max_steps`. It writes the reward, its three terms, the flags, and the episode's step count, previous position, potential and targets left.
- The box test is the CPU's golden-section search along the segment to a 1e-6 m bracket, in float32 (capped at 64 passes, which only a degenerate bracket would reach). The CPU's reach-0 slab test of a box is not ported: survive's region test is a point, which `inside` does as the slab test does for a segment of zero length.
- `observe` writes the 21-number `state` and 10-number `goal` for the worlds in a mask, from the refreshed `xpos`, `xmat`, `subtree_com` and `subtree_linvel`, in the facing frame. The goal's target is the nearest live target's centre (a capsule's midpoint) to the shell; its fraction is the targets left over all of them, or survive's time left over the limit. Once a world has succeeded its goal vector is zeros apart from the task one-hot.
- `look_markers` writes the episode task's marker colours into the world's `geom_rgba` from the targets left: a collected item or key alpha 0, a waypoint locked until it is next and passed after, the unlock goal locked until every key is in, the rest in their kind's colour. It is derived from the targets left each env step, after the autoreset, so a world that just reset shows its next episode's colours (the CPU's `_look` calls change them as targets fall). A world's marker slots hold only its own task's markers, so there are no other tasks' markers to hide.
- Per-world episode state, all device arrays owned by the vector env: the targets left (a bool array over the task's targets, a bitmask's equal without a cap on the count), step count, previous shell position, potential. The previous sim time the spec lists is not kept: still markers don't read it.

## Invariants and checks

- Against the CPU env through `step_to`, on recorded episodes: observations and rewards at `rtol=1e-5, atol=1e-5`, flags, success and the failure code exactly, the marker colours on every unfinished step. `uv run --extra gpu pytest tests/test_warp_env.py tests/test_warp_tasks.py`: reach_goal in the first, the other four tasks (success, hazard, kill plane, left region, truncation, several targets in one step, waypoints touched out of order, the camera flag, shaping, `frame_skip`, `truncate=False`) and the failure order in the second.
- A kernel that changes here must keep the CPU function it names in step, the way the core-step kernels do ([batch_core.md](batch_core.md)): a change to `_advance` or `_observe` fails `test_step_to_matches_the_cpu_env_on_recorded_episodes` until the kernel matches.
- A `look_markers` change must keep `HamsterballEnv._look` calls in step: the same parity test compares colours.

## Known quirks

- float32: a segment whose closest approach to a marker lies within about 1e-5 m of the touch threshold may trigger on one backend and not the other (the CPU is float64, and its state loads into Warp rounded to float32). The parity test's scenarios avoid it; the spec allows the exception.
- `info` arrays that the CPU gives as Python values (`failure_reason` a string or `None`) are int codes here.

## Decisions

- 2026-10-27, ticket gpu-backend/04: the targets-left state is a bool array over the padded target list, not a packed bitmask; the spec says bitmask, and this needs no cap on the item count and no bit arithmetic in the kernels.
- 2026-10-27, ticket gpu-backend/04: the CPU's rerun-until-nothing-new loop is one ordered pass, since liveness reads earlier rows only (waypoint i after waypoint i - 1, the goal after the keys), so the fixed point is the same.
- 2026-10-27, ticket gpu-backend/04: marker colours are derived from the targets left after the autoreset rather than written when a target falls, so they cost one small kernel and need no event log.
- 2026-10-27, ticket gpu-backend/03: the marker table has the goal and hazards as rows of one table, tested by one function, so later tasks add rows and a bitmask over them rather than new tests.
- 2026-10-27, ticket gpu-backend/03: the kernels live in the graph, so `advance` always runs on all worlds and masks nothing; a finished world's results are stashed and its reset follows in the same graph.

## Tried and rejected

- Nothing yet.

## History

- 2026-10-27, ticket gpu-backend/03: first version, reach_goal on one level. Parity with the CPU env on 8 recorded scenarios (sphere, capsule and box goals and hazards, a fall, a truncation): largest observation difference 5.7e-6, largest reward difference 4.7e-8.
- 2026-10-27, ticket gpu-backend/04: all five tasks on still markers. Parity with the CPU env on 14 recorded scenarios of the four new tasks plus the eight reach_goal ones (Warp CPU device): largest observation difference 7.6e-6 (a fall), largest reward difference 1.6e-7; flags, success, failure codes and marker colours matched on every step.
- 2026-10-27, ticket gpu-backend/04: CUDA run on an NVIDIA GeForce RTX 2080 Ti (sm_75) on a GPU cluster, Warp 1.18.0: `HAMSTERBALL_WARP_DEVICE=cuda:0 pytest -m gpu tests/test_warp_tasks.py tests/test_warp_env.py tests/test_batch_core.py` passed 62 of 62 (the env step as one captured graph); largest observation difference 7.6e-6, reward 1.6e-7, as on the CPU device.
- 2026-10-27, ticket gpu-backend/05: the marker tables and the kill plane and step limit became per-entry tables indexed by each world's entry, for a bank of levels; the kernels' logic is unchanged (ticket 04's scalars `mode`, `ntargets`, `goal_row`, `survive`, `time_penalty` and `max_steps` became per-entry arrays).
