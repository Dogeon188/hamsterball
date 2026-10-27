# env_step.py (gpu)

## Purpose

The env-step kernels: what `HamsterballEnv._advance` and `_observe` ([env.md](../env.md)) do after the physics, one thread per world, so the Warp vector env ([vector_env.md](vector_env.md)) runs them inside the step graph. This slice covers reach_goal on still markers; the other tasks, moving markers, attached markers and survive's point test come with later tickets of the `gpu-backend` effort.

## Interface

Needs the `gpu` extra. Nothing but [vector_env.py](vector_env.md) calls it.

- `Markers(entries, device)`: the marker tables on the device, one block of `stride` rows per bank entry (`entries` is a list of marker lists, goal first, then hazards; `n[e]` counts the rows, the rest of the block is padding), `stride` the longest list. Each row is a sphere, a capsule (a sphere is a capsule of zero length), or a box, in the form `core.marker` reads (the box's frame is built with the same `mju_euler2Quat` and `mju_quat2Mat` calls).
- `FAILURE_CODES = ("none", "hazard", "kill_plane", "left_region")`: the table for `info["failure_reason"]`, an int32 per world, the index into this tuple. `left_region` is for survive and unused so far.
- Kernels, named for the CPU function they port: `advance` (`_advance`), `observe` (`_observe`'s `state` and `goal`), `stash_final` and `reset_episode` (autoreset), `set_spawn` (the spawn pose of the masked worlds from their entry's table), `either` (an or of two masks). `advance`, `observe` and `reset_episode` read each world's entry (the bank's (level, task) pair) for its marker rows, kill plane height and step limit.
- Helpers: `seg_seg_dist`, `box_dist`, `seg_box_dist`, `touches` (`core.marker`'s `_seg_seg_dist`, `_seg_box_dist` and `crosses` with reach `R`).

## How it works

- `advance` adds a step, then tests the segment from the previous env step's shell centre to the current one against the goal with reach `R`. A touch is success, `+1`. Otherwise each hazard is tested the same way (`fail = 1`), then the centre's height against the kill plane (`fail = 2`); either is `-1`. Success wins over failure, as on the CPU. The time penalty is `-0.001`, the shaping term `gamma * phi' - phi` with `phi' = 0` at a terminated state, and `truncated` is `truncate and not terminated and steps >= max_steps`. It writes the reward, its three terms, the flags, and the episode's step count, previous position and potential.
- The box test is the CPU's golden-section search along the segment to a 1e-6 m bracket, in float32 (capped at 64 passes, which only a degenerate bracket would reach). The CPU's reach-0 slab test is not ported: every task here touches with reach `R`, and survive's centre test arrives with survive.
- `observe` writes the 21-number `state` and 10-number `goal` for the worlds in a mask, from the refreshed `xpos`, `xmat`, `subtree_com` and `subtree_linvel`, in the facing frame. The goal's target is the goal marker's centre (a capsule's midpoint); once a world has succeeded its goal vector is zeros apart from the task one-hot.
- Per-world episode state, all device arrays owned by the vector env: step count, previous shell position, potential. The previous sim time the spec lists is not kept: still markers don't read it.

## Invariants and checks

- Against the CPU env through `step_to`, on recorded episodes: observations and rewards at `rtol=1e-5, atol=1e-5`, flags, success and the failure code exactly. `uv run pytest tests/test_warp_env.py`.
- A kernel that changes here must keep the CPU function it names in step, the way the core-step kernels do ([batch_core.md](batch_core.md)): a change to `_advance` or `_observe` fails `test_step_to_matches_the_cpu_env_on_recorded_episodes` until the kernel matches.

## Known quirks

- float32: a segment whose closest approach to a marker lies within about 1e-5 m of the touch threshold may trigger on one backend and not the other (the CPU is float64, and its state loads into Warp rounded to float32). The parity test's scenarios avoid it; the spec allows the exception.
- `info` arrays that the CPU gives as Python values (`failure_reason` a string or `None`) are int codes here.

## Decisions

- 2026-10-27, ticket gpu-backend/03: the marker table has the goal and hazards as rows of one table, tested by one function, so later tasks add rows and a bitmask over them rather than new tests.
- 2026-10-27, ticket gpu-backend/03: the kernels live in the graph, so `advance` always runs on all worlds and masks nothing; a finished world's results are stashed and its reset follows in the same graph.

## Tried and rejected

- Nothing yet.

## History

- 2026-10-27, ticket gpu-backend/03: first version, reach_goal on one level. Parity with the CPU env on 8 recorded scenarios (sphere, capsule and box goals and hazards, a fall, a truncation): largest observation difference 5.7e-6, largest reward difference 4.7e-8.
- 2026-10-27, ticket gpu-backend/05: the marker table and the kill plane and step limit became per-entry tables indexed by each world's entry, for a bank of levels; the kernels' logic is unchanged.
