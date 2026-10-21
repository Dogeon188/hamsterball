# env.py

## Purpose

The Gymnasium env of spec §10 over the core: it picks a level and task per episode, steps the core, runs the marker test, and returns the `state` and `goal` observations, sparse reward, and `info`.

## Interface

- `import hamsterball` registers `Hamsterball-v0` with `entry_point="hamsterball.env:HamsterballEnv"`.
- `HamsterballEnv(tasks, levels=None, tier="abstract", camera_action=False, frame_skip=10, time_limit_s=None)`. `levels` holds `Level` objects, bundled level names (stems in `hamsterball/levels/`), or paths; `None` is every bundled level. Construction raises if no pool level supports any of `tasks`.
- Action: the core's (`[run_x, run_y]`, plus `cam_yaw` under `camera_action`).
- Observation: `Dict(state=Box(19), goal=Box(10))`, float32, laid out as spec §10's tables.
- `info` on every step: `core_state`, `state`, `goal`, `success`, `failure_reason` (`None`, `"hazard"`, `"kill_plane"`, or `"left_region"`), `reward_terms` (`event`, `time`, `shaping`). On reset: `core_state`, `state`, `goal`.
- Bundled levels: `first_goal` (reach goal), a 20 m square box platform, goal sphere (1 m) 8 m along +y, kill plane at −5 m, 30 s; `first_survive` (survive), the same platform with a capsule region (radius 2 m, axis x = −1 to 1 m) around the spawn, 10 s.

## How it works

- Reset: draw a pool level (among those supporting a requested task), then a task it supports, from `np_random`. One `Core` per level id compiles on first use; `Core.reset` places the shell at the spawn with the spawn yaw. The step limit is `ceil(time_limit_s / (frame_skip·dt))`.
- Step: `Core.step(action, n=frame_skip)`, then success: reach goal runs the swept goal test from the previous env step's shell centre (a point at reset); survive succeeds when the step count reaches the limit with the current centre inside the region (a point test). Without success, failure checks in order: a hazard (swept, every task), the kill plane, and for survive leaving the region. Success wins over failure in one step. Reward is the event (+1 or −1) plus −0.001 per step, except survive, which has no time penalty.
- Survive never truncates: at its step limit the shell is either inside (success) or outside (`left_region`), and both terminate.
- Goal vector: the current target is the goal (reach goal) or the region's centre (survive). Fraction remaining is 1 for reach goal; for survive it is `1 − steps / step_limit`, time left over the limit in whole env steps.
- Observation: after stepping, the env recomputes derived quantities (`mj_kinematics`, `mj_comPos`, `mj_comVel`, `mj_subtreeVel`), which leaves the core state untouched. Vectors rotate into the camera yaw frame by −yaw about z; the shell's angular velocity first goes from the free joint's body frame to the world. Lean and relative velocity use the hamster subtree's centre of mass and its velocity, so an articulated tier reads the same.

## Invariants and checks

- `uv run pytest tests/test_env.py`: frames (yaw 0 faces +y), the swept goal, success, the kill plane, truncation at the step count, `info` keys, and the 120°/s orbit.
- `uv run pytest tests/test_survive_and_hazards.py`: survive success at the limit and `left_region`, hazards on both tasks, a swept hazard, and success over a hazard in one step.

## Known quirks

- The step-limit conversion subtracts 1e-9 before `ceil`, so a whole number of env steps doesn't round up through float error.
- Compiled models are cached without bound (one per level id); spec §7's `model_cache` LRU is for later.
- Not yet: `reset` options and `info['reset_options']`, the other three tasks, shaping, views, `obs_state`, the five preset ids, and `make_vec`. Those are tickets 03 to 07 and 14.

## Decisions

- 2026-10-22, ticket hamsterball-v1/02: `env` is one module, not a package, until it needs a second file; `hamsterball.env:HamsterballEnv` resolves the same either way.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/02: added as the reach-goal tracer.
- 2026-10-22, ticket hamsterball-v1/04: added the survive task, hazards, and the `first_survive` level.
