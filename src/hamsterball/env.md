# env.py

## Purpose

The Gymnasium env of spec §10 over the core: it picks a level and task per episode, steps the core, runs the marker tests in task order, and returns the `state` and `goal` observations, the reward (sparse, plus optional potential shaping), and `info`.

## Interface

- `import hamsterball` registers `Hamsterball-v0` with `entry_point="hamsterball.env:HamsterballEnv"`.
- `HamsterballEnv(tasks, levels=None, tier="abstract", camera_action=False, frame_skip=10, time_limit_s=None, shaped=False, shaping_gamma=0.99)`. `levels` holds `Level` objects, bundled level names (stems in `hamsterball/levels/`), or paths; `None` is every bundled level. Construction raises if no pool level supports any of `tasks`.
- Action: the core's (`[run_x, run_y]`, plus `cam_yaw` under `camera_action`).
- Observation: `Dict(state=Box(19), goal=Box(10))`, float32, laid out as spec §10's tables.
- `info` on every step: `core_state`, `state`, `goal`, `success`, `failure_reason` (`None`, `"hazard"`, `"kill_plane"`, or `"left_region"`), `reward_terms` (`event`, `time`, `shaping`). On reset: `core_state`, `state`, `goal`.
- Bundled levels, one per task so far, each a box platform over a kill plane at −5 m:
  - `first_goal` (reach goal): 20 m square, goal sphere (1 m) 8 m along +y, 30 s.
  - `waypoint_loop` (waypoint race): 30 m square, four waypoint spheres looping round the spawn, 45 s.
  - `item_scatter` (collect items): 30 m square, three item spheres (0.8 m), 45 s.
  - `locked_goal` (unlock goal): 30 m square, two keys (0.8 m) to the sides, the goal 9 m ahead, 45 s.
  - `first_survive` (survive): 20 m square, a capsule region (radius 2 m, axis x = −1 to 1 m) around the spawn, 10 s.

## How it works

- Reset: draw a pool level (among those supporting a requested task), then a task it supports, from `np_random`. One `Core` per level id compiles on first use; `Core.reset` places the shell at the spawn with the spawn yaw. The step limit is `ceil(time_limit_s / (frame_skip·dt))`.
- Targets: at reset the env lists the task's targets in task order as `(marker key, index)` pairs (`("goal", None)` for a lone goal). The live targets are the ones that can trigger now: the next waypoint only; every remaining key, and the unlock goal only once no key is left; every remaining item; the reach goal. The current target is the live target whose centre (`marker.centre`) is nearest the shell centre, and the fraction remaining is targets left over targets. Survive's one target is `("region", None)`, its centre the current target; its fraction remaining is `1 − steps / step_limit`, time left over the limit in whole env steps.
- Step: `Core.step(action, n=frame_skip)`, then the swept marker test from the previous env step's shell centre (a point at reset) against every live target, repeated until nothing new triggers, so one step completes waypoints i and i + 1, or the last key and the goal. Each waypoint, item, or key is +1; an empty target list is success, +1 more. Survive skips the swept test: its region is a point test on the current centre, and its target list empties (success) when the step count reaches the limit with the centre inside. Then failure, −1 added to that step's event, checked in order: a hazard (swept, every task), the kill plane, and for survive leaving the region (`left_region`). Success wins over failure in one step. Reward is `event + time + shaping`, with time −0.001, or 0 for survive.
- Survive never truncates: at its step limit the shell is either inside (success) or outside (`left_region`), and both terminate.
- Shaping (`shaped=True`): Φ = −0.01 × the distance from the shell centre to the current target (its centre), and Φ = 0 at a terminated state (success or failure; truncation isn't terminal). The step's term is `γ·Φ(s′) − Φ(s)`, where Φ(s) is the value stored at the previous step, so a target switch within the step shows as a jump.
- Marker looks: marker geoms are named `<task>/<key>` or `<task>/<key>/<i>` (see [level.md](core/level.md)). Reset sets every marker geom's `rgba`: the episode task's markers take their kind's colour, other tasks' markers alpha 0, and the unlock goal `LOCKED_GOAL_RGBA` (grey, fainter). A collected item or key goes to alpha 0, and the unlock goal takes the goal colour when the last key is in. Episodes on a level share its cached model, so reset restores every look.
- Observation: after stepping, the env recomputes derived quantities (`mj_kinematics`, `mj_comPos`, `mj_comVel`, `mj_subtreeVel`), which leaves the core state untouched. Vectors rotate into the camera yaw frame by −yaw about z; the shell's angular velocity first goes from the free joint's body frame to the world. Lean and relative velocity use the hamster subtree's centre of mass and its velocity, so an articulated tier reads the same.

## Invariants and checks

- `uv run pytest tests/test_env.py`: frames (yaw 0 faces +y), the swept goal, success, the kill plane, truncation at the step count, `info` keys, and the 120°/s orbit.
- `uv run pytest tests/test_tasks.py`: waypoints i and i + 1 in one step, waypoints out of order, the nearest item, the unlock goal ignored then unlocked, marker looks, shaping with γ 0.99 and 0.5, and each multi-target task played to success on its bundled level by a scripted controller (velocity tracking toward the current target, 3 m/s cap).
- `uv run pytest tests/test_survive_and_hazards.py`: survive success at the limit and `left_region`, a hazard at the spawn on every task, a swept hazard, and success over a hazard in one step.

## Known quirks

- The step-limit conversion subtracts 1e-9 before `ceil`, so a whole number of env steps doesn't round up through float error.
- Compiled models are cached without bound (one per level id); spec §7's `model_cache` LRU is for later.
- Waypoints stay visible once passed; spec §7 hides only collected items and keys.
- A marker test runs on the whole step's segment, not the part after the previous target, so a step that crosses waypoint i + 1 before i (in space) still counts both when it also crosses i.
- Not yet: `reset` options and `info['reset_options']`, views, `obs_state`, the five preset ids, and `make_vec`. Those are tickets 05 to 07 and 14.

## Decisions

- 2026-10-22, ticket hamsterball-v1/02: `env` is one module, not a package, until it needs a second file; `hamsterball.env:HamsterballEnv` resolves the same either way.

- 2026-10-22, ticket hamsterball-v1/03: markers of tasks other than the episode's hide (alpha 0), so a level that supports several tasks shows one task's targets. The spec says only that collected items and keys hide; this extends it.
- 2026-10-22, ticket hamsterball-v1/03: a failure step adds −1 to that step's other events rather than replacing them, since spec §9 sums the reward terms.

## Tried and rejected

- 2026-10-22, ticket hamsterball-v1/03: a scripted controller that runs straight at the current target overshot every turn (fell off `waypoint_loop` after the first waypoint; ran out the clock on the other two). The tests use velocity tracking instead.

## History

- 2026-10-22, ticket hamsterball-v1/02: added as the reach-goal tracer.
- 2026-10-22, ticket hamsterball-v1/03: waypoint race, collect items, unlock goal, marker looks, and `shaped`/`shaping_gamma`, with a bundled level per task.
- 2026-10-22, ticket hamsterball-v1/04: added the survive task, hazards, and the `first_survive` level.
