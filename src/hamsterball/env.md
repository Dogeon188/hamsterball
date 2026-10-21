# env.py

## Purpose

The Gymnasium env of spec §10 over the core: it picks a level and task per episode, steps the core, runs the marker tests in task order, and returns the `state` and `goal` observations, the reward (sparse, plus optional potential shaping), and `info`.

## Interface

- `import hamsterball` registers spec §2's six ids at `hamsterball.ENV_VERSION` (N = 0), each with `entry_point="hamsterball.env:HamsterballEnv"`: `Hamsterball-v0` (every kwarg free) and the presets `HamsterballReachGoal`, `HamsterballWaypointRace`, `HamsterballCollectItems`, `HamsterballUnlockGoal`, `HamsterballSurvive`, each fixing `tasks` to its one task. `hamsterball.register_ids(n)` also registers every version below n as an id whose entry point raises `gymnasium.error.DeprecatedEnv` naming the release to install (`hamsterball==<old N>.*`).
- `HamsterballEnv(tasks, levels=None, tier="abstract", camera_action=False, frame_skip=10, time_limit_s=None, shaped=False, shaping_gamma=0.99, model_cache=16, difficulty=0.5)`. `levels` is a `LevelGenerator` (such as `ProceduralGenerator()`, see [generator.md](core/generator.md)) or a list of `Level` objects, bundled level names (stems in `hamsterball/levels/`), level file paths, or directories (their `*.yaml`); `None` is every bundled level. Construction raises if no pool level (or the generator's `tasks`) supports any of `tasks`. `difficulty` goes to the generator; a value other than 0.5 with a file pool raises `ValueError`.
- `env.set_levels(levels, difficulty=None)`: replaces the pool from the next reset, with the same entries and the same checks; `None` keeps a generator pool's current difficulty, and a file pool resets it to 0.5. On a vector env, `venv.call("set_levels", levels, difficulty=d)`.
- `reset(seed=None, options=None)`: `options` may hold `level` (a `Level` or canonical YAML string), `task`, and `difficulty`; see How it works.
- `DiscreteAction(env)`: a wrapper with `Discrete(25)`: 0 is a no-op; `1 + 8·e + k` runs toward `k·45°` clockwise from camera forward (k = 0 forward, 2 right, 4 back) at effort `(0.25, 0.5, 1.0)[e]`. Under the camera flag `cam_yaw` stays 0.
- Action: the core's (`[run_x, run_y]`, plus `cam_yaw` under `camera_action`).
- Observation: `Dict(state=Box(19), goal=Box(10))`, float32, laid out as spec §10's tables.
- `info` on every step: `core_state`, `state`, `goal`, `success`, `failure_reason` (`None`, `"hazard"`, `"kill_plane"`, or `"left_region"`), `reward_terms` (`event`, `time`, `shaping`). On reset: `reset_options` (`{level: canonical YAML, level_id, task}`, plus `difficulty` for a generated level), `core_state`, `state`, `goal`.
- Bundled levels, at least one per task, each a box platform over a kill plane at −5 m:
  - `first_goal` (reach goal): 20 m square, goal sphere (1 m) 8 m along +y, 30 s.
  - `waypoint_loop` (waypoint race): 30 m square, four waypoint spheres looping round the spawn, 45 s.
  - `item_scatter` (collect items): 30 m square, three item spheres (0.8 m), 45 s.
  - `locked_goal` (unlock goal): 30 m square, two keys (0.8 m) to the sides, the goal 9 m ahead, 45 s.
  - `hill` (reach goal): `first_goal` with a 0.4 m round hfield hill (`hill.png`, 11×11, 5 m square) centred 4 m along +y, on the way to the goal.
  - `first_survive` (survive): 20 m square, a capsule region (radius 2 m, axis x = −1 to 1 m) around the spawn, 10 s.

## How it works

- Reset: options override kwargs, and the env draws each missing field from `np_random`, level first (spec §10): `{level, task}` as given; `{task}` a pool level supporting the task (whether or not the task is in `tasks`); `{level}` an allowed task the level supports, else any task it supports; `{}` a pool level supporting an allowed task, then an allowed task. A level that lacks the task, or a `{task}` no pool level supports, raises `ValueError`. `info['reset_options']` replays the episode on any env, whatever its kwargs.
- Generator pool: with no `level` option, the env draws a 63-bit seed (`np_random.integers(2**63)`), then the task if none is given (an allowed task the generator supports; a given task must be one of the generator's `tasks`), and calls `generator(task, seed, difficulty)` with `options['difficulty']` or else the env's difficulty. A `level` option skips the generator, as with a file pool. A `difficulty` option other than 0.5 raises with a file pool when the env would draw from it, and is ignored beside a `level` option.
- `reset_options['difficulty']`: the level's provenance `difficulty` when it has a `generator` block, else the difficulty the env passed to the generator, else absent. So a generated episode replayed from `{level, task}` on any env reports the same `reset_options`.
- Model cache: an LRU (`OrderedDict`) of `Core`s keyed by level id, at most `model_cache`; a hit moves the level to most recent, and a miss compiles and evicts the least recent. `Core.reset` calls `mj_resetData`, then places the shell at the spawn with the spawn yaw. The step limit is `ceil(time_limit_s / (frame_skip·dt))`.
- Targets: at reset the env lists the task's targets in task order as `(marker key, index)` pairs (`("goal", None)` for a lone goal). The live targets are the ones that can trigger now: the next waypoint only; every remaining key, and the unlock goal only once no key is left; every remaining item; the reach goal. The current target is the live target whose centre (`marker.centre`) is nearest the shell centre, and the fraction remaining is targets left over targets. Survive's one target is `("region", None)`, its centre the current target; its fraction remaining is `1 − steps / step_limit`, time left over the limit in whole env steps.
- Step: `Core.step(action, n=frame_skip)`, then the swept marker test from the previous env step's shell centre (a point at reset) against every live target, repeated until nothing new triggers, so one step completes waypoints i and i + 1, or the last key and the goal. Each waypoint, item, or key is +1; an empty target list is success, +1 more. Survive skips the swept test: its region is a point test on the current centre, and its target list empties (success) when the step count reaches the limit with the centre inside. Then failure, −1 added to that step's event, checked in order: a hazard (swept, every task), the kill plane, and for survive leaving the region (`left_region`). Success wins over failure in one step. Reward is `event + time + shaping`, with time −0.001, or 0 for survive.
- Survive never truncates: at its step limit the shell is either inside (success) or outside (`left_region`), and both terminate.
- Shaping (`shaped=True`): Φ = −0.01 × the distance from the shell centre to the current target (its centre), and Φ = 0 at a terminated state (success or failure; truncation isn't terminal). The step's term is `γ·Φ(s′) − Φ(s)`, where Φ(s) is the value stored at the previous step, so a target switch within the step shows as a jump.
- Marker looks: marker geoms are named `<task>/<key>` or `<task>/<key>/<i>` (see [level.md](core/level.md)). Reset sets every marker geom's `rgba`: the episode task's markers take their kind's colour, other tasks' markers alpha 0, and the unlock goal `LOCKED_GOAL_RGBA` (grey, fainter). A collected item or key goes to alpha 0, and the unlock goal takes the goal colour when the last key is in. Episodes on a level share its cached model, so reset restores every look.
- Observation: after stepping, the env recomputes derived quantities (`mj_kinematics`, `mj_comPos`, `mj_comVel`, `mj_subtreeVel`), which leaves the core state untouched. Vectors rotate into the camera yaw frame by −yaw about z; the shell's angular velocity first goes from the free joint's body frame to the world. Lean and relative velocity use the hamster subtree's centre of mass and its velocity, so an articulated tier reads the same.

## Invariants and checks

- `uv run pytest tests/test_env.py`: frames (yaw 0 faces +y), the swept goal, success, the kill plane, truncation at the step count, `info` keys, the 120°/s orbit, and a run over the `hill` hfield. The tests pin `levels=["first_goal"]` unless they name another, as the default pool is every bundled level.
- `uv run pytest tests/test_tasks.py`: waypoints i and i + 1 in one step, waypoints out of order, the nearest item, the unlock goal ignored then unlocked, marker looks, shaping with γ 0.99 and 0.5, and each multi-target task played to success on its bundled level by a scripted controller (velocity tracking toward the current target, 3 m/s cap).
- `uv run pytest tests/test_episode_setup.py`: same seed same (level, task), each option case, replay from `reset_options` on an env with other kwargs, pool entry kinds, `set_levels` direct and through `SyncVectorEnv.call`, LRU hits and eviction with `mj_resetData` on a hit, every preset with its default pool, a retired id's error, and the `DiscreteAction` table.
- `uv run pytest tests/test_generator.py` (its env section): a generator pool's draws and `reset_options['difficulty']`, the difficulty option, replay of a generated episode on a file-pool env, a non-default difficulty with a file pool raising (kwarg, `set_levels`, option), `set_levels` swapping pools, and any protocol object as `levels`.
- `uv run pytest tests/test_survive_and_hazards.py`: survive success at the limit and `left_region`, a hazard at the spawn on every task, a swept hazard, and success over a hazard in one step.

## Known quirks

- The step-limit conversion subtracts 1e-9 before `ceil`, so a whole number of env steps doesn't round up through float error.
- Waypoints stay visible once passed; spec §7 hides only collected items and keys.
- A marker test runs on the whole step's segment, not the part after the previous target, so a step that crosses waypoint i + 1 before i (in space) still counts both when it also crosses i.
- Not yet: views, `obs_state`, `make_vec` and the ids' `vector_entry_point`. Those are tickets 06 and 14.
- `action_space` comes from compiling the pool's first level, which takes one LRU slot. A generator pool has no level at construction, so the env compiles a throwaway `Core` on the default flat floor for it.
- Every generated level has its own id, so a generator pool misses the model LRU on every reset and compiles a model each time.

## Decisions

- 2026-10-22, ticket hamsterball-v1/02: `env` is one module, not a package, until it needs a second file; `hamsterball.env:HamsterballEnv` resolves the same either way.

- 2026-10-22, ticket hamsterball-v1/03: markers of tasks other than the episode's hide (alpha 0), so a level that supports several tasks shows one task's targets. The spec says only that collected items and keys hide; this extends it.
- 2026-10-22, ticket hamsterball-v1/03: a failure step adds −1 to that step's other events rather than replacing them, since spec §9 sums the reward terms.

- 2026-10-22, ticket hamsterball-v1/07: the release named for a retired id is `hamsterball==<N>.*`, since spec §2 makes each N bump a major release and both start at 0; no release table to keep.
- 2026-10-22, ticket hamsterball-v1/07: a `{task}` reset option draws from the whole pool, not only levels with an allowed task, since options override kwargs.
- 2026-10-22, ticket hamsterball-v1/10: `reset_options['difficulty']` comes from the level's provenance when it has one, so replaying a generated episode from `{level, task}` on a file-pool env reports the same options; a `difficulty` reset option is ignored beside a `level` option and raises only when the env would draw from a file pool. Spec §10 says only that a non-default kwarg with a file pool raises.
- 2026-10-22, ticket hamsterball-v1/10: with a generator the env draws the seed before the task, as spec §10 orders level before task, though the generator needs the task to build the level.

## Tried and rejected

- 2026-10-22, ticket hamsterball-v1/03: a scripted controller that runs straight at the current target overshot every turn (fell off `waypoint_loop` after the first waypoint; ran out the clock on the other two). The tests use velocity tracking instead.

## History

- 2026-10-22, ticket hamsterball-v1/02: added as the reach-goal tracer.
- 2026-10-22, ticket hamsterball-v1/03: waypoint race, collect items, unlock goal, marker looks, and `shaped`/`shaping_gamma`, with a bundled level per task.
- 2026-10-22, ticket hamsterball-v1/04: added the survive task, hazards, and the `first_survive` level.
- 2026-10-22, ticket hamsterball-v1/05: bundled the `hill` hfield level; a run straight at the goal climbs over the hill (shell centre above 0.8 m) and succeeds.
- 2026-10-22, ticket hamsterball-v1/07: reset options and `info['reset_options']`, directories in the pool, `set_levels`, the `model_cache` LRU, the six ids with retired versions, and `DiscreteAction`.
- 2026-10-22, ticket hamsterball-v1/10: generator pools (`levels=` a `LevelGenerator`), the `difficulty` kwarg, reset option, and `set_levels` argument, and `difficulty` in `reset_options`.
