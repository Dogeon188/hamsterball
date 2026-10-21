# \_\_init\_\_.py

## Purpose

The package root: it picks the headless GL backend before anything imports `mujoco`, registers spec §2's env ids at the env version, and holds `make_vec`'s per-worker env fn, which must load without `mujoco`.

## Interface

- `import hamsterball` sets `MUJOCO_GL=egl` on Linux unless it's already set (`osmesa` is the CPU fallback, set by hand); macOS leaves it unset (CGL). glfw is never used for env rendering; an env with views raises under `MUJOCO_GL=glfw` (see [env.md](env.md)).
- `ENV_VERSION`: N of spec §2 (0), bumped with every change that can alter a trajectory, as a major release.
- `PRESETS`: id name → the one task its `tasks` kwarg fixes, or `None` for `Hamsterball`, whose `tasks` defaults to every task id.
- `register_ids(n=ENV_VERSION)`: registers the six ids `<name>-v<n>` with `entry_point="hamsterball.env:HamsterballEnv"` and `vector_entry_point="hamsterball.env:make_vec"`, and every version below n as an id whose entry point raises `gymnasium.error.DeprecatedEnv` naming the release to install (`hamsterball==<old N>.*`). Import runs it once.
- `_worker_env(i, devices, kwargs)`: `make_vec`'s env fn for worker i (see [env.md](env.md)): with `devices` above 1 it sets `MUJOCO_EGL_DEVICE_ID` to `i mod devices` unless set, then unpickles `kwargs` and builds `HamsterballEnv(**kwargs)`.

## How it works

- The `MUJOCO_GL` default runs at the top of the module, before the imports below it, so it applies whenever `hamsterball` is imported before `mujoco`.
- The module imports only `gymnasium` and the stdlib at load; `hamsterball.env` (and so `mujoco`) loads only when an env is made. That is why `_worker_env` lives here: unpickling it in a worker loads no `mujoco`, so the device id is set first. Its kwargs come pickled for the same reason, since a generator or `Level` in them imports `mujoco` on unpickle.

## Invariants and checks

- `uv run pytest tests/test_episode_setup.py`: every id made with no kwargs and its default pool, and a retired id's error.
- `uv run pytest tests/test_vector.py`: a worker's env fn unpickles without loading `mujoco` and sets the device id round-robin.
- `uv run pytest tests/test_views.py`: the `MUJOCO_GL` default.

## Known quirks

- The `MUJOCO_GL` default only takes effect if `hamsterball` is imported before `mujoco`.

## Decisions

- 2026-10-22, ticket hamsterball-v1/07: the release named for a retired id is `hamsterball==<N>.*` (see [env.md](env.md)).
- 2026-10-22, code review of hamsterball-v1: `_worker_env` lives in the package root rather than `env.py`, which imports `mujoco`.

## Tried and rejected

## History

- 2026-10-22, code review of hamsterball-v1: doc added; the root's description moved here from [env.md](env.md). The registration and `MUJOCO_GL` code came in tickets 02, 06, 07, and 14 (commits `bb2b439`, `e40241e`, `cc5be87`, `0fff3f9`), and `_worker_env` in `ff2f49d`.
