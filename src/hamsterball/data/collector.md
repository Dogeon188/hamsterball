# collector.py

## Purpose

Records episodes of a hamsterball env to a Minari 0.5 dataset with Arrow storage (spec §12), storing what [replay.md](replay.md) needs to re-step them and what baselines need to filter them.

## Interface

- `Collector(env, **kw)`: a `minari.DataCollector` over a `gym.make`d hamsterball env (it needs `env.spec`). `kw` passes through to Minari (`action_space`, `jpeg_encoding`); the observation space (`OBSERVATION`), step callback, episode metadata callback, `record_infos`, and `data_format="arrow"` are fixed.
- `reset`, `step`, `create_dataset(dataset_id, **kw)`, `add_to_dataset(dataset)`, `close()`: Minari's.
- `discard_episode()`: drops the episode in progress (an abandoned run). Call `reset` before stepping again.
- `provenance()`: `{hamsterball_version, mujoco_version, platform}` of this build, `platform` as `<OS>-<arch>` from `platform.system()` and `platform.machine()`.
- `SPACE_KWARGS`: the env kwargs written into the dataset's `env_spec`.
- `OBSERVATION`: the stored observation space, `Dict(state=Box(19), goal=Box(10))` float32, whatever the env's.

## How it works

- Per dataset: construction writes every `SPACE_KWARGS` attribute of the env into `env.unwrapped.spec.kwargs`, before Minari copies the spec, so defaults the caller left out are explicit and `recover_environment()` rebuilds the same spaces. `create_dataset` adds `provenance()` to the dataset metadata.
- Per episode: `reset` replaces the buffered episode's options with the env's `info['reset_options']`, so an episode replays from its seed and options alone. The episode metadata callback adds Minari's reward statistics, the last step's `success` and `failure_reason`, `return`, and `length`.
- Per step: the action, `info['state']` and `info['goal']` as the observation, and `info['core_state']` only. The step callback takes `state` and `goal` from `info`, which has them with `obs_state=False`, and drops the view pixels and the other info keys. `step` casts a `Box` action to the action space's dtype before stepping, so the env steps with exactly the float32 action Minari stores.

## Invariants and checks

- `uv run pytest tests/test_data.py`: reset options and seeds per episode, `discard_episode`, dataset metadata and `env_spec` kwargs with `recover_environment()`, episode outcomes, a pixel env (`views`, `obs_state=False`) storing `state` and `goal` and replaying, and the replay tests of [replay.md](replay.md).
- `discard_episode` sets Minari's private `DataCollector._buffer` to None; `reset` and `create_dataset` flush `_buffer` and skip it when None. `test_discard_episode_drops_the_episode_in_progress` fails if that internal changes.
- `step` reads Minari's private `DataCollector._episode_id` to key each episode's outcome; `test_episode_metadata_holds_the_outcome_return_and_length` fails if it changes.

## Known quirks

- Minari's `DataCollector` needs `jax` (`minari[create]`), and its Arrow storage imports `PIL` without declaring it, so the `data` extra holds `minari[arrow,create]==0.5.*` and `pillow`. Install `hamsterball[data]` to record or replay.
- The episode metadata callback runs only in `create_dataset`. Minari's `add_to_dataset` copies episodes without running it, so appended episodes lack `success`, `failure_reason`, `return`, and `length`.
- `levels` must be JSON-serialisable in the spec kwargs (names or paths, not `Level` objects), or Minari can't write `env_spec`.
- `views` and `view_size` come back from `env_spec` as JSON lists; the env turns them into tuples.
- Not yet: the low-level policy's run vector as an extra per-step field, and the game's overrun count.

## Decisions

- 2026-10-22, ticket hamsterball-v1/08: outcomes come from the env's per-step `info`, tracked by episode id in the Collector, rather than stored as per-step fields, so a step stores only spec §12's action, observation, and `core_state`.
- 2026-10-22, ticket hamsterball-v1/08: the Collector casts actions to the action space dtype, since Minari stores them as float32 and a float64 action would replay differently.

- 2026-10-22, ticket hamsterball-v1/08: the stored observation is always `state` and `goal` from `info`, never pixels (spec §12), so a pixel env's dataset has a different observation space from the env that `recover_environment()` rebuilds.
- 2026-10-22, user ruling: Minari and pillow move out of the core dependencies into a `data` extra, so the env installs without jax (~58 MB). Spec §2 updated. `tests/test_data.py` skips without the extra.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/08: added. Found that minari 0.5.4's `DataCollector` imports `jax` and its Arrow storage imports `PIL`, neither in the `arrow` extra.
- 2026-10-22, ticket hamsterball-v1/08: after merging ticket 06's views, the Collector stores `state` and `goal` from `info` instead of the env's observation, and writes `views`, `view_size`, and `obs_state` into `env_spec`.
