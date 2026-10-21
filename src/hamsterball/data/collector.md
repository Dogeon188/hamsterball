# collector.py

## Purpose

Records episodes of a hamsterball env to a Minari 0.5 dataset with Arrow storage (spec §12), storing what [replay.md](replay.md) needs to re-step them and what baselines need to filter them.

## Interface

- `Collector(env, **kw)`: a `minari.DataCollector` over a `gym.make`d hamsterball env (it needs `env.spec`). `kw` passes through to Minari (`action_space`, `jpeg_encoding`); the observation space (`OBSERVATION`), step callback, episode metadata callback, `record_infos`, and `data_format="arrow"` are fixed.
- `reset`, `step`, `create_dataset(dataset_id, **kw)`, `close()`: Minari's.
- `add_to_dataset(dataset)`: Minari's, but each appended episode keeps its seed and reset options and gets the episode metadata callback's fields, as in `create_dataset`.
- `discard_episode()`: drops the episode in progress (an abandoned run). Call `reset` before stepping again.
- `provenance()`: `{hamsterball_version, mujoco_version, platform}` of this build, `platform` as `<OS>-<arch>` from `platform.system()` and `platform.machine()`.
- `SPACE_KWARGS`: the env kwargs written into the dataset's `env_spec`: the space-shaping ones, plus `truncate` and `time_limit_s`, which decide when an episode ends.
- `OBSERVATION`: the stored observation space, `Dict(goal=Box(10), state=Box(19))` float32, whatever the env's.

## How it works

- Per dataset: construction writes every `SPACE_KWARGS` attribute of the env into `env.unwrapped.spec.kwargs`, before Minari copies the spec, so defaults the caller left out are explicit and `recover_environment()` rebuilds the same spaces. `create_dataset` adds `provenance()` to the dataset metadata.
- Per episode: `reset` replaces the buffered episode's options with the env's `info['reset_options']`, so an episode replays from its seed and options alone. The episode metadata callback adds Minari's reward statistics, the last step's `success` and `failure_reason`, `return`, and `length`.
- Per step: the action, `info['state']` and `info['goal']` as the observation, and `info['core_state']` only. The step callback takes `state` and `goal` from `info`, which has them with `obs_state=False`, and drops the view pixels and the other info keys. `step` casts a `Box` action to the action space's dtype before stepping, so the env steps with exactly the float32 action Minari stores.
- `core_state` is stored as a variable-length `list<double>` rather than Minari's fixed-size list: the collector module replaces Minari's `arrow_storage._encode_info` on import with one that encodes `core_state` that way (other info keys as Minari does). Its size depends on the level (kinematic bodies add qpos and qvel), and Minari reads a dataset as a union of per-episode Arrow files that must share one schema, so a fixed-size list fails on read once two sizes meet (`ArrowTypeError: ... fixed_size_list<item: double>[110] ... list<item: double>`). Reading needs no change: each episode's rows share a size, so Minari's `np.stack` returns `(steps, size)` float64 as before.

## Invariants and checks

- `uv run pytest tests/test_data.py`: reset options and seeds per episode, `discard_episode`, dataset metadata and `env_spec` kwargs (`truncate` and `time_limit_s` among them, defaults written out) with `recover_environment()`, episode outcomes, a pixel env (`views`, `obs_state=False`) storing `state` and `goal` and replaying, and the replay tests of [replay.md](replay.md).
- `discard_episode` sets Minari's private `DataCollector._buffer` to None; `reset` and `create_dataset` flush `_buffer` and skip it when None. `test_discard_episode_drops_the_episode_in_progress` fails if that internal changes.
- `step` reads Minari's private `DataCollector._episode_id` to key each episode's outcome; `test_episode_metadata_holds_the_outcome_return_and_length` fails if it changes.
- `add_to_dataset` calls Minari's private `_flush_to_storage` and reads `_storage` and `_episode_metadata_callback`; `test_appended_episodes_get_the_outcome_metadata_too` fails if they change, and replays the appended episode.
- The `_encode_info` replacement patches a Minari internal for every Arrow write in the process (re-render's output included); `tests/test_data.py::test_one_dataset_holds_levels_whose_core_states_differ_in_size` creates, appends to, and replays a dataset of two sizes, and fails if the internal changes.
- `OBSERVATION`'s keys stay in sorted order (see Known quirks); the same test fails on a mismatch.

## Known quirks

- Minari's `DataCollector` needs `jax` (`minari[create]`), and its Arrow storage imports `PIL` without declaring it, so the `data` extra holds `minari[arrow,create]==0.5.*` and `pillow`. Install `hamsterball[data]` to record or replay.
- Minari 0.5.4 runs the episode metadata callback only in `create_dataset`, and its `add_to_dataset` copies episodes through `MinariStorage.update_from_storage`, which rebuilds each from its steps and drops the seed and options. The Collector's `add_to_dataset` writes them back with the callback's fields.
- Minari's episode buffer stores a `Dict` observation in sorted key order (jax's `tree_map`), but `update_from_storage` re-encodes in the space's key order. With any other order an appended episode gets a different Arrow schema and reading the dataset fails with `ArrowTypeError`, so `OBSERVATION` lists `goal` before `state`.
- `levels` must be JSON-serialisable in the spec kwargs (names or paths, not `Level` objects), or Minari can't write `env_spec`. A `LevelGenerator` pool is the exception: the Collector drops `levels` and `difficulty` from the spec, since each episode's options hold its generated level and difficulty; the recovered env has the default pool.
- `views` and `view_size` come back from `env_spec` as JSON lists; the env turns them into tuples.
- A dataset written before ticket survive-movers/03 stores `core_state` as a fixed-size list, so appending a new episode to it (the game's per-player dataset) fails on read. Start a new dataset, or rewrite the old one.
- Not yet: the low-level policy's run vector as an extra per-step field, and the game's overrun count.

## Decisions

- 2026-10-22, ticket hamsterball-v1/08: outcomes come from the env's per-step `info`, tracked by episode id in the Collector, rather than stored as per-step fields, so a step stores only spec §12's action, observation, and `core_state`.
- 2026-10-22, ticket hamsterball-v1/08: the Collector casts actions to the action space dtype, since Minari stores them as float32 and a float64 action would replay differently.
- 2026-10-22, ticket hamsterball-v1/08: the stored observation is always `state` and `goal` from `info`, never pixels (spec §12), so a pixel env's dataset has a different observation space from the env that `recover_environment()` rebuilds.
- 2026-10-22, user ruling: Minari and pillow move out of the core dependencies into a `data` extra, so the env installs without jax (~58 MB). Spec §2 updated. `tests/test_data.py` skips without the extra.
- 2026-10-22, ticket survive-movers/03: one dataset holds levels whose core states differ in size (the game's per-player dataset holds every level played; the default survive pool mixes levels with and without bodies), by storing `core_state` as a variable-length list.

## Tried and rejected

- 2026-10-22, ticket survive-movers/03: considered and dropped without a run: padding the core state to a fixed size (no fixed maximum, and replay and re-render would have to strip it); a leaf type for the step info that Minari encodes as a variable list (its buffer stacks `np.ndarray` leaves into a fixed-size list, and pyarrow can't infer a type for `array.array`, tried); keying datasets by core-state size (splits a player's games).

## History

- 2026-10-22, ticket hamsterball-v1/08: added. Found that minari 0.5.4's `DataCollector` imports `jax` and its Arrow storage imports `PIL`, neither in the `arrow` extra.
- 2026-10-22, ticket hamsterball-v1/08: after merging ticket 06's views, the Collector stores `state` and `goal` from `info` instead of the env's observation, and writes `views`, `view_size`, and `obs_state` into `env_spec`.
- 2026-10-22, ticket hamsterball-v1/10: a generator pool's `env_spec` failed to serialise, so its dataset had no `env_spec` and `replay` raised; the Collector now drops `levels` and `difficulty` for a generator pool. `tests/test_generator.py::test_a_generated_episode_records_and_replays`.
- 2026-10-22, ticket hamsterball-v1/13: agent finding while building baselines over appended episodes: Minari's `add_to_dataset` dropped the seed, reset options, and callback fields, and an appended episode's observation schema clashed with the first's. The Collector now overrides `add_to_dataset`, and `OBSERVATION` lists `goal` first.
- 2026-10-22, user play-test request (no time-up for a human player, except in survive): the Collector writes the env's `truncate` and `time_limit_s` into `env_spec`, so `recover_environment()` rebuilds an env with the same episode ending. `tests/test_data.py::test_an_episode_recorded_past_the_time_limit_with_truncate_false_replays_bit_for_bit`.
- 2026-10-22, ticket survive-movers/03: measured by the exploration agent and reproduced here: a Minari 0.5 dataset mixing core-state sizes 110 and 115 failed at `create_dataset` (the episode metadata callback reads the union of episodes) with `ArrowTypeError`. Fixed by the variable-length encoding.
