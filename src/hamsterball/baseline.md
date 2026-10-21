# baseline.py

## Purpose

`hamsterball-baseline`, spec §13's human baselines: per group of episodes, how often and how fast players succeed, read offline from Minari datasets that [data/collector.md](data/collector.md) wrote. The game keeps no scores; this is where they come from.

## Interface

- `hamsterball-baseline DATASET...` (`main`, a click command): Minari dataset ids (e.g. `hamsterball/human-ana-v0`, under `~/.minari` or `MINARI_DATASETS_PATH`). Prints one row per group: task, level, episodes, success rate, median and best time to success in sim seconds (`-` without a success), and mean return. File levels come first, the level shown as its id's first 12 hex digits; generated levels show `d=<difficulty>`. Without the `data` extra it exits 1 asking for `hamsterball[data]`.
- `baselines(dataset_ids)`: `{group: Stats}`, the numbers behind the table. A group is `(task, level_id)` for a file level and `(task, difficulty rounded to 0.1)` for a generated one.
- `Stats(episodes, success_rate, median_time, best_time, mean_return)`: times are None when the group has no success.
- `group(options)`: an episode's group from its reset options.

## How it works

- Reads only the dataset's `env_spec` and each episode's metadata (`storage.get_episode_metadata`), never its steps: the group from `options` (the env's `info['reset_options']`), and `success`, `length`, and `return` from the Collector's episode metadata callback.
- An episode is generated when its options carry `difficulty`, which the env adds only for generated levels.
- Time to success is `length × frame_skip × DT`, with each dataset's own `frame_skip` from `env_spec`, so datasets recorded at different control rates pool correctly.
- Groups pool across all the datasets given, whatever their tier, action space, or player: grouping never looks at `env_spec` beyond `frame_skip`.
- Minari is imported inside the functions, so the module imports without the `data` extra.

## Invariants and checks

- `uv run --locked --all-extras pytest tests/test_baseline.py`: two synthetic datasets (different `frame_skip` and action space) pool into the expected groups with hand-worked numbers, generated difficulties 0.26 and 0.34 share the 0.3 group, the CLI table rows, and the missing-Minari message.

## Known quirks

- Episodes need `success`, `length`, and `return` in their metadata. Episodes appended by the Collector before ticket 13's `add_to_dataset` fix lack them, and `baselines` raises `KeyError` on them.
- Rounding is Python's `round`, so a difficulty exactly halfway between tenths goes to whichever the binary float lies nearer (0.25 rounds to 0.2).
- File levels show a 12-digit id prefix, not a name: the canonical level has no name field.

## Decisions

- 2026-10-22, ticket hamsterball-v1/13: a top-level module `hamsterball.baseline`, not inside `hamsterball.data`, since `hamsterball.data` imports Minari on import and the CLI must say which extra to install when Minari is missing.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/13: added. While building it, found that Minari's `add_to_dataset` dropped appended episodes' seed, options, and outcomes; fixed in the Collector (see [data/collector.md](data/collector.md)).
