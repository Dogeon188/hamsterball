# session.py

## Purpose

The game's episodes and their recording (spec §13), without a display: each finished episode is appended to the player's Minari dataset the moment it ends, and an abandoned run is discarded. [window.py](window.md) drives it; [collector.md](../data/collector.md) records.

## Interface

- `Session(task, level, difficulty=0.5, seed=None, player=None, record=True, tier="abstract")`: `level` is a level file path or `"generated"`; a file level ignores `difficulty`. Recording needs `player` to pass `name_ok` (`ValueError` otherwise) and the `data` extra; `record=False` needs neither.
- `env`: the `Hamsterball-v<N>` env (`camera_action=True`, `truncate=False`), wrapped in a `Collector` when recording; the window reads `env.unwrapped` for drawing and the HUD.
- `recording`, `dataset_id` (`hamsterball/human-<player>-v<N>`, or `hamsterball/human-<player>-articulated-v<N>` with `tier="articulated"`, apart from the abstract dataset, N = `hamsterball.ENV_VERSION`), `running` (an episode is in progress).
- `start(retry=False)`: abandons a run in progress, then resets. A file level resets with `{task}`; a generated level draws a fresh seed, except the first episode when `seed` is given (`{level: generator(task, seed, difficulty), task}`). `retry=True` resets with the last episode's `{level, task}`, so the same level and seed.
- `step(action, overruns=0)`: one env step; `action` is always the game's 4-D action. In the articulated tier the session runs the scripted gait (`lowlevel.DEFAULT`) on the latest observation and `action[:2]`, steps the env with the torques then `action[2:]` (`turn`, `cam_pitch`), and passes the run vector to `Collector.step(run_vector=...)` (not when `record=False`); `start()` resets the gait. The dataset carries `low_level_policy` and says "articulated" in its description. When the step ends the episode, a recording session appends it to the dataset (creating the dataset on the first), and sets the episode's `overruns` metadata.
- `abandon()`: discards the run in progress with `Collector.discard_episode()`. `close()` abandons, then closes the env.

## How it works

- Saving: `minari.load_dataset(dataset_id)` then `Collector.add_to_dataset`, or on `FileNotFoundError` `Collector.create_dataset` (author the player, no `algorithm_name`: spec §12 names a human in `author` only). Each save holds one episode, since Minari starts fresh storage after either call. Then `update_episode_metadata([{"overruns": n}], [last index])`, which merges into the episode's metadata.
- Before wrapping, the session drops `levels` and `difficulty` from the env spec's kwargs, so the dataset's `env_spec` names no level: replay resets each episode from its own options (`{level, task}`), and one dataset holds every level and task the player plays. The `env_spec` keeps the first session's `tasks`; replay ignores it, since a `{level, task}` reset works on any env.
- Crash safety: a finished episode is on disk before `step` returns. The run in progress lives only in the Collector's buffer and temporary directory under Minari's root, so a crash loses that run and nothing else.

## Invariants and checks

- `uv run --all-extras pytest tests/test_session.py`: two finished episodes append with overruns 3 and 0 and replay, the dataset naming the player in `author` and holding no `algorithm_name`; a dataset recorded from a level file replays after the file is deleted, across a file and a generated level (seed, then a retry after an abandoned run); a crash after one finished episode leaves exactly it; a file level constructs with a non-default difficulty; the env doesn't truncate, and its spec says so; quitting mid-episode writes nothing and `record=False` never writes; a bad player name raises.

## Known quirks

- A crash leaves the Collector's temporary directory (`tmp*` in Minari's datasets root) behind; Minari ignores it, but nothing cleans it up.
- The overrun count is set after Minari's write, so a crash between the two leaves an episode without `overruns`. Baselines don't read it.
- `minari.load_dataset` runs on every save; it reads only metadata, cheap next to a human episode.

## Decisions

- 2026-10-22, ticket hamsterball-v1/12: the overrun count goes into the episode metadata from the session, through `update_episode_metadata` after the append, so `collector.py` stays untouched while ticket 09 changes it.
- 2026-10-22, ticket hamsterball-v1/12: `--seed` sets only the first episode of a generated session; Enter (next) draws a fresh seed as spec §13 says, and Retry replays the current one.
- 2026-10-22, ticket hamsterball-v1/12: the dataset's `env_spec` names no level (see How it works), so a human dataset replays on a machine without the player's level files.
- 2026-10-22, ticket hamsterball-v1/12: the `game` extra depends on the `data` extra, rather than the game running with recording off when Minari is missing, since spec §13 records by default. The session still imports Minari only when recording.

- 2026-10-22, bug fix: a file level drops `difficulty` before `gym.make`, rather than the window passing 0.5, so the menu and the `--level FILE --difficulty X` flags are both fixed. The menu hides the slider for a file level but still remembers the last generated level's difficulty, and the env raises on a non-default difficulty with a file pool.
- 2026-10-22, user play-test request: a human player gets no time-up except in survive, so the session's env has `truncate=False`; an episode ends in success, failure, or an abandoned run. Survive still ends at its limit (its success). The flag is in the dataset's `env_spec`, so replay rebuilds the same ending.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/12: added, test-first. Agent finding: with a file level, `env_spec` held the level's absolute path, and `replay` rebuilds the env from `env_spec`, so the session drops it.
- 2026-10-22, code review of hamsterball-v1: `_save` set both `author` and `algorithm_name="human"`; spec §12 names a human player in `author` only, so `algorithm_name` is dropped (Minari warns that it's unset).
- 2026-10-22, user play-test request: no time-up for a human player, except in survive; the session's env sets `truncate=False`.
- 2026-10-27, ticket articulated-tier/06: the session owns the gait rather than the window, so the headless tests cover the whole articulated step and recording; `test_an_articulated_session_steps_the_gait_and_records_to_its_own_dataset` checks the id, 10-D actions, run vectors, gait name, and replay.
