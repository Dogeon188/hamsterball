# menu.py

## Purpose

The game menu's logic without a display (spec §13): the player name check, the level list, the seed field, and the choices remembered in `~/.config/hamsterball/game.json`. [window.py](window.md) draws the menu with Dear ImGui over these functions.

## Interface

- `name_ok(name)`: whether `name` matches `[a-z0-9_]+`.
- `level_choices(levels_dir, task)`: the stems of the `*.yaml` files in `levels_dir` that have a `task` block, sorted, then `"generated"` (`GENERATED`).
- `resolve(level, levels_dir)`: what to play for a menu or `--level` choice: `"generated"`, an existing file path as given, or `levels_dir/<stem>.yaml`.
- `parse_seed(text)`: the seed field; blank is `None` (random), else a non-negative integer; raises `ValueError`.
- `CONFIG`: the config file's path. `load_config(path=CONFIG)` returns the remembered choices, `{}` when the file is missing or isn't a JSON object. `save_config(settings, path=CONFIG)` writes `player`, `task`, `level`, and `difficulty` (`SAVED`), making the directory.
- `prefill(flags, config)`: the menu's starting `{player, task, level, difficulty, seed}`: each flag that isn't `None`, else the remembered choice, else `DEFAULTS` (`""`, `reach_goal`, `None`, 0.5, `None`).

## How it works

- `level_choices` loads every file with `Level.load` for each call; a file that raises `ValueError`, `OSError`, or a YAML error is left out, so one bad file doesn't stop the menu. The window calls it once per task when the menu opens.
- The config stores the level as the menu chose it: a stem, `"generated"`, or the `--level` path.

## Invariants and checks

- `uv run --all-extras pytest tests/test_menu.py`: names, the level list on the bundled levels and on a directory with a malformed file, the config round trip (no seed saved), flags over config over defaults, a corrupt config, `resolve`, and `parse_seed`.

## Known quirks

- `CONFIG` is `~/.config/hamsterball/game.json` on every OS, as spec §13 says; it ignores `XDG_CONFIG_HOME`. Tests point it elsewhere by passing `path` or patching `menu.CONFIG` (window.py reads the attribute at call time).
- The seed isn't remembered: spec §13 lists the last player, task, level, and difficulty.

## Decisions

- 2026-10-22, ticket hamsterball-v1/12: "levels whose spawns support the task" is read as levels with a block for the task, since every task block has a spawn (see [level.md](../core/level.md)).

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/12: added, test-first.
