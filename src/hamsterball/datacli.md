# datacli.py

## Purpose

`hamsterball-data`, [later efforts](../../.scratch/later-efforts/spec.md) §7: the command line for the dataset functions in `hamsterball.data`, which v1 shipped as Python functions only. Each subcommand wraps one function and adds nothing to it.

## Interface

`hamsterball-data` (`main`, a click group). A DATASET is a Minari dataset id, under `~/.minari` or `MINARI_DATASETS_PATH`. Without the `data` extra every subcommand exits 1 asking for `hamsterball[data]`.

- `replay DATASET [--episode N]...`: [replay.md](data/replay.md)'s `replay()` on the given episodes (default all). Prints `<n> episodes replay bit for bit` and exits 0; a `ReplayMismatch` exits 1 with its message (episode, step, field, difference). Provenance warnings go to stderr.
- `rerender DATASET OUT_ID --view V... [--size HxW]`: [rerender.md](data/rerender.md)'s `rerender()`. `--view` is required and repeatable (`chase`, `follow`, `first_person`); `--size` defaults to 84x84, the env's default.
- `video DATASET EPISODE VIEW PATH [--size HxW]`: `to_video()`. `--size` defaults to the dataset's view size and must be even.
- `lerobot DATASET REPO_ID ROOT [--images]`: [lerobot.md](data/lerobot.md)'s `to_lerobot()`, lossy video by default, lossless PNG frames with `--images`. Exits 1 when the `lerobot` extra is missing or `to_lerobot` raises `ValueError` (no views, or a frame skip without a whole-number fps).

A `--size` that doesn't parse as `HxW` is a usage error (exit 2). Other errors, such as an unknown dataset or episode id, pass through as tracebacks.

## How it works

- The group callback imports Minari before any subcommand runs, so a missing `data` extra gives one message.
- Each subcommand imports its function from `hamsterball.data` inside its body: `hamsterball.data` imports Minari on import, and LeRobot loads only for `lerobot`.

## Invariants and checks

- `uv run --all-extras pytest tests/test_datacli.py`: replay passes a clean dataset and counts its episodes, exits 1 naming the diverging step on a recording made under another drag constant, a bad `--size` is a usage error, rerender then video then lerobot run end to end (the lerobot half skips without its extra, so on Python 3.11), and a missing Minari names the extra.

## Known quirks

- `replay` stops at the first mismatch across all episodes, as `replay()` does.

## Decisions

- 2026-10-23, ticket data-clis/01: one group with four subcommands, not four scripts, so one module holds the Minari check and one `[project.scripts]` entry covers them.
- 2026-10-23, ticket data-clis/01: a top-level module, not inside `hamsterball.data`, for the reason [baseline.md](baseline.md) logs: the CLI must name the extra when Minari is missing.
- 2026-10-23, ticket data-clis/01: `replay` keeps stopping at the first mismatch, so `data/replay.py` stays unchanged.

## Tried and rejected

## History

- 2026-10-23, ticket data-clis/01: added.
