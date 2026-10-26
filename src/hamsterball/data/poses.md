# poses.py

## Purpose

The dataset pose stream ([pose-stream spec](../../../.scratch/pose-stream/spec.md)): a recorded episode's static header and per-env-step poses, computed without stepping physics, so it works on any platform, as [re-render](rerender.md) does.

## Interface

- `pose_stream(dataset, episode)`: a generator. Yields a header dict, then n + 1 poses for an n-step episode. The header: `pose_format` (1), `level` (the canonical level YAML, hfield samples inline), `level_id`, `task`, `rate` (env steps per second, `1 / (frame_skip * DT)`), `step_limit` (in env steps). Each pose is `HamsterballEnv.pose()`'s dict ([env.md](../env.md)).
- `hamsterball-data poses DATASET EPISODE PATH` writes it as JSON Lines ([datacli.md](../datacli.md)).

## How it works

Recovers the env with no views, then drives it with [rerender.md](rerender.md)'s `walk` (`reset(seed, options)`, then `step_to` over each stored core state) and calls `pose()` after each. Marker progress and the outcome come from `step_to`'s recomputation, so a pose never needs history beyond the core states.

## Invariants and checks

- `uv run --all-extras pytest tests/test_pose_stream.py`: on `locked_goal`, `waypoint_loop`, `survive_raft`, `survive_seesaw`, `survive_collapse`, and `survive_sweepers`, the stream's poses equal the live `pose()` list collected while recording, with `physics_step` patched to raise; the header's YAML loads to its level id; the CLI writes 1 + (n + 1) lines equal to the function's.

## Known quirks

- `step_limit` reads the env's `_max_steps`, which is private; there is no public accessor.
- An unknown episode id raises, as the other data functions do.

## Decisions

- 2026-10-27, ticket pose-stream/03: header key `rate` holds the env-step rate in Hz, `step_limit` the limit in env steps; spec names the contents, not the keys.
- 2026-10-27, ticket pose-stream/03: one function yielding the header first, so the CLI is a line-per-item writer.

## Tried and rejected

## History

- 2026-10-27, ticket pose-stream/03: added.
