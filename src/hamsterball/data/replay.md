# replay.py

## Purpose

Re-step replay (spec §12): a determinism check that recorded episodes reproduce bit for bit on this build. See [collector.md](collector.md) for what a dataset stores.

## Interface

- `replay(dataset, episodes=None)`: re-steps each episode (ids, or None for every episode in the dataset) and returns None when every core state matches.
- `ReplayMismatch(AssertionError)`: raised at the first differing step, with `episode`, `step` (0 is the reset), `field` (`<component>[<index>]`, e.g. `qvel[3]`), and `diff` (the largest absolute difference).
- `field(model, i)`: the name of core state element `i`.

## How it works

1. Warns once per key when the dataset's `hamsterball_version`, `mujoco_version`, or `platform` differs from `provenance()`, then carries on.
2. Builds the env with `dataset.recover_environment()`.
3. Per episode: `reset(seed, options)` from the episode metadata, compares the reset's `core_state`, then steps each stored action and compares each step's. Comparison is on raw bytes, so `-0.0` vs `0.0` and NaN payloads count as differences.
4. `field` walks the single-bit `mjtState` components in `mjSTATE_INTEGRATION`, in `mj_getState` order, sized by `mj_stateSize`.

## Invariants and checks

- `uv run pytest tests/test_data.py`: an episode of every task replays bit for bit (CI runs it on macOS arm64 and Linux x86_64, Python 3.11 and 3.14), a recording under a different drag constant fails at step 1, and a different stored MuJoCo version and platform warn.

## Known quirks

- Replay stops at the first mismatch across all episodes; it doesn't continue to the next episode.

## Decisions

- 2026-10-22, ticket hamsterball-v1/08: a mismatch raises `ReplayMismatch`, an `AssertionError`, rather than returning a report, so a test or script fails on it without extra checks.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/08: added.
