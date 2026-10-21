# rerender.py

## Purpose

Re-render (spec §12): turns a state-only dataset from the [Collector](collector.md) into pixels without stepping physics, so it works on any platform, whatever the recording's build.

## Interface

- `rerender(dataset, views, view_size, out_id)`: writes and returns a new Minari dataset `out_id` (Arrow, `jpeg_encoding=False`) whose observations are `state`, `goal`, and one `(H, W, 3)` uint8 frame per view. Actions, rewards, terminations, truncations, infos (`core_state`), seeds, reset options, and episode metadata carry over; so do the source's `algorithm_name`, `author`, `author_email`, `code_permalink`, `description`, `requirements`, and provenance (`hamsterball_version`, `mujoco_version`, `platform`: where the physics ran). The `env_spec` gains `views`, `view_size`, and `obs_state=True`, so `recover_environment()` rebuilds an env with the new observation space.
- `episodes(dataset, views, view_size, ids=None)`: yields `(episode, metadata, observations)` per episode, the observations stacked per key over the episode's steps (reset included).

## How it works

1. Builds the env with `dataset.recover_environment(views=..., view_size=..., obs_state=True)`.
2. Per episode: `reset(seed, options)` from the episode metadata. Reset is `mj_resetData` plus the spawn, with no physics stepped, so it lands on the stored first core state exactly.
3. Each later stored core state goes through `HamsterballEnv.step_to(core_state)` (see [env.md](../env.md)): the env sets the state in place of stepping physics, then runs the same post-physics step as `step`: the swept marker tests from the previous shell centre (so progress, hidden items and keys, the unlocked goal, and an inactive goal after success all follow the recording), the reward, and the observation, which poses the views and renders them.
4. `rerender` writes one episode at a time with `storage.update_episodes`, so only one episode's frames are in memory, then rewrites each episode's metadata from the source.

## Invariants and checks

- `uv run --locked --all-extras pytest tests/test_rerender.py`: with `physics_step` patched to raise, re-render recomputes `goal` and `state` equal bit for bit to the recorded ones on an `unlock_goal` episode played to success, and its frames equal bit for bit the frames a live pixel env renders re-stepping the same actions.

## Known quirks

- Re-render trusts the stored core states; it doesn't check them against physics. Run [replay](replay.md) for that.
- `rewards` are copied, not recomputed, though `step_to` recomputes the same ones.

## Decisions

- 2026-10-22, ticket hamsterball-v1/09: the env's step splits into the physics and `_advance()` (everything after it), and the public `step_to(core_state)` swaps the physics for `Core.set_state`. Re-render reuses the env's marker tests, marker looks, and view posing through it rather than copying them.
- 2026-10-22, ticket hamsterball-v1/09: the re-rendered `state` and `goal` are the recomputed ones, not copies, so a test comparing them to the recording checks the recomputation.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/09: added.
