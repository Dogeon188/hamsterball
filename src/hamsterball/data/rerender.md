# rerender.py

## Purpose

Re-render (spec §12): turns a state-only dataset from the [Collector](collector.md) into pixels without stepping physics, so it works on any platform, whatever the recording's build.

## Interface

- `rerender(dataset, views, view_size, out_id)`: writes and returns a new Minari dataset `out_id` (Arrow, `jpeg_encoding=False`) whose observations are `state`, `goal`, and one `(H, W, 3)` uint8 frame per view. Actions, rewards, terminations, truncations, infos (`core_state`), seeds, reset options, and episode metadata carry over; so do the source's `algorithm_name`, `author`, `author_email`, `code_permalink`, `description`, `requirements`, and provenance (`hamsterball_version`, `mujoco_version`, `platform`: where the physics ran). The `env_spec` gains `views`, `view_size`, and `obs_state=True`, so `recover_environment()` rebuilds an env with the new observation space.
- `to_video(dataset, episode, view, path, view_size=None)`: writes episode id `episode`, re-rendered from `view`, as an H.264 (`libx264`, yuv420p) mp4 at the env-step rate, `1 / (frame_skip · DT)` (60 fps at the default `frame_skip` 10). `view_size` is `(H, W)`, by default the dataset's `env_spec` one. One frame per stored core state, the reset's included.
- `walk(env, ep, meta)`: the per-episode loop `episodes` and [poses.md](poses.md) share: yields `env.reset(seed, options)`'s observation, then `env.step_to(core_state)`'s for each later stored core state. The caller reads the env between yields.
- `episodes(dataset, views, view_size, ids=None)`: yields `(episode, metadata, observations)` per episode, the observations stacked per key over the episode's steps (reset included).

## How it works

1. Builds the env with `dataset.recover_environment(views=..., view_size=..., obs_state=True)`.
2. Per episode, `walk` does `reset(seed, options)` from the episode metadata. Reset is `mj_resetData` plus the spawn, with no physics stepped, so it lands on the stored first core state exactly.
3. Each later stored core state goes through `HamsterballEnv.step_to(core_state)` (see [env.md](../env.md)): the env sets the state in place of stepping physics, then runs the same post-physics step as `step`: the swept marker tests from the previous shell centre (so progress, hidden items and keys, the unlocked goal, and an inactive goal after success all follow the recording), the reward, and the observation, which poses the views and renders them.
4. `rerender` writes one episode at a time with `storage.update_episodes`, so only one episode's frames are in memory, then rewrites each episode's metadata from the source.

## Invariants and checks

- `uv run --locked --all-extras pytest tests/test_rerender.py`: with `physics_step` patched to raise, re-render recomputes `goal` and `state` equal bit for bit to the recorded ones on an `unlock_goal` episode played to success, and its frames equal bit for bit the frames a live pixel env renders re-stepping the same actions, the hamster's features ([views.md](../core/views.md)) among them after a turn. `to_video` writes one decodable frame per core state at 30 fps under `frame_skip=20`.

## Known quirks

- Re-render trusts the stored core states; it doesn't check them against physics. Run [replay](replay.md) for that.
- `to_video` needs even `view_size` numbers (yuv420p halves the chroma), and its default is the dataset's `view_size`, 84×84 unless the recording set one; pass a larger size for people to watch.
- `to_video` uses PyAV (`av`, in the `data` extra), whose wheels bundle FFmpeg with `libx264`, so no system FFmpeg is needed. Spec §2's `data` group doesn't list it.
- `rewards` are copied, not recomputed, though `step_to` recomputes the same ones.

## Decisions

- 2026-10-22, ticket hamsterball-v1/09: the env's step splits into the physics and `_advance()` (everything after it), and the public `step_to(core_state)` swaps the physics for `Core.set_state`. Re-render reuses the env's marker tests, marker looks, and view posing through it rather than copying them.
- 2026-10-22, ticket hamsterball-v1/09: the re-rendered `state` and `goal` are the recomputed ones, not copies, so a test comparing them to the recording checks the recomputation.
- 2026-10-22, ticket hamsterball-v1/09: `to_video` takes an optional `view_size` beyond spec §12's `(dataset, episode, view, path)`, since 84×84 is too small to watch; it encodes lossy H.264, since the mp4 is for people, and the lossless frames live in `rerender`'s dataset.

- 2026-10-27, ticket pose-stream/03: the reset-then-`step_to` loop became `walk`, so the pose stream reuses it rather than copying it; `episodes` behaves as before.

- 2026-10-27, ticket articulated-tier/05: articulated datasets re-render with no change: `proprio`, the wall-contact flag in `state`, and progress are recomputed by the env from each core state (collision included) and stack like any observation key; the stored `run_vector` info carries over, and the dataset metadata key `low_level_policy` is copied with the provenance keys. `tests/test_articulated_data.py` checks the three equal the live record with `physics_step` disabled.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/09: added.
- 2026-10-23, code review of mouse-camera tickets 04–05: the tail's re-render test moved here from `test_views.py` and goes through `rerender` (it called `env.step_to` directly): 12 turning steps recorded, the chase re-rendered at 48×64, bit for bit the live frames. At that size the tail covers 4 chase pixels and none of the follow view's, so the test checks the chase only.
- 2026-10-27, ticket pose-stream/03: extracted `walk` for the pose stream; no rerender test changed.
