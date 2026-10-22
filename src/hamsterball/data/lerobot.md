# lerobot.py

## Purpose

LeRobot export (spec §12): a one-way copy of a [re-rendered](rerender.md) Minari dataset as a LeRobot v3 dataset, for LeRobot's multi-view imitation policies. Minari stays the source of truth.

## Interface

- `to_lerobot(dataset, repo_id, root, use_videos=True)`: writes and returns a `LeRobotDataset` at `root`. `dataset` must hold views (from `rerender`), or it raises `ValueError`.
- Needs the `lerobot` extra (`lerobot[dataset]>=0.6`), which installs only on Python 3.12 and 3.13; `import hamsterball.data` doesn't import it.

## How it works

- Features: `action`; `observation.state` (21) and `observation.goal` (10), float32; `observation.images.<view>` per view, `(H, W, 3)`; `next.reward` (float32) and `next.done` (bool, terminated or truncated), shape 1.
- A LeRobot frame pairs observation t with action t, so each episode gives one frame per action and drops the last observation.
- Each frame's task string is the episode's task (`options["task"]`).
- `fps` is the env-step rate, `1 / (frame_skip · DT)`; LeRobot takes an int, so a `frame_skip` that doesn't divide 600 raises `ValueError`.
- `use_videos=True` is LeRobot's default, lossy AV1 video (`libsvtav1`, in PyAV's wheels); `False` stores PNG frames, lossless.

## Invariants and checks

- `uv run --locked --all-extras --python 3.13 pytest tests/test_rerender.py`: an exported episode loads back with LeRobot, with the frame count, fps, task, action, state, goal, reward, done, and (with `use_videos=False`) the frames equal to the re-rendered dataset's. The test skips where the extra isn't installed (Python 3.11 and 3.14, so CI skips it).

## Known quirks

- `lerobot` 0.6 requires Python ≥ 3.12, and pins `numpy<2.3`, `pillow<13`, and `av<16`. numpy 2.2 has no Python 3.14 wheels; built from source on 3.14 its `assert_array_equal` failed on equal uint8 arrays. So the extra stops below 3.14, and the lock holds numpy 2.2.6, pillow 12, and av 15 for Python 3.12 and 3.13 (core and the other extras included), the newer ones elsewhere. The `data` extra's `pillow` and `av` are unbounded for the same reason.
- The `all` extra doesn't include `lerobot`, though spec §2 says `all` is every extra but `gpu`.
- No seed, level, or reset options reach LeRobot (it has no slot for them); keep the Minari dataset to replay or re-render.

## Decisions

- 2026-10-22, ticket hamsterball-v1/09: `state` and `goal` stay two features, `observation.state` and `observation.goal`, rather than one concatenated `observation.state`, so nothing is relabelled.

## Tried and rejected

- 2026-10-22, ticket hamsterball-v1/09: `lerobot` without a version bound resolved to 0.1.0 (pre-v3 format), since 0.6 conflicts with `pillow>=13` and `av>=18` and needs Python ≥ 3.12. Bounded to `>=0.6`, with the marker above.

## History

- 2026-10-22, ticket hamsterball-v1/09: added. Agent check: a `use_videos=True` export decodes back to `(3, H, W)` frames.
- 2026-10-22, ticket mouse-camera/02: `observation.state` is 21 (the camera pitch's sin and cos).
