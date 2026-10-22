# play.py

## Purpose

The game's logic without a display (spec §13): input to the 4-D action, real-time pacing with overrun counting, the play/pause/result flow, and the HUD and result-card text. [window.py](window.md) is the shell that feeds it.

## Interface

- `action(keys, left=(0, 0), right=(0, 0), mouse=(0, 0))`: `[run_x, run_y, turn, cam_pitch]` (camera frame, x right, y forward) from the held key names (`w a s d shift ctrl q e`), the GLFW stick axes (y down), and this frame's `Mouse.move` pair.
- `Mouse(sens=MOUSE_SENS)`: `move(dx, dy)` turns pixels of motion into this frame's `(turn, cam_pitch)`; `clear()` drops the carried-over motion on both axes.
- `Pacer(period, clock=perf_counter, sleep=time.sleep, slack=0.002)`: `wait()` before each env step sleeps to the frame's deadline; `overruns` counts late frames; `reset()` starts a fresh schedule (after a pause or at an episode start).
- `Flow`: `state` is `"play"`, `"paused"`, `"result"` (then `"menu"` or `"quit"` once left); `end(outcome)` shows the result card; `press(name)` takes `esc`, `enter`, `r`, the pause buttons `resume retry menu quit`, or `close`, and returns the shell's command (`retry`, `next`, `menu`, `quit`) or `None`.
- `outcome(terminated, truncated, info)`: the result card's headline (`Success`, `Failed: hazard`, `Failed: fell off`, `Failed: left the region`, `Time up`) or `None`.
- `hud(env, action, recording)`: five lines: `t s` (`t / limit s` for survive), task progress, `REC` or `not recording`, `effort e`, `speed v m/s`. `sim_time(env)` and `progress(env)` are its parts.

## How it works

- Keys: WASD sum to a direction, normalised, at effort 0.5; Shift 0.25, Ctrl 1.0, both 0.25. Q gives `turn` +1 (the facing rises, so the view turns left), E −1, both 0.
- Sticks: a radial deadzone of 0.15, effort `(m − 0.15) / 0.85` capped at 1. The left stick outside its deadzone replaces the WASD run; the right stick's x outside its deadzone replaces Q/E, right turning right, and its y sets `cam_pitch`, pushing up looking up (GLFW y is down and the pitch is below the horizon, so y passes through unflipped). The right stick's deadzone is per axis, so each axis overrides only its own input. No keys drive the pitch.
- Mouse: `move(dx, dy)` adds `(−dx, dy) · sens` to a two-axis carry (right turns right, up looks up), returns it clipped to ±1 per axis, and keeps the rest, so a flick turns at the full 120°/s orbit rate (or looks up and down at 90°/s) over as many frames as it takes, each axis on its own. A non-zero mouse turn overrides the right stick's x and Q/E; a non-zero mouse pitch overrides the right stick's y. With no motion and nothing carried, an axis is 0 and the stick and keys work as before.
- Pacer: deadlines follow an ideal 1/60 s schedule. A `wait` past its deadline by more than `slack` counts an overrun and restarts the schedule at now, so a slow machine slows the game and never bursts steps to catch up; an early `wait` sleeps.
- Flow is a table `state → {input: (next state, command)}`. Esc toggles pause; R and Enter work only on the result card; `close` quits from anywhere. Leaving by retry, menu, quit, or close before the result is an abandoned run (ticket 12 discards it).
- HUD progress: waypoint `passed+1 / n` (capped at `n/n` after a win), items and keys `collected / n` (then `keys n/n: reach the goal`), `reach the goal`, survive `limit − t` left. Speed is the shell's world speed (`qvel[0:3]`).

## Invariants and checks

- `uv run --all-extras pytest tests/test_game.py`: key, stick, and mouse mapping (a flick spent over frames, the two axes' independent carries, each axis's override of its own stick axis and keys, the right stick's pitch direction), deadzone override, pacing and overruns on a fake clock, the flow table, the HUD at reset on every bundled task level (elapsed time only, except survive) and after one second of sprinting, waypoint progress capped after a win, the outcome names, and the window module's import and `--help` (skipped without the `game` extra).

## Known quirks

- `progress` and `sim_time` read the env's private `_todo`, `_steps`, and `_max_steps`, to keep env.py untouched while tickets 08, 10, and 14 change it. A public accessor is the fix if a second reader appears.
- `slack` 2 ms and the deadzone 0.15 need play-test; the slack keeps a late OS wake-up from counting as an overrun.
- The gamepad has no walk or sprint buttons: the stick's deflection is the effort.
- **Needs play-test:** `MOUSE_SENS` 0.05 `turn` per pixel, i.e. 0.1° per pixel (`turn` 1 turns 2° in one 1/60 s step). A guess from the plan. The pitch shares it: 0.075° per pixel at the 90°/s pitch rate.
- **Needs play-test:** the right stick's pitch direction (pushing up looks up, not the flight-style inverse); there is no invert option.

## Decisions

- 2026-10-22, [ADR 0001](../../../docs/adr/0001-mujoco-core-game-wraps-env.md): the game drives `HamsterballEnv` with `camera_action=True`; nothing here steps the core directly.
- 2026-10-22, ticket hamsterball-v1/11: Q turns the view left (+`cam_yaw`) and E right, and pushing the right stick right turns right. Spec §13 says only "Q/E set `cam_yaw` to ±1".
- 2026-10-22, ticket hamsterball-v1/11: each stick overrides only its own keys (left stick the run, right stick the turn), so a player can steer with the stick and turn with Q/E.
- 2026-10-22, ticket hamsterball-v1/11: an overrun is a frame more than `slack` past its deadline, and the schedule restarts from it (no catch-up), per spec §13's "slows down rather than skipping steps".

- 2026-10-22, user play-test request: no time-up for a human player except in survive (the session's env has `truncate=False`), so the HUD's time line shows elapsed sim time only for every other task; survive keeps `t / limit s` and its `s left` progress line. The result card already showed elapsed time only; its `Time up` headline can now come only from an env that truncates.

- 2026-10-22, ticket mouse-camera/01: mouse motion caps at `cam_yaw` ±1 per step and carries the rest, so the env and the recorded 3-D action stay as they are; a frame with mouse motion, carry included, overrides the right stick and Q/E.
- 2026-10-22, ticket mouse-camera/03: the mouse pitch shares `MOUSE_SENS` with the yaw, and each mouse axis overrides only its own stick axis, so a mouse player can still tilt with the stick while turning, and vice versa. The right stick keeps the per-axis deadzone its x already had (the ticket called it radial), so a mostly-horizontal push doesn't also tilt.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/11: added, test-first.
- 2026-10-22, code review of hamsterball-v1: the waypoint HUD read `waypoint 5/4` after a win; it now caps at `n/n`.
- 2026-10-22, code review of hamsterball-v1: `Pacer.tick()` renamed `Pacer.wait()`, since GLOSSARY.md avoids "tick" (it means a physics step there).
- 2026-10-22, user play-test request: no time-up for a human player, except in survive; the HUD drops the limit from the time line of every other task.
- 2026-10-22, ticket mouse-camera/01: added `Mouse` and `action`'s `mouse_cam`, test-first.
- 2026-10-22, ticket mouse-camera/02: `action` returns 4-D with `cam_pitch` 0, matching the env's camera-flag action.
- 2026-10-22, ticket mouse-camera/03: the mouse's y and the right stick's y drive `cam_pitch`, test-first; `action` takes `right` and `mouse` as pairs in place of `right_x` and `mouse_cam`.
- 2026-10-22, code review of mouse-camera: doc wording only: the mouse "looks up and down" at 90°/s (was "tilts"; tilt is a mover's rocking in GLOSSARY.md).
- 2026-10-22, ticket mouse-camera/04 (rename, no behaviour change): "camera yaw" is now **facing** (GLOSSARY.md): the action dim `cam_yaw` is `turn`.
