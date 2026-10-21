# play.py

## Purpose

The game's logic without a display (spec §13): input to the 3-D action, real-time pacing with overrun counting, the play/pause/result flow, and the HUD and result-card text. [window.py](window.md) is the shell that feeds it.

## Interface

- `action(keys, left=(0, 0), right_x=0)`: `[run_x, run_y, cam_yaw]` (camera frame, x right, y forward) from the held key names (`w a s d shift ctrl q e`) and GLFW stick axes (y down).
- `Pacer(period, clock=perf_counter, sleep=time.sleep, slack=0.002)`: `tick()` before each env step sleeps to the frame's deadline; `overruns` counts late frames; `reset()` starts a fresh schedule (after a pause or at an episode start).
- `Flow`: `state` is `"play"`, `"paused"`, `"result"` (then `"menu"` or `"quit"` once left); `end(outcome)` shows the result card; `press(name)` takes `esc`, `enter`, `r`, the pause buttons `resume retry menu quit`, or `close`, and returns the shell's command (`retry`, `next`, `menu`, `quit`) or `None`.
- `outcome(terminated, truncated, info)`: the result card's headline (`Success`, `Failed: hazard`, `Failed: fell off`, `Failed: left the region`, `Time up`) or `None`.
- `hud(env, action, recording)`: five lines: `t / limit s`, task progress, `REC` or `not recording`, `effort e`, `speed v m/s`. `sim_time(env)` and `progress(env)` are its parts.

## How it works

- Keys: WASD sum to a direction, normalised, at effort 0.5; Shift 0.25, Ctrl 1.0, both 0.25. Q gives `cam_yaw` +1 (the yaw rises, so the view turns left), E −1, both 0.
- Sticks: a radial deadzone of 0.15, effort `(m − 0.15) / 0.85` capped at 1. The left stick outside its deadzone replaces the WASD run; the right stick's x outside its deadzone replaces Q/E, right turning right. Each stick overrides only its own keys.
- Pacer: deadlines follow an ideal 1/60 s schedule. A tick past its deadline by more than `slack` counts an overrun and restarts the schedule at now, so a slow machine slows the game and never bursts steps to catch up; an early tick sleeps.
- Flow is a table `state → {input: (next state, command)}`. Esc toggles pause; R and Enter work only on the result card; `close` quits from anywhere. Leaving by retry, menu, quit, or close before the result is an abandoned run (ticket 12 discards it).
- HUD progress: waypoint `passed+1 / n` (capped at `n/n` after a win), items and keys `collected / n` (then `keys n/n: reach the goal`), `reach the goal`, survive `limit − t` left. Speed is the shell's world speed (`qvel[0:3]`).

## Invariants and checks

- `uv run --all-extras pytest tests/test_game.py`: key and stick mapping, deadzone override, pacing and overruns on a fake clock, the flow table, the HUD at reset on every bundled task level and after one second of sprinting, waypoint progress capped after a win, the outcome names, and the window module's import and `--help` (skipped without the `game` extra).

## Known quirks

- `progress` and `sim_time` read the env's private `_todo`, `_steps`, and `_max_steps`, to keep env.py untouched while tickets 08, 10, and 14 change it. A public accessor is the fix if a second reader appears.
- `slack` 2 ms and the deadzone 0.15 need play-test; the slack keeps a late OS wake-up from counting as an overrun.
- The gamepad has no walk or sprint buttons: the stick's deflection is the effort.

## Decisions

- 2026-10-22, [ADR 0001](../../../docs/adr/0001-mujoco-core-game-wraps-env.md): the game drives `HamsterballEnv` with `camera_action=True`; nothing here steps the core directly.
- 2026-10-22, ticket hamsterball-v1/11: Q turns the view left (+`cam_yaw`) and E right, and pushing the right stick right turns right. Spec §13 says only "Q/E set `cam_yaw` to ±1".
- 2026-10-22, ticket hamsterball-v1/11: each stick overrides only its own keys (left stick the run, right stick the turn), so a player can steer with the stick and turn with Q/E.
- 2026-10-22, ticket hamsterball-v1/11: an overrun is a frame more than `slack` past its deadline, and the schedule restarts from it (no catch-up), per spec §13's "slows down rather than skipping steps".

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/11: added, test-first.
- 2026-10-22, code review of hamsterball-v1: the waypoint HUD read `waypoint 5/4` after a win; it now caps at `n/n`.
