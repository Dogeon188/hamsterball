# window.py

## Purpose

The game's window (spec §13): one resizable GLFW window on the main thread, the chase view drawn by `mjr_render` with shadows on, and the Dear ImGui HUD, pause menu, and result card on top. It's a thin shell over [play.py](play.md), which holds the logic. Ships in the `game` extra (`imgui-bundle`, `glfw`, `click`).

## Interface

- `hamsterball-play --task T --level FILE` (`main`, a click command): `--task` is one of the five tasks; `--level` a level YAML or a bundled level name. Both are required until the menu (ticket 12).
- `run(task, level, recording=False)`: plays episodes until Menu, Quit, or the window closes; returns the last command.

## How it works

- Setup: GLFW's default context (legacy GL 2.1 on macOS, compatibility elsewhere), 1280×720, `swap_interval(0)`; imgui-bundle's native GLFW backend and its **OpenGL 2** renderer, so imgui and MuJoCo share one compatibility context; no `imgui.ini`.
- The env is `HamsterballEnv([task], levels=[level], camera_action=True)` with no views; the window has its own `MjrContext`, `MjvScene` (shadow flag on), and a fixed `MjvCamera` on the model's `chase` camera with `views.option("chase")`. The env poses the view mocaps every step; the window runs `mj_kinematics` and `mj_camlight` (derived only) before `mjv_updateScene`. The context is rebuilt when the episode's model changes.
- Each frame: poll events, read held keys with `imgui.is_key_down` and presses with `imgui.is_key_pressed`, read the first gamepad's sticks; while playing, `Pacer.tick()`, then one `env.step(action(...))`, ending the episode on `outcome`. Paused or on the result card, physics doesn't step and the loop sleeps one period. Then draw the chase view, the HUD, and the pause or result window, swap, and run the flow's commands: `retry` resets with `{level: the canonical YAML, task}`, `next` with `{task}`, `menu` and `quit` leave.
- The overrun count resets each episode and shows on the result card.

## Invariants and checks

- `uv run --all-extras pytest tests/test_game.py -k window`: the module imports and `--help` works.
- Agent smoke runs (scratch scripts, not in CI) on an M-series Mac: 240 frames of `item_scatter` opened the window, rendered the chase view with the shell's shadow at 2560×1440 framebuffer, and counted 2 overruns: the second frame (after the first draw builds the `MjrContext`) and the frame the script read pixels back; a scripted result card, Enter, pause, and resume stepped only while playing.

## Known quirks

- **Needs play-test:** the whole window behaviour: feel of the keys and sticks, pacing and tearing without vsync, the HUD and cards' look and placement, mouse clicks on the pause buttons, retina scaling, resizing, and the shadow's jagged edge at the default shadow map size.
- Menu leaves the game (as Quit does) until ticket 12 adds the menu screen. The HUD always shows "not recording" until ticket 12 records.
- No vsync: vsync on top of the Pacer's schedule would beat against it and count false overruns; the cost is possible tearing.

## Decisions

- 2026-10-22, spec §13 and issue 18: one GLFW window and GL context on the main thread, `mjr_render` into the framebuffer, imgui-bundle's GLFW backend on top; the env's headless renderer stays separate.
- 2026-10-22, ticket hamsterball-v1/11: imgui's OpenGL 2 renderer, not OpenGL 3, since MuJoCo needs a compatibility context and macOS gives OpenGL 3 only as a core profile.
- 2026-10-22, ticket hamsterball-v1/11: the CLI uses click (in the `game` extra), the project's choice for new CLIs.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/11: added. Agent finding: an episode's second frame overruns, since the first frame builds the `MjrContext`; otherwise a scripted 4 s run on an M-series Mac held 60 Hz.
