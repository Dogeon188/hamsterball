# window.py

## Purpose

The game's window (spec §13): one resizable GLFW window on the main thread, the chase view drawn by `mjr_render` with shadows on, and the Dear ImGui HUD, pause menu, and result card on top. It's a thin shell over [play.py](play.md), [menu.py](menu.md), and [session.py](session.md), which hold the logic. Ships in the `game` extra (`imgui-bundle`, `glfw`, and the `data` extra for recording); `click` is a core dependency.

## Interface

- `hamsterball-play [--levels DIR] [--player NAME] [--no-record] [--task T] [--level FILE|NAME|generated] [--seed S] [--difficulty D]` (`main`, a click command). `--levels` defaults to the bundled levels; `--level` is a level YAML, a stem in `--levels`, or `generated`; `--seed` is an integer ≥ 0 and `--difficulty` in [0, 1]. With `--task` and `--level`, and a valid `--player` (or a remembered one) unless `--no-record`, the game skips the menu.
- `run(settings, levels_dir, record, skip_menu)`: the menu (unless skipped), then episodes with its choices, back to the menu on Menu, until Quit or the window closes.

## How it works

- Start: `main` prefills `{player, task, level, difficulty, seed}` with `menu.prefill(flags, load_config(menu.CONFIG))`. Each time play starts, `run` saves the choices with `save_config`, resolves the level, and builds a `Session`.
- Menu: one full-window imgui window with the player name, a task combo, the level combo (`level_choices` for every task, computed when the menu opens), and for "generated" a difficulty slider and a seed field. Problems (a bad name while recording, a bad seed) show in red and disable Play; the menu shows the dataset id or "not recording". Quit or closing the window ends the game.
- Setup: GLFW's default context (legacy GL 2.1 on macOS, compatibility elsewhere), 1280×720, `swap_interval(0)`; imgui-bundle's native GLFW backend and its **OpenGL 2** renderer, so imgui and MuJoCo share one compatibility context; no `imgui.ini`.
- The env is the session's (`Hamsterball-v<N>` with `camera_action=True`, wrapped in the Collector when recording) with no views; the window has its own `MjrContext`, `MjvScene` (shadow flag on), and a fixed `MjvCamera` on the model's `chase` camera with `views.option("chase")`. The env poses the view mocaps every step; the window runs `mj_kinematics` and `mj_camlight` (derived only) before `mjv_updateScene`. The context is rebuilt when the episode's model changes.
- Each frame: poll events, read held keys with `imgui.is_key_down` and presses with `imgui.is_key_pressed`, read the first gamepad's sticks; while playing, `Pacer.wait()`, then one `session.step(action(...), overruns)`, ending the episode on `outcome`. Paused or on the result card, physics doesn't step and the loop sleeps one period. Then draw the chase view, the HUD, and the pause or result window, swap, and run the flow's commands: `retry` is `session.start(retry=True)` (same level and seed), `next` is `session.start()`, `menu` and `quit` leave. `session.step` gets the Pacer's overrun count, so a finished episode is saved with it; leaving `_play` closes the session, which discards a run in progress.
- The overrun count resets each episode and shows on the result card, with "saved to <dataset id>" or "not recorded".

## Invariants and checks

- `uv run --all-extras pytest tests/test_game.py -k "window or cli"`: the module imports, `--help` lists every spec §13 flag, and `main` skips the menu only with a task, a level, and a valid name or `--no-record`, prefilling from flags over the config file.
- Agent smoke runs (scratch scripts, not in CI) on an M-series Mac: 240 frames of `item_scatter` opened the window, rendered the chase view with the shell's shadow at 2560×1440 framebuffer, and counted 2 overruns: the second frame (after the first draw builds the `MjrContext`) and the frame the script read pixels back; a scripted result card, Enter, pause, and resume stepped only while playing.

## Known quirks

- **Needs play-test:** the whole window behaviour: feel of the keys and sticks, pacing and tearing without vsync, the HUD and cards' look and placement, mouse clicks on the pause buttons, retina scaling, resizing, and the shadow's jagged edge at the default shadow map size.
- **Needs play-test (ticket 12):** the menu screen: its layout, typing in the name and seed fields, the combos, the disabled Play button, Menu → menu → Play again, and the REC indicator. A level file that fails to compile with `--task`/`--level` raises out of the window rather than showing an error.
- No vsync: vsync on top of the Pacer's schedule would beat against it and count false overruns; the cost is possible tearing.

## Decisions

- 2026-10-22, spec §13 and issue 18: one GLFW window and GL context on the main thread, `mjr_render` into the framebuffer, imgui-bundle's GLFW backend on top; the env's headless renderer stays separate.
- 2026-10-22, ticket hamsterball-v1/11: imgui's OpenGL 2 renderer, not OpenGL 3, since MuJoCo needs a compatibility context and macOS gives OpenGL 3 only as a core profile.
- 2026-10-22, ticket hamsterball-v1/11: the CLI uses click (in the `game` extra), the project's choice for new CLIs.
- 2026-10-22, user ruling: keep Click for every CLI; it moves to the core dependencies, so the `game` extra drops it.
- 2026-10-22, ticket hamsterball-v1/12: `--task` and `--level` skip the menu only when recording has a valid player name (from `--player` or the config file) or `--no-record` is set; otherwise the menu opens prefilled, to ask for the name. Spec §13 doesn't say what happens without a name.
- 2026-10-22, ticket hamsterball-v1/12: the menu is drawn as a full-window imgui window, which also paints the background, so the menu needs no GL clear call and no PyOpenGL.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/11: added. Agent finding: an episode's second frame overruns, since the first frame builds the `MjrContext`; otherwise a scripted 4 s run on an M-series Mac held 60 Hz.
- 2026-10-22, ticket hamsterball-v1/12: added the menu, the `--levels`, `--player`, `--no-record`, `--seed`, and `--difficulty` flags, and recording through [session.py](session.md). Agent smoke run (scratch script, not in CI) on an M-series Mac: 40 frames of the menu with "generated" chosen, then 40 frames each of a recording `first_survive` session and a non-recording generated one, each closed by a faked window close; no errors.
