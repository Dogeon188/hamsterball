# model.py

## Purpose

Builds the hamsterball model of spec §3: the shell, a control tier's hamster fragment, and a level's static geoms, compiled into one `MjModel` at the frozen 1/600 s physics step.

## Interface

- `build(tier, level=FLAT, view_size=GAME_SIZE, bodies="", slots=0) -> MjModel`: `view_size` is `(height, width)`; the model's offscreen buffer (`<visual><global offwidth offheight>`) is the larger of it and `GAME_SIZE` (720, 1280, the game window) in each dimension, so env views and the game both fit. `level` is top-level MJCF sections (`<asset>`, `<worldbody>` with geoms only, no bodies), placed after the light's worldbody and before the shell's. `FLAT` is an infinite ground plane. `bodies` is top-level MJCF sections too (`Level.body_mjcf()`: a `<worldbody>` of kinematic bodies and seesaws, and a `<contact>` of excludes), placed last, so its bodies follow the view mocaps (then `Level.tile_mjcf()`'s tiles); `""` leaves the model as before. `slots` adds userdata slots after the tier's (one per collapsing tile); 0 leaves `nuserdata` as before.
- Constants the rest of the core imports: `GAME_SIZE`, `DT`, `R`, `T`, `M_SHELL`, `M_HAM`, `SOLREF`, `TILE_NORMALS` (320×3, shell frame), and the fixed ids `SHELL`, `HAM`, `SHELL_QPOS`, `HAM_QPOS`, `SHELL_DOF`, `HAM_DOF`.

## How it works

- The shell is a free body whose origin is the shell centre: 320 box tiles on a Fibonacci sphere carry its 0.8 kg (contype and conaffinity 2, group 3), and a massless outer sphere (contype 1, `condim 6`, friction `1 0.05 0.04`, group 1) touches the level.
- The tier's `fragment` goes in the worldbody after the shell, then the view mocaps of [views.md](views.md), and `nuserdata` is 1 (the core's camera yaw) plus the tier's slots plus `slots` (a level's tiles).
- Floats go into the MJCF with `repr` of plain Python floats (NumPy 2's `repr` of `np.float64` isn't valid MJCF), so tile poses match the spec's formulas to full precision.
- Integrator `implicitfast`, as in the prototype that set the feel.
- `<visual><map znear="0.001"/>`: the near clip plane is a fraction of the model's extent, which large levels grow; the default 0.01 cut the ground below the first-person eye on a 30 m level. It changes pixels only.

## Invariants and checks

- The shell is body 1 and the hamster body 2, both free, so the shell's qpos starts at 0 and the hamster root's at 7 (dofs 0 and 6). `build()` asserts this and that the hamster subtree weighs 2.4 kg. Later bodies, such as the view mocaps and then a level's kinematic bodies and seesaws, must come after the hamster fragment.
- `uv run pytest tests/test_model.py` checks the model against spec §3. A compiled `MjModel` keeps no per-geom mass, so the test checks the shell body's 0.8 kg, which also shows the outer sphere is massless.

## Known quirks

- The prototype wrote tile poses with 4 decimals; this build writes them exactly, a sub-0.1 mm difference.

## Decisions

- 2026-10-21, [ADR-0001](../../../docs/adr/0001-mujoco-core-game-wraps-env.md): primitive geoms only (box tiles, a sphere), so MJX and MuJoCo Warp load the same model.
- 2026-10-21, [ADR-0002](../../../docs/adr/0002-exact-mujoco-pin-bumps-env-version.md): `mujoco==3.16.0`, pinned exactly in `pyproject.toml`.
- 2026-10-21, ticket hamsterball-v1/01: hard-coded body, qpos, and dof ids for the shell and the hamster root, in place of name lookups, so the per-step code reads like a kernel.
- 2026-10-22, ticket survive-movers/03: kinematic bodies compile after the view mocaps, as free-jointed, gravity-compensated 1000 kg bodies excluded from contact with the world and with each other; see [step.md](step.md) for how they are driven and the rejected alternatives.

## Tried and rejected

## History

- 2026-10-21, ticket hamsterball-v1/01: ported from `prototypes/honest_drive.py` (branch `prototype/honest-physics`) with the issue 12 and 13 verdicts fixed and the drop-lane course dropped.
- 2026-10-22, ticket hamsterball-v1/05: `level` became top-level MJCF sections instead of worldbody content, so a level can carry hfield assets.
- 2026-10-22, ticket hamsterball-v1/06: the three view mocap bodies follow the tier's fragment.
- 2026-10-22, ticket hamsterball-v1/06: `znear` 0.001 of the extent, after the first-person view lost the near ground on `locked_goal` (see [views.md](views.md)).
- 2026-10-22, code review of hamsterball-v1: the offscreen buffer was a fixed 1280×720, so a `view_size` wider or taller raised in `mujoco.Renderer`; spec §7 asks for at least `view_size`. `build` now takes `view_size` and sizes the buffer to it, never below the game's 1280×720.
- 2026-10-22, ticket survive-movers/03: added `bodies` to `build`. A level without bodies passes `""` and compiles the same model (one more newline in the XML).
- 2026-10-22, ticket survive-movers/06: added `slots` to `build` for the collapsing tiles' trigger times.
