# model.py

## Purpose

Builds the hamsterball model of spec §3: the shell, a control tier's hamster fragment, and a level's static geoms, compiled into one `MjModel` at the frozen 1/600 s physics step.

## Interface

- `build(tier, level=FLAT) -> MjModel`: `level` is worldbody MJCF (geoms only, no bodies). `FLAT` is an infinite ground plane.
- Constants the rest of the core imports: `DT`, `R`, `T`, `M_SHELL`, `M_HAM`, `SOLREF`, `TILE_NORMALS` (320×3, shell frame), and the fixed ids `SHELL`, `HAM`, `SHELL_QPOS`, `HAM_QPOS`, `SHELL_DOF`, `HAM_DOF`.

## How it works

- The shell is a free body whose origin is the shell centre: 320 box tiles on a Fibonacci sphere carry its 0.8 kg (contype and conaffinity 2, group 3), and a massless outer sphere (contype 1, `condim 6`, friction `1 0.05 0.04`, group 1) touches the level.
- The tier's `fragment` goes in the worldbody after the shell, then the view mocaps of [views.md](views.md), and `nuserdata` is 1 (the core's camera yaw) plus the tier's slots.
- Floats go into the MJCF with `repr` of plain Python floats (NumPy 2's `repr` of `np.float64` isn't valid MJCF), so tile poses match the spec's formulas to full precision.
- Integrator `implicitfast`, as in the prototype that set the feel.

## Invariants and checks

- The shell is body 1 and the hamster body 2, both free, so the shell's qpos starts at 0 and the hamster root's at 7 (dofs 0 and 6). `build()` asserts this and that the hamster subtree weighs 2.4 kg. Later bodies, such as the view mocaps, must come after the hamster fragment.
- `uv run pytest tests/test_model.py` checks the model against spec §3. A compiled `MjModel` keeps no per-geom mass, so the test checks the shell body's 0.8 kg, which also shows the outer sphere is massless.

## Known quirks

- The prototype wrote tile poses with 4 decimals; this build writes them exactly, a sub-0.1 mm difference.

## Decisions

- 2026-10-21, [ADR-0001](../../../docs/adr/0001-mujoco-core-game-wraps-env.md): primitive geoms only (box tiles, a sphere), so MJX and MuJoCo Warp load the same model.
- 2026-10-21, [ADR-0002](../../../docs/adr/0002-exact-mujoco-pin-bumps-env-version.md): `mujoco==3.16.0`, pinned exactly in `pyproject.toml`.
- 2026-10-21, ticket hamsterball-v1/01: hard-coded body, qpos, and dof ids for the shell and the hamster root, in place of name lookups, so the per-step code reads like a kernel.

## Tried and rejected

## History

- 2026-10-21, ticket hamsterball-v1/01: ported from `prototypes/honest_drive.py` (branch `prototype/honest-physics`) with the issue 12 and 13 verdicts fixed and the drop-lane course dropped.
- 2026-10-22, ticket hamsterball-v1/06: the three view mocap bodies follow the tier's fragment.
