# shell_contact.py

Throwaway prototype that answers `.scratch/hamsterball/issues/12-shell-wall-contact.md`: which outer collider, inner tile count, and contact softness the shell should use. It is not production code. Model C and its constants are copied from `hamster_feel.py` at 2307efc, the feel chosen in issue 06. Only the shell's contacts vary.

## Interface

```
uv run prototypes/shell_contact.py           # interactive window
uv run prototypes/shell_contact.py --sweep   # headless table over every variant, about 25 s
```

Keys: `O` outer collider, `N` tile count, `K` softness, `M` margin (each reloads the shell), `WASD` run (camera-relative), `Shift` full effort, `Q`/`E` orbit the camera, `R` reset, `Esc` quit. A gamepad's left stick sets direction and effort.

## How it works

A variant is a tuple `(outer, n_tiles, soft, margin)` drawn from `OUTERS`, `TILE_COUNTS`, `SOFTS`, and `MARGINS`. The first value of each axis is the 2307efc baseline.

- **Outer** `tiles`: the tiles are also the outer collider. `sphere`: a massless sphere of radius R meets the world on contype bit 1, and the tiles move to bit 2 so they touch only the hamster. `sphere+roll`: the same sphere with condim 6 and friction `1 TORSION ROLL`.
- **Softness** sets `solref` on the tiles and the hamster geom, so their pair uses it. The ground pair keeps MuJoCo's defaults.
- **Margin** sets the hamster geom's margin.

`measure(v)` runs three phases and returns the sweep's columns:

1. Full effort along +x on flat ground for 6 s. Over the last 4 s it records rattle (the share of steps with the wall gap above margin + 3 mm), the std of the shell centre's height, and the worst penetration. It also times `mj_step` alone.
2. Release, then the slowdown between 1 s and 3 s after release. The first second is skipped because the hamster swings back to the bottom and pushes the shell on.
3. A fresh model under gravity tilted 20°, full effort uphill for 3 s, and the distance climbed.

## Known quirks

- The non-colinear foot-force pair from issue 06 still feeds energy at zero input. On the frictionless `sphere` the shell speeds up from 4.2 to 4.7 m/s over 40 s with no input. Rolling friction hides this; it doesn't remove it.
- Stiff contact (`solref 0.004 1`) rattles on 60–90% of steps and loses a third of the run speed.
- 320 tiles cost about 25% of CPU step rate against 180.

## Decisions

- [ADR-0001](../docs/adr/0001-mujoco-core-game-wraps-env.md), MuJoCo core: every collider is a MuJoCo primitive, so MJX and Warp can load the model. MJX filters pairs by contype/conaffinity at compile time, so with the outer sphere the tiles never pair with a level hfield.
- Issue 12 verdict, from play-testing on 2026-10-20: **`sphere+roll`, 320 tiles, `soft`, 2 mm margin**, with torsional friction 0.05.
  - The coast drag comes from the outer collider's rolling friction (0.04 m), not from facets or a speed-dependent force.
  - The torsional friction (0.05 m) stops leftover yaw after a turn.
  - The hamster-tile pair uses solref `0.04 2` and a 2 mm hamster margin, so the airborne cut's touch test is margin + 3 mm.

## Tried and rejected

All numbers are from `--sweep` at 7c9ec0b, 2 mm margin, unless noted.

- **Tiles as the outer collider (baseline)**: 33% rattle, 1.4 mm shell jitter, and 8.6 m climbed in 3 s against 9.8 for `sphere+roll`. Every tile would also pair with every hfield in MJX.
- **Smooth sphere without rolling friction**: 5% rattle and 0.1 mm jitter at 320 soft, but the coast lost about 0 m/s², so the shell rolled on forever.
- **80 tiles**: 48–89% rattle and up to 10 mm penetration in every combination.
- **Stiff contact, solref `0.004 1`**: worst rattle and run speed in every combination.
- **Torsional friction 0.005, the plane's default**: after a turn the stopped shell kept spinning in yaw, 0.44 rad/s 8 s after stopping and 0.08 rad/s after 28 s. Raised to 0.05 in 143836a.

## History

- 2026-10-20, design questions (AskUserQuestion), issue 12: the user chose a new script with model C only, a sweep table plus play-testing, MJX cost argued from contype filtering rather than timed, and outer candidates tiles, sphere, and sphere with rolling friction. They approved the full 54-run grid.
- 2026-10-20, commit 7c9ec0b: built the sweep and play mode. Agent finding: the first coast window counted the hamster's swing-back as acceleration, and a fixed 3 mm rattle threshold counted the margin's standoff as rattle. Both fixed before the commit.
- 2026-10-20, play-test verdict, issue 12: the user found `sphere+roll` / 320 / soft / 2 mm felt best and chose the outer collider as the source of coast drag. They reported the stopped shell still spinning on its own.
- 2026-10-20, agent finding: the spin is yaw left by turns, plus energy from the non-colinear foot-force pair. Torsional friction 0.05 stops it within 10 s with run speed, coast and climb nearly unchanged. The user chose that fix (commit 143836a) and confirmed in play that the spin is gone.
