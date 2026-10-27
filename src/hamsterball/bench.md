# bench.py

## Purpose

`hamsterball-bench`, spec §14's throughput check: env steps/s for each configuration, and the one acceptance gate, checked by hand on an M2 before a release, not in CI.

## Interface

- `hamsterball-bench [--tier abstract|articulated] [--secs 5] [--reps 3] [--copies N]`: prints a table of env steps/s (the median of `--reps` runs of `--secs` timed seconds) for `state` and `pixels` configs, on `flat` and `hfield` levels, at 1 and N copies, then the gate line. Exits 1 if the gate fails. With `--tier articulated` it prints the same table and no gate line (exit 0): the gate is the abstract tier's. N defaults to the performance cores (`sysctl hw.perflevel0.physicalcpu` on macOS, else every core).
- `hfield_level()`: the bench's hfield `Level`, built in memory.
- `rate(kwargs, copies, secs)`: env steps/s summed over the copies for `HamsterballEnv(**kwargs)`.

## How it works

- Levels: `flat` is the bundled `first_goal` (a box platform). `hfield` is issue 22's terrain: 256×256 samples of rolling sines over 120 m square, 0.6 m high, with the goal 50 m out. The bundled `hill` is mostly box, so it would understate the hfield cost.
- Configs: `state` is the default env (`state` and `goal` only); `pixels` adds the three views at 84×84.
- One copy is a plain `HamsterballEnv` stepped in a loop that resets on episode end. N copies is `make_vec` (`AsyncVectorEnv`, spawn on macOS), which resets itself. Actions are uniform random from the action space, seeded 0. The first 20 env steps are untimed (compilation of the first renderer).
- `--tier` goes into every config's env kwargs. The articulated tier steps at about 0.7 to 0.9 of the abstract tier's rate on state configs (a collision pass per observation, 8 joints, more bodies) and 0.5 to 0.9 on pixels (laptop, 4 copies, 2 s runs, noisy).
- The gate reads the `state`/`hfield`/1-copy median against `GATE` = 1,500.

## Invariants and checks

- `uv run pytest tests/test_bench.py`: a short run prints all eight rows and the gate line, and the exit code follows PASS/FAIL. It checks the format, not the speed.
- Gate (by hand, on an M2): `uv run hamsterball-bench`.

## Known quirks

- Gate results on the M2 (Apple M2, 4 performance + 4 efficiency cores):

  | Date | Commit | Power | Python | state hfield 1 copy | Result |
  |---|---|---|---|---|---|
  | 2026-10-22 | ticket 14 branch | battery | 3.12.9 | 1,784 | PASS |

- The full table of that run (defaults: median of 3 × 5 s, N = 4), with other agents' test runs sharing the machine:

  | config | level | 1 copy | 4 copies |
  |---|---|---|---|
  | state | flat | 2,401 | 3,451 |
  | state | hfield | 1,784 | 4,405 |
  | pixels | flat | 672 | 1,391 |
  | pixels | hfield | 134 | 281 |

- The 4-copy rows are under issue 22's guidance (about 6,000 state-only): they ran on battery while other processes loaded the machine, and lockstep stepping makes copies on a busy core hold back the batch. Re-measure on AC power with the machine idle before a release.
- `pixels` on the hfield level runs at a fifth of the flat rate: the renderer draws the 256×256 hfield mesh in each of the three views, which the physics doesn't pay for. Spec §14's ~300 env steps/s pixel guidance came from flat ground.

## Decisions

- 2026-10-22, issue 23 (resolved by grilling 2026-10-20): one gate, ≥1,500 state-only env steps/s in one process on an hfield level on an M2, checked by hand; every other number is guidance.
- 2026-10-22, ticket hamsterball-v1/14: `bench` is one module, not a subpackage (spec §2 lists it among the subpackages), until it needs a second file, as with `env`.
- 2026-10-22, ticket hamsterball-v1/14: the CLI uses Click, a new core dependency (`click>=8.1`) that spec §2's dependency table doesn't list, per the user's rule that new Python CLIs use Click.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/14: added. Agent finding while running the gate: before `Level` cached its id, a reset on the 256×256 hfield level cost about 0.63 s of YAML serialisation (see [core/level.md](core/level.md)), which cut a straight-run probe that fell off the edge every 472 steps to about 870 env steps/s. With the cache the gate run above passed. A cProfile of steady stepping on that level put about 41% of an env step in `mj_step`, 27% in the abstract tier's `drive`, and the rest in the step's other Python (`mj_kinematics`, camera yaw, drag, containment, marker test, observation).
- 2026-10-27, ticket articulated-tier/03: `--tier` option; the articulated tier reports throughput with no gate. Test: `tests/test_bench.py` (rows only, no gate line).
