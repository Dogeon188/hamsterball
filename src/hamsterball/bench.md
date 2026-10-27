# bench.py

## Purpose

`hamsterball-bench`, spec §14's throughput check: env steps/s for each configuration, and the one acceptance gate, checked by hand on an M2 before a release, not in CI.

## Interface

- `hamsterball-bench [--backend cpu] [--secs 5] [--reps 3] [--copies N]`: prints a table of env steps/s (the median of `--reps` runs of `--secs` timed seconds) for `state` and `pixels` configs, on `flat` and `hfield` levels, at 1 and N copies, then the gate line. Exits 1 if the gate fails. N defaults to the performance cores (`sysctl hw.perflevel0.physicalcpu` on macOS, else every core).
- `hamsterball-bench --backend warp [--device cuda:0] [--num-envs N]... [--secs 5] [--reps 3]`: the GPU-batched env (needs the `gpu` extra). Same table, with `envs` (worlds of one `WarpVectorEnv`; `--num-envs` repeats, default 16, 256, 4096) in place of copies, a header naming the GPU, and no gate: it exits 0 and prints `no gate`.
- `hfield_level()`: the bench's hfield `Level`, built in memory.
- `warp_rate(kwargs, num_envs, secs, device)`: env steps/s of one `WarpVectorEnv`.
- `rate(kwargs, copies, secs)`: env steps/s summed over the copies for `HamsterballEnv(**kwargs)`.

## How it works

- Levels: `flat` is the bundled `first_goal` (a box platform). `hfield` is issue 22's terrain: 256×256 samples of rolling sines over 120 m square, 0.6 m high, with the goal 50 m out. The bundled `hill` is mostly box, so it would understate the hfield cost.
- Configs: `state` is the default env (`state` and `goal` only); `pixels` adds the three views at 84×84.
- One copy is a plain `HamsterballEnv` stepped in a loop that resets on episode end. N copies is `make_vec` (`AsyncVectorEnv`, spawn on macOS), which resets itself. Actions are uniform random from the action space, seeded 0. The first 20 env steps are untimed (compilation of the first renderer).
- The gate reads the `state`/`hfield`/1-copy median against `GATE` = 1,500.
- The warp backend steps one `WarpVectorEnv` on the same two levels (`first_goal`, the hfield level) and configs, with the bank's one level, frame skip 10 and autoreset on the device. Random actions are 32 batches drawn on the device up front and cycled, so the clock holds the env, not the sampler. The first 20 steps (graph capture, kernel compilation, the first renderer) are untimed; the device is synchronised every 8 steps and at both clock readings, so queued work counts.

## Invariants and checks

- `uv run pytest tests/test_bench.py`: a short run prints all eight rows and the gate line, and the exit code follows PASS/FAIL. It checks the format, not the speed.
- Gate (by hand, on an M2): `uv run hamsterball-bench`.
- `uv run --extra gpu pytest tests/test_bench.py -k warp`: the warp table prints every row on the Warp CPU device with no PASS or FAIL; format only, no speed.

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

- **GPU numbers** (2026-10-27, ticket gpu-backend/11; `hamsterball-bench --backend warp --num-envs 16 --num-envs 256 --num-envs 1024 --num-envs 4096`, median of 3 x 5 s, an NVIDIA GeForce RTX 2080 Ti (11 GiB, sm_75) on a GPU cluster, Warp 1.18.0, CUDA 13.4, the step as one captured graph; env steps/s, each env step 10 physics steps, uniform random actions). `pixels` is the three views at 84x84:

| config | level | 16 envs | 256 envs | 1024 envs | 4096 envs |
|---|---|---|---|---|---|
| state | flat | 4,008 | 36,346 | 111,174 | 213,816 |
| state | hfield | 2,877 | 27,529 | 65,894 | 102,096 |
| pixels | flat | 2,753 | 27,790 | 64,065 | 93,014 |
| pixels | hfield | 2,349 | 18,416 | 31,167 | 36,509 |

  Guidance, not a gate. State-only flat ground reaches 214,000 env steps/s at 4,096 worlds; at 16 worlds it is bound by kernel launches (4,000/s, about 4 ms per env step). The hfield rows run at 59% (1,024 worlds) and 48% (4,096) of the flat rate. `pixels` runs at 58% of the state rate on flat ground and 47% on hfield at 1,024 worlds, 43% and 36% at 4,096. Against the M2 table above (4 copies, state, hfield: 4,405), 1,024 GPU worlds are about 15 times faster. Other tasks, tiers and banks were not measured; one bank level means the padded model is as small as it gets.

## Decisions

- 2026-10-22, issue 23 (resolved by grilling 2026-10-20): one gate, ≥1,500 state-only env steps/s in one process on an hfield level on an M2, checked by hand; every other number is guidance.
- 2026-10-22, ticket hamsterball-v1/14: `bench` is one module, not a subpackage (spec §2 lists it among the subpackages), until it needs a second file, as with `env`.
- 2026-10-22, ticket hamsterball-v1/14: the CLI uses Click, a new core dependency (`click>=8.1`) that spec §2's dependency table doesn't list, per the user's rule that new Python CLIs use Click.
- 2026-10-27, ticket gpu-backend/11: `--backend warp` reuses the table's configs and levels and gates nothing (spec: "a throughput gate" is out of scope); `--backend` is `cpu` or `warp` as `make_vec`'s is, and `--device` defaults to `cuda:0` as `WarpVectorEnv`'s does.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/14: added. Agent finding while running the gate: before `Level` cached its id, a reset on the 256×256 hfield level cost about 0.63 s of YAML serialisation (see [core/level.md](core/level.md)), which cut a straight-run probe that fell off the edge every 472 steps to about 870 env steps/s. With the cache the gate run above passed. A cProfile of steady stepping on that level put about 41% of an env step in `mj_step`, 27% in the abstract tier's `drive`, and the rest in the step's other Python (`mj_kinematics`, camera yaw, drag, containment, marker test, observation).
- 2026-10-27, ticket gpu-backend/11: added `--backend warp`, `--num-envs` and `--device`, with the RTX 2080 Ti numbers above.
