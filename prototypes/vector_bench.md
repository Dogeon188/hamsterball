# vector_bench.py

Throwaway benchmark that answers `.scratch/hamsterball/issues/22-measure-cpu-vector-throughput.md`: how fast state-only env copies run on the CPU, single process and vectorized, on an M2 Mac. It is not production code. The core step is `honest_drive.control()` with `PICK` at 1/600 s, the setup "Re-check the feel at 1/600 s" validated, and drag k is frozen at 0.059.

## Interface

```
uv run prototypes/vector_bench.py check                    # tidied core == control(), bitwise, both levels
uv run prototypes/vector_bench.py raw|single|vector|all [--secs 5] [--reps 3] [--cores asis,fast]
```

`vector_bench_results.txt` holds the run the ticket's answer quotes.

## How it works

- **Levels:** `flat` is a plane. `hfield` is a 256×256 grid of rolling sines over 120 m, up to 0.6 m high. Both use `honest_drive.shell_wall()` and the hamster from `honest_drive.build()`, with no drop-lane course.
- **`BenchEnv`:** a minimal reach-goal env. Each env step runs 10 physics steps of control, `mj_step`, and the chase yaw (τ 0.55 s above 0.8 m/s). Then it runs one swept segment-vs-sphere marker test and builds the 19-D `state` and 10-D `goal` from "Observation spec". Episodes truncate at 600 env steps or near the hfield edge.
- **Cores:** `asis` calls `honest_drive.control()`. `fast` is `FastCore`, the same maths with ids, tile normals, and constants hoisted and a hand-written cross product. `check` asserts the two give bitwise-equal `qpos` after 1,200 env steps.
- **Actions:** a random run vector per copy, held for 60 env steps, at effort 0.2 to 1.0.
- **Modes:**
  - `raw` times `mj_step` alone.
  - `single` times one env and splits each env step's time into its parts.
  - `vector` times `SyncVectorEnv` at 1/4/16 copies and `AsyncVectorEnv(context="spawn")` at 1/2/4/8/16 copies.
  - Each figure is the median of 3 runs of 5 s.

## Log

- 2026-10-20: the first `fast` core only hoisted lookups and saved 10%. A profile showed `np.cross` took about 60% of `control()` (about 20 µs a call on axis handling). The hand-written cross cut `control()` from 490 to 250 µs per env step, and the results file has both runs.
