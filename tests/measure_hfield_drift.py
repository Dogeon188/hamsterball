"""Hfield contact-cap drift (spec gpu-backend, "Measured, not asserted"; issue 20's measurement): drop the shell at many
spots over each hfield level, still, on the CPU core and on the batched core, and compare where each comes to rest.
Both GPU backends cap hfield contacts at 4; the CPU core uses as many as it finds. Prints one row per level.

    uv run --extra gpu python tests/measure_hfield_drift.py [--device cuda:0] [--spots 200] [--secs 3]
"""

import click
import numpy as np

from hamsterball.bench import hfield_level
from hamsterball.core import Core
from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from hamsterball.core.step import FRAME_SKIP
from hamsterball.env import LEVELS_DIR

DROP = 0.6  # m above the terrain's highest point


def levels():
    """The bundled hfield level, the bench's rolling terrain, and generated reach_goal terrain at three difficulties."""
    gen = ProceduralGenerator()
    return {"hill (bundled)": Level.load(LEVELS_DIR / "hill.yaml"), "bench rolling": hfield_level(),
            **{f"generated d={d}": gen("reach_goal", seed, d) for seed, d in ((1, 0.3), (3, 0.6), (2, 0.9))}}


def spots(level, n, rng):
    """`n` drop points over the level's first hfield, inside 90% of its extent, DROP above its highest point."""
    g = next(g for g in level.geoms if g["type"] == "hfield")
    x, y, z = g.get("pos", [0, 0, 0])
    rx, ry, top = g["size"][:3]
    return np.stack([x + rng.uniform(-.9, .9, n) * rx, y + rng.uniform(-.9, .9, n) * ry, np.full(n, z + top + DROP)], 1)


def measure(level, n, secs, device, seed=0):
    """Rest positions (n, 3) of the shell from `n` drops on the CPU core and on the batched core, after `secs` still."""
    import warp as wp

    from hamsterball.gpu.batch_core import BatchCore

    drops = spots(level, n, np.random.default_rng(seed)).astype(np.float32)
    frames, still = round(secs * 600 / FRAME_SKIP), np.zeros(2)
    core, cpu = Core(level=level), np.zeros((n, 3))
    for i, p in enumerate(drops):
        core.reset(p)
        for _ in range(frames):
            core.step(still)
        cpu[i] = core.data.qpos[:3]
    batch = BatchCore(levels=[level], num_envs=n, device=device, tasks=["reach_goal"])
    with wp.ScopedDevice(batch.device):
        batch.reset_masked(wp.ones(n, dtype=wp.bool), wp.array(drops, dtype=wp.vec3), wp.zeros(n))
    for _ in range(frames):
        batch.step(np.zeros((n, 2), np.float32))
    return cpu, batch.data.qpos.numpy()[:, :3]


def summary(cpu, gpu):
    """Rest height differences |dz| in mm (median, 90th percentile, max), the spots more than 1 cm apart in height, and
    in xy (the ball rolled to a different place); and among the spots where both came to rest within 1 cm of each other
    in xy, how many more than 1 cm apart in height (the cap's own effect, free of rolling off differently)."""
    dz, dxy = np.abs(gpu[:, 2] - cpu[:, 2]), np.linalg.norm(gpu[:, :2] - cpu[:, :2], axis=1)
    return {"median_mm": np.median(dz) * 1e3, "p90_mm": np.percentile(dz, 90) * 1e3, "max_mm": dz.max() * 1e3,
            "over_1cm": int((dz > 0.01).sum()), "xy_over_1cm": int((dxy > 0.01).sum()),
            "same_spot": int((dxy <= 0.01).sum()), "same_spot_over_1cm": int(((dxy <= 0.01) & (dz > 0.01)).sum())}


@click.command()
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--spots", "n", default=200, show_default=True)
@click.option("--secs", default=3.0, show_default=True, help="Seconds the shell is left still after the drop.")
def main(device, n, secs):
    print(f"{'level':<18}{'median mm':>10}{'p90 mm':>9}{'max mm':>9}{'dz>1cm':>8}{'dxy>1cm':>9}{'same-spot dz>1cm':>18}   of {n} drops, {secs:g} s")
    for name, level in levels().items():
        s = summary(*measure(level, n, secs, device))
        print(f"{name:<18}{s['median_mm']:>10.2f}{s['p90_mm']:>9.2f}{s['max_mm']:>9.1f}{s['over_1cm']:>8}{s['xy_over_1cm']:>9}{s['same_spot_over_1cm']:>10} of {s['same_spot']}")


if __name__ == "__main__":
    main()
