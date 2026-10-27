"""hamsterball-bench (spec §14): env steps/s for state-only and pixel envs, 1 and N copies, flat and hfield levels."""

import base64
import os
import platform
import statistics
import subprocess
import time

import click
import mujoco
import numpy as np

from .core.level import Level
from .core.views import VIEWS
from .env import HamsterballEnv, make_vec

GATE = 1500  # spec §14: one state-only process on an hfield level, env steps/s on an M2
WARMUP = 20  # env steps before timing: the first render makes the renderer


def hfield_level():
    """A 256x256 rolling-terrain level, 120 m square and 0.6 m high: the hfield of issue 22's measurement."""
    x = np.linspace(-60, 60, 256)
    X, Y = np.meshgrid(x, x)
    h = np.clip(0.5 + 0.25 * np.sin(X / 4) * np.cos(Y / 5) + 0.25 * np.sin((X + Y) / 9), 0, 1)
    samples = base64.b64encode((h * 65535).round().astype("<u2").tobytes()).decode()
    return Level.from_yaml(f"""
version: 1
kill_plane_z: -5
geoms:
  - {{type: hfield, size: [60, 60, 0.6], nrow: 256, ncol: 256, samples: {samples}}}
tasks:
  reach_goal:
    spawn: {{pos: [0, 0, 1.2]}}
    goal: {{type: sphere, pos: [0, 50, 1], radius: 1}}
    time_limit_s: 30
""")


def rate(kwargs, copies, secs):
    """Env steps/s summed over the copies, under uniform random actions: one plain env, or make_vec's async env."""
    env = HamsterballEnv(**kwargs) if copies == 1 else make_vec(copies, **kwargs)
    try:
        env.action_space.seed(0)
        env.reset(seed=0)
        n, t0 = -WARMUP, None
        while True:
            if n == 0:
                t0 = time.perf_counter()
            elif n > 0 and (t := time.perf_counter() - t0) >= secs:
                return n * copies / t
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if copies == 1 and (terminated or truncated):  # make_vec resets its copies itself
                env.reset()
            n += 1
    finally:
        env.close()


def performance_cores():
    """The machine's performance cores (spec §10's recommended num_envs); all cores where the OS doesn't say."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True)
        return int(out.stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return os.cpu_count()


@click.command()
@click.option("--secs", default=5.0, show_default=True, help="Timed seconds per run.")
@click.option("--reps", default=3, show_default=True, help="Runs per configuration; the median is printed.")
@click.option("--copies", type=int, default=None, help="N, the vector env's copies.  [default: performance cores]")
@click.option("--tier", type=click.Choice(["abstract", "articulated"]), default="abstract", show_default=True,
              help="Control tier; the articulated tier prints its rates with no gate.")
def main(secs, reps, copies, tier):
    """Print env steps/s for state-only and pixel envs, 1 and N copies, flat and hfield levels, and the M2 gate.

    Exits 1 if one state-only process on the hfield level runs under 1,500 env steps/s (spec §14); the gate is
    the abstract tier's, so another tier only prints its rates."""
    copies = copies or performance_cores()
    click.echo(f"{platform.platform()}, Python {platform.python_version()}, mujoco {mujoco.__version__}")
    click.echo(f"median of {reps} x {secs:g} s, random actions, {tier} tier; N copies run in make_vec's AsyncVectorEnv\n")
    click.echo(f"{'config':<8}{'level':<8}{'copies':>6}  {'env steps/s':>11}")
    levels = {"flat": "first_goal", "hfield": hfield_level()}
    configs = {"state": {}, "pixels": {"views": VIEWS}}  # pixels: the three views at 84x84, plus the vectors
    gate = None
    for config, extra in configs.items():
        for name, level in levels.items():
            for n in sorted({1, copies}):
                kwargs = {"tasks": ["reach_goal"], "levels": [level], "tier": tier, **extra}
                r = statistics.median(rate(kwargs, n, secs) for _ in range(reps))
                click.echo(f"{config:<8}{name:<8}{n:>6}  {r:>11,.0f}")
                if (config, name, n) == ("state", "hfield", 1):
                    gate = r
    if tier != "abstract":
        return
    ok = gate >= GATE
    click.echo(f"\ngate: one state-only process on hfield >= {GATE:,}: {gate:,.0f} env steps/s, {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
