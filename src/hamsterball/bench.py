"""hamsterball-bench (spec §14): env steps/s for state-only and pixel envs, 1 and N copies, flat and hfield levels; with
--backend warp, the GPU-batched env at a few num_envs."""

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


def warp_rate(kwargs, num_envs, secs, device):
    """Env steps/s of the Warp vector env's `num_envs` worlds under uniform random actions (32 drawn up front, cycled, so
    the clock holds the env). The device is synchronised before each reading of the clock."""
    import torch

    from .gpu.vector_env import WarpVectorEnv

    env = WarpVectorEnv(num_envs, device=device, **kwargs)
    try:
        sync = (lambda: torch.cuda.synchronize()) if env.core.device.is_cuda else (lambda: None)
        actions = torch.rand(32, num_envs, 2, device=env.torch_device) * 2 - 1
        env.reset(seed=0)
        for i in range(WARMUP):  # graph capture, kernel compilation, the first renderer
            env.step(actions[i % 32])
        sync()
        n, t0 = 0, time.perf_counter()
        while (t := time.perf_counter() - t0) < secs:
            env.step(actions[n % 32])
            n += 1
            if n % 8 == 0:
                sync()  # keep the queue short without a sync per step
        sync()
        return n * num_envs / (time.perf_counter() - t0)
    finally:
        env.close()


def main_warp(secs, reps, num_envs, device):
    """The GPU-batched table: state and pixels configs on the flat and hfield levels, per num_envs. Gates nothing."""
    import torch
    import warp as wp

    gpu = torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "Warp CPU device"
    click.echo(f"{platform.platform()}, Python {platform.python_version()}, mujoco {mujoco.__version__}, warp {wp.config.version}, {gpu}")
    click.echo(f"median of {reps} x {secs:g} s, random actions, one WarpVectorEnv (frame_skip 10, autoreset on the device)\n")
    click.echo(f"{'config':<8}{'level':<8}{'envs':>6}  {'env steps/s':>11}")
    levels = {"flat": "first_goal", "hfield": hfield_level()}
    configs = {"state": {}, "pixels": {"views": VIEWS}}
    for config, extra in configs.items():
        for name, level in levels.items():
            for n in num_envs:
                kwargs = {"tasks": ["reach_goal"], "levels": [level], **extra}
                r = statistics.median(warp_rate(kwargs, n, secs, device) for _ in range(reps))
                click.echo(f"{config:<8}{name:<8}{n:>6}  {r:>11,.0f}")
    click.echo("\nno gate: the GPU numbers are guidance (spec gpu-backend)")


@click.command()
@click.option("--secs", default=5.0, show_default=True, help="Timed seconds per run.")
@click.option("--reps", default=3, show_default=True, help="Runs per configuration; the median is printed.")
@click.option("--copies", type=int, default=None, help="N, the vector env's copies (cpu).  [default: performance cores]")
@click.option("--backend", type=click.Choice(["cpu", "warp"]), default="cpu", show_default=True, help="warp: the GPU-batched env.")
@click.option("--num-envs", type=int, multiple=True, help="warp: worlds per run; repeat for several.  [default: 16, 256, 4096]")
@click.option("--device", default="cuda:0", show_default=True, help="warp: the Warp device, e.g. cpu or cuda:0.")
def main(secs, reps, copies, backend, num_envs, device):
    """Print env steps/s for state-only and pixel envs, 1 and N copies, flat and hfield levels, and the M2 gate.

    Exits 1 if one state-only process on the hfield level runs under 1,500 env steps/s (spec §14). With --backend warp it
    prints the GPU-batched env's steps/s at a few num_envs and gates nothing."""
    if backend == "warp":
        return main_warp(secs, reps, num_envs or (16, 256, 4096), device)
    copies = copies or performance_cores()
    click.echo(f"{platform.platform()}, Python {platform.python_version()}, mujoco {mujoco.__version__}")
    click.echo(f"median of {reps} x {secs:g} s, random actions; N copies run in make_vec's AsyncVectorEnv\n")
    click.echo(f"{'config':<8}{'level':<8}{'copies':>6}  {'env steps/s':>11}")
    levels = {"flat": "first_goal", "hfield": hfield_level()}
    configs = {"state": {}, "pixels": {"views": VIEWS}}  # pixels: the three views at 84x84, plus the vectors
    gate = None
    for config, extra in configs.items():
        for name, level in levels.items():
            for n in sorted({1, copies}):
                kwargs = {"tasks": ["reach_goal"], "levels": [level], **extra}
                r = statistics.median(rate(kwargs, n, secs) for _ in range(reps))
                click.echo(f"{config:<8}{name:<8}{n:>6}  {r:>11,.0f}")
                if (config, name, n) == ("state", "hfield", 1):
                    gate = r
    ok = gate >= GATE
    click.echo(f"\ngate: one state-only process on hfield >= {GATE:,}: {gate:,.0f} env steps/s, {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
