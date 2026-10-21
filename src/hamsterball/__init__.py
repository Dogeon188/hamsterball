"""Hamsterball: a hamster running inside a rolling shell, as a Gymnasium env and a game on one MuJoCo core."""

import os
import sys

# Headless GL for env views, before anything imports mujoco: EGL on Linux (set MUJOCO_GL=osmesa for the CPU
# fallback); macOS leaves it unset, which selects CGL. glfw is never used for env rendering.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from gymnasium.envs.registration import register  # noqa: E402

register(id="Hamsterball-v0", entry_point="hamsterball.env:HamsterballEnv")
