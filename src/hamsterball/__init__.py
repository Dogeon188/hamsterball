"""Hamsterball: a hamster running inside a rolling shell, as a Gymnasium env and a game on one MuJoCo core."""

import os
import sys

# Headless GL for env views, before anything imports mujoco: EGL on Linux (set MUJOCO_GL=osmesa for the CPU
# fallback); macOS leaves it unset, which selects CGL. glfw is never used for env rendering.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from functools import partial  # noqa: E402

from gymnasium.envs.registration import register  # noqa: E402
from gymnasium.error import DeprecatedEnv  # noqa: E402

ENV_VERSION = 0  # N of spec §2: bumps with every change that can alter a trajectory, as a major release
PRESETS = {
    "Hamsterball": None,
    "HamsterballReachGoal": "reach_goal",
    "HamsterballWaypointRace": "waypoint_race",
    "HamsterballCollectItems": "collect_items",
    "HamsterballUnlockGoal": "unlock_goal",
    "HamsterballSurvive": "survive",
}


def _worker_env(i, devices, kwargs):
    """make_vec's env fn for worker i (spec §11): on a multi-GPU machine, EGL device i mod devices unless
    MUJOCO_EGL_DEVICE_ID is already set, chosen before mujoco loads in this process. It lives here, beside no mujoco
    import, and takes the env kwargs pickled, since unpickling them (a generator, a Level) imports mujoco."""
    if devices > 1:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(i % devices))
    import pickle

    from .env import HamsterballEnv

    return HamsterballEnv(**pickle.loads(kwargs))


def _retired(env_id, n, **_):
    raise DeprecatedEnv(
        f"{env_id} is retired: this hamsterball registers v{ENV_VERSION}. "
        f"For v{n}, install the release that registers it: pip install 'hamsterball=={n}.*'"
    )


def register_ids(n=ENV_VERSION):
    """Register the six ids at version n, and every older version as an id that raises naming its release."""
    for name, task in PRESETS.items():
        register(
            id=f"{name}-v{n}",
            entry_point="hamsterball.env:HamsterballEnv",
            vector_entry_point="hamsterball.env:make_vec",
            kwargs={"tasks": [task]} if task else {},
        )
        for old in range(n):
            register(id=f"{name}-v{old}", entry_point=partial(_retired, f"{name}-v{old}", old))


register_ids()
