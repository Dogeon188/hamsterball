"""Hamsterball: a hamster running inside a rolling shell, as a Gymnasium env and a game on one MuJoCo core."""

from functools import partial

from gymnasium.envs.registration import register
from gymnasium.error import DeprecatedEnv

ENV_VERSION = 0  # N of spec §2: bumps with every change that can alter a trajectory, as a major release
PRESETS = {
    "Hamsterball": None,
    "HamsterballReachGoal": "reach_goal",
    "HamsterballWaypointRace": "waypoint_race",
    "HamsterballCollectItems": "collect_items",
    "HamsterballUnlockGoal": "unlock_goal",
    "HamsterballSurvive": "survive",
}


def _retired(env_id, n, **_):
    raise DeprecatedEnv(
        f"{env_id} is retired: this hamsterball registers v{ENV_VERSION}. "
        f"For v{n}, install the release that registers it: pip install 'hamsterball=={n}.*'"
    )


def register_ids(n=ENV_VERSION):
    """Register the six ids at version n, and every older version as an id that raises naming its release."""
    for name, task in PRESETS.items():
        register(id=f"{name}-v{n}", entry_point="hamsterball.env:HamsterballEnv", kwargs={"tasks": [task]} if task else {})
        for old in range(n):
            register(id=f"{name}-v{old}", entry_point=partial(_retired, f"{name}-v{old}", old))


register_ids()
