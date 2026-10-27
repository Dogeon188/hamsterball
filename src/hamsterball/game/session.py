"""The game's episodes and their recording (spec §13): every finished episode is appended to the player's Minari
dataset as it ends, and an abandoned run is discarded. Runs without a display; recording needs the `data` extra."""

import gymnasium as gym
import numpy as np

from .. import ENV_VERSION
from ..core.generator import ProceduralGenerator
from ..lowlevel import DEFAULT, get
from .menu import GENERATED, dataset_id, name_ok


class Session:
    """Episodes of one task on one level (a level file, or "generated" at `difficulty`), recorded to
    `hamsterball/human-<player>-v<N>` (`-articulated-v<N>` with `tier="articulated"`) unless `record` is False.
    In the articulated tier `step` still takes the game's 4-D action; the scripted gait turns its run vector into torques.

    `start()` begins the next episode (a fresh seed on a generated level; `seed`, if given, sets only the first's),
    `start(retry=True)` replays the last level and seed. `step(action, overruns)` steps, and appends the episode with its
    overrun count when the step ends it. Starting again, or `close()`, before an episode ends abandons it.
    """

    def __init__(self, task, level, difficulty=0.5, seed=None, player=None, record=True, tier="abstract"):
        if record and not name_ok(player or ""):
            raise ValueError(f"player name {player!r} must match [a-z0-9_]+")
        self.task, self.recording, self.player = task, record, player
        self.tier = tier
        self.gait = get(DEFAULT) if tier == "articulated" else None
        self.dataset_id = dataset_id(player, tier)
        if level == GENERATED:
            pool = gen = ProceduralGenerator()
            self._first = {"level": gen(task, seed, difficulty), "task": task} if seed is not None else None
        else:  # the menu remembers a generated level's difficulty, which a file pool refuses
            pool, self._first, difficulty = [str(level)], None, None
        env = gym.make(f"Hamsterball-v{ENV_VERSION}", tasks=[task], levels=pool, difficulty=difficulty,
                       camera_action=True, truncate=False, tier=tier)  # no time-up for a human (survive keeps its limit)
        if record:
            from ..data import Collector  # the data extra; only recording needs it

            # every episode's options hold its level, so the dataset's env_spec names none: a level path works only
            # on this machine, and one dataset holds every level and task the player plays
            for k in ("levels", "difficulty"):
                env.unwrapped.spec.kwargs.pop(k)
            env = Collector(env)
        self.env, self.running, self._retry, self._obs = env, False, None, None

    def start(self, retry=False):
        """Abandon any run in progress, then reset for the next episode, or the last one's level and seed."""
        self.abandon()
        options = self._retry if retry else self._first or {"task": self.task}
        self._first = None
        obs, info = self.env.reset(options=options)
        self._obs = obs
        if self.gait:
            self.gait.reset()
        self._retry = {"level": info["reset_options"]["level"], "task": self.task}
        self.running = True
        return obs, info

    def step(self, action, overruns=0):
        """One env step; an episode this step ends is appended to the dataset with `overruns` in its metadata."""
        if self.gait:  # the run vector is the abstract action's first half; the camera dims ride after the torques
            run = np.asarray(action[:2], np.float32)
            torques = self.gait.act(self._obs, run)
            action = np.concatenate([torques, action[2:]])
            out = self.env.step(action, run_vector=run) if self.recording else self.env.step(action)
            self._obs = out[0]
        else:
            out = self.env.step(action)
        if out[2] or out[3]:
            self.running = False
            if self.recording:
                self._save(overruns)
        return out

    def _save(self, overruns):
        import minari

        try:
            ds = minari.load_dataset(self.dataset_id)
        except FileNotFoundError:
            ds = self.env.create_dataset(
                self.dataset_id, author=self.player,  # a human: `author` only, no algorithm_name (spec §12)
                description=f"Hamsterball games played by {self.player}{' in the articulated tier' * bool(self.gait)} "
                "(hamsterball-play).",
                **({"low_level_policy": self.gait.name} if self.gait else {}),
            )
        else:
            self.env.add_to_dataset(ds)
        ds.storage.update_episode_metadata([{"overruns": overruns}], [ds.storage.total_episodes - 1])

    def abandon(self):
        """Discard the run in progress, if any (Retry, Menu, Quit, or closing the window before the episode ends)."""
        if self.running and self.recording:
            self.env.discard_episode()
        self.running = False

    def close(self):
        self.abandon()
        self.env.close()
