"""The game's episodes and their recording (spec §13): every finished episode is appended to the player's Minari
dataset as it ends, and an abandoned run is discarded. Runs without a display; recording needs the `data` extra."""

import gymnasium as gym

from .. import ENV_VERSION
from ..core.generator import ProceduralGenerator
from .menu import GENERATED, name_ok


class Session:
    """Episodes of one task on one level (a level file, or "generated" at `difficulty`), recorded to
    `hamsterball/human-<player>-v<N>` unless `record` is False.

    `start()` begins the next episode (a fresh seed on a generated level; `seed`, if given, sets only the first's),
    `start(retry=True)` replays the last level and seed. `step(action, overruns)` steps, and appends the episode with its
    overrun count when the step ends it. Starting again, or `close()`, before an episode ends abandons it.
    """

    def __init__(self, task, level, difficulty=0.5, seed=None, player=None, record=True):
        if record and not name_ok(player or ""):
            raise ValueError(f"player name {player!r} must match [a-z0-9_]+")
        self.task, self.recording, self.player = task, record, player
        self.dataset_id = f"hamsterball/human-{player}-v{ENV_VERSION}"
        if level == GENERATED:
            pool = gen = ProceduralGenerator()
            self._first = {"level": gen(task, seed, difficulty), "task": task} if seed is not None else None
        else:
            pool, self._first = [str(level)], None
        env = gym.make(f"Hamsterball-v{ENV_VERSION}", tasks=[task], levels=pool, difficulty=difficulty,
                       camera_action=True)
        if record:
            from ..data import Collector  # the data extra; only recording needs it

            # every episode's options hold its level, so the dataset's env_spec names none: a level path works only
            # on this machine, and one dataset holds every level and task the player plays
            for k in ("levels", "difficulty"):
                env.unwrapped.spec.kwargs.pop(k)
            env = Collector(env)
        self.env, self.running, self._retry = env, False, None

    def start(self, retry=False):
        """Abandon any run in progress, then reset for the next episode, or the last one's level and seed."""
        self.abandon()
        options = self._retry if retry else self._first or {"task": self.task}
        self._first = None
        obs, info = self.env.reset(options=options)
        self._retry = {"level": info["reset_options"]["level"], "task": self.task}
        self.running = True
        return obs, info

    def step(self, action, overruns=0):
        """One env step; an episode this step ends is appended to the dataset with `overruns` in its metadata."""
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
                self.dataset_id, algorithm_name="human", author=self.player,
                description=f"Hamsterball games played by {self.player} (hamsterball-play).",
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
