"""The Collector (spec §12): Minari's DataCollector, storing what replay needs from a hamsterball env."""

import dataclasses
import importlib.metadata
import platform

import gymnasium as gym
import minari
import mujoco
import numpy as np
from minari.data_collector.callbacks import EpisodeMetadataCallback, StepDataCallback


class _StepData(StepDataCallback):
    """Keeps only `core_state` from info: the other keys are derived, and reset's `reset_options` goes per episode."""

    def __call__(self, env, **kw):
        step = super().__call__(env, **kw)
        step["info"] = {"core_state": step["info"]["core_state"]}
        return step


# every env kwarg that shapes a space or the steps, so recover_environment() rebuilds an env with the same spaces
SPACE_KWARGS = ("tier", "camera_action", "views", "view_size", "obs_state", "shaped", "frame_skip")


def provenance():
    """What a replay must match: the hamsterball and MuJoCo versions, and the platform (OS and CPU architecture)."""
    return {
        "hamsterball_version": importlib.metadata.version("hamsterball"),
        "mujoco_version": mujoco.__version__,
        "platform": f"{platform.system()}-{platform.machine()}",
    }


class Collector(minari.DataCollector):
    """Minari's DataCollector over a `gym.make`d hamsterball env, storing per step the action, observation, and
    `info['core_state']`, and per episode the seed and `info['reset_options']`."""

    def __init__(self, env, **kw):
        u = env.unwrapped
        # fill the spec's defaults in, before Minari copies the spec; views and obs_state arrive with ticket 06
        u.spec.kwargs.update({k: getattr(u, k) for k in SPACE_KWARGS if hasattr(u, k)})
        outcomes = self._outcomes = {}  # episode id -> its last step's success and failure_reason

        class Metadata(EpisodeMetadataCallback):
            def __call__(self, episode):
                return {
                    **super().__call__(episode),
                    **outcomes[episode["id"]],
                    "return": float(episode["rewards"].sum()),
                    "length": len(episode["rewards"]),
                }

        super().__init__(
            env, step_data_callback=_StepData, episode_metadata_callback=Metadata, record_infos=True,
            data_format="arrow", **kw,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # the env's own options name the level and task it drew, so the episode replays from them alone
        self._buffer = dataclasses.replace(self._buffer, options=info["reset_options"])
        return obs, info

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Box):
            # step with the action as it will be stored, so a replay of the stored action steps identically
            action = np.asarray(action, self.action_space.dtype)
        episode = self._episode_id  # Minari moves to the next id when the episode ends
        obs, rew, terminated, truncated, info = super().step(action)
        self._outcomes[episode] = {"success": info["success"], "failure_reason": info["failure_reason"]}
        return obs, rew, terminated, truncated, info

    def create_dataset(self, dataset_id, **kw):
        """Minari's create_dataset, with `provenance()` added to the dataset metadata."""
        ds = super().create_dataset(dataset_id, **kw)
        ds.storage.update_metadata(provenance())
        return ds

    def discard_episode(self):
        """Drop the episode in progress (an abandoned run): nothing of it reaches the dataset."""
        self._buffer = None  # Minari internal: reset and create_dataset flush _buffer, and skip it when None
