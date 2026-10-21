"""The Collector (spec §12): Minari's DataCollector, storing what replay needs from a hamsterball env."""

import dataclasses
import importlib.metadata
import platform

import gymnasium as gym
import minari
import mujoco
import numpy as np
import pyarrow as pa
from minari.data_collector.callbacks import EpisodeMetadataCallback, StepDataCallback
from minari.dataset._storages import arrow_storage

from ..core.generator import LevelGenerator

_encode_info = arrow_storage._encode_info


def _encode_core_state(info):
    """Minari's info encoder, but `core_state` as a variable-length list: its size depends on the level (kinematic
    bodies add qpos and qvel), and Minari's fixed-size list per episode can't be read back across sizes. Decoding
    needs no change: each episode's rows have one size, so Minari's np.stack gives a (steps, size) array as before."""
    if "core_state" not in info:
        return _encode_info(info)
    rows = list(np.stack(info["core_state"]))
    arrays, fields = [pa.array(rows, pa.list_(pa.float64()))], [pa.field("core_state", pa.list_(pa.float64()))]
    if rest := {k: v for k, v in info.items() if k != "core_state"}:
        other = _encode_info(rest)
        arrays += other.flatten()
        fields += list(other.type)
    return pa.StructArray.from_arrays(arrays, fields=fields)


arrow_storage._encode_info = _encode_core_state  # Minari internal: every arrow write in this process goes through it


class _StepData(StepDataCallback):
    """Stores `state` and `goal` (from info, so with or without views or `obs_state`) and `core_state`.

    No pixels: re-render makes them from the core states. reset's `reset_options` goes per episode instead.
    """

    def __call__(self, env, **kw):
        step = super().__call__(env, **kw)
        info = step["info"]
        step["observation"] = {"state": info["state"], "goal": info["goal"]}
        step["info"] = {"core_state": info["core_state"]}
        return step


_inf = np.float32(np.inf)
# what a dataset stores per step: spec §10's state and goal, never pixels. Keys in sorted order, the order Minari's
# episode buffer writes them in, so add_to_dataset (which re-encodes in the space's order) writes the same Arrow schema
OBSERVATION = gym.spaces.Dict(
    goal=gym.spaces.Box(-_inf, _inf, (10,), np.float32), state=gym.spaces.Box(-_inf, _inf, (19,), np.float32)
)
# every env kwarg that shapes a space or the steps, so recover_environment() rebuilds an env with the same spaces and
# the same episode ending (truncate, time_limit_s)
SPACE_KWARGS = (
    "tier", "camera_action", "views", "view_size", "obs_state", "shaped", "frame_skip", "truncate", "time_limit_s"
)


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
        u.spec.kwargs.update({k: getattr(u, k) for k in SPACE_KWARGS})  # the defaults too, before Minari copies the spec
        if isinstance(u.spec.kwargs.get("levels"), LevelGenerator):
            # a generator object isn't JSON, and every episode stores its level in its options anyway; drop the pool
            # and its difficulty, so the recovered env takes the default pool, which accepts the default difficulty
            for k in ("levels", "difficulty"):
                u.spec.kwargs.pop(k, None)
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
            data_format="arrow", observation_space=OBSERVATION, **kw,
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

    def add_to_dataset(self, dataset):
        """Minari's add_to_dataset, keeping each episode's metadata: Minari's copy drops the seed and options, and it
        runs the episode metadata callback only in create_dataset."""
        self._flush_to_storage()  # Minari internal: the episode in progress joins _storage, as add_to_dataset does
        n = self._storage.total_episodes
        metas = [  # read before Minari resets _storage; the id is the dataset's, not this storage's
            {k: v for k, v in {**stored, **extra}.items() if k != "id"}
            for stored, extra in zip(
                self._storage.get_episode_metadata(range(n)), self._storage.apply(self._episode_metadata_callback)
            )
        ]
        first = dataset.storage.total_episodes
        super().add_to_dataset(dataset)
        dataset.storage.update_episode_metadata(metas, range(first, first + n))

    def discard_episode(self):
        """Drop the episode in progress (an abandoned run): nothing of it reaches the dataset."""
        self._buffer = None  # Minari internal: reset and create_dataset flush _buffer, and skip it when None
