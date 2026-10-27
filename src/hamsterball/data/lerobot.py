"""LeRobot export (spec §12): a one-way copy of a re-rendered dataset as a LeRobot v3 dataset (the `lerobot` extra)."""

from fractions import Fraction

import numpy as np

from ..core.model import DT
from .collector import ARTICULATED_OBSERVATION


def to_lerobot(dataset, repo_id, root, use_videos=True):
    """Write `dataset`, a re-rendered Minari dataset (see `rerender`), as LeRobot v3 dataset `repo_id` at `root`.

    Each frame pairs observation t with action t, so an episode's last observation (it has no action) is dropped.
    `state`, `goal` and (articulated tier) `proprio` map to `observation.state`, `observation.goal` and
    `observation.proprio`, each view to `observation.images.<view>`,
    the reward and episode end to `next.reward` and `next.done`, and the episode's task to LeRobot's task string.
    `use_videos=True` (LeRobot's default) encodes lossy video; False stores lossless PNG frames.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # the lerobot extra (Python 3.12+)

    kw = dataset.env_spec.kwargs
    fps = Fraction(1 / (DT * kw["frame_skip"])).limit_denominator(1000)
    if fps.denominator != 1:
        raise ValueError(f"LeRobot needs a whole-number fps; frame_skip {kw['frame_skip']} gives {fps}")
    views = [k for k in dataset.observation_space.keys() if k not in ARTICULATED_OBSERVATION.spaces]
    if not views:
        raise ValueError("no views in the observations: rerender the dataset first")
    act = dataset.action_space
    proprio = "proprio" in dataset.observation_space.spaces
    features = {
        "action": {"dtype": "float32", "shape": act.shape, "names": None},
        "observation.state": {"dtype": "float32", "shape": (21,), "names": None},
        "observation.goal": {"dtype": "float32", "shape": (10,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        **({"observation.proprio": {"dtype": "float32", "shape": (26,), "names": None}} if proprio else {}),
        **{
            f"observation.images.{v}": {
                "dtype": "video" if use_videos else "image",
                "shape": dataset.observation_space[v].shape,
                "names": ["height", "width", "channels"],
            }
            for v in views
        },
    }
    out = LeRobotDataset.create(repo_id, int(fps), features, root=root, use_videos=use_videos)
    ids = dataset.episode_indices
    for ep, meta in zip(dataset.iterate_episodes(ids), dataset.storage.get_episode_metadata(ids)):
        obs, n = ep.observations, len(ep.actions)
        done = ep.terminations | ep.truncations
        for t in range(n):
            out.add_frame({
                "action": np.asarray(ep.actions[t], np.float32),
                "observation.state": obs["state"][t],
                "observation.goal": obs["goal"][t],
                "next.reward": np.array([ep.rewards[t]], np.float32),
                "next.done": np.array([done[t]]),
                **({"observation.proprio": obs["proprio"][t]} if proprio else {}),
                **{f"observation.images.{v}": obs[v][t] for v in views},
                "task": meta["options"]["task"],
            })
        out.save_episode()
    out.finalize()
    return out
