"""Pose stream (pose-stream spec): a recorded episode's header and poses, with no physics stepped."""

from ..core.model import DT
from .rerender import walk

POSE_FORMAT = 1


def pose_stream(dataset, episode):
    """Yield the header dict, then episode `episode`'s n + 1 poses (`HamsterballEnv.pose()` after reset and each step)."""
    env = dataset.recover_environment(views=()).unwrapped
    try:
        (ep,) = dataset.iterate_episodes([episode])
        (meta,) = dataset.storage.get_episode_metadata([episode])
        for i, _ in enumerate(walk(env, ep, meta)):
            if i == 0:
                yield {
                    "pose_format": POSE_FORMAT, "level": env.level.canonical(), "level_id": env.level.id,
                    "task": env.task, "rate": 1 / (env.frame_skip * DT), "step_limit": env.step_limit,
                }
            yield env.pose()
    finally:
        env.close()
