"""Re-render (spec §12): pixels from a dataset's stored core states, with no physics stepped, so on any platform."""

from fractions import Fraction

import minari
import numpy as np
from minari.data_collector import EpisodeBuffer

from ..core.model import DT
from .collector import provenance

CARRIED = ("algorithm_name", "author", "author_email", "code_permalink", "description", "requirements")


def episodes(dataset, views, view_size, ids=None):
    """Yield (episode, its metadata, observations) per episode, observations re-rendered from the stored core states.

    The observations are what an env with `views` at `view_size` returns: `state` and `goal` recomputed by the env's
    marker tests from each core state in order (so hidden items and an inactive goal match the recording), and a
    frame per view.
    """
    env = dataset.recover_environment(views=tuple(views), view_size=tuple(view_size), obs_state=True).unwrapped
    ids = dataset.episode_indices if ids is None else ids
    try:
        for ep, meta in zip(dataset.iterate_episodes(ids), dataset.storage.get_episode_metadata(ids)):
            # reset is mj_resetData plus the spawn, no physics stepped: it sets the stored first core state exactly
            obs = [env.reset(seed=meta.get("seed"), options=meta["options"])[0]]
            obs += [env.step_to(s)[0] for s in ep.infos["core_state"][1:]]
            yield ep, meta, {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    finally:
        env.close()


def rerender(dataset, views, view_size, out_id):
    """Write a new Minari dataset `out_id`: `dataset` with `views` at `view_size` added to its observations, lossless.

    Actions, rewards, terminations, infos, seeds, and episode metadata carry over; the env_spec gains the views, so
    `recover_environment()` rebuilds an env with the new observation space.
    """
    env = dataset.recover_environment(views=tuple(views), view_size=tuple(view_size), obs_state=True)
    src = dataset.storage.metadata
    out = minari.create_dataset_from_buffers(
        out_id, [], env=env, action_space=dataset.action_space, data_format="arrow", jpeg_encoding=False,
        **{k: src[k] for k in CARRIED if src.get(k) is not None},
    )
    env.close()
    for ep, meta, obs in episodes(dataset, views, view_size):  # one episode in memory at a time
        out.storage.update_episodes([EpisodeBuffer(
            id=int(ep.id), seed=meta.get("seed"), options=meta["options"], observations=obs, actions=ep.actions,
            rewards=ep.rewards, terminations=ep.terminations, truncations=ep.truncations, infos=ep.infos,
        )])
        out.storage.update_episode_metadata([meta], [int(ep.id)])
    # the physics was recorded on the source's build and platform, not this one
    out.storage.update_metadata({k: src[k] for k in provenance() if k in src})
    return minari.load_dataset(out_id)


def to_video(dataset, episode, view, path, view_size=None):
    """Write episode `episode` of `dataset`, re-rendered from `view`, as an H.264 mp4 at the env-step rate.

    `view_size` is (H, W), even numbers (yuv420p), by default the dataset's.
    """
    import av  # the data extra

    view_size = view_size or dataset.env_spec.kwargs["view_size"]
    _, _, obs = next(episodes(dataset, (view,), view_size, [episode]))
    rate = Fraction(1 / (DT * dataset.env_spec.kwargs["frame_skip"])).limit_denominator(1000)
    with av.open(str(path), "w") as f:
        stream = f.add_stream("libx264", rate=rate)
        stream.height, stream.width = view_size
        stream.pix_fmt = "yuv420p"
        for frame in obs[view]:
            f.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        f.mux(stream.encode())  # flush
