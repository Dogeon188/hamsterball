"""Re-step replay (spec §12): a determinism check of recorded episodes against this build."""

import warnings

import mujoco
import numpy as np

from ..core.step import STATE
from .collector import provenance

# the core state's components in mj_getState order, each one bit of mjtState
COMPONENTS = sorted(
    (int(v), k.removeprefix("mjSTATE_").lower())
    for k, v in mujoco.mjtState.__members__.items()
    if int(v) & STATE and int(v) & (int(v) - 1) == 0
)


class ReplayMismatch(AssertionError):
    """A re-stepped core state differs from the recorded one."""

    def __init__(self, episode, step, field, diff):
        self.episode, self.step, self.field, self.diff = episode, step, field, diff
        super().__init__(f"episode {episode} diverges at step {step}: largest difference {diff:.3g} in {field}")


def field(model, i):
    """Name core state element i, as `<component>[<index>]`."""
    for bit, name in COMPONENTS:
        n = mujoco.mj_stateSize(model, bit)
        if i < n:
            return f"{name}[{i}]"
        i -= n
    raise IndexError(i)


def replay(dataset, episodes=None):
    """Re-step each episode from its stored seed, reset options, and actions, comparing every core state bit for bit.

    `episodes` is a list of episode ids, or None for all. Raises ReplayMismatch at the first step that differs
    (step 0 is the reset); warns first when the dataset was recorded on other versions or another platform.
    """
    stored = dataset.storage.metadata
    for k, v in provenance().items():
        if stored.get(k) != v:
            warnings.warn(f"dataset {k} is {stored.get(k)}, this build has {v}: replay may diverge", stacklevel=2)
    env = dataset.recover_environment()
    ids = dataset.episode_indices if episodes is None else episodes
    for ep, meta in zip(dataset.iterate_episodes(ids), dataset.storage.get_episode_metadata(ids)):
        _, info = env.reset(seed=meta.get("seed"), options=meta["options"])
        recorded = ep.infos["core_state"]
        for t in range(len(recorded)):
            if t:
                _, _, _, _, info = env.step(ep.actions[t - 1])
            got, want = info["core_state"], recorded[t]
            if got.tobytes() != want.tobytes():
                d = np.abs(got - want)
                i = int(np.nanargmax(d)) if not np.isnan(d).all() else 0
                raise ReplayMismatch(ep.id, t, field(env.unwrapped.core.model, i), float(d[i]))
