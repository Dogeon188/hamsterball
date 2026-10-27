"""Pixel views on the Warp vector env (GPU effort, ticket 10): shapes and dtype, the mocap poses against the CPU's,
and each world's own marker state in its pixels. No pixel comparison with the CPU renderer."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from test_warp_env import DEVICE, SCENARIOS, record, warp_env

from hamsterball.core.level import Level
from hamsterball.core.model import HAM_QPOS, SHELL_QPOS
from hamsterball.core.views import VIEWS

pytestmark = pytest.mark.gpu

SIZE = (20, 28)  # (height, width): small, and not square, so a swapped axis shows


def px(t):
    return t.cpu().numpy()


@pytest.mark.parametrize("views", [VIEWS, ("follow",), ("first_person", "chase")])
def test_each_view_is_a_uint8_tensor_of_the_view_size(views):
    import torch

    lv, _, _ = SCENARIOS["sphere_goal"]
    venv = warp_env(lv, 3, views=views, view_size=SIZE)
    obs, _ = venv.reset(seed=0)
    assert list(obs) == ["state", "goal", *views]
    assert sorted(venv.single_observation_space) == sorted(["state", "goal", *views])  # a Dict space sorts its keys
    assert venv.single_observation_space[views[0]] == gym.spaces.Box(0, 255, (*SIZE, 3), np.uint8)
    assert venv.observation_space[views[0]].shape == (3, *SIZE, 3)
    for _ in range(2):
        for v in views:
            assert isinstance(obs[v], torch.Tensor) and obs[v].device == venv.torch_device
            assert obs[v].dtype == torch.uint8 and obs[v].shape == (3, *SIZE, 3)
            assert px(obs[v]).std() > 0  # something is drawn
        obs, *_ = venv.step(np.zeros((3, 2), np.float32))
    assert not np.array_equal(px(obs[views[0]])[0], px(obs[views[-1]])[0]) or len(views) == 1


def test_a_pixels_only_env_has_only_the_views():
    lv, _, _ = SCENARIOS["sphere_goal"]
    venv = warp_env(lv, 2, views=("chase",), obs_state=False, view_size=SIZE)
    obs, _ = venv.reset(seed=0)
    assert list(obs) == ["chase"] and list(venv.single_observation_space) == ["chase"]
    obs, reward, terminated, truncated, info = venv.step(np.zeros((2, 2), np.float32))
    assert list(obs) == ["chase"] and reward.shape == (2,)
    with pytest.raises(ValueError, match="at least one view"):
        warp_env(lv, 2, obs_state=False)


def test_an_unknown_view_is_an_error():
    lv, _, _ = SCENARIOS["sphere_goal"]
    with pytest.raises(ValueError, match="unknown views"):
        warp_env(lv, 1, views=("birdseye",))


def test_the_final_observation_holds_the_views_of_the_step_that_ended_the_episode():
    lv, _, _ = SCENARIOS["sphere_goal"]
    venv = warp_env(lv, 2, views=("chase",), view_size=SIZE, time_limit_s=1 / 6)  # 10 env steps
    first, _ = venv.reset(seed=0)
    for _ in range(10):
        obs, _, _, truncated, info = venv.step(np.array([[0, 1.0]] * 2, np.float32))
    assert truncated.all() and info["_final_obs"].all()
    assert set(info["final_obs"]) == {"state", "goal", "chase"}
    np.testing.assert_array_equal(px(obs["chase"]), px(first["chase"]))  # the next episode starts where the first did
    assert info["final_obs"]["chase"].dtype == obs["chase"].dtype
    assert not np.array_equal(px(info["final_obs"]["chase"]), px(obs["chase"]))  # the ball had rolled


def mocap_slice(model):
    """Where the mocap poses sit in a core state: just before the userdata."""
    n = 7 * model.nmocap
    end = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION) - model.nuserdata
    return slice(end - n, end), model.nmocap


def test_the_views_and_features_are_posed_as_on_the_cpu():
    lv = Level.from_dict({
        "kill_plane_z": -5, "follow_yaw_deg": 35.0, "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}],
        "tasks": {"reach_goal": {"spawn": {"pos": [0, 0, 0.5], "yaw_deg": 0}, "time_limit_s": 30,
                                 "goal": {"type": "sphere", "pos": [0, 8, 0.5], "radius": 1}}}})
    ep = record(lv, [0.2, 1.0, 0.6, -0.5], {"time_limit_s": 5, "camera_action": True})
    venv = warp_env(lv, 1, views=VIEWS, view_size=SIZE, time_limit_s=5, camera_action=True)
    venv.reset(seed=0)
    sl, nmocap = mocap_slice(venv.core.cpu_model)
    for t in range(0, 40, 3):
        want = ep["steps"][t][0]
        venv.step_to(want[None])
        got = venv.core.state(0)
        gp, wp_ = got[sl][:3 * nmocap].reshape(-1, 3), want[sl][:3 * nmocap].reshape(-1, 3)
        np.testing.assert_allclose(gp, wp_, atol=2e-5, err_msg=f"mocap_pos at step {t}")
        gq, wq = got[sl][3 * nmocap:].reshape(-1, 4), want[sl][3 * nmocap:].reshape(-1, 4)
        for i in range(nmocap):  # q and -q are one rotation
            sign = np.sign(gq[i] @ wq[i])
            np.testing.assert_allclose(sign * gq[i], wq[i], atol=2e-5, err_msg=f"mocap_quat {i} at step {t}")


def two_spawns(task, block, *spawns):
    """One level per spawn: the same platform and markers, only the spawn differs."""
    return [Level.from_dict({
        "kill_plane_z": -5, "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [20, 20, 0.5]}],
        "tasks": {task: {"spawn": {"pos": [*s, 0.5], "yaw_deg": 0}, "time_limit_s": 30, **block}}}) for s in spawns]


def same_state_different_history(task, block, view, spawns):
    """Worlds on `spawns`' levels, all put at the first level's spawn pose: their pixels differ only by what each swept
    through to get there (a marker the swept segment touched is collected, one it didn't isn't)."""
    from hamsterball.gpu.vector_env import WarpVectorEnv

    venv = WarpVectorEnv(len(spawns), levels=two_spawns(task, block, *spawns), tasks=[task], device=DEVICE,
                         views=(view,), view_size=(48, 48))
    venv.reset(seed=0, options={"level_index": np.arange(len(spawns))})
    target = venv.core.states()[0]  # at the first spawn: where every world ends
    obs, *_ = venv.step_to(np.stack([target] * len(spawns)))
    return px(obs[view]).astype(float) / 255


def count(img, pick):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return int(pick(r, g, b).sum())


YELLOW = lambda r, g, b: (r > 0.3) & (g > 0.75 * r) & (g < 0.95 * r) & (b < 0.3 * r)  # noqa: E731
GREEN = lambda r, g, b: (g > 1.6 * r) & (g > 1.6 * b)  # noqa: E731


@pytest.mark.parametrize("view", ["chase", "follow"])
def test_a_collected_item_vanishes_in_its_own_world_only(view):
    items = [{"type": "sphere", "pos": [2, 0, 0.5], "radius": 0.5}, {"type": "sphere", "pos": [0, -15, 0.5], "radius": 0.5}]  # the second is behind the camera
    imgs = same_state_different_history("collect_items", {"items": items}, view, [(0.0, 0.0), (6.0, 0.0)])
    kept, taken = imgs  # world 0 swept nothing; world 1 swept through the item at (2, 0)
    assert count(kept, YELLOW) > 5, "the uncollected item is drawn"
    assert count(taken, YELLOW) == 0, "the collected item is gone"


@pytest.mark.parametrize("view", ["chase", "follow"])
def test_the_unlock_goal_changes_colour_in_the_world_that_has_every_key(view):
    block = {"keys": [{"type": "sphere", "pos": [2, 0, 0.5], "radius": 0.5}], "goal": {"type": "sphere", "pos": [0, 6, 0.5], "radius": 1}}
    imgs = same_state_different_history("unlock_goal", block, view, [(0.0, 0.0), (6.0, 0.0)])
    locked, open_ = imgs
    assert count(locked, GREEN) == 0, "a locked goal is not the goal's green"
    assert count(open_, GREEN) > 5, "an unlocked goal is green"


def test_first_person_looks_along_the_facing_from_the_eye():
    block = {"goal": {"type": "sphere", "pos": [0, 6, 0.5], "radius": 1}}
    imgs = same_state_different_history("reach_goal", block, "first_person", [(0.0, 0.0), (1.0, 0.0)])
    assert count(imgs[0], GREEN) > 5, "the goal ahead is in view"
