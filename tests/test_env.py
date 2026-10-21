"""The reach-goal env (spec §9, §10) on the bundled level, through gymnasium.make."""

import math
from importlib.resources import files

import gymnasium as gym
import numpy as np
import pytest
import yaml

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level

BUNDLED = yaml.safe_load((files("hamsterball") / "levels/first_goal.yaml").read_text())
INFO_KEYS = {"core_state", "state", "goal", "success", "failure_reason", "reward_terms"}


def level(spawn_yaw_deg=0.0, goal=None):
    d = yaml.safe_load(yaml.safe_dump(BUNDLED))
    d["tasks"]["reach_goal"]["spawn"]["yaw_deg"] = spawn_yaw_deg
    if goal:
        d["tasks"]["reach_goal"]["goal"] = goal
    return Level.from_yaml(yaml.safe_dump(d))


def make(**kw):
    return gym.make("Hamsterball-v0", tasks=["reach_goal"], **kw)


def toward_goal(obs, sign=1.0):
    """A scripted controller: run straight at (or away from) the goal, read in the camera yaw frame."""
    g = obs["goal"][:2]
    return (sign * g / np.linalg.norm(g)).astype(np.float32)


def run(env, policy, max_steps=5000):
    obs, info = env.reset(seed=0)
    for n in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        assert INFO_KEYS <= info.keys()
        if terminated or truncated:
            return n, reward, terminated, truncated, info
    raise AssertionError("episode never ended")


def test_observations_are_the_19_d_state_and_10_d_goal():
    env = make()
    assert env.action_space.shape == (2,)
    assert env.observation_space["state"].shape == (19,) and env.observation_space["goal"].shape == (10,)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    s, g = obs["state"], obs["goal"]
    np.testing.assert_allclose(s[0:3], [0, 0, 0.5])  # shell position, world
    assert s[3] == pytest.approx(5.5)  # above the kill plane at -5
    np.testing.assert_allclose(s[4:6], [0, 1])  # sin, cos of camera yaw 0
    np.testing.assert_allclose(s[6:12], 0, atol=1e-6)  # at rest
    np.testing.assert_allclose(s[12:15], [0, 0, -0.34], atol=1e-6)  # hamster 1 cm above the bottom
    assert s[18] == 1  # inside the airborne grace
    np.testing.assert_allclose(g, [0, 8, 0, 1, 1, 1, 0, 0, 0, 0], atol=1e-6)
    assert {"core_state", "state", "goal"} <= info.keys()
    np.testing.assert_array_equal(info["state"], s)


def test_yaw_0_faces_plus_y_and_vectors_read_in_the_camera_yaw_frame():
    env = make(levels=[level(spawn_yaw_deg=90)])
    obs, _ = env.reset(seed=0)
    s = obs["state"]
    np.testing.assert_allclose(s[4:6], [1, 0], atol=1e-6)
    # camera forward is world -x, so the goal 8 m along world +y is 8 m to camera right
    np.testing.assert_allclose(obs["goal"][:3], [8, 0, 0], atol=1e-6)
    for _ in range(60):
        obs, *_ = env.step(np.array([0, 1], np.float32))  # run camera-forward for a second
    s = obs["state"]
    assert s[0] < -1 and abs(s[1]) < 0.3 * -s[0]  # world -x, give or take the chase yaw's drift
    assert s[7] > 1 and abs(s[6]) < 0.05 * s[7]  # shell velocity: camera forward
    assert s[9] < -1 and abs(s[10]) < 0.05 * -s[9]  # rolling forward spins the shell about camera -x
    assert s[13] > 0.1  # the hamster leans forward


def test_a_goal_crossed_within_one_env_step_triggers():
    # a 10 cm thick wall across the path; one env step is a whole second, so the shell skips clean over it
    wall = {"type": "box", "pos": [0, 4, 0.5], "size": [3, 0.05, 3]}
    env = make(levels=[level(goal=wall)], frame_skip=600)
    obs, _ = env.reset(seed=0)
    ys = [obs["state"][1]]
    for _ in range(5):
        obs, reward, terminated, _, info = env.step(np.array([0, 1], np.float32))
        ys.append(obs["state"][1])
        if terminated:
            break
    assert info["success"]
    assert ys[-2] < 3.95 and ys[-1] > 4.05  # neither end of the last segment is inside the wall


def test_running_at_the_goal_succeeds():
    n, reward, terminated, truncated, info = run(make(), toward_goal)
    assert terminated and not truncated and info["success"] and info["failure_reason"] is None
    assert reward == pytest.approx(1 - 0.001)
    assert info["reward_terms"] == {"event": 1.0, "time": -0.001, "shaping": 0.0}
    np.testing.assert_allclose(info["goal"], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # after success: zeros, one-hot stays


def test_running_off_the_edge_fails_at_the_kill_plane():
    n, reward, terminated, truncated, info = run(make(), lambda obs: toward_goal(obs, -1))
    assert terminated and not info["success"] and info["failure_reason"] == "kill_plane"
    assert reward == pytest.approx(-1 - 0.001)
    assert info["state"][2] < -5


@pytest.mark.parametrize("frame_skip, time_limit_s, steps", [(10, 30, 1800), (7, 0.505, 44)])
def test_standing_still_truncates_at_the_time_limit(frame_skip, time_limit_s, steps):
    env = make(frame_skip=frame_skip, time_limit_s=None if time_limit_s == 30 else time_limit_s)
    n, reward, terminated, truncated, info = run(env, lambda obs: np.zeros(2, np.float32))
    assert (n, truncated, terminated) == (steps, True, False)
    assert reward == pytest.approx(-0.001) and info["failure_reason"] is None and not info["success"]


def test_camera_action_adds_cam_yaw_and_orbits_at_120_deg_per_s():
    env = make(camera_action=True)
    assert env.action_space.shape == (3,)
    env.reset(seed=0)
    for _ in range(30):  # half a second at 60 Hz
        obs, *_ = env.step(np.array([0, 0, 1], np.float32))
    sin, cos = obs["state"][4:6]
    assert math.degrees(math.atan2(sin, cos)) == pytest.approx(60, abs=1e-3)
