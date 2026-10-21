"""Survive and hazards (spec §9) through gymnasium.make."""

from importlib.resources import files

import gymnasium as gym
import numpy as np
import pytest
import yaml

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level

STILL = np.zeros(2, np.float32)
FORWARD = np.array([0, 1], np.float32)


def toward_target(obs):
    g = obs["goal"][:2]
    return (g / np.linalg.norm(g)).astype(np.float32)


def play(env, policy, max_steps=5000):
    """Reset, then step until the episode ends; returns the step count, last step, and goal history.

    `policy` is a fixed action or a function of the observation."""
    obs, _ = env.reset(seed=0)
    goals = [obs["goal"]]
    for n in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = env.step(policy(obs) if callable(policy) else policy)
        goals.append(obs["goal"])
        if terminated or truncated:
            return n, reward, terminated, truncated, info, goals
    raise AssertionError("episode never ended")


def test_surviving_to_the_time_limit_terminates_in_success():
    env = gym.make("Hamsterball-v0", tasks=["survive"])
    n, reward, terminated, truncated, info, goals = play(env, STILL)
    assert (n, terminated, truncated) == (600, True, False)  # 10 s at 60 Hz
    assert info["success"] and info["failure_reason"] is None
    assert reward == 1.0 and info["reward_terms"] == {"event": 1.0, "time": 0.0, "shaping": 0.0}
    # region centre, active, fraction remaining = time left / time limit, one-hot survive
    np.testing.assert_allclose(goals[0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 1], atol=1e-6)
    assert goals[150][4] == pytest.approx(0.75)
    np.testing.assert_allclose(goals[-1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


def test_survive_with_truncate_false_still_succeeds_at_its_time_limit():
    env = gym.make("Hamsterball-v0", tasks=["survive"], truncate=False)
    n, reward, terminated, truncated, info, _ = play(env, STILL)
    assert (n, terminated, truncated, info["success"]) == (600, True, False, True)


def test_leaving_the_survive_region_fails():
    env = gym.make("Hamsterball-v0", tasks=["survive"])
    n, reward, terminated, truncated, info, _ = play(env, FORWARD)
    assert terminated and not info["success"] and info["failure_reason"] == "left_region"
    assert reward == -1.0 and info["state"][1] > 2  # out past the 2 m radius


def with_hazards(name, *hazards):
    """A bundled level with hazards added."""
    d = yaml.safe_load((files("hamsterball") / f"levels/{name}.yaml").read_text())
    d["hazards"] = list(hazards)
    return Level.from_yaml(yaml.safe_dump(d))


@pytest.mark.parametrize(
    "name, task",
    [
        ("first_goal", "reach_goal"),
        ("waypoint_loop", "waypoint_race"),
        ("item_scatter", "collect_items"),
        ("locked_goal", "unlock_goal"),
        ("first_survive", "survive"),
    ],
)
def test_a_hazard_fails_every_task(name, task):
    env = gym.make("Hamsterball-v0", tasks=[task], levels=[with_hazards(name, {"type": "sphere", "pos": [0, 0, 0.5], "radius": 0.2})])
    n, reward, terminated, truncated, info, _ = play(env, STILL)
    assert (n, terminated, info["failure_reason"], info["success"]) == (1, True, "hazard", False)
    assert info["reward_terms"]["event"] == -1.0


def test_a_hazard_crossed_within_one_env_step_triggers():
    # a 10 cm thick wall across the path to the goal; one env step is a whole second, so the shell skips clean over it
    wall = {"type": "box", "pos": [0, 4, 0.5], "size": [3, 0.05, 3]}
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=[with_hazards("first_goal", wall)], frame_skip=600)
    obs, _ = env.reset(seed=0)
    ys = [obs["state"][1]]
    for _ in range(5):
        obs, reward, terminated, _, info = env.step(FORWARD)
        ys.append(obs["state"][1])
        if terminated:
            break
    assert info["failure_reason"] == "hazard"
    assert ys[-2] < 3.95 and ys[-1] > 4.05  # neither end of the last segment is inside the wall


def test_a_step_that_reaches_the_goal_and_crosses_a_hazard_succeeds():
    over_goal = {"type": "sphere", "pos": [0, 8, 0.5], "radius": 1}
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=[with_hazards("first_goal", over_goal)])
    n, reward, terminated, truncated, info, _ = play(env, toward_target)
    assert terminated and info["success"] and info["failure_reason"] is None
    assert info["reward_terms"]["event"] == 1.0
