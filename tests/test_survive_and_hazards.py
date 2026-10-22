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
    env = gym.make("Hamsterball-v2", tasks=["survive"], levels=["first_survive"])
    n, reward, terminated, truncated, info, goals = play(env, STILL)
    assert (n, terminated, truncated) == (600, True, False)  # 10 s at 60 Hz
    assert info["success"] and info["failure_reason"] is None
    assert reward == 1.0 and info["reward_terms"] == {"event": 1.0, "time": 0.0, "shaping": 0.0}
    # region centre, active, fraction remaining = time left / time limit, one-hot survive
    np.testing.assert_allclose(goals[0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 1], atol=1e-6)
    assert goals[150][4] == pytest.approx(0.75)
    np.testing.assert_allclose(goals[-1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


def test_survive_with_truncate_false_still_succeeds_at_its_time_limit():
    env = gym.make("Hamsterball-v2", tasks=["survive"], levels=["first_survive"], truncate=False)
    n, reward, terminated, truncated, info, _ = play(env, STILL)
    assert (n, terminated, truncated, info["success"]) == (600, True, False, True)


def test_leaving_the_survive_region_fails():
    env = gym.make("Hamsterball-v2", tasks=["survive"], levels=["first_survive"])
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
    env = gym.make("Hamsterball-v2", tasks=[task], levels=[with_hazards(name, {"type": "sphere", "pos": [0, 0, 0.5], "radius": 0.2})])
    n, reward, terminated, truncated, info, _ = play(env, STILL)
    assert (n, terminated, info["failure_reason"], info["success"]) == (1, True, "hazard", False)
    assert info["reward_terms"]["event"] == -1.0


def test_a_hazard_crossed_within_one_env_step_triggers():
    # a 10 cm thick wall across the path to the goal; one env step is a whole second, so the shell skips clean over it
    wall = {"type": "box", "pos": [0, 4, 0.5], "size": [3, 0.05, 3]}
    d = yaml.safe_load((files("hamsterball") / "levels/first_goal.yaml").read_text())
    d["hazards"] = [wall]
    d["tasks"]["reach_goal"]["goal"]["pos"] = [0, 9, 0.5]  # out of that step's reach: success would beat the hazard
    env = gym.make("Hamsterball-v2", tasks=["reach_goal"], levels=[Level.from_yaml(yaml.safe_dump(d))], frame_skip=600)
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
    env = gym.make("Hamsterball-v2", tasks=["reach_goal"], levels=[with_hazards("first_goal", over_goal)])
    n, reward, terminated, truncated, info, _ = play(env, toward_target)
    assert terminated and info["success"] and info["failure_reason"] is None
    assert info["reward_terms"]["event"] == 1.0


# --- the touch rule: a marker triggers once the shell (radius 0.5 m) touches it; the region still tests the centre


def ball(y, radius=0.2):
    """A sphere marker on the +y axis at shell-centre height, its surface 0.5 m + a gap from the spawned centre."""
    return {"type": "sphere", "pos": [0, y, 0.5], "radius": radius}


TOUCH, CLEAR = ball(0.69), ball(0.75)  # 1 cm into the shell at spawn; 5 cm clear of it
FAR = ball(-12)


def touch_level(task, hazards=(), **markers):
    block = {"spawn": {"pos": [0, 0, 0.5]}, "time_limit_s": 30, **markers}
    d = {"kill_plane_z": -5, "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [20, 20, 0.5]}],
         "hazards": list(hazards), "tasks": {task: block}}
    return Level.from_yaml(yaml.safe_dump(d))


def first_steps(task, lv, n=30):
    env = gym.make("Hamsterball-v2", tasks=[task], levels=[lv])
    env.reset(seed=0)
    return [env.step(STILL) for _ in range(n)]


def test_a_shell_touching_a_hazard_fails():
    (_, reward, terminated, _, info), *_ = first_steps("reach_goal", touch_level("reach_goal", [TOUCH], goal=FAR), 1)
    assert terminated and info["failure_reason"] == "hazard"


def test_a_shell_a_hair_clear_of_a_hazard_does_not_fail():
    steps = first_steps("reach_goal", touch_level("reach_goal", [CLEAR], goal=FAR))
    assert not any(s[2] for s in steps)


def test_a_goal_counts_on_touch():
    (_, _, terminated, _, info), *_ = first_steps("reach_goal", touch_level("reach_goal", goal=TOUCH), 1)
    assert terminated and info["success"]


def test_an_item_is_collected_on_touch():
    (_, _, terminated, _, info), *_ = first_steps("collect_items", touch_level("collect_items", items=[TOUCH, FAR]), 1)
    assert not terminated and info["reward_terms"]["event"] == 1.0


def test_a_locked_goal_touched_does_nothing():
    steps = first_steps("unlock_goal", touch_level("unlock_goal", keys=[FAR], goal=TOUCH))
    assert not any(s[2] for s in steps) and all(s[4]["reward_terms"]["event"] == 0 for s in steps)


def test_the_survive_region_fails_only_once_the_centre_leaves():
    env = gym.make("Hamsterball-v2", tasks=["survive"], levels=["first_survive"])
    obs, _ = env.reset(seed=0)
    ys = [obs["state"][1]]
    while True:
        obs, _, terminated, _, info = env.step(FORWARD)
        ys.append(obs["state"][1])
        if terminated:
            break
    assert info["failure_reason"] == "left_region"
    assert 1.5 < ys[-2] <= 2 < ys[-1]  # the shell had crossed the edge a step earlier; only the centre leaving fails
