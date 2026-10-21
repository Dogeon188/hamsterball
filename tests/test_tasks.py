"""Multi-target tasks and shaping (spec §9): waypoint race, collect items, unlock goal."""

import gymnasium as gym
import numpy as np
import pytest
import yaml

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level

FORWARD = np.array([0, 1], np.float32)


def toward_target(obs):
    """A scripted controller: steer the shell's velocity toward the current target, in the camera yaw frame."""
    g, v = obs["goal"][:2], obs["state"][6:8]
    want = g / np.linalg.norm(g) * min(3.0, np.linalg.norm(g))  # aim at 3 m/s, slowing on the last metres
    a = want - v
    return (a / max(1.0, np.linalg.norm(a))).astype(np.float32)


def wall(y, x=0.0):
    """A 4 cm thick marker across the +y path at y; the shell skips over it within one env step."""
    return {"type": "box", "pos": [x, y, 0.5], "size": [3, 0.02, 3]}


def far(x, y):
    return {"type": "sphere", "pos": [x, y, 0.5], "radius": 1}


def level(task, **markers):
    block = {"spawn": {"pos": [0, 0, 0.5]}, "time_limit_s": 30, **markers}
    d = {"kill_plane_z": -5, "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [20, 20, 0.5]}], "tasks": {task: block}}
    return Level.from_yaml(yaml.safe_dump(d))


def make(task, lv, **kw):
    env = gym.make("Hamsterball-v0", tasks=[task], levels=[lv], **kw)
    obs, info = env.reset(seed=0)
    return env, obs


def roll_forward(env, steps):
    """Run camera-forward (+y), returning each step's (obs, reward, terminated, info)."""
    out = []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(FORWARD)
        out.append((obs, reward, terminated, info))
        if terminated:
            break
    return out


def test_one_step_crossing_waypoints_i_and_i_plus_1_counts_both():
    env, obs = make("waypoint_race", level("waypoint_race", waypoints=[wall(3), wall(3.1), far(15, -15)]), frame_skip=600)
    np.testing.assert_allclose(obs["goal"][:5], [0, 3, 0, 1, 1], atol=1e-6)
    events = [(o, i["reward_terms"]["event"]) for o, _, _, i in roll_forward(env, 3)]
    hit = [(o, e) for o, e in events if e]
    assert len(hit) == 1 and hit[0][1] == 2
    assert hit[0][0]["goal"][4] == pytest.approx(1 / 3)


def test_a_passed_waypoint_turns_dim_green_and_reset_restores_it():
    from hamsterball.core.level import MARKER_RGBA, PASSED_RGBA

    env, _ = make("waypoint_race", level("waypoint_race", waypoints=[wall(3), far(15, -15)]))
    for _ in range(300):
        if env.step(FORWARD)[4]["reward_terms"]["event"]:  # waypoint 0 passed
            break
    else:
        raise AssertionError("never reached waypoint 0")
    assert rgba(env, "waypoint_race/waypoints/0") == pytest.approx(PASSED_RGBA)
    assert rgba(env, "waypoint_race/waypoints/1") == pytest.approx(MARKER_RGBA["waypoints"])
    env.reset(seed=0)
    assert rgba(env, "waypoint_race/waypoints/0") == pytest.approx(MARKER_RGBA["waypoints"])


def test_a_waypoint_out_of_order_does_not_count():
    env, _ = make("waypoint_race", level("waypoint_race", waypoints=[far(15, -15), wall(3)]), frame_skip=60)
    steps = roll_forward(env, 25)
    assert steps[-1][0]["state"][1] > 5  # well past the second waypoint
    assert all(i["reward_terms"]["event"] == 0 for *_, i in steps)
    assert steps[-1][0]["goal"][4] == 1


def rgba(env, name):
    m = env.unwrapped.core.model
    return m.geom_rgba[m.geom(name).id].tolist()


def test_collect_items_targets_the_nearest_item_and_hides_it_once_collected():
    env, obs = make("collect_items", level("collect_items", items=[far(0, -6), wall(3)]), frame_skip=600)
    np.testing.assert_allclose(obs["goal"][:5], [0, 3, 0, 1, 1], atol=1e-6)
    assert rgba(env, "collect_items/items/1")[3] > 0
    obs, _, _, info = next(s for s in roll_forward(env, 3) if s[3]["reward_terms"]["event"])
    assert info["reward_terms"]["event"] == 1 and not info["success"]
    assert rgba(env, "collect_items/items/1")[3] == 0 and rgba(env, "collect_items/items/0")[3] > 0
    other = np.linalg.norm(np.array([0, -6, 0.5]) - obs["state"][:3])  # the camera yaw frame turns; distance doesn't
    assert np.linalg.norm(obs["goal"][:3]) == pytest.approx(other, rel=1e-5)
    assert obs["goal"][4] == pytest.approx(0.5)


def test_unlock_goal_ignores_the_goal_until_every_key_is_collected():
    env, obs = make("unlock_goal", level("unlock_goal", keys=[far(-10, -10)], goal=wall(3)), frame_skip=60)
    np.testing.assert_allclose(obs["goal"][:5], [-10, -10, 0, 1, 1], atol=1e-6)  # the key, not the nearer goal
    locked = rgba(env, "unlock_goal/goal")
    steps = roll_forward(env, 25)
    assert steps[-1][0]["state"][1] > 5  # straight through the goal
    assert all(i["reward_terms"]["event"] == 0 and not i["success"] for *_, i in steps)
    assert rgba(env, "unlock_goal/goal") == locked


def test_collecting_the_last_key_unlocks_the_goal_and_its_look():
    env, obs = make("unlock_goal", level("unlock_goal", keys=[wall(3)], goal=far(10, 10)), frame_skip=600)
    locked = rgba(env, "unlock_goal/goal")
    obs, _, _, info = next(s for s in roll_forward(env, 3) if s[3]["reward_terms"]["event"])
    assert info["reward_terms"]["event"] == 1 and not info["success"]
    assert rgba(env, "unlock_goal/keys/0")[3] == 0
    assert rgba(env, "unlock_goal/goal") != locked
    assert obs["goal"][4] == pytest.approx(0.5)  # (0 keys + goal) / (1 key + 1)
    env.reset(seed=0)
    assert rgba(env, "unlock_goal/goal") == locked and rgba(env, "unlock_goal/keys/0")[3] > 0  # reset restores looks


def test_only_the_episode_task_markers_show():
    d = {
        "kill_plane_z": -5,
        "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [20, 20, 0.5]}],
        "tasks": {
            "reach_goal": {"spawn": {"pos": [0, 0, 0.5]}, "time_limit_s": 30, "goal": far(0, 8)},
            "collect_items": {"spawn": {"pos": [0, 0, 0.5]}, "time_limit_s": 30, "items": [far(0, 4)]},
        },
    }
    env, _ = make("collect_items", Level.from_yaml(yaml.safe_dump(d)))
    assert rgba(env, "reach_goal/goal")[3] == 0 and rgba(env, "collect_items/items/0")[3] > 0


@pytest.mark.parametrize("gamma", [0.99, 0.5])
def test_shaping_adds_gamma_phi_next_minus_phi_and_phi_is_0_at_the_terminal_state(gamma):
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=["first_goal"], shaped=True, shaping_gamma=gamma)
    obs, _ = env.reset(seed=0)
    dist = 8.0  # first_goal: the goal sits 8 m ahead of the spawn
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(toward_target(obs))
        t = info["reward_terms"]
        assert reward == pytest.approx(t["event"] + t["time"] + t["shaping"])
        new = np.linalg.norm(obs["goal"][:3])
        if terminated:
            assert info["success"] and t["shaping"] == pytest.approx(0.01 * dist)  # γ·0 − (−0.01·d)
            return
        assert t["shaping"] == pytest.approx(gamma * -0.01 * new + 0.01 * dist, abs=1e-7)
        dist = new
    raise AssertionError("never reached the goal")


def test_shaping_is_off_by_default():
    env, _ = make("collect_items", level("collect_items", items=[far(0, 6)]))
    assert all(i["reward_terms"]["shaping"] == 0 for *_, i in roll_forward(env, 30))


@pytest.mark.parametrize("task, targets", [("waypoint_race", "waypoints"), ("collect_items", "items"), ("unlock_goal", "keys")])
def test_each_task_plays_to_success_on_its_bundled_level(task, targets):
    env = gym.make("Hamsterball-v0", tasks=[task])  # the default pool: every bundled level
    obs, _ = env.reset(seed=0)
    n = len(env.unwrapped.level.tasks[task][targets])
    events = []
    for _ in range(5000):
        obs, reward, terminated, truncated, info = env.step(toward_target(obs))
        events.append(info["reward_terms"]["event"])
        if terminated or truncated:
            break
    assert terminated and info["success"]
    assert sum(events) == n + 1  # +1 per waypoint, item, or key, +1 on success
    onehot = np.eye(5)[["reach_goal", "waypoint_race", "collect_items", "unlock_goal", "survive"].index(task)]
    np.testing.assert_array_equal(obs["goal"], [0, 0, 0, 0, 0, *onehot])
