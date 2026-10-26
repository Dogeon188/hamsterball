"""env.unwrapped.pose() (pose-stream ticket 01): what a front end needs, from the public env."""

import json
import math

import gymnasium as gym
import mujoco
import numpy as np
import pytest
import yaml

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level
from test_tasks import far, level, toward_target

ZERO = np.zeros(2, np.float32)


def make(task, levels, **kw):
    return gym.make("Hamsterball-v2", tasks=[task], levels=levels, **kw)


def states(pose):
    return {k: m["state"] for k, m in pose["markers"].items()}


def play(env, obs, limit=5000):
    """Drive toward the target to the end, yielding each env step's pose (reset's first)."""
    yield env.unwrapped.pose()
    for _ in range(limit):
        obs, _, terminated, truncated, _ = env.step(toward_target(obs))
        yield env.unwrapped.pose()
        if terminated or truncated:
            return


def test_pose_is_json_plain_data_and_holds_the_shell_and_hamster_at_reset():
    env = make("reach_goal", ["first_goal"])
    env.reset(seed=0)
    p = env.unwrapped.pose()
    assert json.loads(json.dumps(p)) == p
    assert (p["step"], p["time"], p["facing"]) == (0, 0.0, 0.0)
    assert p["pitch"] == pytest.approx(math.radians(15))
    assert set(p["bodies"]) == {"shell", "hamster"}
    np.testing.assert_allclose(p["bodies"]["shell"]["pos"], [0, 0, 0.5])
    np.testing.assert_allclose(p["bodies"]["shell"]["quat"], [1, 0, 0, 0], atol=1e-9)
    assert "outcome" not in p and p["target"] == "goal" and states(p) == {"goal": "active"}
    env.step(ZERO)
    assert env.unwrapped.pose()["step"] == 1


def test_pose_does_not_change_the_trajectory_or_info():
    a, b = make("reach_goal", ["first_goal"]), make("reach_goal", ["first_goal"])
    a.reset(seed=0), b.reset(seed=0)
    for _ in range(20):
        a.unwrapped.pose()
        oa, ra, _, _, ia = a.step(np.array([0.3, 1], np.float32))
        ob, rb, _, _, ib = b.step(np.array([0.3, 1], np.float32))
        assert "pose" not in ia and ia.keys() == ib.keys() and ra == rb
        np.testing.assert_array_equal(oa["state"], ob["state"])


def test_locked_goal_keys_collect_then_the_goal_unlocks_and_succeeds():
    env = make("unlock_goal", ["locked_goal"])
    obs, _ = env.reset(seed=0)
    poses = list(play(env, obs))
    first = states(poses[0])
    assert first == {"keys[0]": "active", "keys[1]": "active", "goal": "locked"}
    seen = [states(p) for p in poses]
    last = poses[-1]
    assert states(last) == {"keys[0]": "done", "keys[1]": "done", "goal": "done"}
    assert last["target"] is None and last["outcome"] == "success"
    assert all("outcome" not in p for p in poses[:-1])
    # the goal turns active exactly when the last key is done, before success
    unlocked = [s for s in seen if s["goal"] == "active"]
    assert unlocked and all(s["keys[0]"] == s["keys[1]"] == "done" for s in unlocked)
    assert any(s["keys[0]"] == "done" and s["keys[1]"] == "active" or s["keys[1]"] == "done" and s["keys[0]"] == "active"
               for s in seen)


def test_waypoint_loop_passes_each_waypoint_and_wakes_the_next():
    env = make("waypoint_race", ["waypoint_loop"])
    obs, _ = env.reset(seed=0)
    poses = list(play(env, obs))
    keys = list(poses[0]["markers"])
    assert states(poses[0]) == {keys[0]: "active", **{k: "locked" for k in keys[1:]}}
    for p in poses:  # in order: done..., one active (until the end), locked...
        s = [states(p)[k] for k in keys]
        assert s == sorted(s, key=["done", "active", "locked"].index)
        assert s.count("active") <= 1
    assert set(states(poses[-1]).values()) == {"done"} and poses[-1]["outcome"] == "success"


def test_target_names_the_marker_the_goal_vector_points_at():
    env = make("collect_items", ["item_scatter"])
    obs, _ = env.reset(seed=0)
    e = env.unwrapped
    for _ in range(300):
        p = e.pose()
        if p["target"] is None:
            break
        m = e._marker(next(t for t in e._todo if (t[0] if t[1] is None else f"{t[0]}[{t[1]}]") == p["target"]))
        c = np.array(m["pos"]) - e.core.data.qpos[:3]
        facing = p["facing"]
        want = [c[0] * math.cos(facing) + c[1] * math.sin(facing), -c[0] * math.sin(facing) + c[1] * math.cos(facing)]
        np.testing.assert_allclose(obs["goal"][:2], want, atol=1e-5)
        obs, *_ = env.step(toward_target(obs))
    assert p["target"] is None or p["target"].startswith("items[")


def box_level(task, hazards=(), **markers):
    block = {"spawn": {"pos": [0, 0, 0.5]}, "time_limit_s": 30, **markers}
    d = {"kill_plane_z": -5, "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [20, 20, 0.5]}],
         "hazards": list(hazards), "tasks": {task: block}}
    return Level.from_yaml(yaml.safe_dump(d))


def last_pose(env, action, limit=5000):
    for _ in range(limit):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return env.unwrapped.pose()


def test_outcome_is_on_the_last_pose_only_and_names_how_it_ended():
    goal = {"type": "sphere", "pos": [0, 3, 0.5], "radius": 1}
    env = make("reach_goal", [box_level("reach_goal", goal=goal)])
    env.reset(seed=0)
    assert last_pose(env, np.array([0, 1], np.float32))["outcome"] == "success"

    haz = {"type": "sphere", "pos": [0, 0, 0.5], "radius": 0.2}
    env = make("reach_goal", [box_level("reach_goal", [haz], goal=far(0, 15))])
    env.reset(seed=0)
    p = last_pose(env, ZERO)
    assert p["outcome"] == "hazard" and states(p) == {"goal": "active", "hazards[0]": "active"}

    env = make("reach_goal", [box_level("reach_goal", goal=far(0, 15))], time_limit_s=1)
    env.reset(seed=0)
    assert last_pose(env, ZERO)["outcome"] == "truncated"

    env = make("reach_goal", [box_level("reach_goal", goal=far(0, 15))])
    env.reset(seed=0)
    assert last_pose(env, np.array([1, 0], np.float32))["outcome"] == "kill_plane"

    env = make("survive", [box_level("survive", region={"type": "sphere", "pos": [0, 0, 0.5], "radius": 1})])
    env.reset(seed=0)
    p = last_pose(env, np.array([1, 0], np.float32))
    assert p["outcome"] == "left_region" and p["target"] == "region"

    env = make("survive", [box_level("survive", region={"type": "sphere", "pos": [0, 0, 0.5], "radius": 1})],
               time_limit_s=1)
    env.reset(seed=0)
    p = last_pose(env, ZERO)
    assert p["outcome"] == "success" and p["target"] is None


def test_other_tasks_markers_stay_out():
    env = gym.make("Hamsterball-v2", tasks=["reach_goal"], levels=["first_goal"])
    env.reset(seed=0)
    assert set(env.unwrapped.pose()["markers"]) == {"goal"}


def test_views_equal_the_mujoco_cameras_with_pitch_applied():
    env = make("reach_goal", ["first_goal"], camera_action=True)  # no views kwarg: the poses are still there
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.array([0, 0, 0.5, 1], np.float32))
    e = env.unwrapped
    p = e.pose()
    m, d = e.core.model, e.core.data
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    assert set(p["views"]) == {"chase", "follow", "first_person"}
    for v, view in p["views"].items():
        c = m.camera(v)
        np.testing.assert_allclose(view["pos"], d.cam_xpos[c.id], atol=1e-9)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, d.cam_xmat[c.id])
        assert abs(np.dot(view["quat"], q)) == pytest.approx(1.0, abs=1e-9)  # q and -q are one rotation
        assert view["fovy"] == pytest.approx(m.cam_fovy[c.id])
    assert p["facing"] == pytest.approx(d.userdata[0]) and p["pitch"] == pytest.approx(d.userdata[1])
    assert p["pitch"] > math.radians(15)
