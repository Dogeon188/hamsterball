"""Episode setup (spec §10 Reset, §7 model cache) and the env ids (spec §2)."""

import gymnasium as gym
import numpy as np
import pytest

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level
from hamsterball.env import LEVELS_DIR, TASKS

ALL = list(TASKS)


def make(**kw):
    return gym.make("Hamsterball-v0", **{"tasks": ALL, **kw})


def drawn(env, seed):
    _, info = env.reset(seed=seed)
    return info["reset_options"]["level_id"], info["reset_options"]["task"]


def test_the_same_seed_draws_the_same_level_and_task():
    env = make()
    draws = [drawn(env, s) for s in range(20)]
    assert draws == [drawn(env, s) for s in range(20)]
    assert len({t for _, t in draws}) > 1 and len({lv for lv, _ in draws}) > 1


def test_reset_options_report_the_canonical_level_and_task():
    env = make()
    _, info = env.reset(seed=3)
    opts = info["reset_options"]
    assert opts.keys() == {"level", "level_id", "task"}
    lv = Level.from_yaml(opts["level"])
    assert lv.id == opts["level_id"] and opts["task"] in lv.tasks


def test_an_episode_replays_from_its_reset_options_alone():
    env = make()
    env.reset(seed=5)
    actions = [np.array([0.3, 1.0], np.float32)] * 30
    _, info = env.reset(seed=7)
    opts = info["reset_options"]
    states = [env.step(a)[4]["core_state"] for a in actions]
    other = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=["first_goal"])  # different kwargs
    _, info2 = other.reset(options={"level": opts["level"], "task": opts["task"]})
    assert info2["reset_options"] == opts
    np.testing.assert_array_equal(states, [other.step(a)[4]["core_state"] for a in actions])


def test_level_and_task_options_are_used_as_given():
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=["first_goal"])
    lv = Level.load(LEVELS_DIR / "first_survive.yaml")
    _, info = env.reset(options={"level": lv, "task": "survive"})
    assert info["reset_options"] == {"level": lv.canonical(), "level_id": lv.id, "task": "survive"}


def test_a_task_option_draws_a_pool_level_that_supports_it():
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"])  # options override the task filter
    for s in range(5):
        _, info = env.reset(seed=s, options={"task": "collect_items"})
        assert info["reset_options"]["task"] == "collect_items"
        assert "collect_items" in Level.from_yaml(info["reset_options"]["level"]).tasks


def test_a_task_option_no_pool_level_supports_raises():
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=["first_goal"])
    with pytest.raises(ValueError):
        env.reset(options={"task": "survive"})


def two_task_level():
    """first_goal plus a survive block around its spawn."""
    goal = Level.load(LEVELS_DIR / "first_goal.yaml")
    survive = Level.load(LEVELS_DIR / "first_survive.yaml")
    goal.tasks["survive"] = survive.tasks["survive"]
    return goal


def test_a_level_option_takes_an_allowed_task_then_falls_back_to_any():
    lv = two_task_level()
    env = gym.make("Hamsterball-v0", tasks=["survive"], levels=["first_survive"])
    for s in range(5):
        assert env.reset(seed=s, options={"level": lv})[1]["reset_options"]["task"] == "survive"
    env = gym.make("Hamsterball-v0", tasks=["waypoint_race"], levels=["waypoint_loop"])
    tasks = {env.reset(seed=s, options={"level": lv})[1]["reset_options"]["task"] for s in range(20)}
    assert tasks == {"reach_goal", "survive"}


def test_a_level_that_lacks_the_task_raises():
    env = make()
    with pytest.raises(ValueError):
        env.reset(options={"level": Level.load(LEVELS_DIR / "first_goal.yaml"), "task": "survive"})


def test_no_options_draws_a_pool_level_then_an_allowed_task():
    env = gym.make("Hamsterball-v0", tasks=["survive", "reach_goal"], levels=[two_task_level(), "waypoint_loop"])
    seen = {drawn(env, s)[1] for s in range(20)}
    assert seen == {"survive", "reach_goal"}  # waypoint_loop has no allowed task, so it is never drawn


def test_construction_raises_when_no_pool_level_supports_an_allowed_task():
    with pytest.raises(ValueError):
        gym.make("Hamsterball-v0", tasks=["survive"], levels=["first_goal", "waypoint_loop"])


def test_levels_take_names_paths_directories_and_level_objects(tmp_path):
    (tmp_path / "a.yaml").write_text((LEVELS_DIR / "first_goal.yaml").read_text())
    obj = Level.load(LEVELS_DIR / "item_scatter.yaml")
    env = make(levels=["waypoint_loop", LEVELS_DIR / "locked_goal.yaml", tmp_path, obj])
    names = {Level.load(LEVELS_DIR / f"{n}.yaml").id: n for n in ("first_goal", "waypoint_loop", "locked_goal", "item_scatter")}
    assert {names[drawn(env, s)[0]] for s in range(30)} == set(names.values())


def test_set_levels_replaces_the_pool_from_the_next_reset():
    env = make(levels=["first_goal"])
    env.reset(seed=0)
    env.unwrapped.set_levels(["first_survive"])
    assert {drawn(env, s)[1] for s in range(5)} == {"survive"}
    with pytest.raises(ValueError):
        gym.make("Hamsterball-v0", tasks=["reach_goal"]).unwrapped.set_levels(["first_survive"])


def test_set_levels_works_through_a_vector_envs_call():
    venv = gym.vector.SyncVectorEnv([lambda: make(levels=["first_goal"])] * 2)
    venv.call("set_levels", ["first_survive"])
    _, info = venv.reset(seed=0)
    assert list(info["reset_options"]["task"]) == ["survive", "survive"]


def test_the_model_cache_is_an_lru_keyed_by_level_id(monkeypatch):
    names = ["first_goal", "waypoint_loop", "item_scatter"]
    env = make(model_cache=2).unwrapped
    lv = {n: Level.load(LEVELS_DIR / f"{n}.yaml") for n in names}
    reset = lambda n: env.reset(options={"level": lv[n]}) and env.core  # noqa: E731
    a, b = reset("first_goal"), reset("waypoint_loop")
    assert reset("first_goal") is a  # a hit; first_goal is now the most recent
    reset("item_scatter")  # evicts waypoint_loop, the least recent
    assert reset("first_goal") is a and reset("waypoint_loop") is not b
    resets = []
    real = __import__("mujoco").mj_resetData
    monkeypatch.setattr("mujoco.mj_resetData", lambda m, d: resets.append(m) or real(m, d))
    _, info = env.reset(options={"level": lv["first_goal"]})
    env.step(np.array([1, 1], np.float32))
    _, again = env.reset(options={"level": lv["first_goal"]})
    assert resets[-1] is a.model
    np.testing.assert_array_equal(again["core_state"], info["core_state"])


PRESETS = {
    "HamsterballReachGoal-v0": "reach_goal",
    "HamsterballWaypointRace-v0": "waypoint_race",
    "HamsterballCollectItems-v0": "collect_items",
    "HamsterballUnlockGoal-v0": "unlock_goal",
    "HamsterballSurvive-v0": "survive",
}


@pytest.mark.parametrize("env_id", PRESETS)
def test_each_preset_constructs_with_its_default_pool(env_id):
    env = gym.make(env_id)
    assert {drawn(env, s)[1] for s in range(5)} == {PRESETS[env_id]}


def test_the_general_id_takes_every_task():
    assert {drawn(make(), s)[1] for s in range(40)} == set(TASKS)


def test_an_older_id_raises_naming_the_release_to_install(monkeypatch):
    from gymnasium.envs import registration

    monkeypatch.setattr(registration, "registry", {})
    hamsterball.register_ids(2)  # as a release at N = 2 would
    assert "HamsterballSurvive-v2" in registration.registry
    with pytest.raises(gym.error.DeprecatedEnv, match=r"hamsterball==1\.\*"):
        gym.make("HamsterballSurvive-v1")
    with pytest.raises(gym.error.DeprecatedEnv, match=r"hamsterball==0\.\*"):
        gym.make("Hamsterball-v0", tasks=["reach_goal"])


def test_discrete_action_is_a_no_op_plus_8_directions_by_3_efforts():
    from hamsterball.env import DiscreteAction

    env = DiscreteAction(gym.make("HamsterballReachGoal-v0"))
    assert env.action_space == gym.spaces.Discrete(25)
    np.testing.assert_array_equal(env.action(0), [0, 0])
    np.testing.assert_allclose(env.action(1), [0, 0.25], atol=1e-7)  # forward, quarter effort
    np.testing.assert_allclose(env.action(3), [0.25, 0], atol=1e-7)  # right
    np.testing.assert_allclose(env.action(8 + 2), [0.5 / 2**0.5] * 2, atol=1e-7)  # forward-right, half
    np.testing.assert_allclose(env.action(16 + 5), [0, -1], atol=1e-7)  # back, full
    acts = np.array([env.action(i) for i in range(25)])
    assert len(np.unique(acts.round(6), axis=0)) == 25 and acts.dtype == np.float32
    env.reset(seed=0)
    env.step(17)
    cam = DiscreteAction(gym.make("HamsterballReachGoal-v0", camera_action=True))
    np.testing.assert_allclose(cam.action(1), [0, 0.25, 0], atol=1e-7)  # cam_yaw held at 0
