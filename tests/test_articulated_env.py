"""The articulated tier through the env (spec: articulated-tier, ticket 03)."""


import gymnasium as gym
import numpy as np
import pytest

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.articulated_tier import STAND
from hamsterball.core.generator import ProceduralGenerator
from hamsterball.env import TASKS, HamsterballEnv


def test_every_task_constructs_and_steps_on_the_default_pool_with_the_camera_flag_on_and_off():
    for camera in (False, True):
        env = HamsterballEnv(tier="articulated", camera_action=camera)
        assert env.action_space.shape == (10 if camera else 8,)
        assert env.observation_space["state"].shape == (21,)
        assert env.observation_space["proprio"].shape == (26,)
        env = HamsterballEnv(tier="articulated", camera_action=camera, tasks=list(TASKS))
        env.action_space.seed(0)
        seen = set()
        for seed in range(25):
            obs, info = env.reset(seed=seed)
            seen.add(env.task)
            for _ in range(3):
                obs, *_ = env.step(env.action_space.sample())
                assert env.observation_space.contains(obs)
        assert seen == set(TASKS)


def test_a_generated_level_steps_with_the_articulated_tier():
    env = HamsterballEnv(tier="articulated", levels=ProceduralGenerator(), tasks=list(TASKS))
    assert env.action_space.shape == (8,)
    env.reset(seed=1)
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)


def test_the_abstract_tier_has_no_proprio():
    env = HamsterballEnv(tasks=["reach_goal"], levels=["first_goal"])
    assert set(env.observation_space.spaces) == {"state", "goal"}
    assert "proprio" not in env.reset(seed=0)[0]


def test_proprio_at_reset_is_the_standing_pose_in_the_facing_frame():
    env = HamsterballEnv(tier="articulated", tasks=["reach_goal"], levels=["first_goal"])
    p = env.reset(seed=0)[0]["proprio"]
    assert p.shape == (26,) and p.dtype == np.float32
    np.testing.assert_allclose(p[:8], STAND, atol=1e-4)  # joint positions
    np.testing.assert_allclose(p[8:16], 0, atol=1e-6)  # joint velocities
    np.testing.assert_allclose(p[20:23], [0, 1, 0], atol=1e-6)  # torso forward is the facing (+y at facing 0)
    np.testing.assert_allclose(p[23:26], [0, 0, 1], atol=1e-6)  # level


def test_foot_contact_flags_follow_the_feet():
    env = HamsterballEnv(tier="articulated", tasks=["reach_goal"], levels=["first_goal"])
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(8, np.float32))
    assert obs["proprio"][16:20].sum() >= 2  # standing on the wall
    assert set(obs["proprio"][16:20]) <= {0.0, 1.0}


def test_a_live_observation_equals_the_one_recomputed_from_the_stored_core_state():
    env = HamsterballEnv(tier="articulated", tasks=["reach_goal"], levels=["first_goal"])
    env.action_space.seed(3)
    env.reset(seed=0)
    for _ in range(40):
        obs, _, _, _, info = env.step(env.action_space.sample())
        again = env.step_to(info["core_state"])[0]
        for k in ("state", "proprio"):
            np.testing.assert_array_equal(obs[k], again[k])


def test_the_wall_contact_flag_is_a_hamster_geom_on_an_inner_tile():
    env = HamsterballEnv(tier="articulated", tasks=["reach_goal"], levels=["first_goal"])
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(8, np.float32))
    assert obs["state"][20] == 1


@pytest.mark.gl
def test_each_view_renders_and_the_first_person_camera_sits_at_the_eye():
    env = HamsterballEnv(tier="articulated", tasks=["reach_goal"], levels=["first_goal"],
                         views=("chase", "follow", "first_person"), view_size=(32, 32))
    obs, _ = env.reset(seed=0)
    for v in ("chase", "follow", "first_person"):
        assert obs[v].shape == (32, 32, 3) and obs[v].std() > 0
    m, d = env.core.model, env.core.data
    np.testing.assert_allclose(d.mocap_pos[m.body("view_first_person").mocapid[0]], d.site("hamster_eye").xpos)


def test_make_vec_runs_the_articulated_tier_on_cpu():
    venv = gym.make_vec("HamsterballReachGoal-v2", num_envs=2, levels=["first_goal"], tier="articulated")
    try:
        obs, _ = venv.reset(seed=0)
        assert obs["proprio"].shape == (2, 26)
        obs, *_ = venv.step(venv.action_space.sample())
        assert obs["proprio"].shape == (2, 26) and venv.action_space.shape == (2, 8)
    finally:
        venv.close()
