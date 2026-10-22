"""Moving hazards (survive-movers ticket 02): any hazard can take a motion, tested on the relative swept path."""

import gymnasium as gym
import numpy as np
import pytest

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level
from hamsterball.core.marker import centre, crosses
from hamsterball.core.motion import load
from hamsterball.env import LEVELS_DIR

STILL = np.zeros(2, np.float32)
SWEEP = {"type": "path", "points": [[-1, 0, 0], [1, 0, 0]], "period": 4}  # 1 m/s: -1 at t = 0, +1 at t = 2


def hazard(shape, motion=None):
    return {**shape, "motion": load(motion)} if motion else shape


# --- the relative swept test


def test_a_hazard_sweeping_through_a_still_shell_is_caught_within_one_step():
    h = hazard({"type": "sphere", "pos": [0, 0, 0], "radius": 0.3}, SWEEP)
    p = np.zeros(3)
    assert not crosses(h, p, p, t=0) and not crosses(h, p, p, t=2)  # either end alone misses
    assert crosses(h, p, p, t=2, t0=0)


def test_a_shell_riding_alongside_a_hazard_is_not_caught():
    h = hazard({"type": "sphere", "pos": [0, 0, 0], "radius": 0.3}, SWEEP)
    p0, p1 = np.array([-1.5, 0, 0]), np.array([0.5, 0, 0])  # 0.5 m behind the hazard the whole step
    assert crosses(h, p0, p1, t=0)  # holding the hazard still at t0 would flag it
    assert not crosses(h, p0, p1, t=2, t0=0)


def test_a_shell_and_hazard_passing_head_on_are_caught():
    h = hazard({"type": "sphere", "pos": [0, 0, 0], "radius": 0.3}, SWEEP)  # -1 -> +1
    p0, p1 = np.array([0.5, 0, 0]), np.array([-0.5, 0, 0])
    assert not crosses(h, p0, p0, t=0) and not crosses(h, p1, p1, t=2)
    assert crosses(h, p0, p1, t=2, t0=0)


def test_a_spinning_bar_sweeping_past_a_still_shell_is_caught():
    bar = hazard({"type": "capsule", "fromto": [0, 0, 0, 2, 0, 0], "radius": 0.1},
                 {"type": "oscillate", "axis": [1, 0, 0], "amplitude": 0, "period": 1, "spin": 90})
    p = np.array([1, 0.8, 0])  # 0.8 m off the bar's centre; the bar passes over it at 90 deg
    assert not crosses(bar, p, p, t=0) and not crosses(bar, p, p, t=2)
    assert crosses(bar, p, p, t=2, t0=0)


# --- the loader


def goal_level(*hazards):
    return Level.from_dict({
        "kill_plane_z": -5,
        "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}],
        "tasks": {"reach_goal": {"spawn": {"pos": [0, 0, 0.5]}, "goal": {"type": "sphere", "pos": [0, 8, 0.5], "radius": 0.5},
                                 "time_limit_s": 30}},
        "hazards": list(hazards),
    })


def test_a_hazard_motion_loads_with_its_defaults_filled():
    h = {"type": "sphere", "pos": [3, 0, 0.5], "radius": 0.4}
    lv = goal_level({**h, "motion": SWEEP})
    assert lv.hazards[0]["motion"] == load(SWEEP)
    assert goal_level({**h, "motion": dict(SWEEP, phase=1, spin=0)}).id == lv.id
    assert "motion" not in goal_level(h).hazards[0]
    with pytest.raises(ValueError):
        goal_level({**h, "motion": {"type": "path"}})


def test_existing_hazard_levels_keep_their_id():
    h = {"type": "sphere", "pos": [3, 0, 0.5], "radius": 0.4}
    assert goal_level(h).id == goal_level(dict(h)).id and "motion" not in goal_level(h).canonical()


# --- through the env


def test_a_moving_hazard_fails_reach_goal_and_is_drawn_where_it_is():
    # sweeps along x through the spawn; at t = 0 it is 3 m away, so the shell sits still until it arrives
    lv = goal_level({"type": "sphere", "pos": [0, 0, 0.5], "radius": 0.4,
                     "motion": {"type": "oscillate", "axis": [1, 0, 0], "amplitude": 3, "period": 4, "phase": 0.25}})
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=[lv])
    e = env.unwrapped
    env.reset(seed=0)
    g = e.core.model.geom("hazard0").id
    np.testing.assert_allclose(e.core.data.geom_xpos[g], [3, 0, 0.5], atol=1e-9)
    for n in range(1, 200):
        _, reward, terminated, _, info = env.step(STILL)
        np.testing.assert_allclose(e.core.data.geom_xpos[g], centre(lv.hazards[0], e.core.data.time), atol=1e-9)
        if terminated:
            break
    assert info["failure_reason"] == "hazard"
    assert 40 < n < 60  # it reaches the 0.4 m radius at about 0.87 s


def test_a_fast_hazard_cannot_skip_over_the_shell_between_env_steps():
    # 0.1 m radius, 30 m/s along x: 0.5 m per env step, five times its diameter
    lv = goal_level({"type": "sphere", "pos": [0, 0, 0.5], "radius": 0.1,
                     "motion": {"type": "path", "points": [[-15, 0, 0], [15, 0, 0]], "period": 2, "phase": 0.1}})
    env = gym.make("Hamsterball-v0", tasks=["reach_goal"], levels=[lv])
    env.reset(seed=0)
    for _ in range(60):
        *_, terminated, _, info = env.step(STILL)
        if terminated:
            break
    assert info["failure_reason"] == "hazard"


# --- survive_sweepers


def sweepers_env(**kw):
    return gym.make("Hamsterball-v0", tasks=["survive"], levels=["survive_sweepers"], **kw)


def dodge(obs):
    """Roll to the corner at (2.5, 2.5), clear of both sweepers' lines and the spinning bar, and hold there."""
    g, v = np.array([2.5, 2.5]) - obs["state"][:2], obs["state"][6:8]
    a = g / max(np.linalg.norm(g), 1e-9) * min(3.0, np.linalg.norm(g)) - v
    return (a / max(1.0, np.linalg.norm(a))).astype(np.float32)


def run(env, policy):
    obs, _ = env.reset(seed=0)
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            return info


def test_an_inputless_hamsterball_is_hit_by_a_sweeper():
    assert run(sweepers_env(), lambda obs: STILL)["failure_reason"] == "hazard"


def test_dodging_to_a_clear_corner_survives():
    info = run(sweepers_env(), dodge)
    assert info["success"], info["failure_reason"]
