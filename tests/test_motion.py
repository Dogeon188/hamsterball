"""Motion (survive-movers ticket 01): a mover's pose as a pure function of its motion and sim time."""

import math

import gymnasium as gym
import numpy as np
import pytest

import hamsterball  # noqa: F401  registers the env ids
from hamsterball.core.level import Level
from hamsterball.core.marker import centre, crosses
from hamsterball.core.motion import load, pose
from hamsterball.env import LEVELS_DIR

SQUARE = load({"type": "path", "points": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], "period": 16})


@pytest.mark.parametrize(
    "t, offset",
    [(0, [0, 0, 0]), (2, [2, 0, 0]), (4, [4, 0, 0]), (6, [4, 2, 0]), (14, [0, 2, 0]), (16, [0, 0, 0]), (18, [2, 0, 0])],
)
def test_a_path_moves_at_constant_speed_round_its_closed_polyline(t, offset):
    np.testing.assert_allclose(pose(SQUARE, t)[0], offset, atol=1e-12)
    assert pose(SQUARE, t)[1] == 0.0


def test_a_paths_phase_is_a_fraction_of_its_period():
    m = load({"type": "path", "points": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], "period": 16, "phase": 0.25})
    np.testing.assert_allclose(pose(m, 0)[0], [4, 0, 0], atol=1e-12)


def test_oscillate_is_a_sine_offset_along_its_axis():
    m = load({"type": "oscillate", "axis": [0, 2, 0], "amplitude": 3, "period": 4})
    for t, y in [(0, 0), (1, 3), (2, 0), (3, -3)]:
        np.testing.assert_allclose(pose(m, t)[0], [0, y, 0], atol=1e-12)
    m = load({"type": "oscillate", "axis": [1, 0, 0], "amplitude": 3, "period": 4, "phase": 0.25})
    np.testing.assert_allclose(pose(m, 0)[0], [3, 0, 0], atol=1e-12)


def test_spin_turns_the_mover_about_z_at_a_constant_rate():
    m = load({"type": "oscillate", "axis": [1, 0, 0], "amplitude": 0, "period": 1, "spin": 30})
    assert pose(m, 3)[1] == pytest.approx(math.radians(90))


@pytest.mark.parametrize(
    "motion",
    [
        {"type": "path", "points": [[0, 0, 0], [1, 0, 0]], "period": 2.0, "phase": 0.0, "spin": 0.0},
        {"type": "path", "points": [[0, 0, 0], [1, 0, 0]], "period": 2, "phase": 1},
        {"type": "path", "points": [[0, 0, 0], [1, 0, 0]], "period": 2},
    ],
)
def test_equivalent_spellings_load_the_same(motion):
    assert load(motion) == load({"type": "path", "points": [[0, 0, 0], [1, 0, 0]], "period": 2})


@pytest.mark.parametrize(
    "motion",
    [
        {"type": "spline", "points": [[0, 0, 0], [1, 0, 0]], "period": 2},
        {"type": "path", "points": [[0, 0, 0]], "period": 2},
        {"type": "path", "points": [[1, 0, 0], [1, 0, 0]], "period": 2},
        {"type": "path", "points": [[0, 0], [1, 0]], "period": 2},
        {"type": "path", "points": [[0, 0, 0], [1, 0, 0]], "period": 0},
        {"type": "path", "points": [[0, 0, 0], [1, 0, 0]]},
        {"type": "path", "points": [[0, 0, 0], [1, 0, 0]], "period": 2, "speed": 1},
        {"type": "oscillate", "axis": [0, 0, 0], "amplitude": 1, "period": 2},
        {"type": "oscillate", "axis": [1, 0, 0], "amplitude": 1, "period": -2},
        {"type": "oscillate", "axis": [1, 0, 0], "period": 2},
        [1, 2, 3],
    ],
)
def test_a_malformed_motion_raises(motion):
    with pytest.raises(ValueError):
        load(motion)


# --- the survive region's motion, through the level loader and the marker test


DRIFT = {"type": "path", "points": [[0, 0, 0], [4, 0, 0]], "period": 8}


def survive(region, spawn=(0, 0, 0.5)):
    return {
        "kill_plane_z": -5,
        "geoms": [{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}],
        "tasks": {"survive": {"spawn": {"pos": list(spawn)}, "region": region, "time_limit_s": 10}},
    }


def test_a_region_motion_loads_with_its_defaults_filled():
    lv = Level.from_dict(survive({"type": "sphere", "pos": [0, 0, 0.5], "radius": 1, "motion": DRIFT}))
    assert lv.tasks["survive"]["region"]["motion"] == load(DRIFT)
    spelt = dict(DRIFT, phase=0, spin=0.0)
    assert Level.from_dict(survive({"type": "sphere", "pos": [0, 0, 0.5], "radius": 1, "motion": spelt})).id == lv.id


def test_a_level_without_motion_has_no_motion_key():
    lv = Level.load(LEVELS_DIR / "first_survive.yaml")
    assert "motion" not in lv.canonical()


def test_a_malformed_region_motion_raises():
    with pytest.raises(ValueError):
        Level.from_dict(survive({"type": "sphere", "pos": [0, 0, 0.5], "radius": 1, "motion": {"type": "path"}}))


def test_the_spawn_check_uses_the_regions_pose_at_time_0():
    away = dict(DRIFT, phase=0.25)  # starts 2 m along +x
    with pytest.raises(ValueError, match="outside its region"):
        Level.from_dict(survive({"type": "sphere", "pos": [0, 0, 0.5], "radius": 1, "motion": away}))
    Level.from_dict(survive({"type": "sphere", "pos": [0, 0, 0.5], "radius": 1, "motion": away}, spawn=(2, 0, 0.5)))


def test_the_marker_test_and_centre_follow_the_motion():
    region = Level.from_dict(survive({"type": "sphere", "pos": [0, 0, 0.5], "radius": 1, "motion": DRIFT}))
    region = region.tasks["survive"]["region"]
    p = np.array([2, 0, 0.5])
    assert not crosses(region, p, p) and crosses(region, p, p, t=2)  # at t = 2 s the region is 2 m along +x
    np.testing.assert_allclose(centre(region, t=2), [2, 0, 0.5])


def test_a_spinning_box_region_turns_about_its_centre():
    box = {"type": "box", "pos": [1, 0, 0.5], "size": [2, 0.5, 1], "motion": {"type": "oscillate", "axis": [1, 0, 0],
           "amplitude": 0, "period": 1, "spin": 90}}
    region = Level.from_dict(survive(box, spawn=(1, 0, 0.5))).tasks["survive"]["region"]
    p = np.array([1, 1.5, 0.5])  # outside the box's 0.5 m half-width in y, inside its 2 m half-length
    assert not crosses(region, p, p) and crosses(region, p, p, t=1)  # a quarter turn at 90 deg/s


# --- survive_drift through the env


def drift_env(**kw):
    return gym.make("Hamsterball-v0", tasks=["survive"], levels=["survive_drift"], **kw)


def track(obs):
    """Steer the shell's velocity toward the region centre, in the camera yaw frame (tests/test_tasks.py's)."""
    g, v = obs["goal"][:2], obs["state"][6:8]
    want = g / max(np.linalg.norm(g), 1e-9) * min(3.0, np.linalg.norm(g))
    a = want - v
    return (a / max(1.0, np.linalg.norm(a))).astype(np.float32)


def run(env, policy):
    obs, _ = env.reset(seed=0)
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            return info


def test_an_inputless_hamsterball_drifts_out_of_the_region():
    info = run(drift_env(), lambda obs: np.zeros(2, np.float32))
    assert info["failure_reason"] == "left_region"


def test_tracking_the_region_centre_survives():
    info = run(drift_env(), track)
    assert info["success"], info["failure_reason"]


def test_the_goal_vector_and_drawn_region_follow_the_regions_pose_now():
    env = drift_env()
    e = env.unwrapped
    env.reset(seed=0)
    for _ in range(60):  # 1 s
        obs, *_ = env.step(np.zeros(2, np.float32))
    region = e.level.tasks["survive"]["region"]
    want = centre(region, t=1.0)
    np.testing.assert_allclose(e.core.data.time, 1.0)
    np.testing.assert_allclose(obs["goal"][:3], want - obs["state"][:3], atol=1e-5)  # camera yaw is still 0
    g = e.core.model.geom("survive/region").id
    np.testing.assert_allclose(e.core.data.geom_xpos[g], want, atol=1e-9)
    yaw = np.arctan2(e.core.data.geom_xmat[g][3], e.core.data.geom_xmat[g][0])
    np.testing.assert_allclose(yaw, np.radians(18), atol=1e-9)
    env.reset(seed=0)  # the cached model's region goes back to its pose at time 0
    np.testing.assert_allclose(e.core.data.geom_xpos[g], centre(region), atol=1e-9)


def test_shaping_reads_the_regions_distance_now():
    env = drift_env(shaped=True, shaping_gamma=1.0)
    _, start = env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, np.float32))
    e = env.unwrapped
    d0 = np.linalg.norm(centre(e.level.tasks["survive"]["region"], t=0) - start["state"][:3])
    d1 = np.linalg.norm(centre(e.level.tasks["survive"]["region"], t=e.core.data.time) - info["state"][:3])
    assert info["reward_terms"]["shaping"] == pytest.approx(-0.01 * (d1 - d0), abs=1e-6)
