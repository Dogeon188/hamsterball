"""The procedural level generator (spec §8) and the env's generator pool and `difficulty` (spec §10)."""

import numpy as np
import pytest

from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from hamsterball.env import HamsterballEnv

GEN = ProceduralGenerator()


def test_a_generated_level_carries_its_provenance_and_the_one_task():
    level = GEN("reach_goal", 7, 0.5)
    assert level.generator == {"name": "procedural", "version": GEN.version, "task": "reach_goal", "seed": 7,
                               "difficulty": 0.5}
    assert list(level.tasks) == ["reach_goal"]
    assert Level.from_yaml(level.canonical()).id == level.id


def test_the_hamsterball_rests_at_a_generated_spawn():
    env = HamsterballEnv(["reach_goal"], levels=[GEN("reach_goal", 3, 0.5)])
    env.reset(seed=0)
    spawn = np.array(env.level.tasks["reach_goal"]["spawn"]["pos"])
    for _ in range(30):  # half a second of no input
        info = env.step(np.zeros(2, np.float32))[4]
    assert info["failure_reason"] is None
    assert np.linalg.norm(info["state"][:3] - spawn) < 0.3


def toward_target(obs):
    """test_tasks.py's scripted controller: steer the shell's velocity toward the current target at up to 3 m/s."""
    g, v = obs["goal"][:2], obs["state"][6:8]
    a = g / np.linalg.norm(g) * min(3.0, np.linalg.norm(g)) - v
    return (a / max(1.0, np.linalg.norm(a))).astype(np.float32)


@pytest.mark.parametrize("task, seed", [("reach_goal", 0), ("waypoint_race", 1), ("unlock_goal", 2)])
def test_a_scripted_controller_finishes_an_easy_generated_level_within_its_time_limit(task, seed):
    env = HamsterballEnv([task], levels=[GEN(task, seed, 0.2)])
    obs, _ = env.reset(seed=0)
    while True:
        obs, _, terminated, truncated, info = env.step(toward_target(obs))
        if terminated or truncated:
            break
    assert info["success"]


def test_the_hamsterball_survives_an_easy_plateau_with_no_input():
    env = HamsterballEnv(["survive"], levels=[GEN("survive", 0, 0.0)], time_limit_s=3)
    env.reset(seed=0)
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
        if terminated or truncated:
            break
    assert info["success"]


def test_every_table_row_interpolates_linearly():
    p = GEN.params(0.25)
    assert p["route_length"] == 13.75 and p["corridor_width"] == 9.75 and p["time_slack"] == 2.625
    assert p["climb_grade"] == 7.5 and p["descent_grade"] == 11.25 and p["survive_duration"] == 22.5
    assert GEN.params(0.5)["waypoints"] == 6  # 5.5 rounds up to a whole marker
    assert [GEN.params(d)["cliff_height"] for d in (0, 0.49, 0.5, 0.75, 1)] == [0, 0, 0, 1.5, 3]


@pytest.mark.parametrize("task, key, d0, d1", [
    ("waypoint_race", "waypoints", 3, 8), ("collect_items", "items", 2, 8), ("unlock_goal", "keys", 1, 4),
])
def test_marker_counts_follow_difficulty(task, key, d0, d1):
    assert len(GEN(task, 0, 0.0).tasks[task][key]) == d0
    assert len(GEN(task, 0, 1.0).tasks[task][key]) == d1


def test_each_table_row_is_a_constructor_override():
    gen = ProceduralGenerator(waypoints=(5, 5), survive_duration=(12, 12))
    assert len(gen("waypoint_race", 0, 0.0).tasks["waypoint_race"]["waypoints"]) == 5
    assert len(gen("waypoint_race", 0, 1.0).tasks["waypoint_race"]["waypoints"]) == 5
    assert gen("survive", 0, 0.9).tasks["survive"]["time_limit_s"] == 12
    with pytest.raises(TypeError):
        ProceduralGenerator(gravity=(1, 2))


def heights(level):
    """The hfield's node heights as MuJoCo sees them, NaN below the kill plane."""
    import base64

    g = level.geoms[0]
    q = np.frombuffer(base64.b64decode(g["samples"]), "<u2").reshape(g["nrow"], g["ncol"])
    h = g["pos"][2] + q / 65535 * g["size"][2]
    return np.where(h > level.kill_plane_z, h, np.nan)


def biggest_drop(level):
    """The largest height step between two neighbouring nodes above the kill plane."""
    h = heights(level)
    return max(np.nanmax(np.abs(np.diff(h, axis=a))) for a in (0, 1))


def test_cliffs_appear_from_difficulty_one_half():
    for task in ("reach_goal", "waypoint_race"):
        assert all(biggest_drop(GEN(task, s, d)) < 0.5 for s in range(5) for d in (0.0, 0.25, 0.45))
        assert any(biggest_drop(GEN(task, s, 1.0)) > 1.0 for s in range(5))


def test_a_level_with_no_solvable_layout_raises_after_32_attempts():
    from hamsterball.core.generator import GenerationError

    gen = ProceduralGenerator(climb_grade=(60, 60))  # every climb steeper than the flood fill's 15 degrees
    with pytest.raises(GenerationError):
        gen("reach_goal", 0, 1.0)
    assert gen.generate("survive", 0, 1.0)[1] == 1  # survive builds no route


def test_the_time_limit_covers_the_route_at_5_m_s_times_the_slack():
    for task in ("reach_goal", "waypoint_race", "collect_items", "unlock_goal"):
        for d in (0.0, 0.5, 1.0):
            level = GEN(task, 4, d)
            block = level.tasks[task]
            route = [block["spawn"]["pos"]] + [m["pos"] for k in ("keys", "waypoints", "items", "goal") if k in block
                                               for m in (block[k] if isinstance(block[k], list) else [block[k]])]
            straight = sum(np.linalg.norm(np.subtract(b, a)[:2]) for a, b in zip(route, route[1:]))
            assert block["time_limit_s"] >= straight / 5 * GEN.params(d)["time_slack"] - 0.01


def test_survive_picks_a_plateau_disc_or_a_ridge_by_seed():
    p = GEN.params(0.0)
    shapes = []
    for s in range(20):
        block = GEN("survive", s, 0.0).tasks["survive"]
        region, spawn = block["region"], block["spawn"]["pos"]
        f = region["fromto"]
        ridge = f[:2] != f[3:5]
        shapes.append(ridge)
        assert region["radius"] == (p["survive_ridge_width"] / 2 if ridge else p["survive_disc_radius"])
        assert spawn[:2] == [(f[0] + f[3]) / 2, (f[1] + f[4]) / 2]  # the region's centre
        assert block["time_limit_s"] == p["survive_duration"]
    assert 5 <= sum(shapes) <= 15


def test_hazards_appear_with_difficulty():
    assert GEN("reach_goal", 0, 0.0).hazards == []
    assert len(GEN("reach_goal", 0, 1.0).hazards) == 5  # 1 per 8 m of a 40 m route


# sha256 over the space-joined level ids of the sweep, per task. A change here is a change to the generator's
# output: bump ProceduralGenerator.version and update these (spec §8 Determinism).
GOLDEN = {
    "reach_goal": "4b9612a079b89fc4a8332e032d1945f10dad7933986d3ca6482948d731f7d95f",
    "waypoint_race": "e41b48e03e3e7219d282c6be5add30885004aa836a19acfd4ee586b8ccd1745d",
    "collect_items": "d02974cabca37357e20062a3b0e184f48e20d9d35dce4006eb37644ee3bc3e67",
    "unlock_goal": "d38e82312238c601e82a896ef9910cf649067440015262b6d5a32e57d57ce1b4",
    "survive": "f941cf8e966ed34f2f56a5e20b6deddf5a6175e3057e853f61321cdea15ef029",
}


def test_the_seed_difficulty_sweep_matches_its_golden_hashes(capsys):
    import hashlib

    attempts = levels = 0
    got = {}
    for task in GOLDEN:
        ids = []
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            for seed in range(8):
                level, n = GEN.generate(task, seed, d)
                ids.append(level.id)
                attempts, levels = attempts + n, levels + 1
        got[task] = hashlib.sha256(" ".join(ids).encode()).hexdigest()
    rate = (attempts - levels) / attempts
    with capsys.disabled():
        print(f"\ngenerator resample failure rate: {rate:.3f} ({attempts - levels} of {attempts} attempts)")
    assert got == GOLDEN
    assert rate < 0.25
    assert GEN("waypoint_race", 3, 0.5).id == GEN("waypoint_race", 3, 0.5).id
