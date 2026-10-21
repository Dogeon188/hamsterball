"""The procedural level generator (spec §8) and the env's generator pool and `difficulty` (spec §10)."""

import numpy as np
import pytest

from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from hamsterball.core.marker import centre
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
    assert gen.generate("survive", 0, 0.0)[1] == 1  # survive builds no route, and nothing moves at d = 0


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
        assert spawn == list(centre(region))  # the region's centre, at shell-centre height (spec §9)
        assert block["time_limit_s"] == p["survive_duration"]
    assert 5 <= sum(shapes) <= 15


def survive_levels(d, seeds=range(8)):
    return [GEN("survive", s, d) for s in seeds]


def test_easy_survive_levels_hold_still():
    for level in survive_levels(0.0):
        assert "motion" not in level.tasks["survive"]["region"] and level.hazards == []


def test_the_region_drifts_faster_with_difficulty_and_its_centre_stays_on_the_flat_top():
    from hamsterball.core.motion import pose

    for d in (0.25, 0.5, 1.0):
        for level in survive_levels(d):
            region = level.tasks["survive"]["region"]
            m = region["motion"]
            assert m["type"] == "oscillate"
            peak = m["amplitude"] * 2 * np.pi / m["period"]
            assert abs(peak - GEN.params(d)["survive_drift_speed"]) < 0.01
            h, g = heights(level), level.geoms[0]
            for t in np.linspace(0, m["period"], 17):
                c = np.array(centre(region)) + pose(m, t)[0]
                col = round((c[0] - g["pos"][0] + g["size"][0]) / 0.25)
                row = round((g["pos"][1] + g["size"][1] - c[1]) / 0.25)
                assert abs(h[row, col]) < 1e-3  # the flat top sits at z = 0


@pytest.mark.parametrize("d, count", [(0.25, 1), (0.5, 2), (1.0, 3)])
def test_survive_hazards_move_more_and_faster_with_difficulty_and_start_clear_of_the_spawn(d, count):
    from hamsterball.core.marker import crosses

    for level in survive_levels(d):
        block = level.tasks["survive"]
        assert len(level.hazards) == count
        for hz in level.hazards:
            m = hz["motion"]
            assert abs(m["amplitude"] * 2 * np.pi / m["period"] - GEN.params(d)["hazard_speed"]) < 0.01
            spawn = block["spawn"]["pos"]
            assert not crosses(hz, spawn, spawn, 0.0)


@pytest.mark.parametrize("d", [0.5, 0.75, 1.0])
def test_survive_from_mid_difficulty_fails_with_no_input(d):
    for seed in range(4):
        env = HamsterballEnv(["survive"], levels=[GEN("survive", seed, d)])
        env.reset(seed=0)
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
            if terminated or truncated:
                break
        assert not info["success"], (seed, d)


def test_hazards_appear_with_difficulty():
    assert GEN("reach_goal", 0, 0.0).hazards == []
    assert len(GEN("reach_goal", 0, 1.0).hazards) == 5  # 1 per 8 m of a 40 m route


# sha256 over the space-joined level ids of the sweep, per task. A change here is a change to the generator's
# output: bump ProceduralGenerator.version and update these (spec §8 Determinism).
GOLDEN = {
    "reach_goal": "c9aea83e946be0d48f1a20c51b871310ad5a16129e5fadee081a29789df4337f",
    "waypoint_race": "86cbe3377440abbca2539b7aad2b1a1200854ce32996637da4564d5e620b631b",
    "collect_items": "6b5c27b24b355ba4c251d21a8d35b7e4c9f6cc5cba4f66f4fe1efda3e1d2297f",
    "unlock_goal": "670ec8bbe8553fe4918d328d96ee198ee885bd962d2b5789c0913ca2bce5cc08",
    "survive": "7a61d6e2572477aec5b5cce0d3709ba29382eb3b2853289d055b913aefd8a9ff",
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


# --- the env's generator pool and difficulty (spec §10) ---


def test_the_env_draws_generated_levels_and_reports_their_difficulty():
    env = HamsterballEnv(["waypoint_race", "survive"], levels=GEN, difficulty=0.3)
    seen = set()
    for s in range(6):
        _, info = env.reset(seed=s)
        opts = info["reset_options"]
        level = Level.from_yaml(opts["level"])
        assert opts["difficulty"] == 0.3 == level.generator["difficulty"]
        assert opts["task"] == level.generator["task"] in ("waypoint_race", "survive")
        seen.add(opts["task"])
    assert seen == {"waypoint_race", "survive"}
    _, a = env.reset(seed=11)
    _, b = env.reset(seed=11)
    assert a["reset_options"] == b["reset_options"]


def test_reset_options_override_the_difficulty_and_pick_the_task():
    env = HamsterballEnv(["reach_goal"], levels=GEN)
    _, info = env.reset(seed=0)
    assert info["reset_options"]["difficulty"] == 0.5  # the default
    _, info = env.reset(seed=0, options={"difficulty": 0.9, "task": "collect_items"})
    assert info["reset_options"]["difficulty"] == 0.9 and info["reset_options"]["task"] == "collect_items"


def test_a_generated_episode_replays_from_its_reset_options_on_a_file_pool_env():
    env = HamsterballEnv(["unlock_goal"], levels=GEN, difficulty=0.6)
    _, info = env.reset(seed=4)
    opts = info["reset_options"]
    other = HamsterballEnv(["reach_goal"], levels=["first_goal"])
    _, info2 = other.reset(options={"level": opts["level"], "task": opts["task"]})
    assert info2["reset_options"] == opts  # difficulty comes from the level's provenance


def test_a_non_default_difficulty_with_a_file_pool_raises():
    with pytest.raises(ValueError):
        HamsterballEnv(["reach_goal"], levels=["first_goal"], difficulty=0.7)
    env = HamsterballEnv(["reach_goal"], levels=["first_goal"])
    with pytest.raises(ValueError):
        env.set_levels(["first_goal"], difficulty=0.7)
    with pytest.raises(ValueError):
        env.reset(options={"difficulty": 0.7})
    _, info = env.reset(seed=0)
    assert "difficulty" not in info["reset_options"]


def test_set_levels_swaps_in_a_generator_with_a_difficulty():
    env = HamsterballEnv(["reach_goal"], levels=["first_goal"])
    env.set_levels(GEN, difficulty=0.2)
    _, info = env.reset(seed=0)
    assert info["reset_options"]["difficulty"] == 0.2
    env.set_levels(["first_goal"])
    _, info = env.reset(seed=0)
    assert "difficulty" not in info["reset_options"]


def test_levels_accepts_any_object_that_satisfies_the_protocol():
    class Flat:
        tasks = ("reach_goal",)
        version = 3

        def __call__(self, task, seed, difficulty):
            return GEN(task, seed % 4, difficulty)

    env = HamsterballEnv(["reach_goal", "survive"], levels=Flat(), difficulty=0.1)
    _, info = env.reset(seed=0)
    assert info["reset_options"]["task"] == "reach_goal" and info["reset_options"]["difficulty"] == 0.1
    with pytest.raises(ValueError):
        HamsterballEnv(["survive"], levels=Flat())  # no task the generator supports



def test_a_generated_episode_records_and_replays(tmp_path, monkeypatch):
    pytest.importorskip("minari")
    import gymnasium as gym

    import hamsterball  # noqa: F401  registers the env ids
    from hamsterball.data import Collector, replay

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path))
    env = Collector(gym.make("HamsterballCollectItems-v0", levels=GEN, difficulty=0.7))
    env.action_space.seed(0)
    env.reset(seed=3)
    for _ in range(20):
        env.step(env.action_space.sample())
    ds = env.create_dataset("test/generated-v0", algorithm_name="random")
    assert next(iter(ds.storage.get_episode_metadata([0])))["options"]["difficulty"] == 0.7
    replay(ds)
