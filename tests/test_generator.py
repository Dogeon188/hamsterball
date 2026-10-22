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
    a = g / max(np.linalg.norm(g), 1e-9) * min(3.0, np.linalg.norm(g)) - v
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
    level = GEN("survive", 3, 0.0)
    assert layout(level) == "plateau"
    env = HamsterballEnv(["survive"], levels=[level], time_limit_s=3)
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


def layout(level):
    """Which of survive's six layouts a generated level is."""
    region = level.tasks["survive"]["region"]
    if level.collapse:
        return "collapse"
    if level.bodies:
        return "seesaw" if "joint" in level.bodies[0] else "platform" if "attach" in region else "pillars"
    f = region["fromto"]
    return "ridge" if f[:2] != f[3:5] else "plateau"


def test_survive_picks_one_of_six_layouts_by_seed():
    from collections import Counter

    counts = Counter(layout(GEN("survive", s, 0.0)) for s in range(120))
    assert set(counts) == {"plateau", "ridge", "pillars", "platform", "seesaw", "collapse"}
    assert all(10 <= n <= 30 for n in counts.values()), counts  # 1/6 each


def test_plateau_and_ridge_regions_follow_the_table():
    p = GEN.params(0.0)
    for s in range(40):
        level = GEN("survive", s, 0.0)
        if layout(level) not in ("plateau", "ridge"):
            continue
        block = level.tasks["survive"]
        region, spawn = block["region"], block["spawn"]["pos"]
        ridge = layout(level) == "ridge"
        assert region["radius"] == (p["survive_ridge_width"] / 2 if ridge else p["survive_disc_radius"])
        assert spawn == list(centre(region))  # the region's centre, at shell-centre height (spec §9)
        assert block["time_limit_s"] == p["survive_duration"]


def spiral_walk(obs, walk):
    """Collapse's scripted controller: roll at 2 m/s through the tile centres in a square spiral out from the spawn's
    tile (`walk` = [the spiral's points, the next one's index]), so each tile is left before it falls."""
    pts, i = walk
    p, (sy, cy) = obs["state"][:2], obs["state"][4:6]
    if np.linalg.norm(pts[i] - p) < 0.4:
        walk[1] = i = i + 1
    w = pts[i] - p
    g = np.array([w[0] * cy + w[1] * sy, -w[0] * sy + w[1] * cy])  # world -> camera yaw frame, as the goal vector
    a = g / max(np.linalg.norm(g), 1e-9) * 2.0 - obs["state"][6:8]
    return (a / max(1.0, np.linalg.norm(a))).astype(np.float32)


def spiral(n, pitch):
    out, x, y, dx, dy, leg = [(0, 0)], 0, 0, 1, 0, 1
    while len(out) < n:
        for _ in range(2):
            for _ in range(leg):
                x, y = x + dx, y + dy
                out.append((x, y))
            dx, dy = -dy, dx
        leg += 1
    return np.array(out[:n], float) * pitch


def test_every_survive_layout_at_difficulty_zero_is_survived_by_a_scripted_controller():
    from hamsterball.core.generator import TILE_PITCH

    seen = set()
    for seed in range(36):
        level = GEN("survive", seed, 0.0)
        seen.add(layout(level))
        walk = [spiral(100, TILE_PITCH), 0]
        env = HamsterballEnv(["survive"], levels=[level])
        obs, _ = env.reset(seed=0)
        while True:
            a = spiral_walk(obs, walk) if level.collapse else toward_target(obs)
            obs, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert info["success"], (seed, layout(level), info["failure_reason"])
    assert len(seen) == 6


def survive_levels(d, n=8):
    """The first n plateau or ridge levels by seed, the layouts that carry ticket 07's movers."""
    levels = (GEN("survive", s, d) for s in range(1000))
    return [lv for lv in levels if layout(lv) in ("plateau", "ridge")][:n]


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
    for seed in range(12):
        env = HamsterballEnv(["survive"], levels=[GEN("survive", seed, d)])
        env.reset(seed=0)
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
            if terminated or truncated:
                break
        assert not info["success"], (seed, d)


def first(kind, d):
    return next(lv for lv in (GEN("survive", s, d) for s in range(1000)) if layout(lv) == kind)


@pytest.mark.parametrize("d, pillars, speed, tilt, k, off, delay", [
    (0.0, 1, 0.5, None, 400, 0.25, 2.0), (1.0, 3, 2.0, 12, 50, 1.5, 0.5),
])
def test_the_body_layouts_scale_with_difficulty(d, pillars, speed, tilt, k, off, delay):
    peak = lambda m: m["amplitude"] * 2 * np.pi / m["period"]  # noqa: E731
    bodies = first("pillars", d).bodies
    assert len(bodies) == pillars and all(abs(peak(b["motion"]) - speed) < 0.01 for b in bodies)
    m = first("platform", d).bodies[0]["motion"]
    assert abs(peak(m) - GEN.params(d)["platform_speed"]) < 0.01
    assert (m["tilt"]["amplitude"] if "tilt" in m else None) == tilt
    seesaw = first("seesaw", d)
    assert seesaw.bodies[0]["joint"]["stiffness"] == k
    assert abs(np.linalg.norm(seesaw.tasks["survive"]["spawn"]["pos"][:2]) - off) < 0.01
    collapse = first("collapse", d).collapse
    assert collapse["delay"] == delay and len(collapse["tiles"]) < 100  # tickets 06: under about 100 tiles


def test_a_collapse_time_limit_never_outlasts_its_tiles():
    for d in (0.0, 0.5, 1.0):
        level = first("collapse", d)
        c = level.collapse
        assert level.tasks["survive"]["time_limit_s"] <= 0.5 * len(c["tiles"]) * c["delay"] + 1e-9


def test_hazards_appear_with_difficulty():
    assert GEN("reach_goal", 0, 0.0).hazards == []
    assert len(GEN("reach_goal", 0, 1.0).hazards) == 5  # 1 per 8 m of a 40 m route


# sha256 over the space-joined level ids of the sweep, per task. A change here is a change to the generator's
# output: bump ProceduralGenerator.version and update these (spec §8 Determinism).
GOLDEN = {
    "reach_goal": "1ab838ff7faf8f1137c185493ae1033d625004640ce34739e4af88e61345ab59",
    "waypoint_race": "559b001ff63d82bb59597ea0040339bacd63d0bb83bc39f6d7575e89814529b8",
    "collect_items": "1ecd136f0503cb7a6bf173bc61aaaa769d9e4147c1d87742d3f56f4d078fca1d",
    "unlock_goal": "b4c6c6028116478b3c10de6b9f0e9dc85b769a3f57a7e7caad6a0401cf8514f4",
    "survive": "f130e29280f328d8922da9a0cec9730d75abdf5ce17d1fa02687957cb1e95833",
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
