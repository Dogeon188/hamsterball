"""All five tasks on the Warp vector env against the CPU env (GPU effort, seam 2): recorded CPU episodes of each task,
replayed through `step_to`, on static levels."""

import numpy as np
import pytest
from test_warp_env import CODES, DEVICE, OBS

from hamsterball.core.level import TASK_MARKERS, Level
from hamsterball.core.model import SHELL_QPOS
from hamsterball.env import LEVELS_DIR, TASKS, HamsterballEnv

pytestmark = pytest.mark.gpu

PLATFORM = {"type": "box", "pos": [0, 0, -0.5], "size": [20, 20, 0.5]}


def sph(x, y, r=1.0):
    return {"type": "sphere", "pos": [x, y, 0.5], "radius": r}


def make(task, geoms=(PLATFORM,), hazards=(), **block):
    return Level.from_dict({
        "kill_plane_z": -5, "geoms": list(geoms), "hazards": list(hazards),
        "tasks": {task: {"spawn": {"pos": [0, 0, 0.5], "yaw_deg": 0}, "time_limit_s": 20, **block}}})


def bundled(name):
    return Level.load(LEVELS_DIR / f"{name}.yaml")


def greedy(obs, _):
    """Steer at the current target: its direction in the facing frame, which the goal vector holds."""
    g = obs["goal"][:2]
    d = max(float(np.linalg.norm(g)), 1e-6)
    return (g / d * min(1.0, d / 6)).astype(np.float32)  # slower near the target: the ball coasts


def constant(*a):
    return lambda obs, _: np.array(a, np.float32)


def camera(obs, i):
    """Greedy steering plus a held turn, and a pitch that swings, so the camera flag's two inputs both run."""
    return np.append(greedy(obs, i), [0.05, 1.0 if i % 20 < 10 else -1.0]).astype(np.float32)


LEFT = make("survive", region={"type": "box", "pos": [0, 0, 0.5], "size": [2, 2, 1]})
EDGE = make("survive", region={"type": "box", "pos": [0, 0, 0.5], "size": [20, 20, 1]},
            hazards=[{"type": "sphere", "pos": [0, 4, 0.5], "radius": 0.3}])
STACKED = make("waypoint_race", waypoints=[sph(0, 3), sph(0, 3.4), sph(3, 6)])
OUT_OF_ORDER = make("waypoint_race", waypoints=[sph(0, 8), sph(0, 4), sph(0, 1)])
SHORT = make("collect_items", items=[sph(0, 3, 0.5)], hazards=[])
# id: (level, task, policy, env kwargs, expected outcome)
SCENARIOS = {
    "waypoints": (bundled("waypoint_loop"), "waypoint_race", greedy, {}, "success"),
    "waypoints_camera_shaped": (STACKED, "waypoint_race", camera, {"camera_action": True, "shaped": True}, "success"),
    "waypoints_stacked": (STACKED, "waypoint_race", greedy, {"shaped": True}, "success"),
    "waypoints_passed_out_of_order": (OUT_OF_ORDER, "waypoint_race", greedy, {}, "success"),
    "items": (bundled("item_scatter"), "collect_items", greedy, {"shaped": True, "shaping_gamma": 0.9}, "success"),
    "items_truncated_frame_skip": (bundled("item_scatter"), "collect_items", constant(0, 0), {"time_limit_s": 2, "frame_skip": 5}, "truncated"),
    "items_no_truncation": (SHORT, "collect_items", constant(1, 0), {"time_limit_s": 1, "truncate": False}, "running"),
    "keys_then_goal": (bundled("locked_goal"), "unlock_goal", greedy, {}, "success"),
    "keys_then_goal_camera": (bundled("locked_goal"), "unlock_goal", camera, {"camera_action": True}, "success"),
    "survive_time_up": (bundled("first_survive"), "survive", constant(0, 0), {"shaped": True}, "success"),
    "survive_left_region": (LEFT, "survive", constant(0, 1), {"shaped": True}, "left_region"),
    "survive_hazard": (EDGE, "survive", constant(0, 1), {}, "hazard"),
    "hazard_in_waypoints": (make("waypoint_race", waypoints=[sph(0, 9)], hazards=[sph(0, 4, 0.3)]), "waypoint_race", constant(0, 1), {}, "hazard"),
    "waypoints_fall": (make("waypoint_race", waypoints=[sph(0, 9)]), "waypoint_race", constant(1, 0), {}, "kill_plane"),
}


def record(lv, task, policy, kw):
    """One CPU episode: the reset, then per step the core state, what step returns, and the geoms' colours."""
    env = HamsterballEnv(levels=[lv], tasks=[task], **kw)
    obs, info = env.reset(seed=0)
    rgba = lambda: {env.core.model.geom(g).name: env.core.model.geom_rgba[g].copy() for g in range(env.core.model.ngeom)}  # noqa: E731
    ep = {"reset_obs": obs, "reset_rgba": rgba(), "steps": [], "model": env.core.model}
    for i in range(env._max_steps + 5):
        obs, r, term, trunc, info = env.step(policy(obs, i))
        ep["steps"].append((info["core_state"], obs, r, term, trunc, info, rgba()))
        if term or trunc:
            break
    last = ep["steps"][-1]
    ep["outcome"] = "success" if last[5]["success"] else last[5]["failure_reason"] or ("truncated" if last[4] else "running")
    ep["events"] = [s[5]["reward_terms"]["event"] for s in ep["steps"]]
    return ep


def warp_env(lv, task, n, **kw):
    from hamsterball.gpu.vector_env import WarpVectorEnv

    return WarpVectorEnv(n, levels=[lv], **{"tasks": [task], "device": DEVICE, **kw})


def looks(venv, w):
    """The episode's markers' colours in world w, by geom name (`task/key[/i]`): the padded model keeps each marker in
    a slot, which the vector env maps from its row."""
    task = TASKS[int(venv.task_idx.numpy()[w])]
    e = int(venv.core.entry.numpy()[w])
    bank = venv.core.bank
    block = bank.levels[bank.entries[e][0]].tasks[task]
    rgba = venv.core.model.geom_rgba.numpy()[w]
    geoms = venv.row_geom.numpy()[e * venv.markers.stride:]
    names = [f"{task}/{k}" + (f"/{i}" if isinstance(block[k], list) else "")
             for k in TASK_MARKERS[task] for i in (range(len(block[k])) if isinstance(block[k], list) else [0])]
    return {name: rgba[geoms[i]] for i, name in enumerate(names)}


def same_looks(got, want, where):
    """The CPU env hides other tasks' markers with alpha 0; the bank holds only the episode task's, so those are absent."""
    for name, c in want.items():
        if "/" in name:  # marker geoms are named task/key[/i]
            if name in got:
                np.testing.assert_allclose(got[name], c, atol=1e-6, err_msg=f"{where} {name}")
            else:
                assert c[3] == 0, f"{where} {name} is drawn on the CPU but not in the bank"


@pytest.mark.parametrize("name", SCENARIOS)
def test_step_to_matches_the_cpu_env_on_recorded_episodes(name):
    lv, task, policy, kw, want = SCENARIOS[name]
    ep = record(lv, task, policy, kw)
    assert ep["outcome"] == want  # the scenario does what it says
    venv = warp_env(lv, task, 1, **kw)
    obs, _ = venv.reset(seed=0)
    for k in ("state", "goal"):
        np.testing.assert_allclose(obs[k][0].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
    same_looks(looks(venv, 0), ep["reset_rgba"], "reset")
    worst = worst_r = 0.0
    for t, (state, cpu_obs, cpu_r, cpu_term, cpu_trunc, cpu_info, cpu_rgba) in enumerate(ep["steps"]):
        obs, reward, terminated, truncated, info = venv.step_to(state[None])
        where = f"{name} step {t}"
        assert bool(terminated[0]) == cpu_term and bool(truncated[0]) == cpu_trunc, where
        assert bool(info["success"][0]) == cpu_info["success"], where
        assert int(info["failure_reason"][0]) == CODES[cpu_info["failure_reason"]], where
        done = cpu_term or cpu_trunc
        got = info["final_obs"] if done else obs  # a finished world has already reset
        for k in ("state", "goal"):
            g = got[k][0].cpu().numpy()
            np.testing.assert_allclose(g, cpu_obs[k], **OBS, err_msg=f"{where} {k}")
            worst = max(worst, float(np.abs(g - cpu_obs[k]).max()))
        np.testing.assert_allclose(float(reward[0]), cpu_r, **OBS, err_msg=where)
        worst_r = max(worst_r, abs(float(reward[0]) - cpu_r))
        for term_name, v in cpu_info["reward_terms"].items():
            np.testing.assert_allclose(float(info["reward_terms"][term_name][0]), v, **OBS, err_msg=f"{where} {term_name}")
        if not done:  # a finished world's colours are already its next episode's
            same_looks(looks(venv, 0), cpu_rgba, where)
    print(f"{name}: worst observation difference {worst:.3g}, reward {worst_r:.3g}")


def test_one_env_step_can_complete_several_targets():
    ep = record(STACKED, "waypoint_race", greedy, {})
    assert max(ep["events"]) >= 2.0  # the CPU completes two waypoints in one step: the scenario exercises the rerun


def test_failure_order_is_hazard_then_kill_plane_then_left_region():
    """Three states that each fail every test they can: far outside the region, below the kill plane, in a hazard."""
    hz = {"type": "sphere", "pos": [30, 0, -6], "radius": 1}
    region = {"type": "box", "pos": [0, 0, 0.5], "size": [2, 2, 1]}
    cases = {
        "left_region": (make("survive", region=region), [30, 0, 0.5]),
        "kill_plane": (make("survive", region=region), [30, 0, -6]),
        "hazard": (make("survive", region=region, hazards=[hz]), [30, 0, -6]),
    }
    for want, (lv, pos) in cases.items():
        cpu = HamsterballEnv(levels=[lv], tasks=["survive"])
        _, info = cpu.reset(seed=0)
        s = info["core_state"].copy()
        s[1 + SHELL_QPOS : 4 + SHELL_QPOS] = pos  # state layout: time, then qpos
        _, r, term, _, cpu_info = cpu.step_to(s)
        assert term and cpu_info["failure_reason"] == want
        venv = warp_env(lv, "survive", 1)
        venv.reset(seed=0)
        _, reward, terminated, _, info = venv.step_to(s[None])
        assert bool(terminated[0]) and int(info["failure_reason"][0]) == CODES[want], want
        np.testing.assert_allclose(float(reward[0]), r, **OBS)


def test_survive_has_no_time_penalty_and_a_still_hamster_wins_at_time_up():
    venv = warp_env(bundled("first_survive"), "survive", 2, time_limit_s=1 / 6)  # 10 env steps
    venv.reset(seed=0)
    for i in range(10):
        _, reward, terminated, truncated, info = venv.step(np.zeros((2, 2), np.float32))
        if i < 9:
            assert (info["reward_terms"]["time"] == 0).all() and not terminated.any()
    assert terminated.all() and info["success"].all() and not truncated.any()
    np.testing.assert_allclose(reward.cpu().numpy(), 1.0, atol=1e-6)


def test_the_cameras_turn_and_pitch_set_the_facing_and_pitch_in_the_observation():
    lv = make("collect_items", items=[sph(0, 9)])
    cpu = HamsterballEnv(levels=[lv], tasks=["collect_items"], camera_action=True)
    cpu.reset(seed=0)
    venv = warp_env(lv, "collect_items", 1, camera_action=True)
    venv.reset(seed=0)
    for i in range(12):
        a = np.array([[0.0, 0.0, 0.4, -1.0 if i < 6 else 1.0]], np.float32)
        obs, *_ = cpu.step(a[0])
        got, *_ = venv.step(a)
    np.testing.assert_allclose(got["state"][0, 4:8].cpu().numpy(), obs["state"][4:8], atol=1e-5)
    assert abs(float(got["state"][0, 4])) > 0.1  # it did turn


def test_each_bundled_static_task_level_runs_through_make_vec():
    import gymnasium as gym

    ids = {"reach_goal": "HamsterballReachGoal-v2", "waypoint_race": "HamsterballWaypointRace-v2",
           "collect_items": "HamsterballCollectItems-v2", "unlock_goal": "HamsterballUnlockGoal-v2",
           "survive": "HamsterballSurvive-v2"}
    levels = {"reach_goal": "first_goal", "waypoint_race": "waypoint_loop", "collect_items": "item_scatter",
              "unlock_goal": "locked_goal", "survive": "first_survive"}
    assert set(ids) == set(TASKS)
    for task, id_ in ids.items():
        venv = gym.make_vec(id_, num_envs=2, vectorization_mode="vector_entry_point", backend="warp",
                            levels=[levels[task]], device=DEVICE)
        obs, info = venv.reset(seed=0)
        assert (info["task_index"].cpu().numpy() == TASKS.index(task)).all()
        assert (obs["goal"][:, 5 + TASKS.index(task)] == 1).all()
        venv.step(np.zeros((2, 2), np.float32))


@pytest.mark.parametrize("lv, task", [
    (bundled("survive_seesaw"), "survive"),  # a seesaw
    (bundled("survive_collapse"), "survive"),  # tiles
])
def test_levels_with_seesaws_or_tiles_raise(lv, task):
    with pytest.raises(NotImplementedError):
        warp_env(lv, task, 1)
