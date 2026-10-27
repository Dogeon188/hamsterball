"""The Warp vector env on a level bank (GPU effort, seam 2 and sampling): pools of levels, a draw per world at each reset."""

import os

import numpy as np
import pytest
import torch

from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from hamsterball.env import LEVELS_DIR, HamsterballEnv
from test_warp_env import AHEAD, BOX, CAPSULE, CODES, FORWARD, OBS, SIDEWAYS, SPHERE, level, record

pytestmark = pytest.mark.gpu

DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")  # "cuda:0" runs the same tests on a GPU
HILL = Level.load(LEVELS_DIR / "hill.yaml")
SURVIVE_ONLY = Level.load(LEVELS_DIR / "first_survive.yaml")
LOW = level({"type": "sphere", "pos": [3, 6, 0.5], "radius": 0.5})  # its own goal
LOW.kill_plane_z = -9.0
FAR = level(SPHERE)  # a floor nowhere near the spawn: a ball dropped on it falls
FAR.geoms = [{"type": "box", "pos": [50, 50, -0.5], "size": [10, 10, 0.5], "euler": [0, 0, 0]}]


CODES_NAME = {v: k for k, v in CODES.items()}


def warp_env(n, levels, **kw):
    from hamsterball.gpu.vector_env import WarpVectorEnv

    return WarpVectorEnv(n, levels=levels, tasks=["reach_goal"], device=DEVICE, **{"bank_size": 4, **kw})


def test_each_world_draws_its_level_and_a_seed_replays_the_draw():
    pool = [level(SPHERE), LOW, HILL, SURVIVE_ONLY]  # the last has no reach_goal: never drawn
    venv = warp_env(64, pool)
    _, info = venv.reset(seed=3)
    drawn = info["level_index"].cpu().numpy().copy()
    assert set(drawn) == {0, 1, 2}
    assert (info["task_index"] == 0).all()  # reach_goal's index in the goal one-hot
    venv.reset(seed=3)
    assert np.array_equal(venv.reset(seed=3)[1]["level_index"].cpu().numpy(), drawn)
    assert not np.array_equal(venv.reset(seed=4)[1]["level_index"].cpu().numpy(), drawn)
    _, info = venv.reset()  # no seed: the stream goes on
    assert set(info["level_index"].cpu().numpy()) == {0, 1, 2}


def test_levels_are_the_banks_canonical_levels():
    venv = warp_env(2, ["first_goal", "hill", LOW])
    assert len(venv.levels) == 3
    for lv in venv.levels:
        assert Level.from_yaml(lv.canonical()).id == lv.id


def test_forced_draws_put_each_world_on_its_level_and_match_the_cpu_reset():
    pool = [level(SPHERE), LOW, HILL]
    venv = warp_env(6, pool)
    forced = torch.tensor([2, 0, 1, 1, 0, 2])
    obs, info = venv.reset(seed=0, options={"level_index": forced, "task_index": torch.zeros(6, dtype=torch.long)})
    assert np.array_equal(info["level_index"].cpu().numpy(), forced.numpy())
    for w, i in enumerate(forced.tolist()):  # kill plane height, goal and spawn differ per level
        cpu = HamsterballEnv(levels=[pool[i]], tasks=["reach_goal"])
        want, _ = cpu.reset(seed=0)
        for k in ("state", "goal"):
            np.testing.assert_allclose(obs[k][w].cpu().numpy(), want[k], **OBS, err_msg=f"world {w} level {i} {k}")
    # a later reset without options draws freely again
    _, info = venv.reset()
    assert len(set(info["level_index"].cpu().numpy())) > 1
    _, info = venv.reset(options={"level_index": 1})  # an int forces every world
    assert (info["level_index"] == 1).all()


def test_each_world_stands_on_its_own_level_and_falls_off_a_level_without_a_floor():
    venv = warp_env(4, [level(SPHERE), FAR])
    venv.reset(seed=0, options={"level_index": torch.tensor([0, 1, 0, 1])})
    first = {}
    for t in range(100):
        _, _, term, _, info = venv.step(np.zeros((4, 2), np.float32))
        for w in np.flatnonzero(term.cpu().numpy()):
            first.setdefault(int(w), CODES_NAME[int(info["failure_reason"][w])])
    assert first == {1: "kill_plane", 3: "kill_plane"}  # the other two stood on their floor the whole time


def test_a_forced_draw_the_pool_cannot_make_is_an_error():
    venv = warp_env(2, [level(SPHERE), SURVIVE_ONLY])
    with pytest.raises(ValueError):
        venv.reset(options={"level_index": 1})  # no reach_goal there
    with pytest.raises(ValueError):
        venv.reset(options={"level_index": 5})
    with pytest.raises(ValueError):
        venv.reset(options={"task_index": 4})  # survive: no pool level has it for these tasks
    with pytest.raises(ValueError):
        venv.reset(options={"level": "x"})


def test_a_pool_with_seesaws_raises_until_they_are_built():
    raft = Level.from_dict({
        "kill_plane_z": -5, "geoms": [], "bodies": [{"pos": [0, 0, -0.1], "geoms": [{"type": "box", "size": [2, 2, 0.1]}],
                                                      "joint": {"type": "ball", "stiffness": 50, "range": 15}}],
        "tasks": {"reach_goal": {"spawn": {"pos": [0, 0, 0.5]}, "goal": SPHERE, "time_limit_s": 30}}})
    with pytest.raises(NotImplementedError, match="seesaws"):
        warp_env(2, [level(SPHERE), raft])


def replay(venv, eps, labels):
    """Feed each world the core states of its CPU episode; compare observations, rewards, flags and codes."""
    worst = 0.0
    for t in range(max(len(e["steps"]) for e in eps)):
        states = np.stack([e["steps"][min(t, len(e["steps"]) - 1)][0] for e in eps])
        obs, reward, terminated, truncated, info = venv.step_to(states)
        for w, ep in enumerate(eps):
            if t >= len(ep["steps"]):
                continue
            cpu_obs, cpu_r, cpu_term, cpu_trunc, cpu_info = ep["steps"][t][1:6]
            where = f"{labels[w]} world {w} step {t}"
            assert bool(terminated[w]) == cpu_term and bool(truncated[w]) == cpu_trunc, where
            assert bool(info["success"][w]) == cpu_info["success"], where
            assert int(info["failure_reason"][w]) == CODES[cpu_info["failure_reason"]], where
            done = cpu_term or cpu_trunc
            got = info["final_obs"] if done else obs
            for k in ("state", "goal"):
                g = got[k][w].cpu().numpy()
                np.testing.assert_allclose(g, cpu_obs[k], **OBS, err_msg=f"{where} {k}")
                worst = max(worst, float(np.abs(g - cpu_obs[k]).max()))
            np.testing.assert_allclose(float(reward[w]), cpu_r, **OBS, err_msg=where)
    return worst


def test_step_to_matches_the_cpu_env_on_episodes_from_several_bank_levels():
    """Worlds forced onto different levels of one bank replay CPU episodes recorded on those levels."""
    kw = {"time_limit_s": 6, "shaped": True}
    cases = [(level(SPHERE), FORWARD, "success"), (level(CAPSULE), FORWARD, "success"), (level(BOX), [0.1, 1.0], "success"),
             (level(SPHERE, [AHEAD]), FORWARD, "hazard"), (level(SPHERE), SIDEWAYS, "kill_plane"), (LOW, [0.0, 0.0], "truncated")]
    eps = [record(lv, a, kw) for lv, a, _ in cases]
    assert [e["outcome"] for e in eps] == [c[2] for c in cases]
    bank = [lv for lv, _, _ in cases][::-1] + [HILL]  # in another order than the episodes, and with an hfield level
    venv = warp_env(len(eps), bank, **kw)
    forced = torch.tensor([bank.index(lv) for lv, _, _ in cases])
    obs, _ = venv.reset(seed=0, options={"level_index": forced})
    for w, ep in enumerate(eps):
        for k in ("state", "goal"):
            np.testing.assert_allclose(obs[k][w].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
    print(f"mixed bank: worst observation difference {replay(venv, eps, [c[2] for c in cases]):.3g}")


def test_step_to_matches_the_cpu_env_on_episodes_of_different_tasks_in_one_bank():
    """One bank of levels of five tasks' kinds: each world forced onto its level and task, with markers and looks."""
    from test_warp_tasks import EDGE, LEFT, OUT_OF_ORDER, greedy, looks, make, same_looks, sph
    from test_warp_tasks import record as record_task

    from hamsterball.env import TASKS

    kw = {"shaped": True}
    stand = lambda obs, i: np.zeros(2, np.float32)  # noqa: E731
    run = lambda obs, i: np.array([0.0, 1.0], np.float32)  # noqa: E731
    cases = [
        (level(SPHERE), "reach_goal", greedy, "success"),
        (OUT_OF_ORDER, "waypoint_race", greedy, "success"),
        (make("collect_items", items=[sph(0, 3, 0.5), sph(1, 5, 0.5)]), "collect_items", greedy, "success"),
        (make("unlock_goal", keys=[sph(0, 3)], goal=sph(0, 6)), "unlock_goal", greedy, "success"),
        (EDGE, "survive", run, "hazard"),
        (LEFT, "survive", run, "left_region"),
        (Level.load(LEVELS_DIR / "first_survive.yaml"), "survive", stand, "success"),
    ]
    eps = [record_task(lv, task, policy, kw) for lv, task, policy, _ in cases]
    assert [e["outcome"] for e in eps] == [c[3] for c in cases]
    bank = [c[0] for c in cases][::-1]
    from hamsterball.gpu.vector_env import WarpVectorEnv

    venv = WarpVectorEnv(len(eps), levels=bank, device=DEVICE, bank_size=4, **kw)  # all five tasks
    forced = {"level_index": torch.tensor([bank.index(c[0]) for c in cases]),
              "task_index": torch.tensor([TASKS.index(c[1]) for c in cases])}
    obs, info = venv.reset(seed=0, options=forced)
    assert np.array_equal(info["task_index"].cpu().numpy(), forced["task_index"].numpy())
    for w, ep in enumerate(eps):
        for k in ("state", "goal"):
            np.testing.assert_allclose(obs[k][w].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
        same_looks(looks(venv, w), ep["reset_rgba"], f"world {w} reset")
    print(f"mixed tasks: worst observation difference {replay(venv, eps, [c[3] for c in cases]):.3g}")


def test_a_level_with_two_tasks_draws_both_and_the_pool_honours_tasks():
    from hamsterball.env import TASKS

    spawn = {"pos": [0, 0, 0.5]}
    both = Level.from_dict({"kill_plane_z": -5, "geoms": FAR.geoms, "tasks": {
        "reach_goal": {"spawn": spawn, "goal": SPHERE, "time_limit_s": 5},
        "survive": {"spawn": spawn, "region": {"type": "box", "pos": [0, 0, 0.5], "size": [2, 2, 1]}, "time_limit_s": 5}}})
    from hamsterball.gpu.vector_env import WarpVectorEnv

    venv = WarpVectorEnv(64, levels=[both, level(SPHERE)], tasks=["reach_goal", "survive"], device=DEVICE)
    _, info = venv.reset(seed=0)
    drawn = set(zip(info["level_index"].cpu().tolist(), info["task_index"].cpu().tolist()))
    assert drawn == {(0, 0), (0, 4), (1, 0)}  # the second level has no survive; each of the first level's tasks is drawn
    _, info = venv.reset(options={"task_index": TASKS.index("survive")})
    assert (info["level_index"] == 0).all() and (info["task_index"] == 4).all()
    only = WarpVectorEnv(8, levels=[both, level(SPHERE)], tasks=["reach_goal"], device=DEVICE)
    _, info = only.reset(seed=0)
    assert (info["task_index"] == 0).all()


def test_a_generator_pool_is_a_bank_of_generated_levels():
    gen = ProceduralGenerator()
    venv = warp_env(3, gen, bank_size=4)
    assert len(venv.levels) == 4 and all(lv.generator["task"] == "reach_goal" for lv in venv.levels)
    assert len({lv.id for lv in venv.levels}) == 4
    venv.reset(seed=0)
    obs, *_ = venv.step(np.zeros((3, 2), np.float32))
    assert np.isfinite(obs["state"].cpu().numpy()).all()


def test_set_levels_swaps_the_pool_and_worlds_switch_at_their_next_reset():
    venv = warp_env(4, [level(SPHERE)], time_limit_s=0.5)  # 30 env steps, on a floor
    venv.reset(seed=0)
    venv.step(np.zeros((4, 2), np.float32))
    old = venv.levels
    venv.set_levels([FAR, HILL])
    assert venv.levels is not old and len(venv.levels) == 2
    for _ in range(20):  # mid-episode: on the old level, which is gone from the bank
        obs, _, term, trunc, info = venv.step(np.zeros((4, 2), np.float32))
    assert (info["level_index"] == -1).all() and not trunc.any()
    assert obs["state"][:, 2].min() > 0.4  # still on the floor of the level it started on: the new bank has none here
    for _ in range(40):
        obs, _, term, trunc, info = venv.step(np.zeros((4, 2), np.float32))
        if trunc.any():
            break
    assert trunc.all()
    assert set(info["level_index"].cpu().numpy()) <= {0, 1}  # reset onto the new pool
    venv.set_levels(ProceduralGenerator())  # a generator pool: a new bank of bank_size levels
    assert len(venv.levels) == 4 and venv.levels[0].generator is not None
    venv.reset(seed=1)
    venv.step(np.zeros((4, 2), np.float32))
