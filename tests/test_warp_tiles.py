"""Collapsing tiles on the batched core and the Warp vector env (GPU effort): seam 1 against `Core` (tiles that trigger
within a step, tiles already falling, the CPU-layout state) and seam 2 against the CPU env (recorded episodes of collapse
levels replayed through `step_to`, tile colours included)."""

import os

import numpy as np
import pytest
import torch

from hamsterball.core import Core
from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.level import Level
from hamsterball.core.step import TILE_DROP
from hamsterball.env import LEVELS_DIR, HamsterballEnv

from test_warp_env import CODES, OBS

pytestmark = pytest.mark.gpu

DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")  # "cuda:0" runs the same tests on a GPU
FLOAT32 = dict(rtol=1e-5, atol=1e-6)  # what the spec asks of the core-step kernels' writes
TILE_QVEL = dict(rtol=1e-5, atol=3e-5)  # a secant over the step where a tile reaches its bottom: the stored trigger's float32
# rounding (2e-9 s at 0.1 s) times g * s (11 m/s) is 2e-8 m of z, which over dt is 1e-5 m/s; see tiles.md
LOOSE = dict(rtol=1e-4, atol=1e-5)  # shell and hamster dofs after a physics step; see batch_core.md
GEN = ProceduralGenerator()
STILL = np.zeros(2, np.float32)


def bundled(name):
    return Level.load(LEVELS_DIR / f"{name}.yaml")


def survive(**extra):
    return {"kill_plane_z": -5, "geoms": extra.pop("geoms", []),
            "tasks": {"survive": {"spawn": {"pos": [0, 0, 0.5]}, "time_limit_s": 10,
                                  "region": {"type": "box", "pos": [0, 0, 0.5], "size": [8, 8, 2]}}}, **extra}


# a 2 by 2 floor of boxes, 0.3 s delay, the ball spawning over the middle of all four
SMALL = Level.from_dict(survive(collapse={"delay": 0.3, "tiles": [
    {"type": "box", "pos": [x, y, -0.1], "size": [0.74, 0.74, 0.1], "rgba": [0.2 + 0.1 * i, 0.4, 0.6, 1]}
    for i, (x, y) in enumerate([(-0.75, -0.75), (-0.75, 0.75), (0.75, -0.75), (0.75, 0.75)])]}))
# tiles of every shape, and a solid under part of them
MIXED = Level.from_dict(survive(geoms=[{"type": "box", "pos": [4, 0, -0.5], "size": [1, 4, 0.5]}], collapse={"delay": 0.1, "tiles": [
    {"type": "box", "pos": [0, 0, -0.1], "size": [1, 1, 0.1]},
    {"type": "sphere", "pos": [2.2, 0, -0.4], "size": [0.5]},
    {"type": "capsule", "pos": [0, 2, -0.2], "size": [0.3, 1.0], "euler": [0, 90, 0]}]}))
SURVIVE_COLLAPSE = bundled("survive_collapse")  # 81 tiles
GENERATED = GEN("survive", 9, 0.5)  # 63 tiles, delay 1.25 s
FLOOR = Level.from_dict(survive(geoms=[{"type": "box", "pos": [0, 0, -0.5], "size": [10, 10, 0.5]}]))
assert [len(lv.collapse["tiles"]) if lv.collapse else 0 for lv in (SMALL, MIXED, SURVIVE_COLLAPSE, GENERATED, FLOOR)] == [4, 3, 81, 63, 0]
BANK = [SMALL, MIXED, SURVIVE_COLLAPSE, GENERATED, FLOOR]


def batch(levels, **kw):
    from hamsterball.gpu.batch_core import BatchCore

    return BatchCore(levels=levels, num_envs=len(levels), device=DEVICE, tasks=["survive"], **kw)


def cpu_core(lv, env_steps, action=STILL):
    core = Core(level=lv)
    core.reset(lv.tasks["survive"]["spawn"]["pos"])
    for _ in range(env_steps):
        core.step(action)
    return core


def tile_slices(core):
    """Where each tile's qpos, qvel and trigger-time userdata sit in the CPU core-state vector."""
    nq, nv = core.model.nq, core.model.nv
    nu = len(core.data.userdata)
    t = core.tiles
    return [(slice(1 + a.qpos, 8 + a.qpos), slice(1 + nq + a.dof, 7 + nq + a.dof), 1 + nq + 3 * nv + 6 * core.model.nbody + 7 * core.model.nmocap + t.slot + i)
            for i, a in enumerate(t.tiles)], nu


def test_every_level_of_a_bank_with_tiles_round_trips_its_cpu_state():
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        s = cpu_core(lv, 40).state().astype(np.float32).astype(np.float64)
        b.reset([w], bank_index=w)
        b.set_state(w, s)
        assert np.array_equal(b.state(w), s), w
    states = [cpu_core(lv, 5).state().astype(np.float32) for lv in BANK]
    assert len({len(s) for s in states}) > 1
    b.set_states(states)
    for w, s in enumerate(states):
        assert np.array_equal(b.state(w), s), w
    assert all(np.array_equal(a, s) for a, s in zip(b.cpu_states(), states))


def test_a_state_with_touched_and_untouched_tiles_round_trips_with_its_infinities():
    b = batch(BANK)
    core = cpu_core(SMALL, 20)
    s = core.state().astype(np.float32)
    _, nu = tile_slices(core)
    assert np.isinf(s[-4:]).any() and np.isfinite(s[-4:]).any()  # some tiles touched, some not
    b.reset([0], bank_index=0)
    b.set_state(0, s)
    got = b.state(0)
    assert np.array_equal(got[-4:], s[-4:])  # inf stays inf, a stored time stays


DROP = 0.05  # m: a ball spawned this high lands on the tiles after 0.1 s (a spawn at rest on them touches at distance 0,
# which MuJoCo lists as a contact and Warp does not: the trigger comes one physics step later, see tiles.md)


def drop_core(lv, steps=None):
    """A Core on the level with the ball DROP above its spawn: run still until just before a tile triggers (steps None), or
    for `steps` env steps."""
    core = Core(level=lv)
    x, y, z = lv.tasks["survive"]["spawn"]["pos"]
    core.reset((x, y, z + DROP))
    if steps is not None:
        for _ in range(steps):
            core.step(STILL)
        return core
    while True:
        before = core.state()
        core.step(STILL, 1)
        if np.isfinite(core.data.userdata[core.tiles.slot:]).any() if core.tiles else True:
            break
    core.set_state(before)
    return core


@pytest.mark.parametrize("env_steps", [None, 30, 90, 200])
def test_drive_and_touch_tiles_match_the_cpu_core_after_one_physics_step(env_steps):
    """Tiles that trigger within the step (None: the state just before the ball lands), and tiles already falling (the
    later ones): each tile's qpos, qvel and trigger time after a step are the CPU's, whatever the padded slots beyond the
    level's own tiles hold."""
    b = batch(BANK)
    cores = []
    for w, lv in enumerate(BANK):
        core = drop_core(lv, env_steps)
        b.reset([w], bank_index=w)
        b.set_state(w, core.state())
        cores.append(core)
    b.step(np.zeros((len(BANK), 2), np.float32), 1)
    for w, core in enumerate(cores):
        core.step(STILL, 1)
        want, got = core.state(), b.state(w)
        assert len(got) == len(want)
        np.testing.assert_allclose(got[0], want[0], atol=1e-6)
        if core.tiles:
            for i, (qp, qv, ud) in enumerate(tile_slices(core)[0]):
                np.testing.assert_allclose(got[qp], want[qp], **FLOAT32, err_msg=f"world {w} tile {i} qpos")
                np.testing.assert_allclose(got[qv], want[qv], **TILE_QVEL, err_msg=f"world {w} tile {i} qvel")
                assert np.isinf(got[ud]) == np.isinf(want[ud]), f"world {w} tile {i} trigger"
                if np.isfinite(want[ud]):
                    np.testing.assert_allclose(got[ud], want[ud], atol=2e-6, err_msg=f"world {w} tile {i} trigger time")
        np.testing.assert_allclose(got[1:15], want[1:15], **LOOSE, err_msg=f"world {w} shell and hamster qpos")


def test_a_tile_falls_at_g_and_is_held_under_the_kill_plane():
    """A world driven by `step`, the ball over one tile of SMALL: the tile it touches drops `delay` later, is at the CPU's
    height at every step, and stops TILE_DROP under the kill plane."""
    b = batch([SMALL])
    core = drop_core(SMALL, 0)
    b.reset(spawn=(0.0, 0.0, 0.5 + DROP))
    t = core.tiles
    for env_step in range(130):  # 2.2 s: 0.3 s of delay, then 1.9 s of fall (the tile reaches the kill plane at 1.5 s)
        core.step(STILL)
        b.step(np.zeros((1, 2), np.float32))
        want, got = core.state(), b.state(0)
        for i, (qp, qv, ud) in enumerate(tile_slices(core)[0]):
            np.testing.assert_allclose(got[qp][:3], want[qp][:3], rtol=0, atol=2e-3, err_msg=f"step {env_step} tile {i}")
    z = [got[qp][2] - a.pos[2] for (qp, _, _), a in zip(tile_slices(core)[0], t.tiles)]
    assert min(z) == pytest.approx(SMALL.kill_plane_z - TILE_DROP + 0.1, abs=1e-5), "held under the kill plane"


def test_reset_leaves_every_tile_untouched_at_rest_and_unused_slots_alone():
    b = batch(BANK)
    for w, lv in enumerate(BANK):
        b.reset([w], bank_index=w)
        core = Core(level=lv)
        core.reset(lv.tasks["survive"]["spawn"]["pos"])
        np.testing.assert_allclose(b.state(w), core.state(), **FLOAT32, err_msg=f"world {w}")


# ---- seam 2: the Warp vector env against the CPU env on recorded episodes -------------------------------------------


def go_east(obs, _):
    return np.array([1.0, 0.0], np.float32)


def record(lv, policy, kw):
    """One CPU episode: the reset, then per step the core state, what step returns and every tile's colour."""
    env = HamsterballEnv(levels=[lv], tasks=["survive"], **kw)
    obs, info = env.reset(seed=0)
    m, t = env.core.model, env.core.tiles
    colours = lambda: np.array([m.geom_rgba[m.body_geomadr[t.body + i]] for i in range(len(t.tiles))])  # noqa: E731
    ep = {"reset_obs": obs, "reset_colours": colours(), "steps": []}
    for i in range(env._max_steps + 5):
        obs, r, term, trunc, info = env.step(policy(obs, i))
        ep["steps"].append((info["core_state"], obs, r, term, trunc, info, colours()))
        if term or trunc:
            break
    last = ep["steps"][-1]
    ep["outcome"] = "success" if last[5]["success"] else last[5]["failure_reason"] or ("truncated" if last[4] else "running")
    return ep


def venv_for(levels, n, **kw):
    from hamsterball.gpu.vector_env import WarpVectorEnv

    return WarpVectorEnv(n, levels=levels, tasks=["survive"], device=DEVICE, **kw)


def warp_colours(venv, w):
    e = int(venv.core.entry.numpy()[w])
    geoms = [venv.core.bank.slot_ids[s] for s in venv.core.bank.tile_slots[e]]
    return venv.core.model.geom_rgba.numpy()[w, geoms]


def replay(venv, eps, labels):
    """Feed each world the core states of its CPU episode; compare observations, rewards, flags, codes and tile colours.
    Returns the worst observation, reward and colour differences."""
    worst = worst_r = worst_c = 0.0
    for w, ep in enumerate(eps):
        np.testing.assert_allclose(warp_colours(venv, w), ep["reset_colours"], atol=1e-6, err_msg=f"{labels[w]} reset colours")
    for t in range(max(len(e["steps"]) for e in eps)):
        held = venv.core.cpu_states()  # a finished world has reset onto a level of the draw: hold its state
        states = [e["steps"][t][0] if t < len(e["steps"]) else held[w] for w, e in enumerate(eps)]
        obs, reward, terminated, truncated, info = venv.step_to(states)
        for w, ep in enumerate(eps):
            if t >= len(ep["steps"]):
                continue
            cpu_obs, cpu_r, cpu_term, cpu_trunc, cpu_info, cpu_c = ep["steps"][t][1:]
            where = f"{labels[w]} step {t}"
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
            worst_r = max(worst_r, abs(float(reward[w]) - cpu_r))
            if not done:  # a finished world's tiles are already its next episode's
                c = warp_colours(venv, w)
                np.testing.assert_allclose(c, cpu_c, atol=1e-6, err_msg=f"{where} colours")
                worst_c = max(worst_c, float(np.abs(c - cpu_c).max()))
    return worst, worst_r, worst_c


SCENARIOS = {
    "bundled_floor_a_still_ball_falls_through_its_dropped_tile": (SURVIVE_COLLAPSE, "still", {"time_limit_s": 15}, "left_region"),
    "generated_floor_rolled_east_off_the_edge": (GENERATED, "east", {"time_limit_s": 30}, "left_region"),
    "small_floor_a_still_ball_falls_through": (SMALL, "still", {"time_limit_s": 5}, "left_region"),
    "mixed_shapes_a_still_ball_falls_through": (MIXED, "still", {"time_limit_s": 5}, "left_region"),
}
POLICIES = {"still": lambda obs, i: STILL, "east": go_east}


@pytest.mark.parametrize("name", SCENARIOS)
def test_step_to_matches_the_cpu_env_on_recorded_episodes_on_collapse_levels(name):
    lv, policy, kw, want = SCENARIOS[name]
    ep = record(lv, POLICIES[policy], kw)
    assert ep["outcome"] == want, ep["outcome"]  # the scenario does what it says
    venv = venv_for([lv], 1, **kw)
    obs, _ = venv.reset(seed=0)
    for k in ("state", "goal"):
        np.testing.assert_allclose(obs[k][0].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
    worst = replay(venv, [ep], [name])
    print(f"{name}: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}, tile colour {worst[2]:.3g}")


def test_episodes_on_different_collapse_levels_of_one_bank_replay_together():
    """Worlds forced onto each of five levels of one padded bank, three of them with tiles: each replays its own CPU
    episode, so the other worlds' tiles change nothing."""
    lvs = [SMALL, MIXED, SURVIVE_COLLAPSE, FLOOR]
    kw = {"time_limit_s": 6}
    eps = [record(lv, POLICIES["still"], kw) if lv.collapse else record_plain(lv, kw) for lv in lvs]
    venv = venv_for(lvs, len(lvs), **kw)
    venv.reset(seed=0, options={"level_index": torch.arange(len(lvs))})
    worst = replay_mixed(venv, eps, [f"level{i}" for i in range(len(lvs))])
    print(f"mixed bank: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}")


def record_plain(lv, kw):
    """record() for a level without tiles: no colours."""
    env = HamsterballEnv(levels=[lv], tasks=["survive"], **kw)
    obs, _ = env.reset(seed=0)
    ep = {"reset_obs": obs, "reset_colours": None, "steps": []}
    for _ in range(env._max_steps + 5):
        obs, r, term, trunc, info = env.step(STILL)
        ep["steps"].append((info["core_state"], obs, r, term, trunc, info, None))
        if term or trunc:
            break
    return ep


def replay_mixed(venv, eps, labels):
    """`replay` where some worlds have no tiles to compare."""
    tiled = [w for w, e in enumerate(eps) if e["reset_colours"] is not None]
    plain = [w for w in range(len(eps)) if w not in tiled]
    worst = worst_r = 0.0
    for w in tiled:
        np.testing.assert_allclose(warp_colours(venv, w), eps[w]["reset_colours"], atol=1e-6)
    for t in range(max(len(e["steps"]) for e in eps)):
        held = venv.core.cpu_states()
        states = [e["steps"][t][0] if t < len(e["steps"]) else held[w] for w, e in enumerate(eps)]
        obs, reward, terminated, truncated, info = venv.step_to(states)
        for w, ep in enumerate(eps):
            if t >= len(ep["steps"]):
                continue
            cpu_obs, cpu_r, cpu_term, cpu_trunc, cpu_info, cpu_c = ep["steps"][t][1:]
            where = f"{labels[w]} step {t}"
            assert bool(terminated[w]) == cpu_term and bool(truncated[w]) == cpu_trunc, where
            done = cpu_term or cpu_trunc
            got = info["final_obs"] if done else obs
            for k in ("state", "goal"):
                g = got[k][w].cpu().numpy()
                np.testing.assert_allclose(g, cpu_obs[k], **OBS, err_msg=f"{where} {k}")
                worst = max(worst, float(np.abs(g - cpu_obs[k]).max()))
            worst_r = max(worst_r, abs(float(reward[w]) - cpu_r))
            if w in tiled and not done:
                np.testing.assert_allclose(warp_colours(venv, w), cpu_c, atol=1e-6, err_msg=f"{where} colours")
    assert plain
    return worst, worst_r


def test_live_physics_collapses_each_world_from_its_own_trigger():
    """Through `step` on a bank of a tile level and a floor: the ball on the tiles loses them one by one and the world
    finishes (falls out), while the floor world carries on; colours turn from the authored one toward the warning."""
    venv = venv_for([SMALL, FLOOR], 2, time_limit_s=6)
    venv.reset(seed=0, options={"level_index": torch.tensor([0, 1])})
    authored = warp_colours(venv, 0).copy()
    first = None
    for i in range(300):
        obs, r, term, trunc, info = venv.step(torch.zeros(2, 2))
        if first is None and not np.allclose(warp_colours(venv, 0), authored):
            first = i
        if bool(term[0]):
            assert int(info["failure_reason"][0]) in (2, 3)
            break
    else:
        pytest.fail("the ball on four falling tiles never fell")
    assert first is not None and first < i, "the tiles warned before the ball fell"
    assert not bool(term[1]), "the floor world carries on"


def test_a_generator_pool_with_collapse_layouts_trains_on_the_device():
    """A bank of generated survive levels, collapse floors among them, steps, ends and resets worlds in the same step."""
    class NoSeesaw:
        def __init__(self):
            self.tasks, self.version = GEN.tasks, GEN.version

        def __call__(self, task, seed, difficulty):
            while (lv := GEN(task, seed, difficulty)) and any("joint" in b for b in lv.bodies):
                seed += 1
            return lv

    venv = venv_for(NoSeesaw(), 4, bank_size=8, difficulty=0.5)
    assert venv.core.bank.ntiles >= 1
    obs, info = venv.reset(seed=3)
    for _ in range(60):
        obs, r, term, trunc, info = venv.step(torch.zeros(4, 2))
    assert torch.isfinite(obs["state"]).all()
