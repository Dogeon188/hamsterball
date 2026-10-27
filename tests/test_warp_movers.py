"""Moving markers on the Warp vector env against the CPU env (GPU effort, seam 2): recorded CPU episodes of levels whose
region or hazards move, replayed through `step_to`, with each moving marker's geom pose compared too."""

import mujoco
import numpy as np
import pytest
import torch
from test_warp_env import CODES, DEVICE, OBS
from test_warp_tasks import bundled, constant, make, warp_env

from hamsterball.core.generator import ProceduralGenerator
from hamsterball.core.model import DT
from hamsterball.core.step import FRAME_SKIP
from hamsterball.env import HamsterballEnv

pytestmark = pytest.mark.gpu

GEN = ProceduralGenerator()
DRIFT_AND_HAZARDS = GEN("survive", 2, 0.5)  # a drifting region and two moving hazards
assert "motion" in DRIFT_AND_HAZARDS.tasks["survive"]["region"] and len(DRIFT_AND_HAZARDS.hazards) == 2
PERIOD = 2 * FRAME_SKIP * DT  # two env steps: the hazard is across the ball at one step from where it was at the last
FAST = {"type": "sphere", "pos": [0, 0, 0.5], "radius": 0.3,  # 4 m each way
        "motion": {"type": "oscillate", "axis": [1, 0, 0], "amplitude": 4, "period": PERIOD, "phase": 0.1}}
SPINNER = {"type": "capsule", "fromto": [1, -1, 0.5, 1, 1, 0.5], "radius": 0.1,  # a bar turning 360 deg/s about its middle
           "motion": {"type": "oscillate", "axis": [1, 0, 0], "amplitude": 0, "period": 1, "spin": 360}}
TRAVELLER = {"type": "box", "pos": [0, 0, 0.5], "size": [0.5, 0.3, 0.5], "euler": [0, 0, 20],  # a spinning box back and forth along x
             "motion": {"type": "path", "points": [[-5, 0, 0], [5, 0, 0]], "period": 3, "phase": 0.1, "spin": 90}}
WIDE = {"type": "box", "pos": [0, 0, 0.5], "size": [20, 20, 1]}


def chase(obs, _):
    """Steer at the region's centre, slowing within 4 m of it (the drifting region outruns a gentler gain)."""
    g = obs["goal"][:2]
    return (g / max(float(np.linalg.norm(g)), 1e-6) * min(1.0, float(np.linalg.norm(g)) / 4)).astype(np.float32)


AWAY = constant(0, -1)  # rolls off to -y: out of the drifting region
# id: (level, policy, env kwargs, expected outcome)
SCENARIOS = {
    "drift_region_leaves_a_still_ball": (bundled("survive_drift"), constant(0, 0), {}, "left_region"),
    "drift_region_chased": (bundled("survive_drift"), chase, {"time_limit_s": 8}, "success"),
    "sweepers_hit_a_still_ball": (bundled("survive_sweepers"), constant(0, 0), {}, "hazard"),
    "sweepers_dodged_to_time_up": (bundled("survive_sweepers"), constant(0, 0), {"time_limit_s": 1}, "success"),
    "generated_drift_and_hazards_hit_shaped": (DRIFT_AND_HAZARDS, constant(0, 0), {"shaped": True}, "hazard"),
    "generated_drift_and_hazards_chased": (DRIFT_AND_HAZARDS, chase, {"time_limit_s": 8}, "success"),
    "fast_hazard_crosses_a_still_ball_between_steps": (make("survive", region=WIDE, hazards=[FAST]), constant(0, 0), {}, "hazard"),
    "spinning_bar_sweeps_a_still_ball": (make("survive", region=WIDE, hazards=[SPINNER]), constant(0, 0), {}, "hazard"),
    "box_hazard_travels_a_path_and_spins": (make("survive", region=WIDE, hazards=[TRAVELLER]), constant(0, 0), {}, "hazard"),
    "shaped_goal_follows_the_drifting_region": (bundled("survive_drift"), AWAY, {"shaped": True}, "left_region"),
}


def mover_geoms(env):
    """name -> (geom id) of each moving marker in the CPU env's model."""
    m = env.core.model
    return {name: m.geom(name).id for name, _ in env.level.movers()}


def record(lv, policy, kw):
    """One CPU episode: the reset, then per step the core state, what step returns and each mover's world pose."""
    env = HamsterballEnv(levels=[lv], tasks=["survive"], **kw)
    obs, info = env.reset(seed=0)
    geoms = mover_geoms(env)
    d, m = env.core.data, env.core.model
    poses = lambda: {n: (m.geom_pos[g].copy(), m.geom_quat[g].copy(), d.geom_xpos[g].copy(), d.geom_xmat[g].copy())  # noqa: E731
                     for n, g in geoms.items()}
    ep = {"reset_obs": obs, "reset_poses": poses(), "steps": []}
    for i in range(env._max_steps + 5):
        obs, r, term, trunc, info = env.step(policy(obs, i))
        ep["steps"].append((info["core_state"], obs, r, term, trunc, info, poses()))
        if term or trunc:
            break
    last = ep["steps"][-1]
    ep["outcome"] = "success" if last[5]["success"] else last[5]["failure_reason"] or ("truncated" if last[4] else "running")
    ep["names"] = list(geoms)
    ep["capsules"] = {name for name, mk in env.level.movers() if mk["type"] == "capsule"}
    return ep


def warp_poses(venv, w, names):
    """The same poses in world w of the Warp env: the slot geom of each mover's marker row. The region is row 0, hazard
    i the row after the task's targets (the region only)."""
    e = int(venv.core.entry.numpy()[w])
    rows = venv.row_geom.numpy()[e * venv.markers.stride:]
    model, data = venv.core.model, venv.core.data
    out = {}
    for name in names:
        row = 0 if name.endswith("region") else 1 + int(name.removeprefix("hazard"))
        g = int(rows[row])
        out[name] = (model.geom_pos.numpy()[w, g], model.geom_quat.numpy()[w, g], data.geom_xpos.numpy()[w, g],
                     data.geom_xmat.numpy()[w, g])
    return out


def mat(q):
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def same_poses(got, want, capsules, where):
    """Positions equal, orientations equal; a capsule's orientation only up to its axis's sign (the bank's table points
    a capsule's axis from the fromto's start to its end, the compiler's the other way: the same capsule). The geom's
    quaternion must be the one its matrix is."""
    for name, (pos, quat, xpos, xmat) in want.items():
        g_pos, g_quat, g_xpos, g_xmat = got[name]
        for a, b, what in ((g_pos, pos, "geom_pos"), (g_xpos, xpos, "geom_xpos")):
            np.testing.assert_allclose(a, b, atol=1e-5, err_msg=f"{where} {name} {what}")
        np.testing.assert_allclose(mat(g_quat), g_xmat.reshape(3, 3), atol=1e-5, err_msg=f"{where} {name} quat vs xmat")
        if name in capsules:
            axis, want_axis = g_xmat.reshape(3, 3)[:, 2], xmat.reshape(3, 3)[:, 2]
            assert abs(float(axis @ want_axis)) > 1 - 1e-5, f"{where} {name} capsule axis {axis} against {want_axis}"
        else:
            np.testing.assert_allclose(g_xmat.reshape(3, 3), xmat.reshape(3, 3), atol=1e-5, err_msg=f"{where} {name} geom_xmat")


def replay(venv, eps, labels):
    """Feed each world the core states of its CPU episode; compare observations, rewards, flags, codes and the movers'
    poses. Returns the worst observation, reward and pose differences."""
    worst = worst_r = worst_pose = 0.0
    for w, ep in enumerate(eps):
        same_poses(warp_poses(venv, w, ep["names"]), ep["reset_poses"], ep["capsules"], f"{labels[w]} reset")
    for t in range(max(len(e["steps"]) for e in eps)):
        states = np.stack([e["steps"][min(t, len(e["steps"]) - 1)][0] for e in eps])
        obs, reward, terminated, truncated, info = venv.step_to(states)
        for w, ep in enumerate(eps):
            if t >= len(ep["steps"]):
                continue
            cpu_obs, cpu_r, cpu_term, cpu_trunc, cpu_info, cpu_poses = ep["steps"][t][1:]
            where = f"{labels[w]} step {t}"
            assert bool(terminated[w]) == cpu_term and bool(truncated[w]) == cpu_trunc, where
            assert bool(info["success"][w]) == cpu_info["success"], where
            assert int(info["failure_reason"][w]) == CODES[cpu_info["failure_reason"]], where
            done = cpu_term or cpu_trunc
            got = info["final_obs"] if done else obs  # a finished world has already reset
            for k in ("state", "goal"):
                g = got[k][w].cpu().numpy()
                np.testing.assert_allclose(g, cpu_obs[k], **OBS, err_msg=f"{where} {k}")
                worst = max(worst, float(np.abs(g - cpu_obs[k]).max()))
            np.testing.assert_allclose(float(reward[w]), cpu_r, **OBS, err_msg=where)
            worst_r = max(worst_r, abs(float(reward[w]) - cpu_r))
            for term_name, v in cpu_info["reward_terms"].items():
                np.testing.assert_allclose(float(info["reward_terms"][term_name][w]), v, **OBS, err_msg=f"{where} {term_name}")
            if not done:  # a finished world's markers are already at its next episode's start
                got_poses = warp_poses(venv, w, ep["names"])
                same_poses(got_poses, cpu_poses, ep["capsules"], where)
                worst_pose = max([worst_pose, *(float(np.abs(got_poses[n][k] - cpu_poses[n][k]).max()) for n in cpu_poses for k in (0, 2))])
    return worst, worst_r, worst_pose


@pytest.mark.parametrize("name", SCENARIOS)
def test_step_to_matches_the_cpu_env_on_recorded_episodes_with_moving_markers(name):
    lv, policy, kw, want = SCENARIOS[name]
    ep = record(lv, policy, kw)
    assert ep["outcome"] == want  # the scenario does what it says
    venv = warp_env(lv, "survive", 1, **kw)
    obs, _ = venv.reset(seed=0)
    for k in ("state", "goal"):
        np.testing.assert_allclose(obs[k][0].cpu().numpy(), ep["reset_obs"][k], **OBS, err_msg=f"reset {k}")
    worst = replay(venv, [ep], [name])
    print(f"{name}: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}, mover pose {worst[2]:.3g}")


def test_episodes_on_different_levels_of_one_bank_replay_together():
    """Worlds forced onto a drifting region, moving hazards, a generated level with both, and a still level in one
    padded bank: each replays its own CPU episode, so padding and the other worlds' movers change nothing."""
    cases = [("drift", bundled("survive_drift"), constant(0, 0)), ("sweepers", bundled("survive_sweepers"), constant(0, 0)),
             ("generated", DRIFT_AND_HAZARDS, constant(0, 0)), ("still", bundled("first_survive"), constant(0, 0))]
    eps = [record(lv, policy, {}) for _, lv, policy in cases]
    from hamsterball.gpu.vector_env import WarpVectorEnv

    venv = WarpVectorEnv(len(cases), levels=[lv for _, lv, _ in cases], tasks=["survive"], device=DEVICE)
    venv.reset(seed=0, options={"level_index": torch.arange(len(cases))})
    worst = replay(venv, eps, [name for name, *_ in cases])
    print(f"mixed bank: worst observation difference {worst[0]:.3g}, reward {worst[1]:.3g}, mover pose {worst[2]:.3g}")


def test_the_fast_hazard_scenario_jumps_the_ball_between_env_steps():
    """The scenario's hit comes from the relative path, not from either end: at the last two steps the hazard is on
    opposite sides of the still ball, each farther than the touch distance (hazard radius plus shell radius)."""
    ep = record(make("survive", region=WIDE, hazards=[FAST]), constant(0, 0), {})
    assert ep["outcome"] == "hazard"
    times = [0.0] + [s[0][0] for s in ep["steps"]]  # core state 0 is the sim time
    x = [4 * np.sin(2 * np.pi * (t / PERIOD + 0.1)) for t in times[-2:]]
    assert x[0] * x[1] < 0 and min(abs(x[0]), abs(x[1])) > 0.3 + 0.5, x


def test_stepping_moves_each_worlds_markers_with_its_own_sim_time():
    """Through `step` (live physics) and then `step_to` with the second world's clock moved on: the hazard's geom sits
    at its authored pose plus `motion.pose` at that world's sim time."""
    from hamsterball.core.motion import pose

    lv = bundled("survive_sweepers")
    venv = warp_env(lv, "survive", 2, time_limit_s=10)
    venv.reset(seed=0)
    hazard, g = lv.hazards[0], int(venv.row_geom.numpy()[1])  # survive's region is row 0, the first hazard row 1

    def check():
        t = venv.core.data.time.numpy()
        got = venv.core.model.geom_pos.numpy()[:, g]
        for w in range(2):
            np.testing.assert_allclose(got[w], np.array(hazard["pos"]) + pose(hazard["motion"], float(t[w]))[0], atol=1e-5)
        return t

    for _ in range(5):
        venv.step(torch.zeros(2, 2))
    t = check()
    assert abs(t[0] - 5 * FRAME_SKIP * DT) < 1e-4
    states = venv.core.states().copy()
    states[1, 0] += 0.7  # a state's first number is its sim time
    venv.step_to(states)
    assert abs(check()[1] - t[0] - 0.7) < 1e-4
