"""Pixel views (spec §11): what each view sees, through the env's observation."""

import math
import re
import os
from importlib.resources import files
import subprocess
import sys
import threading

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from hamsterball.core import views
from hamsterball.core import Core
from hamsterball.core.model import HAM, M_SHELL, SHELL
from hamsterball.core.level import Level
from test_env import level, make

VIEWS = ("chase", "follow", "first_person")


def goal_x(img):
    """Mean column of the goal's green pixels, as a fraction of the width; None if the goal isn't in view."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero((g > r + 40) & (g > b + 40))
    return None if len(xs) == 0 else xs.mean() / img.shape[1]


def test_views_are_uint8_rgb_at_view_size():
    env = make(views=VIEWS, view_size=(48, 64))
    obs, _ = env.reset(seed=0)
    for o in (obs, env.step(np.zeros(2, np.float32))[0]):
        for v in VIEWS:
            assert o[v].shape == (48, 64, 3) and o[v].dtype == np.uint8
            assert env.observation_space[v].contains(o[v])
    env.close()


def test_views_render_wider_and_taller_than_the_games_window():
    env = make(views=("chase",), view_size=(800, 1600))  # past the 1280 x 720 offscreen buffer the game needs
    obs, _ = env.reset(seed=0)
    assert obs["chase"].shape == (800, 1600, 3)
    env.close()


def test_chase_and_first_person_face_the_facing():
    # facing 0 faces +y, where the goal is: centred
    obs, _ = make(levels=[level(0.0)], views=VIEWS).reset(seed=0)
    assert abs(goal_x(obs["chase"]) - 0.5) < 0.05
    assert abs(goal_x(obs["first_person"]) - 0.5) < 0.05
    # facing 30 deg turns the view left, so the goal sits right of centre
    obs, _ = make(levels=[level(30.0)], views=VIEWS).reset(seed=0)
    assert goal_x(obs["chase"]) > 0.6
    assert goal_x(obs["first_person"]) > 0.6


def test_chase_and_first_person_follow_the_camera_pitch_and_follow_keeps_its_own():
    env = make(camera_action=True, views=VIEWS)
    env.reset(seed=0)

    def pitch(v):  # degrees below the horizon that the view's camera looks
        d = env.unwrapped.core.data
        fwd = -d.cam_xmat[d.model.camera(v).id].reshape(3, 3)[:, 2]  # a MuJoCo camera looks along its -z
        return math.degrees(math.asin(-fwd[2]))

    assert (pitch("chase"), pitch("first_person"), pitch("follow")) == pytest.approx((15, 0, 35))
    for _ in range(2):  # 15 deg an env step: 30 deg further down
        env.step(np.array([0, 0, 0, 1], np.float32))
    assert (pitch("chase"), pitch("first_person"), pitch("follow")) == pytest.approx((45, 30, 35))
    env.close()


def test_follow_keeps_the_level_yaw():
    lv = level(30.0)  # follow_yaw_deg 0: the follow view still looks along +y, at the goal
    obs, _ = make(levels=[lv], views=VIEWS).reset(seed=0)
    assert abs(goal_x(obs["follow"]) - 0.5) < 0.05
    lv.follow_yaw_deg = 30.0  # now the level turns the follow view left, whatever the facing
    obs, _ = make(levels=[level(0.0)], views=VIEWS).reset(seed=0)
    obs30, _ = make(levels=[lv], views=VIEWS).reset(seed=0)
    assert abs(goal_x(obs["follow"]) - 0.5) < 0.05 and goal_x(obs30["follow"]) > 0.6


def test_geom_groups_per_view():
    # level 0, outer sphere 1, hamster 2, inner tiles 3. Pixels can't show this: from inside, back-face culling
    # already drops the shell and the hamster, so the option is checked directly.
    assert list(views.option("chase").geomgroup) == list(views.option("follow").geomgroup) == [1, 1, 1, 0, 0, 0]
    assert list(views.option("first_person").geomgroup) == [1, 0, 0, 0, 0, 0]
    for v in VIEWS:  # hitboxes 4: off by default, so env pixels never show them; the flag turns only them on
        on = list(views.option(v).geomgroup)
        on[4] = 1
        assert list(views.option(v, hitboxes=True).geomgroup) == on


def test_renders_on_the_thread_that_steps_the_env():
    env = make(views=("chase",))
    out = []
    t = threading.Thread(target=lambda: out.append(env.reset(seed=0)[0]["chase"]))
    t.start()
    t.join()
    assert goal_x(out[0]) is not None
    env.close()


def test_mujoco_gl_defaults_to_egl_on_linux_and_stays_unset_on_macos():
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    code = "import hamsterball, os; print(os.environ.get('MUJOCO_GL'))"
    got = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert got == {"linux": "egl", "darwin": "None"}[sys.platform]


def test_glfw_is_refused_for_env_views(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    with pytest.raises(RuntimeError, match="glfw"):
        make(views=("chase",))


def test_obs_state_false_returns_pixels_only_with_state_and_goal_in_info():
    env = make(views=("chase",), obs_state=False)
    obs, info = env.reset(seed=0)
    assert set(obs) == set(env.observation_space.spaces) == {"chase"}
    assert info["state"].shape == (21,) and info["goal"].shape == (10,)
    obs, *_, info = env.step(np.zeros(2, np.float32))
    assert set(obs) == {"chase"} and {"state", "goal"} <= info.keys()


def test_first_person_sees_the_ground_at_its_feet_on_a_large_level():
    # the near clip plane scales with the model's extent; on a 30 m level it once cut the ground below the eye
    env = gym.make("Hamsterball-v1", tasks=["unlock_goal"], levels=["locked_goal"], views=("first_person",))
    obs, _ = env.reset(seed=0)
    assert obs["first_person"][-1].min() > 0


def test_views_survive_model_cache_eviction():
    # model_cache=1: every switch evicts the other level's model and its renderer, and both come back on demand
    env = gym.make("Hamsterball-v1", tasks=["reach_goal", "unlock_goal"], levels=["first_goal", "locked_goal"], views=("chase",), model_cache=1)
    for name, task in [("first_goal", "reach_goal"), ("locked_goal", "unlock_goal"), ("first_goal", "reach_goal")]:
        obs, _ = env.reset(options={"level": Level.load(files("hamsterball") / f"levels/{name}.yaml"), "task": task})
        assert len(env.unwrapped._cores) == 1
        if name == "first_goal":
            assert abs(goal_x(obs["chase"]) - 0.5) < 0.05
    env.close()


PARTS = ("tail", "ear_left", "ear_right")


def part_offsets(env, facing):
    """Each part's centre from the hamster centre, as (right, forward, up) in the facing frame."""
    m, d = env.unwrapped.core.model, env.unwrapped.core.data
    right = np.array([math.cos(facing), math.sin(facing), 0.0])
    fwd = np.array([-math.sin(facing), math.cos(facing), 0.0])  # facing 0 faces +y
    return {p: (lambda r: (r @ right, r @ fwd, r[2]))(d.geom_xpos[m.geom(p).id] - d.xpos[HAM]) for p in PARTS}


def test_the_tail_sits_behind_and_the_ears_on_top_whatever_the_facing():
    env = make(camera_action=True, views=("chase",))  # rendering carries the mocap pose to the geoms
    env.reset(seed=0)
    for facing in (0.0, math.radians(60)):  # 60 deg: after two full turns left
        o = part_offsets(env, facing)
        assert abs(o["tail"][0]) < 1e-9 and o["tail"][1] < -0.12  # straight behind, out past the hamster's radius
        (lx, ly, lz), (rx, ry, rz) = o["ear_left"], o["ear_right"]
        assert lx < 0 < rx and abs(lx + rx) < 1e-9  # one each side
        assert ly == pytest.approx(ry) and ly > 0 and lz == pytest.approx(rz) and lz > 0.08  # forward, on top
        for _ in range(2):
            env.step(np.array([0, 0, 1, 0], np.float32))
    env.close()


def test_the_tail_and_ears_show_in_chase_and_follow_only_and_never_touch_the_physics():
    env = make()
    env.reset(seed=0)
    m = env.unwrapped.core.model
    for p in PARTS:
        g = m.geom(p).id
        group = m.geom_group[g]
        assert views.option("chase").geomgroup[group] and views.option("follow").geomgroup[group]
        assert not views.option("first_person").geomgroup[group]
        assert (m.geom_contype[g], m.geom_conaffinity[g], m.body_mass[m.geom_bodyid[g]]) == (0, 0, 0)
        assert m.body_mocapid[m.geom_bodyid[g]] >= 0  # a mocap body: welded to the world, so nothing pushes it



def test_hitbox_geoms_are_the_shell_centre_and_the_kill_plane_and_never_touch_the_physics():
    lv = level()
    lv.kill_plane_z = -3.5
    m = Core(level=lv).model
    dot, plane = m.geom("hitbox_centre").id, m.geom("hitbox_kill_plane").id
    assert m.geom_bodyid[dot] == SHELL and list(m.geom_pos[dot]) == [0, 0, 0] and m.geom_rgba[dot][3] == 1
    assert m.geom_bodyid[plane] == 0 and m.geom_pos[plane][2] == -3.5 and m.geom_rgba[plane][3] < 1
    for g in (dot, plane):
        assert (m.geom_group[g], m.geom_contype[g], m.geom_conaffinity[g]) == (4, 0, 0)
    assert m.body_mass[SHELL] == pytest.approx(M_SHELL)  # the dot weighs nothing
    assert (m.body("shell").id, m.body("hamster").id) == (SHELL, HAM) == (1, 2)


def test_hitbox_geoms_change_no_trajectory(monkeypatch):
    def run():
        core = Core(level=Level.load(files("hamsterball") / "levels/survive_seesaw.yaml"))
        for a in ([1, 0], [0.3, 1], [-1, 0.5]):
            for _ in range(20):
                core.step(np.array(a, np.float32))
        return core.state()

    with_hitboxes = run()
    compile_ = mujoco.MjModel.from_xml_string
    monkeypatch.setattr(mujoco.MjModel, "from_xml_string",
                        lambda xml: compile_(re.sub(r'<geom name="hitbox_[^>]*/>', "", xml)))
    without = run()
    assert len(with_hitboxes) == len(without) and np.array_equal(with_hitboxes, without)
