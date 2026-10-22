"""The compiled hamsterball model matches spec §3."""

import math

import mujoco
import numpy as np
import pytest

from hamsterball.core import Core


@pytest.fixture(scope="module")
def model():
    return Core().model


def shell_geoms(m):
    shell = m.body("shell").id
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == shell]


def test_shell_wall_is_320_fibonacci_tiles_carrying_the_shell_mass(model):
    tiles = [g for g in shell_geoms(model) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    assert len(tiles) == 320
    # the outer sphere is massless (next test), so the shell body's mass is the tiles'
    assert model.body_mass[model.body("shell").id] == pytest.approx(0.8)
    assert set(model.geom_contype[tiles]) == {2} and set(model.geom_conaffinity[tiles]) == {2}
    assert set(model.geom_group[tiles]) == {3}
    np.testing.assert_allclose(model.geom_solref[tiles], [[0.04, 2]] * 320)
    # tile 7 sits on the Fibonacci sphere at (R - T/2) n
    z = 1 - 2 * 7.5 / 320
    r, phi = math.sqrt(1 - z * z), 7 * math.pi * (3 - math.sqrt(5))
    np.testing.assert_allclose(model.geom_pos[tiles[7]], 0.485 * np.array([r * math.cos(phi), r * math.sin(phi), z]))
    side = 1.25 * math.sqrt(4 * math.pi * 0.25 / 320)
    np.testing.assert_allclose(model.geom_size[tiles[0]], [side / 2, side / 2, 0.015])


def test_outer_collider_is_a_massless_rolling_sphere(model):
    # the one colliding sphere: the hitbox view's centre dot is a contactless sphere too
    (outer,) = [g for g in shell_geoms(model) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]]
    assert model.geom_size[outer][0] == 0.5
    assert model.body_mass[model.body("shell").id] == pytest.approx(0.8)  # a solid 0.5 m sphere would add 524 kg
    assert model.geom_contype[outer] == 1 and model.geom_condim[outer] == 6 and model.geom_group[outer] == 1
    np.testing.assert_allclose(model.geom_friction[outer], [1, 0.05, 0.04])


def test_hamster_is_a_frictionless_sphere_with_an_eye(model):
    ham = model.body("hamster")
    (g,) = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ham.id]
    assert model.body_subtreemass[ham.id] == pytest.approx(2.4)
    assert model.geom_size[g][0] == 0.12 and model.geom_group[g] == 2
    assert model.geom_contype[g] == 2 and model.geom_conaffinity[g] == 2
    assert model.geom_condim[g] == 1 and model.geom_priority[g] == 1
    np.testing.assert_allclose(model.geom_solref[g], [0.04, 2])
    assert model.geom_margin[g] == 0.002
    assert model.site("hamster_eye").bodyid == ham.id
    np.testing.assert_allclose(model.site("hamster_eye").pos, 0)


def test_physics_step_is_one_six_hundredth_of_a_second(model):
    assert model.opt.timestep == 1 / 600
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
