"""The scripted gait through the policy seam: `act(obs, run_vector) -> torques` on a Core with the articulated tier
(spec: articulated-tier, ticket 04). The feel reference is frozen like test_feel.py's, within a few percent."""

import math

import numpy as np
import pytest

from hamsterball import lowlevel
from hamsterball.core import Core
from hamsterball.core.articulated_tier import ArticulatedTier

FORWARD = np.array([0.0, 1.0])  # facing 0: the run direction +y, uphill in the climb


def new_core(tilt_deg=0.0):
    core = Core(ArticulatedTier())
    core.reset(facing=0.0)
    if tilt_deg:
        g = math.radians(tilt_deg)
        core.model.opt.gravity[:] = [0, -9.81 * math.sin(g), -9.81 * math.cos(g)]  # uphill is +y
    return core


def drive(core, policy, run_vector, seconds, each=None):
    """One policy step per env step (60 Hz); returns False if the torso tips past 0.3 up."""
    for _ in range(round(seconds * 60)):
        obs = core.tier.obs(core.model, core.data, core.facing)
        if obs["proprio"][25] < 0.3:
            return False
        core.step(policy.act(obs, run_vector))
        assert np.isfinite(core.data.qpos).all()
        if each:
            each(obs["proprio"])
    return True


def speed(core):
    return math.hypot(*core.data.qvel[:2])


def heading(proprio):
    """Degrees the torso's forward axis points left of the facing's +y."""
    return math.degrees(math.atan2(-proprio[20], proprio[21]))


def test_the_registry_looks_a_gait_up_by_name_and_an_unknown_name_raises():
    policy = lowlevel.get(lowlevel.DEFAULT)
    assert policy.name == lowlevel.DEFAULT
    assert policy.act(np.zeros(26, np.float32), np.zeros(2)).shape == (8,)
    with pytest.raises(KeyError):
        lowlevel.get("no_such_gait")


def test_a_zero_run_vector_stands_upright_with_its_feet_on_the_wall():
    core = new_core()
    policy = lowlevel.get(lowlevel.DEFAULT)
    tilt, feet = [], []
    assert drive(core, policy, np.zeros(2), 6, lambda p: (tilt.append(p[25]), feet.append(p[16:20].min())))
    assert min(tilt) > 0.9
    assert min(feet[60:]) == 1.0  # all four feet on the wall once settled
    assert speed(core) < 0.05


def test_a_full_forward_run_vector_pushes_the_shell_along_the_run_direction_for_20_s_without_tumbling():
    core = new_core()
    policy = lowlevel.get(lowlevel.DEFAULT)
    vy = []
    assert drive(core, policy, FORWARD, 20, lambda p: vy.append(core.data.qvel[1]))
    d = core.data
    assert np.mean(vy[-300:]) > 0.4  # about 3% of the abstract tier's 14.3 m/s top speed
    assert abs(d.qpos[0]) < 0.2 * d.qpos[1]  # and mostly straight


def test_the_zero_run_vector_holds_a_3_degree_slope():
    core = new_core(3.0)
    assert drive(core, lowlevel.get(lowlevel.DEFAULT), np.zeros(2), 10)


def test_feel_reference_of_the_scripted_trot():
    """Frozen with the gait: a registered gait's torques never change (a change registers a new name), so these
    hold to a few percent. Measured on flat ground, run vector (0, 1), shell speed along the run. The abstract tier
    reaches 14.3 m/s; the 20 deg climb is out of reach (the hamster would have to sit about 54 deg up the wall; see
    lowlevel.md), so the climb is recorded as a limit instead."""
    def vy_mean(rv, a, b):
        core, policy, vy = new_core(), lowlevel.get(lowlevel.DEFAULT), []
        assert drive(core, policy, rv, b, lambda p: vy.append(core.data.qvel[1]))
        return float(np.mean(vy[round(a * 60):round(b * 60)])), core.data.qpos[1]

    assert vy_mean(FORWARD, 0, 3)[1] == pytest.approx(1.237, rel=0.05)  # m travelled in the first 3 s
    assert vy_mean(FORWARD, 10, 20)[0] == pytest.approx(0.487, rel=0.05)  # top speed
    assert vy_mean(FORWARD * 0.5, 10, 20)[0] == pytest.approx(0.093, rel=0.10)  # half effort
    core = new_core(20.0)
    assert not drive(core, lowlevel.get(lowlevel.DEFAULT), FORWARD, 10)  # the climb: tumbles at once


@pytest.mark.parametrize("run_vector, facing_after", [((1.0, 0.0), -90.0), ((-1.0, 0.0), 90.0)])
def test_a_sideways_run_vector_turns_the_hamster_toward_it(run_vector, facing_after):
    core = new_core()
    policy = lowlevel.get(lowlevel.DEFAULT)
    last = []
    assert drive(core, policy, np.array(run_vector), 10, last.append)
    assert abs((heading(last[-1]) - facing_after + 180) % 360 - 180) < 35
