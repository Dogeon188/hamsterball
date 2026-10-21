"""Feel regression: the core reproduces the spec §5 feel reference, measured as in the prototype that set it
(prototype/honest-physics: prototypes/honest_drive.py, measure() and contain_row(), at 1/600 s).

The run goes along +x, as the prototype's did: the Fibonacci tiles aren't isotropic, and running along +y
accelerates about 3% faster. The chase yaw is off, as the prototype's drive took a world-frame direction: on the
climb the shell first rolls back faster than 0.8 m/s, and the chase would swing the run direction off the slope.
"""

import math

import numpy as np
import pytest

from hamsterball.core import Core, step

STEPS_PER_S = 600
FULL, HALF, IDLE = np.array([0.0, 1.0]), np.array([0.0, 0.5]), np.zeros(2)


def new_core():
    core = Core()
    core.reset(yaw=-math.pi / 2)  # camera forward is +x
    return core


def run(core, until, action, each=None):
    while core.data.time < until - 0.5 / STEPS_PER_S:
        core.step(action, n=1)
        assert np.isfinite(core.data.qpos).all()
        if each:
            each()


def speed(core):
    return math.hypot(core.data.qvel[0], core.data.qvel[1])


def ham_offset(core):
    q = core.data.qpos
    return q[7:10] - q[0:3]


def lean_deg(core):
    u = ham_offset(core)
    return math.degrees(math.acos(-u[2] / np.linalg.norm(u)))


@pytest.fixture(scope="module")
def feel():
    """All of §5's measurements, one simulation each."""
    out = {}

    # acceleration is measured uncapped, as the prototype's accel() did
    patch = pytest.MonkeyPatch()
    patch.setattr(step, "CHASE_MIN_SPEED2", math.inf)
    with pytest.MonkeyPatch.context() as uncapped:
        uncapped.setattr(step, "DRAG_K", 0.0)
        core = new_core()
        run(core, 1.0, FULL)
        v1 = speed(core)
        run(core, 3.0, FULL)
        out["accel"] = (speed(core) - v1) / 2

    core = new_core()
    run(core, 3.0, HALF)
    out["v3_half"] = speed(core)

    core = new_core()
    leans = []
    run(core, 3.0, FULL, lambda: core.data.time > 2.0 and leans.append(lean_deg(core)))
    out["v3_full"], out["lean"] = speed(core), float(np.mean(leans))
    run(core, 20.0, FULL)
    out["top"] = speed(core)
    run(core, 21.0, IDLE)  # the hamster swings back to the bottom in the first second
    v1 = speed(core)
    run(core, 23.0, IDLE)
    out["coast"] = (v1 - speed(core)) / 2

    core = new_core()
    g = math.radians(20)
    core.model.opt.gravity[:] = [-9.81 * math.sin(g), 0, -9.81 * math.cos(g)]  # uphill is +x
    run(core, 3.0, FULL)
    out["climb"] = core.data.qpos[0]

    for vr in (0, 4, 8, 12):
        worst = 0.0
        for direction in [(0, 0, -1), (0, 0, 1), (1, 0, 0)]:
            core = new_core()
            d = core.data
            d.qpos[7:10] = d.qpos[0:3]
            d.qvel[6:9] = np.array(direction) * vr

            def track():
                nonlocal worst
                worst = max(worst, np.linalg.norm(ham_offset(core)) - 0.35)

            run(core, 1.0, IDLE, track)
        out[f"sink{vr}"] = worst
    patch.undo()
    print({k: round(float(v), 4) for k, v in out.items()})
    return out


# Spec §5 values. Measured on an M2 (see step.md, History): every row within 1.3%, so 2% catches a real drift.
@pytest.mark.parametrize(
    "key, expected",
    [
        ("accel", 4.06),
        ("v3_half", 5.6),
        ("v3_full", 9.6),
        ("lean", 54),
        ("top", 14.3),
        ("coast", 2.49),
        ("climb", 5.1),
    ],
)
def test_drive_feels_like_the_play_tested_prototype(feel, key, expected):
    assert feel[key] == pytest.approx(expected, rel=0.02)


# Not spec §5's 16 / 21 / 30 / 36 mm: the core reads fresh positions, so containment acts a step earlier than in the
# prototype, and the sink stays under the 20 mm hard stop plus one step's travel (see step.md, Known quirks).
@pytest.mark.parametrize("vr, expected_mm", [(0, 14), (4, 17), (8, 20), (12, 23)])
def test_containment_holds_a_thrown_hamster_like_the_prototype(feel, vr, expected_mm):
    assert feel[f"sink{vr}"] * 1000 == pytest.approx(expected_mm, abs=1.5)
