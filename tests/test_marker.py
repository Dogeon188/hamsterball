"""The swept marker test (spec §9): a marker triggers when the shell centre's path over one env step crosses it."""

import pytest

from hamsterball.core.marker import crosses

SPHERE = {"type": "sphere", "pos": [0.0, 5.0, 0.5], "radius": 0.2}
BOX = {"type": "box", "pos": [0.0, 5.0, 0.5], "size": [2.0, 0.1, 1.0], "euler": [0.0, 0.0, 90.0]}  # a thin wall along y
CAPSULE = {"type": "capsule", "fromto": [-1.0, 5.0, 0.5, 1.0, 5.0, 0.5], "radius": 0.2}


@pytest.mark.parametrize("marker", [SPHERE, CAPSULE])
def test_a_step_that_jumps_clean_over_a_marker_still_triggers_it(marker):
    assert crosses(marker, [0, 4, 0.5], [0, 6, 0.5])
    assert not crosses(marker, [0, 4, 0.5], [0, 4.7, 0.5])
    assert not crosses(marker, [0, 6, 0.5], [0, 7, 0.5])


def test_a_rotated_box_triggers_by_its_own_frame():
    # turned 90 degrees about z, the 4 m x 0.2 m slab spans y 3..7 and x -0.1..0.1
    assert crosses(BOX, [-1, 6.5, 0.5], [1, 6.5, 0.5])
    assert not crosses(BOX, [-1, 7.5, 0.5], [1, 7.5, 0.5])
    assert not crosses(BOX, [0.5, 3, 0.5], [0.5, 7, 0.5])


def test_at_reset_the_segment_is_a_point():
    assert crosses(SPHERE, [0.1, 5, 0.5], [0.1, 5, 0.5])
    assert crosses(BOX, [0, 6, 0.5], [0, 6, 0.5])
    assert crosses(CAPSULE, [0.9, 5, 0.6], [0.9, 5, 0.6])
    assert not crosses(CAPSULE, [1.3, 5, 0.5], [1.3, 5, 0.5])


# The touch rule: with reach R (the shell radius), a marker triggers once the shell's outer sphere reaches it.
R, HAIR = 0.5, 1e-4


@pytest.mark.parametrize("marker", [SPHERE, CAPSULE])
def test_a_round_marker_triggers_within_its_radius_plus_reach(marker):
    edge = 0.2 + R  # radius + reach, straight up from the centre (sphere) or axis (capsule)
    assert crosses(marker, [0, 5, 0.5 + edge - HAIR], [0, 5, 0.5 + edge - HAIR], reach=R)
    assert not crosses(marker, [0, 5, 0.5 + edge + HAIR], [0, 5, 0.5 + edge + HAIR], reach=R)


CUBE = {"type": "box", "pos": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "euler": [0.0, 0.0, 0.0]}


@pytest.mark.parametrize("out", [(1, 0, 0), (1, 1, 0), (1, 1, 1)], ids=["face", "edge", "corner"])
def test_a_box_triggers_within_reach_of_its_face_edge_and_corner(out):
    import numpy as np

    surface, n = np.array(out, float), np.array(out, float) / np.linalg.norm(out)  # outward from that feature
    near, far = surface + (R - HAIR) * n, surface + (R + HAIR) * n
    assert crosses(CUBE, near, near, reach=R)
    assert not crosses(CUBE, far, far, reach=R)


def test_a_box_corner_is_rounded_not_grown_per_axis():
    # 0.6 m out along the corner diagonal: within 0.5 m of the corner on every axis, but 0.6 m from the box
    p = [1 + 0.6 / 3**0.5] * 3
    assert not crosses(CUBE, p, p, reach=R)


@pytest.mark.parametrize("marker", [SPHERE, CAPSULE, BOX])
def test_a_step_passing_within_reach_mid_step_triggers(marker):
    # from x -3 to 3 along a line 0.65 m above the marker's top (z 0.7 round, 1.5 box): both ends far out of reach
    top = 1.5 if marker is BOX else 0.7
    p0, p1 = [-3, 5, top + R - 0.05], [3, 5, top + R - 0.05]
    assert crosses(marker, p0, p1, reach=R)
    assert not crosses(marker, p0, p0, reach=R) and not crosses(marker, p1, p1, reach=R)
    assert not crosses(marker, p0, p1)  # reach 0: the centre test misses


def test_a_moving_box_with_reach_is_tested_on_the_relative_path():
    from hamsterball.core.motion import load

    box = {"type": "box", "pos": [0, 0, 0], "size": [0.3, 0.3, 0.3], "euler": [0, 0, 0],
           "motion": load({"type": "path", "points": [[-1, 0, 0], [1, 0, 0]], "period": 4})}  # x -1 at t 0, +1 at t 2
    p = [0, 0.7, 0]  # still, 0.4 m beside the box as it passes
    assert not crosses(box, p, p, t=0, reach=R) and not crosses(box, p, p, t=2, reach=R)
    assert crosses(box, p, p, t=2, t0=0, reach=R)
    assert not crosses(box, p, p, t=2, t0=0)
