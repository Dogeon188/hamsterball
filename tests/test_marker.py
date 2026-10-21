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
