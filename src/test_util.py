"""
Tests for the util module.
"""

from action import Orientation
from util import Transform, TransformSeries


def test_transform_series_methods():
    ts = TransformSeries([Transform.TRANSPOSE, Transform.MIRROR])

    assert len(ts) == 2
    assert ts[0] == Transform.TRANSPOSE
    assert ts[1] == Transform.MIRROR

    assert ts.merge(Transform.TRANSPOSE) == TransformSeries(
        [Transform.TRANSPOSE, Transform.MIRROR, Transform.TRANSPOSE]
    )
    assert ts.merge(Transform.MIRROR) == TransformSeries([Transform.TRANSPOSE])


def test_indexable_enum():
    assert Orientation.LENGTH() == 4

    assert int(Orientation.NORTH) == 0
    assert str(Orientation.NORTH) == "N"
    assert Orientation("N") == Orientation.NORTH
    assert Orientation.from_index(0) == Orientation.NORTH

    assert int(Orientation.EAST) == 1
    assert str(Orientation.EAST) == "E"
    assert Orientation("E") == Orientation.EAST
    assert Orientation.from_index(1) == Orientation.EAST

    assert int(Orientation.SOUTH) == 2
    assert str(Orientation.SOUTH) == "S"
    assert Orientation("S") == Orientation.SOUTH
    assert Orientation.from_index(2) == Orientation.SOUTH

    assert int(Orientation.WEST) == 3
    assert str(Orientation.WEST) == "W"
    assert Orientation("W") == Orientation.WEST
    assert Orientation.from_index(3) == Orientation.WEST

    # assert that it fails to initialize with an invalid value
    try:
        Orientation("Z")
        assert False
    except ValueError:
        assert True
    except:
        assert False
