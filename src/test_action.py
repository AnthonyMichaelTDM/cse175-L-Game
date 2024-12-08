"""
Tests for the action module.
"""

from action import Orientation


def test_orientation_index():
    assert Orientation.NORTH.index() == 0
    assert Orientation.EAST.index() == 1
    assert Orientation.SOUTH.index() == 2
    assert Orientation.WEST.index() == 3


def test_orientation_int():
    assert int(Orientation.NORTH) == 0
    assert int(Orientation.EAST) == 1
    assert int(Orientation.SOUTH) == 2
    assert int(Orientation.WEST) == 3


def test_orientation_transpose():
    assert Orientation.NORTH.transpose() == Orientation.WEST
    assert Orientation.EAST.transpose() == Orientation.SOUTH
    assert Orientation.SOUTH.transpose() == Orientation.EAST
    assert Orientation.WEST.transpose() == Orientation.NORTH


def test_orientation_mirror():
    assert Orientation.NORTH.mirror() == Orientation.NORTH
    assert Orientation.EAST.mirror() == Orientation.WEST
    assert Orientation.SOUTH.mirror() == Orientation.SOUTH
    assert Orientation.WEST.mirror() == Orientation.EAST
