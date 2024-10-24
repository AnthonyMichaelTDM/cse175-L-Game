"""
Tests for the grid module.
"""

from action import Coordinate, LPiecePosition, NeutralPiecePosition, Orientation
from grid import Grid, GridCell


def test_cell_chars():
    assert str(GridCell.RED) == "+"
    assert str(GridCell.BLUE) == "x"
    assert str(GridCell.EMPTY) == "."
    assert str(GridCell.NEUTRAL) == "#"


def test_render_initial_grid():
    grid = Grid()
    expected = """# + + .
. x + .
. x + .
. x x #"""
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(2, 0)
    assert grid.red_position.orientation == Orientation.WEST
    assert grid.blue_position.corner == Coordinate(1, 3)
    assert grid.blue_position.orientation == Orientation.EAST


def test_rotate_grid_1():
    grid = Grid()
    grid = grid.rotate(1)
    expected = """. . . #
x x x +
x + + +
# . . ."""
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(3, 2)
    assert grid.red_position.orientation == Orientation.NORTH
    assert grid.blue_position.corner == Coordinate(0, 1)
    assert grid.blue_position.orientation == Orientation.SOUTH

    assert grid == Grid().rotate(-3)


def test_rotate_grid_2():
    grid = Grid()
    grid = grid.rotate(2)
    expected = """# x x .
. + x .
. + x .
. + + #"""
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(1, 3)
    assert grid.red_position.orientation == Orientation.EAST
    assert grid.blue_position.corner == Coordinate(2, 0)
    assert grid.blue_position.orientation == Orientation.WEST

    assert grid == Grid().rotate(-2)


def test_rotate_grid_3():
    grid = Grid()
    grid = grid.rotate(3)
    expected = """. . . #
+ + + x
+ x x x
# . . ."""
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(0, 1)
    assert grid.red_position.orientation == Orientation.SOUTH
    assert grid.blue_position.corner == Coordinate(3, 2)
    assert grid.blue_position.orientation == Orientation.NORTH

    assert grid == Grid().rotate(-1)


def test_rotate_grid_4():
    grid = Grid()
    grid = grid.rotate(4)
    expected = """# + + .
. x + .
. x + .
. x x #"""
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(2, 0)
    assert grid.red_position.orientation == Orientation.WEST
    assert grid.blue_position.corner == Coordinate(1, 3)
    assert grid.blue_position.orientation == Orientation.EAST

    assert grid == Grid().rotate(0)


def test_normalize_grid():
    grid = Grid()
    grid, n, mirrored = grid.normalize()
    expected = """# . . .
+ x x x
+ + + x
. . . #"""

    assert n == 1
    assert mirrored
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(0, 2)
    assert grid.red_position.orientation == Orientation.NORTH
    assert grid.blue_position.corner == Coordinate(3, 1)
    assert grid.blue_position.orientation == Orientation.SOUTH


def test_new_with():
    grid = (
        Grid._new_with(
            LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
            LPiecePosition(Coordinate(2, 2), Orientation.WEST),
            (
                NeutralPiecePosition(Coordinate(0, 1)),
                NeutralPiecePosition(Coordinate(3, 2)),
            ),
        ),
    )
    expected = """. . x .
# . x .
+ x x #
+ + + ."""

    grid = Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 1), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(2, 2)),
            NeutralPiecePosition(Coordinate(3, 3)),
        ),
    )
    expected = """. . x .
x x x .
+ . # .
+ + + #"""
    assert grid.render() == expected
