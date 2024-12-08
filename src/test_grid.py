"""
Tests for the grid module.
"""

from action import (
    Coordinate,
    LGameAction,
    LPiecePosition,
    NeutralPiecePosition,
    Orientation,
)
from cell import GridCell
from game import LGameState
from grid import Grid
from util import Transform, TransformSeries


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


def test_transpose_grid():
    grid = Grid()
    grid = grid.transpose()
    expected = """# . . .
+ x x x
+ + + x
. . . #"""

    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(0, 2)
    assert grid.red_position.orientation == Orientation.NORTH
    assert grid.blue_position.corner == Coordinate(3, 1)
    assert grid.blue_position.orientation == Orientation.SOUTH
    assert grid.transformations == TransformSeries([Transform.TRANSPOSE])


def test_flip_grid():
    grid = Grid()
    grid = grid.flip()
    expected = """. x x #
. x + .
. x + .
# + + ."""
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(2, 3)
    assert grid.red_position.orientation == Orientation.WEST
    assert grid.blue_position.corner == Coordinate(1, 0)
    assert grid.blue_position.orientation == Orientation.EAST
    assert grid.transformations == TransformSeries([Transform.FLIP])


def test_mirror_grid():
    grid = Grid()
    grid = grid.mirror()
    expected = """. + + #
. + x .
. + x .
# x x ."""
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(1, 0)
    assert grid.red_position.orientation == Orientation.EAST
    assert grid.blue_position.corner == Coordinate(2, 3)
    assert grid.blue_position.orientation == Orientation.WEST
    assert grid.transformations == TransformSeries([Transform.MIRROR])


def test_normalize_grid():
    grid = Grid()
    grid = grid.normalize()
    expected = """# . . .
+ x x x
+ + + x
. . . #"""

    assert grid.transformations == TransformSeries([Transform.TRANSPOSE])
    assert grid.render() == expected
    assert grid.red_position.corner == Coordinate(0, 2)
    assert grid.red_position.orientation == Orientation.NORTH
    assert grid.blue_position.corner == Coordinate(3, 1)
    assert grid.blue_position.orientation == Orientation.SOUTH


def test_normalize_denormalize_grid():
    grid = Grid()
    normalized_grid = grid.normalize()
    denormalized_grid = normalized_grid.unapply_transformations(
        normalized_grid.transformations
    )
    assert (
        grid == denormalized_grid
    ), "grid should be the same after normalizing and denormalizing"


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


def test_mask_checks():
    grid = Grid()
    """
    grid state:
    # + + .
    . x + .
    . x + .
    . x x #
    """

    some_valid_masks = [
        (
            LPiecePosition(Coordinate(3, 0), Orientation.SOUTH).grid_mask(GridCell.RED),
            GridCell.RED,
        ),
        (
            LPiecePosition(Coordinate(3, 2), Orientation.WEST).grid_mask(GridCell.RED),
            GridCell.RED,
        ),
        (
            LPiecePosition(Coordinate(0, 1), Orientation.EAST).grid_mask(GridCell.BLUE),
            GridCell.BLUE,
        ),
        (
            LPiecePosition(Coordinate(0, 3), Orientation.NORTH).grid_mask(
                GridCell.BLUE
            ),
            GridCell.BLUE,
        ),
        (NeutralPiecePosition(Coordinate(0, 1)).grid_mask(), GridCell.NEUTRAL),
    ]

    for mask, color in some_valid_masks:
        assert grid.is_mask_valid(
            mask, color
        ), f"mask: {mask} should be valid for {str(grid)}"

    some_invalid_masks = [
        (
            LPiecePosition(Coordinate(0, 2), Orientation.EAST).grid_mask(GridCell.RED),
            GridCell.RED,
        ),
        (
            LPiecePosition(Coordinate(3, 2), Orientation.SOUTH).grid_mask(GridCell.RED),
            GridCell.RED,
        ),
        (
            LPiecePosition(Coordinate(0, 0), Orientation.SOUTH).grid_mask(
                GridCell.BLUE
            ),
            GridCell.BLUE,
        ),
        (
            LPiecePosition(Coordinate(2, 3), Orientation.EAST).grid_mask(GridCell.BLUE),
            GridCell.BLUE,
        ),
        (NeutralPiecePosition(Coordinate(1, 0)).grid_mask(), GridCell.NEUTRAL),
    ]

    for mask, color in some_invalid_masks:
        assert not grid.is_mask_valid(
            mask, color
        ), f"mask: {mask} should be valid for {str(grid)}"


def test_normalize_and_denormalize_all_grids():
    # first, gather all the possible grid states
    from computer import ComputerAgent, defensive_heuristic

    class MockAgent(ComputerAgent):
        def __init__(self, id: int):
            super().__init__(id, 1, defensive_heuristic)

        def get_action(self, state: LGameState) -> LGameAction: ...

        def get_cache_info(self, id: int) -> dict[str, dict[str, int]]: ...

    DEPTH_LIMIT = 3

    initial_state = LGameState(
        (
            MockAgent(0),
            MockAgent(1),
        )
    ).normalize()

    frontier = [(initial_state, 0, 0)]  # (state, agent_id, depth)
    visited: set[Grid] = set()
    while len(frontier) > 0:
        (state, agent_id, depth) = frontier.pop()

        if depth >= DEPTH_LIMIT:
            continue

        visited.add(state.grid)

        for action in state.get_legal_actions(agent_id):
            new_state = state.generate_successor(action, agent_id)
            if new_state.grid not in visited:
                if depth + 1 < DEPTH_LIMIT:
                    frontier.append((new_state, 1 - agent_id, depth + 1))
                visited.add(new_state.grid)

    for grid in visited:
        grid = grid.unapply_transformations(grid.transformations)
        assert len(grid.transformations) == 0
        assert grid == grid.unapply_transformations(
            grid.transformations
        ), f"""
grid:
{str(grid)}
should be the same after denormalizing
"""
        normalized_grid = grid.normalize()
        normalized_red = normalized_grid.red_position
        assert normalized_red.orientation == Orientation.NORTH
        assert normalized_red.corner.x < 2

        assert grid == normalized_grid.unapply_transformations(
            normalized_grid.transformations
        ), f"""
grid:
{str(grid)}
should be the same after normalizing and denormalizing, but got
{str(normalized_grid.unapply_transformations(normalized_grid.transformations))}
instead
"""
