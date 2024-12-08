"""
Some constants used in the project

such as all the possible terminal states of the game
"""

from action import Coordinate, LPiecePosition, NeutralPiecePosition, Orientation
from grid import Grid


def _grid_swap_red_blue(grid: Grid) -> Grid:
    return Grid._new_with(grid.blue_position, grid.red_position, grid.neutral_positions)


# States, Red to move, where Blue has won
# Note: although elsewhere red is considered to be the first player, here red is just "the player to move"
TERMINAL_STATES = [
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 1), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(2, 0)),
            NeutralPiecePosition(Coordinate(3, 3)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 1), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(3, 0)),
            NeutralPiecePosition(Coordinate(3, 3)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 1), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(3, 1)),
            NeutralPiecePosition(Coordinate(3, 3)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 2), Orientation.WEST),
        (
            NeutralPiecePosition(Coordinate(0, 1)),
            NeutralPiecePosition(Coordinate(3, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 2), Orientation.WEST),
        (
            NeutralPiecePosition(Coordinate(0, 1)),
            NeutralPiecePosition(Coordinate(3, 3)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 1), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(2, 2)),
            NeutralPiecePosition(Coordinate(3, 3)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(1, 2), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(1, 0)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(1, 2), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(2, 0)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(1, 2), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(3, 2)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 2), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(1, 0)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 2), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(2, 0)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 2), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(1, 2)),
            NeutralPiecePosition(Coordinate(0, 1)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 1), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 2), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(2, 0)),
            NeutralPiecePosition(Coordinate(3, 1)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(0, 1), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 2), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(2, 0)),
            NeutralPiecePosition(Coordinate(0, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 3), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 2), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(1, 1)),
            NeutralPiecePosition(Coordinate(0, 2)),
        ),
    ),
]


def is_losing_state(grid: Grid, red_to_move: bool = True) -> bool:
    """
    Returns True if the grid is a losing position for the player to move
    """
    if red_to_move:
        return grid.normalize() in TERMINAL_STATES
    else:
        return _grid_swap_red_blue(grid).normalize() in TERMINAL_STATES


def is_winning_state(grid: Grid, red_to_move: bool = True) -> bool:
    """
    Returns True if the grid is a winning position for the player to move
    """
    return is_losing_state(grid, not red_to_move)


def is_terminal_state(grid: Grid) -> bool:
    """
    Returns True if the grid is a terminal state
    """
    return is_losing_state(grid, True) or is_losing_state(grid, False)
