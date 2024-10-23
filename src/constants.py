"""
Some constants used in the project

such as all the possible terminal states of the game
"""

from action import Coordinate, LPiecePosition, NeutralPiecePosition, Orientation
from grid import Grid


def __grid_swap_red_blue(grid: Grid) -> Grid:
    return Grid._new_with(grid.blue_position, grid.red_position, grid.neutral_positions)


# States, Red to move, where Blue has won
# Note: although elsewhere red is considered to be the first player, here red is just "the player to move"
TERMINAL_STATES = [
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 2), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(3, 1)),
            NeutralPiecePosition(Coordinate(4, 4)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 2), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(4, 1)),
            NeutralPiecePosition(Coordinate(4, 4)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 2), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(4, 2)),
            NeutralPiecePosition(Coordinate(4, 4)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 3), Orientation.WEST),
        (
            NeutralPiecePosition(Coordinate(1, 2)),
            NeutralPiecePosition(Coordinate(4, 3)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 3), Orientation.WEST),
        (
            NeutralPiecePosition(Coordinate(1, 2)),
            NeutralPiecePosition(Coordinate(4, 4)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 2), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(3, 3)),
            NeutralPiecePosition(Coordinate(4, 4)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 3), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(2, 1)),
            NeutralPiecePosition(Coordinate(1, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 3), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(3, 1)),
            NeutralPiecePosition(Coordinate(1, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(2, 3), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(4, 3)),
            NeutralPiecePosition(Coordinate(1, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(4, 3), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(2, 1)),
            NeutralPiecePosition(Coordinate(1, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(4, 3), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(3, 1)),
            NeutralPiecePosition(Coordinate(1, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 3), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(2, 3)),
            NeutralPiecePosition(Coordinate(1, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 2), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 3), Orientation.SOUTH),
        (
            NeutralPiecePosition(Coordinate(3, 1)),
            NeutralPiecePosition(Coordinate(4, 2)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(1, 2), Orientation.NORTH),
        LPiecePosition(Coordinate(4, 3), Orientation.NORTH),
        (
            NeutralPiecePosition(Coordinate(3, 1)),
            NeutralPiecePosition(Coordinate(1, 3)),
        ),
    ),
    Grid._new_with(
        LPiecePosition(Coordinate(2, 4), Orientation.NORTH),
        LPiecePosition(Coordinate(3, 3), Orientation.EAST),
        (
            NeutralPiecePosition(Coordinate(2, 2)),
            NeutralPiecePosition(Coordinate(1, 3)),
        ),
    ),
]


def is_terminal_state(grid: Grid, red_to_move: bool) -> bool:
    """
    Returns True if the grid is a terminal state
    """
    if red_to_move:
        return grid.normalize()[0] in TERMINAL_STATES
    else:
        return __grid_swap_red_blue(grid).normalize()[0] in TERMINAL_STATES
