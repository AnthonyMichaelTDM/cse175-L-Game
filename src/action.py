"""
Code for actions in the game
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cell import GridCell
from util import IndexableEnum, Transformable


@dataclass(frozen=True, slots=True)
class Coordinate(Transformable):
    """
    An (x,y) coordinate on the game board (4x4 grid), where (1,1) is the top-left corner

    x is the column, and
    y is the row
    """

    x: int
    y: int

    def __add__(self, other: "Coordinate") -> "Coordinate":
        return Coordinate(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        return Coordinate(self.x - other.x, self.y - other.y)

    def transpose(self) -> "Coordinate":
        """
        Transpose the coordinate
        """
        return Coordinate(self.y, self.x)

    def flip(self) -> "Coordinate":
        """
        Flip the coordinate across the horizontal axis
        """
        return Coordinate(self.x, 3 - self.y)

    def mirror(self) -> "Coordinate":
        """
        Mirror the coordinate across the vertical axis
        """
        return Coordinate(3 - self.x, self.y)

    def is_in_bounds(self) -> bool:
        """
        Returns True if the coordinate is within the bounds of the 4x4 grid
        """
        return 0 <= self.x <= 3 and 0 <= self.y <= 3

    def to_index(self) -> tuple[int, int]:
        """
        convert into an index into the grid
        """
        return self.y, self.x


OrientationDirections = [
    Coordinate(0, -1),
    Coordinate(1, 0),
    Coordinate(0, 1),
    Coordinate(-1, 0),
]


class Orientation(IndexableEnum):
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"

    def transpose(self) -> "Orientation":
        """
        Transpose the orientation
        """
        MAPPING = [
            3,
            2,
            1,
            0,
        ]

        return Orientation.from_index(MAPPING[int(self)])

    def flip(self) -> "Orientation":
        """
        Flip the orientation across the horizontal axis
        """
        MAPPING = [
            2,
            1,
            0,
            3,
        ]

        return Orientation.from_index(MAPPING[int(self)])

    def mirror(self) -> "Orientation":
        """
        Mirror the orientation across the vertical axis
        """
        MAPPING = [
            0,
            3,
            2,
            1,
        ]

        return Orientation.from_index(MAPPING[int(self)])

    def direction(self) -> Coordinate:
        return OrientationDirections[int(self)]


@dataclass(frozen=True, slots=True)
class LPiecePosition(Transformable):
    """
    An L-piece position, consisting of a coordinate and an orientation

    the coordinate is the coordinate of the corner of the L-piece, and the orientation is the direction that the foot of the L is facing (relative to the corner)

    Ex: (1, 2), Orientation.NORTH represents an L-piece with the corner at (1, 2) and the foot facing north
    """

    corner: Coordinate
    orientation: Orientation

    def __post_init__(self):
        if not self.corner.is_in_bounds():
            raise ValueError("Corner must be in bounds")
        if not (self.corner + self.orientation.direction()).is_in_bounds():
            raise ValueError("Foot must be in bounds")

    def grid_mask(self, color: GridCell) -> np.ndarray:
        """
        Returns a 4x4 grid mask of the L-piece
        """
        grid = np.full((4, 4), GridCell.EMPTY)

        # color corner and foot
        corner = self.corner
        grid[corner.to_index()] = color
        grid[(corner + self.orientation.direction()).to_index()] = color

        # color head1 and head2
        head_direction = (
            direction
            if (
                corner + (direction := self.orientation.next().direction()) + direction
            ).is_in_bounds()
            else self.orientation.previous().direction()
        )
        head1 = corner + head_direction
        grid[head1.to_index()] = color
        grid[(head1 + head_direction).to_index()] = color

        return grid

    def transpose(self) -> "LPiecePosition":
        """
        Transpose the L-piece
        """
        return LPiecePosition(self.corner.transpose(), self.orientation.transpose())

    def flip(self) -> "LPiecePosition":
        """
        Flip the L-piece across the horizontal axis
        """
        return LPiecePosition(self.corner.flip(), self.orientation.flip())

    def mirror(self) -> "LPiecePosition":
        """
        Mirror the L-piece across the vertical axis
        """
        return LPiecePosition(self.corner.mirror(), self.orientation.mirror())

    def __str__(self) -> str:
        return f"{self.corner.x + 1} {self.corner.y + 1} {self.orientation.value}"


# Precompute all valid L-piece positions (4x4 grid)
ALL_VALID_LPIECE_POSITIONS: list[LPiecePosition] = [
    LPiecePosition(coord, Orientation(o))
    for coord in [Coordinate(x, y) for x in range(4) for y in range(4)]
    for o in Orientation
    if (coord + o.direction()).is_in_bounds()
]
# Precompute the grid masks for all valid L-piece positions
ALL_VALID_LPIECE_POSITIONS_GRID_MASKS: dict[
    GridCell, dict[LPiecePosition, np.ndarray]
] = {
    GridCell.RED: {
        pos: pos.grid_mask(GridCell.RED) for pos in ALL_VALID_LPIECE_POSITIONS
    },
    GridCell.BLUE: {
        pos: pos.grid_mask(GridCell.BLUE) for pos in ALL_VALID_LPIECE_POSITIONS
    },
}


@dataclass(frozen=True, slots=True)
class NeutralPiecePosition(Transformable):
    """
    A neutral piece position
    """

    position: Coordinate

    def __post_init__(self):
        if not self.position.is_in_bounds():
            raise ValueError("Position must be in bounds")

    def transpose(self) -> "NeutralPiecePosition":
        """
        Transpose the neutral piece
        """
        return NeutralPiecePosition(self.position.transpose())

    def flip(self) -> "NeutralPiecePosition":
        """
        Flip the neutral piece across the horizontal axis
        """
        return NeutralPiecePosition(self.position.flip())

    def mirror(self) -> "NeutralPiecePosition":
        """
        Mirror the neutral piece across the vertical axis
        """
        return NeutralPiecePosition(self.position.mirror())

    def grid_mask(self) -> np.ndarray:
        """
        Returns a 4x4 grid mask of the neutral piece
        """
        grid = np.full((4, 4), GridCell.EMPTY)
        grid[self.position.to_index()] = GridCell.NEUTRAL
        return grid


@dataclass(frozen=True, slots=True)
class LGameAction(Transformable):
    """
    An action in the L-game
    """

    l_piece_move: LPiecePosition
    neutral_piece_move: Optional[tuple[NeutralPiecePosition, NeutralPiecePosition]] = (
        None
    )

    def transpose(self) -> "LGameAction":
        """
        transpose the action
        """
        l_piece_move = self.l_piece_move.transpose()
        neutral_piece_move = (
            (
                self.neutral_piece_move[0].transpose(),
                self.neutral_piece_move[1].transpose(),
            )
            if self.neutral_piece_move
            else None
        )
        return LGameAction(l_piece_move, neutral_piece_move)

    def flip(self) -> "LGameAction":
        """
        flip the action across the horizontal axis
        """
        l_piece_move = self.l_piece_move.flip()
        neutral_piece_move = (
            (self.neutral_piece_move[0].flip(), self.neutral_piece_move[1].flip())
            if self.neutral_piece_move
            else None
        )
        return LGameAction(l_piece_move, neutral_piece_move)

    def mirror(self) -> "LGameAction":
        """
        mirror the action across the vertical axis
        """
        l_piece_move = self.l_piece_move.mirror()
        neutral_piece_move = (
            (self.neutral_piece_move[0].mirror(), self.neutral_piece_move[1].mirror())
            if self.neutral_piece_move
            else None
        )
        return LGameAction(l_piece_move, neutral_piece_move)

    def __str__(self) -> str:
        string = str(self.l_piece_move)
        if self.neutral_piece_move is not None:
            old = self.neutral_piece_move[0].position
            new = self.neutral_piece_move[1].position
            string += f" {old.x + 1} {old.y + 1} {new.x + 1} {new.y+1}"
        return string
