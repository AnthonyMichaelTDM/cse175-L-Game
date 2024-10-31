"""
Code for actions in the game
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from cell import GridCell

from util import IndexableEnum


@dataclass(frozen=True)
class Coordinate:
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

    def rotate(self, n: int = 1) -> "Coordinate":
        """
        Rotates the coordinate 90 degrees clockwise `n` times
        """
        n = n % 4

        if n < 0:
            n += 4

        x, y = self.x, self.y

        for _ in range(n):
            x, y = 3 - y, x

        return Coordinate(x, y)

    def mirror(self) -> "Coordinate":
        """
        Mirror the coordinate across the x=2 line
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

    def rotate(self, n: int = 1) -> "Orientation":
        """
        Rotates the orientation `n` times
        """
        n = n % 4

        if n < 0:
            n += 4

        return Orientation.from_index((int(self) + n) % 4)

    def mirror(self) -> "Orientation":
        """
        Mirror the orientation across the x=2 line
        """
        return (
            self
            if self == Orientation.NORTH or self == Orientation.SOUTH
            else self.rotate(2)
        )

    def direction(self) -> Coordinate:
        return OrientationDirections[int(self)]


@dataclass(frozen=True)
class LPiecePosition:
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
                corner
                + (direction := self.orientation.rotate(1).direction())
                + direction
            ).is_in_bounds()
            else self.orientation.rotate(-1).direction()
        )
        head1 = corner + head_direction
        grid[head1.to_index()] = color
        grid[(head1 + head_direction).to_index()] = color

        return grid

    def rotate(self, n: int = 1) -> "LPiecePosition":
        """
        Rotate the L-piece `n` times
        """
        return LPiecePosition(self.corner.rotate(n), self.orientation.rotate(n))

    def mirror(self) -> "LPiecePosition":
        """
        Mirror the L-piece across the x=2 line
        """
        return LPiecePosition(self.corner.mirror(), self.orientation.mirror())


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


@dataclass(frozen=True)
class NeutralPiecePosition:
    """
    A neutral piece position
    """

    position: Coordinate

    def __post_init__(self):
        if not self.position.is_in_bounds():
            raise ValueError("Position must be in bounds")

    def rotate(self, n: int = 1) -> "NeutralPiecePosition":
        """
        Rotate the neutral piece `n` times
        """
        return NeutralPiecePosition(self.position.rotate(n))

    def mirror(self) -> "NeutralPiecePosition":
        """
        Mirror the neutral piece across the x=2 line
        """
        return NeutralPiecePosition(self.position.mirror())

    def grid_mask(self) -> np.ndarray:
        """
        Returns a 4x4 grid mask of the neutral piece
        """
        grid = np.full((4, 4), GridCell.EMPTY)
        grid[self.position.to_index()] = GridCell.NEUTRAL
        return grid


@dataclass(frozen=True)
class LGameAction:
    """
    An action in the L-game
    """

    l_piece_move: LPiecePosition
    neutral_piece_move: Optional[tuple[NeutralPiecePosition, NeutralPiecePosition]] = (
        None
    )
