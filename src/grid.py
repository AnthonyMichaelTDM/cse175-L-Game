"""
Code for the game grid
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from action import (
    Coordinate,
    LPiecePosition,
    NeutralPiecePosition,
    Orientation,
)


# use a list instead of a dict because we can index into this list with the GridCell since it's an integere backed Enum
CellChars = ["+", "x", ".", "#"]


class GridCell(Enum):
    """
    The possible states of a cell in the grid
    """

    # red and blue are 0 and 1 so that they line up with the player IDs
    RED = 0
    BLUE = 1
    EMPTY = 2
    NEUTRAL = 3

    def __str__(self) -> str:
        return CellChars[self.value]


@dataclass
class Grid:
    """
    The 4x4 grid for the L-game

    Starting layout:

    ```
    # + + .
    . x + .
    . x + .
    . x x #
    ```

    `.` represents an empty cell
    `+` represents a red cell
    `x` represents a blue cell
    `#` represents a neutral cell

    the top left corner is (0,0) and the bottom right corner is (3,3)

    """

    # grid is a 2d array of cells
    grid: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [GridCell.NEUTRAL, GridCell.RED, GridCell.RED, GridCell.EMPTY],
                [GridCell.EMPTY, GridCell.BLUE, GridCell.RED, GridCell.EMPTY],
                [GridCell.EMPTY, GridCell.BLUE, GridCell.RED, GridCell.EMPTY],
                [GridCell.EMPTY, GridCell.BLUE, GridCell.BLUE, GridCell.NEUTRAL],
            ]
        )
    )
    red_position: LPiecePosition = LPiecePosition(Coordinate(2, 0), Orientation.WEST)
    blue_position: LPiecePosition = LPiecePosition(Coordinate(1, 3), Orientation.EAST)
    neutral_positions: tuple[NeutralPiecePosition, NeutralPiecePosition] = field(
        default_factory=lambda: (
            NeutralPiecePosition(Coordinate(0, 0)),
            NeutralPiecePosition(Coordinate(3, 3)),
        )
    )

    @classmethod
    def _new_with(
        cls,
        red_position: LPiecePosition,
        blue_position: LPiecePosition,
        neutral_positions: tuple[NeutralPiecePosition, NeutralPiecePosition],
    ) -> "Grid":
        """
        Create a new grid with the specified positions

        Args:
            red_position (LPiecePosition): the new position of the red L-piece
            blue_position (LPiecePosition): the new position of the blue L-piece
            neutral_positions (tuple[NeutralPiecePosition, NeutralPiecePosition]): the new positions of the neutral pieces
        """
        grid = np.full((4, 4), GridCell.EMPTY)

        for cell in red_position.get_cells():
            grid[cell.to_index()] = GridCell.RED
        for cell in blue_position.get_cells():
            grid[cell.to_index()] = GridCell.BLUE
        grid[neutral_positions[0].position.to_index()] = GridCell.NEUTRAL
        grid[neutral_positions[1].position.to_index()] = GridCell.NEUTRAL

        grid = Grid(
            grid=grid,
            red_position=red_position,
            blue_position=blue_position,
            neutral_positions=neutral_positions,
        )
        grid, rotated, mirrored = grid.normalize()

        return grid

    def move_red(self, new_position: LPiecePosition):
        """
        Move the red L-piece to the new position

        Args:
            new_position (LPiecePosition): the new position of the red L-piece
        """
        old_position = self.red_position or None
        old_cells = old_position.get_cells() if old_position else []
        new_cells = new_position.get_cells()

        # check if the new position is valid
        if any(
            not cell.is_in_bounds()
            or not (self.grid[cell.to_index()] == GridCell.EMPTY or GridCell.RED)
            for cell in new_cells
        ):
            raise ValueError("Invalid position")

        for cell in old_cells:
            self.grid[cell.to_index()] = GridCell.EMPTY
        for cell in new_cells:
            self.grid[cell.to_index()] = GridCell.RED

    def move_blue(self, new_position: LPiecePosition):
        """
        Move the blue L-piece to the new position

        Args:
            new_position (LPiecePosition): the new position of the blue L-piece
        """

        old_position = self.blue_position or None
        old_cells = old_position.get_cells() if old_position else []
        new_cells = new_position.get_cells()

        # check if the new position is valid
        if any(
            not cell.is_in_bounds()
            or not (self.grid[cell.to_index()] == GridCell.EMPTY or GridCell.BLUE)
            for cell in new_cells
        ):
            raise ValueError("Invalid position")

        for cell in old_cells:
            self.grid[cell] = GridCell.EMPTY
        for cell in new_cells:
            self.grid[cell] = GridCell.BLUE

    def move_neutral(
        self, old_position: NeutralPiecePosition, new_position: NeutralPiecePosition
    ):
        """
        Move the neutral pieces to the new positions

        Args:
            old_position (NeutralPiecePosition): the old position of the neutral piece
            new_position (NeutralPiecePosition): the new position of the neutral piece
        """

        old_cell = old_position.position
        new_cell = new_position.position

        # check if the new position is valid
        if (
            not old_cell.is_in_bounds()
            or not new_cell.is_in_bounds()
            or self.grid[old_cell.to_index()] != GridCell.NEUTRAL
            or self.grid[new_cell.to_index()] != GridCell.EMPTY
        ):
            raise ValueError("Invalid position")

        self.grid[old_cell] = GridCell.EMPTY
        self.grid[new_cell] = GridCell.NEUTRAL

    def render(self) -> str:
        """
        Render the grid, alias to __str__
        """
        return str(self)

    def __str__(self) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)

    def rotate(self, n: int = 1) -> "Grid":
        """
        Rotate the grid 90 degrees clockwise n times

        Args:
            n (int): the number of times to rotate the grid
        """
        return Grid(
            grid=np.rot90(self.grid, -n),
            red_position=self.red_position.rotate(n),
            blue_position=self.blue_position.rotate(n),
            neutral_positions=(
                self.neutral_positions[0].rotate(n),
                self.neutral_positions[1].rotate(n),
            ),
        )

    def normalize(self) -> tuple["Grid", int, bool]:
        """
        Normalize the grid by rotating it such that the red L-piece is oriented such that the long end points to the right and the short end points up

        Returns the normalized grid, the number of times the grid was rotated, and whether the grid was mirrored
        """
        n = Orientation.LENGTH() - self.red_position.orientation.index()
        grid = self.rotate(n)

        # check if we need to mirror the grid
        if grid.red_position.corner.x > 1:
            return grid.mirror(), n, True

        return grid, n, False

    def mirror(self) -> "Grid":
        """
        Mirror the grid along the vertical axis
        """
        return Grid(
            grid=np.fliplr(self.grid),
            red_position=self.red_position.mirror(),
            blue_position=self.blue_position.mirror(),
            neutral_positions=(
                self.neutral_positions[0].mirror(),
                self.neutral_positions[1].mirror(),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return (
            np.array_equal(self.grid, other.grid)
            and self.red_position == other.red_position
            and self.blue_position == other.blue_position
            and (
                (self.neutral_positions == other.neutral_positions)
                or (
                    self.neutral_positions[0] == other.neutral_positions[1]
                    and self.neutral_positions[1] == other.neutral_positions[0]
                )
            )
        )
