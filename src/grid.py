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

STRICT_CELL_ARITHMETIC = True

# use a list instead of a dict because we can index into this list with the GridCell since it's an integer backed Enum
CellChars = [".", "+", "x", "", "#"]


class GridCell(Enum):
    """
    The possible states of a cell in the grid
    """

    # the values are chosen such that the empty cell is 0 and the other cells are powers of 2,
    # this is useful for arithmetic operations on the cells as we can use bitwise operations
    EMPTY = 0b0000
    RED = 0b0001
    BLUE = 0b0010
    NEUTRAL = 0b0100

    def __str__(self) -> str:
        return CellChars[self.value]

    def __sub__(self, other: object) -> "GridCell":
        if not isinstance(other, GridCell):
            return NotImplemented
        match self, other:
            case a, b if a == b:
                return GridCell.EMPTY
            case a, GridCell.EMPTY:
                return a
            case GridCell.EMPTY, b:
                if STRICT_CELL_ARITHMETIC:
                    raise ValueError(
                        f"Subtracting {b} from an empty cell is not allowed"
                    )
                return GridCell.EMPTY
            case a, b:
                if STRICT_CELL_ARITHMETIC:
                    raise ValueError(f"Subtracting {b} from {a} is not allowed")
                return GridCell.EMPTY

    def __add__(self, other: object) -> "GridCell":
        if not isinstance(other, GridCell):
            return NotImplemented
        match self, other:
            case GridCell.EMPTY, b:
                return b
            case a, GridCell.EMPTY:
                return a
            case a, b:
                if STRICT_CELL_ARITHMETIC:
                    raise ValueError(f"Adding {a} and {b} is not allowed")
                return GridCell.EMPTY


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

    def __hash__(self) -> int:
        return hash(
            (
                self.red_position,
                self.blue_position,
                self.neutral_positions,
            )
        )

    def get_red_legal_moves(self) -> list:
        legal_moves = []
        current_cells = self.red_position.get_cells()
        # goes over all cell check for empty slots to fit a L with at most three overlaps
        for x in range(4):
            for y in range(4):
                if self.grid[x, y] == GridCell.EMPTY:
                    for orientation in [
                        Orientation.NORTH,
                        Orientation.EAST,
                        Orientation.SOUTH,
                        Orientation.WEST,
                    ]:
                        possible_position = LPiecePosition(
                            Coordinate(x, y), orientation
                        )
                        new_cells = possible_position.get_cells()
                        overlap_cell_count = sum(
                            1 for cell in new_cells if cell in current_cells
                        )
                        on_empty_cell_count = sum(
                            1
                            for cell in new_cells
                            if cell.is_in_bounds()
                            and self.grid[cell.to_index()] == GridCell.EMPTY
                        )
                        if overlap_cell_count < 4 and (
                            overlap_cell_count + on_empty_cell_count == 4
                        ):
                            legal_moves.append(possible_position)
        return legal_moves

    def get_blue_legal_moves(self) -> list:
        legal_moves = []
        current_cells = self.blue_position.get_cells()
        # goes over all cell check for empty slots to fit a L with at most three overlaps
        for x in range(4):
            for y in range(4):
                if self.grid[x, y] == GridCell.EMPTY:
                    for orientation in [
                        Orientation.NORTH,
                        Orientation.EAST,
                        Orientation.SOUTH,
                        Orientation.WEST,
                    ]:
                        possible_position = LPiecePosition(
                            Coordinate(x, y), orientation
                        )
                        new_cells = possible_position.get_cells()
                        overlap_cell_count = sum(
                            1 for cell in new_cells if cell in current_cells
                        )
                        on_empty_cell_count = sum(
                            1
                            for cell in new_cells
                            if cell.is_in_bounds()
                            and self.grid[cell.to_index()] == GridCell.EMPTY
                        )
                        if overlap_cell_count < 4 and (
                            overlap_cell_count + on_empty_cell_count == 4
                        ):
                            legal_moves.append(possible_position)
        return legal_moves

    def get_neutral_legal_moves(self) -> list:
        legal_moves = []
        for neutral in self.neutral_positions:
            x, y = neutral.position.to_index()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_neutral = NeutralPiecePosition(Coordinate(x + dx, y + dy))
                if new_neutral.position.is_in_bounds():
                    legal_moves.append([neutral, new_neutral])
        return legal_moves
