"""
Code for the game grid
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from action import (
    ALL_VALID_LPIECE_POSITIONS_GRID_MASKS,
    Coordinate,
    LPiecePosition,
    NeutralPiecePosition,
    Orientation,
)
from cell import GridCell

STRICT_MOVES = True


@dataclass(frozen=True, slots=True)
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
    # grid: np.ndarray = field(
    #     default_factory=lambda: np.array(
    #         [
    #             [GridCell.EMPTY, GridCell.NEUTRAL, GridCell.EMPTY, GridCell.EMPTY],
    #             [GridCell.EMPTY, GridCell.NEUTRAL, GridCell.RED, GridCell.RED],
    #             [GridCell.BLUE, GridCell.BLUE, GridCell.BLUE, GridCell.RED],
    #             [GridCell.BLUE, GridCell.EMPTY, GridCell.EMPTY, GridCell.RED],
    #         ]
    #     )
    # )
    # red_position: LPiecePosition = LPiecePosition(Coordinate(3, 1), Orientation.WEST)
    # blue_position: LPiecePosition = LPiecePosition(Coordinate(0, 2), Orientation.SOUTH)
    grid: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [GridCell.BLUE, GridCell.BLUE, GridCell.EMPTY, GridCell.EMPTY],
                [GridCell.BLUE, GridCell.NEUTRAL, GridCell.EMPTY, GridCell.EMPTY],
                [GridCell.BLUE, GridCell.EMPTY, GridCell.EMPTY, GridCell.RED],
                [GridCell.NEUTRAL, GridCell.RED, GridCell.RED, GridCell.RED],
            ]
        )
    )
    red_position: LPiecePosition = LPiecePosition(
        Coordinate(3, 3), Orientation.NORTH)
    blue_position: LPiecePosition = LPiecePosition(
        Coordinate(0, 0), Orientation.EAST)
    
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
        inner_grid = np.full((4, 4), GridCell.EMPTY)

        inner_grid += red_position.grid_mask(GridCell.RED)
        inner_grid += blue_position.grid_mask(GridCell.BLUE)
        inner_grid += neutral_positions[0].grid_mask()
        inner_grid += neutral_positions[1].grid_mask()

        grid = Grid(
            grid=inner_grid,
            red_position=red_position,
            blue_position=blue_position,
            neutral_positions=neutral_positions,
        )
        grid, _, _ = grid.normalize()

        return grid

    def is_mask_valid(self, mask: np.ndarray, color: GridCell) -> bool:
        """
        Assert that for every non-Empty cell in the grid mask, the corresponding cell in the grid is either color or empty

        Let l be the proposition that the cell in the grid is color or empty
        Let r be the proposition that the cell in the grid mask is color

        we want to assert that for all cells in the grid, the proposition (l | r) == l is true

        This has the following truth table:

        | l | r | result |
        |---|---|--------|
        | 0 | 0 | 1      |
        | 0 | 1 | 0      |
        | 1 | 0 | 1      |
        | 1 | 1 | 1      |
        """
        l = self.grid & (GridCell.EMPTY | color)
        r = mask & color
        result = (l | r).astype(bool) == l.astype(bool)
        result = bool(np.all(result))

        return result

    def move_red(self, new_position: LPiecePosition) -> "Grid":
        """
        Move the red L-piece to the new position

        Args:
            new_position (LPiecePosition): the new position of the red L-piece
        """
        if STRICT_MOVES:
            assert self.is_mask_valid(
                new_position.grid_mask(GridCell.RED), GridCell.RED
            ), "Invalid move, can't move to a non-empty cell"

        grid = self.grid.copy()

        if self.red_position is not None:
            if self.red_position == new_position:
                raise ValueError("Invalid move, can't move to the same position")
            grid = grid - self.red_position.grid_mask(GridCell.RED)
        grid = grid + new_position.grid_mask(GridCell.RED)

        return Grid(
            grid=grid,
            red_position=new_position,
            blue_position=self.blue_position,
            neutral_positions=self.neutral_positions,
        )

    def move_blue(self, new_position: LPiecePosition) -> "Grid":
        """
        Move the blue L-piece to the new position

        Args:
            new_position (LPiecePosition): the new position of the blue L-piece
        """
        # assert that for every non-Empty cell in the grid mask, the corresponding cell in the grid is either blue or empty
        if STRICT_MOVES:
            assert self.is_mask_valid(
                new_position.grid_mask(GridCell.BLUE), GridCell.BLUE
            ), "Invalid move, can't move to a non-empty cell"

        grid = self.grid.copy()

        if self.blue_position is not None:
            if self.blue_position == new_position:
                raise ValueError("Invalid move, can't move to the same position")
                # pass
            grid = grid - self.blue_position.grid_mask(GridCell.BLUE)
        grid = grid + new_position.grid_mask(GridCell.BLUE)

        return Grid(
            grid=grid,
            red_position=self.red_position,
            blue_position=new_position,
            neutral_positions=self.neutral_positions,
        )

    def move_neutral(
        self, old_position: NeutralPiecePosition, new_position: NeutralPiecePosition
    ) -> "Grid":
        """
        Move the neutral pieces to the new positions

        Args:
            old_position (NeutralPiecePosition): the old position of the neutral piece
            new_position (NeutralPiecePosition): the new position of the neutral piece
        """

        assert (
            self.grid[old_position.position.to_index()] == GridCell.NEUTRAL
        ), "Invalid move, can't move a non-neutral piece"
        if STRICT_MOVES:
            assert (
                self.grid[new_position.position.to_index()] == GridCell.EMPTY
            ), "Invalid move, can't move to a non-empty cell"

        grid = self.grid - old_position.grid_mask() + new_position.grid_mask()

        if self.neutral_positions[0] == old_position:
            neutral_positions = (new_position, self.neutral_positions[1])
        elif self.neutral_positions[1] == old_position:
            neutral_positions = (self.neutral_positions[0], new_position)
        else:
            raise ValueError("Invalid move, can't move a non-neutral piece")

        return Grid(
            grid=grid,
            red_position=self.red_position,
            blue_position=self.blue_position,
            neutral_positions=neutral_positions,
        )

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

    def get_red_legal_moves(self) -> list[LPiecePosition] | None:
        """Get the legal moves for the red L-piece

        Returns:
            list[LPiecePosition] | None: a list of all possible moves for the red L-piece, or None if there are no legal moves
        """
        # Idea: remove the L piece from the grid, then check all possible positions for the L piece to see if it fits
        # then remove the "current" red position from that list (because you can't move to the same position)
        # then return the list of possible positions, or None if that list is empty
        legal_moves = [
            position
            for position, mask in ALL_VALID_LPIECE_POSITIONS_GRID_MASKS[
                GridCell.RED
            ].items()
            if position != self.red_position and self.is_mask_valid(mask, GridCell.RED)
        ]
        return legal_moves if legal_moves else None

    def get_blue_legal_moves(self) -> list[LPiecePosition] | None:
        """Get the legal moves for the blue L-piece

        Returns:
            list[LPiecePosition] | None: a list of all possible moves for the blue L-piece, or None if there are no legal moves
        """
        legal_moves = [
            position
            for position, mask in ALL_VALID_LPIECE_POSITIONS_GRID_MASKS[
                GridCell.BLUE
            ].items()
            if position != self.blue_position
            and self.is_mask_valid(mask, GridCell.BLUE)
        ]
        return legal_moves if legal_moves else None

    def get_neutral_legal_moves(
        self, proposed_l_move: LPiecePosition, color: GridCell
    ) -> list[Tuple[NeutralPiecePosition, NeutralPiecePosition]]:
        """returns a list of all possible moves for the neutral pieces, not including the option to not move any, given the proposed move for the L-piece

        Args:
            proposed_l_move (LPiecePosition): the proposed move for the L-piece
            color (GridCell): the color of the L-piece that moved

        Returns:
            list[Tuple[NeutralPiecePosition, NeutralPiecePosition]]: a list of all possible moves for the neutral pieces
        """
        assert color in (GridCell.RED, GridCell.BLUE), "Invalid color"
        piece = self.red_position if color == GridCell.RED else self.blue_position

        # this is in row major order, but we need it in column major order
        empty_cells = np.argwhere(
            (self.grid - piece.grid_mask(color) + proposed_l_move.grid_mask(color))
            == GridCell.EMPTY
        )

        legal_moves: list[Tuple[NeutralPiecePosition, NeutralPiecePosition]] = [
            (neutral, new)
            for y, x in empty_cells
            for neutral in self.neutral_positions
            if (new := NeutralPiecePosition(Coordinate(x, y))) != neutral.position
        ]

        if STRICT_MOVES:
            assert len(legal_moves) == 12, "Invalid number of legal moves: " + str(
                len(legal_moves)
            )

        return legal_moves
