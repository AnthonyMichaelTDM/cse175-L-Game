"""
Implements the GridCell enum which represents the possible states of a cell in the grid.
"""

from enum import Enum


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
