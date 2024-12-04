"""
Implements the GridCell enum which represents the possible states of a cell in the grid.
"""

from enum import Enum

STRICT_CELL_ARITHMETIC = True

# use a list instead of a dict because we can index into this list with the GridCell since it's an integer backed Enum
CellChars = [".", "+", "x", "#"]


class GridCell(Enum):
    """
    The possible states of a cell in the grid
    """

    # the values are chosen such that the empty cell is 0 and the other cells are powers of 2,
    # this is useful for arithmetic operations on the cells as we can use bitwise operations
    EMPTY = 0b0001
    RED = 0b0010
    BLUE = 0b0100
    NEUTRAL = 0b1000

    def __str__(self) -> str:
        # index into CellChars with the position of the highest set bit
        return CellChars[self.value.bit_length() - 1]

    # bitwise operations for arithmetic on the cells

    def __sub__(self, other: object) -> "GridCell":
        if not isinstance(other, GridCell):
            return NotImplemented
        match self, other:
            case a, b if a == b:
                return GridCell.EMPTY
            case a, GridCell.EMPTY:
                return a
            case GridCell.EMPTY, b if STRICT_CELL_ARITHMETIC:
                raise ValueError(
                    f"Subtracting {repr(b)} from an empty cell is not allowed"
                )
            case a, b if STRICT_CELL_ARITHMETIC:
                raise ValueError(f"Subtracting {repr(b)} from {repr(a)} is not allowed")
        return GridCell.EMPTY

    def __add__(self, other: object) -> "GridCell":
        if not isinstance(other, GridCell):
            return NotImplemented
        match self, other:
            case GridCell.EMPTY, b:
                return b
            case a, GridCell.EMPTY:
                return a
            case a, b if STRICT_CELL_ARITHMETIC:
                raise ValueError(f"Adding {repr(a)} and {repr(b)} is not allowed")
        return GridCell.EMPTY

    def __and__(self, other: object) -> int:
        match other:
            case int():
                return self.value & other
            case GridCell():
                return self.value & other.value
            case _:
                return NotImplemented

    def __or__(self, other: object) -> int:
        match other:
            case int():
                return self.value | other
            case GridCell():
                return self.value | other.value
            case _:
                return NotImplemented

    def __xor__(self, other: object) -> int:
        match other:
            case int():
                return self.value ^ other
            case GridCell():
                return self.value ^ other.value
            case _:
                return NotImplemented

    def __invert__(self) -> int:
        return ~self.value

    def __lshift__(self, other: int) -> int:
        return self.value << other

    def __rshift__(self, other: int) -> int:
        return self.value >> other
