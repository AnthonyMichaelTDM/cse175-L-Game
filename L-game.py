"""
Main file for the project,
parses the command line arguments to determine the mode of operation

usage: main.py -p1 <AgentType> [-d1 <int>] [-h1 <heuristic>] -p2 <AgentType> [-d2 <int>] [-h2 <heuristic>] [-d <int>]

- `-p1` and `-p2` are the types of agents to use for player 1 and player 2, respectively. The agent types can be one of the following:
    - `human`: a human player
    - `minimax`: a computer player that uses the minimax algorithm to choose its moves
    - `alphabeta`: a computer player that uses a heuristic alpha-beta pruning algorithm to choose its moves
- `-d1` and `-d2` are optional arguments that specify the depth of the search tree for the first and second computer players, respectively. The default depth is 3.
    - if the agent type is `human`, the depth argument is ignored
    - if `-d` is specified instead, it sets the depth for both computer players
        - if `-d1` or `-d2` is also specified, they take precedence over `-d` for the corresponding player
- `-h1` and `-h2` are optional arguments that specify the heuristic function to use for the first and second computer players, respectively.
    The default heuristic is the aggressive heuristic.
    - the options for the heuristic are:
        - `aggressive`: the aggressive heuristic
        - `defensive`: the defensive heuristic

Then runs the game loop with the specified agents
"""

import abc
import argparse
import time
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from random import randint
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeVar,
    override,
)

import numpy as np

T = TypeVar("T")

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


class Transform(Enum):
    TRANSPOSE = 0
    FLIP = 1
    MIRROR = 2


@dataclass(frozen=True, slots=True)
class TransformSeries:
    transformations: list[Transform] = field(default_factory=list)

    def merge(self, transformation: Transform):
        """
        Create a new transformation series with the given transformation appended to the end if it is different from the last transformation
        otherwise, returns a new transformation series with the last transformation removed
        """
        if self.transformations and self.transformations[-1] == transformation:
            return TransformSeries(self.transformations[:-1])
        return TransformSeries(self.transformations + [transformation])

    def __len__(self):
        return len(self.transformations)

    def __getitem__(self, index):
        return self.transformations[index]

    def __iter__(self):
        return iter(self.transformations)


class Transformable(abc.ABC):
    @abc.abstractmethod
    def transpose(self) -> Self:
        pass

    @abc.abstractmethod
    def flip(self) -> Self:
        pass

    @abc.abstractmethod
    def mirror(self) -> Self:
        pass

    def apply_transformations(self, transformations: Iterable[Transform]) -> Self:
        obj = self
        for transformation in transformations:
            if transformation == Transform.TRANSPOSE:
                obj = obj.transpose()
            elif transformation == Transform.FLIP:
                obj = obj.flip()
            elif transformation == Transform.MIRROR:
                obj = obj.mirror()
        return obj

    def unapply_transformations(self, transformations: TransformSeries):
        return self.apply_transformations(reversed(transformations))


class IndexableEnum(Generic[T], Enum):
    """
    An enum type whose variants are of type T and who has an additional index attribute that is the 0-based index of the variant in the enum (in order of definition).

    But can be initialized without specifying the index

    For instance:

    ```
    class Orientation(IndexableEnum):
        NORTH = "N"
        EAST = "E"
        SOUTH = "S"
        WEST = "W"

    Orientation.NORTH.index()  # 0
    Orientation.NORTH.value  # "N"
    Orientation("N")  # Orientation.NORTH
    Orientation(0) # Orientation.NORTH
    Orientation.EAST.index()  # 1
    ```
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        value = start if start is not None else name.lower()
        index = count
        return value, index

    def __new__(cls, value) -> Self:
        index = len(cls.__members__)

        obj = object.__new__(cls)
        obj._value_ = value
        obj.__setattr__("_index", index)
        if not hasattr(cls, "_member_index_"):
            cls._member_index_ = [obj]
        else:
            cls._member_index_.append(obj)  # type: ignore
        return obj

    @classmethod
    def from_index(cls, index: int) -> Self:
        member_index: list[Self] = cls._member_index_  # type: ignore
        if index < 0 or index >= len(member_index):
            raise ValueError(f"Invalid index: {index}")
        return member_index[index]

    def next(self) -> Self:
        return self.from_index((self.index() + 1) % self.LENGTH())

    def previous(self) -> Self:
        return self.from_index((self.index() - 1) % self.LENGTH())

    def __str__(self) -> str:
        return str(self.value)

    def index(self) -> int:
        return self._index  # type: ignore

    @classmethod
    def LENGTH(cls) -> int:
        """
        Returns the number of variants in the enum class
        """
        return len(cls)

    def __int__(self) -> int:
        return self.index()


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


class AgentRules[Action, State](abc.ABC):
    """
    An abstract base class for the rules a game-playing agent must follow
    """

    @abc.abstractmethod
    def get_legal_actions(self, state: State, agent_id: int) -> list[Action]:
        """
        Get the legal actions for the given state

        Args:
            state (State): the current game state
            agent_id (int): the agent's ID

        Returns:
            list[Action]: the legal actions
        """
        ...

    @abc.abstractmethod
    def apply_action(self, state: State, action: Action, agent_id: int) -> State:
        """
        Apply the specified action to the state

        Args:
            state (State): the current game state
            action (Action): the action to apply
            agent_id (int): the agent's ID

        Returns:
            State: the new state after applying the action
        """
        ...


@dataclass(slots=True)
class Agent[Action, State](abc.ABC):
    """
    An abstract base class for a game-playing agent
    """

    id: int

    def agent_id(self) -> int:
        """
        Get the agent's ID

        Returns:
            int: the agent's ID
        """
        return self.id

    @abc.abstractmethod
    def get_action(self, state: State) -> Action:
        """
        Get next action from the agent given the current state

        Args:
            state (State): the current game state

        Returns:
            Action: the next action to take
        """
        ...

    @classmethod
    @abc.abstractmethod
    def get_rules(cls) -> AgentRules[Action, State]:
        """
        Get the rules for the agent

        Returns:
            AgentRules[Action, State]: the rules for the agent
        """
        ...

    def __hash__(self) -> int:
        return hash(self.id)


STRICT_MOVES = True


@dataclass(frozen=True, slots=True)
class Grid(Transformable):
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
    transformations: TransformSeries = field(default_factory=lambda: TransformSeries())

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
            transformations=TransformSeries(),
        )
        grid = grid.normalize()

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
            transformations=self.transformations,
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
            transformations=self.transformations,
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
            transformations=self.transformations,
        )

    def render(self) -> str:
        """
        Render the grid, alias to __str__
        """
        return str(self)

    def __str__(self) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)

    def transpose(self) -> "Grid":
        """
        Transpose the grid
        """
        return Grid(
            grid=self.grid.T,
            red_position=self.red_position.transpose(),
            blue_position=self.blue_position.transpose(),
            neutral_positions=(
                self.neutral_positions[0].transpose(),
                self.neutral_positions[1].transpose(),
            ),
            transformations=self.transformations.merge(Transform.TRANSPOSE),
        )

    def flip(self) -> "Grid":
        """
        Flip the grid along the horizontal axis
        """
        return Grid(
            grid=np.flipud(self.grid),
            red_position=self.red_position.flip(),
            blue_position=self.blue_position.flip(),
            neutral_positions=(
                self.neutral_positions[0].flip(),
                self.neutral_positions[1].flip(),
            ),
            transformations=self.transformations.merge(Transform.FLIP),
        )

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
            transformations=self.transformations.merge(Transform.MIRROR),
        )

    def normalize(self) -> "Grid":
        """
        Normalize the grid by
        performing one or more of the following transformations:
        - mirror the grid along the vertical axis
        - flip the grid along the horizontal axis
        - transpose the grid
        such that the red L-piece is oriented such that the long end points to the right and the short end points up

        Returns the normalized grid
        """
        result = self.unapply_transformations(self.transformations)
        assert len(result.transformations) == 0
        if result.red_position.orientation in [Orientation.WEST, Orientation.EAST]:
            result = result.transpose()
        if result.red_position.orientation == Orientation.SOUTH:
            result = result.flip()
        if result.red_position.corner.x > 1:
            result = result.mirror()
        return result

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


class LegalActionsCache:
    """
    Singleton cache for storing legal actions for game states
    """

    _instance = None
    _cache: dict[int, dict[Grid, list[LGameAction]]] = {
        0: {state: [] for state in TERMINAL_STATES},
        1: {_grid_swap_red_blue(state): [] for state in TERMINAL_STATES},
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LegalActionsCache, cls).__new__(cls)
        return cls._instance

    def get(self, state: "LGameState", agent_id: int) -> list[LGameAction] | None:
        return (
            agent_cache.get(state.grid)
            if (agent_cache := self._cache.get(agent_id))
            else None
        )

    def set(self, state: "LGameState", agent_id: int, actions: list[LGameAction]):
        self._cache[agent_id][state.grid] = actions

    def __len__(self):
        return sum(len(agent_cache) for agent_cache in self._cache.values())


# Initialize the singleton cache
legal_actions_cache = LegalActionsCache()


@dataclass(frozen=True, slots=True)
class GameState[Action](abc.ABC):
    """
    An abstract base class for a game state
    """

    agents: Sequence[Agent[Action, Self]]

    def get_legal_actions(self, agent_id: int) -> list[Action]:
        """
        Get the legal actions for the given agent in the state

        By convention, the agent ID is 0-indexed where 0 is the first agent (e.g., the human player in a human-vs-computer game)

        Returns:
            list[Action]: the legal actions
        """
        # if self.is_terminal():
        #     return []

        if agent_id < 0 or agent_id >= len(self.agents):
            raise ValueError(f"Invalid agent ID: {agent_id}")

        actions = self.agents[agent_id].get_rules().get_legal_actions(self, agent_id)

        return actions

    def generate_successor(self, action: Action, agent_id: int) -> Self:
        """
        Generate the successor state  after the specified agent takes the specified action

        Args:
            action (Action): the action to apply
            agent_id (int): the agent ID of the agent taking the action

        Returns:
            GameState: the successor state
        """
        if self.is_loss(agent_id):
            raise ValueError("Cannot generate successor of terminal state")

        if agent_id < 0 or agent_id >= len(self.agents):
            raise ValueError(f"Invalid agent ID: {agent_id}")

        return (
            self.agents[agent_id]
            .get_rules()
            .apply_action(self, action, agent_id)
            .normalize()
        )

    @abc.abstractmethod
    def normalize(self) -> Self:
        """
        Normalize the state

        Some games have symmetries that make some states equivalent. This method should transform the state into a canonical form that is unique for each equivalence class.

        Returns:
            GameState: the normalized state
        """
        ...

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the state is terminal

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        ...

    @abc.abstractmethod
    def is_win(self, agent_id: int) -> bool:
        """
        Check if the state is a winning state for the specified agent

        Returns:
            bool: True if the state is a winning state, False otherwise
        """
        ...

    @abc.abstractmethod
    def is_loss(self, agent_id: int) -> bool:
        """
        Check if the state is a losing state for the specified agent

        Returns:
            bool: True if the state is a losing state, False otherwise
        """
        ...


@dataclass(frozen=True, slots=True)
class LGameState(GameState[LGameAction]):
    """
    The state of the L-game
    """

    agents: tuple[Agent[LGameAction, "LGameState"], Agent[LGameAction, "LGameState"]]

    # the internal grid, should always be normalized
    grid: Grid = field(default_factory=Grid)

    @override
    def get_legal_actions(self, agent_id: int) -> list[LGameAction]:
        global legal_actions_cache

        if cached := legal_actions_cache.get(self, agent_id):
            return cached

        actions = super(LGameState, self).get_legal_actions(agent_id)
        legal_actions_cache.set(self, agent_id, actions)
        return actions

    def get_sorted_legal_actions(self, agent_id: int) -> list[LGameAction]:
        global legal_actions_cache

        if cached := legal_actions_cache.get(self, agent_id):
            return cached

        actions = self.get_legal_actions(agent_id)
        game_rules = LGameRules()

        actions.sort(
            key=lambda action: len(
                game_rules.get_legal_actions(
                    game_rules.apply_action(self, action, agent_id), 1 - agent_id
                )
            )
            / 13.0,
            # reverse=True, # if this is uncommented, it makes alpha-beta prune as few nodes as possible, which makes it way slower
        )

        legal_actions_cache.set(self, agent_id, actions)
        return actions

    def render(self) -> str:
        """
        Render the game state
        """
        return self.grid.unapply_transformations(self.grid.transformations).render()

        # return self.grid.render()

    def normalize(self) -> "LGameState":
        """
        Normalize the state, by transforming (through symmetry) into a state where the red L-piece is oriented such that the long end points to the right and the short end points up

        like so:

        #
        ###


        Returns:
            LGameState: the normalized state
        """
        return LGameState(agents=self.agents, grid=self.grid.normalize())

    def is_terminal(self) -> bool:
        """
        Check if the state is terminal

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        return self.is_loss(0) or self.is_loss(1)

    def is_win(self, agent_id: int) -> bool:
        """
        Check if the state is a winning state for the red player

        Returns:
            bool: True if the state is a winning state, False otherwise
        """
        return self.is_loss(1 - agent_id)

    def is_loss(self, agent_id: int) -> bool:
        """
        Check if the state is a losing state for the red player

        Returns:
            bool: True if the state is a losing state, False otherwise
        """
        return self.get_legal_actions(agent_id) == []

    def copy(self, **kwargs) -> "LGameState":
        """
        Create a copy of the game state with the specified modifications

        Args:
            kwargs: the modifications to make

        Returns:
            LGameState: the new game state
        """
        if "agents" not in kwargs:
            kwargs["agents"] = self.agents
        if "grid" not in kwargs:
            kwargs["grid"] = self.grid

        return LGameState(
            **kwargs,
        )


class LGameRules(AgentRules[LGameAction, LGameState]):
    """
    The rules for the L-game, which all agents must follow
    """

    def get_legal_actions(self, state: LGameState, agent_id: int) -> list[LGameAction]:
        """
        Get the legal actions for the given state, assuming it is the agent's turn

        Args:
            state (LGameState): the current game state

        Returns:
            list[LGameAction]: the legal actions
        """
        l_piece_moves = self.get_l_piece_moves(state, agent_id)
        if not l_piece_moves:
            return []
        legal_actions: list[LGameAction] = [
            LGameAction(l_move, neutral_move)
            for l_move in l_piece_moves
            if l_move
            != (state.grid.red_position if agent_id == 0 else state.grid.blue_position)
            for neutral_move in (
                self.get_neutral_legal_moves(state, l_move, agent_id) + [None]
            )
        ]

        return legal_actions

    def get_l_piece_moves(
        self, state: LGameState, agent_id: int
    ) -> list[LPiecePosition] | None:
        """
        Get legal moves for the current player's L-piece

        Args:
            state (LGameState): the current game state

        Returns:
            list: the legal moves for the L-piece
        """
        if agent_id == 0:
            return state.grid.get_red_legal_moves()
        else:
            return state.grid.get_blue_legal_moves()

    def get_neutral_legal_moves(
        self, state: LGameState, proposed_l_move: LPiecePosition, agent_id: int
    ) -> list:
        """
        Determine the legal moves for the neutral pieces based on the L-piece move, not including the option to move no neutral piece

        Args:
            state (LGameState): the current game state
            l_piece_move: the selected move for the L-piece

        Returns:
            list: legal moves for the neutral pieces based on the L-piece move
        """
        move_color = GridCell.RED if agent_id == 0 else GridCell.BLUE

        legal_moves = state.grid.get_neutral_legal_moves(proposed_l_move, move_color)

        return legal_moves

    def apply_action(
        self, state: LGameState, action: LGameAction, agent_id: int
    ) -> LGameState:
        """
        Apply the specified action to the state

        Args:
            state (LGameState): the current game state
            action (LGameAction): the action to apply

        Returns:
            LGameState: the new state after applying the action
        """

        if agent_id == 0:
            new_grid = state.grid.move_red(action.l_piece_move)
        else:
            new_grid = state.grid.move_blue(action.l_piece_move)

        if action.neutral_piece_move:
            new_grid = new_grid.move_neutral(*action.neutral_piece_move)

        return state.copy(grid=new_grid)


@dataclass(slots=True)
class _CacheInfo:
    hits: int
    misses: int
    length: int

    def __str__(self):
        return f"{{ hits: {self.hits}, misses: {self.misses}, length: {self.length} }}"


def minimax_cache():
    """
    A decorator that caches the results of a computer agents min and max functions with the game state grid as the key

    This one is specifically for the minimax agent, as it only considers the depth of the search for cache invalidation

    this brings a significant performance improvement to the agents as it avoids recalculating the same states multiple times

    Problem, as descirbed it would ignore the depth, this would affect the ability of the agents to explore the game tree, and lead to suboptimal moves.

    The solution is to store the depth next to the result in the cache, and only use the cached result if its depth is higher than the current depth.
    With this approach, the agents can still explore the game tree properly, while still benefiting from the cache.

    Results(minimax depth=1, alphabeta depth=2)
    - without caching: tests run in about 2 minutes
    - with caching (first implementations): tests run in ~25 seconds, but agents perform worse
    - with caching (second implementation): tests run in ~35 seconds, minimax agent performance unaffected (haven't validated alphabeta, but I assume it's fine too)

    with minimax depth = 2, alphabeta depth = 4, this implementation finishes the tests in about 5 minutes, an order of magnitude faster than it would take without caching

    Caching works particularly well for this problem because although the branching factor is massive, there are relatively few unique states, so the cache hit rate is high
    ( the cache should never need to be larger than 2296 * 2 = 4592 entries, which is very manageable )
    """

    def decorator(func):
        # Constants shared by all lru cache instances:
        sentinel = object()  # unique object used to signal cache misses
        # build a key from the function arguments
        make_key = lambda args, _: args[1].grid
        # get the depth from the function arguments
        get_depth = lambda args: args[2]
        # get the agent from the function arguments
        get_self = lambda args: args[0]

        cache: dict[Any, dict[Grid, tuple[int, tuple]]] = {}
        hits, misses = {}, {}
        # bound method to lookup a key or return None
        cache_get = lambda id: cache[id].get
        cache_len = lambda id: cache[id].__len__  # get cache size without calling len()

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds)
            depth = get_depth(args)
            self_ = get_self(args)
            id = args[0].id

            # if the agent doesn't have a cache, create one
            if id not in cache:
                cache[id] = {}
                hits[id], misses[id] = 0, 0

            result = cache_get(id)(key, sentinel)
            if (
                result is not sentinel
                and isinstance(result, tuple)
                and result[0] >= depth
            ):
                if depth == self_.depth and cache_len(id)() == 4592:
                    # if we get a cache hit at the root, then try bumping the depth up by 1 so we can get a better result
                    self_.depth += 1
                    args = list(args)
                    args[2] += 1
                    print(f"Bumping depth up by 1, new max depth: {self_.depth}")
                    result = func(*args, **kwds)
                    cache[id][key] = (self_.depth, result)
                    return result

                hits[id] = (1 + hits[id]) if id in hits else 1
                return result[1]
            misses[id] = (1 + misses[id]) if id in misses else 1
            result = func(*args, **kwds)
            # if (misses + hits) % 100 == 0:
            #     print(f"Cache stats: hits={hits}, misses={misses}")
            if get_depth(args) == 0:
                return result
            cache[id][key] = (depth, result)
            return result

        def cache_info(id):
            """Report cache statistics"""
            return _CacheInfo(
                hits[id] if id in hits else 0,
                misses[id] if id in misses else 0,
                cache_len(id)(),
            )

        def cache_clear():
            """Clear the cache and cache statistics"""
            nonlocal hits, misses
            cache.clear()
            hits = misses = {}

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        wrapper.__wrapped__ = func

        return wrapper

    return decorator


def alpha_beta_cache():
    """
    A decorator that caches the results of a computer agents min and max functions with the game state grid as the key

    This one is specifically for the alpha-beta agent, as it considers the values of alpha and beta as well as the depth for cache invalidation
    """

    def decorator(func):
        # Constants shared by all lru cache instances:
        sentinel = object()
        # build a key from the function arguments
        make_key = lambda args, _: args[1].grid
        # get the depth from the function arguments
        get_depth = lambda args: args[2]
        # get the alpha value from the function arguments
        get_alpha = lambda args: args[3]
        # get the beta value from the function arguments
        get_beta = lambda args: args[4]

        cache: dict[Any, dict[Grid, tuple[int, float, float, LGameAction | None]]] = {}
        hits, misses = {}, {}
        # bound method to lookup a key or return None
        cache_get = lambda id: cache[id].get
        cache_len = lambda id: cache[id].__len__  # get cache size without calling len()

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds)
            depth = get_depth(args)
            alpha = get_alpha(args)
            beta = get_beta(args)
            id = args[0].id

            # if the agent doesn't have a cache, create one
            if id not in cache:
                cache[id] = {}
                hits[id], misses[id] = 0, 0

            result = cache_get(id)(key, sentinel)
            if (
                result is not sentinel
                and isinstance(result, tuple)
                and result[0] >= depth
                and result[1] <= alpha
                and result[2] >= beta
            ):
                hits[id] = (1 + hits[id]) if id in hits else 1
                return result[3]
            misses[id] = (1 + misses[id]) if id in misses else 1
            result = func(*args, **kwds)
            # if (misses + hits) % 100 == 0:
            #     print(f"Cache stats: hits={hits}, misses={misses}")
            if get_depth(args) == 0:
                return result
            cache[id][key] = (depth, alpha, beta, result)
            return result

        def cache_info(id):
            """Report cache statistics"""
            return _CacheInfo(
                hits[id] if id in hits else 0,
                misses[id] if id in misses else 0,
                cache_len(id)(),
            )

        def cache_clear():
            """Clear the cache and cache statistics"""
            nonlocal hits, misses
            cache.clear()
            hits = misses = {}

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        wrapper.__wrapped__ = func

        return wrapper

    return decorator


def aggressive_heuristic(state: LGameState, agent_id: int) -> float:
    """
    A heuristic function that evaluates the mobility of agents in the L-game

    specifically, the heuristic value is the reciprocal of the number of available moves for the other agent (the opponent)

    The idea is that less available moves for the opponent is better for the agent

    Args:
        state (LGameState): the current game state
        agent_id (int): the ID of the agent to evaluate

    Returns:
        float: the heuristic value
    """
    available_moves = len(state.get_legal_actions(1 - agent_id)) // 13
    try:
        return 1.0 / available_moves
    except ZeroDivisionError:
        return float("inf")


def defensive_heuristic(state: LGameState, agent_id: int) -> float:
    """
    A heuristic function that evaluates the mobility of agents in the L-game

    specifically, the heuristic value is the number of available moves for the agent

    The idea is that more available moves for the agent is better for the agent

    Args:
        state (LGameState): the current game state
        agent_id (int): the ID of the agent to evaluate

    Returns:
        float: the heuristic value
    """
    available_moves = len(state.get_legal_actions(agent_id)) // 13
    return available_moves


class ComputerAgent(Agent[LGameAction, LGameState]):
    """
    An abstract base class for a computer agent
    """

    __slots__ = ["id", "depth", "evaluation_function"]

    def __init__(
        self,
        agent_id: int,
        depth: int,
        evaluation_function: Callable[[LGameState, int], float],
    ):
        self.id = agent_id
        self.depth = depth
        self.evaluation_function = evaluation_function

    @classmethod
    def get_rules(cls) -> AgentRules[LGameAction, LGameState]:
        """
        Get the rules for this agent

        Returns:
            AgentRules[LGameAction, LGameState]: the rules for the agent
        """
        return LGameRules()

    def other_id(self) -> int:
        """
        Get the ID of the other agent

        Returns:
            int: the ID of the other agent
        """
        return 1 - self.id

    @classmethod
    @abc.abstractmethod
    def get_cache_info(cls, id: int) -> dict[str, _CacheInfo]:
        """
        Get the cacheinfo for cached functions
        """
        ...


class MinimaxAgent(ComputerAgent):
    """
    A computer agent that uses minimax to play the L-game
    """

    def get_action(self, state: LGameState) -> LGameAction:
        """
        Get the next action for the computer agent
        """
        (_, action) = self.max_value(state, self.depth)
        if not action:
            raise ValueError("No legal actions")
        return action

    @minimax_cache()
    def max_value(
        self, state: LGameState, depth: int
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the max player (red)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore

        Returns:
            (float, Direction): a utility action pair
        """
        if depth <= 0:
            return (self.evaluation_function(state, self.agent_id()), None)
        if state.is_loss(self.agent_id()):
            return (float("-inf"), None)

        max_value, best_action = float("-inf"), None
        legal_actions = state.get_legal_actions(self.agent_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.agent_id())
            (min_value, _) = self.min_value(successor, depth)
            if min_value > max_value or best_action is None:
                max_value, best_action = min_value, action

        return (max_value, best_action)

    @minimax_cache()
    def min_value(
        self, state: LGameState, depth: int
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the min player (blue)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore

        Returns:
            (float, LGameAction): a utility action pair
        """
        if depth <= 0:
            return (self.evaluation_function(state, self.other_id()), None)
        if state.is_loss(self.other_id()):
            return (float("inf"), None)

        min_value, best_action = float("inf"), None
        legal_actions = state.get_legal_actions(self.other_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.other_id())
            (max_value, _) = self.max_value(successor, depth - 1)
            if max_value < min_value or best_action is None:
                min_value, best_action = max_value, action
        return (min_value, best_action)

    @classmethod
    def get_cache_info(cls, id: int) -> dict[str, _CacheInfo]:
        return {
            "max_value": cls.max_value.cache_info(id),
            "min_value": cls.min_value.cache_info(id),
        }


class AlphaBetaAgent(ComputerAgent):
    """
    A computer agent that uses heuristic Alpha-Beta pruning to play the L-game
    """

    def get_action(self, state: LGameState) -> LGameAction:
        """
        Get the next action for the computer agent
        """
        (_, action) = self.max_value(state, self.depth, float("-inf"), float("inf"))
        if not action:
            raise ValueError("No legal actions")
        return action

    @alpha_beta_cache()
    def max_value(
        self, state: LGameState, depth: int, alpha: float, beta: float
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the max player (red)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore
            alpha (float): the best value for the max player
            beta (float): the best value for the min player

        Returns:
            (float, LGameAction): a utility action pair
        """
        if depth <= 0:
            return (self.evaluation_function(state, self.agent_id()), None)
        if state.is_loss(self.agent_id()):
            return (float("-inf"), None)

        value, best_action = float("-inf"), None
        legal_actions = state.get_legal_actions(self.agent_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.agent_id())
            (min_value, _) = self.min_value(successor, depth, alpha, beta)
            if min_value > value or best_action is None:
                value, best_action = min_value, action
                alpha = max(alpha, value)
            if value > beta:
                return (value, best_action)
        return (value, best_action)

    @alpha_beta_cache()
    def min_value(
        self, state: LGameState, depth: int, alpha: float, beta: float
    ) -> tuple[float, LGameAction | None]:
        """find the optimal action for the min player (blue)

        Args:
            state (LGameState): the current game state
            depth (int): the remaining depth to explore
            alpha (float): the best value for the max player
            beta (float): the best value for the min player

        Returns:
            (float, LGameAction): a utility action pair
        """
        if depth <= 0:
            return (self.evaluation_function(state, self.other_id()), None)
        if state.is_loss(self.other_id()):
            return (float("inf"), None)

        value, best_action = float("inf"), None
        legal_actions = state.get_legal_actions(self.other_id())
        for action in legal_actions:
            successor = state.generate_successor(action, self.other_id())

            (max_value, _) = self.max_value(successor, depth - 1, alpha, beta)

            if max_value < value or best_action is None:
                value, best_action = max_value, action
                beta = min(beta, value)
            if value < alpha:
                return (value, best_action)
        return (value, best_action)

    @classmethod
    def get_cache_info(cls, id: int) -> dict[str, _CacheInfo]:
        return {
            "max_value": cls.max_value.cache_info(id),
            "min_value": cls.min_value.cache_info(id),
        }


class HumanAgent(Agent[LGameAction, LGameState]):
    """
    A human agent for the L-game
    """

    def get_action(self, state: LGameState) -> LGameAction:
        """
        Get the next action from the human player

        Args:
            state (LGameState): the current game state

        Returns:
            LGameAction: the next action to take
        """

        # prompt user for action
        command = input(
            """
Enter your move (type `help` for formatting instructions, `legal` for legal l-moves, `exit` to quit,
`transpose` to transpose the board, `flip` to flip the board accross X axis, 'mirror' to flip the board accross Y axis):
"""
        ).strip()

        if command == "help":
            print(
                "Enter the coordinates of the L-piece's corcer and orientation, and optionally the current and desired position of a neutral piece"
            )
            print(
                """Example: 1 2 E 4 3 1 1 
where (1,2) are the (x,y) coordinates of the corner of the L (where (1,1) is the top left grid position) 
and E is the orientation of the foot of the L (out of North, South, East, West); 
and a neutral piece is moved from (4,3) to (1,1). If not moving a neutral piece, omit the part 4 3 1 1."""
            )
            return self.get_action(state)
        if command == "legal":
            print("Legal L-moves:")
            if self.agent_id() == 0:
                moves = state.grid.get_red_legal_moves() or []
            if self.agent_id() == 1:
                moves = state.grid.get_blue_legal_moves() or []

            for move in moves:
                denorm_move = move.unapply_transformations(state.grid.transformations)
                print(
                    f"\t{denorm_move.corner.x + 1} {denorm_move.corner.y +
                                                    1} {denorm_move.orientation.value}"
                )
            return self.get_action(state)
        if command == "exit":
            exit()
        if command == "transpose":
            state = state.copy(grid=state.grid.transpose())
            print(state.render())
            return self.get_action(state)
        if command == "mirror":
            state = state.copy(grid=state.grid.mirror())
            print(state.render())
            return self.get_action(state)

        # parse the command
        try:
            action = self.parse_command(command)
        except ValueError as e:
            print(f"{e}")
            return self.get_action(state)

        # transform action to normalized grid
        action = action.apply_transformations(state.grid.transformations)

        # check if the action is legal
        if action not in self.get_rules().get_legal_actions(state, self.id):
            print("Illegal move")
            return self.get_action(state)

        return action

    @staticmethod
    def parse_command(command: str) -> LGameAction:
        args = command.split(" ")
        if len(args) != 3 and len(args) != 7:
            raise ValueError(
                "Invalid number of arguments, type `help` for instructions"
            )

        x, y = map(int, args[:2])
        orientation = Orientation(args[2])

        if len(args) == 3:
            return LGameAction(LPiecePosition(Coordinate(x - 1, y - 1), orientation))
        else:
            nx1, ny1, nx2, ny2 = map(int, args[3:])
            return LGameAction(
                LPiecePosition(Coordinate(x - 1, y - 1), orientation),
                (
                    NeutralPiecePosition(Coordinate(nx1 - 1, ny1 - 1)),
                    NeutralPiecePosition(Coordinate(nx2 - 1, ny2 - 1)),
                ),
            )

    @classmethod
    def get_rules(cls) -> AgentRules[LGameAction, LGameState]:
        """
        Get the rules for the human agent

        Returns:
            AgentRules[LGameAction, LGameState]: the rules for the agent
        """
        return LGameRules()


def prepopulate_legal_actions_cache():
    """
    Prepopulate the legal actions cache for terminal states

    We do this by performing a breadth-first graph search over the state space, starting from the initial state and depth-limited to a depth of 3

    This doesn't actually check every possible state, but it gets over 99.7% of them
    """
    print("Prepopulating legal actions cache")

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
    visited = set()
    while len(frontier) > 0:
        (state, agent_id, depth) = frontier.pop()

        if depth >= DEPTH_LIMIT:
            continue

        visited.add(state.grid)

        for action in state.get_sorted_legal_actions(agent_id):
            new_state = state.generate_successor(action, agent_id)
            if new_state.grid not in visited:
                if depth + 1 < DEPTH_LIMIT:
                    frontier.append((new_state, 1 - agent_id, depth + 1))
                visited.add(new_state.grid)

                if len(visited) % 100 == 0:
                    print(f"visited {len(visited)} states")

    print(f"Done prepopulating legal actions cache, visited {len(visited)} states")


prepopulate_legal_actions_cache()


@dataclass(frozen=True, slots=True)
class LGame:
    """
    Handles the game loop and such for the L-game
    """

    initial_state: LGameState

    def run_step(self, state: LGameState) -> tuple[LGameState, int | None]:
        """
        Play a single step of the game
        returns the updated state
        """

        new_state = state.normalize()

        for i, agent in enumerate(state.agents):
            if isinstance(agent, ComputerAgent):
                print(f"\nplayer {i+1} is thinking")
                start_time = time.time()

            # get the next action
            action = agent.get_action(new_state)

            if isinstance(agent, ComputerAgent):
                denormed_action = action.unapply_transformations(
                    new_state.grid.transformations
                )
                ellapsed = time.time() - start_time
                print(
                    f"\tplayer {i+1} chose: {str(denormed_action)} in {ellapsed:.2f}s"
                )
                for func_name, info in agent.get_cache_info(i).items():
                    print(f"{func_name} stats:{info}")

            # generate the successor state
            new_state = new_state.generate_successor(action, i)

            # render new state
            print()
            print(new_state.render())

            if new_state.is_win(i):
                return new_state, i

        return new_state, None

    def prepopulate_caches(self):
        """
        Runs a step of the game to populate the caches of the computer agents up to some depth
        """

        state = self.initial_state.normalize()
        for i, agent in enumerate(state.agents):
            if not isinstance(agent, ComputerAgent):
                continue

            start_time = time.time()

            if isinstance(agent, MinimaxAgent):
                print(f"partially prepopulating Minimax cache for player {i+1}")
                agent.max_value(state, min(agent.depth, 1))
            elif isinstance(agent, AlphaBetaAgent):
                print(f"partially populate AlphaBeta cache for player {i+1}")
                agent.max_value(state, min(agent.depth, 2), float("-inf"), float("inf"))

            ellapsed = time.time() - start_time
            print(f"took {ellapsed:.2f}s")
            for func_name, info in agent.get_cache_info(i).items():
                print(f"{func_name} stats:{info}")

    def run(self):
        """
        Run the game loop
        """

        # TODO: do something special if the state is terminal (e.g., print the winner)
        state = self.initial_state.normalize()
        # render the starting state
        print(state.render())
        while not state.is_terminal():
            state, winner = self.run_step(state)

            if winner is not None:
                print(f"player {winner + 1} wins!")
        print("Game over")


# parse command line arguments
class AgentType(StrEnum):
    HUMAN = "human"
    MINIMAX = "minimax"
    ALPHABETA = "alphabeta"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(s: str) -> "AgentType":
        try:
            return AgentType(s.lower())
        except ValueError:
            raise ValueError(f"Invalid agent type: {s}")

    def get_agent(
        self, id: int, depth: int | None = None, heuristic: str | None = None
    ) -> Agent[LGameAction, LGameState]:
        """
        Get an agent of the specified type
        """
        if depth is None:
            depth = randint(1, 3)
            print(f"Agent {id} given random depth: {depth}")

        if heuristic is None:
            heuristic = "aggressive"

        match heuristic:
            case "aggressive":
                agent_heuristic = aggressive_heuristic
            case "defensive":
                agent_heuristic = defensive_heuristic
            case _:
                raise ValueError(f"Invalid heuristic: {heuristic}")

        match self:
            case AgentType.HUMAN:

                return HumanAgent(id)
            case AgentType.MINIMAX:

                return MinimaxAgent(id, depth, agent_heuristic)
            case AgentType.ALPHABETA:

                return AlphaBetaAgent(id, depth, agent_heuristic)


# parse command line arguments to create the agents
parser = argparse.ArgumentParser(description="Play the L-game")
parser.add_argument(
    "-p1", type=AgentType.from_str, required=True, help="Type of agent for player 1"
)
parser.add_argument("-d1", type=int, help="Depth limit of the search tree for player 1")
parser.add_argument(
    "-h1",
    type=str,
    help="Heuristic function for player 1",
    choices=["aggressive", "defensive"],
)
parser.add_argument(
    "-p2", type=AgentType.from_str, required=True, help="Type of agent for player 2"
)
parser.add_argument("-d2", type=int, help="Depth limit of the search tree for player 2")
parser.add_argument(
    "-h2",
    type=str,
    help="Heuristic function for player 2",
    choices=["aggressive", "defensive"],
)
parser.add_argument(
    "-d",
    type=int,
    help="Depth limit of the search tree for computer players (overridden by -d1 and -d2)",
)

args = parser.parse_args()

if hasattr(args, "help"):
    parser.print_help()
    exit()

# create the agents
player1 = args.p1.get_agent(0, args.d1 or args.d)
player2 = args.p2.get_agent(1, args.d2 or args.d)

# initialize the game
game = LGame(LGameState((player1, player2)))

game.prepopulate_caches()

# run the game loop
game.run()
