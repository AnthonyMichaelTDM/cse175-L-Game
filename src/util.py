"""
Utility classes and functions.
"""

from enum import Enum
from typing import Generic, Self, TypeVar

T = TypeVar("T")


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

    Orientation.NORTH.index  # 0
    Orientation.NORTH.value  # "N"
    Orientation("N")  # Orientation.NORTH
    Orientation(0) # Orientation.NORTH
    Orientation.EAST.index  # 1
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
