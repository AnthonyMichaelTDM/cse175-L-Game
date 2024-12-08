"""
Utility classes and functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Iterable, Self, TypeVar

T = TypeVar("T")


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


class Transformable(ABC):
    @abstractmethod
    def transpose(self) -> Self:
        pass

    @abstractmethod
    def flip(self) -> Self:
        pass

    @abstractmethod
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

    Orientation.NORTH.index() # 0
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
