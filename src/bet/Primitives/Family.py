from __future__ import annotations

from typing import TYPE_CHECKING, List

# This is for preventing a circular import
if TYPE_CHECKING:
    from bet.GeneticAlgorithm import Individual

    from .Primitive import Primitive


class Family:
    def __init__(self, 
        name: str, 
        description: str | None = None
    ) -> None:
        self.name = name
        self.description = description if description is not None else ""

    def compatible_with_family(self, _: Family) -> bool:
        return True

    def compatible_with_primitive(self, primitive: Primitive) -> bool:
        return all(
            [
                self.compatible_with_family(family)
                and family.compatible_with_family(self)
                for family in primitive.families
            ]
        )

    def compatible_with_primitives(self, primitives: List[Primitive]) -> bool:
        return all(
            [self.compatible_with_primitive(primitive) for primitive in primitives]
        )

    def compatible_with_individual(
        self, other: Individual, _type: str = "instruction"
    ) -> bool:
        if _type not in ["instruction", "request"]:
            raise ValueError(
                f"Invalid type: {_type}, must be 'instruction' or 'request'"
            )

        return self.compatible_with_primitives(other.__dict__[f"{_type}_primitives"])

    def __str__(self) -> str:
        return type(self).__name__ + self.name

    def __eq__(self, other: Family) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class NoSameFamily(Family):
    def compatible_with_family(self, other: Family) -> bool:
        return other.name != self.name


class MaxSameFamily(NoSameFamily):
    def __init__(self, name: str, max_same: int) -> None:
        super().__init__(name)
        self.max_same = max_same

    def compatible_with_primitives(self, primitives: List[Primitive]) -> bool:
        same = 0
        for primitive in primitives:
            if not self.compatible_with_primitive(primitive):
                same += 1

        return same <= self.max_same
