from __future__ import annotations

import asyncio
from typing import Dict, List

import minillmlib as mll

from bet.Factories import (EXECUTION_ORDER, AddFactory,
                           AdditionalGenerationInstructionFactory, Factory,
                           FactoryType, IndividualTransformFactory,
                           WrapFactory)
from bet.PromptBlock import Ensemble
from bet.utils import flatten_dict, logger

from .Family import Family


class Primitive:
    def __init__(self,
        factories: List[Factory],
        complexity: int,
        levels: List[int] | int = 0,
        families: List[Family] = [],
        base: bool = False,
        duplicate_in_other: bool = False,
        prevent_in_other: bool = False,
        craft_time: int = 60,  # In second
        linked_primitives_names: (
            List[str] | None
        ) = None,  # Note that here it must be object that is in the other lib, not the simple_name() of the primitive
        can_be_selected: bool = True,  # If False, primitive can only be added through linking
        nefarious: bool = False,  # If this is true, when we do the "over-refusal" studies, we will not use those primitives as this is framing the prompt in a bad way, which should be refused
        readable_name: str | None = None,
        description: str | None = None
    ) -> None:

        # If this is true, and this is selected by either the instruction or the request, then it will be duplicated in the other one
        self.duplicate_in_other = duplicate_in_other
        self.prevent_in_other = prevent_in_other

        # Store the names of primitives this should be linked to in the other list
        self.linked_primitives_names: List[str] = (
            linked_primitives_names if linked_primitives_names is not None else []
        )

        self.levels = levels

        # Sort factories in their correct categories
        self.actions = {
            FactoryType.ADD: self.add,
            FactoryType.INDIVIDUAL_TRANSFORM: self.individual_transform,
            FactoryType.REWRITE_GLOBAL_TRANSFORM: self.rewrite_global_transform,
            FactoryType.FUZZING_GLOBAL_TRANSFORM: self.fuzzing_global_transform,
            FactoryType.ENCRYPTION_GLOBAL_TRANSFORM: self.encryption_global_transform,
            FactoryType.WRAP: self.wrap,
        }

        if isinstance(levels, int):
            levels = [levels] * len(factories)

        if len(factories) != len(levels):
            raise ValueError("Number of factories and levels must be the same")

        if 0 not in levels and len(factories) > 0:
            raise ValueError("At least one level must be 0 for the default primitive")

        self.additional_generation_instructions = []
        self.factories: Dict[str, List[Factory]] = {k: [] for k in EXECUTION_ORDER}
        self.factories[FactoryType.ADDITIONAL_GENERATION_INSTRUCTION] = []

        factory: Factory
        for level, factory in zip(levels, factories):
            if factory.type in self.factories and factory.type in self.actions:
                saved_factory = factory.deepcopy()
                saved_factory.update_level(level)
                self.factories[factory.type].append(saved_factory)
            elif factory.type == FactoryType.ADDITIONAL_GENERATION_INSTRUCTION:
                factory: AdditionalGenerationInstructionFactory
                self.additional_generation_instructions.append(
                    factory.additional_generation_instruction
                )
                self.factories[factory.type].append(factory)
            else:
                raise ValueError(
                    f"FactoryType not added to execution order, or the function hasn't been implemented: {factory.type}. To remove this error, you must add this new type to the EXECUTION_ORDER, and implement an action for this type (Ideally, do that in the parent Primitive class)"
                )
        # Between 0 (normal prompting), 3 (technique that you'd have to know about), and 5 (our best and not available on internet)
        self.complexity = complexity

        self.level = 0
        self.max_level = max(levels, default=0)

        self.families = families

        self.craft_time = craft_time

        self.base = base

        self.can_be_selected = can_be_selected
        self.nefarious = nefarious

        self.readable_name = readable_name

        if readable_name is not None and " x " in readable_name:
            raise ValueError("Readable name cannot contain ' x ' as this will conflict when doing the shap value")

        self.description = description if description is not None else ""

    def get_readable_name(self) -> str:
        return self.readable_name if self.readable_name is not None else self.simple_name()

    def set_primitive_to_request_mode(self) -> None:
        for factory in flatten_dict(self.factories):
            factory.request = True

    def post_init_factories(self, overwrite: bool = False) -> None:
        for factory in flatten_dict(self.factories):
            factory.post_init(overwrite=overwrite)

    def random_select_parameters(
        self, overwrite: bool = False, p_overwrite: float = 1
    ) -> None:
        for factory in flatten_dict(self.factories):
            factory.random_select_parameters(
                overwrite=overwrite, p_overwrite=p_overwrite
            )

    async def add(self, 
        ensemble: Ensemble, 
        _assistant: mll.GeneratorInfo | None
    ) -> None:
        factory: AddFactory
        for factory in self.factories[FactoryType.ADD]:
            if factory.level <= self.level:
                ensemble.add_unit(
                    await factory.make_unit(
                        prompt_item_collection=ensemble.prompt_item_collection,
                        assistant=_assistant
                    )
                )

    async def individual_transform(self, 
        ensemble: Ensemble,
        _assistant: mll.GeneratorInfo | None
    ) -> None:
        factory: IndividualTransformFactory
        for factory in self.factories[FactoryType.INDIVIDUAL_TRANSFORM]:
            if factory.level <= self.level:
                await ensemble.individual_transform(
                    transform=factory.transform,
                    assistant=_assistant
                )

    async def global_transform(self, 
        ensemble: Ensemble, 
        _type: FactoryType
    ) -> None:
        for factory in self.factories[_type]:
            if factory.level <= self.level:
                ensemble.global_transform(transform=factory.transform, _type=_type)

    async def rewrite_global_transform(self, 
        ensemble: Ensemble, 
        _assistant: mll.GeneratorInfo | None
    ) -> None:
        await self.global_transform(
            ensemble=ensemble, _type=FactoryType.REWRITE_GLOBAL_TRANSFORM
        )

    async def fuzzing_global_transform(self, 
        ensemble: Ensemble, 
        _assistant: mll.GeneratorInfo | None
    ) -> None:
        await self.global_transform(
            ensemble=ensemble, _type=FactoryType.FUZZING_GLOBAL_TRANSFORM
        )

    async def encryption_global_transform(self, 
        ensemble: Ensemble, 
        _assistant: mll.GeneratorInfo | None
    ) -> None:
        await self.global_transform(
            ensemble=ensemble, _type=FactoryType.ENCRYPTION_GLOBAL_TRANSFORM
        )

    async def wrap(self, 
        ensemble: Ensemble, 
        _assistant: mll.GeneratorInfo | None
    ) -> None:
        factory: WrapFactory
        for factory in self.factories[FactoryType.WRAP]:
            if factory.level <= self.level:
                factory.connect(ensemble=ensemble)

    async def is_viable(self, tested_model: mll.GeneratorInfo) -> bool:
        try:
            results = await asyncio.gather(
                *[
                    factory.is_viable(tested_model=tested_model)
                    for factory in flatten_dict(self.factories)
                ]
            )
            return all(results)
        except Exception as e:
            logger.error(
                {
                    "type": "viable_error",
                    "error": str(e),
                    "factory": self.full_name(),
                }
            )
            return False

    def family_distance(self, other: Primitive) -> float:
        if not self.families or not other.families:
            return 1

        shared_families = set(self.families) & set(other.families)
        total_families = set(self.families) | set(other.families)

        return 1 - len(shared_families) / len(total_families) if total_families else 1

    def increase_level(self) -> None:
        self.level = min(self.level + 1, self.max_level)

    def decrease_level(self) -> None:
        self.level = max(self.level - 1, 0)

    def full_name(self) -> str:
        return (
            "P:"
            + "-".join(
                sorted(
                    [factory.full_name() for factory in flatten_dict(self.factories)]
                )
            )
            + f"|Lvl:{self.level}|"
        )
    
    def prebuild(self, *args, **kwargs) -> bool:
        valid = []

        for factory in flatten_dict(self.factories):
            valid.append(factory.prebuild(*args, **kwargs))

        return all(valid)

    def __str__(self) -> str:
        return (
            "P:"
            + "-".join(
                sorted([str(factory) for factory in flatten_dict(self.factories)])
            )
            + f"|Lvl:{self.level}|"
        )

    def simple_name(self) -> str:
        return "P:" + "-".join(
            sorted([factory.simple_name() for factory in flatten_dict(self.factories)])
        )

    def compatible_with_primitive(self, other: Primitive) -> bool:
        return all(family.compatible_with_primitive(other) for family in self.families)

    def compatible_with_primitives(self, others: List[Primitive]) -> bool:
        return all(
            family.compatible_with_primitives(others) for family in self.families
        )

    def __eq__(self, other: Primitive) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __deepcopy__(self, _) -> Primitive:
        new_factories = []
        for _, factories in self.factories.items():
            new_factories += [factory.deepcopy() for factory in factories]

        new_prim = Primitive(
            factories=new_factories,
            complexity=self.complexity,
            families=self.families,
            levels=self.levels,
            base=self.base,
            duplicate_in_other=self.duplicate_in_other,
            prevent_in_other=self.prevent_in_other,
            craft_time=self.craft_time,
            linked_primitives_names=self.linked_primitives_names,
            can_be_selected=self.can_be_selected,
            nefarious=self.nefarious,
            readable_name=self.readable_name,
            description=self.description,
        )

        new_prim.level = self.level

        return new_prim
