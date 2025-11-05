from __future__ import annotations

import asyncio
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import minillmlib as mll
from pymongo.collection import Collection

from bet.Factories.Factory import FactoryType
from bet.Factories.Transforms import GlobalTransform, IndividualTransform
from bet.utils import flatten_dict, logger

from .Unit import Unit


class Ensemble:
    def __init__(self,
        units: List[Unit] | None = None,
        role: str | None = None,
        merge_with: str = None,  # one of [None, "suffix", "prefix"]
        prompt_item_collection: Collection | None = None,
    ):
        self.units: List[Unit] = units if units else []

        self._apply_scenario: GlobalTransform = None

        self._global_transform: Dict[str, List[GlobalTransform]] = {
            FactoryType.REWRITE_GLOBAL_TRANSFORM: [],
            FactoryType.FUZZING_GLOBAL_TRANSFORM: [],
            FactoryType.ENCRYPTION_GLOBAL_TRANSFORM: [],
        }

        self._individual_transform: List[IndividualTransform] = []

        assert merge_with in [
            None,
            "suffix",
            "prefix",
        ], "merge_with must be one of [None, 'suffix', 'prefix']"

        self.prefix: Ensemble = None
        self.suffix: Ensemble = None
        self.merge_with = merge_with

        # Can be None and set later on, but it must be set to generate a prompt
        self.role = role
        self.prompt_item_collection = prompt_item_collection

    def add_unit(self,
        unit: Unit,
        end: bool = True
    ):
        if end:
            self.units.append(unit)
        else:
            self.units = [unit] + self.units

    async def individual_transform(self,
        transform: IndividualTransform,
        assistant: mll.GeneratorInfo
    ):
        tasks = [
            unit.add_transform(
                transform=deepcopy(transform), 
                assistant=assistant
            ) for unit in self.units
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                {
                    "type": "transform_error",
                    "error": str(e),
                    "transform": transform.full_name(),
                }
            )
            raise e

        self._individual_transform.append(deepcopy(transform))

    def global_transform(self,
        transform: GlobalTransform,
        _type: FactoryType
    ):
        self._global_transform[_type].append(deepcopy(transform))

    def apply_scenario(self,
        transform: GlobalTransform
    ):
        self._apply_scenario = deepcopy(transform)

    def add_fix(self,
        ensemble: Ensemble,
        prefix: bool
    ):
        anchor_point = getattr(self, "prefix" if prefix else "suffix")

        if anchor_point is not None:
            anchor_point.add_fix(ensemble=ensemble, prefix=prefix)
        else:
            if prefix:
                last_suffix = ensemble.get_last_suffix()
                last_suffix.suffix = self
                self.prefix = last_suffix
            else:
                first_prefix = ensemble.get_first_prefix()
                first_prefix.prefix = self
                self.suffix = first_prefix

    def get_applied_global_transform(self) -> List[str]:
        return [
            str(transform)
            for transform in flatten_dict(self._global_transform, deduplicate=True)
            if transform.applied
        ]

    def get_individual_transform(self) -> List[str]:
        return [str(transform) for transform in self._individual_transform]

    def __str__(self) -> str:
        # Unit order has an impact
        units_name = [str(unit) for unit in self.units]

        # Only select the global that are applied or just going to be applied
        # We do that because when retrieving pre-computed transform, you can only
        # use precomputation that had the same transform applied before.
        global_transforms = sorted([gt for gt in self.get_applied_global_transform()])

        return (
            ";".join(units_name)
            + "-"
            + "scenario:"
            + str(self._apply_scenario)
            + (f"|{'-'.join(global_transforms)}|" if len(global_transforms) > 0 else "")
        )

    async def generate_prompts(self,
        n: int,
        resolve_suffix: bool = True,
        assistant: mll.GeneratorInfo | None = None
    ) -> List[mll.ChatNode]:
        if n <= 0:
            raise ValueError("n must be greater than 0")

        if self.role is None:
            raise ValueError(
                f"{str(self)} has no role, one must be assigned before generating the prompts"
            )

        if resolve_suffix and self.suffix:
            # Step 1: Go to the last suffix
            return await self.suffix.generate_prompts(
                n=n, 
                assistant=assistant
            )

        nodes: List[mll.ChatNode] = [None] * n

        # While going up the chain, merge ensemble that need to be merged
        current_ensemble = self
        while current_ensemble.prefix:
            if (
                current_ensemble.merge_with == "prefix"
                or current_ensemble.prefix.merge_with == "suffix"
            ):
                current_ensemble = merge_ensembles(
                    current_ensemble.prefix, current_ensemble
                )
            else:
                break

        if current_ensemble.prefix:
            # Step 2: If there is still a prefix, keep accumulating all prefixes into a node
            nodes = await current_ensemble.prefix.generate_prompts(
                n=n, 
                resolve_suffix=False,
                assistant=assistant
            )

        # Step 3: sample random content from units
        content: List[str] = [""] * n
        for i in range(n):
            content[i] = " ".join(
                [
                    " ".join(unit.generate_random_variation())
                    for unit in current_ensemble.units
                ]
            )

        data_buffer = defaultdict(dict)

        # Step 4: apply scenario
        if current_ensemble._apply_scenario is not None:
            for i in range(n):
                if nodes[i] is not None and "brainstorming" in nodes[i].metadata:
                    # current_ensemble._apply_scenario.parent is an ApplyScenarioFactory here
                    data_buffer["brainstorming"][str(i)] = nodes[i].metadata["brainstorming"]

            content = await current_ensemble._apply_scenario.apply_transform_lst(
                content=content,
                previous_transform=(
                    current_ensemble.get_individual_transform(),
                    current_ensemble.get_applied_global_transform(),
                ),
                prompt_item_collection=current_ensemble.prompt_item_collection,
                data_buffer=data_buffer,
                assistant=assistant
            )

        # Step 5: apply each global transform
        for cat in [
            FactoryType.REWRITE_GLOBAL_TRANSFORM,
            FactoryType.FUZZING_GLOBAL_TRANSFORM,
            FactoryType.ENCRYPTION_GLOBAL_TRANSFORM,
        ]:
            for transform in current_ensemble._global_transform[cat]:
                content = await transform.apply_transform_lst(
                    content=content,
                    previous_transform=(
                        current_ensemble.get_individual_transform(),
                        current_ensemble.get_applied_global_transform(),
                    ),
                    prompt_item_collection=current_ensemble.prompt_item_collection,
                    assistant=assistant
                )

        # Step 6: Create new node from ensemble content and add them to the nodes
        for i in range(n):
            new_node = mll.ChatNode(role=current_ensemble.role, content=content[i])

            if nodes[i] is not None and "brainstorming" in nodes[i].metadata:
                new_node.metadata["brainstorming"] = nodes[i].metadata["brainstorming"]

            # NOTE: Maybe we could aggregate brainstorming instead
            if "brainstorming" in data_buffer:
                new_node.metadata["brainstorming"] = data_buffer["brainstorming"][str(i)]

            nodes[i] = (
                nodes[i].add_child(new_node) if nodes[i] is not None else new_node
            )

        return nodes

    def get_first_prefix(self) -> Ensemble:
        prefix = self
        while prefix.prefix is not None:
            prefix = prefix.prefix
        return prefix

    def get_last_suffix(self) -> Ensemble:
        suffix = self
        while suffix.suffix is not None:
            suffix = suffix.suffix
        return suffix

    def __deepcopy__(self, memo):
        """Custom deepcopy that handles the non-copyable prompt_item_collection."""
        # Create a new instance without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes with special handling for some
        for key, value in self.__dict__.items():
            if key == 'prompt_item_collection':
                # Share the collection reference instead of copying
                setattr(result, key, value)
            elif key == 'units':
                # Deepcopy each unit individually
                setattr(result, key, [unit.deepcopy() for unit in value])
            else:
                # Deepcopy other attributes
                setattr(result, key, deepcopy(value, memo))

        return result

    def deepcopy(self) -> Ensemble:
        """Convenience method for deepcopy."""
        return deepcopy(self)


def merge_ensembles(
    ensemble_a: Ensemble,
    ensemble_b: Ensemble,
) -> Ensemble:
    new_merge_with = None
    if ensemble_a.merge_with == "suffix" and ensemble_b.merge_with != "prefix":
        new_merge_with = ensemble_b.merge_with

    new_ensemble = Ensemble(
        units=ensemble_a.units + ensemble_b.units,
        role=ensemble_b.role if ensemble_a.role is None else ensemble_a.role,
        merge_with=new_merge_with,
        prompt_item_collection=ensemble_b.prompt_item_collection if ensemble_a.prompt_item_collection is None else ensemble_a.prompt_item_collection,
    )

    new_ensemble._apply_scenario = (
        ensemble_b._apply_scenario
        if ensemble_a._apply_scenario is None
        else ensemble_a._apply_scenario
    )

    for cat in ensemble_a._global_transform:
        new_ensemble._global_transform[cat] = (
            ensemble_a._global_transform[cat] + ensemble_b._global_transform[cat]
        )

    new_ensemble._individual_transform = (
        ensemble_a._individual_transform + ensemble_b._individual_transform
    )

    new_ensemble.prefix = ensemble_a.prefix
    new_ensemble.suffix = ensemble_b.suffix

    return new_ensemble
