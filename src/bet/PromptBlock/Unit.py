from __future__ import annotations

from copy import deepcopy
from typing import List

import minillmlib as mll
from numpy import random
from pymongo.collection import Collection

from bet.Factories.Transforms import IndividualTransform
from bet.utils import unauthorized_tks_msg, validate_name


class Unit:
    def __init__(self,
        name: str,
        content: List[List[str]],
        selected_sentences: int,
        select_from_end: bool = False,  # Start from the last sentence instead of the first when selecting sentences
        prompt_item_collection: Collection = None,
    ):
        if len(content) == 0:
            raise ValueError(
                "content should not be empty, you must create it before setting up a unit"
            )

        if selected_sentences < 1:
            raise ValueError(
                f"selected_sentences should be at least 1: {selected_sentences}"
            )

        if any(
            len(variation) < selected_sentences
            or any(len(sentence) < 1 for sentence in variation)
            for variation in content
        ):
            raise ValueError(
                f"content should not contain empty variations, each variations must have at least {selected_sentences} sentences: {content}"
            )

        if not validate_name(name, exclude=[":"]):
            raise ValueError(unauthorized_tks_msg.format(entity="Unit name", name=name))

        # initial content saved in case it is needed (content before any transform is applied)
        self.initial_content = content

        self.content = content
        self.applied_transforms: List[IndividualTransform] = []
        self.name = name
        self.select_from_end = select_from_end
        self.prompt_item_collection = prompt_item_collection

        self.selected_sentences = selected_sentences

    def __str__(self) -> str:
        return "-".join(
            [self.name] + sorted([str(it) for it in self.applied_transforms])
        )

    async def add_transform(self, 
        transform: IndividualTransform, 
        assistant: mll.GeneratorInfo
    ) -> None:
        self.applied_transforms.append(transform)
        search_unit = (
            self.prompt_item_collection.find_one({"name": str(self)}) if self.prompt_item_collection is not None else None
        )
        if search_unit is not None:
            self.content = search_unit["content"]
        else:
            self.content = await transform.apply_transform_lst(
                content=self.content,
                previous_transform=[str(it) for it in self.applied_transforms[:-1]],
                prompt_item_collection=self.prompt_item_collection,
                assistant=assistant
            )

            if self.prompt_item_collection is not None:
                self.prompt_item_collection.insert_one({"name": str(self), "content": self.content})

    def generate_variation(self, 
        n: int
    ) -> List[str]:
        if not self.select_from_end:
            return self.content[n][: self.selected_sentences]
        else:
            return self.content[n][::-1][: self.selected_sentences][::-1]

    def generate_random_variation(self) -> List[str]:
        return self.generate_variation(random.randint(len(self.content)))

    def __deepcopy__(self, memo):
        """Custom deepcopy that handles the non-copyable prompt_item_collection."""
        # Create a new instance without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except prompt_item_collection
        for key, value in self.__dict__.items():
            if key == 'prompt_item_collection':
                # Share the collection reference instead of copying
                setattr(result, key, value)
            else:
                # Deepcopy other attributes
                setattr(result, key, deepcopy(value, memo))
        
        return result
    
    def deepcopy(self) -> Unit:
        """Convenience method for deepcopy."""
        return deepcopy(self)

    def __len__(self):
        return self.selected_sentences
