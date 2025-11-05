from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any, Dict, List, TypeVar

import minillmlib as mll
from numpy import random
from pymongo.collection import Collection

from bet.utils import logger, unauthorized_tks_msg, validate_name

ParameterOption = TypeVar("ParameterOption")


class FactoryType(StrEnum):
    ADD = "Add"
    INDIVIDUAL_TRANSFORM = "IndividualTransform"
    ADDITIONAL_GENERATION_INSTRUCTION = "AdditionalGenerationInstruction"
    REWRITE_GLOBAL_TRANSFORM = "RewriteGlobalTransform"
    FUZZING_GLOBAL_TRANSFORM = "FuzzingGlobalTransform"
    ENCRYPTION_GLOBAL_TRANSFORM = "EncryptionGlobalTransform"
    WRAP = "Wrap"


EXECUTION_ORDER = [
    FactoryType.ADD,
    FactoryType.INDIVIDUAL_TRANSFORM,
    FactoryType.REWRITE_GLOBAL_TRANSFORM,
    FactoryType.FUZZING_GLOBAL_TRANSFORM,
    FactoryType.ENCRYPTION_GLOBAL_TRANSFORM,
    FactoryType.WRAP,
]

# NOTE: When making a factory name, you must be extra careful that it doesn't conflict with any other factory name, that it doesn't contain the '-' character, and that the same name will not be used for different stuff because it will retrive it from database where-as it should be generated. 
# For example, the ApplyScenarioFactory should have the entire scenario included in it's name because each different scenario will generate completely different prompts and shouldn't be saved in the database under the same name for different stuff.


class Factory:
    def __init__(self, _type: FactoryType, name: str, request: bool = False) -> None:
        """
        Initializes the Factory with the provided type and name.

        Args:
            _type (FactoryType): The type of the Factory.
            name (str): The name of the Factory.
        """
        if not validate_name(name):
            raise ValueError(
                unauthorized_tks_msg.format(entity="Factory name", name=name)
            )

        self.type = _type
        self.name = name

        self.parameters: Dict[str, List[ParameterOption]] = {}
        self.selected: Dict[str, ParameterOption] = {}
        # Save the important parameters in the database name
        self.important_for_db: List[str] = []

        self.request = request
        self.level = 0

        self.saved_info = (
            {}
        )  # When saving the transformed_content, this will also be saved. This is empty but a child class can fill it (e.g. saving the language when translating text)
        self.additionnal_search_info = (
            {}
        )  # When retrieving the transformed_content, this will also be used as a key. This is empty but a child class can fill it (e.g. when retriving for apply scenario, this will also add the additional instructions to the search)

        self.post_init()

    def post_init(self, overwrite=False) -> None:
        self.setup_ga_parameters()
        self.random_select_parameters(overwrite=overwrite)

    def update_level(self, level: int) -> None:
        self.level = level

    def setup_ga_parameters(self) -> None:
        # check that data that will be saved in the db is well formated
        for name in self.important_for_db:
            if not validate_name(name):
                raise ValueError(
                    unauthorized_tks_msg.format(entity="Parameter name", name=name)
                )

        # remove all the parameters with no options:
        processed_params = {}
        for name, parameter in self.parameters.items():
            if type(parameter) == list and len(parameter) != 0:
                processed_params[name] = parameter
        self.parameters = processed_params

    def random_select_parameters(
        self, overwrite: bool = False, p_overwrite: float = 1
    ) -> None:
        """
        Randomly select parameters for the factory.
        """
        for name, parameter in self.parameters.items():
            if (
                overwrite and random.random() < p_overwrite
            ) or name not in self.selected:
                self.selected[name] = parameter[random.randint(len(parameter))]

    def retrieve_selected_parameter(self, name: str) -> Any:
        """
        Retrieve the selected parameter for the given name.
        """
        if name not in self.selected:
            raise ValueError(
                f"{name} not in selected parameter. Did you call random_select_parameters?"
            )
        return self.selected[name]

    # NOTE: I tried to make this a decorator but it didn't work with any function due to args being different each time. Maybe later try to make this a decorator
    def retrieve_and_save(
        self, prompt_item_collection: Collection | None, content: List[str] | str | None = None
    ):
        # Try to retrieve the data from the database in a data_buffer. If called again, it will save self.data_buffer in the database.
        if prompt_item_collection is not None:
            search = {"name": str(self)}

            if content:
                search["content"] = content

            if len(self.additionnal_search_info) > 0:
                for k, v in self.additionnal_search_info.items():
                    search[k] = v

            self.data_buffer: Dict[str, Any] | None = prompt_item_collection.find_one(search)
        else:
            self.data_buffer = None

        yield

        if prompt_item_collection is not None:
            if self.data_buffer is None:
                logger.warning(
                    {
                        "type": "data_buffer_warning",
                        "message": "data buffer is None, it should be filled before calling this method",
                        "factory": str(self),
                    }
                )
                self.data_buffer = {}

            if "name" not in self.data_buffer:
                self.data_buffer["name"] = str(self)

            if "content" not in self.data_buffer and content is not None:
                self.data_buffer["content"] = content

            if len(self.saved_info) > 0:
                for k, v in self.saved_info.items():
                    self.data_buffer[k] = v

            if len(self.additionnal_search_info) > 0:
                for k, v in self.additionnal_search_info.items():
                    self.data_buffer[k] = v

            # First double check if another one has not been inserted while this was running before saving
            if prompt_item_collection.find_one(search) is None:
                prompt_item_collection.insert_one(self.data_buffer)

        yield

    def get_selected_idx(self, param: str) -> int:
        return self.parameters[param].index(self.retrieve_selected_parameter(param))

    async def is_viable(self, tested_model: mll.GeneratorInfo) -> bool:
        return True

    def sorted_selected_keys(self) -> List[str]:
        return sorted(list(self.selected.keys()))

    def full_name(self) -> str:
        if len(self.selected) > 0:
            suffix = f"|{';'.join(
                [
                    f"{param}:{str(self.get_selected_idx(param))}" 
                    for param in self.sorted_selected_keys()
                ]
            )}|"
        else:
            suffix = ""

        return f"F:{self.type}:{self.name}{suffix}"

    def prebuild(self, 
        *args, **kwargs
    ) -> bool:
        return True

    def __str__(self) -> str:
        if len(self.important_for_db) > 0:
            suffix = f"|{';'.join(
                [
                    f"{param}:{str(self.get_selected_idx(param))}"
                    for param in self.sorted_selected_keys()
                    if param in self.important_for_db
                ]
            )}|"
        else:
            suffix = ""

        return f"F:{self.type}:{self.name}{suffix}"

    def simple_name(self) -> str:
        return f"F:{self.type}:{self.name}"

    def __eq__(self, other: Factory) -> bool:
        return str(self) == str(other) and all(
            [
                other.get_selected_idx(param) == self.get_selected_idx(param)
                for param in self.selected
            ]
        )

    def __hash__(self) -> int:
        return hash(str(self))

    def deepcopy(self) -> Factory:
        copied = deepcopy(self)
        copied.reset_pointers()
        return copied

    def create_transform_lambda(self):
        raise NotImplementedError(
            "This factory is a transform that requires a lambda function to be use. Please implement a create_transform_lambda() method for your class, you can look at EncodeFactory and ApplyScenarioFactory in GlobalTransformFactories for inspiration."
        )

    def reset_pointers(self):
        # In case of the factory having a transform, reset the parent and the transform pointers
        if "transform" in self.__dict__:
            self.transform.parent = self
            self.transform.transform = self.create_transform_lambda()
