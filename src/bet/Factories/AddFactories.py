import json
from typing import List, override

import minillmlib as mll
from pymongo.collection import Collection

from bet.PromptBlock import Unit
from bet.utils import (add_request_instruction_prompt, list_to_str_dict,
                       split_sentences, split_variations,
                       strip_merge_and_split)

from .Factory import Factory, FactoryType


class AddFactory(Factory):
    def __init__(self, 
        name: str, 
        request: bool = False
    ) -> None:
        super().__init__(_type=FactoryType.ADD, name=name, request=request)

    async def make_unit(self, 
        prompt_item_collection: Collection, 
        assistant: mll.GeneratorInfo | None = None
    ) -> Unit:
        raise NotImplementedError


class AddTextFactory(AddFactory):
    def __init__(self, 
        name: str, 
        n_sentences: int, 
        request: bool = False
    ) -> None:
        self.n_sentences = n_sentences
        super().__init__(name=name, request=request)

    # NOTE: If you want to add more parameters, don't forget to super this
    def setup_ga_parameters(self) -> None:
        """
        Setup GA parameters for the factory by adding a parameter based on the name and a range of the number of sentences.
        """
        self.parameters["n_sentences"] = [*range(1, self.n_sentences + 1)]
        super().setup_ga_parameters()

    async def make_unit(self, 
        prompt_item_collection: Collection, 
        variations: List[List[str]]
    ) -> Unit:
        """
        Create a unit based on the selected parameters

        Args:
            prompt_item_collection (Collection): The MongoDB collection to use for the unit.
            variations (List[List[str]]): A list of variations to create the unit from.

        Returns:
            Unit: A Unit object created with the specified name, content, and selected sentences.
        """
        _n_sentences: int = self.retrieve_selected_parameter("n_sentences")
        return Unit(
            name="U:" + str(self),
            content=variations,
            selected_sentences=_n_sentences,
            prompt_item_collection=prompt_item_collection,
        )


# AddStaticText takes a list of string (list of variations) and split the strings by sentences and create a unit with this content.
class AddStaticTextFactory(AddTextFactory):
    def __init__(self, 
        name: str, 
        content: List[str],
        request: bool = False
    ) -> None:
        """
        Initializes the AddStaticTextFactory with the provided name and content list.

        Args:
            name (str): The name of the factory.
            content (List[str]): A list of strings to be split into sentences.
        """
        self.content = split_variations(content)
        n_sentences = len(self.content[0])
        assert all(
            len(content) == n_sentences for content in self.content
        ), f"All variations should have the same number of sentences. AddStaticTextFactory: {name}"
        super().__init__(name=name, n_sentences=n_sentences, request=request)

    async def make_unit(
        self, 
        prompt_item_collection: Collection, 
        assistant: mll.GeneratorInfo | None = None
    ) -> Unit:
        """
        Create a unit based on the selected parameters

        Args:
            prompt_item_collection (Collection): The MongoDB collection to use for the unit.

        Returns:
            Unit: A Unit object created with the specified name, content, and selected sentences.
        """
        return await super().make_unit(prompt_item_collection, self.content)

# AddGeneratedText takes a description of a behavior agnostic paragraphe, generate it, and create a unit from it
class AddGeneratedTextFactory(AddTextFactory):
    def __init__(self,
        name: str,
        n_variations: int,
        n_sentences: int,
        description: str,
        example: str,
        additional_instructions: List[str] = None,
        request: bool = False,
    ) -> None:
        """
        Initializes the AddGeneratedTextFactory with the provided parameters.

        Args:
            name (str): The name of the factory.
            n_variations (int): The number of variations to generate.
            n_sentences (int): The number of sentences in each variation.
            description (str): A description of the behavior agnostic paragraphe.
            example (str): An example of the behavior agnostic paragraphe.
        """

        self.n_variations = n_variations
        self.description = description

        split_example = split_sentences(example)
        if len(split_example) > n_sentences:
            raise ValueError(
                f"Example has more sentences than n_sentences. Example: {example}, n_sentences: {n_sentences}. You should set n_sentences to at least {len(split_example)}"
            )
        # Format the example for the assistant
        self.example = list_to_str_dict(split_example, n_sentences)

        self.generation_parameters = mll.GeneratorCompletionParameters(
            max_tokens=2048,
            temperature=1,
            response_format={"type": "json_object"},
        )

        self.additional_instructions = (
            "".join(["\n- " + instr for instr in additional_instructions])
            if additional_instructions is not None
            else ""
        )

        super().__init__(name=name, n_sentences=n_sentences, request=request)

    async def make_unit(self, 
        prompt_item_collection: Collection,
        assistant: mll.GeneratorInfo | None = None
    ) -> Unit:
        """
        Generate or retrieve variations of a behavior agnostic paragraphe and create a unit from it.

        Args:
            prompt_item_collection (Collection): The database to retrieve variations from.

        Returns:
            Unit: A Unit object created with the specified name, content, and selected sentences.
        """
        if assistant is None:
            raise ValueError("Assistant is not set make_unit")

        variations = []

        # Decorator that handle retrival and saving of variations
        retrieve_and_save = super().retrieve_and_save(prompt_item_collection=prompt_item_collection)
        next(retrieve_and_save)

        if self.data_buffer is not None:
            variations = self.data_buffer["variations"]
        else:
            try:
                executant = mll.ChatNode.from_thread(
                    ["prompts/pre_prompt.json", "prompts/add_generated_text.json"]
                )
            except FileNotFoundError:
                executant = mll.ChatNode.from_thread(
                    ["src/eval_task/prompts/pre_prompt.json", "src/eval_task/prompts/add_generated_text.json"]
                )


            self.additional_instructions = add_request_instruction_prompt(
                additional_instructions=self.additional_instructions,
                is_request=self.request,
            )

            executant.update_format_kwargs(
                propagate=True,
                n_sentences=self.n_sentences,
                description=self.description,
                example=self.example,
                additional_instructions=self.additional_instructions,
            )
            result = await executant.complete_async(
                mll.NodeCompletionParameters(
                    gi=assistant,
                    generation_parameters=self.generation_parameters,
                    parse_json=True,
                    crash_on_refusal=True,
                    retry=10,
                    n=self.n_variations
                )
            )

            variations = [
                strip_merge_and_split(
                    [v for _, v in json.loads(variation.content).items()],
                    self.n_sentences,
                )
                for variation in result
            ]

            if len(variations) != self.n_variations:
                raise ValueError(
                    f"Expected {self.n_variations} variations, but got {len(variations)}"
                )

            if not all(len(variation) == self.n_sentences for variation in variations):
                raise ValueError(
                    f"Expected {self.n_sentences} sentences in each variation, but got {variations}"
                )

            self.data_buffer = {"variations": variations}

            next(retrieve_and_save)

        return await super().make_unit(prompt_item_collection, variations)


class AddPersonaGeneratedTextFactory(AddGeneratedTextFactory):
    def __init__(self,
        short_persona: str,
        n_variations: int,
        n_sentences: int,
        persona: str,
        additional_instructions: List[str] = None,
        request: bool = False,
    ):
        additional_instructions = (
            additional_instructions if additional_instructions is not None else []
        )

        additional_instructions.append(
            "The example here is automatically generated, you should not copy it, it's just to give you an idea. It's probably not very relevant, use your own judgment. And adapt it to the specific persona."
        )

        description = f'Create a paragraph that explains that the behavior must be simulated with the following persona: "{persona}"'

        example = f"You have been attributed the persona of {short_persona} who __|behaves in a certain way|__. This {short_persona} is __|details about the persona and their behavior|__."

        super().__init__(
            name=f"{'_'.join(short_persona.split())}_persona_af",
            n_variations=n_variations,
            n_sentences=n_sentences,
            description=description,
            example=example,
            additional_instructions=additional_instructions,
            request=request,
        )
