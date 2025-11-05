from __future__ import annotations

import functools
import json
from typing import Awaitable, List, Tuple

import minillmlib as mll
from pymongo.collection import Collection

from bet.utils import add_request_instruction_prompt, strip_merge_and_split

from .Factory import Factory, FactoryType
from .Transforms import IndividualTransform, IndividualTransformCallable


# NOTE: When naming a transform, you must be extremely clear as the name will be used to make sure that the transformation hasn't been removed
class IndividualTransformFactory(Factory):
    def __init__(self,
        name: str,
        request: bool = False,
    ) -> None:
        """
        Initializes the IndividualTransformFactory with the provided name and transform.

        Parameters:
            name (str): The name of the factory.
            transform (IndividualTransform): The individual transform associated with the factory.
        """
        self.transform = IndividualTransform(
            parent=self, transform=self.create_transform_lambda()
        )
        super().__init__(
            _type=FactoryType.INDIVIDUAL_TRANSFORM, name=name, request=request
        )

async def rewrite_with_llm_wrapper(
    instance: RewriteWithLLMIndividualFactory, 
    content: str, 
    previous_transform: Tuple[List[str], List[str]], 
    prompt_item_collection: Collection,
    assistant: mll.GeneratorInfo | None = None,
    *_args, **_kwargs
) -> Awaitable[str]:
    return await instance.rewrite_with_llm(
        instructions=instance.instructions,
        content=content,
        previous_transform=previous_transform,
        prompt_item_collection=prompt_item_collection,
        assistant=assistant
    )

class RewriteWithLLMIndividualFactory(IndividualTransformFactory):
    def __init__(self,
        name: str,
        instructions: str,
        assistant: mll.GeneratorInfo | None = None,
        additional_instructions: List[str] = None,
        request: bool = False,
    ) -> None:
        """
        Initializes the LLMIndividualTransformFactory with the provided name and instructions.

        Parameters:
            name (str): The name of the factory.
            instructions (str): The instructions for the transformation.
        """
        self.generation_parameters = mll.GeneratorCompletionParameters(
            max_tokens=2048, temperature=0.6, response_format={"type": "json_object"}
        )

        name = "rewrite_" + name

        self.instructions = instructions

        self.additional_instructions = (
            "".join(["\n- " + instr for instr in additional_instructions])
            if additional_instructions is not None
            else ""
        )

        super().__init__(name=name, request=request)

    def create_transform_lambda(self) -> IndividualTransformCallable:
        return functools.partial(rewrite_with_llm_wrapper, self)

    async def rewrite_with_llm(self,
        instructions: str,
        content: List[str],
        previous_transform: List[str],
        prompt_item_collection: Collection,
        assistant: mll.GeneratorInfo | None = None
    ) -> Awaitable[List[str]]:
        """
        Apply the instructions on the content.

        Args:
            instructions (str): The instructions for the transformation.
            content (List[str]): The content to be transformed.
            previous_transform (List[str]): The name of the previous individual transforms that have been applied.

        Returns:
            List[str]: The transformed content.
        """

        if assistant is None:
            raise ValueError("Assistant is not set rewrite_with_llm individual")

        retrieve_and_save = super().retrieve_and_save(prompt_item_collection=prompt_item_collection, content=content)
        next(retrieve_and_save)

        if self.data_buffer is not None:
            return self.data_buffer["transformed_content"]
        else:
            try:
                executant = mll.ChatNode.from_thread(
                    ["prompts/pre_prompt.json", "prompts/rewrite_with_llm.json"]
                )
            except FileNotFoundError:
                executant = mll.ChatNode.from_thread(
                    ["src/eval_task/prompts/pre_prompt.json", "src/eval_task/prompts/rewrite_with_llm.json"]
                )

            self.additional_instructions = add_request_instruction_prompt(
                self.additional_instructions, self.request
            )

            executant.update_format_kwargs(
                propagate=True,
                instructions=instructions,
                previous_transform="[" + ", ".join(previous_transform) + "]",
                to_transform=json.dumps({i: v for i, v in enumerate(content)}),
                n_sentences=len(content),
                additional_instructions=self.additional_instructions,
            )

            completion = (
                await executant.complete_async(
                    mll.NodeCompletionParameters(
                        gi=assistant,
                        generation_parameters=self.generation_parameters,
                        parse_json=True,
                        crash_on_refusal=True,
                        retry=3,
                    )
                )
            ).content
            result = json.loads(completion)

            parsed = strip_merge_and_split([v for _, v in result.items()], len(content))

            if not len(parsed) == len(content):
                raise ValueError(
                    f'Failed to rewrite with LLM: "{parsed}"\n\n "{content}"\n\n "{result}"'
                )

            self.data_buffer = {"transformed_content": parsed}
            next(retrieve_and_save)

            return parsed