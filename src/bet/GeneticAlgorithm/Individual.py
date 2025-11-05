from __future__ import annotations

import asyncio
import json
import os
import random
import re
from copy import deepcopy
from typing import List, Optional, Tuple

import minillmlib as mll
import numpy as np
from pymongo.collection import Collection
from scipy.spatial.distance import cdist

from bet.Factories import (EXECUTION_ORDER, ApplyScenarioFactory, WrapFactory,
                           WrapType)
from bet.Primitives import (Primitive, PrimitiveLib, primitive_name_pattern,
                            primitive_name_to_obj)
from bet.PromptBlock import Ensemble
from bet.utils import Scenario, evaluate_prompt_async, logger


def distance_matrix(
    vector_instruction: np.ndarray, # (n, d)
    vector_request: np.ndarray, # (n, d)
    other_vector_instruction: np.ndarray, # (m, d)
    other_vector_request: np.ndarray, # (m, d)
) -> np.ndarray[np.ndarray[float]]: # (n, m)
    return cdist(
        vector_instruction, other_vector_instruction, 'cityblock'
    ) + cdist(
        vector_request, other_vector_request, 'cityblock'
    )

def distance_one_to_one(
    vector_instruction: np.ndarray,  # (n, d)
    vector_request: np.ndarray,      # (n, d)
    other_vector_instruction: np.ndarray,  # (n, d)
    other_vector_request: np.ndarray,      # (n, d)
) -> np.ndarray:  # (n,)
    return np.abs(vector_instruction - other_vector_instruction).sum(axis=1) + np.abs(vector_request - other_vector_request).sum(axis=1)


def vectorise_individuals(population: List[Individual]) -> Tuple[np.ndarray, np.ndarray]:
    n_pop = len(population)
    
    # Create instruction and request matrices in one go
    vectors_instruction = np.empty((n_pop, len(population[0].family_index_instruction)))
    vectors_request = np.empty((n_pop, len(population[0].family_index_request)))

    # Fill matrices
    for i, individual in enumerate(population):
        vectors_instruction[i] = individual.family_vector_instruction
        vectors_request[i] = individual.family_vector_request

    return vectors_instruction, vectors_request

def distance_matrix_individuals(
    population: List[Individual],
    other_population: List[Individual],
) -> np.ndarray:
    vectors_instruction, vectors_request = vectorise_individuals(population)
    other_vectors_instruction, other_vectors_request = vectorise_individuals(other_population)

    return distance_matrix(
        vector_instruction=vectors_instruction, 
        vector_request=vectors_request, 
        other_vector_instruction=other_vectors_instruction, 
        other_vector_request=other_vectors_request
    )

def distance_one_to_one_individuals(
    population: List[Individual],
    other_population: List[Individual],
    retry_penalty: int=0
) -> np.ndarray:
    return np.array([
        indiv.distance(other, retry_penalty=retry_penalty)
        for indiv, other in zip(population, other_population)
    ])

class Individual:
    def __init__(
        self,
        instruction_primitives: List[Primitive],
        request_primitives: List[Primitive],
        instr_primitive_lib: PrimitiveLib,
        req_primitive_lib: PrimitiveLib,
        use_system: bool,
        orphan: bool,
    ):
        self.instruction_primitives = instruction_primitives
        self.request_primitives = request_primitives
        self.instr_set = set()
        self.req_set = set()
        self.use_system = use_system

        self.family_index_instruction = instr_primitive_lib._family_index_readable_names
        self.family_index_request = req_primitive_lib._family_index_readable_names

        self.family_computed_instruction = instr_primitive_lib._family_computed
        self.family_computed_request = req_primitive_lib._family_computed

        self.family_vector_instruction = np.zeros(len(self.family_index_instruction))
        self.family_vector_request = np.zeros(len(self.family_index_request))

        # Handle linked primitives
        self._instantiate_linked_primitives(instr_primitive_lib, req_primitive_lib)

        self.post_init_factories(select_new_parameters=orphan)

        self.post_modification()

    def get_roles(self) -> Tuple[str, str]:
        if self.use_system:
            return ("system", "user")
        return ("user", "user")

    def post_init_factories(self, select_new_parameters: bool = False):
        for primitive in self.instruction_primitives + self.request_primitives:
            primitive.post_init_factories(overwrite=select_new_parameters)

    def primitive_names(self, full: bool = False) -> Tuple[List[str], List[str]]:
        return (
            [
                str(primitive) if not full else primitive.full_name()
                for primitive in self.instruction_primitives
            ],
            [
                str(primitive) if not full else primitive.full_name()
                for primitive in self.request_primitives
            ],
        )

    def simple_primitive_names(self) -> Tuple[List[str], List[str]]:
        return (
            [primitive.simple_name() for primitive in self.instruction_primitives],
            [primitive.simple_name() for primitive in self.request_primitives],
        )

    def get_instruction_primitive(self, name: str) -> None | Primitive:
        filter_name = lambda p: str(p) == name
        return next(filter(filter_name, self.instruction_primitives), None)

    def get_request_primitive(self, name: str) -> None | Primitive:
        filter_name = lambda p: str(p) == name
        return next(filter(filter_name, self.request_primitives), None)
    def compute_family_stats(self):
        # Reset family vector
        self.family_vector_instruction = np.zeros(len(self.family_index_instruction))
        self.family_vector_request = np.zeros(len(self.family_index_request))

        try:
            # Compute family vector
            for primitive in self.instruction_primitives:
                if primitive.get_readable_name() not in self.family_index_instruction:
                    continue
                self.family_vector_instruction[self.family_index_instruction[primitive.get_readable_name()]] += 1

                if self.family_computed_instruction:
                    for family in primitive.families:
                        if str(family) not in self.family_index_instruction:
                            continue
                        self.family_vector_instruction[self.family_index_instruction[str(family)]] += 1

            for primitive in self.request_primitives:
                if primitive.get_readable_name() not in self.family_index_request:
                    continue
                self.family_vector_request[self.family_index_request[primitive.get_readable_name()]] += 1

                if self.family_computed_request:
                    for family in primitive.families:
                        if str(family) not in self.family_index_request:
                            continue
                        self.family_vector_request[self.family_index_request[str(family)]] += 1
        except Exception as e:
            #print(self.family_index_instruction)
            #print(self.family_vector_instruction)
            raise e

    def distance(self, other: Individual, retry_penalty: int = 0) -> float:
        if retry_penalty > 0:
            prim_distance = (
                len(self.instr_set ^ other.instr_set) +
                len(self.req_set ^ other.req_set)
            ) 
            return prim_distance + (
                retry_penalty if prim_distance == 0 else 0
            )
        
        # Compute family distance using family vector
        return (
            np.abs(
                self.family_vector_instruction - other.family_vector_instruction
            ).sum() + np.abs(
                self.family_vector_request - other.family_vector_request
            ).sum()
        )
    
    def distance_multiple(self, 
        others: List[Individual] | None = None,
        other_vectors_instruction: np.ndarray | None = None,
        other_vectors_request: np.ndarray | None = None,
        retry_penalty: int = 0
    ) -> np.ndarray:
        if others is None and (other_vectors_instruction is None or other_vectors_request is None or retry_penalty > 0):
            raise ValueError("Either 'others' or 'other_vectors_instruction' and 'other_vectors_request' must be provided. If 'retry_penalty' is greater than 0, 'others' must be provided.")
        
        if retry_penalty > 0:
            return [self.distance(other, retry_penalty=retry_penalty) for other in others]
        
        if other_vectors_instruction is None or other_vectors_request is None:
            n_others = len(others)

            # Create instruction and request matrices in one go
            other_vectors_instruction = np.empty((n_others, len(self.family_index_instruction)))
            other_vectors_request = np.empty((n_others, len(self.family_index_request)))
        
            # Fill matrices
            for i, other in enumerate(others):
                other_vectors_instruction[i] = other.family_vector_instruction
                other_vectors_request[i] = other.family_vector_request

        # Compute distances in one operation
        return (
            np.abs(
                self.family_vector_instruction[None, :] - other_vectors_instruction
            ).sum(axis=1) +
            np.abs(
                self.family_vector_request[None, :] - other_vectors_request
            ).sum(axis=1)
        )

    def instruction_base_primitive_len(self) -> int:
        return len([p for p in self.instruction_primitives if p.base])

    def request_base_primitive_len(self) -> int:
        return len([p for p in self.request_primitives if p.base])

    def post_modification(self) -> None:
        # First remove duplicates keeping ordering
        self.instruction_primitives = list(dict.fromkeys(self.instruction_primitives))
        self.request_primitives = list(dict.fromkeys(self.request_primitives))

        # Update sets
        self.instr_set = set(self.instruction_primitives)
        self.req_set = set(self.request_primitives)

        # Then remove primitives that are not allowed in the other list
        same_prim = self.instr_set & self.req_set
        for prim in same_prim:
            if prim.prevent_in_other:
                self.request_primitives.remove(prim)
        
        # Finally, re-compute family stats
        self.compute_family_stats()

    def distance_from_population(self, population: List[Individual]) -> float:
        return self.distance_multiple(others=population).mean()

    async def build_ensemble_by_type(
        self,
        prompt_item_collection: Collection,
        role: str,
        prompt_type: str,
        scenario: str,
        assistant: mll.GeneratorInfo,
        request: bool = False,
        skip_rewrite: bool = False,
    ) -> Ensemble:
        prompt_ensemble = Ensemble(prompt_item_collection=prompt_item_collection, role=role)

        # Just in case we forgot to select parameter somewhere else in the code, doing it here (but it shouldn't be necessary)
        self.post_init_factories()

        primitive_instructions = []
        for primitive in self.__dict__[f"{prompt_type}_primitives"]:
            for factory_type in EXECUTION_ORDER:
                await primitive.actions[factory_type](
                    ensemble=prompt_ensemble,
                    _assistant=assistant
                )
            primitive_instructions.extend(primitive.additional_generation_instructions)

        if not skip_rewrite:
            if os.path.exists("prompts/apply_behavior.json"):
                instruction_path = "prompts/apply_behavior.json" if not request else "prompts/apply_action.json"
            else:
                instruction_path = "src/eval_task/prompts/apply_behavior.json" if not request else "src/eval_task/prompts/apply_action.json"
                

            apply_scenario = ApplyScenarioFactory(
                scenario=scenario,
                instructions_path=instruction_path,
                assistant=assistant,
                request=request,
                primitive_instructions=primitive_instructions,
            )

            prompt_ensemble.apply_scenario(transform=apply_scenario.transform)

        return prompt_ensemble

    async def build_prompts(self,
        n_prompts: int,
        prompt_item_collection: Collection,
        behavior: str,
        action: str,
        assistant: mll.GeneratorInfo,
        skip_rewrite: Tuple[bool, bool] | None = None,
        catch_exceptions: bool = False,
    ) -> List[mll.ChatNode]:

        if skip_rewrite is None:
            skip_rewrite = (False, False)

        logger.debug(
            {
                "type": "building_prompts",
                "individual": str(self),
                "behavior": behavior,
                "action": action,
                "n_prompts": n_prompts,
            }
        )

        try:
            instruction_role, request_role = self.get_roles()

            request_ensemble = await self.build_ensemble_by_type(
                prompt_item_collection=prompt_item_collection,
                role=request_role,
                prompt_type="request",
                scenario=action,
                assistant=assistant,
                request=True,
                skip_rewrite=skip_rewrite[1],
            )

            if len(self.instruction_primitives) > 0:
                instruction_ensemble = await self.build_ensemble_by_type(
                    prompt_item_collection=prompt_item_collection,
                    role=instruction_role,
                    prompt_type="instruction",
                    scenario=behavior,
                    assistant=assistant,
                    skip_rewrite=skip_rewrite[0],
                )

                connect_instruction_and_request = WrapFactory(
                    name="connect_instruction_and_request",
                    wrap_inside=False,
                    wrap_type=WrapType.PREFIX,
                    ensemble=instruction_ensemble,
                )
                connect_instruction_and_request.connect(ensemble=request_ensemble)

            return await request_ensemble.generate_prompts(
                n=n_prompts, 
                assistant=assistant
            )
        except Exception as e:
            logger.debug(
                {
                    "type": "prompt_build_error",
                    "individual": str(self),
                    "behavior": behavior,
                    "action": action,
                    "n_prompts": n_prompts,
                    "error": str(e),
                }
            )
            if catch_exceptions:
                return [None] * n_prompts
            raise e

    async def evaluate(self,
        n_prompts: int,
        prompt_item_collection: Collection,
        scenarios: List[Scenario],
        evaluated_model: mll.GeneratorInfo,
        assistant_builder: mll.GeneratorInfo,
        assistant_evaluator: mll.GeneratorInfo,
        evaluator_prompt_path: str,
        timeout: int = 420,
        inverse_score: bool = False
    ) -> Tuple[
        float, List[float], List[mll.ChatNode], List[str], List[str], List[Scenario], List[Tuple[int, str, str, Scenario, mll.ChatNode]]
    ]:
        prompts = None
        cleaned_results = []
        results = []
        try:
            selected_scenarios: List[Scenario] = random.choices(scenarios, k=n_prompts)

            async def build_prompts_async(scenario: Scenario):
                try:
                    result = await asyncio.wait_for(
                        self.build_prompts(
                            n_prompts=1,
                            prompt_item_collection=prompt_item_collection,
                            behavior=scenario.behavior,
                            action=scenario.action,
                            assistant=assistant_builder,
                        ),
                        timeout=timeout,
                    )
                    return (result[0], scenario)
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"build_prompts timed out after {timeout // 60} minutes"
                        + (
                            f" and {timeout % 60} seconds."
                            if (timeout % 60) > 0
                            else ""
                        )
                    )
                except Exception as e:
                    logger.error({
                        "type": "build_prompts_error",
                        "error": str(e),
                        "individual": str(self),
                    })
                    raise e

            prompts: List[Tuple[mll.ChatNode, Scenario]] = await asyncio.gather(
                *[build_prompts_async(scenario) for scenario in selected_scenarios]
            )

            async def process_evaluation(prompt: Tuple[mll.ChatNode, Scenario]):
                return (
                    await evaluate_prompt_async(
                        prompt=prompt[0],
                        behavior=prompt[1].behavior,
                        action=prompt[1].action,
                        evaluated_model=evaluated_model,
                        assistant=assistant_evaluator,
                        evaluator_prompt_path=evaluator_prompt_path,
                        inverse_score=inverse_score,
                    )
                ) + (prompt[1], prompt[0])

            _results = await asyncio.gather(
                *[process_evaluation(prompt) for prompt in prompts]
            )
            for result in _results:
                if "ERROR" in result[2]:
                    cleaned_results.append(result)
                else:
                    results.append(result)
            if len(results) < 1:
                raise ValueError("No valid prompt evaluated")

        except Exception as e:
            logger.error(
                {
                    "type": "prompt_eval_error",
                    "error": str(e),
                    "individual": self.full_name(),
                }
            )
            try:
                return (0, [], [
                    prompt[0] for prompt in prompts
                ] if prompts is not None else [], [], [], [
                    prompt[1] for prompt in prompts
                ] if prompts is not None else [], cleaned_results)
            except Exception as inner_e:
                logger.error(
                    {
                        "type": "prompt_eval_error_return",
                        "error": f"Exception in error return: {inner_e}",
                        "individual": self.full_name(),
                    }
                )
                return (0, [], [], [], [], [], [])

        scores, llm_answers, analyses, scenarios, processed_prompts = zip(*results)

        # TODO: Tweak the score with stuff like the len of the prompt to optimize for other stuff than just the prompts working
        # NOTE: The score of an individual is between -25 and 100, where everything under or equal to 0 means that all of the answers where bad, and 100 means that all of them were perfect
        return (
            ((sum(scores) / len(scores)) / 4) * 100,
            scores,
            processed_prompts,
            llm_answers,
            analyses,
            scenarios,
            cleaned_results
        )

    def estimate_craft_time(self) -> int:
        return sum(
            [
                p.craft_time
                for p in self.instruction_primitives + self.request_primitives
            ]
        )

    def max_complexity(self) -> int:
        return max(
            [
                p.complexity
                for p in self.instruction_primitives + self.request_primitives
            ]
        )

    def full_name(self):
        return (
            "Idv:"
            + "_"
            + str(self.primitive_names(full=True))
            + "_"
            + str(self.get_roles())
        )

    def __str__(self) -> str:
        return "Idv:" + "_" + str(self.primitive_names()) + "_" + str(self.get_roles())

    def _instantiate_linked_primitives(
        self,
        instr_primitive_lib: PrimitiveLib,
        req_primitive_lib: PrimitiveLib,
    ):
        """Create linked primitives in the request list for any instruction primitives that need them"""

        # For each instruction primitive, setup its linked primitives
        attempt_linking_primitives(
            self.instruction_primitives, req_primitive_lib, self.request_primitives
        )

        # For each request primitive, setup its linked primitives
        attempt_linking_primitives(
            self.request_primitives, instr_primitive_lib, self.instruction_primitives
        )

def individual_name_to_obj(
    individual_name: str,
    instr_primitive_lib: PrimitiveLib,
    req_primitive_lib: PrimitiveLib,
) -> Individual:
    primitive_list_pattern = rf"\[(?:'{primitive_name_pattern.replace('(', '(?:').replace('(?:?:', '(?:')}'(?:, )?)*\]"

    individual_name_pattern = rf"Idv:_\(({primitive_list_pattern}), ({primitive_list_pattern})\)_\((.*), (.*)\)"  # Making the primitive groups non-capturing

    groups = re.search(individual_name_pattern, individual_name)

    instruction_primitives_lst = groups.group(1).replace("'", '"')
    request_primitives_lst = groups.group(2).replace("'", '"')
    role_instr = groups.group(3)

    if (
        instruction_primitives_lst is None
        or request_primitives_lst is None
        or role_instr is None
    ):
        raise Exception(f"Invalid individual name: {individual_name}")

    instruction_primitives = [
        primitive_name_to_obj(primitive_name, instr_primitive_lib)
        for primitive_name in json.loads(instruction_primitives_lst)
    ]
    request_primitives = [
        primitive_name_to_obj(primitive_name, req_primitive_lib)
        for primitive_name in json.loads(request_primitives_lst)
    ]

    return Individual(
        instruction_primitives=instruction_primitives,
        request_primitives=request_primitives,
        instr_primitive_lib=instr_primitive_lib,
        req_primitive_lib=req_primitive_lib,
        use_system="system" in role_instr,
        orphan=False,
    )


def attempt_linking_primitives(
    primitive_list: List[Primitive],
    other_primitive_lib: PrimitiveLib,
    other_primitive_list: List[Primitive],
) -> None:
    to_remove = []
    for primitive in primitive_list:
        if len(primitive.linked_primitives_names) > 0:
            to_link: List[Primitive] = []
            removed = False

            # Get each linked primitive from other_primitive_lib
            for linked_primitive_name in primitive.linked_primitives_names:
                try:
                    linked_primitive = primitive_name_to_obj(
                        linked_primitive_name, other_primitive_lib
                    )
                except:
                    linked_primitive = None

                if isinstance(linked_primitive, Primitive) and any(
                    [
                        prim.simple_name() == linked_primitive.simple_name()
                        for prim in other_primitive_list
                    ]
                ):
                    continue  # It already exists in the other primitive list so skipping this one

                if (
                    linked_primitive is not None
                    and isinstance(linked_primitive, Primitive)
                    and linked_primitive.compatible_with_primitives(
                        other_primitive_list
                    )
                ):
                    to_link.append(deepcopy(linked_primitive))
                else:  # This means that for some reason we can't link the primitive, so we should remove the primitive from the instruction list
                    logger.debug(
                        {
                            "type": "primitive_removal",
                            "removed": primitive.simple_name(),
                            "reason": f"{linked_primitive_name} couldn't be linked to {[prim.simple_name() for prim in other_primitive_list]}",
                        }
                    )
                    to_remove.append(primitive)
                    removed = True
                    break

            if not removed:
                other_primitive_list.extend(to_link)
                primitive.linked_primitives_names = [prim.simple_name() for prim in to_link]

    for prim in to_remove:
        primitive_list.remove(prim)
