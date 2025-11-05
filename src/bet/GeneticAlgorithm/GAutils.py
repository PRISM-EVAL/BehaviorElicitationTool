from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

# import time
import numpy as np
from numpy import random

from bet.Primitives import Primitive, PrimitiveLib

from .Individual import Individual


@dataclass
class EffectivenessEstimationParameters:
    good_job_threshold: float = 2
    base_success_score: float = 10
    exploration_factor: float = 50
    min_temperature: float = 5
    exploration_decay_rate: float = 2 # How fast exploration decreases

    def to_json(self):
        return {
            "good_job_threshold": self.good_job_threshold,
            "base_success_score": self.base_success_score,
            "exploration_factor": self.exploration_factor,
            "min_temperature": self.min_temperature,
            "exploration_decay_rate": self.exploration_decay_rate,
        }


def floats_to_probabilities(
    float_list: List[float],
    temperature: float = 1.0,
    reverse: bool = False,
) -> List[float]:
    float_array = np.array(float_list) if not reverse else -np.array(float_list)
    return np.exp(float_array / temperature) / np.sum(np.exp(float_array / temperature))

def estimate_effectiveness(
    scores: List[float], 
    parameters: EffectivenessEstimationParameters
) -> float:
    if not scores:
        return 0.0
    
    # Count good_job (scores >= good_job_threshold)
    good_job = [s for s in scores if s >= parameters.good_job_threshold]
    failures = [s for s in scores if s < parameters.good_job_threshold]

    if not good_job:
        return sum(scores) / len(scores)  # If no good_job, just return mean
    
    # Success quality
    success_rate = len(good_job) / len(scores)
    success_quality = sum(good_job) / len(good_job)
    success_score = success_quality * success_rate * parameters.base_success_score
    
    # Failure penalty
    failure_score = (sum(failures) / len(failures)) if len(failures) > 0 else 0
    failure_penalty = failure_score * (1 - success_rate)
    
    return parameters.base_success_score + success_score + failure_penalty

def compute_probabilities(
    available_primitives: List[Primitive],
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters(),
    primitive_effectiveness: Dict[str, List[float]] | None = None,
) -> List[float]:
    p = None
    if primitive_effectiveness is not None:

        selected_effectiveness = []
        min_len = float('inf')
        for primitive in available_primitives:
            p_name = primitive.simple_name()
            if p_name in primitive_effectiveness:
                selected_effectiveness.append(primitive_effectiveness[p_name])
                min_len = min(min_len, len(primitive_effectiveness[p_name]))
            else:
                selected_effectiveness.append([])
                min_len = 0

        estimated_effectiveness = [
            estimate_effectiveness(
                scores=ef,
                parameters=effectiveness_estimation_parameters,
            )
            for ef in selected_effectiveness
        ]

        # NOTE: Here we are computing the probabilities of each primitive to be selected based on their estimated effectiveness, and the temperature is computed based on the amount of exploration that has been done (the more exploration, the less temperature). Temperature is basically noise added on top of the probability, the higher the temperature, the higher the noise, the more chances of selecting an underperforming primitive
        p = floats_to_probabilities(
            estimated_effectiveness,
            temperature=max(
                effectiveness_estimation_parameters.min_temperature, 
                effectiveness_estimation_parameters.exploration_factor - effectiveness_estimation_parameters.exploration_decay_rate * min_len
            )
        )
    return p

def parallel_select_random_primitive(
    primitive_list: List[Primitive],
    primitive_lib: PrimitiveLib,
    already_selected: List[List[Primitive]] | None = None, # List of already selected primitives for each individual
    n_individuals: int = 1,
    primitive_effectiveness: List[Dict[str, List[float]]] | None = None,
    max_complexity: int = -1,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters(),
    _timing_obj: Optional[Dict[str, float]] = None,
    deepcopy_primitives: bool = True,
) -> List[Primitive] | None:

    if already_selected is None:
        already_selected = [[] for _ in range(n_individuals)]
    elif len(already_selected) == 1:
        already_selected = already_selected * n_individuals
    else:
        n_individuals = len(already_selected)
    

    # First filter the compatibility matrix to get only those in primitive_list, and in order
    available_primitives_idx = np.zeros((n_individuals, len(primitive_lib._compatibility_matrix)), dtype=bool)

    test_primitive: Primitive
    for test_primitive in primitive_list:
        pname = test_primitive.simple_name()
        idx = primitive_lib._compatibility_matrix[pname]["idx"]

        if test_primitive.can_be_selected and (
            test_primitive.complexity <= max_complexity or 
            max_complexity < 0
        ):
            available_primitives_idx[:, idx] = True

    for idx, _selected in enumerate(already_selected):
        for selected in _selected:
            selected_name = selected.simple_name()
            if selected_name not in primitive_lib._compatibility_matrix:
                continue

            available_primitives_idx[idx] &= primitive_lib._compatibility_matrix[selected_name]["array"]

    available_primitives = [
        np.array(primitive_lib.to_list())[_available_primitives_idx] 
        for _available_primitives_idx in available_primitives_idx
    ]

    if primitive_effectiveness is not None and len(primitive_effectiveness) == n_individuals and type(primitive_effectiveness) is list:
        p = [
            compute_probabilities(
                available_primitives=_available_primitives,
                primitive_effectiveness=_primitive_effectiveness,
                effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            ) for _available_primitives, _primitive_effectiveness in zip(available_primitives, primitive_effectiveness)
        ]
    elif primitive_effectiveness is not None and (
            (len(primitive_effectiveness) > 1 and len(primitive_effectiveness) != n_individuals) or (type(primitive_effectiveness) is not list and type(primitive_effectiveness) is not np.ndarray)
        ):
        raise ValueError(f"Primitive effectiveness must be a list of length 1 or n_individuals, got {len(primitive_effectiveness)} {type(primitive_effectiveness)} {n_individuals}")
    else:
        used_effectiveness = primitive_effectiveness[0] if primitive_effectiveness is not None else None
        p = [
            compute_probabilities(
                available_primitives=_available_primitives,
                primitive_effectiveness=used_effectiveness,
                effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            ) for _available_primitives in available_primitives
        ]

    result = [
        random.choice(a=_available_primitives, p=_p) if len(_available_primitives) > 0 else None
        for _available_primitives, _p in zip(available_primitives, p)
    ]

    return deepcopy(result) if deepcopy_primitives else result

def select_random_primitive(
    primitive_list: List[Primitive],
    primitive_lib: PrimitiveLib,
    already_selected: List[Primitive] | None = None,
    primitive_effectiveness: Dict[str, List[float]] | None = None,
    max_complexity: int = -1,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters(),
    _timing_obj: Optional[Dict[str, float]] = None,
    deepcopy_primitives: bool = True,
) -> Primitive | None:

    if already_selected is None:
        already_selected = []

    # First filter the compatibility matrix to get only those in primitive_list, and in order
    # start_time = time.time()

    available_primitives_idx = np.zeros(len(primitive_lib._compatibility_matrix), dtype=bool)

    test_primitive: Primitive
    for test_primitive in primitive_list:
        pname = test_primitive.simple_name()
        idx = primitive_lib._compatibility_matrix[pname]["idx"]

        if test_primitive.can_be_selected and (
            test_primitive.complexity <= max_complexity or 
            max_complexity < 0
        ):
            available_primitives_idx[idx] = True

    for selected in already_selected:
        selected_name = selected.simple_name()
        if selected_name not in primitive_lib._compatibility_matrix:
            continue

        available_primitives_idx &= primitive_lib._compatibility_matrix[selected_name]["array"]

    if np.sum(available_primitives_idx) == 0:
        return None

    available_primitives = np.array(primitive_lib.to_list())[available_primitives_idx]
    
    p = compute_probabilities(
        available_primitives=available_primitives,
        primitive_effectiveness=primitive_effectiveness,
        effectiveness_estimation_parameters=effectiveness_estimation_parameters,
    )

    result = random.choice(a=available_primitives, p=p)
    return deepcopy(result) if deepcopy_primitives else result

def _parallel_select_random_primitives(
    primitive_list: List[Primitive],
    primitive_lib: PrimitiveLib,
    n_primitives: List[int],
    already_selected: List[List[Primitive]] | None = None,
    primitive_effectiveness: List[Dict[str, List[float]]] | None = None,
    max_complexity: int = -1,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters(),
    deepcopy_primitives: bool = True,
) -> List[List[Primitive]]:
    if already_selected is None:
        already_selected = [[] for _ in range(len(n_primitives))]

    if len(already_selected) != len(n_primitives):
        raise ValueError("already_selected must have the same length as n_primitives")

    selected_primitives: List[List[Primitive]] = [[] for _ in range(len(n_primitives))]
    track_n_primitives = np.array(n_primitives.copy())

    active_indices = np.arange(len(n_primitives))

    for _ in range(min(max(n_primitives), len(primitive_list))):
        mask = track_n_primitives > 0
        
        if not np.any(mask):
            break

        active = active_indices[mask]

        select_new = parallel_select_random_primitive(
            primitive_list=primitive_list,
            primitive_lib=primitive_lib,
            already_selected=[already_selected[i] + selected_primitives[i] for i in active],
            primitive_effectiveness=[primitive_effectiveness[i] for i in active] if primitive_effectiveness is not None and len(primitive_effectiveness) > 1 else primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives
        )

        # Process results directly into target arrays
        for _, (idx, prim) in enumerate(zip(active, select_new)):
            if prim is None:
                # Mark this individual as done
                track_n_primitives[idx] = 0
            else:
                selected_primitives[idx].append(prim)
                track_n_primitives[idx] -= 1

    return selected_primitives
    

def _select_random_primitives(
    primitive_list: List[Primitive],
    primitive_lib: PrimitiveLib,
    n_primitives: int,
    already_selected: List[Primitive] | None = None,
    primitive_effectiveness: Dict[str, List[float]] | None = None,
    max_complexity: int = -1,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters(),
    deepcopy_primitives: bool = True,
) -> List[Primitive]:

    if already_selected is None:
        already_selected = []

    selected_primitives: List[Primitive] = []

    for _ in range(min(n_primitives, len(primitive_list))):
        select_new = select_random_primitive(
            primitive_list=primitive_list,
            primitive_lib=primitive_lib,
            already_selected=already_selected + selected_primitives,
            primitive_effectiveness=primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives
        )

        if select_new is None:
            break
        selected_primitives.append(select_new)

    return selected_primitives

def parallel_select_random_primitives(
    primitive_lib: PrimitiveLib,
    n_primitives: List[int],
    base: bool = True,
    already_selected: List[List[Primitive]] | None = None,
    primitive_effectiveness: List[Dict[str, List[float]]] | None = None,
    max_complexity: int = -1,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters(),
    deepcopy_primitives: bool = True,
) -> List[List[Primitive]]:


    primitive_range = range(len(n_primitives))
    if already_selected is None:
        already_selected = [[] for _ in primitive_range]

    if len(already_selected) != len(n_primitives):
        raise ValueError("already_selected must have the same length as n_primitives")

    if base:
        add_primitives = primitive_lib.get_base_primitives()

        base_primitives = parallel_select_random_primitive(
            primitive_list=add_primitives,
            primitive_lib=primitive_lib,
            already_selected=already_selected,
            primitive_effectiveness=primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives,
        )

        if any(base_primitive is None for base_primitive in base_primitives):
            raise ValueError(
                "Something went wrong, there was no base to be selected in the whole list of primitive"
            )
        
        other_random_primitives = _parallel_select_random_primitives(
            primitive_list=primitive_lib.to_list(),
            primitive_lib=primitive_lib,
            n_primitives=[n - 1 for n in n_primitives],
            already_selected=[[base_primitives[i]] + already_selected[i] for i in primitive_range],
            primitive_effectiveness=primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives,
        )
        
        return [[base_primitives[i]] + other_random_primitives[i] for i in primitive_range]
    else:
        if any(n_primitive <= 0 for n_primitive in n_primitives):
            raise ValueError("n_primitives must be greater than 0 if no base")
        
        return _parallel_select_random_primitives(
            primitive_list=primitive_lib.to_list(),
            primitive_lib=primitive_lib,
            n_primitives=n_primitives,
            already_selected=already_selected,
            primitive_effectiveness=primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives,
        )

def select_random_primitives(
    primitive_lib: PrimitiveLib,
    n_primitives: int,
    base: bool = True,
    already_selected: List[Primitive] = None,
    primitive_effectiveness: Dict[str, List[float]] | None = None,
    max_complexity: int = -1,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters(),
    deepcopy_primitives: bool = True,
) -> List[Primitive]:
    # Fuck python, lost another 20 minutes of my life
    if already_selected is None:
        already_selected = []

    if base:
        add_primitives = primitive_lib.get_base_primitives()
        base_primitive = select_random_primitive(
            primitive_list=add_primitives,
            primitive_lib=primitive_lib,
            already_selected=already_selected,
            primitive_effectiveness=primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives,
        )

        if base_primitive is None:
            raise ValueError(
                "Something went wrong, there was no base to be selected in the whole list of primitive"
            )
        
        primitives_list = []

        for primitive in primitive_lib.to_list():
            if primitive != base_primitive:
                primitives_list.append(primitive)

        return [base_primitive] + _select_random_primitives(
            primitive_list=primitives_list,
            n_primitives=n_primitives - 1,
            primitive_lib=primitive_lib,
            already_selected=already_selected + [base_primitive],
            primitive_effectiveness=primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives,
        )

    else:
        if n_primitives <= 0:
            raise ValueError("n_primitives must be greater than 0 if no base")

        return _select_random_primitives(
            primitive_list=primitive_lib.to_list(),
            n_primitives=n_primitives,
            primitive_lib=primitive_lib,
            already_selected=already_selected,
            primitive_effectiveness=primitive_effectiveness,
            max_complexity=max_complexity,
            effectiveness_estimation_parameters=effectiveness_estimation_parameters,
            deepcopy_primitives=deepcopy_primitives,
        )

def parallel_make_random_individuals(
    max_instruction_primitives: int,
    max_request_primitives: int,
    instruction_primitive_lib: PrimitiveLib,
    request_primitive_lib: PrimitiveLib,
    n_individuals: int,
    use_system: bool = True,
    primitive_effectiveness: List[Dict[str, List[float]]] | None = None,
    max_complexity: int = -1,
    deepcopy_primitives: bool = True,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters()
) -> List[Individual]:

    range_n_individuals = range(n_individuals)

    n_instruction_primitives = [
        random.randint(0, max_instruction_primitives) if max_instruction_primitives > 0 else 0
        for _ in range_n_individuals
    ]
    n_request_primitives = [
        random.randint(0, max_request_primitives) if max_request_primitives > 0 else 0
        for _ in range_n_individuals
    ]

    if use_system:
        use_system = random.random() < 0.5

    _selected_instruction_primitives = parallel_select_random_primitives(
        primitive_lib=instruction_primitive_lib,
        n_primitives=n_instruction_primitives,
        primitive_effectiveness=primitive_effectiveness,
        max_complexity=max_complexity,
        effectiveness_estimation_parameters=effectiveness_estimation_parameters,
        deepcopy_primitives=deepcopy_primitives,
    )

    _selected_request_primitives = parallel_select_random_primitives(
        primitive_lib=request_primitive_lib,
        n_primitives=n_request_primitives,
        already_selected=_selected_instruction_primitives,
        primitive_effectiveness=primitive_effectiveness,
        max_complexity=max_complexity,
        effectiveness_estimation_parameters=effectiveness_estimation_parameters,
        deepcopy_primitives=deepcopy_primitives,
    )

    for i in range_n_individuals:
        for primitive in _selected_instruction_primitives[i] + _selected_request_primitives[i]:
            if primitive.duplicate_in_other:
                if primitive not in _selected_request_primitives[i]:
                    _selected_request_primitives[i].append(primitive)
                if primitive not in _selected_instruction_primitives[i]:
                    _selected_instruction_primitives[i].append(primitive)

    return [
        Individual(
            instruction_primitives=_selected_instruction_primitives[i],
            request_primitives=_selected_request_primitives[i],
            instr_primitive_lib=instruction_primitive_lib,
            req_primitive_lib=request_primitive_lib,
            use_system=use_system,
            orphan=True,
        )
        for i in range_n_individuals
    ]
    


def make_random_individual(
    max_instruction_primitives: int,
    max_request_primitives: int,
    instruction_primitive_lib: PrimitiveLib,
    request_primitive_lib: PrimitiveLib,
    use_system: bool = True,
    primitive_effectiveness: Dict[str, List[float]] | None = None,
    max_complexity: int = -1,
    deepcopy_primitives: bool = True,
    effectiveness_estimation_parameters: EffectivenessEstimationParameters = EffectivenessEstimationParameters()
) -> Individual:
    n_instruction_primitives = (
        random.randint(0, max_instruction_primitives)
        if max_instruction_primitives > 0
        else 0
    )
    n_request_primitives = (
        random.randint(0, max_request_primitives) if max_request_primitives > 0 else 0
    )
    if use_system:
        use_system = random.random() < 0.5

    old_selected_instruction_primitives = select_random_primitives(
        primitive_lib=instruction_primitive_lib,
        n_primitives=n_instruction_primitives,
        primitive_effectiveness=primitive_effectiveness,
        max_complexity=max_complexity,
        effectiveness_estimation_parameters=effectiveness_estimation_parameters,
        deepcopy_primitives=deepcopy_primitives,
    )

    old_selected_request_primitives = select_random_primitives(
        primitive_lib=request_primitive_lib,
        n_primitives=n_request_primitives,
        already_selected=old_selected_instruction_primitives,
        primitive_effectiveness=primitive_effectiveness,
        max_complexity=max_complexity,
        effectiveness_estimation_parameters=effectiveness_estimation_parameters,
        deepcopy_primitives=deepcopy_primitives,
    )

    selected_instruction_primitives = []
    selected_request_primitives = []
    for primitive in old_selected_instruction_primitives:
        selected_instruction_primitives.append(primitive)
        if primitive.duplicate_in_other:
            selected_request_primitives.append(primitive)

    for primitive in old_selected_request_primitives:
        selected_request_primitives.append(primitive)
        if primitive.duplicate_in_other:
            selected_instruction_primitives.append(primitive)

    new_individual = Individual(
        selected_instruction_primitives,
        selected_request_primitives,
        instr_primitive_lib=instruction_primitive_lib,
        req_primitive_lib=request_primitive_lib,
        use_system=use_system,
        orphan=True,
    )
    return new_individual

# For testing
def generate_indiv_with_part(
    part: str, 
    instruction_primitive_lib: PrimitiveLib,
    request_primitive_lib: PrimitiveLib,
    max_try: int = 10000,
    max_instruction_primitives: int = 8,
    max_request_primitives: int = 8,
    use_system: bool = False,
) -> Individual:
    for _ in range(max_try):
        indiv = make_random_individual(
            max_instruction_primitives=max_instruction_primitives,
            max_request_primitives=max_request_primitives,
            instruction_primitive_lib=instruction_primitive_lib,
            use_system=use_system,
            request_primitive_lib=request_primitive_lib,
        )
        name = "\n-\n".join(["|".join(p) for p in indiv.simple_primitive_names()])
        if part in name:
            print(name)
            break
    return indiv

