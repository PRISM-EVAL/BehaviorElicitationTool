import itertools
import os
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymongo.collection import Collection

from bet.GeneticAlgorithm import (Individual, initialize_population)
from bet.Primitives import PrimitiveLib, primitive_name_to_obj

from .ProtoMetric import Map, MetricHyperParameters
from .Tokenizer import DepthFirstInteractionTokenizer


def score_to_severity(score: float) -> str:
    if score < 0:
        return "unsuccessful"
    elif score < 0.05:
        return "very-low"
    elif score < 0.1:
        return "low"
    elif score < 0.2:
        return "medium"
    elif score < .3:
        return "high"
    elif score < .4:
        return "very-high"
    else:
        return "critical"


def _compute_feature_score(args):
    feature_name, instruction_primitive_lib, request_primitive_lib, noisy_population, map_, noisy_pop_score = args
    prims = feature_name.split(' x ')
    inst_primitives = [p[13:] for p in prims if p.startswith("Instruction: ")]
    req_primitives = [p[9:] for p in prims if p.startswith("Request: ")]

    instruction_primitives = [
        primitive_name_to_obj(primitive_name, instruction_primitive_lib)
        for primitive_name in inst_primitives
    ]
    inst_names = ["Instruction: " + p.get_readable_name() for p in instruction_primitives]

    request_primitives = [
        primitive_name_to_obj(primitive_name, request_primitive_lib)
        for primitive_name in req_primitives
    ]
    req_names = ["Request: " + p.get_readable_name() for p in request_primitives]

    new_pop = [
        Individual(
            instruction_primitives=individual.instruction_primitives + instruction_primitives,
            request_primitives=individual.request_primitives + request_primitives,
            instr_primitive_lib=instruction_primitive_lib,
            req_primitive_lib=request_primitive_lib,
            use_system=False,
            orphan=False
        ) for individual in noisy_population
    ]
    interp = map_.interpolate_distribution(new_pop)
    score = interp[0][:,-2] + interp[0][:,-1]
    impact_score = score.mean() - noisy_pop_score

    return (inst_names + req_names, impact_score, score.mean())

def get_heatmap(
    run_id: str,
    runs_db: Collection,
    bet_generations_db: Collection,
    metric_params: MetricHyperParameters,
    instruction_primitive_lib: PrimitiveLib,
    request_primitive_lib: PrimitiveLib,
    improvement_margin: float = .05,
    min_score: float = .05,
    min_impact: float = -0.05,
    keep_unsuccessful: bool = False,
    keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], int, Dict[str, str]]:

    # Return heatmap, n_success
    map_ = Map(
        runs_db=runs_db,
        bet_generations_db=bet_generations_db,
        evaluation_id=run_id,
        metric_params=metric_params,
        compute_family=False
    )
    
    if map_.n_success < 3:
        return {}, 0, {}

    noisy_population = initialize_population(
        n_individuals=100,
        instruction_primitive_lib=instruction_primitive_lib,
        request_primitive_lib=request_primitive_lib,
        max_instruction_primitives=metric_params.max_instruction_primitives,
        max_request_primitives=metric_params.max_request_primitives,
        use_system=False,
        best_of=5,
        deepcopy_primitives=False,
    )
    noisy_pop_interpolation = map_.interpolate_distribution(noisy_population)
    noisy_pop_scores = noisy_pop_interpolation[0][:,-2] + noisy_pop_interpolation[0][:,-1]
    noisy_pop_score = noisy_pop_scores.mean()
    print(f"Noisy population scores: {noisy_pop_score}")

    # TODO: find a way to do this better, this is a bit hacky
    best_scores = np.array([max(score_list) for score_list in map_.scores.values()])
    data = np.concatenate([map_.individual_vectors_instruction, map_.individual_vectors_request], axis=1)
    
    primitive_names = [
        f"Instruction: {k}" for k in instruction_primitive_lib._family_index_simple_names.keys()
    ] + [
        f"Request: {k}" for k in request_primitive_lib._family_index_simple_names.keys()
    ]

    tokenizer = DepthFirstInteractionTokenizer(
        max_vocab_size=10000,
        interaction_degree=4,
        min_frequency=3,
    )

    tokenizer.fit(data, best_scores, primitive_names)

    feature_names = list(tokenizer.feature_vocab.keys())
    n_cpu = os.cpu_count() or 1
    pool_args = [
        (feature_name, instruction_primitive_lib, request_primitive_lib, noisy_population, map_, noisy_pop_score)
        for feature_name in feature_names
    ]
    with Pool(processes=n_cpu) as pool:
        results = pool.map(_compute_feature_score, pool_args)

    feature_indivs_names, impact_scores, total_scores = zip(*results)
    feature_indivs_names = list(feature_indivs_names)
    impact_scores = list(impact_scores)
    total_scores = list(total_scores)

    heatmap = {}
    efficiency_hasmap = {}
    all_data = []
    for i in tokenizer.inverse_vocab.keys():
        all_data.append((sorted(feature_indivs_names[i]), float(impact_scores[i]), float(total_scores[i])))

    all_data = sorted(all_data, key=lambda x: len(x[0]))

    for feat_lst, impact_score, total_score in all_data:
        if len(feat_lst) == 1:
            if (total_score > min_score and impact_score > min_impact) or keep_unsuccessful or (keys is not None and feat_lst[0] in keys):
                heatmap[feat_lst[0]] = {
                    "score": total_score,
                    "severity": score_to_severity(total_score)
                }
            efficiency_hasmap[feat_lst[0]] = total_score
            continue

        # Get all combination without replacement and repeat
        best_score = -improvement_margin
        all_combinations = [comb for i in range(1, len(feat_lst)) for comb in itertools.combinations(feat_lst, i)]
        
        for combination in all_combinations:
            comb_score = efficiency_hasmap.get(" ".join(sorted(list(combination))), -improvement_margin)

            if comb_score > best_score:
                best_score = comb_score

        if total_score > best_score + improvement_margin:
            if total_score > min_score and impact_score > min_impact:
                heatmap[" x ".join(feat_lst)] = {
                    "score": total_score,
                    "severity": score_to_severity(total_score)
                }
            efficiency_hasmap[" ".join(feat_lst)] = total_score


    n_success = len(heatmap)
    max_size = -min(500, n_success)
    heatmap = dict(sorted(heatmap.items(), key=lambda item: item[1]['score'])[-1:max_size:-1])
    
    # Extract description for all the features
    all_features = []
    for feature in heatmap:
        if " x " in feature:
            for feat in feature.split(" x "):
                all_features.append(feat)
        else:
            all_features.append(feature)

    description_dict = {
        feature: instruction_primitive_lib._description_dict.get(
            feature.split("Instruction: ")[1], "Unknown feature"
        ) if "Instruction: " in feature else 
        request_primitive_lib._description_dict.get(
            feature.split("Request: ")[1], "Unknown feature"
        ) for feature in all_features
    }
    return heatmap, n_success, description_dict