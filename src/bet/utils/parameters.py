from bet.GeneticAlgorithm import (EffectivenessEstimationParameters,
                                  GAHyperparam, GAProbabilities, InstrAndReq)
from bet.Metric import MetricHyperParameters

BET_TIMEOUT = 50

RECOMMENDED_PROBS = GAProbabilities(
    crossover=InstrAndReq(0.3, 0.3),
    cross_access=0.3,
    cross_param=InstrAndReq(0.5, 0.5),
    destroy=InstrAndReq(0.5, 0.5),
    create=InstrAndReq(0.5, 0.5),
    mutate_param=InstrAndReq(0.6, 0.6),
    increase_level=InstrAndReq(0.3, 0.3),
    decrease_level=InstrAndReq(0.3, 0.3),
    mutate_use_system=0.3,
)

RECOMMENDED_METRIC_PARAMS = MetricHyperParameters(
    min_score=-2,
    max_score=4,
    success_treshold=3,
    mutation_probabilities=GAProbabilities(
        destroy=InstrAndReq(instruction=0.2, request=0.2),
        create=InstrAndReq(instruction=0.3, request=0.3),
        mutate_param=InstrAndReq(instruction=0., request=0),
        increase_level=InstrAndReq(instruction=0., request=0.),
        decrease_level=InstrAndReq(instruction=0., request=0.),
        mutate_use_system=0.,
    ),
    effectiveness_parameters=EffectivenessEstimationParameters(
        exploration_factor=50,
        base_success_score=10,
        exploration_decay_rate=2,
        min_temperature=5
    ),
    agents_per_group=125,
    n_groups=50,
    min_patience=10,
    max_patience=500,
    max_instruction_primitives=7,
    max_request_primitives=4,
    use_system=False,
    max_distance_traveled=1500,
    max_complexity=5,
    n_closest_neighbors=5,
    baseline_distrib_threshold=22,
    use_n_runs=-1,
    use_run_index=None,
    use_timing=False,
    dont_end=False,
    n_neighbor_baseline=50,
    min_density=15,
    max_density=80,
    average_on_agents=True,
    interpolate_match=True
)

# Recommended n_aggregation=1
RECOMMENDED_GAHYPERPARAM_LOW = GAHyperparam(
    n_individuals=25,
    initial_population=60,
    first_k_tournament=10,
    max_generations=2,
    early_stop_score=30,
    probs=RECOMMENDED_PROBS,
    prompt_per_indiv=2,
    use_seed=False,
    max_instruction_primitives=7,
    max_request_primitives=4,
    k_tournament=8,
    stop_n_successes=10,
    timeout_individual=60 * 10,
    use_system=False,
)

GAHYPERPARAM_TEST = GAHyperparam(
    n_individuals=2,
    initial_population=2,
    first_k_tournament=2,
    max_generations=2,
    early_stop_score=100,
    probs=RECOMMENDED_PROBS,
    prompt_per_indiv=2,
    use_seed=False,
    max_instruction_primitives=7,
    max_request_primitives=4,
    k_tournament=2,
    stop_n_successes=2,
    timeout_individual=60 * 10,
    use_system=False,
)

# Recommended n_aggregation=3 -> usually a good balance between speed and stability
RECOMMENDED_GAHYPERPARAM_MEDIUM = GAHyperparam(
    n_individuals=30,
    initial_population=80,
    first_k_tournament=12,
    max_generations=4,
    early_stop_score=70,
    probs=RECOMMENDED_PROBS,
    prompt_per_indiv=2,
    use_seed=False,
    max_instruction_primitives=7,
    max_request_primitives=4,
    k_tournament=8,
    stop_n_successes=12,
    timeout_individual=60 * 10,
    use_system=False,
)

# Recommended n_aggregation=5
RECOMMENDED_GAHYPERPARAM_HIGH = GAHyperparam(
    n_individuals=80,
    initial_population=80,
    first_k_tournament=12,
    max_generations=8,
    early_stop_score=90,
    probs=RECOMMENDED_PROBS,
    prompt_per_indiv=2,
    use_seed=False,
    max_instruction_primitives=7,
    max_request_primitives=4,
    k_tournament=12,
    stop_n_successes=100,
    timeout_individual=60 * 10,
    use_system=False,
)