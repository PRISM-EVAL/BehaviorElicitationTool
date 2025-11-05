from typing import Any, TypeVar

from .GAutils import EffectivenessEstimationParameters

T = TypeVar("T")


class InstrAndReq:
    def __init__(self, 
        instruction: T, 
        request: T
    ) -> None:
        self.instr = instruction
        self.req = request

    def to_json(self) -> dict[str, Any]:
        return {"instruction": self.instr, "request": self.req}


class GAProbabilities:
    def __init__(self,
        crossover: InstrAndReq | None = None,
        cross_access: float | None = None,
        cross_param: InstrAndReq | None = None,
        destroy: InstrAndReq | None = None,
        create: InstrAndReq | None = None,
        mutate_param: InstrAndReq | None = None,
        increase_level: InstrAndReq | None = None,
        decrease_level: InstrAndReq | None = None,
        mutate_use_system: float | None = None,
    ):
        # Crossover
        self.crossover = crossover
        self.cross_use_system = cross_access
        self.cross_param = cross_param

        # Mutation
        self.destroy = destroy
        self.create = create
        self.mutate_param = mutate_param
        self.increase_level = increase_level
        self.decrease_level = decrease_level
        self.mutate_use_system = mutate_use_system

    def to_json(self) -> dict[str, Any]:
        return {
            "crossover": self.crossover.to_json() if self.crossover else None,
            "cross_access": self.cross_use_system,
            "cross_param": self.cross_param.to_json() if self.cross_param else None,
            "destroy": self.destroy.to_json() if self.destroy else None,
            "create": self.create.to_json() if self.create else None,
            "mutate_param": self.mutate_param.to_json() if self.mutate_param else None,
            "increase_level": self.increase_level.to_json() if self.increase_level else None,
            "decrease_level": self.decrease_level.to_json() if self.decrease_level else None,
            "mutate_use_system": self.mutate_use_system,
        }


class GAHyperparam:
    def __init__(self,
        n_individuals: int,
        max_generations: int,
        probs: GAProbabilities,
        prompt_per_indiv: int,
        early_stop_score: float,
        stop_n_successes: int = -1,  # If >= 1, stop the GA after `stop_n_successes` successes
        use_system: bool = True,
        use_seed: bool = False,
        initial_population: int = -1,
        seed: int = 42,
        max_instruction_primitives: int = 8,
        max_request_primitives: int = 8,
        k_tournament: int = 5,
        first_k_tournament: int = -1,
        timeout_individual: int = 420,
        inverse_score: bool = False,
        remove_nefarious: bool = False,
        effectiveness_estimation_parameters: EffectivenessEstimationParameters | None = EffectivenessEstimationParameters(),
    ):
        self.n_individuals = n_individuals
        self.max_generations = max_generations
        self.early_stop_score = early_stop_score
        self.probs = probs
        self.prompt_per_indiv = prompt_per_indiv
        self.use_system = use_system
        self.use_seed = use_seed
        self.seed = seed
        self.max_instruction_primitives = max_instruction_primitives
        self.max_request_primitives = max_request_primitives
        self.k_tournament = k_tournament
        self.initial_population = initial_population
        self.stop_n_successes = stop_n_successes
        self.first_k_tournament = first_k_tournament
        self.timeout_individual = timeout_individual
        self.inverse_score = inverse_score
        self.remove_nefarious = remove_nefarious
        self.effectiveness_estimation_parameters = effectiveness_estimation_parameters

    def to_json(self) -> dict[str, Any]:
        return {
            "n_individuals": self.n_individuals,
            "max_generations": self.max_generations,
            "early_stop_score": self.early_stop_score,
            "probs": self.probs.to_json(),
            "prompt_per_indiv": self.prompt_per_indiv,
            "use_system": self.use_system,
            "use_seed": self.use_seed,
            "seed": self.seed,
            "max_instruction_primitives": self.max_instruction_primitives,
            "max_request_primitives": self.max_request_primitives,
            "k_tournament": self.k_tournament,
            "initial_population": self.initial_population,
            "stop_n_successes": self.stop_n_successes,
            "first_k_tournament": self.first_k_tournament,
            "timeout_individual": self.timeout_individual,
            "inverse_score": self.inverse_score,
            "remove_nefarious": self.remove_nefarious,
            "effectiveness_estimation_parameters": self.effectiveness_estimation_parameters.to_json(),
        }
