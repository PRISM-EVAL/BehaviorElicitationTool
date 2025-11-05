
from dotenv import load_dotenv

load_dotenv()

import asyncio
import os
import time
import uuid
from copy import deepcopy
from typing import List
from uuid import uuid4

import minillmlib as mll
import nest_asyncio

from bet.GeneticAlgorithm import BET_optimisation, GAHyperparam
from bet.Metric import MetricHyperParameters, get_heatmap, proto_metric
from bet.Primitives import PrimitiveLib
from bet.utils import Scenario, database, logger
from bet.utils.parameters import BET_TIMEOUT


def run_evaluation(
    scenarios: List[Scenario],
    instruction_primitive_lib: PrimitiveLib,
    request_primitive_lib: PrimitiveLib,
    hyperparam: GAHyperparam,
    run_metric: bool = True,
    metric_params: MetricHyperParameters | None = None,
    evaluator_prompt_path: str | None = "./prompts/evaluate_behavior.json",
    n_aggregation: int | None = None,
    evaluated_model_name: str | None = None,
    llm_system_id: str | None = None, # By default use the model name, but if you're trying multiple model with the same name, you can override it here to run multiple evals
    evaluated_model: mll.GeneratorInfo | None = None,
    evaluation_id: str | None = None,
    bet_run_ids: List[str] | None = None,
    agent_context: str | None = None,
    timeout: int = BET_TIMEOUT, # in minutes
    parallelize: bool = False
):
    # If you fix bet_run_ids and evaluation_id, you can be sure it won't be run mutliple time. 
    # It will resume from where it left if a run failed or is stopped in the middle

    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

    ## Step 1: Preprocess params
    if bet_run_ids is None:
        if n_aggregation is None:
            n_aggregation = 1
        bet_run_ids = ["bet_run_" + str(uuid4()) for _ in range(n_aggregation)]
    elif n_aggregation is None:
        n_aggregation = len(bet_run_ids)
    elif n_aggregation < len(bet_run_ids):
        raise ValueError("if you set n_aggregation and bet_run_ids, len(bet_run_ids) must be smaller or equal to n_aggregation")
    elif n_aggregation > len(bet_run_ids):
        # Expand bet_run_ids to fit n_aggregation
        bet_run_ids += ["bet_run_" + str(uuid4()) for _ in range(n_aggregation - len(bet_run_ids))]
    
    # Calculate parallelization strategy
    max_parallel_runs = n_aggregation
    if parallelize:
        total_cpus = os.cpu_count() or 1
        
        if total_cpus == 1:
            logger.warning({
                "message": "Running with only 1 CPU - no CPU will be reserved for system",
                "total_cpus": 1,
                "warning": "System may be unresponsive during execution"
            })
        
        # Reserve 1 CPU for user, use remaining for parallel runs
        max_parallel_runs = max(1, total_cpus - 1)
        
        if max_parallel_runs < n_aggregation:
            logger.warning({
                "message": "Not enough CPUs for full parallelization, will run in chunks",
                "total_cpus": total_cpus,
                "requested_parallel_runs": n_aggregation,
                "max_parallel_runs": max_parallel_runs,
                "reserved_cpus": 0 if total_cpus == 1 else 1
            })
    
    if evaluation_id is None:
        evaluation_id = "eval_" + str(uuid4())

    if (n_finished := database.collections["runs"].count_documents({
        "evaluation_id": evaluation_id, 
        "finished": True, 
        "bet_run_id": {"$nin": bet_run_ids}
    })) > 0:
        logger.warning(
            "Careful, you have finished runs that are not in your bet_run_ids. "
            "Maybe you forgot to update your evaluation_id? "
            "Do you want to proceed anyway? "
            "(It means that the previous runs might be used to compute the metric)"
        )
        answer = input(
            "This is expected only if you want to accumulate more results into the same evaluation. "
            f"Type 'y' if you wish to proceed, this will expand n_aggregations to fit the number of finished runs + the current n_aggregations ({n_finished} + {n_aggregation}). "
            "If you instead want n_aggregation in total, please add the existing run_ids to bet_run_ids: "
        )
        if answer.lower() != "y":
            raise ValueError("User declined to proceed with existing runs in evaluation_id")
        else:
            n_aggregation += n_finished
            logger.info(f"Expanded n_aggregation to {n_aggregation} (including {n_finished} existing finished runs)")

    if run_metric and metric_params is None:
        raise ValueError("metric_params must be provided if run_metric is True")

    # Create model
    if evaluated_model_name is None and evaluated_model is None:
        raise ValueError("evaluated_model_name or evaluated_model must be provided")

    if evaluated_model_name is not None and evaluated_model is None:
        evaluated_model = mll.GeneratorInfo(
            model=evaluated_model_name,
            _format="url",
            api_url=f"https://openrouter.ai/api/v1/chat/completions",
            api_key=OPENROUTER_API_KEY,
            completion_parameters=mll.GeneratorCompletionParameters(
                provider={
                    # NOTE: see https://openrouter.ai/docs/features/provider-routing
                    "sort": "throughput",
                    "data_collection": "deny"
                },
                usage={"include": False}
            )
        )

    # Building assistant builder and evaluator
    assistant_builder: mll.GeneratorInfo = mll.GeneratorInfo(
        model="deepseek/deepseek-chat-v3-0324",
        _format="url",
        api_url=f"https://openrouter.ai/api/v1/chat/completions",
        api_key=OPENROUTER_API_KEY,
        completion_parameters=mll.GeneratorCompletionParameters(
            provider={
                # NOTE: might need update later if providers change
                "order": ["BaseTen", "Fireworks", "GMICloud"],
                "data_collection": "deny"
            },
            usage={
                "include": True
            }
        ),
        usage_tracking_type="openrouter",
        usage_id_key="bet_run_id",
        usage_key="cost"
    )

    assistant_evaluator: mll.GeneratorInfo = mll.GeneratorInfo(
        model="deepseek/deepseek-chat-v3-0324",
        _format="url",
        api_url=f"https://openrouter.ai/api/v1/chat/completions",
        api_key=OPENROUTER_API_KEY,
        completion_parameters=mll.GeneratorCompletionParameters(
            provider={
                # NOTE: might need update later if providers change
                "order": ["BaseTen", "Fireworks", "GMICloud"],
                "data_collection": "deny"
            },
            usage={
                "include": True
            }
        ),
        usage_tracking_type="openrouter",
        usage_id_key="bet_run_id",
        usage_key="cost"
    )

    request_primitive_lib.prebuild(agent_context=agent_context)
    instruction_primitive_lib.prebuild(agent_context=agent_context)

    ## Step 2: Run BET

    async def _run_with_timeout(bet_run_id: str, chunk_size: int):
        """Run BET_optimisation in a separate thread to enable true parallelization"""
        def _run_in_thread():
            _assistant_builder = deepcopy(assistant_builder)
            _assistant_builder.usage_id_value = bet_run_id
            _assistant_evaluator = deepcopy(assistant_evaluator)
            _assistant_evaluator.usage_id_value = bet_run_id
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    BET_optimisation(
                        hyperparam=hyperparam,
                        scenarios=scenarios,
                        evaluated_llm=evaluated_model,
                        llm_system_id=evaluated_model if llm_system_id is None else llm_system_id,
                        assistant_builder=_assistant_builder,
                        assistant_evaluator=_assistant_evaluator,
                        instruction_primitives=deepcopy(instruction_primitive_lib),
                        request_primitives=deepcopy(request_primitive_lib),
                        evaluation_id=evaluation_id,
                        bet_run_id=bet_run_id,
                        skip_existing=n_aggregation,
                        evaluator_prompt_path=evaluator_prompt_path,
                        n_parallel_runs=chunk_size
                    )
                )
            finally:
                loop.close()
        
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _run_in_thread),
            timeout=timeout * 60,
        )

    async def _run_all():
        if parallelize:
            # Run in chunks based on available CPUs
            for i in range(0, len(bet_run_ids), max_parallel_runs):
                chunk = bet_run_ids[i:i + max_parallel_runs]
                chunk_size = len(chunk)
                logger.info({
                    "message": "Running parallel chunk",
                    "chunk_index": i // max_parallel_runs + 1,
                    "total_chunks": (len(bet_run_ids) + max_parallel_runs - 1) // max_parallel_runs,
                    "runs_in_chunk": chunk_size
                })
                await asyncio.gather(*[_run_with_timeout(bet_run_id, chunk_size) for bet_run_id in chunk])
        else:
            for bet_run_id in bet_run_ids:
                await _run_with_timeout(bet_run_id, 1)

    try:
        logger.info({"message": "Running BET mapping"})
        try:
            # Enable nested event loops for Jupyter notebooks
            nest_asyncio.apply()
            asyncio.run(_run_all())

            logger.info(
                {"message": "BET mapping completed"}
            )
        except TimeoutError:
            logger.error({
                "message": "BET mapping timed out. You can retry with those parameters to get back where it stopped",
                "bet_run_ids": bet_run_ids,
                "evaluation_id": evaluation_id,
                "scenarios": scenarios
            })
            return
        except Exception as e:
            logger.exception({
                "message": "BET mapping failed. You can re-run with the same parameters to take up where it failed. Additionally, if you can't find how to fix the error on some of the runs, you can just change the bet_run_id that are failing to random ones to restart just those ones.",
                "bet_run_ids": bet_run_ids,
                "evaluation_id": evaluation_id,
                "scenarios": scenarios,
                "error": str(e)
            })
            return

        if not run_metric:
            logger.info({
                "message": "BET has finished running, you can check your database for the results. If you want to run the metric, don't forget to set run_metric=True and metric_params. If you reuse the same parameters it will skip directly to the metric.",
                "evaluation_id": evaluation_id,
                "bet_run_ids": bet_run_ids
            })
            return

        logger.info({"message": "Computing metrics"})
        
        metric_params.instr_primitive_lib = instruction_primitive_lib
        metric_params.req_primitive_lib = request_primitive_lib
        metric_params.use_n_runs = n_aggregation

        metric_results = proto_metric(
            evaluation_id=evaluation_id,
            runs_db=database.collections["runs"],
            bet_generations_db=database.collections["bet_generations"],
            metric_params=metric_params,
        )

        logger.info({
            "message": "Computing the heatmap"
        })
        heatmap, all_success, description_dict = get_heatmap(
            run_id=evaluation_id,
            runs_db=database.collections["runs"],
            bet_generations_db=database.collections["bet_generations"],
            metric_params=metric_params,
            instruction_primitive_lib=instruction_primitive_lib,
            request_primitive_lib=request_primitive_lib
        )

        all_prompts = []

        for run in database.collections["runs"].find({
            "evaluation_id": evaluation_id, "finished": True
        }):
            for _, generation_id in run["generations"].items():
                generation_data = database.collections["bet_generations"].find_one({
                    "generation_id": generation_id
                })
                for individual_data in generation_data["evals"]:
                    for prompt in individual_data["details"]:
                        if prompt["score"] >= 3:
                            all_prompts.append({
                                "answer": prompt["llm_answer"],
                                "score": prompt["score"],
                                "prompt": prompt["prompt"]
                            })

        all_prompts = sorted(all_prompts, key=lambda x: x["score"], reverse=True)
        all_outputs = [prompt["answer"] for prompt in all_prompts]
        all_prompts = [prompt["prompt"] for prompt in all_prompts]

        # Generate a uuid for the result
        result_id = str(uuid.uuid4())
        database.collections["results"].insert_one(
            {
                "scenarios": [{
                    "behavior": scenario.behavior,
                    "action": scenario.action
                } for scenario in scenarios],
                "bet_params": hyperparam.to_json(),
                "metric_params": metric_params.to_json(),
                "model": evaluated_model.model,
                "use_system": hyperparam.use_system,
                "result_id": result_id,
                "steps_to_elicit": metric_results["median"] if metric_results["status"] == "success" else None,
                "min": metric_results["min"] if metric_results["status"] == "success" else None,
                "p_10": metric_results["p_10"] if metric_results["status"] == "success" else None,
                "p_25": metric_results["p_25"] if metric_results["status"] == "success" else None,
                "p_75": metric_results["p_75"] if metric_results["status"] == "success" else None,
                "p_90": metric_results["p_90"] if metric_results["status"] == "success" else None,
                "max": metric_results["max"] if metric_results["status"] == "success" else None,
                "output_sample": all_outputs,
                "all_prompts": all_prompts,
                "timestamp": time.time(),
                "evaluation_id": evaluation_id,
                "heatmap": heatmap,
                "all_success": all_success,
                "description_dict": description_dict,
                "agent_context": agent_context,
            }
        )
    except Exception as e:
        logger.exception(
            {
                "message": "Error running BET",
                "error": str(e),
            }
        )
        return

    logger.info({"message": "Evaluation completed"})
