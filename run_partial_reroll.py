# Copyright Sierra

import argparse
from tau_bench.types import RunConfig
from tau_bench.partial_reroll import run_partial_reroll
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Partial-rerolling control (aka Bo-N from the failure point): reuses "
            "the real intervention pipeline's failure-point identification "
            "(agent.run_intervention) and truncates the trajectory at the exact "
            "same point intervention/placeholder would insert their message at, "
            "but inserts NO text at all - the agent just continues generating from "
            "there, relying on sampling stochasticity alone. Use a non-zero "
            "--temperature or every reroll attempt will be identical. Compare its "
            "Bo-N number against both the real intervention Bo-N and the "
            "placeholder Bo-N to see whether the intervention TEXT is doing "
            "anything beyond what re-rolling the dice from the failure point does."
        )
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to the completed baseline run folder (must contain transcript.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--intervention_model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="The model used for failure-point identification (its intervention text is discarded)",
    )
    parser.add_argument(
        "--intervention-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the failure-point identification model (defaults to openai)",
    )
    parser.add_argument(
        "--search_model",
        type=str,
        help="Model backing the intervenor's transcript-search tool (defaults to the intervenor model)",
    )
    parser.add_argument(
        "--search-model-provider",
        type=str,
        choices=provider_list,
        help="Provider for the search-tool model (defaults to the intervenor's provider)",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="react-intervened",
        choices=["react-intervened"],
        help="Only react-intervened supports run_intervention/solve_with_intervention",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The sampling temperature for the action model (must be > 0 for Bo-N reroll attempts to differ at all)",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
    )
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) only run partial reroll on these task IDs")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument(
        "--best_of_N",
        type=int,
        default=3,
        help="Number of reroll attempts per failing task (should match the real intervention run's N)",
    )

    args = parser.parse_args()
    print(args)

    config = RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        intervention_model=args.intervention_model,
        intervention_model_provider=args.intervention_model_provider,
        search_model=args.search_model,
        search_model_provider=args.search_model_provider,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        user_strategy=args.user_strategy,
        best_of_N=args.best_of_N,
        run_intervention=False,
        baseline_path=args.baseline_path,
    )
    return config


def main():
    config = parse_args()
    run_partial_reroll(config)


if __name__ == "__main__":
    main()
