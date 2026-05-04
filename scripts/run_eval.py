import argparse
import os

import pandas as pd
import yaml

from src.agent_tools.base import get_tools
from src.experiment.load_datasets import get_tasks
from src.experiment.utils import get_logger, seed_everything
from src.mas import MASPredictor
from src.prompt_structures import AgentPromptBatch, AgentPromptSet


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--step", type=int, default=None)
    return parser.parse_args()


def run_eval(args, eval_dir, logger):
    # Ensure the run is marked finished
    finished_flag = os.path.join(args.run_dir, ".finished")
    if not os.path.exists(finished_flag):
        raise Exception(f"Run not marked finished: {args.run_dir}")

    config_path = f"{args.run_dir}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    exp_cfg = config["experiment"]

    seed_everything(exp_cfg["seed"])

    tools_adapter = get_tools(exp_cfg["dataset"])

    history_path = f"{args.run_dir}/prompts_per_step.parquet"
    hist_df = pd.read_parquet(history_path)

    if args.step is not None:
        hist_df = hist_df[hist_df["step"] == args.step]
    elif not args.eval_all:
        # Only evaluate the final step, with highest score
        hist_df["total_tokens"] = (
            hist_df["downstream_input_tokens"]
            + hist_df["downstream_output_tokens"]
            + hist_df["meta_input_tokens"]
            + hist_df["meta_output_tokens"]
        )
        max_token = config["experiment"]["max_token_budget"]
        last_valid_step = hist_df[hist_df["total_tokens"] <= max_token].step.max()
        hist_df = hist_df[hist_df["step"] == last_valid_step]

    hist_df = hist_df.drop_duplicates(subset=["prompt"])
    agent_prompt_batch_strs = hist_df["prompt"].tolist()
    apb = AgentPromptBatch.from_iterable(
        [AgentPromptSet.from_string(aps) for aps in agent_prompt_batch_strs]
    )

    test_task = get_tasks(
        exp_cfg["dataset"],
        split="test",
        seed=exp_cfg["seed"],
        eval_strategy="full",
        tools_adapter=tools_adapter,
    )

    api_key = open("token.txt").read().strip()
    model_kwargs = dict(api_key=api_key, **config["task_llm_kwargs"])
    predictor = MASPredictor(
        config["setup_dict"],
        model_kwargs,
        seed=exp_cfg["seed"],
        tool_adapter=tools_adapter,
        recursion_limit=exp_cfg["recursion_limit"],
        predict_use_tqdm=exp_cfg.get("predict_use_tqdm", True),
        max_tokens_per_field=exp_cfg["max_tokens_per_field"],
    )

    logger.warning(f"Setup finished. Starting evaluation on {len(apb)} agent prompt sets.")
    logger.warning("STARTING EVALUATION")

    scores, seqs = test_task.evaluate(apb, predictor)

    logger.warning("EVAL FINISHED")
    # logger.warning(f"Test sequences: {seqs}")
    logger.warning(f"Test scores: {scores.mean(axis=-1)}")
    pd.DataFrame(
        {"agent_prompt_set": agent_prompt_batch_strs, "test_score": scores.mean(axis=-1)}
    ).to_parquet(f"{eval_dir}/scores_per_step.parquet", index=False)

    token_usage = predictor.llm.get_token_count()
    logger.warning(f"Token usage: {token_usage}")
    with open(f"{eval_dir}/token_usage.yaml", "w") as f:
        yaml.safe_dump(token_usage, f)


def main(logger):
    args = _parse_args()
    if args.step is not None:
        eval_dir = f"{args.run_dir}/eval_step_{args.step}"
    elif args.eval_all:
        eval_dir = f"{args.run_dir}/eval_all_final"
    else:
        eval_dir = f"{args.run_dir}/eval"
    os.makedirs(eval_dir, exist_ok=True)

    # Create lock file to prevent concurrent evaluations
    lock_file = os.path.join(args.run_dir, ".eval.lock")
    if os.path.exists(lock_file):
        raise Exception(
            f"Lock file exists at {lock_file}. Another process may be running evaluation."
        )

    with open(lock_file, "w") as f:
        f.write(f"Started at: {os.getpid()}\n")

    try:
        run_eval(args, eval_dir, logger)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        os.remove(lock_file)
        raise e

    os.remove(lock_file)


if __name__ == "__main__":
    args = _parse_args()
    if args.step is not None:
        eval_dir = f"{args.run_dir}/eval_step_{args.step}"
    elif args.eval_all:
        eval_dir = f"{args.run_dir}/eval_all_final"
    else:
        eval_dir = f"{args.run_dir}/eval"
    os.makedirs(eval_dir, exist_ok=True)
    logger = get_logger(__name__, log_to_file=True, log_file_path=f"{eval_dir}/eval.log")
    main(logger)
