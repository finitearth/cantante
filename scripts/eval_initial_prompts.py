"""Evaluate initial prompts from a config file.

For each dataset in the config grid, samples N_SAMPLES prompt sets (using the
config seed for reproducibility), then evaluates each set on one of the
EVAL_SEEDS. Each evaluation is stored in its own run dir immediately:

  {output_dir}/seed_{eval_seed}_dataset_{dataset}_optimizer_initial/
    config.yaml
    runinfo.json
    eval/
      eval.log
      scores_per_step.parquet
      token_usage.yaml

Usage:
  uv run experiments/eval_initial_prompts.py --config_path config/main_experiments/main_exp.yaml
"""

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from src.agent_tools.base import get_tools
from src.experiment.load_datasets import get_tasks
from src.experiment.utils import get_logger, inject_tool_descriptions, seed_everything
from src.mas import MASPredictor
from src.prompt_structures import AgentPromptBatch, AgentPromptPool

EVAL_SEEDS = [7, 42, 47]
N_SAMPLES = 3


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ignore-locks", action="store_true")
    return parser.parse_args()


def _load_config(config_path, dataset):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config.pop("grid", None)
    config["experiment"]["dataset"] = dataset
    config["config_path"] = config_path

    setup_dict_path = config["setup_dict_folder"] + "/" + dataset + ".yaml"
    with open(setup_dict_path, "r") as f:
        setup_dict = yaml.safe_load(f) or {}

    initial_prompts = setup_dict.pop("init_agent_prompt_pool")
    config["setup_dict"] = setup_dict
    config["init_agent_prompt_pool"] = initial_prompts

    return config


def _grid_datasets(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["grid"]["experiment.dataset"]


def _write_runinfo(run_dir, config):
    run_info = {
        "start_time": datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d-%m-%Y %H:%M"),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "config_path": config.get("config_path"),
    }
    run_info["git_hash"] = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
        .decode("ascii")
        .strip()
    )
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=os.getcwd()).decode(
        "utf-8"
    )
    run_info["git_dirty"] = len(status.strip()) > 0

    with open(os.path.join(run_dir, "runinfo.json"), "w") as f:
        json.dump(run_info, f, indent=2)


def main():
    args = _parse_args()
    datasets = _grid_datasets(args.config_path)

    for dataset in datasets:
        config = _load_config(args.config_path, dataset)
        exp_cfg = config["experiment"]
        base_output_dir = exp_cfg["output_dir"].rstrip("/")

        seed_everything(0)
        tools_adapter = get_tools(dataset)
        api_key = open("token.txt").read().strip()
        model_kwargs = dict(api_key=api_key, **config["task_llm_kwargs"])

        init_prompts_with_tools = inject_tool_descriptions(
            config["init_agent_prompt_pool"],
            config["setup_dict"],
            tools_adapter,
        )
        pool = AgentPromptPool.from_mapping(init_prompts_with_tools)
        prompt_sets = pool.sample_batch(N_SAMPLES).prompt_sets

        for eval_seed, prompt_set in zip(EVAL_SEEDS, prompt_sets):
            run_dir = f"{base_output_dir}/seed_{eval_seed}_dataset_{dataset}_optimizer_initial"
            eval_dir = f"{run_dir}/eval"
            os.makedirs(eval_dir, exist_ok=True)

            logger = get_logger(
                f"{__name__}.{eval_seed}.{dataset}",
                log_to_file=True,
                log_file_path=f"{eval_dir}/eval.log",
            )

            lock_file = os.path.join(run_dir, ".eval.lock")
            if not args.ignore_locks and os.path.exists(lock_file):
                logger.warning(f"Skipping seed={eval_seed} dataset={dataset}: lock file exists.")
                continue
            if os.path.exists(os.path.join(run_dir, ".finished")):
                logger.warning(f"Skipping seed={eval_seed} dataset={dataset}: already finished.")
                continue

            with open(lock_file, "w") as f:
                f.write(f"Started at PID: {os.getpid()}\n")

            _write_runinfo(run_dir, config)
            with open(os.path.join(run_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(config, f)

            logger.warning(f"Starting eval: seed={eval_seed} dataset={dataset} -> {run_dir}")

            try:
                seed_everything(eval_seed)
                predictor = MASPredictor(
                    config["setup_dict"],
                    model_kwargs,
                    seed=eval_seed,
                    tool_adapter=tools_adapter,
                    recursion_limit=exp_cfg["recursion_limit"],
                    predict_use_tqdm=exp_cfg.get("predict_use_tqdm", True),
                    max_tokens_per_field=exp_cfg["max_tokens_per_field"],
                )

                test_task = get_tasks(
                    dataset,
                    split="test",
                    seed=eval_seed,
                    eval_strategy="full",
                    tools_adapter=tools_adapter,
                )

                apb = AgentPromptBatch.from_iterable([prompt_set])
                scores, _ = test_task.evaluate(apb, predictor)
                score = scores.mean(axis=-1)[0]
                logger.warning(f"Score: {score}")

                pd.DataFrame(
                    {"agent_prompt_set": [prompt_set.to_string()], "test_score": [score]}
                ).to_parquet(f"{eval_dir}/scores_per_step.parquet", index=False)

                token_usage = predictor.llm.get_token_count()
                logger.warning(f"Token usage: {token_usage}")
                with open(f"{eval_dir}/token_usage.yaml", "w") as f:
                    yaml.safe_dump(token_usage, f)

                with open(os.path.join(run_dir, ".finished"), "w") as f:
                    f.write("ok\n")

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                os.remove(lock_file)
                raise

            os.remove(lock_file)


if __name__ == "__main__":
    main()
