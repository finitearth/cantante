"""
Resume an interrupted experiment from a checkpoint produced by CheckpointCallback.

Usage:
  uv run experiments/restart_experiment.py \
      --config_path dataset_gsm8k_optimizer_cantante_seed_7/config.yaml \
      --checkpoint dataset_gsm8k_optimizer_cantante_seed_7/checkpoints/checkpoint_step_0003.json
"""

import argparse
import json
import logging
import os

import yaml
from promptolution.llms import APILLM
from promptolution.utils.callbacks import TokenCountCallback

from optimization.cantante import build_mas_optimizer
from src.agent_tools.base import get_tools
from src.callbacks import CheckpointCallback, OptimizationCallback
from src.experiment.load_datasets import get_tasks
from src.experiment.utils import get_logger, inject_tool_descriptions, seed_everything
from src.mas import MASPredictor
from src.prompt_structures import AgentPromptPool

logging.getLogger("httpx").setLevel(logging.WARNING)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", required=True, help="Path to the original experiment config YAML."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the checkpoint JSON file to resume from."
    )
    return parser.parse_args()


def main():
    args = get_args()

    # ------------------------------------------------------------------ config
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    setup_dict_path = config["setup_dict_folder"] + "/" + config["experiment"]["dataset"] + ".yaml"
    with open(setup_dict_path) as f:
        setup_dict = yaml.safe_load(f) or {}

    initial_prompts = setup_dict.pop("init_agent_prompt_pool")
    config["setup_dict"] = setup_dict
    config["init_agent_prompt_pool"] = initial_prompts
    config["config_path"] = args.config_path

    # ---------------------------------------------------------------- checkpoint
    with open(args.checkpoint) as f:
        checkpoint = json.load(f)

    resumed_step = checkpoint["_global_step"]
    total_steps = config["experiment"]["n_steps"]
    remaining_steps = total_steps - resumed_step
    logging_dir = config["experiment"]["output_dir"]
    os.makedirs(logging_dir, exist_ok=True)

    logger = get_logger(
        __name__,
        logging.ERROR,
        log_to_file=True,
        log_file_path=os.path.join(logging_dir, "experiment_restart.log"),
        file_level=logging.INFO,
    )

    logger.info(f"Resuming from step {resumed_step}, running {remaining_steps} more steps.")

    # Remove stale lock file if present (original run likely crashed)
    lock_file = os.path.join(logging_dir, ".experiment.lock")
    if os.path.exists(lock_file):
        logger.warning(f"Removing stale lock file at {lock_file}")
        os.remove(lock_file)

    # ------------------------------------------------- remaining token budget
    # Tokens already consumed are recorded in the checkpoint by CheckpointCallback.
    token_counts = checkpoint.get("_token_counts", {})
    tokens_spent = (
        token_counts.get("downstream_input", 0)
        + token_counts.get("downstream_output", 0)
        + token_counts.get("meta_input", 0)
        + token_counts.get("meta_output", 0)
    )
    remaining_budget = max(0, config["experiment"]["max_token_budget"] - tokens_spent)
    logger.info(
        f"Tokens spent before checkpoint: {tokens_spent}, remaining budget: {remaining_budget}"
    )

    # ---------------------------------------------------------- infrastructure
    seed_everything(config["experiment"]["seed"])

    tools_adapter = get_tools(config["experiment"]["dataset"])

    dev_task = get_tasks(
        config["experiment"]["dataset"],
        split="train",
        block_size=config["experiment"]["block_size"],
        dev_size=config["experiment"].get("dev_size", 300),
        seed=config["experiment"]["seed"],
        eval_strategy=config["experiment"]["eval_strategy"],
        tools_adapter=tools_adapter,
    )

    if config["experiment"]["node_optimizer"] == "EvoPrompt":
        dev_task.eval_strategy = "subsample"

    api_key = open("token.txt").read().strip()

    predictor = MASPredictor(
        config["setup_dict"],
        dict(api_key=api_key, **config["task_llm_kwargs"]),
        seed=config["experiment"]["seed"],
        tool_adapter=tools_adapter,
        recursion_limit=config["experiment"]["recursion_limit"],
        predict_use_tqdm=config["experiment"].get("predict_use_tqdm", True),
        max_tokens_per_field=config["experiment"]["max_tokens_per_field"],
    )

    meta_llm_cfg = config["meta_llm_kwargs"]
    meta_llm = APILLM(
        api_url=meta_llm_cfg["base_url"],
        model_id=meta_llm_cfg["model"],
        max_tokens=meta_llm_cfg["max_tokens"],
        api_key=api_key,
        max_retries=meta_llm_cfg["max_retries"],
        call_timeout_s=meta_llm_cfg["timeout"],
        call_kwargs={
            "temperature": meta_llm_cfg["temperature"],
            "seed": config["experiment"]["seed"],
        },
    )

    checkpoint_dir = os.path.join(logging_dir, "checkpoints")
    callbacks = [
        TokenCountCallback(remaining_budget, "total_tokens"),
        OptimizationCallback(log_dir=logging_dir, meta_llm=meta_llm),
        CheckpointCallback(checkpoint_dir=checkpoint_dir, meta_llm=meta_llm),
    ]

    init_prompts_with_tools = inject_tool_descriptions(
        config["init_agent_prompt_pool"],
        config["setup_dict"],
        tools_adapter,
    )
    init_agent_prompt_pool = AgentPromptPool.from_mapping(init_prompts_with_tools)

    optimizer = build_mas_optimizer(
        config=config,
        predictor=predictor,
        task=dev_task,
        init_agent_prompt_pool=init_agent_prompt_pool,
        meta_llm=meta_llm,
        callbacks=callbacks,
    )

    # ------------------------------------------------ setup + inject checkpoint
    # _pre_optimization_loop builds proxy_tasks, agent_optimizers, and runs each
    # CAPO's _pre_optimization_loop (sets up max_prompt_length etc.).
    # load_state_dict then overwrites the evolving state with checkpoint values.
    optimizer._pre_optimization_loop()
    optimizer.load_state_dict(checkpoint)
    logger.info(f"Loaded checkpoint from {args.checkpoint} (step {resumed_step}).")

    # ---------------------------------------------------- run remaining steps
    best_prompts = optimizer.optimize(n_steps=remaining_steps, skip_pre_loop=True)
    logger.info("Restart complete.")
    logger.info(best_prompts)

    with open(os.path.join(logging_dir, ".finished"), "w") as f:
        f.write("ok\n")


if __name__ == "__main__":
    main()
