import argparse
import json
import logging
import os
import platform
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml
from promptolution.llms import APILLM

from src.optimization.cantante import build_mas_optimizer
from src.agent_tools.base import get_tools
from src.callbacks import CheckpointCallback, OptimizationCallback
from src.experiment.load_datasets import get_tasks
from src.experiment.utils import get_logger, inject_tool_descriptions, seed_everything
from src.mas import MASPredictor
from src.prompt_structures import AgentPromptPool

logging.getLogger("httpx").setLevel(logging.WARNING)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ignore-locks", action="store_true")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    setup_dict_path = config["setup_dict_folder"] + "/" + config["experiment"]["dataset"] + ".yaml"
    with open(setup_dict_path, "r") as setup_file:
        setup_dict = yaml.safe_load(setup_file) or {}

    initial_prompts = setup_dict.pop("init_agent_prompt_pool")
    config["setup_dict"] = setup_dict
    config["init_agent_prompt_pool"] = initial_prompts

    # Add origin of config for run info
    config["config_path"] = args.config_path
    config["ignore_locks"] = args.ignore_locks

    return config


def write_runinfo(logging_dir: str, config: dict):
    """Collect and write run information JSON to logging_dir."""
    start_dt = datetime.now(ZoneInfo("Europe/Berlin"))
    run_info = {
        "start_time": start_dt.strftime("%d-%m-%Y %H:%M"),
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

    os.makedirs(logging_dir, exist_ok=True)
    with open(os.path.join(logging_dir, "runinfo.json"), "w") as f:
        json.dump(run_info, f, indent=2)


def run_experiment(config, logging_dir, logger):
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

    model_kwargs = dict(api_key=api_key, **config["task_llm_kwargs"])
    predictor = MASPredictor(
        config["setup_dict"],
        model_kwargs,
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
    logger.info("Starting optimization...")
    best_prompts = optimizer.optimize(n_steps=config["experiment"]["n_steps"])
    logger.info("Best Prompts:")
    logger.info(best_prompts)
    return best_prompts


def main(config=None):
    if config is None:
        config = get_config()
    logging_dir = config["experiment"]["output_dir"]
    os.makedirs(logging_dir, exist_ok=True)

    logger = get_logger(
        __name__,
        logging.ERROR,
        log_to_file=True,
        log_file_path=f"{logging_dir}/experiment.log",
        file_level=logging.INFO,
    )

    logger.info("Starting experiment! Logging to %s/experiment.log", logging_dir)
    ignore_locks = config.get("ignore_locks", False)

    # Create lock file to prevent concurrent runs
    lock_file = os.path.join(logging_dir, ".experiment.lock")
    if not ignore_locks:
        if os.path.exists(lock_file):
            raise Exception(
                f"Lock file exists at {lock_file}. Another process may be running this experiment."
            )
        with open(lock_file, "w") as f:
            f.write(f"Started at PID: {os.getpid()}\n")

    # Write run info JSON
    write_runinfo(logging_dir, config)

    with open(os.path.join(logging_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    try:
        run_experiment(config, logging_dir, logger)
        # Mark run as finished for the evaluator
        with open(os.path.join(logging_dir, ".finished"), "w") as f:
            f.write("ok\n")
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        raise e

    if not ignore_locks and os.path.exists(lock_file):
        os.remove(lock_file)


if __name__ == "__main__":
    main()
