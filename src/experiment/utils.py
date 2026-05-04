import hashlib
import logging
import os
import random
import string
from typing import Optional

import numpy as np

# import torch


def seed_everything(seed: int = 42):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_random_hash(length: int = 5) -> str:
    text = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    hash_object = hashlib.sha256(text.encode())
    hash_string = hash_object.hexdigest()

    return hash_string[:length]


def get_logger(
    name: str,
    level: int = logging.WARNING,
    log_to_file: bool = False,
    log_file_path: Optional[str] = None,
    file_level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger for scripts/modules."""
    # Root logger level must be at most the file level so messages aren't
    # filtered before they reach the file handler.
    root_level = min(level, file_level) if log_to_file else level

    logging.basicConfig(
        level=root_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )

    if log_to_file and log_file_path:
        root_logger = logging.getLogger()
        assert not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(file_level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)

    for noisy in [
        "httpx",
        "httpcore",
        "openai",
        "langchain",
        "langchain_core",
        "langgraph",
        "anthropic",
        "urllib3",
        "asyncio",
    ]:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    return logging.getLogger(name)


def flatten_config(cfg, prefix="", sep="."):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_config(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def inject_tool_descriptions(init_prompt_pool_cfg, setup_dict, tools_adapter):
    agent_tools = {agent["name"]: agent.get("tools", []) for agent in setup_dict.get("agents", [])}
    updated = {}
    for agent_name, prompts in init_prompt_pool_cfg.items():
        tool_names = agent_tools.get(agent_name, [])
        if tool_names:
            tool_description_block = tools_adapter.get_tool_descriptions(tool_names=tool_names)
            updated_prompts = [
                f"{p}\n\nAvailable tools:\n{tool_description_block}" for p in prompts
            ]
        else:
            updated_prompts = prompts
        updated[agent_name] = updated_prompts
    return updated
