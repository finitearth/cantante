"""Callbacks for MAS optimization tracking and monitoring."""

import json
import logging
import os
from time import perf_counter
from typing import TYPE_CHECKING

import pandas as pd
from promptolution.utils.callbacks import BaseCallback

if TYPE_CHECKING:
    from src.optimization.base_mas_optimizer import BaseMASOptimizer

logger = logging.getLogger(__name__)


class OptimizationCallback(BaseCallback):
    """Tracks best prompt sets and cumulative token usage (downstream + meta) at each step."""

    def __init__(self, log_dir=None, meta_llm=None, max_token_budget=10_000_000):
        self.log_dir = log_dir
        self.meta_llm = meta_llm
        self.max_token_budget = max_token_budget
        self.best_agent_prompt_set = None

        os.makedirs(self.log_dir, exist_ok=True)

    def on_step_end(self, optimizer):
        step = getattr(optimizer, "_global_step", None)
        downstream = optimizer.predictor.llm.get_token_count()
        meta = self.meta_llm.get_token_count()

        total_tokens = (
            downstream["input_tokens"]
            + downstream["output_tokens"]
            + meta["input_tokens"]
            + meta["output_tokens"]
        )
        new_row = {
            "prompt": optimizer.get_best_candidate().to_string(),
            "step": step,
            "downstream_input_tokens": downstream["input_tokens"],
            "downstream_output_tokens": downstream["output_tokens"],
            "meta_input_tokens": meta["input_tokens"],
            "meta_output_tokens": meta["output_tokens"],
            "total_tokens": total_tokens,
            "timestamp": perf_counter(),
        }

        logger.warning(
            f"Step {step} token usage - Downstream input: {downstream['input_tokens']}, "
            f"Downstream output: {downstream['output_tokens']}, "
            f"Meta input: {meta['input_tokens']}, Meta output: {meta['output_tokens']}"
        )

        # If file exists, append, else create
        step_file = f"{self.log_dir}/prompts_per_step.parquet"
        if os.path.exists(step_file):
            old_df = pd.read_parquet(step_file)
            new_df = pd.concat([old_df, pd.DataFrame([new_row])], ignore_index=True)
            new_df.to_parquet(step_file, index=False)
        else:
            pd.DataFrame([new_row]).to_parquet(step_file, index=False)

        if total_tokens >= self.max_token_budget:
            logger.warning("Max token budget tokens reached. Stopping optimization.")
            return False  # Stop optimization
        return True


class CheckpointCallback(BaseCallback):
    """Saves a JSON checkpoint of the optimizer state after every step.

    The checkpoint includes cumulative token counts so that a restart can
    compute the remaining token budget and initialize TokenCountCallback
    correctly.
    """

    def __init__(self, checkpoint_dir: str, meta_llm=None):
        self.checkpoint_dir = checkpoint_dir
        self.meta_llm = meta_llm
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_step_end(self, optimizer: "BaseMASOptimizer") -> bool:
        step = optimizer._global_step
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:04d}.json")

        state = optimizer.state_dict()

        downstream = optimizer.predictor.llm.get_token_count()
        meta = (
            self.meta_llm.get_token_count()
            if self.meta_llm is not None
            else {"input_tokens": 0, "output_tokens": 0}
        )
        state["_token_counts"] = {
            "downstream_input": downstream["input_tokens"],
            "downstream_output": downstream["output_tokens"],
            "meta_input": meta["input_tokens"],
            "meta_output": meta["output_tokens"],
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")
        return True
