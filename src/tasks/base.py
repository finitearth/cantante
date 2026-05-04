from typing import Any, List, Tuple

import numpy as np
from promptolution.tasks.base_task import BaseTask

from src.mas import MASPredictor
from src.prompt_structures import AgentPromptBatch

# See `prompt_structures.py` for canonical terminology around pools/sets/batches.


class BaseMASTask(BaseTask):
    """
    Abstract base class for a Multi-Agent System (MAS) evaluation task.

    This class defines the API expected by the MASOptimizer:
    - It accepts a *batch* of 'agent_prompt_batch'.
    - It returns a *batch* of scores and intermediate results,
      with a 1:1 mapping to the input batch.

    This class handles the 'promptolution' subsampling and the
    critical batching logic, fixing the averaging bug.
    """

    def set_tools_adapter(self, tools_adapter):
        self.tools_adapter = tools_adapter

    def get_tools_adapter(self):
        return self.tools_adapter

    def evaluate(  # type: ignore
        self,
        agent_prompt_batch,
        mas_predictor: "MASPredictor",
        eval_strategy=None,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Evaluates a batch of agent_prompt_batch and returns scores
        and intermediate results, with no aggregation.

        This method overrides the BaseTask.evaluate() to implement
        the correct batching logic for the MASOptimizer.
        """

        eval_strategy = eval_strategy or self.eval_strategy

        # Get the list of datasets to run on for this entire batch
        xs, ys = self.subsample()
        prompt_hash_map, hash_prompts = agent_prompt_batch.get_prompt_hash_map()

        # Create string-keyed lookup map for consistent caching with _prepare_batch
        hash_str_to_prompt_set = {str(h): prompt_hash_map[h] for h in hash_prompts}

        (
            prompts_to_evaluate,
            xs_to_evaluate,
            ys_to_evaluate,
            cache_keys,
        ) = self._prepare_batch(hash_prompts, xs, ys, eval_strategy=eval_strategy)
        if prompts_to_evaluate:
            # Convert hash keys back to AgentPromptSet objects
            prompt_sets_to_eval = [
                hash_str_to_prompt_set[str(hash_key)] for hash_key in prompts_to_evaluate
            ]
            eval_batch = AgentPromptBatch.from_iterable(prompt_sets_to_eval)
            scores, states = self._evaluate(
                eval_batch,
                xs_to_evaluate,
                ys_to_evaluate,
                mas_predictor,
            )
        else:
            scores = []
            states = []
            prompt_sets_to_eval = []

        for i, k in enumerate(cache_keys):
            self.eval_cache[k] = scores[i]
            self.seq_cache[k] = states[i]

        cached_scores, _, cached_states = self._collect_results_from_cache(
            hash_prompts,
            xs,
            ys,
        )

        return cached_scores, cached_states  # type: ignore
