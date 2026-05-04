"""Intelligent attributer using LLM-based credit assignment."""

import json
import logging

import numpy as np

from src.attribution.base import BaseAttributer
from src.meta_prompts import ATTRIBUTION_PROMPT
from src.prompt_utils import constrain_prompt_length

logger = logging.getLogger(__name__)


class AbsoluteAttributer(BaseAttributer):
    """Attribute per-query rewards across agents using intermediate state.

    Expected shapes:
    - `system_scores`: list[run] of list[query] floats
    - `intermediate_results`: list[run] of list[query] dict

    Conservation guarantee:
    For each run/query, sum_agent credit == system_scores[run][query].

    Attribution approach:
    Groups all prompt-sets together per query for comparative analysis by LLM.
    Uses seeded random shuffling of prompt-sets to avoid order bias.

    Difference to intelligent attributer:
    - No constraints on Sum of credits per query to equal system score.
    LLM can assign any credit distribution.
    """

    def __init__(
        self,
        llm=None,
        agent_var_mapping=None,
        prompt_template=None,
        group_size=4,
        max_tokens_per_field=1024,
        seed=42,
    ):
        super().__init__(llm=llm, agent_var_mapping=agent_var_mapping)
        self.prompt_template = prompt_template or ATTRIBUTION_PROMPT
        self.group_size = group_size
        self.max_tokens_per_field = max_tokens_per_field
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _parse_intermediate_results(self, intermediate_results, agent_names):
        cleaned = []
        for ir in intermediate_results:
            cleaned_ir = []
            for agent in agent_names:
                input_vars = self.agent_var_mapping[agent]["input_vars"]
                output_vars = self.agent_var_mapping[agent]["output_vars"]
                state = self._clean_state(ir, only_include=input_vars + output_vars)
                cleaned_ir.append(state)
            cleaned.append(cleaned_ir)

        return cleaned

    def _build_all_prompts(self, agent_names, system_scores, intermediate_results):
        """Build all LLM prompts for attribution, batching prompt-sets by group_size."""
        # Determine number of queries (assume all runs have same query count)
        n_queries = len(system_scores[0]) if len(system_scores) > 0 else 0
        n_runs = len(system_scores)

        prompts = []
        prompt_metadata = []  # List of (query_idx, run_indices_in_group, shuffle_mapping)

        # Build prompts for each query, grouping runs by group_size
        for qi in range(n_queries):
            # Extract query (same across all runs)
            query = self._clean_state(intermediate_results[0][qi])["query"]

            # Process runs in groups
            for group_start in range(0, n_runs, self.group_size):
                group_end = min(group_start + self.group_size, n_runs)
                run_indices = list(range(group_start, group_end))

                # Shuffle run_indices to avoid order bias
                shuffled_indices = run_indices.copy()
                self.rng.shuffle(shuffled_indices)

                # Create mapping from shuffled position to original run_idx
                shuffle_mapping = {i: shuffled_indices[i] for i in range(len(shuffled_indices))}

                # Collect prompt-set results for this group (in shuffled order)
                rollouts = []
                for run_idx in shuffled_indices:
                    state = self._clean_state(intermediate_results[run_idx][qi])
                    rollout = {
                        "system_score": float(system_scores[run_idx][qi]),
                        # "prediction": state.get("prediction", "FAILED"),
                        "agent_outputs": self._state_to_agent_outputs(state),
                    }
                    rollouts.append(rollout)

                query_and_sets = {
                    "query": query,
                    "executions": rollouts,
                }

                query_and_sets = self._truncate_field_value(query_and_sets)

                prompt = self._safe_format_prompt(
                    self.prompt_template,
                    # agent_context=agent_context_json,
                    query_and_prompt_sets=json.dumps(query_and_sets, ensure_ascii=False, indent=2),
                    n_parametrizations=str(len(rollouts)),
                    agent_names=str(agent_names),
                )
                prompts.append(prompt)
                prompt_metadata.append((qi, run_indices, shuffle_mapping))

        prompts = [
            constrain_prompt_length(prompt, self.llm.max_tokens - self.max_tokens_per_field)
            for prompt in prompts
        ]

        return prompts, prompt_metadata

    def _safe_format_prompt(self, prompt, **kwargs):
        """Format prompt template robustly via granular placeholder replacement."""
        for key, value in kwargs.items():
            placeholder = "{" + str(key) + "}"
            try:
                prompt = prompt.replace(placeholder, str(value))
            except Exception as e:
                logger.warning(f"Failed to replace placeholder '{key}': {e}")

        return prompt

    def _parse_attributions_response(self, response, agent_names, group_size):
        """Parse LLM response and return attribution matrix (group_size × n_agents)."""
        attribution_matrix = np.zeros((group_size, len(agent_names)))

        payload = self._extract_tag(response, "attribution")
        try:
            attributions_by_run = json.loads(payload)
            attributions_by_run = attributions_by_run["executions"]
            for group_idx in range(group_size):
                attributions = attributions_by_run[group_idx]
                attributions = attributions["agent_credits"]
                attributions = np.array([float(attributions[a]) for a in agent_names])
                attribution_matrix[group_idx] = np.clip(
                    attributions, -1, 1
                )  # Clip according to constraint

        except Exception as e:
            logger.warning(f"Failed to parse attribution weights from response: {response}")
            logger.warning(f"Error: {e}. Falling back to zeros.")

        return attribution_matrix

    def attribute(self, agent_prompt_batch, system_scores, intermediate_results):
        """
        Attributes scores across agents using LLM-based credit assignment.

        Shape documentation:
        - system_scores: list[n_runs] of list[n_queries_per_run] of float
        - intermediate_results: list[n_runs] of list[n_queries_per_run] of dict

        Returns:
        - Dict[agent_name -> list[n_runs] of list[n_queries_per_run] of float]
        """
        if self.llm is None:
            raise ValueError("llm must be provided for IntelligentAttributer")
        agent_names = agent_prompt_batch.get_agent_names()

        n_runs = len(system_scores)
        n_queries = len(system_scores[0]) if len(system_scores) > 0 else 0
        # Build all prompts (grouped by group_size prompt-sets per query)
        prompts, prompt_metadata = self._build_all_prompts(
            agent_names, system_scores, intermediate_results
        )
        try:
            responses = self.llm.get_response(prompts)
        except Exception as e:
            logger.error(f"LLM call for attribution failed: {e}")
            responses = [""] * len(prompts)  # Fallback to empty responses
        attributed = {a: [[None] * n_queries for _ in range(n_runs)] for a in agent_names}
        for response, (qi, run_indices, shuffle_mapping) in zip(responses, prompt_metadata):
            group_size = len(run_indices)
            attribution_matrix = self._parse_attributions_response(
                response, agent_names, group_size
            )

            for shuffled_idx in range(group_size):
                run_idx = shuffle_mapping[shuffled_idx]
                for ai, agent_name in enumerate(agent_names):
                    credit = attribution_matrix[shuffled_idx, ai]
                    attributed[agent_name][run_idx][qi] = credit

        return attributed
