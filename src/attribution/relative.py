"""Intelligent attributer using LLM-based credit assignment."""

import json
import logging

import numpy as np

from src.attribution.base import BaseAttributer
from src.meta_prompts import CONSTRAINED_ATTRIBUTION_PROMPT

logger = logging.getLogger(__name__)


class RelativeAttributer(BaseAttributer):
    """Attribute per-query rewards across agents using intermediate state.

    Expected shapes:
    - `system_scores`: list[run] of list[query] floats
    - `intermediate_results`: list[run] of list[query] dict

    Conservation guarantee:
    For each run/query, sum_agent credit == system_scores[run][query].

    Attribution approach:
    Groups all prompt-sets together per query for comparative analysis by LLM.
    Uses seeded random shuffling of prompt-sets to avoid order bias.
    """

    def __init__(
        self,
        llm=None,
        agent_var_mapping=None,
        prompt_template=None,
        group_size=4,
        seed=42,
    ):
        super().__init__(llm=llm, agent_var_mapping=agent_var_mapping)
        self.prompt_template = prompt_template or CONSTRAINED_ATTRIBUTION_PROMPT
        self.group_size = group_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _build_all_prompts(self, agent_names, system_scores, intermediate_results):
        """Build all LLM prompts for attribution, batching prompt-sets by group_size."""
        # Build agent context once (shared across all prompts)
        agent_context = [
            {agent: self.agent_var_mapping[agent]["task_description"]} for agent in agent_names
        ]
        agent_context_json = json.dumps(agent_context, ensure_ascii=False)

        # Determine number of queries (assume all runs have same query count)
        n_queries = len(system_scores[0]) if len(system_scores) > 0 else 0
        n_runs = len(system_scores)

        prompts = []
        prompt_metadata = []  # List of (query_idx, run_indices_in_group, shuffle_mapping)

        # Build prompts for each query, grouping runs by group_size
        for qi in range(n_queries):
            # Extract query (should be same across all runs)
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
                prompt_sets = []
                for run_idx in shuffled_indices:
                    state = self._clean_state(intermediate_results[run_idx][qi])
                    prompt_sets.append(
                        {
                            "system_score": float(system_scores[run_idx][qi]),
                            "prediction": state["prediction"],
                            "agent_outputs": self._state_to_agent_outputs(state),
                        }
                    )

                query_and_sets = {
                    "query": query,
                    "parametrizations": prompt_sets,
                    "expected_output": {
                        "n_parametrizations": len(prompt_sets),
                        "agent_names": agent_names,
                    },
                }

                prompt = self.prompt_template.format(
                    agent_context=agent_context_json,
                    query_and_prompt_sets=json.dumps(query_and_sets, ensure_ascii=False),
                    n_parametrizations=str(len(prompt_sets)),
                    agent_names=str(agent_names),
                )
                prompts.append(prompt)
                prompt_metadata.append((qi, run_indices, shuffle_mapping))

        return prompts, prompt_metadata

    def _parse_weights_response(self, response, agent_names, group_size):
        """Parse LLM response and return normalized weight matrix (group_size × n_agents)."""
        payload = self._extract_tag(response, "weights")
        weight_matrix = np.ones((group_size, len(agent_names))) / len(agent_names)

        try:
            weights_by_run = json.loads(payload)

            for group_idx in range(group_size):
                weights = weights_by_run[group_idx]
                weights = np.array([max(0.0, float(weights[a])) for a in agent_names])
                weight_matrix[group_idx] = weights / weights.sum()

        except Exception as e:
            logger.warning(f"Failed to parse attribution weights from response: {response}")
            logger.warning(f"Error: {e}. Using uniform distribution for all groups.")

        return weight_matrix

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

        # Batch LLM call
        logger.info("Calling LLM for attribution...")
        try:
            responses = self.llm.get_response(prompts)
        except Exception as e:
            logger.error(f"LLM call for attribution failed: {e}")
            responses = [""] * len(prompts)  # Fallback to empty responses
        logger.info(f"Received {len(responses)} LLM responses")

        # Initialize attributed structure: {agent_name -> [[query_credits]...]}
        attributed = {a: [[None] * n_queries for _ in range(n_runs)] for a in agent_names}

        # Process each response
        for response, (qi, run_indices, shuffle_mapping) in zip(responses, prompt_metadata):
            group_size = len(run_indices)

            # Get normalized weight matrix (group_size × n_agents) for this group
            weight_matrix = self._parse_weights_response(response, agent_names, group_size)

            # For each run in the group, compute credits for this query
            # Map shuffled positions back to original run indices
            for shuffled_idx in range(group_size):
                run_idx = shuffle_mapping[shuffled_idx]
                score = system_scores[run_idx][qi]
                for ai, agent_name in enumerate(agent_names):
                    credit = weight_matrix[shuffled_idx, ai] * score
                    attributed[agent_name][run_idx][qi] = credit

        logger.info("Attribution complete")
        return attributed
