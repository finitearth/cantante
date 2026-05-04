"""Base attributer interface."""

import re
from abc import ABC, abstractmethod

from src.prompt_utils import constrain_prompt_length


def build_agent_var_mapping(predictor):
    """Build agent variable mapping with input/output vars and task descriptions.

    Args:
        predictor: MASPredictor instance.

    Returns:
        Dict mapping agent_name -> {input_vars, output_vars, task_description}
        Example: {"AgentA": {"input_vars": [...], "output_vars": [...], "task_description": "..."}}
    """
    mapping = {}
    for agent_name, agent in predictor.agents.items():
        mapping[agent_name] = {
            "input_vars": agent.input_vars,
            "output_vars": agent.output_vars,
            "task_description": agent.task_description,
        }
    return mapping


class BaseAttributer(ABC):
    """
    Abstract base class for credit assignment (attribution).

    It defines the interface for distributing a system-level score
    back to the individual agents that contributed to it.
    """

    def __init__(self, llm=None, agent_var_mapping=None, *args, **kwargs):
        """Create an attributer.

        Args:
            llm: Optional LLM client/wrapper for model-based attribution.
            agent_var_mapping: Dict mapping agent -> {input_vars, output_vars, task_description}.
        """
        self.llm = llm
        self.agent_var_mapping = agent_var_mapping or {}

    def _clean_state(self, state, only_include=None):
        """Drop non-attribution fields from a single state dict (returns a copy)."""
        if state is None:
            return {}
        ignore = {"few_shots", "agent_prompt_set"}
        state = {k: v for k, v in state.copy().items() if k not in ignore}
        if only_include is not None:
            state = {k: v for k, v in state.items() if k in only_include}
        return state

    def _state_to_agent_outputs(self, state):
        """Convert a state dict into {agent_name: {key: value}}."""
        state = self._clean_state(state)
        by_agent = {agent_name: {} for agent_name in self.agent_var_mapping}
        for agent_name, var_info in self.agent_var_mapping.items():
            for output_key in var_info["output_vars"]:
                if output_key in state:
                    by_agent[agent_name][output_key] = state[output_key]
        return by_agent

    def _extract_tag(self, text, tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", str(text), flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    def _truncate_field_value(self, value):
        if self.max_tokens_per_field is None:
            return value

        if isinstance(value, str):
            return constrain_prompt_length(value, self.max_tokens_per_field)

        if isinstance(value, dict):
            return {k: self._truncate_field_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._truncate_field_value(v) for v in value]

        return value

    @abstractmethod
    def attribute(
        self,
        agent_prompt_batch,
        system_scores,
        intermediate_results,
    ):
        """
        Assigns credit to each agent based on the system's performance.

        ---
        **Design Rule: No Aggregation**
        This method operates on a batch level. The 'system_scores' input is a
        list of floats, where each float corresponds to one 'agent_prompt_set'.

        This method MUST return a dictionary where each value is a list
        of scores of the *exact same length* as the 'system_scores' input.
        Aggregation is NOT performed at this step.
        ---

        Args:
            agent_prompt_batch (List[Dict[str, str]]): The batch of prompt sets
                that were evaluated. agent_prompt_batch[i] corresponds
                to system_scores[i].
            system_scores (List[float]): The list of system-level scores,
                one for each agent_prompt_set.
            intermediate_results (List[Any]): A list of intermediate outputs
                from the task, one for each agent_prompt_set.
                intermediate_results[i] corresponds to system_scores[i].

        Returns:
            Dict[str, List[float]]: A dictionary mapping each agent's name
                to a list of attributed scores.
        """
        raise NotImplementedError("Subclasses must implement the attribute method.")
