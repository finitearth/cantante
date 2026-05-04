# Prompt structure terminology now lives in `prompt_structures.py`.
from itertools import cycle

from src.prompt_structures import AgentPromptBatch, AgentPromptPool


class BaseCandidateSelector:
    """
    Abstract base class for a candidate selector.

    It defines the interface for constructing a batch of
    'agent_prompt_set's from pending and fallback pools.
    """

    def __init__(self):
        pass

    def select_candidates(
        self,
        pending_by_agent: dict,
        agent_optimizers: dict,
    ) -> AgentPromptBatch:
        """
        Selects and combines agent prompts to create a batch for evaluation.
        Handles all pool preparation, fallback logic, and candidate selection.

        Args:
            pending_by_agent (dict): Maps agent names to pending objects with .prompts attribute.
            agent_optimizers (dict): Maps agent names to optimizer objects with .prompts attribute.

        Returns:
            AgentPromptBatch: Batch of prompt sets ready for evaluation.
        """
        # Extract prompts from objects
        pending_prompts = {name: obj.prompts for name, obj in pending_by_agent.items()}
        fallback_prompts = {name: obj.prompts for name, obj in agent_optimizers.items()}

        # Build pools
        pending_pool = AgentPromptPool.from_mapping(pending_prompts)
        fallback_pool = AgentPromptPool.from_mapping(fallback_prompts).sample_set_as_pool()

        # Build full pool with fallbacks
        agent_prompt_pool = pending_pool.with_fallbacks(
            required_agents=list(agent_optimizers.keys()),
            fallback_pool=fallback_pool,
        )

        return self._select_from_pool(agent_prompt_pool)

    def _select_from_pool(self, agent_prompt_pool: AgentPromptPool) -> AgentPromptBatch:
        """
        Internal method to select candidates from a complete pool.
        Subclasses should override this method.
        """
        raise NotImplementedError


class NaiveSelector(BaseCandidateSelector):
    """
    A naive candidate selector that pairs the first prompt of agent A with
    the first prompt of agent B, the second with the second, and so on.

    If agents have different numbers of prompts, this selector will
    stop at the length of the shortest prompt list (due to zip).
    """

    def _select_from_pool(self, agent_prompt_pool: AgentPromptPool) -> AgentPromptBatch:
        agent_names = agent_prompt_pool.get_agent_names()
        prompts_by_agent = agent_prompt_pool.get_agent_prompt_lists()

        # Turn each list into an infinite cycle
        cycled_lists = [cycle(lst) for lst in prompts_by_agent]

        # Find the max length (so we produce that many sets)
        max_len = max(len(lst) for lst in prompts_by_agent)

        agent_prompt_batch = []
        for _ in range(max_len):
            tuple_prompts = [next(c) for c in cycled_lists]
            agent_prompt_batch.append(dict(zip(agent_names, tuple_prompts)))

        return AgentPromptBatch.from_iterable(agent_prompt_batch)


ALL_SELECTORS = {"naive": NaiveSelector}
