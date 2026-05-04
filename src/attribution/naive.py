"""Naive attributer implementation."""

import logging

from src.attribution.base import BaseAttributer

logger = logging.getLogger(__name__)


class IdentityAttributer(BaseAttributer):
    """
    A naive attributer that assigns the full, unmodified system-level score
    to every agent that participated in each run.
    """

    def attribute(
        self,
        agent_prompt_batch,
        system_scores,
        intermediate_results,
    ):
        agent_names = agent_prompt_batch.get_agent_names()
        attributed_scores = {agent_name: [] for agent_name in agent_names}

        for score in system_scores:
            for agent_name in agent_names:
                attributed_scores[agent_name].append(score)

        return attributed_scores
