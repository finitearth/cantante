"""
Shared prompt data structures.

Key terms (normalized by these classes):
- agent prompt: a single template string or CustomPrompt.
- prompt pool: Dict[str, List[prompt]] mapping agent -> candidate prompts.
- prompt set: Dict[str, prompt] representing one candidate per agent.
- prompt batch: List[prompt set] for batch evaluation/selection.

Use AgentPromptPool/AgentPromptSet/AgentPromptBatch to convert between
string dicts and CustomPrompt instances without ad-hoc list-of-dicts handling.
"""

import json
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Union

from promptolution.utils import Prompt


class CustomPrompt:
    """Custom prompt class that handles few-shot examples differently than promptolution."""

    def __init__(self, instruction, few_shots=None):
        if isinstance(instruction, Prompt) or isinstance(instruction, CustomPrompt):
            few_shots = instruction.few_shots
            instruction = instruction.instruction
        self.instruction = instruction.strip()
        self.few_shots = few_shots or []

    def construct_prompt(self):
        few_shot_str = "\n\n".join(self.few_shots).strip()
        if "{few_shots}" in self.instruction:
            if few_shot_str:
                # Replace placeholder with actual fewshots, then strip tags while keeping content
                prompt = self.instruction.replace("{few_shots}", few_shot_str)
                prompt = re.sub(
                    r"<few_shots_section>(.*?)</few_shots_section>", r"\1", prompt, flags=re.DOTALL
                )
            else:
                # No shots: remove the entire block (header + placeholder) to avoid orphaned text
                prompt = re.sub(
                    r"<few_shots_section>.*?</few_shots_section>",
                    "",
                    self.instruction,
                    flags=re.DOTALL,
                )
                # Fallback for prompts that use {few_shots} without the convention
                prompt = prompt.replace("{few_shots}", "")
        else:
            prompt = self.instruction + ("\n\n" + few_shot_str if few_shot_str else "")
        return prompt.strip()

    def __str__(self):
        return self.construct_prompt()


def convert_prompts(prompts):
    """Converts promptolution prompt objects into custom prompt objects."""
    if not isinstance(prompts, list):
        return CustomPrompt(prompts.instruction, prompts.few_shots)

    converted = []
    for prompt in prompts:
        converted.append(CustomPrompt(prompt.instruction, prompt.few_shots))
    return converted


PromptLike = Union[str, CustomPrompt]


def _normalize_pool_mapping(
    mapping: Mapping[str, Sequence[PromptLike]]
) -> Dict[str, List[CustomPrompt]]:
    return {agent: [CustomPrompt(p) for p in prompts] for agent, prompts in mapping.items()}


@dataclass(frozen=True)
class AgentPromptSet:
    prompts: Dict[str, CustomPrompt]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, PromptLike]) -> "AgentPromptSet":
        return cls(prompts={agent: CustomPrompt(prompt) for agent, prompt in mapping.items()})

    def as_custom_dict(self) -> Dict[str, CustomPrompt]:
        return self.prompts

    def as_string_dict(self) -> Dict[str, str]:
        return {agent: str(prompt) for agent, prompt in self.prompts.items()}

    def to_string(self) -> str:
        """Serialize this prompt set to a JSON string."""
        return json.dumps(self.as_string_dict())

    @classmethod
    def from_string(cls, prompt_string: str) -> "AgentPromptSet":
        """Deserialize a prompt set from a JSON string."""
        prompt_dict = json.loads(prompt_string)
        return cls.from_mapping(prompt_dict)

    def items(self):
        return self.prompts.items()

    def __iter__(self):
        return iter(self.prompts)

    def __len__(self):
        return len(self.prompts)

    def construct_prompt(self) -> str:
        """Construct a string for this prompt set, concatenating agent prompts in sorted order."""
        return "\n\n".join(str(self.prompts[agent]) for agent in sorted(self.prompts.keys()))

    def __str__(self) -> str:
        return self.construct_prompt()


@dataclass
class AgentPromptBatch:
    prompt_sets: List[AgentPromptSet]

    @classmethod
    def from_iterable(
        cls, prompt_sets: Iterable[Union[AgentPromptSet, Mapping[str, PromptLike]]]
    ) -> "AgentPromptBatch":
        normalized: List[AgentPromptSet] = []
        for prompt_set in prompt_sets:
            if isinstance(prompt_set, AgentPromptSet):
                normalized.append(prompt_set)
            else:
                normalized.append(AgentPromptSet.from_mapping(prompt_set))
        return cls(prompt_sets=normalized)

    def get_prompt_hash_map(self):
        """
        Returns a dict mapping CustomPrompt(hash) -> prompt set, using a canonical hash function.
        Also returns a list of CustomPrompt hashes in order for consistent iteration.
        """
        hash_prompts = []
        hash_map = {}
        for aps in self.prompt_sets:
            prompt_set = aps.as_custom_dict()
            # Canonical hash: join agent:prompt for sorted agent names
            hash_str = "|".join(
                f"{agent}:{str(prompt)}" for agent, prompt in sorted(prompt_set.items())
            )
            hash_prompt = CustomPrompt(hash_str)
            hash_prompts.append(hash_prompt)
            hash_map[hash_prompt] = aps
        return hash_map, hash_prompts

    def get_agent_names(self) -> List[str]:
        """Return agent names from the first prompt set in the batch."""
        return list(self.prompt_sets[0].prompts.keys())

    def as_flattened_strings(self) -> List[str]:
        """Return all prompts as a flat list of strings for logging/callbacks."""
        flattened = []
        for aps in self.prompt_sets:
            for agent_name in sorted(aps.prompts.keys()):
                prompt_str = str(aps.prompts[agent_name])
                flattened.append(prompt_str)
        return flattened

    def as_custom_dicts(self) -> List[Dict[str, CustomPrompt]]:
        return [aps.as_custom_dict() for aps in self.prompt_sets]

    def as_string_dicts(self) -> List[Dict[str, str]]:
        return [aps.as_string_dict() for aps in self.prompt_sets]

    def __iter__(self):
        return iter(self.prompt_sets)

    def __len__(self):
        return len(self.prompt_sets)


@dataclass
class AgentPromptPool:
    pool: Dict[str, List[CustomPrompt]]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[PromptLike]]) -> "AgentPromptPool":
        return cls(pool=_normalize_pool_mapping(mapping))

    @classmethod
    def ensure(
        cls, pool: Union["AgentPromptPool", Mapping[str, Sequence[PromptLike]]]
    ) -> "AgentPromptPool":
        if isinstance(pool, AgentPromptPool):
            return pool
        return cls.from_mapping(pool)

    def with_overrides(
        self, overrides: Union["AgentPromptPool", Mapping[str, Sequence[PromptLike]]]
    ) -> "AgentPromptPool":
        override_pool = (
            overrides
            if isinstance(overrides, AgentPromptPool)
            else AgentPromptPool.from_mapping(overrides)
        )
        merged = dict(self.pool)
        for agent, prompts in override_pool.pool.items():
            merged[agent] = prompts
        return AgentPromptPool(pool=merged)

    def sample_set_as_pool(self) -> "AgentPromptPool":
        """Sample one random prompt per agent and return as a pool with single prompts."""
        result_pool = {}
        for agent, prompts in self.pool.items():
            if prompts:
                result_pool[agent] = [random.choice(prompts)]
        return AgentPromptPool(pool=result_pool)

    def with_fallbacks(
        self, required_agents: Iterable[str], fallback_pool: "AgentPromptPool"
    ) -> "AgentPromptPool":
        merged: Dict[str, List[CustomPrompt]] = {}
        for agent in required_agents:
            if agent in self.pool:
                merged[agent] = self.pool[agent]
            elif agent in fallback_pool.pool:
                merged[agent] = fallback_pool.pool[agent]
        return AgentPromptPool(pool=merged)

    def first_set(self) -> AgentPromptSet:
        return AgentPromptSet.from_mapping(
            {agent: prompts[0] for agent, prompts in self.pool.items()}
        )

    def sample_set(self) -> AgentPromptSet:
        """Randomly pick one prompt per agent from the pool."""
        return AgentPromptSet.from_mapping(
            {agent: random.choice(prompts) for agent, prompts in self.pool.items()}
        )

    def sample_batch(self, n: int) -> "AgentPromptBatch":
        """Return a batch of `n` randomly sampled prompt sets (with replacement)."""
        prompt_sets = [self.sample_set() for _ in range(n)]
        return AgentPromptBatch.from_iterable(prompt_sets)

    def as_string_pool(self) -> Dict[str, List[str]]:
        return {agent: [str(prompt) for prompt in prompts] for agent, prompts in self.pool.items()}

    def items(self):
        return self.pool.items()

    def __contains__(self, agent: str) -> bool:
        return agent in self.pool

    def __getitem__(self, agent: str) -> List[CustomPrompt]:
        return self.pool[agent]

    def get_agent_names(self) -> List[str]:
        """Return agent names in the pool."""
        return list(self.pool.keys())

    def get_agent_prompt_lists(self) -> List[List[CustomPrompt]]:
        """Return prompts per agent in the same order as `get_agent_names`."""
        return list(self.pool.values())


def to_prompt_batch(agent_prompt_sets) -> AgentPromptBatch:
    if isinstance(agent_prompt_sets, AgentPromptBatch):
        return agent_prompt_sets
    if isinstance(agent_prompt_sets, AgentPromptSet):
        return AgentPromptBatch(prompt_sets=[agent_prompt_sets])
    return AgentPromptBatch.from_iterable(agent_prompt_sets)


def to_prompt_pool(agent_prompt_pool) -> AgentPromptPool:
    return AgentPromptPool.ensure(agent_prompt_pool)
