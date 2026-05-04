from promptolution.optimizers import CAPO, EvoPromptGA

from src.prompt_structures import CustomPrompt, convert_prompts


class CustomCAPO(CAPO):
    """
    CAPO wrapper that records the genealogy of generated prompts.

    The unmodified CAPO optimizer performs mutation and crossover steps
    without keeping lineage information. This wrapper reimplements
    `_crossover` and hooks into `_mutate` to track how new prompts originate.
    """

    def __init__(
        self,
        agent_name=None,
        initial_prompts=None,
        *args,
        **kwargs,
    ):
        self.agent_name = agent_name or "unknown"

        super().__init__(
            initial_prompts=initial_prompts, create_fs_reasoning=False, *args, **kwargs
        )
        self.prompts = convert_prompts(self.prompts)
        self.population_size = len(self.prompts)

    def _initialize_population(self, init_prompts):
        prompts = super()._initialize_population(init_prompts)
        prompts = convert_prompts(prompts)
        return prompts

    def _crossover(self, parents):
        offsprings = super()._crossover(parents)
        offsprings = convert_prompts(offsprings)
        return offsprings

    def _mutate(self, offsprings):
        mutants = super()._mutate(offsprings)
        mutants = convert_prompts(mutants)
        return mutants

    def get_best_candidate(self):
        best_idx = self.scores.index(max(self.scores))
        return self.prompts[best_idx]

    def state_dict(self) -> dict:
        """Serialize the evolving CAPO state (population + scores + length normalizer)."""
        return {
            "prompts": [
                {"instruction": p.instruction, "few_shots": list(p.few_shots)} for p in self.prompts
            ],
            "scores": list(self.scores),
            "max_prompt_length": self.max_prompt_length,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore population, scores, and length normalizer from a checkpoint."""
        self.prompts = [CustomPrompt(p["instruction"], p["few_shots"]) for p in state["prompts"]]
        self.scores = list(state["scores"])
        self.max_prompt_length = state["max_prompt_length"]


class CustomEvoPrompt(EvoPromptGA):
    def __init__(
        self,
        task,
        predictor,
        meta_llm,
        agent_name=None,
        initial_prompts=None,
        *args,
        **kwargs,
    ):
        self.agent_name = agent_name or "unknown"

        super().__init__(predictor, task, meta_llm, initial_prompts=initial_prompts)
        self.prompts = convert_prompts(self.prompts)
        self.population_size = len(self.prompts)

    def _pre_optimization_loop(self) -> None:
        self.prompts = convert_prompts(self.prompts)
        self.scores = [0.0] * len(self.prompts)

    def _crossover(self, prompts, scores):
        offspring = super()._crossover(prompts, scores)
        offspring = convert_prompts(offspring)
        return offspring

    def get_best_candidate(self):
        best_idx = self.scores.index(max(self.scores))
        return self.prompts[best_idx]

    def state_dict(self) -> dict:
        """Serialize the evolving CAPO state (population + scores + length normalizer)."""
        return {
            "prompts": [{"instruction": p.instruction} for p in self.prompts],
            "scores": list(self.scores),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore population, scores, and length normalizer from a checkpoint."""
        self.prompts = [
            CustomPrompt(p["instruction"], []) for p in state["prompts"]
        ]  # Initialize with empty few-shot examples
        self.scores = list(state["scores"])
