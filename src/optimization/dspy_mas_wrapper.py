import copy
from types import SimpleNamespace
from typing import Any

import dspy
import numpy as np
from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from src.prompt_structures import AgentPromptBatch


class PromptolutionDSPYLM(dspy.LM):
    def __init__(self, llm, model_name: str = "promptolution_lm"):
        super().__init__(model=model_name)
        self.promptolution_llm = llm

    def forward(self, prompt=None, messages=None, **kwargs):
        if prompt is None:
            system_prompt = messages[0]["content"]
            prompt = messages[1]["content"]
        else:
            system_prompt = DEFAULT_SYS_PROMPT
        response = self.promptolution_llm.get_response(
            system_prompts=[system_prompt], prompts=[prompt]
        )[0]

        choice = SimpleNamespace(message=SimpleNamespace(content=str(response)))
        return SimpleNamespace(
            choices=[choice],
            usage={},
            model=self.model,
            _hidden_params={},
        )

    def copy(self, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def deepcopy(self):
        return self

    def reset_copy(self):
        return self


class PromptMutationProxy:
    class Signature:
        def __init__(self, instructions: str):
            self.instructions = str(instructions)

        def with_instructions(self, instructions: str):
            return PromptMutationProxy.Signature(instructions=instructions)

    def __init__(
        self,
        instructions: str,
        demos: list[Any] | None = None,
    ):
        self.signature = PromptMutationProxy.Signature(instructions=instructions)
        self.demos = list(demos or [])


class DSPYMASPredictorWrapper(dspy.Module):
    def __init__(
        self,
        mas_predictor,
        seed_candidate: dict[str, str],
    ):
        super().__init__()
        self.mas_predictor = mas_predictor
        self.agent_order = list(self.mas_predictor.agents.keys())
        self._predictors_by_agent: dict[str, PromptMutationProxy] = {}

        for agent_name in self.agent_order:
            seed_prompt = str(seed_candidate.get(agent_name, ""))
            self._predictors_by_agent[agent_name] = PromptMutationProxy(instructions=seed_prompt)

    def predictors(self) -> list[Any]:
        return [self._predictors_by_agent[agent_name] for agent_name in self.agent_order]

    def to_candidate(self) -> dict[str, str]:
        candidate: dict[str, str] = {}
        for agent_name in self.agent_order:
            predictor = self._predictors_by_agent[agent_name]
            candidate[agent_name] = str(predictor.signature.instructions)
        return candidate

    def deepcopy(self):
        copied = DSPYMASPredictorWrapper(
            mas_predictor=self.mas_predictor,
            seed_candidate=self.to_candidate(),
        )
        for agent_name in self.agent_order:
            copied._predictors_by_agent[agent_name].demos = copy.deepcopy(
                self._predictors_by_agent[agent_name].demos
            )
        return copied

    def forward(self, query):
        candidate = self.to_candidate()
        return dspy.Prediction(
            prediction="", candidate=candidate
        )  # prediction will be performed inside task.evaluate, to allow for tool calls


def build_default_metric(task, mas_predictor):
    def _metric(example, pred, trace=None):
        y_true = example.answer
        y_pred = pred.prediction

        if y_pred != "":  # task_interface_scoring, just reuse prediction made before
            score = task._score_pred([y_true], [y_pred])[0]
            return float(score)

        # else, we need to compute the prediction first
        prompt_batch = AgentPromptBatch.from_iterable([pred.candidate])
        scores, _ = task.evaluate(prompt_batch, mas_predictor)
        return float(np.mean(scores[0]))

    return _metric
