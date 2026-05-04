import os
import shutil
from dataclasses import dataclass

import numpy as np
from dspy import Example
from dspy.teleprompt import MIPROv2
from gepa.api import optimize as gepa_optimize

from src.experiment.utils import generate_random_hash, get_logger
from src.optimization.base_mas_optimizer import BaseMASOptimizer
from src.optimization.dspy_mas_wrapper import (
    DSPYMASPredictorWrapper,
    PromptolutionDSPYLM,
    build_default_metric,
)
from src.prompt_structures import AgentPromptBatch, AgentPromptPool, AgentPromptSet

logger = get_logger(__name__)


@dataclass
class GEPAEvalOut:
    outputs: list[str]
    scores: list[float]
    trajectories: list[dict] | None
    objective_scores: None = None  # required by GEPA, even if not used


class GEPAAdapter:
    def __init__(self, task, predictor):
        self.task = task
        self.predictor = predictor
        self.propose_new_texts = None

    def evaluate(
        self, batch: list, candidate: dict[str, str], capture_traces: bool = False
    ) -> GEPAEvalOut:
        pairs = list(batch)
        prompt_batch = AgentPromptBatch.from_iterable([candidate])

        raw_scores, states = self.task.evaluate(prompt_batch, self.predictor)
        scores = raw_scores[0]
        outputs = states[0]

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "trace_id": idx,
                    "input": pairs[idx][0],
                    "target": pairs[idx][1],
                    "score": scores[idx],
                    "prediction": outputs[idx],
                    "state": states[0][idx],
                    "candidate": candidate,
                }
                for idx in range(len(pairs))
            ]

        return GEPAEvalOut(outputs=list(outputs), scores=list(scores), trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update=None):
        records = [
            {
                "Inputs": {"input": trace["input"]},
                "Generated Outputs": {
                    "prediction": trace["prediction"],
                    "state": trace["state"],
                },
                "Feedback": f"Target: {trace['target']}",
                "score": trace["score"],
                "trace_id": trace["trace_id"],
            }
            for trace in eval_batch.trajectories
        ]

        dataset = {agent_name: records for agent_name in candidate.keys()}
        if components_to_update:
            dataset = {k: v for k, v in dataset.items() if k in components_to_update}
        return dataset


class BaseDSPYMASOptimizer(BaseMASOptimizer):
    def __init__(
        self,
        predictor,
        task,
        init_agent_prompt_pool,
        callbacks=None,
        **backend_kwargs,
    ):
        seed_prompt_dict = (
            AgentPromptPool.ensure(init_agent_prompt_pool).sample_set().as_string_dict()
        )
        seed_prompts = list(seed_prompt_dict.values())
        super().__init__(
            predictor=predictor,
            task=task,
            initial_prompts=seed_prompts,
            callbacks=callbacks,
        )
        self.predictor = predictor
        self.task = task
        self.init_agent_prompt_pool = init_agent_prompt_pool
        self.run_dir = f"./runs/dspy_mas_{generate_random_hash()}"
        self.backend_kwargs = backend_kwargs

        self.seed_candidate: dict[str, str] = {}
        self.trainset = []
        self.prompts: list[dict[str, str]] = []
        self.scores: list[float] = []

    def _prepare_run_dir(self) -> None:
        if os.path.isdir(self.run_dir):
            shutil.rmtree(self.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)

    def _extract_seed_candidate(self) -> dict[str, str]:
        pool = AgentPromptPool.ensure(self.init_agent_prompt_pool)
        return pool.sample_set().as_string_dict()

    def _eval_candidate(self, candidate: dict[str, str]) -> float:
        batch = AgentPromptBatch.from_iterable([candidate])
        system_scores, _ = self.task.evaluate(batch, self.predictor)
        return system_scores.mean()

    def get_best_candidate(self):
        best_idx = self.scores.index(max(self.scores))
        return self.prompts[best_idx]

    def step(self):
        return self._step()

    def _step(self):
        raise NotImplementedError

    def _pre_optimization_loop(self):
        self._prepare_run_dir()
        self.seed_candidate = self._extract_seed_candidate()
        self.trainset = list(zip(self.task.xs, self.task.ys))


class Gepa(BaseDSPYMASOptimizer):
    def __init__(
        self,
        predictor,
        task,
        init_agent_prompt_pool,
        meta_llm,
        step_budget: int = 150,
        callbacks=None,
        **backend_kwargs,
    ):
        self.meta_llm = PromptolutionDSPYLM(meta_llm)
        self.reflection_lm = lambda prompt: str(
            self.meta_llm(prompt=prompt)[0]
        )  # patch required due to dspy's Interface
        super().__init__(
            predictor=predictor,
            task=task,
            init_agent_prompt_pool=init_agent_prompt_pool,
            callbacks=callbacks,
            **backend_kwargs,
        )
        self.cum_budget = 0
        self.step_budget = step_budget
        self.adapter = GEPAAdapter(task=task, predictor=predictor)

    def _step(self):
        self.cum_budget += self.step_budget
        result = gepa_optimize(
            run_dir=self.run_dir,
            seed_candidate=self.seed_candidate,
            adapter=self.adapter,  # type: ignore
            trainset=self.trainset,
            reflection_lm=self.reflection_lm,
            max_metric_calls=self.cum_budget,
        )
        self.prompts = [AgentPromptSet.from_mapping(candidate) for candidate in result.candidates]
        self.scores = [float(score) for score in result.val_aggregate_scores]

        best_idx = int(np.argmax(self.scores))
        self.seed_candidate = self.prompts[best_idx]

        return self.prompts


class Mipro(BaseDSPYMASOptimizer):
    def __init__(
        self,
        predictor,
        task,
        init_agent_prompt_pool,
        teacher=None,
        run_dir: str | None = None,
        step_budget: int = 10,
        num_candidates: int = 5,
        callbacks=None,
        **backend_kwargs,
    ):
        self.teacher = teacher
        super().__init__(
            predictor=predictor,
            task=task,
            init_agent_prompt_pool=init_agent_prompt_pool,
            run_dir=run_dir,
            callbacks=callbacks,
            **backend_kwargs,
        )
        self.dspy_lm = PromptolutionDSPYLM(self.teacher)
        self.step_budget = step_budget
        self.optimizer = MIPROv2(
            metric=build_default_metric(self.task, self.predictor),
            prompt_model=self.dspy_lm,
            task_model=self.dspy_lm,
            auto=None,
            num_candidates=num_candidates,
        )
        self.dspy_trainset = []

    def _pre_optimization_loop(self):
        super()._pre_optimization_loop()
        self.dspy_trainset = [
            Example(query=x, answer=y).with_inputs("query") for x, y in self.trainset
        ]

    def _step(self):
        student = DSPYMASPredictorWrapper(
            mas_predictor=self.predictor,
            seed_candidate=dict(self.seed_candidate),
        )
        compiled = self.optimizer.compile(
            student=student,
            trainset=self.dspy_trainset,
            num_trials=self.step_budget,
        )
        best = compiled.candidate_programs
        self.prompts = [AgentPromptSet.from_mapping(b["program"].to_candidate()) for b in best]
        self.scores = [b["score"] for b in compiled.candidate_programs]

        self.seed_candidate = best[0]["program"].to_candidate()

        return self.prompts
