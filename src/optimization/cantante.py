from typing import Dict

import pandas as pd
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.tasks.base_task import BaseTask

from src.attribution import ALL_ATTRIBUTERS, build_agent_var_mapping
from src.attribution.base import BaseAttributer
from src.candidate_selector import ALL_SELECTORS, BaseCandidateSelector
from src.experiment.utils import get_logger
from src.mas import MASPredictor
from src.meta_prompts import ATTRIBUTION_PROMPT_ALTERNATIVE, CROSSOVER_PROMPT, MUTATION_PROMPT
from src.optimization.base_mas_optimizer import BaseMASOptimizer
from src.optimization.broker_agent_optimizers import broker_agent_optimizer_step
from src.optimization.dspy_mas_optimizer import Gepa, Mipro
from src.optimization.local_optimizer import CustomCAPO, CustomEvoPrompt
from src.optimization.proxy_task import AttributionProxyTask
from src.prompt_structures import AgentPromptPool, AgentPromptSet

ALL_NODELEVEL_OPTIMIZER = {
    "CAPO": CustomCAPO,
    "EvoPrompt": CustomEvoPrompt,
}

logger = get_logger(__name__)


def build_mas_optimizer(
    config,
    predictor,
    task,
    init_agent_prompt_pool,
    meta_llm,
    callbacks=None,
):
    experiment_cfg = config["experiment"]
    optimizer_choice = experiment_cfg["optimizer"]
    optimizer_kwargs = dict(config.get("optimizer_kwargs", {}))

    if optimizer_choice == "gepa":
        return Gepa(
            predictor=predictor,
            task=task,
            init_agent_prompt_pool=init_agent_prompt_pool,
            meta_llm=meta_llm,
            callbacks=callbacks,
            **optimizer_kwargs,
        )

    if optimizer_choice == "mipro":
        return Mipro(
            predictor=predictor,
            task=task,
            teacher=meta_llm,
            init_agent_prompt_pool=init_agent_prompt_pool,
            callbacks=callbacks,
            **optimizer_kwargs,
        )

    selector = ALL_SELECTORS[experiment_cfg["selector"]]()
    agent_var_mapping = build_agent_var_mapping(predictor)

    attributer_kwargs = config.get("attributer_kwargs", {})
    # Ablation flag: swap in the alternative attribution meta-prompt (defined in meta_prompts.py)
    use_alt_prompt = attributer_kwargs.pop("use_alternative_attribution_prompt", False)
    attributer = ALL_ATTRIBUTERS[experiment_cfg["attributer"]](
        llm=meta_llm,
        agent_var_mapping=agent_var_mapping,
        seed=experiment_cfg["seed"],
        max_tokens_per_field=experiment_cfg["max_tokens_per_field"],
        prompt_template=ATTRIBUTION_PROMPT_ALTERNATIVE if use_alt_prompt else None,
        **attributer_kwargs,
    )

    optimizer_cls = ALL_NODELEVEL_OPTIMIZER[experiment_cfg["node_optimizer"]]

    return Cantante(
        predictor=predictor,
        task=task,
        selector=selector,
        attributer=attributer,
        init_agent_prompt_pool=init_agent_prompt_pool,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=dict(meta_llm=meta_llm, **optimizer_kwargs),
        n_fs=experiment_cfg.get("n_fs", 0),
        callbacks=callbacks,
    )


def render_meta_prompt(templates, task_description, input_vars, output_vars):
    rendered = []
    for template in templates:
        prompt = template.replace("<task_desc>", task_description)
        prompt = prompt.replace("<input_vars>", ", ".join(input_vars))
        prompt = prompt.replace("<output_vars>", ", ".join(output_vars))
        prompt = prompt.strip()
        rendered.append(prompt)
    return rendered


def extract_prompts_dict(obj_dict):
    """Given a dict of objects with a .prompts attribute, return {name: obj.prompts}."""
    return {name: obj.prompts for name, obj in obj_dict.items()}


class Cantante(BaseMASOptimizer):
    def __init__(
        self,
        predictor: MASPredictor,
        task: BaseTask,
        selector: BaseCandidateSelector,
        attributer: BaseAttributer,
        init_agent_prompt_pool: AgentPromptPool,
        optimizer_cls,
        optimizer_kwargs=None,
        n_fs: int = 0,
        callbacks=None,
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
        self.selector = selector
        self.attributer = attributer
        self.n_fs = n_fs
        self.init_agent_prompt_pool = init_agent_prompt_pool
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.proxy_tasks: Dict[str, AttributionProxyTask] = {}
        self.agent_optimizers: Dict[str, BaseOptimizer] = {}

        self.callbacks = callbacks or []

        self._global_step = 0
        self.prompts = []  # Track agent prompt sets evaluated each step
        self.scores = []  # Track corresponding system scores

    def _pre_optimization_loop(self):
        self._setup()
        for optim in self.agent_optimizers.values():
            optim._pre_optimization_loop()

    def _setup(self):
        df_fs = self._sample_fs()
        for agent_name, agent in self.predictor.agents.items():
            proxy_task = AttributionProxyTask(
                super_task=self.task, task_description=agent.task_description
            )
            mutation_prompt, crossover_prompt = render_meta_prompt(
                [MUTATION_PROMPT, CROSSOVER_PROMPT],
                agent.task_description,
                agent.input_vars,
                agent.output_vars,
            )

            agent_optimizer = self.optimizer_cls(
                initial_prompts=self.init_agent_prompt_pool[agent_name],
                task=proxy_task,
                predictor=agent,
                mutation_template=mutation_prompt,
                crossover_template=crossover_prompt,
                df_few_shots=df_fs[agent_name],
                agent_name=agent_name,
                **self.optimizer_kwargs,
            )

            self.proxy_tasks[agent_name] = proxy_task
            self.agent_optimizers[agent_name] = agent_optimizer

    def _sample_fs(self):
        """Sample few-shot examples for each node-level optimizer to use during prompt evolution."""
        few_shot_dataframes = {}
        # pop from training task
        df = self.task.df.sample(n=self.n_fs)
        # run self.n_fs steps of predictions to generate few-shot examples
        few_shot_batch = self.init_agent_prompt_pool.sample_batch(self.n_fs)

        _, states = self.task._evaluate(
            few_shot_batch,
            df[self.task.x_column],
            df[self.task.y_column],
            self.predictor,
        )
        # for each agent, extract their input/output variables to create few-shot records
        for agent_name, agent in self.predictor.agents.items():
            # IMPORTANT: for promptolution reasons the tag <few_shots> is dangerous!
            # per agent, extract input and output variables from each state
            few_shot_records = []
            for state in states:
                # remove fewshots in prompt for few shot creation
                input_vars = [
                    var for var in agent.input_vars if var in state and var != "few_shots"
                ]
                output_vars = [
                    var for var in agent.output_vars if var in state and var != "few_shots"
                ]
                input_str = "\n".join([f"{var}: `{state.get(var)}`" for var in input_vars])
                output_str = "\n".join([f"<{var}>{state.get(var)}</{var}>" for var in output_vars])

                record = dict(input=input_str, target=output_str)
                few_shot_records.append(record)
            few_shot_dataframes[agent_name] = pd.DataFrame(few_shot_records)
        return few_shot_dataframes

    def state_dict(self) -> dict:
        return {
            "_global_step": self._global_step,
            "prompts": [p.to_string() for p in self.prompts],
            "scores": list(self.scores),
            "agent_optimizers": {
                name: optim.state_dict() for name, optim in self.agent_optimizers.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self._global_step = state["_global_step"]
        self.prompts = [AgentPromptSet.from_string(s) for s in state["prompts"]]
        self.scores = list(state["scores"])
        for name, agent_state in state["agent_optimizers"].items():
            if name in self.agent_optimizers:
                self.agent_optimizers[name].load_state_dict(agent_state)

    def step(self):
        """Execute one optimization step across all agent optimizers."""
        logger.warning(f"Starting optimization step {self._global_step}...")

        # clear prompts and scores for callbacks; filled by select_evaluate_attribute
        self.prompts = []
        self.scores = []

        return broker_agent_optimizer_step(self)

    def select_evaluate_attribute(self, pending_by_agent):
        """
        Runs the core optimization cycle:
        - selects candidates
        - evaluates them
        - attributes scores
        - logs results
        """

        # --- 1. Select ---
        agent_prompt_batch = self.selector.select_candidates(
            pending_by_agent=pending_by_agent,
            agent_optimizers=self.agent_optimizers,
        )

        # --- 2. Evaluate ---
        system_scores, inter = self.task.evaluate(
            agent_prompt_batch,
            self.predictor,
        )
        self.prompts.extend(agent_prompt_batch.prompt_sets)
        self.scores.extend(system_scores.mean(-1).tolist())

        # --- 3. Attribute ---
        attributed_scores = self.attributer.attribute(agent_prompt_batch, system_scores, inter)

        return attributed_scores

    def get_best_candidate(self):
        best_candidate = {}
        for agent_name, optim in self.agent_optimizers.items():
            best_candidate[agent_name] = optim.get_best_candidate()

        best_candidate = AgentPromptSet.from_mapping(best_candidate)
        return best_candidate
