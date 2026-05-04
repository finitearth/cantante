from dataclasses import dataclass
from threading import Event, Lock
from typing import List, Optional

import numpy as np
from promptolution.tasks.base_task import BaseTask, EvalResult, EvalStrategy


@dataclass
class _Pending:
    prompts: List[str]
    done: Event
    eval_strategy: Optional[EvalStrategy] = None
    scores: Optional[List[List[float]]] = None
    error: Optional[Exception] = None


class AttributionProxyTask(BaseTask):
    def __init__(
        self, super_task, eval_timeout_s: float = 360.0, task_description=None, *args, **kwargs
    ):
        super().__init__(
            df=super_task.df,
            x_column=super_task.x_column,
            y_column=super_task.y_column,
            n_subsamples=super_task.n_subsamples,
            eval_strategy=super_task.eval_strategy,
            task_description=task_description,
            *args,
            **kwargs,
        )
        self._eval_timeout_s = eval_timeout_s
        self._pending: Optional[_Pending] = None
        self._lock = Lock()

        # prompt -> per-datapoint scores for this proxy task
        self.proxy_eval_cache: dict[str, List[float]] = {}

    def evaluate(
        self,
        prompts: List[str],
        predictor=None,
        eval_strategy=None,
        *args,
        **kwargs,
    ) -> EvalResult:
        if eval_strategy == "evaluated":
            scores = [self.proxy_eval_cache[p] for p in prompts]

        else:
            with self._lock:
                if self._pending is not None:
                    raise RuntimeError("Concurrent evaluate() on the same proxy")
                pending = _Pending(prompts=prompts, done=Event())
                self._pending = pending  # broker will pop this

            pending.done.wait()

            if pending.error is not None:
                raise pending.error

            if pending.scores is None:
                raise RuntimeError("Broker finished but did not set scores")
            scores = pending.scores

            # cache the scores for future "evaluated" calls
            for p, s in zip(prompts, scores):
                self.proxy_eval_cache[p] = s

        assert len(scores) == len(prompts), f"Expected {len(prompts)} scores but got {len(scores)}"

        scores_array = np.asarray(scores, dtype=float)

        agg_scores = np.nanmean(scores_array, axis=1)

        n_prompts = len(prompts)
        n_datapoints = scores_array.shape[1]
        prompt_token_lengths = np.asarray([len(str(p).split()) for p in prompts], dtype=float)
        input_tokens = np.repeat(prompt_token_lengths[:, None], n_datapoints, axis=1)
        output_tokens = np.zeros((n_prompts, n_datapoints), dtype=float)
        sequences = np.full((n_prompts, n_datapoints), "", dtype=object)

        return EvalResult(
            scores=scores_array,
            agg_scores=agg_scores,
            sequences=sequences,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agg_input_tokens=np.nanmean(input_tokens, axis=1),
            agg_output_tokens=np.nanmean(output_tokens, axis=1),
        )

    # called by broker to poll
    def try_pop_pending(self) -> Optional[_Pending]:
        with self._lock:
            p = self._pending
            self._pending = None
            return p

    def cancel_pending(self):
        with self._lock:
            if self._pending is not None:
                self._pending.done.set()

    def _evaluate(self, *_, **__) -> np.ndarray:
        raise NotImplementedError("AttributionProxyTask._evaluate is not used")
