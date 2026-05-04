from typing import TYPE_CHECKING

from promptolution.tasks import ClassificationTask

from src.tasks.base import BaseMASTask

if TYPE_CHECKING:
    from src.mas import MASPredictor


class GSM8KTask(ClassificationTask, BaseMASTask):
    def _evaluate(self, agent_prompt_batch, xs, ys, mas_predictor: "MASPredictor"):  # type: ignore
        y_preds, states = mas_predictor.predict(xs, agent_prompt_batch)
        # Convert metric scores to +1 for correct, 0 for incorrect
        scores = self._score_pred(ys, y_preds)
        return scores, states  # pyright: ignore[reportReturnType]

    def _score_pred(self, y_trues, y_preds):
        # Simple exact match scoring for GSM8K
        scores = [
            1.0 if self.metric([y_true], [y_pred]) > 0.5 else 0.0
            for y_true, y_pred in zip(y_trues, y_preds)
        ]

        return scores
