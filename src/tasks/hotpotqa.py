import re
import string
from typing import TYPE_CHECKING

from promptolution.tasks import ClassificationTask

from src.tasks.base import BaseMASTask

if TYPE_CHECKING:
    from src.mas import MASPredictor


class HotpotQATask(ClassificationTask, BaseMASTask):
    def _make_tools_adapter_for_query(self, query):
        env = {
            "question": str(query),
            "question_column": self.x_column,
            "task_df": self.df.copy(),
            "row": self.df[self.df[self.x_column] == query].iloc[0].to_dict(),
        }

        return self.tools_adapter.copy().set_env(env)

    def _evaluate(self, agent_prompt_batch, xs, ys, mas_predictor: "MASPredictor"):  # type: ignore
        tool_adapters = [self._make_tools_adapter_for_query(query) for query in xs]

        y_preds, states = mas_predictor.predict(xs, agent_prompt_batch, tool_adapters=tool_adapters)
        scores = self._score_pred(ys, y_preds)
        return scores, states  # pyright: ignore[reportReturnType]

    def _score_pred(self, y_trues, y_preds):
        return [
            1.0 if self._normalize_answer(y_true) == self._normalize_answer(y_pred) else 0.0
            for y_true, y_pred in zip(y_trues, y_preds)
        ]

    @staticmethod
    def _normalize_answer(text):
        if text is None:
            return ""

        value = str(text).lower()
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        value = "".join(ch for ch in value if ch not in string.punctuation)
        value = " ".join(value.split())
        return value
