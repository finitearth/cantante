import subprocess
import sys
from typing import TYPE_CHECKING

from promptolution.tasks import ClassificationTask

from src.agent_tools.mbpp import _check_safety, _resource_limits
from src.experiment.utils import get_logger
from src.tasks.base import BaseMASTask

if TYPE_CHECKING:
    from src.mas import MASPredictor

logger = get_logger(__name__)


class MBPPTask(ClassificationTask, BaseMASTask):
    def __init__(self, *args, n_visible_tests: int = 1, eval_timeout_sec: int = 30, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_visible_tests = n_visible_tests
        self.eval_timeout_sec = eval_timeout_sec

    def _evaluate(self, agent_prompt_batch, xs, ys, mas_predictor: "MASPredictor"):
        tool_adapters = []
        all_test_lists = []
        for x in xs:
            test_list = list(self.df.loc[self.df[self.x_column] == x, "test_list"].iloc[0])
            all_test_lists.append(test_list)
            visible = test_list[: self.n_visible_tests]
            adapter = self.tools_adapter.copy().set_env({"code": "", "test_list": visible})
            tool_adapters.append(adapter)

        y_preds, states = mas_predictor.predict(xs, agent_prompt_batch, tool_adapters=tool_adapters)
        scores = []
        for test_list, y_pred, state, adapter in zip(
            all_test_lists, y_preds, states, tool_adapters
        ):
            y_pred = adapter.env.get("code") or y_pred
            state["prediction"] = y_pred
            code = y_pred.replace("```python", "").replace("```", "").strip()
            passed = 0
            if _check_safety(code) is not None:
                scores.append(0.0)
                continue
            for test in test_list:
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", code + "\n" + test],
                        timeout=self.eval_timeout_sec,
                        capture_output=True,
                        preexec_fn=_resource_limits,
                    )
                    if result.returncode == 0:
                        passed += 1
                except (subprocess.TimeoutExpired, Exception):
                    pass
            score = 1 if passed == len(test_list) else 0
            scores.append(score)

        return scores, states  # pyright: ignore[reportReturnType]
