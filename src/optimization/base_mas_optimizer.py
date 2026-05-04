import json
import os

from promptolution.optimizers.base_optimizer import BaseOptimizer

from src.prompt_structures import AgentPromptSet


class BaseMASOptimizer(BaseOptimizer):
    def __init__(self, predictor, task, *args, **kwargs):
        super().__init__(predictor=predictor, task=task, *args, **kwargs)
        self._global_step = 0

    def step(self):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    def optimize(self, n_steps: int, skip_pre_loop: bool = False) -> list:
        if not skip_pre_loop:
            self._pre_optimization_loop()

        for _ in range(n_steps):
            self._global_step += 1
            self.step()

            continue_optimization = self._on_step_end()
            if not continue_optimization:
                break

        self._on_train_end()
        return self.prompts

    def _pre_optimization_loop(self):
        return

    def state_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the evolving optimizer state."""
        prompts_serialized = []
        for p in self.prompts:
            if isinstance(p, AgentPromptSet):
                prompts_serialized.append(p.to_string())
            else:
                # Fallback: try to convert dict-like to JSON
                prompts_serialized.append(json.dumps(p) if not isinstance(p, str) else p)

        return {
            "_global_step": self._global_step,
            "prompts": prompts_serialized,
            "scores": list(self.scores),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore evolving state from a snapshot produced by state_dict()."""
        self._global_step = state["_global_step"]
        self.prompts = [AgentPromptSet.from_string(s) for s in state.get("prompts", [])]
        self.scores = list(state.get("scores", []))

    def save(self, path: str) -> None:
        """Persist the current state dict as a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)
