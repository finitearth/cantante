import copy
import inspect

from langchain_core.tools import StructuredTool


def tool_method(func):
    func.__is_adapter_tool__ = True
    return func


class BaseToolsAdapter:
    def copy(self):
        return copy.copy(self)

    def set_env(self, env):
        self.env = env

        return self

    def _iter_tool_functions(self):
        for _, method in inspect.getmembers(self, predicate=callable):
            if getattr(method, "__is_adapter_tool__", False):
                yield method

    def get_tools(self):
        return [StructuredTool.from_function(func) for func in self._iter_tool_functions()]

    def get_tool_descriptions(self, tool_names=None) -> str:
        tools = self.get_tools()
        if tool_names is not None:
            tools = [tool for tool in tools if tool.name in tool_names]

        lines = []
        for tool in tools:
            schema = tool.args_schema.model_json_schema()

            properties = schema["properties"]
            arg_names = properties.keys()
            arg_part = f" Arguments: `{', '.join(arg_names)}`."
            lines.append(f"- {tool.name}: {tool.description}{arg_part}")

        return "\n".join(lines)


def get_tools(dataset_name: str):
    if dataset_name == "gsm8k":
        from src.agent_tools.gsm8k import GSM8KTools

        return GSM8KTools()
    elif dataset_name == "hotpotqa":
        from src.agent_tools.hotpotqa import HotpotQATools

        return HotpotQATools()
    elif dataset_name == "mbpp":
        from src.agent_tools.mbpp import MBPPTools

        return MBPPTools()
    else:
        raise ValueError(f"No tools adapter defined for dataset {dataset_name}")
