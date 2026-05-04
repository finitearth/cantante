import json
import operator
import re
from types import SimpleNamespace
from typing import Annotated, Any, Dict, Optional
from uuid import uuid4

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from promptolution.predictors.base_predictor import BasePredictor
from pydantic import create_model
from tqdm.auto import tqdm

from src.experiment.utils import get_logger
from src.prompt_utils import constrain_prompt_length, count_tokens

SYSTEM_PROMPT = "You are a helpful Assistant!"

logger = get_logger(__name__)


def extract_placeholders(prompt):
    return set(re.findall(r"{(\w+)}", prompt))


def safe_format(
    template: str, values: Dict[str, str], max_tokens_per_field: Optional[int] = None
) -> str:
    # constrain length of values
    if max_tokens_per_field is not None:
        values = {k: constrain_prompt_length(v, max_tokens_per_field) for k, v in values.items()}

    pattern = re.compile(r"(?<!\{)\{([^{}]*)\}(?!\})")
    out = []
    last = 0

    try:
        for m in pattern.finditer(template):
            out.append(template[last : m.start()])
            inner = m.group(1)
            # Determine base identifier for presence check in `values`
            base = inner.split("!")[0].split(":")[0]
            base = re.split(r"[\.\[]", base)[0] if base else base

            # Escape if positional (empty or digits) or unknown named
            should_escape = (base == "") or base.isdigit() or (base not in values)
            if should_escape:
                out.append("{{" + inner + "}}")
            else:
                out.append(m.group(0))
            last = m.end()
        out.append(template[last:])

        safe_template = "".join(out)
        return safe_template.format(**values)
    except Exception as e:
        logger.error(f"Error in safe_format regex processing: {e}")
        return template


class Agent:
    def __init__(
        self,
        model,
        name,
        tools=None,
        tools_adapter=None,
        input_vars=None,
        output_vars=None,
        task_description=None,
        max_tool_retries=5,
        max_tokens=4096,
        max_tokens_per_field=512,
    ):
        self.model = model

        self.name = name
        self.input_vars = input_vars or []
        self.output_vars = output_vars or []
        self.task_description = task_description
        self.max_tool_retries = max_tool_retries
        self.max_tokens = max_tokens
        self.max_tokens_per_field = max_tokens_per_field

        self.tool_names = tools or []
        self.tools_adapter = tools_adapter
        self.tools = {}
        self._setup_tools()

        self.prompt = None

        # for promptolution compatibility
        self.llm = SimpleNamespace()
        self.llm.tokenizer = None
        self.llm.get_token_count = self._get_token_count

        self.token_count = {"input_tokens": 0, "output_tokens": 0}

        self.extraction_description = (
            "The output variables will be extracted from the LLM response"
            " by looking for tags like <var>...</var>."
        )

    def _render_prompt(self, state):
        prompt = state.agent_prompt_set[self.name]

        place_holders = extract_placeholders(prompt)

        values = {
            place_holder: getattr(state, place_holder, "UNKNOWN") for place_holder in place_holders
        }
        values = {k: (v if v is not None else "") for k, v in values.items()}
        values["few_shots"] = ""

        rendered_prompt = safe_format(prompt, values, self.max_tokens_per_field)
        prompt = constrain_prompt_length(rendered_prompt, self.max_tokens)

        return HumanMessage(prompt)

    def _extract_outputs(self, response):
        outputs = {}
        for var in self.output_vars:
            pattern = f"<{var}>(.*?)</{var}>"
            match = re.search(pattern, response.content, re.DOTALL)
            if match:
                outputs[var] = match.group(1).strip()
            else:
                outputs[var] = response.content
        return outputs

    def _invoke_model(self, messages):
        try:
            response = self.model.invoke(messages)
        except Exception as e:
            logger.error(f"LLM invocation failed; Last message: {messages[-1].content}\nError: {e}")
            raise e
        return response

    def call(self, state):
        messages = [SystemMessage(content=SYSTEM_PROMPT), self._render_prompt(state)]

        response = self._invoke_model(messages)
        action_type, payload = self._parse_structured_model_output(response.content)
        active_tools = self._get_tools_for_state(state)

        tool_records = []
        if action_type == "tool_call":
            messages, final_response, tool_records = self._resolve_tool_calls(
                messages, payload, active_tools
            )
        else:
            messages.append(response)
            final_response = response

        outputs = self._extract_outputs(final_response)
        self._update_token_count(messages, final_response)

        return dict(
            messages=messages,
            call_history=self._append_call_history(state, tool_records, outputs),
            **outputs,
        )

    def _append_call_history(self, state, tool_records, outputs):
        history = getattr(state, "call_history", None) or []
        loop_n = (
            sum(1 for e in history if e.get("type") == "agent_call" and e.get("agent") == self.name)
            + 1
        )
        return tool_records + [
            {"type": "agent_call", "agent": self.name, "loop": loop_n, "outputs": outputs}
        ]

    def _resolve_tool_calls(self, messages, payload, active_tools):
        tool_call_count = 0
        current_payload = payload
        response = None
        tool_records = []

        while tool_call_count < self.max_tool_retries:
            tool_name, tool_args = self._validate_tool_payload(current_payload)

            synthetic_tool_call_message, tool_call_id = self._build_synthetic_tool_call_message(
                tool_name=tool_name,
                tool_args=tool_args,
            )
            messages.append(synthetic_tool_call_message)

            if tool_name not in active_tools:
                msg = {
                    "name": tool_name,
                    "error": "Unknown tool",
                    "available_tools": sorted(active_tools.keys()),
                }
                tool_result_message = f"<tool_result>\n{msg}\n</tool_result>"
                tool_result = None
            else:
                tool_result = self._execute_tool_call(tool_name, tool_args, active_tools)
                tool_result_message = self._build_tool_result_message(
                    tool_name, tool_args, tool_result
                )

            tool_records.append(
                {
                    "type": "tool_call",
                    "agent": self.name,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                }
            )
            messages.append(ToolMessage(content=tool_result_message, tool_call_id=tool_call_id))

            response = self._invoke_model(messages)
            action_type, next_payload = self._parse_structured_model_output(response.content)

            current_payload = next_payload
            tool_call_count += 1

            if action_type == "plain_final":
                messages.append(response)
                return messages, response, tool_records
            if action_type == "invalid":
                logger.warning(
                    "Agent '%s': invalid tool_call output, stopping tool loop: %s",
                    self.name,
                    next_payload,
                )
                tool_records.append(
                    {"type": "tool_call_failed", "agent": self.name, "error": str(next_payload)}
                )
                messages.append(response)
                return messages, response, tool_records

        messages.append(response)
        return messages, response, tool_records

    def _build_synthetic_tool_call_message(self, tool_name: str, tool_args: dict):
        tool_call_id = f"call_{uuid4().hex}"
        tool_call = {
            "id": tool_call_id,
            "name": tool_name,
            "args": tool_args,
            "type": "tool_call",
        }
        return AIMessage(content="", tool_calls=[tool_call]), tool_call_id

    def _validate_tool_payload(self, payload):
        if not isinstance(payload, dict):
            logger.error("Agent '%s': parsed payload is not a dict", self.name)
            raise ValueError("Invalid parsed tool payload.")

        tool_name = payload.get("name")
        tool_args = payload.get("arguments")
        if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
            logger.error(
                "Agent '%s': invalid payload shape name_type=%s args_type=%s",
                self.name,
                type(tool_name).__name__,
                type(tool_args).__name__,
            )
            raise ValueError(
                "Invalid tool payload fields. Expected {'name': string, 'arguments': object}."
            )

        return tool_name, tool_args

    def _build_tool_result_message(self, tool_name: str, tool_args: dict, tool_result: str) -> str:
        result_text = constrain_prompt_length(tool_result, self.max_tokens)

        arg_keys = sorted(tool_args.keys())
        payload = {
            "name": tool_name,
            "argument_keys": arg_keys,
            "result": result_text,
        }
        return f"<tool_result>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_result>"

    def _update_token_count(self, messages, response):
        input_tokens = sum(count_tokens(str(m.content)) for m in messages[:-1])
        output_tokens = count_tokens(str(response.content))
        self.token_count["input_tokens"] += input_tokens
        self.token_count["output_tokens"] += output_tokens

    def _get_token_count(self):
        """Get token count from this agent's model."""
        return self.token_count

    def _execute_tool_call(self, tool_name, tool_args, active_tools):
        tool = active_tools[tool_name]
        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            logger.exception("Agent '%s': tool '%s' failed", self.name, tool_name)
            result = f"Tool '{tool_name}' failed with error: {e}"
        return result

    def _parse_structured_model_output(self, content: str):
        tool_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
        if not tool_match:
            return "plain_final", content

        payload_raw = tool_match.group(1).strip()
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError as e:
            logger.error("Agent '%s': tool_call JSON parse failed: %s", self.name, e)
            return "invalid", f"Invalid tool_call JSON: {e}"

        if not isinstance(payload, dict):
            logger.error("Agent '%s': tool_call payload is not a JSON object", self.name)
            return "invalid", "Invalid tool_call payload: expected a JSON object."

        tool_name = payload.get("name")
        tool_args = payload.get("arguments")
        if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
            logger.error("Agent '%s': tool_call payload missing valid name/arguments", self.name)
            return (
                "invalid",
                "Invalid tool_call payload: 'name' must be string and 'arguments' must be object.",
            )

        logger.info("Agent '%s': parsed tool_call for tool '%s'", self.name, tool_name)
        return "tool_call", {"name": tool_name, "arguments": tool_args}

    def _setup_tools(self):
        if self.tools_adapter is None:
            return

        tools = self.tools_adapter.get_tools()
        tools = [tool for tool in tools if tool.name in self.tool_names]
        self.tools = {tool.name: tool for tool in tools}

    def _get_tools_for_state(self, state):
        if state.tools_adapter is None:
            return {}
        tools = state.tools_adapter.get_tools()
        tools = [tool for tool in tools if tool.name in self.tool_names]
        return {tool.name: tool for tool in tools}


class ConditionalRouter:
    """Callable router that checks a boolean-ish state field and optional retry cap.

    Retry counts are tracked internally, keyed by query, so concurrent batch
    invocations do not interfere with each other.
    """

    def __init__(self, target_node, condition_field, max_retries=5, fallback_target="__end__"):
        self.target_node = target_node
        self.condition_field = condition_field
        self.max_retries = max_retries
        self.fallback_target = fallback_target
        self._counters: dict[str, int] = {}

    def __call__(self, state):
        condition_value = getattr(state, self.condition_field)
        key = str(state.query) + str(state.agent_prompt_set)
        n_retries = self._counters.get(key, 0)

        if self._process_bool(condition_value) and n_retries < self.max_retries:
            self._counters[key] = n_retries + 1
            return self.target_node

        self._counters.pop(key, None)
        return self.fallback_target

    def _process_bool(self, value):
        if isinstance(value, bool):
            return value

        value = str(value)
        value = value.strip().lower()
        if value in ["true", "y", "1", "yes"]:
            return True

        return False


class MASPredictor(BasePredictor):
    def __init__(
        self,
        setup_dict,
        model_kwargs,
        seed=42,
        tool_adapter=None,
        recursion_limit=100,
        max_tokens_per_field=1024,
        predict_use_tqdm=True,
    ):
        """
        Initializes the MASPredictor with model configuration.

        Args:
            model_kwargs (dict): Keyword arguments for initializing the chat model.
                Must contain keys: model_name, temperature, base_url, token, ...
        """
        self.tool_adapter = tool_adapter
        self.recursion_limit = recursion_limit
        self.predict_use_tqdm = predict_use_tqdm
        self._validate_setup(setup_dict)
        model = self._build_chat_model(model_kwargs=model_kwargs, seed=seed)
        self.agents = {}

        vars = set()
        for agent in setup_dict["agents"]:
            self.agents[agent["name"]] = Agent(
                name=agent["name"],
                tools=agent["tools"],
                tools_adapter=tool_adapter,
                input_vars=agent["input_vars"],
                output_vars=agent["output_vars"],
                model=model,
                task_description=agent["task_description"],
                max_tool_retries=agent["max_tool_calls"],
                max_tokens=model_kwargs["max_tokens"],
                max_tokens_per_field=max_tokens_per_field,
                # rate_limiter=self.rate_limiter,
            )

            vars.update(agent["input_vars"])
            vars.update(agent["output_vars"])

        # Create llm wrapper for token counting that aggregates across all agents
        self.llm = SimpleNamespace()
        self.llm.get_token_count = self._aggregate_token_count

        # Build state schema
        state_schema = {var: (Optional[str], None) for var in vars}
        state_schema["agent_prompt_set"] = (Optional[Dict[str, str]], None)
        state_schema["tools_adapter"] = (Any, None)
        state_schema["call_history"] = (Annotated[list, operator.add], [])

        self.state_schema = create_model("State", **state_schema)
        self.graph = StateGraph(self.state_schema)

        for agent_name, agent in self.agents.items():
            self.graph.add_node(
                agent_name,
                agent.call,
            )

        # Process edges: unconditional by default, conditional if condition_fix is present
        for edge in setup_dict["edges"]:
            if "condition_fix" in edge:
                # Ensure the condition field is defined in the state schema
                condition_field = edge["condition_fix"]
                assert condition_field in state_schema
                # Conditional edge: only route if condition field is true
                self.graph.add_conditional_edges(
                    edge["from"],
                    ConditionalRouter(
                        edge["to"],
                        condition_field,
                        max_retries=edge["max_retries"],
                        fallback_target=edge["fallback_target"],
                    ),
                )
            else:
                # Unconditional edge
                self.graph.add_edge(
                    start_key=edge["from"],
                    end_key=edge["to"],
                )

        self.graph = self.graph.compile()

    def _build_chat_model(self, model_kwargs, seed):
        assert (
            model_kwargs["model_provider"] == "openai"
        ), "MASPredictor currently expects 'openai' as the model provider."

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=model_kwargs["requests_per_second"],
            max_bucket_size=model_kwargs["max_bucket_size"],
        )
        return ChatOpenAI(
            model=model_kwargs["model"],
            base_url=model_kwargs["base_url"],
            api_key=model_kwargs["api_key"],
            temperature=model_kwargs["temperature"],
            timeout=model_kwargs["timeout"],
            max_retries=model_kwargs["max_retries"],
            seed=seed,
            max_tokens=model_kwargs["max_tokens"],
            rate_limiter=rate_limiter,
            cache=False,
        )

    def _aggregate_token_count(self):
        """Aggregate token counts from all agents' models."""
        total_count = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for agent in self.agents.values():
            count = agent.llm.get_token_count()
            total_count["input_tokens"] += count["input_tokens"]
            total_count["output_tokens"] += count["output_tokens"]
        total_count["total_tokens"] = total_count["input_tokens"] + total_count["output_tokens"]
        return total_count

    def predict(self, queries, agent_prompt_batch, tool_adapters=None):  # type: ignore[override]
        agent_prompt_sets = agent_prompt_batch.as_string_dicts()
        assert len(agent_prompt_sets) == len(
            queries
        ), "Length of agent_prompt_batch must match number of queries"

        if tool_adapters is None:
            tool_adapters = [None] * len(queries)
        elif not isinstance(tool_adapters, list):
            tool_adapters = [tool_adapters] * len(queries)
        assert len(tool_adapters) == len(
            queries
        ), "Length of tool_adapters must match number of queries"

        states = [
            self.state_schema(
                query=query,
                agent_prompt_set=agent_prompt_set,
                tools_adapter=tool_adapter,
            ).model_dump()
            for agent_prompt_set, query, tool_adapter in zip(
                agent_prompt_sets, queries, tool_adapters
            )
        ]

        out_states = [None] * len(states)
        batch_iterator = self.graph.batch_as_completed(
            states,
            config={"recursion_limit": self.recursion_limit},
        )
        if self.predict_use_tqdm:
            batch_iterator = tqdm(batch_iterator, total=len(states), desc="MAS predict")

        for idx, res in batch_iterator:
            out_states[idx] = res

        preds = [res["prediction"] for res in out_states]
        preds = [p if p is not None else "" for p in preds]
        return preds, out_states

    def _validate_setup(self, setup_dict):
        # ensure the following:
        # 1. at least one agent is accepting "query" as input
        # 2. at least one agent is producing "prediction" as output
        # 3. each agent has non-empty input_vars and output_vars
        # 4. every conditional edge has required fields: max_retries, fallback_target
        input_vars = set()
        output_vars = set()
        for agent in setup_dict["agents"]:
            input_vars.update(agent["input_vars"])
            output_vars.update(agent["output_vars"])

        if "query" not in input_vars:
            raise ValueError("No agent accepts 'query' as input")
        if "prediction" not in output_vars:
            raise ValueError("No agent produces 'prediction' as output")

        required_conditional_fields = {"max_retries", "fallback_target"}
        for edge in setup_dict.get("edges", []):
            if "condition_fix" not in edge:
                continue
            missing = required_conditional_fields - edge.keys()
            if missing:
                raise ValueError(
                    f"Conditional edge from '{edge.get('from')}' is missing required "
                    f"field(s): {sorted(missing)}"
                )

    def _extract_preds(self, preds):
        raise NotImplementedError
