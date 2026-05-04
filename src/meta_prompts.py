CROSSOVER_PROMPT = """You receive two prompt TEMPLATES for the following task:
`<task_desc>`

Merge them into one coherent template while preserving all input placeholders exactly as they appear (e.g., {query}), without renaming, removing, or inventing any. Any output tags (e.g., "<output>...</output>") from the parents must be persevered. 

The following input variables should remain present with the {place_holder} -format: `<input_vars>`
The following output variables should remain present with the <output></output> -format: `<output_vars>`

Combine the essential style and instructions of both parents into a clear, consistent final template.

Prompt 1: `<mother>`
Prompt 2: `<father>`

Return only the merged prompt:
<prompt>new prompt</prompt>
"""

MUTATION_PROMPT = """You receive a prompt TEMPLATE for the task following task:
`<task_desc>`

Rephrase it while preserving its meaning, keeping all input placeholders exactly as written (e.g., {query}) and maintaining the <output>...</output> blocks for output variables.

The following input variables should remain present with the {place_holder} -format: `<input_vars>`
The following output variables should remain present with the <output></output> -format: `<output_vars>`

Do not rename, remove, or add placeholders. You may change the order of the placeholders.

The goal is to vary the linguistic style slightly while keeping the structure functional. 

Prompt: 
`<instruction>`

Return only the rewritten prompt:
<prompt>new prompt</prompt>
"""

# ---------------- Attribution prompts ----------------

ATTRIBUTION_PROMPT = """# Task
You are an attribution agent for a multi-agent system.

You will receive multiple executions of the SAME query.
Each execution uses different agent parametrizations but follows the same workflow.

Your goal is to estimate how each agent's behavior contributed to the outcome of its execution by comparing differences across executions.

Base your attribution on:
- the system score differences across executions,
- the system prediction
- how agent outputs differ across executions,
- whether these differences improve or degrade downstream reasoning,
- whether an agent introduces useful reasoning or merely follows,
- whether an agent propagates, corrects, or ignores errors.

## Attribution Scale
Assign a float score in [-1.0, 1.0] to each agent for each execution:
- positive = helpful contribution
- negative = harmful contribution
- 0 = neutral or unclear impact

## Critical Reasoning Guidelines
- Identical predictions or system scores do not imply identical contributions; base attribution on differences in reasoning and role adherence.

## Input
You will receive JSON with:
- "query": same for all executions
- "executions": list of execution results, each containing:
  - "system_score": this is the reward signal for attribution, computed by the evaluator based on the execution's prediction. The goal is to maximize this quantity.
  - "agent_outputs": mapping from agent name to that agent's output. The output with the key "prediction" is the final prediction, on which the system is scored on.

## Output
First, briefly compare the executions and identify the key behavioral differences that lead to differences in the system score.

Then return JSON inside <attribution>...</attribution>.

## Critical Formatting Rules
- The content inside <attribution>...</attribution> must be valid JSON.
- Do not use markdown code fences inside the <attribution> block.
- The top-level JSON object inside <attribution> must have exactly one key: "executions".
- "executions" must be a list with exactly {n_parametrizations} items.
- Preserve the same execution order as given in the input.
- Each item in "executions" must be an object with exactly one key: "agent_credits".
- "agent_credits" must be an object containing exactly these agent names and no others:
  {agent_names}
- Every credit must be a float between -1.0 and 1.0.
- Do not omit any execution.
- Do not put explanations, summaries, or any other text inside the JSON.
- Any reasoning or explanation must be placed outside the <attribution> block.

# Few-Shot Example

## Input
{
  "query": "A number doubled and then increased by 3 equals 11. What is the number?",
  "executions": [
    {
      "system_score": 1.0,
      "agent_outputs": {
        "planner": {
            "plan": "Let x be the number. Then 2x + 3 = 11. Subtract 3 to get 2x = 8. Divide by 2 to get x = 4."
        },
        "formatter": {
            "formatted_answer": "4"
        },
        "critic": {
            "correct: "Yes",
            "prediction": "4"
        }
      }
    },
    {
      "system_score": 0.0,
      "agent_outputs": {
        "planner": {
            "plan": "A possible number is 5 because doubling gives 10 and that is close to 11."
        },
        "formatter": {
            "formatted_answer": "5"
        },
        "critic": {
            "correct: "True",
            "prediction": "5"
        }
      }
    },   
    {
      "system_score": 0.0,
      "agent_outputs": {
        "planner": {
            "plan": "Let x be the number. Then 2x + 3 = 11. Subtract 3 to get 2x = 8. Divide by 2 to get x = 4."
        },
        "formatter": {
            "formatted_answer": "x = 4"
        },
        "critic": {
            "correct: "True",
            "prediction": "x = 4"
        }
      }
    }
  ]
}

## Output

The first execution is fully correct. The planner sets up and solves the equation properly, the formatter states the correct answer, and the critic confirms consistency between reasoning and output. An attribution of 1.0 for all agents is appropriate as all agents performed their roles without any error.

The second execution fails primarily due to the planner. The planner does not formulate the equation and instead relies on an unsupported guess, which leads to an incorrect solution. The formatter correctly follows its role by presenting the answer clearly and in the required format, but it propagates the incorrect result. The critic fails to identify that the reasoning is invalid and does not challenge the incorrect answer. Therefore, the planner deserves an attribution of -1.0, as it is the source of the error, while the critic shares some blame for not catching the error, giving an attribution of -0.8. The formatter, while it does propagate the error, is not the cause and still fulfills its role in terms of format and clarity, so an attribution of 1.0 is reasonable.

The third execution shows a localized failure due to formatting. The planner correctly solves the equation, and the critic confirms that the reasoning is valid. However, the formatter outputs the answer in an incorrect format, violating the output requirements. A system score of 0.4 is reasonable here: the underlying reasoning and value are correct, but the final output is not usable due to formatting errors.

<attribution>
{
  "executions": [
    {
      "agent_credits": {
        "planner": 1.0,
        "formatter": 1.0,
        "critic": 1.0
      }
    },
    {
      "agent_credits": {
        "planner": -1.0,
        "formatter": 1.0,
        "critic": -0.8
      }
    },
    {
      "agent_credits": {
        "planner": 1.0,
        "formatter": -1.0,
        "critic": 0.9
      }
    }
  ]
}
</attribution>

# Attribution

Now perform attribution for the following context.

## Query and Executions
{query_and_prompt_sets}

Compare all {n_parametrizations} executions and assign scores for agents in {agent_names}."""

# Alternative attribution prompt for ablation studies
ATTRIBUTION_PROMPT_ALTERNATIVE =  """You are an attribution agent in a multi-agent pipeline. Your job is to assess how much each agent contributed — positively or negatively — to the final outcome of an execution.

## What You Receive
A JSON object containing:
- **`agent_context`**: A mapping from agent name to its intended role
- **`query`**: The shared input across all executions
- **`executions`**: A list of runs of the same query, each with:
  - `system_score`: A numeric quality signal for the final output, calculated by the evaluator. The goal is to maximize this score.
  - `agent_outputs`: A mapping from agent name to that agent's output. The output with the key "prediction" is the final prediction, which is scored by the evaluator to produce `system_score`.

## Your Task
Compare executions to identify *what differed* and *why those differences mattered*. Then score each agent's contribution per execution. Remember that the goal is to maximize the system score.

Focus on:
- What is the final system score?
- Did the agent fulfill its described role?
- Did this agent's behavior lead to better or worse system scores compared to other executions?
- Did its output improve, degrade, or leave unchanged the quality of downstream reasoning?
- Did it introduce useful reasoning, or mainly follow what was already present?
- Did it catch and correct errors, propagate them, or ignore them?

> **Key principle**: Similar outputs across executions do not imply similar contributions. Look at *how* and *why* outputs differed and what effect they had on the final outcome.

## Scoring
Assign a float in **[-1.0, 1.0]** per agent per execution:

| Range | Meaning |
|-------|---------|
| 0.7 - 1.0 | Clearly helpful; drove correct reasoning |
| 0.1 - 0.6 | Mildly helpful or supportive |
| 0.0 | Neutral or ambiguous impact |
| -0.1 - -0.6 | Mildly harmful; introduced noise or errors |
| -0.7 - -1.0 | Clearly harmful; caused or failed to catch critical errors |

## Output Format
1. **Briefly explain** key behavioral differences across executions outside the JSON block.
2. **Return attribution JSON** inside `<attribution>...</attribution>` tags.

### JSON Schema
```json
{
  "executions": [
    {
      "agent_credits": {
        "<agent_name>": <float>,
        ...
      }
    }
  ]
}
````

### Formatting Rules

* The `<attribution>` block must contain **valid JSON only** — no markdown, no comments.
* `executions` must have exactly **{n_parametrizations}** items, in the **same order** as the input.
* Each item in `executions` must be an object with exactly one key: `agent_credits`.
* `agent_credits` must include **exactly** these agents: `{agent_names}`
* All scores must be floats in `[-1.0, 1.0]`.
* Do not include any explanation inside the `<attribution>` block.

---

# Few-Shot Example

## Input
{
  "query": "A number doubled and then increased by 3 equals 11. What is the number?",
  "executions": [
    {
      "system_score": 1.0,
      "agent_outputs": {
        "planner": {
            "plan": "Let x be the number. Then 2x + 3 = 11. Subtract 3 to get 2x = 8. Divide by 2 to get x = 4."
        },
        "formatter": {
            "formatted_answer": "4"
        },
        "critic": {
            "correct: "Yes",
            "prediction": "4"
        }
      }
    },
    {
      "system_score": 0.0,
      "agent_outputs": {
        "planner": {
            "plan": "A possible number is 5 because doubling gives 10 and that is close to 11."
        },
        "formatter": {
            "formatted_answer": "5"
        },
        "critic": {
            "correct: "True",
            "prediction": "5"
        }
      }
    },   
    {
      "system_score": 0.0,
      "agent_outputs": {
        "planner": {
            "plan": "Let x be the number. Then 2x + 3 = 11. Subtract 3 to get 2x = 8. Divide by 2 to get x = 4."
        },
        "formatter": {
            "formatted_answer": "x = 4"
        },
        "critic": {
            "correct: "True",
            "prediction": "x = 4"
        }
      }
    }
  ]
}

## Output

The first execution is fully correct. The planner sets up and solves the equation properly, the formatter states the correct answer, and the critic confirms consistency between reasoning and output. An attribution of 1.0 for all agents is appropriate as all agents performed their roles without any error.

The second execution fails primarily due to the planner. The planner does not formulate the equation and instead relies on an unsupported guess, which leads to an incorrect solution. The formatter correctly follows its role by presenting the answer clearly and in the required format, but it propagates the incorrect result. The critic fails to identify that the reasoning is invalid and does not challenge the incorrect answer. Therefore, the planner deserves an attribution of -1.0, as it is the source of the error, while the critic shares some blame for not catching the error, giving an attribution of -0.8. The formatter, while it does propagate the error, is not the cause and still fulfills its role in terms of format and clarity, so an attribution of 1.0 is reasonable.

The third execution shows a localized failure due to formatting. The planner correctly solves the equation, and the critic confirms that the reasoning is valid. However, the formatter outputs the answer in an incorrect format, violating the output requirements. A system score of 0.4 is reasonable here: the underlying reasoning and value are correct, but the final output is not usable due to formatting errors.

<attribution>
{
  "executions": [
    {
      "agent_credits": {
        "planner": 1.0,
        "formatter": 1.0,
        "critic": 1.0
      }
    },
    {
      "agent_credits": {
        "planner": -1.0,
        "formatter": 1.0,
        "critic": -0.8
      }
    },
    {
      "agent_credits": {
        "planner": 1.0,
        "formatter": -1.0,
        "critic": 0.9
      }
    }
  ]
}
</attribution>

---

## Now Perform Attribution

{query_and_prompt_sets}

Compare all {n_parametrizations} executions and assign scores for agents: {agent_names}."""


CONSTRAINED_ATTRIBUTION_PROMPT = """# Challenge
The following example illustrates reward attribution across parametrizations of agents (e.g., prompt variants, temperature, or backbone models).

Your attribution should explicitly compare agent behavior across different parametrizations to identify which agent configurations contribute to better system performance, as this signal will be used to optimize the agentic system.

## Your Task
You will receive multiple results for the SAME query. Each run uses different agent parametrizations but processes the same input query.

For EACH parametrization, you must:
1. Compare the system score, agent outputs, and final predictions across ALL parametrizations
2. Analyze how each agent's outputs contributed to the quality of that run's prediction
3. Assign a weight to each agent in the range [0, 1]
4. Ensure weights sum to exactly 1.0 for each parametrization

NOTE: The system score may be positive or negative.
For positive scores, the attribution weights represent how much each agent contributed to the system's success (credit assignment).
For negative scores, the attribution weights represent how much each agent contributed to the failure (blame assignment).
In both cases, weights must sum to 1.0 and reflect relative responsibility among agents.

## Input Format
- **query**: The input question/task (SAME for all)
- **parametrizations**: List of results, one per parametrization. Each contains:
  - **system_score**: The reward achieved (higher = better)
  - **agent_outputs**: What each agent produced (mapping agent_name -> {{output_key: output_value}}). The variable with key "prediction" is the final system prediction, which is scored by the evaluator to produce system_score.

## Output Format
Before assigning weights, briefly reflect on how differences in agent behavior across configurations influenced the outcome.
Return a JSON object containing a list with the agent reward attribution weights. The list must be in the same order as the input parametrizations.
Enclose your response in <weights>...</weights> tags.

**Expected dimensions:** You will receive <n_parametrizations> parametrizations and must assign weights for <agent_names> agents.
Your output must be a JSON list with exactly <n_parametrizations> entries (one per parametrization), where each entry contains weights for all agents in <agent_names>.

## Few-Shot Example

Suppose we evaluate several different parametrizations of a planner-proposer-critic system on the same query. Although the agents share the same roles across runs, differences in their parametrizations lead to distinct reasoning behaviors and outcomes, which must be compared for reward attribution.

{{
  "query": "A train travels 120 km at a constant speed. If the speed had been 10 km/h faster, the trip would have taken 30 minutes less. What was the original speed?",
  "parametrizations": [
    {{
      "system_score": 0.8,
      "prediction": "44.24 km/h",
      "agent_outputs": [
        {{
          "agent": "planner",
          "output": "Let v be original speed. Time difference condition gives 120/v - 120/(v+10) = 0.5. Solve symbolically: v^2 + 10v - 2400 = 0. Positive root is ≈44.24. Verify numerically: time difference ≈0.5h."
        }},
        {{ "agent": "proposer_1", "output": "Solving the quadratic gives v ≈ 44.24 km/h." }},
        {{ "agent": "proposer_2", "output": "Maybe round to 45 km/h." }},
        {{ "agent": "critic", "output": "Planner and proposer_1 are consistent and numerically verified; proposer_2 is only a rough approximation." }}
      ]
    }},
    {{
      "system_score": -0.6,
      "prediction": "50 km/h",
      "agent_outputs": [
        {{
          "agent": "planner",
          "output": "Estimate by trying values: at 50 km/h the time is 2.4h, at 60 km/h it is 2.0h, difference is 0.4h, close to 0.5h."
        }},
        {{ "agent": "proposer_1", "output": "50 km/h seems reasonable from estimation." }},
        {{ "agent": "proposer_2", "output": "Probably 60 km/h since it's a nice number." }},
        {{ "agent": "critic", "output": "50 km/h is close enough; accept." }}
      ]
    }}
  ]
}}



### Expected Output

Comparing the two parametrizations, the first configuration clearly performs better. Relative to the second, the planner here not only derives the correct equation but also verifies the solution numerically, which strongly reduces uncertainty and justifies a high weight. Proposer_1 directly supports this reasoning, while proposer_2 only offers a coarse approximation and should be downweighted. The critic is stricter than in the second configuration, where it accepts an imprecise argument, and therefore deserves meaningful credit.
→ Weights: {{ "planner": 0.40, "proposer_1": 0.30, "proposer_2": 0.05, "critic": 0.25 }}

In the second configuration, the negative system score indicates failure and thus blame assignment. Compared to the first configuration, the planner relies on rough estimation rather than enforcing the exact constraint, and the critic passively accepts a “close enough” argument instead of challenging it. Proposer_1 contributes to the incorrect conclusion, while proposer_2's suggestion is largely unhelpful. Blame is therefore shared mainly between planner, proposer_1, and critic.
→ Weights: {{ "planner": 0.35, "proposer_1": 0.30, "proposer_2": 0.15, "critic": 0.20 }}

<weights>
[
  {{ "planner": 0.40, "proposer_1": 0.30, "proposer_2": 0.05, "critic": 0.25 }},
  {{ "planner": 0.35, "proposer_1": 0.30, "proposer_2": 0.15, "critic": 0.20 }}
]
</weights>


## Agent Context

Each agent has a specific role, inputs, and outputs. Here is the metadata for all agents:

{agent_context}


## Query and configurations to Process

{query_and_prompt_sets}

Now produce your attribution weights, consisting of {n_parametrizations} entries (one per parametrization), where each entry contains weights for all agents in {agent_names}.
"""