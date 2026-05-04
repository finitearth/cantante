# CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution

![Python](https://img.shields.io/badge/python-3.12%2B-blue) ![uv](https://img.shields.io/badge/package%20manager-uv-purple)

> CANTANTE converts system-level rewards into per-agent optimization signals via contrastive in-group attribution, enabling principled local optimization of multi-agent systems.

## Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Installation

```bash
git clone [REPO URL PLACEHOLDER]
cd cantante
uv sync
```

### API key

All LLM calls use an OpenAI-compatible API. Place your key in `token.txt`. `run_experiment.py` reads it with `open("token.txt").read().strip()`. Update `task_llm_kwargs.base_url` and `meta_llm_kwargs.base_url` in your config to point to your endpoint.

---

## Interactive inference notebook

For manual inspection and experimentation, `notebooks/inference.ipynb` provides an interactive way to load resulting configurations and run single-example inference.

---

## Quick start — single experiment

**1. Run the optimization**

```bash
uv run scripts/run_experiment.py --config configs/ablations/group_size.yaml
```


**2. Evaluate on the test set**

```bash
uv run scripts/run_eval.py --run_dir results/ablations/<run_dir>
```

Results land in `<run_dir>/eval/`. Add `--eval_all` to score every step, or `--step N` for a specific step.

**3. Evaluate the initial prompts as a baseline**

```bash
uv run scripts/eval_initial_prompts.py --config configs/main_experiments/main_exp.yaml
```

**Resuming an interrupted run**

```bash
uv run scripts/restart_experiment.py --run_dir results/main_experiments/<run_dir>
```

---

## Reproducing paper experiments

`run_batch.py` expands the `grid:` block of a config into individual runs, executes each one, and automatically triggers evaluation afterwards. Multiple instances can be launched simultaneously — each worker acquires a file-system lock before starting a job, so there is no double execution and you can parallelise across machines by pointing them at the same `results/` directory (e.g. on a shared filesystem).

```bash
# Main experiments:
uv run scripts/run_batch.py --config configs/main_experiments

# Ablation studies
uv run scripts/run_batch.py --config configs/ablations/
```

**Flags**

| Flag | Effect |
|------|--------|
| `--dry-run` | Print the job list without running anything |
| `--skip-completed` | Skip runs whose output directory already has `.finished` (default: on) |
| `--eval-only` | Skip optimization; only evaluate already-finished runs |
| `--ignore-locks` | Start even if a lock file exists |

---

## Generating paper artefacts

Once all results are in place:

```bash
# LaTeX tables → tables/
uv run scripts/create_tables.py

# PDF figures → figures/
uv run scripts/create_plots.py
```

Both scripts read from `results/main_experiments/` and `results/ablations/`.

---

## Code execution (MBPP only)

The MBPP task executes LLM-generated code in a subprocess. Before running any MBPP experiment you must explicitly opt in:

```bash
export ALLOW_CODE_EXECUTION=1
```

We strongly recommend running inside a sandbox (Docker, gVisor, nsjail). See `src/agent_tools/mbpp.py` for details.

---

## Config reference

| Field | Description | Example |
|---|---|---|
| `experiment.dataset` | benchmark to run | `gsm8k`, `hotpotqa`, `mbpp` |
| `experiment.optimizer` | optimization algorithm | `cantante`, `mipro`, `gepa` |
| `experiment.seed` | global random seed | `42` |
| `experiment.max_token_budget` | total token cap | `10000000` |
| `experiment.output_dir` | where run artefacts are written | `./results/ablations/my_run` |
| `task_llm_kwargs.model` | model used by the MAS agents | any OpenAI-compatible model ID |
| `meta_llm_kwargs.model` | model used for attribution and optimization | any OpenAI-compatible model ID |
| `grid` | key–value lists expanded by `run_batch.py` | `experiment.seed: [7, 42, 47]` |
| `setup_dict_folder` | folder containing per-dataset setup YAMLs | `configs/setups` |

---

## Adding a custom task or graph

1. **Dataset loader** — add a case in `src/experiment/load_datasets.py` returning a `BaseMASTask` subclass (`src/tasks/gsm8k.py` is the simplest reference).

2. **Agent setup YAML** — create `configs/setups/<your_task>.yaml` with:

   ```yaml
   edges:
     - from: "__start__"
       to: "agent_a"
     - from: "agent_a"
       to: "__end__"

   agents:
     - name: "agent_a"
       task_description: "..."
       input_vars: ["query"]
       output_vars: ["prediction"]
       tools: []
       max_tool_calls: 0

   init_agent_prompt_pool:
     agent_a:
       - "Prompt variant 1 ..."
       - "Prompt variant 2 ..."
       # at least 6 variants recommended
   ```

3. **Tools (optional)** — add a `BaseToolsAdapter` subclass in `src/agent_tools/` and register it in `src/agent_tools/base.py::get_tools()`.

4. **Experiment config** — copy any existing config, set `experiment.dataset` to your task name, and point `setup_dict_folder` at `configs/setups`.

5. Run with `run_experiment.py` or `run_batch.py` as normal.

---

## Repository structure

```
Cantante/
├── configs/
│   ├── main_experiments/          grid configs for main paper runs
│   ├── ablations/                 one config per ablation study
│   └── setups/                    per-dataset MAS graph + initial prompt pools
├── scripts/
│   ├── run_experiment.py          run a single experiment from a YAML config
│   ├── run_batch.py               expand a grid config and run all jobs (+ eval)
│   ├── run_eval.py                evaluate a finished run on the test split
│   ├── eval_initial_prompts.py    baseline: evaluate seed prompts only
│   ├── restart_experiment.py      resume an interrupted run from checkpoint
│   ├── create_tables.py           generate LaTeX tables  →  tables/
│   └── create_plots.py            generate paper figures →  figures/
├── src/
│   ├── mas.py                     MASPredictor — LangGraph-based MAS engine
│   ├── meta_prompts.py            system prompts for mutation & crossover
│   ├── prompt_structures.py       AgentPromptPool / Set / Batch data structures
│   ├── prompt_utils.py            token counting and length utilities
│   ├── callbacks.py               CheckpointCallback, OptimizationCallback
│   ├── candidate_selector.py      prompt candidate selection strategies
│   ├── agent_tools/
│   │   ├── base.py                tool registry (get_tools) and BaseToolsAdapter
│   │   ├── gsm8k.py               maths tools
│   │   ├── hotpotqa.py            QA retrieval tools
│   │   └── mbpp.py                sandboxed code execution (⚠ see Setup)
│   ├── tasks/
│   │   ├── base.py                BaseMASTask — evaluation loop and scoring
│   │   ├── gsm8k.py               GSM8K maths task
│   │   ├── hotpotqa.py            HotpotQA multi-hop QA task
│   │   └── mbpp.py                MBPP code generation task
│   ├── optimization/
│   │   ├── cantante.py            CANTANTE — attribution-guided genetic optimiser
│   │   ├── local_optimizer.py     CAPO and EvoPrompt node-level optimisers
│   │   ├── dspy_mas_optimizer.py  wrappers for GEPA and MIPROv2 (DSPy)
│   │   ├── dspy_mas_wrapper.py    DSPy integration adapters
│   │   ├── base_mas_optimizer.py  base optimiser interface
│   │   ├── broker_agent_optimizers.py  multi-agent optimization coordination
│   │   └── proxy_task.py          attribution proxy task wrapper
│   ├── attribution/
│   │   ├── base.py                BaseAttributer
│   │   ├── absolute.py            absolute token importance scoring
│   │   ├── relative.py            relative (gradient-based) attribution
│   │   └── naive.py               identity / uniform baseline
│   ├── analysis/
│   │   ├── utils.py               DataFrame loading, load_main_results_df, compute_ranks
│   │   ├── tables.py              LatexTable, render_table, get_agg_table
│   │   └── style.py               matplotlib style configuration
│   └── experiment/
│       ├── load_datasets.py       get_tasks() — dataset loader factory
│       ├── configs.py             dataset configuration registry
│       └── utils.py               seed_everything, get_logger, inject_tool_descriptions
├── notebooks/
│   ├── generate_seed_prompts.ipynb   create initial prompt pools
│   └── inference.ipynb               single-example inference testing
├── results/
│   ├── main_experiments/          one sub-directory per completed run
│   │   └── <run_name>/
│   │       ├── prompts_per_step.parquet   full optimization trajectory
│   │       ├── .finished                  flag written on clean exit
│   │       ├── checkpoints/               per-step checkpoint JSONs
│   │       └── eval/
│   │           ├── scores_per_step.parquet   test-set scores per prompt set
│   │           └── token_usage.yaml
│   └── ablations/                 same structure, one dir per ablation run
├── figures/                       generated PDF figures
├── tables/                        generated LaTeX table files
├── token.txt                      API key — not committed, see Setup
└── pyproject.toml
```
